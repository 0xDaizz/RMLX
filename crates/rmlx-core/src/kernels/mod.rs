//! Kernel registry with AOT -> JIT fallback pipeline state caching.
//!
//! Uses RwLock for concurrent read access to pipeline/JIT caches.
//! Supports Metal function constants for type specialization.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::RwLock;

use metal::{CompileOptions, FunctionConstantValues, MTLDataType};
use smallvec::SmallVec;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::pipeline_cache::DiskPipelineCache;
use rmlx_metal::FunctionConstant;

use crate::dtype::DType;

/// Errors from kernel operations.
#[derive(Debug)]
pub enum KernelError {
    /// Kernel not found in registry.
    NotFound(String),
    /// Shader compilation failed.
    CompilationFailed(String),
    /// Pipeline creation failed.
    PipelineFailed(String),
    /// Function constant specialization failed (e.g., wrong constant index or type).
    Specialization(String),
    /// Invalid shape for the operation.
    InvalidShape(String),
    /// Internal error (e.g., lock poisoned).
    Internal(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::NotFound(name) => write!(f, "kernel not found: {name}"),
            KernelError::CompilationFailed(e) => write!(f, "shader compilation failed: {e}"),
            KernelError::PipelineFailed(e) => write!(f, "pipeline creation failed: {e}"),
            KernelError::Specialization(e) => {
                write!(f, "function constant specialization failed: {e}")
            }
            KernelError::InvalidShape(e) => write!(f, "invalid shape: {e}"),
            KernelError::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// A function constant value for Metal pipeline specialization.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionConstantValue {
    Bool(bool),
    U32(u32),
    F32(f32),
}

impl Eq for FunctionConstantValue {}

impl std::hash::Hash for FunctionConstantValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            FunctionConstantValue::Bool(v) => v.hash(state),
            FunctionConstantValue::U32(v) => v.hash(state),
            FunctionConstantValue::F32(v) => v.to_bits().hash(state),
        }
    }
}

/// Cache key for compiled pipelines.
///
/// Uses `Cow<'static, str>` for kernel names — zero-cost `Borrowed` for the
/// common case (string literals) while still supporting dynamically built
/// names (e.g. `format!`).  Constants use `SmallVec<[...; 4]>` to avoid heap
/// allocation when there are ≤4 function constants.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    pub kernel_name: Cow<'static, str>,
    pub dtype: DType,
    /// Optional function constants for specialization.
    pub constants: SmallVec<[(u32, FunctionConstantValue); 4]>,
}

/// Kernel registry managing AOT libraries and JIT compilation with pipeline caching.
///
/// Lookup order: in-memory pipeline cache -> disk pipeline cache -> AOT metallib
/// -> JIT cache -> error.
///
/// Uses `RwLock` instead of `Mutex` for the caches, allowing concurrent readers
/// (the common case — pipeline lookups) without contention.
pub struct KernelRegistry {
    device: GpuDevice,
    aot_lib: Option<metal::Library>,
    /// JIT cache: name -> (source text, compiled Library).
    /// The source text is retained so that [`DiskPipelineCache`] can compute a
    /// stable SHA-256 key when a pipeline miss occurs.
    jit_cache: RwLock<HashMap<String, JitEntry>>,
    pipelines: RwLock<HashMap<PipelineKey, metal::ComputePipelineState>>,
    /// Persistent disk-backed pipeline cache.  `None` only when explicitly
    /// disabled (e.g., in tests via `new_without_disk_cache`).
    disk_cache: Option<DiskPipelineCache>,
    /// Frozen snapshot of pipelines for lock-free reads.
    ///
    /// After calling [`freeze()`](Self::freeze), this points to a heap-allocated
    /// `HashMap` clone of `pipelines`. Pipeline lookups check this pointer first
    /// (lock-free `Acquire` load) before falling back to the `RwLock`.
    ///
    /// Pipelines added after `freeze()` (e.g., JIT compilation of new kernels)
    /// go into the `RwLock` map; call `freeze()` again to capture them.
    frozen_pipelines: AtomicPtr<HashMap<PipelineKey, metal::ComputePipelineState>>,
}

/// A JIT-compiled library together with the source text it was compiled from.
struct JitEntry {
    source: String,
    library: metal::Library,
}

impl KernelRegistry {
    /// Create a new registry. Tries to load AOT metallib from `METALLIB_PATH`.
    ///
    /// A [`DiskPipelineCache`] is created automatically and will persist
    /// compiled metallib binaries to `~/.cache/rmlx/pipelines/`.
    pub fn new(device: GpuDevice) -> Self {
        let metallib_path = crate::METALLIB_PATH;
        let aot_lib = if metallib_path.is_empty() {
            None
        } else {
            let path = std::path::Path::new(metallib_path);
            if path.exists() {
                device.raw().new_library_with_file(path).ok()
            } else {
                None
            }
        };

        let disk_cache = Some(DiskPipelineCache::new(device.raw()));

        Self {
            device,
            aot_lib,
            jit_cache: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            disk_cache,
            frozen_pipelines: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Create a registry without a disk pipeline cache.
    ///
    /// Useful for tests or environments where disk persistence is undesirable.
    #[cfg(test)]
    pub fn new_without_disk_cache(device: GpuDevice) -> Self {
        let metallib_path = crate::METALLIB_PATH;
        let aot_lib = if metallib_path.is_empty() {
            None
        } else {
            let path = std::path::Path::new(metallib_path);
            if path.exists() {
                device.raw().new_library_with_file(path).ok()
            } else {
                None
            }
        };

        Self {
            device,
            aot_lib,
            jit_cache: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            disk_cache: None,
            frozen_pipelines: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Get or create a compute pipeline for the given kernel and dtype.
    ///
    /// Tries: in-memory pipeline cache -> disk pipeline cache -> AOT library
    /// -> JIT cache -> error.
    pub fn get_pipeline(
        &self,
        kernel_name: &str,
        dtype: DType,
    ) -> Result<metal::ComputePipelineState, KernelError> {
        // Fast path: build a Borrowed key for the cache lookup (no allocation).
        let lookup_key = PipelineKey {
            kernel_name: Cow::Borrowed(
                // SAFETY: the borrow only lives for the cache lookup below;
                // we never store this key.  The transmute extends the lifetime
                // to 'static so it fits into Cow<'static, str>, which is
                // required by PipelineKey.
                unsafe { &*(kernel_name as *const str) },
            ),
            dtype,
            constants: SmallVec::new(),
        };

        // 1a. Check frozen snapshot (lock-free)
        if let Some(pipeline) = self.frozen_lookup(&key) {
            return Ok(pipeline);
        }

        // 1b. Check in-memory pipeline cache (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            if let Some(pipeline) = cache.get(&lookup_key) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: allocate an owned key for insertion.
        let key = PipelineKey {
            kernel_name: Cow::Owned(kernel_name.to_string()),
            dtype,
            constants: SmallVec::new(),
        };

        // 2. Try disk pipeline cache — needs JIT source for the cache key.
        if let Some(ref disk) = self.disk_cache {
            if let Some(source) = self.find_jit_source_for_kernel(kernel_name) {
                if let Ok(pso) = disk.get_or_compile(&source, kernel_name, &[]) {
                    // Store in the in-memory cache for future fast-path hits.
                    if let Ok(mut cache) = self.pipelines.write() {
                        cache.insert(key, pso.clone());
                    }
                    return Ok(pso);
                }
            }
        }

        // 3. Try AOT library
        let function = self
            .aot_lib
            .as_ref()
            .and_then(|lib| lib.get_function(kernel_name, None).ok());

        // 4. If AOT miss, try JIT cache (read lock)
        let function = match function {
            Some(f) => f,
            None => {
                let jit = self
                    .jit_cache
                    .read()
                    .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
                let jit_fn = jit
                    .values()
                    .find_map(|entry| entry.library.get_function(kernel_name, None).ok());
                match jit_fn {
                    Some(f) => f,
                    None => {
                        return Err(KernelError::NotFound(format!(
                            "{kernel_name} (dtype={dtype}): not in AOT or JIT cache"
                        )));
                    }
                }
            }
        };

        // 5. Create pipeline from the function
        let pipeline = self
            .device
            .raw()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| KernelError::PipelineFailed(e.to_string()))?;

        // 6. Cache it (write lock)
        {
            let mut cache = self
                .pipelines
                .write()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            cache.insert(key, pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Look up a kernel function from AOT or JIT libraries, applying function
    /// constants if provided.
    ///
    /// When `fcv` is `Some(...)`, this method distinguishes between:
    /// - The kernel name not existing in any library -> `KernelError::NotFound`
    /// - The kernel existing but constant specialization failing (wrong index,
    ///   wrong type, etc.) -> `KernelError::Specialization` with the Metal error
    ///
    /// This avoids the bug where `.ok()` on `get_function` collapses real
    /// specialization errors into a misleading "not found" message.
    fn lookup_function_with_constants(
        &self,
        kernel_name: &str,
        fcv: Option<FunctionConstantValues>,
        dtype: DType,
    ) -> Result<metal::Function, KernelError> {
        // Helper: try to get a specialized function from a single library.
        // Returns Ok(function) on success, Err(Some(metal_error)) if the
        // function exists but specialization failed, or Err(None) if the
        // function simply doesn't exist in this library.
        let try_library = |lib: &metal::Library,
                           fcv: Option<FunctionConstantValues>|
         -> Result<metal::Function, Option<String>> {
            match fcv {
                None => lib.get_function(kernel_name, None).map_err(|_| None),
                Some(constants) => {
                    // First, check whether the function name exists at all
                    // (plain lookup without constants — cheap, no NSError path).
                    if lib.get_function(kernel_name, None).is_err() {
                        return Err(None); // not in this library
                    }
                    // The function exists; now apply constants. Any error here
                    // is a real specialization failure from Metal.
                    lib.get_function(kernel_name, Some(constants)).map_err(Some)
                }
            }
        };

        // Try AOT library first.
        if let Some(aot) = self.aot_lib.as_ref() {
            match try_library(aot, fcv.clone()) {
                Ok(f) => return Ok(f),
                Err(Some(metal_err)) => {
                    return Err(KernelError::Specialization(format!(
                        "{kernel_name}: {metal_err}"
                    )));
                }
                Err(None) => { /* not in AOT, try JIT */ }
            }
        }

        // Try JIT cache.
        let jit = self
            .jit_cache
            .read()
            .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;

        for entry in jit.values() {
            match try_library(&entry.library, fcv.clone()) {
                Ok(f) => return Ok(f),
                Err(Some(metal_err)) => {
                    return Err(KernelError::Specialization(format!(
                        "{kernel_name}: {metal_err}"
                    )));
                }
                Err(None) => continue,
            }
        }

        Err(KernelError::NotFound(format!(
            "{kernel_name} (dtype={dtype}): not in AOT or JIT cache"
        )))
    }

    /// Get or create a compute pipeline with function constant specialization.
    ///
    /// Function constants allow specializing a single compiled kernel for
    /// different types, layouts, or modes without recompilation. Intended uses:
    /// - `traditional` (bool) in RoPE
    /// - `forward` (bool) in RoPE
    /// - `has_w` (bool) in RMS norm
    /// - Type selection constants
    ///
    /// When `constants` is non-empty, builds `MTLFunctionConstantValues`, applies
    /// them via `library.get_function(name, Some(fcv))`, and creates a specialized
    /// pipeline. When empty, falls back to the plain function lookup.
    pub fn get_pipeline_with_constants(
        &self,
        kernel_name: &str,
        dtype: DType,
        constants: &[(u32, FunctionConstantValue)],
    ) -> Result<metal::ComputePipelineState, KernelError> {
        // Fast path: build a Borrowed key for the cache lookup (no allocation).
        let lookup_key = PipelineKey {
            kernel_name: Cow::Borrowed(unsafe { &*(kernel_name as *const str) }),
            dtype,
            constants: constants.iter().cloned().collect(),
        };

        // 1a. Check frozen snapshot (lock-free)
        if let Some(pipeline) = self.frozen_lookup(&key) {
            return Ok(pipeline);
        }

        // 1b. Check in-memory pipeline cache (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            if let Some(pipeline) = cache.get(&lookup_key) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: allocate an owned key for insertion.
        let key = PipelineKey {
            kernel_name: Cow::Owned(kernel_name.to_string()),
            dtype,
            constants: lookup_key.constants,
        };

        // 2. Try disk pipeline cache with function constants.
        if let Some(ref disk) = self.disk_cache {
            let disk_constants = Self::to_disk_constants(constants);
            if let Some(source) = self.find_jit_source_for_kernel(kernel_name) {
                if let Ok(pso) = disk.get_or_compile(&source, kernel_name, &disk_constants) {
                    if let Ok(mut cache) = self.pipelines.write() {
                        cache.insert(key, pso.clone());
                    }
                    return Ok(pso);
                }
            }
        }

        // 3. Build FunctionConstantValues if needed
        let fcv = if constants.is_empty() {
            None
        } else {
            let values = FunctionConstantValues::new();
            for (index, constant) in constants {
                match constant {
                    FunctionConstantValue::Bool(v) => {
                        let val: u8 = u8::from(*v);
                        values.set_constant_value_at_index(
                            &val as *const u8 as *const c_void,
                            MTLDataType::Bool,
                            *index as u64,
                        );
                    }
                    FunctionConstantValue::U32(v) => {
                        values.set_constant_value_at_index(
                            v as *const u32 as *const c_void,
                            MTLDataType::UInt,
                            *index as u64,
                        );
                    }
                    FunctionConstantValue::F32(v) => {
                        values.set_constant_value_at_index(
                            v as *const f32 as *const c_void,
                            MTLDataType::Float,
                            *index as u64,
                        );
                    }
                }
            }
            Some(values)
        };

        // 4. Find the function from library (AOT then JIT), applying constants.
        //
        // When function constants are provided, we must distinguish between
        // "kernel not found" (name doesn't exist in the library) and
        // "specialization failed" (name exists but constants are invalid).
        // Metal's `newFunctionWithName:constantValues:error:` returns an
        // NSError for both cases, so we probe with a plain lookup first to
        // determine whether the function exists, then apply constants.
        let function = self.lookup_function_with_constants(kernel_name, fcv, dtype)?;

        // 5. Create pipeline from the (possibly specialized) function
        let pipeline = self
            .device
            .raw()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| KernelError::PipelineFailed(e.to_string()))?;

        // 6. Cache it (write lock)
        {
            let mut cache = self
                .pipelines
                .write()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            cache.insert(key, pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Register a JIT-compiled shader source under the given name.
    pub fn register_jit_source(&self, name: &str, source: &str) -> Result<(), KernelError> {
        let options = CompileOptions::new();
        // Metal 3.1+ required for bfloat16 support.
        // metal-rs CompileOptions sets language version automatically based on SDK.
        let lib = self
            .device
            .raw()
            .new_library_with_source(source, &options)
            .map_err(|e| KernelError::CompilationFailed(format!("{name}: {e}")))?;

        let mut cache = self
            .jit_cache
            .write()
            .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
        cache.insert(
            name.to_string(),
            JitEntry {
                source: source.to_string(),
                library: lib,
            },
        );
        Ok(())
    }

    /// Register a JIT-compiled shader source only if not already registered.
    pub fn register_jit_source_if_absent(
        &self,
        name: &str,
        source: &str,
    ) -> Result<(), KernelError> {
        {
            let cache = self
                .jit_cache
                .read()
                .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
            if cache.contains_key(name) {
                return Ok(());
            }
        }
        self.register_jit_source(name, source)
    }

    /// Reference to the GPU device.
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Check if AOT library is loaded.
    pub fn has_aot(&self) -> bool {
        self.aot_lib.is_some()
    }

    /// Number of cached pipeline states.
    pub fn cached_pipeline_count(&self) -> Result<usize, KernelError> {
        let cache = self
            .pipelines
            .read()
            .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
        Ok(cache.len())
    }

    /// Number of JIT-compiled libraries.
    pub fn jit_library_count(&self) -> Result<usize, KernelError> {
        let cache = self
            .jit_cache
            .read()
            .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
        Ok(cache.len())
    }

    /// Register a kernel source with compile-time constant specialization.
    ///
    /// This implements C15 (Function constants for kernel specialization) by
    /// generating a specialized kernel source that prepends `constant constexpr`
    /// definitions to the shader, then compiles it as a new JIT library.
    ///
    /// This approach works around metal-rs not yet exposing `MTLFunctionConstantValues`
    /// by achieving specialization at the source level. The Metal compiler will
    /// optimize the kernel based on these constants (e.g., unrolling loops, removing
    /// dead branches).
    ///
    /// # Arguments
    /// - `name`: unique name for this specialized library (used as JIT cache key)
    /// - `base_source`: the Metal shader source to specialize
    /// - `specializations`: constant name-value pairs to prepend as `constant constexpr`
    ///
    /// # Example
    /// ```ignore
    /// registry.register_specialized_source(
    ///     "sdpa_d128",
    ///     SDPA_SHADER_SOURCE,
    ///     &[("HEAD_DIM", 128), ("GROUP_SIZE", 32)],
    /// )?;
    /// ```
    pub fn register_specialized_source(
        &self,
        name: &str,
        base_source: &str,
        specializations: &[(&str, u32)],
    ) -> Result<(), KernelError> {
        let mut specialized_source =
            String::from("#include <metal_stdlib>\nusing namespace metal;\n");
        for (const_name, value) in specializations {
            specialized_source.push_str(&format!(
                "constant constexpr uint {const_name} = {value};\n"
            ));
        }
        // Strip the #include and using lines from base_source to avoid duplicates
        let cleaned = base_source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.starts_with("#include <metal_stdlib>")
                    && !trimmed.starts_with("using namespace metal;")
            })
            .collect::<Vec<_>>()
            .join("\n");
        specialized_source.push_str(&cleaned);
        self.register_jit_source(name, &specialized_source)
    }

    /// Build a specialized pipeline cache key string from constants.
    ///
    /// Useful for callers that want to look up specialized pipelines
    /// by a deterministic name based on their constant values.
    ///
    /// # Example
    /// ```ignore
    /// let key = KernelRegistry::specialized_key("sdpa", &[("HEAD_DIM", 128)]);
    /// // Returns "sdpa_HEAD_DIM_128"
    /// ```
    pub fn specialized_key(base_name: &str, specializations: &[(&str, u32)]) -> String {
        let mut key = base_name.to_string();
        for (name, value) in specializations {
            key.push('_');
            key.push_str(name);
            key.push('_');
            key.push_str(&value.to_string());
        }
        key
    }

    /// Invalidate all cached pipelines (including any frozen snapshot).
    ///
    /// Useful after registering new specialized sources to force recompilation
    /// on next access.
    pub fn clear_pipeline_cache(&self) -> Result<(), KernelError> {
        // Drop the frozen snapshot first so stale entries aren't served.
        let old = self
            .frozen_pipelines
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: `old` was created by `freeze()` via `Box::into_raw`.
            unsafe {
                drop(Box::from_raw(old));
            }
        }

        let mut cache = self
            .pipelines
            .write()
            .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
        cache.clear();
        Ok(())
    }

    /// Reference to the disk pipeline cache, if available.
    pub fn disk_cache(&self) -> Option<&DiskPipelineCache> {
        self.disk_cache.as_ref()
    }

    /// Snapshot the current pipeline cache for lock-free reads.
    ///
    /// After this call, [`get_pipeline`](Self::get_pipeline) and
    /// [`get_pipeline_with_constants`](Self::get_pipeline_with_constants) will
    /// first probe the frozen snapshot via an `Acquire` atomic load — no `RwLock`
    /// contention on the hot path.
    ///
    /// Call this once after model warm-up (i.e., after all kernels have been
    /// compiled at least once). Pipelines JIT-compiled after `freeze()` will
    /// still be found via the `RwLock` fallback; call `freeze()` again to
    /// capture them in the snapshot.
    pub fn freeze(&self) {
        let snap = Box::new(
            self.pipelines
                .read()
                .expect("pipeline cache lock poisoned during freeze")
                .clone(),
        );
        let old = self
            .frozen_pipelines
            .swap(Box::into_raw(snap), Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: `old` was created by a prior `freeze()` call via
            // `Box::into_raw` and is no longer accessible through the atomic.
            unsafe {
                drop(Box::from_raw(old));
            }
        }
    }

    /// Returns `true` if the registry has a frozen pipeline snapshot.
    pub fn is_frozen(&self) -> bool {
        !self.frozen_pipelines.load(Ordering::Acquire).is_null()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Lock-free lookup in the frozen snapshot. Returns `None` if no snapshot
    /// exists or the key is not found.
    fn frozen_lookup(&self, key: &PipelineKey) -> Option<metal::ComputePipelineState> {
        let ptr = self.frozen_pipelines.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` was created by `freeze()` via `Box::into_raw` and remains
        // valid until the next `freeze()` or `Drop`. The `Acquire` ordering
        // ensures we see the fully initialized `HashMap`.
        let map = unsafe { &*ptr };
        map.get(key).cloned()
    }

    /// Find the MSL source string for a kernel by scanning the JIT cache.
    ///
    /// Returns the source of the first JIT entry whose compiled library
    /// contains a function named `kernel_name`.
    fn find_jit_source_for_kernel(&self, kernel_name: &str) -> Option<String> {
        let jit = self.jit_cache.read().ok()?;
        for entry in jit.values() {
            if entry.library.get_function(kernel_name, None).is_ok() {
                return Some(entry.source.clone());
            }
        }
        None
    }

    /// Convert `FunctionConstantValue` slice to `FunctionConstant` slice
    /// for use with [`DiskPipelineCache`].
    fn to_disk_constants(
        constants: &[(u32, FunctionConstantValue)],
    ) -> Vec<(u32, FunctionConstant)> {
        constants
            .iter()
            .map(|(idx, val)| {
                let fc = match val {
                    FunctionConstantValue::Bool(v) => FunctionConstant::Bool(*v),
                    FunctionConstantValue::U32(v) => FunctionConstant::U32(*v),
                    FunctionConstantValue::F32(v) => FunctionConstant::F32(*v),
                };
                (*idx, fc)
            })
            .collect()
    }
}

impl Drop for KernelRegistry {
    fn drop(&mut self) {
        let ptr = self.frozen_pipelines.load(Ordering::Acquire);
        if !ptr.is_null() {
            // SAFETY: `ptr` was created by `freeze()` via `Box::into_raw`.
            // We hold `&mut self`, so no concurrent access is possible.
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::foreign_types::ForeignType;

    /// A minimal Metal shader with a function constant.
    ///
    /// `constant bool IS_FORWARD [[function_constant(0)]];`
    /// The kernel body branches on `IS_FORWARD` so the compiler can specialize.
    const SHADER_WITH_CONSTANT: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant bool IS_FORWARD [[function_constant(0)]];

kernel void test_fc_kernel(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = IS_FORWARD ? 1.0f : -1.0f;
}
"#;

    /// A plain shader without function constants (for plain-path tests).
    const SHADER_PLAIN: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void test_plain_kernel(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = 42.0f;
}
"#;

    /// Helper: create a GpuDevice, returning None if Metal is unavailable.
    fn try_gpu_device() -> Option<GpuDevice> {
        GpuDevice::system_default().ok()
    }

    #[test]
    fn test_function_constants_pipeline() {
        let device = match try_gpu_device() {
            Some(d) => d,
            None => {
                eprintln!("skipping test_function_constants_pipeline: no Metal device");
                return;
            }
        };

        let registry = KernelRegistry::new(device);

        // Register a shader that uses function_constant(0).
        registry
            .register_jit_source("test_fc", SHADER_WITH_CONSTANT)
            .expect("should compile shader with function constant");

        // 1. Specialized with IS_FORWARD = true (index 0, bool).
        let constants_true = vec![(0u32, FunctionConstantValue::Bool(true))];
        let specialized_true = registry
            .get_pipeline_with_constants("test_fc_kernel", DType::Float32, &constants_true)
            .expect("specialized pipeline (true) should succeed");

        // 2. Specialized with IS_FORWARD = false (index 0, bool).
        let constants_false = vec![(0u32, FunctionConstantValue::Bool(false))];
        let specialized_false = registry
            .get_pipeline_with_constants("test_fc_kernel", DType::Float32, &constants_false)
            .expect("specialized pipeline (false) should succeed");

        // 3. The two specialized pipelines should be distinct objects.
        //    (Different constant values -> different compiled functions.)
        assert_ne!(
            specialized_true.as_ptr(),
            specialized_false.as_ptr(),
            "pipelines with different constants should be distinct"
        );

        // 4. Hitting the cache: second lookup with same constants should return
        //    the cached pipeline (same pointer).
        let cached = registry
            .get_pipeline_with_constants("test_fc_kernel", DType::Float32, &constants_true)
            .expect("cached pipeline lookup should succeed");
        assert_eq!(
            specialized_true.as_ptr(),
            cached.as_ptr(),
            "second lookup should return cached pipeline"
        );

        // 5. Pipeline cache should contain 2 entries (two distinct specializations).
        let count = registry
            .cached_pipeline_count()
            .expect("should read pipeline count");
        assert_eq!(count, 2, "expected 2 cached pipelines");
    }

    #[test]
    fn test_function_constants_empty_delegates_to_plain() {
        let device = match try_gpu_device() {
            Some(d) => d,
            None => {
                eprintln!(
                    "skipping test_function_constants_empty_delegates_to_plain: no Metal device"
                );
                return;
            }
        };

        let registry = KernelRegistry::new(device);
        registry
            .register_jit_source("test_plain", SHADER_PLAIN)
            .expect("should compile plain shader");

        // Empty constants should behave identically to get_pipeline.
        let via_constants = registry
            .get_pipeline_with_constants("test_plain_kernel", DType::Float32, &[])
            .expect("empty-constants pipeline should succeed");

        let via_plain = registry
            .get_pipeline("test_plain_kernel", DType::Float32)
            .expect("plain pipeline should succeed");

        // Both should produce the same cached pipeline since the key is identical.
        assert_eq!(
            via_constants.as_ptr(),
            via_plain.as_ptr(),
            "empty constants should produce the same cached pipeline as plain"
        );
    }

    #[test]
    fn test_function_constants_not_found() {
        let device = match try_gpu_device() {
            Some(d) => d,
            None => {
                eprintln!("skipping test_function_constants_not_found: no Metal device");
                return;
            }
        };

        let registry = KernelRegistry::new(device);

        // Lookup a kernel that does not exist -- should return NotFound.
        let result = registry.get_pipeline_with_constants(
            "nonexistent_kernel",
            DType::Float32,
            &[(0, FunctionConstantValue::Bool(true))],
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::NotFound(msg) => {
                assert!(
                    msg.contains("nonexistent_kernel"),
                    "error should mention kernel name: {msg}"
                );
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_kernel_error_specialization() {
        // 1. Create the error variant.
        let err = KernelError::Specialization("test error".to_string());

        // 2. Verify Display output contains expected substrings.
        let display = format!("{err}");
        assert!(
            display.contains("specialization"),
            "Display should contain 'specialization', got: {display}"
        );
        assert!(
            display.contains("test error"),
            "Display should contain 'test error', got: {display}"
        );

        // 3. Pattern-match the variant to confirm the inner message.
        match err {
            KernelError::Specialization(msg) => {
                assert_eq!(msg, "test error");
            }
            other => panic!("expected Specialization, got {other:?}"),
        }
    }
}
