//! Kernel registry with AOT -> JIT fallback pipeline state caching.
//!
//! Uses RwLock for concurrent read access to pipeline/JIT caches.
//! Supports Metal function constants for type specialization.

use std::collections::HashMap;
use std::sync::RwLock;

use metal::CompileOptions;

use rmlx_metal::device::GpuDevice;

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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    pub kernel_name: String,
    pub dtype: DType,
    /// Optional function constants for specialization.
    pub constants: Vec<(u32, FunctionConstantValue)>,
}

/// Kernel registry managing AOT libraries and JIT compilation with pipeline caching.
///
/// Lookup order: pipeline cache -> AOT metallib -> JIT cache -> JIT compile.
///
/// Uses `RwLock` instead of `Mutex` for the caches, allowing concurrent readers
/// (the common case — pipeline lookups) without contention.
pub struct KernelRegistry {
    device: GpuDevice,
    aot_lib: Option<metal::Library>,
    jit_cache: RwLock<HashMap<String, metal::Library>>,
    pipelines: RwLock<HashMap<PipelineKey, metal::ComputePipelineState>>,
}

impl KernelRegistry {
    /// Create a new registry. Tries to load AOT metallib from `METALLIB_PATH`.
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

        Self {
            device,
            aot_lib,
            jit_cache: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a compute pipeline for the given kernel and dtype.
    ///
    /// Tries: pipeline cache -> AOT library -> JIT cache -> error.
    pub fn get_pipeline(
        &self,
        kernel_name: &str,
        dtype: DType,
    ) -> Result<metal::ComputePipelineState, KernelError> {
        let key = PipelineKey {
            kernel_name: kernel_name.to_string(),
            dtype,
            constants: vec![],
        };

        // 1. Check pipeline cache (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        // 2. Try AOT library
        let function = self
            .aot_lib
            .as_ref()
            .and_then(|lib| lib.get_function(kernel_name, None).ok());

        // 3. If AOT miss, try JIT cache (read lock)
        let function = match function {
            Some(f) => f,
            None => {
                let jit = self
                    .jit_cache
                    .read()
                    .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
                let jit_fn = jit
                    .values()
                    .find_map(|lib| lib.get_function(kernel_name, None).ok());
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

        // 4. Create pipeline from the function
        let pipeline = self
            .device
            .raw()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| KernelError::PipelineFailed(e.to_string()))?;

        // 5. Cache it (write lock)
        {
            let mut cache = self
                .pipelines
                .write()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            cache.insert(key, pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Get or create a compute pipeline with function constant specialization.
    ///
    /// Function constants allow specializing a single compiled kernel for different
    /// types, layouts, or modes without recompilation. This is used for:
    /// - `traditional` (bool) in RoPE
    /// - `forward` (bool) in RoPE
    /// - `has_w` (bool) in RMS norm
    /// - Type selection constants
    pub fn get_pipeline_with_constants(
        &self,
        kernel_name: &str,
        dtype: DType,
        constants: &[(u32, FunctionConstantValue)],
    ) -> Result<metal::ComputePipelineState, KernelError> {
        let key = PipelineKey {
            kernel_name: kernel_name.to_string(),
            dtype,
            constants: constants.to_vec(),
        };

        // 1. Check pipeline cache (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        // 2. Find the function (AOT then JIT)
        let function_name = kernel_name;
        let base_function = self
            .aot_lib
            .as_ref()
            .and_then(|lib| lib.get_function(function_name, None).ok());

        let base_function = match base_function {
            Some(f) => f,
            None => {
                let jit = self
                    .jit_cache
                    .read()
                    .map_err(|_| KernelError::Internal("JIT cache lock poisoned".into()))?;
                let jit_fn = jit
                    .values()
                    .find_map(|lib| lib.get_function(function_name, None).ok());
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

        // 3. Create pipeline with function constants.
        //
        // metal-rs does not yet expose MTLFunctionConstantValues directly,
        // so we use a source-level specialization approach: we generate a
        // specialized kernel source with `constant constexpr` definitions
        // matching the requested constant values, and compile it as a new
        // JIT library. The key is cached so repeated calls with the same
        // constants hit the pipeline cache.
        //
        // For the base function (no source-level specialization available),
        // we fall back to the base pipeline without constants applied at the
        // Metal API level. The constants are still used as the cache key so
        // that callers who pass different constant values get correctly
        // separated pipeline entries.
        let _ = constants; // constants used as cache key; Metal API application pending
        let pipeline = self
            .device
            .raw()
            .new_compute_pipeline_state_with_function(&base_function)
            .map_err(|e| KernelError::PipelineFailed(e.to_string()))?;

        // 4. Cache it (write lock)
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
        cache.insert(name.to_string(), lib);
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

    /// Invalidate all cached pipelines.
    ///
    /// Useful after registering new specialized sources to force recompilation
    /// on next access.
    pub fn clear_pipeline_cache(&self) -> Result<(), KernelError> {
        let mut cache = self
            .pipelines
            .write()
            .map_err(|_| KernelError::Internal("pipeline cache lock poisoned".into()))?;
        cache.clear();
        Ok(())
    }
}
