//! Disk-backed pipeline cache using compiled metallib bytes.
//!
//! Shader compilation (MSL -> metallib) is expensive (~5-50ms).  This module
//! provides [`DiskPipelineCache`] which persists compiled `MTLLibrary` data to
//! `~/.cache/rmlx/pipelines/` so that subsequent launches can skip the
//! compilation step entirely.
//!
//! **Cache key:** SHA-256 of (kernel source + function name + serialized
//! specialization constants).
//!
//! **Cache value:** The metallib binary produced by Metal's runtime compiler,
//! serialized via the Obj-C `[MTLLibrary serializeToURL:error:]` API (macOS 14+).
//! On older macOS versions, serialization is skipped and the library is always
//! compiled from source.
//!
//! This module is intentionally *not* wired into the main compilation flow yet.
//! Integration is planned for a follow-up PR.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;

use metal::{CompileOptions, ComputePipelineState, FunctionConstantValues, Library, MTLDataType};
use sha2::{Digest, Sha256};

use crate::pipeline::FunctionConstant;
use crate::MetalError;

// ---------------------------------------------------------------------------
// DiskPipelineCache
// ---------------------------------------------------------------------------

/// Disk-backed pipeline cache.
///
/// On a cache miss the shader source is compiled, the resulting metallib is
/// serialized to `~/.cache/rmlx/pipelines/<sha256>.metallib`, and the PSO is
/// returned.  On a cache hit the metallib is loaded from disk via
/// `new_library_with_file`, avoiding the MSL compiler entirely.
///
/// An in-memory layer (`HashMap`) avoids redundant disk reads within the same
/// process lifetime.
pub struct DiskPipelineCache {
    device: metal::Device,
    cache_dir: PathBuf,
    /// In-memory cache: cache-key -> (Library, HashMap<function_name, PSO>)
    memory: RwLock<HashMap<String, CacheEntry>>,
}

struct CacheEntry {
    /// Retained so the library (and its functions) stay alive as long as the
    /// PSOs that reference them.
    #[allow(dead_code)]
    library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
}

// SAFETY: `metal::Device`, `Library`, and `ComputePipelineState` are Obj-C
// objects that are internally reference-counted and thread-safe.  The `RwLock`
// provides Rust-side synchronization.
unsafe impl Send for DiskPipelineCache {}
unsafe impl Sync for DiskPipelineCache {}

impl DiskPipelineCache {
    /// Create a new disk pipeline cache for the given device.
    ///
    /// The cache directory defaults to `~/.cache/rmlx/pipelines/`.
    /// It is created lazily on the first cache miss.
    pub fn new(device: &metal::Device) -> Self {
        let cache_dir = default_cache_dir();
        Self {
            device: device.clone(),
            cache_dir,
            memory: RwLock::new(HashMap::new()),
        }
    }

    /// Create a disk pipeline cache with a custom cache directory.
    ///
    /// Useful for testing or non-standard deployments.
    pub fn with_cache_dir(device: &metal::Device, cache_dir: PathBuf) -> Self {
        Self {
            device: device.clone(),
            cache_dir,
            memory: RwLock::new(HashMap::new()),
        }
    }

    /// Return the cache directory path.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }

    /// Get or compile a compute pipeline state.
    ///
    /// 1. Compute a SHA-256 cache key from `(source, function_name, constants)`.
    /// 2. Check in-memory cache.
    /// 3. Check disk cache (`<cache_dir>/<key>.metallib`).
    /// 4. On full miss, compile from source, persist metallib to disk.
    /// 5. Create PSO from the library and return it.
    pub fn get_or_compile(
        &self,
        source: &str,
        function_name: &str,
        constants: &[(u32, FunctionConstant)],
    ) -> Result<ComputePipelineState, MetalError> {
        let cache_key = compute_cache_key(source, function_name, constants);

        // --- In-memory fast path (read lock) ---
        {
            let mem = self.memory.read().map_err(|_| {
                MetalError::PipelineCreate("disk pipeline cache lock poisoned".to_string())
            })?;
            if let Some(entry) = mem.get(&cache_key) {
                if let Some(pso) = entry.pipelines.get(function_name) {
                    return Ok(pso.clone());
                }
            }
        }

        // --- Disk / compile path (write lock) ---
        let mut mem = self.memory.write().map_err(|_| {
            MetalError::PipelineCreate("disk pipeline cache lock poisoned".to_string())
        })?;

        // Double-check after acquiring write lock.
        if let Some(entry) = mem.get(&cache_key) {
            if let Some(pso) = entry.pipelines.get(function_name) {
                return Ok(pso.clone());
            }
        }

        let metallib_path = self.cache_dir.join(format!("{cache_key}.metallib"));

        // Try loading from disk first.
        let library = if metallib_path.exists() {
            self.load_library_from_disk(&metallib_path)?
        } else {
            // Full miss: compile and persist.
            let lib = self.compile_source(source)?;
            // Best-effort write to disk; don't fail the pipeline on I/O errors.
            let _ = self.persist_library_to_disk(&lib, &metallib_path);
            lib
        };

        // Create PSO from the library.
        let pso = self.create_pso(&library, function_name, constants)?;

        let cloned = pso.clone();
        let entry = mem.entry(cache_key).or_insert_with(|| CacheEntry {
            library,
            pipelines: HashMap::new(),
        });
        entry.pipelines.insert(function_name.to_string(), pso);

        Ok(cloned)
    }

    /// Number of in-memory cache entries.
    pub fn len(&self) -> usize {
        self.memory.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the in-memory cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear in-memory cache (does not remove disk files).
    pub fn clear_memory(&self) {
        if let Ok(mut mem) = self.memory.write() {
            mem.clear();
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn compile_source(&self, source: &str) -> Result<Library, MetalError> {
        let options = CompileOptions::new();
        self.device
            .new_library_with_source(source, &options)
            .map_err(|e| MetalError::ShaderCompile(e.to_string()))
    }

    fn load_library_from_disk(&self, path: &std::path::Path) -> Result<Library, MetalError> {
        self.device
            .new_library_with_file(path)
            .map_err(|e| MetalError::LibraryLoad(format!("failed to load cached metallib: {e}")))
    }

    /// Serialize a compiled library to disk using Obj-C
    /// `[MTLLibrary serializeToURL:error:]` (macOS 14+).
    ///
    /// Returns `Ok(())` on success or if the API is not available (best-effort).
    fn persist_library_to_disk(
        &self,
        library: &Library,
        path: &std::path::Path,
    ) -> Result<(), MetalError> {
        // Ensure cache directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| MetalError::LibraryLoad(format!("failed to create cache dir: {e}")))?;
        }

        // Call [MTLLibrary serializeToURL:error:] via objc::msg_send!
        // This API is available on macOS 14.0+ / iOS 17.0+.
        // We check at runtime via respondsToSelector:.
        let path_str = path.to_string_lossy();
        let url_string = format!("file://{path_str}");

        unsafe {
            // Create NSString for the URL string.
            let c_url_string = std::ffi::CString::new(url_string.as_str())
                .map_err(|e| MetalError::LibraryLoad(format!("invalid URL string: {e}")))?;
            let nsstring_class = objc::runtime::Class::get("NSString")
                .ok_or_else(|| MetalError::LibraryLoad("NSString class not found".to_string()))?;
            let ns_url_string: *mut objc::runtime::Object = msg_send![
                nsstring_class,
                stringWithUTF8String: c_url_string.as_ptr()
            ];

            // Create NSURL from the string.
            let nsurl_class = objc::runtime::Class::get("NSURL")
                .ok_or_else(|| MetalError::LibraryLoad("NSURL class not found".to_string()))?;
            let url: *mut objc::runtime::Object =
                msg_send![nsurl_class, URLWithString: ns_url_string];
            if url.is_null() {
                return Err(MetalError::LibraryLoad(format!(
                    "failed to create NSURL from: {url_string}"
                )));
            }

            // Get a reference to LibraryRef (which implements Message).
            let lib_ref: &metal::LibraryRef = library;

            // Check if the library responds to serializeToURL:error:
            let sel = objc::runtime::Sel::register("serializeToURL:error:");
            let responds: bool = msg_send![lib_ref, respondsToSelector: sel];
            if !responds {
                // API not available on this macOS version — skip silently.
                return Ok(());
            }

            let mut err: *mut objc::runtime::Object = std::ptr::null_mut();
            let _result: bool = msg_send![lib_ref, serializeToURL: url error: &mut err];

            if !err.is_null() {
                let desc: *mut objc::runtime::Object = msg_send![err, localizedDescription];
                let c_msg: *const std::os::raw::c_char = msg_send![desc, UTF8String];
                let message = std::ffi::CStr::from_ptr(c_msg)
                    .to_string_lossy()
                    .into_owned();
                return Err(MetalError::LibraryLoad(format!(
                    "serializeToURL failed: {message}"
                )));
            }
        }

        Ok(())
    }

    fn create_pso(
        &self,
        library: &Library,
        function_name: &str,
        constants: &[(u32, FunctionConstant)],
    ) -> Result<ComputePipelineState, MetalError> {
        let function = if constants.is_empty() {
            library
                .get_function(function_name, None)
                .map_err(|_| MetalError::KernelNotFound(function_name.to_string()))?
        } else {
            let fcv = FunctionConstantValues::new();
            for (index, constant) in constants {
                match constant {
                    FunctionConstant::Bool(v) => {
                        let val: u8 = u8::from(*v);
                        fcv.set_constant_value_at_index(
                            &val as *const u8 as *const std::ffi::c_void,
                            MTLDataType::Bool,
                            *index as u64,
                        );
                    }
                    FunctionConstant::U32(v) => {
                        fcv.set_constant_value_at_index(
                            v as *const u32 as *const std::ffi::c_void,
                            MTLDataType::UInt,
                            *index as u64,
                        );
                    }
                    FunctionConstant::F32(v) => {
                        fcv.set_constant_value_at_index(
                            v as *const f32 as *const std::ffi::c_void,
                            MTLDataType::Float,
                            *index as u64,
                        );
                    }
                }
            }
            library
                .get_function(function_name, Some(fcv))
                .map_err(|_| MetalError::KernelNotFound(function_name.to_string()))?
        };

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreate(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Default cache directory: `~/.cache/rmlx/pipelines/`
fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| {
            // Fallback: use $HOME/.cache
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
                .join(".cache")
        })
        .join("rmlx")
        .join("pipelines")
}

/// Compute a stable SHA-256 hex key from source, function name, and constants.
fn compute_cache_key(
    source: &str,
    function_name: &str,
    constants: &[(u32, FunctionConstant)],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\x00");
    hasher.update(function_name.as_bytes());
    hasher.update(b"\x00");

    for (index, constant) in constants {
        hasher.update(index.to_le_bytes());
        match constant {
            FunctionConstant::Bool(v) => {
                hasher.update([0u8]); // discriminant
                hasher.update([u8::from(*v)]);
            }
            FunctionConstant::U32(v) => {
                hasher.update([1u8]);
                hasher.update(v.to_le_bytes());
            }
            FunctionConstant::F32(v) => {
                hasher.update([2u8]);
                hasher.update(v.to_le_bytes());
            }
        }
    }

    let hash = hasher.finalize();
    // Hex-encode the hash.
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KERNEL_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_add(device float* out [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            out[id] = float(id) + 1.0;
        }
    "#;

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = compute_cache_key("src", "fn", &[]);
        let k2 = compute_cache_key("src", "fn", &[]);
        assert_eq!(k1, k2);
        // SHA-256 hex = 64 chars
        assert_eq!(k1.len(), 64);
    }

    #[test]
    fn test_cache_key_varies_with_source() {
        let k1 = compute_cache_key("src_a", "fn", &[]);
        let k2 = compute_cache_key("src_b", "fn", &[]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_function_name() {
        let k1 = compute_cache_key("src", "fn_a", &[]);
        let k2 = compute_cache_key("src", "fn_b", &[]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_constants() {
        let k1 = compute_cache_key("src", "fn", &[(0, FunctionConstant::Bool(true))]);
        let k2 = compute_cache_key("src", "fn", &[(0, FunctionConstant::Bool(false))]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_default_cache_dir_exists_or_creatable() {
        let dir = default_cache_dir();
        // Should be something like /Users/<user>/Library/Caches/rmlx/pipelines
        // or /Users/<user>/.cache/rmlx/pipelines
        assert!(dir.to_str().unwrap().contains("rmlx"));
        assert!(dir.to_str().unwrap().contains("pipelines"));
    }

    #[test]
    fn test_new_cache_is_empty() {
        let device = metal::Device::system_default().unwrap();
        let cache = DiskPipelineCache::new(&device);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_dir_creation() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_test_{}", std::process::id()));
        let cache_dir = tmp.join("rmlx_test_pipelines");

        // Directory should not exist yet.
        assert!(!cache_dir.exists());

        let cache = DiskPipelineCache::with_cache_dir(&device, cache_dir.clone());

        // Compile a kernel — this should create the cache directory.
        let result = cache.get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[]);
        assert!(result.is_ok());
        assert!(cache_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_cold_miss_compiles_and_stores() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_cold_{}", std::process::id()));
        let cache_dir = tmp.join("pipelines");
        let cache = DiskPipelineCache::with_cache_dir(&device, cache_dir.clone());

        // Cold miss: compile and persist.
        let pso = cache
            .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
            .unwrap();
        assert_eq!(cache.len(), 1);

        // Check if metallib was persisted (may not be on macOS < 14).
        let key = compute_cache_key(TEST_KERNEL_SOURCE, "test_add", &[]);
        let metallib_path = cache_dir.join(format!("{key}.metallib"));
        // On macOS 14+ this should exist; on older versions it won't.
        // We only assert the PSO is usable.
        if metallib_path.exists() {
            // Verify the file has non-zero size.
            let meta = fs::metadata(&metallib_path).unwrap();
            assert!(meta.len() > 0, "metallib should be non-empty");
        }

        // PSO should be usable (non-null max total threads).
        assert!(pso.max_total_threads_per_threadgroup() > 0);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_warm_hit_loads_from_disk() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_warm_{}", std::process::id()));
        let cache_dir = tmp.join("pipelines");

        // First cache: cold miss, writes to disk.
        let disk_file_exists;
        {
            let cache = DiskPipelineCache::with_cache_dir(&device, cache_dir.clone());
            cache
                .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
                .unwrap();
            let key = compute_cache_key(TEST_KERNEL_SOURCE, "test_add", &[]);
            disk_file_exists = cache_dir.join(format!("{key}.metallib")).exists();
        }

        if !disk_file_exists {
            // serializeToURL not available — skip the warm-hit portion.
            let _ = fs::remove_dir_all(&tmp);
            return;
        }

        // Second cache (new instance, empty in-memory): should load from disk.
        {
            let cache = DiskPipelineCache::with_cache_dir(&device, cache_dir.clone());
            assert!(cache.is_empty(), "in-memory cache should be empty");

            let pso = cache
                .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
                .unwrap();
            assert!(pso.max_total_threads_per_threadgroup() > 0);
            assert_eq!(cache.len(), 1);
        }

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_in_memory_hit() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_mem_{}", std::process::id()));
        let cache_dir = tmp.join("pipelines");
        let cache = DiskPipelineCache::with_cache_dir(&device, cache_dir);

        // First call: cold miss.
        cache
            .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
            .unwrap();

        // Second call: in-memory hit (no disk I/O needed).
        let pso = cache
            .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
            .unwrap();
        assert!(pso.max_total_threads_per_threadgroup() > 0);
        assert_eq!(cache.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_clear_memory() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_clr_{}", std::process::id()));
        let cache = DiskPipelineCache::with_cache_dir(&device, tmp.join("p"));

        cache
            .get_or_compile(TEST_KERNEL_SOURCE, "test_add", &[])
            .unwrap();
        assert_eq!(cache.len(), 1);

        cache.clear_memory();
        assert!(cache.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_invalid_source_returns_error() {
        let device = metal::Device::system_default().unwrap();
        let tmp = std::env::temp_dir().join(format!("rmlx_inv_{}", std::process::id()));
        let cache = DiskPipelineCache::with_cache_dir(&device, tmp.join("p"));

        let result = cache.get_or_compile("not valid MSL", "nope", &[]);
        assert!(result.is_err());
        assert!(cache.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }
}
