//! Kernel registry with AOT -> JIT fallback pipeline state caching.

use std::collections::HashMap;
use std::sync::Mutex;

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
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::NotFound(name) => write!(f, "kernel not found: {name}"),
            KernelError::CompilationFailed(e) => write!(f, "shader compilation failed: {e}"),
            KernelError::PipelineFailed(e) => write!(f, "pipeline creation failed: {e}"),
            KernelError::InvalidShape(e) => write!(f, "invalid shape: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Cache key for compiled pipelines.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    pub kernel_name: String,
    pub dtype: DType,
}

/// Kernel registry managing AOT libraries and JIT compilation with pipeline caching.
///
/// Lookup order: pipeline cache -> AOT metallib -> JIT cache -> JIT compile.
pub struct KernelRegistry {
    device: GpuDevice,
    aot_lib: Option<metal::Library>,
    jit_cache: Mutex<HashMap<String, metal::Library>>,
    pipelines: Mutex<HashMap<PipelineKey, metal::ComputePipelineState>>,
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
            jit_cache: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
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
        };

        // 1. Check pipeline cache
        {
            let cache = self.pipelines.lock().unwrap();
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        // 2. Try AOT library
        let function = self
            .aot_lib
            .as_ref()
            .and_then(|lib| lib.get_function(kernel_name, None).ok());

        // 3. If AOT miss, try JIT cache
        let function = match function {
            Some(f) => f,
            None => {
                let jit = self.jit_cache.lock().unwrap();
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

        // 5. Cache it
        {
            let mut cache = self.pipelines.lock().unwrap();
            cache.insert(key, pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Register a JIT-compiled shader source under the given name.
    pub fn register_jit_source(&self, name: &str, source: &str) -> Result<(), KernelError> {
        let options = CompileOptions::new();
        let lib = self
            .device
            .raw()
            .new_library_with_source(source, &options)
            .map_err(|e| KernelError::CompilationFailed(format!("{name}: {e}")))?;

        let mut cache = self.jit_cache.lock().unwrap();
        cache.insert(name.to_string(), lib);
        Ok(())
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
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    /// Number of JIT-compiled libraries.
    pub fn jit_library_count(&self) -> usize {
        self.jit_cache.lock().unwrap().len()
    }
}
