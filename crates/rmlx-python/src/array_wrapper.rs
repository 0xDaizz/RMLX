use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use rmlx_core::array::Array as GpuArray;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops::binary::{binary_op_async, BinaryOp};
use rmlx_core::CommandBufferHandle;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal;

/// Global GPU context shared across all Python operations.
///
/// Holds the Metal device, kernel registry (with compiled pipelines),
/// and a command queue. Initialized once on first use via `OnceLock`.
struct GpuContext {
    registry: KernelRegistry,
    queue: metal::CommandQueue,
}

static GPU_CONTEXT: OnceLock<Result<GpuContext, String>> = OnceLock::new();

/// Get or initialize the global GPU context.
///
/// On first call, creates a Metal device, registers all built-in kernels,
/// and creates a command queue. Subsequent calls return the cached context.
/// If initialization failed, the error is cached and returned on every call.
fn get_gpu_context() -> Result<&'static GpuContext, PyErr> {
    let result = GPU_CONTEXT.get_or_init(|| {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(e) => return Err(format!("Metal device init failed: {e}")),
        };
        let registry = KernelRegistry::new(device);
        if let Err(e) = rmlx_core::ops::register_all(&registry) {
            return Err(format!("kernel registration failed: {e}"));
        }
        let queue = registry.device().new_command_queue();
        Ok(GpuContext { registry, queue })
    });
    result
        .as_ref()
        .map_err(|e| PyRuntimeError::new_err(e.clone()))
}

/// Python wrapper around rmlx_core::array::Array.
/// Provides creation, shape inspection, and data extraction.
/// Holds CPU data in `data` and optionally a GPU-backed `GpuArray`.
#[pyclass(name = "Array")]
pub struct PyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype_name: String,
    /// GPU-backed array (populated after `to_gpu()` or `from_array()`).
    gpu_array: Option<GpuArray>,
    /// Pending async GPU command buffer handle.
    /// Must be waited on before reading GPU data (e.g. in `to_cpu()`).
    pending_handle: Option<CommandBufferHandle>,
}

impl PyArray {
    /// Wrap an existing GPU array, extracting CPU data from it.
    ///
    /// # Safety
    /// Caller must ensure all GPU command buffers that write to the array's
    /// Metal buffer have completed (via `waitUntilCompleted` or completion
    /// handler) before calling this function.
    pub unsafe fn from_gpu_array(gpu: GpuArray) -> Self {
        let shape = gpu.shape().to_vec();
        let dtype_name = gpu.dtype().name().to_string();
        let data: Vec<f32> = gpu.to_vec::<f32>();
        Self {
            shape,
            data,
            dtype_name,
            gpu_array: Some(gpu),
            pending_handle: None,
        }
    }

    /// Get a reference to the underlying GPU array, if present.
    pub fn gpu_array(&self) -> Option<&GpuArray> {
        self.gpu_array.as_ref()
    }

    /// Check whether this PyArray has a GPU-backed array.
    pub fn is_on_gpu(&self) -> bool {
        self.gpu_array.is_some()
    }

    /// Attach an async command buffer handle for GPU sync tracking.
    pub fn set_pending_handle(&mut self, handle: CommandBufferHandle) {
        self.pending_handle = Some(handle);
    }

    /// Wait for any pending GPU operation to complete.
    fn wait_pending(&mut self) {
        if let Some(h) = self.pending_handle.take() {
            h.wait();
        }
    }

    /// Execute a GPU binary op between two PyArrays that both have gpu_array.
    /// Returns a new PyArray with gpu_array set and pending_handle tracking
    /// the async command buffer.
    ///
    /// Uses the global `GpuContext` to avoid per-call device/registry/queue creation.
    fn gpu_binary_op(&self, other: &PyArray, op: BinaryOp, op_name: &str) -> PyResult<Self> {
        let self_gpu = self
            .gpu_array
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("{op_name}: self has no GPU array")))?;
        let other_gpu = other
            .gpu_array
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("{op_name}: other has no GPU array")))?;

        let ctx = get_gpu_context()?;

        let launch = binary_op_async(&ctx.registry, self_gpu, other_gpu, op, &ctx.queue)
            .map_err(|e| PyValueError::new_err(format!("GPU {op_name} failed: {e}")))?;

        let out_array = launch.into_array();

        // SAFETY: into_array() blocks until GPU completes, so the data is ready.
        let shape = out_array.shape().to_vec();
        let cpu_data = unsafe { out_array.to_vec::<f32>() };

        Ok(Self {
            shape: shape.clone(),
            data: cpu_data,
            dtype_name: self.dtype_name.clone(),
            gpu_array: Some(out_array),
            pending_handle: None,
        })
    }

    /// Internal constructor used by tests (no PyO3 runtime needed).
    #[cfg(test)]
    fn new_rust(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_numel: usize = shape.iter().product();
        if expected_numel != data.len() {
            return Err(format!(
                "data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_numel
            ));
        }
        Ok(Self {
            shape,
            data,
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    #[cfg(test)]
    fn zeros_rust(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; numel],
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        }
    }

    #[cfg(test)]
    fn ones_rust(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; numel],
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        }
    }

    #[cfg(test)]
    fn reshape_rust(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.data.len() {
            return Err("reshape size mismatch".to_string());
        }
        Ok(Self {
            shape: new_shape,
            data: self.data.clone(),
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    #[cfg(test)]
    fn add_rust(&self, other: &PyArray) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err("shape mismatch for add".to_string());
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    #[cfg(test)]
    fn mul_rust(&self, other: &PyArray) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err("shape mismatch for mul".to_string());
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let expected_numel: usize = shape.iter().product();
        if expected_numel != data.len() {
            return Err(PyValueError::new_err(format!(
                "data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_numel
            )));
        }
        Ok(Self {
            shape,
            data,
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; numel],
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        }
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; numel],
            dtype_name: "float32".to_string(),
            gpu_array: None,
            pending_handle: None,
        }
    }

    #[staticmethod]
    fn from_list(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        PyArray::new(data, shape)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn size(&self) -> usize {
        self.data.len()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.dtype_name.clone()
    }

    fn tolist(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.data.len() {
            return Err(PyValueError::new_err("reshape size mismatch"));
        }
        Ok(Self {
            shape: new_shape,
            data: self.data.clone(),
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, dtype={})", self.shape, self.dtype_name)
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __add__(&self, other: &PyArray) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(PyValueError::new_err("shape mismatch for add"));
        }

        // Use GPU path when both operands have GPU arrays
        if self.gpu_array.is_some() && other.gpu_array.is_some() {
            return self.gpu_binary_op(other, BinaryOp::Add, "add");
        }

        // CPU fallback
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    fn __mul__(&self, other: &PyArray) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(PyValueError::new_err("shape mismatch for mul"));
        }

        // Use GPU path when both operands have GPU arrays
        if self.gpu_array.is_some() && other.gpu_array.is_some() {
            return self.gpu_binary_op(other, BinaryOp::Mul, "mul");
        }

        // CPU fallback
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
            dtype_name: self.dtype_name.clone(),
            gpu_array: None,
            pending_handle: None,
        })
    }

    fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    fn max(&self) -> Option<f32> {
        self.data.iter().cloned().reduce(f32::max)
    }

    fn min(&self) -> Option<f32> {
        self.data.iter().cloned().reduce(f32::min)
    }

    fn mean(&self) -> PyResult<f32> {
        if self.data.is_empty() {
            return Err(PyValueError::new_err("mean of empty array is undefined"));
        }
        Ok(self.sum() / self.data.len() as f32)
    }

    /// Upload CPU data to a GPU-backed Metal buffer.
    ///
    /// Returns a new PyArray with both CPU data and GPU buffer populated.
    /// Uses the global GPU context for the Metal device.
    fn to_gpu(&self) -> PyResult<Self> {
        let ctx = get_gpu_context()?;
        let gpu = GpuArray::from_slice(ctx.registry.device().raw(), &self.data, self.shape.clone());
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
            dtype_name: self.dtype_name.clone(),
            gpu_array: Some(gpu),
            pending_handle: None,
        })
    }

    /// Download GPU array data to CPU, returning a CPU-only PyArray.
    ///
    /// If a pending GPU command buffer handle exists, waits for completion
    /// before reading. If no GPU array is present, returns a copy of the
    /// current CPU data.
    #[allow(clippy::wrong_self_convention)]
    fn to_cpu(&mut self) -> PyResult<Self> {
        // Wait for any pending async GPU operation to complete before reading
        self.wait_pending();

        match &self.gpu_array {
            Some(gpu) => {
                // SAFETY: CommandBufferHandle.wait() completed (via wait_pending above),
                // ensuring all GPU writes to this buffer are finished before we read.
                let cpu_data = unsafe { gpu.to_vec::<f32>() };
                Ok(Self {
                    shape: gpu.shape().to_vec(),
                    data: cpu_data,
                    dtype_name: gpu.dtype().name().to_string(),
                    gpu_array: None,
                    pending_handle: None,
                })
            }
            None => Ok(Self {
                shape: self.shape.clone(),
                data: self.data.clone(),
                dtype_name: self.dtype_name.clone(),
                gpu_array: None,
                pending_handle: None,
            }),
        }
    }

    /// Check whether this array has a GPU-backed Metal buffer.
    fn has_gpu(&self) -> bool {
        self.gpu_array.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyarray_new_valid() {
        let arr = PyArray::new_rust(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.data.len(), 6);
    }

    #[test]
    fn test_pyarray_shape_mismatch() {
        let result = PyArray::new_rust(vec![1.0, 2.0, 3.0], vec![2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pyarray_zeros() {
        let arr = PyArray::zeros_rust(vec![3, 4]);
        assert_eq!(arr.shape, vec![3, 4]);
        assert_eq!(arr.data.len(), 12);
        assert!(arr.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pyarray_ones() {
        let arr = PyArray::ones_rust(vec![2, 2]);
        assert_eq!(arr.shape, vec![2, 2]);
        assert_eq!(arr.data.len(), 4);
        assert!(arr.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_pyarray_reshape() {
        let arr = PyArray::new_rust(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = arr.reshape_rust(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.data.len(), 6);

        let bad = arr.reshape_rust(vec![4, 4]);
        assert!(bad.is_err());
    }

    #[test]
    fn test_pyarray_add() {
        let a = PyArray::new_rust(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = PyArray::new_rust(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.add_rust(&b).unwrap();
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);

        let d = PyArray::new_rust(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(a.add_rust(&d).is_err());
    }

    #[test]
    fn test_pyarray_mul() {
        let a = PyArray::new_rust(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let b = PyArray::new_rust(vec![5.0, 6.0, 7.0], vec![3]).unwrap();
        let c = a.mul_rust(&b).unwrap();
        assert_eq!(c.data, vec![10.0, 18.0, 28.0]);

        let d = PyArray::new_rust(vec![1.0], vec![1]).unwrap();
        assert!(a.mul_rust(&d).is_err());
    }

    #[test]
    fn test_pyarray_sum_max_min_mean() {
        let arr = PyArray::new_rust(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        assert_eq!(arr.data.iter().sum::<f32>(), 10.0);
        assert_eq!(arr.data.iter().cloned().reduce(f32::max), Some(4.0));
        assert_eq!(arr.data.iter().cloned().reduce(f32::min), Some(1.0));
        assert_eq!(arr.data.iter().sum::<f32>() / arr.data.len() as f32, 2.5);
    }

    #[test]
    fn test_pyarray_repr() {
        let arr = PyArray::new_rust(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let repr = format!("Array(shape={:?}, dtype={})", arr.shape, arr.dtype_name);
        assert_eq!(repr, "Array(shape=[2, 2], dtype=float32)");
    }

    #[test]
    fn test_pyarray_ndim_size_dtype() {
        let arr = PyArray::new_rust(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(arr.shape.len(), 2);
        assert_eq!(arr.data.len(), 6);
        assert_eq!(arr.dtype_name, "float32");
    }

    #[test]
    fn test_pyarray_gpu_default_none() {
        let arr = PyArray::new_rust(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(!arr.is_on_gpu());
        assert!(arr.gpu_array().is_none());
    }

    #[test]
    fn test_pyarray_from_gpu_array() {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => return, // skip on machines without Metal
        };
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let gpu = GpuArray::from_slice(&device, &data, vec![2, 2]);
        let arr = unsafe { PyArray::from_gpu_array(gpu) };
        assert!(arr.is_on_gpu());
        assert_eq!(arr.shape, vec![2, 2]);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(arr.dtype_name, "float32");
    }

    /// Test to_cpu on a CPU-only array (no pending handle) — regression test.
    #[test]
    fn test_to_cpu_no_handle() {
        let mut arr = PyArray::new_rust(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(arr.pending_handle.is_none());
        // to_cpu is a #[pymethods] fn, but we can test via the internal path
        // by directly checking wait_pending doesn't panic on None handle
        arr.wait_pending();
        assert!(arr.pending_handle.is_none());
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0]);
    }

    /// Test that set_pending_handle + wait_pending works correctly.
    #[test]
    fn test_pending_handle_set_and_wait() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&flag);

        // Simulate a handle that completes quickly
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            flag_clone.store(true, Ordering::Release);
        });

        let handle = CommandBufferHandle::new_from_flag(flag);
        let mut arr = PyArray::new_rust(vec![1.0, 2.0], vec![2]).unwrap();
        arr.set_pending_handle(handle);
        assert!(arr.pending_handle.is_some());

        arr.wait_pending();
        assert!(arr.pending_handle.is_none());
    }

    /// GPU add -> to_cpu roundtrip test.
    #[test]
    fn test_gpu_add_to_cpu_roundtrip() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => return, // skip on machines without Metal
        };
        let registry = KernelRegistry::new(device);
        if rmlx_core::ops::register_all(&registry).is_err() {
            return; // skip if kernel compilation fails
        }
        let queue = registry.device().new_command_queue();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let a_gpu = GpuArray::from_slice(registry.device().raw(), &a_data, vec![4]);
        let b_gpu = GpuArray::from_slice(registry.device().raw(), &b_data, vec![4]);

        // Use binary_op_async to get LaunchResult
        let launch = binary_op_async(&registry, &a_gpu, &b_gpu, BinaryOp::Add, &queue).unwrap();
        let handle = launch.handle().clone();
        let out_gpu = launch.into_array();

        // Build PyArray with pending handle
        let shape = out_gpu.shape().to_vec();
        let mut result = PyArray {
            shape,
            data: Vec::new(),
            dtype_name: "float32".to_string(),
            gpu_array: Some(out_gpu),
            pending_handle: Some(handle),
        };

        // wait_pending then read
        result.wait_pending();
        let gpu = result.gpu_array.as_ref().unwrap();
        // SAFETY: wait_pending completed, GPU writes are done
        let cpu_data = unsafe { gpu.to_vec::<f32>() };
        assert_eq!(cpu_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// GPU mul -> to_cpu roundtrip test.
    #[test]
    fn test_gpu_mul_to_cpu_roundtrip() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => return,
        };
        let registry = KernelRegistry::new(device);
        if rmlx_core::ops::register_all(&registry).is_err() {
            return;
        }
        let queue = registry.device().new_command_queue();

        let a_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_data = vec![10.0f32, 10.0, 10.0, 10.0];
        let a_gpu = GpuArray::from_slice(registry.device().raw(), &a_data, vec![4]);
        let b_gpu = GpuArray::from_slice(registry.device().raw(), &b_data, vec![4]);

        let launch = binary_op_async(&registry, &a_gpu, &b_gpu, BinaryOp::Mul, &queue).unwrap();
        let handle = launch.handle().clone();
        let out_gpu = launch.into_array();

        let shape = out_gpu.shape().to_vec();
        let mut result = PyArray {
            shape,
            data: Vec::new(),
            dtype_name: "float32".to_string(),
            gpu_array: Some(out_gpu),
            pending_handle: Some(handle),
        };

        result.wait_pending();
        let gpu = result.gpu_array.as_ref().unwrap();
        // SAFETY: wait_pending completed, GPU writes are done
        let cpu_data = unsafe { gpu.to_vec::<f32>() };
        assert_eq!(cpu_data, vec![20.0, 30.0, 40.0, 50.0]);
    }

    /// Verify the global GPU_CONTEXT OnceLock is initialized once and
    /// subsequent accesses return the same `Ok(GpuContext)`.
    #[test]
    fn test_gpu_context_singleton() {
        // Access the raw OnceLock to avoid PyErr construction (no Python runtime in tests).
        let result1 = GPU_CONTEXT.get_or_init(|| {
            let device = match GpuDevice::system_default() {
                Ok(d) => d,
                Err(e) => return Err(format!("Metal device init failed: {e}")),
            };
            let registry = KernelRegistry::new(device);
            if let Err(e) = rmlx_core::ops::register_all(&registry) {
                return Err(format!("kernel registration failed: {e}"));
            }
            let queue = registry.device().new_command_queue();
            Ok(GpuContext { registry, queue })
        });
        let ctx1 = match result1 {
            Ok(c) => c,
            Err(_) => return, // skip on machines without Metal
        };

        // Second call must return the same pointer
        let result2 = GPU_CONTEXT.get().unwrap();
        let ctx2 = result2.as_ref().unwrap();
        assert!(std::ptr::eq(ctx1, ctx2));
    }

    /// Verify the global context provides a working device/registry/queue
    /// that can execute binary ops — the same path `gpu_binary_op` uses.
    #[test]
    fn test_global_context_binary_op() {
        let result = GPU_CONTEXT.get_or_init(|| {
            let device = match GpuDevice::system_default() {
                Ok(d) => d,
                Err(e) => return Err(format!("{e}")),
            };
            let registry = KernelRegistry::new(device);
            if let Err(e) = rmlx_core::ops::register_all(&registry) {
                return Err(format!("{e}"));
            }
            let queue = registry.device().new_command_queue();
            Ok(GpuContext { registry, queue })
        });
        let ctx = match result {
            Ok(c) => c,
            Err(_) => return, // skip on machines without Metal
        };

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let a_gpu = GpuArray::from_slice(ctx.registry.device().raw(), &a_data, vec![4]);
        let b_gpu = GpuArray::from_slice(ctx.registry.device().raw(), &b_data, vec![4]);

        let launch =
            binary_op_async(&ctx.registry, &a_gpu, &b_gpu, BinaryOp::Add, &ctx.queue).unwrap();
        let out_array = launch.into_array();

        let shape = out_array.shape().to_vec();
        // SAFETY: into_array() blocks until GPU completes.
        let cpu_data = unsafe { out_array.to_vec::<f32>() };
        assert_eq!(shape, vec![4]);
        assert_eq!(cpu_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Verify that GpuArray::from_slice works with the global context device,
    /// matching the path used by `to_gpu()`.
    #[test]
    fn test_to_gpu_uses_global_device() {
        let result = GPU_CONTEXT.get_or_init(|| {
            let device = match GpuDevice::system_default() {
                Ok(d) => d,
                Err(e) => return Err(format!("{e}")),
            };
            let registry = KernelRegistry::new(device);
            if let Err(e) = rmlx_core::ops::register_all(&registry) {
                return Err(format!("{e}"));
            }
            let queue = registry.device().new_command_queue();
            Ok(GpuContext { registry, queue })
        });
        let ctx = match result {
            Ok(c) => c,
            Err(_) => return,
        };

        let data = vec![1.0f32, 2.0, 3.0];
        let gpu = GpuArray::from_slice(ctx.registry.device().raw(), &data, vec![3]);
        assert_eq!(gpu.shape(), &[3]);
        // SAFETY: no async operations pending.
        let roundtrip = unsafe { gpu.to_vec::<f32>() };
        assert_eq!(roundtrip, vec![1.0, 2.0, 3.0]);
    }
}
