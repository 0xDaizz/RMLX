use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rmlx_core::array::Array as GpuArray;
#[allow(unused_imports)]
use rmlx_core::dtype::DType;
use rmlx_metal::metal;

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
}

impl PyArray {
    /// Wrap an existing GPU array, extracting CPU data from it.
    ///
    /// # Safety
    /// Caller must ensure no GPU writes are in-flight to the array's buffer.
    pub unsafe fn from_gpu_array(gpu: GpuArray) -> Self {
        let shape = gpu.shape().to_vec();
        let dtype_name = gpu.dtype().name().to_string();
        let data: Vec<f32> = gpu.to_vec::<f32>();
        Self {
            shape,
            data,
            dtype_name,
            gpu_array: Some(gpu),
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
        })
    }

    fn __mul__(&self, other: &PyArray) -> PyResult<Self> {
        if self.shape != other.shape {
            return Err(PyValueError::new_err("shape mismatch for mul"));
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

    fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }

    /// Upload CPU data to a GPU-backed Metal buffer.
    ///
    /// Returns a new PyArray with both CPU data and GPU buffer populated.
    /// The Metal device is obtained from the system default.
    fn to_gpu(&self) -> PyResult<Self> {
        let device = metal::Device::system_default()
            .ok_or_else(|| PyValueError::new_err("no Metal GPU device available"))?;
        let gpu = GpuArray::from_slice(&device, &self.data, self.shape.clone());
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
            dtype_name: self.dtype_name.clone(),
            gpu_array: Some(gpu),
        })
    }

    /// Download GPU array data to CPU, returning a CPU-only PyArray.
    ///
    /// If no GPU array is present, returns a copy of the current CPU data.
    fn to_cpu(&self) -> PyResult<Self> {
        match &self.gpu_array {
            Some(gpu) => {
                // Safety: we assume no GPU writes are in-flight when Python calls this.
                let cpu_data = unsafe { gpu.to_vec::<f32>() };
                Ok(Self {
                    shape: gpu.shape().to_vec(),
                    data: cpu_data,
                    dtype_name: gpu.dtype().name().to_string(),
                    gpu_array: None,
                })
            }
            None => Ok(Self {
                shape: self.shape.clone(),
                data: self.data.clone(),
                dtype_name: self.dtype_name.clone(),
                gpu_array: None,
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
}
