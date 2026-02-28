use pyo3::prelude::*;
use rmlx_core::dtype::DType;

#[pyclass(name = "DType", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyDType {
    pub(crate) inner: DType,
}

#[pymethods]
impl PyDType {
    #[staticmethod]
    fn float32() -> Self {
        Self {
            inner: DType::Float32,
        }
    }

    #[staticmethod]
    fn float16() -> Self {
        Self {
            inner: DType::Float16,
        }
    }

    #[staticmethod]
    fn bfloat16() -> Self {
        Self {
            inner: DType::Bfloat16,
        }
    }

    #[staticmethod]
    fn q4_0() -> Self {
        Self { inner: DType::Q4_0 }
    }

    #[staticmethod]
    fn q4_1() -> Self {
        Self { inner: DType::Q4_1 }
    }

    #[staticmethod]
    fn q8_0() -> Self {
        Self { inner: DType::Q8_0 }
    }

    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn size(&self) -> usize {
        self.inner.size_of()
    }

    fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    fn packed_block_size(&self) -> Option<usize> {
        self.inner.packed_block_size()
    }

    fn block_size(&self) -> Option<usize> {
        self.inner.block_size()
    }

    fn numel_to_bytes(&self, numel: usize) -> usize {
        self.inner.numel_to_bytes(numel)
    }

    fn __repr__(&self) -> String {
        format!("DType.{}", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pydtype_float32() {
        let dt = PyDType::float32();
        assert_eq!(dt.name(), "float32");
        assert_eq!(dt.size(), 4);
    }

    #[test]
    fn test_pydtype_float16() {
        let dt = PyDType::float16();
        assert_eq!(dt.name(), "float16");
        assert_eq!(dt.size(), 2);
    }

    #[test]
    fn test_pydtype_bfloat16() {
        let dt = PyDType::bfloat16();
        assert_eq!(dt.name(), "bfloat16");
        assert_eq!(dt.size(), 2);
    }

    #[test]
    fn test_pydtype_repr() {
        let dt = PyDType::float32();
        assert_eq!(dt.__repr__(), "DType.float32");
    }

    #[test]
    fn test_pydtype_str() {
        let dt = PyDType::float16();
        assert_eq!(dt.__str__(), "float16");
    }

    #[test]
    fn test_pydtype_quantized() {
        let dt = PyDType::q4_0();
        assert!(dt.is_quantized());
        assert_eq!(dt.packed_block_size(), Some(18));
        assert_eq!(dt.block_size(), Some(32));
        assert_eq!(dt.numel_to_bytes(32), 18);
    }

    #[test]
    fn test_pydtype_float_not_quantized() {
        let dt = PyDType::float32();
        assert!(!dt.is_quantized());
        assert_eq!(dt.packed_block_size(), None);
        assert_eq!(dt.block_size(), None);
        assert_eq!(dt.numel_to_bytes(100), 400);
    }
}
