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

    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn size(&self) -> usize {
        self.inner.size_of()
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
}
