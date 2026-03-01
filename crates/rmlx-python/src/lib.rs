//! PyO3 Python bindings for RMLX.
//! Provides `import rmlx` with Array, DType, and kernel operations.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod array_wrapper;
mod dtype_wrapper;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
mod rmlx {
    #[pymodule_export]
    use super::array_wrapper::PyArray;
    #[pymodule_export]
    use super::dtype_wrapper::PyDType;
    #[pymodule_export]
    use super::version;
}
