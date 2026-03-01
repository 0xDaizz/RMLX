# rmlx-python — Python Bindings

## Overview

`rmlx-python` is a PyO3 0.28-based Python binding that enables using RMLX's Array and DType from Python via `import rmlx`. It is built with maturin and supports Python 3.9+.

> **Status:** PyDType (3 float types) and PyArray (creation, arithmetic, reductions, reshape) are implemented.

---

## Module Structure

```
rmlx-python/
├── Cargo.toml          # cdylib, PyO3 0.28, maturin
└── src/
    ├── lib.rs           # PyO3 module declaration (rmlx)
    ├── dtype_wrapper.rs # PyDType class
    └── array_wrapper.rs # PyArray class
```

### Cargo.toml Key Configuration

```toml
[lib]
name = "rmlx"
crate-type = ["cdylib"]

[dependencies]
rmlx-core = { path = "../rmlx-core" }
rmlx-nn = { path = "../rmlx-nn" }
rmlx-metal = { path = "../rmlx-metal" }
pyo3 = { version = "0.28", features = ["extension-module"] }
```

---

## Module Entry Point (`lib.rs`)

```rust
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
```

Names exposed to Python:
- `rmlx.Array` -> `PyArray`
- `rmlx.DType` -> `PyDType`
- `rmlx.version()` -> package version string

---

## PyDType Class (`dtype_wrapper.rs`)

A frozen Python class wrapping `rmlx_core::dtype::DType`.

```rust
#[pyclass(name = "DType", frozen, skip_from_py_object)]
pub struct PyDType {
    pub(crate) inner: DType,
}
```

### Static Creation Methods

| Python | Internal DType |
|--------|---------------|
| `DType.float32()` | `DType::Float32` |
| `DType.float16()` | `DType::Float16` |
| `DType.bfloat16()` | `DType::Bfloat16` |

### Instance Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `name()` | `str` | Type name (e.g., `"float32"`) |
| `size()` | `int` | Element size in bytes |
| `__repr__()` | `str` | `"DType.float32"` format |
| `__str__()` | `str` | `"float32"` format |

---

## PyArray Class (`array_wrapper.rs`)

An N-dimensional array holding f32 data. Supports shape validation, arithmetic operations, and reductions.

```rust
#[pyclass(name = "Array")]
pub struct PyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype_name: String,   // currently fixed to "float32"
}
```

### Constructors

| Python | Description |
|--------|-------------|
| `Array(data, shape)` | Creates from data and shape (ValueError on length mismatch) |
| `Array.zeros(shape)` | Initializes with zeros |
| `Array.ones(shape)` | Initializes with 1.0 |
| `Array.from_list(data, shape)` | Same as `Array(data, shape)` |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `list[int]` | Array shape |
| `ndim` | `int` | Number of dimensions |
| `size` | `int` | Total number of elements |
| `dtype` | `str` | Data type name |

### Operation Methods

| Method | Description |
|--------|-------------|
| `reshape(new_shape)` | Reshapes to new shape (total element count must match) |
| `tolist()` | Extracts data as `list[float]` |
| `__add__(other)` | Element-wise addition (shapes must match) |
| `__mul__(other)` | Element-wise multiplication (shapes must match) |
| `__len__()` | Total number of elements |
| `__repr__()` | `"Array(shape=[2, 3], dtype=float32)"` format |

### Reductions

| Method | Returns | Description |
|--------|---------|-------------|
| `sum()` | `float` | Total sum |
| `max()` | `float` or `None` | Maximum value |
| `min()` | `float` or `None` | Minimum value |
| `mean()` | `float` | Mean |

---

## Build and Installation

### Building with maturin

```bash
# Development build (installs into current venv)
cd crates/rmlx-python
maturin develop --release

# Wheel build
maturin build --release
```

### Requirements

- Python 3.9+
- Rust stable
- maturin (`pip install maturin`)

---

## Python Usage Example

```python
import rmlx

# Check version
print(rmlx.version())

# DType
dt = rmlx.DType.float32()
print(dt.name())   # "float32"
print(dt.size())   # 4

# Array creation
a = rmlx.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
print(a.shape)     # [2, 3]
print(a.ndim)      # 2
print(a.size)      # 6
print(a.dtype)     # "float32"

# zeros / ones
z = rmlx.Array.zeros([3, 4])
o = rmlx.Array.ones([2, 2])

# Arithmetic operations
b = rmlx.Array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [2, 3])
c = a + b          # element-wise addition
d = a * b          # element-wise multiplication

# Reductions
print(a.sum())     # 21.0
print(a.max())     # 6.0
print(a.min())     # 1.0
print(a.mean())    # 3.5

# reshape
r = a.reshape([3, 2])
print(r.shape)     # [3, 2]

# Data extraction
data = a.tolist()  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

---

## Dependencies

```toml
[dependencies]
rmlx-core = { path = "../rmlx-core" }
rmlx-nn = { path = "../rmlx-nn" }
rmlx-metal = { path = "../rmlx-metal" }
pyo3 = { version = "0.28", features = ["extension-module"] }
```
