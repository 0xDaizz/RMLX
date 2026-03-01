# rmlx-python — Python 바인딩

## 개요

`rmlx-python`은 PyO3 0.28 기반의 Python 바인딩으로, `import rmlx`로 Python에서 RMLX의 Array 및 DType을 사용할 수 있게 합니다. maturin으로 빌드하며, Python 3.9+를 지원합니다.

> **상태:** PyDType (3종 float 타입), PyArray (생성, 산술, 리덕션, reshape)가 구현되어 있습니다.

---

## 모듈 구조

```
rmlx-python/
├── Cargo.toml          # cdylib, PyO3 0.28, maturin
└── src/
    ├── lib.rs           # PyO3 모듈 선언 (rmlx)
    ├── dtype_wrapper.rs # PyDType 클래스
    └── array_wrapper.rs # PyArray 클래스
```

### Cargo.toml 핵심 설정

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

## 모듈 진입점 (`lib.rs`)

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

Python에서 노출되는 이름:
- `rmlx.Array` → `PyArray`
- `rmlx.DType` → `PyDType`
- `rmlx.version()` → 패키지 버전 문자열

---

## PyDType 클래스 (`dtype_wrapper.rs`)

`rmlx_core::dtype::DType`을 래핑하는 불변(frozen) Python 클래스입니다.

```rust
#[pyclass(name = "DType", frozen, skip_from_py_object)]
pub struct PyDType {
    pub(crate) inner: DType,
}
```

### Static 생성 메서드

| Python | 내부 DType |
|--------|-----------|
| `DType.float32()` | `DType::Float32` |
| `DType.float16()` | `DType::Float16` |
| `DType.bfloat16()` | `DType::Bfloat16` |

### 인스턴스 메서드

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `name()` | `str` | 타입 이름 (예: `"float32"`) |
| `size()` | `int` | 요소 크기 (바이트) |
| `__repr__()` | `str` | `"DType.float32"` 형식 |
| `__str__()` | `str` | `"float32"` 형식 |

---

## PyArray 클래스 (`array_wrapper.rs`)

f32 데이터를 가진 N차원 배열입니다. shape 검증, 산술 연산, 리덕션을 지원합니다.

```rust
#[pyclass(name = "Array")]
pub struct PyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype_name: String,   // 현재 "float32" 고정
}
```

### 생성자

| Python | 설명 |
|--------|------|
| `Array(data, shape)` | 데이터와 형상으로 생성 (길이 불일치 시 ValueError) |
| `Array.zeros(shape)` | 0으로 초기화 |
| `Array.ones(shape)` | 1.0으로 초기화 |
| `Array.from_list(data, shape)` | `Array(data, shape)`와 동일 |

### 속성 (Properties)

| 속성 | 타입 | 설명 |
|------|------|------|
| `shape` | `list[int]` | 배열 형상 |
| `ndim` | `int` | 차원 수 |
| `size` | `int` | 총 요소 수 |
| `dtype` | `str` | 데이터 타입 이름 |

### 연산 메서드

| 메서드 | 설명 |
|--------|------|
| `reshape(new_shape)` | 새 형상으로 재배치 (총 요소 수 일치 필수) |
| `tolist()` | `list[float]`로 데이터 추출 |
| `__add__(other)` | 원소별 덧셈 (shape 일치 필수) |
| `__mul__(other)` | 원소별 곱셈 (shape 일치 필수) |
| `__len__()` | 총 요소 수 |
| `__repr__()` | `"Array(shape=[2, 3], dtype=float32)"` 형식 |

### 리덕션

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `sum()` | `float` | 전체 합 |
| `max()` | `float` or `None` | 최댓값 |
| `min()` | `float` or `None` | 최솟값 |
| `mean()` | `float` | 평균 |

---

## 빌드 및 설치

### maturin으로 빌드

```bash
# 개발 빌드 (현재 venv에 설치)
cd crates/rmlx-python
maturin develop --release

# 휠 빌드
maturin build --release
```

### 요구사항

- Python 3.9+
- Rust stable
- maturin (`pip install maturin`)

---

## Python 사용 예시

```python
import rmlx

# 버전 확인
print(rmlx.version())

# DType
dt = rmlx.DType.float32()
print(dt.name())   # "float32"
print(dt.size())   # 4

# Array 생성
a = rmlx.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
print(a.shape)     # [2, 3]
print(a.ndim)      # 2
print(a.size)      # 6
print(a.dtype)     # "float32"

# zeros / ones
z = rmlx.Array.zeros([3, 4])
o = rmlx.Array.ones([2, 2])

# 산술 연산
b = rmlx.Array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [2, 3])
c = a + b          # 원소별 덧셈
d = a * b          # 원소별 곱셈

# 리덕션
print(a.sum())     # 21.0
print(a.max())     # 6.0
print(a.min())     # 1.0
print(a.mean())    # 3.5

# reshape
r = a.reshape([3, 2])
print(r.shape)     # [3, 2]

# 데이터 추출
data = a.tolist()  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

---

## 의존성

```toml
[dependencies]
rmlx-core = { path = "../rmlx-core" }
rmlx-nn = { path = "../rmlx-nn" }
rmlx-metal = { path = "../rmlx-metal" }
pyo3 = { version = "0.28", features = ["extension-module"] }
```
