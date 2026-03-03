# 첫 번째 Metal Compute 커널 실행하기

이 튜토리얼에서는 rmlx-metal을 사용하여 GPU에서 벡터 덧셈을 수행하는 과정을 단계별로 안내합니다.

---

## 개요

Metal compute 커널을 실행하는 전체 파이프라인은 다음과 같습니다.

```
디바이스 획득 → 버퍼 생성 → 셰이더 컴파일 → 파이프라인 생성 → 디스패치 → 결과 확인
```

이 튜토리얼의 전체 코드는 `crates/rmlx-metal/tests/metal_compute.rs`에서 확인할 수 있습니다.

---

## 1단계: 프로젝트 설정

`Cargo.toml`에 `rmlx-metal` 의존성을 추가합니다.

```toml
[dependencies]
rmlx-metal = { path = "../rmlx/crates/rmlx-metal" }
```

필요한 모듈을 import합니다.

```rust
use rmlx_metal::buffer::read_buffer;
use rmlx_metal::command::encode_compute_1d;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::library::compile_source;
use rmlx_metal::pipeline::PipelineCache;
use rmlx_metal::queue::GpuQueue;
```

---

## 2단계: GPU 디바이스 획득

시스템의 기본 Metal GPU 디바이스를 획득합니다.

```rust
let device = match GpuDevice::system_default() {
    Ok(d) => d,
    Err(_) => {
        eprintln!("skipping: no Metal device available");
        return;
    }
};
let queue = GpuQueue::new(&device);
```

- `GpuDevice::system_default()`는 시스템의 기본 Metal 디바이스를 반환합니다.
- Apple Silicon Mac에서는 통합 GPU가 기본 디바이스로 선택됩니다.
- Metal 디바이스가 없는 환경(VM, Intel Mac 등)에서는 에러를 반환합니다.
- `GpuQueue`는 GPU 명령을 제출하기 위한 커맨드 큐를 생성합니다.

---

## 3단계: 버퍼 생성

GPU 연산에 필요한 입력/출력 버퍼를 생성합니다.

```rust
// 입력 버퍼: 데이터와 함께 생성
let buffer_a = device.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0]);
let buffer_b = device.new_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0]);

// 출력 버퍼: 크기만 지정하여 빈 버퍼 생성
let buffer_out = device.new_buffer(
    16, // 4 floats * 4 bytes = 16 bytes
    rmlx_metal::metal::MTLResourceOptions::StorageModeShared,
);
```

- `new_buffer_with_data`는 Rust 슬라이스의 데이터를 GPU 버퍼로 복사합니다.
- `new_buffer`는 지정된 크기의 빈 버퍼를 생성합니다.
- `StorageModeShared`는 CPU와 GPU가 동일한 메모리를 공유하는 모드입니다.
  Apple Silicon의 UMA 구조에서는 실제로 데이터 복사가 발생하지 않습니다.

---

## 4단계: MSL 셰이더 작성 및 컴파일

Metal Shading Language(MSL)로 벡터 덧셈 커널을 작성합니다.

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
```

### MSL 속성(Attribute) 설명

| 속성 | 의미 |
|------|------|
| `kernel void` | GPU에서 실행되는 compute 함수를 선언합니다 |
| `device const float *a` | GPU 디바이스 메모리에 있는 읽기 전용 버퍼 포인터입니다 |
| `device float *out` | GPU 디바이스 메모리에 있는 쓰기 가능 버퍼 포인터입니다 |
| `[[buffer(0)]]` | Rust에서 `set_buffer(0, ...)` 으로 바인딩한 버퍼와 연결됩니다. 인덱스 0, 1, 2 순서로 매핑됩니다 |
| `[[thread_position_in_grid]]` | 현재 스레드의 전역 인덱스입니다. GPU가 각 스레드에 고유한 값을 자동 할당합니다. 이 예시에서는 0, 1, 2, 3이 됩니다 |

이 셰이더를 JIT 컴파일합니다.

```rust
const VECTOR_ADD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
"#;

// JIT 컴파일: 런타임에 MSL 소스를 컴파일합니다
let library = compile_source(device.raw(), VECTOR_ADD_SOURCE)
    .expect("shader compilation");
```

> **참고**: `compile_source`는 `MTLDevice::newLibraryWithSource`를 호출하여 MSL 소스를 런타임에 컴파일합니다.
> Xcode 없이도 동작하며, AOT 컴파일된 `.metallib` 파일을 사용하는 것과 동일한 결과를 생성합니다.

---

## 5단계: 파이프라인 생성

컴파일된 라이브러리에서 커널 함수를 찾아 compute pipeline state를 생성합니다.

```rust
let mut cache = PipelineCache::new(device.raw());
let pipeline = cache
    .get_or_create("vector_add_float", &library)
    .expect("pipeline creation");
```

- `PipelineCache`는 동일 커널의 파이프라인을 캐싱하여 중복 생성을 방지합니다.
- `get_or_create`는 함수 이름(`"vector_add_float"`)으로 라이브러리에서 커널 함수를 찾아 파이프라인을 생성합니다.
- 이미 캐시에 존재하면 기존 파이프라인을 반환합니다.

---

## 6단계: 디스패치 및 실행

커널을 GPU에 제출하고 실행합니다.

```rust
// 커맨드 버퍼 생성
let cmd_buf = queue.new_command_buffer();

// 1D compute 인코딩: 버퍼 바인딩 + 스레드 개수 지정
encode_compute_1d(
    cmd_buf,
    pipeline,
    &[(&buffer_a, 0), (&buffer_b, 0), (&buffer_out, 0)],
    4, // 스레드 수 = 원소 수
);

// GPU에 제출하고 완료 대기
cmd_buf.commit();
cmd_buf.wait_until_completed();
```

- `encode_compute_1d`는 compute command encoder를 생성하고, 버퍼를 바인딩하고, 스레드를 디스패치하는 과정을 하나의 호출로 처리합니다.
- 튜플 `(&buffer_a, 0)`에서 두 번째 값 `0`은 버퍼 내 오프셋(바이트)입니다.
- 스레드 수 `4`는 처리할 원소 수와 일치합니다. 각 스레드가 하나의 원소를 담당합니다.
- `commit()`은 커맨드 버퍼를 GPU 큐에 제출합니다.
- `wait_until_completed()`는 GPU 실행이 완료될 때까지 현재 스레드를 블록합니다.

---

## 7단계: 결과 확인

GPU에서 계산된 결과를 CPU에서 읽어옵니다.

```rust
// SAFETY: buffer_out uses StorageModeShared and GPU work has completed
// (wait_until_completed returned). We read exactly 4 floats = 16 bytes
// which matches the buffer allocation size.
let result: &[f32] = unsafe { read_buffer(&buffer_out, 4) };
assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
```

- `read_buffer`는 `unsafe` 함수입니다. GPU 버퍼의 내용을 Rust 슬라이스로 해석합니다.
- `StorageModeShared`를 사용했으므로 CPU와 GPU가 같은 메모리를 공유하여 별도의 데이터 전송이 필요 없습니다.
- `wait_until_completed()` 이후에 호출해야 GPU 연산 결과가 보장됩니다.

**결과**: `[1.0+5.0, 2.0+6.0, 3.0+7.0, 4.0+8.0]` = `[6.0, 8.0, 10.0, 12.0]`

---

## 전체 코드

아래는 위 단계들을 하나로 합친 전체 코드입니다.

```rust
use rmlx_metal::buffer::read_buffer;
use rmlx_metal::command::encode_compute_1d;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::library::compile_source;
use rmlx_metal::pipeline::PipelineCache;
use rmlx_metal::queue::GpuQueue;

const VECTOR_ADD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_float(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out     [[buffer(2)]],
    uint idx              [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}
"#;

fn main() {
    // 1. GPU 디바이스 획득
    let device = GpuDevice::system_default().expect("Metal device required");
    let queue = GpuQueue::new(&device);

    // 2. 입력/출력 버퍼 생성
    let buffer_a = device.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0]);
    let buffer_b = device.new_buffer_with_data(&[5.0f32, 6.0, 7.0, 8.0]);
    let buffer_out = device.new_buffer(
        16,
        rmlx_metal::metal::MTLResourceOptions::StorageModeShared,
    );

    // 3. 셰이더 JIT 컴파일 + 파이프라인 생성
    let library = compile_source(device.raw(), VECTOR_ADD_SOURCE)
        .expect("shader compilation");
    let mut cache = PipelineCache::new(device.raw());
    let pipeline = cache
        .get_or_create("vector_add_float", &library)
        .expect("pipeline creation");

    // 4. GPU 디스패치
    let cmd_buf = queue.new_command_buffer();
    encode_compute_1d(
        cmd_buf,
        pipeline,
        &[(&buffer_a, 0), (&buffer_b, 0), (&buffer_out, 0)],
        4,
    );

    // 5. 실행 및 대기
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // 6. 결과 확인
    // SAFETY: StorageModeShared buffer, GPU work completed, reading 4 f32 = 16 bytes
    let result: &[f32] = unsafe { read_buffer(&buffer_out, 4) };
    println!("결과: {:?}", result); // [6.0, 8.0, 10.0, 12.0]
}
```

---

## 테스트 실행

이 튜토리얼과 동일한 코드가 통합 테스트로 포함되어 있습니다.

```bash
# 통합 테스트 실행
cargo test -p rmlx-metal -- test_basic_metal_compute

# 출력과 함께 실행 (println! 출력 확인)
cargo test -p rmlx-metal -- test_basic_metal_compute --nocapture
```

---

## 다음 단계

이 튜토리얼에서 다룬 내용은 rmlx의 Metal compute 파이프라인 기초입니다.
프로젝트의 전체 구현 계획은 [구현 로드맵](../roadmap/phases.md)을 참고하세요.

- **Phase 2**에서 matmul, softmax, RoPE 등 10종의 GPU 연산 핵심 커널이 추가됩니다
- **Phase 3**에서 MTLSharedEvent 기반 동기화와 듀얼 큐 파이프라인이 구현됩니다
- **Phase 4**에서 Expert Parallelism을 위한 MoE 커널이 추가됩니다
- [GPU Pipeline](../gpu-pipeline.md) — ExecGraph가 연산을 배칭하여 16.15x 속도 향상을 달성하는 방법
