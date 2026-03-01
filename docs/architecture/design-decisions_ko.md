# 설계 결정

RMLX의 주요 기술 결정과 그 근거를 정리합니다. 각 결정은 PoC(Proof of Concept) 검증 결과 또는 MLX 아키텍처 분석에 기반합니다.

---

## 1. Rust를 선택한 이유 (C++ 대신)

MLX는 C++로 작성되어 있지만, RMLX는 Rust를 선택했습니다.

### 근거

| 관점 | MLX (C++) | RMLX (Rust) |
|------|-----------|-------------|
| **메모리 안전성** | 수동 관리, MTL::Buffer 수명 추적이 수동 | 소유권 시스템으로 컴파일 타임에 안전성 보장 |
| **동시성** | `std::mutex` 수동 잠금, data race 가능 | `Send`/`Sync` 트레이트로 컴파일 타임 스레드 안전성 |
| **빌드 시스템** | CMake + Python setuptools + nanobind | Cargo workspace 단일 빌드 체인 |
| **에러 처리** | C++ exception, Metal 에러 런타임 확인 | `Result<T, E>` 타입으로 명시적 에러 전파 |
| **추상화 비용** | virtual dispatch 오버헤드 | trait object 또는 monomorphization 선택 가능 (zero-cost) |
| **Python 바인딩** | nanobind (C++ → Python) | 불필요 (Rust 네이티브 프레임워크) |

### 핵심 판단

GPU 인퍼런스 엔진은 `unsafe` 블록(Metal FFI, RDMA FFI)이 불가피하지만, Rust의 소유권 시스템은 `unsafe` 경계를 **명시적으로 격리**할 수 있습니다. 안전한 코드(`ZeroCopyBuffer` 외부 API)와 위험한 코드(`posix_memalign`, `ibv_post_send` 호출부)가 타입 시스템 레벨에서 구분되므로, C++보다 안전하면서도 동일한 성능을 유지할 수 있습니다.

또한 Cargo workspace는 6개 크레이트의 의존성, 버전, 빌드 설정을 단일 `Cargo.toml`로 관리합니다. CMake + setuptools 조합 대비 빌드 복잡도가 현저히 낮습니다.

---

## 2. metal-rs 0.31을 선택한 이유

Metal API 바인딩으로 metal-rs 0.31을 사용합니다.

### 근거

- **PoC Phase 4에서 완전 검증**: `posix_memalign` → `newBufferWithBytesNoCopy` → `ibv_reg_mr` → RDMA 전송 → Metal compute 전체 파이프라인을 metal-rs 0.31로 성공적으로 재현했습니다.
- **직접 MTLDevice/MTLBuffer 접근**: Objective-C++ 브릿지 없이 Rust에서 Metal 객체에 직접 접근할 수 있습니다.
- **타입 안전한 API**: `Device`, `CommandQueue`, `Buffer`, `ComputePipelineState` 등이 Rust 타입으로 래핑되어 있어 misuse를 컴파일 타임에 방지합니다.
- **Metal 3 기능 지원**: `MTLSharedEvent`, `ResidencySet` 등 M3 이상에서 필요한 API를 지원합니다.

### 대안 분석

| 대안 | 미채택 사유 |
|------|------------|
| metal-cpp (C++) | Objective-C++ 브릿지 필요, Rust 프로젝트와 빌드 통합 복잡 |
| objc2 직접 사용 | 모든 Metal API를 수동 바인딩해야 하므로 개발 속도 저하 |
| wgpu | Metal 전용 최적화(SharedEvent, NoCopy buffer) 접근 불가 |

---

## 3. Eager-first 실행 모델

MLX의 lazy evaluation 대신 eager-first 실행을 채택합니다.

### 근거

MLX는 lazy evaluation을 기본으로 사용합니다. 연산을 즉시 실행하지 않고 그래프를 빌드한 후 `eval()` 호출 시 한꺼번에 실행합니다. 이 방식은 배치 연산(prefill)에서는 최적화 기회를 제공하지만, **단일 토큰 디코드**에서는 문제가 됩니다.

```
디코드 단계: 매 스텝마다 1개 토큰만 처리
→ 그래프 빌드 오버헤드 > 실제 연산 시간
→ per-op 94μs 중 84μs가 소프트웨어 오버헤드
```

### RMLX의 접근법

- **디코드(autoregressive)**: Eager 실행 — 연산을 즉시 Metal CommandBuffer에 인코딩하여 제출합니다. 그래프 빌드 오버헤드가 없습니다.
- **프리필(batch)**: 선택적 tracing — 반복되는 연산 패턴을 tracing으로 기록하고, 최적화된 그래프로 컴파일하여 실행합니다.

이 하이브리드 접근법은 디코드의 per-step 레이턴시를 최소화하면서, 프리필에서는 연산 퓨전 등의 최적화 혜택을 유지합니다.

---

## 4. Zero-copy RDMA 데이터 경로

RDMA 통신 경로에서 `memcpy`/`std::copy`를 완전히 제거합니다.

### 근거

MLX에서 RDMA 통신 시 데이터를 `std::copy`로 별도 버퍼에 복사한 후 전송합니다. 이는 대규모 텐서 전송에서 상당한 오버헤드를 발생시킵니다.

### 구현 방식

```
Step 1: posix_memalign(page_size)로 페이지 정렬 메모리 할당
Step 2: newBufferWithBytesNoCopy로 Metal 버퍼 뷰 생성 (복사 없음)
Step 3: ibv_reg_mr로 RDMA MR 등록 (동일 물리 주소)
```

이렇게 생성된 3개의 뷰(raw pointer, Metal buffer, RDMA MR)는 **동일한 물리 주소**를 공유합니다.

```
Metal GPU compute → [동일 물리 버퍼] → ibv_post_send → wire
→ ibv_recv → [동일 물리 버퍼] → Metal GPU compute
```

- PoC Phase 2에서 `buffer.contents() == raw_ptr == mr.addr` 주소 일치를 확인했습니다.
- PoC Phase 4에서 전체 파이프라인(할당 → Metal compute → RDMA 전송 → 수신 → Metal compute)을 검증했습니다.

### 범위 한정

"Zero-copy"는 **반복 실행되는 RDMA 통신 hot path**에 한정됩니다. 다음 구간에서는 복사가 존재하며, 이는 의도된 것입니다.

| 구간 | 방식 | 이유 |
|------|------|------|
| 모델 가중치 로딩 | `mmap` → `copy_nonoverlapping` → Metal buffer | safetensors mmap은 read-only, GPU 가중치는 persistent resident 필요 |
| KV 캐시 초기화 | `calloc` → Metal buffer 또는 heap 할당 | 초기 1회, 이후 GPU 내부에서만 갱신 |

---

## 5. MTLSharedEvent 기반 동기화

`waitUntilCompleted` 대신 `MTLSharedEvent`를 사용합니다.

### 근거

MLX에서 GPU 작업 완료를 확인하기 위해 `waitUntilCompleted`를 호출합니다. 이 API는 커맨드 버퍼가 완료될 때까지 CPU 스레드를 **블로킹**합니다.

### 성능 비교

| 방식 | 레이턴시 | 비고 |
|------|----------|------|
| `waitUntilCompleted` | 424.9μs | CPU 블로킹, 컨텍스트 스위치 발생 |
| `MTLSharedEvent` spin-wait | 263.9μs | 비블로킹, signal/wait 패턴 |
| **개선율** | **1.61x** | |

### 동작 원리

```
1. GPU CommandBuffer에 encodeSignalEvent(event, value: N)을 인코딩합니다.
2. CPU 측에서 event.signaledValue를 spin-wait로 폴링합니다.
3. value >= N이 되면 GPU 작업 완료를 확인합니다.
```

`MTLSharedEvent`는 cross-queue 동기화에도 사용됩니다. 듀얼 큐 파이프라인에서 compute 큐와 transfer 큐 간의 데이터 의존성을 event signal/wait으로 관리합니다.

---

## 6. UC QP 선택 (RC 대신)

RDMA Queue Pair 타입으로 UC(Unreliable Connection)를 사용합니다.

### 근거

일반적으로 RDMA 통신에서는 RC(Reliable Connection) QP를 사용합니다. 그러나 **Thunderbolt 5 RDMA는 RC를 지원하지 않습니다.** 하드웨어 제약으로 인해 UC QP만 사용 가능합니다.

### UC QP의 특성

| 특성 | 설명 |
|------|------|
| 연결형 | 1:1 QP 연결 (connectionless가 아님) |
| 비신뢰성 | 하드웨어 레벨 재전송/확인 없음 |
| Send/Recv 지원 | ibv_post_send/recv로 데이터 전송 |
| RDMA Write | 지원 (단, 원격 키 교환 필요) |
| RDMA Read | **미지원** |

### 신뢰성 보장

UC는 하드웨어 레벨 신뢰성이 없으므로, 애플리케이션 레벨에서 신뢰성을 보장합니다.

- **2-phase exchange**: count 메시지 → payload 순서로 전송하여 수신 측이 예상 데이터량을 사전에 파악합니다.
- **CQ polling**: Completion Queue에서 `IBV_WC_SUCCESS`를 확인하여 전송 완료를 검증합니다.
- **Warmup**: 연결 수립 후 더미 데이터로 경로를 예열하여 초기 패킷 손실을 방지합니다.

실제로 Thunderbolt 5는 로컬 연결(케이블 1~2m)이므로 패킷 손실 확률이 매우 낮습니다. 네트워크 수준의 복잡한 재전송 프로토콜은 불필요합니다.

---

## 7. 듀얼 큐 파이프라인

Compute 큐와 Transfer 큐를 분리하여 GPU 레벨에서 오버랩합니다.

### 근거

단일 `MTLCommandQueue`에서는 compute와 데이터 전송이 순차적으로 실행됩니다.

```
단일 큐: [compute layer N] → [transfer layer N] → [compute layer N+1] → ...
듀얼 큐: [compute layer N  ] → [compute layer N+1  ] → ...
         [transfer layer N-1] → [transfer layer N    ] → ...
```

듀얼 큐를 사용하면 layer N의 compute가 진행되는 동안 layer N-1의 RDMA 전송 결과를 동시에 처리할 수 있습니다.

### 구현 세부사항

```rust
pub struct StreamManager {
    compute_queue: CommandQueue,    // 주 연산 큐
    transfer_queue: CommandQueue,   // RDMA 동기화/데이터 준비 큐
}
```

- **compute_queue**: matmul, softmax, attention 등 주요 연산을 실행합니다.
- **transfer_queue**: RDMA send/recv 완료 대기 및 수신 데이터 준비(reformat 등)를 처리합니다.

Cross-queue 동기화는 `MTLSharedEvent`로 관리합니다. `HazardTrackingModeUntracked` 옵션을 사용하여 Metal의 자동 hazard tracking을 비활성화하고, 모든 cross-queue 접근을 명시적으로 동기화합니다. 이 패턴은 PoC Phase 3.6에서 검증되었습니다.

### 주의사항

`HazardTrackingModeUntracked` 사용 시, 모든 cross-queue 버퍼 접근에 대해 반드시 `MTLSharedEvent` 또는 `MTLFence`로 동기화해야 합니다. 새로운 cross-queue 접근 패턴을 추가할 때는 반드시 데이터 정합성 테스트를 동반해야 합니다.
