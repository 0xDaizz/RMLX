# 크레이트 구조

RMLX는 Cargo workspace로 구성된 6개의 크레이트와 부가 디렉토리로 이루어져 있습니다.

---

## 워크스페이스 전체 구조

```
rmlx/
├── Cargo.toml                    # workspace 루트
├── PLAN.md                       # 구현 계획서
├── crates/
│   ├── rmlx-metal/               # Metal GPU 추상화
│   │   ├── Cargo.toml            # deps: metal-rs 0.31, objc2, block2
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── device.rs         # MTLDevice 래퍼, 아키텍처 감지
│   │       ├── queue.rs          # MTLCommandQueue 관리 (듀얼 큐 지원)
│   │       ├── command.rs        # CommandBuffer/Encoder 추상화
│   │       ├── buffer.rs         # MTLBuffer, zero-copy 생성
│   │       ├── event.rs          # MTLSharedEvent 래퍼
│   │       ├── fence.rs          # MTLFence + fast-fence (shared buffer spin)
│   │       ├── pipeline.rs       # ComputePipelineState 캐시
│   │       ├── library.rs        # MTLLibrary 로드/JIT 컴파일
│   │       ├── resident.rs       # ResidencySet 관리
│   │       └── self_check.rs     # 시동 진단 (startup diagnostics)
│   │
│   ├── rmlx-alloc/               # 메모리 할당자
│   │   ├── Cargo.toml            # deps: rmlx-metal, libc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── zero_copy.rs      # posix_memalign + newBufferWithBytesNoCopy
│   │       ├── pool.rs           # 이중 등록 버퍼 풀 (Metal + ibv_mr)
│   │       ├── cache.rs          # MLX-style size-binned 캐시
│   │       ├── stats.rs          # 할당 통계, 피크 메모리 추적
│   │       └── leak_detector.rs  # 메모리 누수 감지
│   │
│   ├── rmlx-rdma/                # RDMA 통신
│   │   ├── Cargo.toml            # deps: rmlx-alloc, libc (ibverbs FFI)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── ffi.rs            # ibverbs C FFI 바인딩 (bindgen)
│   │   │   ├── context.rs        # ibv_context, PD, CQ 관리
│   │   │   ├── qp.rs             # UC QP 생성/관리, GID 처리
│   │   │   ├── mr.rs             # ibv_reg_mr, 이중 등록 버퍼 관리
│   │   │   ├── exchange.rs       # blocking_exchange (2-phase count→payload)
│   │   │   ├── collective.rs     # all-to-all, ring allreduce
│   │   │   ├── connection.rs     # hosts.json 파싱, 연결 수립, warmup
│   │   │   ├── multi_port.rs     # 듀얼 TB5 포트 스트라이핑
│   │   │   └── rdma_metrics.rs   # RDMA 메트릭 수집
│   │   └── build.rs              # ibverbs 링크 설정
│   │
│   ├── rmlx-core/                # 연산 엔진
│   │   ├── Cargo.toml            # deps: rmlx-metal, rmlx-alloc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dtype.rs          # 데이터 타입 (f32, f16, bf16, quantized)
│   │       ├── array.rs          # N-dim array 타입 (Buffer 소유권)
│   │       ├── ops/
│   │       │   ├── mod.rs
│   │       │   ├── matmul.rs     # GEMM (metal-rs dispatch)
│   │       │   ├── quantized.rs  # QMM (4bit, 8bit, FP4, FP8)
│   │       │   ├── softmax.rs
│   │       │   ├── rms_norm.rs
│   │       │   ├── rope.rs
│   │       │   ├── binary.rs     # element-wise 연산
│   │       │   ├── reduce.rs
│   │       │   ├── copy.rs
│   │       │   └── indexing.rs
│   │       ├── kernels/
│   │       │   ├── mod.rs        # 커널 레지스트리
│   │       │   ├── loader.rs     # .metallib AOT 로드
│   │       │   └── jit.rs        # 소스 문자열 JIT 컴파일
│   │       ├── graph.rs          # 연산 그래프 (선택적 tracing)
│   │       ├── scheduler.rs      # 스트림별 실행 스케줄러
│   │       ├── vjp.rs            # VJP (Vector-Jacobian Product) autodiff
│   │       ├── lora.rs           # LoRA fine-tuning 지원
│   │       ├── logging.rs        # Structured logging (tracing)
│   │       ├── metrics.rs        # 메트릭 수집 (Prometheus 호환)
│   │       ├── precision_guard.rs # 정밀도 가드 (dtype 안전성)
│   │       └── shutdown.rs       # Graceful shutdown
│   │
│   ├── rmlx-distributed/         # 분산 프리미티브
│   │   ├── Cargo.toml            # deps: rmlx-core, rmlx-rdma
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── group.rs          # 분산 그룹 추상화 (rank, world_size)
│   │       ├── primitives.rs     # AllReduce, AllGather, Send, Recv, AllToAll
│   │       ├── moe/
│   │       │   ├── mod.rs
│   │       │   ├── dispatch.rs   # MoeDispatchExchange (3-zone auto backend)
│   │       │   ├── combine.rs    # MoeCombineExchange
│   │       │   ├── policy.rs     # 3-zone 정책 (CPU/Metal/byte threshold)
│   │       │   ├── kernels.rs    # MoE Metal 커널 7종 관리
│   │       │   └── warmup.rs     # RDMA + Metal JIT 사전 워밍업
│   │       ├── moe_exchange.rs   # MoE 데이터 교환 프리미티브
│   │       ├── moe_policy.rs     # MoE 라우팅 정책
│   │       ├── pipeline.rs       # layer-level compute↔RDMA 파이프라인
│   │       ├── sparse_guard.rs   # 희소 디스패치 안전 가드
│   │       ├── warmup.rs         # 분산 워밍업
│   │       └── metrics.rs        # 분산 메트릭 수집
│   │
│   └── rmlx-nn/                  # 신경망 레이어
│       ├── Cargo.toml            # deps: rmlx-core
│       └── src/
│           ├── lib.rs
│           ├── linear.rs         # Linear (quantized 지원)
│           ├── embedding.rs
│           ├── attention.rs      # Multi-head/GQA attention
│           ├── transformer.rs    # Transformer block
│           ├── moe.rs            # MoE gate + expert 라우팅
│           └── models/
│               ├── mod.rs
│               ├── llama.rs      # LLaMA 아키텍처
│               ├── qwen.rs       # Qwen/Qwen2.5
│               ├── deepseek.rs   # DeepSeek-V3 (MoE)
│               └── mixtral.rs    # Mixtral (MoE)
│
├── shaders/                      # Metal 셰이더 소스
│   ├── mlx_compat/               # MLX에서 가져온 .metal 파일
│   │   ├── gemv.metal
│   │   ├── gemm.metal
│   │   ├── quantized.metal
│   │   ├── softmax.metal
│   │   ├── rms_norm.metal
│   │   ├── rope.metal
│   │   ├── binary.metal
│   │   ├── reduce.metal
│   │   ├── copy.metal
│   │   └── fence.metal
│   ├── moe/                      # EP 전용 커널
│   │   ├── dispatch_local.metal
│   │   ├── dispatch_scatter.metal
│   │   ├── combine_gather.metal
│   │   ├── combine_weighted_sum.metal
│   │   ├── packet_gather.metal
│   │   └── packet_scatter.metal
│   └── custom/                   # RMLX 전용 커널
│       ├── fused_attention.metal
│       └── fused_moe_gate.metal
│
├── tests/                        # 통합 테스트
│   ├── metal_basic.rs
│   ├── zero_copy.rs
│   ├── rdma_exchange.rs
│   ├── moe_dispatch.rs
│   └── e2e_inference.rs
│
├── benches/                      # 벤치마크
│   ├── matmul.rs
│   ├── rdma_latency.rs
│   ├── moe_step.rs
│   └── e2e_throughput.rs
│
└── examples/
    ├── basic_compute.rs
    ├── zero_copy_demo.rs
    └── two_node_ep.rs
```

---

## 크레이트별 상세 정보

### rmlx-metal — Metal GPU 추상화

| 항목 | 내용 |
|------|------|
| **목적** | Apple Metal API를 Rust로 안전하게 추상화합니다. `metal-rs` 0.31을 기반으로 MTLDevice, MTLCommandQueue, MTLBuffer, MTLSharedEvent 등을 래핑합니다. |
| **핵심 모듈** | `device.rs` (디바이스 + 아키텍처 감지), `queue.rs` (듀얼 큐 관리), `command.rs` (CommandBuffer/Encoder), `event.rs` (MTLSharedEvent 래퍼), `pipeline.rs` (PSO 캐시), `self_check.rs` (시동 진단) |
| **의존성** | metal-rs 0.31, objc2, block2 |
| **현재 상태** | 완료 — GpuDevice, StreamManager, DeviceStream, GpuEvent, SharedEvent 동기화, 듀얼 큐 파이프라인, 시동 진단 전체 구현 |

---

### rmlx-alloc — 메모리 할당자

| 항목 | 내용 |
|------|------|
| **목적** | Zero-copy 메모리 할당 및 버퍼 풀 관리를 담당합니다. `posix_memalign` → `newBufferWithBytesNoCopy` 패턴으로 복사 없는 Metal 버퍼를 생성하고, RDMA `ibv_mr` 이중 등록을 지원합니다. |
| **핵심 모듈** | `zero_copy.rs` (ZeroCopyBuffer, CompletionFence), `pool.rs` (이중 등록 버퍼 풀), `cache.rs` (size-binned 캐시), `stats.rs` (할당 통계), `leak_detector.rs` (메모리 누수 감지) |
| **의존성** | `rmlx-metal`, libc |
| **현재 상태** | 완료 — ZeroCopyBuffer, DualRegPool, MetalAllocator, size-binned 캐시, 누수 감지 전체 구현 |

---

### rmlx-rdma — RDMA 통신

| 항목 | 내용 |
|------|------|
| **목적** | Thunderbolt 5 RDMA를 통한 노드 간 고성능 통신을 제공합니다. ibverbs C 라이브러리를 FFI로 바인딩하며, UC QP 기반 send/recv와 collective 연산을 지원합니다. |
| **핵심 모듈** | `ffi.rs` (ibverbs 바인딩), `context.rs` (ibv_context/PD/CQ), `qp.rs` (UC QP), `mr.rs` (메모리 영역 등록), `collective.rs` (all-to-all, allreduce), `multi_port.rs` (포트 스트라이핑), `rdma_metrics.rs` (RDMA 메트릭) |
| **의존성** | `rmlx-alloc`, libc (ibverbs FFI) |
| **현재 상태** | 완료 — ibverbs FFI, UC QP, blocking_exchange, ConnectionManager, 듀얼 포트 스트라이핑, RDMA 메트릭 전체 구현 |

---

### rmlx-core — 연산 엔진

| 항목 | 내용 |
|------|------|
| **목적** | 연산 그래프, Op 레지스트리, 커널 디스패치를 통합 관리하는 핵심 엔진입니다. N-dim array 타입과 dtype 시스템을 정의하며, eager-first 실행과 선택적 tracing 컴파일을 지원합니다. |
| **핵심 모듈** | `dtype.rs` (f32, f16, bf16, quantized), `array.rs` (N-dim array), `ops/` (matmul, softmax 등 7종 커널), `kernels/` (AOT/JIT 커널 관리), `graph.rs` (연산 그래프), `scheduler.rs` (스트림별 스케줄러), `vjp.rs` (VJP autodiff), `lora.rs` (LoRA fine-tuning), `logging.rs` (structured logging), `metrics.rs` (메트릭 수집), `precision_guard.rs` (정밀도 가드), `shutdown.rs` (graceful shutdown) |
| **의존성** | `rmlx-metal`, `rmlx-alloc` |
| **현재 상태** | 완료 — Array 타입, 7종 Metal 커널 디스패치, VJP autodiff, LoRA, 프로덕션 하드닝 전체 구현 |

---

### rmlx-distributed — 분산 프리미티브

| 항목 | 내용 |
|------|------|
| **목적** | 분산 추론에 필요한 통신 프리미티브와 MoE Expert Parallelism을 구현합니다. layer-level 파이프라인을 통해 compute와 RDMA를 오버랩합니다. |
| **핵심 모듈** | `group.rs` (rank/world_size 추상화), `primitives.rs` (AllReduce, AllGather 등), `moe/` (3-zone auto backend, MoE dispatch/combine), `moe_exchange.rs`, `moe_policy.rs`, `pipeline.rs` (compute-RDMA 오버랩), `sparse_guard.rs` (희소 디스패치 가드), `warmup.rs` (분산 워밍업), `metrics.rs` (분산 메트릭) |
| **의존성** | `rmlx-core`, `rmlx-rdma` |
| **현재 상태** | 완료 — EP dispatch/combine, 3-zone auto backend, compute-RDMA 파이프라인, MoE 교환, 분산 메트릭 전체 구현 |

---

### rmlx-nn — 신경망 레이어

| 항목 | 내용 |
|------|------|
| **목적** | Transformer 기반 아키텍처의 신경망 레이어를 제공합니다. Linear, Attention, MoE 등의 고수준 모듈과 LLaMA, Qwen, DeepSeek-V3 등 모델 아키텍처를 포함합니다. |
| **핵심 모듈** | `linear.rs` (quantized Linear, `prepare_weight_t()`), `attention.rs` (Multi-head/GQA), `transformer.rs` (Transformer block, `forward_graph()`, `forward_into_cb()`), `moe.rs` (gate + routing), `models/` (llama.rs, qwen.rs, deepseek.rs, mixtral.rs) |
| **의존성** | `rmlx-core` |
| **현재 상태** | 완료 — Transformer 블록, Linear/Attention/MoE 레이어, LLaMA/Qwen/DeepSeek-V3/Mixtral 모델 아키텍처 전체 구현 |

---

## 워크스페이스 설정

```toml
# Cargo.toml (workspace 루트)

[workspace]
resolver = "2"
members = [
    "crates/rmlx-metal",
    "crates/rmlx-alloc",
    "crates/rmlx-rdma",
    "crates/rmlx-core",
    "crates/rmlx-distributed",
    "crates/rmlx-nn",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
license = "MIT OR Apache-2.0"

[workspace.dependencies]
metal = "0.31"
objc = "0.2"
block = "0.1"
libc = "0.2"
```

`workspace.dependencies`를 활용하여 크레이트 간 의존성 버전을 통일합니다. 각 크레이트의 `Cargo.toml`에서 `metal.workspace = true` 형태로 참조할 수 있습니다.

---

## 부가 디렉토리

### shaders/

Metal 셰이더 소스 파일을 저장합니다. 3개 하위 디렉토리로 분류됩니다.

- **mlx_compat/**: MLX에서 가져온 호환 커널 (gemv, gemm, quantized, softmax 등)
- **moe/**: Expert Parallelism 전용 커널 (dispatch, combine, packet 관련 6종)
- **custom/**: RMLX 전용 최적화 커널 (fused_attention, fused_moe_gate)

### tests/

통합 테스트입니다. 단위 테스트는 각 크레이트 내부에 위치합니다.

### benches/

Criterion 기반 벤치마크입니다. matmul, RDMA latency, MoE step, e2e throughput을 측정합니다.

### examples/

사용 예제입니다. 기본 compute, zero-copy 데모, 2-node EP 예제를 포함합니다.
