# 크레이트 구조

RMLX는 Cargo workspace로 구성된 8개의 크레이트(rmlx-metal, rmlx-alloc, rmlx-rdma, rmlx-core, rmlx-distributed, rmlx-nn, rmlx-cli, rmlx-macros)와 부가 디렉토리로 이루어져 있습니다.

---

## 워크스페이스 전체 구조

```
rmlx/
├── Cargo.toml                    # workspace 루트
├── PLAN.md                       # 구현 계획서
├── crates/
│   ├── rmlx-metal/               # Metal GPU 추상화
│   │   ├── Cargo.toml            # deps: objc2-metal 0.3, objc2 0.6, block2 0.6, objc2-foundation 0.3, bytemuck
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── device.rs         # MTLDevice 래퍼, 아키텍처 감지
│   │       ├── queue.rs          # MTLCommandQueue 관리 (듀얼 큐 지원)
│   │       ├── command.rs        # CommandBuffer/Encoder 추상화
│   │       ├── compute_pass.rs   # ComputePass zero-cost 뉴타입 래퍼
│   │       ├── buffer.rs         # MTLBuffer, zero-copy 생성
│   │       ├── managed_buffer.rs # 관리 버퍼 수명 관리
│   │       ├── event.rs          # MTLSharedEvent 래퍼
│   │       ├── fence.rs          # MTLFence + fast-fence (shared buffer spin)
│   │       ├── pipeline.rs       # ComputePipelineState 생성
│   │       ├── pipeline_cache.rs # PipelineCache — PSO 캐싱 계층
│   │       ├── library.rs        # MTLLibrary 로드/JIT 컴파일
│   │       ├── library_cache.rs  # 라이브러리 캐싱
│   │       ├── types.rs          # 타입 별칭 (MtlDevice, MtlBuffer, MtlPipeline 등)
│   │       ├── autorelease.rs    # Autorelease 풀 관리
│   │       ├── capture.rs        # Metal GPU 캡처 매니저
│   │       ├── msl_version.rs    # MSL 버전 감지
│   │       ├── self_check.rs     # 시동 진단 (startup diagnostics)
│   │       ├── stream.rs         # 스트림 관리
│   │       ├── batcher.rs        # CommandBatcher — CB 그룹화
│   │       ├── exec_graph.rs     # ExecGraph — ICB replay 지원 결정론적 op 재생
│   │       ├── icb.rs            # Indirect Command Buffer 지원
│   │       ├── icb_sparse.rs     # Sparse ICB — MoE 가변 디스패치 패턴용
│   │       └── metal4/           # Feature-gated Metal 4 (macOS 26+) 지원
│   │           ├── mod.rs
│   │           ├── command.rs    # MTL4CommandBuffer/Allocator/Queue
│   │           ├── compiler.rs   # MTL4Compiler/CompilerTask
│   │           ├── compute.rs    # MTL4ComputePipeline/ComputeCommandEncoder
│   │           └── counter_heap.rs # MTL4Counters
│   │
│   ├── rmlx-alloc/               # 메모리 할당자
│   │   ├── Cargo.toml            # deps: rmlx-metal, libc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── allocator.rs      # MetalAllocator — 통합 할당 진입점
│   │       ├── zero_copy.rs      # posix_memalign + newBufferWithBytesNoCopy
│   │       ├── buffer_pool.rs    # 이중 등록 버퍼 풀 (Metal + ibv_mr)
│   │       ├── cache.rs          # MLX-style size-binned 캐시
│   │       ├── bfc.rs            # Best-Fit Coalescing 할당자
│   │       ├── small_alloc.rs    # 소규모 할당 fast-path
│   │       ├── residency.rs      # ResidencyManager (MTLResidencySet 백엔드)
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
│   │       │   ├── mod.rs          # Op 레지스트리 + shape-aware GEMM 디스패치 테이블
│   │       │   ├── matmul.rs       # GEMM (Metal dispatch via ComputePass)
│   │       │   ├── gemv.rs         # GEMV 경로
│   │       │   ├── quantized.rs    # QMM (4bit, 8bit, FP4, FP8)
│   │       │   ├── softmax.rs
│   │       │   ├── rms_norm.rs
│   │       │   ├── layer_norm.rs   # LayerNorm
│   │       │   ├── rope.rs
│   │       │   ├── binary.rs       # element-wise 연산
│   │       │   ├── unary.rs        # 단항 연산
│   │       │   ├── reduce.rs
│   │       │   ├── argreduce.rs    # ArgReduce (argmin/argmax)
│   │       │   ├── copy.rs
│   │       │   ├── indexing.rs
│   │       │   ├── concat.rs       # 텐서 연결
│   │       │   ├── select.rs       # 텐서 선택
│   │       │   ├── slice.rs        # 텐서 슬라이싱
│   │       │   ├── sort.rs         # 텐서 정렬
│   │       │   ├── scan.rs         # Prefix scan
│   │       │   ├── random.rs       # 난수 생성
│   │       │   ├── silu.rs         # SiLU activation + fused SwiGLU
│   │       │   ├── gelu.rs         # GELU activation
│   │       │   ├── fp8.rs          # FP8 양자화 연산
│   │       │   ├── conv.rs         # Conv1d/Conv2d
│   │       │   ├── conv_tiled.rs   # 타일 컨볼루션
│   │       │   ├── gather_mm.rs    # 그룹/수집 행렬 곱셈
│   │       │   ├── sdpa.rs         # Fused Scaled Dot-Product Attention
│   │       │   ├── sdpa_backward.rs # SDPA 역전파
│   │       │   ├── topk_route.rs   # GPU-native top-k 라우팅
│   │       │   ├── moe_kernels.rs  # MoE 전용 커널
│   │       │   ├── fused.rs        # 융합 커널 연산
│   │       │   ├── buffer_slots.rs # 버퍼 슬롯 관리
│   │       │   └── vjp_gpu.rs      # VJP autodiff GPU 연산
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
│   │       ├── init.rs           # 분산 초기화
│   │       ├── transport.rs      # 전송 추상화
│   │       ├── moe_exchange.rs   # MoE 데이터 교환 프리미티브
│   │       ├── moe_policy.rs     # MoE 라우팅 정책
│   │       ├── moe_kernels.rs    # MoE Metal 커널 관리
│   │       ├── ep_runtime.rs     # Expert Parallelism 런타임
│   │       ├── fp8_exchange.rs   # FP8 와이어 교환
│   │       ├── slab_ring.rs      # RDMA용 slab 링 버퍼
│   │       ├── v3_protocol.rs    # 가변 길이 v3 교환 프로토콜
│   │       ├── pipeline.rs       # layer-level compute↔RDMA 파이프라인
│   │       ├── sparse_guard.rs   # 희소 디스패치 안전 가드
│   │       ├── credit_manager.rs # 흐름 제어 크레딧
│   │       ├── health.rs         # 분산 건강 모니터링
│   │       ├── perf_counters.rs  # 성능 카운터
│   │       ├── progress_tracker.rs # 진행 추적
│   │       ├── warmup.rs         # 분산 워밍업
│   │       └── metrics.rs        # 분산 메트릭 수집
│   │
│   ├── rmlx-nn/                  # 신경망 레이어
│   │   ├── Cargo.toml            # deps: rmlx-core
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── linear.rs         # Linear (quantized 지원, prepare_weight_t())
│   │       ├── quantized_linear.rs # QuantizedLinear
│   │       ├── embedding.rs
│   │       ├── attention.rs      # Multi-head/GQA attention
│   │       ├── mla.rs            # Multi-Latent Attention
│   │       ├── sliding_window.rs # 슬라이딩 윈도우 어텐션
│   │       ├── layer_norm.rs     # LayerNorm
│   │       ├── rms_norm.rs       # RMSNorm
│   │       ├── rope.rs           # Rotary Position Embedding
│   │       ├── activations.rs    # 14개 활성화 함수
│   │       ├── conv.rs           # Conv1d/Conv2d
│   │       ├── transformer.rs    # Transformer block (forward_graph(), forward_into_cb())
│   │       ├── moe.rs            # MoE gate + expert 라우팅 + GPU 라우팅
│   │       ├── moe_pipeline.rs   # MoE 파이프라인 실행
│   │       ├── expert_group.rs   # Expert 그룹 관리
│   │       ├── parallel.rs       # Megatron-LM TP (Column/RowParallel)
│   │       ├── paged_kv_cache.rs # 페이지드 KV 캐시
│   │       ├── prefix_cache.rs   # 접두사 캐시
│   │       ├── dynamic.rs        # 동적 shape
│   │       ├── sampler.rs        # 토큰 샘플링
│   │       ├── scheduler.rs      # 요청 스케줄링
│   │       ├── prefill_plan.rs   # 프리필 계획
│   │       ├── prefill_pool.rs   # 프리필 풀 관리
│   │       ├── gguf_loader.rs    # GGUF 모델 로딩
│   │       ├── safetensors_loader.rs # Safetensors 모델 로딩
│   │       └── models/
│   │           ├── mod.rs
│   │           ├── qwen.rs       # Qwen 3.5
│   │           ├── deepseek.rs   # DeepSeek-V3 (MoE)
│   │           └── mixtral.rs    # Mixtral (MoE)
│   │
│   ├── rmlx-macros/              # Proc-macro 유틸리티
│   │   ├── Cargo.toml            # deps: syn, quote, proc-macro2
│   │   └── src/
│   │       └── lib.rs            # 프레임워크 타입용 derive 매크로
│   │
│   └── rmlx-cli/                 # 네이티브 CLI 도구
│       ├── Cargo.toml            # deps: clap, serde, serde_json
│       └── src/
│           └── main.rs           # rmlx config, rmlx launch 서브커맨드
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
| **목적** | Apple Metal API를 Rust로 안전하게 추상화합니다. `objc2-metal` 0.3을 기반으로 MTLDevice, MTLCommandQueue, MTLBuffer, MTLSharedEvent 등을 래핑합니다. `compute_pass.rs`의 `ComputePass` 뉴타입 래퍼가 zero-cost 추상화 계층을 제공하며, `types.rs`에서 타입 별칭(`MtlDevice`, `MtlBuffer`, `MtlPipeline` 등)을 정의합니다. 메인 코드베이스에 18개 `unsafe` 블록이 있으며, 모두 100% 안전한 public API 뒤에 캡슐화되어 있습니다. |
| **핵심 모듈** | `device.rs` (디바이스 + 아키텍처 감지), `queue.rs` (듀얼 큐 관리), `command.rs` (CommandBuffer 추상화), `compute_pass.rs` (ComputePass zero-cost 래퍼), `types.rs` (MtlDevice, MtlBuffer, MtlPipeline 타입 별칭), `event.rs` (MTLSharedEvent 래퍼), `pipeline.rs` + `pipeline_cache.rs` (PSO 생성 + 캐싱), `library_cache.rs` (라이브러리 캐싱), `self_check.rs` (시동 진단), `batcher.rs` (CommandBatcher), `exec_graph.rs` (ICB replay 지원 ExecGraph), `icb.rs` + `icb_sparse.rs` (Indirect Command Buffer + sparse 변형), `autorelease.rs` (autorelease 풀), `capture.rs` (GPU 캡처), `managed_buffer.rs` (관리 버퍼), `stream.rs` (스트림 관리), `metal4/` (feature-gated Metal 4 지원) |
| **의존성** | objc2-metal 0.3, objc2 0.6, block2 0.6, objc2-foundation 0.3, bytemuck |
| **현재 상태** | 완료 — GpuDevice, StreamManager, DeviceStream, GpuEvent, SharedEvent 동기화, 듀얼 큐 파이프라인, 시동 진단, ExecGraph (5 CBs/layer + ICB replay), CommandBatcher, ICB + ICB Sparse, Metal 4 feature-gated 지원 (MTL4CommandAllocator, MTL4ComputePipeline, MTL4Counters) 전체 구현 |

---

### rmlx-alloc — 메모리 할당자

| 항목 | 내용 |
|------|------|
| **목적** | Zero-copy 메모리 할당 및 버퍼 풀 관리를 담당합니다. `posix_memalign` → `newBufferWithBytesNoCopy` 패턴으로 복사 없는 Metal 버퍼를 생성하고, RDMA `ibv_mr` 이중 등록을 지원합니다. |
| **핵심 모듈** | `allocator.rs` (MetalAllocator 통합 진입점), `zero_copy.rs` (ZeroCopyBuffer, CompletionFence), `buffer_pool.rs` (이중 등록 버퍼 풀), `cache.rs` (size-binned 캐시), `bfc.rs` (best-fit coalescing 할당자), `small_alloc.rs` (소규모 할당 fast-path), `residency.rs` (MTLResidencySet 백엔드 ResidencyManager), `stats.rs` (할당 통계), `leak_detector.rs` (메모리 누수 감지) |
| **의존성** | `rmlx-metal`, libc |
| **현재 상태** | 완료 — MetalAllocator, ZeroCopyBuffer, DualRegPool, size-binned 캐시, BFC 할당자, 소규모 할당 fast-path, ResidencyManager (MTLResidencySet 백엔드), 누수 감지 전체 구현 |

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
| **핵심 모듈** | `dtype.rs` (f32, f16, bf16, quantized), `array.rs` (N-dim array), `ops/` (matmul, gemv, softmax, rms_norm, layer_norm, rope, quantized, binary, unary, reduce, argreduce, copy, indexing, concat, select, slice, sort, scan, random, sdpa, sdpa_backward, silu, gelu, fp8, conv, conv_tiled, gather_mm, topk_route, moe_kernels, fused, buffer_slots, vjp_gpu 등 32개 op 모듈), `kernels/` (AOT/JIT 커널 관리), `graph.rs` (연산 그래프), `scheduler.rs` (스트림별 스케줄러), `vjp.rs` (VJP autodiff), `lora.rs` (LoRA fine-tuning), `logging.rs` (structured logging), `metrics.rs` (메트릭 수집), `precision_guard.rs` (정밀도 가드), `shutdown.rs` (graceful shutdown) |
| **의존성** | `rmlx-metal`, `rmlx-alloc` |
| **현재 상태** | 완료 — Array 타입, 32개 op 모듈, shape-aware GEMM 디스패치 테이블 (M 차원 기반 Tiled/Split-K/NAX 자동 선택), ExecMode, CommandBufferHandle, LaunchResult, VJP autodiff, LoRA, 프로덕션 하드닝 전체 구현 |

---

### rmlx-distributed — 분산 프리미티브

| 항목 | 내용 |
|------|------|
| **목적** | 분산 추론에 필요한 통신 프리미티브와 MoE Expert Parallelism을 구현합니다. layer-level 파이프라인을 통해 compute와 RDMA를 오버랩합니다. |
| **핵심 모듈** | `group.rs` (rank/world_size 추상화), `init.rs` (초기화), `transport.rs` (전송 추상화), `moe_exchange.rs` (MoE 데이터 교환), `moe_policy.rs` (MoE 라우팅 정책), `moe_kernels.rs` (MoE Metal 커널), `ep_runtime.rs` (EP 런타임), `fp8_exchange.rs` (FP8 와이어 교환), `slab_ring.rs` (slab 링 버퍼), `v3_protocol.rs` (가변 길이 교환), `pipeline.rs` (compute-RDMA 오버랩), `sparse_guard.rs` (희소 디스패치 가드), `credit_manager.rs` (흐름 제어), `health.rs` (건강 모니터링), `perf_counters.rs` (성능 카운터), `progress_tracker.rs` (진행 추적), `warmup.rs` (분산 워밍업), `metrics.rs` (분산 메트릭) |
| **의존성** | `rmlx-core`, `rmlx-rdma` |
| **현재 상태** | 완료 — EP dispatch/combine, EP 런타임, FP8 와이어 교환, slab 링 버퍼, v3 프로토콜, compute-RDMA 파이프라인, MoE 교환, 분산 메트릭 전체 구현 |

---

### rmlx-nn — 신경망 레이어

| 항목 | 내용 |
|------|------|
| **목적** | Transformer 기반 아키텍처의 신경망 레이어를 제공합니다. Linear, Attention, MoE 등의 고수준 모듈과 Qwen 3.5, DeepSeek-V3, Mixtral 등 모델 아키텍처를 포함합니다. |
| **핵심 모듈** | `linear.rs` (Linear, `prepare_weight_t()`), `quantized_linear.rs` (QuantizedLinear), `attention.rs` (Multi-head/GQA), `mla.rs` (Multi-Latent Attention), `sliding_window.rs` (슬라이딩 윈도우), `layer_norm.rs` (LayerNorm), `rms_norm.rs` (RMSNorm), `rope.rs` (RoPE), `activations.rs` (14개 활성화), `conv.rs` (Conv1d/Conv2d), `transformer.rs` (Transformer block, `forward_graph()`, `forward_into_cb()`), `moe.rs` (gate + routing + GPU routing), `moe_pipeline.rs` (MoE 파이프라인), `expert_group.rs` (expert 그룹), `paged_kv_cache.rs` (페이지드 KV 캐시), `prefix_cache.rs` (접두사 캐시), `dynamic.rs` (동적 shape), `sampler.rs` (토큰 샘플링), `scheduler.rs` (요청 스케줄링), `prefill_plan.rs` + `prefill_pool.rs` (프리필 최적화), `gguf_loader.rs` (GGUF 로딩), `safetensors_loader.rs` (safetensors 로딩), `parallel.rs` (ColumnParallel/RowParallel), `models/` (qwen.rs, deepseek.rs, mixtral.rs) |
| **의존성** | `rmlx-core` |
| **현재 상태** | 완료 — Transformer 블록, Linear/QuantizedLinear/Attention/MLA/MoE 레이어, KV 캐시 (static/rotating/paged/prefix), 슬라이딩 윈도우, LayerNorm, 14개 활성화, 병렬 Linear, GGUF/safetensors 로더, Qwen/DeepSeek-V3/Mixtral/Kimi 모델, ExecGraph 호환 `forward_graph()`, 가중치 사전 캐싱 전체 구현 |

---

### rmlx-macros — Proc-Macro 유틸리티

| 항목 | 내용 |
|------|------|
| **목적** | 프레임워크 타입을 위한 절차적 매크로(derive 매크로)를 제공하여 크레이트 간 보일러플레이트를 줄입니다. |
| **핵심 모듈** | `lib.rs` (proc-macro 진입점) |
| **의존성** | syn 2, quote 1, proc-macro2 1 |
| **현재 상태** | 완료 — 프레임워크 타입용 derive 매크로 제공 |

---

### rmlx-cli — 네이티브 CLI 도구

| 항목 | 내용 |
|------|------|
| **목적** | 분산 클러스터 관리를 위한 `rmlx` 명령줄 인터페이스를 제공합니다. MLX의 `mlx.distributed_config`와 `mlx.launch`를 모델로 한 `rmlx config`(호스트파일 생성 및 기본 설정)과 `rmlx launch`(멀티 노드 프로세스 오케스트레이션)를 구현합니다. |
| **핵심 명령** | `rmlx config` (호스트 탐색, RDMA 백엔드 설정, 호스트파일 출력), `rmlx launch` (SSH 기반 멀티 노드 명령 전달) |
| **의존성** | `rmlx-distributed`, `rmlx-rdma` |
| **현재 상태** | 초기 구현 — config 및 launch 서브커맨드 동작 |

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
    "crates/rmlx-cli",
    "crates/rmlx-macros",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
license = "MIT"

[workspace.dependencies]
objc2 = "0.6"
objc2-metal = "0.3"
objc2-foundation = "0.3"
block2 = "0.6"
bytemuck = { version = "1", features = ["derive"] }
libc = "0.2"
libloading = "0.8"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
crc32fast = "1"
half = "2"
sha2 = "0.10"
dirs = "5"
tracing = "0.1"
smallvec = { version = "1", features = ["const_generics"] }
bumpalo = "3"
```

`workspace.dependencies`를 활용하여 크레이트 간 의존성 버전을 통일합니다. 각 크레이트의 `Cargo.toml`에서 `objc2-metal.workspace = true` 형태로 참조할 수 있습니다.

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
