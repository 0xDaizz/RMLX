# Implementation Roadmap — Phases 0-9B Complete

The rmlx project implementation roadmap. All phases have been completed through 9B-opt. Implementation phases: 9 (all complete through 9B-opt).

---

## Overview

| Phase | Name | Key Content | Prerequisites | Status |
|:-----:|------|------------|:------------:|:------:|
| 0 | Scaffolding | Workspace, metal-rs wrappers, CI | -- | Complete |
| 1 | Zero-Copy + RDMA | ZeroCopyBuffer, DualRegPool, ibverbs FFI, blocking_exchange | Phase 0 | Complete |
| 1-hotfix | IbvSendWr FFI Layout Fix | FFI layout fix | Phase 1 | Complete |
| 2A | Metal Compute Foundation | Shader vendoring, DType/Array, KernelRegistry | Phase 0 | Complete |
| 2A | Metal Compute Kernels | 7 GPU kernels + integration tests | Phase 2A foundation | Complete |
| 2B | Steel GEMM + Quantization | Steel GEMM, quantized matmul, indexing | Phase 2A | Complete |
| 3 | Pipeline Overlap | MTLSharedEvent, dual-queue pipeline | Phase 2 | Complete |
| 4 | Expert Parallelism | EP dispatch/combine, 3-zone auto backend, sparse dispatch | Phase 1 + 3 | Complete |
| 5A | NN Inference Core | LLaMA, Qwen, DeepSeek, Mixtral | Phase 4 | Complete |
| 5B | Model Serving | rmlx-lm (separate repo ~/rmlx-lm) | Phase 5A | Complete |
| 6 | Multi-Port | Dual TB5 multi-port striping, multi-node topology | Phase 4 | Complete |
| 7A | Production Hardening | Hardening, observability | Phase 5 | Complete |
| 7B | VJP Autodiff | VJP autodiff + LoRA fine-tuning | Phase 7A | Complete |
| 8 | KV Cache + API Surface | KV cache, parallel linear, API ergonomics | Phase 7B | Complete |
| 9A | GPU Pipeline — ExecGraph | CommandBatcher, ExecGraph, ICB, `_into_cb()` pattern | Phase 8 | Complete |
| 9B-opt | GPU Pipeline — Optimization | Weight pre-caching, contiguous transpose, 16.15x speedup | Phase 9A | Complete |

---

## Phase Completion History

| Phase | Commit | Tests | Status |
|-------|--------|-------|--------|
| Phase 0: Scaffolding + Metal GPU abstraction | 7071c73 | baseline | Complete |
| Phase 1: Zero-copy memory + RDMA ibverbs | d541bb3 | + alloc/rdma tests | Complete |
| Phase 1-hotfix: IbvSendWr FFI layout fix | 9cca9a9 | 23 tests | Complete |
| Phase 2A-1~4: Shader vendoring, DType/Array, KernelRegistry | 3179bde | foundation | Complete |
| Phase 2A-5~9: 7 GPU kernels + integration tests | 5ef6a07 | 40 tests | Complete |
| Phase 2B: Steel GEMM, quantized matmul, indexing | e4d9c14 | 43 tests | Complete |
| Phase 3: SharedEvent sync + dual queue + layer pipeline | f9cadcf | 52 tests | Complete |
| Phase 4: EP 3-Zone dispatch + MoE exchange | 6fb3296 | 62 tests | Complete |
| Phase 5A: rmlx-nn inference core (LLaMA, Qwen, DeepSeek, Mixtral) | d126aaf | + nn tests | Complete |
| Phase 5B: rmlx-lm serving engine (separate repo ~/rmlx-lm) | 98e50c1 | 75 tests | Complete |
| Phase 6: Dual TB5 multi-port striping + multi-node topology | 8c8b25f | + distributed tests | Complete |
| Phase 7A: Production hardening / observability | 0fa70bb | 98 tests | Complete |
| Phase 7B: VJP autodiff + LoRA fine-tuning | 025ed8f | 108 tests | Complete |
| Phase 8: KV Cache + API Surface | squash merge | 339 tests | Complete |
| Phase 9A: GPU Pipeline — ExecGraph | Phase 9 merge commit | 339+ tests | Complete |
| Phase 9B-opt: GPU Pipeline — Optimization | optimization merge | 339+ tests | Complete |

---

## Phase Dependency Diagram

```mermaid
graph LR
    P0["Phase 0<br/>Scaffolding<br/>Complete"]
    P1["Phase 1<br/>Zero-Copy + RDMA<br/>Complete"]
    P2["Phase 2<br/>Metal Compute<br/>Complete"]
    P3["Phase 3<br/>Pipeline Overlap<br/>Complete"]
    P4["Phase 4<br/>Expert Parallelism<br/>Complete"]
    P5A["Phase 5A<br/>NN Inference Core<br/>Complete"]
    P5B["Phase 5B<br/>Model Serving<br/>Complete"]
    P6["Phase 6<br/>Multi-Port<br/>Complete"]
    P7["Phase 7<br/>Production<br/>Complete"]
    P8["Phase 8<br/>KV Cache + API<br/>Complete"]
    P9["Phase 9<br/>GPU Pipeline<br/>Complete"]

    P0 --> P1
    P0 --> P2
    P2 --> P3
    P1 --> P4
    P3 --> P4
    P4 --> P5A
    P5A --> P5B
    P4 --> P6
    P5B --> P7
    P7 --> P8
    P8 --> P9

    style P0 fill:#22c55e,color:#fff
    style P1 fill:#22c55e,color:#fff
    style P2 fill:#22c55e,color:#fff
    style P3 fill:#22c55e,color:#fff
    style P4 fill:#22c55e,color:#fff
    style P5A fill:#22c55e,color:#fff
    style P5B fill:#22c55e,color:#fff
    style P6 fill:#22c55e,color:#fff
    style P7 fill:#22c55e,color:#fff
    style P8 fill:#22c55e,color:#fff
    style P9 fill:#22c55e,color:#fff
```

---

## Phase 0: Scaffolding — Complete (`7071c73`)

### Goal

Establish the Cargo workspace structure, validate metal-rs basic operations, and set up CI.

### Key Deliverables

- Cargo workspace initialization (6 crate skeletons)
- `rmlx-metal`: MTLDevice creation, basic command buffer/encoder wrappers
- `rmlx-metal`: Simple Metal compute kernel execution (vector add)
- Build system: `.metal` -> `.metallib` AOT compilation pipeline in `build.rs`
- CI: GitHub Actions (macOS runner, `cargo test`, `cargo clippy`)

### Definition of Done (DoD)

- [x] `cargo build --workspace` succeeds (0 errors)
- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `cargo test --workspace` -- `test_basic_metal_compute` PASS
- [x] `build.rs` `.metal` -> `.metallib` AOT compilation succeeds
- [x] Codex review: SAFETY comments present on unsafe blocks

---

## Phase 1: Zero-Copy + RDMA — Complete (`d541bb3`, hotfix `9cca9a9`)

### Goal

Convert PoC Phase 1-4 validation results into production-quality code. Implement zero-copy transfers by registering GPU buffers directly with RDMA.

### Key Deliverables

- `rmlx-alloc`: ZeroCopyBuffer (`posix_memalign` + NoCopy)
- `rmlx-alloc`: DualRegPool (Metal + `ibv_mr` dual-registered pool)
- `rmlx-alloc`: MetalAllocator (heap + cache, MLX compatible)
- `rmlx-rdma`: ibverbs FFI bindings (`bindgen`)
- `rmlx-rdma`: IbContext, PD, CQ, UC QP wrappers
- `rmlx-rdma`: `ibv_reg_mr` wrapper + dual registration tests
- `rmlx-rdma`: `blocking_exchange` (2-phase count -> payload)
- `rmlx-rdma`: ConnectionManager (`hosts.json` parsing, warmup)
- Integration test: 2-node zero-copy RDMA round-trip

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `test_zero_copy_buffer_lifecycle` -- InFlightToken drop-then-free verified
- [x] `test_dual_registration` -- Metal + ibv_mr same-address verified
- [x] `test_rdma_exchange_2node` -- 4MB round-trip, 0 mismatch
- [x] `test_rdma_startup_probe` -- GID/MR/QP runtime discovery succeeds
- [x] `test_recv_before_send_invariant` -- Error returned when recv not posted
- [x] Benchmark: RDMA bandwidth > 6 GB/s (single port)
- [x] Codex review: FFI boundary safety, lifetime verification

---

## Phase 2: Metal Compute — Complete (2A: `3179bde`, `5ef6a07` / 2B: `e4d9c14`)

### Goal

Build the core Metal kernel execution pipeline needed for LLM inference. Reuse MLX's Metal shaders to dispatch 10 kernel types from Rust.

### Key Deliverables

- `rmlx-core`: Array type (N-dim, dtype, ownership management)
- `rmlx-core`: dtype system (f32, f16, bf16, q4_0, q4_1, q8_0)
- MLX `.metal` kernel porting (Rust dispatch wrappers):
  - matmul (GEMM/GEMV)
  - quantized matmul (QMM 4bit/8bit)
  - softmax
  - RMS normalization
  - RoPE (rotary position embedding)
  - Element-wise binary ops (add, mul, etc.)
  - reduce (sum, max, argmax)
  - copy / transpose
  - indexing (gather, scatter)
- `rmlx-core`: KernelRegistry (AOT + JIT)
- `rmlx-core`: Per-stream CommandEncoder management
- Benchmarks: Per-kernel performance comparison vs. MLX

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] 10 kernels each within +/-5% of MLX performance
- [x] `test_matmul_correctness` -- fp16/bf16 accuracy (ulp < 2)
- [x] `test_quantized_matmul` -- q4/q8 accuracy
- [x] `test_dispatch_geometry` -- threadgroup vs. thread size verified
- [x] Codex review: kernel binding index consistency verified

---

## Phase 3: Pipeline Overlap — Complete (`f9cadcf`)

### Goal

Implement MTLSharedEvent-based GPU synchronization and dual queue pipeline to overlap compute and RDMA transfers.

### Key Deliverables

- `rmlx-metal`: GpuEvent (MTLSharedEvent wrapper)
- `rmlx-metal`: FenceImpl (fast fence + SharedEvent fallback)
- `rmlx-metal`: StreamManager (dual queue management)
- `rmlx-distributed`: LayerPipeline (compute <-> RDMA overlap)
- GPU -> CPU sync: event spin-wait (263.9 us target)
- GPU -> GPU sync: encodeSignal/WaitForEvent cross-queue

Pipeline overlap effect:

```
Non-pipelined: 60 x (20ms + 7ms) = 1,620ms
Pipelined:     60 x 20ms + 7ms   = 1,207ms  (25% improvement)
```

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `test_shared_event_latency` -- spin-wait < 280 us
- [x] `test_dual_queue_overlap` -- concurrent execution of both queues confirmed
- [x] `test_layer_pipeline_correctness` -- pipeline result == serial result
- [x] `test_event_deadline_cancel` -- timeout/cancel behavior confirmed
- [x] Benchmark: sync latency histogram (p50/p95/p99)
- [x] Codex review: synchronization protocol correctness

---

## Phase 4: Expert Parallelism — Complete (`6fb3296`)

### Goal

Reimplement MLX EP optimizations in RMLX, achieving additional performance gains through zero-copy. Achieve 2-node Mixtral decode step < 35ms.

### Key Deliverables

- `rmlx-distributed`: Group abstraction (rank, world_size, EP topology)
- `rmlx-distributed`: AllToAll primitive
- `rmlx-distributed/moe`: MoeDispatchExchange
  - CPU backend (N <= 64)
  - Metal backend (N >= 320, 7 kernels)
  - Byte threshold for intermediate range
- `rmlx-distributed/moe`: MoeCombineExchange
  - Single-source weighted sum
  - Dual-source weighted sum (local + remote, zero-copy)
- `rmlx-distributed/moe`: MoePolicy (3-zone auto + cooldown)
- 7 MoE Metal kernels JIT-compiled

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `test_1rank_vs_2rank_parity` -- single-node result == 2-node EP result
- [x] `test_3zone_policy` -- correct backend selection for N=1/64/256/1024
- [x] `test_sparse_dispatch_correctness` -- matmul scatter == dense result
- [x] `test_interleaved_exchange_stress` -- 1000 consecutive exchanges with 0 errors
- [x] `test_capacity_overflow_detection` -- overflow_count metric accuracy
- [x] Benchmark: 2-node decode step < 35ms
- [x] Codex review: exchange protocol, metric collection accuracy

---

## Phase 5A: NN Inference Core — Complete (`d126aaf`)

### Goal

Implement core neural network modules required for LLM inference in the rmlx-nn crate.

### Key Deliverables

**rmlx framework** (`~/rmlx/`):
- `rmlx-nn`: Transformer block (Linear, Attention, FFN, MoE)
- `rmlx-nn`: Model architectures (LLaMA, Qwen, DeepSeek-V3, Mixtral)

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] Model architecture accuracy verification
- [x] Codex review: nn module safety

---

## Phase 5B: Model Serving — Complete (`98e50c1`, separate repo ~/rmlx-lm)

### Goal

Build a complete LLM inference serving engine. Implemented in a separate repository (`~/rmlx-lm/`) that references rmlx as a dependency.

### Key Deliverables

**rmlx-lm serving application** (`~/rmlx-lm/`):
- safetensors model loader (quantized weight decoding)
- KV cache management (paged attention)
- Token sampler (temperature, top-p, top-k, repetition penalty)
- Tokenizer integration (tokenizers-rs, HuggingFace compatible)
- Continuous batching scheduler
- HTTP server (OpenAI Chat Completions API compatible)
- CLI interface (generate, serve, benchmark)

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0 (both repositories)
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings (both)
- [x] `test_e2e_single_token_generation` -- single-token generation accuracy
- [x] `test_continuous_batching` -- concurrent multi-request handling
- [x] `test_kv_cache_paged` -- cache allocation/deallocation stress
- [x] `test_openai_api_compat` -- /v1/chat/completions response format
- [x] `test_metrics_prometheus` -- /metrics endpoint parsing succeeds
- [x] Benchmark: 2-node EP decode > 28 tok/s
- [x] Codex review: serving stability, memory leaks

---

## Phase 6: Multi-Port — Complete (`8c8b25f`)

### Goal

Expand bandwidth by utilizing multiple TB5 ports and support 3+ nodes. Achieve ~1.8x bandwidth over single port with dual port striping.

### Key Deliverables

- `rmlx-rdma/multi_port`: Dual TB5 port striping
- `rmlx-rdma/multi_port`: Automatic striping based on transfer size (N >= 8 threshold)
- Multi-node topology manager (ring, mesh, hybrid)
- 3+ node EP support (all-to-all with > 2 ranks)

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `test_dual_port_striping` -- 2-port concurrent transfer, data integrity
- [x] `test_single_port_fallback` -- graceful fallback on 1-port failure
- [x] Benchmark: dual-port bandwidth > 12 GB/s
- [x] Codex review: port independence, error isolation

---

## Phase 7A: Production Hardening / Observability — Complete (`0fa70bb`)

### Goal

Ensure production stability and observability.

### Key Deliverables

- Structured logging (`tracing` crate)
- Metrics collection (Prometheus compatible)
- Graceful shutdown + error recovery
- GID table corruption detection and automatic alerts
- Memory leak detection (allocation statistics-based)

### Definition of Done (DoD)

- [x] Structured logging applied across all crates
- [x] Prometheus /metrics endpoint operational
- [x] Graceful shutdown scenario tested

---

## Phase 7B: VJP Autodiff + LoRA Fine-tuning — Complete (`025ed8f`)

### Goal

Build a VJP framework and LoRA fine-tuning foundation for training support.

### Key Deliverables

- VJP (Vector-Jacobian Product) framework
- Basic training loop (LoRA fine-tuning)

### Definition of Done (DoD)

- [x] VJP gradient accuracy for basic operations (matmul, softmax)
- [x] LoRA fine-tuning functional verification

---

## Phase 8: KV Cache + API Surface — Complete (squash merged to main)

### Goal

Add incremental decoding support via KV cache in rmlx-nn and improve API ergonomics across the framework.

### Key Deliverables

- `rmlx-nn`: `LayerKvCache` struct for incremental KV caching in attention
- `rmlx-nn`: Cache-aware `forward()` in Attention, TransformerBlock, TransformerModel
- `rmlx-nn`: Per-expert MoE routing metrics (`MoeForwardMetrics.expert_tokens`)
- `rmlx-nn`: Megatron-LM parallel linear layers (`parallel.rs`: ColumnParallelLinear, RowParallelLinear)
- `rmlx-distributed`: Per-expert histogram in `MoeMetrics`
- `rmlx-metal`: Top-level re-exports (`GpuDevice`, `GpuEvent`, `Architecture`)
- `rmlx-core`: `prelude` module (Array, DType, KernelError, KernelRegistry)
- `rmlx-nn`: Re-exports (`LayerKvCache`, `FeedForward`)

### Definition of Done (DoD)

- [x] `cargo fmt --all --check` -- diff 0
- [x] `cargo clippy --workspace -- -D warnings` -- 0 warnings
- [x] `cargo test --workspace` -- 339 tests passing, 0 failures
- [x] KV cache: decode step processes only the last token (O(n^2) → O(n))
- [x] Backward compatible: cache=None preserves existing behavior
- [x] Codex review: 0 Critical/High issues

---

## Phase 9: GPU Pipeline — Complete

### Phase 9A: ExecGraph + CommandBatcher

#### Goal

Eliminate per-op CPU overhead by batching multiple GPU operations into minimal command buffers using ExecGraph.

#### Key Deliverables

- `rmlx-metal`: `CommandBatcher` — batches encoder work into shared command buffers
- `rmlx-metal`: `ExecGraph` — pre-built execution graph that replays deterministic op sequences
- `rmlx-metal`: `IcbBuilder`/`IcbReplay`/`IcbCache` — Indirect Command Buffer support
- `rmlx-core`: `_into_cb()` pattern for all 14 ops — encode into caller's command buffer
- `rmlx-nn`: `forward_graph()` for Attention, TransformerBlock, TransformerModel
- `rmlx-nn`: `forward_into_cb()` for Linear
- Benchmark: 65 CBs/layer → 5 CBs/layer (92.3% reduction)

### Phase 9B-opt: Weight Pre-caching + Optimization

#### Goal

Pre-cache contiguous transposed weight matrices to eliminate transpose overhead during inference.

#### Key Deliverables

- `rmlx-nn`: `prepare_weight_t()` / `weight_transposed_contiguous()` for Linear
- `rmlx-nn`: `prepare_weights_for_graph()` for TransformerModel/Block/Attention/FeedForward
- Benchmark: 110.4ms → 6.8ms per layer (16.15x speedup)
- Numerical parity: max_diff=6.4e-6

#### Definition of Done (DoD)

- [x] 16.15x speedup (110.4ms → 6.8ms)
- [x] 92.3% CB reduction (65 → 5)
- [x] Numerical parity (max_diff=6.4e-6)
- [x] All 339+ tests passing

---

## Phase 10: Flash Attention + Paged KV Cache — Planned

### Goal

Implement Flash Attention 2 with tiled K/V processing and paged attention for dynamic KV cache management.

### Key Deliverables

- Flash Attention 2 Metal kernel (tiled Q/K/V with online softmax)
- Paged KV cache with block-level allocation/deallocation
- Integration with ExecGraph pipeline

---

## Phase 11: Continuous Batching + Serving Optimization — Planned

### Goal

Add continuous batching for production serving and optimize the full inference pipeline.

### Key Deliverables

- Continuous batching scheduler (in-flight request management)
- Speculative decoding (draft + verify pattern)
- Request-level KV cache management

---

## Phase 12: Advanced Quantization + Model Support — Planned

### Goal

Expand quantization format support and add more model architectures.

### Key Deliverables

- GGUF format support
- AWQ/GPTQ quantization
- FP8 support
- Additional model architectures

---

## CI Required Test Matrix

The CI pipeline applied across all phases:

```yaml
# .github/workflows/ci.yml
jobs:
  build-and-test:
    runs-on: macos-15  # Apple Silicon runner
    steps:
      - cargo build --workspace
      - cargo test --workspace
      - cargo clippy --workspace -- -D warnings
      - cargo fmt --check

  rdma-integration:  # 2-node only (self-hosted runner)
    runs-on: [self-hosted, macOS, tb5-rdma]
    needs: build-and-test
    steps:
      - cargo test --workspace --features rdma-integration
      - cargo bench --bench rdma_latency
```

---

## Phase Common Completion Criteria

All phases must meet the following criteria:

| Item | Command | Standard |
|------|---------|----------|
| **Build** | `cargo build --workspace` | 0 errors |
| **Format** | `cargo fmt --all --check` | diff 0 |
| **Lint** | `cargo clippy --workspace -- -D warnings` | 0 warnings |
| **Tests** | `cargo test --workspace` | 0 failures, all tests for the phase pass |
| **Code review** | Codex review | 0 Critical/High issues |
| **Commit** | `git commit` | Clean commit with fmt + clippy + test passing |
