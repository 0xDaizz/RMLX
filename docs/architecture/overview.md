# Architecture Overview

RMLX is a layered architecture composed of four layers, organized into 8 Cargo crates (rmlx-metal, rmlx-alloc, rmlx-rdma, rmlx-core, rmlx-distributed, rmlx-nn, rmlx-cli, rmlx-macros). All Phases (0 through 9B-opt + S1-S5 + Audit Remediation) and EP-1 through EP-6 have been completed and the system is fully implemented with 1,298+ tests. Each layer has clear responsibility boundaries and is separated into individual Cargo crates. Phase 7 additions include VJP autodiff, LoRA fine-tuning, and production hardening (structured logging, metrics, precision guard, graceful shutdown). Phase 9 additions include ExecGraph (5 CBs/layer with ICB replay, 92.3% reduction), CommandBatcher, Indirect Command Buffers (including sparse ICB), and weight pre-caching for 17.4x speedup. Serving support phases (S1-S5) add Flash Attention 2, GELU, FP8/GGUF/AWQ/GPTQ, KV cache variants, Conv1d/Conv2d, and dynamic shapes. The full-crate audit remediation (76 items) adds GatherMM, LayerNorm, unary ops, QuantizedLinear, MLA, sliding window attention, GGUF loading, 14 activations, ring/allreduce collectives, connection manager, coordinator, fence manager, library cache, residency management, and more. EP optimization adds GPU-native top-k routing, grouped expert GEMM, variable-length v3 exchange, TBO/SBO overlap, FP8 wire exchange, and sparse ICB + slab-ring execution. Phase 8c adds CachedDecode (pre-resolved PSOs + pre-allocated scratch buffers), 2-encoder decode path, and `_preresolved_into_encoder` pattern for zero per-token CPU overhead. Phase A adds prefill (seq_len=N) single-layer optimization: single-CB pipeline (54 sync points to 1), GQA slab SDPA (32 per-head dispatches to 1), GEMM threadgroup swizzle, and new `_into_cb` ops — achieving 3.5-7.3x speedup over baseline. The objc2-metal migration (from metal-rs) reduced unsafe surface to 18 blocks in rmlx-metal, 100% encapsulated behind safe APIs, with the `ComputePass` zero-cost newtype wrapper and `types.rs` alias layer providing ergonomic access.

### Performance Milestones

| Metric | Value |
|--------|-------|
| FP16 GEMM throughput | 24.05 TFLOPS |
| QMM Q4 throughput | 17.43 TFLOPS |
| Decode latency | 699.3 us/layer |
| Shape-aware GEMM dispatch | Tiled/Split-K/NAX selection by M dimension |

---

## Full Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ~/rmlx/ (this repository — ML framework)                                  │
│                        rmlx-core (compute engine)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Compute Graph / Op Registry (32 op modules)     │   │
│  │  matmul · softmax · rms_norm · rope · quantized_matmul · moe_gate   │   │
│  │  sdpa · silu · gelu · fp8 · conv · binary · reduce · copy · indexing · ...  │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────┼───────────────────────────────────────┐   │
│  │                ExecGraph / CommandBatcher / ICB Layer                │   │
│  │  ┌────────────┐  ┌──────────────────┐  ┌─────────────────────┐     │   │
│  │  │ ExecGraph   │  │ CommandBatcher   │  │ IcbBuilder/         │     │   │
│  │  │ (5 CBs/     │  │ (encoder         │  │  IcbReplay/         │     │   │
│  │  │  layer)     │  │  grouping)       │  │  IcbCache           │     │   │
│  │  └────────────┘  └──────────────────┘  └─────────────────────┘     │   │
│  └──────────────────────────────┼───────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────┼───────────────────────────────────────┐   │
│  │                        Metal Pipeline Layer                          │   │
│  │  ┌────────────┐  ┌──────────┴──────────┐  ┌─────────────────────┐   │   │
│  │  │ Kernel     │  │ CommandEncoder       │  │ Pipeline            │   │   │
│  │  │ Manager    │  │ (barrier, fence,     │  │ Scheduler           │   │   │
│  │  │ (JIT/AOT)  │  │  concurrent ctx)     │  │ (dual-queue)        │   │   │
│  │  └─────┬──────┘  └──────────┬───────────┘  └─────────┬───────────┘   │   │
│  └────────┼─────────────────────┼──────────────────────────┼────────────┘   │
│           │                     │                          │                │
│  ┌────────┼─────────────────────┼──────────────────────────┼────────────┐   │
│  │        ▼                     ▼                          ▼            │   │
│  │                       Sync & Memory Layer                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │   │
│  │  │ SharedEvent      │  │ Zero-Copy       │  │ Buffer Pool        │   │   │
│  │  │ Manager          │  │ Allocator       │  │ (Metal + ibv_mr    │   │   │
│  │  │ (signal/wait)    │  │ (posix_memalign │  │  dual-registered)  │   │   │
│  │  │                  │  │  + NoCopy)       │  │                    │   │   │
│  │  └─────────────────┘  └─────────────────┘  └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
          │                                           │
┌─────────┼───────────────────────────────────────────┼───────────────────────┐
│         ▼                                           ▼                       │
│                      rmlx-rdma (communication layer)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │
│  │ ibverbs FFI     │  │ Connection      │  │ EP Collective              │   │
│  │ (UC QP,         │  │ Manager         │  │ (all-to-all, ring          │   │
│  │  send/recv,     │  │ (hosts.json,    │  │  allreduce, send/recv)     │   │
│  │  CQ polling)    │  │  GID, warmup)   │  │                            │   │
│  └─────────────────┘  └─────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hardware: M3 Ultra (80-core GPU, 512GB UMA) x 2, TB5 RDMA (16MB max_mr)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### 1. Compute Engine — rmlx-core

The core engine responsible for computation graphs and kernel dispatch.

| Component | Role |
|-----------|------|
| **Op Registry** | Registers 32 op modules including matmul, gemv, softmax, rms_norm, rope, quantized, moe_kernels, sdpa (+ backward), silu, gelu, fp8, conv, conv_tiled, gather_mm, layer_norm, unary, binary, reduce, copy, indexing, concat, select, slice, sort, scan, random, argreduce, topk_route, fused, buffer_slots, vjp_gpu |
| **Shape-aware Dispatch** | GEMM dispatch table selects optimal kernel by M dimension: Tiled (M=1, BM=16 pad), Split-K (M=2-32), Tiled MlxMicro 16x32 (M=33-128), MlxSmall 32x32 (M=256), NAX 64x128 (M=512+) |
| **Compute Graph** | Selective tracing-based computation graph (eager-first, tracing during prefill) |
| **Kernel Dispatch** | Maps ops to Metal kernels and executes them, selecting optimal kernels based on dtype/shape |

rmlx-core depends on rmlx-metal and rmlx-alloc, and provides computation APIs to upper layers (rmlx-nn, rmlx-distributed).

---

### ExecGraph / CommandBatcher Layer

Reduces per-op CPU overhead by batching multiple GPU operations into minimal command buffers.

| Component | Role |
|-----------|------|
| **ExecGraph** | Pre-built execution graph replaying deterministic op sequences with 5 CBs/layer (92.3% reduction from 65), supports ICB replay for near-zero CPU re-encoding cost |
| **CommandBatcher** | Groups encoder work into shared command buffers, eliminating per-op CB creation |
| **ICB / ICB Sparse** | Indirect Command Buffers for zero-CPU-overhead replay of pre-encoded command sequences; sparse variant for MoE variable-dispatch patterns |

---

### EP Optimization Layer (Post-Audit)

Adds six EP phases that keep MoE routing/exchange/compute on-GPU and reduce communication overhead.

| Component | Role |
|-----------|------|
| **TopKRoute (EP-1)** | Fused softmax -> top-k -> normalize -> histogram -> prefix-scan routing kernel |
| **ExpertGroup + GatherMM (EP-2)** | Expert weight stacking + grouped batched GEMM path (Gate -> Up -> fused SwiGLU -> Down) |
| **v3 Protocol + Slab Ring (EP-3/EP-6)** | Variable-length payload exchange + pre-registered zero-copy RDMA ring buffers |
| **MoePipeline (EP-4)** | TBO/SBO compute-communication overlap with `GpuEvent` signal/wait chains |
| **FP8 Exchange (EP-5)** | Per-token E4M3 wire quantization + fused dequant-scatter path |

---

### 2. Metal Pipeline Layer

Manages the Metal GPU execution pipeline. The rmlx-metal crate implements this layer.

| Component | Role |
|-----------|------|
| **Kernel Manager** | `.metallib` AOT loading and source string JIT compilation, ComputePipelineState caching via `PipelineCache` |
| **ComputePass** | Zero-cost newtype wrapper over the raw Metal compute encoder; `types.rs` defines type aliases (`MtlDevice`, `MtlBuffer`, `MtlPipeline`) for ergonomic usage |
| **Pipeline Scheduler** | Dual `MTLCommandQueue` management — separates compute and transfer queues for GPU-level overlap |
| **Metal 4 Support** | Feature-gated (`metal4`) macOS 26+ API support: `MTL4CommandAllocator`, `MTL4ComputePipeline`, `MTL4Counters`, etc. |

The dual queue architecture was validated in PoC Phase 3.6. It maximizes pipeline efficiency by executing compute operations and RDMA data transfer preparation simultaneously.

---

### 3. Sync & Memory Layer

Handles synchronization and memory management. rmlx-metal (events) and rmlx-alloc (allocator, buffer pool) implement this layer.

| Component | Role |
|-----------|------|
| **SharedEvent Manager** | Non-blocking synchronization via `MTLSharedEvent` signal/wait (263.9 us, 1.61x improvement over `waitUntilCompleted`) |
| **Zero-Copy Allocator** | `posix_memalign` -> `newBufferWithBytesNoCopy` — creates a Metal view on the same physical memory for copy-free GPU access |
| **Buffer Pool** | Pre-allocates dual Metal + `ibv_mr` registered buffers, eliminating runtime registration overhead with size-binned caching |

The zero-copy path is limited to the RDMA communication hot path. Copies during one-time initialization such as model weight loading are not optimization targets.

---

### 4. Communication Layer — rmlx-rdma

Handles inter-node communication via Thunderbolt 5 RDMA.

| Component | Role |
|-----------|------|
| **ibverbs FFI** | ibverbs C library FFI bindings, UC QP creation/management, CQ polling |
| **Connection Manager** | Peer management via `hosts.json`, GID exchange, connection establishment and warmup |
| **EP Collective** | Distributed communication primitives: all-to-all, ring allreduce, send/recv |
| **Multi-port Striping** | Striping for dual TB5 port bandwidth utilization |

---

## Crate Dependency Graph

```
          rmlx-nn  rmlx-distributed
              │       │       │
              └───┬───┘       │
                  ▼           │
              rmlx-core       │
               │    │         │
               ▼    ▼         ▼
          rmlx-metal  rmlx-alloc
                        │
                        ▼
                    rmlx-rdma
```

The exact dependency relationships are as follows:

| Crate | Dependencies |
|-------|-------------|
| `rmlx-metal` | (external: objc2-metal 0.3, objc2 0.6, block2 0.6, objc2-foundation 0.3, bytemuck) |
| `rmlx-alloc` | `rmlx-metal`, libc |
| `rmlx-rdma` | `rmlx-alloc`, libc (ibverbs FFI) |
| `rmlx-core` | `rmlx-metal`, `rmlx-alloc` |
| `rmlx-distributed` | `rmlx-core`, `rmlx-rdma` |
| `rmlx-nn` | `rmlx-core` |
| `rmlx-macros` | (proc-macro crate: syn, quote, proc-macro2) |

### Unsafe Surface

The `rmlx-metal` crate contains 18 `unsafe` blocks in the main (non-metal4) codebase, all 100% encapsulated behind safe public APIs. The `ComputePass` wrapper and `types.rs` alias layer ensure that downstream crates (rmlx-core, rmlx-nn) never touch raw Metal pointers directly.

**Dependency principle**: Lower layers (rmlx-metal, rmlx-alloc) are unaware of upper layers (rmlx-core, rmlx-nn). All dependencies are unidirectional, and circular dependencies are not allowed.

---

## Hardware Target

| Item | Specification |
|------|-------------|
| GPU | Apple M3 Ultra, 80-core GPU |
| Memory | 512GB Unified Memory Architecture (UMA) |
| Nodes | 2 (Mac Studio) |
| Interconnect | Thunderbolt 5 RDMA (16MB max_mr) |
| RDMA mode | UC QP (Unreliable Connection) — RC not supported on TB5 |
