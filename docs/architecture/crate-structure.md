# Crate Structure

RMLX consists of 7 crates organized as a Cargo workspace, along with supplementary directories.

---

## Full Workspace Layout

```
rmlx/
├── Cargo.toml                    # workspace root
├── PLAN.md                       # implementation plan
├── crates/
│   ├── rmlx-metal/               # Metal GPU abstractions
│   │   ├── Cargo.toml            # deps: metal-rs 0.31, objc2, block2
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── device.rs         # MTLDevice wrapper, architecture detection
│   │       ├── queue.rs          # MTLCommandQueue management (dual queue support)
│   │       ├── command.rs        # CommandBuffer/Encoder abstraction
│   │       ├── buffer.rs         # MTLBuffer, zero-copy creation
│   │       ├── event.rs          # MTLSharedEvent wrapper
│   │       ├── fence.rs          # MTLFence + fast-fence (shared buffer spin)
│   │       ├── pipeline.rs       # ComputePipelineState cache
│   │       ├── library.rs        # MTLLibrary load/JIT compilation
│   │       ├── resident.rs       # ResidencySet management
│   │       ├── self_check.rs     # Startup diagnostics
│   │       ├── batcher.rs       # CommandBatcher — CB grouping
│   │       ├── exec_graph.rs    # ExecGraph — deterministic op replay
│   │       └── icb.rs           # Indirect Command Buffer support
│   │
│   ├── rmlx-alloc/               # Memory allocator
│   │   ├── Cargo.toml            # deps: rmlx-metal, libc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── zero_copy.rs      # posix_memalign + newBufferWithBytesNoCopy
│   │       ├── pool.rs           # Dual-registered buffer pool (Metal + ibv_mr)
│   │       ├── cache.rs          # MLX-style size-binned cache
│   │       ├── stats.rs          # Allocation statistics, peak memory tracking
│   │       └── leak_detector.rs  # Memory leak detection
│   │
│   ├── rmlx-rdma/                # RDMA communication
│   │   ├── Cargo.toml            # deps: rmlx-alloc, libc (ibverbs FFI)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── ffi.rs            # ibverbs C FFI bindings (bindgen)
│   │   │   ├── context.rs        # ibv_context, PD, CQ management
│   │   │   ├── qp.rs             # UC QP creation/management, GID handling
│   │   │   ├── mr.rs             # ibv_reg_mr, dual-registered buffer management
│   │   │   ├── exchange.rs       # blocking_exchange (2-phase count -> payload)
│   │   │   ├── collective.rs     # all-to-all, ring allreduce
│   │   │   ├── connection.rs     # hosts.json parsing, connection setup, warmup
│   │   │   ├── multi_port.rs     # Dual TB5 port striping
│   │   │   └── rdma_metrics.rs   # RDMA metrics collection
│   │   └── build.rs              # ibverbs link configuration
│   │
│   ├── rmlx-core/                # Compute engine
│   │   ├── Cargo.toml            # deps: rmlx-metal, rmlx-alloc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dtype.rs          # Data types (f32, f16, bf16, quantized)
│   │       ├── array.rs          # N-dim array type (Buffer ownership)
│   │       ├── ops/
│   │       │   ├── mod.rs
│   │       │   ├── matmul.rs     # GEMM (metal-rs dispatch)
│   │       │   ├── quantized.rs  # QMM (4bit, 8bit, FP4, FP8)
│   │       │   ├── softmax.rs
│   │       │   ├── rms_norm.rs
│   │       │   ├── rope.rs
│   │       │   ├── binary.rs     # Element-wise operations
│   │       │   ├── reduce.rs
│   │       │   ├── copy.rs
│   │       │   ├── indexing.rs
│   │       │   ├── silu.rs         # SiLU activation + fused SwiGLU
│   │       │   └── sdpa.rs         # Fused Scaled Dot-Product Attention
│   │       ├── kernels/
│   │       │   ├── mod.rs        # Kernel registry
│   │       │   ├── loader.rs     # .metallib AOT loader
│   │       │   └── jit.rs        # Source string JIT compilation
│   │       ├── graph.rs          # Computation graph (selective tracing)
│   │       ├── scheduler.rs      # Per-stream execution scheduler
│   │       ├── vjp.rs            # VJP (Vector-Jacobian Product) autodiff
│   │       ├── lora.rs           # LoRA fine-tuning support
│   │       ├── logging.rs        # Structured logging (tracing)
│   │       ├── metrics.rs        # Metrics collection (Prometheus compatible)
│   │       ├── precision_guard.rs # Precision guard (dtype safety)
│   │       └── shutdown.rs       # Graceful shutdown
│   │
│   ├── rmlx-distributed/         # Distributed primitives
│   │   ├── Cargo.toml            # deps: rmlx-core, rmlx-rdma
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── group.rs          # Distributed group abstraction (rank, world_size)
│   │       ├── primitives.rs     # AllReduce, AllGather, Send, Recv, AllToAll
│   │       ├── moe/
│   │       │   ├── mod.rs
│   │       │   ├── dispatch.rs   # MoeDispatchExchange (3-zone auto backend)
│   │       │   ├── combine.rs    # MoeCombineExchange
│   │       │   ├── policy.rs     # 3-zone policy (CPU/Metal/byte threshold)
│   │       │   ├── kernels.rs    # MoE Metal kernel management (7 kernels)
│   │       │   └── warmup.rs     # RDMA + Metal JIT pre-warmup
│   │       ├── moe_exchange.rs   # MoE data exchange primitives
│   │       ├── moe_policy.rs     # MoE routing policy
│   │       ├── pipeline.rs       # Layer-level compute <-> RDMA pipeline
│   │       ├── sparse_guard.rs   # Sparse dispatch safety guard
│   │       ├── warmup.rs         # Distributed warmup
│   │       └── metrics.rs        # Distributed metrics collection
│   │
│   ├── rmlx-nn/                  # Neural network layers
│   │   ├── Cargo.toml            # deps: rmlx-core
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── linear.rs         # Linear (with quantization support)
│   │       ├── embedding.rs
│   │       ├── attention.rs      # Multi-head/GQA attention
│   │       ├── transformer.rs    # Transformer block
│   │       ├── moe.rs            # MoE gate + expert routing
│   │       ├── parallel.rs       # Megatron-LM TP (Column/RowParallel)
│   │       └── models/
│   │           ├── mod.rs
│   │           ├── llama.rs      # LLaMA architecture
│   │           ├── qwen.rs       # Qwen/Qwen2.5
│   │           ├── deepseek.rs   # DeepSeek-V3 (MoE)
│   │           └── mixtral.rs    # Mixtral (MoE)
│   │
│   └── rmlx-cli/                 # Native CLI tooling
│       ├── Cargo.toml            # deps: clap, serde, serde_json
│       └── src/
│           └── main.rs           # rmlx config, rmlx launch subcommands
│
├── shaders/                      # Metal shader sources
│   ├── mlx_compat/               # .metal files ported from MLX
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
│   ├── moe/                      # EP-specific kernels
│   │   ├── dispatch_local.metal
│   │   ├── dispatch_scatter.metal
│   │   ├── combine_gather.metal
│   │   ├── combine_weighted_sum.metal
│   │   ├── packet_gather.metal
│   │   └── packet_scatter.metal
│   └── custom/                   # RMLX-specific kernels
│       ├── fused_attention.metal
│       └── fused_moe_gate.metal
│
├── tests/                        # Integration tests
│   ├── metal_basic.rs
│   ├── zero_copy.rs
│   ├── rdma_exchange.rs
│   ├── moe_dispatch.rs
│   └── e2e_inference.rs
│
├── benches/                      # Benchmarks
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

## Crate Details

### rmlx-metal — Metal GPU Abstraction

| Item | Details |
|------|---------|
| **Purpose** | Provides a safe Rust abstraction over the Apple Metal API. Wraps MTLDevice, MTLCommandQueue, MTLBuffer, MTLSharedEvent, and more, built on `metal-rs` 0.31. |
| **Key modules** | `device.rs` (device + architecture detection), `queue.rs` (dual queue management), `command.rs` (CommandBuffer/Encoder), `event.rs` (MTLSharedEvent wrapper), `pipeline.rs` (PSO cache), `self_check.rs` (startup diagnostics), `batcher.rs` (CommandBatcher), `exec_graph.rs` (ExecGraph), `icb.rs` (Indirect Command Buffer) |
| **Dependencies** | metal-rs 0.31, objc2, block2 |
| **Status** | Complete — GpuDevice, StreamManager, DeviceStream, GpuEvent, SharedEvent sync, dual queue pipeline, startup diagnostics, top-level re-exports (GpuDevice, GpuEvent), ExecGraph (5 CBs/layer), CommandBatcher, ICB, fence manager, library cache, MSL version detection, autorelease pool, capture manager, managed buffers, Phase 0+1+2 audit remediation (M1-M8) fully implemented |

---

### rmlx-alloc — Memory Allocator

| Item | Details |
|------|---------|
| **Purpose** | Handles zero-copy memory allocation and buffer pool management. Creates copy-free Metal buffers using the `posix_memalign` -> `newBufferWithBytesNoCopy` pattern, with support for RDMA `ibv_mr` dual registration. |
| **Key modules** | `zero_copy.rs` (ZeroCopyBuffer, CompletionFence), `pool.rs` (dual-registered buffer pool), `cache.rs` (size-binned cache), `stats.rs` (allocation statistics), `leak_detector.rs` (memory leak detection) |
| **Dependencies** | `rmlx-metal`, libc |
| **Status** | Complete — ZeroCopyBuffer, DualRegPool, MetalAllocator, size-binned cache, leak detection, small allocation fast-path, Phase 0+1+2 audit remediation (A1-A12) fully implemented. `ResidencyManager` API exists, but Metal `MTLResidencySet` backend is currently a documented stub pending `metal-rs` support. |

---

### rmlx-rdma — RDMA Communication

| Item | Details |
|------|---------|
| **Purpose** | Provides high-performance inter-node communication via Thunderbolt 5 RDMA. Binds the ibverbs C library via FFI and supports UC QP-based send/recv and collective operations. |
| **Key modules** | `ffi.rs` (ibverbs bindings), `context.rs` (ibv_context/PD/CQ), `qp.rs` (UC QP), `mr.rs` (memory region registration), `collective.rs` (all-to-all, allreduce), `multi_port.rs` (port striping), `rdma_metrics.rs` (RDMA metrics) |
| **Dependencies** | `rmlx-alloc`, libc (ibverbs FFI) |
| **Status** | Complete — ibverbs FFI, UC QP, blocking_exchange, ConnectionManager, dual port striping, RDMA metrics, ring/allreduce/allgather collectives, connection manager, coordinator, Phase 0+1+2 audit remediation (R1-R3) fully implemented |

---

### rmlx-core — Compute Engine

| Item | Details |
|------|---------|
| **Purpose** | The core engine that integrates the computation graph, Op registry, and kernel dispatch. Defines the N-dim array type and dtype system, and supports eager-first execution with selective tracing compilation. |
| **Key modules** | `dtype.rs` (f32, f16, bf16, quantized), `array.rs` (N-dim array), `ops/` (25 op modules: matmul, softmax, rms_norm, rope, quantized, binary, reduce, copy, indexing, sdpa, sdpa_backward, silu, gelu, fp8, conv, conv_tiled, gather_mm, layer_norm, unary, concat, select, vjp_gpu, etc.), `kernels/` (AOT/JIT kernel management), `graph.rs` (computation graph), `scheduler.rs` (per-stream scheduler), `vjp.rs` (VJP autodiff), `lora.rs` (LoRA fine-tuning), `prelude.rs` (convenience re-exports), `logging.rs` (structured logging), `metrics.rs` (metrics collection), `precision_guard.rs` (precision guard), `shutdown.rs` (graceful shutdown) |
| **Dependencies** | `rmlx-metal`, `rmlx-alloc` |
| **Status** | Complete — Array type, 25 op modules (including sdpa + backward, silu, gelu, gather_mm, layer_norm, unary, concat, select, conv_tiled, vjp_gpu), ExecMode, CommandBufferHandle, LaunchResult, VJP autodiff, LoRA, production hardening, Phase 0+1+2 audit remediation (C1-C9) fully implemented |

---

### rmlx-distributed — Distributed Primitives

| Item | Details |
|------|---------|
| **Purpose** | Implements communication primitives and MoE Expert Parallelism for distributed inference. Overlaps compute and RDMA through a layer-level pipeline. |
| **Key modules** | `group.rs` (rank/world_size abstraction), `primitives.rs` (AllReduce, AllGather, etc.), `moe/` (3-zone auto backend, MoE dispatch/combine), `moe_exchange.rs`, `moe_policy.rs`, `pipeline.rs` (compute-RDMA overlap), `sparse_guard.rs` (sparse dispatch guard), `warmup.rs` (distributed warmup), `metrics.rs` (distributed metrics) |
| **Dependencies** | `rmlx-core`, `rmlx-rdma` |
| **Status** | Complete — EP dispatch/combine (loop ordering + capacity + caching fixed), 3-zone auto backend (threshold + hysteresis + cooldown fixed), compute-RDMA pipeline, MoE exchange, shared expert, EP integration, distributed metrics, Phase 0+1+2 audit remediation (D1-D10) fully implemented |

---

### rmlx-nn — Neural Network Layers

| Item | Details |
|------|---------|
| **Purpose** | Provides neural network layers for Transformer-based architectures. Includes high-level modules such as Linear, Attention, and MoE, as well as model architectures for LLaMA, Qwen, DeepSeek-V3, and others. |
| **Key modules** | `linear.rs` (Linear, `prepare_weight_t()`), `quantized_linear.rs` (QuantizedLinear), `attention.rs` (Multi-head/GQA), `mla.rs` (Multi-Latent Attention), `sliding_window.rs` (Sliding Window), `layer_norm.rs` (LayerNorm), `activations.rs` (14 activations), `transformer.rs` (Transformer block, `forward_graph()`, `forward_into_cb()`), `moe.rs` (gate + routing + shared expert + EP integration + GPU routing), `gguf_loader.rs` (GGUF model loading), `parallel.rs` (ColumnParallel/RowParallel), `models/` (llama.rs, qwen.rs, deepseek.rs, mixtral.rs) |
| **Dependencies** | `rmlx-core` |
| **Status** | Complete — Transformer block, Linear/QuantizedLinear/Attention/MLA/MoE layers, KV cache (static/rotating/batch/quantized), sliding window attention, LayerNorm, 14 activations, parallel linear layers, GGUF model loader, LLaMA/Qwen/DeepSeek-V3/Mixtral model architectures, ExecGraph-compatible `forward_graph()`, weight pre-caching, Phase 0+1+2 audit remediation (N1-N8) fully implemented |

---

### rmlx-cli — Native CLI Tooling

| Item | Details |
|------|---------|
| **Purpose** | Provides the `rmlx` command-line interface for distributed cluster management. Implements `rmlx config` (hostfile generation and baseline setup) and `rmlx launch` (multi-node process orchestration), modeled after MLX's `mlx.distributed_config` and `mlx.launch`. |
| **Key commands** | `rmlx config` (host discovery, RDMA backend setup, hostfile output), `rmlx launch` (SSH-based multi-node command dispatch) |
| **Dependencies** | `rmlx-distributed`, `rmlx-rdma` |
| **Status** | Initial implementation — config and launch subcommands functional |

---

## Workspace Configuration

```toml
# Cargo.toml (workspace root)

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
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
license = "MIT"

[workspace.dependencies]
metal = "0.31"
objc = "0.2"
block = "0.1"
libc = "0.2"
```

`workspace.dependencies` is used to unify dependency versions across crates. Each crate's `Cargo.toml` can reference them using `metal.workspace = true`.

---

## Supplementary Directories

### shaders/

Stores Metal shader source files, organized into 3 subdirectories:

- **mlx_compat/**: Compatibility kernels ported from MLX (gemv, gemm, quantized, softmax, etc.)
- **moe/**: Expert Parallelism-specific kernels (6 kernels for dispatch, combine, and packet operations)
- **custom/**: RMLX-specific optimized kernels (fused_attention, fused_moe_gate)

### tests/

Integration tests. Unit tests reside within each individual crate.

### benches/

Criterion-based benchmarks measuring matmul, RDMA latency, MoE step, and e2e throughput.

### examples/

Usage examples including basic compute, zero-copy demo, and 2-node EP.
