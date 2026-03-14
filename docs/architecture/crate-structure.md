# Crate Structure

RMLX consists of 8 crates organized as a Cargo workspace, along with supplementary directories.

---

## Full Workspace Layout

```
rmlx/
├── Cargo.toml                    # workspace root
├── PLAN.md                       # implementation plan
├── crates/
│   ├── rmlx-metal/               # Metal GPU abstractions
│   │   ├── Cargo.toml            # deps: objc2-metal 0.3, objc2 0.6, block2 0.6, objc2-foundation 0.3, bytemuck
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── device.rs         # MTLDevice wrapper, architecture detection
│   │       ├── queue.rs          # MTLCommandQueue management (dual queue support)
│   │       ├── command.rs        # CommandBuffer/Encoder abstraction
│   │       ├── compute_pass.rs   # ComputePass zero-cost newtype wrapper over Metal encoder
│   │       ├── buffer.rs         # MTLBuffer, zero-copy creation
│   │       ├── managed_buffer.rs # Managed buffer lifecycle
│   │       ├── event.rs          # MTLSharedEvent wrapper
│   │       ├── fence.rs          # MTLFence + fast-fence (shared buffer spin)
│   │       ├── pipeline.rs       # ComputePipelineState creation
│   │       ├── pipeline_cache.rs # PipelineCache — PSO caching layer
│   │       ├── library.rs        # MTLLibrary load/JIT compilation
│   │       ├── library_cache.rs  # Library caching
│   │       ├── types.rs          # Type aliases (MtlDevice, MtlBuffer, MtlPipeline, etc.)
│   │       ├── autorelease.rs    # Autorelease pool management
│   │       ├── capture.rs        # Metal GPU capture manager
│   │       ├── msl_version.rs    # MSL version detection
│   │       ├── self_check.rs     # Startup diagnostics
│   │       ├── stream.rs         # Stream management
│   │       ├── batcher.rs        # CommandBatcher — CB grouping
│   │       ├── exec_graph.rs     # ExecGraph — deterministic op replay with ICB replay
│   │       ├── icb.rs            # Indirect Command Buffer support
│   │       ├── icb_sparse.rs     # Sparse ICB for MoE variable-dispatch patterns
│   │       └── metal4/           # Feature-gated Metal 4 (macOS 26+) support
│   │           ├── mod.rs
│   │           ├── command.rs    # MTL4CommandBuffer/Allocator/Queue
│   │           ├── compiler.rs   # MTL4Compiler/CompilerTask
│   │           ├── compute.rs    # MTL4ComputePipeline/ComputeCommandEncoder
│   │           └── counter_heap.rs # MTL4Counters
│   │
│   ├── rmlx-alloc/               # Memory allocator
│   │   ├── Cargo.toml            # deps: rmlx-metal, libc
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── allocator.rs      # MetalAllocator — unified allocation entry point
│   │       ├── zero_copy.rs      # posix_memalign + newBufferWithBytesNoCopy
│   │       ├── buffer_pool.rs    # Dual-registered buffer pool (Metal + ibv_mr)
│   │       ├── cache.rs          # MLX-style size-binned cache
│   │       ├── bfc.rs            # Best-Fit Coalescing allocator
│   │       ├── small_alloc.rs    # Small allocation fast-path
│   │       ├── residency.rs      # ResidencyManager (MTLResidencySet backend)
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
│   │       │   ├── mod.rs          # Op registry + shape-aware GEMM dispatch table
│   │       │   ├── matmul.rs       # GEMM (Metal dispatch via ComputePass)
│   │       │   ├── gemv.rs         # GEMV path
│   │       │   ├── quantized.rs    # QMM (4bit, 8bit, FP4, FP8)
│   │       │   ├── softmax.rs
│   │       │   ├── rms_norm.rs
│   │       │   ├── layer_norm.rs   # LayerNorm
│   │       │   ├── rope.rs
│   │       │   ├── binary.rs       # Element-wise operations
│   │       │   ├── unary.rs        # Unary operations
│   │       │   ├── reduce.rs
│   │       │   ├── argreduce.rs    # ArgReduce (argmin/argmax)
│   │       │   ├── copy.rs
│   │       │   ├── indexing.rs
│   │       │   ├── concat.rs       # Tensor concatenation
│   │       │   ├── select.rs       # Tensor selection
│   │       │   ├── slice.rs        # Tensor slicing
│   │       │   ├── sort.rs         # Tensor sorting
│   │       │   ├── scan.rs         # Prefix scan
│   │       │   ├── random.rs       # Random number generation
│   │       │   ├── silu.rs         # SiLU activation + fused SwiGLU
│   │       │   ├── gelu.rs         # GELU activation
│   │       │   ├── fp8.rs          # FP8 quantization ops
│   │       │   ├── conv.rs         # Conv1d/Conv2d
│   │       │   ├── conv_tiled.rs   # Tiled convolution
│   │       │   ├── gather_mm.rs    # Grouped/gathered matrix multiply
│   │       │   ├── sdpa.rs         # Fused Scaled Dot-Product Attention
│   │       │   ├── sdpa_backward.rs # SDPA backward pass
│   │       │   ├── topk_route.rs   # GPU-native top-k routing
│   │       │   ├── moe_kernels.rs  # MoE-specific kernels
│   │       │   ├── fused.rs        # Fused kernel operations
│   │       │   ├── buffer_slots.rs # Buffer slot management
│   │       │   └── vjp_gpu.rs      # VJP autodiff GPU ops
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
│   │       ├── init.rs           # Distributed initialization
│   │       ├── transport.rs      # Transport abstraction
│   │       ├── moe_exchange.rs   # MoE data exchange primitives
│   │       ├── moe_policy.rs     # MoE routing policy
│   │       ├── moe_kernels.rs    # MoE Metal kernel management
│   │       ├── ep_runtime.rs     # Expert Parallelism runtime
│   │       ├── fp8_exchange.rs   # FP8 wire exchange
│   │       ├── slab_ring.rs      # Slab ring buffer for RDMA
│   │       ├── v3_protocol.rs    # Variable-length v3 exchange protocol
│   │       ├── pipeline.rs       # Layer-level compute <-> RDMA pipeline
│   │       ├── sparse_guard.rs   # Sparse dispatch safety guard
│   │       ├── credit_manager.rs # Flow control credits
│   │       ├── health.rs         # Distributed health monitoring
│   │       ├── perf_counters.rs  # Performance counters
│   │       ├── progress_tracker.rs # Progress tracking
│   │       ├── warmup.rs         # Distributed warmup
│   │       └── metrics.rs        # Distributed metrics collection
│   │
│   ├── rmlx-nn/                  # Neural network layers
│   │   ├── Cargo.toml            # deps: rmlx-core
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── linear.rs         # Linear (with quantization support, prepare_weight_t())
│   │       ├── quantized_linear.rs # QuantizedLinear
│   │       ├── embedding.rs
│   │       ├── attention.rs      # Multi-head/GQA attention
│   │       ├── mla.rs            # Multi-Latent Attention
│   │       ├── sliding_window.rs # Sliding window attention
│   │       ├── layer_norm.rs     # LayerNorm
│   │       ├── rms_norm.rs       # RMSNorm
│   │       ├── rope.rs           # Rotary Position Embedding
│   │       ├── activations.rs    # 14 activation functions
│   │       ├── conv.rs           # Conv1d/Conv2d
│   │       ├── transformer.rs    # Transformer block (forward_graph(), forward_into_cb())
│   │       ├── moe.rs            # MoE gate + expert routing + GPU routing
│   │       ├── moe_pipeline.rs   # MoE pipeline execution
│   │       ├── expert_group.rs   # Expert group management
│   │       ├── parallel.rs       # Megatron-LM TP (Column/RowParallel)
│   │       ├── paged_kv_cache.rs # Paged KV cache
│   │       ├── prefix_cache.rs   # Prefix cache
│   │       ├── dynamic.rs        # Dynamic shapes
│   │       ├── sampler.rs        # Token sampling
│   │       ├── scheduler.rs      # Request scheduling
│   │       ├── prefill_plan.rs   # Prefill planning
│   │       ├── prefill_pool.rs   # Prefill pool management
│   │       ├── gguf_loader.rs    # GGUF model loading
│   │       ├── safetensors_loader.rs # Safetensors model loading
│   │       └── models/
│   │           ├── mod.rs
│   │           ├── qwen.rs       # Qwen 3.5
│   │           ├── deepseek.rs   # DeepSeek-V3 (MoE)
│   │           └── mixtral.rs    # Mixtral (MoE)
│   │
│   ├── rmlx-macros/              # Proc-macro utilities
│   │   ├── Cargo.toml            # deps: syn, quote, proc-macro2
│   │   └── src/
│   │       └── lib.rs            # Derive macros for framework types
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
| **Purpose** | Provides a safe Rust abstraction over the Apple Metal API. Wraps MTLDevice, MTLCommandQueue, MTLBuffer, MTLSharedEvent, and more, built on `objc2-metal` 0.3. The `ComputePass` newtype wrapper (in `compute_pass.rs`) provides a zero-cost abstraction layer over the raw Metal encoder, and `types.rs` defines type aliases (`MtlDevice`, `MtlBuffer`, `MtlPipeline`, etc.) for ergonomic usage. Contains 18 `unsafe` blocks in the main codebase, all 100% encapsulated behind safe public APIs. |
| **Key modules** | `device.rs` (device + architecture detection), `queue.rs` (dual queue management), `command.rs` (CommandBuffer abstraction), `compute_pass.rs` (ComputePass zero-cost wrapper), `types.rs` (MtlDevice, MtlBuffer, MtlPipeline type aliases), `event.rs` (MTLSharedEvent wrapper), `pipeline.rs` + `pipeline_cache.rs` (PSO creation + caching), `library_cache.rs` (library caching), `self_check.rs` (startup diagnostics), `batcher.rs` (CommandBatcher), `exec_graph.rs` (ExecGraph with ICB replay), `icb.rs` + `icb_sparse.rs` (Indirect Command Buffer + sparse variant), `autorelease.rs` (autorelease pool), `capture.rs` (GPU capture manager), `managed_buffer.rs` (managed buffer lifecycle), `stream.rs` (stream management), `metal4/` (feature-gated Metal 4 support) |
| **Dependencies** | objc2-metal 0.3, objc2 0.6, block2 0.6, objc2-foundation 0.3, bytemuck |
| **Status** | Complete — GpuDevice, StreamManager, DeviceStream, GpuEvent, SharedEvent sync, dual queue pipeline, startup diagnostics, top-level re-exports (GpuDevice, GpuEvent), ExecGraph (5 CBs/layer with ICB replay), CommandBatcher, ICB + ICB Sparse, fence manager, library cache, MSL version detection, autorelease pool, capture manager, managed buffers, Metal 4 feature-gated support (MTL4CommandAllocator, MTL4ComputePipeline, MTL4Counters), Phase 0+1+2 audit remediation (M1-M8) fully implemented |

---

### rmlx-alloc — Memory Allocator

| Item | Details |
|------|---------|
| **Purpose** | Handles zero-copy memory allocation and buffer pool management. Creates copy-free Metal buffers using the `posix_memalign` -> `newBufferWithBytesNoCopy` pattern, with support for RDMA `ibv_mr` dual registration. |
| **Key modules** | `allocator.rs` (MetalAllocator unified entry point), `zero_copy.rs` (ZeroCopyBuffer, CompletionFence), `buffer_pool.rs` (dual-registered buffer pool), `cache.rs` (size-binned cache), `bfc.rs` (best-fit coalescing allocator), `small_alloc.rs` (small allocation fast-path), `residency.rs` (ResidencyManager with MTLResidencySet backend), `stats.rs` (allocation statistics), `leak_detector.rs` (memory leak detection) |
| **Dependencies** | `rmlx-metal`, libc |
| **Status** | Complete — ZeroCopyBuffer, DualRegPool, MetalAllocator, size-binned cache, leak detection, small allocation fast-path, Phase 0+1+2 audit remediation (A1-A12) fully implemented. `ResidencyManager` is now backed by `objc2-metal`'s `MTLResidencySet` support. |

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
| **Key modules** | `dtype.rs` (f32, f16, bf16, quantized), `array.rs` (N-dim array), `ops/` (32 op modules: matmul, gemv, softmax, rms_norm, layer_norm, rope, quantized, binary, unary, reduce, argreduce, copy, indexing, concat, select, slice, sort, scan, random, sdpa, sdpa_backward, silu, gelu, fp8, conv, conv_tiled, gather_mm, topk_route, moe_kernels, fused, buffer_slots, vjp_gpu), `kernels/` (AOT/JIT kernel management), `graph.rs` (computation graph), `scheduler.rs` (per-stream scheduler), `vjp.rs` (VJP autodiff), `lora.rs` (LoRA fine-tuning), `prelude.rs` (convenience re-exports), `logging.rs` (structured logging), `metrics.rs` (metrics collection), `precision_guard.rs` (precision guard), `shutdown.rs` (graceful shutdown) |
| **Dependencies** | `rmlx-metal`, `rmlx-alloc` |
| **Status** | Complete — Array type, 32 op modules (including gemv, argreduce, slice, sort, scan, random, topk_route, moe_kernels, fused, buffer_slots, and all previous modules), shape-aware GEMM dispatch table (Tiled/Split-K/NAX by M dimension), ExecMode, CommandBufferHandle, LaunchResult, VJP autodiff, LoRA, production hardening, Phase 0+1+2 audit remediation (C1-C9) fully implemented |

---

### rmlx-distributed — Distributed Primitives

| Item | Details |
|------|---------|
| **Purpose** | Implements communication primitives and MoE Expert Parallelism for distributed inference. Overlaps compute and RDMA through a layer-level pipeline. |
| **Key modules** | `group.rs` (rank/world_size abstraction), `init.rs` (initialization), `transport.rs` (transport abstraction), `moe_exchange.rs` (MoE data exchange), `moe_policy.rs` (MoE routing policy), `moe_kernels.rs` (MoE Metal kernels), `ep_runtime.rs` (Expert Parallelism runtime), `fp8_exchange.rs` (FP8 wire exchange), `slab_ring.rs` (slab ring buffer), `v3_protocol.rs` (variable-length exchange), `pipeline.rs` (compute-RDMA overlap), `sparse_guard.rs` (sparse dispatch guard), `credit_manager.rs` (flow control), `health.rs` (health monitoring), `perf_counters.rs` (performance counters), `progress_tracker.rs` (progress tracking), `warmup.rs` (distributed warmup), `metrics.rs` (distributed metrics) |
| **Dependencies** | `rmlx-core`, `rmlx-rdma` |
| **Status** | Complete — EP dispatch/combine (loop ordering + capacity + caching fixed), 3-zone auto backend (threshold + hysteresis + cooldown fixed), compute-RDMA pipeline, MoE exchange, shared expert, EP integration, distributed metrics, Phase 0+1+2 audit remediation (D1-D10) fully implemented |

---

### rmlx-nn — Neural Network Layers

| Item | Details |
|------|---------|
| **Purpose** | Provides neural network layers for Transformer-based architectures. Includes high-level modules such as Linear, Attention, and MoE, as well as model architectures for Qwen 3.5, DeepSeek-V3, Mixtral, and others. |
| **Key modules** | `linear.rs` (Linear, `prepare_weight_t()`), `quantized_linear.rs` (QuantizedLinear), `attention.rs` (Multi-head/GQA), `mla.rs` (Multi-Latent Attention), `sliding_window.rs` (Sliding Window), `layer_norm.rs` (LayerNorm), `rms_norm.rs` (RMSNorm), `rope.rs` (RoPE), `activations.rs` (14 activations), `conv.rs` (Conv1d/Conv2d), `transformer.rs` (Transformer block, `forward_graph()`, `forward_into_cb()`), `moe.rs` (gate + routing + GPU routing), `moe_pipeline.rs` (MoE pipeline), `expert_group.rs` (expert group management), `paged_kv_cache.rs` (paged KV cache), `prefix_cache.rs` (prefix cache), `dynamic.rs` (dynamic shapes), `sampler.rs` (token sampling), `scheduler.rs` (request scheduling), `prefill_plan.rs` + `prefill_pool.rs` (prefill optimization), `gguf_loader.rs` (GGUF loading), `safetensors_loader.rs` (safetensors loading), `parallel.rs` (ColumnParallel/RowParallel), `models/` (qwen.rs, deepseek.rs, mixtral.rs) |
| **Dependencies** | `rmlx-core` |
| **Status** | Complete — Transformer block, Linear/QuantizedLinear/Attention/MLA/MoE layers, KV cache (static/rotating/batch/quantized), sliding window attention, LayerNorm, 14 activations, parallel linear layers, GGUF model loader, Qwen/DeepSeek-V3/Mixtral/Kimi model architectures, ExecGraph-compatible `forward_graph()`, weight pre-caching, Phase 0+1+2 audit remediation (N1-N8) fully implemented |

---

### rmlx-macros — Proc-Macro Utilities

| Item | Details |
|------|---------|
| **Purpose** | Provides procedural macros (derive macros) for framework types, reducing boilerplate across crates. |
| **Key modules** | `lib.rs` (proc-macro entry point) |
| **Dependencies** | syn 2, quote 1, proc-macro2 1 |
| **Status** | Complete — proc-macro crate providing derive macros for framework types |

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

`workspace.dependencies` is used to unify dependency versions across crates. Each crate's `Cargo.toml` can reference them using `objc2-metal.workspace = true`.

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
