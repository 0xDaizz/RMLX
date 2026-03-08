# RMLX vs MLX vs CUDA -- Architecture Comparison

> An honest, engineering-focused comparison of three GPU inference stacks.
> RMLX has clear advantages in certain areas and clear disadvantages in others.
> This document presents both sides without marketing spin.
>
> Related docs: [Architecture Overview](architecture/overview.md) | [Design Decisions](architecture/design-decisions.md) | [Roadmap](roadmap/phases.md)

---

## 1. Architecture Comparison

| Feature | RMLX | MLX | CUDA (PyTorch / vLLM) |
|---------|------|-----|-----------------------|
| **Language** | Rust 1.80+ | C++ / Python (nanobind) | C++ / Python (pybind11) |
| **GPU API** | Apple Metal (metal-rs 0.31) | Apple Metal (metal-cpp) | NVIDIA CUDA |
| **Memory model** | Unified Memory (UMA) | Unified Memory (UMA) | Discrete VRAM + host RAM |
| **Execution model** | Eager-first (selective tracing for prefill) | Lazy evaluation (graph-level fusion) | Eager (PyTorch) / Graph (CUDA Graphs) |
| **CB management** | ExecGraph + 9-dispatch: 65 CBs/layer -> 1 CB with 9 dispatches (64x speedup vs per-op baseline, 6.34x faster than MLX at 60L) | 1 CB per eval() batch | Stream-ordered, CUDA Graphs capture-replay |
| **Sync mechanism** | MTLSharedEvent signal/wait (263.9us) | waitUntilCompleted (424.9us) | CUDA events / streams |
| **RDMA** | Zero-copy: posix_memalign + NoCopy + ibv_reg_mr | std::copy to transfer buffer | GPUDirect RDMA / NVLink |
| **Expert Parallelism** | Native EP (3-zone auto backend, 7 MoE kernels, EP-1~EP-6 optimized path) | No EP support | DeepSpeed-MoE, Tutel |
| **Quantization** | Q4_0, Q4_1, Q8_0, FP8, GGUF, AWQ, GPTQ | Q4_0, Q4_1, Q8_0, GGUF, AWQ, GPTQ | FP8, AWQ, GPTQ, INT4/INT8 |
| **Flash Attention** | Flash Attention 2 (D≤256) | No (fused SDPA) | FlashAttention-2/3 |
| **KV cache** | Static, Rotating, Batch, Quantized | Static per-layer cache | PagedAttention (vLLM) |
| **Training** | LoRA fine-tuning only | Full training support | Full training + LoRA + QLoRA |
| **Op modules** | 27 | ~50+ | Hundreds |
| **NN activations** | 14 | ~10 | Dozens |
| **GatherMM** | Yes | Yes | Yes |
| **LayerNorm** | Yes | Yes | Yes |
| **QuantizedLinear** | Yes | Yes | Yes |
| **MLA** | Yes (DeepSeek-V3) | No | Partial |
| **Sliding Window Attn** | Yes | Yes | Yes |
| **GGUF model loading** | Yes | Yes | Yes |
| **Test suite** | 1,298+ tests | Extensive | Extensive |
| **Prefill optimization** | Phase A: single-CB pipeline, GQA slab SDPA, 3.5-7.3x speedup | Lazy eval graph fusion | CUDA Graphs + cuBLAS |
| **Phases complete** | 0-9B-opt + S1-S5 + Phase KO + Phase 8c + Phase 9 + Phase 10 + Phase 11 + Phase A | N/A (stable release) | N/A (stable release) |

---

## 2. Where RMLX is Superior to MLX

### 2.1 MTLSharedEvent Synchronization

MLX uses `waitUntilCompleted` to block the CPU thread until a command buffer finishes. RMLX replaces this with non-blocking `MTLSharedEvent` signal/wait.

| Method | Latency | Mechanism |
|--------|---------|-----------|
| `waitUntilCompleted` (MLX) | 424.9us | CPU blocks until CB completion |
| `MTLSharedEvent` spin-wait (RMLX) | 263.9us | Non-blocking poll on signaledValue |
| **Improvement** | **1.61x** | |

This difference compounds across layers. In a 60-layer model, synchronization overhead alone drops by tens of milliseconds.

### 2.2 ExecGraph Command Buffer Batching

MLX submits individual command buffers per operation. RMLX's ExecGraph pre-encodes a deterministic sequence of operations into a small number of command buffers.

| Metric | Baseline (per-op) | ExecGraph |
|--------|-------------------|-----------|
| CBs per layer | 65 | 5 |
| CB reduction | -- | 92.3% |
| CPU-GPU sync points per layer | ~65 | ~1 |
| CPU-GPU sync reduction | -- | 98.5% |
| Latency per layer | ~112ms | ~6.4ms |
| **Speedup** | -- | **17.4x** |
| Latency reduction | -- | 94.3% |

Numerical parity is maintained: max_diff = 6.4e-6 between baseline and ExecGraph outputs.

**Phase KO Update:** The ExecGraph pipeline has been further optimized with the 9-dispatch decode path:

| Metric | Phase 9B-opt | Phase KO (9-dispatch) |
|--------|--------------|-----------------------|
| Dispatches per layer | 65 in 5 CBs | 9 in 1 CB (4 encoders) |
| Latency (single layer) | ~6.4ms | ~1.7ms |
| Latency (60L pipeline) | — | 751 us/L |
| Speedup vs baseline | 17.4x | 64x |
| vs MLX compiled (60L) | ~4.8x slower | **6.34x faster** |
| Latency (Cached 2-enc, 60L) | — | 714 us/L (6x lower σ) |
| Latency (Fused 7-dispatch, 60L) | — | **703.4 us/L** (Phase 10 best) |

The 9-dispatch path is 6.34x faster than MLX's compiled execution at 60-layer depth through merged QKV/gate_up weight projections, batched RoPE, slab-layout SDPA decode, fused GEMV+bias, StorageModePrivate weights, and Array::uninit for output buffers. Multi-layer CB amortization reduces per-layer overhead from 1,739 us (single) to 751 us/L (60L).

### 2.3 Zero-Copy RDMA Data Path

MLX copies data to a separate buffer via `std::copy` before RDMA transmission. RMLX eliminates all copies on the hot path.

```
posix_memalign  ->  newBufferWithBytesNoCopy  ->  ibv_reg_mr
    (CPU)              (Metal GPU view)           (RDMA MR)
         \                   |                    /
          +----- same physical address ----------+
```

Three views share a single physical address. GPU compute results are transmitted over RDMA without any intermediate copy.

### 2.4 CPU-Minimal Execution

ExecGraph reduces the CPU's role to a thin submission layer. Per-layer CPU-GPU synchronization drops from ~65 points to ~1 (98.5% reduction). This is critical on Apple Silicon where the CPU and GPU share the same memory bus -- reducing CPU activity during inference frees bandwidth for the GPU.

### 2.5 Single Rust Binary

| Aspect | MLX | RMLX |
|--------|-----|------|
| Build system | CMake + Python setuptools + nanobind | `cargo build` (single command) |
| Runtime | Python interpreter + C++ shared libraries | Single static binary |
| Deployment | pip install + dependencies | Copy binary |
| Package management | pip + CMake find_package | Cargo.toml |

No Python interpreter overhead. No dynamic library resolution at startup.

### 2.6 Expert Parallelism (EP)

MLX has no built-in Expert Parallelism support. Users must implement MoE dispatch/combine manually. RMLX provides a complete EP stack as a first-class feature:

| Component | RMLX | MLX |
|-----------|------|-----|
| MoE dispatch/combine | 3-zone auto backend (CPU/Metal/RDMA) | Manual implementation |
| MoE Metal kernels | 7 dedicated GPU kernels | None |
| GPU routing (EP-1) | Fused softmax -> top-k -> normalize -> histogram -> prefix-scan | N/A |
| Expert compute (EP-2) | Grouped expert GEMM + stacked weights (ExpertGroup + GatherMM f16/bf16) | N/A |
| Wire protocol (EP-3) | Variable-length v3 protocol with packed PacketMeta + 16B alignment | N/A |
| Overlap engine (EP-4) | TBO + SBO MoePipeline with GpuEvent chains | N/A |
| FP8 exchange (EP-5) | Per-token E4M3 wire format + fused dequant-scatter | N/A |
| Sparse launch transport (EP-6) | ICB sparse expert launch + RDMA slab ring | N/A |
| SparseGuard | Overflow monitoring + capacity auto-tuning | N/A |
| Zero-copy EP path | ibv_mr + Metal buffer on same physical address | N/A |

The 3-zone policy selects CPU for small payloads (N ≤ 64), Metal GPU for medium (N ≥ 320), and RDMA for inter-node communication. EP-1 through EP-6 further eliminate routing/launch/protocol bottlenecks and reduce communication bytes under load imbalance.

### 2.7 Ownership Safety

Rust's ownership system isolates `unsafe` Metal/RDMA FFI boundaries at compile time. The public API surface is entirely safe Rust. Internally, `unsafe` blocks for `posix_memalign`, `newBufferWithBytesNoCopy`, and `ibv_post_send` are explicitly bounded with `SAFETY` comments. Data races in concurrent GPU/RDMA code paths are caught at compile time via `Send`/`Sync` traits.

---

## 3. Where RMLX is Superior to CUDA

### 3.1 Unified Memory Architecture (UMA)

Apple Silicon's UMA means GPU and CPU share the same physical memory. There is no PCIe bus between them.

| Operation | CUDA | RMLX (UMA) |
|-----------|------|------------|
| Host-to-device transfer | `cudaMemcpy` over PCIe (32-64 GB/s) | Not needed -- same memory |
| Device-to-host readback | `cudaMemcpyDeviceToHost` | Direct pointer access |
| Memory capacity | Limited by VRAM (24-80GB typical) | Full system RAM (up to 512GB) |

For models that exceed VRAM capacity on discrete GPUs, UMA provides a significant advantage. A Mac Studio M3 Ultra with 512GB can hold a ~250B parameter model in memory without offloading.

### 3.2 Single-Binary Deployment

| Aspect | CUDA stack | RMLX |
|--------|------------|------|
| Driver | NVIDIA driver (version-specific) | macOS built-in Metal driver |
| Runtime | CUDA toolkit + cuDNN + cuBLAS | None (Metal ships with macOS) |
| Python | Python + PyTorch + transformers | Not needed |
| Binary | Multiple shared libraries | Single static binary |

No driver compatibility matrix. No `CUDA_HOME` environment variable. No `nvidia-smi` debugging.

### 3.3 Power Efficiency

Apple Silicon (5nm / 3nm process) operates at significantly lower power than datacenter GPUs.

| Platform | Typical Power | Use Case |
|----------|---------------|----------|
| M3 Ultra (Mac Studio) | ~100W total system | Desktop inference |
| NVIDIA H100 (SXM5) | 700W per GPU | Datacenter |
| NVIDIA A100 (SXM4) | 400W per GPU | Datacenter |

For inference workloads that fit in UMA, the performance-per-watt advantage is substantial.

### 3.4 Zero-Copy Buffer Sharing

RMLX's dual-registration technique (Metal buffer + RDMA MR on the same physical address) provides a simpler zero-copy path than CUDA's GPUDirect RDMA, which requires kernel-level pinning and driver coordination.

---

## 4. Where RMLX Falls Behind MLX

This section is intentionally thorough. Honest assessment of limitations is more useful than marketing.

### 4.1 Flash Attention — Now Implemented

RMLX now implements Flash Attention 2 with K/V outer loop, head_dim up to 256, a decode fast path for single-query tokens, and causal mask block skipping. This closes the attention efficiency gap with MLX's fused SDPA.

### 4.2 Smaller Ecosystem

MLX has a large Python community with pip-installable packages, Hugging Face integrations, and active third-party contributions. RMLX is a single-team project with no public package registry presence.

### 4.3 No Python API

The machine learning research community overwhelmingly uses Python. RMLX is Rust-only, which limits adoption among researchers. Providing a Python binding layer (via PyO3) is not yet on the roadmap.

### 4.4 Less Mature Quantization

| Format | RMLX | MLX |
|--------|------|-----|
| Q4_0 / Q4_1 | Yes | Yes |
| Q8_0 | Yes | Yes |
| GGUF | Yes | Yes |
| AWQ | Yes | Yes |
| GPTQ | Yes | Yes |
| FP8 | Yes (E4M3/E5M2) | Partial |

RMLX now supports loading GGUF files and dequantizing AWQ/GPTQ packed weights. FP8 (E4M3/E5M2) dtypes with dequant/quant kernels are also available. The quantization gap with MLX has been closed.

### 4.5 No Graph Optimization

MLX uses lazy evaluation with graph-level fusion. The graph compiler can fuse element-wise operations, eliminate dead code, and optimize memory allocation across an entire computation graph. RMLX uses eager-first execution with selective tracing, which provides lower per-op latency but misses graph-level optimization opportunities.

### 4.6 No Training Support

MLX supports full training with automatic differentiation, optimizer states, and gradient accumulation. RMLX has only VJP autodiff with LoRA fine-tuning support. Full pre-training or full-parameter fine-tuning is not supported.

---

## 5. Where RMLX Falls Behind CUDA

The gap between RMLX and the CUDA ecosystem is larger than the gap between RMLX and MLX. This is expected -- CUDA has had decades of investment.

### 5.1 Flash Attention Gap Narrowed

RMLX now implements Flash Attention 2 with online softmax and tiled K/V processing. However, NVIDIA's FlashAttention-3 leverages Tensor Core-specific optimizations (warp-level MMA instructions, asynchronous copy) that are not applicable to Apple Silicon. The CUDA implementation remains more optimized for its hardware.

### 5.2 Interconnect Bandwidth

| Interconnect | Per-Link Bandwidth | Typical Config | Total |
|--------------|-------------------|----------------|-------|
| Thunderbolt 5 | ~16 GB/s per port | 2 ports | ~32 GB/s |
| NVLink (H100) | 50 GB/s per link | 12 links | 600 GB/s |
| **Ratio** | | | **~18.7x disadvantage** |

For distributed inference, this bandwidth gap limits RMLX's scaling efficiency for communication-heavy workloads (e.g., tensor parallelism).

### 5.6 No CUDA Graphs Capture-Replay

CUDA Graphs captures a sequence of GPU operations and replays them with near-zero CPU overhead. RMLX's ExecGraph re-encodes a deterministic sequence each time, which is faster than per-op submission but slower than true replay. See Section 6 for a detailed comparison.

### 5.7 No Tensor Cores Equivalent

NVIDIA GPUs have dedicated Tensor Cores for matrix multiplication (FP16, BF16, FP8, INT8). Apple GPUs use SIMD groups for matrix operations, which provide good throughput but lack dedicated fixed-function matrix hardware.

### 5.8 Ecosystem Maturity

CUDA has decades of optimization across compilers (NVCC, Triton), libraries (cuBLAS, cuDNN, NCCL, cuSPARSE), profiling tools (Nsight, nvprof), and community knowledge. RMLX cannot match this breadth.

---

## 6. ExecGraph vs CUDA Graphs -- Detailed Comparison

| Aspect | ExecGraph (RMLX) | CUDA Graphs |
|--------|-------------------|-------------|
| **Mechanism** | Re-encode deterministic op sequence into batched CBs | Capture GPU operations, replay recorded stream |
| **CPU overhead** | Re-encoding cost per inference step | Near-zero (replay only) |
| **Flexibility** | Always deterministic; works for any shape | Requires re-capture when input shapes change |
| **CB reduction** | 65 -> 1 per layer (98.5%) | Full coalescing into single graph |
| **Sync model** | MTLSharedEvent (non-blocking) | CUDA events (stream-ordered) |
| **Shape dynamism** | Re-encode handles shape changes naturally | Must re-capture or use CUDA Graph updates |
| **Latency** | ~0.70ms per layer at 60L (fused 7-dispatch) | Sub-millisecond replay |
| **Speedup over baseline** | 64x | Typically 2-5x (already from efficient baseline) |
| **Implementation complexity** | Moderate (deterministic sequencing) | Low (capture API is straightforward) |
| **Memory overhead** | Minimal (re-encode reuses buffers) | Graph storage (captured operations) |

**Key insight**: ExecGraph's 64x speedup comes from collapsing the decode path into 9 dispatches in a single command buffer. CUDA's baseline is already more efficient due to stream-ordered execution, so CUDA Graphs provides a smaller relative improvement from a stronger starting point.

**Phase A update**: The prefill path now also uses a single-CB pipeline (54 sync points reduced to 1), GQA slab SDPA (32 per-head dispatches reduced to 1), and GEMM threadgroup swizzle. Single-layer prefill achieves 3.5-7.3x speedup over baseline, with MLX parity within 1.2-3.4x. The remaining gap is primarily in GEMM throughput (rmlx 13 TFLOPS vs MLX 24 TFLOPS).

---

## 7. Memory Model Comparison

### UMA (Apple Silicon) vs Discrete GPU Memory (NVIDIA)

```
Apple Silicon UMA:
┌────────────────────────────────────┐
│        512GB Unified Memory        │
│   ┌──────────┐  ┌──────────────┐   │
│   │ CPU cores │  │ GPU cores    │   │
│   │ (read/    │  │ (read/       │   │
│   │  write)   │  │  write)      │   │
│   └──────────┘  └──────────────┘   │
└────────────────────────────────────┘

NVIDIA Discrete:
┌───────────────┐    PCIe/NVLink    ┌─────────────────┐
│  Host RAM     │ <===============> │  GPU VRAM       │
│  (CPU access) │    32-64 GB/s     │  (24-80GB HBM)  │
│               │                   │  ~3.35 TB/s BW  │
└───────────────┘                   └─────────────────┘
```

| Aspect | UMA (RMLX / MLX) | Discrete GPU (CUDA) |
|--------|-------------------|---------------------|
| **Memory capacity** | Up to 512GB (M3 Ultra) | 24-80GB VRAM typical |
| **CPU-GPU transfer** | Zero-copy (same physical memory) | Explicit cudaMemcpy required |
| **Memory bandwidth** | ~400 GB/s | ~3.35 TB/s (H100 HBM3e) |
| **Bandwidth ratio** | 1x | ~8.4x advantage to discrete |
| **Large model support** | Can hold ~250B params in 512GB | Requires multi-GPU or offloading beyond VRAM |
| **RDMA integration** | Dual-register same physical address | GPUDirect RDMA (kernel driver required) |

**UMA advantages**: No copy overhead, larger effective capacity, simpler programming model, zero-copy RDMA path.

**Discrete advantages**: Far higher memory bandwidth (HBM3e: 3.35 TB/s vs UMA: ~400 GB/s), dedicated VRAM isolates GPU workloads from CPU memory pressure.

---

## 8. Quantization Support Comparison

| Format | RMLX | MLX | CUDA Ecosystem |
|--------|------|-----|----------------|
| Q4_0 (4-bit block, absmax) | Yes | Yes | Via llama.cpp |
| Q4_1 (4-bit block, min+scale) | Yes | Yes | Via llama.cpp |
| Q8_0 (8-bit block) | Yes | Yes | Via llama.cpp |
| GGUF (llama.cpp format) | Yes | Yes | Yes (native llama.cpp) |
| AWQ (activation-aware) | Yes | Yes | Yes (AutoAWQ) |
| GPTQ (post-training) | Yes | Yes | Yes (AutoGPTQ) |
| FP8 (E4M3/E5M2) | Yes | Partial | Yes (H100+, native) |
| INT4 (NVIDIA) | No | No | Yes (TensorRT-LLM) |
| INT8 (smooth quant) | No | Partial | Yes (TensorRT-LLM) |
| W4A16 (weight 4-bit, act 16-bit) | Yes (Q4_0/Q4_1) | Yes | Yes |

RMLX now supports GGUF, AWQ, GPTQ, and FP8 formats in addition to the three basic block-quantization formats. The remaining gaps are NVIDIA-specific formats (INT4/INT8 via TensorRT-LLM).

---

## 9. Numerical Parity

RMLX maintains strict numerical parity between its baseline execution path and the ExecGraph-optimized path.

| Metric | Value |
|--------|-------|
| max_diff (baseline vs ExecGraph) | 6.4e-6 |
| Tolerance threshold | 1e-4 |
| Test coverage | 1,298+ tests |
| Verification method | Element-wise comparison across all ops |

This ensures that the 17.4x performance improvement from ExecGraph introduces no meaningful numerical drift.

---

## 10. Completed Parity Items

The following gaps identified in earlier versions have been closed:

| Phase | Focus | Key Deliverables | Gap Addressed |
|-------|-------|-----------------|---------------|
| **Phase S3a** | Attention Optimization | Flash Attention 2 (K/V outer loop, D≤256, decode fast path) | Sections 4.1 (closed) |
| **Phase S2** | Advanced Quantization | GGUF loader, AWQ/GPTQ dequant, FP8 dtypes | Section 4.4 (closed) |
| **Phase A** | Prefill Optimization | Single-CB pipeline, GQA slab SDPA, GEMM swizzle, 3.5-7.3x speedup | Prefill performance gap narrowed |

The Phase 0+1+2 full-crate audit added 76 items including GatherMM, LayerNorm, unary ops, QuantizedLinear, MLA, sliding window attention, GGUF loading, 14 activation functions, ring/allreduce collectives, connection manager, and coordinator.

Remaining gaps: Python API, speculative decoding, and NVIDIA-specific quantization formats (INT4/INT8).

---

## Summary

RMLX is not a replacement for MLX or CUDA. It occupies a specific niche: **high-performance Metal GPU ML framework for Apple Silicon, written in Rust, optimized for distributed execution over Thunderbolt 5 RDMA**.

**Choose RMLX when**: You need maximum GPU performance on Apple Silicon hardware, you want Expert Parallelism with zero-copy RDMA for MoE models on Mac Studio clusters, you want a single Rust binary with no Python dependency, or you are building a distributed inference cluster.

**Choose MLX when**: You need Python compatibility, a mature ecosystem, training support, or broad quantization format support.

**Choose CUDA when**: You need maximum absolute performance, production serving at scale, speculative decoding, or access to the largest ecosystem of tools and libraries.
