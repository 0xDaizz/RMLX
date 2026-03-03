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
| **CB management** | ExecGraph: 65 CBs/layer -> 5 CBs/layer (92.3% reduction) | 1 CB per eval() batch | Stream-ordered, CUDA Graphs capture-replay |
| **Sync mechanism** | MTLSharedEvent signal/wait (263.9us) | waitUntilCompleted (424.9us) | CUDA events / streams |
| **RDMA** | Zero-copy: posix_memalign + NoCopy + ibv_reg_mr | std::copy to transfer buffer | GPUDirect RDMA / NVLink |
| **Quantization** | Q4_0, Q4_1, Q8_0 | Q4_0, Q4_1, Q8_0, GGUF, AWQ, GPTQ | FP8, AWQ, GPTQ, INT4/INT8 |
| **Flash Attention** | No (fused SDPA only) | No (fused SDPA) | FlashAttention-2/3 |
| **KV cache** | Static per-layer cache | Static per-layer cache | PagedAttention (vLLM) |
| **Training** | LoRA fine-tuning only | Full training support | Full training + LoRA + QLoRA |
| **Op modules** | 14 | ~50+ | Hundreds |
| **Test suite** | 339+ tests | Extensive | Extensive |
| **Phases complete** | 0-9B-opt | N/A (stable release) | N/A (stable release) |

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
| Latency per layer | 110.4ms | 6.8ms |
| **Speedup** | -- | **16.15x** |
| Latency reduction | -- | 93.8% |

Numerical parity is maintained: max_diff = 6.4e-6 between baseline and ExecGraph outputs.

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

### 2.6 Ownership Safety

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

### 4.1 No Flash Attention

RMLX implements fused SDPA (Scaled Dot-Product Attention) but not Flash Attention 2 with tiled K/V and online softmax. Flash Attention reduces memory usage from O(N^2) to O(N) for the attention matrix, which is critical for long-context inference.

MLX also lacks full Flash Attention, but its fused SDPA implementation is more mature.

### 4.2 Smaller Ecosystem

MLX has a large Python community with pip-installable packages, Hugging Face integrations, and active third-party contributions. RMLX is a single-team project with no public package registry presence.

### 4.3 No Python API

The machine learning research community overwhelmingly uses Python. RMLX is Rust-only, which limits adoption among researchers. Providing a Python binding layer (via PyO3) is not yet on the roadmap.

### 4.4 Less Mature Quantization

| Format | RMLX | MLX |
|--------|------|-----|
| Q4_0 / Q4_1 | Yes | Yes |
| Q8_0 | Yes | Yes |
| GGUF | No | Yes |
| AWQ | No | Yes |
| GPTQ | No | Yes |
| FP8 | No | Partial |

MLX supports loading GGUF, AWQ, and GPTQ quantized models directly. RMLX supports only block-quantized Q4_0, Q4_1, and Q8_0.

### 4.5 No Graph Optimization

MLX uses lazy evaluation with graph-level fusion. The graph compiler can fuse element-wise operations, eliminate dead code, and optimize memory allocation across an entire computation graph. RMLX uses eager-first execution with selective tracing, which provides lower per-op latency but misses graph-level optimization opportunities.

### 4.6 No Training Support

MLX supports full training with automatic differentiation, optimizer states, and gradient accumulation. RMLX has only VJP autodiff with LoRA fine-tuning support. Full pre-training or full-parameter fine-tuning is not supported.

---

## 5. Where RMLX Falls Behind CUDA

The gap between RMLX and the CUDA ecosystem is larger than the gap between RMLX and MLX. This is expected -- CUDA has had decades of investment.

### 5.1 No Flash Attention

FlashAttention-2 and FlashAttention-3 are standard in the CUDA ecosystem. They provide both memory efficiency (O(N) instead of O(N^2)) and computational efficiency (better SRAM utilization on NVIDIA hardware). RMLX's fused SDPA does not achieve comparable efficiency for long sequences.

### 5.2 No Paged KV Cache

vLLM's PagedAttention manages KV cache memory like virtual memory pages, enabling efficient memory sharing across requests and near-zero waste. RMLX uses static per-layer KV cache allocation.

### 5.3 No Speculative Decoding

Speculative decoding (Medusa, EAGLE, draft-model approaches) is standard in CUDA serving stacks. It can provide 2-3x throughput improvement for autoregressive generation. RMLX does not implement any form of speculative decoding.

### 5.4 No Continuous Batching

Production CUDA serving engines (vLLM, TensorRT-LLM, SGLang) use continuous batching to maximize GPU utilization by dynamically adding and removing requests from a batch. RMLX does not support this.

### 5.5 Interconnect Bandwidth

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
| **CB reduction** | 65 -> 5 per layer (92.3%) | Full coalescing into single graph |
| **Sync model** | MTLSharedEvent (non-blocking) | CUDA events (stream-ordered) |
| **Shape dynamism** | Re-encode handles shape changes naturally | Must re-capture or use CUDA Graph updates |
| **Latency** | 6.8ms per layer | Sub-millisecond replay |
| **Speedup over baseline** | 16.15x | Typically 2-5x (already from efficient baseline) |
| **Implementation complexity** | Moderate (deterministic sequencing) | Low (capture API is straightforward) |
| **Memory overhead** | Minimal (re-encode reuses buffers) | Graph storage (captured operations) |

**Key insight**: ExecGraph's 16.15x speedup is relative to a per-op baseline that is more overhead-heavy than CUDA's default execution model. CUDA's baseline is already more efficient due to stream-ordered execution, so CUDA Graphs provides a smaller relative improvement from a stronger starting point.

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
| GGUF (llama.cpp format) | No | Yes | Yes (native llama.cpp) |
| AWQ (activation-aware) | No | Yes | Yes (AutoAWQ) |
| GPTQ (post-training) | No | Yes | Yes (AutoGPTQ) |
| FP8 (E4M3/E5M2) | No | Partial | Yes (H100+, native) |
| INT4 (NVIDIA) | No | No | Yes (TensorRT-LLM) |
| INT8 (smooth quant) | No | Partial | Yes (TensorRT-LLM) |
| W4A16 (weight 4-bit, act 16-bit) | Yes (Q4_0/Q4_1) | Yes | Yes |

RMLX's quantization coverage is limited to the three block-quantization formats needed for basic inference. MLX and the CUDA ecosystem support significantly more formats, enabling better model quality at lower bit widths.

---

## 9. Numerical Parity

RMLX maintains strict numerical parity between its baseline execution path and the ExecGraph-optimized path.

| Metric | Value |
|--------|-------|
| max_diff (baseline vs ExecGraph) | 6.4e-6 |
| Tolerance threshold | 1e-4 |
| Test coverage | 339+ tests |
| Verification method | Element-wise comparison across all ops |

This ensures that the 16.15x performance improvement from ExecGraph introduces no meaningful numerical drift.

---

## 10. Roadmap to Parity

RMLX is actively working to close the gaps identified in Sections 4 and 5. The planned phases are:

| Phase | Focus | Key Deliverables | Gap Addressed |
|-------|-------|-----------------|---------------|
| **Phase 10** | Attention Optimization | Flash Attention 2 with tiled K/V, Paged KV Cache | Sections 4.1, 5.1, 5.2 |
| **Phase 11** | Serving Optimization | Continuous batching, speculative decoding, dynamic scheduling | Sections 5.3, 5.4 |
| **Phase 12** | Advanced Quantization | GGUF loader, AWQ, GPTQ, FP8 support | Section 4.4, 8 |

These phases are not yet scheduled. The current focus (Phases 0-9B-opt) prioritizes the Metal GPU pipeline and ExecGraph optimization.

---

## Summary

RMLX is not a replacement for MLX or CUDA. It occupies a specific niche: **high-performance Metal GPU inference on Apple Silicon, written in Rust, optimized for distributed execution over Thunderbolt 5 RDMA**.

**Choose RMLX when**: You need maximum inference performance on Apple Silicon hardware, you want a single Rust binary with no Python dependency, or you are building a distributed inference cluster of Mac Studios.

**Choose MLX when**: You need Python compatibility, a mature ecosystem, training support, or broad quantization format support.

**Choose CUDA when**: You need maximum absolute performance, production serving at scale, Flash Attention, speculative decoding, or access to the largest ecosystem of tools and libraries.
