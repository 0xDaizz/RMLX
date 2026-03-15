<p align="center">
  <!-- Logo coming soon -->
  <h1 align="center">RMLX</h1>
  <p align="center">
    <strong>Rust ML runtime for Apple Silicon — zero-copy Metal GPU pipeline with RDMA distributed inference</strong>
  </p>
  <p align="center">
    <em>156× decode speedup · 12.1× faster than MLX compiled · 46× Split-CB TP · 5.0× vs MLX at TP=2 · EP 30-178× vs MLX</em>
  </p>
  <p align="center">
    <a href="https://github.com/0xDaizz/RMLX/actions/workflows/ci.yml"><img src="https://github.com/0xDaizz/RMLX/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.80%2B-orange.svg" alt="Rust 1.80+"></a>
    <img src="https://img.shields.io/badge/tests-1298%20passing-brightgreen.svg" alt="Tests">
    <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg" alt="macOS Apple Silicon">
  </p>
  <p align="center">
    <a href="#-install">Install</a> ·
    <a href="#-quickstart">Quickstart</a> ·
    <a href="#-performance">Performance</a> ·
    <a href="#-features">Features</a> ·
    <a href="#%EF%B8%8F-architecture">Architecture</a> ·
    <a href="#-roadmap">Roadmap</a> ·
    <a href="#-docs">Docs</a>
  </p>
</p>

---

> 🇰🇷 한국어 문서: [docs/README_ko.md](docs/README_ko.md)

## 🧠 What is RMLX?

RMLX reimplements Apple's [MLX](https://github.com/ml-explore/mlx) Metal GPU pipeline **entirely in Rust**, built on `objc2-metal` / `objc2` / `block2` / `objc2-foundation`. The framework is organized into **7 crates** spanning GPU compute, memory allocation, neural network layers, and RDMA-based distributed inference.

The fused 7-dispatch decode path achieves **703 μs/layer** (73.6% bandwidth efficiency) — the practical floor for f16 decode on Apple Silicon. FP16 GEMM reaches **24.05 TFLOPS** (MLX parity), QMM Q4 hits **17.43 TFLOPS** (+28% vs MLX), and 2-node RDMA tensor parallelism delivers **5.0× faster** decode than MLX TP=2 (378 μs vs 1,880 μs). Expert parallelism achieves **30–178× vs MLX** per-expert, with buffer-pooled MoE grouped forward cutting latency by **68×**.

Single Rust binary. Zero-copy unified memory. No Python runtime, no framework overhead.

## 📦 Install

**Prerequisites:** macOS 14+ on Apple Silicon (M1 or later). See [full prerequisites](docs/getting-started/prerequisites.md).

```bash
git clone https://github.com/0xDaizz/RMLX.git
cd RMLX
cargo build --workspace
```

## 🚀 Quickstart

### Build & Test

```bash
cargo build --workspace           # Build all 7 crates
cargo test  --workspace           # Run 1,298 tests
```

### Benchmark

```bash
cargo bench -p rmlx-nn --bench pipeline_bench
```

### Distributed RDMA (2-node)

```bash
# One-time install
cargo install --path crates/rmlx-cli

# Auto-detect TB5 topology, assign IPs, configure interfaces
rmlx config --hosts node1,node2 --auto-setup --output rmlx-hosts.json --verbose

# Launch distributed job
rmlx launch --backend rdma --hostfile rmlx-hosts.json -- ibv_devices
```

> `--auto-setup` discovers Thunderbolt connections via `system_profiler`, assigns point-to-point IPs, and configures RDMA interfaces automatically.

## 📊 Performance

All numbers measured on Apple Silicon (M3 Ultra), single transformer layer, Qwen 3.5 MoE A22B config, f16 unless noted.

### ⚡ Decode — 110 ms → 703 μs (156×)

| Stage | Latency | vs Naive | Key Technique |
|:------|--------:|---------:|:--------------|
| Naive (per-op sync) | 110 ms | 1× | 65 CBs, GPU idle between dispatches |
| ExecGraph | 2.8 ms | 39× | CB batching 65 → 5 |
| 9-Dispatch + PSO Cache | 1,081 μs | 102× | Single-CB decode, kernel optimization |
| f16 + CachedDecode (60L) | 714 μs | 154× | f16 default, buffer reuse, 2-encoder pipeline |
| **7-Dispatch Fusion (60L)** | **703 μs** | **156×** | fused_rms_gemv + fused_swiglu_down |

> **BW efficiency 73.6%** — practical floor for f16 decode on Apple Silicon (theoretical min ~520 μs).

### ⚡ TP Decode — 12.1× vs MLX (2-node TB5 RDMA)

| Config | RMLX | MLX | Speedup |
|:-------|-----:|----:|--------:|
| TP=1 single-CB | 182 μs | 2,197 μs (mx.compile) | **12.1×** |
| TP=2 Split-CB | 378 μs | 1,880 μs (JACCL) | **5.0×** |
| Per-op → Split-CB | 18,193 μs → 378 μs | — | **48×** (architecture) |
| RDMA allreduce (1×) | 14 μs | 87 μs (JACCL 2×) | **6.2×** |

> Split-CB TP batches an entire layer into 2 command buffers with inter-CB allreduce, eliminating per-op GPU sync overhead.

### ⚡ TP Prefill — Split-CB (2-node TB5 RDMA)

| Config | M=128 | M=512 |
|:-------|------:|------:|
| RMLX TP=1 | 5,048 μs | 12,616 μs |
| RMLX TP=2 Split-CB | 1,925 μs | 5,466 μs |
| MLX TP=1 | 3,890 μs | 11,795 μs |
| MLX TP=2 JACCL | 3,700 μs | 8,049 μs |

> New `forward_prefill_with_group_split_cb()` method enables TP=2 prefill with proper GPU timing via commandBuffer() isolation.

### ⚡ Prefill GEMM — 24.05 TFLOPS

| Metric | RMLX | MLX | Delta |
|:-------|-----:|----:|:------|
| FP16 GEMM (M=512) | 24.05T | 24T | Pipe parity |
| FP16 GEMM peak | 46.3T | ~23T | 2× MLX |
| Small-M dispatch (M=32–256) | 1.24–2.83× | 1× | RMLX faster |
| QMM Q4 (M=512) | 17.43T | 13.6T | +28% |
| E2E prefill (seq ≥ 256) | — | — | MLX parity |

### ⚡ Expert Parallelism — 30–178× vs MLX

| Config | RMLX | MLX (mx.compile) | Speedup |
|:-------|-----:|------------------:|--------:|
| Single expert FFN (M=1..512) | 42–54 μs | 1,338–9,609 μs | **30–178×** |
| MoE grouped seq=4 (8 experts) | 359 μs | — | 68× vs pre-pooling |
| MoE grouped seq=32 | 665 μs | — | — |
| MoE grouped seq=128 | 1,658 μs | — | — |

> Buffer-pooled `grouped_forward`: 32 allocations → 4 bulk allocations (14 ms → 359 μs, 39×). `commandBufferWithUnretainedReferences` removes CB retain/release overhead.

### ⚡ EP-2 End-to-End (4 experts/rank + RDMA)

| Seq Length | RMLX EP-2 | MLX EP-2 (JACCL) | Speedup |
|:-----------|----------:|------------------:|--------:|
| seq=4 | 233 μs | 6,895 μs | **30×** |
| seq=32 | 429 μs | — | — |
| seq=64 | 672 μs | — | — |

> EP-2 e2e seq=4: was 12,537 μs before buffer pooling — **54× improvement**.

### ⚡ RDMA vs JACCL Transport

| Payload | RMLX RDMA | MLX JACCL | Speedup |
|:--------|----------:|----------:|--------:|
| 28 KB | 12 μs | 79 μs | **6.6×** |
| 896 KB | 97 μs | 308 μs | **3.2×** |

## ✨ Features

### RMLX vs MLX vs CUDA

| Feature | RMLX | MLX | CUDA |
|:--------|:----:|:---:|:----:|
| Unified memory (zero-copy) | ✅ | ✅ | ❌ |
| 7-dispatch fused decode | ✅ | ❌ | ❌ |
| Single-CB prefill pipeline | ✅ | ❌ | ❌ |
| Expert parallelism (MoE, 30–178×) | ✅ | ❌ | ⚠️ |
| Zero-copy RDMA | ✅ | ❌ | ❌ |
| Flash Attention 2 | ✅ | ✅ | ✅ |
| MLA (DeepSeek-V3) | ✅ | ❌ | ⚠️ |
| GGUF model loading | ✅ | ✅ | ✅ |
| Quantized inference (Q4/Q8) | ✅ | ✅ | ✅ |
| Single Rust binary | ✅ | ❌ | ❌ |
| Metal 4 support (macOS 26+) | ✅ | ❌ | ❌ |

### 🔧 Key Capabilities

<details open>
<summary><strong>32+ GPU Ops</strong></summary>

- Flash Attention 2 Metal kernel (tiled online softmax, D up to 256)
- SIMD group MMA matmul, BM=8 GEMV with dynamic tile selection
- Batched SDPA decode with slab KV cache
- FP8 (E4M3/E5M2), AWQ/GPTQ INT4, K-quant (Q2K–Q6K)
- Fused kernels: SiLU-mul, RMSNorm+residual, GEMV+bias, GEMM+residual epilogue
- GEMM: MLX-architecture kernel (BK=16, 2 SG, serpentine MMA)
- QMM MMA Q4/Q8, QMV qdot pattern — no CPU fallback

</details>

<details open>
<summary><strong>Infrastructure</strong></summary>

- **ExecGraph**: command buffer batching (65 CB → 5)
- **CachedDecode**: pre-resolved PSOs, zero per-token allocation
- **Metal**: `objc2-metal 0.3` with ComputePass zero-cost abstraction, ChipTuning (M1–M4), DiskPipelineCache
- **Allocator**: zero-copy (posix_memalign + MTLBuffer), BFC, residency manager
- **RDMA**: ibverbs FFI, TB5 multi-port, ring/allreduce/allgather collectives
- **Distributed**: TP with Split-CB, expert parallelism (3-zone auto), tree allreduce, topology-aware CLI

</details>

<details>
<summary><strong>Neural Network Layers</strong></summary>

- **Models**: Qwen 3.5, DeepSeek-V3, Mixtral, Kimi K2.5
- **Attention**: Multi-Head, GQA, MLA, Sliding Window
- **KV cache**: static, rotating, paged (vLLM-style), quantized, slab decode
- **Quantization**: QuantizedLinear, AWQ, GPTQ, K-quant
- **Loading**: GGUF v2/v3 with tensor mapping

</details>

## 🏗️ Architecture

```mermaid
graph TD
    CLI[🖥️ rmlx-cli] --> NN[🧠 rmlx-nn]
    NN --> CORE[⚙️ rmlx-core]
    CORE --> METAL[🔩 rmlx-metal]
    CORE --> ALLOC[📦 rmlx-alloc]
    DIST[🌐 rmlx-distributed] --> CORE
    DIST --> RDMA[🔗 rmlx-rdma]
    ALLOC --> RDMA
    METAL -.-> ALLOC
```

| Crate | Role |
|:------|:-----|
| **rmlx-cli** | Launch, config, topology discovery |
| **rmlx-nn** | Models, attention, MoE, KV cache, GGUF loader |
| **rmlx-core** | 32+ op modules, Array/DType, autodiff |
| **rmlx-metal** | Device, ExecGraph, ChipTuning, pipeline cache, ComputePass (`objc2-metal`), Metal 4 |
| **rmlx-alloc** | Zero-copy allocator, BFC, residency manager |
| **rmlx-distributed** | Expert parallelism, allreduce, topology, TP |
| **rmlx-rdma** | ibverbs FFI, TB5 multi-port, collectives |

## 🗺️ Roadmap

| Era | Phases | Key Result | Status |
|:----|:-------|:-----------|:------:|
| **Foundation** | Phase 0 → 7C | Core framework, Metal bindings, ExecGraph, RDMA infra | ✅ Complete |
| **Decode Optimization** | KO → Phase 11 | 110 ms → 703 μs/layer (156×, 73.6% BW) | ✅ Complete |
| **GEMM & Prefill** | Phase A → D | 24.05T TFLOPS, MLX parity, single-CB prefill | ✅ Complete |
| **Quantized Kernels** | Phase F → J | QMM Q4 17.43T (+28% vs MLX), QMV near-parity | ✅ Complete |
| **Distributed RDMA** | EP-1 → 6, RDMA-7 | TP=2 5.0× vs MLX, Split-CB, 14 μs allreduce | ✅ Complete |
| **EP + Buffer Pooling** | Phase 8 | EP 30–178× vs MLX, MoE 68× improvement, EP-2 e2e 54× | ✅ Complete |
| **Next** | KO-2, KO-3, EP-7 | Multi-token decode, speculative decoding, EP scaling | 🔜 Planned |

> See [full roadmap](docs/roadmap/phases.md) and [benchmark report](docs/reports/phase-f-i-benchmark-2026-03-08.md) for details.

## 📚 Docs

| Document | Description |
|:---------|:------------|
| [Architecture Overview](docs/architecture/overview.md) | System design and crate responsibilities |
| [GPU Pipeline](docs/gpu-pipeline.md) | Metal compute pipeline internals |
| [Implementation Roadmap](docs/roadmap/phases.md) | Full phase-by-phase history |
| [RMLX vs MLX vs CUDA](docs/comparison.md) | Detailed framework comparison |
| [Getting Started](docs/getting-started/prerequisites.md) | Prerequisites and setup guide |

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple — the Metal GPU compute framework that RMLX reimplements in Rust
- [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple — LLM inference patterns and Metal kernel references
- [vllm-mlx](https://github.com/nicholasgasior/vllm-mlx) — distributed inference architecture and RDMA transport patterns

## 📄 License

MIT — see [LICENSE](LICENSE).
