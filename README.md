# RMLX

**Rust ML runtime for Apple Silicon — 6.34x faster than MLX at model-scale decode, prefill within 1.2-3.4x of MLX**

[![CI](https://github.com/0xDaizz/RMLX/actions/workflows/ci.yml/badge.svg)](https://github.com/0xDaizz/RMLX/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.80+](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-1298%20passing-brightgreen.svg)]()
[![macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)]()

> 한국어 문서: [docs/README_ko.md](docs/README_ko.md)

---

RMLX reimplements Apple's [MLX](https://github.com/ml-explore/mlx) Metal GPU pipeline **entirely in Rust**. The fused 7-dispatch decode path achieves **703.4 us/layer at 60-layer depth** — **6.34x faster than MLX compiled** (4,525 us/layer) on identical hardware (M3 Ultra). Phase 11 kernel optimization experiments confirmed this as the practical floor for f16 decode on Apple Silicon (73.6% bandwidth efficiency).

## ⚡ Performance

Single transformer layer decode (Llama-2 7B shapes, f16, M3 Ultra):

| Path | Latency | Speedup |
|------|--------:|--------:|
| Baseline (per-op sync) | 111,083 us | 1x |
| ExecGraph (5 CB) | 2,846 us | 39x |
| Single-CB (44 enc) | 2,143 us | 52x |
| **9-Dispatch (single layer)** | **1,081 us** | **103x** |
| **60-layer pipeline** | **751 us/L** | **6.34x vs MLX** |
| **Cached 2-encoder (60L)** | **714 us/L** | **8% faster, 6x lower σ** |
| **Fused 7-dispatch (60L)** | **703.4 us/L** | **Phase 10 best** |
| **Phase A prefill (single-layer)** | **3.5-7.3x vs baseline** | **MLX parity within 1.2-3.4x** |
| MLX compiled (60L, f16) | 4,525 us/L | — |

## ✨ RMLX vs MLX vs CUDA

| Feature | RMLX | MLX | CUDA |
|---------|:----:|:---:|:----:|
| Unified memory (zero-copy) | ✅ | ✅ | ❌ |
| 9-dispatch decode (6.34x vs MLX) | ✅ | ➖ | ➖ |
| Prefill single-CB pipeline | ✅ | ➖ | ➖ |
| Expert parallelism | ✅ | ❌ | ⚠️ |
| Zero-copy RDMA | ✅ | ❌ | ❌ |
| Flash Attention 2 | ✅ | ✅ | ✅ |
| MLA (DeepSeek-V3) | ✅ | ❌ | ⚠️ |
| GGUF model loading | ✅ | ✅ | ✅ |
| Quantized inference | ✅ | ✅ | ✅ |
| Single Rust binary | ✅ | ❌ | ❌ |

## 🛠️ What's Inside

<details open>
<summary><b>32+ GPU ops</b> — matmul, softmax, RMS norm, RoPE, GEMV, SDPA, conv, scan, sort, argreduce, random, ...</summary>

- Flash Attention 2 Metal kernel (tiled online softmax, D up to 256)
- BM=8 GEMV with dynamic tile selection, SIMD group MMA matmul, barrier-free BM8, 4×float4 vectorization
- Batched SDPA decode with slab KV cache
- FP8 (E4M3/E5M2), AWQ/GPTQ INT4, K-quant (Q2K–Q6K)
- Single-pass layer norm, register-cached RMS norm
- Fused kernels: silu-mul, RMSNorm+residual, GEMV+bias
</details>

<details open>
<summary><b>Infrastructure</b> — ExecGraph, 9-dispatch decode, RDMA, BFC allocator</summary>

- **7-dispatch fused decode**: merged QKV/gate-up weights, batched SDPA, fused_rms_gemv + fused_swiglu_down kernel fusion, 703.4 us/layer at 60L depth (6.34x faster than MLX); **CachedDecode** path with pre-resolved PSOs + zero per-token allocation
- **Phase A prefill**: single-CB pipeline (54 sync points to 1), GQA slab SDPA (32 per-head dispatches to 1), GEMM threadgroup swizzle, 3.5-7.3x speedup over baseline
- **ExecGraph**: command buffer batching (65 CB down to 5)
- **Metal**: ChipTuning (M1–M4), DiskPipelineCache, fence manager, dual queues
- **Allocator**: zero-copy (posix_memalign + MTLBuffer), BFC, residency manager
- **RDMA**: ibverbs FFI, TB5 multi-port, ring/allreduce/allgather collectives
- **Distributed**: expert parallelism (3-zone auto), tree allreduce, topology-aware CLI
</details>

<details>
<summary><b>Neural network layers</b> — 4 model architectures, 16 activations, MoE, MLA</summary>

- **Models**: Llama, Qwen2, DeepSeek-V3, Mixtral
- **Attention**: Multi-Head, GQA, MLA, Sliding Window
- **KV cache**: static, rotating, paged (vLLM-style), quantized, slab decode
- **Quantization**: QuantizedLinear, AWQ, GPTQ, K-quant
- **Loading**: GGUF v2/v3 with tensor mapping
</details>

## 🏗️ Architecture

```mermaid
graph TD
    NN[rmlx-nn]
    CORE[rmlx-core]
    DIST[rmlx-distributed]
    METAL[rmlx-metal]
    ALLOC[rmlx-alloc]
    RDMA[rmlx-rdma]
    CLI[rmlx-cli]

    NN --> CORE
    CORE --> METAL
    CORE --> ALLOC
    DIST --> CORE
    DIST --> RDMA
    ALLOC --> RDMA
    METAL -.-> ALLOC
    CLI --> NN
```

| Crate | Role |
|-------|------|
| **rmlx-nn** | Models, attention, MoE, KV cache, GGUF loader |
| **rmlx-core** | 32+ op modules, Array/DType, autodiff |
| **rmlx-metal** | Device, ExecGraph, ChipTuning, pipeline cache |
| **rmlx-alloc** | Zero-copy allocator, BFC, residency |
| **rmlx-distributed** | EP, allreduce, topology |
| **rmlx-rdma** | ibverbs FFI, collectives |
| **rmlx-cli** | Launch, config, topology discovery |

## 🚀 Quick Start

```bash
git clone https://github.com/0xDaizz/RMLX.git && cd RMLX

cargo build --workspace        # Build
cargo test --workspace         # 1,298 tests
cargo bench -p rmlx-nn --bench pipeline_bench  # Benchmark
```

> Requires macOS 14+ on Apple Silicon. See [Prerequisites](docs/getting-started/prerequisites.md).

Distributed 2-node RDMA runbook (minimal):

```bash
# cargo install --path crates/rmlx-cli   (one-time)
rmlx config --hosts node1,node2 --backend rdma --over thunderbolt --output rmlx-hosts.json --verbose
rmlx launch --backend rdma --hostfile rmlx-hosts.json -- ibv_devices
```

## 📊 Stats

| Metric | Value |
|--------|------:|
| Crates | 7 |
| Tests | 1,298 |
| GPU ops | 32+ |
| Activations | 16 |
| Model architectures | 4 |
| Decode latency | 703.4 us/L (60L fused 7-dispatch) |
| Prefill speedup | 3.5-7.3x vs baseline |
| Prefill vs MLX | within 1.2-3.4x |
| vs MLX (60L compiled) | 6.34x faster |

## 🗺️ Roadmap

seq_len=1 decode optimization is concluded at 703.4 us/layer (73.6% bandwidth efficiency — practical floor for f16 on Apple Silicon). Next phases focus on broader workloads.

| Phase | Focus | Scope | Status |
|:-----:|-------|:-----:|:------:|
| 12 | **GEMM optimization (seq_len=N)** | Framework | Planned |
| 13 | **Paged attention + speculative decode kernels** | Framework | Planned |
| 14 | **SDPA / attention optimization** | Framework | Planned |
| 15 | **Multi-node RDMA optimization (TP/EP)** | Framework | Planned |
| 16 | **Memory efficiency** | Framework | Planned |

> Framework = rmlx-core / rmlx-nn / rmlx-distributed kernel-level work.
> Serving-layer concerns (scheduling, eviction, orchestration) belong in rmlx-serve.
> See [full roadmap](docs/roadmap/phases.md) for details.

## 📚 Docs

- [Architecture Overview](docs/architecture/overview.md)
- [GPU Pipeline](docs/gpu-pipeline.md)
- [Implementation Roadmap](docs/roadmap/phases.md)
- [RMLX vs MLX vs CUDA](docs/comparison.md)
- [Getting Started](docs/getting-started/prerequisites.md)

## 📄 License

MIT — see [LICENSE](LICENSE).
