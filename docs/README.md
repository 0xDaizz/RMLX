# RMLX — Rust ML Framework for Apple Silicon

> **A Rust-based Metal GPU ML framework optimized for Apple Silicon**
>
> Status: All Phases complete (0-9B-opt + S1-S5 + Audit Remediation) (543 tests, 0 failures) | License: MIT | Rust 1.80+ | macOS (Apple Silicon)

---

## 🔍 What is RMLX?

RMLX is a project that **reimplements the core Metal GPU inference pipeline of Apple's MLX framework in Rust**. The goal is to reach the theoretical performance ceiling for Expert Parallelism (EP) based distributed ML inference on a Mac Studio M3 Ultra cluster.

It fundamentally addresses the structural bottlenecks identified in MLX's C++/Python architecture by leveraging Rust's language-level strengths.

---

## 💡 Why is RMLX Needed?

MLX is an excellent framework, but it carries the following software overheads in distributed inference scenarios:

| Bottleneck | MLX Behavior | RMLX Solution |
|------------|-------------|---------------|
| **Per-op overhead** | 84 of 94 us/op is software overhead | Eager execution + Rust zero-cost abstractions |
| **Sync inefficiency** | `waitUntilCompleted` blocking (424.9 us) | `MTLSharedEvent` spin-wait (263.9 us, 1.61x improvement) |
| **Memory copies** | `std::copy`-based RDMA transfers | Zero-copy: dual Metal + RDMA registration on the same physical address |
| **No pipeline** | Compute -> Transfer runs sequentially | Dual `MTLCommandQueue` for GPU-level overlap |
| **CB overhead** | 65 CBs/layer, per-op CPU sync | ExecGraph: 5 CBs/layer (92.3% reduction, 16.15x speedup) |
| **Lazy evaluation** | Graph build overhead even for single-token decode | Eager-first + selective tracing compilation |

---

## 🎯 Core Objective

```
2-node EP decode: 64ms/step -> 33ms/step (~30 tok/s)
Achieve near-parity with single-node 32ms/step

ExecGraph result: 110.4ms/layer -> 6.8ms/layer (16.15x speedup)
92.3% CB reduction, 98.5% CPU-GPU sync reduction
```

The ultimate goal is to connect two Mac Studio M3 Ultras via Thunderbolt 5 RDMA and achieve inference performance nearly identical to a single node.

---

## ⚡ Key Differentiators vs. MLX

1. **Zero-copy RDMA data path**
   `posix_memalign` -> `newBufferWithBytesNoCopy` -> `ibv_reg_mr` — three views share the same physical address, completely eliminating `memcpy` from the RDMA hot path.

2. **MTLSharedEvent-based synchronization**
   Achieves 1.61x synchronization performance improvement using event signal/wait instead of `waitUntilCompleted`.

3. **Dual queue pipeline**
   Separates compute and transfer queues to overlap computation and communication at the GPU level.

4. **Eager-first execution model**
   Eliminates lazy evaluation graph build overhead during single-token decode. Selective tracing is applied for batch operations such as prefill.

5. **Unified buffer pool**
   Pre-allocates dual Metal + RDMA registered buffers, eliminating runtime registration overhead.

6. **ExecGraph CB batching**
   Reduces 65 command buffers per transformer layer to 5 through deterministic operation
   grouping, achieving 16.15x speedup (110.4ms → 6.8ms) with 92.3% CB reduction.

7. **Expert Parallelism (EP)**
   MLX has no built-in EP support. RMLX provides a complete EP stack: 3-zone auto backend policy (CPU/Metal/RDMA) that selects the optimal path by data size, 7 dedicated MoE Metal kernels, SparseGuard overflow monitoring with capacity auto-tuning, and compute-RDMA pipeline overlap for distributed MoE inference on models like Mixtral and DeepSeek-V3.

---

## 🛠️ Tech Stack

| Area | Technology |
|------|-----------|
| Language | Rust 1.80+ (edition 2021) |
| GPU | metal-rs 0.31 (Apple Metal API) |
| RDMA | ibverbs FFI (Thunderbolt 5 UC QP) |
| Hardware | Apple Silicon UMA (M3 Ultra, 80-core GPU, 512GB) |
| Build | Cargo workspace (6 crates) |

---

## 📚 Next Steps

- [Architecture Overview](architecture/overview.md) — Full system layer diagram and design philosophy
- [Crate Structure](architecture/crate-structure.md) — Workspace layout and role of each crate
- [Design Decisions](architecture/design-decisions.md) — Rationale behind key technical decisions
- [GPU Pipeline](gpu-pipeline.md) — ExecGraph architecture and benchmark results
- [RMLX vs MLX vs CUDA](comparison.md) — Honest architecture comparison

---

## 🔧 Distributed RDMA Runbook (2-node minimal)

Use the built-in RMLX helpers (modeled after `mlx.distributed_config` and `mlx.launch`):

```bash
# 1) Generate hostfile + baseline setup
python3 scripts/rmlx_distributed_config.py \
  --hosts node1,node2 \
  --backend rdma \
  --over thunderbolt \
  --control-iface en0 \
  --auto-setup \
  --output rmlx-hosts.json \
  --verbose

# 2) Validate RDMA visibility on each host
python3 scripts/rmlx_launch.py \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- ibv_devices
```

For full prerequisites and caveats (SSH + passwordless sudo), see
[Getting Started: Prerequisites](getting-started/prerequisites.md).

---

## 📁 Project Structure

```
rmlx/
├── crates/
│   ├── rmlx-metal/          # Metal GPU abstractions
│   ├── rmlx-alloc/          # Zero-copy memory allocator
│   ├── rmlx-rdma/           # RDMA communication (ibverbs)
│   ├── rmlx-core/           # Compute engine (Op registry, VJP autodiff, LoRA)
│   ├── rmlx-distributed/    # Distributed primitives (EP, AllReduce, MoE)
│   └── rmlx-nn/             # Neural network layers (Transformer, MoE)
├── shaders/                 # Metal shader sources
├── tests/                   # Integration tests
├── benches/                 # Benchmarks
└── examples/                # Usage examples
```
