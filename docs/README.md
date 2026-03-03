# RMLX — Rust ML Framework for Apple Silicon

> **A Rust-based Metal GPU ML framework optimized for Apple Silicon**
>
> Status: All Phases complete (339 tests, 0 failures) | License: MIT OR Apache-2.0 | Rust 1.80+ | macOS (Apple Silicon)

---

## What is RMLX?

RMLX is a project that **reimplements the core Metal GPU inference pipeline of Apple's MLX framework in Rust**. The goal is to reach the theoretical performance ceiling for Expert Parallelism (EP) based distributed ML inference on a Mac Studio M3 Ultra cluster.

It fundamentally addresses the structural bottlenecks identified in MLX's C++/Python architecture by leveraging Rust's language-level strengths.

---

## Why is RMLX Needed?

MLX is an excellent framework, but it carries the following software overheads in distributed inference scenarios:

| Bottleneck | MLX Behavior | RMLX Solution |
|------------|-------------|---------------|
| **Per-op overhead** | 84 of 94 us/op is software overhead | Eager execution + Rust zero-cost abstractions |
| **Sync inefficiency** | `waitUntilCompleted` blocking (424.9 us) | `MTLSharedEvent` spin-wait (263.9 us, 1.61x improvement) |
| **Memory copies** | `std::copy`-based RDMA transfers | Zero-copy: dual Metal + RDMA registration on the same physical address |
| **No pipeline** | Compute -> Transfer runs sequentially | Dual `MTLCommandQueue` for GPU-level overlap |
| **Lazy evaluation** | Graph build overhead even for single-token decode | Eager-first + selective tracing compilation |

---

## Core Objective

```
2-node EP decode: 64ms/step -> 33ms/step (~30 tok/s)
Achieve near-parity with single-node 32ms/step
```

The ultimate goal is to connect two Mac Studio M3 Ultras via Thunderbolt 5 RDMA and achieve inference performance nearly identical to a single node.

---

## Key Differentiators vs. MLX

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

---

## Tech Stack

| Area | Technology |
|------|-----------|
| Language | Rust 1.80+ (edition 2021) |
| GPU | metal-rs 0.31 (Apple Metal API) |
| RDMA | ibverbs FFI (Thunderbolt 5 UC QP) |
| Hardware | Apple Silicon UMA (M3 Ultra, 80-core GPU, 512GB) |
| Build | Cargo workspace (6 crates) |

---

## Next Steps

- [Architecture Overview](architecture/overview.md) — Full system layer diagram and design philosophy
- [Crate Structure](architecture/crate-structure.md) — Workspace layout and role of each crate
- [Design Decisions](architecture/design-decisions.md) — Rationale behind key technical decisions

---

## Project Structure

This repository (`~/rmlx/`) is **framework-only**. The serving layer (`rmlx-serve`) is managed in a [separate repository](https://github.com/rmlx-serve).

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
