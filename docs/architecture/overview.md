# Architecture Overview

RMLX is a layered architecture composed of five layers. All Phases (0 through 7B) have been completed and the system is fully implemented. Each layer has clear responsibility boundaries and is separated into individual Cargo crates. Phase 7 additions include VJP autodiff, LoRA fine-tuning, and production hardening (structured logging, metrics, precision guard, graceful shutdown).

---

## Full Layer Diagram

```
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  ~/rmlx-serve/ (separate repository — references rmlx as a Cargo dependency)
│ ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
  │  Model Zoo   │  │  Scheduler   │  │  KV Cache    │  │ PP/TP/EP         │
│ │ (safetensors)│  │ (continuous  │  │  Manager     │  │ Orchestrator     │ │
  │              │  │  batching)   │  │  (paged)     │  │                  │
│ └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
 ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ┼─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ┘
         │  Cargo dep:      │  rmlx = {       │  path = "../rmlx" │
         │  rmlx-nn         │  rmlx-distributed                   │
┌────────┼─────────────────┼─────────────────┼───────────────────┼───────────┐
│        ▼                 ▼                 ▼                   ▼           │
│  ~/rmlx/ (this repository — framework only)                               │
│                        rmlx-core (compute engine)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Compute Graph / Op Registry                     │   │
│  │  matmul · softmax · rms_norm · rope · quantized_matmul · moe_gate   │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
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

### 1. Application Layer — rmlx-serve (Separate Repository)

Handles application logic required for model serving. References the RMLX framework as a Cargo dependency.

| Component | Role |
|-----------|------|
| **Model Zoo** | Loading model weights in safetensors format and quantization decoding |
| **Scheduler** | Request scheduling based on continuous batching |
| **KV Cache Manager** | Paged KV cache management (dynamic allocation/deallocation) |
| **PP/TP/EP Orchestrator** | Pipeline/Tensor/Expert Parallelism orchestration |

**Rationale for separation**: The framework (`rmlx`) and the application (`rmlx-serve`) have different release cycles, dependencies, and testing strategies. `rmlx` can be reused by other applications (training, benchmark tools, etc.).

---

### 2. Compute Engine — rmlx-core

The core engine responsible for computation graphs and kernel dispatch.

| Component | Role |
|-----------|------|
| **Op Registry** | Registers operations such as matmul, softmax, rms_norm, rope, quantized_matmul, moe_gate |
| **Compute Graph** | Selective tracing-based computation graph (eager-first, tracing during prefill) |
| **Kernel Dispatch** | Maps ops to Metal kernels and executes them, selecting optimal kernels based on dtype/shape |

rmlx-core depends on rmlx-metal and rmlx-alloc, and provides computation APIs to upper layers (rmlx-nn, rmlx-distributed).

---

### 3. Metal Pipeline Layer

Manages the Metal GPU execution pipeline. The rmlx-metal crate implements this layer.

| Component | Role |
|-----------|------|
| **Kernel Manager** | `.metallib` AOT loading and source string JIT compilation, ComputePipelineState caching |
| **CommandEncoder** | CommandBuffer/Encoder lifetime management, barrier/fence insertion, concurrent dispatch context |
| **Pipeline Scheduler** | Dual `MTLCommandQueue` management — separates compute and transfer queues for GPU-level overlap |

The dual queue architecture was validated in PoC Phase 3.6. It maximizes pipeline efficiency by executing compute operations and RDMA data transfer preparation simultaneously.

---

### 4. Sync & Memory Layer

Handles synchronization and memory management. rmlx-metal (events) and rmlx-alloc (allocator, buffer pool) implement this layer.

| Component | Role |
|-----------|------|
| **SharedEvent Manager** | Non-blocking synchronization via `MTLSharedEvent` signal/wait (263.9 us, 1.61x improvement over `waitUntilCompleted`) |
| **Zero-Copy Allocator** | `posix_memalign` -> `newBufferWithBytesNoCopy` — creates a Metal view on the same physical memory for copy-free GPU access |
| **Buffer Pool** | Pre-allocates dual Metal + `ibv_mr` registered buffers, eliminating runtime registration overhead with size-binned caching |

The zero-copy path is limited to the RDMA communication hot path. Copies during one-time initialization such as model weight loading are not optimization targets.

---

### 5. Communication Layer — rmlx-rdma

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
| `rmlx-metal` | (external: metal-rs 0.31, objc2, block2) |
| `rmlx-alloc` | `rmlx-metal`, libc |
| `rmlx-rdma` | `rmlx-alloc`, libc (ibverbs FFI) |
| `rmlx-core` | `rmlx-metal`, `rmlx-alloc` |
| `rmlx-distributed` | `rmlx-core`, `rmlx-rdma` |
| `rmlx-nn` | `rmlx-core` |

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
