# Design Decisions

This document summarizes RMLX's key technical decisions and their rationale. Each decision is based on Proof of Concept (PoC) validation results or analysis of the MLX architecture.

---

## 1. Why Rust (Instead of C++)

MLX is written in C++, but RMLX chose Rust.

### Rationale

| Aspect | MLX (C++) | RMLX (Rust) |
|--------|-----------|-------------|
| **Memory safety** | Manual management; MTL::Buffer lifetime tracking is manual | Ownership system guarantees safety at compile time |
| **Concurrency** | Manual `std::mutex` locking; data races possible | `Send`/`Sync` traits provide compile-time thread safety |
| **Build system** | CMake + Python setuptools + nanobind | Single Cargo workspace build chain |
| **Error handling** | C++ exceptions; Metal errors checked at runtime | Explicit error propagation via `Result<T, E>` types |
| **Abstraction cost** | Virtual dispatch overhead | Choice between trait objects or monomorphization (zero-cost) |
| **Python bindings** | nanobind (C++ -> Python) | Not required (Rust-native framework) |

### Key Judgment

A GPU inference engine inevitably requires `unsafe` blocks (Metal FFI, RDMA FFI), but Rust's ownership system can **explicitly isolate** `unsafe` boundaries. Safe code (the `ZeroCopyBuffer` public API) and unsafe code (`posix_memalign`, `ibv_post_send` call sites) are distinguished at the type system level, maintaining the same performance as C++ while being safer.

Additionally, the Cargo workspace manages dependencies, versions, and build settings for 7 crates through a single `Cargo.toml`. This is significantly less complex than a CMake + setuptools combination.

---

## 2. Why metal-rs 0.31

metal-rs 0.31 is used as the Metal API binding.

### Rationale

- **Fully validated in PoC Phase 4**: The entire pipeline — `posix_memalign` -> `newBufferWithBytesNoCopy` -> `ibv_reg_mr` -> RDMA transfer -> Metal compute — was successfully reproduced using metal-rs 0.31.
- **Direct MTLDevice/MTLBuffer access**: Metal objects can be accessed directly from Rust without an Objective-C++ bridge.
- **Type-safe API**: `Device`, `CommandQueue`, `Buffer`, `ComputePipelineState`, and other types are wrapped in Rust types, preventing misuse at compile time.
- **Metal 3 feature support**: Supports APIs required for M3 and later, including `MTLSharedEvent` and `ResidencySet`.

### Alternatives Considered

| Alternative | Reason for Rejection |
|-------------|---------------------|
| metal-cpp (C++) | Requires Objective-C++ bridge; complex build integration with Rust projects |
| Direct objc2 usage | All Metal APIs must be manually bound, slowing development |
| wgpu | Cannot access Metal-specific optimizations (SharedEvent, NoCopy buffer) |

---

## 3. Eager-First Execution Model

Eager-first execution is adopted instead of MLX's lazy evaluation.

### Rationale

MLX uses lazy evaluation by default. It does not execute operations immediately but builds a graph and executes everything at once when `eval()` is called. While this approach provides optimization opportunities for batch operations (prefill), it becomes problematic for **single-token decode**:

```
Decode phase: processes only 1 token per step
-> Graph build overhead > actual computation time
-> 84 of 94 us/op is software overhead
```

### RMLX's Approach

- **Decode (autoregressive)**: Eager execution — operations are immediately encoded into Metal CommandBuffers and submitted. No graph build overhead.
- **Prefill (batch)**: Selective tracing — repeated operation patterns are recorded via tracing, compiled into optimized graphs, and executed.

This hybrid approach minimizes per-step latency during decode while preserving optimization benefits such as operation fusion during prefill.

---

## 4. Zero-Copy RDMA Data Path

`memcpy`/`std::copy` is completely eliminated from the RDMA communication path.

### Rationale

In MLX, data is copied to a separate buffer using `std::copy` before RDMA transmission. This causes significant overhead for large tensor transfers.

### Implementation

```
Step 1: Allocate page-aligned memory with posix_memalign(page_size)
Step 2: Create a Metal buffer view with newBufferWithBytesNoCopy (no copy)
Step 3: Register RDMA MR with ibv_reg_mr (same physical address)
```

The three views (raw pointer, Metal buffer, RDMA MR) created this way **share the same physical address**.

```
Metal GPU compute -> [same physical buffer] -> ibv_post_send -> wire
-> ibv_recv -> [same physical buffer] -> Metal GPU compute
```

- PoC Phase 2 confirmed address identity: `buffer.contents() == raw_ptr == mr.addr`.
- PoC Phase 4 validated the full pipeline (allocate -> Metal compute -> RDMA send -> receive -> Metal compute).

### Scope Limitations

"Zero-copy" applies only to the **repeatedly executed RDMA communication hot path**. Copies exist in the following segments, and this is intentional:

| Segment | Method | Reason |
|---------|--------|--------|
| Model weight loading | `mmap` -> `copy_nonoverlapping` -> Metal buffer | safetensors mmap is read-only; GPU weights require persistent residency |
| KV cache initialization | `calloc` -> Metal buffer or heap allocation | One-time initialization; subsequently updated only within the GPU |

---

## 5. MTLSharedEvent-Based Synchronization

`MTLSharedEvent` is used instead of `waitUntilCompleted`.

### Rationale

In MLX, `waitUntilCompleted` is called to verify GPU task completion. This API **blocks** the CPU thread until the command buffer completes.

### Performance Comparison

| Method | Latency | Notes |
|--------|---------|-------|
| `waitUntilCompleted` | 424.9 us | CPU blocking, context switch overhead |
| `MTLSharedEvent` spin-wait | 263.9 us | Non-blocking, signal/wait pattern |
| **Improvement** | **1.61x** | |

### How It Works

```
1. Encode encodeSignalEvent(event, value: N) into the GPU CommandBuffer.
2. CPU-side polls event.signaledValue via spin-wait.
3. When value >= N, GPU task completion is confirmed.
```

`MTLSharedEvent` is also used for cross-queue synchronization. In the dual queue pipeline, data dependencies between the compute queue and transfer queue are managed via event signal/wait.

---

## 6. UC QP (Instead of RC)

UC (Unreliable Connection) is used as the RDMA Queue Pair type.

### Rationale

RDMA communication typically uses RC (Reliable Connection) QPs. However, **Thunderbolt 5 RDMA does not support RC.** Due to hardware constraints, only UC QPs are available.

### UC QP Characteristics

| Characteristic | Description |
|---------------|-------------|
| Connection-oriented | 1:1 QP connections (not connectionless) |
| Unreliable | No hardware-level retransmission/acknowledgment |
| Send/Recv support | Data transfer via ibv_post_send/recv |
| RDMA Write | Supported (requires remote key exchange) |
| RDMA Read | **Not supported** |

### Reliability Guarantees

Since UC lacks hardware-level reliability, reliability is ensured at the application level:

- **2-phase exchange**: Sends a count message followed by the payload, so the receiver knows the expected data volume in advance.
- **CQ polling**: Verifies transmission completion by checking `IBV_WC_SUCCESS` in the Completion Queue.
- **Warmup**: Exchanges dummy data after connection establishment to warm up the path and prevent initial packet loss.

In practice, Thunderbolt 5 is a local connection (1-2m cable), so packet loss probability is very low. Complex network-level retransmission protocols are unnecessary.

---

## 7. Dual Queue Pipeline

Compute and transfer queues are separated for GPU-level overlap.

### Rationale

With a single `MTLCommandQueue`, compute and data transfer execute sequentially:

```
Single queue: [compute layer N] -> [transfer layer N] -> [compute layer N+1] -> ...
Dual queue:   [compute layer N  ] -> [compute layer N+1  ] -> ...
              [transfer layer N-1] -> [transfer layer N    ] -> ...
```

With dual queues, RDMA transfer results from layer N-1 can be processed simultaneously while layer N's compute is in progress.

### Implementation Details

```rust
pub struct StreamManager {
    compute_queue: CommandQueue,    // primary computation queue
    transfer_queue: CommandQueue,   // RDMA sync/data preparation queue
}
```

- **compute_queue**: Executes primary operations such as matmul, softmax, and attention.
- **transfer_queue**: Handles RDMA send/recv completion waiting and received data preparation (reformatting, etc.).

Cross-queue synchronization is managed via `MTLSharedEvent`. The `HazardTrackingModeUntracked` option disables Metal's automatic hazard tracking, and all cross-queue accesses are explicitly synchronized. This pattern was validated in PoC Phase 3.6.

### Cautions

When using `HazardTrackingModeUntracked`, all cross-queue buffer accesses must be synchronized with `MTLSharedEvent` or `MTLFence`. When adding new cross-queue access patterns, data integrity tests must always be included.

---

## 8. ExecGraph Architecture — Re-encode vs Capture-replay

ExecGraph uses deterministic re-encoding rather than CUDA Graphs-style capture-replay.

### Rationale

CUDA Graphs captures a sequence of GPU operations and replays them with near-zero CPU overhead.
Metal does not have an equivalent capture-replay mechanism. Instead, ExecGraph:

1. Pre-analyzes the transformer layer's operation sequence
2. Groups operations into 5 command buffers (down from 65)
3. Re-encodes the same deterministic sequence each forward pass
4. Uses MTLSharedEvent for inter-CB synchronization

### Why 5 Command Buffers?

| CB | Operations | Reason for boundary |
|----|-----------|---------------------|
| CB1 | RMS norm + Q/K/V projections (fused) | Shared input dependency |
| CB2 | Head split + RoPE + cache append | Depends on CB1 output |
| CB3 | SDPA + head concat + O_proj | Depends on CB2 |
| CB4 | Residual + pre-FFN norm | Depends on CB3 |
| CB5 | Gate + up + silu_mul + down + residual | Depends on CB4 |

Each boundary exists because subsequent operations depend on the previous CB's output.
Within each CB, multiple operations share a single encoder.

### Benchmark Evidence

| Metric | Baseline | ExecGraph | Improvement |
|--------|----------|-----------|-------------|
| CBs/layer | 65 | 5 | 92.3% reduction |
| Latency/layer | ~112ms | ~6.4ms | 17.4x speedup |
| CPU-GPU syncs | ~65 | ~1 | 98.5% reduction |

### Trade-offs vs CUDA Graphs

| Aspect | ExecGraph | CUDA Graphs |
|--------|-----------|-------------|
| Mechanism | Re-encode each pass | Capture once, replay |
| CPU overhead | Low (deterministic path) | Near-zero (replay) |
| Shape flexibility | Can handle shape changes | Requires re-capture |
| Implementation | Simpler (no capture state) | Complex capture semantics |

---

## 9. Weight Pre-caching — Memory vs Latency Tradeoff

Pre-computed contiguous transposed weight matrices are cached at model load time.

### Rationale

Linear layers compute `x @ W^T`. The transpose `W^T` creates a non-contiguous view,
requiring a contiguous copy before matmul dispatch. In the baseline, this copy happens
every forward pass.

### Solution

`prepare_weight_t()` pre-computes and caches the contiguous transposed weight at model
load time. This trades ~2x weight memory for zero-cost transpose during inference.

### Memory Impact

For a 7B parameter model with f16 weights:
- Base weights: ~14 GB
- Transposed cache: ~14 GB
- Total: ~28 GB (fits in 512GB UMA with headroom)

### Performance Impact

Eliminates the per-layer transpose + copy overhead from the critical path.
Combined with ExecGraph, this contributes to the 17.4x speedup.
