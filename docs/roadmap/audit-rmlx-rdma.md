# rmlx-rdma Production Readiness Audit

**Date:** 2026-03-06
**Auditors:** Claude Opus 4.6
**Verdict:** CONDITIONALLY PRODUCTION-READY — Well-structured TB5 RDMA layer with strong fundamentals, but needs correctness hardening and feature completion
**Reference:** MLX JACCL (MLX JACCL source (mlx/distributed/jaccl/)), NCCL, Gloo, macOS 26.2 TB5 RDMA

## Executive Summary

rmlx-rdma is a 16-module Thunderbolt 5 RDMA communication crate with ~5,500 LOC. It provides ibverbs FFI bindings (dynamic loading), UC queue pair management, pre-registered MR pools, GPU-visible descriptor rings, ring-based collectives (allreduce/allgather/reduce_scatter), multi-port striping, topology-aware connection management, and TCP-based QP info exchange. Architecture closely mirrors MLX's JACCL backend but adds several innovations (GPU doorbell ring, tiered MR pool, dual-port striping).

The crate has strong foundations but ~65% of code requires RDMA hardware for testing, leaving significant untested surface. Key gaps vs MLX/NCCL include: no pipelined circular buffers, no tree allreduce, no RC (Reliable Connection) mode, no fault tolerance, and limited GPU-compute/network overlap.

## Module Inventory

| Module | File | Purpose | Has Tests? |
|--------|------|---------|------------|
| `ffi` | ffi.rs | ibverbs FFI bindings via dlopen (23 symbols) | No |
| `context` | context.rs | RDMA device context + protection domain + device probing | No |
| `qp` | qp.rs | UC queue pair lifecycle (RESET->INIT->RTR->RTS) | No |
| `mr` | mr.rs | Memory region registration (unsafe, size-limited) | No |
| `exchange_tag` | exchange_tag.rs | 64-bit wr_id encoding (seq/tag/buf_slot/peer_id) | Yes (8 tests) |
| `mr_pool` | mr_pool.rs | Tiered double-buffered MR pool (6 tiers: 4KB->16MB) | No |
| `shared_buffer` | shared_buffer.rs | Triple-view buffer (CPU/Metal/RDMA) with multi-PD registration | No |
| `progress` | progress.rs | Lock-free CQ progress engine (manual + background modes) | Yes (6 tests) |
| `connection` | connection.rs | Full RDMA connection lifecycle + WR posting + completion tracking | No |
| `exchange` | exchange.rs | TCP-based QP info exchange (hub/spoke protocol) | Yes (5 tests) |
| `coordinator` | coordinator.rs | N-way hub-spoke coordination for QP info all-gather | Yes (8 tests) |
| `gpu_doorbell` | gpu_doorbell.rs | GPU-visible descriptor ring + CPU proxy thread | Yes (5 tests) |
| `connection_manager` | connection_manager.rs | Multi-peer connection manager with topology awareness | Yes (8 tests) |
| `multi_port` | multi_port.rs | TB5 dual-port striping + topology abstractions (ring/mesh/hybrid) | No |
| `rdma_metrics` | rdma_metrics.rs | Lock-free atomic performance counters | No |
| `collectives` | collectives.rs | Ring allreduce, allgather, reduce_scatter (f32) | Yes (16 tests) |

**Total: 56 tests, ~35% testable without hardware**

## P0 — Critical Issues

### CRIT-1: Collectives Only Support f32

| Field | Detail |
|-------|--------|
| **File** | `collectives.rs` (entire file) |
| **Description** | `ring_allreduce`, `ring_reduce_scatter` only operate on `Vec<f32>`. No support for f16, bf16, or integer types. `apply_reduce_op` is hardcoded to f32 element-wise operations. |
| **Impact** | Cannot use for ML inference which predominantly uses f16/bf16 tensors. Requires data type conversion before/after collective, adding latency and memory overhead. |
| **MLX Comparison** | MLX JACCL handles arbitrary dtypes by operating on raw bytes for allgather and implementing typed reductions for float16/bfloat16 via MPI custom types. |
| **Fix** | Generic `apply_reduce_op<T: ReduceElement>` with trait implementations for f32, f16, bf16. Use `bytemuck` or manual transmute for type-punning. Add `ring_allreduce_bytes` for dtype-agnostic byte-level operations. |

### CRIT-2: No Reliability Layer for UC Transport

| Field | Detail |
|-------|--------|
| **File** | `qp.rs` (UC mode throughout) |
| **Description** | Uses IBV_QPT_UC (Unreliable Connection) exclusively. UC provides no transport-level retransmission — dropped packets are silently lost. No application-level reliability (checksums, sequence numbers, retransmission). |
| **Impact** | Silent data corruption in production. While TB5 over direct cable is highly reliable, any packet loss (congestion, transient error) will produce incorrect ML results with no detection. |
| **MLX Comparison** | MLX JACCL also uses UC, but MLX operates in a controlled TB5 direct-cable environment where loss is essentially impossible. However, if RMLX targets broader deployment (switches, longer cables), this becomes critical. |
| **Fix Options** | (A) Add optional RC (Reliable Connection) mode for non-TB5 deployments. (B) Add application-level CRC32 checksums in wr_id or descriptor metadata. (C) Document UC-only constraint explicitly and add health-check heartbeat. Recommend (B) + (C) for production. |

## P1 — High Issues

### HIGH-1: No Pipelined Circular Buffers

| Field | Detail |
|-------|--------|
| **Files** | `collectives.rs`, `connection.rs` |
| **Description** | Ring allreduce posts one send + one recv per step, waits for completion, then proceeds to next step. No overlap between network I/O and reduction computation. NCCL uses 8-slot circular pipeline buffers for full overlap. |
| **Impact** | ~2x slower than achievable throughput for large tensors. Network is idle during reduction, computation is idle during transfer. |
| **Fix** | Implement `PipelinedRingBuffer` with N slots (default 4-8). Post sends/recvs for slot K+1 while reducing slot K. Use double-buffered MR pool (already `PIPELINE=2` in mr_pool.rs) as the backing. |

### HIGH-2: SharedBuffer Multi-PD Registration Scalability

| Field | Detail |
|-------|--------|
| **File** | `shared_buffer.rs` |
| **Description** | Each `SharedBuffer` maintains a `HashMap<ConnectionId, MemoryRegion>` — one MR per connection. For N peers with M tiers x PIPELINE buffers, this creates `N x M x PIPELINE` memory registrations. At 16 peers x 6 tiers x 2 pipeline = 192 MRs per node. |
| **Impact** | MR registration is expensive (~100us per ibv_reg_mr). Startup time grows linearly with cluster size. |
| **MLX Comparison** | MLX JACCL pre-registers buffers with all peers' PDs at startup. Same scaling issue, but MLX limits to 8 peers. |
| **Fix** | (A) Lazy registration — only register with PDs for active connections. (B) Use a single PD if all connections share the same RDMA device (common in TB5). (C) Implement MR caching with refcounting. |

### HIGH-3: ProgressEngine Background Thread Health

| Field | Detail |
|-------|--------|
| **File** | `progress.rs` |
| **Description** | Background thread polls CQ in a tight loop. If CQ returns an error (WC status != SUCCESS), the error is recorded but the thread continues polling. No escalation strategy for persistent errors. Also, `is_healthy()` flag is set to false only on mutex poisoning, not on CQ errors. |
| **Fix** | (A) Track consecutive error count; shut down after threshold. (B) Expose error callback hook. (C) Set `healthy=false` on persistent CQ errors (e.g., 10 consecutive WC errors). |

### HIGH-4: Collectives Don't Use ProgressEngine

| Field | Detail |
|-------|--------|
| **Files** | `collectives.rs`, `progress.rs` |
| **Description** | Ring collectives use `connection.wait_completions()` (direct CQ poll) instead of the `ProgressEngine`. The progress engine exists but is not integrated into the collective path. |
| **Impact** | No async completion handling for collectives. No benefit from background polling thread. |
| **Fix** | Refactor collectives to post operations via `ProgressEngine::register_op()` and use `PendingOp::wait()` for completion. This enables future non-blocking collective APIs. |

## P2 — Medium Issues

### MED-1: QueuePair::raw() Dead Code
- **File:** `qp.rs`
- **Description:** `raw()` method marked `#[allow(dead_code)]`
- **Verdict:** KEEP — useful for advanced users who need direct ibv_post_send with custom WRs

### MED-2: rdma Feature Flag Not Used
- **File:** `Cargo.toml`
- **Description:** Feature `rdma = []` is defined but no code is feature-gated by it
- **Fix:** Either gate all RDMA code behind `#[cfg(feature = "rdma")]` or remove the unused feature

### MED-3: PSN Computation is Deterministic
- **File:** `qp.rs`
- **Description:** PSN = `rank * 1000 + 42` — predictable, not randomized. In hostile network environments this could enable replay attacks.
- **Fix:** For TB5 direct cable this is fine. For broader deployment, use `rand::thread_rng().gen::<u32>() & 0xFFFFFF`

### MED-4: Device Probe Fallback to Hardcoded TB5 Defaults
- **File:** `context.rs`
- **Description:** If ibv_query_device/port fail, falls back to hardcoded values (MAX_SEND_WR=8192, CQ_DEPTH=8192, MTU_1024). These may not match actual hardware.
- **Fix:** Log a warning (already done). Consider failing instead of using potentially wrong defaults.

### MED-5: Collectives No-Connection Path Returns Input Unchanged
- **File:** `collectives.rs`
- **Description:** If no ring connections exist, `ring_allreduce` returns the input data unchanged. This silently skips the collective.
- **Fix:** Return error or log warning. Silent success when collective should aggregate across nodes is dangerous.

### MED-6: eprintln! for Logging
- **Files:** `context.rs`, `connection.rs`, `exchange.rs`
- **Description:** Uses `eprintln!` directly instead of a logging framework (tracing/log crate).
- **Fix:** Replace with `tracing::warn!` / `tracing::info!` for production observability.

## Dead Code / Orphan Analysis

| Item | File | Verdict | Rationale |
|------|------|---------|-----------|
| `QueuePair::raw()` | qp.rs | **KEEP** | Escape hatch for advanced ibverbs usage |
| `rdma` feature flag | Cargo.toml | **FIX** — gate code or remove | Currently decorative |
| `PortFailover` struct | multi_port.rs | **KEEP** — wire into connection_manager | Has mark_failed/recovering/active but no callers yet. Should integrate with ConnectionManager for failover. |
| `RdmaMetrics` | rdma_metrics.rs | **KEEP** — wire into connection.rs | Struct exists but record_send/recv not called from connection post paths. Should integrate. |
| `MrPool` in connection.rs | connection.rs | **PARTIALLY WIRED** | `init_mr_pool()` and `acquire_mr()` exist but collectives don't use them (they register ad-hoc MRs). |

## Feature Gap Analysis

### vs MLX JACCL

| # | Feature | MLX JACCL | RMLX | Gap | Priority |
|---|---------|-----------|------|-----|----------|
| J1 | **Multi-dtype collectives** | Raw bytes + typed reductions for float16/bf16 | f32 only | Must generalize | P0 |
| J2 | **Mesh allreduce** | Star/broadcast for <=8 nodes (all-to-all) | Ring only | Add mesh mode for small clusters | P2 |
| J3 | **Pipeline depth 2** | 2 send/recv in flight simultaneously | 1 in flight per step | Already has PIPELINE=2 constant, wire it in | P1 |
| J4 | **8 buffer size classes** | 4KB x (1<<k) for k in [0..7] = 4KB-512KB | 6 tiers: 4KB-16MB | RMLX has larger tiers (16MB vs 512KB max) — this is better for ML | RMLX ahead |
| J5 | **Multi-wire parallelism** | Up to 4 RDMA connections per direction in ring | Dual-port striping (2 max) | Add multi-wire (4+ connections) option | P3 |
| J6 | **Bidirectional ring** | Left + right connections simultaneously | Single-direction ring | Add bidirectional for 2x bandwidth | P2 |
| J7 | **Blocking send/recv/all_to_all** | Full point-to-point + all-to-all support | Collectives only, no standalone send/recv | post_send/recv exist in connection.rs but no high-level API | P2 |
| J8 | **MoE dispatch/combine** | Sophisticated CPU+GPU hybrid with Metal kernels | In rmlx-distributed (separate crate) | Cross-crate: verify integration | N/A |
| J9 | **Warmup** | RDMA warmup rounds + kernel pre-compile | connection.warmup() exists (10 rounds of 4B send/recv) | RMLX has this | — |

### vs NCCL / Gloo / Frontier

| # | Feature | Best Reference | RMLX | Gap | Priority |
|---|---------|---------------|------|-----|----------|
| N1 | **Pipelined circular buffers** | NCCL: 8-slot pipeline for overlapped network I/O + reduction | No pipeline. Sequential send->wait->reduce per step | Must implement for throughput | P1 |
| N2 | **Tree allreduce** | NCCL: tree for latency-sensitive small messages | Ring only | Add tree mode for small messages | P2 |
| N3 | **Transport abstraction trait** | Gloo: clean transport interface (TCP/RDMA/SHM) | RDMA-only. TCP used only for QP exchange | Add `trait Transport { fn send(); fn recv(); }` with TCP fallback | P2 |
| N4 | **Fault tolerance** | NCCL 2.27: communicator abort + re-init | None. Connection failure = unrecoverable | Add reconnection logic + communicator rebuild | P2 |
| N5 | **Zero-copy user buffer registration** | NCCL 2.19: register user buffers for direct send/recv | SharedBuffer provides this. MrPool provides pre-registered buffers. | RMLX has this | — |
| N6 | **Non-blocking collectives** | NCCL: all ops return immediately, sync via stream | All collectives are blocking | Add async versions returning PendingOp. ProgressEngine exists but unused. | P2 |
| N7 | **Gradient compression** | DeepSpeed: 1-bit Adam, LAMB | None | P3 — optimization | P3 |
| N8 | **GPU-native allreduce** | NCCL: entire allreduce runs on GPU | CPU-mediated via GPU doorbell proxy | Not feasible on Metal (no GPU-initiated RDMA) | SKIP |

### RMLX Innovations (Ahead of MLX)

| Feature | RMLX Approach | MLX Comparison |
|---------|--------------|----------------|
| **GPU Doorbell Ring** | GPU-visible descriptor ring with CPU proxy thread. GPU signals via GpuEvent, CPU translates to ibv_post_send/recv. | MLX has no GPU-initiated RDMA path |
| **Tiered MR Pool** | 6 tiers (4KB-16MB) with atomic slot acquisition, double-buffered | MLX has 8 smaller tiers (4KB-512KB) |
| **Dual-Port Striping** | StripeEngine with round-robin chunk assignment across TB5 ports | MLX has multi-wire but not port-level striping |
| **PortFailover** | State machine (Active/Failed/Recovering) per port | MLX has no failover |
| **Topology Abstraction** | Ring/Mesh/Hybrid enum with `peers()` and `connections_per_node()` | MLX hardcodes MeshGroup/RingGroup |
| **Exchange Tag Protocol** | 64-bit wr_id with seq/tag/buf_slot/peer_id encoding | MLX uses simpler wr_id |
| **Connection Manager** | HashMap-based multi-peer with topology-aware expected peers | MLX uses Vec<Connection> |

## Test Coverage Assessment

### Tested (56 tests, ~35% of codebase)
- exchange_tag: roundtrip, truncation, decoding (8)
- exchange: serialization, reconnect, retry exhaustion (5)
- coordinator: single/multi-rank gather, barrier (8)
- gpu_doorbell: descriptor size/alignment, ops (5)
- connection_manager: topology, neighbors, expected peers (8)
- progress: lifecycle, error, timeout, concurrent, poison (6)
- collectives: chunk boundaries, reduce ops, single-rank paths (16)

### Untested (~65% — requires RDMA hardware)
- FFI symbol loading and ibverbs calls
- Device context opening and probing
- QP state machine transitions
- Memory region registration/deregistration
- MR pool allocation/release lifecycle
- SharedBuffer triple-view (CPU/Metal/RDMA)
- Full connection establishment flow
- Multi-node ring allreduce with actual data
- Dual-port striping under load
- PortFailover state transitions
- RdmaMetrics recording accuracy

## Cross-Crate Dependencies

| Dependency | Direction | What's Needed |
|-----------|-----------|---------------|
| `rmlx-alloc` -> `rmlx-rdma` | rmlx-rdma uses rmlx-alloc | `CompletionTicket` for SharedBuffer lifecycle. `ZeroCopyBuffer` not directly used (SharedBuffer is RDMA's own zero-copy). |
| `rmlx-metal` -> `rmlx-rdma` | rmlx-rdma uses rmlx-metal | `GpuEvent` for GPU doorbell signaling. Metal `Buffer` for SharedBuffer's GPU view. |
| `rmlx-distributed` -> `rmlx-rdma` | rmlx-distributed uses rmlx-rdma | EP dispatch/combine uses RDMA connections. Must verify: does EP use MrPool or ad-hoc registration? Does EP use ProgressEngine? |
| `rmlx-core` -> `rmlx-rdma` | Indirect via rmlx-distributed | Tensor dtype information needed for multi-dtype collectives (CRIT-1). |

## Recommended Fix Roadmap

### Phase 1: Correctness (Block release)
1. **CRIT-1** — Generalize collectives to support f16/bf16 (trait-based)
2. **CRIT-2** — Add application-level CRC32 checksums for UC transport; document UC-only constraint
3. **HIGH-3** — ProgressEngine health escalation (consecutive error threshold)
4. **MED-5** — Collectives no-connection path should error, not silently succeed
5. Wire `RdmaMetrics` into connection post_send/post_recv paths
6. Wire `PortFailover` into ConnectionManager

### Phase 2: Performance
7. **HIGH-1** — Pipelined circular buffers (4-8 slots) for ring allreduce
8. **J3** — Wire PIPELINE=2 into collectives (2 in-flight per step)
9. **J6** — Bidirectional ring allreduce (left + right simultaneously)
10. **HIGH-4** — Integrate collectives with ProgressEngine for async path

### Phase 3: Feature Completion
11. **N3** — Transport abstraction trait with TCP fallback
12. **J2** — Mesh allreduce for small clusters
13. **N2** — Tree allreduce for latency-sensitive small messages
14. **N4** — Fault tolerance: reconnection + communicator rebuild
15. **N6** — Non-blocking collective APIs returning PendingOp
16. **MED-6** — Replace eprintln! with tracing crate

---

*Generated by Claude Opus 4.6 — 2026-03-06*
