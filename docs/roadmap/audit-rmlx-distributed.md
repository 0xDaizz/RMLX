# rmlx-distributed Production Readiness Audit

Date: 2026-03-06
Scope: `crates/rmlx-distributed/` (15 modules, ~7000+ LOC)
Auditors: Claude Opus 4.6 (MLX comparison, CUDA frontier comparison, direct code audit)
References: MLX (`local MLX checkout`), NCCL, DeepEP, Megascale, Tutel, FasterMoE, vLLM

## Crate Overview

Distributed communication layer for RMLX, providing MoE (Mixture-of-Experts) token exchange over RDMA/Thunderbolt 5 with Metal GPU integration. Key subsystems:

- **init.rs** (~760 LOC) — RDMA bootstrap with coordinator-mediated all_gather, MPI/SLURM env compat
- **group.rs** (~1117 LOC) — Communication group with collectives (allreduce, allgather, broadcast, barrier, all_to_all, reduce_scatter, split), ring algorithms
- **transport.rs** — RdmaConnectionTransport with SharedBuffer, StripeEngine, dual-port TB5 striping
- **pipeline.rs** (~907 LOC) — LayerPipeline with GpuEvent chain, CompletionTicket, PipelineStage state machine
- **moe_exchange.rs** (~1000+ LOC) — MoeDispatchExchange/MoeCombineExchange with Metal shaders, WireProtocol V2/V3
- **fp8_exchange.rs** (~487 LOC) — FP8 E4M3 quantization, fused dequant+scatter Metal shader
- **v3_protocol.rs** (~958 LOC) — Variable-length wire protocol with PacketMeta, blocking_exchange_v3
- **slab_ring.rs** (~649 LOC) — Pre-registered Metal buffer ring with AtomicU64 producer/consumer
- **moe_policy.rs** (~375 LOC) — 3-zone CPU/Metal/RDMA policy with hysteresis and cooldown
- **ep_runtime.rs** (~199 LOC) — EpRuntimeContext hub with DescriptorProxy, CreditManager, GpuEvents
- **sparse_guard.rs** (~130 LOC) — Overflow EMA monitoring with IncreaseCapacity/DenseFallback actions
- **credit_manager.rs** (~189 LOC) — UC recv-credit window per (peer, tag)
- **metrics.rs** (~131 LOC) — MoeMetrics with atomic counters, per-expert histogram
- **perf_counters.rs** (~328 LOC) — Zero-copy KPI counters (cpu_copy_bytes, mr_reg_calls, gpu_sync_calls)
- **warmup.rs** (~189 LOC) — WarmupState with idempotent run_warmup, ThresholdCalibration

Dependencies: rmlx-alloc, rmlx-core, rmlx-metal, rmlx-rdma

## Positive Findings

- **Zero-copy RDMA pipeline (9/10):** SlabRing → SharedBuffer → RDMA path avoids CPU-side memcpy; StorageModeShared UMA buffers enable GPU→RDMA without staging. Superior to DeepEP's cudaMemcpy staging.
- **FP8 E4M3 quantization (9/10):** Fused Metal shader for dequant+scatter reduces wire bandwidth 2× with single-pass GPU compute. On par with DeepEP's FP8 and ahead of Tutel/FasterMoE (no FP8 support).
- **Async pipeline architecture (8/10):** GpuEvent-driven compute↔RDMA overlap with CompletionTicket tracking. Correct dual-completion fence (GPU + RDMA) in pipeline.rs.
- **v3 variable-length wire protocol:** Handles heterogeneous expert token counts without padding waste. PacketMeta + length-prefixed encoding is clean.
- **SparseGuard overflow monitoring:** EMA-based detection with automatic dense fallback is novel and production-relevant.
- **Dual-port TB5 striping:** StripeEngine in transport.rs leverages both Thunderbolt 5 ports for 2× bandwidth.
- **Credit-based flow control:** CreditManager prevents UC receiver overrun — critical for lossless RDMA.
- **Comprehensive module coverage:** 15 modules covering the full EP lifecycle (init → dispatch → combine → metrics).

## P0 — Critical

### P0-1: `read_buffer_bytes` / `read_buffer_f32` Unsound Unsafe + Panic

**Status: ✅ FIXED (Phase 0, PR #38)** — replaced assert! with Result

- **Location:** `moe_exchange.rs:46-74`
- **Description:** These helpers use `assert!(buf.len() >= offset + len)` followed by raw `unsafe { std::slice::from_raw_parts(...) }`. On bad input the assert panics (crashes the entire distributed job), and the unsafe block has no validation that the Metal buffer pointer is still valid or that the returned slice lifetime is sound. A GPU-side race (buffer reuse before read completes) causes UB.
- **Fix:** Return `Result<&[u8], Error>` instead of panicking. Add GpuEvent fence check before buffer access. Consider safe Metal buffer accessor API from rmlx-metal.
- **Severity rationale:** Panic kills all ranks; UB on buffer race is memory corruption.
- **Source:** Direct code audit

### P0-2: EP Dispatch/Combine Correctness Issues
- **Location:** `moe_exchange.rs` dispatch/combine paths, `ep_runtime.rs`
- **Description:** Previously identified P0 correctness issues documented in `docs/roadmap/audit-ep-dispatch-combine.md`. Token routing indices, combine weight accumulation, and async handle lifecycle have known bugs.
- **Fix:** See existing audit document for detailed remediation plan.
- **Source:** Prior audit (cross-reference)

### P0-3: `blocking_exchange_v3` Deadlock Potential
- **Location:** `v3_protocol.rs` blocking_exchange_v3
- **Description:** All ranks execute send-then-recv in a synchronous loop. If message sizes are asymmetric and RDMA buffers fill up, all ranks can block on send simultaneously with no rank progressing to recv — classic circular deadlock. NCCL avoids this with chunked ring pipeline; DeepEP uses async RDMA with CQ polling.
- **Fix:** Implement non-blocking exchange with progress engine polling, or use chunked send/recv interleaving.
- **Source:** Direct code audit + Frontier CUDA comparison

## P1 — High

### P1-1: Ring Allreduce Chunk Rounding Error
- **Location:** `group.rs` ring_allreduce implementation
- **Description:** Chunk size calculation uses integer division `total_len / world_size` without handling remainder. When `total_len % world_size != 0`, the last chunk is silently truncated, producing incorrect reduction results for non-divisible tensor sizes.
- **Fix:** Pad to next multiple of world_size, or use variable-size last chunk.
- **Source:** Direct code audit

### P1-2: `MoePolicy` Thread Safety — Mixed Atomic/Mutable State
- **Location:** `moe_policy.rs`
- **Description:** MoePolicy uses AtomicU64 for counters alongside non-atomic mutable fields (zone thresholds, cooldown state). Concurrent callers can observe torn reads on non-atomic fields while atomics appear consistent, leading to incorrect zone transitions.
- **Fix:** Either make all shared state atomic, or wrap the entire policy in a Mutex. Consider RwLock for read-heavy access pattern.
- **Source:** Direct code audit

### P1-3: Warmup Calibration Uses Fake Benchmarks
- **Location:** `warmup.rs:146-169`
- **Description:** `ThresholdCalibration` benchmarks CPU `sin()` loops to set Metal/RDMA thresholds. This has zero correlation with actual GPU compute or network latency. Thresholds derived from CPU sin() will be wildly wrong for Metal dispatch decisions.
- **Fix:** Run actual Metal shader micro-benchmarks (e.g., small matmul) and RDMA ping-pong to calibrate thresholds.
- **Source:** Direct code audit

### P1-4: No Fault Tolerance / Node Failure Recovery
- **Location:** Entire crate
- **Description:** No mechanism for detecting node failures, re-routing around dead ranks, or checkpointing distributed state. A single node crash kills the entire job. NCCL has health checks + timeout detection; Megascale has hot-spare failover; DeepEP has per-peer timeout with retry.
- **Fix:** Add heartbeat/watchdog per rank, timeout-based failure detection, and graceful degradation (at minimum, clean error reporting instead of infinite hang).
- **Source:** Frontier CUDA comparison

### P1-5: No Tree/Hierarchical Collective Algorithms
- **Location:** `group.rs`
- **Description:** All collectives use flat ring topology. For >4 nodes, ring allreduce is bandwidth-suboptimal. NCCL uses tree + ring hybrid; Megascale uses hierarchical (intra-node ring, inter-node tree). Apple Silicon clusters with Thunderbolt daisy-chain topology especially benefit from tree algorithms.
- **Fix:** Implement tree allreduce for latency-sensitive small tensors; keep ring for bandwidth-bound large tensors.
- **Source:** Frontier CUDA comparison

### P1-6: f16/bf16 Conversion Edge Cases
- **Location:** `group.rs` f16/bf16 helper functions
- **Description:** Float-to-half and half-to-float conversions don't explicitly handle NaN payload preservation, subnormal flushing, or Inf edge cases. MLX and CUDA libraries preserve NaN payloads and handle subnormals correctly per IEEE 754. Silent precision loss in reductions can accumulate.
- **Fix:** Add explicit NaN/Inf/subnormal handling, or delegate to the half crate's IEEE-compliant conversion.
- **Source:** Direct code audit + MLX comparison

### P1-7: SlabRing No Backpressure — Producer Overwrites Consumer
- **Location:** `slab_ring.rs`
- **Description:** If producer laps consumer (all slabs in-flight), the ring has no blocking backpressure mechanism. The `try_acquire` returns None but `acquire` may spin-wait. Under sustained load, this causes either busy-wait CPU burn or silent data loss if callers don't check return values.
- **Fix:** Add bounded blocking with condvar or GpuEvent-based backpressure. Log warnings on ring full events.
- **Source:** Direct code audit

## P2 — Medium

### P2-1: Dead Code — Metal Kernel Constants Not Integrated
- **Location:** `moe_exchange.rs` (multiple `#[allow(dead_code)]` constants)
- **Description:** Several Metal kernel function name constants (multi_dtype, packet_gather, packet_scatter variants) are declared but never referenced in the dispatch path. These represent partially implemented shader integration.
- **Impact:** Code bloat and confusion about which Metal kernels are actually used.
- **Fix:** Either integrate into dispatch path or remove with TODO comments for future phases.
- **Source:** Direct code audit

### P2-2: No Topology-Aware Routing
- **Location:** `transport.rs`, `group.rs`
- **Description:** RDMA routing is flat — all peers treated as equidistant. Thunderbolt 5 daisy-chain topology has asymmetric hop counts. NCCL does topology detection (NVLink/PCIe/network) and builds optimal rings. Megascale does ECMP-aware placement.
- **Fix:** Read TB5 topology from system_profiler (or rmlx-cli's config output) and build topology-aware rings.
- **Source:** Frontier CUDA comparison

### P2-3: No Multi-Backend Fallback
- **Location:** `init.rs`
- **Description:** MLX supports graceful fallback: JACCL → ring → MPI → single-node. rmlx-distributed only has RDMA and loopback. If RDMA init fails, there's no TCP/shared-memory fallback path.
- **Fix:** Add TCP transport as fallback (lower priority given Apple Silicon RDMA focus).
- **Source:** MLX comparison

### P2-4: Missing Latency/Throughput Metrics in Hot Path
- **Location:** `metrics.rs`, `perf_counters.rs`
- **Description:** Metrics track counts (dispatch_count, combine_count) but not latency histograms or throughput rates. No p50/p99 latency tracking for dispatch→combine round-trip. DeepEP and Megascale track per-operation latency distributions.
- **Fix:** Add Instant-based latency tracking in dispatch/combine hot path. Consider HDR histogram for low-overhead percentile tracking.
- **Source:** Frontier CUDA comparison

### P2-5: `CreditManager` Unsafe Without Send/Sync Audit
- **Location:** `credit_manager.rs`
- **Description:** `ensure_credits` and `replenish` use unsafe blocks for RDMA operations. The struct is shared across threads (used in EpRuntimeContext) but there's no explicit Send/Sync impl audit. RDMA handle thread-safety depends on underlying ibverbs guarantees.
- **Fix:** Add `// SAFETY:` comments documenting Send/Sync invariants. Consider wrapping RDMA ops in Mutex if ibverbs handles aren't thread-safe.
- **Source:** Direct code audit

### P2-6: No Python API / Bindings
- **Location:** Entire crate
- **Description:** MLX exposes full distributed API through Python (mlx.distributed). rmlx has no Python bindings. This blocks adoption by ML researchers who work primarily in Python.
- **Fix:** PyO3 bindings for core types (Group, init, allreduce). Lower priority if Rust-native usage is primary target.
- **Source:** MLX comparison

### P2-7: Coordinator Single Point of Failure in Init
- **Location:** `init.rs` coordinator-mediated all_gather
- **Description:** Bootstrap uses rank 0 as coordinator. If rank 0 is slow or fails during init, all other ranks hang indefinitely. No timeout on coordinator connection.
- **Fix:** Add connection timeout, retry with exponential backoff, and clear error message on coordinator unreachable.
- **Source:** Direct code audit

### P2-8: `PipelineStage` State Machine Lacks Error State
- **Location:** `pipeline.rs` PipelineStage enum
- **Description:** State machine has Computing/Transferring/Idle/Complete states but no Error/Failed state. A GPU fault or RDMA error during transfer leaves the pipeline in an inconsistent state with no recovery path.
- **Fix:** Add Error state with error context, propagate to LayerPipeline for clean shutdown or retry.
- **Source:** Direct code audit

## P3 — Low

### P3-1: `sparse_guard.rs` Magic Constants
- **Location:** `sparse_guard.rs`
- **Description:** EMA alpha (0.1), overflow threshold (0.9), capacity increase factor — all hardcoded. Should be configurable for different model architectures.
- **Fix:** Accept SparseGuardConfig struct with defaults.
- **Source:** Direct code audit

### P3-2: `perf_counters.rs` Global Singleton Pattern
- **Location:** `perf_counters.rs` `global_counters()`
- **Description:** Uses `OnceLock` global singleton. Makes testing harder (counters leak between tests) and prevents per-group counters in multi-tenant scenarios.
- **Fix:** Consider instance-based counters attached to Group/Context.
- **Source:** Direct code audit

### P3-3: No Device-Side Collective API
- **Location:** `group.rs`
- **Description:** All collectives are host-initiated. NCCL 2.x supports device-side launch (kernel-initiated collectives). Not critical for Apple Silicon (no equivalent to CUDA's device-side launch) but relevant if Metal 4 adds this capability.
- **Source:** Frontier CUDA comparison

### P3-4: V2 Wire Protocol Still Present
- **Location:** `moe_exchange.rs` WireProtocol enum
- **Description:** V2 protocol code paths remain alongside V3. If V2 is deprecated, dead code should be removed. If both are supported, there should be negotiation.
- **Fix:** Either remove V2 or add version negotiation in handshake.
- **Source:** Direct code audit

### P3-5: Missing JACCL / MPI Compatibility Layer
- **Location:** `init.rs`
- **Description:** Init reads `RMLX_RANK`/`RMLX_WORLD_SIZE` and `OMPI_COMM_WORLD_RANK`/`SLURM_PROCID` env vars, but there's no actual MPI or JACCL integration. MLX's distributed module has full MPI backend support.
- **Fix:** Low priority — document as intentional omission for Apple Silicon RDMA-first design.
- **Source:** MLX comparison

## Comparative Summary

| Capability | rmlx-distributed | MLX distributed | NCCL/DeepEP/Megascale |
|---|---|---|---|
| Zero-copy RDMA | ★★★★★ | ★★★★ | ★★★★ (GDR) |
| FP8 wire quantization | ★★★★★ | ★★★ | ★★★★ (DeepEP) |
| Async compute↔comm overlap | ★★★★ | ★★★ | ★★★★★ |
| Collective algorithms | ★★ (ring only) | ★★★ (ring+MPI) | ★★★★★ (tree+ring+SHARP) |
| Fault tolerance | ★ | ★★ | ★★★★ (Megascale) |
| Topology awareness | ★ | ★★★★ (TB5 DFS) | ★★★★★ (NVLink/PCIe) |
| Multi-backend support | ★★ (RDMA+loopback) | ★★★★ (JACCL+ring+MPI) | ★★★★★ |
| Python API | ☆ | ★★★★★ | ★★★★★ |
| Wire protocol | ★★★★ (v3 var-len) | ★★★ | ★★★★ |
| Flow control | ★★★★ (credit-based) | ★★★ | ★★★★★ (NACK-based) |
| Production monitoring | ★★ | ★★★ | ★★★★★ |

**Overall Production Readiness: 7/10**

Strengths: Zero-copy RDMA pipeline, FP8 quantization, async overlap architecture, and credit-based flow control are genuinely best-in-class for Apple Silicon. The v3 wire protocol and SlabRing design are clean and efficient.

Critical gaps: EP dispatch/combine correctness (P0-2), blocking exchange deadlock (P0-3), and unsafe buffer access (P0-1) must be fixed before any production deployment. Fault tolerance (P1-4) and collective algorithm diversity (P1-5) are the biggest gaps versus CUDA ecosystem maturity.

## Recommended Priority Order

1. **P0-1, P0-2, P0-3** — Fix unsafe/panic, EP correctness, deadlock potential
2. **P1-1, P1-2** — Ring rounding + MoePolicy thread safety (correctness bugs)
3. **P1-3** — Replace fake warmup benchmarks with real Metal/RDMA calibration
4. **P1-4** — Add basic fault detection (heartbeat + timeout)
5. **P1-7** — SlabRing backpressure
6. **P1-5, P1-6** — Tree algorithms + f16 edge cases (performance/correctness)
7. **P2-*/** — Topology awareness, metrics, Python API (maturity improvements)

## Appendix: Codex Independent Audit

_Generated by `codex exec --full-auto --json` on 2026-03-06_

Codex independently read all 15 `src/*.rs` and 4 `tests/*.rs` files, confirmed `cargo check -p rmlx-distributed --tests` passes, and produced the following findings. Line references point to specific evidence locations.

### P0 — Critical

1. **Async combine uses global expert index against local-only buffer** — `combine_async` indexes `local_expert_outputs` with global expert IDs, so non-zero ranks send zeroed or wrong segments. (`moe_exchange.rs:3158-3164`)
2. **Zero-copy RDMA `PendingOp` does not own buffer/MR lifetime** — `send_zero_copy_async` / `recv_zero_copy_async` return `PendingOp` without preventing the caller from dropping `SharedBuffer` while DMA is in flight, enabling use-after-free. (`transport.rs:549, 765, 802`) **Status: ✅ FIXED (Phase 0, PR #38)** — ZeroCopyPendingOp with Arc<SharedBuffer>, infinite-wait Drop

### P1 — High

1. **Ring reduction chunk size can be non-f32-aligned** — `div_ceil` operates on byte counts, producing chunk sizes not divisible by `size_of::<f32>()`, causing partial or incorrect reductions. (`group.rs:695, 711, 899, 1023`)
2. **`blocking_exchange_v3` deadlock on asymmetric zero-token case** — Asymmetric branch behavior (`send` vs `sendrecv` size/payload phases) deadlocks when one side has zero tokens and the other has non-zero. (`v3_protocol.rs:476, 482, 503, 516`)
3. **`dispatch_async` missing validation parity with `dispatch`** — No token_data divisibility check, no strict expert-range enforcement, missing conn/buffer entries silently skipped. (`moe_exchange.rs:2066, 2107, 2183, 2331`)
4. **Descriptor proxy path does not validate `peer_id` bounds** — Malformed GPU descriptor can panic or post RDMA ops to invalid regions. (`transport.rs:624, 643, 656`)
5. **Metal buffer readers panic in production path** — `read_buffer_f32` can overflow `n * size_of::<f32>()` before bounds check; `assert!` panics crash the process. (`moe_exchange.rs:45, 63, 82`)
6. **`combine_with_layout` blind indexing on route indices** — Only checks non-empty, then indexes without bounds validation; malformed or stale layout causes panic. (`moe_exchange.rs:3009, 3083`)

### P2 — Medium

1. **f16/bf16 conversion edge cases incorrect** — f16 subnormals decoded ~2x too small, f32 subnormals flushed to zero, some NaNs collapse to Inf, bf16 uses truncation with no rounding. (`group.rs:1063, 1073, 1094, 1099, 1114`)
2. **SlabRing TOCTOU race in multi-producer/consumer** — `load/check/fetch_add` sequence without CAS reservation; concurrent producers or consumers can reuse the same slab. (`slab_ring.rs:200, 220, 229, 261`)
3. **Silent fallback to loopback on RDMA failure** — Multi-rank auto-init can silently fall back to rank0/world1 loopback, risking incorrect distributed execution instead of fail-fast. (`init.rs:205, 545, 554`)
4. **Remaining panic paths in non-test code** — `expect` in connection setup (`init.rs:420`), assert in SlabRing constructor (`slab_ring.rs:162`).
5. **Warmup/calibration is placeholder** — No real RDMA ping-pong or Metal JIT calibration; synthetic CPU `sin()` benchmarks set GPU/RDMA thresholds. (`init.rs:481`, `warmup.rs:146`)

### P3 — Low

1. **Dead Metal kernel constants** — Large set of `#[allow(dead_code)]` kernel function name constants not exercised by any runtime path. (`moe_exchange.rs:202, 247, 469, 528`)
2. **`blocking_exchange_v3` exported but not integrated** — Exported in `lib.rs` but MoE dispatch path uses bespoke two-phase logic; V3 protocol is orphaned from the hot path. (`lib.rs:41`, `v3_protocol.rs:430`, `moe_exchange.rs:1498`)
3. **Doc/code drift** — Comments claim RDMA `dispatch()` delegates to `dispatch_async()`, but implementation takes blocking `route_rdma_zero_copy` path. (`moe_exchange.rs:694, 722, 900`)

### Test Coverage Gaps

1. No end-to-end tests for `WireProtocol::V3` in the MoE dispatch/combine path; only pack/unpack unit tests in the protocol module.
2. No tests covering `dispatch_async`, `combine_async_start`, `combine_async_finish` correctness across ranks.
3. No tests for ring reduction chunk misalignment cases (`len % (4*n) != 0`) or f16/bf16 NaN/subnormal edge behavior.
4. Metal-dependent slab ring tests skip via `std::process::exit(0)`, which can prematurely terminate the test process and hide unrelated failures. (`slab_ring.rs:328`)

### Codex Run Metadata

- Input tokens: 4,958,695 (cached: 4,589,952)
- Output tokens: 24,101
- Files analyzed: 19 (15 src + 4 tests)
- `cargo check -p rmlx-distributed --tests`: passed (with Metal AOT skip warning)
