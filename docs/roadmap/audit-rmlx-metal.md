# rmlx-metal Production Readiness Audit

**Date:** 2026-03-06
**Auditors:** Claude Opus 4.6, Codex
**Verdict:** NOT PRODUCTION-READY — 3 Critical, 2 High, multiple Medium issues
**Reference:** MLX (MLX source (mlx/backend/metal/)), PyTorch MPS, metal-rs 0.31

## Executive Summary

rmlx-metal is a 20-module Metal GPU abstraction layer (device, queue, command, buffer, pipeline, library, library_cache, managed_buffer, stream, event, fence, batcher, autorelease, capture, msl_version, icb, icb_sparse, exec_graph, self_check). It provides solid foundational abstractions but has critical FFI bugs, broken dispatch sizing, and significant feature gaps vs MLX and frontier Metal frameworks.

## Module Inventory

| Module | File | Lines | Purpose | Status |
|--------|------|-------|---------|--------|
| `GpuDevice` | device.rs | 113 | Metal device wrapper with arch detection | Functional, needs tuning |
| `GpuQueue` | queue.rs | 80 | Command queue with error store | Functional |
| `CommandBufferManager` | command.rs | 382 | Batching + completion handlers + barrier tracker | Functional, needs tuning |
| `CommandBatcher` | batcher.rs | 370 | Encoder batching with stats | Functional, needs doc |
| `buffer` | buffer.rs | 67 | Buffer creation helpers (unsafe FFI) | Functional |
| `PipelineCache` | pipeline.rs | 204 | Thread-safe RwLock pipeline + specialization cache | Functional, needs disk cache |
| `LibraryCache` | library_cache.rs | 130 | MSL source hash-based library cache | Functional |
| `ManagedBuffer` | managed_buffer.rs | 124 | RAII buffer with allocator trait | Functional |
| `StreamManager` | stream.rs | 168 | Multi-stream (default/compute/copy) with fence sync | Functional, minor dead code |
| `GpuEvent` | event.rs | 120 | MTLSharedEvent with escalating CPU wait | Functional |
| `GpuFence` | fence.rs | 120 | Monotonic fence on MTLSharedEvent | Functional |
| `ScopedPool` | autorelease.rs | 62 | RAII NSAutoreleasePool | Functional, thread-safety gap |
| `CaptureScope` | capture.rs | 175 | GPU trace capture via MTLCaptureManager | **Broken FFI** |
| `MslVersion` / `DeviceInfo` | msl_version.rs | 92 | MSL version detection per architecture | Functional |
| `IcbBuilder` / `IcbReplay` | icb.rs | 164 | CPU-side dispatch replay (not real ICB) | Functional, naming misleading |
| `SparseExpertPlan` | icb_sparse.rs | 340 | MoE sparse expert dispatch | **SwiGLU bug** |
| `ExecGraph` | exec_graph.rs | 200 | Async execution graph with event chain | **Unsignaled token bug** |
| `self_check` | self_check.rs | 80 | Startup validation | Functional |

## P0 — Critical Issues

### CRIT-1: CaptureScope FFI Memory Corruption

**Status: ✅ FIXED (Phase 0, PR #38)** — replaced raw objc with safe metal crate APIs

| Field | Detail |
|-------|--------|
| **Files** | `capture.rs:136`, `capture.rs:150-155` |
| **Description** | Two FFI errors: (1) `device as *const metal::Device as *const Object` gives a pointer to the Rust wrapper struct, not the underlying ObjC `id`. Must use `ForeignType::as_ptr()` to get the raw ObjC pointer. (2) `stringWithUTF8String:` receives `str::as_bytes().as_ptr()` which is NOT null-terminated — reads past buffer. (3) ObjC autorelease objects (`NSString`, `NSURL`) leak because no autorelease pool wraps the calls. |
| **Impact** | UB on capture start — potential crash, memory corruption, or silent wrong-object binding. |
| **Fix** | (A) Extract device pointer: `let raw = device.as_ptr() as *mut Object;`. (B) Use `CString::new(path).unwrap().as_ptr()` for null-terminated string. (C) Wrap all ObjC calls in `objc::rc::autoreleasepool(\|\| { ... })`. |

### CRIT-2: SparseExpertPlan SwiGLU Grid Under-Dispatch

**Status: ✅ FIXED (Phase 0, PR #38)** — changed to 2D grid preserving intermediate_dim

| Field | Detail |
|-------|--------|
| **Files** | `icb_sparse.rs:192-201` (build), `icb_sparse.rs:96-108` (replay) |
| **Description** | `build()` creates SwiGLU dispatch as 1D grid: `MTLSize::new(max_cap * inter, 1, 1)`. `replay_with_count()` replaces grid width with just `token_count`, dropping the `* intermediate_dim` factor. Result: SwiGLU launches `token_count` threads instead of `token_count * intermediate_dim` — under-dispatches by factor of `intermediate_dim`. |
| **Impact** | Incorrect MoE SwiGLU computation — wrong output values. |
| **Fix** | Change SwiGLU in `build()` to 2D grid: `MTLSize::new(max_cap, inter, 1)` with threadgroup `MTLSize::new(min(16, max_cap), min(16, inter), 1)`. This way `replay_with_count()` correctly replaces width=`token_count` while preserving height=`intermediate_dim`. |

### CRIT-3: CommandBatcher Encoder Lifecycle Documentation Gap

**Status: ✅ FIXED (Phase 0, PR #38)** — doc comments + debug_assert

| Field | Detail |
|-------|--------|
| **Files** | `batcher.rs:95-99`, `batcher.rs:124-126` |
| **Description** | `end_encoder()` only flips `encoder_active` bool — does NOT call Metal's `end_encoding()`. The caller must call `end_encoding()` on the returned encoder reference before calling `end_encoder()`. If `encoder()` is called twice without `end_encoding()` on the first, the first encoder leaks. |
| **Verdict** | NOT a code bug — this is the correct pattern because CommandBatcher borrows out the encoder reference and cannot call `end_encoding()` itself. But the API contract is undocumented and fragile. |
| **Fix** | Add doc comments on `encoder()` and `end_encoder()` explicitly stating: "Caller MUST call `end_encoding()` on the returned encoder before calling `end_encoder()` or calling `encoder()` again. Failure to do so leaks the Metal encoder." Consider adding a debug_assert checking encoder_active consistency. |

## P1 — High Issues

### HIGH-1: ExecGraph Unsignaled Token → Deadlock

**Status: ✅ FIXED (Phase 0, PR #38)** — guard returns previous token

| Field | Detail |
|-------|--------|
| **Files** | `exec_graph.rs:128-134`, `batcher.rs:173-181` |
| **Description** | `submit_batch()` increments counter and returns `EventToken`, then calls `batcher.flush_signal()`. If no command buffer exists (`current_cb` is `None`), `flush_signal()` skips the signal. The returned token references an event value that will never be signaled. Any `wait_for(token)` causes GPU deadlock; `sync()` causes CPU hang until timeout. |
| **Fix** | Guard `submit_batch()`: if `!self.batcher.has_pending()`, either (a) return the previous token unchanged, or (b) create an empty CB, signal, and commit (guarantees every token is signaled). Option (b) preferred for invariant safety. |

### HIGH-2: ICB Naming Misleading — No Real MTLIndirectCommandBuffer

| Field | Detail |
|-------|--------|
| **Files** | `icb.rs:1-19`, `icb_sparse.rs:1-10` |
| **Description** | Modules named "ICB" (Indirect Command Buffer) but use CPU-side re-encoding via `CapturedDispatch`. No `MTLIndirectCommandBuffer` is created or used. |
| **Verdict** | KEEP — The current CPU-side approach is actually correct for MoE use case. Metal's true ICB cannot mix different pipeline states per command (each command inherits from one pipeline), but MoE needs 4 different pipelines per expert (gate, up, swiglu, down). The re-encoding cost is minimal since pipeline states and buffers are pre-resolved. |
| **Fix** | Rename modules to `dispatch_replay` and `sparse_dispatch` to avoid confusion. Update doc comments to explain why true ICB is not used. Keep `_device` parameter in `IcbBuilder::build()` as API placeholder for future Metal ICB support. |

## P2 — Medium Issues

### MED-1: StreamState.label Dead Code

**Status: ✅ FIXED (Phase 0, PR #38)** — wired to Metal CommandQueue

- **File:** `stream.rs:30-31`
- **Verdict:** IMPLEMENT — Wire `label` to `queue.set_label()` for Metal debugger/profiler visibility. Remove `#[allow(dead_code)]`.

### MED-2: ScopedPool Thread Safety Gap

**Status: ✅ FIXED (Phase 0, PR #38)** — PhantomData<*mut ()>

- **File:** `autorelease.rs:28`
- **Description:** Raw pointer wrapper with no `!Send` marker. Can be accidentally moved cross-thread.
- **Fix:** Add `impl !Send for ScopedPool {}` or `PhantomData<*mut ()>` to prevent cross-thread moves.

### MED-3: Panic Paths in Production Code

**Status: ✅ FIXED (Phase 0, PR #38)** — eliminated with safe APIs

- **Files:** `capture.rs:150,156`, `autorelease.rs:44`, `stream.rs:155`, `icb_sparse.rs:266`
- **Description:** `unwrap()`, `expect()`, `assert!()` in non-test code paths.
- **Fix:** Replace with `Result` returns or graceful fallbacks.

### MED-4: Completion Handler Drops Rich Error Details
- **File:** `command.rs:261`
- **Description:** Only records status enum text, drops Metal's detailed error description.
- **Fix:** Capture `cb.error()` message string when status is Error/Fault.

### MED-5: StreamManager Uses Wrong Error Variant

**Status: ✅ FIXED (Phase 0, PR #38)** — added to MetalError

- **File:** `stream.rs:107`
- **Description:** Missing stream mapped to `MetalError::KernelNotFound` — semantically wrong.
- **Fix:** Add `MetalError::StreamNotFound(u32)` variant.

## Dead Code / Orphan Analysis — Decisions

| Item | File | Verdict | Rationale |
|------|------|---------|-----------|
| `StreamState.label` | stream.rs:30 | **IMPLEMENT** | Wire to Metal queue label for debugger. Trivial, high-value. |
| `ExpertDispatchGroup.expert_id` | icb_sparse.rs:47 | **KEEP** | Zero-cost structural identifier. Useful for future debug logging, error messages, profiler labels. |
| `IcbBuilder::build(_device)` | icb.rs:86 | **KEEP** | API placeholder for future true ICB. True ICB not viable for MoE (pipeline-per-command limitation). |
| `should_flush()` | batcher.rs:142 | **KEEP** | Advisory query for callers (ExecGraph). Auto-flush would be dangerous — could split dependent encoders across CBs. Design is correct. |
| `encoder_active` flag logic | batcher.rs:97 | **KEEP + DOC** | Not a bug — caller owns encoder lifetime. Add explicit doc comments on API contract. |
| `LibraryCache` u64 hash | library_cache.rs:120 | **KEEP** | Collision probability negligible at <1000 libraries. SipHash is well-distributed. |

## Feature Gap Analysis vs MLX + Frontier

### Features to IMPLEMENT

| # | Feature | Priority | Complexity | Implementation Strategy |
|---|---------|----------|------------|------------------------|
| F1 | **Chip-class tuning** | **P1** | S | Add `ChipTuning` struct to `device.rs` with per-generation thresholds (M1: 64ops/16MB, M3: 96ops/24MB, M4: 128ops/32MB). Modify `CommandBufferManager` and `CommandBatcher` to accept tuning params instead of hardcoded constants. MLX tunes by chip suffix (p/g/s/d). |
| F2 | **Unretained command buffer references** | **P2** | S | metal-rs exposes `CommandQueue::new_command_buffer_with_unretained_references()`. Add `use_unretained: bool` to `CommandBufferManager`. Safe because `ManagedBuffer` RAII guarantees resource lifetimes. ~1-2us savings per CB. |
| F3 | **Per-encoder MTLFence** | **P2** | M | metal-rs exposes `Device::new_fence()`, `ComputeCommandEncoderRef::update_fence()/wait_for_fence()`. Add `MetalFence` wrapper alongside existing `GpuFence` (SharedEvent). Modify `BarrierTracker` to use fence instead of end-encoder/new-encoder strategy. Saves ~5-10us per barrier. |
| F5 | **GPU error buffer** | **P2** | S | 64-byte `StorageModeShared` buffer bound to index 30 of every encoder. Shaders write error codes (NaN, OOB). Check after CB completion. Add `GpuErrorBuffer` struct to `command.rs`. |
| F7 | **Binary archive / disk pipeline cache** | **P1** | M | metal-rs exposes `BinaryArchive` API. Add `DiskPipelineCache` to `pipeline.rs`. On first compile, add pipeline to archive + serialize to disk. On subsequent launches, load from archive — eliminates ~50-200ms cold start per pipeline. Requires NSURL construction from Rust path via objc FFI. |
| F9 | **GPU Counter API** | **P2** | M | metal-rs exposes `counter_sets()`, `CounterSampleBuffer`. Add `profiler.rs` behind `profiling` feature flag. Sample counters at dispatch boundaries for programmatic ALU/cache/occupancy measurement. |
| F10 | **MTLHeap exposure** | **P1** | M | metal-rs exposes full `Heap`/`HeapDescriptor` API. Add `HeapAllocator` to rmlx-metal. Modify rmlx-alloc `SmallBufferPool` to use `HeapAllocator` instead of single-buffer bitmap. Each small alloc returns a real `metal::Buffer` from heap — eliminates offset tracking, enables `HazardTrackingModeUntracked` per-buffer. |

### Features to DEFER

| # | Feature | Priority | Rationale |
|---|---------|----------|-----------|
| F4 | **Fast fence (GPU atomics)** | P3 | Micro-optimization (~0.5-2us) over MTLFence. GPU spinlocks are fragile. Revisit if profiling shows MTLFence is bottleneck. |
| F6 | **Adaptive commit (memory pressure)** | P3 | Needs memory pressure monitoring in rmlx-alloc first. Chip-class tuning (F1) provides coarse version. Revisit for >8B parameter models. |

### Features to SKIP

| # | Feature | Rationale |
|---|---------|-----------|
| F8 | **Metal 4 readiness** | metal-rs has zero Metal 4 API surface. Current abstractions are thin enough to adapt later. Over-engineering risk. |
| — | **MPS integration** | MLX doesn't use MPS either. RMLX has custom kernels. Not needed. |
| — | **Cross-process SharedEvent** | No use case for RMLX (single-process inference). |
| — | **objc2-metal migration** | metal crate still maintained. Migration path exists but no urgency. Track for future. |

## Test Coverage Assessment

**Current:** 79 tests total, 22 pass without Metal device, 57 require GPU hardware.

### Missing Coverage
- CaptureScope FFI (currently crashes without device)
- SwiGLU dispatch grid sizing verification
- ExecGraph empty-submit deadlock scenario
- Multi-threaded pipeline cache stress test
- Encoder lifecycle double-create without end_encoding
- Stream synchronize ordering guarantees
- ManagedBuffer cross-thread usage
- Large-scale batcher throughput benchmark

## Cross-Crate Dependencies

| Dependency | Direction | What's Needed |
|-----------|-----------|---------------|
| `rmlx-metal` → `rmlx-alloc` | rmlx-alloc depends on rmlx-metal | `GpuDevice`, `Buffer`, `GpuEvent`. Need to add: `HeapAllocator` for SmallBufferPool, `HazardTrackingModeUntracked` helper |
| `rmlx-core` → `rmlx-metal` | rmlx-core uses rmlx-metal | Need to verify: pipeline cache lifecycle, command buffer manager ownership |
| `rmlx-nn` → `rmlx-metal` | rmlx-nn dispatches kernels | Need to verify: error buffer binding convention (index 30), barrier tracker integration |
| `rmlx-distributed` → `rmlx-metal` | EP uses GpuEvent/GpuFence | Need to verify: ExecGraph integration with EP dispatch path |

## Recommended Fix Roadmap

### Phase 1: Correctness (Block release)
1. **CRIT-1** — Fix CaptureScope FFI (device pointer, CString, autorelease pool)
2. **CRIT-2** — Fix SwiGLU dispatch to 2D grid
3. **HIGH-1** — Guard ExecGraph submit_batch against empty CB
4. **CRIT-3** — Document encoder lifecycle contract
5. **MED-2** — Add `!Send` to ScopedPool
6. **MED-3** — Remove panic paths from production code
7. **MED-5** — Add `MetalError::StreamNotFound`

### Phase 2: Performance Foundations
8. **F1** — Chip-class tuning (ChipTuning struct)
9. **F2** — Unretained command buffer references
10. **F10** — MTLHeap exposure (enables rmlx-alloc SmallBufferPool upgrade)
11. **F7** — Binary archive / disk pipeline cache
12. **MED-1** — Wire StreamState.label to Metal queue

### Phase 3: Advanced Features
13. **F3** — Per-encoder MTLFence (replace end-encoder barrier strategy)
14. **F5** — GPU error buffer
15. **F9** — GPU Counter API (behind feature flag)
16. **HIGH-2** — Rename ICB modules to dispatch_replay/sparse_dispatch
17. **MED-4** — Capture rich error details from completion handlers

---

*Generated by Claude Opus 4.6 + Codex — 2026-03-06*
