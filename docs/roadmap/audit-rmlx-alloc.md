# rmlx-alloc Production Readiness Audit

**Date:** 2026-03-06
**Auditors:** Claude Opus 4.6, Codex
**Verdict:** NOT PRODUCTION-READY — 1 Critical, 3 High, multiple Medium issues

## Executive Summary

rmlx-alloc is a well-structured GPU memory allocator for Apple Silicon Metal with 8 modules: MetalAllocator, BufferPool, BufferCache, LeakDetector, ResidencyManager, SmallBufferPool, AllocStats, and ZeroCopyBuffer. Architecture follows MLX patterns but has critical correctness bugs, missing features vs both MLX and CUDA frontier, and orphan modules not wired into the main allocation path.

## Module Inventory

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| MetalAllocator | allocator.rs | 262 | Core allocator with cache, GC, limits |
| BufferPool | buffer_pool.rs | 101 | Completion-aware ZeroCopyBuffer pool |
| BufferCache | cache.rs | 173 | Size-binned LRU buffer cache |
| LeakDetector | leak_detector.rs | 108 | Atomic leak detection |
| ResidencyManager | residency.rs | 368 | Metal 3 residency (feature-gated) |
| SmallBufferPool | small_alloc.rs | 234 | Slab allocator for <256B |
| AllocStats | stats.rs | 89 | Allocation statistics |
| ZeroCopyBuffer | zero_copy.rs | 434 | Page-aligned zero-copy with completion tracking |

## P0 — Critical Issues

### CRIT-1: Zero-Size Allocation Cache Poisoning

**Status: ✅ FIXED (Phase 0, PR #38)** — reject zero-size + validate cache buffer sizes

- **Location:** allocator.rs:114, cache.rs:42, cache.rs:73
- **Description:** `alloc(0)` creates a zero-length Metal buffer. `free()` caches it. Cache's `align_size()` rounds 0 up to 16, so later `acquire(n)` for n<=16 can return the zero-length buffer. The caller gets a buffer that's too small.
- **Impact:** Silent data corruption, GPU fault
- **Fix:** Either reject zero-size alloc or don't cache zero-length buffers; validate returned buffer length >= requested size in `acquire()`
- **MLX comparison:** MLX's `align_size` does NOT handle zero — but MLX never calls `malloc(0)` because the framework layer prevents it

## P1 — High Issues

### HIGH-1: Memory/Block Limit Bypass Under Concurrency
- **Location:** allocator.rs:121-130, allocator.rs:171
- **Description:** Limit checks use `stats.active()` (Relaxed atomic load) + `requested` without atomic reservation. Actual `buf.length()` may be larger due to alignment. Multiple threads can race past limits simultaneously.
- **Impact:** OOM despite configured limits
- **Fix:** Use atomic compare-and-swap reservation before allocation, adjust based on actual buffer size after
- **MLX comparison:** MLX uses a single mutex for the entire alloc path, naturally serializing limit checks. RMLX checks limits outside the lock.

### HIGH-2: Stats Underflow/Corruption on Free Path
- **Location:** allocator.rs:190, stats.rs:43
- **Description:** `record_free()` does `fetch_sub` without validation. `free()` accepts any MetalBuffer without ownership verification. A buffer not allocated by this allocator, or double-freed, will corrupt `active_bytes` (wrap around on underflow).
- **Impact:** Broken telemetry, broken OOM detection (active_bytes becomes huge)
- **Fix:** Use `saturating_sub` in stats, and add ownership validation (e.g., track allocated pointers in a HashSet, or use opaque handles)

### HIGH-3: Metal 3 ObjC Runtime Availability Gap
- **Location:** residency.rs:75-85
- **Description:** `msg_send!` calls to `MTLResidencySet` APIs assume runtime availability. On older macOS (pre-15) with `metal3` feature enabled, these will crash with "unrecognized selector."
- **Impact:** Process abort on unsupported platforms
- **Fix:** Add `@available(macOS 15, *)` style runtime check via `objc::runtime::class_exists("MTLResidencySetDescriptor")` before sending messages; also capture NSError from `newResidencySetWithDescriptor:error:`

## P2 — Medium Issues

### MED-1: SmallBufferPool Double-Free Corruption
- **Location:** small_alloc.rs:115-146
- **Description:** `free()` unconditionally sets bitmap slot to `true` and increments `free_count`. Repeated free of same slot inflates counters.
- **Fix:** Check slot state before freeing; panic or return error on double-free

### MED-2: ZeroCopyBuffer Send/Sync Assumptions
- **Location:** zero_copy.rs:183-190
- **Description:** Manual `unsafe impl Send/Sync` relies on Metal buffer thread-safety assumptions that aren't formally guaranteed by Apple docs.
- **Fix:** Document the exact invariants; consider wrapping raw_ptr in a type that prevents aliased mutable access

### MED-3: BufferCache LRU Linear Scan
- **Location:** cache.rs:131, cache.rs:150
- **Description:** `lru_remove` uses `VecDeque::position` (O(n) scan). At scale with thousands of cached buffers, this becomes a bottleneck.
- **Fix:** Replace VecDeque LRU with a doubly-linked list with O(1) removal (like MLX's intrusive linked list), or use `LinkedHashMap`

### MED-4: Mutex Poison Handling Inconsistency
- **Location:** allocator.rs:142 vs 193 vs 205
- **Description:** `alloc()` returns `AllocError::MutexPoisoned`, but `free()`, `clear_cache()`, `set_cache_limit()` silently ignore poisoning. This hides allocator degradation.
- **Fix:** Consistent strategy: either always propagate or document why ignoring is acceptable in each case

### MED-5: Busy-Spin Completion Waits
- **Location:** zero_copy.rs:98-107, zero_copy.rs:329
- **Description:** `wait_all_complete()` and `ZeroCopyBuffer::drop()` use `thread::yield_now()` spin loops. Burns CPU and adds tail-latency stalls.
- **Fix:** Use condvar or Metal shared event signaling (`MTLSharedEvent::notifyListener`) for blocking waits instead of spinning

## Orphan Modules (Not Integrated into MetalAllocator)

These modules exist and are tested independently but are NOT wired into the main `MetalAllocator` allocation path:

| Module | Status | Impact |
|--------|--------|--------|
| SmallBufferPool | Standalone | MetalAllocator allocates small buffers individually instead of using slab. MLX uses MTLHeap for <256B buffers. |
| LeakDetector | Standalone | MetalAllocator doesn't call `record_alloc`/`record_free` on LeakDetector |
| ResidencyManager | Standalone | MetalAllocator doesn't manage residency sets during alloc/free |

### Unused Error Variants
- `AllocError::MetalBufferCreate` — never constructed anywhere in the crate
- `AllocError::PoolExhausted` — never constructed anywhere in the crate

## Feature Gap Analysis vs MLX + CUDA Frontier

### Features MLX Has That RMLX Is Missing

| Feature | MLX Implementation | RMLX Status | Priority |
|---------|-------------------|-------------|----------|
| **MTLHeap for small alloc** | Pre-allocates 1MB MTLHeap, allocates <256B from it (true Metal sub-allocation) | Uses shared MetalBuffer + bitmap slab (not a real MTLHeap) | P1 — MTLHeap reduces per-buffer overhead significantly |
| **HazardTrackingModeUntracked** | All buffers created with `ResourceHazardTrackingModeUntracked` (performance optimization) | Not applied | P1 — free performance win |
| **Resource count limit** | Tracks `num_resources_` and triggers GC when exceeding `resource_limit_` from device info | Only tracks bytes, not buffer count | P2 |
| **make_buffer (external ptr wrap)** | `device->newBuffer(ptr, size, ...)` wraps external memory without copy, freed via `release()` not `free()` | No public API for wrapping external pointers | P2 |
| **Lock-free device allocation** | Drops mutex BEFORE calling `device->newBuffer()`, reacquires after | Device alloc happens inside implicit lock path | P2 |
| **Scoped autorelease pool** | Wraps Metal-cpp calls in `new_scoped_memory_pool()` to prevent ObjC autorelease accumulation | Not present | P2 |
| **Wired memory limit in ResidencyManager** | `ResidencySet` has capacity/wired limit, moves buffers between wired/unwired sets | ResidencyManager has no capacity concept | P2 |
| **Singleton with intentional leak** | Heap-allocated singleton, destructor never called at exit (avoids expensive cleanup) | Not applicable (Rust ownership) but consider `ManuallyDrop` pattern for global allocator | P3 |
| **GC limit formula** | `gc_limit = min(0.95 * max_recommended_working_set, block_limit)` — more conservative than block limit | `gc_limit = 2 GiB` fixed constant | P2 — should derive from device capabilities |

### Features CUDA Frontier Has (Applicable to Metal)

| Feature | Best Implementation | RMLX Status | Priority |
|---------|-------------------|-------------|----------|
| **Block splitting & coalescing** | PyTorch CCA / TF BFC — split oversized free blocks, merge adjacent free blocks on deallocation | Not implemented. Cache stores whole buffers only. | P1 — essential for long-running workloads |
| **Dual pool (small/large)** | PyTorch: <1MB / >=1MB separate pools. RMM: binning_resource with per-size-class fixed pools + fallback | Single pool only | P2 |
| **Graduated OOM recovery** | PyTorch: (1) evict LRU -> (2) flush all cache -> (3) trigger GC callback -> (4) fail | Single-step: evict then fail | P2 |
| **Composable allocator trait** | RMM: `device_memory_resource` trait + decorator pattern (logging, limiting, tracking wrappers) | Monolithic MetalAllocator | P2 — Rust traits are ideal for this |
| **Allocation history / memory snapshots** | PyTorch: `_record_memory_history()`, `_snapshot()` with stack traces, visualization tools | Basic atomic counters only | P2 |
| **Per-stream/queue free lists** | PyTorch: per-CUDA-stream caches. RMM: per-stream free lists. | No queue awareness | P3 — less critical for Metal unified memory but still beneficial |
| **Power-of-2 size rounding option** | PyTorch: configurable via `PYTORCH_CUDA_ALLOC_CONF` | Only power-of-2 for small sizes (cache.rs), page-align for large | P3 |
| **Thread-local arenas** | RMM arena_memory_resource: per-thread small arenas + shared global arena (jemalloc-inspired) | Single mutex | P3 — reduces lock contention |

## Test Coverage Assessment

**Current:** 13 integration tests + 3 unit tests + 6 inline tests = 22 tests total

**Missing coverage:**
- No concurrency tests (multi-threaded alloc/free)
- No stress tests (allocation patterns, fragmentation measurement)
- No zero-size allocation edge case test
- No double-free detection test
- No memory limit enforcement test under concurrency
- No Metal 3 residency integration test (only no-op stub tested)
- No benchmark tests

## Recommended Fix Priority

### Phase 1: Correctness (Block release)
1. Fix CRIT-1: Zero-size cache poisoning
2. Fix HIGH-1: Atomic limit reservation
3. Fix HIGH-2: Stats underflow protection + ownership check
4. Fix HIGH-3: Metal 3 runtime availability guard
5. Fix MED-1: SmallBufferPool double-free guard
6. Wire SmallBufferPool, LeakDetector, ResidencyManager into MetalAllocator

### Phase 2: MLX Parity
7. Add HazardTrackingModeUntracked to all buffer creation
8. Replace SmallBufferPool backing with real MTLHeap
9. Add resource count limit tracking
10. Implement dynamic gc_limit based on device capabilities
11. Add make_buffer API for external pointer wrapping
12. Add wired memory limit to ResidencyManager

### Phase 3: Beyond MLX (CUDA Frontier Features)
13. Implement block splitting and coalescing (BFC-style)
14. Add composable allocator trait (RMM-inspired decorator pattern)
15. Implement graduated OOM recovery with callback hooks
16. Add allocation history recording with backtrace support
17. Add dual pool (small/large threshold)
18. Add concurrency and stress tests

---

*Generated by Claude Opus 4.6 + Codex — 2026-03-06*
