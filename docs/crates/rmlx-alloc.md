# rmlx-alloc — Memory Allocator

## Overview

`rmlx-alloc` is the crate responsible for GPU memory allocation and zero-copy buffer management. It registers page-aligned memory allocated via `posix_memalign` with Metal's `newBufferWithBytesNoCopy`, providing simultaneous CPU/GPU access to a single physical memory region. It follows the MLX MetalAllocator pattern and includes size-binned caching and allocation statistics tracking.

> **Status:** Phase 1 implementation complete + Phase 0+1+2 audit remediation (items A1-A12). `ZeroCopyBuffer`, `MetalAllocator`, `BufferCache`, `AllocStats`, `LeakDetector`, and `SmallAllocator` are implemented. `ResidencyManager` API is present, but its Metal `MTLResidencySet` backend is currently a documented stub until `metal-rs` exposes the required bindings. Under feature `metal3`, current calls intentionally fail-fast with `todo!("TODO(A6)...")` rather than silently no-op.

---

## Module Structure

```mermaid
graph TD
    A[rmlx-alloc] --> B[allocator.rs — MetalAllocator]
    A --> C[zero_copy.rs — ZeroCopyBuffer]
    A --> D[cache.rs — BufferCache]
    A --> E[stats.rs — AllocStats]
    A --> F[leak_detector.rs — LeakDetector]
    B --> D
    B --> E
    C --> G[posix_memalign]
    C --> H[Metal NoCopy]
```

### `allocator.rs` — `MetalAllocator`

A Metal buffer allocator following the MLX MetalAllocator pattern. Checks the cache first on allocation, falling back to direct device allocation on cache miss.

```rust
pub struct MetalAllocator {
    device: Arc<GpuDevice>,
    cache: Mutex<BufferCache>,
    stats: AllocStats,
    block_limit: usize,           // maximum total allocation (0 = unlimited)
}
```

| Method | Description |
|--------|-------------|
| `new(device, max_cache_size)` | Creates a new allocator |
| `set_block_limit(limit)` | Sets the maximum total allocation limit (0 = unlimited) |
| `alloc(size)` | Allocates a Metal buffer (cache first, device fallback) |
| `free(buffer)` | Returns a buffer to the cache for reuse |
| `stats()` | Returns a reference to allocation statistics |
| `clear_cache()` | Clears the cache and releases all cached buffers |

**Allocation flow:**
1. Check `block_limit` (return `OutOfMemory` if exceeded)
2. Attempt `BufferCache::acquire(size)`
3. Cache hit -> record cache hit in stats and return
4. Cache miss -> call `device.new_buffer()` to allocate a new buffer

---

### `zero_copy.rs` — `ZeroCopyBuffer`

A zero-copy buffer where page-aligned memory can be simultaneously accessed by the CPU and Metal GPU.

**Allocation flow:**

```mermaid
flowchart LR
    A[posix_memalign<br/>page-aligned allocation + zero-fill] --> B[newBufferWithBytesNoCopy<br/>Metal GPU mapping]
```

1. **`posix_memalign`** — Allocates page-aligned memory in `sysconf(_SC_PAGESIZE)` units (typically 16KB on Apple Silicon) and zero-fills it
2. **`newBufferWithBytesNoCopy`** — Creates a Metal buffer accessible by the GPU without copying

```rust
pub struct ZeroCopyBuffer {
    raw_ptr: NonNull<u8>,
    metal_buffer: MetalBuffer,
    in_flight: Arc<()>,
    size: usize,
    _alignment: usize,
}
```

| Method | Description |
|--------|-------------|
| `new(device, size)` | Allocates a page-aligned zero-copy buffer |
| `metal_buffer()` | Returns a reference to the Metal buffer |
| `as_ptr()` | Returns a read-only pointer |
| `as_mut_ptr()` | Returns a writable pointer (requires `&mut self`) |
| `size()` | Returns the page-aligned buffer size |
| `acquire_in_flight()` | Acquires an `InFlightToken` to prevent deallocation during GPU/RDMA operations |
| `acquire_fence(op_tag)` | Acquires a `CompletionFence` for hardware completion verification |
| `in_flight_count()` | Returns the current in-flight reference count (including self; 1 if no external references) |

`Send` and `Sync` are manually implemented (guaranteeing thread safety for Metal `StorageModeShared` buffers on UMA-based Apple Silicon).

**Safe deallocation (Drop):**

The `ZeroCopyBuffer` `Drop` implementation waits via a `yield_now()` loop for up to 5 seconds until all in-flight tokens are released. On timeout, it intentionally leaks the memory to prevent use-after-free.

---

### In-Flight Tracking: `InFlightToken`, `CompletionFence`, `GpuCompletionHandler`

```rust
/// Arc<()>-based reference counting — prevents buffer deallocation
pub struct InFlightToken {
    _guard: Arc<()>,
}

/// ManuallyDrop-based — can only be released via release_after_verification()
pub struct CompletionFence {
    token: ManuallyDrop<InFlightToken>,
    op_tag: &'static str,
    verified: Arc<AtomicBool>,
}

/// Safely releases CompletionFence from Metal completedHandler callback
pub struct GpuCompletionHandler {
    fence: Option<CompletionFence>,
}
```

**Safety contract:**
- `CompletionFence::release_after_verification()` must only be called after the GPU command buffer has completed or `IBV_WC_SUCCESS` has been received from the CQ
- If a `CompletionFence` is dropped without verification, it intentionally leaks the `InFlightToken`, preventing `ZeroCopyBuffer::drop()` from freeing the memory (use-after-free prevention)
- `GpuCompletionHandler::on_completed()` safely releases the fence from a Metal completion callback

---

### `cache.rs` — `BufferCache`

A size-binned buffer cache following the MLX BufferCache pattern. Implemented with `BTreeMap<usize, VecDeque<MetalBuffer>>`, rounding sizes up to page boundaries.

| Method | Description |
|--------|-------------|
| `new(max_cache_size)` | Creates a new cache with the specified maximum cache size in bytes |
| `acquire(size)` | Searches the cache for a suitable buffer (exact match first, then next larger bin) |
| `release(buffer)` | Returns a buffer to the cache (evicts from the largest bin when max exceeded) |
| `cache_size()` | Returns the current cache usage in bytes |
| `clear()` | Releases all cached buffers |

**Eviction policy:** When the cache is full, eviction starts from the largest bin to reclaim as much memory as possible per eviction. Single buffers exceeding `max_cache_size` are dropped immediately without caching.

---

### `stats.rs` — `AllocStats`

A thread-safe allocation statistics tracker. All counters are `AtomicUsize`-based.

```rust
pub struct AllocStats {
    active_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    total_allocs: AtomicUsize,
    total_frees: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}
```

| Method | Description |
|--------|-------------|
| `record_alloc(size)` | Records an allocation (peak update via `compare_exchange_weak` CAS loop) |
| `record_free(size)` | Records a deallocation |
| `record_cache_hit()` / `record_cache_miss()` | Records a cache hit/miss |
| `active()` | Returns current active bytes |
| `peak()` | Returns peak bytes |
| `total_allocs()` / `total_frees()` | Returns total allocation/deallocation counts |
| `cache_hits()` / `cache_misses()` | Returns cache hit/miss counts |

---

### `leak_detector.rs` — `LeakDetector`

An `AtomicU64`-based lock-free allocation/deallocation counter for memory leak detection. Tracks the high-water mark of allocated bytes via a CAS loop.

```rust
pub struct LeakDetector {
    alloc_count: AtomicU64,
    free_count: AtomicU64,
    alloc_bytes: AtomicU64,
    free_bytes: AtomicU64,
    high_water_mark_bytes: AtomicU64,
}
```

| Method | Description |
|--------|-------------|
| `new()` | Creates a new `LeakDetector` with all counters at 0 |
| `record_alloc(bytes)` | Records an allocation (count +1, bytes accumulated, high-water-mark CAS update) |
| `record_free(bytes)` | Records a deallocation (count +1, bytes accumulated) |
| `outstanding_allocs()` | Returns the number of unreleased allocations (`alloc_count - free_count`) |
| `outstanding_bytes()` | Returns the number of unreleased bytes (`alloc_bytes - free_bytes`) |
| `has_potential_leak()` | Heuristic leak detection: outstanding allocs > 0 **and** outstanding bytes > 1 MiB |
| `report()` | Returns a `LeakReport` snapshot of the full state |

**`LeakReport` snapshot:**

```rust
#[derive(Debug, Clone)]
pub struct LeakReport {
    pub total_allocs: u64,
    pub total_frees: u64,
    pub outstanding_allocs: u64,
    pub outstanding_bytes: u64,
    pub high_water_mark_bytes: u64,
    pub potential_leak: bool,
}
```

**High-water mark update:** On each `record_alloc()` call, the current outstanding bytes (`alloc_bytes - free_bytes`) are computed. If greater than the existing high-water mark, it is atomically updated via a `compare_exchange_weak` CAS loop.

> The `Default` trait is implemented, so `LeakDetector::default()` can also be used for creation.

---

## Error Handling

```rust
#[derive(Debug)]
pub enum AllocError {
    PosixMemalign(i32),         // posix_memalign failed (error code)
    MetalBufferCreate,          // Metal buffer creation failed
    OutOfMemory {               // Out of memory
        requested: usize,
        available: usize,
    },
    PoolExhausted,              // Buffer pool exhausted
}
```

---

## Memory Management Flow

```mermaid
sequenceDiagram
    participant App
    participant MetalAllocator
    participant BufferCache
    participant GpuDevice

    App->>MetalAllocator: alloc(size)
    MetalAllocator->>BufferCache: acquire(size)
    alt Cache hit
        BufferCache-->>MetalAllocator: Some(buffer)
    else Cache miss
        MetalAllocator->>GpuDevice: new_buffer(size, Shared)
        GpuDevice-->>MetalAllocator: buffer
    end
    MetalAllocator-->>App: buffer
    App->>MetalAllocator: free(buffer)
    MetalAllocator->>BufferCache: release(buffer)
```

---

## Dependencies

```toml
[dependencies]
rmlx-metal = { path = "../rmlx-metal" }
libc = "0.2"
```
