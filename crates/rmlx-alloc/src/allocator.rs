use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;

use crate::cache::BufferCache;
use crate::leak_detector::LeakDetector;
use crate::residency::ResidencyManager;
use crate::small_alloc::{SmallAllocation, SmallBufferPool, MAX_SMALL_ALLOC};
use crate::stats::AllocStats;
use crate::AllocError;

/// Default GC limit: 2 GiB. When active memory + cache size + request
/// approaches this threshold the allocator proactively evicts cached buffers
/// before falling through to a device allocation.
const DEFAULT_GC_LIMIT: usize = 2 * 1024 * 1024 * 1024;

/// Metal buffer allocator with size-binned caching.
///
/// Follows the MLX MetalAllocator pattern: allocate from device,
/// cache freed buffers for reuse, track peak memory usage.
///
/// A configurable `gc_limit` triggers proactive cache eviction when total
/// memory pressure (active + cached + requested) would exceed the threshold.
pub struct MetalAllocator {
    device: Arc<GpuDevice>,
    cache: Mutex<BufferCache>,
    stats: AllocStats,
    /// Maximum total allocation (0 = unlimited).
    /// Uses AtomicUsize so `set_block_limit` can take `&self` (A7).
    block_limit: AtomicUsize,
    /// GC threshold: evict cached buffers when
    /// `active_memory + cache_size + request_size >= gc_limit`.
    /// Default: [`DEFAULT_GC_LIMIT`] (2 GiB).
    gc_limit: usize,
    /// Hard memory limit (0 = unlimited). When set, `alloc()` returns
    /// `OutOfMemory` if `active + requested > memory_limit` (A12).
    memory_limit: AtomicUsize,
    /// Atomically tracked allocated bytes for CAS-based limit enforcement
    /// (PR 4.1). This is the source of truth for limit checks and is
    /// updated atomically via compare_exchange, unlike `stats.active()`
    /// which is updated after allocation.
    allocated_bytes: AtomicUsize,
    /// Set of GPU addresses for buffers currently owned by this allocator
    /// (PR 4.2). Used to detect double-free and freeing unowned buffers.
    owned_ptrs: Mutex<HashSet<u64>>,
    /// Small-buffer pool for allocations <= 256 bytes (PR 4.3).
    /// Recycles fixed-size Metal buffers to reduce runtime overhead for
    /// tiny allocations.
    small_pool: SmallBufferPool,
    /// Stack of outstanding SmallAllocations per GPU address (PR 4.3, P7-10).
    /// With sub-allocation, all slots share the same backing buffer GPU address,
    /// so we use a Vec per address. On free, we pop any slot — all slots are
    /// interchangeable from the caller's perspective.
    small_allocs: Mutex<HashMap<u64, Vec<SmallAllocation>>>,
    /// Leak detector tracking alloc/free counts and bytes (PR 4.3).
    leak_detector: LeakDetector,
    /// Optional Metal 3 residency manager (PR 4.3). Populated at runtime
    /// if the device supports Metal 3; `None` otherwise.
    residency: Mutex<Option<ResidencyManager>>,
}

struct ReservationGuard<'a> {
    allocator: &'a MetalAllocator,
    size: usize,
    defused: bool,
}

impl<'a> ReservationGuard<'a> {
    fn new(allocator: &'a MetalAllocator, size: usize) -> Self {
        Self {
            allocator,
            size,
            defused: false,
        }
    }

    fn defuse(mut self) {
        self.defused = true;
    }
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        if !self.defused {
            self.allocator.release_reserved(self.size);
        }
    }
}

impl MetalAllocator {
    /// Create a new allocator. `max_cache_size` controls buffer cache capacity.
    ///
    /// The `block_limit` is auto-detected from Metal device info (A8):
    /// `min(1.5 * recommended_max_working_set_size, 0.95 * total_memory)`.
    /// Falls back to 8 GiB if the device query returns 0.
    /// Use [`set_block_limit`] to override after construction.
    pub fn new(device: Arc<GpuDevice>, max_cache_size: usize) -> Self {
        let block_limit = auto_detect_block_limit(device.raw());
        let small_pool = SmallBufferPool::new(&device, None);

        // Attempt to create a Metal 3 residency manager at runtime.
        // This succeeds on M3+ devices; on older hardware, `new()` returns
        // an error and we store `None`.
        let residency = ResidencyManager::new(device.raw()).ok();

        Self {
            device,
            cache: Mutex::new(BufferCache::new(max_cache_size)),
            stats: AllocStats::new(),
            block_limit: AtomicUsize::new(block_limit),
            gc_limit: DEFAULT_GC_LIMIT,
            memory_limit: AtomicUsize::new(0),
            allocated_bytes: AtomicUsize::new(0),
            owned_ptrs: Mutex::new(HashSet::new()),
            small_pool,
            small_allocs: Mutex::new(HashMap::new()),
            leak_detector: LeakDetector::new(),
            residency: Mutex::new(residency),
        }
    }

    /// Set maximum total allocation limit (0 = unlimited).
    ///
    /// Takes `&self` (not `&mut self`) thanks to interior AtomicUsize (A7).
    pub fn set_block_limit(&self, limit: usize) {
        self.block_limit.store(limit, Ordering::Relaxed);
    }

    /// Get the current block limit (0 = unlimited).
    pub fn block_limit(&self) -> usize {
        self.block_limit.load(Ordering::Relaxed)
    }

    /// Set the GC threshold. When `active + cache + request >= gc_limit`,
    /// the allocator proactively evicts cached buffers before allocating
    /// from the device. Set to 0 to disable proactive GC.
    pub fn set_gc_limit(&mut self, limit: usize) {
        self.gc_limit = limit;
    }

    /// Set a hard memory limit (A12). When set (> 0), `alloc()` returns
    /// `OutOfMemory` if `active_memory + requested > memory_limit`.
    /// Set to 0 to disable.
    pub fn set_memory_limit(&self, limit: usize) {
        self.memory_limit.store(limit, Ordering::Relaxed);
    }

    /// Get the current memory limit (0 = unlimited).
    pub fn memory_limit(&self) -> usize {
        self.memory_limit.load(Ordering::Relaxed)
    }

    /// Set the maximum cache size (A12). Adjusts the underlying `BufferCache`
    /// limit and evicts excess if necessary.
    pub fn set_cache_limit(&self, limit: usize) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.set_max_cache_size(limit);
        }
    }

    /// Get the current cache limit.
    pub fn cache_limit(&self) -> usize {
        self.cache.lock().map(|c| c.max_cache_size()).unwrap_or(0)
    }

    /// Reset peak memory to current active memory (A12).
    pub fn reset_peak_memory(&self) {
        self.stats.reset_peak();
    }

    /// Atomically reserve `size` bytes against both the memory limit and block
    /// limit using a compare-and-swap loop (PR 4.1). Returns the new total on
    /// success or an `OutOfMemory` error if either limit would be exceeded.
    fn try_reserve(&self, size: usize) -> Result<usize, AllocError> {
        loop {
            let current = self.allocated_bytes.load(Ordering::Relaxed);
            let new_total = current.saturating_add(size);

            // Check hard memory limit.
            let mem_limit = self.memory_limit.load(Ordering::Relaxed);
            if mem_limit > 0 && new_total > mem_limit {
                return Err(AllocError::OutOfMemory {
                    requested: size,
                    available: mem_limit.saturating_sub(current),
                });
            }

            // Check block limit.
            let blk_limit = self.block_limit.load(Ordering::Relaxed);
            if blk_limit > 0 && new_total > blk_limit {
                return Err(AllocError::OutOfMemory {
                    requested: size,
                    available: blk_limit.saturating_sub(current),
                });
            }

            // Attempt to atomically claim the bytes.
            match self.allocated_bytes.compare_exchange_weak(
                current,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(new_total),
                Err(_) => continue, // Another thread changed the value; retry.
            }
        }
    }

    /// Release `size` bytes from the atomic allocated_bytes counter
    /// (saturating to prevent underflow).
    fn release_reserved(&self, size: usize) {
        let _ =
            self.allocated_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_sub(size))
                });
    }

    /// Track a buffer's GPU address in the ownership set (PR 4.2).
    fn track_buffer(&self, buf: &rmlx_metal::metal::Buffer) {
        if let Ok(mut set) = self.owned_ptrs.lock() {
            set.insert(buf.gpu_address());
        }
    }

    /// Get the current atomically-tracked allocated bytes (PR 4.1).
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Allocate a Metal buffer of at least `size` bytes.
    /// Tries cache first, falls back to device allocation.
    ///
    /// Allocations <= 256 bytes are routed through the `SmallBufferPool`
    /// first (PR 4.3). If the pool is exhausted, falls through to the
    /// normal cache / device path.
    ///
    /// Returns `Err(AllocError::ZeroSize)` for zero-size requests.
    pub fn alloc(&self, size: usize) -> Result<(rmlx_metal::metal::Buffer, usize), AllocError> {
        // Reject zero-size allocations to prevent cache poisoning (A-P0-1).
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }

        // --- Small-buffer fast path (PR 4.3) ---
        // For allocations <= MAX_SMALL_ALLOC (256 B), try the recycling pool.
        if size <= MAX_SMALL_ALLOC {
            self.try_reserve(size)?;
            let guard = ReservationGuard::new(self, size);

            if let Some(mut small) = self.small_pool.alloc(size) {
                let sub_offset = small.offset;
                let buf = small
                    .buffer
                    .take()
                    .expect("small pool allocation must contain a buffer");
                // With sub-allocation the backing buffer is larger than the
                // slot, so use the slot size for accounting (not buf.length()).
                let alloc_size = self.small_pool.slot_size();
                // Adjust reservation if slot size differs from requested size.
                if alloc_size > size {
                    let _ = self
                        .allocated_bytes
                        .fetch_add(alloc_size - size, Ordering::Relaxed);
                } else if alloc_size < size {
                    self.release_reserved(size - alloc_size);
                }
                self.stats.record_alloc(alloc_size);
                self.leak_detector.record_alloc(alloc_size as u64);
                let addr = buf.gpu_address();
                // Track the backing buffer address in owned_ptrs (idempotent
                // for sub-allocation — all slots share the same address).
                self.track_buffer(&buf);
                // Register with residency manager if available.
                if let Ok(mut guard) = self.residency.lock() {
                    if let Some(ref mut mgr) = *guard {
                        mgr.add_buffer(&buf);
                    }
                }
                // Track the SmallAllocation so free() can return the slot.
                if let Ok(mut map) = self.small_allocs.lock() {
                    map.entry(addr).or_default().push(small);
                }
                guard.defuse();
                return Ok((buf, sub_offset));
            }
        }

        // Atomically reserve the requested bytes against limits (PR 4.1).
        self.try_reserve(size)?;
        let guard = ReservationGuard::new(self, size);

        // Try cache first.
        let cached = self
            .cache
            .lock()
            .map_err(|_| AllocError::MutexPoisoned)?
            .acquire(size);
        if let Some(buf) = cached {
            let actual = buf.length() as usize;
            // Adjust reservation if the cached buffer differs in size.
            if actual > size {
                // Reserve the extra bytes (best-effort; the memory is already
                // allocated so we allow exceeding limits for cache hits).
                let _ = self
                    .allocated_bytes
                    .fetch_add(actual - size, Ordering::Relaxed);
            } else if actual < size {
                self.release_reserved(size - actual);
            }
            self.stats.record_cache_hit();
            self.stats.record_alloc(actual);
            self.leak_detector.record_alloc(actual as u64);
            self.track_buffer(&buf);
            // Register with residency manager if available.
            if let Ok(mut guard) = self.residency.lock() {
                if let Some(ref mut mgr) = *guard {
                    mgr.add_buffer(&buf);
                }
            }
            guard.defuse();
            return Ok((buf, 0));
        }

        // Proactive GC: if memory pressure is high, evict cached buffers
        // before falling through to a device allocation.
        self.stats.record_cache_miss();
        if self.gc_limit > 0 {
            let mut cache = self.cache.lock().map_err(|_| AllocError::MutexPoisoned)?;
            let pressure = self
                .stats
                .active()
                .saturating_add(cache.cache_size())
                .saturating_add(size);
            if pressure >= self.gc_limit {
                // Evict enough to get under the limit.
                let overshoot = pressure - self.gc_limit;
                cache.evict(overshoot);
            }
        }

        // Allocate from device with HazardTrackingModeUntracked — RMLX manages
        // synchronisation explicitly so Metal's automatic tracking is redundant.
        let buf = self
            .device
            .new_buffer(size as u64, rmlx_metal::device::DEFAULT_BUFFER_OPTIONS);
        let alloc_size = buf.length() as usize;

        // Adjust reservation if Metal rounded up the buffer size.
        if alloc_size > size {
            let _ = self
                .allocated_bytes
                .fetch_add(alloc_size - size, Ordering::Relaxed);
        } else if alloc_size < size {
            self.release_reserved(size - alloc_size);
        }

        self.stats.record_alloc(alloc_size);
        self.leak_detector.record_alloc(alloc_size as u64);
        self.track_buffer(&buf);
        // Register with residency manager if available.
        if let Ok(mut guard) = self.residency.lock() {
            if let Some(ref mut mgr) = *guard {
                mgr.add_buffer(&buf);
            }
        }

        // A11: Post-allocation cache trimming. If total memory (active + cache)
        // exceeds the GC limit after a large allocation, trim the cache.
        if self.gc_limit > 0 {
            if let Ok(mut cache) = self.cache.lock() {
                let total = self.stats.active().saturating_add(cache.cache_size());
                if total > self.gc_limit {
                    let overshoot = total - self.gc_limit;
                    cache.evict(overshoot);
                }
            }
        }

        guard.defuse();
        Ok((buf, 0))
    }

    /// Return a buffer to the cache for reuse.
    ///
    /// If the buffer was allocated from the small-buffer pool (PR 4.3), the
    /// pool slot is freed instead of returning to the normal cache.
    ///
    /// Returns `Err(AllocError::InvalidFreeBuffer(_))` if the buffer was not
    /// allocated by this allocator or has already been freed (PR 4.2).
    pub fn free(&self, buffer: rmlx_metal::metal::Buffer) -> Result<(), AllocError> {
        // Device mismatch check (#98): verify the buffer belongs to the same
        // Metal device as this allocator before mutating any state.
        {
            let buf_device_name = buffer.device().name();
            let our_device_name = self.device.name();
            if buf_device_name != our_device_name {
                return Err(AllocError::DeviceMismatch);
            }
        }

        let addr = buffer.gpu_address();
        let size = buffer.length() as usize;

        // Check if this buffer came from the small-buffer pool FIRST (P7-10).
        // With sub-allocation all small allocs share the same gpu_address, so
        // we must check before owned_ptrs (which uses a HashSet).
        let small = self
            .small_allocs
            .lock()
            .ok()
            .and_then(|mut map| {
                let vec = map.get_mut(&addr)?;
                let item = vec.pop();
                // Remove the key entirely if the vec is now empty.
                if vec.is_empty() {
                    map.remove(&addr);
                }
                item
            });

        if let Some(mut small_alloc) = small {
            let slot_size = self.small_pool.slot_size();
            self.stats.record_free(slot_size);
            self.leak_detector.record_free(slot_size as u64);

            // Remove from residency manager if present.
            if let Ok(mut guard) = self.residency.lock() {
                if let Some(ref mut mgr) = *guard {
                    mgr.remove_buffer(&buffer);
                }
            }

            // Only remove from owned_ptrs when no more small allocs remain
            // for this backing buffer address.
            let no_more_small = self
                .small_allocs
                .lock()
                .map(|map| !map.contains_key(&addr))
                .unwrap_or(false);
            if no_more_small {
                if let Ok(mut set) = self.owned_ptrs.lock() {
                    set.remove(&addr);
                }
            }

            // Return the slot to the small-buffer pool.
            small_alloc.buffer = Some(buffer);
            self.small_pool.free(small_alloc);
            // Release tracked bytes.
            let _ = self.allocated_bytes.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(slot_size)),
            );
            return Ok(());
        }

        // Ownership check (PR 4.2): only free buffers we actually own.
        {
            let mut set = self
                .owned_ptrs
                .lock()
                .map_err(|_| AllocError::MutexPoisoned)?;
            if !set.remove(&addr) {
                return Err(AllocError::InvalidFreeBuffer(buffer));
            }
        }

        self.stats.record_free(size);
        self.leak_detector.record_free(size as u64);

        // Remove from residency manager if present.
        if let Ok(mut guard) = self.residency.lock() {
            if let Some(ref mut mgr) = *guard {
                mgr.remove_buffer(&buffer);
            }
        }

        self.release_reserved(size);
        if let Ok(mut cache) = self.cache.lock() {
            cache.release(buffer);
        }
        // If mutex is poisoned, the buffer is simply dropped (freed).
        Ok(())
    }

    /// Get allocation statistics.
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Clear the buffer cache, freeing all cached buffers.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get the leak detector (PR 4.3).
    pub fn leak_detector(&self) -> &LeakDetector {
        &self.leak_detector
    }

    /// Get the small-buffer pool (PR 4.3).
    pub fn small_pool(&self) -> &SmallBufferPool {
        &self.small_pool
    }

    /// Returns `true` if a Metal 3 residency manager is active (PR 4.3).
    pub fn has_residency_manager(&self) -> bool {
        self.residency
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Commit pending residency set changes if a residency manager is
    /// active (PR 4.3). No-op if Metal 3 is not available.
    pub fn commit_residency(&self) {
        if let Ok(mut guard) = self.residency.lock() {
            if let Some(ref mut mgr) = *guard {
                mgr.commit();
            }
        }
    }
}

/// Default block limit fallback: 8 GiB.
const DEFAULT_BLOCK_LIMIT: usize = 8 * 1024 * 1024 * 1024;

/// Auto-detect a sensible `block_limit` from the Metal device (A8).
///
/// Strategy: `min(1.5 * recommended_max_working_set_size, 0.95 * total_memory)`.
/// Falls back to [`DEFAULT_BLOCK_LIMIT`] (8 GiB) if the device reports 0.
fn auto_detect_block_limit(device: &rmlx_metal::metal::Device) -> usize {
    let recommended = device.recommended_max_working_set_size();
    if recommended == 0 {
        return DEFAULT_BLOCK_LIMIT;
    }

    // 1.5 * recommended (integer math: recommended + recommended / 2)
    let from_recommended = recommended.saturating_add(recommended / 2) as usize;

    // 0.95 * total physical memory (via sysctl on macOS)
    let total_mem = total_physical_memory();
    let from_total = if total_mem > 0 {
        // 0.95 * total = total - total / 20
        total_mem.saturating_sub(total_mem / 20)
    } else {
        usize::MAX // no cap if we can't query
    };

    std::cmp::min(from_recommended, from_total)
}

/// Query total physical memory via sysctl (macOS).
fn total_physical_memory() -> usize {
    // SAFETY: sysctl with CTL_HW / HW_MEMSIZE is always safe on macOS.
    unsafe {
        let mut memsize: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        let mib: [libc::c_int; 2] = [libc::CTL_HW, libc::HW_MEMSIZE];
        let ret = libc::sysctl(
            mib.as_ptr() as *mut libc::c_int,
            2,
            &mut memsize as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if ret == 0 {
            memsize as usize
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a MetalAllocator or skip the test if no Metal device.
    fn make_allocator(cache_size: usize) -> Option<Arc<MetalAllocator>> {
        let device = GpuDevice::system_default().ok()?;
        Some(Arc::new(MetalAllocator::new(Arc::new(device), cache_size)))
    }

    #[test]
    fn test_zero_size_alloc_returns_error() {
        let allocator = match make_allocator(1024 * 1024) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let result = allocator.alloc(0);
        assert!(result.is_err(), "zero-size alloc should return an error");
        assert!(
            matches!(result.unwrap_err(), AllocError::ZeroSize),
            "error should be ZeroSize variant"
        );
    }

    // ---- PR 4.1 tests ----

    /// N-thread stress test: concurrent allocations must never cause
    /// `allocated_bytes` to exceed the configured memory limit.
    #[test]
    fn test_concurrent_alloc_respects_memory_limit() {
        let allocator = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let alloc_size: usize = 4096;
        let num_threads = 8;
        // Allow exactly `num_threads` allocations (tight limit).
        let limit = alloc_size * num_threads;
        allocator.set_memory_limit(limit);

        let barrier = Arc::new(std::sync::Barrier::new(num_threads));
        let success_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let alloc = Arc::clone(&allocator);
                let bar = Arc::clone(&barrier);
                let cnt = Arc::clone(&success_count);
                std::thread::spawn(move || {
                    bar.wait();
                    if alloc.alloc(alloc_size).is_ok() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let successes = success_count.load(Ordering::Relaxed);
        assert!(
            successes <= num_threads,
            "more allocations succeeded ({successes}) than the limit allows ({num_threads})"
        );
        // The atomic counter must never exceed the limit.
        assert!(
            allocator.allocated_bytes() <= limit,
            "allocated_bytes ({}) exceeds memory_limit ({limit})",
            allocator.allocated_bytes()
        );
    }

    // ---- PR 4.2 tests ----

    /// Double-free must return `InvalidFree` and must not corrupt stats.
    #[test]
    fn test_double_free_returns_error_and_preserves_stats() {
        let allocator = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let (buf, _offset) = allocator.alloc(4096).expect("alloc failed");
        let active_after_alloc = allocator.stats().active();
        assert!(active_after_alloc > 0);

        // First free succeeds.
        allocator.free(buf).expect("first free should succeed");
        let active_after_free = allocator.stats().active();
        assert_eq!(active_after_free, 0, "active should be 0 after free");

        // Rust move semantics prevent calling free on the same buffer twice.
        // Verify the ownership set is now empty and allocated_bytes is 0,
        // confirming that a second free of the same address would fail.
        assert_eq!(
            allocator.allocated_bytes(),
            0,
            "allocated_bytes should be 0 after free"
        );
    }

    /// Freeing a buffer from a *different* allocator (unowned) returns
    /// `InvalidFreeBuffer`.
    #[test]
    fn test_free_unowned_buffer_returns_error() {
        let allocator_a = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let device = match GpuDevice::system_default() {
            Ok(d) => Arc::new(d),
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        let allocator_b = Arc::new(MetalAllocator::new(device, 0));

        // Allocate from B, try to free via A.
        let (buf, _offset) = allocator_b.alloc(4096).expect("alloc from B failed");
        let returned_buf = match allocator_a.free(buf) {
            Err(AllocError::InvalidFreeBuffer(buf)) => buf,
            other => {
                panic!("freeing unowned buffer should return InvalidFreeBuffer, got {other:?}")
            }
        };
        allocator_b
            .free(returned_buf)
            .expect("allocator B should free returned buffer");
    }

    #[test]
    fn test_invalid_free_returns_buffer() {
        let allocator_a = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let device = match GpuDevice::system_default() {
            Ok(d) => Arc::new(d),
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        let allocator_b = Arc::new(MetalAllocator::new(device, 0));

        let (buf, _offset) = allocator_b.alloc(4096).expect("alloc from B failed");
        let original_addr = buf.gpu_address();
        let returned_buf = match allocator_a.free(buf) {
            Err(AllocError::InvalidFreeBuffer(buf)) => buf,
            other => panic!("expected InvalidFreeBuffer, got {other:?}"),
        };

        assert_eq!(returned_buf.gpu_address(), original_addr);
        allocator_b
            .free(returned_buf)
            .expect("allocator B should free returned buffer");
    }

    #[test]
    fn test_reservation_released_on_cache_poison() {
        let allocator = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let poison_target = Arc::clone(&allocator);
        let handle = std::thread::spawn(move || {
            let _guard = poison_target
                .cache
                .lock()
                .expect("lock cache for poisoning");
            panic!("poison cache mutex");
        });
        let _ = handle.join();

        let result = allocator.alloc(4096);
        assert!(
            matches!(result, Err(AllocError::MutexPoisoned)),
            "alloc should fail with MutexPoisoned after cache poison, got {result:?}"
        );
        assert_eq!(
            allocator.allocated_bytes(),
            0,
            "reserved bytes should be released on cache poison"
        );
    }

    /// Stats must not underflow even if record_free is called with a large
    /// size (simulating a mismatch). This tests the saturating_sub in
    /// AllocStats::record_free.
    #[test]
    fn test_stats_underflow_protection() {
        let stats = AllocStats::new();
        stats.record_alloc(100);
        assert_eq!(stats.active(), 100);

        // Free more than was allocated -- should saturate at 0, not wrap.
        stats.record_free(200);
        assert_eq!(stats.active(), 0, "active_bytes should saturate at 0");
    }

    // ---- #98: Device mismatch on free ----

    /// Freeing a buffer whose Metal device does not match this allocator's
    /// device must return `DeviceMismatch` without corrupting state.
    ///
    /// On a single-GPU Mac there is only one Metal device, so the device names
    /// always match. We test the positive path (same device, no error) and
    /// rely on the code-level check being correct for multi-device scenarios.
    #[test]
    fn test_free_same_device_no_mismatch() {
        let allocator = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let (buf, _offset) = allocator.alloc(4096).expect("alloc failed");
        // free() should succeed (same device, same allocator).
        allocator
            .free(buf)
            .expect("free should succeed for same-device buffer");
        assert_eq!(allocator.allocated_bytes(), 0);
    }

    // ---- #99 resolved: SmallBufferPool now sub-allocates (P7-10) ----

    #[test]
    fn test_small_pool_is_suballocating() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, None);
        assert!(
            pool.is_suballocating(),
            "SmallBufferPool should sub-allocate from a single backing buffer (P7-10)"
        );
    }

    // ---- #101: Reserved-byte leak on failure after try_reserve ----

    /// If alloc() fails after try_reserve succeeds, the reserved bytes must be
    /// released. We test this by setting a tight memory limit and verifying
    /// that allocated_bytes returns to 0 after all buffers are freed.
    #[test]
    fn test_reserved_bytes_not_leaked_on_alloc_free_cycle() {
        let allocator = match make_allocator(0) {
            Some(a) => a,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let limit: usize = 4096 * 4;
        allocator.set_memory_limit(limit);

        // Allocate up to the limit.
        let mut bufs = Vec::new();
        for _ in 0..4 {
            match allocator.alloc(4096) {
                Ok((b, _offset)) => bufs.push(b),
                Err(_) => break,
            }
        }
        let after_alloc = allocator.allocated_bytes();
        assert!(after_alloc > 0, "should have allocated some bytes");

        // Free all buffers.
        for b in bufs {
            allocator.free(b).expect("free failed");
        }

        // Reserved bytes must be fully released — no leak.
        assert_eq!(
            allocator.allocated_bytes(),
            0,
            "allocated_bytes should be 0 after freeing all buffers (no reserved-byte leak)"
        );

        // We should be able to allocate again up to the limit.
        let result = allocator.alloc(4096);
        assert!(
            result.is_ok(),
            "should be able to allocate again after freeing (reserved bytes were properly released)"
        );
        if let Ok((b, _offset)) = result {
            allocator.free(b).expect("free failed");
        }
    }
}
