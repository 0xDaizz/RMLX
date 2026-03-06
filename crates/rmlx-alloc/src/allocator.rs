use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal::MTLResourceOptions;

use crate::cache::BufferCache;
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
        Self {
            device,
            cache: Mutex::new(BufferCache::new(max_cache_size)),
            stats: AllocStats::new(),
            block_limit: AtomicUsize::new(block_limit),
            gc_limit: DEFAULT_GC_LIMIT,
            memory_limit: AtomicUsize::new(0),
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

    /// Allocate a Metal buffer of at least `size` bytes.
    /// Tries cache first, falls back to device allocation.
    ///
    /// Returns `Err(AllocError::ZeroSize)` for zero-size requests.
    pub fn alloc(&self, size: usize) -> Result<rmlx_metal::metal::Buffer, AllocError> {
        // Reject zero-size allocations to prevent cache poisoning (A-P0-1).
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }

        // Check hard memory limit (A12).
        let mem_limit = self.memory_limit.load(Ordering::Relaxed);
        if mem_limit > 0 && self.stats.active() + size > mem_limit {
            return Err(AllocError::OutOfMemory {
                requested: size,
                available: mem_limit.saturating_sub(self.stats.active()),
            });
        }

        let limit = self.block_limit.load(Ordering::Relaxed);
        if limit > 0 && self.stats.active() + size > limit {
            return Err(AllocError::OutOfMemory {
                requested: size,
                available: limit.saturating_sub(self.stats.active()),
            });
        }

        // Try cache first
        let cached = self
            .cache
            .lock()
            .map_err(|_| AllocError::MutexPoisoned)?
            .acquire(size);
        if let Some(buf) = cached {
            self.stats.record_cache_hit();
            self.stats.record_alloc(buf.length() as usize);
            return Ok(buf);
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

        // Allocate from device
        let buf = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        let alloc_size = buf.length() as usize;
        self.stats.record_alloc(alloc_size);

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

        Ok(buf)
    }

    /// Return a buffer to the cache for reuse.
    pub fn free(&self, buffer: rmlx_metal::metal::Buffer) {
        let size = buffer.length() as usize;
        self.stats.record_free(size);
        if let Ok(mut cache) = self.cache.lock() {
            cache.release(buffer);
        }
        // If mutex is poisoned, the buffer is simply dropped (freed).
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

    #[test]
    fn test_zero_size_alloc_returns_error() {
        let device = match GpuDevice::system_default() {
            Ok(d) => Arc::new(d),
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let allocator = MetalAllocator::new(device, 1024 * 1024);
        let result = allocator.alloc(0);
        assert!(result.is_err(), "zero-size alloc should return an error");
        assert!(
            matches!(result.unwrap_err(), AllocError::ZeroSize),
            "error should be ZeroSize variant"
        );
    }
}
