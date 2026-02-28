use std::sync::Arc;
use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal::MTLResourceOptions;

use crate::cache::BufferCache;
use crate::stats::AllocStats;
use crate::AllocError;

/// Metal buffer allocator with size-binned caching.
///
/// Follows the MLX MetalAllocator pattern: allocate from device,
/// cache freed buffers for reuse, track peak memory usage.
pub struct MetalAllocator {
    device: Arc<GpuDevice>,
    cache: Mutex<BufferCache>,
    stats: AllocStats,
    /// Maximum total allocation (0 = unlimited)
    block_limit: usize,
}

impl MetalAllocator {
    /// Create a new allocator. `max_cache_size` controls buffer cache capacity.
    pub fn new(device: Arc<GpuDevice>, max_cache_size: usize) -> Self {
        Self {
            device,
            cache: Mutex::new(BufferCache::new(max_cache_size)),
            stats: AllocStats::new(),
            block_limit: 0,
        }
    }

    /// Set maximum total allocation limit (0 = unlimited).
    pub fn set_block_limit(&mut self, limit: usize) {
        self.block_limit = limit;
    }

    /// Allocate a Metal buffer of at least `size` bytes.
    /// Tries cache first, falls back to device allocation.
    pub fn alloc(&self, size: usize) -> Result<rmlx_metal::metal::Buffer, AllocError> {
        if self.block_limit > 0 && self.stats.active() + size > self.block_limit {
            return Err(AllocError::OutOfMemory {
                requested: size,
                available: self.block_limit.saturating_sub(self.stats.active()),
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

        // Allocate from device
        self.stats.record_cache_miss();
        let buf = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        self.stats.record_alloc(buf.length() as usize);
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
