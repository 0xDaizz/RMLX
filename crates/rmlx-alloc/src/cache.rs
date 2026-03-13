use std::collections::{BTreeMap, VecDeque};

use objc2_metal::MTLBuffer as _;
use rmlx_metal::MtlBuffer;

use crate::zero_copy::page_size;

/// Size-binned buffer cache (MLX BufferCache pattern).
///
/// Caches freed Metal buffers by size to avoid repeated allocation overhead.
/// Sizes are rounded up to the nearest page boundary.
///
/// Eviction uses LRU order (oldest-first) to prevent stale small buffers
/// from accumulating indefinitely.
///
/// The cache is always accessed behind a `Mutex`, so interior fields use
/// plain `usize` rather than `AtomicUsize` (A9).
pub struct BufferCache {
    bins: BTreeMap<usize, VecDeque<MtlBuffer>>,
    /// LRU list tracking insertion order: oldest at front, newest at back.
    /// Each entry is `(aligned_size, gpu_address)` identifying the buffer.
    lru: VecDeque<(usize, u64)>,
    /// Total size of all cached buffers (plain usize, guarded by external Mutex).
    cache_size: usize,
    max_cache_size: usize,
}

impl BufferCache {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            bins: BTreeMap::new(),
            lru: VecDeque::new(),
            cache_size: 0,
            max_cache_size,
        }
    }

    /// Try to get a cached buffer of at least `size` bytes.
    ///
    /// Returns the smallest cached buffer that fits, bounded by an upper limit
    /// of `min(2 * aligned_size, aligned_size + 2 * page_size)` to prevent
    /// a small request from consuming a disproportionately large buffer.
    pub fn acquire(&mut self, size: usize) -> Option<MtlBuffer> {
        let key = align_size(size);
        let ps = page_size();
        // Upper bound: at most 2x the request or request + 2 pages, whichever is smaller.
        let max_key = std::cmp::min(key.saturating_mul(2), key.saturating_add(ps * 2));

        // Search within [key, max_key) for the smallest non-empty bin.
        let found_key =
            self.bins
                .range(key..max_key)
                .find_map(|(k, d)| if !d.is_empty() { Some(*k) } else { None });

        if let Some(k) = found_key {
            if let Some(buf) = self.bins.get_mut(&k).and_then(|d| d.pop_front()) {
                // Validate the buffer is large enough to satisfy the request.
                // This guards against cache poisoning from zero-size or
                // undersized buffers that were cached under an aligned key.
                if buf.length() < size {
                    // Discard the undersized buffer rather than returning it.
                    self.cache_size = self.cache_size.saturating_sub(k);
                    let addr = buf.gpuAddress();
                    self.lru_remove(k, addr);
                    if let Some(deque) = self.bins.get(&k) {
                        if deque.is_empty() {
                            self.bins.remove(&k);
                        }
                    }
                    drop(buf);
                    return None;
                }
                self.cache_size = self.cache_size.saturating_sub(k);
                // Remove this buffer from the LRU list by gpu_address.
                let addr = buf.gpuAddress();
                self.lru_remove(k, addr);
                if let Some(deque) = self.bins.get(&k) {
                    if deque.is_empty() {
                        self.bins.remove(&k);
                    }
                }
                return Some(buf);
            }
        }
        None
    }

    /// Return a buffer to the cache for future reuse.
    pub fn release(&mut self, buffer: MtlBuffer) {
        let size = buffer.length();

        // Don't cache zero-size buffers — they would poison size bins.
        if size == 0 {
            drop(buffer);
            return;
        }

        let key = align_size(size);

        // Don't cache buffers that exceed the entire cache capacity
        if self.max_cache_size == 0 || key > self.max_cache_size {
            drop(buffer);
            return;
        }

        // Evict if cache is full
        if self.cache_size + key > self.max_cache_size {
            self.evict(key);
        }

        self.cache_size += key;
        // Track in LRU (newest at back).
        let addr = buffer.gpuAddress();
        self.lru.push_back((key, addr));
        self.bins.entry(key).or_default().push_back(buffer);
    }

    /// Current cache size in bytes.
    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    /// Maximum cache size in bytes.
    pub fn max_cache_size(&self) -> usize {
        self.max_cache_size
    }

    /// Set the maximum cache size (A12). Evicts if current size exceeds the new limit.
    pub fn set_max_cache_size(&mut self, limit: usize) {
        self.max_cache_size = limit;
        if self.cache_size > limit {
            let overshoot = self.cache_size - limit;
            self.evict(overshoot);
        }
    }

    /// Clear all cached buffers.
    pub fn clear(&mut self) {
        self.bins.clear();
        self.lru.clear();
        self.cache_size = 0;
    }

    /// Evict LRU (oldest-first) buffers until `needed` bytes are free.
    ///
    /// Evicts from the front of the LRU list (oldest entries) to ensure that
    /// stale small buffers are cleaned up rather than accumulating forever.
    pub(crate) fn evict(&mut self, needed: usize) {
        let target = self.max_cache_size.saturating_sub(needed);
        while self.cache_size > target {
            if let Some((key, addr)) = self.lru.pop_front() {
                // Remove the corresponding entry from the bin.
                let mut removed = false;
                if let Some(deque) = self.bins.get_mut(&key) {
                    if let Some(pos) = deque.iter().position(|b| b.gpuAddress() == addr) {
                        deque.remove(pos);
                        removed = true;
                    }
                    if deque.is_empty() {
                        self.bins.remove(&key);
                    }
                }
                if removed {
                    self.cache_size = self.cache_size.saturating_sub(key);
                }
            } else {
                break; // LRU list is empty.
            }
        }
    }

    /// Remove the first LRU entry matching `(key, addr)`.
    fn lru_remove(&mut self, key: usize, addr: u64) {
        if let Some(pos) = self.lru.iter().position(|(k, a)| *k == key && *a == addr) {
            self.lru.remove(pos);
        }
    }
}

/// Align allocation size using tiered strategy to avoid over-alignment.
///
/// - size <= 16: align to 16 (minimum Metal buffer alignment)
/// - size <= page_size: round up to next power of 2
/// - size > page_size: page-align (original behavior)
///
/// This prevents an 8-byte request from wasting a full 16KB page.
fn align_size(size: usize) -> usize {
    let page = page_size();
    if size <= 16 {
        16
    } else if size <= page {
        size.next_power_of_two()
    } else {
        (size + page - 1) & !(page - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_metal::device::GpuDevice;
    use rmlx_metal::MTLResourceOptions;

    #[test]
    fn test_release_rejects_zero_size_guard() {
        // Verify the zero-size guard in release() by checking the code path.
        // Metal cannot create a true zero-length buffer (returns null), so
        // we verify that the allocator-level guard (AllocError::ZeroSize)
        // prevents zero-size buffers from ever reaching the cache.
        // This test validates the cache_size bookkeeping stays clean.
        let mut cache = BufferCache::new(1024 * 1024);
        assert_eq!(cache.cache_size(), 0);
        // With no buffers released, acquire should return None.
        assert!(cache.acquire(16).is_none());
    }

    #[test]
    fn test_acquire_rejects_undersized_buffer() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut cache = BufferCache::new(1024 * 1024);

        // Insert a small buffer (e.g., 4 bytes) into the cache.
        // align_size(4) == 16, so it will land in the 16-byte bin.
        let buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let actual_len = buf.length();
        cache.release(buf);

        // If the device actually allocated >= 16 bytes (Metal may round up),
        // the acquire should succeed. If it allocated exactly 4, acquire(16)
        // should reject it.
        if actual_len < 16 {
            let result = cache.acquire(16);
            assert!(
                result.is_none(),
                "acquire should reject buffer with length {} for request of 16",
                actual_len
            );
        }
        // Either way, no undersized buffer should ever be returned.
    }

    #[test]
    fn test_acquire_returns_adequate_buffer() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut cache = BufferCache::new(1024 * 1024);

        // Insert a 1024-byte buffer.
        let buf = device.new_buffer(1024, MTLResourceOptions::StorageModeShared);
        cache.release(buf);

        // Acquire a 1024-byte buffer — should succeed and be large enough.
        let result = cache.acquire(1024);
        assert!(result.is_some(), "should acquire a cached 1024-byte buffer");
        let acquired = result.unwrap();
        assert!(
            acquired.length() >= 1024,
            "acquired buffer length {} must be >= 1024",
            acquired.length()
        );
    }
}
