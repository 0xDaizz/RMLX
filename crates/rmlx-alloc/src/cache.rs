use std::collections::{BTreeMap, VecDeque};

use rmlx_metal::metal::Buffer as MetalBuffer;

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
    bins: BTreeMap<usize, VecDeque<MetalBuffer>>,
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
    pub fn acquire(&mut self, size: usize) -> Option<MetalBuffer> {
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
                self.cache_size = self.cache_size.saturating_sub(k);
                // Remove this buffer from the LRU list by gpu_address.
                let addr = buf.gpu_address();
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
    pub fn release(&mut self, buffer: MetalBuffer) {
        let size = buffer.length() as usize;
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
        let addr = buffer.gpu_address();
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
                    if let Some(pos) = deque.iter().position(|b| b.gpu_address() == addr) {
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
