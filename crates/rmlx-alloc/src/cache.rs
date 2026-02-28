use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

use rmlx_metal::metal::Buffer as MetalBuffer;

use crate::zero_copy::page_size;

/// Size-binned buffer cache (MLX BufferCache pattern).
///
/// Caches freed Metal buffers by size to avoid repeated allocation overhead.
/// Sizes are rounded up to the nearest page boundary.
pub struct BufferCache {
    bins: BTreeMap<usize, VecDeque<MetalBuffer>>,
    cache_size: AtomicUsize,
    max_cache_size: usize,
}

impl BufferCache {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            bins: BTreeMap::new(),
            cache_size: AtomicUsize::new(0),
            max_cache_size,
        }
    }

    /// Try to get a cached buffer of at least `size` bytes.
    /// Returns the smallest cached buffer that fits.
    pub fn acquire(&mut self, size: usize) -> Option<MetalBuffer> {
        // Look for exact match first, then smallest larger buffer
        let key = align_to_page(size);
        if let Some(deque) = self.bins.get_mut(&key) {
            if let Some(buf) = deque.pop_front() {
                self.cache_size.fetch_sub(key, Ordering::Relaxed);
                return Some(buf);
            }
        }
        // Try next larger bin
        let next_key = self
            .bins
            .range(key..)
            .find_map(|(k, d)| if !d.is_empty() { Some(*k) } else { None });
        if let Some(k) = next_key {
            let buf = self.bins.get_mut(&k).unwrap().pop_front().unwrap();
            self.cache_size.fetch_sub(k, Ordering::Relaxed);
            return Some(buf);
        }
        None
    }

    /// Return a buffer to the cache for future reuse.
    pub fn release(&mut self, buffer: MetalBuffer) {
        let size = buffer.length() as usize;
        let key = align_to_page(size);

        // Don't cache buffers that exceed the entire cache capacity
        if self.max_cache_size == 0 || key > self.max_cache_size {
            drop(buffer);
            return;
        }

        // Evict if cache is full
        if self.cache_size.load(Ordering::Relaxed) + key > self.max_cache_size {
            self.evict(key);
        }

        self.cache_size.fetch_add(key, Ordering::Relaxed);
        self.bins.entry(key).or_default().push_back(buffer);
    }

    /// Current cache size in bytes.
    pub fn cache_size(&self) -> usize {
        self.cache_size.load(Ordering::Relaxed)
    }

    /// Clear all cached buffers.
    pub fn clear(&mut self) {
        self.bins.clear();
        self.cache_size.store(0, Ordering::Relaxed);
    }

    /// Evict oldest buffers until `needed` bytes are free.
    fn evict(&mut self, needed: usize) {
        let target = self.max_cache_size.saturating_sub(needed);
        while self.cache_size.load(Ordering::Relaxed) > target {
            // Evict from largest bin first (frees most memory per eviction)
            let largest = self.bins.keys().next_back().copied();
            if let Some(key) = largest {
                if let Some(deque) = self.bins.get_mut(&key) {
                    if deque.pop_front().is_some() {
                        self.cache_size.fetch_sub(key, Ordering::Relaxed);
                    }
                    if deque.is_empty() {
                        self.bins.remove(&key);
                    }
                }
            } else {
                break;
            }
        }
    }
}

fn align_to_page(size: usize) -> usize {
    let page = page_size();
    (size + page - 1) & !(page - 1)
}
