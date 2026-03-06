use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe allocation statistics tracker.
pub struct AllocStats {
    active_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    total_allocs: AtomicUsize,
    total_frees: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl AllocStats {
    pub fn new() -> Self {
        Self {
            active_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            total_allocs: AtomicUsize::new(0),
            total_frees: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }

    pub fn record_alloc(&self, size: usize) {
        self.total_allocs.fetch_add(1, Ordering::Relaxed);
        let active = self.active_bytes.fetch_add(size, Ordering::Relaxed) + size;
        // Update peak (relaxed CAS loop)
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while active > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                active,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    pub fn record_free(&self, size: usize) {
        self.total_frees.fetch_add(1, Ordering::Relaxed);
        // Use fetch_update + saturating_sub to prevent underflow on
        // double-free or mismatched free (PR 4.2).
        let _ = self
            .active_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size))
            });
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn active(&self) -> usize {
        self.active_bytes.load(Ordering::Relaxed)
    }
    pub fn peak(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }
    pub fn total_allocs(&self) -> usize {
        self.total_allocs.load(Ordering::Relaxed)
    }
    pub fn total_frees(&self) -> usize {
        self.total_frees.load(Ordering::Relaxed)
    }
    pub fn cache_hits(&self) -> usize {
        self.cache_hits.load(Ordering::Relaxed)
    }
    pub fn cache_misses(&self) -> usize {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Reset peak memory to the current active memory (A12).
    ///
    /// Useful for tracking peak memory within a specific phase of execution
    /// without the watermark from earlier phases.
    pub fn reset_peak(&self) {
        let active = self.active_bytes.load(Ordering::Relaxed);
        self.peak_bytes.store(active, Ordering::Relaxed);
    }
}

impl Default for AllocStats {
    fn default() -> Self {
        Self::new()
    }
}
