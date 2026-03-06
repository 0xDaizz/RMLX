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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stats_zeroed() {
        let s = AllocStats::new();
        assert_eq!(s.active(), 0);
        assert_eq!(s.peak(), 0);
        assert_eq!(s.total_allocs(), 0);
        assert_eq!(s.total_frees(), 0);
        assert_eq!(s.cache_hits(), 0);
        assert_eq!(s.cache_misses(), 0);
    }

    #[test]
    fn test_default_is_zeroed() {
        let s = AllocStats::default();
        assert_eq!(s.active(), 0);
        assert_eq!(s.total_allocs(), 0);
    }

    #[test]
    fn test_record_alloc_updates_active_and_count() {
        let s = AllocStats::new();
        s.record_alloc(1024);
        assert_eq!(s.active(), 1024);
        assert_eq!(s.total_allocs(), 1);

        s.record_alloc(2048);
        assert_eq!(s.active(), 3072);
        assert_eq!(s.total_allocs(), 2);
    }

    #[test]
    fn test_record_free_updates_active_and_count() {
        let s = AllocStats::new();
        s.record_alloc(4096);
        s.record_free(1024);
        assert_eq!(s.active(), 3072);
        assert_eq!(s.total_frees(), 1);
    }

    #[test]
    fn test_peak_tracking() {
        let s = AllocStats::new();
        s.record_alloc(1000);
        s.record_alloc(2000);
        assert_eq!(s.peak(), 3000);

        s.record_free(1000);
        // Peak should remain at 3000.
        assert_eq!(s.peak(), 3000);
        assert_eq!(s.active(), 2000);
    }

    #[test]
    fn test_reset_peak() {
        let s = AllocStats::new();
        s.record_alloc(5000);
        s.record_free(3000);
        assert_eq!(s.peak(), 5000);

        s.reset_peak();
        assert_eq!(s.peak(), 2000); // current active
        assert_eq!(s.active(), 2000);
    }

    #[test]
    fn test_cache_hit_miss_counters() {
        let s = AllocStats::new();
        s.record_cache_hit();
        s.record_cache_hit();
        s.record_cache_miss();

        assert_eq!(s.cache_hits(), 2);
        assert_eq!(s.cache_misses(), 1);
    }

    #[test]
    fn test_free_saturates_at_zero() {
        let s = AllocStats::new();
        s.record_alloc(100);
        s.record_free(200); // free more than allocated
        assert_eq!(s.active(), 0); // should not underflow
    }

    #[test]
    fn test_alloc_free_full_cycle() {
        let s = AllocStats::new();
        for _ in 0..100 {
            s.record_alloc(256);
        }
        assert_eq!(s.total_allocs(), 100);
        assert_eq!(s.active(), 25600);

        for _ in 0..100 {
            s.record_free(256);
        }
        assert_eq!(s.total_frees(), 100);
        assert_eq!(s.active(), 0);
        assert_eq!(s.peak(), 25600);
    }
}
