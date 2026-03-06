//! Lock-free atomic runtime metrics for RMLX.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for runtime performance metrics.
pub struct RuntimeMetrics {
    pub kernel_dispatches: AtomicU64,
    pub kernel_total_time_us: AtomicU64,
    pub buffer_allocs: AtomicU64,
    pub buffer_frees: AtomicU64,
    pub buffer_bytes_allocated: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl RuntimeMetrics {
    pub fn new() -> Self {
        Self {
            kernel_dispatches: AtomicU64::new(0),
            kernel_total_time_us: AtomicU64::new(0),
            buffer_allocs: AtomicU64::new(0),
            buffer_frees: AtomicU64::new(0),
            buffer_bytes_allocated: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Record a kernel dispatch with its duration in microseconds.
    pub fn record_kernel_dispatch(&self, duration_us: u64) {
        self.kernel_dispatches.fetch_add(1, Ordering::Relaxed);
        self.kernel_total_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Record a buffer allocation of `bytes` size.
    pub fn record_buffer_alloc(&self, bytes: u64) {
        self.buffer_allocs.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes_allocated
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a buffer free of `bytes` size.
    pub fn record_buffer_free(&self, bytes: u64) {
        self.buffer_frees.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes_allocated
            .fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a consistent snapshot of all counters.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            kernel_dispatches: self.kernel_dispatches.load(Ordering::Relaxed),
            kernel_total_time_us: self.kernel_total_time_us.load(Ordering::Relaxed),
            buffer_allocs: self.buffer_allocs.load(Ordering::Relaxed),
            buffer_frees: self.buffer_frees.load(Ordering::Relaxed),
            buffer_bytes_allocated: self.buffer_bytes_allocated.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        }
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of runtime metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub kernel_dispatches: u64,
    pub kernel_total_time_us: u64,
    pub buffer_allocs: u64,
    pub buffer_frees: u64,
    pub buffer_bytes_allocated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_metrics_zeroed() {
        let m = RuntimeMetrics::new();
        let s = m.snapshot();
        assert_eq!(s.kernel_dispatches, 0);
        assert_eq!(s.kernel_total_time_us, 0);
        assert_eq!(s.buffer_allocs, 0);
        assert_eq!(s.buffer_frees, 0);
        assert_eq!(s.buffer_bytes_allocated, 0);
        assert_eq!(s.cache_hits, 0);
        assert_eq!(s.cache_misses, 0);
    }

    #[test]
    fn test_default_is_zeroed() {
        let m = RuntimeMetrics::default();
        let s = m.snapshot();
        assert_eq!(s.kernel_dispatches, 0);
    }

    #[test]
    fn test_record_kernel_dispatch() {
        let m = RuntimeMetrics::new();
        m.record_kernel_dispatch(100);
        m.record_kernel_dispatch(250);

        let s = m.snapshot();
        assert_eq!(s.kernel_dispatches, 2);
        assert_eq!(s.kernel_total_time_us, 350);
    }

    #[test]
    fn test_record_buffer_alloc() {
        let m = RuntimeMetrics::new();
        m.record_buffer_alloc(1024);
        m.record_buffer_alloc(2048);

        let s = m.snapshot();
        assert_eq!(s.buffer_allocs, 2);
        assert_eq!(s.buffer_bytes_allocated, 3072);
    }

    #[test]
    fn test_record_buffer_free() {
        let m = RuntimeMetrics::new();
        m.record_buffer_alloc(4096);
        m.record_buffer_free(1024);

        let s = m.snapshot();
        assert_eq!(s.buffer_allocs, 1);
        assert_eq!(s.buffer_frees, 1);
        assert_eq!(s.buffer_bytes_allocated, 3072);
    }

    #[test]
    fn test_cache_hit_miss() {
        let m = RuntimeMetrics::new();
        m.record_cache_hit();
        m.record_cache_hit();
        m.record_cache_miss();
        m.record_cache_hit();

        let s = m.snapshot();
        assert_eq!(s.cache_hits, 3);
        assert_eq!(s.cache_misses, 1);
    }

    #[test]
    fn test_snapshot_is_independent() {
        let m = RuntimeMetrics::new();
        m.record_kernel_dispatch(10);
        let s1 = m.snapshot();

        m.record_kernel_dispatch(20);
        let s2 = m.snapshot();

        // s1 should not be affected by later mutations.
        assert_eq!(s1.kernel_dispatches, 1);
        assert_eq!(s2.kernel_dispatches, 2);
        assert_eq!(s1.kernel_total_time_us, 10);
        assert_eq!(s2.kernel_total_time_us, 30);
    }

    #[test]
    fn test_alloc_free_cycle() {
        let m = RuntimeMetrics::new();
        for i in 0..10 {
            m.record_buffer_alloc((i + 1) * 100);
        }
        for i in 0..10 {
            m.record_buffer_free((i + 1) * 100);
        }
        let s = m.snapshot();
        assert_eq!(s.buffer_allocs, 10);
        assert_eq!(s.buffer_frees, 10);
        assert_eq!(s.buffer_bytes_allocated, 0);
    }
}
