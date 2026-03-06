//! Memory leak detector using atomic counters.

use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks allocation/free counts and bytes to detect potential leaks.
pub struct LeakDetector {
    alloc_count: AtomicU64,
    free_count: AtomicU64,
    alloc_bytes: AtomicU64,
    free_bytes: AtomicU64,
    high_water_mark_bytes: AtomicU64,
}

impl LeakDetector {
    pub fn new() -> Self {
        Self {
            alloc_count: AtomicU64::new(0),
            free_count: AtomicU64::new(0),
            alloc_bytes: AtomicU64::new(0),
            free_bytes: AtomicU64::new(0),
            high_water_mark_bytes: AtomicU64::new(0),
        }
    }

    /// Record an allocation of `bytes` size.
    pub fn record_alloc(&self, bytes: u64) {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.alloc_bytes.fetch_add(bytes, Ordering::Relaxed);
        let current = prev + bytes;
        // Update high water mark (compare-and-swap loop)
        let freed = self.free_bytes.load(Ordering::Relaxed);
        let outstanding = current.saturating_sub(freed);
        loop {
            let hwm = self.high_water_mark_bytes.load(Ordering::Relaxed);
            if outstanding <= hwm {
                break;
            }
            if self
                .high_water_mark_bytes
                .compare_exchange_weak(hwm, outstanding, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Record a free of `bytes` size.
    pub fn record_free(&self, bytes: u64) {
        self.free_count.fetch_add(1, Ordering::Relaxed);
        self.free_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Number of allocations that have not been freed.
    pub fn outstanding_allocs(&self) -> u64 {
        let allocs = self.alloc_count.load(Ordering::Relaxed);
        let frees = self.free_count.load(Ordering::Relaxed);
        allocs.saturating_sub(frees)
    }

    /// Bytes currently allocated (not yet freed).
    pub fn outstanding_bytes(&self) -> u64 {
        let alloc = self.alloc_bytes.load(Ordering::Relaxed);
        let freed = self.free_bytes.load(Ordering::Relaxed);
        alloc.saturating_sub(freed)
    }

    /// Heuristic leak check: outstanding allocs > 0 and outstanding bytes > 1 MiB.
    pub fn has_potential_leak(&self) -> bool {
        self.outstanding_allocs() > 0 && self.outstanding_bytes() > 1024 * 1024
    }

    /// Generate a full leak report.
    pub fn report(&self) -> LeakReport {
        let total_allocs = self.alloc_count.load(Ordering::Relaxed);
        let total_frees = self.free_count.load(Ordering::Relaxed);
        let outstanding_allocs = total_allocs.saturating_sub(total_frees);
        let outstanding_bytes = self.outstanding_bytes();
        let high_water_mark_bytes = self.high_water_mark_bytes.load(Ordering::Relaxed);
        let potential_leak = outstanding_allocs > 0 && outstanding_bytes > 1024 * 1024;
        LeakReport {
            total_allocs,
            total_frees,
            outstanding_allocs,
            outstanding_bytes,
            high_water_mark_bytes,
            potential_leak,
        }
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of leak detector state.
#[derive(Debug, Clone)]
pub struct LeakReport {
    pub total_allocs: u64,
    pub total_frees: u64,
    pub outstanding_allocs: u64,
    pub outstanding_bytes: u64,
    pub high_water_mark_bytes: u64,
    pub potential_leak: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_detector_is_clean() {
        let d = LeakDetector::new();
        assert_eq!(d.outstanding_allocs(), 0);
        assert_eq!(d.outstanding_bytes(), 0);
        assert!(!d.has_potential_leak());
    }

    #[test]
    fn test_default_is_same_as_new() {
        let d = LeakDetector::default();
        assert_eq!(d.outstanding_allocs(), 0);
        assert_eq!(d.outstanding_bytes(), 0);
    }

    #[test]
    fn test_record_alloc_increments_counters() {
        let d = LeakDetector::new();
        d.record_alloc(1024);
        assert_eq!(d.outstanding_allocs(), 1);
        assert_eq!(d.outstanding_bytes(), 1024);
    }

    #[test]
    fn test_record_free_decrements_counters() {
        let d = LeakDetector::new();
        d.record_alloc(2048);
        d.record_free(2048);
        assert_eq!(d.outstanding_allocs(), 0);
        assert_eq!(d.outstanding_bytes(), 0);
    }

    #[test]
    fn test_multiple_alloc_free_cycles() {
        let d = LeakDetector::new();
        d.record_alloc(100);
        d.record_alloc(200);
        d.record_alloc(300);
        assert_eq!(d.outstanding_allocs(), 3);
        assert_eq!(d.outstanding_bytes(), 600);

        d.record_free(100);
        assert_eq!(d.outstanding_allocs(), 2);
        assert_eq!(d.outstanding_bytes(), 500);

        d.record_free(200);
        d.record_free(300);
        assert_eq!(d.outstanding_allocs(), 0);
        assert_eq!(d.outstanding_bytes(), 0);
    }

    #[test]
    fn test_high_water_mark_tracks_peak() {
        let d = LeakDetector::new();
        d.record_alloc(1000);
        d.record_alloc(2000);
        // Peak should be 3000.
        d.record_free(1000);
        // Outstanding is now 2000, but high water mark should still be 3000.
        let report = d.report();
        assert!(report.high_water_mark_bytes >= 3000);
    }

    #[test]
    fn test_has_potential_leak_threshold() {
        let d = LeakDetector::new();
        // Below 1 MiB threshold: no leak.
        d.record_alloc(512 * 1024);
        assert!(!d.has_potential_leak());

        // Above 1 MiB threshold: potential leak.
        d.record_alloc(1024 * 1024);
        assert!(d.has_potential_leak());
    }

    #[test]
    fn test_no_leak_when_all_freed() {
        let d = LeakDetector::new();
        d.record_alloc(2 * 1024 * 1024);
        d.record_free(2 * 1024 * 1024);
        assert!(!d.has_potential_leak());
    }

    #[test]
    fn test_report_fields() {
        let d = LeakDetector::new();
        d.record_alloc(4096);
        d.record_alloc(8192);
        d.record_free(4096);

        let r = d.report();
        assert_eq!(r.total_allocs, 2);
        assert_eq!(r.total_frees, 1);
        assert_eq!(r.outstanding_allocs, 1);
        assert_eq!(r.outstanding_bytes, 8192);
        assert!(!r.potential_leak); // 8192 < 1 MiB
    }

    #[test]
    fn test_report_potential_leak_flag() {
        let d = LeakDetector::new();
        d.record_alloc(2 * 1024 * 1024);
        let r = d.report();
        assert!(r.potential_leak);
        assert_eq!(r.outstanding_allocs, 1);
        assert_eq!(r.outstanding_bytes, 2 * 1024 * 1024);
    }
}
