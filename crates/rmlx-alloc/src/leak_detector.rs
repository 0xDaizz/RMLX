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
