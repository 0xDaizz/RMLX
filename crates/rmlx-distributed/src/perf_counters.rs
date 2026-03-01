//! Performance counters for zero-copy KPI validation.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Performance counters for tracking zero-copy KPIs.
pub struct PerfCounters {
    /// Bytes copied by CPU in hot path (target: 0)
    pub cpu_copy_bytes: AtomicU64,
    /// ibv_reg_mr calls in hot path (target: 0, init excluded)
    pub mr_reg_calls: AtomicU64,
    /// gpu::synchronize() calls in hot path (target: 0)
    pub gpu_sync_calls: AtomicU64,
    /// Total RDMA bytes transferred
    pub rdma_bytes_transferred: AtomicU64,
    /// Total RDMA ops posted
    pub rdma_ops_posted: AtomicU64,
    /// Total RDMA ops that encountered errors
    pub rdma_ops_error: AtomicU64,
    /// Async dispatch calls (via dispatch_async path)
    pub async_dispatch_count: AtomicU64,
    /// Async combine calls (via combine_async_start/finish path)
    pub async_combine_count: AtomicU64,
    /// Fallback to legacy blocking path (should be 0 in steady state)
    pub fallback_count: AtomicU64,
}

impl PerfCounters {
    pub fn new() -> Self {
        Self {
            cpu_copy_bytes: AtomicU64::new(0),
            mr_reg_calls: AtomicU64::new(0),
            gpu_sync_calls: AtomicU64::new(0),
            rdma_bytes_transferred: AtomicU64::new(0),
            rdma_ops_posted: AtomicU64::new(0),
            rdma_ops_error: AtomicU64::new(0),
            async_dispatch_count: AtomicU64::new(0),
            async_combine_count: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> PerfSnapshot {
        PerfSnapshot {
            cpu_copy_bytes: self.cpu_copy_bytes.load(Ordering::Relaxed),
            mr_reg_calls: self.mr_reg_calls.load(Ordering::Relaxed),
            gpu_sync_calls: self.gpu_sync_calls.load(Ordering::Relaxed),
            rdma_bytes_transferred: self.rdma_bytes_transferred.load(Ordering::Relaxed),
            rdma_ops_posted: self.rdma_ops_posted.load(Ordering::Relaxed),
            rdma_ops_error: self.rdma_ops_error.load(Ordering::Relaxed),
            async_dispatch_count: self.async_dispatch_count.load(Ordering::Relaxed),
            async_combine_count: self.async_combine_count.load(Ordering::Relaxed),
            fallback_count: self.fallback_count.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.cpu_copy_bytes.store(0, Ordering::Relaxed);
        self.mr_reg_calls.store(0, Ordering::Relaxed);
        self.gpu_sync_calls.store(0, Ordering::Relaxed);
        self.rdma_bytes_transferred.store(0, Ordering::Relaxed);
        self.rdma_ops_posted.store(0, Ordering::Relaxed);
        self.rdma_ops_error.store(0, Ordering::Relaxed);
        self.async_dispatch_count.store(0, Ordering::Relaxed);
        self.async_combine_count.store(0, Ordering::Relaxed);
        self.fallback_count.store(0, Ordering::Relaxed);
    }

    /// Increment cpu_copy_bytes counter.
    pub fn record_cpu_copy(&self, bytes: u64) {
        self.cpu_copy_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increment mr_reg_calls counter.
    pub fn record_mr_reg(&self) {
        self.mr_reg_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment gpu_sync_calls counter.
    pub fn record_gpu_sync(&self) {
        self.gpu_sync_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record RDMA transfer.
    pub fn record_rdma_transfer(&self, bytes: u64) {
        self.rdma_bytes_transferred
            .fetch_add(bytes, Ordering::Relaxed);
        self.rdma_ops_posted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an async dispatch (dispatch_async path used).
    pub fn record_async_dispatch(&self) {
        self.async_dispatch_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an async combine (combine_async_start/finish path used).
    pub fn record_async_combine(&self) {
        self.async_combine_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fallback to legacy blocking path.
    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for PerfCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PerfSnapshot {
    pub cpu_copy_bytes: u64,
    pub mr_reg_calls: u64,
    pub gpu_sync_calls: u64,
    pub rdma_bytes_transferred: u64,
    pub rdma_ops_posted: u64,
    pub rdma_ops_error: u64,
    pub async_dispatch_count: u64,
    pub async_combine_count: u64,
    pub fallback_count: u64,
}

impl PerfSnapshot {
    fn kpi_status(value: u64, target: u64) -> &'static str {
        if value <= target {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

impl fmt::Display for PerfSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== RMLX Performance Counters ===")?;
        writeln!(
            f,
            "  cpu_copy_bytes:       {:>10}  [{}] (target: 0)",
            self.cpu_copy_bytes,
            Self::kpi_status(self.cpu_copy_bytes, 0)
        )?;
        writeln!(
            f,
            "  mr_reg_calls:         {:>10}  [{}] (target: 0)",
            self.mr_reg_calls,
            Self::kpi_status(self.mr_reg_calls, 0)
        )?;
        writeln!(
            f,
            "  gpu_sync_calls:       {:>10}  [{}] (target: 0)",
            self.gpu_sync_calls,
            Self::kpi_status(self.gpu_sync_calls, 0)
        )?;
        writeln!(
            f,
            "  rdma_bytes_xferred:   {:>10}",
            self.rdma_bytes_transferred
        )?;
        writeln!(f, "  rdma_ops_posted:      {:>10}", self.rdma_ops_posted)?;
        writeln!(
            f,
            "  rdma_ops_error:       {:>10}  [{}] (target: 0)",
            self.rdma_ops_error,
            Self::kpi_status(self.rdma_ops_error, 0)
        )?;
        writeln!(
            f,
            "  async_dispatch:       {:>10}",
            self.async_dispatch_count
        )?;
        writeln!(
            f,
            "  async_combine:        {:>10}",
            self.async_combine_count
        )?;
        writeln!(
            f,
            "  fallback_count:       {:>10}  [{}] (target: 0)",
            self.fallback_count,
            Self::kpi_status(self.fallback_count, 0)
        )?;
        write!(f, "================================")
    }
}

/// Global performance counter instance.
pub fn global_counters() -> &'static PerfCounters {
    static COUNTERS: OnceLock<PerfCounters> = OnceLock::new();
    COUNTERS.get_or_init(PerfCounters::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_record_and_snapshot() {
        let c = PerfCounters::new();
        c.record_cpu_copy(1024);
        c.record_mr_reg();
        c.record_mr_reg();
        c.record_gpu_sync();
        c.record_rdma_transfer(4096);
        c.record_async_dispatch();
        c.record_async_combine();
        c.record_fallback();

        let snap = c.snapshot();
        assert_eq!(snap.cpu_copy_bytes, 1024);
        assert_eq!(snap.mr_reg_calls, 2);
        assert_eq!(snap.gpu_sync_calls, 1);
        assert_eq!(snap.rdma_bytes_transferred, 4096);
        assert_eq!(snap.rdma_ops_posted, 1);
        assert_eq!(snap.async_dispatch_count, 1);
        assert_eq!(snap.async_combine_count, 1);
        assert_eq!(snap.fallback_count, 1);
    }

    #[test]
    fn counter_reset_clears_all() {
        let c = PerfCounters::new();
        c.record_cpu_copy(100);
        c.record_mr_reg();
        c.record_gpu_sync();
        c.record_rdma_transfer(200);
        c.record_async_dispatch();
        c.record_async_combine();
        c.record_fallback();

        c.reset();
        let snap = c.snapshot();
        assert_eq!(snap.cpu_copy_bytes, 0);
        assert_eq!(snap.mr_reg_calls, 0);
        assert_eq!(snap.gpu_sync_calls, 0);
        assert_eq!(snap.rdma_bytes_transferred, 0);
        assert_eq!(snap.rdma_ops_posted, 0);
        assert_eq!(snap.rdma_ops_error, 0);
        assert_eq!(snap.async_dispatch_count, 0);
        assert_eq!(snap.async_combine_count, 0);
        assert_eq!(snap.fallback_count, 0);
    }

    #[test]
    fn kpi_pass_fail_display() {
        let c = PerfCounters::new();
        let snap = c.snapshot();
        let display = format!("{snap}");
        // All zero-target counters should pass when at 0
        assert!(display.contains("[PASS]"));
        assert!(!display.contains("[FAIL]"));

        // Record a cpu_copy — this should cause FAIL
        c.record_cpu_copy(1);
        let snap2 = c.snapshot();
        let display2 = format!("{snap2}");
        assert!(display2.contains("[FAIL]"));
    }

    /// Simulates the zero-copy async path KPI assertion pattern:
    /// After reset, the async zero-copy path should show:
    /// - cpu_copy_bytes == 0 (no memcpy)
    /// - mr_reg_calls == 0 (pre-registered MRs)
    /// - gpu_sync_calls == 0 (async, no blocking wait)
    /// - async_dispatch_count > 0 (async path was used)
    #[test]
    fn kpi_zero_copy_async_path_assertions() {
        let c = PerfCounters::new();
        c.reset();

        // Simulate what the async zero-copy path records:
        // - No cpu_copy (zero-copy)
        // - No mr_reg (pre-registered SharedBuffer)
        // - No gpu_sync (async, non-blocking)
        // - async_dispatch recorded
        c.record_async_dispatch();
        c.record_async_dispatch();
        c.record_async_combine();

        let snap = c.snapshot();
        assert_eq!(snap.cpu_copy_bytes, 0, "zero-copy: no CPU memcpy");
        assert_eq!(snap.mr_reg_calls, 0, "zero-copy: no ibv_reg_mr in hot path");
        assert_eq!(snap.gpu_sync_calls, 0, "async: no blocking GPU sync");
        assert!(
            snap.async_dispatch_count > 0,
            "async dispatch path must be used"
        );
        assert_eq!(snap.fallback_count, 0, "no fallback to legacy path");
    }
}
