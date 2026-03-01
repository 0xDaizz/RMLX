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
}

impl PerfCounters {
    pub fn new() -> Self {
        Self {
            cpu_copy_bytes: AtomicU64::new(0),
            mr_reg_calls: AtomicU64::new(0),
            gpu_sync_calls: AtomicU64::new(0),
            rdma_bytes_transferred: AtomicU64::new(0),
            rdma_ops_posted: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> PerfSnapshot {
        PerfSnapshot {
            cpu_copy_bytes: self.cpu_copy_bytes.load(Ordering::Relaxed),
            mr_reg_calls: self.mr_reg_calls.load(Ordering::Relaxed),
            gpu_sync_calls: self.gpu_sync_calls.load(Ordering::Relaxed),
            rdma_bytes_transferred: self.rdma_bytes_transferred.load(Ordering::Relaxed),
            rdma_ops_posted: self.rdma_ops_posted.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.cpu_copy_bytes.store(0, Ordering::Relaxed);
        self.mr_reg_calls.store(0, Ordering::Relaxed);
        self.gpu_sync_calls.store(0, Ordering::Relaxed);
        self.rdma_bytes_transferred.store(0, Ordering::Relaxed);
        self.rdma_ops_posted.store(0, Ordering::Relaxed);
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
        self.rdma_bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
        self.rdma_ops_posted.fetch_add(1, Ordering::Relaxed);
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
}

impl PerfSnapshot {
    fn kpi_status(value: u64, target: u64) -> &'static str {
        if value <= target { "PASS" } else { "FAIL" }
    }
}

impl fmt::Display for PerfSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== RMLX Performance Counters ===")?;
        writeln!(f, "  cpu_copy_bytes:       {:>10}  [{}] (target: 0)",
            self.cpu_copy_bytes, Self::kpi_status(self.cpu_copy_bytes, 0))?;
        writeln!(f, "  mr_reg_calls:         {:>10}  [{}] (target: 0)",
            self.mr_reg_calls, Self::kpi_status(self.mr_reg_calls, 0))?;
        writeln!(f, "  gpu_sync_calls:       {:>10}  [{}] (target: 0)",
            self.gpu_sync_calls, Self::kpi_status(self.gpu_sync_calls, 0))?;
        writeln!(f, "  rdma_bytes_xferred:   {:>10}", self.rdma_bytes_transferred)?;
        writeln!(f, "  rdma_ops_posted:      {:>10}", self.rdma_ops_posted)?;
        write!(f, "================================")
    }
}

/// Global performance counter instance.
pub fn global_counters() -> &'static PerfCounters {
    static COUNTERS: OnceLock<PerfCounters> = OnceLock::new();
    COUNTERS.get_or_init(PerfCounters::new)
}
