//! RDMA transfer metrics using atomic counters.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for RDMA send/recv performance tracking.
pub struct RdmaMetrics {
    pub send_count: AtomicU64,
    pub recv_count: AtomicU64,
    pub send_bytes: AtomicU64,
    pub recv_bytes: AtomicU64,
    pub send_errors: AtomicU64,
    pub recv_errors: AtomicU64,
    pub cq_polls: AtomicU64,
    pub connection_resets: AtomicU64,
}

impl RdmaMetrics {
    pub fn new() -> Self {
        Self {
            send_count: AtomicU64::new(0),
            recv_count: AtomicU64::new(0),
            send_bytes: AtomicU64::new(0),
            recv_bytes: AtomicU64::new(0),
            send_errors: AtomicU64::new(0),
            recv_errors: AtomicU64::new(0),
            cq_polls: AtomicU64::new(0),
            connection_resets: AtomicU64::new(0),
        }
    }

    /// Record a successful send operation.
    pub fn record_send(&self, bytes: u64) {
        self.send_count.fetch_add(1, Ordering::Relaxed);
        self.send_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a successful recv operation.
    pub fn record_recv(&self, bytes: u64) {
        self.recv_count.fetch_add(1, Ordering::Relaxed);
        self.recv_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a send error.
    pub fn record_send_error(&self) {
        self.send_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a recv error.
    pub fn record_recv_error(&self) {
        self.recv_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a CQ poll.
    pub fn record_cq_poll(&self) {
        self.cq_polls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a connection reset.
    pub fn record_connection_reset(&self) {
        self.connection_resets.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a snapshot of all counters.
    pub fn snapshot(&self) -> RdmaMetricsSnapshot {
        RdmaMetricsSnapshot {
            send_count: self.send_count.load(Ordering::Relaxed),
            recv_count: self.recv_count.load(Ordering::Relaxed),
            send_bytes: self.send_bytes.load(Ordering::Relaxed),
            recv_bytes: self.recv_bytes.load(Ordering::Relaxed),
            send_errors: self.send_errors.load(Ordering::Relaxed),
            recv_errors: self.recv_errors.load(Ordering::Relaxed),
            cq_polls: self.cq_polls.load(Ordering::Relaxed),
            connection_resets: self.connection_resets.load(Ordering::Relaxed),
        }
    }
}

impl Default for RdmaMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of RDMA metrics.
#[derive(Debug, Clone)]
pub struct RdmaMetricsSnapshot {
    pub send_count: u64,
    pub recv_count: u64,
    pub send_bytes: u64,
    pub recv_bytes: u64,
    pub send_errors: u64,
    pub recv_errors: u64,
    pub cq_polls: u64,
    pub connection_resets: u64,
}
