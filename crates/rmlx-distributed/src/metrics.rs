//! MoE dispatch metrics using atomic counters.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for MoE dispatch/combine operations.
pub struct MoeMetrics {
    pub dispatch_count: AtomicU64,
    pub combine_count: AtomicU64,
    pub cpu_dispatches: AtomicU64,
    pub metal_dispatches: AtomicU64,
    pub rdma_dispatches: AtomicU64,
    pub overflow_events: AtomicU64,
    pub zone_switches: AtomicU64,
    pub total_tokens_routed: AtomicU64,
    pub dense_fallback_count: AtomicU64,
}

impl MoeMetrics {
    pub fn new() -> Self {
        Self {
            dispatch_count: AtomicU64::new(0),
            combine_count: AtomicU64::new(0),
            cpu_dispatches: AtomicU64::new(0),
            metal_dispatches: AtomicU64::new(0),
            rdma_dispatches: AtomicU64::new(0),
            overflow_events: AtomicU64::new(0),
            zone_switches: AtomicU64::new(0),
            total_tokens_routed: AtomicU64::new(0),
            dense_fallback_count: AtomicU64::new(0),
        }
    }

    pub fn record_dispatch(&self, tokens: u64) {
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_routed
            .fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn record_combine(&self) {
        self.combine_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cpu_dispatch(&self) {
        self.cpu_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_metal_dispatch(&self) {
        self.metal_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rdma_dispatch(&self) {
        self.rdma_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_overflow(&self) {
        self.overflow_events.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_zone_switch(&self) {
        self.zone_switches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_dense_fallback(&self) {
        self.dense_fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MoeMetricsSnapshot {
        MoeMetricsSnapshot {
            dispatch_count: self.dispatch_count.load(Ordering::Relaxed),
            combine_count: self.combine_count.load(Ordering::Relaxed),
            cpu_dispatches: self.cpu_dispatches.load(Ordering::Relaxed),
            metal_dispatches: self.metal_dispatches.load(Ordering::Relaxed),
            rdma_dispatches: self.rdma_dispatches.load(Ordering::Relaxed),
            overflow_events: self.overflow_events.load(Ordering::Relaxed),
            zone_switches: self.zone_switches.load(Ordering::Relaxed),
            total_tokens_routed: self.total_tokens_routed.load(Ordering::Relaxed),
            dense_fallback_count: self.dense_fallback_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for MoeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of MoE metrics.
#[derive(Debug, Clone)]
pub struct MoeMetricsSnapshot {
    pub dispatch_count: u64,
    pub combine_count: u64,
    pub cpu_dispatches: u64,
    pub metal_dispatches: u64,
    pub rdma_dispatches: u64,
    pub overflow_events: u64,
    pub zone_switches: u64,
    pub total_tokens_routed: u64,
    pub dense_fallback_count: u64,
}
