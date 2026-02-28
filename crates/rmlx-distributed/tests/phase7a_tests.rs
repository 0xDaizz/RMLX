//! Phase 7A tests for rmlx-distributed: moe_metrics, sparse_guard.

use rmlx_distributed::metrics::MoeMetrics;
use rmlx_distributed::sparse_guard::{GuardAction, SparseGuard};

#[test]
fn test_moe_metrics_basic() {
    let m = MoeMetrics::new();

    m.record_dispatch(100);
    m.record_dispatch(200);
    m.record_combine();
    m.record_cpu_dispatch();
    m.record_metal_dispatch();
    m.record_rdma_dispatch();
    m.record_overflow();
    m.record_zone_switch();
    m.record_dense_fallback();

    let snap = m.snapshot();
    assert_eq!(snap.dispatch_count, 2);
    assert_eq!(snap.total_tokens_routed, 300);
    assert_eq!(snap.combine_count, 1);
    assert_eq!(snap.cpu_dispatches, 1);
    assert_eq!(snap.metal_dispatches, 1);
    assert_eq!(snap.rdma_dispatches, 1);
    assert_eq!(snap.overflow_events, 1);
    assert_eq!(snap.zone_switches, 1);
    assert_eq!(snap.dense_fallback_count, 1);
}

#[test]
fn test_sparse_guard_normal() {
    let mut guard = SparseGuard::new();
    // Feed 100 steps with no overflow
    for _ in 0..100 {
        guard.record_step(0, 100);
    }
    let action = guard.evaluate();
    assert_eq!(action, GuardAction::None);
    assert!(!guard.should_increase_capacity());
    assert!(!guard.should_dense_fallback());
    assert!(!guard.is_dense_fallback());
    assert_eq!(guard.capacity_factor(), 1.0);
}

#[test]
fn test_sparse_guard_overflow_increase_capacity() {
    let mut guard = SparseGuard::new();
    // Feed high overflow (10%) for many windows to build up EMA past 0.05
    // EMA converges as: ema_n = alpha*ratio + (1-alpha)*ema_{n-1}
    // With alpha=0.1, ratio=0.10, need ~15 windows to reach ~0.078
    for _ in 0..15 {
        for _ in 0..100 {
            guard.record_step(10, 100);
        }
        guard.evaluate();
    }
    // EMA should be above 0.05 threshold
    assert!(guard.should_increase_capacity());
    assert!(!guard.should_dense_fallback());
    assert!(guard.capacity_factor() > 1.0);
}

#[test]
fn test_sparse_guard_dense_fallback() {
    let mut guard = SparseGuard::new();
    // Feed extreme overflow (50%) for many windows
    for _ in 0..20 {
        for _ in 0..100 {
            guard.record_step(50, 100);
        }
        guard.evaluate();
    }
    // EMA should be well above 0.20
    assert!(guard.is_dense_fallback());
    assert!(guard.should_dense_fallback());
}
