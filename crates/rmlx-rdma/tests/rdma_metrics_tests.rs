//! Phase 7A tests for rmlx-rdma: rdma_metrics.

use rmlx_rdma::rdma_metrics::RdmaMetrics;

#[test]
fn test_rdma_metrics_basic() {
    let m = RdmaMetrics::new();

    m.record_send(4096);
    m.record_send(8192);
    m.record_recv(2048);
    m.record_send_error();
    m.record_recv_error();
    m.record_cq_poll();
    m.record_cq_poll();
    m.record_connection_reset();

    let snap = m.snapshot();
    assert_eq!(snap.send_count, 2);
    assert_eq!(snap.send_bytes, 12288);
    assert_eq!(snap.recv_count, 1);
    assert_eq!(snap.recv_bytes, 2048);
    assert_eq!(snap.send_errors, 1);
    assert_eq!(snap.recv_errors, 1);
    assert_eq!(snap.cq_polls, 2);
    assert_eq!(snap.connection_resets, 1);
}
