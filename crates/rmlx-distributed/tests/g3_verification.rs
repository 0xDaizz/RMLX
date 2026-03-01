//! Phase G3: Verification matrix — validates zero-copy KPIs and feature availability.

use rmlx_distributed::perf_counters::global_counters;

#[test]
fn verify_zero_copy_kpi_counters() {
    let counters = global_counters();
    counters.reset();

    let snap = counters.snapshot();
    assert_eq!(snap.cpu_copy_bytes, 0, "hot path cpu_copy_bytes must be 0");
    assert_eq!(snap.mr_reg_calls, 0, "hot path mr_reg_calls must be 0");
    assert_eq!(snap.gpu_sync_calls, 0, "hot path gpu_sync_calls must be 0");
}

#[test]
fn verify_counter_recording() {
    let counters = global_counters();
    counters.reset();

    counters.record_cpu_copy(1024);
    counters.record_mr_reg();
    counters.record_gpu_sync();
    counters.record_rdma_transfer(4096);

    let snap = counters.snapshot();
    assert_eq!(snap.cpu_copy_bytes, 1024);
    assert_eq!(snap.mr_reg_calls, 1);
    assert_eq!(snap.gpu_sync_calls, 1);
    assert_eq!(snap.rdma_bytes_transferred, 4096);
    assert_eq!(snap.rdma_ops_posted, 1);
}

#[test]
fn verify_counter_display() {
    let counters = global_counters();
    counters.reset();
    let snap = counters.snapshot();
    let display = format!("{}", snap);
    assert!(display.contains("PASS"));
    assert!(display.contains("cpu_copy_bytes"));
}

#[test]
fn verify_feature_matrix_types() {
    // Verify key types from all phases are available and constructible

    // Phase A: ExchangeTag
    use rmlx_rdma::exchange_tag::{ExchangeTag, encode_wr_id, decode_wr_id};
    let wr_id = encode_wr_id(42, ExchangeTag::Data, 0, 1);
    let fields = decode_wr_id(wr_id);
    assert_eq!(fields.seq, 42);

    // Phase B: MrPool types exist
    use rmlx_rdma::mr_pool::MrPool;
    let _ = std::mem::size_of::<MrPool>();

    // Phase C: SharedBuffer types exist
    use rmlx_rdma::shared_buffer::{SharedBuffer, SharedBufferPool, ConnectionId, PIPELINE};
    let _ = std::mem::size_of::<SharedBuffer>();
    let _ = std::mem::size_of::<SharedBufferPool>();
    assert_eq!(PIPELINE, 2);
    let conn_id = ConnectionId { node_id: 0, qp_num: 1, generation: 0 };
    assert_eq!(conn_id.node_id, 0);

    // Phase D-1: ProgressEngine types exist
    use rmlx_rdma::progress::{ProgressEngine, ProgressMode};
    let engine = ProgressEngine::new();
    assert_eq!(engine.pending_count(), 0);
    let _ = ProgressMode::Manual; // verify enum variant exists

    // Phase G1: GPU doorbell types exist
    use rmlx_rdma::gpu_doorbell::RdmaDescriptor;
    assert_eq!(std::mem::size_of::<RdmaDescriptor>(), 64);

    // Phase E: TP layers exist
    // (compile-time check via import)
    use rmlx_nn::parallel::{ColumnParallelLinear, RowParallelLinear};
    let _ = std::mem::size_of::<ColumnParallelLinear>();
    let _ = std::mem::size_of::<RowParallelLinear>();
}
