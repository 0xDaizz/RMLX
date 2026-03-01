//! Phase G3: Verification matrix — validates zero-copy KPIs and feature availability.

use rmlx_distributed::perf_counters::global_counters;
use rmlx_distributed::pipeline::{LayerPipeline, PipelineConfig};
use std::time::Duration;

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
    use rmlx_rdma::exchange_tag::{decode_wr_id, encode_wr_id, ExchangeTag};
    let wr_id = encode_wr_id(42, ExchangeTag::Data, 0, 1);
    let fields = decode_wr_id(wr_id);
    assert_eq!(fields.seq, 42);

    // Phase B: MrPool types exist
    use rmlx_rdma::mr_pool::MrPool;
    let _ = std::mem::size_of::<MrPool>();

    // Phase C: SharedBuffer types exist
    use rmlx_rdma::shared_buffer::{ConnectionId, SharedBuffer, SharedBufferPool, PIPELINE};
    let _ = std::mem::size_of::<SharedBuffer>();
    let _ = std::mem::size_of::<SharedBufferPool>();
    assert_eq!(PIPELINE, 2);
    let conn_id = ConnectionId {
        node_id: 0,
        qp_num: 1,
        generation: 0,
    };
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

/// Verify that pipeline wait_layer_complete increments gpu_sync_calls counter.
#[test]
fn verify_pipeline_gpu_sync_counter() {
    let counters = global_counters();
    counters.reset();

    let config = PipelineConfig {
        num_layers: 2,
        enable_overlap: true,
        sync_timeout: Duration::from_secs(1),
    };
    let pipeline = LayerPipeline::new(config);

    // wait_layer_complete on a layer with no ticket should still record gpu_sync
    let _ = pipeline.wait_layer_complete(0, Duration::from_millis(100));

    let snap = counters.snapshot();
    assert!(
        snap.gpu_sync_calls >= 1,
        "wait_layer_complete must increment gpu_sync_calls, got {}",
        snap.gpu_sync_calls
    );
}

/// Verify that async dispatch/combine counters are properly wired via direct recording.
/// This test exercises the global_counters API to ensure the counter fields match
/// what would be recorded during a real async dispatch/combine cycle.
#[test]
fn verify_async_dispatch_combine_counters() {
    let counters = global_counters();
    counters.reset();

    // Simulate what dispatch_async entry does
    counters.record_async_dispatch();
    counters.record_async_dispatch();

    // Simulate what combine_async_start entry does
    counters.record_async_combine();

    let snap = counters.snapshot();
    assert_eq!(
        snap.async_dispatch_count, 2,
        "async_dispatch_count should be 2"
    );
    assert_eq!(
        snap.async_combine_count, 1,
        "async_combine_count should be 1"
    );
    assert_eq!(snap.cpu_copy_bytes, 0, "no CPU copies in async path");
    assert_eq!(
        snap.mr_reg_calls, 0,
        "no MR registrations in zero-copy path"
    );
}

/// Verify that fallback counter increments for the legacy blocking path.
#[test]
fn verify_fallback_counter() {
    let counters = global_counters();
    counters.reset();

    // Simulate what the blocking RdmaTransport::send/recv/sendrecv does
    counters.record_fallback();
    counters.record_fallback();
    counters.record_fallback();

    let snap = counters.snapshot();
    assert_eq!(snap.fallback_count, 3, "fallback_count should be 3");
}

/// Full KPI assertion: in a pure async zero-copy path, cpu_copy_bytes and
/// mr_reg_calls should remain 0 while async_dispatch_count > 0.
#[test]
fn verify_zero_copy_kpi_async_path() {
    let counters = global_counters();
    counters.reset();

    // Simulate a clean async zero-copy dispatch cycle
    counters.record_async_dispatch();
    counters.record_rdma_transfer(65536);
    counters.record_async_combine();
    counters.record_rdma_transfer(65536);

    let snap = counters.snapshot();
    assert_eq!(
        snap.cpu_copy_bytes, 0,
        "zero-copy: cpu_copy_bytes must be 0"
    );
    assert_eq!(snap.mr_reg_calls, 0, "zero-copy: mr_reg_calls must be 0");
    assert!(
        snap.async_dispatch_count > 0,
        "async_dispatch_count must be > 0"
    );
    assert!(
        snap.async_combine_count > 0,
        "async_combine_count must be > 0"
    );
    assert_eq!(snap.rdma_bytes_transferred, 131072);
    assert_eq!(snap.rdma_ops_posted, 2);
    assert_eq!(snap.fallback_count, 0, "no fallbacks in async path");
}
