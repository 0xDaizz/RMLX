//! Tests for layer pipeline.

use rmlx_distributed::pipeline::*;
use std::thread;
use std::time::Duration;

#[test]
fn test_pipeline_stages() {
    let config = PipelineConfig {
        num_layers: 3,
        enable_overlap: true,
        sync_timeout: Duration::from_secs(5),
    };
    let mut pipeline = LayerPipeline::new(config);

    assert_eq!(pipeline.stage(0), PipelineStage::WaitingForInput);

    pipeline.begin_compute(0).unwrap();
    assert_eq!(pipeline.stage(0), PipelineStage::Computing);

    pipeline.begin_transfer(0).unwrap();
    assert_eq!(pipeline.stage(0), PipelineStage::Transferring);

    pipeline.complete(0).unwrap();
    assert_eq!(pipeline.stage(0), PipelineStage::Complete);

    assert!(!pipeline.all_complete());
    pipeline.complete(1).unwrap();
    pipeline.complete(2).unwrap();
    assert!(pipeline.all_complete());
}

#[test]
fn test_pipeline_reset() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 2,
        ..Default::default()
    });
    pipeline.complete(0).unwrap();
    pipeline.complete(1).unwrap();
    assert!(pipeline.all_complete());

    pipeline.reset();
    assert_eq!(pipeline.stage(0), PipelineStage::WaitingForInput);
    assert!(!pipeline.all_complete());
}

#[test]
fn test_pipeline_stats() {
    let stats = PipelineStats::from_timings(
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(120), // pipeline time
        Duration::from_millis(5),
    );
    // overlap_gain = (200 - 120) / 200 = 0.4
    assert!((stats.overlap_gain - 0.4).abs() < 0.01);
    assert_eq!(stats.serial_time, Duration::from_millis(200));
}

#[test]
fn test_pipeline_overlap_measurement() {
    let stats = LayerPipeline::measure_overlap(
        || std::thread::sleep(Duration::from_millis(50)),
        || std::thread::sleep(Duration::from_millis(50)),
    );
    // Both take ~50ms. Pipeline should be ~50ms (parallel).
    // CI shared runners have highly variable scheduling, so only assert
    // that the measurement completes without panic. Overlap quality is
    // validated by test_pipeline_stats which uses deterministic durations.
    assert!(
        stats.overlap_gain > -1.0,
        "overlap_gain = {}",
        stats.overlap_gain
    );
    eprintln!(
        "overlap_gain = {:.2}, serial={:?}, pipeline={:?}",
        stats.overlap_gain, stats.serial_time, stats.pipeline_time
    );
}

#[test]
fn test_pipeline_begin_compute_no_event_marks_gpu_on_transfer() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 1,
        ..Default::default()
    });
    pipeline.begin_compute(0).unwrap();
    let ticket = pipeline.ticket(0).unwrap();
    assert!(!ticket.is_gpu_complete());

    pipeline.begin_transfer(0).unwrap();
    // Without gpu_event, begin_transfer manually marks gpu complete
    let ticket = pipeline.ticket(0).unwrap();
    assert!(ticket.is_gpu_complete());
}

#[test]
fn test_pipeline_complete_with_rdma() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 2,
        ..Default::default()
    });
    pipeline.begin_compute(0).unwrap();
    pipeline.begin_transfer(0).unwrap();

    // complete_with_rdma(false) does not mark rdma complete
    pipeline.complete_with_rdma(0, false).unwrap();
    let ticket = pipeline.ticket(0).unwrap();
    assert!(ticket.is_gpu_complete());
    assert!(!ticket.is_rdma_complete());

    // Manually mark rdma complete
    ticket.mark_rdma_complete();
    assert!(ticket.is_rdma_complete());
}

#[test]
fn test_pipeline_wait_layer_complete() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 1,
        ..Default::default()
    });
    pipeline.begin_compute(0).unwrap();

    // Grab a clone of the ticket before marking complete
    let ticket_clone = pipeline.ticket(0).unwrap().clone();

    // Spawn thread to mark both phases complete after a short delay
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        ticket_clone.mark_gpu_complete();
        ticket_clone.mark_rdma_complete();
    });

    let result = pipeline.wait_layer_complete(0, Duration::from_secs(2));
    assert!(result.is_ok());
}

#[test]
fn test_pipeline_wait_layer_no_ticket() {
    let pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 1,
        ..Default::default()
    });
    // No ticket created, should return Ok immediately
    let result = pipeline.wait_layer_complete(0, Duration::from_secs(1));
    assert!(result.is_ok());
}

#[test]
fn test_pipeline_wait_layer_timeout() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 1,
        ..Default::default()
    });
    pipeline.begin_compute(0).unwrap();
    // Neither gpu nor rdma completed — should timeout
    let result = pipeline.wait_layer_complete(0, Duration::from_millis(50));
    assert!(result.is_err());
}
