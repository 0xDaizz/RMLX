//! Tests for layer pipeline.

use rmlx_distributed::pipeline::*;
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

    pipeline.begin_compute(0);
    assert_eq!(pipeline.stage(0), PipelineStage::Computing);

    pipeline.begin_transfer(0);
    assert_eq!(pipeline.stage(0), PipelineStage::Transferring);

    pipeline.complete(0);
    assert_eq!(pipeline.stage(0), PipelineStage::Complete);

    assert!(!pipeline.all_complete());
    pipeline.complete(1);
    pipeline.complete(2);
    assert!(pipeline.all_complete());
}

#[test]
fn test_pipeline_reset() {
    let mut pipeline = LayerPipeline::new(PipelineConfig {
        num_layers: 2,
        ..Default::default()
    });
    pipeline.complete(0);
    pipeline.complete(1);
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
    // overlap_gain should be significant (>0.2 easily for sleep-based test)
    assert!(
        stats.overlap_gain > 0.1,
        "overlap_gain = {}",
        stats.overlap_gain
    );
    eprintln!(
        "overlap_gain = {:.2}, serial={:?}, pipeline={:?}",
        stats.overlap_gain, stats.serial_time, stats.pipeline_time
    );
}
