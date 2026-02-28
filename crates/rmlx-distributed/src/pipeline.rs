//! Layer-level pipeline for compute↔RDMA overlap.
//!
//! Overlaps GPU computation of layer N with RDMA transfer of layer N-1's output.
//! This is the key performance optimization for distributed inference.

use std::time::{Duration, Instant};

/// Pipeline stage for tracking layer execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Waiting for input from RDMA transfer
    WaitingForInput,
    /// GPU compute in progress
    Computing,
    /// Transferring output via RDMA
    Transferring,
    /// Layer complete
    Complete,
}

/// Statistics for pipeline overlap measurement.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub serial_time: Duration,
    pub pipeline_time: Duration,
    pub overlap_gain: f64,
    pub compute_time: Duration,
    pub transfer_time: Duration,
    pub sync_overhead: Duration,
}

impl PipelineStats {
    /// Calculate overlap gain: (serial - pipeline) / serial
    pub fn from_timings(
        compute_time: Duration,
        transfer_time: Duration,
        pipeline_time: Duration,
        sync_overhead: Duration,
    ) -> Self {
        let serial_time = compute_time + transfer_time;
        let overlap_gain = if serial_time.as_secs_f64() > 0.0 {
            (serial_time.as_secs_f64() - pipeline_time.as_secs_f64()) / serial_time.as_secs_f64()
        } else {
            0.0
        };
        Self {
            serial_time,
            pipeline_time,
            overlap_gain,
            compute_time,
            transfer_time,
            sync_overhead,
        }
    }
}

/// Layer pipeline configuration.
pub struct PipelineConfig {
    /// Number of layers to pipeline
    pub num_layers: usize,
    /// Whether to use dual-queue overlap (false = single queue, serialized)
    pub enable_overlap: bool,
    /// Timeout for waiting on events
    pub sync_timeout: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            enable_overlap: true,
            sync_timeout: Duration::from_secs(5),
        }
    }
}

/// Layer pipeline manager for compute↔transfer overlap.
///
/// In overlap mode, uses two streams:
/// - Stream 0 (compute): GPU kernel execution
/// - Stream 1 (transfer): RDMA send/recv and buffer copies
///
/// SharedEvent synchronization ensures correctness while maximizing overlap.
pub struct LayerPipeline {
    config: PipelineConfig,
    stages: Vec<PipelineStage>,
}

impl LayerPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let stages = vec![PipelineStage::WaitingForInput; config.num_layers];
        Self { config, stages }
    }

    /// Mark a layer as computing.
    pub fn begin_compute(&mut self, layer: usize) {
        assert!(layer < self.config.num_layers);
        self.stages[layer] = PipelineStage::Computing;
    }

    /// Mark a layer as transferring.
    pub fn begin_transfer(&mut self, layer: usize) {
        assert!(layer < self.config.num_layers);
        self.stages[layer] = PipelineStage::Transferring;
    }

    /// Mark a layer as complete.
    pub fn complete(&mut self, layer: usize) {
        assert!(layer < self.config.num_layers);
        self.stages[layer] = PipelineStage::Complete;
    }

    /// Get current stage for a layer.
    pub fn stage(&self, layer: usize) -> PipelineStage {
        self.stages[layer]
    }

    /// Whether overlap is enabled.
    pub fn overlap_enabled(&self) -> bool {
        self.config.enable_overlap
    }

    /// Sync timeout.
    pub fn sync_timeout(&self) -> Duration {
        self.config.sync_timeout
    }

    /// Check if all layers are complete.
    pub fn all_complete(&self) -> bool {
        self.stages.iter().all(|s| *s == PipelineStage::Complete)
    }

    /// Reset all stages to WaitingForInput.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            *stage = PipelineStage::WaitingForInput;
        }
    }

    /// Measure serial vs pipeline execution time.
    pub fn measure_overlap(
        compute_fn: impl Fn() + Sync,
        transfer_fn: impl Fn() + Sync,
    ) -> PipelineStats {
        // Measure serial
        let start = Instant::now();
        compute_fn();
        let compute_time = start.elapsed();

        let start = Instant::now();
        transfer_fn();
        let transfer_time = start.elapsed();

        // Measure pipeline (both started together)
        let start = Instant::now();
        // In a real implementation, these would run on separate queues.
        // Here we simulate: the overlap depends on whether the GPU can
        // actually overlap compute and transfer.
        std::thread::scope(|s| {
            let h1 = s.spawn(&compute_fn);
            let h2 = s.spawn(&transfer_fn);
            h1.join().unwrap();
            h2.join().unwrap();
        });
        let pipeline_time = start.elapsed();

        PipelineStats::from_timings(compute_time, transfer_time, pipeline_time, Duration::ZERO)
    }
}
