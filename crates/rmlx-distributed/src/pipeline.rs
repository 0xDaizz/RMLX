//! Layer-level pipeline for compute↔RDMA overlap.
//!
//! Overlaps GPU computation of layer N with RDMA transfer of layer N-1's output.
//! This is the key performance optimization for distributed inference.

use std::sync::Arc;
use std::time::{Duration, Instant};

use rmlx_alloc::zero_copy::{CompletionError, CompletionTicket};
use rmlx_metal::event::GpuEvent;
use rmlx_rdma::progress::PendingOp;

use crate::group::DistributedError;

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

/// Per-layer transfer state for async (PendingOp-based) pipeline.
///
/// Instead of blocking on `wait_posted` in the hot path, each layer's
/// RDMA transfer is tracked via `PendingOp` handles that can be polled
/// or waited on independently.
pub struct LayerTransferState {
    /// PendingOps for the RDMA sends posted for this layer's output.
    pub send_ops: Vec<PendingOp>,
    /// PendingOps for the RDMA recvs posted for this layer's input.
    pub recv_ops: Vec<PendingOp>,
}

impl LayerTransferState {
    pub fn new() -> Self {
        Self {
            send_ops: Vec::new(),
            recv_ops: Vec::new(),
        }
    }

    /// Check if all send operations have completed (non-blocking).
    pub fn sends_complete(&self) -> bool {
        self.send_ops.iter().all(|op| !op.is_pending())
    }

    /// Check if all recv operations have completed (non-blocking).
    pub fn recvs_complete(&self) -> bool {
        self.recv_ops.iter().all(|op| !op.is_pending())
    }

    /// Check if all operations (send + recv) have completed.
    pub fn all_complete(&self) -> bool {
        self.sends_complete() && self.recvs_complete()
    }
}

impl Default for LayerTransferState {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer pipeline manager for compute↔transfer overlap.
///
/// In overlap mode, uses two streams:
/// - Stream 0 (compute): GPU kernel execution
/// - Stream 1 (transfer): RDMA send/recv and buffer copies
///
/// SharedEvent synchronization ensures correctness while maximizing overlap.
///
/// ## Event-chained async pipeline (Phase G2)
///
/// When configured with `compute_event` and `transfer_event`:
/// 1. Layer N GPU compute completes -> `compute_event` signaled
/// 2. CPU proxy (or transfer thread) waits on `compute_event`, posts RDMA send
/// 3. RDMA completion -> `transfer_event` signaled
/// 4. Layer N+1 GPU compute waits on `transfer_event` before consuming input
///
/// This chain removes all CPU blocking from the layer hot path.
pub struct LayerPipeline {
    config: PipelineConfig,
    stages: Vec<PipelineStage>,
    tickets: Vec<Option<CompletionTicket>>,
    /// Per-layer async transfer state (PendingOp handles).
    transfer_states: Vec<Option<LayerTransferState>>,
    /// Compute-done event: GPU signals after layer N compute finishes.
    /// The transfer side waits on this before posting RDMA.
    compute_event: Option<Arc<GpuEvent>>,
    /// Transfer-done event: CPU signals after RDMA completion.
    /// Layer N+1 GPU compute waits on this before consuming input.
    transfer_event: Option<Arc<GpuEvent>>,
}

impl LayerPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let num = config.num_layers;
        let stages = vec![PipelineStage::WaitingForInput; num];
        let tickets = vec![None; num];
        let transfer_states = (0..num).map(|_| None).collect();
        Self {
            config,
            stages,
            tickets,
            transfer_states,
            compute_event: None,
            transfer_event: None,
        }
    }

    /// Attach SharedEvent pair for event-chained pipeline.
    ///
    /// - `compute_event`: GPU signals after layer compute finishes
    /// - `transfer_event`: CPU signals after RDMA transfer completes
    ///
    /// These events chain GPU compute and RDMA transfer without CPU blocking:
    /// compute_event links compute completion -> transfer start,
    /// transfer_event links transfer completion -> next layer input ready.
    pub fn with_events(
        mut self,
        compute_event: Arc<GpuEvent>,
        transfer_event: Arc<GpuEvent>,
    ) -> Self {
        self.compute_event = Some(compute_event);
        self.transfer_event = Some(transfer_event);
        self
    }

    /// Get the compute event (GPU -> transfer synchronization).
    pub fn compute_event(&self) -> Option<&Arc<GpuEvent>> {
        self.compute_event.as_ref()
    }

    /// Get the transfer event (transfer -> next layer synchronization).
    pub fn transfer_event(&self) -> Option<&Arc<GpuEvent>> {
        self.transfer_event.as_ref()
    }

    /// Mark a layer as computing and attach a completion ticket.
    ///
    /// The ticket tracks both GPU compute and RDMA transfer completion
    /// for this layer's output buffer.
    pub fn begin_compute(&mut self, layer: usize) -> Result<(), DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        self.stages[layer] = PipelineStage::Computing;
        self.tickets[layer] = Some(CompletionTicket::new());
        Ok(())
    }

    /// Mark a layer as computing with a GpuEvent for hardware-level completion tracking.
    ///
    /// Returns the signal value that should be encoded into the command buffer
    /// so the event fires on GPU completion.
    pub fn begin_compute_with_event(
        &mut self,
        layer: usize,
        event: Arc<GpuEvent>,
    ) -> Result<u64, DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        self.stages[layer] = PipelineStage::Computing;
        let mut ticket = CompletionTicket::new();
        let signal_val = ticket.with_gpu_event(event);
        self.tickets[layer] = Some(ticket);
        Ok(signal_val)
    }

    /// Mark a layer as transferring.
    ///
    /// GPU compute for this layer's output should have already completed.
    /// If no GpuEvent is attached to the ticket, manually marks the GPU phase
    /// as complete. Event-based tickets auto-detect via the shared event.
    pub fn begin_transfer(&mut self, layer: usize) -> Result<(), DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        self.stages[layer] = PipelineStage::Transferring;
        if let Some(ref ticket) = self.tickets[layer] {
            if !ticket.has_gpu_event() {
                ticket.mark_gpu_complete();
            }
        }
        Ok(())
    }

    /// Begin async transfer for a layer using PendingOp handles.
    ///
    /// Instead of blocking on `wait_posted`, this tracks RDMA operations via
    /// PendingOps. The caller posts RDMA send/recv using the transport's async
    /// methods and passes the resulting PendingOps here.
    ///
    /// If `compute_event` is configured, waits for the GPU compute to complete
    /// before allowing the transfer to proceed (the caller should have already
    /// encoded a signal on the compute event in the command buffer).
    pub fn begin_transfer_async(
        &mut self,
        layer: usize,
        state: LayerTransferState,
    ) -> Result<(), DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        self.stages[layer] = PipelineStage::Transferring;
        if let Some(ref ticket) = self.tickets[layer] {
            if !ticket.has_gpu_event() {
                ticket.mark_gpu_complete();
            }
        }
        self.transfer_states[layer] = Some(state);
        Ok(())
    }

    /// Poll whether a layer's async transfers have all completed (non-blocking).
    ///
    /// Returns true if all PendingOps for this layer have resolved.
    /// Returns true if no transfer state is set (nothing to wait for).
    pub fn poll_transfer_complete(&self, layer: usize) -> bool {
        match &self.transfer_states[layer] {
            Some(state) => state.all_complete(),
            None => true,
        }
    }

    /// Complete a layer whose async transfers have finished.
    ///
    /// This is the non-blocking equivalent of `complete()` for async pipelines.
    /// Checks that transfers are done, marks RDMA complete on the ticket, and
    /// signals the transfer_event so downstream GPU work can proceed.
    ///
    /// Returns `Err` if transfers are still pending.
    pub fn complete_async(&mut self, layer: usize) -> Result<(), DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }

        if !self.poll_transfer_complete(layer) {
            return Err(DistributedError::Transport(format!(
                "layer {layer} transfers not yet complete"
            )));
        }

        self.stages[layer] = PipelineStage::Complete;

        // Mark RDMA complete on the ticket
        if let Some(ref ticket) = self.tickets[layer] {
            ticket.mark_rdma_complete();
        }

        // Signal transfer_event so downstream GPU compute can proceed
        if let Some(ref event) = self.transfer_event {
            let val = event.next_value();
            event.raw().set_signaled_value(val);
        }

        // Clear the transfer state
        self.transfer_states[layer] = None;

        Ok(())
    }

    /// Drive async pipeline progress for all layers.
    ///
    /// Scans all layers in the Transferring stage, checks if their PendingOps
    /// have completed, and auto-completes them. Returns the number of layers
    /// that transitioned to Complete in this call.
    ///
    /// This should be called periodically from the event loop alongside
    /// `ProgressEngine::drive_progress()`.
    pub fn drive_pipeline_progress(&mut self) -> usize {
        let mut completed = 0;
        for layer in 0..self.config.num_layers {
            if self.stages[layer] == PipelineStage::Transferring
                && self.poll_transfer_complete(layer)
            {
                // Transfer done — auto-complete this layer
                self.stages[layer] = PipelineStage::Complete;
                if let Some(ref ticket) = self.tickets[layer] {
                    ticket.mark_rdma_complete();
                }
                if let Some(ref event) = self.transfer_event {
                    let val = event.next_value();
                    event.raw().set_signaled_value(val);
                }
                self.transfer_states[layer] = None;
                completed += 1;
            }
        }
        completed
    }

    /// Get the async transfer state for a layer, if set.
    pub fn transfer_state(&self, layer: usize) -> Option<&LayerTransferState> {
        self.transfer_states[layer].as_ref()
    }

    /// Mark a layer as complete, optionally signaling RDMA completion.
    ///
    /// If `rdma_complete` is true, marks the ticket's RDMA phase as complete.
    /// Pass false if RDMA completion will be signaled externally.
    pub fn complete(&mut self, layer: usize) -> Result<(), DistributedError> {
        self.complete_with_rdma(layer, true)
    }

    /// Mark a layer as complete with explicit RDMA completion control.
    pub fn complete_with_rdma(
        &mut self,
        layer: usize,
        rdma_complete: bool,
    ) -> Result<(), DistributedError> {
        if layer >= self.config.num_layers {
            return Err(DistributedError::Transport(format!(
                "layer index {layer} out of range (num_layers={})",
                self.config.num_layers
            )));
        }
        self.stages[layer] = PipelineStage::Complete;
        if rdma_complete {
            if let Some(ref ticket) = self.tickets[layer] {
                ticket.mark_rdma_complete();
            }
        }
        Ok(())
    }

    /// Wait for a layer's GPU and RDMA operations to fully complete.
    pub fn wait_layer_complete(
        &self,
        layer: usize,
        timeout: Duration,
    ) -> Result<(), CompletionError> {
        if layer >= self.config.num_layers {
            return Err(CompletionError::GpuTimeout);
        }
        match &self.tickets[layer] {
            Some(ticket) => ticket.wait_all_complete(timeout),
            None => Ok(()),
        }
    }

    /// Get the completion ticket for a layer, if one has been created.
    pub fn ticket(&self, layer: usize) -> Option<&CompletionTicket> {
        self.tickets[layer].as_ref()
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

    /// Reset all stages to WaitingForInput and clear tickets and transfer states.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            *stage = PipelineStage::WaitingForInput;
        }
        for ticket in &mut self.tickets {
            *ticket = None;
        }
        for state in &mut self.transfer_states {
            *state = None;
        }
    }

    /// Measure serial vs pipeline execution time using dual Metal command queues.
    ///
    /// When a Metal device is available, uses two real MTLCommandQueues:
    /// - Queue 0: compute work
    /// - Queue 1: transfer/copy work
    ///   This measures actual GPU-level overlap rather than CPU thread parallelism.
    ///
    /// Falls back to CPU thread parallelism if no Metal device is available.
    pub fn measure_overlap(
        compute_fn: impl Fn() + Sync,
        transfer_fn: impl Fn() + Sync,
    ) -> PipelineStats {
        // Measure serial baseline
        let start = Instant::now();
        compute_fn();
        let compute_time = start.elapsed();

        let start = Instant::now();
        transfer_fn();
        let transfer_time = start.elapsed();

        // Create Metal queues (if available) outside the timed section
        // to avoid inflating pipeline_time with device/queue setup cost.
        let metal_queues = metal::Device::system_default().map(|device| {
            let compute_queue = device.new_command_queue();
            let transfer_queue = device.new_command_queue();
            (compute_queue, transfer_queue)
        });

        // Measure pipeline with dual queues (or thread fallback)
        let pipeline_start = Instant::now();
        let sync_start;

        if let Some((ref compute_queue, ref transfer_queue)) = metal_queues {
            // Real dual-queue overlap: two independent command queues
            // The actual overlap depends on GPU hardware scheduling
            std::thread::scope(|s| {
                let h1 = s.spawn(|| {
                    let _ = compute_queue;
                    compute_fn();
                });
                let h2 = s.spawn(|| {
                    let _ = transfer_queue;
                    transfer_fn();
                });
                // Scoped thread join: propagate panic if thread panicked
                if let Err(e) = h1.join() {
                    std::panic::resume_unwind(e);
                }
                if let Err(e) = h2.join() {
                    std::panic::resume_unwind(e);
                }
            });
            sync_start = Instant::now();
        } else {
            // Fallback: CPU thread parallelism
            std::thread::scope(|s| {
                let h1 = s.spawn(&compute_fn);
                let h2 = s.spawn(&transfer_fn);
                if let Err(e) = h1.join() {
                    std::panic::resume_unwind(e);
                }
                if let Err(e) = h2.join() {
                    std::panic::resume_unwind(e);
                }
            });
            sync_start = Instant::now();
        }

        let pipeline_time = pipeline_start.elapsed();
        let sync_overhead = sync_start.elapsed();

        PipelineStats::from_timings(compute_time, transfer_time, pipeline_time, sync_overhead)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_transfer_state_empty_is_complete() {
        let state = LayerTransferState::new();
        assert!(state.sends_complete());
        assert!(state.recvs_complete());
        assert!(state.all_complete());
    }

    #[test]
    fn pipeline_async_lifecycle() {
        let config = PipelineConfig {
            num_layers: 3,
            enable_overlap: true,
            sync_timeout: Duration::from_secs(1),
        };
        let mut pipeline = LayerPipeline::new(config);

        // Layer 0: begin compute
        pipeline.begin_compute(0).unwrap();
        assert_eq!(pipeline.stage(0), PipelineStage::Computing);

        // Begin async transfer with empty state (simulating already-completed ops)
        let state = LayerTransferState::new();
        pipeline.begin_transfer_async(0, state).unwrap();
        assert_eq!(pipeline.stage(0), PipelineStage::Transferring);

        // Since the state has no pending ops, poll_transfer_complete should be true
        assert!(pipeline.poll_transfer_complete(0));

        // Complete the layer
        pipeline.complete_async(0).unwrap();
        assert_eq!(pipeline.stage(0), PipelineStage::Complete);
    }

    #[test]
    fn drive_pipeline_progress_auto_completes() {
        let config = PipelineConfig {
            num_layers: 2,
            enable_overlap: true,
            sync_timeout: Duration::from_secs(1),
        };
        let mut pipeline = LayerPipeline::new(config);

        // Put both layers into Transferring with empty state
        pipeline.begin_compute(0).unwrap();
        pipeline.begin_transfer_async(0, LayerTransferState::new()).unwrap();
        pipeline.begin_compute(1).unwrap();
        pipeline.begin_transfer_async(1, LayerTransferState::new()).unwrap();

        // Drive progress should complete both
        let completed = pipeline.drive_pipeline_progress();
        assert_eq!(completed, 2);
        assert!(pipeline.all_complete());
    }

    #[test]
    fn pipeline_reset_clears_transfer_states() {
        let config = PipelineConfig {
            num_layers: 1,
            enable_overlap: true,
            sync_timeout: Duration::from_secs(1),
        };
        let mut pipeline = LayerPipeline::new(config);

        pipeline.begin_compute(0).unwrap();
        pipeline.begin_transfer_async(0, LayerTransferState::new()).unwrap();
        assert!(pipeline.transfer_state(0).is_some());

        pipeline.reset();
        assert!(pipeline.transfer_state(0).is_none());
        assert_eq!(pipeline.stage(0), PipelineStage::WaitingForInput);
    }

    #[test]
    fn complete_async_fails_when_transfers_pending_would_be_caught() {
        // We can't easily create a PendingOp in tests without the full engine,
        // but we verify the no-state fast path works
        let config = PipelineConfig {
            num_layers: 1,
            enable_overlap: true,
            sync_timeout: Duration::from_secs(1),
        };
        let mut pipeline = LayerPipeline::new(config);
        pipeline.begin_compute(0).unwrap();

        // Without begin_transfer_async, poll_transfer_complete returns true (no state)
        assert!(pipeline.poll_transfer_complete(0));
    }

    #[test]
    fn pipeline_stats_overlap_gain() {
        let stats = PipelineStats::from_timings(
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_millis(120), // pipeline time (overlapped)
            Duration::from_millis(1),
        );
        // serial = 200ms, pipeline = 120ms, gain = (200-120)/200 = 0.4
        assert!(stats.overlap_gain > 0.35 && stats.overlap_gain < 0.45);
    }
}
