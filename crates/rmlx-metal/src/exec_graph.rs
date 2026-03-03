//! Async execution graph with GPU-side dependency tracking.
//!
//! Uses `MTLSharedEvent` signal/wait to chain command buffer batches on the GPU
//! side, eliminating CPU stalls between batches. The CPU only blocks once at the
//! very end of the forward pass.
//!
//! # Architecture
//!
//! ```text
//! [Batch 1: norm+projections] --signal(1)--> [Batch 2: RoPE] --signal(2)--> [Batch 3: SDPA]
//!                                                                               |
//! [Batch 6: down+residual] <--signal(5)-- [Batch 5: norm+ffn] <--signal(4)-- [Batch 4: O_proj]
//!                 |
//!          signal(6) → CPU wait (ONCE per forward pass)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let event = GpuEvent::new(device);
//! let mut graph = ExecGraph::new(queue, &event, 32);
//!
//! // Batch 1: encode ops, then submit
//! let enc = graph.encoder();
//! // ... encode norm + projections ...
//! enc.end_encoding();
//! graph.end_encoder();
//! let t1 = graph.submit_batch();
//!
//! // Batch 2: waits for batch 1, then encodes
//! graph.wait_for(t1);
//! let enc = graph.encoder();
//! // ... encode RoPE ...
//! enc.end_encoding();
//! graph.end_encoder();
//! let t2 = graph.submit_batch();
//!
//! // ... more batches ...
//!
//! // Final sync: CPU blocks once
//! graph.sync()?;
//! ```

use std::time::Duration;

use metal::CommandQueue;

use crate::batcher::CommandBatcher;
use crate::event::{EventError, GpuEvent};

/// Token representing a submitted batch in the execution graph.
///
/// Used to express dependencies between batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventToken {
    value: u64,
}

impl EventToken {
    /// The event signal value this token represents.
    pub fn value(&self) -> u64 {
        self.value
    }
}

/// Execution graph that chains command buffer batches via GPU-side events.
///
/// Wraps a `CommandBatcher` and a `GpuEvent` to provide:
/// - Batched encoding (multiple ops per command buffer)
/// - GPU-side dependency chains (no CPU stalls between batches)
/// - Single CPU sync point at the end
pub struct ExecGraph<'q, 'e> {
    batcher: CommandBatcher<'q>,
    event: &'e GpuEvent,
    counter: u64,
    total_batches: usize,
    sync_timeout: Duration,
}

impl<'q, 'e> ExecGraph<'q, 'e> {
    /// Create a new execution graph.
    ///
    /// - `queue`: The Metal command queue for dispatch
    /// - `event`: Shared event for GPU-side synchronization
    /// - `max_encoders_per_batch`: Max encoders per command buffer
    pub fn new(
        queue: &'q CommandQueue,
        event: &'e GpuEvent,
        max_encoders_per_batch: usize,
    ) -> Self {
        Self {
            batcher: CommandBatcher::new(queue, max_encoders_per_batch),
            event,
            counter: 0,
            total_batches: 0,
            sync_timeout: Duration::from_secs(10),
        }
    }

    /// Set the timeout for the final CPU sync.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.sync_timeout = timeout;
        self
    }

    /// Get a new compute command encoder in the current batch.
    ///
    /// The encoder is created on the current command buffer. Call
    /// `end_encoder()` after encoding.
    pub fn encoder(&mut self) -> &metal::ComputeCommandEncoderRef {
        self.batcher.encoder()
    }

    /// End the current compute command encoder.
    pub fn end_encoder(&mut self) {
        self.batcher.end_encoder();
    }

    /// Get the current command buffer for direct encoding.
    pub fn command_buffer(&mut self) -> &metal::CommandBufferRef {
        self.batcher.command_buffer()
    }

    /// Submit the current batch, signaling the event.
    ///
    /// Returns an `EventToken` that subsequent batches can wait on.
    /// The command buffer is committed but the CPU does not block.
    pub fn submit_batch(&mut self) -> EventToken {
        self.counter += 1;
        let value = self.counter;
        self.batcher.flush_signal(self.event, value);
        self.total_batches += 1;
        EventToken { value }
    }

    /// Make the next batch wait for a previous batch's completion.
    ///
    /// Encodes a GPU-side wait on the event. The next batch's command buffer
    /// will not start executing until the event reaches the token's value.
    pub fn wait_for(&mut self, token: EventToken) {
        self.batcher.begin_waiting(self.event, token.value);
    }

    /// Submit the current batch and immediately wait for a token.
    ///
    /// Convenience for the common pattern of `submit_batch()` + `wait_for()`.
    pub fn submit_and_wait(&mut self, wait_token: EventToken) -> EventToken {
        let token = self.submit_batch();
        self.wait_for(wait_token);
        token
    }

    /// CPU-side sync: block until all submitted batches complete.
    ///
    /// This is the **only** point where the CPU blocks on GPU work.
    /// Should be called once at the end of a forward pass.
    pub fn sync(&mut self) -> Result<Duration, EventError> {
        // Flush any pending work
        if self.batcher.has_pending() {
            self.counter += 1;
            self.batcher.flush_signal(self.event, self.counter);
            self.total_batches += 1;
        }

        if self.counter == 0 {
            return Ok(Duration::ZERO);
        }

        self.event.cpu_wait(self.counter, self.sync_timeout)
    }

    /// Sync and reset for the next forward pass.
    pub fn sync_and_reset(&mut self) -> Result<Duration, EventError> {
        let elapsed = self.sync()?;
        self.reset();
        Ok(elapsed)
    }

    /// Reset the graph for a new forward pass.
    ///
    /// Resets the event counter and batch state. Does NOT flush pending work.
    pub fn reset(&mut self) {
        self.counter = 0;
        self.total_batches = 0;
        self.event.reset();
    }

    /// Total batches submitted in this graph.
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }

    /// Current event counter value.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// Access the underlying batcher for stats.
    pub fn batcher(&self) -> &CommandBatcher<'q> {
        &self.batcher
    }

    /// Access the underlying queue.
    pub fn queue(&self) -> &CommandQueue {
        self.batcher.queue()
    }

    /// Access the event.
    pub fn event(&self) -> &GpuEvent {
        self.event
    }
}

/// Statistics snapshot from an ExecGraph.
#[derive(Debug, Clone)]
pub struct ExecGraphStats {
    pub total_batches: usize,
    pub total_cbs: usize,
    pub total_encoders: usize,
    pub encoders_per_cb: f64,
}

impl ExecGraphStats {
    pub fn from_graph(graph: &ExecGraph<'_, '_>) -> Self {
        let batcher_stats = crate::batcher::BatcherStats::from_batcher(graph.batcher());
        Self {
            total_batches: graph.total_batches(),
            total_cbs: batcher_stats.total_cbs,
            total_encoders: batcher_stats.total_encoders,
            encoders_per_cb: batcher_stats.encoders_per_cb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_graph_basic_lifecycle() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        assert_eq!(graph.total_batches(), 0);
        assert_eq!(graph.counter(), 0);
    }

    #[test]
    fn exec_graph_submit_batch() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Encode a no-op and submit
        let enc = graph.encoder();
        enc.end_encoding();
        graph.end_encoder();
        let t1 = graph.submit_batch();

        assert_eq!(t1.value(), 1);
        assert_eq!(graph.total_batches(), 1);
        assert_eq!(graph.counter(), 1);
    }

    #[test]
    fn exec_graph_chained_batches() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Batch 1
        let enc = graph.encoder();
        enc.end_encoding();
        graph.end_encoder();
        let t1 = graph.submit_batch();

        // Batch 2 waits for batch 1
        graph.wait_for(t1);
        let enc = graph.encoder();
        enc.end_encoding();
        graph.end_encoder();
        let t2 = graph.submit_batch();

        // Batch 3 waits for batch 2
        graph.wait_for(t2);
        let enc = graph.encoder();
        enc.end_encoding();
        graph.end_encoder();
        let _t3 = graph.submit_batch();

        assert_eq!(graph.total_batches(), 3);

        // Final sync
        let elapsed = graph.sync().expect("sync should succeed");
        assert!(elapsed.as_secs() < 5);
    }

    #[test]
    fn exec_graph_sync_and_reset() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        let enc = graph.encoder();
        enc.end_encoding();
        graph.end_encoder();
        let _token = graph.submit_batch();

        graph.sync_and_reset().expect("sync should succeed");
        assert_eq!(graph.counter(), 0);
        assert_eq!(graph.total_batches(), 0);
    }

    #[test]
    fn exec_graph_sync_empty() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Sync with nothing submitted should be instant
        let elapsed = graph.sync().expect("sync should succeed");
        assert_eq!(elapsed, Duration::ZERO);
    }

    #[test]
    fn exec_graph_stats() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let event = GpuEvent::new(&device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Batch 1: 3 encoders
        for _ in 0..3 {
            let enc = graph.encoder();
            enc.end_encoding();
            graph.end_encoder();
        }
        let _ = graph.submit_batch();

        // Batch 2: 2 encoders
        for _ in 0..2 {
            let enc = graph.encoder();
            enc.end_encoding();
            graph.end_encoder();
        }
        let _ = graph.submit_batch();

        graph.sync().expect("sync");

        let stats = ExecGraphStats::from_graph(&graph);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_encoders, 5);
    }
}
