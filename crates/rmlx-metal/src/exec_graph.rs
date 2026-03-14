//! Async execution graph with GPU-side dependency tracking.
//!
//! Uses `MTLSharedEvent` signal/wait to chain command buffer batches on the GPU
//! side, eliminating CPU stalls between batches. The CPU only blocks once at the
//! very end of the forward pass.
//!
//! # Architecture
//!
//! On a single `MTLCommandQueue`, command buffers execute in FIFO order.
//! Each `submit_batch()` commits the current CB and starts a new one.
//! GPU FIFO ordering guarantees that CB N completes before CB N+1 begins
//! execution, so explicit `wait_for()` is typically not needed between
//! sequential batches on the same queue. The `wait_for()` API remains
//! available for cases requiring cross-queue synchronization.
//!
//! ```text
//! [Batch 1: norm+proj] --commit--> [Batch 2: RoPE] --commit--> [Batch 3: SDPA+O_proj]
//!                                                                       |
//! [Batch 5: FFN] <--commit-- [Batch 4: residual+norm]                   |
//!        |                                                              |
//!  GPU FIFO ordering ensures sequential execution across all batches
//!        |
//!  sync() -> CPU wait (ONCE per forward pass)
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
//! let pass = ComputePass::new(&enc);
//! // ... encode norm + projections ...
//! pass.end();
//! graph.end_encoder();
//! let _t1 = graph.submit_batch();
//!
//! // Batch 2: GPU FIFO ensures batch 1 completes first
//! let enc = graph.encoder();
//! let pass = ComputePass::new(&enc);
//! // ... encode RoPE ...
//! pass.end();
//! graph.end_encoder();
//! let _t2 = graph.submit_batch();
//!
//! // ... more batches ...
//!
//! // Final sync: CPU blocks once
//! graph.sync()?;
//! ```

use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::batcher::CommandBatcher;
use crate::event::{EventError, GpuEvent};
use crate::types::*;

#[cfg(feature = "metal4")]
use crate::metal4::{CommandAllocator, CommandQueue4, Mtl4CommandBuffer};

/// Configuration for GPU memory backpressure in ExecGraph.
///
/// When the caller reports allocation sizes via [`ExecGraph::add_tracked_bytes`],
/// the graph can automatically flush and sync to relieve memory pressure once
/// the tracked total exceeds configurable thresholds.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Allocated bytes threshold that triggers a warning (default: 80 GB).
    pub warn_threshold_bytes: u64,
    /// Allocated bytes threshold that forces CB commit + sync (default: 100 GB).
    pub force_commit_bytes: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            warn_threshold_bytes: 80 * 1024 * 1024 * 1024, // 80 GB
            force_commit_bytes: 100 * 1024 * 1024 * 1024,  // 100 GB
        }
    }
}

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
    /// Last committed command buffer, retained for `wait_until_completed` in `sync()`.
    /// `MTLSharedEvent.signaledValue` polling is unreliable on Apple Silicon when
    /// multiple command buffers are in flight across iterations; using the CB's own
    /// completion mechanism is more robust.
    last_cb: Option<MtlCB>,
    /// Memory backpressure configuration.
    memory_config: MemoryConfig,
    /// Caller-reported GPU allocation total in bytes. Updated via
    /// [`add_tracked_bytes`] / [`sub_tracked_bytes`] since metal-rs does not
    /// expose `MTLDevice.currentAllocatedSize`.
    tracked_bytes: u64,
    /// Number of ops (dispatches) recorded since the last actual CB commit.
    /// Used by chunked pipeline to batch multiple layers into fewer CBs.
    ops_since_submit: usize,
    /// Maximum ops per command buffer before auto-submit. When set to
    /// `usize::MAX` (default), every `submit_batch()` commits immediately
    /// (legacy behavior). Lower values (e.g. 50) enable chunked pipeline
    /// where `submit_batch()` becomes a no-op until the threshold is reached.
    max_ops_per_batch: usize,
}

impl<'q, 'e> ExecGraph<'q, 'e> {
    /// Create a new execution graph.
    ///
    /// - `queue`: The Metal command queue for dispatch
    /// - `event`: Shared event for GPU-side synchronization
    /// - `max_encoders_per_batch`: Max encoders per command buffer
    pub fn new(
        queue: &'q ProtocolObject<dyn MTLCommandQueue>,
        event: &'e GpuEvent,
        max_encoders_per_batch: usize,
    ) -> Self {
        Self {
            batcher: CommandBatcher::new(queue, max_encoders_per_batch),
            event,
            counter: 0,
            total_batches: 0,
            sync_timeout: Duration::from_secs(10),
            last_cb: None,
            memory_config: MemoryConfig::default(),
            tracked_bytes: 0,
            ops_since_submit: 0,
            max_ops_per_batch: usize::MAX,
        }
    }

    /// Set the timeout for the final CPU sync.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.sync_timeout = timeout;
        self
    }

    /// Set a custom memory backpressure configuration.
    pub fn with_memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = config;
        self
    }

    /// Set the maximum number of ops per command buffer for chunked pipeline.
    ///
    /// When set (e.g. to 50), `submit_batch()` becomes a soft hint that only
    /// actually commits the CB when the ops threshold is reached. This batches
    /// multiple transformer layers into fewer, larger command buffers, reducing
    /// CB creation overhead while maintaining CPU-GPU overlap.
    ///
    /// Default is `usize::MAX` (every `submit_batch()` commits immediately).
    pub fn with_max_ops_per_batch(mut self, max_ops: usize) -> Self {
        self.max_ops_per_batch = max_ops.max(1);
        self
    }

    /// Set the max ops per batch (non-builder variant).
    pub fn set_max_ops_per_batch(&mut self, max_ops: usize) {
        self.max_ops_per_batch = max_ops.max(1);
    }

    /// Record that `count` dispatch ops have been encoded into the current CB.
    ///
    /// If the accumulated ops exceed `max_ops_per_batch`, the current CB is
    /// automatically committed (equivalent to `submit_batch()`).
    ///
    /// Returns `Some(EventToken)` if an auto-submit occurred, `None` otherwise.
    pub fn record_ops(&mut self, count: usize) -> Option<EventToken> {
        self.ops_since_submit += count;
        if self.ops_since_submit >= self.max_ops_per_batch {
            Some(self.force_submit_batch())
        } else {
            None
        }
    }

    /// Current ops accumulated since last commit.
    pub fn ops_since_submit(&self) -> usize {
        self.ops_since_submit
    }

    /// Report that `bytes` have been allocated on the GPU.
    ///
    /// Callers should invoke this after allocating Metal buffers so that
    /// [`check_memory_pressure`] can trigger backpressure when appropriate.
    pub fn add_tracked_bytes(&mut self, bytes: u64) {
        self.tracked_bytes = self.tracked_bytes.saturating_add(bytes);
    }

    /// Report that `bytes` have been freed on the GPU.
    pub fn sub_tracked_bytes(&mut self, bytes: u64) {
        self.tracked_bytes = self.tracked_bytes.saturating_sub(bytes);
    }

    /// Current tracked GPU allocation in bytes.
    pub fn tracked_bytes(&self) -> u64 {
        self.tracked_bytes
    }

    /// Check current tracked GPU memory and force a commit + sync if above
    /// the force threshold.
    ///
    /// Returns `true` if a forced sync occurred.
    pub fn check_memory_pressure(&mut self) -> bool {
        if self.tracked_bytes > self.memory_config.force_commit_bytes {
            let _ = self.submit_batch();
            let _ = self.sync();
            return true;
        }
        // warn_threshold_bytes exceeded but below force — caller may choose
        // to act on the return value or query `tracked_bytes()` directly.
        false
    }

    /// Whether tracked bytes exceed the warning threshold.
    pub fn is_memory_warning(&self) -> bool {
        self.tracked_bytes > self.memory_config.warn_threshold_bytes
    }

    /// Get a new compute command encoder in the current batch.
    ///
    /// The encoder is created on the current command buffer. Call
    /// `end_encoder()` after encoding.
    pub fn encoder(&mut self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.batcher.encoder()
    }

    /// End the current compute command encoder.
    pub fn end_encoder(&mut self) {
        self.batcher.end_encoder();
    }

    /// Get the current command buffer for direct encoding.
    pub fn command_buffer(&mut self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        self.batcher.command_buffer()
    }

    /// Submit the current batch, signaling the event.
    ///
    /// Returns an `EventToken` that subsequent batches can wait on.
    /// The command buffer is committed but the CPU does not block.
    ///
    /// When `max_ops_per_batch` is set (via [`with_max_ops_per_batch`]), this
    /// method becomes a **soft hint**: the CB is only actually committed when
    /// the accumulated ops reach the threshold. Use [`force_submit_batch`] to
    /// unconditionally commit regardless of the threshold.
    ///
    /// If there is no pending work (no encoders were created since the last
    /// submit), this returns the previous token unchanged without incrementing
    /// the counter or submitting an empty command buffer.
    pub fn submit_batch(&mut self) -> EventToken {
        if !self.batcher.has_pending() {
            return EventToken {
                value: self.counter,
            };
        }
        // Chunked pipeline: skip commit if below ops threshold
        if self.max_ops_per_batch < usize::MAX && self.ops_since_submit < self.max_ops_per_batch {
            return EventToken {
                value: self.counter,
            };
        }
        self.force_submit_batch()
    }

    /// Unconditionally submit the current batch, ignoring the ops threshold.
    ///
    /// Use this when you need a hard CB boundary (e.g., cross-queue sync).
    pub fn force_submit_batch(&mut self) -> EventToken {
        if !self.batcher.has_pending() {
            return EventToken {
                value: self.counter,
            };
        }
        self.counter += 1;
        let value = self.counter;
        self.last_cb = self.batcher.flush_signal(self.event, value);
        self.total_batches += 1;
        self.ops_since_submit = 0;
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
        // Flush any pending work (unconditionally, ignoring ops threshold)
        if self.batcher.has_pending() {
            self.counter += 1;
            self.last_cb = self.batcher.flush_signal(self.event, self.counter);
            self.total_batches += 1;
            self.ops_since_submit = 0;
        }

        if self.counter == 0 {
            return Ok(Duration::ZERO);
        }

        // Use CB's own wait_until_completed instead of SharedEvent polling.
        // MTLSharedEvent.signaledValue polling is unreliable on Apple Silicon
        // when GPU has prior in-flight work from previous iterations.
        let start = std::time::Instant::now();
        if let Some(ref cb) = self.last_cb {
            cb.waitUntilCompleted();
        }
        self.last_cb = None;
        Ok(start.elapsed())
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
        self.last_cb = None;
        self.tracked_bytes = 0;
        self.ops_since_submit = 0;
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
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
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

// ===========================================================================
// Metal 4 execution graph
// ===========================================================================

/// Metal 4 execution graph using explicit command buffer lifecycle.
///
/// Instead of creating individual command buffers from the queue and committing
/// them one by one (Metal 3 path), this graph:
///
/// 1. Allocates CBs from a [`CommandAllocator`] (memory reuse via `reset()`).
/// 2. Uses explicit `begin(allocator)` / `end()` around encoding.
/// 3. Collects completed CBs and batch-commits them via [`CommandQueue4::commit_batch`].
/// 4. Calls `allocator.reset()` between decode iterations.
///
/// The caller chooses between `ExecGraph` (Metal 3) and `ExecGraph4` (Metal 4)
/// based on runtime device capability detection.
#[cfg(feature = "metal4")]
pub struct ExecGraph4 {
    allocator: CommandAllocator,
    queue4: CommandQueue4,
    event: GpuEvent,
    /// Device reference for creating new command buffers.
    device: MtlDevice,
    /// Command buffers that have been `end()`-ed and are ready for batch commit.
    pending_cbs: Vec<Mtl4CommandBuffer>,
    /// The currently encoding command buffer (between `begin()` and `end()`).
    current_cb: Option<Mtl4CommandBuffer>,
    /// Last committed command buffer, retained for `waitUntilCompleted` in `sync()`.
    /// `MTLSharedEvent.signaledValue` polling is unreliable on Apple Silicon when
    /// multiple command buffers are in flight across iterations; using the CB's own
    /// completion mechanism is more robust.
    last_cb: Option<Mtl4CommandBuffer>,
    counter: u64,
    total_batches: usize,
    sync_timeout: Duration,
    /// Number of ops recorded since the last batch commit.
    ops_since_submit: usize,
    /// Maximum ops per command buffer before auto-submit.
    max_ops_per_batch: usize,
}

#[cfg(feature = "metal4")]
impl ExecGraph4 {
    /// Create a new Metal 4 execution graph.
    ///
    /// Returns `None` if the device does not support Metal 4 or if
    /// allocator/queue creation fails.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let allocator = CommandAllocator::new(device)?;
        let queue4 = CommandQueue4::new(device)?;
        let event = GpuEvent::new(device);
        Some(Self {
            allocator,
            queue4,
            event,
            device: crate::types::retain_proto(device),
            pending_cbs: Vec::new(),
            current_cb: None,
            last_cb: None,
            counter: 0,
            total_batches: 0,
            sync_timeout: Duration::from_secs(10),
            ops_since_submit: 0,
            max_ops_per_batch: usize::MAX,
        })
    }

    /// Set the timeout for the final CPU sync.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.sync_timeout = timeout;
        self
    }

    /// Set the maximum ops per command buffer for chunked pipeline.
    pub fn with_max_ops_per_batch(mut self, max_ops: usize) -> Self {
        self.max_ops_per_batch = max_ops.max(1);
        self
    }

    /// Set the max ops per batch (non-builder variant).
    pub fn set_max_ops_per_batch(&mut self, max_ops: usize) {
        self.max_ops_per_batch = max_ops.max(1);
    }

    /// Begin a new command buffer for encoding.
    ///
    /// If there is already a current CB in encoding state, this is a no-op
    /// and returns the existing encoder-ready CB. Otherwise, allocates a new
    /// CB from the allocator and calls `begin()`.
    fn ensure_current_cb(&mut self) -> &Mtl4CommandBuffer {
        if self.current_cb.is_none() {
            let cb = self
                .allocator
                .new_command_buffer(&self.device)
                .expect("Metal 4: failed to allocate command buffer");
            cb.begin(&self.allocator);
            self.current_cb = Some(cb);
        }
        self.current_cb.as_ref().unwrap()
    }

    /// Get a compute command encoder on the current Metal 4 command buffer.
    ///
    /// Automatically begins a new CB if none is active.
    pub fn encoder(&mut self) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
        self.ensure_current_cb();
        self.current_cb
            .as_ref()
            .unwrap()
            .compute_encoder()
            .expect("Metal 4: failed to create compute encoder")
    }

    /// End the current compute command encoder.
    ///
    /// Note: With Metal 4 explicit encoding, the caller is responsible for
    /// calling `endEncoding()` on the encoder returned by [`encoder()`].
    /// This method is provided for API symmetry with `ExecGraph`.
    pub fn end_encoder(&mut self) {
        // Metal 4 encoders are ended by the caller via endEncoding().
        // This is a no-op but maintains API parity with ExecGraph.
    }

    /// Finalize the current command buffer (call `end()`) and move it to
    /// the pending list for batch commit.
    fn finalize_current_cb(&mut self) {
        if let Some(cb) = self.current_cb.take() {
            cb.end();
            self.pending_cbs.push(cb);
        }
    }

    /// Record that `count` dispatch ops have been encoded.
    ///
    /// If accumulated ops exceed `max_ops_per_batch`, auto-submits.
    pub fn record_ops(&mut self, count: usize) -> Option<EventToken> {
        self.ops_since_submit += count;
        if self.ops_since_submit >= self.max_ops_per_batch {
            Some(self.force_submit_batch())
        } else {
            None
        }
    }

    /// Current ops accumulated since last commit.
    pub fn ops_since_submit(&self) -> usize {
        self.ops_since_submit
    }

    /// Submit the current batch of command buffers.
    ///
    /// Finalizes the current CB (if any), then batch-commits all pending CBs
    /// via `CommandQueue4::commit_batch()`.
    ///
    /// When `max_ops_per_batch` is set, this is a soft hint that only commits
    /// when the threshold is reached.
    pub fn submit_batch(&mut self) -> EventToken {
        if self.current_cb.is_none() && self.pending_cbs.is_empty() {
            return EventToken {
                value: self.counter,
            };
        }
        // Chunked pipeline: skip commit if below ops threshold
        if self.max_ops_per_batch < usize::MAX && self.ops_since_submit < self.max_ops_per_batch {
            return EventToken {
                value: self.counter,
            };
        }
        self.force_submit_batch()
    }

    /// Unconditionally submit all pending command buffers.
    pub fn force_submit_batch(&mut self) -> EventToken {
        self.finalize_current_cb();

        if self.pending_cbs.is_empty() {
            return EventToken {
                value: self.counter,
            };
        }

        self.counter += 1;
        let value = self.counter;

        // Batch commit all pending CBs
        let cb_refs: Vec<&Mtl4CommandBuffer> = self.pending_cbs.iter().collect();
        self.queue4.commit_batch(&cb_refs);

        // Signal event after commit for cross-queue sync
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(self.event.raw());
        self.queue4.signal_event(event, value);

        // Save the last committed CB for waitUntilCompleted in sync().
        self.last_cb = self.pending_cbs.pop();
        self.pending_cbs.clear();
        self.total_batches += 1;
        self.ops_since_submit = 0;

        EventToken { value }
    }

    /// CPU-side sync: block until all submitted batches complete.
    ///
    /// Uses the last committed CB's `waitUntilCompleted` instead of SharedEvent
    /// polling. `MTLSharedEvent.signaledValue` polling is unreliable on Apple
    /// Silicon when GPU has prior in-flight work from previous iterations.
    pub fn sync(&mut self) -> Result<Duration, EventError> {
        // Flush any pending work
        if self.current_cb.is_some() || !self.pending_cbs.is_empty() {
            self.finalize_current_cb();
            if !self.pending_cbs.is_empty() {
                self.counter += 1;
                let cb_refs: Vec<&Mtl4CommandBuffer> = self.pending_cbs.iter().collect();
                self.queue4.commit_batch(&cb_refs);
                let event: &ProtocolObject<dyn MTLEvent> =
                    ProtocolObject::from_ref(self.event.raw());
                self.queue4.signal_event(event, self.counter);
                // Save the last committed CB for waitUntilCompleted.
                self.last_cb = self.pending_cbs.pop();
                self.pending_cbs.clear();
                self.total_batches += 1;
                self.ops_since_submit = 0;
            }
        }

        if self.counter == 0 {
            return Ok(Duration::ZERO);
        }

        // Use CB's own waitUntilCompleted instead of SharedEvent polling.
        let start = std::time::Instant::now();
        if let Some(ref cb) = self.last_cb {
            cb.as_legacy_cb().waitUntilCompleted();
        }
        self.last_cb = None;
        Ok(start.elapsed())
    }

    /// Sync and reset for the next forward pass.
    pub fn sync_and_reset(&mut self) -> Result<Duration, EventError> {
        let elapsed = self.sync()?;
        self.reset();
        Ok(elapsed)
    }

    /// Reset the graph for a new forward pass.
    ///
    /// Resets the allocator (reclaims encoding memory), event counter,
    /// and all batch state. The caller must ensure all prior GPU work
    /// has completed before calling this.
    pub fn reset(&mut self) {
        // SAFETY: reset() is only called after sync() which waits for all GPU
        // work to complete via waitUntilCompleted on the last committed CB.
        unsafe { self.allocator.reset() };
        self.counter = 0;
        self.total_batches = 0;
        self.current_cb = None;
        self.last_cb = None;
        self.pending_cbs.clear();
        self.ops_since_submit = 0;
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

    /// Access the underlying allocator.
    pub fn allocator(&self) -> &CommandAllocator {
        &self.allocator
    }

    /// Access the underlying Metal 4 queue.
    pub fn queue4(&self) -> &CommandQueue4 {
        &self.queue4
    }

    /// Access the event.
    pub fn event(&self) -> &GpuEvent {
        &self.event
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn test_device() -> &'static MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests"))
    }

    #[test]
    fn exec_graph_basic_lifecycle() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let graph = ExecGraph::new(&queue, &event, 32);

        assert_eq!(graph.total_batches(), 0);
        assert_eq!(graph.counter(), 0);
    }

    #[test]
    fn exec_graph_submit_batch() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Encode a no-op and submit
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let t1 = graph.submit_batch();

        assert_eq!(t1.value(), 1);
        assert_eq!(graph.total_batches(), 1);
        assert_eq!(graph.counter(), 1);
    }

    #[test]
    fn exec_graph_chained_batches() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Batch 1
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let t1 = graph.submit_batch();

        // Batch 2 waits for batch 1
        graph.wait_for(t1);
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let t2 = graph.submit_batch();

        // Batch 3 waits for batch 2
        graph.wait_for(t2);
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let _t3 = graph.submit_batch();

        assert_eq!(graph.total_batches(), 3);

        // Final sync
        let elapsed = graph.sync().expect("sync should succeed");
        assert!(elapsed.as_secs() < 5);
    }

    #[test]
    fn exec_graph_sync_and_reset() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let _token = graph.submit_batch();

        graph.sync_and_reset().expect("sync should succeed");
        assert_eq!(graph.counter(), 0);
        assert_eq!(graph.total_batches(), 0);
    }

    #[test]
    fn exec_graph_sync_empty() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Sync with nothing submitted should be instant
        let elapsed = graph.sync().expect("sync should succeed");
        assert_eq!(elapsed, Duration::ZERO);
    }

    #[test]
    fn submit_batch_no_pending_returns_previous_token() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // submit_batch with no pending work should return token with value 0
        // and not increment the counter or total_batches
        let t = graph.submit_batch();
        assert_eq!(t.value(), 0);
        assert_eq!(graph.counter(), 0);
        assert_eq!(graph.total_batches(), 0);

        // Now do real work and submit
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let t1 = graph.submit_batch();
        assert_eq!(t1.value(), 1);
        assert_eq!(graph.total_batches(), 1);

        // Another empty submit should return the previous token (value 1)
        let t2 = graph.submit_batch();
        assert_eq!(t2.value(), 1);
        assert_eq!(graph.total_batches(), 1); // unchanged
        assert_eq!(graph.counter(), 1); // unchanged
    }

    #[test]
    fn wait_for_empty_submit_completes_immediately() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // submit_batch with no pending work
        let t = graph.submit_batch();
        assert_eq!(t.value(), 0);

        // sync should complete immediately since counter is 0
        let elapsed = graph.sync().expect("sync should succeed");
        assert_eq!(elapsed, Duration::ZERO);
    }

    #[test]
    fn exec_graph_stats() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        // Batch 1: 3 encoders
        for _ in 0..3 {
            let enc = graph.encoder();
            enc.endEncoding();
            graph.end_encoder();
        }
        let _ = graph.submit_batch();

        // Batch 2: 2 encoders
        for _ in 0..2 {
            let enc = graph.encoder();
            enc.endEncoding();
            graph.end_encoder();
        }
        let _ = graph.submit_batch();

        graph.sync().expect("sync");

        let stats = ExecGraphStats::from_graph(&graph);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_encoders, 5);
    }

    #[test]
    fn chunked_pipeline_defers_submit() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        // Set max_ops_per_batch to 20 — submit_batch() won't commit until 20 ops
        let mut graph = ExecGraph::new(&queue, &event, 64).with_max_ops_per_batch(20);

        // Simulate 2 "layers" of 9 ops each (total 18 < 20)
        for _ in 0..2 {
            let enc = graph.encoder();
            enc.endEncoding();
            graph.end_encoder();
            // submit_batch should be a no-op (below threshold)
            let t = graph.submit_batch();
            assert_eq!(t.value(), 0); // no commit happened
            graph.record_ops(9);
        }
        // 18 ops accumulated, 0 CBs committed
        assert_eq!(graph.total_batches(), 0);
        assert_eq!(graph.ops_since_submit(), 18);

        // One more "layer" (9 ops) — record_ops should auto-submit at 27 >= 20
        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();
        let auto_token = graph.record_ops(9);
        assert!(auto_token.is_some()); // auto-submit triggered
        assert_eq!(graph.total_batches(), 1);
        assert_eq!(graph.ops_since_submit(), 0); // reset after submit

        graph.sync().expect("sync");
    }

    #[test]
    fn chunked_pipeline_force_submit_ignores_threshold() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 64).with_max_ops_per_batch(50);

        let enc = graph.encoder();
        enc.endEncoding();
        graph.end_encoder();

        // submit_batch is a no-op (0 ops < 50)
        let t = graph.submit_batch();
        assert_eq!(t.value(), 0);
        assert_eq!(graph.total_batches(), 0);

        // force_submit_batch always commits
        let t = graph.force_submit_batch();
        assert_eq!(t.value(), 1);
        assert_eq!(graph.total_batches(), 1);

        graph.sync().expect("sync");
    }

    #[test]
    fn chunked_pipeline_sync_flushes_remaining() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 64).with_max_ops_per_batch(100);

        // Encode work but never hit threshold
        for _ in 0..5 {
            let enc = graph.encoder();
            enc.endEncoding();
            graph.end_encoder();
            graph.record_ops(9);
        }
        assert_eq!(graph.total_batches(), 0); // nothing committed yet

        // sync() should flush remaining work
        let elapsed = graph.sync().expect("sync");
        assert!(elapsed.as_secs() < 5);
        assert_eq!(graph.total_batches(), 1); // flushed in sync
    }
}
