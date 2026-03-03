//! Metal command buffer and encoder abstractions
//!
//! Provides [`CommandBufferManager`] for batching multiple dispatches into a
//! single command buffer, and [`BarrierTracker`] for inserting memory barriers
//! between dependent kernel dispatches.

use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// M1 — auto-commit thresholds
// ---------------------------------------------------------------------------
const MAX_OPS_PER_BATCH: usize = 64;
const MAX_BYTES_PER_BATCH: u64 = 16 * 1024 * 1024; // 16 MB

/// Manages command buffer batching so that multiple dispatches are encoded into
/// a single command buffer, amortising the ~10-50 µs overhead of creating a new
/// one per dispatch.
///
/// The manager tracks the current command buffer, the number of ops batched,
/// and the cumulative bytes dispatched.  It auto-commits when either threshold
/// is reached.
pub struct CommandBufferManager<'q> {
    queue: &'q metal::CommandQueue,
    current_cb: Option<&'q CommandBufferRef>,
    ops_batched: usize,
    bytes_dispatched: u64,
    /// Track GPU errors reported via completion handlers.
    error_store: std::sync::Arc<GpuErrorStore>,
}

impl<'q> CommandBufferManager<'q> {
    /// Create a new manager that draws command buffers from `queue`.
    pub fn new(queue: &'q metal::CommandQueue) -> Self {
        Self {
            queue,
            current_cb: None,
            ops_batched: 0,
            bytes_dispatched: 0,
            error_store: std::sync::Arc::new(GpuErrorStore::new()),
        }
    }

    /// Return the current command buffer, creating one if necessary.
    pub fn get_or_create_buffer(&mut self) -> &CommandBufferRef {
        if self.current_cb.is_none() {
            let cb = self.queue.new_command_buffer();
            self.current_cb = Some(cb);
        }
        self.current_cb.unwrap()
    }

    /// Create a compute command encoder on the current (or new) command buffer.
    pub fn get_or_create_encoder(&mut self) -> &ComputeCommandEncoderRef {
        let cb = self.get_or_create_buffer();
        cb.new_compute_command_encoder()
    }

    /// Record that one dispatch of `bytes` was encoded.
    /// If the batch thresholds are reached, the command buffer is committed
    /// automatically.
    pub fn maybe_commit(&mut self, bytes: u64) {
        self.ops_batched += 1;
        self.bytes_dispatched += bytes;
        if self.ops_batched >= MAX_OPS_PER_BATCH || self.bytes_dispatched >= MAX_BYTES_PER_BATCH {
            self.force_commit();
        }
    }

    /// Commit the current command buffer immediately (if one exists), register
    /// a completion handler (M4), and reset the batch counters.
    pub fn force_commit(&mut self) {
        if let Some(cb) = self.current_cb.take() {
            register_completion_handler(cb, &self.error_store);
            cb.commit();
            self.ops_batched = 0;
            self.bytes_dispatched = 0;
        }
    }

    /// Number of dispatches batched into the current command buffer.
    pub fn ops_batched(&self) -> usize {
        self.ops_batched
    }

    /// Total bytes dispatched into the current command buffer.
    pub fn bytes_dispatched(&self) -> u64 {
        self.bytes_dispatched
    }

    /// Access the GPU error store to check for asynchronous errors.
    pub fn error_store(&self) -> &std::sync::Arc<GpuErrorStore> {
        &self.error_store
    }
}

impl Drop for CommandBufferManager<'_> {
    fn drop(&mut self) {
        // Flush any remaining work.
        self.force_commit();
    }
}

// ---------------------------------------------------------------------------
// M2 — BarrierTracker
// ---------------------------------------------------------------------------

/// Lightweight tracker that detects read-after-write hazards between
/// consecutive dispatches on the same compute command encoder.
///
/// When a buffer written by one dispatch is read (or written) by the next,
/// the tracker signals that a memory barrier is required.  Because the
/// `metal` crate (v0.31) only exposes `memory_barrier_with_resources` on
/// `ComputeCommandEncoderRef` (not a scope-based barrier), we use
/// the end-encoder / new-encoder strategy as an encoder-level barrier
/// when the resource-based API is not available.
pub struct BarrierTracker {
    /// Buffers that were bound as *output* (write) in the previous dispatch.
    previous_outputs: HashSet<u64>,
}

impl BarrierTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            previous_outputs: HashSet::new(),
        }
    }

    /// Record a set of input/output buffer addresses for the current dispatch
    /// and determine whether a barrier is needed before encoding.
    ///
    /// Returns `true` if any of the current `inputs` (or `outputs`) overlap
    /// with the previous dispatch's outputs, meaning a memory barrier (or
    /// encoder restart) is required.
    ///
    /// After the call, the tracker's state is updated so the *current*
    /// `outputs` become the "previous outputs" for the next dispatch.
    pub fn needs_barrier(&mut self, inputs: &[&Buffer], outputs: &[&Buffer]) -> bool {
        let mut needs = false;

        // Check if any current input was written in the previous dispatch.
        for buf in inputs {
            if self.previous_outputs.contains(&buf.gpu_address()) {
                needs = true;
                break;
            }
        }

        // Also check output-after-output (WAW hazard).
        if !needs {
            for buf in outputs {
                if self.previous_outputs.contains(&buf.gpu_address()) {
                    needs = true;
                    break;
                }
            }
        }

        // Update state: current outputs become previous outputs.
        self.previous_outputs.clear();
        for buf in outputs {
            self.previous_outputs.insert(buf.gpu_address());
        }

        needs
    }

    /// Reset the tracker (e.g. when a new command buffer begins).
    pub fn reset(&mut self) {
        self.previous_outputs.clear();
    }
}

impl Default for BarrierTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// M4 — GPU error checking via completion handler
// ---------------------------------------------------------------------------

/// Thread-safe store for GPU execution errors reported asynchronously via
/// Metal completion handlers.
pub struct GpuErrorStore {
    errors: std::sync::Mutex<Vec<GpuError>>,
}

/// A GPU execution error captured from a completed command buffer.
#[derive(Debug, Clone)]
pub struct GpuError {
    pub status: metal::MTLCommandBufferStatus,
    pub message: String,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU command buffer error (status {:?}): {}",
            self.status, self.message
        )
    }
}

impl std::error::Error for GpuError {}

impl GpuErrorStore {
    /// Create an empty error store.
    pub fn new() -> Self {
        Self {
            errors: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Push a new error (called from completion handler).
    pub fn push(&self, error: GpuError) {
        if let Ok(mut errors) = self.errors.lock() {
            errors.push(error);
        }
    }

    /// Drain and return all recorded errors.
    pub fn take_errors(&self) -> Vec<GpuError> {
        match self.errors.lock() {
            Ok(mut errors) => std::mem::take(&mut *errors),
            Err(_) => Vec::new(),
        }
    }

    /// Check if any errors have been recorded.
    pub fn has_errors(&self) -> bool {
        match self.errors.lock() {
            Ok(errors) => !errors.is_empty(),
            Err(_) => false,
        }
    }
}

impl Default for GpuErrorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Register a completion handler on `cb` that checks the command buffer status
/// after GPU execution and stores any error in `error_store`.
fn register_completion_handler(cb: &CommandBufferRef, error_store: &std::sync::Arc<GpuErrorStore>) {
    let store = std::sync::Arc::clone(error_store);
    let handler = block::ConcreteBlock::new(move |cmd_buf: &CommandBufferRef| {
        let status = cmd_buf.status();
        if status == metal::MTLCommandBufferStatus::Error {
            let msg = format!("command buffer completed with error status: {status:?}");
            store.push(GpuError {
                status,
                message: msg,
            });
        }
    });
    let handler = handler.copy();
    cb.add_completed_handler(&handler);
}

// ---------------------------------------------------------------------------
// Original helper — kept for backwards compatibility
// ---------------------------------------------------------------------------

/// Convenience function to encode a simple 1D compute dispatch.
///
/// Creates a compute command encoder on `cmd_buf`, sets the pipeline state,
/// binds each buffer at consecutive indices (0, 1, 2, ...) with the given
/// offsets, dispatches `num_threads` threads in a 1D grid, and ends encoding.
pub fn encode_compute_1d(
    cmd_buf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, u64)],
    num_threads: u64,
) {
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);

    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }

    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup_size = std::cmp::min(max_threads, num_threads);

    let grid_size = MTLSize::new(num_threads, 1, 1);
    let group_size = MTLSize::new(threadgroup_size, 1, 1);

    encoder.dispatch_threads(grid_size, group_size);
    encoder.end_encoding();
}

/// Encode a 1D compute dispatch with barrier tracking.
///
/// Like [`encode_compute_1d`], but uses a [`BarrierTracker`] to automatically
/// insert an encoder-level memory barrier (end + new encoder) when a buffer
/// written in the previous dispatch is read in this one.
///
/// `input_bufs` are the indices into `buffers` that are read-only inputs.
/// `output_bufs` are the indices into `buffers` that are written outputs.
///
/// When a barrier is needed, the current encoder is ended and a new one is
/// created, which acts as a full memory barrier on Metal.
pub fn encode_compute_1d_tracked(
    cmd_buf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, u64)],
    input_indices: &[usize],
    output_indices: &[usize],
    num_threads: u64,
    tracker: &mut BarrierTracker,
) {
    let inputs: Vec<&Buffer> = input_indices.iter().map(|&i| buffers[i].0).collect();
    let outputs: Vec<&Buffer> = output_indices.iter().map(|&i| buffers[i].0).collect();

    if tracker.needs_barrier(&inputs, &outputs) {
        // Insert a barrier by ending and creating a new encoder.
        // The encoder created here acts as a fresh synchronization point.
        // (We create a no-op encoder just to force the barrier, then create
        // the real one below.)
        let barrier_encoder = cmd_buf.new_compute_command_encoder();
        barrier_encoder.end_encoding();
    }

    encode_compute_1d(cmd_buf, pipeline, buffers, num_threads);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_tracker_no_conflict() {
        let device = metal::Device::system_default().unwrap();
        let a = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);
        let b = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);

        let mut tracker = BarrierTracker::new();
        // First dispatch: write to A
        let needs = tracker.needs_barrier(&[], &[&a]);
        assert!(!needs, "first dispatch should never need a barrier");

        // Second dispatch: read B (no conflict with A)
        let needs = tracker.needs_barrier(&[&b], &[]);
        assert!(!needs, "no RAW hazard expected");
    }

    #[test]
    fn test_barrier_tracker_raw_hazard() {
        let device = metal::Device::system_default().unwrap();
        let a = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);

        let mut tracker = BarrierTracker::new();
        // Dispatch 1: write A
        let needs = tracker.needs_barrier(&[], &[&a]);
        assert!(!needs);

        // Dispatch 2: read A -> RAW hazard
        let needs = tracker.needs_barrier(&[&a], &[]);
        assert!(needs, "RAW hazard on buffer A expected");
    }

    #[test]
    fn test_barrier_tracker_waw_hazard() {
        let device = metal::Device::system_default().unwrap();
        let a = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);

        let mut tracker = BarrierTracker::new();
        // Dispatch 1: write A
        let needs = tracker.needs_barrier(&[], &[&a]);
        assert!(!needs);

        // Dispatch 2: write A -> WAW hazard
        let needs = tracker.needs_barrier(&[], &[&a]);
        assert!(needs, "WAW hazard on buffer A expected");
    }

    #[test]
    fn test_barrier_tracker_reset() {
        let device = metal::Device::system_default().unwrap();
        let a = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);

        let mut tracker = BarrierTracker::new();
        let _ = tracker.needs_barrier(&[], &[&a]);
        tracker.reset();

        // After reset, no hazard should be reported
        let needs = tracker.needs_barrier(&[&a], &[]);
        assert!(!needs, "no hazard after reset");
    }

    #[test]
    fn test_gpu_error_store() {
        let store = GpuErrorStore::new();
        assert!(!store.has_errors());

        store.push(GpuError {
            status: metal::MTLCommandBufferStatus::Error,
            message: "test error".to_string(),
        });
        assert!(store.has_errors());

        let errors = store.take_errors();
        assert_eq!(errors.len(), 1);
        assert!(!store.has_errors());
    }

    #[test]
    fn test_command_buffer_manager_thresholds() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let mut mgr = CommandBufferManager::new(&queue);

        assert_eq!(mgr.ops_batched(), 0);
        assert_eq!(mgr.bytes_dispatched(), 0);

        // Record some ops below threshold
        mgr.get_or_create_buffer();
        mgr.maybe_commit(1024);
        assert_eq!(mgr.ops_batched(), 1);
        assert_eq!(mgr.bytes_dispatched(), 1024);

        // Force commit resets counters
        mgr.force_commit();
        assert_eq!(mgr.ops_batched(), 0);
        assert_eq!(mgr.bytes_dispatched(), 0);
    }

    #[test]
    fn test_command_buffer_manager_auto_commit_ops() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let mut mgr = CommandBufferManager::new(&queue);

        // Batch MAX_OPS_PER_BATCH dispatches — should auto-commit.
        mgr.get_or_create_buffer();
        for _ in 0..(MAX_OPS_PER_BATCH - 1) {
            mgr.maybe_commit(0);
        }
        assert_eq!(mgr.ops_batched(), MAX_OPS_PER_BATCH - 1);

        // This one triggers the auto-commit
        mgr.maybe_commit(0);
        assert_eq!(mgr.ops_batched(), 0);
    }

    #[test]
    fn test_command_buffer_manager_auto_commit_bytes() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let mut mgr = CommandBufferManager::new(&queue);

        mgr.get_or_create_buffer();
        mgr.maybe_commit(MAX_BYTES_PER_BATCH);
        // Should have auto-committed
        assert_eq!(mgr.ops_batched(), 0);
        assert_eq!(mgr.bytes_dispatched(), 0);
    }

    #[test]
    fn test_command_buffer_manager_completion_handler() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let mut mgr = CommandBufferManager::new(&queue);

        let store = std::sync::Arc::clone(mgr.error_store());

        // Create a buffer, commit, and wait — should complete without error.
        let cb = mgr.get_or_create_buffer();
        // Encode a trivial no-op so the command buffer isn't empty.
        let encoder = cb.new_compute_command_encoder();
        encoder.end_encoding();
        mgr.force_commit();

        // Give the GPU a moment to complete.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // No errors expected for a valid no-op command buffer.
        assert!(!store.has_errors());
    }
}
