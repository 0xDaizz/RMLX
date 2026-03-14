//! Metal command buffer and encoder abstractions
//!
//! Provides [`CommandBufferManager`] for batching multiple dispatches into a
//! single command buffer, and [`BarrierTracker`] for inserting memory barriers
//! between dependent kernel dispatches.

use rustc_hash::FxHashSet;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::compute_pass::ComputePass;
use crate::types::*;

// ---------------------------------------------------------------------------
// M1 — auto-commit thresholds
// ---------------------------------------------------------------------------
/// Fallback ops-per-batch threshold. Prefer `ChipTuning::max_ops_per_batch`
/// when a device-specific `ChipTuning` is available.
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
    queue: &'q ProtocolObject<dyn MTLCommandQueue>,
    /// Owned command buffer — `Retained` keeps it alive until we commit.
    current_cb: Option<MtlCB>,
    ops_batched: usize,
    bytes_dispatched: u64,
    /// Track GPU errors reported via completion handlers.
    error_store: std::sync::Arc<GpuErrorStore>,
}

impl<'q> CommandBufferManager<'q> {
    /// Create a new manager that draws command buffers from `queue`.
    pub fn new(queue: &'q ProtocolObject<dyn MTLCommandQueue>) -> Self {
        Self {
            queue,
            current_cb: None,
            ops_batched: 0,
            bytes_dispatched: 0,
            error_store: std::sync::Arc::new(GpuErrorStore::new()),
        }
    }

    /// Return the current command buffer, creating one if necessary.
    ///
    /// The command buffer is retained (owned) to prevent deallocation
    /// before commit.
    pub fn get_or_create_buffer(&mut self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        if self.current_cb.is_none() {
            let cb = crate::queue::fast_command_buffer_owned(self.queue);
            self.current_cb = Some(cb);
        }
        self.current_cb.as_ref().unwrap()
    }

    /// Create a compute command encoder on the current (or new) command buffer.
    ///
    /// Returns an owned `Retained` encoder. The caller must end it (via
    /// `ComputePass::end()` or raw `endEncoding()`) before the command buffer
    /// is committed. The `Retained` keeps the encoder alive for the caller's
    /// scope.
    pub fn get_or_create_encoder(&mut self) -> MtlEncoder {
        let cb = self.get_or_create_buffer();
        cb.computeCommandEncoder().unwrap()
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
        if let Some(ref cb) = self.current_cb.take() {
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
/// the tracker signals that a memory barrier is required.
pub struct BarrierTracker {
    /// Buffers that were bound as *output* (write) in the previous dispatch.
    previous_outputs: FxHashSet<u64>,
    /// Buffers that were bound as *input* (read) in the previous dispatch(es).
    /// Used by `check_concurrent` to detect WAR (write-after-read) hazards.
    previous_inputs: FxHashSet<u64>,
    /// Buffers marked as temporary — exempted from barrier checks.
    /// These are intermediate buffers that are only used within a single
    /// encoder scope (e.g., `normed`, `qkv`, `gate_up`, `hidden` in
    /// the 9-dispatch decode path).
    temporaries: FxHashSet<u64>,
}

impl BarrierTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            previous_outputs: FxHashSet::default(),
            previous_inputs: FxHashSet::default(),
            temporaries: FxHashSet::default(),
        }
    }

    /// Mark a buffer as temporary (exempt from barrier checks).
    ///
    /// Temporary buffers are intermediate results that exist only within
    /// a single encoder scope. They are always produced and consumed in
    /// order, so no inter-dispatch barrier is needed for them.
    ///
    /// # Safety invariant
    ///
    /// Callers **must** call [`clear_temporaries`] at encoder boundaries
    /// or before the underlying allocator could reuse the buffer's GPU
    /// address for a different allocation. Failing to do so may cause
    /// a later, unrelated buffer at the same address to be incorrectly
    /// exempted from barrier tracking.
    pub fn mark_temporary(&mut self, buf: &ProtocolObject<dyn MTLBuffer>) {
        self.temporaries.insert(buf.gpuAddress());
    }

    /// Clear all temporary buffer markings.
    ///
    /// Should be called at encoder boundaries or when the temporary
    /// buffer set changes (e.g., between layers).
    pub fn clear_temporaries(&mut self) {
        self.temporaries.clear();
    }

    /// Number of buffers currently marked as temporary.
    pub fn temporary_count(&self) -> usize {
        self.temporaries.len()
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
    pub fn needs_barrier(
        &mut self,
        inputs: &[&ProtocolObject<dyn MTLBuffer>],
        outputs: &[&ProtocolObject<dyn MTLBuffer>],
    ) -> bool {
        let mut needs = false;

        // Check if any current input was written in the previous dispatch.
        for buf in inputs {
            let addr = buf.gpuAddress();
            if self.temporaries.contains(&addr) {
                continue;
            }
            if self.previous_outputs.contains(&addr) {
                needs = true;
                break;
            }
        }

        // Also check output-after-output (WAW hazard).
        if !needs {
            for buf in outputs {
                let addr = buf.gpuAddress();
                if self.temporaries.contains(&addr) {
                    continue;
                }
                if self.previous_outputs.contains(&addr) {
                    needs = true;
                    break;
                }
            }
        }

        // Update state: current outputs become previous outputs.
        self.previous_outputs.clear();
        for buf in outputs {
            self.previous_outputs.insert(buf.gpuAddress());
        }

        needs
    }

    /// Check and track dependencies, suitable for concurrent encoder mode.
    ///
    /// Unlike `needs_barrier` which replaces previous_outputs wholesale,
    /// this method accumulates outputs across dispatches that don't need
    /// barriers, and only rotates when a barrier IS needed. This correctly
    /// tracks multi-dispatch dependency chains on a concurrent encoder.
    pub fn check_concurrent(
        &mut self,
        inputs: &[&ProtocolObject<dyn MTLBuffer>],
        outputs: &[&ProtocolObject<dyn MTLBuffer>],
    ) -> bool {
        let mut needs = false;

        // RAW check: current inputs vs previous outputs
        for buf in inputs {
            let addr = buf.gpuAddress();
            if self.temporaries.contains(&addr) {
                continue;
            }
            if self.previous_outputs.contains(&addr) {
                needs = true;
                break;
            }
        }

        // WAW check: current outputs vs previous outputs
        if !needs {
            for buf in outputs {
                let addr = buf.gpuAddress();
                if self.temporaries.contains(&addr) {
                    continue;
                }
                if self.previous_outputs.contains(&addr) {
                    needs = true;
                    break;
                }
            }
        }

        // WAR check: current outputs vs previous inputs
        if !needs {
            for buf in outputs {
                let addr = buf.gpuAddress();
                if self.temporaries.contains(&addr) {
                    continue;
                }
                if self.previous_inputs.contains(&addr) {
                    needs = true;
                    break;
                }
            }
        }

        if needs {
            // Barrier needed: clear both tracking sets.
            // After barrier, only the current dispatch's buffers are "pending".
            self.previous_outputs.clear();
            self.previous_inputs.clear();
        }

        // Track current outputs and inputs (accumulate)
        for buf in outputs {
            self.previous_outputs.insert(buf.gpuAddress());
        }
        for buf in inputs {
            self.previous_inputs.insert(buf.gpuAddress());
        }

        needs
    }

    /// Reset the tracker (e.g. when a new command buffer begins).
    pub fn reset(&mut self) {
        self.previous_outputs.clear();
        self.previous_inputs.clear();
        self.temporaries.clear();
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
    errors: parking_lot::Mutex<Vec<GpuError>>,
}

/// A GPU execution error captured from a completed command buffer.
#[derive(Debug, Clone)]
pub struct GpuError {
    pub status: MTLCommandBufferStatus,
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
            errors: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Push a new error (called from completion handler).
    pub fn push(&self, error: GpuError) {
        self.errors.lock().push(error);
    }

    /// Drain and return all recorded errors.
    pub fn take_errors(&self) -> Vec<GpuError> {
        std::mem::take(&mut *self.errors.lock())
    }

    /// Check if any errors have been recorded.
    pub fn has_errors(&self) -> bool {
        !self.errors.lock().is_empty()
    }
}

impl Default for GpuErrorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Register a completion handler on `cb` that checks the command buffer status
/// after GPU execution and stores any error in `error_store`.
pub(crate) fn register_completion_handler(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    error_store: &std::sync::Arc<GpuErrorStore>,
) {
    let store = std::sync::Arc::clone(error_store);
    let handler = block2::RcBlock::new(
        move |cmd_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let cmd_buf = unsafe { cmd_buf.as_ref() };
            let status = cmd_buf.status();
            if status == MTLCommandBufferStatus::Error {
                let msg = format!("command buffer completed with error status: {status:?}");
                store.push(GpuError {
                    status,
                    message: msg,
                });
            }
        },
    );
    unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
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
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, u64)],
    num_threads: u64,
) {
    let encoder = cmd_buf.computeCommandEncoder().unwrap();
    let pass = ComputePass::new(&encoder);

    pass.set_pipeline(pipeline);

    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        pass.set_buffer(index as u32, Some(*buffer), *offset as usize);
    }

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = std::cmp::min(max_threads, num_threads as usize);

    let grid_size = MTLSize {
        width: num_threads as usize,
        height: 1,
        depth: 1,
    };
    let group_size = MTLSize {
        width: threadgroup_size,
        height: 1,
        depth: 1,
    };

    pass.dispatch_threads(grid_size, group_size);
    pass.end();
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
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, u64)],
    input_indices: &[usize],
    output_indices: &[usize],
    num_threads: u64,
    tracker: &mut BarrierTracker,
) {
    let inputs: Vec<&ProtocolObject<dyn MTLBuffer>> =
        input_indices.iter().map(|&i| buffers[i].0).collect();
    let outputs: Vec<&ProtocolObject<dyn MTLBuffer>> =
        output_indices.iter().map(|&i| buffers[i].0).collect();

    if tracker.needs_barrier(&inputs, &outputs) {
        // Insert a barrier by ending and creating a new encoder.
        // The encoder created here acts as a fresh synchronization point.
        // (We create a no-op encoder just to force the barrier, then create
        // the real one below.)
        let barrier_encoder = cmd_buf.computeCommandEncoder().unwrap();
        barrier_encoder.endEncoding();
    }

    encode_compute_1d(cmd_buf, pipeline, buffers, num_threads);
}

// ---------------------------------------------------------------------------
// Concurrent dispatch support
// ---------------------------------------------------------------------------

/// Create a compute command encoder with concurrent dispatch type.
///
/// A concurrent encoder allows multiple dispatches to run in parallel on
/// the GPU. Memory dependencies between dispatches must be managed
/// explicitly via `memory_barrier_scope_buffers()`.
///
/// Falls back to a serial encoder if the concurrent dispatch API is not
/// available (should not happen on any Apple Silicon device).
///
/// # Safety
///
/// The caller must ensure all data dependencies between dispatches are
/// covered by explicit `memory_barrier_scope_buffers()` calls. Without
/// proper barriers, concurrent dispatches may read stale data.
pub fn new_concurrent_encoder(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> MtlEncoder {
    let desc = MTLComputePassDescriptor::new();
    desc.setDispatchType(MTLDispatchType::Concurrent);
    cb.computeCommandEncoderWithDescriptor(&desc).unwrap()
}

/// Insert a scope-level memory barrier on a compute command encoder.
///
/// This ensures all buffer writes from previous dispatches on this encoder
/// are visible to subsequent dispatches.
///
/// Use this between dependent dispatches on a concurrent encoder to
/// prevent data races.
pub fn memory_barrier_scope_buffers(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
}

/// Check if concurrent dispatch is available by attempting to create
/// a concurrent encoder. Returns true if successful.
///
/// Called once at init to determine if the concurrent dispatch path
/// can be used. If not, falls back to serial encoders.
pub fn probe_concurrent_dispatch(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    let queue = device.newCommandQueue();
    let Some(queue) = queue else {
        return false;
    };
    let Some(cb) = queue.commandBuffer() else {
        return false;
    };

    let desc = MTLComputePassDescriptor::new();
    desc.setDispatchType(MTLDispatchType::Concurrent);

    let encoder = cb.computeCommandEncoderWithDescriptor(&desc);
    match encoder {
        Some(enc) => {
            enc.endEncoding();
            true
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::OnceLock;

    fn test_device() -> Option<&'static MtlDevice> {
        static DEVICE: OnceLock<Option<MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| objc2::rc::autoreleasepool(|_| MTLCreateSystemDefaultDevice()))
            .as_ref()
    }

    fn system_device() -> Option<&'static MtlDevice> {
        test_device()
    }

    #[test]
    fn test_barrier_tracker_no_conflict() {
        let Some(device) = system_device() else {
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let b = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

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
        let Some(device) = system_device() else {
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

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
        let Some(device) = system_device() else {
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

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
        let Some(device) = system_device() else {
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        let _ = tracker.needs_barrier(&[], &[&a]);
        tracker.reset();

        // After reset, no hazard should be reported
        let needs = tracker.needs_barrier(&[&a], &[]);
        assert!(!needs, "no hazard after reset");
    }

    #[test]
    fn test_barrier_tracker_temporary_exemption() {
        let Some(device) = system_device() else {
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let temp = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        tracker.mark_temporary(&temp);

        // Write to temp
        let needs = tracker.needs_barrier(&[], &[&temp]);
        assert!(!needs, "first dispatch never needs barrier");

        // Read temp (would be RAW, but temp is exempt)
        let needs = tracker.needs_barrier(&[&temp], &[&a]);
        assert!(!needs, "temporary buffer should be exempt from RAW check");
    }

    #[test]
    fn test_barrier_tracker_temporary_waw_exemption() {
        let Some(device) = system_device() else {
            return;
        };
        let temp = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        tracker.mark_temporary(&temp);

        let needs = tracker.needs_barrier(&[], &[&temp]);
        assert!(!needs);

        // WAW on temp (exempt)
        let needs = tracker.needs_barrier(&[], &[&temp]);
        assert!(!needs, "temporary buffer should be exempt from WAW check");
    }

    #[test]
    fn test_barrier_tracker_clear_temporaries() {
        let Some(device) = system_device() else {
            return;
        };
        let temp = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        tracker.mark_temporary(&temp);
        assert_eq!(tracker.temporary_count(), 1);

        tracker.clear_temporaries();
        assert_eq!(tracker.temporary_count(), 0);

        // Now temp is no longer exempt
        let needs = tracker.needs_barrier(&[], &[&temp]);
        assert!(!needs, "first dispatch");

        let needs = tracker.needs_barrier(&[&temp], &[]);
        assert!(needs, "after clearing temporaries, RAW should be detected");
    }

    #[test]
    fn test_barrier_tracker_mixed_temp_and_regular() {
        let Some(device) = system_device() else {
            return;
        };
        let regular = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let temp = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        tracker.mark_temporary(&temp);

        // Write both regular and temp
        let needs = tracker.needs_barrier(&[], &[&regular, &temp]);
        assert!(!needs, "first dispatch");

        // Read regular (RAW) and temp (exempt)
        let needs = tracker.needs_barrier(&[&regular, &temp], &[]);
        assert!(needs, "regular buffer RAW should still trigger barrier");
    }

    #[test]
    fn test_barrier_tracker_reset_clears_temporaries() {
        let Some(device) = system_device() else {
            return;
        };
        let temp = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();
        tracker.mark_temporary(&temp);
        assert_eq!(tracker.temporary_count(), 1);

        tracker.reset();
        assert_eq!(tracker.temporary_count(), 0);
    }

    #[test]
    fn test_gpu_error_store() {
        let store = GpuErrorStore::new();
        assert!(!store.has_errors());

        store.push(GpuError {
            status: MTLCommandBufferStatus::Error,
            message: "test error".to_string(),
        });
        assert!(store.has_errors());

        let errors = store.take_errors();
        assert_eq!(errors.len(), 1);
        assert!(!store.has_errors());
    }

    #[test]
    fn test_command_buffer_manager_thresholds() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
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
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
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
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
        let mut mgr = CommandBufferManager::new(&queue);

        mgr.get_or_create_buffer();
        mgr.maybe_commit(MAX_BYTES_PER_BATCH);
        // Should have auto-committed
        assert_eq!(mgr.ops_batched(), 0);
        assert_eq!(mgr.bytes_dispatched(), 0);
    }

    #[test]
    fn test_command_buffer_manager_completion_handler() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
        let mut mgr = CommandBufferManager::new(&queue);

        let store = std::sync::Arc::clone(mgr.error_store());

        // Create a buffer, commit, and wait — should complete without error.
        let cb = mgr.get_or_create_buffer();
        // Encode a trivial no-op so the command buffer isn't empty.
        let encoder = cb.computeCommandEncoder().unwrap();
        encoder.endEncoding();
        mgr.force_commit();

        // Give the GPU a moment to complete.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // No errors expected for a valid no-op command buffer.
        assert!(!store.has_errors());
    }

    // ----- Concurrent dispatch tests -----

    #[test]
    fn test_probe_concurrent_dispatch() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let supported = probe_concurrent_dispatch(device);
        // Apple Silicon always supports concurrent dispatch
        assert!(
            supported,
            "concurrent dispatch should be supported on Apple Silicon"
        );
    }

    #[test]
    fn test_concurrent_encoder_creation() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
        let cb = queue.commandBuffer().unwrap();

        let encoder = new_concurrent_encoder(&cb);
        // Should be able to end encoding without crash
        encoder.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn test_memory_barrier_scope() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();
        let cb = queue.commandBuffer().unwrap();

        let encoder = new_concurrent_encoder(&cb);
        // Should be able to insert barrier without crash
        memory_barrier_scope_buffers(&encoder);
        encoder.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn test_barrier_tracker_concurrent_mode() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let a = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let b = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut tracker = BarrierTracker::new();

        // D1: write a
        let needs = tracker.check_concurrent(&[], &[&a]);
        assert!(!needs, "first dispatch");

        // D2: write b (no conflict)
        let needs = tracker.check_concurrent(&[], &[&b]);
        assert!(!needs, "independent write");

        // D3: read a (RAW with D1, which is still in previous_outputs)
        let needs = tracker.check_concurrent(&[&a], &[]);
        assert!(needs, "RAW with D1");

        // D4: read b (was cleared by barrier, but b was also in accumulated outputs)
        // After the barrier in D3, previous_outputs was cleared, then D3's outputs
        // (none in this case) were added. So b is no longer tracked.
        let needs = tracker.check_concurrent(&[&b], &[]);
        assert!(!needs, "b was cleared by the barrier");
    }
}
