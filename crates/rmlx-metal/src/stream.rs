//! Multi-stream command queue architecture.
//!
//! Generalizes the previous dual-queue model to N named streams, each
//! with its own `CommandQueue` and independent `CommandBufferManager`.
//!
//! Predefined stream IDs:
//! - `STREAM_DEFAULT  = 0` — general purpose
//! - `STREAM_COMPUTE  = 1` — GPU kernel dispatch (matmul, softmax, etc.)
//! - `STREAM_COPY     = 2` — DMA/copy operations and RDMA coordination
//!
//! Inter-stream synchronization uses [`GpuFence`] from the `fence` module.

use std::collections::HashMap;

use metal::{CommandBuffer, CommandQueue, Device};

use crate::command::CommandBufferManager;
use crate::fence::GpuFence;
use crate::MetalError;

/// Well-known stream IDs.
pub const STREAM_DEFAULT: u32 = 0;
pub const STREAM_COMPUTE: u32 = 1;
pub const STREAM_COPY: u32 = 2;

/// State for a single stream: its command queue and label.
struct StreamState {
    queue: CommandQueue,
    /// Human-readable label for debugging / GPU profiler output.
    /// Wired to [`CommandQueue::set_label`] so it appears in Metal GPU profiler traces.
    /// The field itself is not read by Rust code but is kept for diagnostics.
    #[allow(dead_code)]
    label: String,
}

impl StreamState {
    /// Create a new stream state, wiring the label to the Metal `CommandQueue`
    /// so it shows up in GPU profiler / capture traces.
    fn new(device: &Device, label: &str) -> Self {
        let queue = device.new_command_queue();
        queue.set_label(label);
        Self {
            queue,
            label: label.to_string(),
        }
    }
}

/// Multi-stream manager supporting N named command queues with
/// cross-stream synchronization via [`GpuFence`].
pub struct StreamManager {
    device: Device,
    streams: HashMap<u32, StreamState>,
    /// Shared fence used for `synchronize()` calls between streams.
    sync_fence: GpuFence,
}

// SAFETY: Metal CommandQueue and Device are Objective-C objects that are
// internally reference-counted and thread-safe. HashMap is only accessed
// via &self or &mut self which Rust enforces at compile time.
// GpuFence is already Send + Sync.
unsafe impl Send for StreamManager {}
unsafe impl Sync for StreamManager {}

/// Cross-stream synchronization helper.
///
/// Wraps a [`GpuFence`] to provide typed signal/wait operations
/// between compute and copy streams. Thread-safe for sharing across
/// dispatch threads.
pub struct StreamSync {
    fence: GpuFence,
}

impl StreamSync {
    /// Create a new stream synchronization primitive on `device`.
    pub fn new(device: &metal::Device) -> Self {
        Self {
            fence: GpuFence::new(device),
        }
    }

    /// Signal from a compute command buffer after compute work completes.
    ///
    /// Returns the signal value that the copy stream should wait on.
    pub fn signal_from_compute(&self, compute_cb: &metal::CommandBufferRef) -> u64 {
        let value = self.fence.next_value();
        self.fence.signal(compute_cb, value);
        value
    }

    /// Wait on a copy command buffer until compute signals the given value.
    pub fn wait_on_copy(&self, copy_cb: &metal::CommandBufferRef, value: u64) {
        self.fence.wait(copy_cb, value);
    }

    /// Signal from a copy command buffer after copy/transfer completes.
    ///
    /// Returns the signal value that the compute stream should wait on.
    pub fn signal_from_copy(&self, copy_cb: &metal::CommandBufferRef) -> u64 {
        let value = self.fence.next_value();
        self.fence.signal(copy_cb, value);
        value
    }

    /// Wait on a compute command buffer until copy signals the given value.
    pub fn wait_on_compute(&self, compute_cb: &metal::CommandBufferRef, value: u64) {
        self.fence.wait(compute_cb, value);
    }

    /// Access the underlying fence.
    pub fn fence(&self) -> &GpuFence {
        &self.fence
    }
}

impl StreamManager {
    /// Create a new stream manager.
    ///
    /// Automatically creates the three predefined streams
    /// (`STREAM_DEFAULT`, `STREAM_COMPUTE`, `STREAM_COPY`).
    pub fn new(device: &Device) -> Self {
        let mut streams = HashMap::new();

        streams.insert(STREAM_DEFAULT, StreamState::new(device, "default"));
        streams.insert(STREAM_COMPUTE, StreamState::new(device, "compute"));
        streams.insert(STREAM_COPY, StreamState::new(device, "copy"));

        let sync_fence = GpuFence::new(device);

        Self {
            device: device.clone(),
            streams,
            sync_fence,
        }
    }

    /// Get or create a stream with the given `id`.
    ///
    /// If the stream already exists, returns its command queue.
    /// Otherwise, creates a new queue on the device and inserts it.
    pub fn get_or_create_stream(&mut self, id: u32) -> &CommandQueue {
        let device = &self.device;
        let state = self
            .streams
            .entry(id)
            .or_insert_with(|| StreamState::new(device, &format!("stream-{id}")));
        &state.queue
    }

    /// Get the command queue for an existing stream.
    ///
    /// Returns `None` if the stream has not been created yet.
    pub fn queue(&self, id: u32) -> Option<&CommandQueue> {
        self.streams.get(&id).map(|s| &s.queue)
    }

    /// Create a command buffer on the given stream.
    ///
    /// Returns an error if `stream_id` does not exist (use `get_or_create_stream`
    /// first or stick to the predefined IDs).
    pub fn command_buffer(&self, stream_id: u32) -> Result<CommandBuffer, MetalError> {
        let state = self.streams.get(&stream_id).ok_or_else(|| {
            MetalError::KernelNotFound(format!("stream {stream_id} does not exist"))
        })?;
        Ok(crate::queue::fast_command_buffer_owned(&state.queue))
    }

    /// Create a [`CommandBufferManager`] for the given stream.
    ///
    /// The manager provides batching (M1) and completion-handler error checking (M4).
    ///
    /// Returns an error if `stream_id` does not exist.
    pub fn create_buffer_manager(
        &self,
        stream_id: u32,
    ) -> Result<CommandBufferManager<'_>, MetalError> {
        let state = self.streams.get(&stream_id).ok_or_else(|| {
            MetalError::KernelNotFound(format!("stream {stream_id} does not exist"))
        })?;
        Ok(CommandBufferManager::new(&state.queue))
    }

    /// Synchronize `dst_stream` after `src_stream`.
    ///
    /// Encodes a signal on `src_cb` and a wait on `dst_cb` using the
    /// internal [`GpuFence`].  Returns the fence value used.
    ///
    /// This creates a happens-before relationship: all commands encoded
    /// on `src_cb` before this call will complete before any commands
    /// encoded on `dst_cb` after this call begin executing.
    pub fn synchronize(
        &self,
        src_cb: &metal::CommandBufferRef,
        dst_cb: &metal::CommandBufferRef,
    ) -> u64 {
        let value = self.sync_fence.next_value();
        self.sync_fence.signal(src_cb, value);
        self.sync_fence.wait(dst_cb, value);
        value
    }

    /// Access the synchronization fence.
    pub fn sync_fence(&self) -> &GpuFence {
        &self.sync_fence
    }

    /// Convenience: get the compute queue.
    pub fn compute_queue(&self) -> &CommandQueue {
        self.queue(STREAM_COMPUTE)
            .expect("STREAM_COMPUTE not found")
    }

    /// Convenience: get the transfer/copy queue.
    pub fn transfer_queue(&self) -> &CommandQueue {
        self.queue(STREAM_COPY).expect("STREAM_COPY not found")
    }

    /// Create a command buffer on the compute stream.
    pub fn compute_command_buffer(&self) -> Result<CommandBuffer, MetalError> {
        self.command_buffer(STREAM_COMPUTE)
    }

    /// Create a command buffer on the copy/transfer stream.
    pub fn transfer_command_buffer(&self) -> Result<CommandBuffer, MetalError> {
        self.command_buffer(STREAM_COPY)
    }

    /// Insert a dependency: transfer waits for compute to complete.
    /// Returns the signal value used for synchronization.
    pub fn sync_transfer_after_compute(
        &self,
        compute_cb: &metal::CommandBufferRef,
        transfer_cb: &metal::CommandBufferRef,
    ) -> u64 {
        self.synchronize(compute_cb, transfer_cb)
    }

    /// Insert a dependency: compute waits for transfer to complete.
    pub fn sync_compute_after_transfer(
        &self,
        transfer_cb: &metal::CommandBufferRef,
        compute_cb: &metal::CommandBufferRef,
    ) -> u64 {
        self.synchronize(transfer_cb, compute_cb)
    }

    /// Number of streams currently managed.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Whether a stream with `id` exists.
    pub fn has_stream(&self, id: u32) -> bool {
        self.streams.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predefined_streams_created() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        assert!(mgr.has_stream(STREAM_DEFAULT));
        assert!(mgr.has_stream(STREAM_COMPUTE));
        assert!(mgr.has_stream(STREAM_COPY));
        assert_eq!(mgr.stream_count(), 3);
    }

    #[test]
    fn test_get_or_create_new_stream() {
        let device = metal::Device::system_default().unwrap();
        let mut mgr = StreamManager::new(&device);

        assert!(!mgr.has_stream(42));
        let _ = mgr.get_or_create_stream(42);
        assert!(mgr.has_stream(42));
        assert_eq!(mgr.stream_count(), 4);
    }

    #[test]
    fn test_get_or_create_existing_stream() {
        let device = metal::Device::system_default().unwrap();
        let mut mgr = StreamManager::new(&device);

        let _ = mgr.get_or_create_stream(STREAM_COMPUTE);
        // Should not create a duplicate.
        assert_eq!(mgr.stream_count(), 3);
    }

    #[test]
    fn test_command_buffer_creation() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        // Should not fail for predefined streams.
        let _cb = mgr.compute_command_buffer().unwrap();
        let _cb = mgr.transfer_command_buffer().unwrap();
        let _cb = mgr.command_buffer(STREAM_DEFAULT).unwrap();
    }

    #[test]
    fn test_synchronize_returns_monotonic_values() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        let cb1 = mgr.compute_command_buffer().unwrap();
        let cb2 = mgr.transfer_command_buffer().unwrap();

        let v1 = mgr.synchronize(&cb1, &cb2);
        assert_eq!(v1, 1);

        let cb3 = mgr.compute_command_buffer().unwrap();
        let cb4 = mgr.transfer_command_buffer().unwrap();
        let v2 = mgr.synchronize(&cb3, &cb4);
        assert_eq!(v2, 2);
    }

    #[test]
    fn test_convenience_sync_methods() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        let compute_cb = mgr.compute_command_buffer().unwrap();
        let transfer_cb = mgr.transfer_command_buffer().unwrap();

        let v = mgr.sync_transfer_after_compute(&compute_cb, &transfer_cb);
        assert!(v > 0);
    }

    #[test]
    fn test_compute_copy_queue_independence() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        // Compute and copy queues should be distinct objects.
        let compute_q = mgr.compute_queue();
        let copy_q = mgr.transfer_queue();

        // Create command buffers on each — both should succeed independently.
        let compute_cb = compute_q.new_command_buffer().to_owned();
        let copy_cb = copy_q.new_command_buffer().to_owned();

        // Encode no-ops on both and commit independently.
        let enc1 = compute_cb.new_compute_command_encoder();
        enc1.end_encoding();
        compute_cb.commit();

        let enc2 = copy_cb.new_blit_command_encoder();
        enc2.end_encoding();
        copy_cb.commit();

        compute_cb.wait_until_completed();
        copy_cb.wait_until_completed();
    }

    #[test]
    fn test_stream_sync_signal_wait() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);
        let sync = super::StreamSync::new(&device);

        let compute_cb = mgr.compute_command_buffer().unwrap();
        let copy_cb = mgr.transfer_command_buffer().unwrap();

        // Encode a no-op compute, then signal.
        let enc = compute_cb.new_compute_command_encoder();
        enc.end_encoding();
        let value = sync.signal_from_compute(&compute_cb);

        // Copy waits for compute.
        sync.wait_on_copy(&copy_cb, value);
        let enc2 = copy_cb.new_blit_command_encoder();
        enc2.end_encoding();

        // Commit both — copy should wait for compute signal.
        compute_cb.commit();
        copy_cb.commit();

        compute_cb.wait_until_completed();
        copy_cb.wait_until_completed();

        // Verify signal value was set.
        assert!(sync.fence().signaled_value() >= value);
    }

    #[test]
    fn test_stream_manager_thread_safe() {
        // Compile-time check that StreamManager is Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<StreamManager>();
        assert_send_sync::<super::StreamSync>();
    }

    #[test]
    fn test_command_buffer_nonexistent_stream_fails() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        // Stream 999 was never created.
        let result = mgr.command_buffer(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_queue_returns_none_for_missing_stream() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        assert!(mgr.queue(999).is_none());
        assert!(mgr.queue(STREAM_COMPUTE).is_some());
    }

    #[test]
    fn test_stream_constants() {
        assert_eq!(STREAM_DEFAULT, 0);
        assert_eq!(STREAM_COMPUTE, 1);
        assert_eq!(STREAM_COPY, 2);
    }

    #[test]
    fn test_create_buffer_manager_success() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        let _bm = mgr.create_buffer_manager(STREAM_COMPUTE).unwrap();
        let _bm = mgr.create_buffer_manager(STREAM_COPY).unwrap();
    }

    #[test]
    fn test_create_buffer_manager_nonexistent_stream() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        let result = mgr.create_buffer_manager(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_sync_fence_accessible() {
        let device = metal::Device::system_default().unwrap();
        let mgr = StreamManager::new(&device);

        let fence = mgr.sync_fence();
        // Initial signaled value should be 0.
        assert_eq!(fence.signaled_value(), 0);
    }

    #[test]
    fn test_stream_sync_fence_accessor() {
        let device = metal::Device::system_default().unwrap();
        let sync = StreamSync::new(&device);
        let fence = sync.fence();
        assert_eq!(fence.signaled_value(), 0);
    }
}
