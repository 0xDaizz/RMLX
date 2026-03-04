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
    #[allow(dead_code)]
    label: String,
}

/// Multi-stream manager supporting N named command queues with
/// cross-stream synchronization via [`GpuFence`].
pub struct StreamManager {
    device: Device,
    streams: HashMap<u32, StreamState>,
    /// Shared fence used for `synchronize()` calls between streams.
    sync_fence: GpuFence,
}

impl StreamManager {
    /// Create a new stream manager.
    ///
    /// Automatically creates the three predefined streams
    /// (`STREAM_DEFAULT`, `STREAM_COMPUTE`, `STREAM_COPY`).
    pub fn new(device: &Device) -> Self {
        let mut streams = HashMap::new();

        streams.insert(
            STREAM_DEFAULT,
            StreamState {
                queue: device.new_command_queue(),
                label: "default".to_string(),
            },
        );
        streams.insert(
            STREAM_COMPUTE,
            StreamState {
                queue: device.new_command_queue(),
                label: "compute".to_string(),
            },
        );
        streams.insert(
            STREAM_COPY,
            StreamState {
                queue: device.new_command_queue(),
                label: "copy".to_string(),
            },
        );

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
        let state = self.streams.entry(id).or_insert_with(|| StreamState {
            queue: device.new_command_queue(),
            label: format!("stream-{id}"),
        });
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
        Ok(state.queue.new_command_buffer().to_owned())
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
}
