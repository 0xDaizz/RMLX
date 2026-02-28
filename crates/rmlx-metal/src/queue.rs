//! Metal command queue management

use metal::{CommandBufferRef, CommandQueue};

use crate::device::GpuDevice;

/// Thin wrapper around a Metal command queue.
///
/// For single-queue usage. See [`crate::stream::StreamManager`] for
/// dual-queue (compute + transfer) scheduling with inter-queue sync.
pub struct GpuQueue {
    queue: CommandQueue,
}

impl GpuQueue {
    /// Create a new command queue on the given device.
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            queue: device.new_command_queue(),
        }
    }

    /// Create a new command buffer from this queue.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        self.queue.new_command_buffer()
    }

    /// Access the underlying `metal::CommandQueue`.
    pub fn raw(&self) -> &CommandQueue {
        &self.queue
    }
}
