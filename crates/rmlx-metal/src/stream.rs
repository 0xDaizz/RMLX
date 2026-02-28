//! Stream manager for dual command queue execution.
//!
//! Manages two independent Metal command queues:
//! - Compute queue: for GPU kernel dispatch (matmul, softmax, etc.)
//! - Transfer queue: for DMA/copy operations and RDMA coordination
//!
//! SharedEvents synchronize between queues when needed.

use metal::{CommandBuffer, CommandQueue, Device};

use crate::event::GpuEvent;

/// Dual-queue stream manager.
pub struct StreamManager {
    compute_queue: CommandQueue,
    transfer_queue: CommandQueue,
    sync_event: GpuEvent,
}

impl StreamManager {
    /// Create a new stream manager with dual queues.
    pub fn new(device: &Device) -> Self {
        let compute_queue = device.new_command_queue();
        let transfer_queue = device.new_command_queue();
        let sync_event = GpuEvent::new(device);
        Self {
            compute_queue,
            transfer_queue,
            sync_event,
        }
    }

    /// Get the compute queue.
    pub fn compute_queue(&self) -> &CommandQueue {
        &self.compute_queue
    }

    /// Get the transfer queue.
    pub fn transfer_queue(&self) -> &CommandQueue {
        &self.transfer_queue
    }

    /// Create a command buffer on the compute queue.
    pub fn compute_command_buffer(&self) -> CommandBuffer {
        self.compute_queue.new_command_buffer().to_owned()
    }

    /// Create a command buffer on the transfer queue.
    pub fn transfer_command_buffer(&self) -> CommandBuffer {
        self.transfer_queue.new_command_buffer().to_owned()
    }

    /// Insert a dependency: transfer waits for compute to complete.
    /// Returns the signal value used for synchronization.
    pub fn sync_transfer_after_compute(
        &self,
        compute_cb: &metal::CommandBufferRef,
        transfer_cb: &metal::CommandBufferRef,
    ) -> u64 {
        let value = self.sync_event.next_value();
        self.sync_event
            .signal_from_command_buffer(compute_cb, value);
        self.sync_event.wait_from_command_buffer(transfer_cb, value);
        value
    }

    /// Insert a dependency: compute waits for transfer to complete.
    pub fn sync_compute_after_transfer(
        &self,
        transfer_cb: &metal::CommandBufferRef,
        compute_cb: &metal::CommandBufferRef,
    ) -> u64 {
        let value = self.sync_event.next_value();
        self.sync_event
            .signal_from_command_buffer(transfer_cb, value);
        self.sync_event.wait_from_command_buffer(compute_cb, value);
        value
    }

    /// Access the synchronization event.
    pub fn sync_event(&self) -> &GpuEvent {
        &self.sync_event
    }
}
