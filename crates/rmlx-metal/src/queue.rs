//! Metal command queue management

use metal::{CommandBufferRef, CommandQueue};

use crate::command::{CommandBufferManager, GpuError, GpuErrorStore};
use crate::device::GpuDevice;

/// Thin wrapper around a Metal command queue.
///
/// For single-queue usage. See [`crate::stream::StreamManager`] for
/// dual-queue (compute + transfer) scheduling with inter-queue sync.
pub struct GpuQueue {
    queue: CommandQueue,
    /// Shared error store for completion-handler error reporting (M4).
    error_store: std::sync::Arc<GpuErrorStore>,
}

impl GpuQueue {
    /// Create a new command queue on the given device.
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            queue: device.new_command_queue(),
            error_store: std::sync::Arc::new(GpuErrorStore::new()),
        }
    }

    /// Create a new command buffer from this queue.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        self.queue.new_command_buffer()
    }

    /// Create a new command buffer and register a completion handler that
    /// checks for GPU errors after execution (M4).
    pub fn new_checked_command_buffer(&self) -> &CommandBufferRef {
        let cb = self.queue.new_command_buffer();
        let store = std::sync::Arc::clone(&self.error_store);
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
        cb
    }

    /// Create a [`CommandBufferManager`] backed by this queue's underlying
    /// Metal command queue.  The manager provides batching (M1) and
    /// completion-handler error checking (M4).
    pub fn create_buffer_manager(&self) -> CommandBufferManager<'_> {
        CommandBufferManager::new(&self.queue)
    }

    /// Drain and return any GPU errors reported by completion handlers.
    pub fn take_errors(&self) -> Vec<GpuError> {
        self.error_store.take_errors()
    }

    /// Check if any GPU errors have been reported.
    pub fn has_errors(&self) -> bool {
        self.error_store.has_errors()
    }

    /// Access the underlying `metal::CommandQueue`.
    pub fn raw(&self) -> &CommandQueue {
        &self.queue
    }
}
