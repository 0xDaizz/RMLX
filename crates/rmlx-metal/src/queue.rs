//! Metal command queue management

use metal::{CommandBuffer, CommandBufferRef, CommandQueue};

use crate::command::{CommandBufferManager, GpuError, GpuErrorStore};
use crate::device::GpuDevice;

/// Create a command buffer using `commandBufferWithUnretainedReferences`.
///
/// This avoids the retain/release overhead for every Metal resource referenced
/// by the command buffer.  Safe as long as all referenced buffers (Arrays,
/// weights, etc.) outlive the command buffer — which is always true in RMLX
/// because `Array` storage is `Arc`-wrapped.
///
/// All Apple Silicon (M1+) supports this API.  On hypothetical older hardware
/// the regular `new_command_buffer()` path would be needed, but RMLX targets
/// M-series exclusively so we use unretained unconditionally.
#[inline]
pub fn fast_command_buffer(queue: &CommandQueue) -> &CommandBufferRef {
    queue.new_command_buffer_with_unretained_references()
}

/// Owned variant: creates an unretained-references command buffer and calls
/// `.to_owned()` so it survives the autorelease pool.
#[inline]
pub fn fast_command_buffer_owned(queue: &CommandQueue) -> CommandBuffer {
    queue
        .new_command_buffer_with_unretained_references()
        .to_owned()
}

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
    ///
    /// Uses `commandBufferWithUnretainedReferences` to skip retain/release
    /// overhead on referenced resources.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        fast_command_buffer(&self.queue)
    }

    /// Create a new command buffer and register a completion handler that
    /// checks for GPU errors after execution (M4).
    pub fn new_checked_command_buffer(&self) -> &CommandBufferRef {
        let cb = fast_command_buffer(&self.queue);
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
