//! Ticket-aware buffer pool for ZeroCopyBuffer recycling.
//!
//! Buffers with attached CompletionTickets are only returned to the pool
//! once both GPU and RDMA operations have completed. Non-complete buffers
//! are queued in a pending list and drained on subsequent pool operations.

use std::collections::VecDeque;

use rmlx_metal::device::GpuDevice;

use crate::zero_copy::ZeroCopyBuffer;
use crate::AllocError;

/// A pool of reusable ZeroCopyBuffers with completion-aware recycling.
///
/// Buffers returned via `release()` are checked for completion status.
/// Complete buffers go directly into the free list; incomplete buffers
/// are queued in a pending list that is drained on each `acquire()` call.
pub struct BufferPool {
    free: Vec<ZeroCopyBuffer>,
    pending: VecDeque<ZeroCopyBuffer>,
    buffer_size: usize,
    max_pool_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool.
    ///
    /// - `buffer_size`: size of each buffer in the pool
    /// - `max_pool_size`: maximum number of free buffers to keep
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            free: Vec::with_capacity(max_pool_size),
            pending: VecDeque::new(),
            buffer_size,
            max_pool_size,
        }
    }

    /// Acquire a buffer from the pool.
    ///
    /// First drains any pending buffers that have completed, then attempts
    /// to return a free buffer. If none available, allocates a new one.
    pub fn acquire(&mut self, device: &GpuDevice) -> Result<ZeroCopyBuffer, AllocError> {
        self.drain_pending();

        if let Some(buf) = self.free.pop() {
            Ok(buf)
        } else {
            ZeroCopyBuffer::new(device, self.buffer_size)
        }
    }

    /// Release a buffer back to the pool.
    ///
    /// If the buffer's completion ticket indicates all operations are done,
    /// it goes directly into the free list. Otherwise, it's queued in the
    /// pending list.
    pub fn release(&mut self, buf: ZeroCopyBuffer) {
        if buf.is_safe_to_free() {
            if self.free.len() < self.max_pool_size {
                self.free.push(buf);
            }
            // else: drop the buffer (pool is full)
        } else {
            self.pending.push_back(buf);
        }
    }

    /// Drain pending buffers, moving completed ones to the free list.
    fn drain_pending(&mut self) {
        let mut still_pending = VecDeque::new();
        while let Some(buf) = self.pending.pop_front() {
            if buf.is_safe_to_free() {
                if self.free.len() < self.max_pool_size {
                    self.free.push(buf);
                }
                // else: drop the buffer
            } else {
                still_pending.push_back(buf);
            }
        }
        self.pending = still_pending;
    }

    /// Number of buffers available in the free list.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Number of buffers waiting for completion.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Configured buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}
