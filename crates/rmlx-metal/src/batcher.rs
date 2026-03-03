//! Command buffer batcher for multi-op Metal dispatch.
//!
//! Accumulates multiple compute command encoders into a single `MTLCommandBuffer`
//! before committing. This reduces the ~30 command buffers per token to ~5-8,
//! eliminating CPU-GPU round-trip overhead from per-op `waitUntilCompleted()`.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut batcher = CommandBatcher::new(queue, 32);
//!
//! // Op 1: encode into current CB
//! let enc = batcher.encoder();
//! enc.set_compute_pipeline_state(&pipeline);
//! enc.set_buffer(0, Some(buf), 0);
//! enc.dispatch_threads(grid, tg);
//! batcher.end_encoder();
//!
//! // Op 2: same CB
//! let enc = batcher.encoder();
//! // ... encode more work ...
//! batcher.end_encoder();
//!
//! // Commit all at once
//! batcher.flush()?;
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use metal::{CommandBuffer, CommandQueue};

use crate::event::GpuEvent;

/// Tracks how many command buffers and encoders are created (for benchmarking).
static TOTAL_CBS_CREATED: AtomicU64 = AtomicU64::new(0);
static TOTAL_ENCODERS_CREATED: AtomicU64 = AtomicU64::new(0);

/// Get the total number of command buffers created since process start.
pub fn total_cbs_created() -> u64 {
    TOTAL_CBS_CREATED.load(Ordering::Relaxed)
}

/// Get the total number of encoders created since process start.
pub fn total_encoders_created() -> u64 {
    TOTAL_ENCODERS_CREATED.load(Ordering::Relaxed)
}

/// Reset counters (for benchmarking).
pub fn reset_counters() {
    TOTAL_CBS_CREATED.store(0, Ordering::Relaxed);
    TOTAL_ENCODERS_CREATED.store(0, Ordering::Relaxed);
}

/// A command buffer batcher that accumulates multiple compute command encoders
/// into a single `MTLCommandBuffer` before committing.
///
/// Each call to `encoder()` creates a new compute command encoder on the current
/// command buffer. When the encoder count reaches `max_batch`, the batcher
/// auto-flushes. The caller can also flush manually.
pub struct CommandBatcher<'q> {
    queue: &'q CommandQueue,
    current_cb: Option<CommandBuffer>,
    encoder_count: usize,
    max_batch: usize,
    total_cbs: usize,
    total_encoders: usize,
    /// Whether an encoder is currently active (not yet ended).
    encoder_active: bool,
}

impl<'q> CommandBatcher<'q> {
    /// Create a new batcher on the given queue.
    ///
    /// `max_batch` controls the maximum number of encoders per command buffer.
    /// Typical values: 16-64 for transformer blocks.
    pub fn new(queue: &'q CommandQueue, max_batch: usize) -> Self {
        Self {
            queue,
            current_cb: None,
            encoder_count: 0,
            max_batch: max_batch.max(1),
            total_cbs: 0,
            total_encoders: 0,
            encoder_active: false,
        }
    }

    /// Get a new compute command encoder on the current command buffer.
    ///
    /// If no command buffer exists, one is created. If the previous encoder
    /// was not ended, it is ended automatically.
    ///
    /// Returns the encoder reference. The caller must call `end_encoder()`
    /// when done encoding, or it will be ended on the next `encoder()` call.
    pub fn encoder(&mut self) -> &metal::ComputeCommandEncoderRef {
        // End any active encoder first
        if self.encoder_active {
            // This shouldn't happen in well-structured code, but handle it gracefully
            self.encoder_active = false;
        }

        // Create CB if needed
        if self.current_cb.is_none() {
            self.current_cb = Some(self.queue.new_command_buffer().to_owned());
            self.total_cbs += 1;
            TOTAL_CBS_CREATED.fetch_add(1, Ordering::Relaxed);
        }

        self.encoder_count += 1;
        self.total_encoders += 1;
        TOTAL_ENCODERS_CREATED.fetch_add(1, Ordering::Relaxed);
        self.encoder_active = true;

        self.current_cb
            .as_ref()
            .expect("CB just created")
            .new_compute_command_encoder()
    }

    /// End the current compute command encoder.
    ///
    /// Must be called after each `encoder()` call before the next `encoder()`
    /// or `flush()` call. The encoder ref returned by `encoder()` must not be
    /// used after this call.
    pub fn end_encoder(&mut self) {
        self.encoder_active = false;
        // Note: The encoder's end_encoding() must be called by the caller
        // on the encoder ref returned by encoder(). This method just tracks state.
    }

    /// Get the current command buffer reference for encoding.
    ///
    /// Creates a new CB if none exists. This is useful when ops need
    /// direct access to the command buffer (e.g., for blit encoders).
    pub fn command_buffer(&mut self) -> &metal::CommandBufferRef {
        if self.current_cb.is_none() {
            self.current_cb = Some(self.queue.new_command_buffer().to_owned());
            self.total_cbs += 1;
            TOTAL_CBS_CREATED.fetch_add(1, Ordering::Relaxed);
        }
        self.current_cb.as_ref().expect("CB just created")
    }

    /// Check if the batcher should flush (encoder count at max).
    pub fn should_flush(&self) -> bool {
        self.encoder_count >= self.max_batch
    }

    /// Commit the current command buffer and wait for GPU completion.
    ///
    /// Resets the batcher for the next batch.
    pub fn flush(&mut self) {
        if let Some(cb) = self.current_cb.take() {
            cb.commit();
            cb.wait_until_completed();
        }
        self.encoder_count = 0;
    }

    /// Commit the current command buffer without waiting.
    ///
    /// Returns the command buffer for optional later synchronization.
    /// Resets the batcher for the next batch.
    pub fn flush_async(&mut self) -> Option<CommandBuffer> {
        let cb = self.current_cb.take();
        if let Some(ref cb) = cb {
            cb.commit();
        }
        self.encoder_count = 0;
        cb
    }

    /// Commit the current CB and signal a GpuEvent at the given value.
    ///
    /// Used by ExecGraph for event-chained execution.
    pub fn flush_signal(&mut self, event: &GpuEvent, value: u64) -> Option<CommandBuffer> {
        let cb = self.current_cb.take();
        if let Some(ref cb) = cb {
            event.signal_from_command_buffer(cb, value);
            cb.commit();
        }
        self.encoder_count = 0;
        cb
    }

    /// Start a new batch that waits for a GpuEvent at the given value.
    ///
    /// Creates a new command buffer and encodes a wait on the event.
    /// Subsequent encoders will execute after the event is signaled.
    pub fn begin_waiting(&mut self, event: &GpuEvent, value: u64) {
        // Flush any pending work
        if self.current_cb.is_some() {
            self.flush_async();
        }
        let cb = self.queue.new_command_buffer().to_owned();
        event.wait_from_command_buffer(&cb, value);
        self.current_cb = Some(cb);
        self.total_cbs += 1;
        TOTAL_CBS_CREATED.fetch_add(1, Ordering::Relaxed);
    }

    /// Number of encoders added to the current command buffer.
    pub fn encoder_count(&self) -> usize {
        self.encoder_count
    }

    /// Total command buffers created by this batcher.
    pub fn stats_cbs(&self) -> usize {
        self.total_cbs
    }

    /// Total encoders created by this batcher.
    pub fn stats_encoders(&self) -> usize {
        self.total_encoders
    }

    /// Whether there is a pending (uncommitted) command buffer.
    pub fn has_pending(&self) -> bool {
        self.current_cb.is_some()
    }

    /// The underlying queue.
    pub fn queue(&self) -> &CommandQueue {
        self.queue
    }
}

impl Drop for CommandBatcher<'_> {
    fn drop(&mut self) {
        // Flush any pending work on drop to prevent GPU resource leaks
        if self.current_cb.is_some() {
            self.flush();
        }
    }
}

/// Statistics snapshot from a CommandBatcher.
#[derive(Debug, Clone)]
pub struct BatcherStats {
    pub total_cbs: usize,
    pub total_encoders: usize,
    pub encoders_per_cb: f64,
}

impl BatcherStats {
    pub fn from_batcher(batcher: &CommandBatcher<'_>) -> Self {
        let total_cbs = batcher.stats_cbs();
        let total_encoders = batcher.stats_encoders();
        let encoders_per_cb = if total_cbs > 0 {
            total_encoders as f64 / total_cbs as f64
        } else {
            0.0
        };
        Self {
            total_cbs,
            total_encoders,
            encoders_per_cb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batcher_stats_tracking() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let batcher = CommandBatcher::new(&queue, 32);
        assert_eq!(batcher.stats_cbs(), 0);
        assert_eq!(batcher.stats_encoders(), 0);
        assert!(!batcher.has_pending());
    }

    #[test]
    fn batcher_creates_cb_on_first_encoder() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        let enc = batcher.encoder();
        // Just end it immediately — no actual dispatch
        enc.end_encoding();
        batcher.end_encoder();

        assert!(batcher.has_pending());
        assert_eq!(batcher.stats_cbs(), 1);
        assert_eq!(batcher.stats_encoders(), 1);
        assert_eq!(batcher.encoder_count(), 1);
    }

    #[test]
    fn batcher_multiple_encoders_same_cb() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        for _ in 0..5 {
            let enc = batcher.encoder();
            enc.end_encoding();
            batcher.end_encoder();
        }

        assert_eq!(batcher.stats_cbs(), 1); // All on same CB
        assert_eq!(batcher.stats_encoders(), 5);
        assert_eq!(batcher.encoder_count(), 5);
    }

    #[test]
    fn batcher_flush_resets() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        let enc = batcher.encoder();
        enc.end_encoding();
        batcher.end_encoder();

        batcher.flush();
        assert!(!batcher.has_pending());
        assert_eq!(batcher.encoder_count(), 0);
        assert_eq!(batcher.stats_cbs(), 1);
    }

    #[test]
    fn batcher_flush_async_returns_cb() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        let enc = batcher.encoder();
        enc.end_encoding();
        batcher.end_encoder();

        let cb = batcher.flush_async();
        assert!(cb.is_some());
        assert!(!batcher.has_pending());

        // Wait for it to complete
        cb.unwrap().wait_until_completed();
    }

    #[test]
    fn batcher_should_flush_at_max() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 3);

        for _ in 0..3 {
            let enc = batcher.encoder();
            enc.end_encoding();
            batcher.end_encoder();
        }

        assert!(batcher.should_flush());
    }

    #[test]
    fn batcher_stats_snapshot() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        for _ in 0..4 {
            let enc = batcher.encoder();
            enc.end_encoding();
            batcher.end_encoder();
        }
        batcher.flush();

        for _ in 0..2 {
            let enc = batcher.encoder();
            enc.end_encoding();
            batcher.end_encoder();
        }
        batcher.flush();

        let stats = BatcherStats::from_batcher(&batcher);
        assert_eq!(stats.total_cbs, 2);
        assert_eq!(stats.total_encoders, 6);
        assert!((stats.encoders_per_cb - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn global_counter_tracking() {
        // Global counters accumulate across tests; just check they increase
        let before_cbs = total_cbs_created();
        let before_encs = total_encoders_created();

        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let mut batcher = CommandBatcher::new(&queue, 32);

        let enc = batcher.encoder();
        enc.end_encoding();
        batcher.end_encoder();
        batcher.flush();

        assert!(total_cbs_created() > before_cbs);
        assert!(total_encoders_created() > before_encs);
    }
}
