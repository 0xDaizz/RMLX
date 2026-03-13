//! GPU fence abstraction for cross-queue command buffer synchronization.
//!
//! Wraps `MTLSharedEvent` (available on all Apple Silicon) to provide
//! explicit signal/wait semantics between command buffers on different
//! queues.  This is the primitive used by [`crate::stream::StreamManager`]
//! to implement `synchronize(src, dst)`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::types::*;

/// Cross-queue GPU synchronization fence backed by `MTLSharedEvent`.
///
/// Each fence maintains a monotonically increasing counter.  Callers
/// encode `signal(cb, value)` on the producer queue and
/// `wait(cb, value)` on the consumer queue.  The GPU hardware stalls
/// the consumer command buffer until the event reaches the required value.
///
/// A CPU-side `cpu_wait` is also provided for host synchronization.
pub struct GpuFence {
    event: MtlEvent,
    counter: AtomicU64,
}

impl GpuFence {
    /// Create a new fence on `device` with initial signal value 0.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let event = device.newSharedEvent().unwrap();
        Self {
            event,
            counter: AtomicU64::new(0),
        }
    }

    /// Allocate the next signal value (monotonically increasing).
    pub fn next_value(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Current counter value (the last value returned by `next_value`).
    pub fn current_value(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Encode a signal on `command_buffer` that fires after the CB completes.
    ///
    /// The GPU will set the shared event to `value` once all preceding
    /// commands in this command buffer have finished executing.
    pub fn signal(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>, value: u64) {
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
        command_buffer.encodeSignalEvent_value(event, value);
    }

    /// Encode a wait on `command_buffer` until the shared event reaches `value`.
    ///
    /// The GPU will stall this command buffer until the event's signaled
    /// value is >= `value`.
    pub fn wait(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>, value: u64) {
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
        command_buffer.encodeWaitForEvent_value(event, value);
    }

    /// Block the CPU until the shared event reaches `value` or `timeout` elapses.
    ///
    /// Uses a spin -> yield -> sleep escalation strategy for low latency.
    ///
    /// Returns `Ok(elapsed)` on success, `Err(FenceError::Timeout)` if the
    /// deadline is exceeded.
    pub fn cpu_wait(&self, value: u64, timeout: Duration) -> Result<Duration, FenceError> {
        let start = Instant::now();
        let spin_threshold = Duration::from_micros(10);
        let yield_threshold = Duration::from_micros(100);

        loop {
            let current = self.event.signaledValue();
            if current >= value {
                return Ok(start.elapsed());
            }

            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(FenceError::Timeout {
                    expected: value,
                    actual: self.event.signaledValue(),
                    elapsed,
                });
            }

            if elapsed < spin_threshold {
                std::hint::spin_loop();
            } else if elapsed < yield_threshold {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    /// Read the current GPU-side signaled value.
    pub fn signaled_value(&self) -> u64 {
        self.event.signaledValue()
    }

    /// Reset the fence to value 0 (for reuse across pipeline iterations).
    pub fn reset(&self) {
        self.counter.store(0, Ordering::SeqCst);
        self.event.setSignaledValue(0);
    }

    /// Access the underlying `MTLSharedEvent`.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }
}

/// Errors from fence operations.
#[derive(Debug)]
pub enum FenceError {
    /// CPU wait timed out before the event reached the expected value.
    Timeout {
        expected: u64,
        actual: u64,
        elapsed: Duration,
    },
}

impl std::fmt::Display for FenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FenceError::Timeout {
                expected,
                actual,
                elapsed,
            } => write!(
                f,
                "fence wait timed out after {elapsed:?}: expected value {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for FenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fence_counter_monotonic() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let fence = GpuFence::new(&device);

        assert_eq!(fence.current_value(), 0);
        assert_eq!(fence.next_value(), 1);
        assert_eq!(fence.next_value(), 2);
        assert_eq!(fence.next_value(), 3);
        assert_eq!(fence.current_value(), 3);
    }

    #[test]
    fn test_fence_reset() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let fence = GpuFence::new(&device);

        fence.next_value();
        fence.next_value();
        fence.reset();

        assert_eq!(fence.current_value(), 0);
        assert_eq!(fence.signaled_value(), 0);
        assert_eq!(fence.next_value(), 1);
    }

    #[test]
    fn test_fence_signal_wait_roundtrip() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let queue = device.newCommandQueue().unwrap();
        let fence = GpuFence::new(&device);

        let value = fence.next_value();

        // Create a command buffer that signals the fence.
        let cb = queue.commandBuffer().unwrap();
        let encoder = cb.computeCommandEncoder().unwrap();
        encoder.endEncoding();
        fence.signal(&cb, value);
        cb.commit();

        // CPU-wait should succeed quickly.
        let result = fence.cpu_wait(value, Duration::from_secs(2));
        assert!(result.is_ok(), "fence cpu_wait should succeed");
    }

    #[test]
    fn test_fence_cpu_wait_already_signaled() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let fence = GpuFence::new(&device);

        // Value 0 is the initial signaled value, so waiting for 0 should return immediately.
        let result = fence.cpu_wait(0, Duration::from_millis(10));
        assert!(result.is_ok());
    }
}
