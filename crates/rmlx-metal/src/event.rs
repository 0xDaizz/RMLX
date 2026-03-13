//! GPU event for CPU-GPU synchronization via MTLSharedEvent.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::types::*;

/// GPU synchronization event wrapping MTLSharedEvent.
///
/// Provides CPU-side waiting with a spin→yield→sleep escalation strategy
/// for low-latency synchronization with GPU command buffer completion.
pub struct GpuEvent {
    event: MtlEvent,
    counter: AtomicU64,
    cancelled: AtomicBool,
}

impl GpuEvent {
    /// Create a new GPU event on the given device.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let event = device.newSharedEvent().unwrap();
        Self {
            event,
            counter: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
        }
    }

    /// Get the next signal value and increment the counter.
    pub fn next_value(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Current counter value.
    pub fn current_value(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Signal this event from a command buffer at the given value.
    pub fn signal_from_command_buffer(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        value: u64,
    ) {
        // MTLSharedEvent extends MTLEvent; upcast via pointer cast.
        let event: &ProtocolObject<dyn MTLEvent> = unsafe {
            &*(&*self.event as *const ProtocolObject<dyn MTLSharedEvent>
                as *const ProtocolObject<dyn MTLEvent>)
        };
        command_buffer.encodeSignalEvent_value(event, value);
    }

    /// Wait on the event from a command buffer.
    pub fn wait_from_command_buffer(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        value: u64,
    ) {
        let event: &ProtocolObject<dyn MTLEvent> = unsafe {
            &*(&*self.event as *const ProtocolObject<dyn MTLSharedEvent>
                as *const ProtocolObject<dyn MTLEvent>)
        };
        command_buffer.encodeWaitForEvent_value(event, value);
    }

    /// CPU-side wait for the event to reach the given value.
    /// Uses spin→yield→sleep escalation strategy.
    /// Returns Ok(elapsed) on success, Err on timeout or cancellation.
    pub fn cpu_wait(&self, value: u64, deadline: Duration) -> Result<Duration, EventError> {
        let start = Instant::now();
        let spin_threshold = Duration::from_micros(10);
        let yield_threshold = Duration::from_micros(100);

        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(EventError::Cancelled);
            }

            let current = self.event.signaledValue();
            if current >= value {
                return Ok(start.elapsed());
            }

            let elapsed = start.elapsed();
            if elapsed >= deadline {
                return Err(EventError::Timeout(elapsed));
            }

            // Escalation strategy
            if elapsed < spin_threshold {
                std::hint::spin_loop();
            } else if elapsed < yield_threshold {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    /// Cancel any pending waits.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Reset cancel flag.
    pub fn reset_cancel(&self) {
        self.cancelled.store(false, Ordering::Release);
    }

    /// Reset the event counter and signaled value to 0.
    /// Call this when reusing the event for a new pipeline iteration.
    pub fn reset(&self) {
        self.counter.store(0, Ordering::SeqCst);
        unsafe { self.event.setSignaledValue(0) };
    }

    /// Raw shared event reference.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }
}

/// Event wait errors.
#[derive(Debug)]
pub enum EventError {
    Timeout(Duration),
    Cancelled,
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventError::Timeout(d) => write!(f, "event wait timed out after {:?}", d),
            EventError::Cancelled => write!(f, "event wait cancelled"),
        }
    }
}

impl std::error::Error for EventError {}
