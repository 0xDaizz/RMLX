//! Safe wrapper around MTL4CounterHeap for GPU timestamp profiling.
//!
//! Provides zero-overhead access to Metal 4's counter heap API, making it
//! trivial to bracket GPU dispatches with `write_timestamp` calls and read
//! back precise timing information.

use std::ops::Range;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4CommandBuffer as MTL4CommandBufferProtocol,
    MTL4ComputeCommandEncoder as MTL4ComputeEncoderProtocol,
    MTL4CounterHeap as MTL4CounterHeapProtocol, MTL4CounterHeapDescriptor, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTL4TimestampHeapEntry, MTLDevice,
};

use crate::MetalError;

// ---------------------------------------------------------------------------
// GpuTimestamp
// ---------------------------------------------------------------------------

/// A resolved GPU timestamp in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuTimestamp {
    /// Raw tick value from the GPU counter.
    pub ticks: u64,
    /// Timestamp converted to microseconds using the device's timestamp frequency.
    pub microseconds: f64,
}

// ---------------------------------------------------------------------------
// CounterHeap
// ---------------------------------------------------------------------------

/// Safe wrapper around `MTL4CounterHeap` for GPU timestamp profiling.
///
/// # Usage
///
/// ```ignore
/// // Create a heap with capacity for 64 timestamps
/// let heap = CounterHeap::new(device.raw(), 64).unwrap();
///
/// // In your dispatch code:
/// heap.write_timestamp(&encoder, 0); // before dispatch
/// heap.write_timestamp(&encoder, 1); // after dispatch
///
/// // After GPU work completes:
/// let timestamps = heap.read_timestamps(0..2);
/// let elapsed_us = timestamps[1].microseconds - timestamps[0].microseconds;
/// ```
pub struct CounterHeap {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeapProtocol>>,
    capacity: usize,
    /// GPU timestamp frequency in ticks per second, cached from the device.
    tick_frequency: u64,
}

impl CounterHeap {
    /// Create a new counter heap for GPU timestamp profiling.
    ///
    /// - `device`: The Metal device to create the heap on.
    /// - `capacity`: Number of timestamp entries the heap can hold.
    ///
    /// Returns `Err` if the device does not support Metal 4 counter heaps or
    /// if creation fails for any other reason.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        capacity: usize,
    ) -> Result<Self, MetalError> {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: `capacity` is a user-chosen size; the descriptor accepts any
        // NSUInteger value and the device will reject invalid sizes via the
        // error return.
        unsafe {
            descriptor.setCount(capacity);
        }

        let heap = device
            .newCounterHeapWithDescriptor_error(&descriptor)
            .map_err(|e| {
                MetalError::PipelineCreate(format!("counter heap creation failed: {e}"))
            })?;

        let tick_frequency = device.queryTimestampFrequency();

        Ok(Self {
            heap,
            capacity,
            tick_frequency,
        })
    }

    /// Record a GPU timestamp at the given index using a command buffer.
    ///
    /// This captures a timestamp after all prior work in the command buffer
    /// has completed. Call this before and after GPU dispatches to measure
    /// elapsed time.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `index >= capacity`.
    #[inline]
    pub fn write_timestamp(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBufferProtocol>,
        index: usize,
    ) {
        debug_assert!(
            index < self.capacity,
            "counter heap index {index} out of bounds (capacity {})",
            self.capacity
        );
        // SAFETY: Index is debug-checked above; the MTL4 API does not
        // bounds-check but we validated capacity at creation time.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&self.heap, index);
        }
    }

    /// Record a GPU timestamp at the given index using a compute command encoder
    /// with explicit granularity control.
    ///
    /// - `Precise`: maximum accuracy, may incur overhead (encoder splitting).
    /// - `Relaxed`: lower overhead, may sample at encoder boundaries.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `index >= capacity`.
    #[inline]
    pub fn write_timestamp_with_granularity(
        &self,
        encoder: &ProtocolObject<dyn MTL4ComputeEncoderProtocol>,
        index: usize,
        granularity: MTL4TimestampGranularity,
    ) {
        debug_assert!(
            index < self.capacity,
            "counter heap index {index} out of bounds (capacity {})",
            self.capacity
        );
        // SAFETY: Index is debug-checked above.
        unsafe {
            encoder.writeTimestampWithGranularity_intoHeap_atIndex(granularity, &self.heap, index);
        }
    }

    /// Convenience: record a precise timestamp via a compute command encoder.
    #[inline]
    pub fn write_precise_timestamp(
        &self,
        encoder: &ProtocolObject<dyn MTL4ComputeEncoderProtocol>,
        index: usize,
    ) {
        self.write_timestamp_with_granularity(encoder, index, MTL4TimestampGranularity::Precise);
    }

    /// Read back resolved timestamps for the given range of indices.
    ///
    /// The caller must ensure that the GPU work writing these timestamps has
    /// completed (e.g., by waiting on a `MTLSharedEvent` or calling
    /// `waitUntilCompleted` on the command buffer).
    ///
    /// # Panics
    ///
    /// Panics if the range exceeds the heap capacity.
    pub fn read_timestamps(&self, range: Range<usize>) -> Vec<GpuTimestamp> {
        assert!(
            range.end <= self.capacity,
            "timestamp range {}..{} exceeds capacity {}",
            range.start,
            range.end,
            self.capacity
        );

        let ns_range = NSRange::new(range.start, range.len());

        // SAFETY: We validated that the range is within bounds above.
        // The caller is responsible for ensuring GPU work has completed.
        let data = unsafe { self.heap.resolveCounterRange(ns_range) };

        let Some(data) = data else {
            return Vec::new();
        };

        let entry_size = std::mem::size_of::<MTL4TimestampHeapEntry>();
        let entry_count = data.len() / entry_size;

        // SAFETY: The resolved data is immutable (freshly returned from the
        // Metal driver) and we do not mutate it while the slice is alive.
        let bytes = unsafe { data.as_bytes_unchecked() };

        let mut timestamps = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let offset = i * entry_size;
            // SAFETY: `resolveCounterRange` returns tightly-packed
            // MTL4TimestampHeapEntry structs. We verified the byte length.
            let entry = unsafe {
                let ptr = bytes.as_ptr().add(offset) as *const MTL4TimestampHeapEntry;
                *ptr
            };

            let microseconds = if self.tick_frequency > 0 {
                (entry.timestamp as f64 / self.tick_frequency as f64) * 1_000_000.0
            } else {
                0.0
            };

            timestamps.push(GpuTimestamp {
                ticks: entry.timestamp,
                microseconds,
            });
        }

        timestamps
    }

    /// Invalidate a range of entries, resetting them to zero.
    ///
    /// Call this before reusing heap indices to ensure stale data is cleared.
    /// The caller must ensure the heap is not in use on the GPU.
    pub fn invalidate(&self, range: Range<usize>) {
        assert!(
            range.end <= self.capacity,
            "invalidate range {}..{} exceeds capacity {}",
            range.start,
            range.end,
            self.capacity
        );
        let ns_range = NSRange::new(range.start, range.len());
        // SAFETY: Range validated above. Caller ensures no concurrent GPU use.
        unsafe {
            self.heap.invalidateCounterRange(ns_range);
        }
    }

    /// Set a debug label on the counter heap (visible in Xcode GPU tools).
    pub fn set_label(&self, label: &str) {
        let ns_label = NSString::from_str(label);
        self.heap.setLabel(Some(&ns_label));
    }

    /// Number of timestamp entries this heap can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// GPU timestamp tick frequency in Hz.
    #[inline]
    pub fn tick_frequency(&self) -> u64 {
        self.tick_frequency
    }

    /// Access the underlying `MTL4CounterHeap` protocol object.
    #[inline]
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4CounterHeapProtocol> {
        &self.heap
    }
}
