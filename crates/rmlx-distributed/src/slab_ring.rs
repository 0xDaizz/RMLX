//! RDMA Slab Ring for true zero-copy GPU-to-wire transfer.
//!
//! Pre-registers Metal buffers as RDMA memory regions, allowing the GPU to
//! write directly into wire-ready buffers. Uses a ring of slabs with
//! producer/consumer synchronization via `MTLSharedEvent` timeline.
//!
//! # Architecture
//!
//! ```text
//! GPU writes -> [Slab 0] -> RDMA sends -> [Slab 0 free]
//!               [Slab 1] <- GPU writes next batch
//!               [Slab 2] <- waiting
//! ```
//!
//! The ring has `depth` slabs. GPU produces into the next free slab,
//! RDMA consumes from the oldest filled slab. Event timeline values
//! track which slabs are ready for each role.
//!
//! # Zero-copy path
//!
//! Each [`Slab`] wraps a `metal::Buffer` allocated with `StorageModeShared`,
//! giving both the GPU and CPU access to the same physical memory on Apple
//! Silicon UMA. When these buffers are additionally registered as RDMA memory
//! regions (via [`rmlx_rdma::shared_buffer::SharedBuffer`]), the full path
//! from GPU compute output to RDMA wire transfer involves zero CPU copies.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rmlx_metal::event::{EventError, GpuEvent};

// ── Slab ──

/// A single slab in the ring: a Metal buffer allocated with `StorageModeShared`.
///
/// On Apple Silicon UMA, `StorageModeShared` means the GPU and CPU access the
/// same physical memory. When the backing allocation is also registered as an
/// RDMA memory region (done externally), the GPU can write directly into
/// wire-ready memory.
///
/// `Debug` is manually implemented because `metal::Buffer` does not derive it.
pub struct Slab {
    /// Metal buffer (`storageModeShared` for UMA CPU/GPU access).
    pub metal_buffer: metal::Buffer,
    /// Size in bytes.
    pub size: usize,
    /// Slab index in the ring.
    pub index: usize,
}

impl std::fmt::Debug for Slab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slab")
            .field("index", &self.index)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

// ── SlabRingConfig ──

/// Configuration for the slab ring.
#[derive(Debug, Clone)]
pub struct SlabRingConfig {
    /// Number of slabs in the ring (depth). More slabs = more pipelining.
    /// Typical: 2-4 for double/triple/quad buffering.
    pub depth: usize,
    /// Size of each slab in bytes.
    /// Should be large enough for max_tokens * hidden_dim * dtype_size.
    pub slab_size: usize,
}

impl Default for SlabRingConfig {
    fn default() -> Self {
        Self {
            depth: 3,
            slab_size: 4 * 1024 * 1024, // 4 MiB default
        }
    }
}

// ── SlabRingError ──

/// Errors from slab ring operations.
#[derive(Debug)]
pub enum SlabRingError {
    /// Ring is full — all slabs are in use.
    Full,
    /// Timeout waiting for a slab to become available.
    Timeout,
    /// GPU event wait failed.
    EventError(String),
}

impl std::fmt::Display for SlabRingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "slab ring is full"),
            Self::Timeout => write!(f, "slab ring timed out waiting for available slab"),
            Self::EventError(e) => write!(f, "slab ring event error: {e}"),
        }
    }
}

impl std::error::Error for SlabRingError {}

impl From<EventError> for SlabRingError {
    fn from(e: EventError) -> Self {
        match e {
            EventError::Timeout(_) => Self::Timeout,
            EventError::Cancelled => Self::EventError("event cancelled".into()),
        }
    }
}

// ── SlabRing ──

/// A fixed pool of pre-registered Metal buffers arranged as a ring.
///
/// The ring has a producer side (GPU writing) and a consumer side (RDMA sending).
/// Synchronization is done via [`GpuEvent`] timeline values so the CPU can wait
/// for the GPU to finish writing before initiating RDMA transfer.
///
/// # Positions
///
/// Both `producer_pos` and `consumer_pos` are monotonically increasing counters.
/// The actual slab index is `pos % depth`. The invariant is:
///
/// ```text
/// consumer_pos <= producer_pos <= consumer_pos + depth
/// ```
///
/// - `producer_pos - consumer_pos` = number of slabs that are filled / in-flight.
/// - The ring is full when `producer_pos - consumer_pos == depth`.
/// - The ring is empty when `producer_pos == consumer_pos`.
pub struct SlabRing {
    /// The slabs, indexed 0..depth.
    slabs: Vec<Slab>,
    /// Ring depth.
    depth: usize,
    /// Producer position (next slab for GPU to write into).
    /// Monotonically increasing; actual index = producer_pos % depth.
    producer_pos: AtomicU64,
    /// Consumer position (next slab for RDMA to send from).
    /// consumer_pos <= producer_pos always.
    consumer_pos: AtomicU64,
    /// GPU event for producer/consumer synchronization.
    /// Timeline value N means: the slab at position N has been produced.
    event: Arc<GpuEvent>,
    /// Config snapshot.
    config: SlabRingConfig,
}

impl SlabRing {
    /// Create a new slab ring with pre-allocated Metal buffers.
    ///
    /// Each slab is allocated as `storageModeShared` for UMA GPU/CPU access.
    /// The caller is responsible for additionally registering these buffers as
    /// RDMA memory regions if zero-copy RDMA transfer is desired.
    pub fn new(device: &metal::Device, config: SlabRingConfig) -> Self {
        assert!(config.depth > 0, "slab ring depth must be > 0");
        assert!(config.slab_size > 0, "slab ring slab_size must be > 0");

        let mut slabs = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let metal_buffer = device.new_buffer(
                config.slab_size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            slabs.push(Slab {
                metal_buffer,
                size: config.slab_size,
                index: i,
            });
        }

        let event = Arc::new(GpuEvent::new(device));

        Self {
            slabs,
            depth: config.depth,
            producer_pos: AtomicU64::new(0),
            consumer_pos: AtomicU64::new(0),
            event,
            config,
        }
    }

    /// Acquire the next slab for GPU writing.
    ///
    /// Returns the slab index and a reference to its Metal buffer.
    /// Returns [`SlabRingError::Full`] if the ring is full (all slabs are
    /// pending RDMA send).
    ///
    /// After the GPU writes into the slab, the caller must call [`produce()`]
    /// to signal that the data is ready for consumption.
    ///
    /// [`produce()`]: Self::produce
    pub fn acquire_for_write(&self) -> Result<&Slab, SlabRingError> {
        let prod = self.producer_pos.load(Ordering::Acquire);
        let cons = self.consumer_pos.load(Ordering::Acquire);

        if prod - cons >= self.depth as u64 {
            return Err(SlabRingError::Full);
        }

        let idx = (prod % self.depth as u64) as usize;
        Ok(&self.slabs[idx])
    }

    /// Signal that the GPU has finished writing to the current producer slab.
    ///
    /// Encodes an event signal into the given command buffer so that when the
    /// GPU finishes executing `cb`, the event timeline advances and the consumer
    /// side can observe the new data.
    ///
    /// The producer position is advanced atomically after encoding.
    pub fn produce(&self, cb: &metal::CommandBufferRef) {
        let next_val = self.producer_pos.load(Ordering::Acquire) + 1;
        self.event.signal_from_command_buffer(cb, next_val);
        self.producer_pos.fetch_add(1, Ordering::Release);
    }

    /// Try to acquire the next slab for RDMA reading (non-blocking).
    ///
    /// Returns `None` if no slab is ready (the ring is empty or the GPU has
    /// not yet finished writing the next slab).
    pub fn try_consume(&self) -> Option<&Slab> {
        let cons = self.consumer_pos.load(Ordering::Acquire);
        let prod = self.producer_pos.load(Ordering::Acquire);

        if cons >= prod {
            return None;
        }

        let idx = (cons % self.depth as u64) as usize;
        Some(&self.slabs[idx])
    }

    /// Block until a slab is ready for RDMA reading.
    ///
    /// Uses [`GpuEvent::cpu_wait()`] to wait for the GPU to finish writing
    /// the next slab. Returns the slab reference on success.
    pub fn consume(&self, timeout: Duration) -> Result<&Slab, SlabRingError> {
        let cons = self.consumer_pos.load(Ordering::Acquire);
        let target_val = cons + 1;

        // Wait for the event timeline to reach our target value.
        self.event.cpu_wait(target_val, timeout)?;

        let idx = (cons % self.depth as u64) as usize;
        Ok(&self.slabs[idx])
    }

    /// Release a consumed slab back to the pool.
    ///
    /// After the RDMA send is complete, call this to make the slab available
    /// for future GPU writes. The consumer position is advanced.
    pub fn release(&self) {
        self.consumer_pos.fetch_add(1, Ordering::Release);
    }

    /// Number of slabs currently in use (produced but not yet consumed+released).
    pub fn in_flight(&self) -> usize {
        let prod = self.producer_pos.load(Ordering::Acquire);
        let cons = self.consumer_pos.load(Ordering::Acquire);
        (prod - cons) as usize
    }

    /// Whether the ring is empty (no slabs waiting for consumption).
    pub fn is_empty(&self) -> bool {
        self.in_flight() == 0
    }

    /// Whether the ring is full (all slabs are in use).
    pub fn is_full(&self) -> bool {
        self.in_flight() >= self.depth
    }

    /// Total number of slabs (ring depth).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Size of each slab in bytes.
    pub fn slab_size(&self) -> usize {
        self.config.slab_size
    }

    /// Access the GPU event for external synchronization.
    pub fn event(&self) -> &Arc<GpuEvent> {
        &self.event
    }

    /// Get a slab by index (for direct buffer access).
    ///
    /// # Panics
    ///
    /// Panics if `index >= depth`.
    pub fn slab(&self, index: usize) -> &Slab {
        &self.slabs[index]
    }

    /// Current producer position (monotonically increasing).
    pub fn producer_pos(&self) -> u64 {
        self.producer_pos.load(Ordering::Acquire)
    }

    /// Current consumer position (monotonically increasing).
    pub fn consumer_pos_val(&self) -> u64 {
        self.consumer_pos.load(Ordering::Acquire)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: get the default Metal device, skip test if unavailable.
    fn require_device() -> metal::Device {
        match metal::Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("skipping test: no Metal device available");
                std::process::exit(0);
            }
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = SlabRingConfig::default();
        assert_eq!(cfg.depth, 3);
        assert_eq!(cfg.slab_size, 4 * 1024 * 1024);
    }

    #[test]
    fn test_new_ring_is_empty() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.in_flight(), 0);
        assert_eq!(ring.depth(), 3);
        assert_eq!(ring.slab_size(), 1024);
        assert_eq!(ring.producer_pos(), 0);
        assert_eq!(ring.consumer_pos_val(), 0);
    }

    #[test]
    fn test_slab_access() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 4096,
        };
        let ring = SlabRing::new(&device, config);

        let slab0 = ring.slab(0);
        assert_eq!(slab0.index, 0);
        assert_eq!(slab0.size, 4096);

        let slab1 = ring.slab(1);
        assert_eq!(slab1.index, 1);
        assert_eq!(slab1.size, 4096);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_slab_out_of_bounds() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);
        let _ = ring.slab(2);
    }

    #[test]
    fn test_acquire_write_returns_correct_slab() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // First acquire should return slab 0 (producer_pos=0 => index 0).
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 0);
    }

    #[test]
    fn test_try_consume_empty_ring() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Nothing produced yet, try_consume should return None.
        assert!(ring.try_consume().is_none());
    }

    #[test]
    fn test_full_ring_acquire_fails() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Simulate producing 2 slabs by manually advancing producer_pos.
        // We cannot call produce() without a real command buffer, so we advance
        // the atomic directly to test the full-ring guard.
        ring.producer_pos.store(2, Ordering::Release);

        assert!(ring.is_full());
        assert!(ring.acquire_for_write().is_err());
        match ring.acquire_for_write() {
            Err(SlabRingError::Full) => {} // expected
            other => panic!("expected Full, got: {other:?}"),
        }
    }

    #[test]
    fn test_ring_lifecycle_with_manual_positions() {
        // Test the full lifecycle: acquire -> produce -> consume -> release.
        // Since we can't encode real GPU commands in unit tests, we manually
        // advance positions to verify state machine correctness.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Phase 1: acquire slab 0 for write.
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 0);

        // Simulate produce (advance producer_pos without real CB).
        ring.producer_pos.store(1, Ordering::Release);
        assert_eq!(ring.in_flight(), 1);
        assert!(!ring.is_empty());
        assert!(!ring.is_full());

        // Phase 2: acquire slab 1 for write.
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 1);
        ring.producer_pos.store(2, Ordering::Release);
        assert_eq!(ring.in_flight(), 2);

        // Phase 3: consume slab 0 (try_consume).
        let consumed = ring.try_consume().unwrap();
        assert_eq!(consumed.index, 0); // consumer_pos=0 => index 0

        // Release slab 0.
        ring.release();
        assert_eq!(ring.consumer_pos_val(), 1);
        assert_eq!(ring.in_flight(), 1);

        // Phase 4: consume slab 1.
        let consumed = ring.try_consume().unwrap();
        assert_eq!(consumed.index, 1); // consumer_pos=1 => index 1

        ring.release();
        assert!(ring.is_empty());
    }

    #[test]
    fn test_ring_wraps_around() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Fill and drain the ring twice to test wrap-around.
        for round in 0..2u64 {
            let base = round * 2;

            // Produce 2 slabs.
            for i in 0..2u64 {
                let slab = ring.acquire_for_write().unwrap();
                assert_eq!(slab.index, ((base + i) % 2) as usize);
                ring.producer_pos.fetch_add(1, Ordering::Release);
            }
            assert!(ring.is_full());

            // Consume 2 slabs.
            for i in 0..2u64 {
                let slab = ring.try_consume().unwrap();
                assert_eq!(slab.index, ((base + i) % 2) as usize);
                ring.release();
            }
            assert!(ring.is_empty());
        }
    }

    #[test]
    fn test_consume_timeout_on_empty_ring() {
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Blocking consume on empty ring should time out.
        let result = ring.consume(Duration::from_millis(50));
        assert!(result.is_err());
    }

    #[test]
    fn test_produce_and_blocking_consume() {
        // Test the real produce() + consume() path with a Metal command buffer.
        let device = require_device();
        let queue = device.new_command_queue();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Acquire slab 0 for writing.
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 0);

        // Create a command buffer, encode the event signal, and commit.
        let cb = queue.new_command_buffer();
        ring.produce(cb);
        cb.commit();
        cb.wait_until_completed();

        // Now blocking consume should succeed immediately.
        let consumed = ring.consume(Duration::from_secs(1)).unwrap();
        assert_eq!(consumed.index, 0);

        ring.release();
        assert!(ring.is_empty());
    }

    #[test]
    fn test_multi_step_pipeline() {
        // Simulates a 3-deep pipeline: produce slab 0, produce slab 1,
        // consume slab 0, produce slab 2, consume slab 1, consume slab 2.
        let device = require_device();
        let queue = device.new_command_queue();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 512,
        };
        let ring = SlabRing::new(&device, config);

        // Produce slab 0.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.new_command_buffer();
        ring.produce(cb);
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(ring.in_flight(), 1);

        // Produce slab 1.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.new_command_buffer();
        ring.produce(cb);
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(ring.in_flight(), 2);

        // Consume slab 0.
        let slab = ring.consume(Duration::from_secs(1)).unwrap();
        assert_eq!(slab.index, 0);
        ring.release();
        assert_eq!(ring.in_flight(), 1);

        // Produce slab 2.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.new_command_buffer();
        ring.produce(cb);
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(ring.in_flight(), 2);

        // Consume slab 1.
        let slab = ring.consume(Duration::from_secs(1)).unwrap();
        assert_eq!(slab.index, 1);
        ring.release();

        // Consume slab 2.
        let slab = ring.consume(Duration::from_secs(1)).unwrap();
        assert_eq!(slab.index, 2);
        ring.release();

        assert!(ring.is_empty());
    }

    #[test]
    fn test_slab_metal_buffer_is_accessible() {
        // Verify that we can actually write to and read from the slab's Metal buffer.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 1,
            slab_size: 256,
        };
        let ring = SlabRing::new(&device, config);

        let slab = ring.slab(0);
        let ptr = slab.metal_buffer.contents() as *mut u8;
        assert!(!ptr.is_null());

        // Write a pattern and read it back (StorageModeShared = CPU-accessible).
        // SAFETY: ptr is valid for slab.size bytes (StorageModeShared Metal buffer).
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, slab.size);
            let slice = std::slice::from_raw_parts(ptr, slab.size);
            assert!(slice.iter().all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn test_error_display() {
        let full = SlabRingError::Full;
        assert_eq!(full.to_string(), "slab ring is full");

        let timeout = SlabRingError::Timeout;
        assert_eq!(
            timeout.to_string(),
            "slab ring timed out waiting for available slab"
        );

        let event = SlabRingError::EventError("test error".into());
        assert_eq!(event.to_string(), "slab ring event error: test error");
    }
}
