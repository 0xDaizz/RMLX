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
//! Each [`Slab`] wraps an `MTLBuffer` allocated with `StorageModeShared`,
//! giving both the GPU and CPU access to the same physical memory on Apple
//! Silicon UMA. When these buffers are additionally registered as RDMA memory
//! regions (via [`rmlx_rdma::shared_buffer::SharedBuffer`]), the full path
//! from GPU compute output to RDMA wire transfer involves zero CPU copies.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice};
use rmlx_metal::event::{EventError, GpuEvent};
use rmlx_metal::{MTLResourceOptions, MtlBuffer};

// ── Slab ──

/// A single slab in the ring: a Metal buffer allocated with `StorageModeShared`.
///
/// On Apple Silicon UMA, `StorageModeShared` means the GPU and CPU access the
/// same physical memory. When the backing allocation is also registered as an
/// RDMA memory region (done externally), the GPU can write directly into
/// wire-ready memory.
///
/// `Debug` is manually implemented because `MtlBuffer` does not derive it.
pub struct Slab {
    /// Metal buffer (`storageModeShared` for UMA CPU/GPU access).
    pub metal_buffer: MtlBuffer,
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
    /// Mutex + Condvar for backpressure: producers block when ring is full.
    backpressure_lock: Mutex<()>,
    /// Condvar signalled when a slot becomes available (consumer frees a slot).
    not_full: Condvar,
    /// Counter: number of times a producer had to block because the ring was full.
    ring_full_count: AtomicU64,
}

impl SlabRing {
    /// Create a new slab ring with pre-allocated Metal buffers.
    ///
    /// Each slab is allocated as `storageModeShared` for UMA GPU/CPU access.
    /// The caller is responsible for additionally registering these buffers as
    /// RDMA memory regions if zero-copy RDMA transfer is desired.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, config: SlabRingConfig) -> Self {
        assert!(config.depth > 0, "slab ring depth must be > 0");
        assert!(config.slab_size > 0, "slab ring slab_size must be > 0");

        let mut slabs = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let metal_buffer = device
                .newBufferWithLength_options(
                    config.slab_size,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
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
            backpressure_lock: Mutex::new(()),
            not_full: Condvar::new(),
            ring_full_count: AtomicU64::new(0),
        }
    }

    /// Try to acquire the next slab without blocking.
    ///
    /// Returns [`SlabRingError::Full`] immediately if the ring is full.
    /// This is the lock-free fast path used internally and by callers that
    /// do not want to block.
    pub fn try_acquire_for_write(&self) -> Result<&Slab, SlabRingError> {
        loop {
            let prod = self.producer_pos.load(Ordering::Acquire);
            let cons = self.consumer_pos.load(Ordering::Acquire);

            if prod - cons >= self.depth as u64 {
                return Err(SlabRingError::Full);
            }

            // Atomically claim this slot via CAS. If another thread raced us
            // and advanced producer_pos, we retry.
            match self.producer_pos.compare_exchange_weak(
                prod,
                prod + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let idx = (prod % self.depth as u64) as usize;
                    return Ok(&self.slabs[idx]);
                }
                Err(_) => continue, // another thread won the race, retry
            }
        }
    }

    /// Acquire the next slab for GPU writing, blocking if the ring is full.
    ///
    /// Returns the slab index and a reference to its Metal buffer.
    /// If the ring is full, blocks on a [`Condvar`] until a consumer frees
    /// a slot. The lock-free CAS fast path is tried first; the Condvar
    /// blocking path is only entered when the ring is actually full.
    ///
    /// After the GPU writes into the slab, the caller must call [`produce()`]
    /// to signal that the data is ready for consumption.
    ///
    /// [`produce()`]: Self::produce
    pub fn acquire_for_write(&self) -> Result<&Slab, SlabRingError> {
        // Fast path: try lock-free CAS first.
        match self.try_acquire_for_write() {
            Ok(slab) => return Ok(slab),
            Err(SlabRingError::Full) => {}
            Err(e) => return Err(e),
        }

        // Slow path: ring is full, block on Condvar.
        self.ring_full_count.fetch_add(1, Ordering::Relaxed);
        let guard = self.backpressure_lock.lock().unwrap();
        // Use wait_while with a predicate: stay blocked while the ring is full.
        // The mutex is held across both the fullness check and the wait, and
        // consumers acquire the same mutex before calling notify_one, so
        // notifications are never lost.
        let _guard = self.not_full.wait_while(guard, |_| self.is_full()).unwrap();
        // Ring is no longer full; claim a slot.
        self.try_acquire_for_write()
    }

    /// Acquire the next slab for GPU writing, with a timeout.
    ///
    /// Like [`acquire_for_write()`] but returns [`SlabRingError::Timeout`]
    /// if no slot becomes available within `timeout`.
    ///
    /// [`acquire_for_write()`]: Self::acquire_for_write
    pub fn acquire_for_write_timeout(&self, timeout: Duration) -> Result<&Slab, SlabRingError> {
        // Fast path: try lock-free CAS first.
        match self.try_acquire_for_write() {
            Ok(slab) => return Ok(slab),
            Err(SlabRingError::Full) => {}
            Err(e) => return Err(e),
        }

        // Slow path: ring is full, block on Condvar with timeout.
        self.ring_full_count.fetch_add(1, Ordering::Relaxed);
        let guard = self.backpressure_lock.lock().unwrap();
        // Use wait_timeout_while with a predicate: stay blocked while full.
        // The mutex is held across both the fullness check and the wait, and
        // consumers acquire the same mutex before calling notify_one, so
        // notifications are never lost.
        let (guard, wait_result) = self
            .not_full
            .wait_timeout_while(guard, timeout, |_| self.is_full())
            .unwrap();
        drop(guard);
        if wait_result.timed_out() && self.is_full() {
            return Err(SlabRingError::Timeout);
        }
        // Ring is no longer full; claim a slot.
        self.try_acquire_for_write()
            .map_err(|_| SlabRingError::Timeout)
    }

    /// Signal that the GPU has finished writing to a producer slab.
    ///
    /// Encodes an event signal into the given command buffer so that when the
    /// GPU finishes executing `cb`, the event timeline advances and the consumer
    /// side can observe the new data.
    ///
    /// The producer position was already advanced by [`acquire_for_write()`],
    /// so this method only encodes the GPU event signal using the current
    /// producer position (which is the value that `acquire_for_write` claimed).
    ///
    /// [`acquire_for_write()`]: Self::acquire_for_write
    pub fn produce(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>) {
        // producer_pos was already advanced by acquire_for_write via CAS.
        // Signal the event at the current producer_pos value so consumers
        // can observe that the slab is ready.
        let current_val = self.producer_pos.load(Ordering::Acquire);
        self.event.signal_from_command_buffer(cb, current_val);
    }

    /// Try to acquire the next slab for RDMA reading (non-blocking).
    ///
    /// Returns `None` if no slab is ready (the ring is empty or the GPU has
    /// not yet finished writing the next slab).
    ///
    /// Uses a compare-and-swap loop on `consumer_pos` to atomically claim
    /// consumption, eliminating TOCTOU races between concurrent consumers.
    pub fn try_consume(&self) -> Option<&Slab> {
        loop {
            let cons = self.consumer_pos.load(Ordering::Acquire);
            let prod = self.producer_pos.load(Ordering::Acquire);

            if cons >= prod {
                return None;
            }

            // Atomically claim this slot for consumption via CAS.
            match self.consumer_pos.compare_exchange_weak(
                cons,
                cons + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let idx = (cons % self.depth as u64) as usize;
                    // Wake any blocked producer now that a slot is free.
                    // We must hold backpressure_lock while notifying to avoid
                    // a race where the producer checks fullness, finds the ring
                    // full, but hasn't entered wait() yet — the notification
                    // would be lost. Holding the lock ensures the producer
                    // either (a) sees the updated consumer_pos before waiting,
                    // or (b) is already in wait() and receives the notification.
                    let _guard = self.backpressure_lock.lock().unwrap();
                    self.not_full.notify_one();
                    return Some(&self.slabs[idx]);
                }
                Err(_) => continue, // another consumer won the race, retry
            }
        }
    }

    /// Block until a slab is ready for RDMA reading.
    ///
    /// Uses [`GpuEvent::cpu_wait()`] to wait for the GPU to finish writing
    /// the next slab, then atomically claims the slot via CAS on
    /// `consumer_pos`. Returns the slab reference on success.
    pub fn consume(&self, timeout: Duration) -> Result<&Slab, SlabRingError> {
        loop {
            let cons = self.consumer_pos.load(Ordering::Acquire);
            let target_val = cons + 1;

            // Wait for the event timeline to reach our target value.
            self.event.cpu_wait(target_val, timeout)?;

            // Atomically claim this slot. If another consumer raced us, retry.
            match self.consumer_pos.compare_exchange_weak(
                cons,
                cons + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let idx = (cons % self.depth as u64) as usize;
                    // Wake any blocked producer now that a slot is free.
                    // Hold backpressure_lock to prevent lost wakeups (see
                    // try_consume for detailed rationale).
                    let _guard = self.backpressure_lock.lock().unwrap();
                    self.not_full.notify_one();
                    return Ok(&self.slabs[idx]);
                }
                Err(_) => continue,
            }
        }
    }

    /// Release a consumed slab back to the pool (no-op).
    ///
    /// With the CAS-based `try_consume` and `consume`, the consumer position
    /// is advanced atomically at the point of consumption. This method is
    /// retained for API compatibility but performs no work. Existing callers
    /// can safely continue to call `release()` after consumption.
    pub fn release(&self) {
        // Consumer position was already advanced by try_consume/consume CAS.
        // Wake any blocked producer as an extra safety net — the primary
        // notification happens inside try_consume/consume, but legacy callers
        // may expect release() to unblock producers.
        let _guard = self.backpressure_lock.lock().unwrap();
        self.not_full.notify_one();
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

    /// Number of times a producer had to block because the ring was full.
    ///
    /// Useful for monitoring backpressure — a steadily increasing count
    /// indicates the consumer cannot keep up with the producer.
    pub fn ring_full_count(&self) -> u64 {
        self.ring_full_count.load(Ordering::Relaxed)
    }
}

// ── Tests ──

#[cfg(test)]
#[allow(clippy::arc_with_non_send_sync)]
mod tests {
    use super::*;
    use objc2_metal::{MTLBuffer as _, MTLCommandQueue as _};
    use std::time::Instant;

    /// Wrapper to assert Send+Sync for SlabRing in tests.
    /// Metal objects are thread-safe on Apple platforms; objc2-metal
    /// conservatively omits Send/Sync but the underlying API is safe.
    #[derive(Clone)]
    struct SendRing(Arc<SlabRing>);
    unsafe impl Send for SendRing {}
    unsafe impl Sync for SendRing {}
    impl std::ops::Deref for SendRing {
        type Target = SlabRing;
        fn deref(&self) -> &SlabRing {
            &self.0
        }
    }

    /// Helper: get the default Metal device, skip test if unavailable.
    fn require_device() -> rmlx_metal::MtlDevice {
        match objc2_metal::MTLCreateSystemDefaultDevice() {
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
    fn test_full_ring_try_acquire_fails() {
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
        assert!(ring.try_acquire_for_write().is_err());
        match ring.try_acquire_for_write() {
            Err(SlabRingError::Full) => {} // expected
            other => panic!("expected Full, got: {other:?}"),
        }
    }

    #[test]
    fn test_ring_lifecycle_with_cas() {
        // Test the full lifecycle: acquire -> try_consume -> acquire -> try_consume.
        // acquire_for_write now advances producer_pos via CAS, and try_consume
        // advances consumer_pos via CAS, so no manual position management needed.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Phase 1: acquire slab 0 for write (CAS advances producer_pos to 1).
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 0);
        assert_eq!(ring.producer_pos(), 1);
        assert_eq!(ring.in_flight(), 1);
        assert!(!ring.is_empty());
        assert!(!ring.is_full());

        // Phase 2: acquire slab 1 for write (CAS advances producer_pos to 2).
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 1);
        assert_eq!(ring.producer_pos(), 2);
        assert_eq!(ring.in_flight(), 2);

        // Phase 3: consume slab 0 (CAS advances consumer_pos to 1).
        let consumed = ring.try_consume().unwrap();
        assert_eq!(consumed.index, 0);
        assert_eq!(ring.consumer_pos_val(), 1);
        assert_eq!(ring.in_flight(), 1);

        // release is now a no-op but should be safe to call.
        ring.release();
        assert_eq!(ring.consumer_pos_val(), 1);
        assert_eq!(ring.in_flight(), 1);

        // Phase 4: consume slab 1 (CAS advances consumer_pos to 2).
        let consumed = ring.try_consume().unwrap();
        assert_eq!(consumed.index, 1);
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
        // acquire_for_write now advances producer_pos via CAS, and
        // try_consume advances consumer_pos via CAS.
        for round in 0..2u64 {
            let base = round * 2;

            // Produce 2 slabs (acquire_for_write CAS-advances producer_pos).
            for i in 0..2u64 {
                let slab = ring.acquire_for_write().unwrap();
                assert_eq!(slab.index, ((base + i) % 2) as usize);
            }
            assert!(ring.is_full());

            // Consume 2 slabs (try_consume CAS-advances consumer_pos).
            for i in 0..2u64 {
                let slab = ring.try_consume().unwrap();
                assert_eq!(slab.index, ((base + i) % 2) as usize);
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
        let queue = device.newCommandQueue().unwrap();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 1024,
        };
        let ring = SlabRing::new(&device, config);

        // Acquire slab 0 for writing.
        let slab = ring.acquire_for_write().unwrap();
        assert_eq!(slab.index, 0);

        // Create a command buffer, encode the event signal, and commit.
        let cb = queue.commandBuffer().unwrap();
        ring.produce(&cb);
        cb.commit();
        cb.waitUntilCompleted();

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
        let queue = device.newCommandQueue().unwrap();
        let config = SlabRingConfig {
            depth: 3,
            slab_size: 512,
        };
        let ring = SlabRing::new(&device, config);

        // Produce slab 0.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.commandBuffer().unwrap();
        ring.produce(&cb);
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(ring.in_flight(), 1);

        // Produce slab 1.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.commandBuffer().unwrap();
        ring.produce(&cb);
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(ring.in_flight(), 2);

        // Consume slab 0.
        let slab = ring.consume(Duration::from_secs(1)).unwrap();
        assert_eq!(slab.index, 0);
        ring.release();
        assert_eq!(ring.in_flight(), 1);

        // Produce slab 2.
        let _ = ring.acquire_for_write().unwrap();
        let cb = queue.commandBuffer().unwrap();
        ring.produce(&cb);
        cb.commit();
        cb.waitUntilCompleted();
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
        let ptr = slab.metal_buffer.contents().as_ptr() as *mut u8;
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
    fn test_concurrent_acquire_release() {
        // Test that multiple threads can concurrently acquire and consume
        // without TOCTOU races causing double-assignment of the same slab.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 8,
            slab_size: 256,
        };
        let ring = Arc::new(SlabRing::new(&device, config));

        let iterations = 100;
        let num_producers = 4;
        let total_produced = Arc::new(AtomicU64::new(0));
        let total_consumed = Arc::new(AtomicU64::new(0));

        // Producers: each tries to acquire slabs
        let mut handles = vec![];
        for _ in 0..num_producers {
            let ring = SendRing(Arc::clone(&ring));
            let produced = Arc::clone(&total_produced);
            let h = std::thread::spawn(move || {
                for _ in 0..iterations {
                    match ring.try_acquire_for_write() {
                        Ok(_slab) => {
                            produced.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(SlabRingError::Full) => {
                            // Ring full, that is fine under contention
                            std::thread::yield_now();
                        }
                        Err(e) => panic!("unexpected error: {e}"),
                    }
                }
            });
            handles.push(h);
        }

        // Consumer: drains the ring
        let ring_c = SendRing(Arc::clone(&ring));
        let consumed = Arc::clone(&total_consumed);
        let consumer = std::thread::spawn(move || {
            // Keep consuming until we have consumed everything producers produced.
            // We loop with a bounded retry to avoid infinite loops.
            let mut retries = 0;
            loop {
                match ring_c.try_consume() {
                    Some(_slab) => {
                        consumed.fetch_add(1, Ordering::Relaxed);
                        retries = 0;
                    }
                    None => {
                        retries += 1;
                        if retries > 10000 {
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            }
        });

        for h in handles {
            h.join().unwrap();
        }
        consumer.join().unwrap();

        let p = total_produced.load(Ordering::Relaxed);
        let c = total_consumed.load(Ordering::Relaxed);
        // Every produced slab should have been consumed (no double-counting).
        assert_eq!(p, c, "produced={p} consumed={c} must match");
        // The ring should be empty after all operations complete.
        assert!(ring.is_empty());
    }

    #[test]
    fn test_backpressure_producer_blocks_and_resumes() {
        // Verify that a producer blocks when the ring is full and resumes
        // when the consumer makes progress.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 256,
        };
        let ring = Arc::new(SlabRing::new(&device, config));

        // Fill the ring (depth=2).
        let _s0 = ring.acquire_for_write().unwrap();
        let _s1 = ring.acquire_for_write().unwrap();
        assert!(ring.is_full());

        let ring_prod = SendRing(Arc::clone(&ring));
        let produced = Arc::new(AtomicU64::new(0));
        let produced_c = Arc::clone(&produced);

        // Spawn a producer that will block because the ring is full.
        let producer = std::thread::spawn(move || {
            // This call should block until the consumer frees a slot.
            let slab = ring_prod.acquire_for_write().unwrap();
            produced_c.store(1, Ordering::Release);
            slab.index
        });

        // Give the producer a moment to enter the blocking path.
        std::thread::sleep(Duration::from_millis(50));
        // Producer should still be blocked (nothing consumed yet).
        assert_eq!(produced.load(Ordering::Acquire), 0);

        // Consumer frees a slot.
        let consumed = ring.try_consume().unwrap();
        assert_eq!(consumed.index, 0);

        // Producer should now unblock and complete.
        let idx = producer.join().unwrap();
        assert_eq!(produced.load(Ordering::Acquire), 1);
        // The producer got slab index 0 (position 2 % 2 = 0).
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_backpressure_timeout() {
        // Verify that acquire_for_write_timeout returns Timeout when the
        // ring is full and no consumer frees a slot within the timeout.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 1,
            slab_size: 256,
        };
        let ring = SlabRing::new(&device, config);

        // Fill the ring.
        let _s = ring.acquire_for_write().unwrap();
        assert!(ring.is_full());

        let result = ring.acquire_for_write_timeout(Duration::from_millis(50));
        match result {
            Err(SlabRingError::Timeout) => {} // expected
            other => panic!("expected Timeout, got: {other:?}"),
        }
    }

    #[test]
    fn test_backpressure_timeout_succeeds_when_freed() {
        // Verify that acquire_for_write_timeout succeeds if a slot is freed
        // before the timeout expires.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 1,
            slab_size: 256,
        };
        let ring = Arc::new(SlabRing::new(&device, config));

        // Fill the ring.
        let _s = ring.acquire_for_write().unwrap();
        assert!(ring.is_full());

        let ring_prod = SendRing(Arc::clone(&ring));
        let producer = std::thread::spawn(move || {
            ring_prod
                .acquire_for_write_timeout(Duration::from_secs(2))
                .map(|s| s.size) // extract owned data to avoid returning a borrow
        });

        // Free a slot after a short delay.
        std::thread::sleep(Duration::from_millis(50));
        let _ = ring.try_consume().unwrap();

        let result = producer.join().unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ring_full_count_increments() {
        // Verify that ring_full_count increments each time a producer blocks.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 1,
            slab_size: 256,
        };
        let ring = Arc::new(SlabRing::new(&device, config));

        assert_eq!(ring.ring_full_count(), 0);

        // Fill the ring.
        let _s = ring.acquire_for_write().unwrap();

        // First timeout: ring_full_count should increment.
        let _ = ring.acquire_for_write_timeout(Duration::from_millis(10));
        assert_eq!(ring.ring_full_count(), 1);

        // Second timeout: ring_full_count should increment again.
        let _ = ring.acquire_for_write_timeout(Duration::from_millis(10));
        assert_eq!(ring.ring_full_count(), 2);

        // Free the slot, acquire should succeed without incrementing.
        let _ = ring.try_consume().unwrap();
        let _ = ring.acquire_for_write().unwrap();
        assert_eq!(ring.ring_full_count(), 2);
    }

    #[test]
    fn test_backpressure_no_data_loss() {
        // Spin up a fast producer and slow consumer, verify all data is
        // transferred without loss.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 2,
            slab_size: 256,
        };
        let ring = Arc::new(SlabRing::new(&device, config));
        let total_items = 20u64;

        let ring_prod = SendRing(Arc::clone(&ring));
        let producer = std::thread::spawn(move || {
            for _ in 0..total_items {
                // acquire_for_write blocks if ring is full — no data loss.
                let _slab = ring_prod.acquire_for_write().unwrap();
            }
        });

        let ring_cons = SendRing(Arc::clone(&ring));
        let consumer = std::thread::spawn(move || {
            let mut consumed = 0u64;
            let mut retries = 0u64;
            while consumed < total_items {
                match ring_cons.try_consume() {
                    Some(_) => {
                        consumed += 1;
                        retries = 0;
                        // Simulate slow consumer.
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    None => {
                        retries += 1;
                        if retries > 100_000 {
                            panic!("consumer stalled after {consumed} items");
                        }
                        std::thread::yield_now();
                    }
                }
            }
            consumed
        });

        producer.join().unwrap();
        let consumed = consumer.join().unwrap();
        assert_eq!(consumed, total_items);
        // Backpressure should have kicked in since depth=2 < total_items=20.
        assert!(ring.ring_full_count() > 0);
    }

    #[test]
    fn test_backpressure_condvar_no_lost_wakeup() {
        // Stress test for the Condvar-based backpressure: many producers
        // block on a tiny ring while a consumer drains it. This exercises
        // the race between consumer notify_one and producer wait. Before
        // the fix (holding backpressure_lock during notify), producers
        // could miss wakeups and deadlock.
        let device = require_device();
        let config = SlabRingConfig {
            depth: 1, // Minimal ring — maximum contention.
            slab_size: 64,
        };
        let ring = Arc::new(SlabRing::new(&device, config));
        let total_per_producer = 50u64;
        let num_producers = 4;
        let total_items = total_per_producer * num_producers as u64;

        let produced = Arc::new(AtomicU64::new(0));
        let consumed = Arc::new(AtomicU64::new(0));

        // Spawn producers that all use blocking acquire_for_write.
        let mut handles = vec![];
        for _ in 0..num_producers {
            let ring = SendRing(Arc::clone(&ring));
            let produced = Arc::clone(&produced);
            let h = std::thread::spawn(move || {
                for _ in 0..total_per_producer {
                    let _slab = ring.acquire_for_write().unwrap();
                    produced.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(h);
        }

        // Consumer: drain until all items consumed.
        let ring_c = SendRing(Arc::clone(&ring));
        let consumed_c = Arc::clone(&consumed);
        let consumer = std::thread::spawn(move || {
            let mut count = 0u64;
            let deadline = Instant::now() + Duration::from_secs(30);
            while count < total_items {
                match ring_c.try_consume() {
                    Some(_) => {
                        count += 1;
                        consumed_c.fetch_add(1, Ordering::Relaxed);
                    }
                    None => {
                        if Instant::now() > deadline {
                            panic!(
                                "consumer stalled: consumed {count}/{total_items}, \
                                 produced {}, in_flight {}",
                                ring_c.producer_pos(),
                                ring_c.in_flight(),
                            );
                        }
                        std::thread::sleep(Duration::from_micros(10));
                    }
                }
            }
        });

        // Join all with a generous timeout to detect deadlocks.
        for h in handles {
            h.join().unwrap();
        }
        consumer.join().unwrap();

        assert_eq!(produced.load(Ordering::Relaxed), total_items);
        assert_eq!(consumed.load(Ordering::Relaxed), total_items);
        assert!(ring.is_empty());
        // With depth=1 and 200 total items, backpressure must have kicked in.
        assert!(ring.ring_full_count() > 0);
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
