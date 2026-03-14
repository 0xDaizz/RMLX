//! GPU-Initiated RDMA descriptor ring.
//!
//! Provides a lock-free, GPU-visible ring buffer through which GPU compute
//! kernels can submit RDMA work requests without CPU involvement in the
//! data path. The CPU acts as a thin proxy that translates descriptors into
//! ibv_post_send/recv calls.
//!
//! # Data flow
//!
//! 1. GPU kernel writes `RdmaDescriptor`(s) into the ring's Metal buffer view
//! 2. GPU command buffer signals `submit_event` at the new head value
//! 3. CPU proxy thread detects the signal via `GpuEvent::cpu_wait`
//! 4. CPU reads descriptors from `tail..head`, posts ibv_post_send/recv
//! 5. CQ completion resolves the PendingOp in the ProgressEngine
//! 6. CPU signals `complete_event` so downstream GPU work can proceed
//!
//! # Ordering guarantees
//!
//! SharedEvent signal/wait provides happens-before ordering between GPU
//! writes and CPU reads. No atomics needed on head/tail — the event
//! signal is the synchronization point.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use objc2_metal::MTLSharedEvent as _;
use rmlx_metal::event::GpuEvent;

use crate::exchange_tag::{encode_wr_id, ExchangeTag};
use crate::shared_buffer::SharedBuffer;

/// RDMA operation type written by GPU into the descriptor.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmaOp {
    Send = 0,
    Recv = 1,
}

impl RdmaOp {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(RdmaOp::Send),
            1 => Some(RdmaOp::Recv),
            _ => None,
        }
    }
}

/// GPU-written RDMA work descriptor.
///
/// Cache-line aligned (64 bytes) for GPU coherency. The GPU compute kernel
/// writes these into the ring buffer; the CPU proxy reads them to post
/// RDMA operations.
///
/// # Layout (64 bytes)
/// - op (1B): RdmaOp — Send or Recv
/// - peer_id (1B): target node
/// - tag (1B): ExchangeTag as u8
/// - buf_slot (1B): SharedBuffer slot index
/// - offset (4B): byte offset into the SharedBuffer payload
/// - length (4B): transfer size in bytes
/// - _reserved (52B): padding to 64 bytes
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct RdmaDescriptor {
    pub op: u8,
    pub peer_id: u8,
    pub tag: u8,
    pub buf_slot: u8,
    pub offset: u32,
    pub length: u32,
    pub _reserved: [u8; 52],
}

const _: () = {
    assert!(std::mem::size_of::<RdmaDescriptor>() == 64);
    assert!(std::mem::align_of::<RdmaDescriptor>() == 64);
};

impl RdmaDescriptor {
    /// Create a new descriptor. Reserved bytes are zeroed.
    pub fn new(
        op: RdmaOp,
        peer_id: u8,
        tag: ExchangeTag,
        buf_slot: u8,
        offset: u32,
        length: u32,
    ) -> Self {
        Self {
            op: op as u8,
            peer_id,
            tag: tag as u8,
            buf_slot,
            offset,
            length,
            _reserved: [0u8; 52],
        }
    }

    /// Decode the op field.
    pub fn rdma_op(&self) -> Option<RdmaOp> {
        RdmaOp::from_u8(self.op)
    }

    /// Decode the tag field.
    pub fn exchange_tag(&self) -> Option<ExchangeTag> {
        ExchangeTag::from_u8(self.tag)
    }
}

impl std::fmt::Debug for RdmaDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaDescriptor")
            .field("op", &self.rdma_op())
            .field("peer_id", &self.peer_id)
            .field("tag", &self.exchange_tag())
            .field("buf_slot", &self.buf_slot)
            .field("offset", &self.offset)
            .field("length", &self.length)
            .finish()
    }
}

/// GPU-visible descriptor ring buffer.
///
/// The ring is backed by a `SharedBuffer` so both GPU and CPU can access
/// the same physical memory. Head and tail are plain u32 — ordering is
/// guaranteed by the SharedEvent signal/wait happens-before relationship.
pub struct DescriptorRing {
    /// Ring memory (GPU writes descriptors, CPU reads them).
    ring: SharedBuffer,
    /// Maximum number of descriptors in the ring.
    capacity: usize,
    /// CPU consumer position (next descriptor to read).
    /// Updated by CPU after processing descriptors.
    tail: u32,
    /// GPU producer position (next slot GPU will write to).
    /// Communicated via submit_event signal value.
    head: u32,
    /// GPU -> CPU: GPU signals this after writing descriptors.
    /// The signaled value equals the new head position.
    submit_event: Arc<GpuEvent>,
    /// CPU -> GPU: CPU signals this after RDMA completions.
    /// Downstream GPU compute waits on this before consuming results.
    complete_event: Arc<GpuEvent>,
    /// Monotonic sequence counter for wr_id generation.
    seq: u64,
}

// SAFETY: DescriptorRing's SharedBuffer is Send, GpuEvent is behind Arc,
// and the plain u32 head/tail are only accessed by the owning thread.
unsafe impl Send for DescriptorRing {}

impl DescriptorRing {
    /// Create a new descriptor ring.
    ///
    /// `ring` must be a SharedBuffer large enough for `capacity` descriptors
    /// (i.e., `ring.size() >= capacity * 64`).
    ///
    /// Both events should be freshly created (signaled_value == 0).
    pub fn new(
        ring: SharedBuffer,
        capacity: usize,
        submit_event: Arc<GpuEvent>,
        complete_event: Arc<GpuEvent>,
    ) -> Self {
        assert!(
            ring.size() >= capacity * std::mem::size_of::<RdmaDescriptor>(),
            "ring buffer too small: {} < {} * 64",
            ring.size(),
            capacity,
        );
        Self {
            ring,
            capacity,
            tail: 0,
            head: 0,
            submit_event,
            complete_event,
            seq: 0,
        }
    }

    /// The ring's Metal buffer, for binding to GPU compute kernels.
    pub fn metal_buffer(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.ring.metal_buffer()
    }

    /// The submit event (GPU signals after writing descriptors).
    pub fn submit_event(&self) -> &Arc<GpuEvent> {
        &self.submit_event
    }

    /// The complete event (CPU signals after RDMA completions).
    pub fn complete_event(&self) -> &Arc<GpuEvent> {
        &self.complete_event
    }

    /// Ring capacity in descriptor slots.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of descriptors available to read (head - tail).
    pub fn available(&self) -> u32 {
        self.head.wrapping_sub(self.tail)
    }

    /// Check if the submit_event indicates new descriptors are available.
    ///
    /// Updates `head` from the event's signaled value. Returns the number
    /// of new descriptors available.
    pub fn poll_submissions(&mut self) -> u32 {
        let signaled = self.submit_event.raw().signaledValue();
        let new_head = signaled as u32;
        if new_head != self.head {
            self.head = new_head;
        }
        self.available()
    }

    /// Read the next pending descriptor from the ring.
    ///
    /// Returns `None` if tail == head (no descriptors available).
    /// Advances tail by 1.
    pub fn pop_descriptor(&mut self) -> Option<RdmaDescriptor> {
        if self.tail == self.head {
            return None;
        }
        let idx = (self.tail as usize) % self.capacity;
        let desc = unsafe {
            let base = self.ring.as_ptr() as *const RdmaDescriptor;
            std::ptr::read(base.add(idx))
        };
        self.tail = self.tail.wrapping_add(1);
        Some(desc)
    }

    /// Generate a wr_id for a descriptor using the internal sequence counter.
    pub fn next_wr_id(&mut self, desc: &RdmaDescriptor) -> u64 {
        let seq = self.seq;
        self.seq += 1;
        let tag = ExchangeTag::from_u8(desc.tag).unwrap_or(ExchangeTag::Data);
        encode_wr_id(seq, tag, desc.buf_slot, desc.peer_id)
    }

    /// Signal the complete_event to notify the GPU that RDMA work has finished.
    ///
    /// The value should be the tail position after processing.
    pub fn signal_completion(&self, value: u64) {
        self.complete_event.raw().setSignaledValue(value);
    }

    /// Current tail position (CPU consumer).
    pub fn tail(&self) -> u32 {
        self.tail
    }

    /// Current head position (GPU producer, last known).
    pub fn head(&self) -> u32 {
        self.head
    }
}

/// Configuration for the descriptor proxy thread.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Timeout for waiting on submit_event signals.
    pub poll_timeout: Duration,
    /// Maximum descriptors to process per batch.
    pub batch_size: usize,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            poll_timeout: Duration::from_millis(100),
            batch_size: 64,
        }
    }
}

/// CPU proxy that translates GPU-written descriptors into RDMA operations.
///
/// Runs in a dedicated thread, waiting on the submit_event for new
/// descriptors and dispatching them via the ProgressEngine.
pub struct DescriptorProxy {
    shutdown: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Result returned by a `DescriptorHandler` after processing a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerResult {
    /// CQ completion confirmed for this WR — proxy may signal the GPU.
    CqConfirmed,
    /// Handler requests the proxy to stop (e.g., fatal error).
    Stop,
}

/// Callback invoked for each descriptor the proxy reads from the ring.
///
/// The callback receives the descriptor and the wr_id assigned to it.
/// It must post the actual ibv_post_send/recv **and** poll the CQ until
/// the work request completes before returning `CqConfirmed`.
///
/// Returns `HandlerResult::CqConfirmed` when the CQ has confirmed
/// completion for the WR, or `HandlerResult::Stop` to shut down the
/// proxy.
pub type DescriptorHandler = Box<dyn FnMut(&RdmaDescriptor, u64) -> HandlerResult + Send>;

impl DescriptorProxy {
    /// Start the proxy thread.
    ///
    /// The `handler` callback is invoked for each descriptor. The caller
    /// should implement the actual ibv_post_send/recv logic there, since
    /// the proxy does not have direct access to the QueuePair.
    pub fn start(
        mut ring: DescriptorRing,
        config: ProxyConfig,
        mut handler: DescriptorHandler,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("rmlx-gpu-proxy".into())
            .spawn(move || {
                while !shutdown_clone.load(Ordering::Acquire) {
                    // Wait for GPU to signal new descriptors
                    let next_expected = (ring.head.wrapping_add(1)) as u64;
                    match ring
                        .submit_event
                        .cpu_wait(next_expected, config.poll_timeout)
                    {
                        Ok(_) => {}
                        Err(_) => {
                            // Timeout — check shutdown and retry
                            continue;
                        }
                    }

                    // Read all available descriptors
                    ring.poll_submissions();
                    let mut confirmed = 0u32;
                    let mut batch = 0u32;
                    while batch < config.batch_size as u32 {
                        let desc = match ring.pop_descriptor() {
                            Some(d) => d,
                            None => break,
                        };
                        let wr_id = ring.next_wr_id(&desc);
                        batch += 1;

                        match handler(&desc, wr_id) {
                            HandlerResult::CqConfirmed => {
                                confirmed += 1;
                            }
                            HandlerResult::Stop => {
                                return; // handler requested stop
                            }
                        }
                    }

                    // Only signal completion to GPU after CQ has confirmed
                    // all work requests in this batch actually completed.
                    if confirmed > 0 {
                        ring.signal_completion(ring.tail() as u64);
                    }
                }
            })
            .expect("failed to spawn GPU proxy thread");

        Self {
            shutdown,
            handle: Some(handle),
        }
    }

    /// Signal the proxy thread to stop and wait for it to exit.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the proxy is still running.
    pub fn is_running(&self) -> bool {
        self.handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }
}

impl Drop for DescriptorProxy {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_size_and_alignment() {
        assert_eq!(std::mem::size_of::<RdmaDescriptor>(), 64);
        assert_eq!(std::mem::align_of::<RdmaDescriptor>(), 64);
    }

    #[test]
    fn descriptor_new_and_decode() {
        let desc = RdmaDescriptor::new(
            RdmaOp::Send,
            5,
            ExchangeTag::MoeDispatchPayload,
            2,
            1024,
            4096,
        );
        assert_eq!(desc.rdma_op(), Some(RdmaOp::Send));
        assert_eq!(desc.exchange_tag(), Some(ExchangeTag::MoeDispatchPayload));
        assert_eq!(desc.peer_id, 5);
        assert_eq!(desc.buf_slot, 2);
        assert_eq!(desc.offset, 1024);
        assert_eq!(desc.length, 4096);
    }

    #[test]
    fn descriptor_debug_format() {
        let desc = RdmaDescriptor::new(RdmaOp::Recv, 0, ExchangeTag::Data, 0, 0, 64);
        let s = format!("{:?}", desc);
        assert!(s.contains("Recv"));
        assert!(s.contains("Data"));
    }

    #[test]
    fn rdma_op_from_u8() {
        assert_eq!(RdmaOp::from_u8(0), Some(RdmaOp::Send));
        assert_eq!(RdmaOp::from_u8(1), Some(RdmaOp::Recv));
        assert_eq!(RdmaOp::from_u8(2), None);
        assert_eq!(RdmaOp::from_u8(255), None);
    }
}
