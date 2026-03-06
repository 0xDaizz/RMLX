//! EP Runtime Context — unified hub connecting all async/zero-copy infrastructure.

use std::sync::{Arc, Mutex};

use rmlx_alloc::zero_copy::CompletionTicket;
use rmlx_metal::event::GpuEvent;
use rmlx_rdma::gpu_doorbell::{DescriptorProxy, DescriptorRing};
use rmlx_rdma::progress::ProgressEngine;
use rmlx_rdma::shared_buffer::{ConnectionId, SharedBufferPool};

use crate::credit_manager::CreditManager;
use crate::perf_counters::global_counters;
use crate::transport::RdmaConnectionTransport;

/// Handle to a SharedBuffer acquired from the pool.
///
/// Stores the buffer's metadata (pointer, size, slot index) and a
/// `CompletionTicket` that keeps the buffer marked as in-use.
/// When the ticket's GPU and RDMA flags are both marked complete,
/// the pool's `acquire()` will consider the buffer available again.
pub struct AcquiredBuffer {
    /// Raw pointer to the buffer memory.
    pub ptr: *mut u8,
    /// Buffer size in bytes (page-aligned).
    pub size: usize,
    /// Slot index within the pool tier.
    pub slot_index: u8,
    /// Completion ticket — marks the buffer as in-use while held.
    ticket: CompletionTicket,
    /// Retains the pool so the underlying SharedBuffer allocation outlives this handle.
    _pool: Arc<Mutex<SharedBufferPool>>,
    /// Whether the caller actually wrote data into this buffer.
    used: bool,
}

// SAFETY: AcquiredBuffer owns a raw pointer into SharedBuffer storage managed
// by SharedBufferPool. The `_pool` Arc keeps that pool alive for the lifetime
// of this handle, so the backing allocation outlives the raw pointer. The
// pointer is valid for `size` bytes and CompletionTicket is Send+Sync.
unsafe impl Send for AcquiredBuffer {}

impl AcquiredBuffer {
    fn mark_complete(&self) {
        // Marking both flags makes `is_safe_to_free()` return true,
        // which in turn makes `SharedBuffer::is_available()` return true.
        self.ticket.mark_gpu_complete();
        self.ticket.mark_rdma_complete();
    }

    /// Mark this buffer as having been written to by the caller.
    pub fn mark_used(&mut self) {
        self.used = true;
    }

    /// Whether this buffer has been marked as written to.
    pub fn is_used(&self) -> bool {
        self.used
    }

    /// Release this buffer back to the pool by marking the ticket complete.
    ///
    /// After calling this, the underlying SharedBuffer will be eligible for
    /// re-acquisition on the next `pool.acquire()` call.
    pub fn release(self) {
        if !self.used {
            eprintln!(
                "[ep-runtime] releasing unused buffer slot={} size={}",
                self.slot_index, self.size
            );
        }
        self.mark_complete();
    }
}

impl Drop for AcquiredBuffer {
    fn drop(&mut self) {
        // Auto-release on drop: mark the ticket complete so the buffer
        // returns to the pool.
        self.mark_complete();
    }
}

/// Result type for paired send/recv buffer acquisition.
pub type SendRecvBuffers = (Vec<Option<AcquiredBuffer>>, Vec<Option<AcquiredBuffer>>);

/// Unified hub connecting transport, progress engine, shared buffers,
/// GPU events, and the descriptor proxy for EP runtime operation.
pub struct EpRuntimeContext {
    transport: Arc<RdmaConnectionTransport>,
    progress: Arc<ProgressEngine>,
    shared_pool: Arc<Mutex<SharedBufferPool>>,
    conn_ids: Vec<ConnectionId>,
    proxy: Option<DescriptorProxy>,
    credits: CreditManager,
    compute_event: Arc<GpuEvent>,
    transfer_event: Arc<GpuEvent>,
    booted: bool,
}

impl EpRuntimeContext {
    /// Create a new EP runtime context.
    ///
    /// Creates GpuEvents for compute/transfer synchronization.
    /// The proxy starts as `None` — call `boot_proxy` to start descriptor dispatch.
    /// Default recv credit buffer size (64KB — max UC message on TB5).
    const DEFAULT_CREDIT_BUF_SIZE: usize = 64 * 1024;
    /// Default minimum credits per (peer, tag) pair.
    const DEFAULT_MIN_CREDITS: usize = 4;

    pub fn new(
        transport: Arc<RdmaConnectionTransport>,
        progress: Arc<ProgressEngine>,
        shared_pool: Arc<Mutex<SharedBufferPool>>,
        conn_ids: Vec<ConnectionId>,
        device: &metal::Device,
    ) -> Self {
        let compute_event = Arc::new(GpuEvent::new(device));
        let transfer_event = Arc::new(GpuEvent::new(device));
        Self {
            transport,
            progress,
            shared_pool,
            conn_ids,
            proxy: None,
            credits: CreditManager::new(Self::DEFAULT_MIN_CREDITS, Self::DEFAULT_CREDIT_BUF_SIZE),
            compute_event,
            transfer_event,
            booted: false,
        }
    }

    /// Start the descriptor proxy with the given ring.
    ///
    /// The proxy thread reads GPU-written descriptors from the ring and
    /// dispatches them as RDMA operations via `dispatch_descriptor` on the transport.
    /// Each posted op is registered with the ProgressEngine. On CQ completion,
    /// the complete_event on the ring is signaled back to the GPU.
    ///
    /// `buffers` provides the (SharedBuffer, ConnectionId) lookup table used by
    /// `dispatch_descriptor` to find pre-registered MRs for each buf_slot.
    /// Pass an empty slice if buffers are managed externally.
    pub fn boot_proxy(
        &mut self,
        ring: DescriptorRing,
        buffers: Vec<(rmlx_rdma::shared_buffer::SharedBuffer, ConnectionId)>,
    ) -> Result<(), crate::group::DistributedError> {
        if self.booted {
            return Err(crate::group::DistributedError::Transport(
                "proxy already booted".into(),
            ));
        }

        let transport = Arc::clone(&self.transport);
        let progress = Arc::clone(&self.progress);
        let counters = global_counters();

        let handler = Box::new(
            move |desc: &rmlx_rdma::gpu_doorbell::RdmaDescriptor,
                  wr_id: u64|
                  -> rmlx_rdma::gpu_doorbell::HandlerResult {
                use rmlx_rdma::gpu_doorbell::HandlerResult;

                // Register the op first (before dispatch) to avoid CQ race.
                let pending = progress.register_op(wr_id);

                let result = transport.dispatch_descriptor(desc, wr_id, &buffers);
                match result {
                    Ok(()) => {
                        // rdma_ops_posted is already incremented inside dispatch_descriptor
                        // via record_rdma_transfer(). No additional increment needed here.

                        // Wait for CQ to confirm this WR completed before signaling
                        // the GPU that the operation is done.
                        match pending.wait(std::time::Duration::from_secs(5)) {
                            Ok(_completion) => HandlerResult::CqConfirmed,
                            Err(e) => {
                                eprintln!("[ep-runtime] CQ wait failed for wr_id={wr_id}: {e:?}");
                                counters
                                    .rdma_ops_error
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                HandlerResult::Stop
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[ep-runtime] dispatch_descriptor error: {e}");
                        // Cancel the registered op since WR was never posted
                        progress.cancel_op(wr_id);
                        counters
                            .rdma_ops_error
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Do NOT confirm completion — stop the proxy to prevent
                        // the GPU from proceeding on a failed RDMA write.
                        HandlerResult::Stop
                    }
                }
            },
        );

        let proxy = DescriptorProxy::start(
            ring,
            rmlx_rdma::gpu_doorbell::ProxyConfig::default(),
            handler,
        );
        self.proxy = Some(proxy);
        self.booted = true;

        Ok(())
    }

    /// Stop the proxy and clean up.
    pub fn shutdown(&mut self) {
        if let Some(ref mut proxy) = self.proxy {
            proxy.shutdown();
        }
        self.proxy = None;
        self.booted = false;
    }

    /// Whether the proxy has been booted.
    pub fn is_booted(&self) -> bool {
        self.booted
    }

    /// Transport reference.
    pub fn transport(&self) -> &Arc<RdmaConnectionTransport> {
        &self.transport
    }

    /// Progress engine reference.
    pub fn progress(&self) -> &Arc<ProgressEngine> {
        &self.progress
    }

    /// Shared buffer pool reference.
    pub fn shared_pool(&self) -> &Arc<Mutex<SharedBufferPool>> {
        &self.shared_pool
    }

    /// Connection IDs for peer connections.
    pub fn conn_ids(&self) -> &[ConnectionId] {
        &self.conn_ids
    }

    /// Credit manager reference (immutable).
    pub fn credits(&self) -> &CreditManager {
        &self.credits
    }

    /// Credit manager reference (mutable, for ensure/replenish operations).
    pub fn credits_mut(&mut self) -> &mut CreditManager {
        &mut self.credits
    }

    /// Compute-done event (GPU signals after layer compute finishes).
    pub fn compute_event(&self) -> &Arc<GpuEvent> {
        &self.compute_event
    }

    /// Transfer-done event (CPU signals after RDMA transfer completes).
    pub fn transfer_event(&self) -> &Arc<GpuEvent> {
        &self.transfer_event
    }

    /// Acquire paired send/recv SharedBuffers from the pool for each peer.
    ///
    /// Returns two vectors indexed by rank:
    /// - `send_bufs[rank]` — buffer for sending to that peer (None for self-rank)
    /// - `recv_bufs[rank]` — buffer for receiving from that peer (None for self-rank)
    ///
    /// Each acquired buffer has a `CompletionTicket` attached that keeps the
    /// underlying `SharedBuffer` marked as in-use. Dropping the `AcquiredBuffer`
    /// (or calling `.release()`) marks the ticket complete, returning the buffer
    /// to the pool.
    ///
    /// If `size` is 0, returns vectors of all `None` (no buffers acquired).
    pub fn acquire_send_recv_buffers(
        &self,
        size: usize,
        world_size: usize,
        local_rank: usize,
    ) -> Result<SendRecvBuffers, crate::group::DistributedError> {
        if size == 0 || world_size == 0 {
            let empty: Vec<Option<AcquiredBuffer>> = (0..world_size).map(|_| None).collect();
            let empty2: Vec<Option<AcquiredBuffer>> = (0..world_size).map(|_| None).collect();
            return Ok((empty, empty2));
        }

        let mut pool = self.shared_pool.lock().map_err(|e| {
            crate::group::DistributedError::Transport(format!(
                "failed to lock SharedBufferPool: {e}"
            ))
        })?;

        let mut send_bufs: Vec<Option<AcquiredBuffer>> = (0..world_size).map(|_| None).collect();
        let mut recv_bufs: Vec<Option<AcquiredBuffer>> = (0..world_size).map(|_| None).collect();

        for rank in 0..world_size {
            if rank == local_rank {
                continue;
            }

            // Acquire send buffer and attach a ticket to mark it in-use.
            let send = pool.acquire(size).ok_or_else(|| {
                crate::group::DistributedError::Transport(format!(
                    "no send buffer available for peer {rank} (need {size} bytes)"
                ))
            })?;
            let send_ticket = CompletionTicket::new();
            send.set_ticket(send_ticket.clone());
            send_bufs[rank] = Some(AcquiredBuffer {
                ptr: send.as_ptr(),
                size: send.size(),
                slot_index: send.slot_index(),
                ticket: send_ticket,
                _pool: Arc::clone(&self.shared_pool),
                used: false,
            });

            // Acquire recv buffer and attach a ticket to mark it in-use.
            let recv = pool.acquire(size).ok_or_else(|| {
                crate::group::DistributedError::Transport(format!(
                    "no recv buffer available for peer {rank} (need {size} bytes)"
                ))
            })?;
            let recv_ticket = CompletionTicket::new();
            recv.set_ticket(recv_ticket.clone());
            recv_bufs[rank] = Some(AcquiredBuffer {
                ptr: recv.as_ptr(),
                size: recv.size(),
                slot_index: recv.slot_index(),
                ticket: recv_ticket,
                _pool: Arc::clone(&self.shared_pool),
                used: false,
            });
        }

        Ok((send_bufs, recv_bufs))
    }
}

impl Drop for EpRuntimeContext {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_rdma::shared_buffer::SharedBufferPool;

    /// Helper: create a pool with the given tier sizes. Skips if no Metal device.
    fn make_pool(tier_sizes: &[usize]) -> Option<SharedBufferPool> {
        let device = metal::Device::system_default()?;
        SharedBufferPool::new(&device, tier_sizes).ok()
    }

    #[test]
    fn acquire_returns_correct_size_buffers_for_each_peer() {
        let pool = match make_pool(&[4096, 16384, 65536]) {
            Some(p) => p,
            None => return, // skip on CI without Metal
        };
        let pool = Arc::new(Mutex::new(pool));
        let device = metal::Device::system_default().unwrap();
        let progress = Arc::new(rmlx_rdma::progress::ProgressEngine::new());
        let transport = Arc::new(crate::transport::RdmaConnectionTransport::new(vec![], 0));

        let ctx = EpRuntimeContext::new(transport, progress, pool, vec![], &device);

        let world_size = 3;
        let local_rank = 1;
        let requested_size = 4096;

        let (send_bufs, recv_bufs) = ctx
            .acquire_send_recv_buffers(requested_size, world_size, local_rank)
            .expect("acquire should succeed");

        assert_eq!(send_bufs.len(), world_size);
        assert_eq!(recv_bufs.len(), world_size);

        // Self-rank should be None.
        assert!(send_bufs[local_rank].is_none());
        assert!(recv_bufs[local_rank].is_none());

        // Peer ranks should have buffers with size >= requested.
        for rank in 0..world_size {
            if rank == local_rank {
                continue;
            }
            let send = send_bufs[rank].as_ref().expect("send buf should exist");
            assert!(
                send.size >= requested_size,
                "send size {} < {}",
                send.size,
                requested_size
            );
            let recv = recv_bufs[rank].as_ref().expect("recv buf should exist");
            assert!(
                recv.size >= requested_size,
                "recv size {} < {}",
                recv.size,
                requested_size
            );
        }
    }

    #[test]
    fn buffers_returned_to_pool_after_drop() {
        // Create a pool with exactly 2 buffers per tier (PIPELINE=2).
        // Acquiring 2 buffers should exhaust the tier; after drop they should
        // be available again.
        let pool = match make_pool(&[4096]) {
            Some(p) => p,
            None => return,
        };
        let pool = Arc::new(Mutex::new(pool));
        let device = metal::Device::system_default().unwrap();
        let progress = Arc::new(rmlx_rdma::progress::ProgressEngine::new());
        let transport = Arc::new(crate::transport::RdmaConnectionTransport::new(vec![], 0));

        let ctx = EpRuntimeContext::new(transport, progress, Arc::clone(&pool), vec![], &device);

        // world_size=2, local_rank=0 => only peer rank 1 needs buffers (1 send + 1 recv = 2).
        // PIPELINE=2, so the pool has exactly 2 buffers in the 4096 tier.
        {
            let (send_bufs, recv_bufs) = ctx
                .acquire_send_recv_buffers(4096, 2, 0)
                .expect("first acquire should succeed");

            // Both buffers are now in use.
            {
                let mut locked = pool.lock().unwrap();
                assert!(
                    locked.acquire(4096).is_none(),
                    "pool should be exhausted while buffers are held"
                );
            }

            // Drop send and recv buffers (out of scope).
            drop(send_bufs);
            drop(recv_bufs);
        }

        // After drop, buffers should be available again.
        {
            let mut locked = pool.lock().unwrap();
            assert!(
                locked.acquire(4096).is_some(),
                "pool should have buffers available after release"
            );
        }
    }

    #[test]
    fn acquire_with_size_zero_returns_empty() {
        let pool = match make_pool(&[4096]) {
            Some(p) => p,
            None => return,
        };
        let pool = Arc::new(Mutex::new(pool));
        let device = metal::Device::system_default().unwrap();
        let progress = Arc::new(rmlx_rdma::progress::ProgressEngine::new());
        let transport = Arc::new(crate::transport::RdmaConnectionTransport::new(vec![], 0));

        let ctx = EpRuntimeContext::new(transport, progress, pool, vec![], &device);

        let (send_bufs, recv_bufs) = ctx
            .acquire_send_recv_buffers(0, 4, 0)
            .expect("zero-size acquire should succeed");

        assert_eq!(send_bufs.len(), 4);
        assert_eq!(recv_bufs.len(), 4);
        // All should be None since size is 0.
        assert!(send_bufs.iter().all(|b| b.is_none()));
        assert!(recv_bufs.iter().all(|b| b.is_none()));
    }

    #[test]
    fn acquired_buffer_tracks_usage() {
        let pool = match make_pool(&[4096]) {
            Some(p) => p,
            None => return,
        };
        let pool = Arc::new(Mutex::new(pool));
        let device = metal::Device::system_default().unwrap();
        let progress = Arc::new(rmlx_rdma::progress::ProgressEngine::new());
        let transport = Arc::new(crate::transport::RdmaConnectionTransport::new(vec![], 0));

        let ctx = EpRuntimeContext::new(transport, progress, pool, vec![], &device);

        let (mut send_bufs, recv_bufs) = ctx
            .acquire_send_recv_buffers(4096, 2, 0)
            .expect("acquire should succeed");

        let send = send_bufs[1].as_mut().expect("send buf should exist");
        assert!(!send.is_used(), "buffer should start unused");
        send.mark_used();
        assert!(send.is_used(), "buffer should report used after mark_used");

        drop(recv_bufs);
        drop(send_bufs);
    }
}
