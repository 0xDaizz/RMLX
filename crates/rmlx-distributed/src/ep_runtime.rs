//! EP Runtime Context — unified hub connecting all async/zero-copy infrastructure.

use std::sync::{Arc, Mutex};

use rmlx_metal::event::GpuEvent;
use rmlx_rdma::gpu_doorbell::{DescriptorProxy, DescriptorRing};
use rmlx_rdma::progress::ProgressEngine;
use rmlx_rdma::shared_buffer::{ConnectionId, SharedBufferPool};

use crate::credit_manager::CreditManager;
use crate::perf_counters::global_counters;
use crate::transport::RdmaConnectionTransport;

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
}

impl Drop for EpRuntimeContext {
    fn drop(&mut self) {
        self.shutdown();
    }
}
