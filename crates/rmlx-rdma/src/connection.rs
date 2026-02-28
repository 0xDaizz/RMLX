//! RDMA connection manager for multi-node TB5 clusters.
//!
//! Manages QP creation, TCP-based QP info exchange, state transitions,
//! and warmup protocol for all peers.

use std::ffi::c_void;

use crate::context::{ProtectionDomain, RdmaContext};
use crate::exchange;
use crate::ffi::*;
use crate::mr::MemoryRegion;
use crate::qp::{CompletionQueue, QueuePair};
use crate::RdmaError;

/// Configuration for an RDMA connection.
pub struct RdmaConfig {
    /// This node's rank (0 = server, 1+ = client)
    pub rank: u32,
    /// Total number of nodes
    pub world_size: u32,
    /// Peer host address (IP or hostname). For rank 0 this is unused.
    pub peer_host: String,
    /// TCP port for QP info exchange
    pub exchange_port: u16,
    /// TCP port for barrier sync
    pub sync_port: u16,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 2,
            peer_host: String::new(),
            exchange_port: exchange::TCP_EXCHANGE_PORT,
            sync_port: exchange::TCP_SYNC_PORT,
        }
    }
}

/// An established RDMA connection to a peer.
///
/// Wraps the full RDMA stack: context, protection domain, completion queue,
/// and queue pair. Individual components handle their own cleanup via Drop.
pub struct RdmaConnection {
    ctx: RdmaContext,
    _pd: ProtectionDomain,
    cq: CompletionQueue,
    qp: QueuePair,
    config: RdmaConfig,
}

impl RdmaConnection {
    /// Open device, create QP, exchange info with peer, and connect.
    ///
    /// This is the main entry point for establishing an RDMA connection:
    /// 1. Opens the default RDMA device
    /// 2. Allocates PD and CQ
    /// 3. Creates a UC Queue Pair
    /// 4. Exchanges QP info via TCP (server on rank 0, client on rank 1+)
    /// 5. Transitions QP through RESET → INIT → RTR → RTS
    pub fn establish(config: RdmaConfig) -> Result<Self, RdmaError> {
        // 1. Open RDMA device
        let ctx = RdmaContext::open_default()?;
        let pd = ctx.alloc_pd()?;
        let cq = CompletionQueue::new(&ctx)?;

        // 2. Create UC Queue Pair
        let mut qp = QueuePair::create_uc(&pd, &cq)?;
        qp.query_local_info(&ctx, config.rank)?;

        // 3. Exchange QP info via TCP
        let remote_info = if config.rank == 0 {
            exchange::exchange_server(qp.local_info(), config.exchange_port)?
        } else {
            exchange::exchange_client(qp.local_info(), &config.peer_host, config.exchange_port)?
        };

        // 4. Connect QP (RESET → INIT → RTR → RTS)
        qp.connect(&remote_info)?;

        Ok(Self {
            ctx,
            _pd: pd,
            cq,
            qp,
            config,
        })
    }

    /// Register a memory region for RDMA operations.
    ///
    /// # Safety
    /// `ptr` must be valid for `size` bytes and must remain valid until the
    /// returned `MemoryRegion` is dropped.
    pub unsafe fn register_mr(
        &self,
        ptr: *mut c_void,
        size: usize,
    ) -> Result<MemoryRegion, RdmaError> {
        // SAFETY: Caller guarantees ptr is valid for size bytes.
        MemoryRegion::register(&self._pd, ptr, size)
    }

    /// Post a send operation (IBV_WR_SEND).
    pub fn post_send(
        &self,
        mr: &MemoryRegion,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<(), RdmaError> {
        let mut sge = IbvSge {
            addr: (mr.addr() as u64) + offset as u64,
            length,
            lkey: mr.lkey(),
        };

        // SAFETY: sge is valid and points into the registered MR.
        // wr is zero-initialized and all required fields are set.
        let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
        wr.wr_id = wr_id;
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = wr_opcode::SEND;
        wr.send_flags = send_flags::SIGNALED;

        self.qp.post_send(&mut wr)
    }

    /// Post a receive operation.
    pub fn post_recv(
        &self,
        mr: &MemoryRegion,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<(), RdmaError> {
        let mut sge = IbvSge {
            addr: (mr.addr() as u64) + offset as u64,
            length,
            lkey: mr.lkey(),
        };

        let mut wr = IbvRecvWr {
            wr_id,
            next: std::ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
        };

        self.qp.post_recv(&mut wr)
    }

    /// Poll for completions. Returns completed work request count.
    pub fn poll_cq(&self, wc: &mut [IbvWc]) -> Result<usize, RdmaError> {
        self.cq.poll(wc)
    }

    /// Wait for exactly `n` completions, checking each for success status.
    ///
    /// Spins on poll_cq until `n` successful completions are received.
    /// Returns an error immediately if any completion has a non-success status.
    pub fn wait_completions(&self, n: usize) -> Result<(), RdmaError> {
        let mut completed = 0usize;
        // SAFETY: IbvWc is a plain C struct safe to zero-initialize.
        let mut wc: [IbvWc; 16] = core::array::from_fn(|_| unsafe { std::mem::zeroed() });

        while completed < n {
            let count = self.cq.poll(&mut wc)?;
            for (i, completion) in wc.iter().enumerate().take(count) {
                if completion.status != wc_status::SUCCESS {
                    return Err(RdmaError::CqPoll(format!(
                        "wc[{}] status={} (wr_id={})",
                        i, completion.status, completion.wr_id,
                    )));
                }
            }
            completed += count;
        }
        Ok(())
    }

    /// Run warmup: exchange 10 rounds of small dummy messages.
    ///
    /// This ensures the QP is fully operational and any lazy hardware
    /// initialization is completed before real data transfer begins.
    pub fn warmup(&self) -> Result<(), RdmaError> {
        const WARMUP_ROUNDS: usize = 10;
        const WARMUP_SIZE: usize = 4; // 4 bytes

        // Register a small buffer for warmup
        let mut warmup_buf = vec![0u8; WARMUP_SIZE];
        // SAFETY: warmup_buf is valid for WARMUP_SIZE bytes and lives for
        // the duration of this function, outliving the MemoryRegion.
        let mr = unsafe {
            MemoryRegion::register(
                &self._pd,
                warmup_buf.as_mut_ptr() as *mut c_void,
                WARMUP_SIZE,
            )?
        };

        for round in 0..WARMUP_ROUNDS {
            if self.config.rank == 0 {
                // Rank 0: recv then send
                self.post_recv(&mr, 0, WARMUP_SIZE as u32, round as u64)?;
                // Barrier: signal rank 1 that recv is posted
                exchange::tcp_barrier_server(self.config.sync_port)?;
                self.wait_completions(1)?; // wait for recv

                self.post_send(&mr, 0, WARMUP_SIZE as u32, (round + 100) as u64)?;
                self.wait_completions(1)?; // wait for send
            } else {
                // Rank 1: send then recv
                self.post_recv(&mr, 0, WARMUP_SIZE as u32, round as u64)?;
                // Barrier: wait for rank 0's recv to be posted
                exchange::tcp_barrier_client(&self.config.peer_host, self.config.sync_port)?;

                self.post_send(&mr, 0, WARMUP_SIZE as u32, (round + 100) as u64)?;
                self.wait_completions(1)?; // wait for send
                self.wait_completions(1)?; // wait for recv
            }
        }

        Ok(())
    }

    /// This node's rank.
    pub fn rank(&self) -> u32 {
        self.config.rank
    }

    /// Total world size.
    pub fn world_size(&self) -> u32 {
        self.config.world_size
    }

    /// Device name (e.g., TB5 RDMA device identifier).
    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }
}
