//! RDMA connection manager for multi-node TB5 clusters.
//!
//! Manages QP creation, TCP-based QP info exchange, state transitions,
//! and warmup protocol for all peers.

use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use crate::context::{ProtectionDomain, RdmaContext};
use crate::exchange;
use crate::ffi::*;
use crate::mr::MemoryRegion;
use crate::mr_pool::{MrHandle, MrPool};
use crate::qp::{CompletionQueue, QueuePair};
use crate::shared_buffer::{ConnectionId, SharedBuffer};
use crate::RdmaError;

/// The kind of RDMA work request that was posted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostedOpKind {
    Send,
    Recv,
}

/// A posted RDMA operation that borrows a `MemoryRegion`.
///
/// The lifetime `'a` ties this operation to the `MemoryRegion` it references,
/// ensuring at compile time that the MR cannot be dropped while the operation
/// is still outstanding. Call `RdmaConnection::wait_posted` to wait for
/// completion before the borrow is released.
#[derive(Debug)]
pub struct PostedOp<'a> {
    /// The work request ID assigned to this operation.
    pub wr_id: u64,
    /// Whether this is a send or receive operation.
    pub kind: PostedOpKind,
    /// Ties the lifetime of this operation to the MemoryRegion.
    _mr: PhantomData<&'a MemoryRegion>,
}

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
    /// CQ poll timeout in milliseconds (default: 5000)
    pub cq_timeout_ms: u64,
    /// Timeout in seconds for server accept() calls (default: 60)
    pub accept_timeout_secs: u64,
    /// Number of retries for TCP I/O operations (default: 3)
    pub io_max_retries: u32,
    /// Delay between I/O retries in milliseconds (default: 1000)
    pub io_retry_delay_ms: u64,
    /// TCP connect timeout in milliseconds (default: 5000)
    pub connect_timeout_ms: u64,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 2,
            peer_host: String::new(),
            exchange_port: exchange::TCP_EXCHANGE_PORT,
            sync_port: exchange::TCP_SYNC_PORT,
            cq_timeout_ms: 5000,
            accept_timeout_secs: 60,
            io_max_retries: 3,
            io_retry_delay_ms: 1000,
            connect_timeout_ms: 5000,
        }
    }
}

/// Tracks work request completions by wr_id with timeout support.
///
/// Maintains a backlog of completions that arrived but were not yet claimed,
/// allowing out-of-order wr_id matching.
pub struct CompletionTracker {
    expected: Vec<u64>,
    completed: Vec<u64>,
    backlog: Vec<IbvWc>,
}

impl CompletionTracker {
    pub fn new() -> Self {
        Self {
            expected: Vec::new(),
            completed: Vec::new(),
            backlog: Vec::new(),
        }
    }

    /// Register an expected wr_id for tracking.
    pub fn expect(&mut self, wr_id: u64) {
        self.expected.push(wr_id);
    }

    /// Wait for a specific wr_id completion from the CQ, with timeout.
    ///
    /// Checks the backlog first, then polls the CQ until the matching
    /// completion arrives or the timeout expires.
    pub fn wait_for(
        &mut self,
        wr_id: u64,
        cq: &CompletionQueue,
        timeout_ms: u64,
    ) -> Result<IbvWc, RdmaError> {
        // Check backlog first
        if let Some(pos) = self.backlog.iter().position(|wc| wc.wr_id == wr_id) {
            let wc = self.backlog.swap_remove(pos);
            if wc.status != wc_status::SUCCESS {
                return Err(RdmaError::CqPoll(format!(
                    "wc status={} for wr_id={wr_id}",
                    wc.status
                )));
            }
            self.completed.push(wr_id);
            return Ok(wc);
        }

        // Poll CQ with timeout
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| unsafe { std::mem::zeroed() });

        loop {
            let count = cq.poll(&mut wc_buf)?;
            for wc in &wc_buf[..count] {
                if wc.wr_id == wr_id {
                    if wc.status != wc_status::SUCCESS {
                        return Err(RdmaError::CqPoll(format!(
                            "wc status={} for wr_id={wr_id}",
                            wc.status
                        )));
                    }
                    self.completed.push(wr_id);
                    return Ok(*wc);
                }
                // Not our target — stash in backlog
                self.backlog.push(*wc);
            }
            if Instant::now() >= deadline {
                return Err(RdmaError::Timeout(format!(
                    "wr_id={wr_id} not completed within {timeout_ms}ms"
                )));
            }
            std::thread::yield_now();
        }
    }

    /// Drain all expected wr_ids, returning once all are completed or timeout.
    pub fn wait_all(
        &mut self,
        cq: &CompletionQueue,
        timeout_ms: u64,
    ) -> Result<Vec<IbvWc>, RdmaError> {
        let expected: Vec<u64> = self.expected.drain(..).collect();
        let mut results = Vec::with_capacity(expected.len());
        for wr_id in expected {
            results.push(self.wait_for(wr_id, cq, timeout_ms)?);
        }
        Ok(results)
    }
}

impl Default for CompletionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Poll CQ with a timeout. Returns all completions polled before the deadline.
///
/// Spins on poll_cq until at least one completion is available or the
/// timeout expires.
pub fn poll_with_timeout(cq: &CompletionQueue, timeout_ms: u64) -> Result<Vec<IbvWc>, RdmaError> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| unsafe { std::mem::zeroed() });

    loop {
        let count = cq.poll(&mut wc_buf)?;
        if count > 0 {
            return Ok(wc_buf[..count].to_vec());
        }
        if Instant::now() >= deadline {
            return Err(RdmaError::Timeout(format!(
                "no completions within {timeout_ms}ms"
            )));
        }
        std::thread::yield_now();
    }
}

/// Safe wrapper for a send-side memory registration.
///
/// Lifetime-tied to the source data slice: the MR cannot outlive the data
/// it was registered from.
pub struct RegisteredSend<'a> {
    mr: MemoryRegion,
    _lt: PhantomData<&'a [u8]>,
}

impl<'a> RegisteredSend<'a> {
    /// Access the underlying `MemoryRegion`.
    pub fn mr(&self) -> &MemoryRegion {
        &self.mr
    }
}

/// Safe wrapper for a recv-side memory registration.
///
/// Lifetime-tied to the mutable buffer: the MR cannot outlive the buffer
/// it was registered from.
pub struct RegisteredRecv<'a> {
    mr: MemoryRegion,
    _lt: PhantomData<&'a mut [u8]>,
}

impl<'a> RegisteredRecv<'a> {
    /// Access the underlying `MemoryRegion`.
    pub fn mr(&self) -> &MemoryRegion {
        &self.mr
    }
}

/// An established RDMA connection to a peer.
///
/// Wraps the full RDMA stack: context, protection domain, completion queue,
/// and queue pair. Individual components handle their own cleanup via Drop.
pub struct RdmaConnection {
    ctx: RdmaContext,
    pd: ProtectionDomain,
    cq: CompletionQueue,
    qp: QueuePair,
    config: RdmaConfig,
    /// Backlog of CQ completions with unexpected wr_ids, for later retrieval.
    completion_backlog: RefCell<Vec<IbvWc>>,
    /// Pre-registered MR pool (lazily initialized via `init_mr_pool`).
    mr_pool: Option<MrPool>,
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
        let mut qp = QueuePair::create_uc(&pd, &cq, &ctx)?;
        qp.query_local_info(&ctx, config.rank)?;

        // 3. Exchange QP info via TCP
        let exchange_cfg = exchange::ExchangeConfig {
            accept_timeout_secs: config.accept_timeout_secs,
            io_max_retries: config.io_max_retries,
            io_retry_delay_ms: config.io_retry_delay_ms,
            connect_timeout_ms: config.connect_timeout_ms,
        };
        let remote_info = if config.rank == 0 {
            exchange::exchange_server(qp.local_info(), config.exchange_port, &exchange_cfg)?
        } else {
            exchange::exchange_client(
                qp.local_info(),
                &config.peer_host,
                config.exchange_port,
                &exchange_cfg,
            )?
        };

        // 4. Connect QP (RESET → INIT → RTR → RTS)
        qp.connect(&remote_info)?;

        Ok(Self {
            ctx,
            pd,
            cq,
            qp,
            config,
            completion_backlog: RefCell::new(Vec::new()),
            mr_pool: None,
        })
    }

    /// Register a memory region for RDMA operations.
    ///
    /// Uses probed `max_mr_size` from the device if available, otherwise
    /// falls back to `DEFAULT_MAX_MR_SIZE`.
    ///
    /// # Safety
    /// `ptr` must be valid for `size` bytes and must remain valid until the
    /// returned `MemoryRegion` is dropped.
    pub unsafe fn register_mr(
        &self,
        ptr: *mut c_void,
        size: usize,
    ) -> Result<MemoryRegion, RdmaError> {
        let max_mr_size = self.ctx.probe().map(|p| p.max_mr_size).unwrap_or_else(|| {
            eprintln!(
                "[rmlx-rdma] WARN: probe unavailable, using DEFAULT_MAX_MR_SIZE={}",
                crate::mr::DEFAULT_MAX_MR_SIZE
            );
            crate::mr::DEFAULT_MAX_MR_SIZE
        });
        // SAFETY: Caller guarantees ptr is valid for size bytes.
        unsafe { MemoryRegion::register_with_limit(&self.pd, ptr, size, max_mr_size) }
    }

    /// Safe MR registration for a send data slice.
    ///
    /// The returned `RegisteredSend` borrows `data` for lifetime `'a`,
    /// ensuring the MR cannot outlive the data it was registered from.
    pub fn register_send_slice<'a>(&self, data: &'a [u8]) -> Result<RegisteredSend<'a>, RdmaError> {
        // SAFETY: data is a valid &[u8]; lifetime 'a outlives RegisteredSend<'a>.
        let mr = unsafe { self.register_mr(data.as_ptr() as *mut c_void, data.len())? };
        Ok(RegisteredSend {
            mr,
            _lt: PhantomData,
        })
    }

    /// Safe MR registration for a recv buffer.
    ///
    /// The returned `RegisteredRecv` borrows `data` for lifetime `'a`,
    /// ensuring the MR cannot outlive the buffer it was registered from.
    pub fn register_recv_slice<'a>(
        &self,
        data: &'a mut [u8],
    ) -> Result<RegisteredRecv<'a>, RdmaError> {
        // SAFETY: data is a valid &mut [u8]; lifetime 'a outlives RegisteredRecv<'a>.
        let mr = unsafe { self.register_mr(data.as_mut_ptr() as *mut c_void, data.len())? };
        Ok(RegisteredRecv {
            mr,
            _lt: PhantomData,
        })
    }

    /// Post a send operation (IBV_WR_SEND).
    ///
    /// Returns a `PostedOp` that borrows the `MemoryRegion`, preventing the MR
    /// from being dropped until the operation completes via `wait_posted`.
    pub fn post_send<'a>(
        &self,
        mr: &'a MemoryRegion,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<PostedOp<'a>, RdmaError> {
        if (offset as u64) + (length as u64) > mr.length() as u64 {
            return Err(RdmaError::InvalidArgument(format!(
                "SGE out of bounds: offset({}) + length({}) > mr.length({})",
                offset,
                length,
                mr.length()
            )));
        }

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

        self.qp.post_send(&mut wr)?;
        Ok(PostedOp {
            wr_id,
            kind: PostedOpKind::Send,
            _mr: PhantomData,
        })
    }

    /// Post a receive operation.
    ///
    /// Returns a `PostedOp` that borrows the `MemoryRegion`, preventing the MR
    /// from being dropped until the operation completes via `wait_posted`.
    pub fn post_recv<'a>(
        &self,
        mr: &'a MemoryRegion,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<PostedOp<'a>, RdmaError> {
        if (offset as u64) + (length as u64) > mr.length() as u64 {
            return Err(RdmaError::InvalidArgument(format!(
                "SGE out of bounds: offset({}) + length({}) > mr.length({})",
                offset,
                length,
                mr.length()
            )));
        }

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

        self.qp.post_recv(&mut wr)?;
        Ok(PostedOp {
            wr_id,
            kind: PostedOpKind::Recv,
            _mr: PhantomData,
        })
    }

    /// Poll for completions. Returns completed work request count.
    pub fn poll_cq(&self, wc: &mut [IbvWc]) -> Result<usize, RdmaError> {
        self.cq.poll(wc)
    }

    /// Wait for specific completions identified by `wr_id`, with default timeout.
    ///
    /// Polls the CQ until all expected wr_ids are matched. Unexpected completions
    /// are stashed in a backlog buffer for later retrieval.
    pub fn wait_completions(&self, expected_wr_ids: &[u64]) -> Result<(), RdmaError> {
        self.wait_completions_with_timeout(expected_wr_ids, self.config.cq_timeout_ms)
    }

    /// Wait for posted operations to complete, using the default CQ timeout.
    ///
    /// Consumes the `PostedOp` handles, releasing the MR borrow once all
    /// operations have completed. This is the preferred API over raw
    /// `wait_completions` as it provides compile-time lifetime safety.
    pub fn wait_posted(&self, ops: &[PostedOp<'_>]) -> Result<(), RdmaError> {
        let wr_ids: Vec<u64> = ops.iter().map(|op| op.wr_id).collect();
        self.wait_completions(&wr_ids)
    }

    /// Wait for specific completions identified by `wr_id`, with explicit timeout.
    ///
    /// Polls the CQ until all expected wr_ids are matched or `timeout_ms` expires.
    /// Unexpected completions (wr_ids not in the expected set) are stashed in a
    /// backlog and returned via `drain_backlog()`.
    pub fn wait_completions_with_timeout(
        &self,
        expected_wr_ids: &[u64],
        timeout_ms: u64,
    ) -> Result<(), RdmaError> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let mut remaining: Vec<u64> = expected_wr_ids.to_vec();
        let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| unsafe { std::mem::zeroed() });

        // Check backlog first for any previously stashed completions
        {
            let mut backlog = self.completion_backlog.borrow_mut();
            let mut backlog_error: Option<RdmaError> = None;
            remaining.retain(|&wr_id| {
                if let Some(pos) = backlog.iter().position(|wc| wc.wr_id == wr_id) {
                    let wc = backlog.swap_remove(pos);
                    if wc.status != wc_status::SUCCESS {
                        backlog_error = Some(RdmaError::CqPoll(format!(
                            "backlog: wr_id {} failed with status {}",
                            wr_id, wc.status
                        )));
                    }
                    false // found, remove from remaining
                } else {
                    true // keep in remaining
                }
            });
            if let Some(err) = backlog_error {
                return Err(err);
            }
        }

        while !remaining.is_empty() {
            let count = self.cq.poll(&mut wc_buf)?;
            for wc in &wc_buf[..count] {
                if let Some(pos) = remaining.iter().position(|&id| id == wc.wr_id) {
                    if wc.status != wc_status::SUCCESS {
                        return Err(RdmaError::CqPoll(format!(
                            "wc status={} for wr_id={}",
                            wc.status, wc.wr_id,
                        )));
                    }
                    remaining.swap_remove(pos);
                } else {
                    // Unexpected wr_id — stash in backlog
                    self.completion_backlog.borrow_mut().push(*wc);
                }
            }
            if !remaining.is_empty() && Instant::now() >= deadline {
                return Err(RdmaError::Timeout(format!(
                    "wr_ids {:?} not completed within {timeout_ms}ms",
                    remaining,
                )));
            }
            if !remaining.is_empty() {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    /// Drain any stashed backlog completions (wr_ids not matched by prior waits).
    pub fn drain_backlog(&self) -> Vec<IbvWc> {
        self.completion_backlog.borrow_mut().drain(..).collect()
    }

    /// Create a new CompletionTracker for fine-grained wr_id-based matching.
    pub fn new_tracker(&self) -> CompletionTracker {
        CompletionTracker::new()
    }

    /// Access the completion queue (for use with CompletionTracker).
    pub fn cq(&self) -> &CompletionQueue {
        &self.cq
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
                &self.pd,
                warmup_buf.as_mut_ptr() as *mut c_void,
                WARMUP_SIZE,
            )?
        };

        for round in 0..WARMUP_ROUNDS {
            let recv_wr_id = round as u64;
            let send_wr_id = (round + 100) as u64;

            if self.config.rank == 0 {
                // Rank 0: recv then send
                let recv_op = self.post_recv(&mr, 0, WARMUP_SIZE as u32, recv_wr_id)?;
                // Barrier: signal rank 1 that recv is posted
                exchange::tcp_barrier_server(
                    self.config.sync_port,
                    self.config.accept_timeout_secs,
                    10,
                )?;
                self.wait_posted(&[recv_op])?;

                let send_op = self.post_send(&mr, 0, WARMUP_SIZE as u32, send_wr_id)?;
                self.wait_posted(&[send_op])?;
            } else {
                // Rank 1: send then recv
                let recv_op = self.post_recv(&mr, 0, WARMUP_SIZE as u32, recv_wr_id)?;
                // Barrier: wait for rank 0's recv to be posted
                exchange::tcp_barrier_client(
                    &self.config.peer_host,
                    self.config.sync_port,
                    10,
                    30,
                    100,
                )?;

                let send_op = self.post_send(&mr, 0, WARMUP_SIZE as u32, send_wr_id)?;
                self.wait_posted(&[send_op])?;
                self.wait_posted(&[recv_op])?;
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

    /// Access the protection domain.
    pub fn pd(&self) -> &ProtectionDomain {
        &self.pd
    }

    /// Initialize the pre-registered MR pool.
    pub fn init_mr_pool(&mut self) -> Result<(), RdmaError> {
        let pool = MrPool::new(&self.pd)?;
        self.mr_pool = Some(pool);
        Ok(())
    }

    /// Acquire a pre-registered MR from the pool.
    ///
    /// Returns `None` if the pool is not initialized or all slots of the
    /// appropriate tier are in use.
    pub fn acquire_mr(&self, size: usize) -> Option<MrHandle> {
        self.mr_pool.as_ref()?.acquire(size)
    }

    /// Post a send using a SharedBuffer's pre-registered MR.
    ///
    /// Looks up the RDMA memory region registered for `conn_id` on the
    /// given SharedBuffer and delegates to `post_send`.
    pub fn post_send_shared<'a>(
        &self,
        buf: &'a SharedBuffer,
        conn_id: &ConnectionId,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<PostedOp<'a>, RdmaError> {
        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            RdmaError::InvalidArgument(format!(
                "no MR registered for conn_id {:?} on SharedBuffer",
                conn_id
            ))
        })?;
        self.post_send(mr, offset, length, wr_id)
    }

    /// Post a recv using a SharedBuffer's pre-registered MR.
    ///
    /// Looks up the RDMA memory region registered for `conn_id` on the
    /// given SharedBuffer and delegates to `post_recv`.
    pub fn post_recv_shared<'a>(
        &self,
        buf: &'a SharedBuffer,
        conn_id: &ConnectionId,
        offset: usize,
        length: u32,
        wr_id: u64,
    ) -> Result<PostedOp<'a>, RdmaError> {
        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            RdmaError::InvalidArgument(format!(
                "no MR registered for conn_id {:?} on SharedBuffer",
                conn_id
            ))
        })?;
        self.post_recv(mr, offset, length, wr_id)
    }
}

// SAFETY: All inner types (RdmaContext, ProtectionDomain, CompletionQueue, QueuePair)
// already implement Send. The RefCell<Vec<IbvWc>> is Send when IbvWc is Send (it is Copy).
unsafe impl Send for RdmaConnection {}
