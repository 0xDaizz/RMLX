//! RDMA connection manager for multi-node TB5 clusters.
//!
//! Manages QP creation, TCP-based QP info exchange, state transitions,
//! and warmup protocol for all peers.

use bytemuck::Zeroable as _;
use parking_lot::Mutex;
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

/// JACCL-compatible buffer pool constants.
const FRAME_SIZE: usize = 4096; // 4KB base unit
const BUFFER_SIZES: usize = 8; // 8 tiers: 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB
const NUM_BUFFERS: usize = 2; // Ping-pong buffers per tier (PIPELINE depth)

/// Select buffer tier for a message size. Returns (tier_index, chunk_size).
fn buffer_size_from_message(msg_bytes: usize) -> (usize, usize) {
    if msg_bytes == 0 {
        return (0, FRAME_SIZE);
    }
    for k in (1..BUFFER_SIZES).rev() {
        let size = FRAME_SIZE * (1 << k);
        if msg_bytes >= size {
            return (k, size);
        }
    }
    (0, FRAME_SIZE)
}

/// Pre-registered RDMA buffer (JACCL SharedBuffer equivalent).
///
/// Allocated once via `posix_memalign`, registered to a PD with `ibv_reg_mr`,
/// and reused for all send/recv operations on a connection. This eliminates
/// the per-transfer `ibv_reg_mr` / `ibv_dereg_mr` overhead.
pub struct SharedRdmaBuffer {
    buf: *mut c_void,
    size: usize,
    mr: *mut IbvMr,
    lib: &'static IbverbsLib,
}

impl SharedRdmaBuffer {
    /// Allocate a page-aligned buffer and register it as an RDMA memory region.
    pub fn new(pd: &ProtectionDomain, size: usize) -> Result<Self, RdmaError> {
        let lib = pd.lib();
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = if page_size > 0 {
            page_size as usize
        } else {
            16384
        };
        // posix_memalign requires size >= alignment, so allocate at least
        // page_size bytes.  However, ibv_reg_mr should register only the
        // actually requested size — macOS TB5 limits 16 KB MRs to 1 per PD,
        // so registering e.g. 4 KB instead of inflating to 16 KB avoids
        // hitting that limit.
        let alloc_size = size.max(page_size); // allocation must be >= page_size for memalign
        let reg_size = size; // register only what was requested

        let mut buf: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { libc::posix_memalign(&mut buf, page_size, alloc_size) };
        if ret != 0 || buf.is_null() {
            return Err(RdmaError::MrReg(format!(
                "SharedRdmaBuffer posix_memalign failed: ret={ret}, alloc_size={alloc_size}"
            )));
        }
        // Zero-initialize
        unsafe { std::ptr::write_bytes(buf as *mut u8, 0, alloc_size) };

        let flags =
            access_flags::LOCAL_WRITE | access_flags::REMOTE_WRITE | access_flags::REMOTE_READ;
        let mr = unsafe { (lib.reg_mr)(pd.raw(), buf, reg_size, flags) };
        if mr.is_null() {
            unsafe { libc::free(buf) };
            return Err(RdmaError::MrReg(format!(
                "SharedRdmaBuffer ibv_reg_mr failed: buf={buf:?}, reg_size={reg_size}"
            )));
        }

        Ok(Self {
            buf,
            size: reg_size,
            mr,
            lib,
        })
    }

    /// Start address of the registered buffer (for SGE).
    pub fn addr(&self) -> u64 {
        self.buf as u64
    }

    /// Local key of the registered MR.
    pub fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy `data` into the buffer (for send path).
    ///
    /// # Panics
    /// Panics if `data.len() > self.size`.
    pub fn copy_from(&self, data: &[u8]) {
        assert!(
            data.len() <= self.size,
            "SharedRdmaBuffer overflow: {} > {}",
            data.len(),
            self.size
        );
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.buf as *mut u8, data.len());
        }
    }

    /// Read `len` bytes from the buffer into a new Vec (for recv path).
    pub fn read(&self, len: usize) -> Vec<u8> {
        let len = len.min(self.size);
        let mut out = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.buf as *const u8, out.as_mut_ptr(), len);
        }
        out
    }
}

impl Drop for SharedRdmaBuffer {
    fn drop(&mut self) {
        unsafe {
            let ret = (self.lib.dereg_mr)(self.mr);
            if ret != 0 {
                eprintln!(
                    "[SharedRdmaBuffer] WARNING: ibv_dereg_mr returned {ret}, leaking buffer"
                );
                return;
            }
            libc::free(self.buf);
        }
    }
}

// SAFETY: SharedRdmaBuffer owns its allocation and MR handle exclusively.
unsafe impl Send for SharedRdmaBuffer {}

/// Pre-registered buffer pool for pipelined RDMA transfers.
/// Organized as [tier][buffer_index] for send and recv separately.
/// tier 0 = FRAME_SIZE (4KB), tier 7 = 512KB.
pub struct RdmaBufferPool {
    send: Vec<Vec<SharedRdmaBuffer>>, // [BUFFER_SIZES][NUM_BUFFERS]
    recv: Vec<Vec<SharedRdmaBuffer>>, // [BUFFER_SIZES][NUM_BUFFERS]
}

impl RdmaBufferPool {
    /// Allocate all tiered send/recv buffers and register them as MRs.
    pub fn new(pd: &ProtectionDomain) -> Result<Self, RdmaError> {
        let mut send = Vec::with_capacity(BUFFER_SIZES);
        let mut recv = Vec::with_capacity(BUFFER_SIZES);
        for k in 0..BUFFER_SIZES {
            let size = FRAME_SIZE * (1 << k);
            let mut send_tier = Vec::with_capacity(NUM_BUFFERS);
            let mut recv_tier = Vec::with_capacity(NUM_BUFFERS);
            for _ in 0..NUM_BUFFERS {
                send_tier.push(SharedRdmaBuffer::new(pd, size)?);
                recv_tier.push(SharedRdmaBuffer::new(pd, size)?);
            }
            send.push(send_tier);
            recv.push(recv_tier);
        }
        Ok(Self { send, recv })
    }
}

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
            let is_recv = wc.opcode >= 128;
            if !is_recv && wc.status != wc_status::SUCCESS {
                return Err(RdmaError::CqPoll(format!(
                    "wc status={} ({}) for wr_id={wr_id}",
                    wc.status,
                    wc_status_str(wc.status)
                )));
            }
            self.completed.push(wr_id);
            return Ok(wc);
        }

        // Poll CQ with timeout
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| IbvWc::zeroed());

        loop {
            let count = cq.poll(&mut wc_buf)?;
            for wc in &wc_buf[..count] {
                if wc.wr_id == wr_id {
                    let is_recv = wc.opcode >= 128;
                    if !is_recv && wc.status != wc_status::SUCCESS {
                        return Err(RdmaError::CqPoll(format!(
                            "wc status={} ({}) for wr_id={wr_id}",
                            wc.status,
                            wc_status_str(wc.status)
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
    let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| IbvWc::zeroed());

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
    original_ptr: *mut u8,
    original_size: usize,
    _lt: PhantomData<&'a mut [u8]>,
}

impl<'a> RegisteredRecv<'a> {
    /// Access the underlying `MemoryRegion`.
    pub fn mr(&self) -> &MemoryRegion {
        &self.mr
    }

    /// Copy received data from the aligned MR buffer back to the original recv buffer.
    /// Must be called after recv completion to retrieve data written by the remote peer.
    pub fn copy_back(&self) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mr.addr() as *const u8,
                self.original_ptr,
                self.original_size,
            );
        }
    }
}

/// An established RDMA connection to a peer.
///
/// Wraps the full RDMA stack: context, protection domain, completion queue,
/// and queue pair. Individual components handle their own cleanup via Drop.
///
/// **Field order matters**: Rust drops fields in declaration order, so we
/// declare dependent resources first (dropped early) and foundational
/// resources last (dropped late):
///
/// Drop order (top to bottom):
/// 1. completion_backlog, config — plain data (no RDMA teardown)
/// 2. pool — tiered buffer MRs deregister (must precede QP destroy)
/// 3. mr_pool — MR deregister
/// 4. qp — QP destroy
/// 5. cq — CQ destroy
/// 6. pd — PD dealloc
/// 7. ctx — context close (must be last)
pub struct RdmaConnection {
    config: RdmaConfig,
    /// Backlog of CQ completions with unexpected wr_ids, for later retrieval.
    /// Protected by a Mutex for thread-safe access (IbvWc is Copy + Send).
    completion_backlog: Mutex<Vec<IbvWc>>,
    /// Pre-registered tiered buffer pool (JACCL chunked transfer pattern).
    /// Allocated once at connection creation, reused for all sends/recvs.
    pool: Option<RdmaBufferPool>,
    /// Pre-registered MR pool (lazily initialized via `init_mr_pool`).
    mr_pool: Option<MrPool>,
    qp: QueuePair,
    cq: CompletionQueue,
    pd: ProtectionDomain,
    ctx: RdmaContext,
}

impl RdmaConnection {
    /// Construct an `RdmaConnection` from pre-built components.
    ///
    /// This bypasses the integrated TCP exchange used by [`establish`] and is
    /// intended for initialization flows that perform QP info exchange through
    /// a separate mechanism (e.g., coordinator-mediated `all_gather`).
    ///
    /// The caller is responsible for:
    /// 1. Opening the RDMA device and allocating PD / CQ / QP
    /// 2. Querying local QP info (`qp.query_local_info`)
    /// 3. Exchanging QP info with the remote peer (via coordinator, etc.)
    /// 4. Connecting the QP (`qp.connect(&remote_info)`)
    ///
    /// If the QP is intentionally left unconnected (e.g., a placeholder for the
    /// self-rank slot), the caller may skip steps 3-4.
    pub fn from_parts(
        ctx: RdmaContext,
        pd: ProtectionDomain,
        cq: CompletionQueue,
        qp: QueuePair,
        config: RdmaConfig,
        pool: Option<RdmaBufferPool>,
    ) -> Self {
        Self {
            config,
            completion_backlog: Mutex::new(Vec::new()),
            pool,
            mr_pool: None,
            qp,
            cq,
            pd,
            ctx,
        }
    }

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

        // Pre-register tiered buffer pool (best-effort; None on failure)
        let pool = RdmaBufferPool::new(&pd)
            .inspect_err(|e| eprintln!("[rdma] buffer pool alloc failed: {e}"))
            .ok();

        Ok(Self {
            config,
            completion_backlog: Mutex::new(Vec::new()),
            pool,
            mr_pool: None,
            qp,
            cq,
            pd,
            ctx,
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
            tracing::warn!(
                target: "rmlx_rdma",
                default = crate::mr::DEFAULT_MAX_MR_SIZE,
                "probe unavailable, using DEFAULT_MAX_MR_SIZE",
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
        let original_ptr = data.as_mut_ptr();
        let original_size = data.len();
        // SAFETY: data is a valid &mut [u8]; lifetime 'a outlives RegisteredRecv<'a>.
        let mr = unsafe { self.register_mr(data.as_mut_ptr() as *mut c_void, data.len())? };
        Ok(RegisteredRecv {
            mr,
            original_ptr,
            original_size,
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
        // All fields read by ibv_post_send (wr_id, next, sg_list, num_sge, opcode,
        // send_flags) are set immediately below; remaining fields are don't-care for SEND.
        let mut wr: IbvSendWr = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        wr.wr_id = wr_id;
        wr.next = std::ptr::null_mut();
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
        let mut wc_buf: [IbvWc; 16] = core::array::from_fn(|_| IbvWc::zeroed());

        // Check backlog first for any previously stashed completions
        {
            let mut backlog = self.completion_backlog.lock();
            let mut backlog_error: Option<RdmaError> = None;
            remaining.retain(|&wr_id| {
                if let Some(pos) = backlog.iter().position(|wc| wc.wr_id == wr_id) {
                    let wc = backlog.swap_remove(pos);
                    // UC recv completions (opcode >= 128) always report status=4
                    // (LOC_PROT_ERR) on macOS TB5, but data arrives correctly.
                    // Match JACCL behavior: ignore status for recv completions.
                    let is_recv = wc.opcode >= 128;
                    if !is_recv && wc.status != wc_status::SUCCESS {
                        backlog_error = Some(RdmaError::CqPoll(format!(
                            "backlog: wr_id {} failed with status {} ({})",
                            wr_id,
                            wc.status,
                            wc_status_str(wc.status)
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
                    // UC recv completions (opcode >= 128) always report status=4
                    // (LOC_PROT_ERR) on macOS TB5, but data arrives correctly.
                    // Match JACCL behavior: ignore status for recv completions.
                    let is_recv = wc.opcode >= 128;
                    if !is_recv && wc.status != wc_status::SUCCESS {
                        return Err(RdmaError::CqPoll(format!(
                            "wc status={} ({}) for wr_id={}",
                            wc.status,
                            wc_status_str(wc.status),
                            wc.wr_id,
                        )));
                    }
                    remaining.swap_remove(pos);
                } else {
                    // Unexpected wr_id — stash in backlog
                    self.completion_backlog.lock().push(*wc);
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
        self.completion_backlog.lock().drain(..).collect()
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
                let _recv_op = self.post_recv(&mr, 0, WARMUP_SIZE as u32, recv_wr_id)?;
                // Barrier: signal rank 1 that recv is posted
                exchange::tcp_barrier_server(
                    self.config.sync_port,
                    self.config.accept_timeout_secs,
                    10,
                )?;
                // macOS TB5 RDMA driver bug: ibv_poll_cq corrupts RQ state,
                // causing subsequent ibv_post_recv to block forever.
                // Skip completion polling entirely — data arrives regardless.
                std::thread::sleep(Duration::from_millis(1));

                let _send_op = self.post_send(&mr, 0, WARMUP_SIZE as u32, send_wr_id)?;
                std::thread::sleep(Duration::from_millis(1));
            } else {
                // Rank 1: send then recv
                let _recv_op = self.post_recv(&mr, 0, WARMUP_SIZE as u32, recv_wr_id)?;
                // Barrier: wait for rank 0's recv to be posted
                exchange::tcp_barrier_client(
                    &self.config.peer_host,
                    self.config.sync_port,
                    10,
                    30,
                    100,
                )?;

                let _send_op = self.post_send(&mr, 0, WARMUP_SIZE as u32, send_wr_id)?;
                // macOS TB5 RDMA driver bug: skip CQ polling.
                std::thread::sleep(Duration::from_millis(1));
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

    // ─── Chunked RDMA transfer (JACCL pipelined buffer pool pattern) ───

    /// Whether the tiered buffer pool is available.
    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    /// Attempt zero-copy send by registering the caller's buffer directly.
    ///
    /// Only works on UMA (Apple Silicon) where the pointer is already in
    /// device-visible memory.  Falls back (returns `Err`) if MR registration
    /// fails (e.g., TB5 driver limitation).
    fn try_nocopy_send(&self, data: &[u8]) -> Result<(), RdmaError> {
        // SAFETY: data.as_ptr() is page-aligned and valid for data.len() bytes
        // (checked by caller). The slice stays alive until this function returns
        // and the MR is dropped.
        let mr = unsafe {
            MemoryRegion::register_nocopy(
                &self.pd,
                data.as_ptr() as *mut std::ffi::c_void,
                data.len(),
            )?
        };

        let mut sge = IbvSge {
            addr: mr.addr() as u64,
            length: data.len() as u32,
            lkey: mr.lkey(),
        };

        let mut wr: IbvSendWr = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        wr.wr_id = 0x4E00; // nocopy marker
        wr.next = std::ptr::null_mut();
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = wr_opcode::SEND;
        wr.send_flags = send_flags::SIGNALED;

        self.qp.post_send(&mut wr)?;

        // Poll CQ for completion (same 5-second timeout pattern as chunked_send)
        let mut wc_buf: [IbvWc; 1] = [IbvWc::zeroed()];
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let count = self.cq.poll(&mut wc_buf)?;
            if count > 0 {
                if wc_buf[0].status != wc_status::SUCCESS {
                    return Err(RdmaError::CqPoll(format!(
                        "try_nocopy_send: wc status={} ({})",
                        wc_buf[0].status,
                        wc_status_str(wc_buf[0].status)
                    )));
                }
                break;
            }
            if Instant::now() >= deadline {
                // HW may still be accessing the buffer — leak the MR to prevent
                // use-after-deregister. The MR will never be freed, but this is
                // better than a potential crash.
                std::mem::forget(mr);
                return Err(RdmaError::Timeout(
                    "try_nocopy_send: timeout waiting for completion (MR leaked for safety)".into(),
                ));
            }
            std::thread::yield_now();
        }

        // MR dropped here — deregisters the nocopy region
        Ok(())
    }

    /// Chunked send using PIPELINE=2 pipelining through tiered buffer pool.
    /// Handles arbitrary message sizes by streaming through pre-registered buffers.
    pub fn chunked_send(&self, data: &[u8]) -> Result<(), RdmaError> {
        // Fast path: try nocopy for page-aligned, large-enough buffers (UMA optimization)
        const PAGE_SIZE: usize = 16384; // macOS page size
        if data.len() >= PAGE_SIZE
            && (data.as_ptr() as usize) % PAGE_SIZE == 0
            && data.len() <= crate::mr::DEFAULT_MAX_MR_SIZE
        {
            match self.try_nocopy_send(data) {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Nocopy registration failed (e.g., TB5 driver limitation) — fall through
                }
            }
        }

        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| RdmaError::InvalidArgument("buffer pool not available".into()))?;

        if data.is_empty() {
            return Ok(());
        }

        let (tier, chunk_size) = buffer_size_from_message(data.len());
        let send_total = data.len();
        let mut send_offset: usize = 0;
        let mut in_flight: u32 = 0;
        let mut wc_buf: [IbvWc; 4] = core::array::from_fn(|_| IbvWc::zeroed());

        const SEND_WR: u64 = 1;

        // Post initial sends (up to NUM_BUFFERS)
        let mut buf_idx = 0;
        while send_offset < send_total && buf_idx < NUM_BUFFERS {
            let chunk = (send_total - send_offset).min(chunk_size);
            let sbuf = &pool.send[tier][buf_idx];
            sbuf.copy_from(&data[send_offset..send_offset + chunk]);

            let mut sge = IbvSge {
                addr: sbuf.addr(),
                length: sbuf.size() as u32,
                lkey: sbuf.lkey(),
            };
            let wr_id = (SEND_WR << 16) | (buf_idx as u64) << 8;
            let mut wr: IbvSendWr = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
            wr.wr_id = wr_id;
            wr.next = std::ptr::null_mut();
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = wr_opcode::SEND;
            wr.send_flags = send_flags::SIGNALED;

            self.qp.post_send(&mut wr)?;
            in_flight += 1;
            send_offset += chunk_size;
            buf_idx += 1;
        }

        // Poll loop with 5-second rolling timeout (JACCL pattern)
        let mut deadline = Instant::now() + Duration::from_secs(5);
        while in_flight > 0 {
            let count = self.cq.poll(&mut wc_buf)?;
            if count > 0 {
                deadline = Instant::now() + Duration::from_secs(5);
            }

            for wc in &wc_buf[..count] {
                let buf_idx = ((wc.wr_id >> 8) & 0xff) as usize;

                if wc.status != wc_status::SUCCESS {
                    return Err(RdmaError::CqPoll(format!(
                        "chunked_send: send wc status={} ({})",
                        wc.status,
                        wc_status_str(wc.status)
                    )));
                }

                in_flight -= 1;

                // Send completed — post next chunk if available
                if send_offset < send_total {
                    let chunk = (send_total - send_offset).min(chunk_size);
                    let sbuf = &pool.send[tier][buf_idx];
                    sbuf.copy_from(&data[send_offset..send_offset + chunk]);

                    let mut sge = IbvSge {
                        addr: sbuf.addr(),
                        length: sbuf.size() as u32,
                        lkey: sbuf.lkey(),
                    };
                    let wr_id = (SEND_WR << 16) | (buf_idx as u64) << 8;
                    let mut wr: IbvSendWr =
                        unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
                    wr.wr_id = wr_id;
                    wr.next = std::ptr::null_mut();
                    wr.sg_list = &mut sge;
                    wr.num_sge = 1;
                    wr.opcode = wr_opcode::SEND;
                    wr.send_flags = send_flags::SIGNALED;

                    self.qp.post_send(&mut wr)?;
                    in_flight += 1;
                    send_offset += chunk_size;
                }
            }

            if in_flight > 0 && Instant::now() >= deadline {
                return Err(RdmaError::Timeout(format!(
                    "chunked_send: timeout (in_flight={in_flight})"
                )));
            }
            if in_flight > 0 && count == 0 {
                std::thread::yield_now();
            }
        }

        Ok(())
    }

    /// Chunked recv using PIPELINE=2 pipelining through tiered buffer pool.
    /// Handles arbitrary message sizes by streaming through pre-registered buffers.
    pub fn chunked_recv(&self, len: usize) -> Result<Vec<u8>, RdmaError> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| RdmaError::InvalidArgument("buffer pool not available".into()))?;

        if len == 0 {
            return Ok(Vec::new());
        }

        let (tier, chunk_size) = buffer_size_from_message(len);
        let recv_total = len;
        let mut recv_offset: usize = 0;
        let mut in_flight: u32 = 0;
        let mut wc_buf: [IbvWc; 4] = core::array::from_fn(|_| IbvWc::zeroed());
        let mut recv_result = vec![0u8; recv_total];

        const RECV_WR: u64 = 2;

        // Post initial recvs (up to NUM_BUFFERS)
        let mut buf_idx = 0;
        while (chunk_size as u64) * (buf_idx as u64) < recv_total as u64 && buf_idx < NUM_BUFFERS {
            let rbuf = &pool.recv[tier][buf_idx];
            let mut sge = IbvSge {
                addr: rbuf.addr(),
                length: rbuf.size() as u32,
                lkey: rbuf.lkey(),
            };
            let wr_id = (RECV_WR << 16) | (buf_idx as u64) << 8;
            let mut wr = IbvRecvWr {
                wr_id,
                next: std::ptr::null_mut(),
                sg_list: &mut sge,
                num_sge: 1,
            };
            self.qp.post_recv(&mut wr)?;
            in_flight += 1;
            buf_idx += 1;
        }

        // Poll loop with 5-second rolling timeout (JACCL pattern)
        let mut deadline = Instant::now() + Duration::from_secs(5);
        while in_flight > 0 {
            let count = self.cq.poll(&mut wc_buf)?;
            if count > 0 {
                deadline = Instant::now() + Duration::from_secs(5);
            }

            for wc in &wc_buf[..count] {
                let buf_idx = ((wc.wr_id >> 8) & 0xff) as usize;

                // UC recv may report non-zero status but data arrives correctly
                in_flight -= 1;

                // Recv completed — copy data out
                let chunk = (recv_total - recv_offset).min(chunk_size);
                if chunk > 0 {
                    let rbuf = &pool.recv[tier][buf_idx];
                    let src = rbuf.read(chunk);
                    recv_result[recv_offset..recv_offset + chunk].copy_from_slice(&src);
                    recv_offset += chunk_size;
                }

                // Post next recv if more data expected
                let remaining_recvs = if recv_total > recv_offset {
                    (recv_total - recv_offset).div_ceil(chunk_size)
                } else {
                    0
                };
                if remaining_recvs > 0 {
                    let rbuf = &pool.recv[tier][buf_idx];
                    let mut sge = IbvSge {
                        addr: rbuf.addr(),
                        length: rbuf.size() as u32,
                        lkey: rbuf.lkey(),
                    };
                    let wr_id = (RECV_WR << 16) | (buf_idx as u64) << 8;
                    let mut wr = IbvRecvWr {
                        wr_id,
                        next: std::ptr::null_mut(),
                        sg_list: &mut sge,
                        num_sge: 1,
                    };
                    self.qp.post_recv(&mut wr)?;
                    in_flight += 1;
                }
            }

            if in_flight > 0 && Instant::now() >= deadline {
                return Err(RdmaError::Timeout(format!(
                    "chunked_recv: timeout (in_flight={in_flight})"
                )));
            }
            if in_flight > 0 && count == 0 {
                std::thread::yield_now();
            }
        }

        Ok(recv_result)
    }

    /// Chunked bidirectional sendrecv using PIPELINE=2 pipelining.
    /// Handles arbitrary message sizes by streaming through tiered buffer pool.
    pub fn chunked_sendrecv(
        &self,
        send_data: &[u8],
        recv_len: usize,
    ) -> Result<Vec<u8>, RdmaError> {
        let pool = self
            .pool
            .as_ref()
            .ok_or_else(|| RdmaError::InvalidArgument("buffer pool not available".into()))?;

        let max_bytes = send_data.len().max(recv_len).max(1);
        let (tier, chunk_size) = buffer_size_from_message(max_bytes);

        let mut recv_result = vec![0u8; recv_len];
        let mut send_offset: usize = 0;
        let mut recv_offset: usize = 0;
        let send_total = send_data.len();
        let recv_total = recv_len;

        let mut in_flight: u32 = 0;
        let mut wc_buf: [IbvWc; 4] = core::array::from_fn(|_| IbvWc::zeroed());

        // Encode wr_id: (type << 16) | (buf_idx << 8) — JACCL pattern
        const SEND_WR: u64 = 1;
        const RECV_WR: u64 = 2;

        // Prefill: post recvs first (UC mode requires recv before send)
        if recv_total > 0 {
            let mut buf_idx = 0;
            while (chunk_size as u64) * (buf_idx as u64) < recv_total as u64
                && buf_idx < NUM_BUFFERS
            {
                let rbuf = &pool.recv[tier][buf_idx];
                let mut sge = IbvSge {
                    addr: rbuf.addr(),
                    length: rbuf.size() as u32,
                    lkey: rbuf.lkey(),
                };
                let wr_id = (RECV_WR << 16) | (buf_idx as u64) << 8;
                let mut wr = IbvRecvWr {
                    wr_id,
                    next: std::ptr::null_mut(),
                    sg_list: &mut sge,
                    num_sge: 1,
                };
                self.qp.post_recv(&mut wr)?;
                in_flight += 1;
                buf_idx += 1;
            }
        }

        // Post sends
        if send_total > 0 {
            let mut buf_idx = 0;
            while send_offset < send_total && buf_idx < NUM_BUFFERS {
                let chunk = (send_total - send_offset).min(chunk_size);
                let sbuf = &pool.send[tier][buf_idx];
                sbuf.copy_from(&send_data[send_offset..send_offset + chunk]);

                let mut sge = IbvSge {
                    addr: sbuf.addr(),
                    length: sbuf.size() as u32,
                    lkey: sbuf.lkey(),
                };
                let wr_id = (SEND_WR << 16) | (buf_idx as u64) << 8;
                let mut wr: IbvSendWr = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
                wr.wr_id = wr_id;
                wr.next = std::ptr::null_mut();
                wr.sg_list = &mut sge;
                wr.num_sge = 1;
                wr.opcode = wr_opcode::SEND;
                wr.send_flags = send_flags::SIGNALED;

                self.qp.post_send(&mut wr)?;
                in_flight += 1;
                send_offset += chunk_size;
                buf_idx += 1;
            }
        }

        // Poll loop with 5-second rolling timeout (JACCL pattern)
        let mut deadline = Instant::now() + Duration::from_secs(5);
        while in_flight > 0 {
            let count = self.cq.poll(&mut wc_buf)?;
            if count > 0 {
                deadline = Instant::now() + Duration::from_secs(5);
            }

            for wc in &wc_buf[..count] {
                let work_type = wc.wr_id >> 16;
                let buf_idx = ((wc.wr_id >> 8) & 0xff) as usize;

                // UC recv may report non-zero status but data arrives correctly
                let is_recv = work_type == RECV_WR;
                if !is_recv && wc.status != wc_status::SUCCESS {
                    return Err(RdmaError::CqPoll(format!(
                        "chunked_sendrecv: send wc status={} ({})",
                        wc.status,
                        wc_status_str(wc.status)
                    )));
                }

                in_flight -= 1;

                if work_type == SEND_WR {
                    // Send completed — post next chunk if available
                    if send_offset < send_total {
                        let chunk = (send_total - send_offset).min(chunk_size);
                        let sbuf = &pool.send[tier][buf_idx];
                        sbuf.copy_from(&send_data[send_offset..send_offset + chunk]);

                        let mut sge = IbvSge {
                            addr: sbuf.addr(),
                            length: sbuf.size() as u32,
                            lkey: sbuf.lkey(),
                        };
                        let wr_id = (SEND_WR << 16) | (buf_idx as u64) << 8;
                        let mut wr: IbvSendWr =
                            unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
                        wr.wr_id = wr_id;
                        wr.next = std::ptr::null_mut();
                        wr.sg_list = &mut sge;
                        wr.num_sge = 1;
                        wr.opcode = wr_opcode::SEND;
                        wr.send_flags = send_flags::SIGNALED;

                        self.qp.post_send(&mut wr)?;
                        in_flight += 1;
                        send_offset += chunk_size;
                    }
                } else if work_type == RECV_WR {
                    // Recv completed — copy data out and post next recv if needed
                    let chunk = (recv_total - recv_offset).min(chunk_size);
                    if chunk > 0 {
                        let rbuf = &pool.recv[tier][buf_idx];
                        let src = rbuf.read(chunk);
                        recv_result[recv_offset..recv_offset + chunk].copy_from_slice(&src);
                        recv_offset += chunk_size;
                    }
                    // Post next recv if more data expected
                    let remaining_recvs = if recv_total > recv_offset {
                        (recv_total - recv_offset).div_ceil(chunk_size)
                    } else {
                        0
                    };
                    if remaining_recvs > 0 {
                        let rbuf = &pool.recv[tier][buf_idx];
                        let mut sge = IbvSge {
                            addr: rbuf.addr(),
                            length: rbuf.size() as u32,
                            lkey: rbuf.lkey(),
                        };
                        let wr_id = (RECV_WR << 16) | (buf_idx as u64) << 8;
                        let mut wr = IbvRecvWr {
                            wr_id,
                            next: std::ptr::null_mut(),
                            sg_list: &mut sge,
                            num_sge: 1,
                        };
                        self.qp.post_recv(&mut wr)?;
                        in_flight += 1;
                    }
                }
            }

            if in_flight > 0 && Instant::now() >= deadline {
                return Err(RdmaError::Timeout(format!(
                    "chunked_sendrecv: timeout (in_flight={in_flight})"
                )));
            }
            if in_flight > 0 && count == 0 {
                std::thread::yield_now();
            }
        }

        Ok(recv_result)
    }
}

// SAFETY: All inner types (RdmaContext, ProtectionDomain, CompletionQueue, QueuePair)
// already implement Send. The Mutex<Vec<IbvWc>> is both Send and Sync when IbvWc is Send.
unsafe impl Send for RdmaConnection {}
