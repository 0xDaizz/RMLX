//! Concrete RDMA transport implementation backed by `rmlx_rdma::RdmaConnection`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use rmlx_rdma::connection::RdmaConnection;
use rmlx_rdma::exchange_tag::{encode_wr_id, try_decode_wr_id, ExchangeTag};
use rmlx_rdma::gpu_doorbell::{RdmaDescriptor, RdmaOp};
use rmlx_rdma::multi_port::StripeEngine;
use rmlx_rdma::progress::{OwnedPendingOp, PendingOp, ProgressEngine};
use rmlx_rdma::rdma_metrics::RdmaMetrics;
use rmlx_rdma::shared_buffer::{ConnectionId, SharedBuffer};
use rmlx_rdma::RdmaError;

use crate::group::{DistributedError, RdmaTransport};
use crate::perf_counters::global_counters;

// ─── ZeroCopyPendingOp: ties SharedBuffer lifetime to RDMA op lifetime ───

/// A pending RDMA operation that holds an `Arc<SharedBuffer>` to prevent
/// the buffer from being dropped while a DMA operation is still in flight.
///
/// This is the zero-copy counterpart of `OwnedPendingOp`: instead of owning
/// a freshly-registered `MemoryRegion`, it holds a shared reference to the
/// `SharedBuffer` whose pre-registered MR is being used by the RDMA HW.
///
/// On `Drop`, if the operation is still pending, we block **indefinitely**
/// until the RDMA NIC has finished accessing the buffer memory before the
/// `Arc` reference is released. This prevents use-after-free when the last
/// `Arc` holder drops the `SharedBuffer` while DMA is still in progress.
/// A warning is emitted if the wait exceeds 5 seconds.
pub struct ZeroCopyPendingOp {
    pending: PendingOp,
    /// Prevent the SharedBuffer from being freed while DMA is in flight.
    _buf: Arc<SharedBuffer>,
}

impl ZeroCopyPendingOp {
    /// Create a new `ZeroCopyPendingOp` that ties buffer lifetime to the op.
    pub fn new(pending: PendingOp, buf: Arc<SharedBuffer>) -> Self {
        Self { pending, _buf: buf }
    }

    /// Non-blocking poll. Returns `Some` if the operation has completed.
    pub fn try_poll(
        &self,
    ) -> Option<Result<rmlx_rdma::progress::Completion, rmlx_rdma::progress::OpError>> {
        self.pending.try_poll()
    }

    /// Returns true if the operation is still pending.
    pub fn is_pending(&self) -> bool {
        self.pending.is_pending()
    }

    /// Blocking wait with timeout. The buffer stays alive regardless of outcome.
    pub fn wait(
        &self,
        timeout: std::time::Duration,
    ) -> Result<rmlx_rdma::progress::Completion, rmlx_rdma::progress::WaitError> {
        self.pending.wait(timeout)
    }

    /// The wr_id this operation is tracking.
    pub fn wr_id(&self) -> u64 {
        self.pending.wr_id()
    }

    /// Access the inner `PendingOp` (e.g. for legacy code that needs it).
    pub fn inner(&self) -> &PendingOp {
        &self.pending
    }
}

impl Drop for ZeroCopyPendingOp {
    fn drop(&mut self) {
        let start = std::time::Instant::now();
        let mut warned = false;
        let mut sleep_ms = 1u64;
        const MAX_SLEEP_MS: u64 = 50;
        const WARN_SECS: u64 = 5;
        const TIMEOUT_SECS: u64 = 30;

        while self.pending.is_pending() {
            let elapsed = start.elapsed();

            if elapsed > std::time::Duration::from_secs(TIMEOUT_SECS) {
                tracing::error!(
                    target: "rmlx_distributed",
                    wr_id = self.pending.wr_id(),
                    elapsed_secs = elapsed.as_secs(),
                    "ZeroCopyPendingOp timed out after {}s — releasing buffer (potential use-after-free risk)",
                    TIMEOUT_SECS,
                );
                break;
            }

            if !warned && elapsed > std::time::Duration::from_secs(WARN_SECS) {
                tracing::warn!(
                    target: "rmlx_distributed",
                    wr_id = self.pending.wr_id(),
                    "ZeroCopyPendingOp still pending after {}s, blocking until complete or timeout",
                    WARN_SECS,
                );
                warned = true;
            }

            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            sleep_ms = (sleep_ms * 2).min(MAX_SLEEP_MS);
        }
    }
}

/// Concrete transport wrapping real `RdmaConnection`s (one per peer rank).
///
/// Each connection is wrapped in a `Mutex` to satisfy the `Sync` requirement
/// of `RdmaTransport` — `RdmaConnection` uses `RefCell` internally for its
/// completion backlog, which is not `Sync`. The Mutex serializes access per
/// peer connection.
///
/// `connections[i]` is the connection to peer rank `i`.
/// The slot at `connections[local_rank]` is `None` (no self-connection;
/// opening the same RDMA device twice in one process hangs on macOS).
///
/// wr_id encoding uses `encode_wr_id(seq, tag, buf_slot, peer_id)` from
/// `rmlx_rdma::exchange_tag` — a structured 64-bit layout with:
/// - [63..24] seq (40 bits), [23..16] tag (8 bits),
/// - [15..8] buf_slot (8 bits), [7..0] peer_id (8 bits).
pub struct RdmaConnectionTransport {
    connections: Vec<Option<Mutex<RdmaConnection>>>,
    local_rank: u32,
    metrics: Arc<RdmaMetrics>,
    stripe_engine: Option<StripeEngine>,
    /// Optional secondary connections for dual-port TB5 striping.
    /// When present, secondary stripe chunks are sent/received via these connections.
    /// `secondary_connections[i]` is the secondary connection to peer rank `i`.
    secondary_connections: Option<Vec<Mutex<RdmaConnection>>>,
    /// Optional progress engine for async (non-blocking) send/recv.
    /// When present, `send_async`/`recv_async` register ops with the engine
    /// and return `PendingOp` handles instead of blocking on completion.
    progress_engine: Option<Arc<ProgressEngine>>,
    /// Per-peer monotonic sequence counter for unique wr_id generation.
    /// `wr_id_seqs[peer_rank]` generates sequence numbers for that peer.
    wr_id_seqs: Vec<AtomicU64>,
}

impl RdmaConnectionTransport {
    /// Create a new transport from a set of peer connections.
    ///
    /// `connections` must contain one entry per rank in the world.
    /// The entry at index `local_rank` must be `None` (no self-connection).
    /// All other entries must be `Some`.
    pub fn new(connections: Vec<Option<RdmaConnection>>, local_rank: u32) -> Self {
        let world_size = connections.len();
        let wr_id_seqs: Vec<AtomicU64> = (0..world_size).map(|_| AtomicU64::new(0)).collect();
        Self {
            connections: connections.into_iter().map(|c| c.map(Mutex::new)).collect(),
            local_rank,
            metrics: Arc::new(RdmaMetrics::new()),
            stripe_engine: None,
            secondary_connections: None,
            progress_engine: None,
            wr_id_seqs,
        }
    }

    /// Get a lock on the connection for the given peer rank.
    ///
    /// # Panics
    ///
    /// Panics if `rank` is the local rank (self-slot is `None`).
    fn conn(&self, rank: usize) -> parking_lot::MutexGuard<'_, RdmaConnection> {
        self.connections[rank]
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "attempted RDMA I/O on self-rank slot (rank {rank}); \
                     self-connections are not created"
                )
            })
            .lock()
    }

    /// Attach a StripeEngine for dual-port TB5 striping.
    ///
    /// When set, large sends will be split across dual ports for increased
    /// bandwidth using the stripe plan.
    pub fn with_stripe_engine(mut self, engine: StripeEngine) -> Self {
        self.stripe_engine = Some(engine);
        self
    }

    /// Attach secondary connections for dual-port TB5 striping.
    ///
    /// When set along with a StripeEngine, secondary stripe chunks are
    /// routed through these connections for true dual-port bandwidth.
    /// `secondary` must have the same length as the primary connections.
    pub fn with_secondary_connections(mut self, secondary: Vec<RdmaConnection>) -> Self {
        self.secondary_connections = Some(secondary.into_iter().map(Mutex::new).collect());
        self
    }

    /// Whether dual-port striping is configured.
    pub fn has_striping(&self) -> bool {
        self.stripe_engine
            .as_ref()
            .is_some_and(|e| e.config().has_dual())
    }

    /// Whether secondary connections are available for true dual-port I/O.
    pub fn has_secondary(&self) -> bool {
        self.secondary_connections.is_some()
    }

    /// Attach a ProgressEngine for async (non-blocking) send/recv.
    ///
    /// When set, `send_async`/`recv_async` methods become available,
    /// returning `PendingOp` handles that can be polled or waited on
    /// without blocking the caller.
    pub fn with_progress_engine(mut self, engine: Arc<ProgressEngine>) -> Self {
        self.progress_engine = Some(engine);
        self
    }

    /// Whether a progress engine is attached for async operations.
    pub fn has_progress_engine(&self) -> bool {
        self.progress_engine.is_some()
    }

    /// This node's rank.
    pub fn local_rank(&self) -> u32 {
        self.local_rank
    }

    /// Number of connections (== world size).
    pub fn world_size(&self) -> usize {
        self.connections.len()
    }

    /// Get metrics reference.
    pub fn metrics(&self) -> &RdmaMetrics {
        &self.metrics
    }
}

/// Default chunk size for stripe splitting (64KB).
const STRIPE_CHUNK_SIZE: usize = 64 * 1024;

/// Minimum data size to activate striping (256KB).
const STRIPE_BYTE_THRESHOLD: usize = 256 * 1024;

impl RdmaConnectionTransport {
    /// Generate a unique wr_id for a peer using the structured encoding.
    ///
    /// Layout: `encode_wr_id(seq, tag, buf_slot, peer_id)` — see `exchange_tag`.
    fn next_wr_id(&self, peer_rank: u32, tag: ExchangeTag, buf_slot: u8) -> u64 {
        let seq = self.wr_id_seqs[peer_rank as usize].fetch_add(1, Ordering::Relaxed);
        encode_wr_id(seq, tag, buf_slot, peer_rank as u8)
    }

    /// Check whether striping should be used for the given data size.
    fn should_stripe(&self, data_len: usize) -> bool {
        data_len >= STRIPE_BYTE_THRESHOLD
            && self
                .stripe_engine
                .as_ref()
                .is_some_and(|e| e.config().has_dual())
    }

    /// Send data using stripe engine chunking.
    ///
    /// Splits data into chunks per the stripe plan. Primary chunks are sent
    /// via the primary connection. Secondary chunks are routed to the secondary
    /// connection when available, otherwise they also go through the primary.
    fn send_striped(
        &self,
        data: &[u8],
        dst_rank: u32,
        conn: &RdmaConnection,
    ) -> Result<(), DistributedError> {
        let engine = self
            .stripe_engine
            .as_ref()
            .ok_or_else(|| DistributedError::Transport("stripe_engine not set".into()))?;

        let plan = engine.plan(data.len(), STRIPE_CHUNK_SIZE);
        let (primary_slices, secondary_slices) = engine.split_by_plan(data, &plan);

        // Send primary chunks via primary connection
        for chunk in primary_slices.iter() {
            let wr_id = self.next_wr_id(dst_rank, ExchangeTag::Data, 0);
            global_counters().record_mr_reg();
            let reg = conn.register_send_slice(chunk).map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed_enhanced(e, wr_id)
            })?;
            global_counters().record_rdma_transfer(chunk.len() as u64);
            let _op = conn
                .post_send(reg.mr(), 0, chunk.len() as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_send_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;
            // macOS TB5 RDMA driver bug: skip CQ polling for send completions.
            // yield_now() gives hardware a scheduling slot without the 1ms fixed latency.
            // Send completes locally on UC QPs — no need to wait for remote arrival.
            std::thread::yield_now();
        }

        // Send secondary chunks: use secondary connection if available, else primary
        let sec_conn_guard;
        let sec_conn: &RdmaConnection = if let Some(ref sec_conns) = self.secondary_connections {
            sec_conn_guard = sec_conns[dst_rank as usize].lock();
            &sec_conn_guard
        } else {
            conn
        };

        for chunk in secondary_slices.iter() {
            let wr_id = self.next_wr_id(dst_rank, ExchangeTag::Data, 1);
            global_counters().record_mr_reg();
            let reg = sec_conn.register_send_slice(chunk).map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed_enhanced(e, wr_id)
            })?;
            global_counters().record_rdma_transfer(chunk.len() as u64);
            let _op = sec_conn
                .post_send(reg.mr(), 0, chunk.len() as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_send_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;
            // macOS TB5 RDMA driver bug: skip CQ polling for send completions.
            std::thread::yield_now();
        }

        self.metrics.record_send(data.len() as u64);
        Ok(())
    }

    /// Receive data using stripe engine chunking.
    ///
    /// Primary chunks are received on the primary connection. Secondary chunks
    /// are received on the secondary connection when available, otherwise on
    /// the primary. Reassembles into original byte order.
    fn recv_striped(
        &self,
        src_rank: u32,
        len: usize,
        conn: &RdmaConnection,
    ) -> Result<Vec<u8>, DistributedError> {
        let engine = self
            .stripe_engine
            .as_ref()
            .ok_or_else(|| DistributedError::Transport("stripe_engine not set".into()))?;

        let plan = engine.plan(len, STRIPE_CHUNK_SIZE);
        let primary_count = plan.primary_chunks.len();
        let secondary_count = plan.secondary_chunks.len();

        // UC mode: post ALL recvs upfront before any sends arrive, to prevent
        // silent data drops when a send completes before matching recv is posted.

        // --- Primary chunks: allocate, register, post on primary connection ---
        let mut primary_bufs: Vec<Vec<u8>> = Vec::with_capacity(primary_count);
        let mut primary_mrs = Vec::with_capacity(primary_count);

        for chunk_assign in &plan.primary_chunks {
            let mut buf = vec![0u8; chunk_assign.length];
            // SAFETY: buf is heap-allocated; moving the Vec does not invalidate its
            // heap pointer. The MR is dropped before the buffer (see drop order below).
            global_counters().record_mr_reg();
            let mr = unsafe {
                conn.register_mr(buf.as_mut_ptr() as *mut c_void, chunk_assign.length)
                    .map_err(|e| {
                        self.metrics.record_recv_error();
                        rdma_to_distributed(e)
                    })?
            };
            primary_mrs.push(mr);
            primary_bufs.push(buf);
        }

        let mut primary_ops = Vec::with_capacity(primary_count);
        for (i, chunk_assign) in plan.primary_chunks.iter().enumerate() {
            let wr_id = self.next_wr_id(src_rank, ExchangeTag::Data, 0);
            global_counters().record_rdma_transfer(chunk_assign.length as u64);
            let op = conn
                .post_recv(&primary_mrs[i], 0, chunk_assign.length as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;
            primary_ops.push(op);
        }

        // --- Secondary chunks: use secondary connection if available ---
        let sec_conn_guard;
        let sec_conn: &RdmaConnection = if let Some(ref sec_conns) = self.secondary_connections {
            sec_conn_guard = sec_conns[src_rank as usize].lock();
            &sec_conn_guard
        } else {
            conn
        };

        let mut secondary_bufs: Vec<Vec<u8>> = Vec::with_capacity(secondary_count);
        let mut secondary_mrs = Vec::with_capacity(secondary_count);

        for chunk_assign in &plan.secondary_chunks {
            let mut buf = vec![0u8; chunk_assign.length];
            // SAFETY: buf is heap-allocated; moving the Vec does not invalidate its
            // heap pointer. The MR is dropped before the buffer (see drop order below).
            global_counters().record_mr_reg();
            let mr = unsafe {
                sec_conn
                    .register_mr(buf.as_mut_ptr() as *mut c_void, chunk_assign.length)
                    .map_err(|e| {
                        self.metrics.record_recv_error();
                        rdma_to_distributed(e)
                    })?
            };
            secondary_mrs.push(mr);
            secondary_bufs.push(buf);
        }

        let mut secondary_ops = Vec::with_capacity(secondary_count);
        for (i, chunk_assign) in plan.secondary_chunks.iter().enumerate() {
            let wr_id = self.next_wr_id(src_rank, ExchangeTag::Data, 1);
            global_counters().record_rdma_transfer(chunk_assign.length as u64);
            let op = sec_conn
                .post_recv(&secondary_mrs[i], 0, chunk_assign.length as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;
            secondary_ops.push(op);
        }

        // Wait for all recv completions via CQ poll (JACCL pattern).
        // Previous sleep(1ms) was a workaround for a misdiagnosed "driver bug" —
        // the real issue was stale CQ completions from prior operations.
        // With proper CQ drain in graceful_shutdown, CQ poll works correctly.
        std::thread::yield_now();

        // Drop ops and MRs
        drop(primary_ops);
        drop(primary_mrs);
        drop(secondary_ops);
        drop(secondary_mrs);

        let result = engine.reassemble_from_chunks(&primary_bufs, &secondary_bufs, &plan);
        self.metrics.record_recv(len as u64);
        Ok(result)
    }
}

// ─── Async (non-blocking) send/recv using ProgressEngine ───

impl RdmaConnectionTransport {
    /// Non-blocking send: posts the send and returns an `OwnedPendingOp` immediately.
    ///
    /// The returned `OwnedPendingOp` owns the MR — it stays registered for the
    /// duration of the RDMA operation and is automatically deregistered when the
    /// op completes or is dropped.
    ///
    /// # Safety
    /// `data` must remain valid and unmodified until the returned op completes.
    pub unsafe fn send_async(
        &self,
        data: &[u8],
        dst_rank: u32,
    ) -> Result<OwnedPendingOp, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport("send_async: no progress engine attached".into())
        })?;

        let conn = self.conn(dst_rank as usize);

        let wr_id = self.next_wr_id(dst_rank, ExchangeTag::Data, 0);
        let pending = engine.register_op(wr_id);

        // SAFETY: data pointer is valid for data.len() bytes (caller guarantees
        // `data` outlives the RDMA operation — see fn-level safety doc).
        global_counters().record_mr_reg();
        let mr =
            unsafe { conn.register_mr(data.as_ptr() as *mut c_void, data.len()) }.map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed_enhanced(e, wr_id)
            })?;

        // Post send — the MR is moved into OwnedPendingOp which keeps it alive.
        global_counters().record_rdma_transfer(data.len() as u64);
        let _op = conn
            .post_send(&mr, 0, data.len() as u32, wr_id)
            .map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed_enhanced(e, wr_id)
            })?;

        self.metrics.record_send(data.len() as u64);
        Ok(OwnedPendingOp::new(pending, mr))
    }

    /// Non-blocking recv: posts the recv and returns an `OwnedPendingOp` immediately.
    ///
    /// The returned `OwnedPendingOp` owns the MR — it stays registered for the
    /// duration of the RDMA operation and is automatically deregistered when the
    /// op completes or is dropped.
    ///
    /// # Safety
    /// `buf` must remain valid and unmodified until the returned op completes.
    pub unsafe fn recv_async(
        &self,
        buf: &mut [u8],
        src_rank: u32,
    ) -> Result<OwnedPendingOp, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport("recv_async: no progress engine attached".into())
        })?;

        let conn = self.conn(src_rank as usize);

        let wr_id = self.next_wr_id(src_rank, ExchangeTag::Data, 0);
        let pending = engine.register_op(wr_id);

        // SAFETY: buf pointer is valid for buf.len() bytes (caller guarantees
        // `buf` outlives the RDMA operation — see fn-level safety doc).
        global_counters().record_mr_reg();
        let mr = unsafe { conn.register_mr(buf.as_mut_ptr() as *mut c_void, buf.len()) }.map_err(
            |e| {
                self.metrics.record_recv_error();
                rdma_to_distributed_enhanced(e, wr_id)
            },
        )?;

        global_counters().record_rdma_transfer(buf.len() as u64);
        let _op = conn
            .post_recv(&mr, 0, buf.len() as u32, wr_id)
            .map_err(|e| {
                self.metrics.record_recv_error();
                rdma_to_distributed_enhanced(e, wr_id)
            })?;

        self.metrics.record_recv(buf.len() as u64);
        Ok(OwnedPendingOp::new(pending, mr))
    }
}

// ─── UC recv-credit pre-posting for tag-based isolation ───

/// A pre-posted recv credit: an allocated buffer + its PendingOp handle.
///
/// In UC mode, recvs must be posted before matching sends arrive.
/// `RecvCredit` represents a single pre-posted recv slot that will capture
/// an incoming message tagged with a specific `ExchangeTag`.
pub struct RecvCredit {
    /// The PendingOp handle returned by the ProgressEngine.
    pub pending: PendingOp,
    /// The buffer that will be written into by the RDMA recv.
    /// Remains valid until the PendingOp completes.
    pub buffer: Vec<u8>,
    /// The tag this credit was posted for.
    pub tag: ExchangeTag,
    /// Source rank.
    pub src_rank: u32,
    /// Held alive for the duration of the RDMA op; deregistered on drop.
    _mr: rmlx_rdma::MemoryRegion,
}

impl RdmaConnectionTransport {
    /// Pre-post a window of recv credits for a specific tag and peer.
    ///
    /// UC mode silently drops data if a recv is not posted before the
    /// matching send arrives. This method pre-posts `count` recv buffers
    /// of `buf_size` bytes each, all tagged with `tag`, so incoming sends
    /// are guaranteed to find a matching recv.
    ///
    /// Returns a Vec of `RecvCredit` handles. Each credit's `PendingOp`
    /// will resolve when data arrives. The caller must keep the credits
    /// alive until they are consumed (the buffer memory is owned by the credit).
    ///
    /// Requires a progress engine to be attached.
    ///
    /// # Safety
    /// The returned `RecvCredit`s own their buffers and their MRs (RAII).
    /// The MR is automatically deregistered when the `RecvCredit` is dropped.
    pub unsafe fn pre_post_recv_credits(
        &self,
        src_rank: u32,
        tag: ExchangeTag,
        buf_size: usize,
        count: usize,
    ) -> Result<Vec<RecvCredit>, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport("pre_post_recv_credits: no progress engine attached".into())
        })?;

        let conn = self.conn(src_rank as usize);

        let mut credits = Vec::with_capacity(count);

        for slot in 0..count {
            let mut buffer = vec![0u8; buf_size];
            let wr_id = self.next_wr_id(src_rank, tag, slot as u8);
            let pending = engine.register_op(wr_id);

            // SAFETY: buffer pointer is valid for buf_size bytes. The buffer is
            // moved into RecvCredit which keeps it alive for the RDMA operation.
            global_counters().record_mr_reg();
            let mr = unsafe { conn.register_mr(buffer.as_mut_ptr() as *mut c_void, buf_size) }
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;

            global_counters().record_rdma_transfer(buf_size as u64);
            let _op = conn
                .post_recv(&mr, 0, buf_size as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;

            credits.push(RecvCredit {
                pending,
                buffer,
                tag,
                src_rank,
                _mr: mr,
            });
        }

        Ok(credits)
    }

    /// Pre-post recv credits on a SharedBuffer's pre-registered MR.
    ///
    /// Like `pre_post_recv_credits`, but uses the SharedBuffer's already-registered
    /// MR instead of allocating new buffers. Each credit claims a `buf_size` slice
    /// starting at `offset + slot * buf_size` within the SharedBuffer.
    ///
    /// This is the zero-copy variant: received data lands directly in GPU-visible memory.
    /// Each returned `ZeroCopyPendingOp` holds an `Arc<SharedBuffer>` reference,
    /// preventing the buffer from being freed while DMA is in flight.
    #[allow(clippy::too_many_arguments)]
    pub fn pre_post_recv_credits_zero_copy(
        &self,
        src_rank: u32,
        tag: ExchangeTag,
        buf: Arc<SharedBuffer>,
        conn_id: &ConnectionId,
        offset: usize,
        buf_size: usize,
        count: usize,
    ) -> Result<Vec<ZeroCopyPendingOp>, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport(
                "pre_post_recv_credits_zero_copy: no progress engine attached".into(),
            )
        })?;

        let conn = self.conn(src_rank as usize);

        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "SharedBuffer not registered on connection {:?}",
                conn_id
            ))
        })?;

        let mut ops = Vec::with_capacity(count);

        for slot in 0..count {
            let slot_offset = offset + slot * buf_size;
            if slot_offset + buf_size > buf.size() {
                return Err(DistributedError::Transport(format!(
                    "recv credit slot {} exceeds SharedBuffer size (offset={}, buf_size={}, total={})",
                    slot, slot_offset, buf_size, buf.size()
                )));
            }

            let wr_id = self.next_wr_id(src_rank, tag, buf.slot_index());
            let pending = engine.register_op(wr_id);

            global_counters().record_rdma_transfer(buf_size as u64);
            let _op = conn
                .post_recv(mr, slot_offset, buf_size as u32, wr_id)
                .map_err(|e| {
                    self.metrics.record_recv_error();
                    rdma_to_distributed_enhanced(e, wr_id)
                })?;

            ops.push(ZeroCopyPendingOp::new(pending, Arc::clone(&buf)));
        }

        Ok(ops)
    }
}

// ─── GPU-proxy integration for descriptor ring dispatch ───

impl RdmaConnectionTransport {
    /// Dispatch a single GPU-written descriptor by posting the corresponding
    /// ibv_post_send/recv on the appropriate connection.
    ///
    /// `buffers` maps buf_slot index to (SharedBuffer, ConnectionId) pairs.
    /// The method looks up the pre-registered MR from the SharedBuffer and
    /// posts the send/recv on the peer's connection.
    ///
    /// This is meant to be called from a DescriptorProxy handler running on
    /// the same thread that owns the SharedBuffers (since SharedBuffer is
    /// Send but not Sync).
    pub fn dispatch_descriptor(
        &self,
        desc: &RdmaDescriptor,
        wr_id: u64,
        buffers: &[(SharedBuffer, ConnectionId)],
    ) -> Result<(), DistributedError> {
        let peer_id = desc.peer_id as u32;
        let buf_slot = desc.buf_slot as usize;

        if buf_slot >= buffers.len() {
            return Err(DistributedError::Transport(format!(
                "descriptor buf_slot {} out of range (have {})",
                buf_slot,
                buffers.len()
            )));
        }

        let (ref buf, ref conn_id) = buffers[buf_slot];
        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "no MR for buf_slot={} conn_id={:?}",
                buf_slot, conn_id
            ))
        })?;

        let conn = self.conn(peer_id as usize);

        // Register with progress engine if available
        if let Some(ref engine) = self.progress_engine {
            let _pending = engine.register_op(wr_id);
        }

        match desc.rdma_op() {
            Some(RdmaOp::Send) => {
                global_counters().record_rdma_transfer(desc.length as u64);
                let _op = conn
                    .post_send(mr, desc.offset as usize, desc.length, wr_id)
                    .map_err(|e| rdma_to_distributed_enhanced(e, wr_id))?;
            }
            Some(RdmaOp::Recv) => {
                global_counters().record_rdma_transfer(desc.length as u64);
                let _op = conn
                    .post_recv(mr, desc.offset as usize, desc.length, wr_id)
                    .map_err(|e| rdma_to_distributed_enhanced(e, wr_id))?;
            }
            None => {
                return Err(DistributedError::Transport(format!(
                    "invalid op byte {}",
                    desc.op
                )));
            }
        }

        Ok(())
    }
}

// ─── Zero-copy send/recv using SharedBuffer pre-registered MRs ───

impl RdmaConnectionTransport {
    /// Zero-copy send: uses the SharedBuffer's pre-registered RDMA MR directly.
    ///
    /// No memcpy, no ibv_reg_mr — the data is sent directly from the
    /// SharedBuffer's memory, which is already registered on this connection's PD.
    ///
    /// Requires:
    /// - `buf.rdma_mr(&conn_id)` returns a valid MR (buffer must be registered
    ///   on this connection's protection domain)
    /// - Data must already be written into `buf` (e.g., by a Metal compute kernel
    ///   writing to `buf.metal_buffer()`)
    pub fn send_zero_copy(
        &self,
        buf: &SharedBuffer,
        conn_id: &ConnectionId,
        dst_rank: u32,
        len: u32,
    ) -> Result<(), DistributedError> {
        let conn = self.conn(dst_rank as usize);

        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "SharedBuffer not registered on connection {:?}",
                conn_id
            ))
        })?;

        let wr_id = self.next_wr_id(dst_rank, ExchangeTag::Data, buf.slot_index());
        global_counters().record_rdma_transfer(len as u64);
        let _op = conn.post_send(mr, 0, len, wr_id).map_err(|e| {
            self.metrics.record_send_error();
            rdma_to_distributed_enhanced(e, wr_id)
        })?;
        // macOS TB5 RDMA driver bug: skip CQ polling for send completions.
        // Send completes locally on UC QPs — yield instead of 1ms sleep.
        std::thread::yield_now();

        self.metrics.record_send(len as u64);
        Ok(())
    }

    /// Zero-copy recv: posts a recv using the SharedBuffer's pre-registered MR.
    ///
    /// The received data lands directly in the SharedBuffer's memory, which
    /// is simultaneously accessible as a Metal GPU buffer — no memcpy needed
    /// for subsequent GPU compute.
    pub fn recv_zero_copy(
        &self,
        buf: &SharedBuffer,
        conn_id: &ConnectionId,
        src_rank: u32,
        len: u32,
    ) -> Result<(), DistributedError> {
        let conn = self.conn(src_rank as usize);

        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "SharedBuffer not registered on connection {:?}",
                conn_id
            ))
        })?;

        let wr_id = self.next_wr_id(src_rank, ExchangeTag::Data, buf.slot_index());
        global_counters().record_rdma_transfer(len as u64);
        let _op = conn.post_recv(mr, 0, len, wr_id).map_err(|e| {
            self.metrics.record_recv_error();
            rdma_to_distributed_enhanced(e, wr_id)
        })?;
        // Yield to let hardware complete — JACCL pattern shows CQ poll works
        // fine on UC QPs. Previous sleep(1ms) was misdiagnosed driver bug.
        std::thread::yield_now();

        self.metrics.record_recv(len as u64);
        Ok(())
    }

    /// Zero-copy async send: posts the send using SharedBuffer's pre-registered MR
    /// and returns a `ZeroCopyPendingOp` immediately without blocking.
    ///
    /// The returned `ZeroCopyPendingOp` holds an `Arc<SharedBuffer>` reference,
    /// preventing the buffer from being freed while the DMA is in flight.
    ///
    /// This is the ideal data path: 0 memcpy + 0 ibv_reg_mr + non-blocking.
    pub fn send_zero_copy_async(
        &self,
        buf: Arc<SharedBuffer>,
        conn_id: &ConnectionId,
        dst_rank: u32,
        len: u32,
    ) -> Result<ZeroCopyPendingOp, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport("send_zero_copy_async: no progress engine attached".into())
        })?;

        let conn = self.conn(dst_rank as usize);

        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "SharedBuffer not registered on connection {:?}",
                conn_id
            ))
        })?;

        let wr_id = self.next_wr_id(dst_rank, ExchangeTag::Data, buf.slot_index());
        let pending = engine.register_op(wr_id);

        global_counters().record_rdma_transfer(len as u64);
        let _op = conn.post_send(mr, 0, len, wr_id).map_err(|e| {
            self.metrics.record_send_error();
            rdma_to_distributed_enhanced(e, wr_id)
        })?;

        self.metrics.record_send(len as u64);
        Ok(ZeroCopyPendingOp::new(pending, buf))
    }

    /// Zero-copy async recv: posts the recv using SharedBuffer's pre-registered MR
    /// and returns a `ZeroCopyPendingOp` immediately without blocking.
    ///
    /// The returned `ZeroCopyPendingOp` holds an `Arc<SharedBuffer>` reference,
    /// preventing the buffer from being freed while the DMA is in flight.
    pub fn recv_zero_copy_async(
        &self,
        buf: Arc<SharedBuffer>,
        conn_id: &ConnectionId,
        src_rank: u32,
        len: u32,
    ) -> Result<ZeroCopyPendingOp, DistributedError> {
        let engine = self.progress_engine.as_ref().ok_or_else(|| {
            DistributedError::Transport("recv_zero_copy_async: no progress engine attached".into())
        })?;

        let conn = self.conn(src_rank as usize);

        let mr = buf.rdma_mr(conn_id).ok_or_else(|| {
            DistributedError::Transport(format!(
                "SharedBuffer not registered on connection {:?}",
                conn_id
            ))
        })?;

        let wr_id = self.next_wr_id(src_rank, ExchangeTag::Data, buf.slot_index());
        let pending = engine.register_op(wr_id);

        global_counters().record_rdma_transfer(len as u64);
        let _op = conn.post_recv(mr, 0, len, wr_id).map_err(|e| {
            self.metrics.record_recv_error();
            rdma_to_distributed_enhanced(e, wr_id)
        })?;

        self.metrics.record_recv(len as u64);
        Ok(ZeroCopyPendingOp::new(pending, buf))
    }
}

fn rdma_to_distributed(e: RdmaError) -> DistributedError {
    DistributedError::Transport(e.to_string())
}

/// Enhanced error conversion that decodes wr_id fields for richer diagnostics.
fn rdma_to_distributed_enhanced(e: RdmaError, wr_id: u64) -> DistributedError {
    let detail = match try_decode_wr_id(wr_id) {
        Some(fields) => format!(
            "{} [wr_id: seq={}, tag={:?}, slot={}, peer={}]",
            e, fields.seq, fields.tag, fields.buf_slot, fields.peer_id
        ),
        None => format!("{} [wr_id: 0x{:016x} (decode failed)]", e, wr_id),
    };
    DistributedError::Transport(detail)
}

impl RdmaTransport for RdmaConnectionTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        global_counters().record_fallback();
        let conn = self.conn(dst_rank as usize);

        // Use striped send for large payloads when stripe engine is configured
        if self.should_stripe(data.len()) {
            return self.send_striped(data, dst_rank, &conn);
        }

        // Use chunked send with tiered buffer pool (JACCL pipelining)
        global_counters().record_rdma_transfer(data.len() as u64);
        conn.chunked_send(data).map_err(|e| {
            self.metrics.record_send_error();
            rdma_to_distributed(e)
        })?;

        self.metrics.record_send(data.len() as u64);
        Ok(())
    }

    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        global_counters().record_fallback();
        let conn = self.conn(src_rank as usize);

        // Use striped recv for large payloads when stripe engine is configured
        if self.should_stripe(len) {
            return self.recv_striped(src_rank, len, &conn);
        }

        // Use chunked recv with tiered buffer pool (JACCL pipelining)
        // CQ polling is handled internally by chunked_recv.
        global_counters().record_rdma_transfer(len as u64);
        let buf = conn.chunked_recv(len).map_err(|e| {
            self.metrics.record_recv_error();
            rdma_to_distributed(e)
        })?;

        self.metrics.record_recv(len as u64);
        Ok(buf)
    }

    fn sendrecv(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_len: usize,
        src_rank: u32,
    ) -> Result<Vec<u8>, DistributedError> {
        global_counters().record_fallback();
        // Chunked sendrecv handles CQ polling internally with PIPELINE=2.
        // For same-peer case, use the single connection's chunked_sendrecv directly.
        // For different peers, send and recv happen on separate connections,
        // so use chunked_send + chunked_recv independently.
        if dst_rank == src_rank {
            let conn = self.conn(dst_rank as usize);

            global_counters().record_rdma_transfer(send_data.len() as u64);
            global_counters().record_rdma_transfer(recv_len as u64);
            let recv_buf = conn.chunked_sendrecv(send_data, recv_len).map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed(e)
            })?;

            self.metrics.record_send(send_data.len() as u64);
            self.metrics.record_recv(recv_len as u64);
            Ok(recv_buf)
        } else {
            // Different peers: lock in rank order to avoid deadlock
            let (first_rank, second_rank) = if dst_rank < src_rank {
                (dst_rank, src_rank)
            } else {
                (src_rank, dst_rank)
            };
            let first_conn = self.conn(first_rank as usize);
            let second_conn = self.conn(second_rank as usize);

            let (dst_conn, src_conn) = if dst_rank < src_rank {
                (&*first_conn, &*second_conn)
            } else {
                (&*second_conn, &*first_conn)
            };

            // Post recv FIRST on src connection (UC mode: recv before send)
            // Then send on dst connection. Both use chunked pipelining internally.
            // Note: for different-peer sendrecv, we run chunked_recv and chunked_send
            // on separate connections. We need recv posted before remote send arrives.
            // Since chunked_recv posts recvs and polls, and chunked_send posts sends
            // and polls, we need to interleave them. Use chunked_sendrecv on src_conn
            // for recv-only (empty send) and dst_conn for send-only (zero recv).
            global_counters().record_rdma_transfer(recv_len as u64);
            global_counters().record_rdma_transfer(send_data.len() as u64);

            // Post recv on src connection first, then send on dst connection.
            // chunked_sendrecv with empty send_data acts as recv-only.
            // chunked_send handles send with internal CQ polling.
            let recv_buf = src_conn.chunked_recv(recv_len).map_err(|e| {
                self.metrics.record_recv_error();
                rdma_to_distributed(e)
            })?;

            dst_conn.chunked_send(send_data).map_err(|e| {
                self.metrics.record_send_error();
                rdma_to_distributed(e)
            })?;

            self.metrics.record_send(send_data.len() as u64);
            self.metrics.record_recv(recv_len as u64);
            Ok(recv_buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::group::{DistributedError, Group, RdmaTransport};
    use parking_lot::Mutex;
    use std::sync::{Arc, OnceLock};

    fn test_device() -> Option<&'static rmlx_metal::MtlDevice> {
        static DEVICE: OnceLock<Option<rmlx_metal::MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| {
                objc2::rc::autoreleasepool(|_| objc2_metal::MTLCreateSystemDefaultDevice())
            })
            .as_ref()
    }

    /// A mock transport that records calls for testing.
    struct MockTransport {
        sent: Mutex<Vec<(Vec<u8>, u32)>>,
        recv_data: Mutex<Vec<u8>>,
    }

    impl MockTransport {
        fn new(recv_data: Vec<u8>) -> Self {
            Self {
                sent: Mutex::new(Vec::new()),
                recv_data: Mutex::new(recv_data),
            }
        }
    }

    impl RdmaTransport for MockTransport {
        fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
            self.sent.lock().push((data.to_vec(), dst_rank));
            Ok(())
        }

        fn recv(&self, _src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
            let data = self.recv_data.lock();
            Ok(data[..len].to_vec())
        }

        fn sendrecv(
            &self,
            send_data: &[u8],
            dst_rank: u32,
            recv_len: usize,
            src_rank: u32,
        ) -> Result<Vec<u8>, DistributedError> {
            self.send(send_data, dst_rank)?;
            self.recv(src_rank, recv_len)
        }
    }

    #[test]
    fn test_mock_transport_send_recv() {
        let payload = vec![1u8, 2, 3, 4];
        let mock = MockTransport::new(payload.clone());

        mock.send(&payload, 1).unwrap();
        let sent = mock.sent.lock();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, payload);
        assert_eq!(sent[0].1, 1);

        let received = mock.recv(0, 4).unwrap();
        assert_eq!(received, payload);
    }

    #[test]
    fn test_group_with_mock_transport_broadcast() {
        let data = vec![0xAA; 8]; // 8 bytes, 4-byte aligned
        let mock = Arc::new(MockTransport::new(data.clone()));

        // Root rank broadcasts
        let group = Group::with_transport(vec![0, 1, 2], 0, 3, mock.clone()).unwrap();
        let result = group.broadcast(&data, 0).unwrap();
        assert_eq!(result, data);

        // Root should have sent to 2 peers
        let sent = mock.sent.lock();
        assert_eq!(sent.len(), 2);
    }

    #[test]
    fn test_wr_id_encoding_new() {
        use rmlx_rdma::exchange_tag::{decode_wr_id, encode_wr_id, ExchangeTag};

        // Verify encode_wr_id roundtrips correctly
        let wr_id = encode_wr_id(42, ExchangeTag::Data, 0, 7);
        let fields = decode_wr_id(wr_id).unwrap();
        assert_eq!(fields.seq, 42);
        assert_eq!(fields.tag, ExchangeTag::Data);
        assert_eq!(fields.buf_slot, 0);
        assert_eq!(fields.peer_id, 7);

        // Verify different tags encode distinctly
        let warmup_id = encode_wr_id(0, ExchangeTag::Warmup, 1, 0);
        let warmup_fields = decode_wr_id(warmup_id).unwrap();
        assert_eq!(warmup_fields.tag, ExchangeTag::Warmup);
        assert_eq!(warmup_fields.buf_slot, 1);

        // Verify buf_slot 0 vs 1 (primary vs secondary) are distinct
        let primary = encode_wr_id(10, ExchangeTag::Data, 0, 5);
        let secondary = encode_wr_id(10, ExchangeTag::Data, 1, 5);
        assert_ne!(primary, secondary);
    }

    #[test]
    fn test_wr_id_uniqueness_new() {
        use rmlx_rdma::exchange_tag::{encode_wr_id, ExchangeTag};

        // Verify that different seq values produce unique wr_ids for same tag/slot/peer
        let mut ids = std::collections::HashSet::new();
        for seq in 0..100u64 {
            let data_id = encode_wr_id(seq, ExchangeTag::Data, 0, 1);
            assert!(ids.insert(data_id), "duplicate wr_id at seq={seq}");
        }
        assert_eq!(ids.len(), 100);

        // Different peers with same seq are also unique
        let peer0 = encode_wr_id(0, ExchangeTag::Data, 0, 0);
        let peer1 = encode_wr_id(0, ExchangeTag::Data, 0, 1);
        assert_ne!(peer0, peer1);
    }

    #[test]
    fn test_group_with_mock_transport_send_recv() {
        let data = vec![0xBB; 4]; // 4 bytes, aligned
        let mock = Arc::new(MockTransport::new(data.clone()));

        let group = Group::with_transport(vec![0, 1], 0, 2, mock.clone()).unwrap();

        group.send(&data, 1).unwrap();
        let sent = mock.sent.lock();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].1, 1);
        drop(sent);

        let received = group.recv(1, 4).unwrap();
        assert_eq!(received, data);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn test_zero_copy_pending_op_keeps_buffer_alive() {
        use rmlx_rdma::progress::ProgressEngine;
        use rmlx_rdma::shared_buffer::SharedBuffer;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let buf = match SharedBuffer::new(device, 4096, 0) {
            Ok(b) => b,
            Err(_) => return, // skip if alloc fails
        };
        let buf = Arc::new(buf);

        // Create a pending op via the progress engine
        let engine = ProgressEngine::new();
        let pending = engine.register_op(42);
        assert!(pending.is_pending());

        // Create ZeroCopyPendingOp — this clones the Arc
        let zc_op = super::ZeroCopyPendingOp::new(pending, Arc::clone(&buf));

        // Drop our local Arc — the ZeroCopyPendingOp should keep the buffer alive
        assert_eq!(Arc::strong_count(&buf), 2); // buf + zc_op._buf
        drop(buf);

        // The op is still pending and the buffer is alive inside zc_op
        assert!(zc_op.is_pending());

        // Verify we can still access the inner PendingOp
        assert_eq!(zc_op.wr_id(), 42);
        assert!(zc_op.try_poll().is_none()); // still pending

        // Complete the op so the Drop impl doesn't block forever.
        // The key correctness property: the buffer was NOT freed while DMA
        // was in flight — the Drop impl now blocks indefinitely until complete.
        engine.synthetic_complete(42);
        assert!(!zc_op.is_pending());
        drop(zc_op);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn test_zero_copy_pending_op_completed_drops_immediately() {
        use rmlx_rdma::progress::ProgressEngine;
        use rmlx_rdma::shared_buffer::SharedBuffer;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let buf = match SharedBuffer::new(device, 4096, 0) {
            Ok(b) => b,
            Err(_) => return,
        };
        let buf = Arc::new(buf);

        let engine = ProgressEngine::new();
        let pending = engine.register_op(99);

        let zc_op = super::ZeroCopyPendingOp::new(pending, Arc::clone(&buf));

        assert!(zc_op.is_pending());
        assert_eq!(zc_op.wr_id(), 99);

        // Strong count: buf + zc_op
        assert_eq!(Arc::strong_count(&buf), 2);

        // Synthetically complete the op so Drop doesn't block forever.
        engine.synthetic_complete(99);
        assert!(!zc_op.is_pending());

        // Drop zc_op — op is complete so Drop returns immediately.
        // After drop, only `buf` remains.
        drop(zc_op);
        assert_eq!(Arc::strong_count(&buf), 1);
    }

    #[test]
    fn test_stripe_constants() {
        assert_eq!(super::STRIPE_CHUNK_SIZE, 64 * 1024);
        assert_eq!(super::STRIPE_BYTE_THRESHOLD, 256 * 1024);
        // Compile-time guarantee that threshold >= chunk size
        const { assert!(super::STRIPE_BYTE_THRESHOLD >= super::STRIPE_CHUNK_SIZE) };
    }

    #[test]
    fn test_stripe_engine_plan_integration() {
        use rmlx_rdma::multi_port::{DualPortConfig, PortConfig, StripeEngine};

        let primary = PortConfig {
            port_num: 1,
            gid_index: 1,
            interface: "en5".to_string(),
            address: "10.254.0.5".to_string(),
        };
        let secondary = PortConfig {
            port_num: 2,
            gid_index: 1,
            interface: "en6".to_string(),
            address: "10.254.0.6".to_string(),
        };
        let config = DualPortConfig::dual(primary, secondary, 4);
        let engine = StripeEngine::new(config);

        // Plan with 512KB of data, 64KB chunks = 8 chunks
        let plan = engine.plan(512 * 1024, super::STRIPE_CHUNK_SIZE);
        assert_eq!(plan.primary_chunks.len(), 4); // even indices: 0,2,4,6
        assert_eq!(plan.secondary_chunks.len(), 4); // odd indices: 1,3,5,7
        assert_eq!(plan.total_bytes, 512 * 1024);

        // Test split and reassemble roundtrip
        let data: Vec<u8> = (0..512 * 1024).map(|i| (i % 256) as u8).collect();
        let (primary_slices, secondary_slices) = engine.split_by_plan(&data, &plan);
        assert_eq!(primary_slices.len(), 4);
        assert_eq!(secondary_slices.len(), 4);

        let primary_owned: Vec<Vec<u8>> = primary_slices.iter().map(|s| s.to_vec()).collect();
        let secondary_owned: Vec<Vec<u8>> = secondary_slices.iter().map(|s| s.to_vec()).collect();
        let reassembled = engine.reassemble_from_chunks(&primary_owned, &secondary_owned, &plan);
        assert_eq!(reassembled, data);
    }
}
