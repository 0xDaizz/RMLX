//! Communication group abstraction for distributed operations.

use std::fmt;
use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Error type for distributed operations.
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    /// Arrays must be materialized (data resident in GPU buffer) before collective ops.
    #[error("not materialized: {0}")]
    NotMaterialized(String),
    /// RDMA transport error (wraps rmlx_rdma::RdmaError description).
    #[error("transport error: {0}")]
    Transport(String),
    /// Wire protocol or data format error (e.g., byte slice conversion failure).
    #[error("protocol error: {0}")]
    Protocol(String),
    /// Configuration error (missing env vars, invalid configuration values).
    #[error("config error: {0}")]
    Config(String),
    /// Backend unavailable (RDMA hardware not found, backend not available).
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
}

/// Verify all arrays are materialized before a collective operation.
///
/// "Materialized" means the array's backing buffer has actual data:
/// - `numel > 0`: at least one element exists
/// - `byte_size > 0`: allocation is non-zero
/// - `byte_size >= numel`: byte count is at least element count (minimum 1 byte/element)
/// - `byte_size % 4 == 0`: byte size is aligned to a 4-byte boundary (f32 element granularity)
///
/// This must be called before any allreduce, allgather, or broadcast to prevent
/// sending uninitialized memory over RDMA.
pub fn ensure_materialized(shapes: &[(usize, usize)]) -> Result<(), DistributedError> {
    for (i, &(numel, byte_size)) in shapes.iter().enumerate() {
        if numel == 0 || byte_size == 0 {
            return Err(DistributedError::NotMaterialized(format!(
                "array at index {i} has zero elements or zero bytes — \
                 all arrays must be materialized before collective operations"
            )));
        }
        if byte_size < numel {
            return Err(DistributedError::NotMaterialized(format!(
                "array at index {i} has byte_size ({byte_size}) < numel ({numel}) — \
                 byte_size must be at least numel (minimum 1 byte per element)"
            )));
        }
        if byte_size % 4 != 0 {
            return Err(DistributedError::NotMaterialized(format!(
                "array at index {i} has byte_size ({byte_size}) not aligned to 4 bytes — \
                 byte_size must be a multiple of 4 for proper element alignment"
            )));
        }
    }
    Ok(())
}

/// Trait for RDMA transport backends.
///
/// Wraps the underlying RDMA connection to provide send/recv primitives
/// that the Group collectives can use. Implementations are expected to
/// handle memory registration, work request posting, and completion polling.
///
/// # Design note: no async send/recv methods
///
/// `RdmaConnectionTransport` exposes low-level `send_async()` / `recv_async()`
/// that return `OwnedPendingOp` handles for non-blocking RDMA operations.
/// These are intentionally **not** part of this trait because:
///
/// 1. **Type coupling** — `OwnedPendingOp` / `ZeroCopyPendingOp` live in
///    `rmlx_rdma`; pulling them into the generic trait would force every
///    backend (including test mocks) to depend on RDMA-specific types.
///
/// 2. **`sendrecv` already overlaps** — the primary latency-hiding pattern
///    needed by ring collectives is overlapped send+recv to different peers.
///    `sendrecv()` / `sendrecv_into()` already provide this: the concrete
///    `RdmaConnectionTransport` implementation posts recv before send and
///    uses chunked pipelining internally.
///
/// 3. **Downcast escape hatch** — collectives that need fine-grained async
///    control can downcast `&dyn RdmaTransport` to `&RdmaConnectionTransport`
///    and call `send_async()` / `recv_async()` directly, falling back to
///    blocking methods when the concrete type is unavailable.
///
/// If future collectives (e.g. pipelined allgather with compute overlap)
/// require trait-level async, consider a separate `AsyncRdmaTransport`
/// extension trait with an associated `PendingOp` type.
pub trait RdmaTransport: Send + Sync {
    /// Send `data` to the peer at `dst_rank`.
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError>;

    /// Receive `len` bytes from the peer at `src_rank`.
    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError>;

    /// Atomic send-then-receive: send `send_data` to `dst_rank`, then receive
    /// `recv_len` bytes from `src_rank`. This avoids deadlocks in pairwise
    /// exchange patterns by ensuring the send is posted before the recv.
    fn sendrecv(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_len: usize,
        src_rank: u32,
    ) -> Result<Vec<u8>, DistributedError>;

    /// Send data from a borrowed slice (zero-copy friendly path).
    /// Default: delegates to `send()`.
    fn send_ref(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        self.send(data, dst_rank)
    }

    /// Receive data directly into a caller-provided mutable slice.
    /// Default: delegates to `recv()` and copies into the buffer.
    fn recv_into(&self, buf: &mut [u8], src_rank: u32) -> Result<(), DistributedError> {
        let data = self.recv(src_rank, buf.len())?;
        let copy_len = data.len().min(buf.len());
        buf[..copy_len].copy_from_slice(&data[..copy_len]);
        Ok(())
    }

    /// Send-then-receive with a caller-provided mutable recv buffer.
    /// Default: delegates to `sendrecv()` and copies result into recv_buf.
    fn sendrecv_into(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_buf: &mut [u8],
        src_rank: u32,
    ) -> Result<(), DistributedError> {
        let data = self.sendrecv(send_data, dst_rank, recv_buf.len(), src_rank)?;
        let copy_len = data.len().min(recv_buf.len());
        recv_buf[..copy_len].copy_from_slice(&data[..copy_len]);
        Ok(())
    }
}

/// Reduction operation for allreduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum (default).
    Sum,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

/// Data type for reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceDtype {
    /// 32-bit float (default).
    F32,
    /// 16-bit float (IEEE 754 half-precision).
    F16,
    /// 16-bit bfloat.
    Bf16,
}

/// A communication group identifying a set of ranks.
#[derive(Clone)]
pub struct Group {
    /// Ranks in this group (sorted, unique).
    ranks: Vec<u32>,
    /// This node's rank.
    local_rank: u32,
    /// Total world size.
    world_size: u32,
    /// Optional RDMA transport for real multi-node communication.
    /// When None, collectives operate in single-process stub mode.
    transport: Option<Arc<dyn RdmaTransport>>,
}

impl fmt::Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Group")
            .field("ranks", &self.ranks)
            .field("local_rank", &self.local_rank)
            .field("world_size", &self.world_size)
            .field("has_transport", &self.transport.is_some())
            .finish()
    }
}

impl Group {
    /// Create a new group from a list of ranks (single-process stub mode).
    pub fn new(
        ranks: Vec<u32>,
        local_rank: u32,
        world_size: u32,
    ) -> Result<Self, DistributedError> {
        let mut ranks = ranks;
        ranks.sort();
        ranks.dedup();
        if !ranks.contains(&local_rank) {
            return Err(DistributedError::Transport(format!(
                "local_rank {local_rank} not in group {:?}",
                ranks
            )));
        }
        Ok(Self {
            ranks,
            local_rank,
            world_size,
            transport: None,
        })
    }

    /// Create a group with all ranks [0, world_size) (single-process stub mode).
    pub fn world(world_size: u32, local_rank: u32) -> Result<Self, DistributedError> {
        Self::new((0..world_size).collect(), local_rank, world_size)
    }

    /// Create a group with an RDMA transport backend for real multi-node communication.
    pub fn with_transport(
        ranks: Vec<u32>,
        local_rank: u32,
        world_size: u32,
        transport: Arc<dyn RdmaTransport>,
    ) -> Result<Self, DistributedError> {
        let mut ranks = ranks;
        ranks.sort();
        ranks.dedup();
        if !ranks.contains(&local_rank) {
            return Err(DistributedError::Transport(format!(
                "local_rank {local_rank} not in group {:?}",
                ranks
            )));
        }
        Ok(Self {
            ranks,
            local_rank,
            world_size,
            transport: Some(transport),
        })
    }

    /// Whether a real RDMA transport is attached.
    pub fn has_transport(&self) -> bool {
        self.transport.is_some()
    }

    /// Ranks in this group.
    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    /// This node's rank.
    pub fn local_rank(&self) -> u32 {
        self.local_rank
    }

    /// Number of ranks in the group.
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// World size.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Peer ranks (all ranks except local).
    pub fn peers(&self) -> Vec<u32> {
        self.ranks
            .iter()
            .copied()
            .filter(|&r| r != self.local_rank)
            .collect()
    }

    /// Check if a rank is in this group.
    pub fn contains(&self, rank: u32) -> bool {
        self.ranks.contains(&rank)
    }

    /// Split this group into sub-groups based on color and key.
    ///
    /// All ranks with the same `color` end up in the same sub-group.
    /// Within a sub-group, ranks are ordered by `key` (ties broken by rank).
    ///
    /// This mirrors MPI_Comm_split semantics. In single-process stub mode
    /// (no transport), the returned group contains only the local rank.
    /// With transport, the caller is responsible for ensuring all ranks in
    /// the group call `split` with consistent color/key values.
    pub fn split(&self, color: u32, key: u32) -> Result<Group, DistributedError> {
        // In stub mode (no transport), we can only know about ourselves.
        // Build a sub-group containing just the local rank.
        if self.transport.is_none() {
            return Group::new(vec![self.local_rank], self.local_rank, self.world_size);
        }

        // With transport, we need to exchange (color, key) with all peers
        // to determine which ranks share the same color.
        // Encode (color, key, rank) as 12 bytes and allgather.
        let mut my_data = Vec::with_capacity(12);
        my_data.extend_from_slice(&color.to_le_bytes());
        my_data.extend_from_slice(&key.to_le_bytes());
        my_data.extend_from_slice(&self.local_rank.to_le_bytes());

        let transport = self.require_transport("split")?;
        let gathered = ring_allgather(&my_data, &self.ranks, self.local_rank, transport)?;

        // Parse all (color, key, rank) tuples
        let mut entries: Vec<(u32, u32, u32)> = Vec::with_capacity(self.ranks.len());
        for chunk in gathered.chunks_exact(12) {
            let c = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let k = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let r = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
            entries.push((c, k, r));
        }

        // Filter to same color, sort by (key, rank)
        let mut same_color: Vec<(u32, u32)> = entries
            .iter()
            .filter(|(c, _, _)| *c == color)
            .map(|(_, k, r)| (*k, *r))
            .collect();
        same_color.sort();

        let sub_ranks: Vec<u32> = same_color.iter().map(|(_, r)| *r).collect();

        Group::with_transport(
            sub_ranks,
            self.local_rank,
            self.world_size,
            Arc::clone(transport),
        )
    }

    // ─── Collective operations ───
    // All collectives call ensure_materialized() at entry.
    // When transport is Some, real RDMA operations are used.
    // When transport is None (single-process mode), stubs return local data.

    /// All-reduce: sum across all ranks.
    ///
    /// With transport: ring allreduce — each rank sends a chunk to its right
    /// neighbor and receives from its left, accumulating a sum in N-1 rounds,
    /// then a gather phase distributes the final result.
    ///
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn allreduce(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allreduce", data)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("allreduce")?;
        let (result, _algo) = allreduce_auto(data, &self.ranks, self.local_rank, transport)?;
        Ok(result)
    }

    /// All-reduce with configurable reduction operation and dtype.
    ///
    /// Supports Sum, Max, and Min operations on f32, f16, and bf16 data.
    /// For f16/bf16, accumulation is performed in f32 precision.
    ///
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn allreduce_op(
        &self,
        data: &[u8],
        op: ReduceOp,
        dtype: ReduceDtype,
    ) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized_dtype("allreduce_op", data, dtype)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("allreduce_op")?;
        let bf16_threshold = match dtype {
            ReduceDtype::Bf16 => MESH_RING_THRESHOLD_BF16,
            _ => MESH_RING_THRESHOLD,
        };
        if self.ranks.len() > 2 && data.len() >= bf16_threshold {
            ring_allreduce_op(data, &self.ranks, self.local_rank, transport, op, dtype)
        } else {
            mesh_allreduce_op(data, &self.ranks, self.local_rank, transport, op, dtype)
        }
    }

    /// Allreduce with explicit dtype (f16/bf16/f32).
    /// Data is transmitted in its native format — no f32 expansion.
    pub fn allreduce_typed(
        &self,
        data: &[u8],
        dtype: ReduceDtype,
    ) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized_dtype("allreduce_typed", data, dtype)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("allreduce_typed")?;
        ring_allreduce_op_native(
            data,
            &self.ranks,
            self.local_rank,
            transport,
            ReduceOp::Sum,
            dtype,
        )
    }

    /// In-place allreduce: reduces data across all ranks directly in the provided buffer.
    ///
    /// Uses ring allreduce with slice-based sendrecv for zero-copy operation.
    /// The buffer is modified in-place with the reduced result.
    pub fn allreduce_in_place(
        &self,
        data: &mut [u8],
        dtype: ReduceDtype,
    ) -> Result<(), DistributedError> {
        let len = data.len();
        if len == 0 {
            return Ok(());
        }
        let elem_size = match dtype {
            ReduceDtype::F32 => 4,
            ReduceDtype::F16 | ReduceDtype::Bf16 => 2,
        };
        if len % elem_size != 0 {
            return Err(DistributedError::Protocol(format!(
                "allreduce_in_place: data length ({len}) must be a multiple of {elem_size} for {:?}",
                dtype
            )));
        }
        if self.ranks.len() <= 1 {
            return Ok(());
        }
        let transport = self.require_transport("allreduce_in_place")?;
        let n = self.ranks.len();
        let total_elems = len / elem_size;
        let elems_per_chunk = total_elems.div_ceil(n);
        let chunk_size = elems_per_chunk * elem_size;

        // Find our position in the ring
        let my_pos = self
            .ranks
            .iter()
            .position(|&r| r == self.local_rank)
            .unwrap();
        let left = self.ranks[(my_pos + n - 1) % n];
        let right = self.ranks[(my_pos + 1) % n];

        let mut staging = vec![0u8; chunk_size];

        // Phase 1: reduce-scatter
        for step in 0..(n - 1) {
            let send_idx = (my_pos + n - step) % n;
            let recv_idx = (my_pos + n - step - 1) % n;

            let send_offset = send_idx * chunk_size;
            let recv_offset = recv_idx * chunk_size;
            let send_len = chunk_size.min(len.saturating_sub(send_offset));
            let recv_len = chunk_size.min(len.saturating_sub(recv_offset));

            if send_len == 0 || recv_len == 0 {
                continue;
            }

            // Send our chunk, recv peer's chunk into staging
            let send_slice = &data[send_offset..send_offset + send_len];
            transport.sendrecv_into(send_slice, right, &mut staging[..recv_len], left)?;

            // Add staging into local recv chunk (element-wise by dtype)
            reduce_inplace(
                &mut data[recv_offset..recv_offset + recv_len],
                &staging[..recv_len],
                ReduceOp::Sum,
                dtype,
            );
        }

        // Phase 2: allgather
        for step in 0..(n - 1) {
            let send_idx = (my_pos + n - step + 1) % n;
            let recv_idx = (my_pos + n - step) % n;

            let send_offset = send_idx * chunk_size;
            let recv_offset = recv_idx * chunk_size;
            let send_len = chunk_size.min(len.saturating_sub(send_offset));
            let recv_len = chunk_size.min(len.saturating_sub(recv_offset));

            if send_len == 0 || recv_len == 0 {
                continue;
            }

            // Copy send chunk to staging to avoid borrow conflict
            staging[..send_len].copy_from_slice(&data[send_offset..send_offset + send_len]);
            transport.sendrecv_into(
                &staging[..send_len],
                right,
                &mut data[recv_offset..recv_offset + recv_len],
                left,
            )?;
        }

        Ok(())
    }

    /// All-gather: gather data from all ranks into every rank.
    ///
    /// With transport: ring allgather — each rank circulates its chunk around
    /// the ring in N-1 rounds until every rank has all chunks.
    ///
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn allgather(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allgather", data)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("allgather")?;
        if self.ranks.len() <= 2 || data.len() < MESH_RING_THRESHOLD {
            mesh_allgather(data, &self.ranks, self.local_rank, transport)
        } else {
            ring_allgather(data, &self.ranks, self.local_rank, transport)
        }
    }

    /// Broadcast: root rank sends data to all other ranks.
    ///
    /// With transport: root sends to each peer; non-root ranks recv from root.
    ///
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn broadcast(&self, data: &[u8], root: u32) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("broadcast", data)?;
        self.validate_rank("broadcast(root)", root)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("broadcast")?;
        if self.local_rank == root {
            for &rank in &self.ranks {
                if rank != root {
                    transport.send(data, rank)?;
                }
            }
            Ok(data.to_vec())
        } else {
            transport.recv(root, data.len())
        }
    }

    /// Send data to a specific peer rank.
    ///
    /// With transport: posts a real RDMA send.
    /// Single-rank groups: no-op.
    /// Multi-rank groups without transport return an error.
    pub fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        Self::check_materialized("send", data)?;
        self.validate_rank("send", dst_rank)?;
        if self.ranks.len() <= 1 {
            return Ok(());
        }
        let transport = self.require_transport("send")?;
        transport.send(data, dst_rank)
    }

    /// Receive data from a specific peer rank.
    ///
    /// With transport: posts a real RDMA recv and waits for completion.
    /// Single-rank groups: returns zeroed buffer.
    /// Multi-rank groups without transport return an error.
    pub fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        if len == 0 {
            return Err(DistributedError::NotMaterialized(
                "recv: requested zero-length buffer".to_string(),
            ));
        }
        self.validate_rank("recv", src_rank)?;
        if self.ranks.len() <= 1 {
            return Ok(vec![0u8; len]);
        }
        let transport = self.require_transport("recv")?;
        transport.recv(src_rank, len)
    }

    /// Send data to `dst_rank` and receive `recv_len` bytes from `src_rank`
    /// in a single atomic operation.
    ///
    /// With transport: delegates to `RdmaTransport::sendrecv()`.
    /// Single-rank groups: returns zeroed buffer.
    /// Multi-rank groups without transport return an error.
    pub fn sendrecv(
        &self,
        send_data: &[u8],
        dst_rank: u32,
        recv_len: usize,
        src_rank: u32,
    ) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("sendrecv", send_data)?;
        if recv_len == 0 {
            return Err(DistributedError::NotMaterialized(
                "sendrecv: requested zero-length recv buffer".to_string(),
            ));
        }
        self.validate_rank("sendrecv(dst)", dst_rank)?;
        self.validate_rank("sendrecv(src)", src_rank)?;
        if self.ranks.len() <= 1 {
            return Ok(vec![0u8; recv_len]);
        }
        let transport = self.require_transport("sendrecv")?;
        transport.sendrecv(send_data, dst_rank, recv_len, src_rank)
    }

    /// All-to-all: each rank sends a distinct chunk to every other rank.
    ///
    /// With transport: pairwise exchange — data is split into N equal chunks,
    /// chunk[i] is sent to rank i, and chunk from rank i is received.
    ///
    /// Data length must be divisible by the group size.
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn all_to_all(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("all_to_all", data)?;
        let n = self.ranks.len();
        if n > 1 && data.len() % n != 0 {
            return Err(DistributedError::Transport(format!(
                "all_to_all: data length ({}) must be divisible by group size ({n})",
                data.len()
            )));
        }
        if n <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("all_to_all")?;
        pairwise_all_to_all(data, &self.ranks, self.local_rank, transport)
    }

    // ─── Array-level collective operations ───
    // These wrap the byte-level collectives with Array semantics,
    // leveraging Apple Silicon UMA for direct CPU access to Metal buffers.

    /// Array-level all-reduce sum.
    ///
    /// On Apple Silicon UMA, Metal buffer contents are CPU-accessible,
    /// so we extract bytes from the array, perform the ring allreduce,
    /// and reconstruct the array.
    ///
    /// Single-rank groups return the input array unchanged.
    pub fn allreduce_sum(
        &self,
        input: &rmlx_core::array::Array,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<rmlx_core::array::Array, DistributedError> {
        if self.ranks.len() <= 1 {
            // Identity: return a zero-copy view of the same buffer.
            return Ok(input.view(
                input.shape().to_vec(),
                input.strides().to_vec(),
                input.offset(),
            ));
        }

        // Extract raw bytes from the Metal buffer (UMA: CPU-accessible).
        let bytes = input.to_bytes();

        // Perform the byte-level ring allreduce.
        let result_bytes = self.allreduce(bytes)?;

        // Reconstruct the array from the reduced bytes.
        Ok(rmlx_core::array::Array::from_bytes(
            device,
            &result_bytes,
            input.shape().to_vec(),
            input.dtype(),
        ))
    }

    /// Array-level all-gather.
    ///
    /// Gathers arrays from all ranks into a single concatenated array.
    /// The gathering is along the first dimension (outermost).
    ///
    /// Single-rank groups return the input array unchanged.
    pub fn allgather_array(
        &self,
        input: &rmlx_core::array::Array,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<rmlx_core::array::Array, DistributedError> {
        if self.ranks.len() <= 1 {
            // Identity: return a zero-copy view of the same buffer.
            return Ok(input.view(
                input.shape().to_vec(),
                input.strides().to_vec(),
                input.offset(),
            ));
        }

        // Extract raw bytes from the Metal buffer (UMA: CPU-accessible).
        let bytes = input.to_bytes();

        // Perform the byte-level ring allgather.
        let result_bytes = self.allgather(bytes)?;

        // Build the gathered shape: [world_size * dim0, ...rest_dims].
        let input_shape = input.shape();
        let mut gathered_shape = input_shape.to_vec();
        gathered_shape[0] *= self.ranks.len();

        // Reconstruct the array from the gathered bytes.
        Ok(rmlx_core::array::Array::from_bytes(
            device,
            &result_bytes,
            gathered_shape,
            input.dtype(),
        ))
    }

    /// Reduce-scatter: reduce data across all ranks, then scatter chunks.
    ///
    /// The input data is divided into N equal chunks (one per rank). Each
    /// chunk is reduced (summed) across all ranks, and rank i receives chunk i.
    /// The output has size `data.len() / group_size`.
    ///
    /// With transport: ring reduce-scatter — N-1 rounds of send/recv/accumulate.
    ///
    /// Data length must be divisible by group size and by 4 (f32 granularity).
    /// Single-rank groups return input unchanged (identity).
    /// Multi-rank groups without transport return an error.
    pub fn reduce_scatter(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("reduce_scatter", data)?;
        let n = self.ranks.len();
        if n <= 1 {
            return Ok(data.to_vec());
        }
        if data.len() % n != 0 {
            return Err(DistributedError::Protocol(format!(
                "reduce_scatter: data length ({}) must be divisible by group size ({n})",
                data.len()
            )));
        }
        if data.len() % 4 != 0 {
            return Err(DistributedError::Protocol(format!(
                "reduce_scatter: data length ({}) must be a multiple of 4 (f32 element size)",
                data.len()
            )));
        }
        let transport = self.require_transport("reduce_scatter")?;
        if self.ranks.len() <= 2 || data.len() < MESH_RING_THRESHOLD {
            mesh_reduce_scatter(data, &self.ranks, self.local_rank, transport)
        } else {
            ring_reduce_scatter(data, &self.ranks, self.local_rank, transport)
        }
    }

    /// Barrier: block until all ranks in the group have reached this point.
    ///
    /// Uses a ring-based sendrecv pattern: each rank sends a 1-byte token
    /// to its right neighbor and receives from its left, for N-1 rounds.
    ///
    /// Single-rank groups return immediately.
    /// Multi-rank groups without transport return an error.
    pub fn barrier(&self) -> Result<(), DistributedError> {
        let n = self.ranks.len();
        if n <= 1 {
            return Ok(());
        }
        let transport = self.require_transport("barrier")?;

        let my_idx = self
            .ranks
            .iter()
            .position(|&r| r == self.local_rank)
            .ok_or_else(|| {
                DistributedError::Transport(format!(
                    "local_rank {} not found in ranks {:?}",
                    self.local_rank, self.ranks
                ))
            })?;
        let right = self.ranks[(my_idx + 1) % n];
        let left = self.ranks[(my_idx + n - 1) % n];

        let token = [0u8; 1];
        for _ in 0..(n - 1) {
            transport.sendrecv(&token, right, 1, left)?;
        }
        Ok(())
    }

    /// Internal helper: validate data is materialized before a collective.
    ///
    /// Returns `DistributedError::NotMaterialized` if data is invalid.
    fn check_materialized(_op_name: &str, data: &[u8]) -> Result<(), DistributedError> {
        let shapes = [(data.len(), data.len())];
        ensure_materialized(&shapes)
    }

    /// Dtype-aware materialization check for allreduce_op.
    ///
    /// For f16/bf16, requires 2-byte alignment instead of 4-byte.
    fn check_materialized_dtype(
        op_name: &str,
        data: &[u8],
        dtype: ReduceDtype,
    ) -> Result<(), DistributedError> {
        if data.is_empty() {
            return Err(DistributedError::NotMaterialized(format!(
                "{op_name}: data is empty"
            )));
        }
        let elem_size = match dtype {
            ReduceDtype::F32 => 4,
            ReduceDtype::F16 | ReduceDtype::Bf16 => 2,
        };
        if data.len() % elem_size != 0 {
            return Err(DistributedError::NotMaterialized(format!(
                "{op_name}: data length ({}) not aligned to {elem_size} bytes for {:?}",
                data.len(),
                dtype
            )));
        }
        Ok(())
    }

    /// Validate that a rank is within the group.
    fn validate_rank(&self, op_name: &str, rank: u32) -> Result<(), DistributedError> {
        if rank >= self.world_size || !self.ranks.contains(&rank) {
            return Err(DistributedError::Transport(format!(
                "{op_name}: rank {rank} not in group (world_size={}, ranks={:?})",
                self.world_size, self.ranks
            )));
        }
        Ok(())
    }

    /// Require a transport for multi-rank operations.
    ///
    /// Returns an error if this is a multi-rank group but no transport is attached.
    fn require_transport(
        &self,
        op_name: &str,
    ) -> Result<&Arc<dyn RdmaTransport>, DistributedError> {
        self.transport.as_ref().ok_or_else(|| {
            DistributedError::Transport(format!(
                "{op_name}: multi-rank group (size={}) requires transport, but none is attached",
                self.ranks.len()
            ))
        })
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Group(rank={}, size={}, ranks={:?}, transport={})",
            self.local_rank,
            self.ranks.len(),
            self.ranks,
            if self.transport.is_some() {
                "rdma"
            } else {
                "stub"
            }
        )
    }
}

// ─── Collective algorithms ───

/// Ring allreduce: sum data across all ranks using a ring topology.
///
/// Phase 1 (reduce-scatter): Each rank sends one chunk to its right neighbor
/// and accumulates the received chunk from its left neighbor. After N-1 rounds,
/// each rank holds the fully-reduced value for one chunk.
///
/// Phase 2 (allgather): Each rank circulates its reduced chunk around the ring
/// so every rank ends up with the full reduced result.
fn ring_allreduce(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    // allreduce operates on f32 elements — data must be 4-byte aligned
    if data.len() % 4 != 0 {
        return Err(DistributedError::Protocol(format!(
            "allreduce: data length ({}) must be a multiple of 4 (f32 element size)",
            data.len()
        )));
    }

    // Find our index in the sorted rank list
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;
    let right = ranks[(my_idx + 1) % n];
    let left = ranks[(my_idx + n - 1) % n];

    // Split data into N equal chunks (pad if needed).
    // Round up to f32 element boundary (4 bytes) so reduction never reads partial elements.
    let chunk_size = data.len().div_ceil(n);
    let chunk_size = chunk_size.div_ceil(4) * 4;
    let mut buf = data.to_vec();
    buf.resize(chunk_size * n, 0);

    // Phase 1: reduce-scatter (N-1 rounds)
    // Use sendrecv to avoid deadlock: send chunk to right, recv chunk from left.
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        // Accumulate: interpret as f32 and sum
        let recv_start = recv_chunk_idx * chunk_size;
        add_f32_inplace(&mut buf[recv_start..recv_start + chunk_size], &received);
    }

    // Phase 2: allgather (N-1 rounds)
    // Use sendrecv to avoid deadlock: send chunk to right, recv chunk from left.
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + 1 + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        buf[recv_start..recv_start + chunk_size].copy_from_slice(&received);
    }

    buf.truncate(data.len());
    Ok(buf)
}

/// Ring allgather: gather data from all ranks.
fn ring_allgather(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;
    let right = ranks[(my_idx + 1) % n];
    let left = ranks[(my_idx + n - 1) % n];

    let chunk_size = data.len();
    let mut result = vec![0u8; chunk_size * n];
    // Place our data at our index
    result[my_idx * chunk_size..(my_idx + 1) * chunk_size].copy_from_slice(data);

    // N-1 rounds: each round, send the chunk we just received to right,
    // receive a new chunk from left. Use sendrecv to avoid deadlock.
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;

        let received = transport.sendrecv(
            &result[send_start..send_start + chunk_size],
            right,
            chunk_size,
            left,
        )?;

        let recv_start = recv_chunk_idx * chunk_size;
        result[recv_start..recv_start + chunk_size].copy_from_slice(&received);
    }

    Ok(result)
}

/// Pairwise all-to-all exchange.
fn pairwise_all_to_all(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    let chunk_size = data.len() / n;
    if chunk_size == 0 {
        return Err(DistributedError::NotMaterialized(
            "all_to_all: data too small to split among ranks".to_string(),
        ));
    }

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;

    let mut result = vec![0u8; data.len()];
    // Our own chunk stays in place
    result[my_idx * chunk_size..(my_idx + 1) * chunk_size]
        .copy_from_slice(&data[my_idx * chunk_size..(my_idx + 1) * chunk_size]);

    // Exchange with each peer using sendrecv
    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_rank == local_rank {
            continue;
        }
        // Send chunk destined for this peer, receive chunk from this peer
        let send_start = peer_idx * chunk_size;
        let received = transport.sendrecv(
            &data[send_start..send_start + chunk_size],
            peer_rank,
            chunk_size,
            peer_rank,
        )?;
        result[peer_idx * chunk_size..(peer_idx + 1) * chunk_size].copy_from_slice(&received);
    }

    Ok(result)
}

/// Element-wise f32 addition: dst[i] += src[i].
/// Operates on raw byte slices interpreted as f32 arrays.
fn add_f32_inplace(dst: &mut [u8], src: &[u8]) {
    let count = dst.len().min(src.len()) / 4;
    for i in 0..count {
        let offset = i * 4;
        let a = f32::from_ne_bytes([
            dst[offset],
            dst[offset + 1],
            dst[offset + 2],
            dst[offset + 3],
        ]);
        let b = f32::from_ne_bytes([
            src[offset],
            src[offset + 1],
            src[offset + 2],
            src[offset + 3],
        ]);
        let sum = a + b;
        dst[offset..offset + 4].copy_from_slice(&sum.to_ne_bytes());
    }
}

/// Element-wise f32 reduction with configurable op: dst[i] = op(dst[i], src[i]).
fn reduce_f32_inplace(dst: &mut [u8], src: &[u8], op: ReduceOp) {
    let count = dst.len().min(src.len()) / 4;
    for i in 0..count {
        let offset = i * 4;
        let a = f32::from_ne_bytes([
            dst[offset],
            dst[offset + 1],
            dst[offset + 2],
            dst[offset + 3],
        ]);
        let b = f32::from_ne_bytes([
            src[offset],
            src[offset + 1],
            src[offset + 2],
            src[offset + 3],
        ]);
        let result = match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        };
        dst[offset..offset + 4].copy_from_slice(&result.to_ne_bytes());
    }
}

/// Element-wise f16 addition in-place: dst[i] += src[i].
/// Converts each pair to f32 for the addition, then stores back as f16.
fn add_f16_inplace(dst: &mut [u8], src: &[u8]) {
    let count = dst.len().min(src.len()) / 2;
    for i in 0..count {
        let offset = i * 2;
        let a = f16_to_f32(u16::from_ne_bytes([dst[offset], dst[offset + 1]]));
        let b = f16_to_f32(u16::from_ne_bytes([src[offset], src[offset + 1]]));
        let sum = f32_to_f16(a + b);
        dst[offset..offset + 2].copy_from_slice(&sum.to_ne_bytes());
    }
}

/// Element-wise bf16 addition in-place: dst[i] += src[i].
fn add_bf16_inplace(dst: &mut [u8], src: &[u8]) {
    let count = dst.len().min(src.len()) / 2;
    for i in 0..count {
        let offset = i * 2;
        let a = bf16_to_f32(u16::from_ne_bytes([dst[offset], dst[offset + 1]]));
        let b = bf16_to_f32(u16::from_ne_bytes([src[offset], src[offset + 1]]));
        let sum = f32_to_bf16(a + b);
        dst[offset..offset + 2].copy_from_slice(&sum.to_ne_bytes());
    }
}

/// Dispatch element-wise reduction by dtype.
fn reduce_inplace(dst: &mut [u8], src: &[u8], op: ReduceOp, dtype: ReduceDtype) {
    match dtype {
        ReduceDtype::F32 => reduce_f32_inplace(dst, src, op),
        ReduceDtype::F16 => match op {
            ReduceOp::Sum => add_f16_inplace(dst, src),
            _ => reduce_f16_inplace(dst, src, op),
        },
        ReduceDtype::Bf16 => match op {
            ReduceOp::Sum => add_bf16_inplace(dst, src),
            _ => reduce_bf16_inplace(dst, src, op),
        },
    }
}

fn reduce_f16_inplace(dst: &mut [u8], src: &[u8], op: ReduceOp) {
    let count = dst.len().min(src.len()) / 2;
    for i in 0..count {
        let offset = i * 2;
        let a = f16_to_f32(u16::from_ne_bytes([dst[offset], dst[offset + 1]]));
        let b = f16_to_f32(u16::from_ne_bytes([src[offset], src[offset + 1]]));
        let result = match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        };
        let bits = f32_to_f16(result);
        dst[offset..offset + 2].copy_from_slice(&bits.to_ne_bytes());
    }
}

fn reduce_bf16_inplace(dst: &mut [u8], src: &[u8], op: ReduceOp) {
    let count = dst.len().min(src.len()) / 2;
    for i in 0..count {
        let offset = i * 2;
        let a = bf16_to_f32(u16::from_ne_bytes([dst[offset], dst[offset + 1]]));
        let b = bf16_to_f32(u16::from_ne_bytes([src[offset], src[offset + 1]]));
        let result = match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        };
        let bits = f32_to_bf16(result);
        dst[offset..offset + 2].copy_from_slice(&bits.to_ne_bytes());
    }
}

/// Ring reduce-scatter: reduce and scatter data across all ranks.
fn ring_reduce_scatter(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;
    let right = ranks[(my_idx + 1) % n];
    let left = ranks[(my_idx + n - 1) % n];

    let chunk_size = data.len() / n;
    let mut buf = data.to_vec();

    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        add_f32_inplace(&mut buf[recv_start..recv_start + chunk_size], &received);
    }

    let chunk_start = my_idx * chunk_size;
    Ok(buf[chunk_start..chunk_start + chunk_size].to_vec())
}

/// Mesh allreduce: each rank exchanges full data with every other rank,
/// then reduces locally. O(1) communication steps but O(N * data_size) volume.
/// Best for small messages where latency dominates.
fn mesh_allreduce(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {} not found in ranks {:?}",
            local_rank, ranks
        ))
    })?;

    // Ensure f32 alignment
    let aligned_len = (data.len() + 3) & !3;
    let mut buf = data.to_vec();
    buf.resize(aligned_len, 0);

    // Exchange with every peer and reduce in-place
    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_idx == my_idx {
            continue;
        }
        let received = transport.sendrecv(&buf, peer_rank, aligned_len, peer_rank)?;
        add_f32_inplace(&mut buf, &received);
    }

    buf.truncate(data.len());
    Ok(buf)
}

/// Mesh allreduce with configurable ReduceOp (f32 only).
fn mesh_allreduce_op_f32(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
) -> Result<Vec<u8>, DistributedError> {
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {} not found in ranks {:?}",
            local_rank, ranks
        ))
    })?;

    let aligned_len = (data.len() + 3) & !3;
    let mut buf = data.to_vec();
    buf.resize(aligned_len, 0);

    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_idx == my_idx {
            continue;
        }
        let received = transport.sendrecv(&buf, peer_rank, aligned_len, peer_rank)?;
        reduce_f32_inplace(&mut buf, &received, op);
    }

    buf.truncate(data.len());
    Ok(buf)
}

/// Mesh allreduce with native dtype support (f16/bf16/f32 without expansion).
fn mesh_allreduce_op_native(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
    dtype: ReduceDtype,
) -> Result<Vec<u8>, DistributedError> {
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {} not found in ranks {:?}",
            local_rank, ranks
        ))
    })?;

    let elem_size = match dtype {
        ReduceDtype::F32 => 4,
        ReduceDtype::F16 | ReduceDtype::Bf16 => 2,
    };
    let align_mask = elem_size - 1;
    let aligned_len = (data.len() + align_mask) & !align_mask;
    let mut buf = data.to_vec();
    buf.resize(aligned_len, 0);

    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_idx == my_idx {
            continue;
        }
        let received = transport.sendrecv(&buf, peer_rank, aligned_len, peer_rank)?;
        reduce_inplace(&mut buf, &received, op, dtype);
    }

    buf.truncate(data.len());
    Ok(buf)
}

/// Mesh allreduce with configurable ReduceOp and dtype support.
fn mesh_allreduce_op(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
    dtype: ReduceDtype,
) -> Result<Vec<u8>, DistributedError> {
    match dtype {
        ReduceDtype::F32 => mesh_allreduce_op_f32(data, ranks, local_rank, transport, op),
        ReduceDtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "mesh_allreduce_op(f16): data length ({}) must be a multiple of 2",
                    data.len()
                )));
            }
            mesh_allreduce_op_native(data, ranks, local_rank, transport, op, dtype)
        }
        ReduceDtype::Bf16 => {
            if data.len() % 2 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "mesh_allreduce_op(bf16): data length ({}) must be a multiple of 2",
                    data.len()
                )));
            }
            mesh_allreduce_op_native(data, ranks, local_rank, transport, op, dtype)
        }
    }
}

/// Mesh allgather: each rank broadcasts its data to all peers.
/// Result is concatenation of all ranks' data in rank order.
fn mesh_allgather(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {} not found in ranks {:?}",
            local_rank, ranks
        ))
    })?;

    let chunk_size = data.len();
    let mut result = vec![0u8; chunk_size * n];

    // Place own data
    result[my_idx * chunk_size..(my_idx + 1) * chunk_size].copy_from_slice(data);

    // Exchange with every peer
    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_idx == my_idx {
            continue;
        }
        let received = transport.sendrecv(data, peer_rank, chunk_size, peer_rank)?;
        result[peer_idx * chunk_size..(peer_idx + 1) * chunk_size]
            .copy_from_slice(&received[..chunk_size]);
    }

    Ok(result)
}

/// Mesh reduce-scatter: mesh allreduce then extract own chunk.
fn mesh_reduce_scatter(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {} not found in ranks {:?}",
            local_rank, ranks
        ))
    })?;

    let reduced = mesh_allreduce(data, ranks, local_rank, transport)?;
    let chunk_size = reduced.len().div_ceil(n);
    let start = my_idx * chunk_size;
    let end = (start + chunk_size).min(reduced.len());
    Ok(reduced[start..end].to_vec())
}

/// Ring allreduce with configurable reduction operation and dtype.
fn ring_allreduce_op(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
    dtype: ReduceDtype,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    match dtype {
        ReduceDtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "allreduce_op(f32): data length ({}) must be a multiple of 4",
                    data.len()
                )));
            }
            ring_allreduce_op_f32(data, ranks, local_rank, transport, op)
        }
        ReduceDtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "allreduce_op(f16): data length ({}) must be a multiple of 2",
                    data.len()
                )));
            }
            ring_allreduce_op_native(data, ranks, local_rank, transport, op, dtype)
        }
        ReduceDtype::Bf16 => {
            if data.len() % 2 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "allreduce_op(bf16): data length ({}) must be a multiple of 2",
                    data.len()
                )));
            }
            ring_allreduce_op_native(data, ranks, local_rank, transport, op, dtype)
        }
    }
}

/// Ring allreduce on f32 data with configurable reduce op.
fn ring_allreduce_op_f32(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;
    let right = ranks[(my_idx + 1) % n];
    let left = ranks[(my_idx + n - 1) % n];

    // Round up to f32 element boundary (4 bytes) so reduction never reads partial elements.
    let chunk_size = data.len().div_ceil(n);
    let chunk_size = chunk_size.div_ceil(4) * 4;
    let mut buf = data.to_vec();
    buf.resize(chunk_size * n, 0);

    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        reduce_f32_inplace(&mut buf[recv_start..recv_start + chunk_size], &received, op);
    }

    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + 1 + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        buf[recv_start..recv_start + chunk_size].copy_from_slice(&received);
    }

    buf.truncate(data.len());
    Ok(buf)
}

/// Ring allreduce on native-dtype data (f16/bf16/f32) with configurable reduce op.
/// Sends data in its native format over RDMA — no f32 expansion.
fn ring_allreduce_op_native(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    op: ReduceOp,
    dtype: ReduceDtype,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    let elem_size = match dtype {
        ReduceDtype::F32 => 4,
        ReduceDtype::F16 | ReduceDtype::Bf16 => 2,
    };

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;
    let right = ranks[(my_idx + 1) % n];
    let left = ranks[(my_idx + n - 1) % n];

    // Chunk size must be aligned to element size
    let total_elems = data.len() / elem_size;
    let elems_per_chunk = total_elems.div_ceil(n);
    let chunk_size = elems_per_chunk * elem_size;

    // Pad data to be evenly divisible
    let padded_len = chunk_size * n;
    let mut buf = vec![0u8; padded_len];
    buf[..data.len()].copy_from_slice(data);

    // Phase 1: Reduce-scatter
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        reduce_inplace(
            &mut buf[recv_start..recv_start + chunk_size],
            &received,
            op,
            dtype,
        );
    }

    // Phase 2: Allgather
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + 1 + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        let received = transport.sendrecv(&buf[send_start..send_end], right, chunk_size, left)?;

        let recv_start = recv_chunk_idx * chunk_size;
        buf[recv_start..recv_start + chunk_size].copy_from_slice(&received);
    }

    buf.truncate(data.len());
    Ok(buf)
}

// ─── Half-precision conversion utilities ───

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // f16 subnormal: value = 2^(-14) * (mantissa / 1024)
            // Normalize: find leading 1-bit position, then form normal f32.
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            // m now has bit 10 set (the implicit leading 1). Strip it.
            // Exponent: the subnormal exponent is -14, minus the shift count,
            // plus 1 because we found the leading 1 at position 10.
            let f32_exp = (127 - 14 - e) as u32;
            let f32_mantissa = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
        }
    } else if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa)
    } else {
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
    }
}

/// Convert f32 to IEEE 754 half-precision (f16) bits.
///
/// Handles NaN payload preservation: if the f32 is NaN, the resulting f16 is
/// also NaN with as many payload bits as fit. If truncation would zero out the
/// mantissa (turning NaN into Inf), the quiet-NaN bit is forced on.
///
/// Handles subnormals: f32 values in the f16 subnormal range
/// (2^-24 <= |x| < 2^-14) are converted to f16 subnormal representation
/// rather than being flushed to zero.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        // Inf or NaN
        let mut f16_mantissa = (mantissa >> 13) as u16;
        if mantissa != 0 && f16_mantissa == 0 {
            // NaN payload was lost by truncation — force quiet NaN bit
            f16_mantissa = 0x200; // quiet NaN
        }
        (sign << 15) | (0x1F << 10) | f16_mantissa
    } else if exponent > 127 + 15 {
        // Overflow to Inf
        (sign << 15) | (0x1F << 10)
    } else if exponent < 127 - 24 {
        // Too small even for f16 subnormal — flush to zero
        sign << 15
    } else if exponent < 127 - 14 {
        // f16 subnormal range: denormalize the value
        // Add the implicit leading 1 to the mantissa, then shift right
        let shift = (127 - 14) - exponent; // 1..10
        let full_mantissa = mantissa | 0x80_0000; // add implicit 1 bit
        let f16_mantissa = (full_mantissa >> (13 + shift)) as u16;
        (sign << 15) | f16_mantissa
    } else {
        let f16_exp = ((exponent - 127 + 15) as u16) & 0x1F;
        let f16_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | (f16_exp << 10) | f16_mantissa
    }
}

/// Convert bfloat16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ─── Tree allreduce ───

/// Default threshold in bytes for choosing tree vs ring allreduce.
/// For data smaller than this, tree allreduce (lower latency) is preferred.
/// For data at or above this, ring allreduce (higher bandwidth) is preferred.
pub const TREE_ALLREDUCE_THRESHOLD: usize = 1024 * 1024; // 1 MB

/// JACCL-compatible threshold: mesh → ring fallback for large messages.
/// Groups with size > 2 and data >= this threshold use ring instead of mesh.
pub const MESH_RING_THRESHOLD: usize = 8 * 1024 * 1024; // 8 MB

/// Lower threshold for bf16 (JACCL: 65536 elements = 128KB).
pub const MESH_RING_THRESHOLD_BF16: usize = 65536 * 2; // 128 KB

/// Tree allreduce on f32 data (byte slices).
///
/// For small tensors (<1MB default), tree allreduce has lower latency than ring
/// because it completes in O(log N) steps instead of O(N).
///
/// Phase 1 (reduce): binary tree reduction to rank 0.
///   - At each round, the active set is halved. Ranks in the upper half send
///     their data to the corresponding rank in the lower half, which accumulates.
///     Phase 2 (broadcast): rank 0 broadcasts the result back down the tree.
///   - Reverses the tree: the lower-half rank sends to the upper-half rank.
///
/// Operates on raw byte slices interpreted as f32 arrays.
///
/// # Arguments
/// * `data` - raw byte slice (f32 elements in native endian), modified in-place.
/// * `ranks` - sorted rank list for this group.
/// * `local_rank` - this node's rank.
/// * `transport` - RDMA transport for send/recv.
///
/// # Returns
/// The allreduced data as a new `Vec<u8>`.
pub fn tree_allreduce(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let n = ranks.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }

    if data.len() % 4 != 0 {
        return Err(DistributedError::Protocol(format!(
            "tree_allreduce: data length ({}) must be a multiple of 4 (f32 element size)",
            data.len()
        )));
    }

    let my_idx = ranks.iter().position(|&r| r == local_rank).ok_or_else(|| {
        DistributedError::Transport(format!(
            "local_rank {local_rank} not found in ranks {:?}",
            ranks
        ))
    })?;

    let mut buf = data.to_vec();

    // Phase 1: tree reduce to index 0
    // stride doubles each round: 1, 2, 4, ...
    let mut stride = 1;
    while stride < n {
        if my_idx % (2 * stride) == 0 {
            // Receiver: receive from my_idx + stride (if it exists)
            let sender_idx = my_idx + stride;
            if sender_idx < n {
                let received = transport.recv(ranks[sender_idx], buf.len())?;
                add_f32_inplace(&mut buf, &received);
            }
        } else if my_idx % (2 * stride) == stride {
            // Sender: send to my_idx - stride
            let receiver_idx = my_idx - stride;
            transport.send(&buf, ranks[receiver_idx])?;
        }
        // Ranks that are neither sender nor receiver at this level are idle.
        stride *= 2;
    }

    // Phase 2: tree broadcast from index 0
    // stride halves each round, starting from the largest power of 2 < n
    let mut stride = 1;
    while stride * 2 < n {
        stride *= 2;
    }
    while stride >= 1 {
        if my_idx % (2 * stride) == 0 {
            // Sender: send to my_idx + stride (if it exists)
            let receiver_idx = my_idx + stride;
            if receiver_idx < n {
                transport.send(&buf, ranks[receiver_idx])?;
            }
        } else if my_idx % (2 * stride) == stride {
            // Receiver: receive from my_idx - stride
            let sender_idx = my_idx - stride;
            let received = transport.recv(ranks[sender_idx], buf.len())?;
            buf.copy_from_slice(&received);
        }
        stride /= 2;
    }

    Ok(buf)
}

/// Tree allreduce with typed elements.
///
/// Works with any type that can be converted to/from f32 byte representation.
/// The data is treated as raw bytes internally — the type parameter controls
/// element alignment validation.
pub fn tree_allreduce_typed<T: Copy + Default>(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<Vec<u8>, DistributedError> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 || data.len() % elem_size != 0 {
        return Err(DistributedError::Protocol(format!(
            "tree_allreduce_typed: data length ({}) must be a multiple of element size ({})",
            data.len(),
            elem_size
        )));
    }
    // Delegate to the f32 tree allreduce (reduction is always f32 sum)
    tree_allreduce(data, ranks, local_rank, transport)
}

/// Auto-selecting allreduce: picks tree or ring based on data size.
///
/// - Data < `TREE_ALLREDUCE_THRESHOLD` (1MB): uses tree allreduce (lower latency).
/// - Data >= `TREE_ALLREDUCE_THRESHOLD`: uses ring allreduce (higher bandwidth).
///
/// Returns `AllreduceAlgorithm` indicating which algorithm was used, along with
/// the result.
pub fn allreduce_auto(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
) -> Result<(Vec<u8>, AllreduceAlgorithm), DistributedError> {
    allreduce_auto_with_threshold(data, ranks, local_rank, transport, TREE_ALLREDUCE_THRESHOLD)
}

/// Auto-selecting allreduce with a configurable threshold.
///
/// Three-tier dispatch:
/// 1. Large data + large group (>2 ranks, >= MESH_RING_THRESHOLD): ring (bandwidth optimal)
/// 2. Small data (< threshold): tree (log N steps, low latency)
/// 3. Otherwise (medium data or small group): mesh (O(1) steps)
pub fn allreduce_auto_with_threshold(
    data: &[u8],
    ranks: &[u32],
    local_rank: u32,
    transport: &Arc<dyn RdmaTransport>,
    threshold: usize,
) -> Result<(Vec<u8>, AllreduceAlgorithm), DistributedError> {
    let n = ranks.len();
    let data_len = data.len();

    if n > 2 && data_len >= MESH_RING_THRESHOLD {
        // Large data, large group: ring (bandwidth optimal)
        let result = ring_allreduce(data, ranks, local_rank, transport)?;
        Ok((result, AllreduceAlgorithm::Ring))
    } else if data_len < threshold {
        // Small data: tree (log N steps)
        let result = tree_allreduce(data, ranks, local_rank, transport)?;
        Ok((result, AllreduceAlgorithm::Tree))
    } else {
        // Medium data or small group: mesh
        let result = mesh_allreduce(data, ranks, local_rank, transport)?;
        Ok((result, AllreduceAlgorithm::Mesh))
    }
}

/// Which allreduce algorithm was selected by `allreduce_auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllreduceAlgorithm {
    /// Binary tree allreduce (lower latency, better for small data).
    Tree,
    /// Ring allreduce (higher bandwidth, better for large data).
    Ring,
    /// Mesh allreduce (O(1) steps, best for small messages where latency dominates).
    Mesh,
}

// ─── Topology-aware ring ordering ───
//
// NOTE: TopologyRing is currently **not connected** to any collective algorithm.
// Ring collectives (ring_allreduce, ring_allreduce_op, ring_allreduce_op_native)
// use `&self.ranks` directly, which is sorted numerically — not by network proximity.
//
// Future integration path:
//   1. In `Group::allreduce_op()` (and similar), call `TopologyRing::from_env(&self.ranks)`
//      to obtain a topology-ordered rank slice.
//   2. Pass the reordered slice to `ring_allreduce_op()` instead of `&self.ranks`.
//   3. This requires no algorithm changes — the ring collectives already use the
//      `ranks` parameter to derive left/right neighbors, so reordering the input
//      is sufficient to make them topology-aware.
//
// Blocked on: real multi-hop topology (current 2-node setup has no hop variance).

/// Topology-aware ring that reorders ranks based on inter-node hop counts.
///
/// Constructs a ring where consecutive ranks are as "close" as possible
/// in the network topology, minimizing total communication cost.
#[derive(Debug, Clone)]
pub struct TopologyRing {
    /// Ordered list of ranks forming the ring.
    pub order: Vec<u32>,
}

impl TopologyRing {
    /// Construct a topology-aware ring from a hop-count matrix.
    ///
    /// Uses greedy nearest-unvisited ordering: starting from rank 0 (or the
    /// first rank), repeatedly pick the closest unvisited rank.
    ///
    /// # Arguments
    /// * `hops` - NxN matrix where `hops[i][j]` is the hop count from rank i to rank j.
    ///   Must be square and have size matching the number of ranks.
    /// * `ranks` - The ranks to order (indices into the hop matrix).
    ///
    /// # Returns
    /// A `TopologyRing` with ranks ordered for minimal hop-distance ring.
    pub fn from_hops(hops: &[Vec<u32>], ranks: &[u32]) -> Result<Self, DistributedError> {
        let n = ranks.len();
        if n == 0 {
            return Ok(Self { order: Vec::new() });
        }
        if n == 1 {
            return Ok(Self {
                order: ranks.to_vec(),
            });
        }

        // Validate hop matrix dimensions
        for (i, row) in hops.iter().enumerate() {
            if row.len() != hops.len() {
                return Err(DistributedError::Config(format!(
                    "TopologyRing: hop matrix row {i} has length {}, expected {}",
                    row.len(),
                    hops.len()
                )));
            }
        }

        // Validate all ranks are valid indices into the hop matrix
        for &r in ranks {
            if (r as usize) >= hops.len() {
                return Err(DistributedError::Config(format!(
                    "TopologyRing: rank {r} out of bounds for hop matrix of size {}",
                    hops.len()
                )));
            }
        }

        // Greedy nearest-unvisited ordering
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);

        // Start from the first rank
        visited[0] = true;
        order.push(ranks[0]);

        for _ in 1..n {
            let current = *order.last().unwrap();
            let current_idx = current as usize;

            // Find nearest unvisited rank
            let mut best_rank_pos = None;
            let mut best_dist = u32::MAX;

            for (pos, &rank) in ranks.iter().enumerate() {
                if visited[pos] {
                    continue;
                }
                let dist = hops[current_idx][rank as usize];
                if dist < best_dist {
                    best_dist = dist;
                    best_rank_pos = Some(pos);
                }
            }

            if let Some(pos) = best_rank_pos {
                visited[pos] = true;
                order.push(ranks[pos]);
            }
        }

        Ok(Self { order })
    }

    /// Construct a topology-aware ring from the `RMLX_TOPOLOGY` environment variable.
    ///
    /// Expected format: JSON object with a "hops" key containing a 2D array:
    /// ```json
    /// {"hops": [[0,1,2,1], [1,0,1,2], [2,1,0,1], [1,2,1,0]]}
    /// ```
    ///
    /// If the environment variable is not set, falls back to sequential ordering.
    ///
    /// # Arguments
    /// * `ranks` - The ranks to order.
    pub fn from_env(ranks: &[u32]) -> Result<Self, DistributedError> {
        match std::env::var("RMLX_TOPOLOGY") {
            Ok(json_str) => {
                let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| {
                    DistributedError::Config(format!(
                        "TopologyRing: failed to parse RMLX_TOPOLOGY JSON: {e}"
                    ))
                })?;

                let hops_val = parsed.get("hops").ok_or_else(|| {
                    DistributedError::Config(
                        "TopologyRing: RMLX_TOPOLOGY JSON missing 'hops' key".to_string(),
                    )
                })?;

                let hops_array = hops_val.as_array().ok_or_else(|| {
                    DistributedError::Config(
                        "TopologyRing: 'hops' value is not an array".to_string(),
                    )
                })?;

                let hops: Vec<Vec<u32>> = hops_array
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        row.as_array()
                            .ok_or_else(|| {
                                DistributedError::Config(format!(
                                    "TopologyRing: hops row {i} is not an array"
                                ))
                            })?
                            .iter()
                            .enumerate()
                            .map(|(j, v)| {
                                v.as_u64()
                                    .ok_or_else(|| {
                                        DistributedError::Config(format!(
                                            "TopologyRing: hops[{i}][{j}] is not a number"
                                        ))
                                    })
                                    .map(|n| n as u32)
                            })
                            .collect::<Result<Vec<u32>, _>>()
                    })
                    .collect::<Result<Vec<Vec<u32>>, _>>()?;

                Self::from_hops(&hops, ranks)
            }
            Err(_) => {
                // Fallback: sequential ordering
                Ok(Self {
                    order: ranks.to_vec(),
                })
            }
        }
    }

    /// Return the ring ordering as a slice.
    pub fn as_slice(&self) -> &[u32] {
        &self.order
    }
}

/// Convert f32 to bfloat16 bits (truncation).
///
/// bf16 shares the same exponent range as f32, so overflow/subnormal concerns
/// don't apply. However, NaN payload bits in the lower 16 bits could be lost;
/// if truncation would zero the mantissa, force the quiet-NaN bit.
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let result = (bits >> 16) as u16;
    // Check if f32 is NaN (exponent=0xFF, mantissa!=0) but truncated bf16 looks like Inf
    let f32_exp = (bits >> 23) & 0xFF;
    let f32_mantissa = bits & 0x7F_FFFF;
    if f32_exp == 0xFF && f32_mantissa != 0 && (result & 0x7F) == 0 {
        // NaN payload was lost — force quiet NaN
        result | 0x40 // set quiet NaN bit in bf16 mantissa
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use super::*;

    // ─── Chunk rounding tests ───

    #[test]
    fn test_chunk_rounding_aligns_to_4_bytes() {
        // 7 f32 elements = 28 bytes, 3 ranks => chunk_size should be 12 (not 10)
        let n = 3usize;
        let data_len = 28usize; // 7 * 4 bytes
        let chunk_size = data_len.div_ceil(n);
        assert_eq!(chunk_size, 10); // naive: 10 bytes, not aligned
        let aligned = chunk_size.div_ceil(4) * 4;
        assert_eq!(aligned, 12); // aligned to 4 bytes
    }

    #[test]
    fn test_allreduce_non_divisible_size_single_rank() {
        // Single rank should return data unchanged regardless of size
        let group = Group::new(vec![0], 0, 1).unwrap();
        // 7 f32 elements = 28 bytes (not divisible by any rank count > 1)
        let data: Vec<u8> = (1..=7u32).flat_map(|v| (v as f32).to_ne_bytes()).collect();
        let result = group.allreduce(&data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_allreduce_op_non_divisible_f32_single_rank() {
        // 7 f32 elements through allreduce_op with single rank
        let group = Group::new(vec![0], 0, 1).unwrap();
        let data: Vec<u8> = (1..=7u32).flat_map(|v| (v as f32).to_ne_bytes()).collect();
        let result = group
            .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F32)
            .unwrap();
        assert_eq!(result, data);
    }

    // ─── f16 conversion edge case tests ───

    #[test]
    fn test_f16_nan_payload_survives_roundtrip() {
        // Create a f16 NaN with a specific payload
        // f16 NaN: sign=0, exp=0x1F, mantissa!=0
        let nan_payloads: Vec<u16> = vec![
            0x7C01, // quiet NaN, payload=1
            0x7E00, // quiet NaN, payload=0x200
            0x7DFF, // signaling NaN, payload=0x1FF
            0xFC01, // negative NaN
        ];

        for &original in &nan_payloads {
            let f32_val = f16_to_f32(original);
            assert!(
                f32_val.is_nan(),
                "f16 0x{:04X} should convert to f32 NaN",
                original
            );
            let roundtrip = f32_to_f16(f32_val);
            // The roundtrip must also be NaN (not Inf)
            let rt_exp = (roundtrip >> 10) & 0x1F;
            let rt_mantissa = roundtrip & 0x3FF;
            assert_eq!(
                rt_exp, 0x1F,
                "roundtrip of 0x{:04X} must have exp=0x1F",
                original
            );
            assert_ne!(
                rt_mantissa, 0,
                "roundtrip of 0x{:04X} must have non-zero mantissa (NaN, not Inf)",
                original
            );
        }
    }

    #[test]
    fn test_f16_nan_with_low_payload_preserved() {
        // A NaN whose payload is only in the lower 13 bits of f32 mantissa
        // (which get truncated to f16). The conversion must still produce NaN.
        let f32_nan = f32::from_bits(0x7F800001); // NaN with payload=1 (lowest bit)
        assert!(f32_nan.is_nan());
        let f16_bits = f32_to_f16(f32_nan);
        let exp = (f16_bits >> 10) & 0x1F;
        let mantissa = f16_bits & 0x3FF;
        assert_eq!(exp, 0x1F);
        assert_ne!(mantissa, 0, "NaN with small payload must not become Inf");
    }

    #[test]
    fn test_f16_subnormal_roundtrip() {
        // f16 subnormal: smallest positive subnormal = 2^-24 ≈ 5.96e-8
        let f16_subnormal: u16 = 0x0001; // smallest subnormal
        let f32_val = f16_to_f32(f16_subnormal);
        assert!(
            f32_val > 0.0 && f32_val < 6.2e-5,
            "subnormal should be tiny positive"
        );
        let roundtrip = f32_to_f16(f32_val);
        assert_eq!(
            roundtrip, f16_subnormal,
            "f16 subnormal should survive roundtrip"
        );

        // A larger f16 subnormal
        let f16_sub2: u16 = 0x0200; // subnormal with mantissa bit 9 set
        let f32_val2 = f16_to_f32(f16_sub2);
        let roundtrip2 = f32_to_f16(f32_val2);
        assert_eq!(
            roundtrip2, f16_sub2,
            "f16 subnormal 0x0200 should survive roundtrip"
        );
    }

    #[test]
    fn test_f16_inf_preserved() {
        let pos_inf: u16 = 0x7C00;
        let f32_val = f16_to_f32(pos_inf);
        assert!(f32_val.is_infinite() && f32_val > 0.0);
        let roundtrip = f32_to_f16(f32_val);
        assert_eq!(roundtrip, pos_inf);

        let neg_inf: u16 = 0xFC00;
        let f32_val = f16_to_f32(neg_inf);
        assert!(f32_val.is_infinite() && f32_val < 0.0);
        let roundtrip = f32_to_f16(f32_val);
        assert_eq!(roundtrip, neg_inf);
    }

    #[test]
    fn test_f16_normal_values_roundtrip() {
        // Test a few normal f16 values
        let test_values: Vec<u16> = vec![
            0x3C00, // 1.0
            0x4000, // 2.0
            0x3800, // 0.5
            0xC000, // -2.0
            0x0400, // smallest normal: 2^-14
        ];
        for &original in &test_values {
            let f32_val = f16_to_f32(original);
            let roundtrip = f32_to_f16(f32_val);
            assert_eq!(
                roundtrip, original,
                "f16 0x{:04X} (={}) should roundtrip exactly",
                original, f32_val
            );
        }
    }

    // ─── bf16 tests ───

    #[test]
    fn test_bf16_basic_roundtrip() {
        let test_values: Vec<f32> = vec![1.0, -1.0, 0.0, 0.5, 100.0, -0.125];
        for &val in &test_values {
            let bf16_bits = f32_to_bf16(val);
            let roundtrip = bf16_to_f32(bf16_bits);
            assert_eq!(
                roundtrip, val,
                "bf16 roundtrip of {} should be exact for simple values",
                val
            );
        }
    }

    #[test]
    fn test_bf16_nan_preserved() {
        let f32_nan = f32::from_bits(0x7F800001); // NaN with small payload
        assert!(f32_nan.is_nan());
        let bf16_bits = f32_to_bf16(f32_nan);
        let back = bf16_to_f32(bf16_bits);
        assert!(back.is_nan(), "bf16 roundtrip of NaN must remain NaN");
        // Verify it's not Inf
        assert_ne!(bf16_bits & 0x7FFF, 0x7F80, "bf16 NaN must not become Inf");
    }

    #[test]
    fn test_bf16_allreduce_op_single_rank() {
        let group = Group::new(vec![0], 0, 1).unwrap();
        // 3 bf16 elements = 6 bytes
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_ne_bytes())
            .collect();
        let result = group
            .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::Bf16)
            .unwrap();
        assert_eq!(result, data);
    }

    // ─── allreduce_op f16 single-rank identity tests ───

    #[test]
    fn test_f16_allreduce_op_single_rank() {
        let group = Group::new(vec![0], 0, 1).unwrap();
        // 5 f16 elements = 10 bytes (odd count, not divisible by 4)
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_ne_bytes())
            .collect();
        let result = group
            .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
            .unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_f16_allreduce_op_nan_single_rank() {
        let group = Group::new(vec![0], 0, 1).unwrap();
        // Include a NaN in the data
        let f16_nan: u16 = 0x7E00; // quiet NaN
        let f16_one: u16 = 0x3C00; // 1.0
        let data: Vec<u8> = [f16_one, f16_nan, f16_one]
            .iter()
            .flat_map(|&v| v.to_ne_bytes())
            .collect();
        let result = group
            .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
            .unwrap();
        // Element 0 and 2 should be 1.0, element 1 should be NaN
        let elem0 = u16::from_ne_bytes([result[0], result[1]]);
        let elem1 = u16::from_ne_bytes([result[2], result[3]]);
        let elem2 = u16::from_ne_bytes([result[4], result[5]]);
        assert_eq!(elem0, f16_one);
        assert_eq!(elem2, f16_one);
        // elem1 must be NaN (exp=0x1F, mantissa!=0)
        assert_eq!((elem1 >> 10) & 0x1F, 0x1F);
        assert_ne!(elem1 & 0x3FF, 0, "NaN payload must survive allreduce_op");
    }

    // ─── Conversion function unit tests ───

    #[test]
    fn test_add_f32_inplace_basic() {
        let a_vals = [1.0f32, 2.0, 3.0];
        let b_vals = [4.0f32, 5.0, 6.0];
        let mut a: Vec<u8> = a_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let b: Vec<u8> = b_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        add_f32_inplace(&mut a, &b);
        for i in 0..3 {
            let result = f32::from_ne_bytes([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
            assert_eq!(result, a_vals[i] + b_vals[i]);
        }
    }

    #[test]
    fn test_reduce_f32_inplace_max_min() {
        let a_vals = [1.0f32, 5.0, 3.0];
        let b_vals = [4.0f32, 2.0, 6.0];
        let mut a_max: Vec<u8> = a_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let b: Vec<u8> = b_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        reduce_f32_inplace(&mut a_max, &b, ReduceOp::Max);
        let expected_max = [4.0f32, 5.0, 6.0];
        for i in 0..3 {
            let result = f32::from_ne_bytes([
                a_max[i * 4],
                a_max[i * 4 + 1],
                a_max[i * 4 + 2],
                a_max[i * 4 + 3],
            ]);
            assert_eq!(result, expected_max[i]);
        }

        let mut a_min: Vec<u8> = a_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        reduce_f32_inplace(&mut a_min, &b, ReduceOp::Min);
        let expected_min = [1.0f32, 2.0, 3.0];
        for i in 0..3 {
            let result = f32::from_ne_bytes([
                a_min[i * 4],
                a_min[i * 4 + 1],
                a_min[i * 4 + 2],
                a_min[i * 4 + 3],
            ]);
            assert_eq!(result, expected_min[i]);
        }
    }
}
