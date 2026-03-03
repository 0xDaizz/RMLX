//! Communication group abstraction for distributed operations.

use std::fmt;
use std::sync::Arc;

/// Error type for distributed operations.
#[derive(Debug)]
pub enum DistributedError {
    /// Arrays must be materialized (data resident in GPU buffer) before collective ops.
    NotMaterialized(String),
    /// RDMA transport error (wraps rmlx_rdma::RdmaError description).
    Transport(String),
    /// Wire protocol or data format error (e.g., byte slice conversion failure).
    Protocol(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMaterialized(msg) => write!(f, "not materialized: {msg}"),
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Protocol(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for DistributedError {}

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
        ring_allreduce(data, &self.ranks, self.local_rank, transport)
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
        Self::check_materialized("allreduce_op", data)?;
        if self.ranks.len() <= 1 {
            return Ok(data.to_vec());
        }
        let transport = self.require_transport("allreduce_op")?;
        ring_allreduce_op(data, &self.ranks, self.local_rank, transport, op, dtype)
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
        ring_allgather(data, &self.ranks, self.local_rank, transport)
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
        device: &metal::Device,
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
        device: &metal::Device,
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
        ring_reduce_scatter(data, &self.ranks, self.local_rank, transport)
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

    // Split data into N equal chunks (pad if needed)
    let chunk_size = data.len().div_ceil(n);
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
            let n_elems = data.len() / 2;
            let mut f32_data = vec![0u8; n_elems * 4];
            for i in 0..n_elems {
                let bits = u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]);
                let val = f16_to_f32(bits);
                f32_data[i * 4..i * 4 + 4].copy_from_slice(&val.to_ne_bytes());
            }
            let result = ring_allreduce_op_f32(&f32_data, ranks, local_rank, transport, op)?;
            let mut out = vec![0u8; n_elems * 2];
            for i in 0..n_elems {
                let val = f32::from_ne_bytes([
                    result[i * 4],
                    result[i * 4 + 1],
                    result[i * 4 + 2],
                    result[i * 4 + 3],
                ]);
                let bits = f32_to_f16(val);
                out[i * 2..i * 2 + 2].copy_from_slice(&bits.to_ne_bytes());
            }
            Ok(out)
        }
        ReduceDtype::Bf16 => {
            if data.len() % 2 != 0 {
                return Err(DistributedError::Protocol(format!(
                    "allreduce_op(bf16): data length ({}) must be a multiple of 2",
                    data.len()
                )));
            }
            let n_elems = data.len() / 2;
            let mut f32_data = vec![0u8; n_elems * 4];
            for i in 0..n_elems {
                let bits = u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]);
                let val = bf16_to_f32(bits);
                f32_data[i * 4..i * 4 + 4].copy_from_slice(&val.to_ne_bytes());
            }
            let result = ring_allreduce_op_f32(&f32_data, ranks, local_rank, transport, op)?;
            let mut out = vec![0u8; n_elems * 2];
            for i in 0..n_elems {
                let val = f32::from_ne_bytes([
                    result[i * 4],
                    result[i * 4 + 1],
                    result[i * 4 + 2],
                    result[i * 4 + 3],
                ]);
                let bits = f32_to_bf16(val);
                out[i * 2..i * 2 + 2].copy_from_slice(&bits.to_ne_bytes());
            }
            Ok(out)
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

    let chunk_size = data.len().div_ceil(n);
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
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let f32_exp = (127 - 15 - e) as u32;
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
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        let f16_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | (0x1F << 10) | f16_mantissa
    } else if exponent > 127 + 15 {
        (sign << 15) | (0x1F << 10)
    } else if exponent < 127 - 14 {
        sign << 15
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

/// Convert f32 to bfloat16 bits (truncation).
fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}
