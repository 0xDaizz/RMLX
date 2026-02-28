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
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMaterialized(msg) => write!(f, "not materialized: {msg}"),
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
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
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        transport.send(&buf[send_start..send_end], right)?;
        let received = transport.recv(left, chunk_size)?;

        // Accumulate: interpret as f32 and sum
        let recv_start = recv_chunk_idx * chunk_size;
        add_f32_inplace(&mut buf[recv_start..recv_start + chunk_size], &received);
    }

    // Phase 2: allgather (N-1 rounds)
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + 1 + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round) % n;
        let send_start = send_chunk_idx * chunk_size;
        let send_end = send_start + chunk_size;

        transport.send(&buf[send_start..send_end], right)?;
        let received = transport.recv(left, chunk_size)?;

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
    // receive a new chunk from left.
    for round in 0..(n - 1) {
        let send_chunk_idx = (my_idx + n - round) % n;
        let recv_chunk_idx = (my_idx + n - round - 1) % n;
        let send_start = send_chunk_idx * chunk_size;

        transport.send(&result[send_start..send_start + chunk_size], right)?;
        let received = transport.recv(left, chunk_size)?;

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
