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
    pub fn new(ranks: Vec<u32>, local_rank: u32, world_size: u32) -> Self {
        let mut ranks = ranks;
        ranks.sort();
        ranks.dedup();
        assert!(
            ranks.contains(&local_rank),
            "local_rank {local_rank} not in group"
        );
        Self {
            ranks,
            local_rank,
            world_size,
            transport: None,
        }
    }

    /// Create a group with all ranks [0, world_size) (single-process stub mode).
    pub fn world(world_size: u32, local_rank: u32) -> Self {
        Self::new((0..world_size).collect(), local_rank, world_size)
    }

    /// Create a group with an RDMA transport backend for real multi-node communication.
    pub fn with_transport(
        ranks: Vec<u32>,
        local_rank: u32,
        world_size: u32,
        transport: Arc<dyn RdmaTransport>,
    ) -> Self {
        let mut ranks = ranks;
        ranks.sort();
        ranks.dedup();
        assert!(
            ranks.contains(&local_rank),
            "local_rank {local_rank} not in group"
        );
        Self {
            ranks,
            local_rank,
            world_size,
            transport: Some(transport),
        }
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
    /// Without transport: returns input unchanged (single-process identity).
    pub fn allreduce(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allreduce", data)?;
        match &self.transport {
            Some(transport) => ring_allreduce(data, &self.ranks, self.local_rank, transport),
            None => Ok(data.to_vec()),
        }
    }

    /// All-gather: gather data from all ranks into every rank.
    ///
    /// With transport: ring allgather — each rank circulates its chunk around
    /// the ring in N-1 rounds until every rank has all chunks.
    ///
    /// Without transport: returns just the local data (single-process).
    pub fn allgather(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allgather", data)?;
        match &self.transport {
            Some(transport) => ring_allgather(data, &self.ranks, self.local_rank, transport),
            None => Ok(data.to_vec()),
        }
    }

    /// Broadcast: root rank sends data to all other ranks.
    ///
    /// With transport: root sends to each peer; non-root ranks recv from root.
    ///
    /// Without transport: returns input unchanged (single-process).
    pub fn broadcast(&self, data: &[u8], root: u32) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("broadcast", data)?;
        match &self.transport {
            Some(transport) => {
                if self.local_rank == root {
                    // Root sends to all peers
                    for &rank in &self.ranks {
                        if rank != root {
                            transport.send(data, rank)?;
                        }
                    }
                    Ok(data.to_vec())
                } else {
                    // Non-root receives from root
                    transport.recv(root, data.len())
                }
            }
            None => Ok(data.to_vec()),
        }
    }

    /// Send data to a specific peer rank.
    ///
    /// With transport: posts a real RDMA send.
    /// Without transport: no-op (single-process).
    pub fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        Self::check_materialized("send", data)?;
        match &self.transport {
            Some(transport) => transport.send(data, dst_rank),
            None => Ok(()),
        }
    }

    /// Receive data from a specific peer rank.
    ///
    /// With transport: posts a real RDMA recv and waits for completion.
    /// Without transport: returns zeroed buffer (single-process stub).
    pub fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        if len == 0 {
            return Err(DistributedError::NotMaterialized(
                "recv: requested zero-length buffer".to_string(),
            ));
        }
        match &self.transport {
            Some(transport) => transport.recv(src_rank, len),
            None => Ok(vec![0u8; len]),
        }
    }

    /// All-to-all: each rank sends a distinct chunk to every other rank.
    ///
    /// With transport: pairwise exchange — data is split into N equal chunks,
    /// chunk[i] is sent to rank i, and chunk from rank i is received.
    ///
    /// Without transport: returns input unchanged (single-process).
    pub fn all_to_all(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("all_to_all", data)?;
        match &self.transport {
            Some(transport) => pairwise_all_to_all(data, &self.ranks, self.local_rank, transport),
            None => Ok(data.to_vec()),
        }
    }

    /// Internal helper: validate data is materialized before a collective.
    ///
    /// In debug builds (`#[cfg(debug_assertions)]`), unmaterialized data
    /// causes a panic for fast failure during development.
    /// In release builds, returns `DistributedError::NotMaterialized`.
    fn check_materialized(op_name: &str, data: &[u8]) -> Result<(), DistributedError> {
        let shapes = [(data.len(), data.len())];
        let result = ensure_materialized(&shapes);
        if let Err(e) = result {
            #[cfg(debug_assertions)]
            {
                panic!(
                    "{op_name}: data not materialized — {e}. \
                     All buffers must contain valid data before collective operations."
                );
            }
            #[cfg(not(debug_assertions))]
            {
                let _ = op_name;
                return Err(e);
            }
        }
        Ok(())
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
    let my_idx = ranks
        .iter()
        .position(|&r| r == local_rank)
        .expect("local_rank must be in ranks");
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

    let my_idx = ranks
        .iter()
        .position(|&r| r == local_rank)
        .expect("local_rank must be in ranks");
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

    let my_idx = ranks
        .iter()
        .position(|&r| r == local_rank)
        .expect("local_rank must be in ranks");

    let mut result = vec![0u8; data.len()];
    // Our own chunk stays in place
    result[my_idx * chunk_size..(my_idx + 1) * chunk_size]
        .copy_from_slice(&data[my_idx * chunk_size..(my_idx + 1) * chunk_size]);

    // Exchange with each peer
    for (peer_idx, &peer_rank) in ranks.iter().enumerate() {
        if peer_rank == local_rank {
            continue;
        }
        // Send chunk destined for this peer
        let send_start = peer_idx * chunk_size;
        transport.send(&data[send_start..send_start + chunk_size], peer_rank)?;
        // Receive chunk from this peer
        let received = transport.recv(peer_rank, chunk_size)?;
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
