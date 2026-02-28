//! Communication group abstraction for distributed operations.

use std::fmt;

/// Error type for distributed operations.
#[derive(Debug)]
pub enum DistributedError {
    /// Arrays must be materialized (data resident in GPU buffer) before collective ops.
    NotMaterialized(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMaterialized(msg) => write!(f, "not materialized: {msg}"),
        }
    }
}

impl std::error::Error for DistributedError {}

/// Verify all arrays are materialized before a collective operation.
///
/// "Materialized" means the array's backing buffer has actual data (non-zero length,
/// valid GPU buffer). This must be called before any allreduce, allgather, or
/// broadcast to prevent sending uninitialized memory over RDMA.
pub fn ensure_materialized(shapes: &[(usize, usize)]) -> Result<(), DistributedError> {
    for (i, &(numel, byte_size)) in shapes.iter().enumerate() {
        if numel == 0 || byte_size == 0 {
            return Err(DistributedError::NotMaterialized(format!(
                "array at index {i} has zero elements or zero bytes — \
                 all arrays must be materialized before collective operations"
            )));
        }
    }
    Ok(())
}

/// A communication group identifying a set of ranks.
#[derive(Debug, Clone)]
pub struct Group {
    /// Ranks in this group (sorted, unique).
    ranks: Vec<u32>,
    /// This node's rank.
    local_rank: u32,
    /// Total world size.
    world_size: u32,
}

impl Group {
    /// Create a new group from a list of ranks.
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
        }
    }

    /// Create a group with all ranks [0, world_size).
    pub fn world(world_size: u32, local_rank: u32) -> Self {
        Self::new((0..world_size).collect(), local_rank, world_size)
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
    // In debug builds, unmaterialized data causes a panic.
    // In release builds, a DistributedError::NotMaterialized is returned.

    /// All-reduce: sum (or other reduction) across all ranks.
    pub fn allreduce(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allreduce", data)?;
        // Stub: in single-process mode, the result is the input unchanged.
        Ok(data.to_vec())
    }

    /// All-gather: gather data from all ranks into every rank.
    pub fn allgather(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("allgather", data)?;
        // Stub: single-process returns just the local data repeated.
        Ok(data.to_vec())
    }

    /// Broadcast: root rank sends data to all other ranks.
    pub fn broadcast(&self, data: &[u8], _root: u32) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("broadcast", data)?;
        Ok(data.to_vec())
    }

    /// Send data to a specific peer rank.
    pub fn send(&self, data: &[u8], _dst_rank: u32) -> Result<(), DistributedError> {
        Self::check_materialized("send", data)?;
        // Stub: no-op in single-process mode.
        Ok(())
    }

    /// Receive data from a specific peer rank.
    pub fn recv(&self, _src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        if len == 0 {
            return Err(DistributedError::NotMaterialized(
                "recv: requested zero-length buffer".to_string(),
            ));
        }
        // Stub: return zeroed buffer of the expected length.
        Ok(vec![0u8; len])
    }

    /// All-to-all: each rank sends a distinct chunk to every other rank.
    pub fn all_to_all(&self, data: &[u8]) -> Result<Vec<u8>, DistributedError> {
        Self::check_materialized("all_to_all", data)?;
        Ok(data.to_vec())
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
            "Group(rank={}, size={}, ranks={:?})",
            self.local_rank,
            self.ranks.len(),
            self.ranks
        )
    }
}
