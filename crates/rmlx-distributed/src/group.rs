//! Communication group abstraction for distributed operations.

use std::fmt;

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
