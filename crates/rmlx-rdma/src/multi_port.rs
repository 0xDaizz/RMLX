//! Dual Thunderbolt 5 port striping for increased bandwidth.
//!
//! When N >= configurable threshold, data is split across two TB5 ports
//! for approximately 2x bandwidth. Falls back to single port for small transfers.

/// Port configuration for a TB5 link.
#[derive(Debug, Clone)]
pub struct PortConfig {
    /// Port index (1-based, matching IB convention).
    pub port_num: u8,
    /// GID index for this port.
    pub gid_index: i32,
    /// Interface name (e.g., "en5", "en6").
    pub interface: String,
    /// IP address.
    pub address: String,
}

/// Dual-port striping configuration.
#[derive(Debug, Clone)]
pub struct DualPortConfig {
    /// Primary port (always used).
    pub primary: PortConfig,
    /// Secondary port (used when data >= stripe_threshold).
    pub secondary: Option<PortConfig>,
    /// Minimum number of chunks to activate striping.
    pub stripe_threshold: usize,
}

impl DualPortConfig {
    /// Single port configuration (no striping).
    pub fn single(port: PortConfig) -> Self {
        Self {
            primary: port,
            secondary: None,
            stripe_threshold: 8,
        }
    }

    /// Dual port configuration.
    pub fn dual(primary: PortConfig, secondary: PortConfig, threshold: usize) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            stripe_threshold: threshold,
        }
    }

    /// Whether dual-port striping is available.
    pub fn has_dual(&self) -> bool {
        self.secondary.is_some()
    }
}

/// Striping plan: how to split data across ports.
#[derive(Debug)]
pub struct StripePlan {
    /// Chunks assigned to primary port.
    pub primary_chunks: Vec<ChunkAssignment>,
    /// Chunks assigned to secondary port (empty if single-port).
    pub secondary_chunks: Vec<ChunkAssignment>,
    /// Total bytes to transfer.
    pub total_bytes: usize,
}

/// A chunk assignment for a specific port.
#[derive(Debug, Clone)]
pub struct ChunkAssignment {
    /// Byte offset in the source buffer.
    pub offset: usize,
    /// Length in bytes.
    pub length: usize,
    /// Sequence number for reordering.
    pub seq: u32,
}

/// Striping engine for distributing data across dual TB5 ports.
pub struct StripeEngine {
    config: DualPortConfig,
}

impl StripeEngine {
    pub fn new(config: DualPortConfig) -> Self {
        Self { config }
    }

    /// Plan how to stripe data across ports.
    /// Splits into chunks and assigns round-robin to primary/secondary.
    pub fn plan(&self, total_bytes: usize, chunk_size: usize) -> StripePlan {
        let n_chunks = total_bytes.div_ceil(chunk_size);

        let mut primary = Vec::new();
        let mut secondary = Vec::new();

        let use_dual = self.config.has_dual() && n_chunks >= self.config.stripe_threshold;

        for i in 0..n_chunks {
            let offset = i * chunk_size;
            let length = std::cmp::min(chunk_size, total_bytes - offset);
            let assignment = ChunkAssignment {
                offset,
                length,
                seq: i as u32,
            };

            if use_dual && i % 2 == 1 {
                secondary.push(assignment);
            } else {
                primary.push(assignment);
            }
        }

        StripePlan {
            primary_chunks: primary,
            secondary_chunks: secondary,
            total_bytes,
        }
    }

    /// Config reference.
    pub fn config(&self) -> &DualPortConfig {
        &self.config
    }
}

/// Multi-node topology types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Point-to-point / ring (2+ nodes).
    Ring,
    /// Full mesh (all-to-all connections).
    Mesh,
    /// Hybrid: ring backbone + mesh within groups.
    Hybrid { group_size: usize },
}

impl Topology {
    /// Number of connections per node.
    pub fn connections_per_node(&self, world_size: usize) -> usize {
        match self {
            Topology::Ring => std::cmp::min(2, world_size - 1),
            Topology::Mesh => world_size - 1,
            Topology::Hybrid { group_size } => {
                let in_group = std::cmp::min(*group_size - 1, world_size - 1);
                let cross_group = if world_size > *group_size { 1 } else { 0 };
                in_group + cross_group
            }
        }
    }

    /// Peer ranks for a given rank in this topology.
    pub fn peers(&self, rank: u32, world_size: u32) -> Vec<u32> {
        match self {
            Topology::Ring => {
                let mut peers = Vec::new();
                if world_size > 1 {
                    peers.push((rank + 1) % world_size);
                    if world_size > 2 {
                        peers.push((rank + world_size - 1) % world_size);
                    }
                }
                peers
            }
            Topology::Mesh => (0..world_size).filter(|&r| r != rank).collect(),
            Topology::Hybrid { group_size } => {
                let gs = *group_size as u32;
                let group_start = (rank / gs) * gs;
                let group_end = std::cmp::min(group_start + gs, world_size);
                let mut peers: Vec<u32> = (group_start..group_end).filter(|&r| r != rank).collect();
                // Add one cross-group peer
                let next_group_rep = group_end % world_size;
                if next_group_rep != rank && !peers.contains(&next_group_rep) {
                    peers.push(next_group_rep);
                }
                peers
            }
        }
    }
}

/// Failover state for dual-port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortState {
    Active,
    Failed,
    Recovering,
}

/// Dual-port failover manager.
pub struct PortFailover {
    primary_state: PortState,
    secondary_state: PortState,
}

impl PortFailover {
    pub fn new() -> Self {
        Self {
            primary_state: PortState::Active,
            secondary_state: PortState::Active,
        }
    }

    /// Mark a port as failed.
    pub fn mark_failed(&mut self, is_primary: bool) {
        if is_primary {
            self.primary_state = PortState::Failed;
        } else {
            self.secondary_state = PortState::Failed;
        }
    }

    /// Mark a port as recovering.
    pub fn mark_recovering(&mut self, is_primary: bool) {
        if is_primary {
            self.primary_state = PortState::Recovering;
        } else {
            self.secondary_state = PortState::Recovering;
        }
    }

    /// Mark a port as active.
    pub fn mark_active(&mut self, is_primary: bool) {
        if is_primary {
            self.primary_state = PortState::Active;
        } else {
            self.secondary_state = PortState::Active;
        }
    }

    /// Whether dual-port is fully operational.
    pub fn is_dual_active(&self) -> bool {
        self.primary_state == PortState::Active && self.secondary_state == PortState::Active
    }

    /// Whether at least one port is active.
    pub fn has_active_port(&self) -> bool {
        self.primary_state == PortState::Active || self.secondary_state == PortState::Active
    }

    pub fn primary_state(&self) -> PortState {
        self.primary_state
    }

    pub fn secondary_state(&self) -> PortState {
        self.secondary_state
    }
}

impl Default for PortFailover {
    fn default() -> Self {
        Self::new()
    }
}
