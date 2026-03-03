//! ConnectionManager — N-peer RDMA connection management with topology awareness.
//!
//! Manages a map of rank -> RdmaConnection for multi-node clusters.
//! Supports adding/removing peers, topology-aware peer selection,
//! and iteration over all connections.

use std::collections::HashMap;

use crate::connection::RdmaConnection;
use crate::multi_port::Topology;
use crate::RdmaError;

/// Manages RDMA connections to multiple peers, indexed by rank.
///
/// Provides topology-aware peer selection and connection lifecycle
/// management for N-peer clusters.
pub struct ConnectionManager {
    /// This node's rank.
    rank: u32,
    /// Total number of nodes in the cluster.
    world_size: u32,
    /// Topology used for peer selection.
    topology: Topology,
    /// Active connections, keyed by peer rank.
    connections: HashMap<u32, RdmaConnection>,
}

impl ConnectionManager {
    /// Create a new ConnectionManager for the given rank and topology.
    pub fn new(rank: u32, world_size: u32, topology: Topology) -> Self {
        Self {
            rank,
            world_size,
            topology,
            connections: HashMap::new(),
        }
    }

    /// This node's rank.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Total world size.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// The topology used for peer selection.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Add a connection to a peer rank.
    ///
    /// Returns an error if a connection to this rank already exists.
    pub fn add_connection(
        &mut self,
        peer_rank: u32,
        conn: RdmaConnection,
    ) -> Result<(), RdmaError> {
        if peer_rank == self.rank {
            return Err(RdmaError::InvalidArgument(
                "cannot add connection to self".into(),
            ));
        }
        if peer_rank >= self.world_size {
            return Err(RdmaError::InvalidArgument(format!(
                "peer rank {peer_rank} >= world_size {}",
                self.world_size
            )));
        }
        if self.connections.contains_key(&peer_rank) {
            return Err(RdmaError::ConnectionFailed(format!(
                "connection to rank {peer_rank} already exists"
            )));
        }
        self.connections.insert(peer_rank, conn);
        Ok(())
    }

    /// Remove and return the connection to a peer rank.
    ///
    /// Returns `None` if no connection exists for the given rank.
    pub fn remove_connection(&mut self, peer_rank: u32) -> Option<RdmaConnection> {
        self.connections.remove(&peer_rank)
    }

    /// Get a reference to the connection for a peer rank.
    pub fn get(&self, peer_rank: u32) -> Option<&RdmaConnection> {
        self.connections.get(&peer_rank)
    }

    /// Get a mutable reference to the connection for a peer rank.
    pub fn get_mut(&mut self, peer_rank: u32) -> Option<&mut RdmaConnection> {
        self.connections.get_mut(&peer_rank)
    }

    /// Returns true if a connection to the given peer rank exists.
    pub fn has_connection(&self, peer_rank: u32) -> bool {
        self.connections.contains_key(&peer_rank)
    }

    /// Number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Iterate over all (rank, connection) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &RdmaConnection)> {
        self.connections.iter()
    }

    /// Iterate mutably over all (rank, connection) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&u32, &mut RdmaConnection)> {
        self.connections.iter_mut()
    }

    /// Return the set of peer ranks this node should connect to,
    /// based on the configured topology.
    pub fn expected_peers(&self) -> Vec<u32> {
        self.topology.peers(self.rank, self.world_size)
    }

    /// Return the number of expected connections based on topology.
    pub fn expected_connection_count(&self) -> usize {
        self.topology.connections_per_node(self.world_size as usize)
    }

    /// Return true if all expected topology peers are connected.
    pub fn is_fully_connected(&self) -> bool {
        let expected = self.expected_peers();
        expected.iter().all(|r| self.connections.contains_key(r))
    }

    /// Return the ranks of peers that are expected but not yet connected.
    pub fn missing_peers(&self) -> Vec<u32> {
        self.expected_peers()
            .into_iter()
            .filter(|r| !self.connections.contains_key(r))
            .collect()
    }

    /// Get the left (previous) neighbor in the ring topology.
    ///
    /// For ring-based collectives, the left neighbor is `(rank - 1 + world_size) % world_size`.
    pub fn ring_left(&self) -> u32 {
        (self.rank + self.world_size - 1) % self.world_size
    }

    /// Get the right (next) neighbor in the ring topology.
    ///
    /// For ring-based collectives, the right neighbor is `(rank + 1) % world_size`.
    pub fn ring_right(&self) -> u32 {
        (self.rank + 1) % self.world_size
    }

    /// Get the connection to the left ring neighbor.
    pub fn left_connection(&self) -> Option<&RdmaConnection> {
        self.connections.get(&self.ring_left())
    }

    /// Get the connection to the right ring neighbor.
    pub fn right_connection(&self) -> Option<&RdmaConnection> {
        self.connections.get(&self.ring_right())
    }

    /// Return all connected peer ranks, sorted.
    pub fn connected_ranks(&self) -> Vec<u32> {
        let mut ranks: Vec<u32> = self.connections.keys().copied().collect();
        ranks.sort_unstable();
        ranks
    }

    /// Drain all connections, returning them as a vec of (rank, connection) pairs.
    pub fn drain_all(&mut self) -> Vec<(u32, RdmaConnection)> {
        self.connections.drain().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock-friendly test helper: since RdmaConnection requires real RDMA hardware,
    /// we test the manager logic using topology and peer selection methods that
    /// don't require actual connections.

    #[test]
    fn test_new_manager() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        assert_eq!(mgr.rank(), 0);
        assert_eq!(mgr.world_size(), 4);
        assert_eq!(mgr.connection_count(), 0);
        assert!(!mgr.is_fully_connected());
    }

    #[test]
    fn test_expected_peers_ring() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        let peers = mgr.expected_peers();
        assert_eq!(peers, vec![1, 3]);
    }

    #[test]
    fn test_expected_peers_mesh() {
        let mgr = ConnectionManager::new(1, 4, Topology::Mesh);
        let mut peers = mgr.expected_peers();
        peers.sort();
        assert_eq!(peers, vec![0, 2, 3]);
    }

    #[test]
    fn test_expected_peers_hybrid() {
        let mgr = ConnectionManager::new(0, 6, Topology::Hybrid { group_size: 3 });
        let peers = mgr.expected_peers();
        // In group [0,1,2], cross-group peer is 3
        assert!(peers.contains(&1));
        assert!(peers.contains(&2));
        assert!(peers.contains(&3));
    }

    #[test]
    fn test_ring_neighbors() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        assert_eq!(mgr.ring_left(), 3);
        assert_eq!(mgr.ring_right(), 1);

        let mgr2 = ConnectionManager::new(2, 4, Topology::Ring);
        assert_eq!(mgr2.ring_left(), 1);
        assert_eq!(mgr2.ring_right(), 3);
    }

    #[test]
    fn test_ring_neighbors_two_nodes() {
        let mgr = ConnectionManager::new(0, 2, Topology::Ring);
        assert_eq!(mgr.ring_left(), 1);
        assert_eq!(mgr.ring_right(), 1);
    }

    #[test]
    fn test_expected_connection_count() {
        let ring4 = ConnectionManager::new(0, 4, Topology::Ring);
        assert_eq!(ring4.expected_connection_count(), 2);

        let mesh4 = ConnectionManager::new(0, 4, Topology::Mesh);
        assert_eq!(mesh4.expected_connection_count(), 3);
    }

    #[test]
    fn test_missing_peers() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        let missing = mgr.missing_peers();
        assert_eq!(missing, vec![1, 3]);
    }

    #[test]
    fn test_connected_ranks_empty() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        assert!(mgr.connected_ranks().is_empty());
    }
}
