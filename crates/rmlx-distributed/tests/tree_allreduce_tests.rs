//! Tests for tree allreduce, auto-dispatch, and topology-aware ring ordering.

use rmlx_distributed::group::{
    allreduce_auto_with_threshold, tree_allreduce, AllreduceAlgorithm, TopologyRing,
};
use rmlx_distributed::{DistributedError, RdmaTransport};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Mailboxes = Arc<Mutex<HashMap<(u32, u32), Vec<Vec<u8>>>>>;

// ─── Mock transport for simulating multi-rank communication ───

/// A simple in-memory transport that routes messages between simulated ranks.
/// Each rank gets its own `MockTransport` instance sharing the same `MockHub`.
#[derive(Clone)]
struct MockHub {
    /// Mailboxes: (src_rank, dst_rank) -> Vec<messages>
    mailboxes: Mailboxes,
}

impl MockHub {
    fn new() -> Self {
        Self {
            mailboxes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn send(&self, src: u32, dst: u32, data: &[u8]) {
        let mut mb = self.mailboxes.lock().unwrap();
        mb.entry((src, dst)).or_default().push(data.to_vec());
    }

    fn recv(&self, src: u32, dst: u32) -> Option<Vec<u8>> {
        let mut mb = self.mailboxes.lock().unwrap();
        let queue = mb.entry((src, dst)).or_default();
        if queue.is_empty() {
            None
        } else {
            Some(queue.remove(0))
        }
    }
}

struct MockTransport {
    rank: u32,
    hub: MockHub,
}

impl MockTransport {
    fn new(rank: u32, hub: MockHub) -> Self {
        Self { rank, hub }
    }
}

impl RdmaTransport for MockTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        self.hub.send(self.rank, dst_rank, data);
        Ok(())
    }

    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        // Busy-wait (fine for tests) until data arrives
        loop {
            if let Some(data) = self.hub.recv(src_rank, self.rank) {
                assert_eq!(
                    data.len(),
                    len,
                    "MockTransport: expected {} bytes from rank {}, got {}",
                    len,
                    src_rank,
                    data.len()
                );
                return Ok(data);
            }
            std::thread::yield_now();
        }
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

// ─── Helper: run tree_allreduce across multiple simulated ranks ───

/// Simulates tree_allreduce across `world_size` ranks, each starting with
/// `per_rank_data[rank]`. Returns the result from each rank.
fn simulate_tree_allreduce(per_rank_data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let world_size = per_rank_data.len();
    let ranks: Vec<u32> = (0..world_size as u32).collect();
    let hub = MockHub::new();

    let handles: Vec<_> = (0..world_size)
        .map(|r| {
            let data = per_rank_data[r].clone();
            let ranks = ranks.clone();
            let hub = hub.clone();
            std::thread::spawn(move || {
                let transport: Arc<dyn RdmaTransport> = Arc::new(MockTransport::new(r as u32, hub));
                let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
                let result_bytes =
                    tree_allreduce(&data_bytes, &ranks, r as u32, &transport).unwrap();
                // Convert back to f32
                result_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>()
            })
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

// ─── Tests ───

#[test]
fn test_tree_allreduce_2_ranks() {
    // rank 0: [1.0, 2.0, 3.0, 4.0]
    // rank 1: [5.0, 6.0, 7.0, 8.0]
    // expected sum: [6.0, 8.0, 10.0, 12.0]
    let data = vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0f32, 6.0, 7.0, 8.0]];
    let results = simulate_tree_allreduce(&data);

    let expected = vec![6.0f32, 8.0, 10.0, 12.0];
    for (rank, result) in results.iter().enumerate() {
        assert_eq!(
            result, &expected,
            "rank {rank} should have the allreduced result"
        );
    }
}

#[test]
fn test_tree_allreduce_4_ranks() {
    // rank 0: [1.0, 0.0]
    // rank 1: [0.0, 2.0]
    // rank 2: [3.0, 0.0]
    // rank 3: [0.0, 4.0]
    // expected sum: [4.0, 6.0]
    let data = vec![
        vec![1.0f32, 0.0],
        vec![0.0f32, 2.0],
        vec![3.0f32, 0.0],
        vec![0.0f32, 4.0],
    ];
    let results = simulate_tree_allreduce(&data);

    let expected = vec![4.0f32, 6.0];
    for (rank, result) in results.iter().enumerate() {
        assert_eq!(
            result, &expected,
            "rank {rank} should have the allreduced result"
        );
    }
}

#[test]
fn test_allreduce_auto_small_uses_tree() {
    // Data < 1MB should use tree
    let hub = MockHub::new();
    let transport: Arc<dyn RdmaTransport> = Arc::new(MockTransport::new(0, hub));
    let ranks = vec![0u32]; // single rank for simplicity
    let small_data = vec![0u8; 100]; // 100 bytes, well below 1MB

    let (result, algo) =
        allreduce_auto_with_threshold(&small_data, &ranks, 0, &transport, 1024 * 1024).unwrap();
    assert_eq!(algo, AllreduceAlgorithm::Tree);
    assert_eq!(result, small_data);
}

#[test]
fn test_allreduce_auto_large_uses_mesh_for_small_group() {
    // With <= 2 ranks, large data uses mesh (ring requires > 2 ranks)
    let hub = MockHub::new();
    let transport: Arc<dyn RdmaTransport> = Arc::new(MockTransport::new(0, hub));
    let ranks = vec![0u32]; // single rank
    let large_data = vec![0u8; 1024 * 1024]; // exactly 1MB

    let (result, algo) =
        allreduce_auto_with_threshold(&large_data, &ranks, 0, &transport, 1024 * 1024).unwrap();
    assert_eq!(algo, AllreduceAlgorithm::Mesh);
    assert_eq!(result, large_data);
}

#[test]
fn test_topology_ring_from_hops() {
    // 4 nodes with hop matrix:
    //   0  1  2  1
    //   1  0  1  2
    //   2  1  0  1
    //   1  2  1  0
    // Optimal ring starting from 0: 0 -> 1 (hop=1) -> 2 (hop=1) -> 3 (hop=1) -> back to 0
    let hops = vec![
        vec![0, 1, 2, 1],
        vec![1, 0, 1, 2],
        vec![2, 1, 0, 1],
        vec![1, 2, 1, 0],
    ];
    let ranks = vec![0, 1, 2, 3];
    let ring = TopologyRing::from_hops(&hops, &ranks).unwrap();

    // Greedy from 0: nearest is 1 or 3 (both hop=1), pick 1 first (lower index).
    // From 1: nearest unvisited is 2 (hop=1), 3 is hop=2.
    // From 2: only 3 left (hop=1).
    // Expected: [0, 1, 2, 3]
    assert_eq!(ring.order, vec![0, 1, 2, 3]);

    // Now test with a different topology where greedy gives different ordering
    let hops2 = vec![
        vec![0, 3, 1, 2],
        vec![3, 0, 2, 1],
        vec![1, 2, 0, 3],
        vec![2, 1, 3, 0],
    ];
    let ring2 = TopologyRing::from_hops(&hops2, &ranks).unwrap();
    // From 0: nearest is 2 (hop=1)
    // From 2: nearest unvisited: 1 (hop=2) or 3 (hop=3) -> pick 1
    // From 1: only 3 left (hop=1)
    // Expected: [0, 2, 1, 3]
    assert_eq!(ring2.order, vec![0, 2, 1, 3]);
}

#[test]
fn test_topology_ring_fallback() {
    // When RMLX_TOPOLOGY is not set, from_env should return sequential ordering.
    // We unset it to be safe (it shouldn't be set in test env).
    std::env::remove_var("RMLX_TOPOLOGY");

    let ranks = vec![0, 1, 2, 3];
    let ring = TopologyRing::from_env(&ranks).unwrap();
    assert_eq!(ring.order, vec![0, 1, 2, 3]);
}

#[test]
fn test_topology_ring_from_env_json() {
    // Set the env var with a hop matrix and verify parsing
    let json = r#"{"hops":[[0,1,2],[1,0,1],[2,1,0]]}"#;
    std::env::set_var("RMLX_TOPOLOGY", json);

    let ranks = vec![0, 1, 2];
    let ring = TopologyRing::from_env(&ranks).unwrap();
    // From 0: nearest is 1 (hop=1)
    // From 1: nearest is 2 (hop=1)
    // Expected: [0, 1, 2]
    assert_eq!(ring.order, vec![0, 1, 2]);

    // Clean up
    std::env::remove_var("RMLX_TOPOLOGY");
}

#[test]
fn test_topology_ring_invalid_json() {
    std::env::set_var("RMLX_TOPOLOGY", "not valid json");
    let result = TopologyRing::from_env(&[0, 1]);
    assert!(result.is_err());
    std::env::remove_var("RMLX_TOPOLOGY");
}

#[test]
fn test_topology_ring_empty_ranks() {
    let hops: Vec<Vec<u32>> = vec![];
    let ring = TopologyRing::from_hops(&hops, &[]).unwrap();
    assert!(ring.order.is_empty());
}

#[test]
fn test_topology_ring_single_rank() {
    let hops = vec![vec![0]];
    let ring = TopologyRing::from_hops(&hops, &[0]).unwrap();
    assert_eq!(ring.order, vec![0]);
}

#[test]
fn test_tree_allreduce_single_rank() {
    let hub = MockHub::new();
    let transport: Arc<dyn RdmaTransport> = Arc::new(MockTransport::new(0, hub));
    let data: Vec<u8> = [1.0f32, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let result = tree_allreduce(&data, &[0], 0, &transport).unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_tree_allreduce_3_ranks() {
    // Odd number of ranks to test non-power-of-2
    // rank 0: [1.0, 1.0]
    // rank 1: [2.0, 2.0]
    // rank 2: [3.0, 3.0]
    // expected sum: [6.0, 6.0]
    let data = vec![vec![1.0f32, 1.0], vec![2.0f32, 2.0], vec![3.0f32, 3.0]];
    let results = simulate_tree_allreduce(&data);

    let expected = vec![6.0f32, 6.0];
    for (rank, result) in results.iter().enumerate() {
        assert_eq!(
            result, &expected,
            "rank {rank} should have the allreduced result"
        );
    }
}
