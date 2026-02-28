use rmlx_rdma::multi_port::*;

#[test]
fn test_single_port_config() {
    let port = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let config = DualPortConfig::single(port);
    assert!(!config.has_dual());
}

#[test]
fn test_dual_port_config() {
    let p1 = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let p2 = PortConfig {
        port_num: 2,
        gid_index: 1,
        interface: "en6".into(),
        address: "10.254.0.7".into(),
    };
    let config = DualPortConfig::dual(p1, p2, 8);
    assert!(config.has_dual());
}

#[test]
fn test_stripe_plan_single_port() {
    let port = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let engine = StripeEngine::new(DualPortConfig::single(port));
    let plan = engine.plan(1024, 256);
    assert_eq!(plan.primary_chunks.len(), 4);
    assert!(plan.secondary_chunks.is_empty());
    assert_eq!(plan.total_bytes, 1024);
}

#[test]
fn test_stripe_plan_dual_port() {
    let p1 = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let p2 = PortConfig {
        port_num: 2,
        gid_index: 1,
        interface: "en6".into(),
        address: "10.254.0.7".into(),
    };
    let engine = StripeEngine::new(DualPortConfig::dual(p1, p2, 4));
    // 16KB / 1KB chunks = 16 chunks >= threshold 4
    let plan = engine.plan(16384, 1024);
    assert!(
        !plan.secondary_chunks.is_empty(),
        "should use secondary port"
    );
    let total_chunks = plan.primary_chunks.len() + plan.secondary_chunks.len();
    assert_eq!(total_chunks, 16);
    // Round-robin: primary gets even, secondary gets odd
    assert_eq!(plan.primary_chunks.len(), 8);
    assert_eq!(plan.secondary_chunks.len(), 8);
}

#[test]
fn test_stripe_below_threshold() {
    let p1 = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let p2 = PortConfig {
        port_num: 2,
        gid_index: 1,
        interface: "en6".into(),
        address: "10.254.0.7".into(),
    };
    let engine = StripeEngine::new(DualPortConfig::dual(p1, p2, 8));
    // 4 chunks < threshold 8 => all on primary
    let plan = engine.plan(4096, 1024);
    assert_eq!(plan.primary_chunks.len(), 4);
    assert!(plan.secondary_chunks.is_empty());
}

#[test]
fn test_ring_topology() {
    let topo = Topology::Ring;
    assert_eq!(topo.connections_per_node(4), 2);
    let peers = topo.peers(0, 4);
    assert!(peers.contains(&1));
    assert!(peers.contains(&3));
    assert!(!peers.contains(&0));
}

#[test]
fn test_mesh_topology() {
    let topo = Topology::Mesh;
    assert_eq!(topo.connections_per_node(4), 3);
    let peers = topo.peers(1, 4);
    assert_eq!(peers.len(), 3);
    assert!(peers.contains(&0));
    assert!(peers.contains(&2));
    assert!(peers.contains(&3));
}

#[test]
fn test_hybrid_topology() {
    let topo = Topology::Hybrid { group_size: 2 };
    let peers = topo.peers(0, 4);
    assert!(peers.contains(&1), "should have in-group peer");
    assert!(peers.contains(&2), "should have cross-group peer");
}

#[test]
fn test_port_failover() {
    let mut failover = PortFailover::new();
    assert!(failover.is_dual_active());

    failover.mark_failed(false); // secondary failed
    assert!(!failover.is_dual_active());
    assert!(failover.has_active_port());

    failover.mark_failed(true); // primary failed too
    assert!(!failover.has_active_port());

    failover.mark_active(true);
    assert!(failover.has_active_port());
}

#[test]
fn test_chunk_ordering() {
    let p1 = PortConfig {
        port_num: 1,
        gid_index: 1,
        interface: "en5".into(),
        address: "10.254.0.5".into(),
    };
    let p2 = PortConfig {
        port_num: 2,
        gid_index: 1,
        interface: "en6".into(),
        address: "10.254.0.7".into(),
    };
    let engine = StripeEngine::new(DualPortConfig::dual(p1, p2, 4));
    let plan = engine.plan(8192, 1024);
    // Verify sequence numbers allow reassembly
    let mut all_seqs: Vec<u32> = plan.primary_chunks.iter().map(|c| c.seq).collect();
    all_seqs.extend(plan.secondary_chunks.iter().map(|c| c.seq));
    all_seqs.sort();
    assert_eq!(all_seqs, (0..8).collect::<Vec<u32>>());
}
