//! EP RDMA integration tests.
//!
//! Run with: RMLX_TEST_EP=1 cargo test -p rmlx-distributed --test ep_rdma_integration -- --ignored

use rmlx_distributed::group::{DistributedError, Group, RdmaTransport};
use rmlx_distributed::moe_exchange::{
    MoeCombineExchange, MoeDispatchConfig, MoeDispatchExchange, WireProtocol,
};
use rmlx_distributed::moe_policy::{MoeBackend, MoePolicy};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─── Mock transport ───

/// Loopback transport for testing EP dispatch/combine without real RDMA hardware.
/// Shares an in-memory queue map between two rank endpoints.
struct LoopbackTransport {
    local_rank: u32,
    #[allow(clippy::type_complexity)]
    queues: Arc<Mutex<HashMap<(u32, u32), Vec<Vec<u8>>>>>,
}

impl LoopbackTransport {
    fn new_pair() -> (Arc<Self>, Arc<Self>) {
        let queues = Arc::new(Mutex::new(HashMap::new()));
        let t0 = Arc::new(Self {
            local_rank: 0,
            queues: queues.clone(),
        });
        let t1 = Arc::new(Self {
            local_rank: 1,
            queues,
        });
        (t0, t1)
    }
}

impl RdmaTransport for LoopbackTransport {
    fn send(&self, data: &[u8], dst_rank: u32) -> Result<(), DistributedError> {
        let mut q = self.queues.lock().unwrap();
        q.entry((self.local_rank, dst_rank))
            .or_default()
            .push(data.to_vec());
        Ok(())
    }

    fn recv(&self, src_rank: u32, len: usize) -> Result<Vec<u8>, DistributedError> {
        let mut q = self.queues.lock().unwrap();
        let queue = q.entry((src_rank, self.local_rank)).or_default();
        if let Some(msg) = queue.first().cloned() {
            queue.remove(0);
            let mut buf = msg;
            buf.resize(len, 0);
            Ok(buf)
        } else {
            Ok(vec![0u8; len])
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

// ─── Helper ───

fn has_metal_gpu() -> bool {
    objc2_metal::MTLCreateSystemDefaultDevice().is_some()
}

// ─── Tests ───

#[test]
#[ignore = "requires EP test setup; run with RMLX_TEST_EP=1"]
fn test_ep_dispatch_local_only() {
    if !has_metal_gpu() {
        eprintln!("Skipping: no Metal GPU");
        return;
    }

    // Single node (world_size=1) — all 4 experts are local.
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.5,
        group,
        wire_protocol: WireProtocol::V2,
        enable_fp8: false,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());

    // 8 tokens, top-1 routing, round-robin across 4 experts.
    let num_tokens = 8;
    let indices: Vec<u32> = (0..num_tokens as u32).map(|i| i % 4).collect();
    let weights: Vec<f32> = vec![1.0; num_tokens];
    // hidden_dim = 4 floats = 16 bytes per token
    let hidden_dim = 4;
    let token_bytes = num_tokens * hidden_dim * std::mem::size_of::<f32>();
    let token_data = vec![1u8; token_bytes];

    let result = exchange
        .dispatch(num_tokens, &indices, &weights, &token_data)
        .unwrap();

    // All experts are local → range should be (0, 4).
    assert_eq!(result.local_expert_range, (0, 4));
    // 8 tokens / 4 experts = 2 per expert (uniform).
    assert_eq!(result.expert_counts.len(), 4);
    for &count in &result.expert_counts {
        assert_eq!(count, 2, "each expert should receive exactly 2 tokens");
    }
    // No overflow with capacity_factor=1.5.
    assert_eq!(result.overflow_count, 0);
    // Routed data should be non-empty.
    assert!(!result.routed_data.is_empty());
    // Layout should be cached.
    assert_eq!(result.layout.batch_size, num_tokens);
    assert_eq!(result.layout.experts_per_rank, 4);
}

#[test]
#[ignore = "requires EP test setup; run with RMLX_TEST_EP=1"]
fn test_ep_dispatch_two_node_routing_metadata() {
    if !has_metal_gpu() {
        eprintln!("Skipping: no Metal GPU");
        return;
    }

    // 2-node setup: rank 0 owns experts 0-3, rank 1 owns experts 4-7.
    let (t0, _t1) = LoopbackTransport::new_pair();
    let group = Group::with_transport(vec![0, 1], 0, 2, t0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 2,
        capacity_factor: 1.25,
        group,
        wire_protocol: WireProtocol::V2,
        enable_fp8: false,
    };
    // Force RDMA path with low thresholds.
    let policy = MoePolicy::with_thresholds(0, 1, 0);
    policy.set_world_size(2);

    let mut exchange = MoeDispatchExchange::new(config, policy);

    let num_tokens = 16;
    // top-2: 32 assignments, cycling through 8 experts.
    let indices: Vec<u32> = (0..num_tokens * 2).map(|i| (i as u32) % 8).collect();
    let weights: Vec<f32> = vec![0.6, 0.4]
        .into_iter()
        .cycle()
        .take(num_tokens * 2)
        .collect();
    let hidden_dim = 4;
    let token_data = vec![42u8; num_tokens * hidden_dim * std::mem::size_of::<f32>()];

    let result = exchange
        .dispatch(num_tokens, &indices, &weights, &token_data)
        .unwrap();

    // Rank 0 owns experts 0-3.
    assert_eq!(result.local_expert_range, (0, 4));
    // 8 experts total.
    assert_eq!(result.expert_counts.len(), 8);
    // Layout should reflect 2 ranks.
    assert_eq!(result.layout.experts_per_rank, 4);
    // RDMA backend should be selected.
    assert_eq!(result.backend, MoeBackend::Rdma);
    assert_eq!(exchange.metrics().snapshot().rdma_dispatches, 1);
}

#[test]
#[ignore = "requires EP test setup; run with RMLX_TEST_EP=1"]
fn test_ep_combine_cpu_roundtrip() {
    if !has_metal_gpu() {
        eprintln!("Skipping: no Metal GPU");
        return;
    }

    // Single node, 2 experts, top-1 routing, hidden_dim=3.
    // dispatch → identity expert → combine, verify output ≈ input * weight.
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);

    let num_tokens = 4;
    let hidden_dim = 3;
    let top_k = 1;
    let num_experts = 2;

    // Token i → expert (i % 2), weight = 0.5 for all.
    let indices: Vec<u32> = (0..num_tokens)
        .map(|i| (i as u32) % num_experts as u32)
        .collect();
    let weights: Vec<f32> = vec![0.5; num_tokens];

    // Build expert outputs as if each expert applied an identity function.
    // Expert 0 receives tokens 0, 2; Expert 1 receives tokens 1, 3.
    // Token values: token_i = [(i+1)*1.0, (i+1)*2.0, (i+1)*3.0]
    let mut expert0_out = Vec::new(); // tokens 0, 2
    let mut expert1_out = Vec::new(); // tokens 1, 3

    for token_idx in 0..num_tokens {
        let base = (token_idx + 1) as f32;
        let vals = vec![base * 1.0, base * 2.0, base * 3.0];
        if token_idx % 2 == 0 {
            expert0_out.extend_from_slice(&vals);
        } else {
            expert1_out.extend_from_slice(&vals);
        }
    }

    let expert_outputs = vec![expert0_out, expert1_out];

    let result = combine.combine_cpu(
        &expert_outputs,
        &weights,
        &indices,
        num_tokens,
        top_k,
        hidden_dim,
    );

    assert_eq!(result.len(), num_tokens * hidden_dim);

    // Token 0 → expert 0, slot 0: weight=0.5 * [1, 2, 3] = [0.5, 1.0, 1.5]
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.5).abs() < 1e-5);

    // Token 1 → expert 1, slot 0: weight=0.5 * [2, 4, 6] = [1.0, 2.0, 3.0]
    assert!((result[3] - 1.0).abs() < 1e-5);
    assert!((result[4] - 2.0).abs() < 1e-5);
    assert!((result[5] - 3.0).abs() < 1e-5);

    // Token 2 → expert 0, slot 1: weight=0.5 * [3, 6, 9] = [1.5, 3.0, 4.5]
    assert!((result[6] - 1.5).abs() < 1e-5);
    assert!((result[7] - 3.0).abs() < 1e-5);
    assert!((result[8] - 4.5).abs() < 1e-5);

    // Token 3 → expert 1, slot 1: weight=0.5 * [4, 8, 12] = [2.0, 4.0, 6.0]
    assert!((result[9] - 2.0).abs() < 1e-5);
    assert!((result[10] - 4.0).abs() < 1e-5);
    assert!((result[11] - 6.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires EP test setup; run with RMLX_TEST_EP=1"]
fn test_ep_dispatch_layout_caching() {
    if !has_metal_gpu() {
        eprintln!("Skipping: no Metal GPU");
        return;
    }

    // Verify that DispatchLayout from dispatch() captures correct metadata
    // for later reuse in combine_with_layout.
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        group,
        wire_protocol: WireProtocol::V2,
        enable_fp8: false,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());

    let num_tokens = 6;
    let indices: Vec<u32> = vec![0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3];
    let weights: Vec<f32> = vec![0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2, 0.9, 0.1, 0.55, 0.45];
    let hidden_dim = 2;
    let token_data = vec![1u8; num_tokens * hidden_dim * std::mem::size_of::<f32>()];

    let result = exchange
        .dispatch(num_tokens, &indices, &weights, &token_data)
        .unwrap();
    let layout = &result.layout;

    // Layout metadata consistency checks.
    assert_eq!(layout.batch_size, num_tokens);
    assert_eq!(layout.expert_indices.len(), num_tokens * 2); // top_k=2
    assert_eq!(layout.expert_counts.len(), 4);
    assert_eq!(layout.local_expert_range, (0, 4));
    assert_eq!(layout.experts_per_rank, 4);

    // route_indices should have one entry per (n, k) pair.
    assert_eq!(layout.route_indices.len(), num_tokens * 2);

    // Total assigned tokens across experts should equal total non-dropped slots.
    let total_assigned: usize = layout.expert_counts.iter().sum();
    let non_dropped: usize = layout.route_indices.iter().filter(|&&idx| idx >= 0).count();
    assert_eq!(total_assigned, non_dropped);
}

#[test]
#[ignore = "requires EP test setup; run with RMLX_TEST_EP=1"]
fn test_ep_dispatch_overflow_with_capacity() {
    if !has_metal_gpu() {
        eprintln!("Skipping: no Metal GPU");
        return;
    }

    // All tokens routed to a single expert with tight capacity → overflow expected.
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.0,
        group,
        wire_protocol: WireProtocol::V2,
        enable_fp8: false,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());

    let num_tokens = 16;
    // All tokens go to expert 0.
    let indices: Vec<u32> = vec![0; num_tokens];
    let weights: Vec<f32> = vec![1.0; num_tokens];
    let hidden_dim = 2;
    let token_data = vec![1u8; num_tokens * hidden_dim * std::mem::size_of::<f32>()];

    let result = exchange
        .dispatch(num_tokens, &indices, &weights, &token_data)
        .unwrap();

    // capacity = ceil(16 / 4 * 1.0) = 4 per expert. 16 tokens to expert 0 → 12 overflow.
    assert!(result.overflow_count > 0, "should have overflow tokens");
    assert_eq!(result.expert_counts[0], 4, "expert 0 capped at capacity");
    assert_eq!(result.expert_counts[1], 0);
    assert_eq!(result.expert_counts[2], 0);
    assert_eq!(result.expert_counts[3], 0);
    assert!(exchange.metrics().snapshot().overflow_events > 0);
}
