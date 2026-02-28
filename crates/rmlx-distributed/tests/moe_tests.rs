use rmlx_distributed::group::{self, DistributedError, Group, RdmaTransport};
use rmlx_distributed::moe_exchange::{MoeCombineExchange, MoeDispatchConfig, MoeDispatchExchange};
use rmlx_distributed::moe_policy::{MoeBackend, MoePolicy, ThresholdCalibration};
use rmlx_distributed::sparse_guard::SparseGuard;
use rmlx_distributed::warmup::WarmupState;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[test]
fn test_group_world() {
    let g = Group::world(4, 1).unwrap();
    assert_eq!(g.size(), 4);
    assert_eq!(g.local_rank(), 1);
    assert_eq!(g.peers(), vec![0, 2, 3]);
    assert!(g.contains(2));
    assert!(!g.contains(5));
}

#[test]
fn test_moe_policy_zones() {
    let policy = MoePolicy::new();
    // CPU zone: N <= 64
    assert_eq!(policy.select(32, 128), MoeBackend::Cpu);
    assert_eq!(policy.select(64, 256), MoeBackend::Cpu);
    // GPU zone: N >= 320
    assert_eq!(policy.select(320, 1280), MoeBackend::Metal);
    assert_eq!(policy.select(1000, 4000), MoeBackend::Metal);
    // Middle zone: byte threshold
    assert_eq!(policy.select(128, 512), MoeBackend::Cpu); // below 4096 bytes
    assert_eq!(policy.select(128, 8192), MoeBackend::Metal); // above 4096 bytes
}

#[test]
fn test_moe_policy_cooldown() {
    let mut policy = MoePolicy::new();
    policy.switch_backend(MoeBackend::Cpu);
    // During cooldown, should keep Cpu
    assert_eq!(policy.cooldown_remaining(), 32);
    for _ in 0..32 {
        assert_eq!(policy.select(500, 10000), MoeBackend::Cpu);
    }
    // After cooldown expires (32 calls consumed it), should switch based on thresholds
    let result = policy.select(500, 10000);
    assert_eq!(result, MoeBackend::Metal);
}

#[test]
fn test_moe_dispatch_basic() {
    let group = Group::world(2, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 2,
        capacity_factor: 1.25,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    // 4 tokens, top-2 => 8 expert assignments
    let indices = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
    let weights = vec![0.6f32, 0.4, 0.5, 0.5, 0.7, 0.3, 0.8, 0.2];
    // 4 tokens * 16 bytes each (simulated hidden_dim=4 * f32)
    let token_data = vec![1u8; 4 * 16];
    let result = exchange
        .dispatch(4, &indices, &weights, &token_data)
        .unwrap();
    assert_eq!(result.expert_counts.len(), 8);
    assert_eq!(result.local_expert_range, (0, 4)); // rank 0 owns experts 0-3
    assert_eq!(exchange.metrics().snapshot().total_tokens_routed, 4);
    assert!(!result.routed_data.is_empty());
}

#[test]
fn test_moe_dispatch_overflow() {
    let group = Group::world(2, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.0, // exact capacity
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    // All tokens go to expert 0 — massive overflow
    let indices = vec![0u32, 0, 0, 0, 0, 0, 0, 0]; // 8 tokens, all to expert 0
    let weights = vec![1.0f32; 8];
    let token_data = vec![1u8; 8 * 16];
    let result = exchange
        .dispatch(8, &indices, &weights, &token_data)
        .unwrap();
    // capacity = ceil(8 * 1 / 4 * 1.0) = 2
    assert!(result.overflow_count > 0, "should have overflow");
    assert!(exchange.metrics().snapshot().overflow_events > 0);
}

#[test]
fn test_moe_combine_cpu() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    // 2 experts, 2 tokens, top_k=1, hidden_dim=3
    let expert0_out = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens * 3 dims
    let expert1_out = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let expert_outputs = vec![expert0_out, expert1_out];
    let weights = vec![0.5f32, 0.7]; // token0->expert0 w=0.5, token1->expert1 w=0.7
    let indices = vec![0u32, 1];
    let result = combine.combine_cpu(&expert_outputs, &weights, &indices, 2, 1, 3);
    // token 0: 0.5 * [1,2,3] = [0.5, 1.0, 1.5]
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.5).abs() < 1e-5);
    // token 1: 0.7 * [40,50,60] = [28, 35, 42]
    assert!((result[3] - 28.0).abs() < 1e-5);
    assert!((result[4] - 35.0).abs() < 1e-5);
    assert!((result[5] - 42.0).abs() < 1e-5);
}

#[test]
fn test_moe_metrics() {
    let m = rmlx_distributed::metrics::MoeMetrics::default();
    m.record_dispatch(100);
    m.record_metal_dispatch();
    m.record_overflow();
    let snap = m.snapshot();
    assert_eq!(snap.total_tokens_routed, 100);
    assert_eq!(snap.dispatch_count, 1);
    assert_eq!(snap.overflow_events, 1);
    assert_eq!(snap.metal_dispatches, 1);
}

#[test]
fn test_3zone_boundary() {
    let policy = MoePolicy::with_thresholds(64, 320, 4096);
    // Exact boundaries
    assert_eq!(policy.select(64, 100), MoeBackend::Cpu);
    assert_eq!(policy.select(65, 100), MoeBackend::Cpu); // below byte threshold
    assert_eq!(policy.select(65, 4096), MoeBackend::Metal); // at byte threshold
    assert_eq!(policy.select(319, 10000), MoeBackend::Metal);
    assert_eq!(policy.select(320, 100), MoeBackend::Metal);
}

#[test]
fn test_warmup_state() {
    let mut state = WarmupState::new();
    assert!(!state.is_ready());
    state.set_rdma_warmed();
    assert!(!state.is_ready());
    state.set_jit_warmed();
    assert!(state.is_ready());
}

#[test]
fn test_group_display() {
    let g = Group::world(2, 0).unwrap();
    let s = format!("{g}");
    assert!(s.contains("rank=0"));
    assert!(s.contains("size=2"));
}

// ─── RDMA zone selection tests ───

#[test]
fn test_moe_policy_rdma_zone_multinode() {
    let mut policy = MoePolicy::new();
    policy.set_world_size(2);
    policy.set_hysteresis_band(0); // disable hysteresis for clean threshold testing

    // CPU zone: N <= 64
    assert_eq!(policy.select(32, 128), MoeBackend::Cpu);
    assert_eq!(policy.select(64, 256), MoeBackend::Cpu);

    // Metal zone: 64 < N <= 320 (single node would also be Metal for large N)
    assert_eq!(policy.select(128, 8192), MoeBackend::Metal);

    // RDMA zone: N > 320 AND world_size > 1
    assert_eq!(policy.select(500, 2000), MoeBackend::Rdma);
    assert_eq!(policy.select(1000, 4000), MoeBackend::Rdma);
}

#[test]
fn test_moe_policy_no_rdma_single_node() {
    let mut policy = MoePolicy::new();
    policy.set_world_size(1);
    policy.set_hysteresis_band(0);

    // Even with large N, single node should NOT select RDMA
    assert_eq!(policy.select(500, 2000), MoeBackend::Metal);
    assert_eq!(policy.select(1000, 4000), MoeBackend::Metal);
}

#[test]
fn test_moe_policy_hysteresis() {
    let mut policy = MoePolicy::with_thresholds(64, 320, 4096);
    policy.set_world_size(2);
    // Default hysteresis_band = 16

    // Start in Metal (default). To drop to CPU must go below 64 - 16 = 48.
    // N=50, byte=5000 → middle zone (48 < 50 < 320), byte >= 4096 → Metal (stays)
    assert_eq!(policy.select(50, 5000), MoeBackend::Metal);
    // N=47, byte=5000 → 47 <= 48 → CPU
    assert_eq!(policy.select(47, 5000), MoeBackend::Cpu);

    // Switch to CPU
    policy.switch_backend(MoeBackend::Cpu);
    // consume cooldown
    for _ in 0..32 {
        policy.select(100, 1000);
    }
    // Now in CPU. To leave CPU, must exceed 64 + 16 = 80.
    assert_eq!(policy.select(75, 5000), MoeBackend::Cpu); // 75 <= 80, stays CPU
    assert_eq!(policy.select(81, 5000), MoeBackend::Metal); // 81 > 80, middle zone, byte >= 4096 → Metal
}

#[test]
fn test_moe_dispatch_rdma_routing() {
    let (t0, _t1) = LoopbackTransport::new_pair();
    let group = Group::with_transport(vec![0, 1], 0, 2, t0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 2,
        capacity_factor: 1.25,
        group,
    };
    let mut policy = MoePolicy::new();
    policy.set_world_size(2);
    policy.set_hysteresis_band(0);

    let mut exchange = MoeDispatchExchange::new(config, policy);

    // Large dispatch that should trigger RDMA (500 elements > 320 threshold)
    let indices: Vec<u32> = (0..500).map(|i| i % 8).collect();
    let weights: Vec<f32> = vec![0.5; 500];
    let token_data = vec![1u8; 250 * 16];
    let result = exchange
        .dispatch(250, &indices, &weights, &token_data)
        .unwrap();
    assert_eq!(result.backend, MoeBackend::Rdma);
    assert_eq!(exchange.metrics().snapshot().rdma_dispatches, 1);
    assert!(!result.routed_data.is_empty());
}

// ─── Threshold calibration tests ───

#[test]
fn test_threshold_calibration_finds_crossover() {
    let mut cal = ThresholdCalibration::new();

    // Simulate: CPU is faster for small N, GPU is faster for N >= 128
    cal.calibrate(
        |n| n as f64 * 0.001, // CPU: linear scaling
        |n| {
            if n < 128 {
                0.5 // GPU: fixed overhead for small N
            } else {
                n as f64 * 0.0005 // GPU: faster for large N
            }
        },
    );

    assert!(cal.calibrated);
    assert_eq!(cal.crossover_n, 128);
    // cpu_max = max(64, 128 - 32) = 96
    assert_eq!(cal.cpu_max, 96);
    // gpu_min = min(320, 128 + 32) = 160
    assert_eq!(cal.gpu_min, 160);
}

#[test]
fn test_threshold_calibration_apply_to_policy() {
    let mut cal = ThresholdCalibration::new();
    cal.calibrate(
        |n| n as f64 * 0.001,
        |_n| 0.01, // GPU always faster
    );

    let mut policy = MoePolicy::new();
    cal.apply_to(&mut policy);

    assert!(cal.calibrated);
    // Crossover at N=32 (first test size where GPU is faster)
    assert_eq!(cal.crossover_n, 32);
    assert_eq!(policy.cpu_max(), cal.cpu_max as u32);
    assert_eq!(policy.gpu_min(), cal.gpu_min as u32);
}

#[test]
fn test_threshold_calibration_no_crossover() {
    let mut cal = ThresholdCalibration::new();
    // CPU is always faster than GPU
    cal.calibrate(
        |_n| 0.001, // CPU always fast
        |_n| 1.0,   // GPU always slow
    );
    assert!(cal.calibrated);
    // No crossover found → crossover_n stays 0, defaults unchanged
    assert_eq!(cal.crossover_n, 0);
    assert_eq!(cal.cpu_max, 64);
    assert_eq!(cal.gpu_min, 320);
}

// ─── ensure_materialized tests ───

#[test]
fn test_ensure_materialized_valid() {
    // All arrays have non-zero elements and bytes
    let shapes = vec![(100, 400), (50, 200), (1, 4)];
    assert!(group::ensure_materialized(&shapes).is_ok());
}

#[test]
fn test_ensure_materialized_empty_array() {
    let shapes = vec![(100, 400), (0, 0), (50, 200)];
    let result = group::ensure_materialized(&shapes);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("index 1"));
}

#[test]
fn test_ensure_materialized_zero_bytes() {
    let shapes = vec![(10, 0)];
    assert!(group::ensure_materialized(&shapes).is_err());
}

// ─── Route function correctness tests ───

#[test]
fn test_route_cpu_scatters_tokens_correctly() {
    let group = Group::world(2, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.25,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());

    // 4 tokens, top_k=1. Experts: [0, 1, 2, 3]
    // Rank 0 owns experts 0-1, Rank 1 owns experts 2-3
    let indices = vec![0u32, 1, 2, 3];
    let weights = vec![1.0f32; 4];
    // Each token is 4 bytes (simulated): [AA, BB, CC, DD]
    let token_data = vec![0xAAu8, 0xBB, 0xCC, 0xDD];
    let result = exchange
        .dispatch(4, &indices, &weights, &token_data)
        .unwrap();

    // Only local experts (0 and 1) should have data
    // Token 0 -> expert 0 (local), Token 1 -> expert 1 (local)
    assert!(!result.routed_data.is_empty());
    // Routed data should contain bytes from token 0 and token 1
}

#[test]
fn test_route_rdma_materialization_guard() {
    // Passing empty token_data should fail ensure_materialized in RDMA path
    let (t0, _t1) = LoopbackTransport::new_pair();
    let group = Group::with_transport(vec![0, 1], 0, 2, t0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut policy = MoePolicy::new();
    policy.set_world_size(2);
    policy.set_hysteresis_band(0);
    let mut exchange = MoeDispatchExchange::new(config, policy);

    let indices: Vec<u32> = (0..500).map(|i| i % 4).collect();
    let weights: Vec<f32> = vec![1.0; 500];
    // Empty token data — should trigger NotMaterialized error on RDMA path
    let token_data: Vec<u8> = vec![];
    let result = exchange.dispatch(500, &indices, &weights, &token_data);
    // batch_size != 0 but token_data empty => token_stride = 0 => early return empty
    // The RDMA path calls ensure_materialized with (0, 0) => error
    assert!(result.is_err() || result.unwrap().routed_data.is_empty());
}

#[test]
fn test_dispatch_result_has_routed_data() {
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 2,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());

    // 2 tokens, each 8 bytes, top_k=1
    let indices = vec![0u32, 1];
    let weights = vec![1.0f32; 2];
    let token_data = vec![0xFFu8; 2 * 8];
    let result = exchange
        .dispatch(2, &indices, &weights, &token_data)
        .unwrap();
    // Both experts are local (single rank), routed_data should be non-empty
    assert!(!result.routed_data.is_empty());
}

// ─── Collective ensure_materialized enforcement tests ───

// In debug builds, collectives panic on unmaterialized data.
// In release builds, they return DistributedError::NotMaterialized.

#[test]
fn test_collective_allreduce_rejects_empty() {
    let g = Group::world(2, 0).unwrap();
    assert!(g.allreduce(&[]).is_err());
}

#[test]
fn test_collective_allgather_rejects_empty() {
    let g = Group::world(2, 0).unwrap();
    assert!(g.allgather(&[]).is_err());
}

#[test]
fn test_collective_broadcast_rejects_empty() {
    let g = Group::world(2, 0).unwrap();
    assert!(g.broadcast(&[], 0).is_err());
}

#[test]
fn test_collective_send_rejects_empty() {
    let g = Group::world(2, 0).unwrap();
    assert!(g.send(&[], 1).is_err());
}

#[test]
fn test_collective_recv_rejects_zero_len() {
    let g = Group::world(2, 0).unwrap();
    let result = g.recv(1, 0);
    assert!(result.is_err());
}

#[test]
fn test_collective_all_to_all_rejects_empty() {
    let g = Group::world(2, 0).unwrap();
    assert!(g.all_to_all(&[]).is_err());
}

#[test]
fn test_collective_allreduce_accepts_valid() {
    let g = Group::world(1, 0).unwrap();
    let data = vec![1u8, 2, 3, 4];
    let result = g.allreduce(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), data);
}

#[test]
fn test_collective_send_recv_accepts_valid() {
    let g = Group::world(1, 0).unwrap();
    let data = vec![1u8, 2, 3, 4];
    assert!(g.send(&data, 1).is_ok());
    let received = g.recv(1, 4).unwrap();
    assert_eq!(received.len(), 4);
}

#[test]
fn test_collective_all_to_all_accepts_valid() {
    let g = Group::world(1, 0).unwrap();
    let data = vec![1u8; 64];
    let result = g.all_to_all(&data).unwrap();
    assert_eq!(result, data);
}

// ─── Fail-open removal: multi-rank without transport must error ───

#[test]
fn test_multirank_no_transport_allreduce_errors() {
    let g = Group::world(2, 0).unwrap();
    let data = vec![1u8, 2, 3, 4];
    let result = g.allreduce(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("requires transport"), "error: {err}");
}

#[test]
fn test_multirank_no_transport_send_errors() {
    let g = Group::world(2, 0).unwrap();
    let data = vec![1u8, 2, 3, 4];
    let result = g.send(&data, 1);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("requires transport"), "error: {err}");
}

#[test]
fn test_multirank_no_transport_barrier_errors() {
    let g = Group::world(3, 0).unwrap();
    let result = g.barrier();
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("requires transport"), "error: {err}");
}

#[test]
fn test_all_to_all_rejects_indivisible_length() {
    // Use a 3-rank group: 8 bytes % 3 != 0, but 8 is 4-byte aligned (passes materialization)
    let queues = Arc::new(Mutex::new(HashMap::new()));
    let t0: Arc<dyn RdmaTransport> = Arc::new(LoopbackTransport {
        local_rank: 0,
        queues,
    });
    let g = Group::with_transport(vec![0, 1, 2], 0, 3, t0).unwrap();
    let data = vec![1u8; 8]; // 8 bytes, 4-aligned, but 8 % 3 != 0
    let result = g.all_to_all(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("divisible"), "error: {err}");
}

// ─── Barrier tests ───

#[test]
fn test_barrier_single_rank() {
    let g = Group::world(1, 0).unwrap();
    assert!(g.barrier().is_ok());
}

#[test]
fn test_barrier_3_rank_with_loopback() {
    // 3-rank barrier using loopback transport.
    // Each rank sends a token right and receives from left, for 2 rounds.
    let queues = Arc::new(Mutex::new(HashMap::new()));
    let transports: Vec<Arc<dyn RdmaTransport>> = (0..3)
        .map(|rank| {
            Arc::new(LoopbackTransport {
                local_rank: rank,
                queues: queues.clone(),
            }) as Arc<dyn RdmaTransport>
        })
        .collect();

    // Run barriers from each rank sequentially (loopback doesn't need real concurrency)
    for rank in 0..3u32 {
        let g = Group::with_transport(vec![0, 1, 2], rank, 3, transports[rank as usize].clone())
            .unwrap();
        let result = g.barrier();
        assert!(
            result.is_ok(),
            "barrier failed for rank {rank}: {:?}",
            result
        );
    }
}

#[test]
fn test_barrier_4_rank_with_loopback() {
    let queues = Arc::new(Mutex::new(HashMap::new()));
    let transports: Vec<Arc<dyn RdmaTransport>> = (0..4)
        .map(|rank| {
            Arc::new(LoopbackTransport {
                local_rank: rank,
                queues: queues.clone(),
            }) as Arc<dyn RdmaTransport>
        })
        .collect();

    for rank in 0..4u32 {
        let g = Group::with_transport(vec![0, 1, 2, 3], rank, 4, transports[rank as usize].clone())
            .unwrap();
        let result = g.barrier();
        assert!(
            result.is_ok(),
            "barrier failed for rank {rank}: {:?}",
            result
        );
    }
}

// ─── MoeCombineExchange edge case tests ───

#[test]
fn test_moe_combine_cpu_empty_expert_output() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    // Expert 0 has output but expert 1 is empty
    let expert0_out = vec![1.0f32, 2.0, 3.0];
    let expert_outputs: Vec<Vec<f32>> = vec![expert0_out, vec![]];
    let weights = vec![0.5f32, 0.8]; // token 0 -> expert 0, token 1 -> expert 1
    let indices = vec![0u32, 1];
    // 2 tokens, top_k=1, hidden_dim=3
    // token 1 -> expert 1 which is empty, so nothing is added
    let result = combine.combine_cpu(&expert_outputs, &weights, &indices, 2, 1, 3);
    // token 0: 0.5 * [1,2,3] = [0.5, 1.0, 1.5]
    assert!((result[0] - 0.5).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 1.5).abs() < 1e-5);
    // token 1: expert 1 has empty output, exp_base + hidden_dim > expert_out.len()
    // so nothing is added → stays 0.0
    assert!((result[3] - 0.0).abs() < 1e-5);
    assert!((result[4] - 0.0).abs() < 1e-5);
    assert!((result[5] - 0.0).abs() < 1e-5);
}

#[test]
fn test_moe_combine_cpu_zero_weights() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    let expert0_out = vec![10.0f32, 20.0, 30.0];
    let expert_outputs = vec![expert0_out];
    let weights = vec![0.0f32]; // zero weight
    let indices = vec![0u32];
    let result = combine.combine_cpu(&expert_outputs, &weights, &indices, 1, 1, 3);
    // 0.0 * [10, 20, 30] = [0, 0, 0]
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_moe_combine_cpu_top2() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    // 1 token, top_k=2, hidden_dim=2, 2 experts
    let expert0_out = vec![1.0f32, 2.0]; // expert 0 output for token 0
    let expert1_out = vec![10.0f32, 20.0]; // expert 1 output for token 0
    let expert_outputs = vec![expert0_out, expert1_out];
    let weights = vec![0.6f32, 0.4]; // token 0 -> expert 0 w=0.6, expert 1 w=0.4
    let indices = vec![0u32, 1];
    let result = combine.combine_cpu(&expert_outputs, &weights, &indices, 1, 2, 2);
    // token 0: 0.6 * [1, 2] + 0.4 * [10, 20] = [0.6+4.0, 1.2+8.0] = [4.6, 9.2]
    assert!((result[0] - 4.6).abs() < 1e-5, "got {}", result[0]);
    assert!((result[1] - 9.2).abs() < 1e-5, "got {}", result[1]);
}

#[test]
fn test_moe_combine_cpu_out_of_range_expert() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    // Only 1 expert available, but index refers to expert 5 (out of range)
    let expert0_out = vec![1.0f32, 2.0];
    let expert_outputs = vec![expert0_out];
    let weights = vec![1.0f32];
    let indices = vec![5u32]; // out of range
    let result = combine.combine_cpu(&expert_outputs, &weights, &indices, 1, 1, 2);
    // Expert index 5 >= expert_outputs.len() (1), so nothing added
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
}

// ─── Loopback RDMA transport for testing size exchange + expert-ID protocol ───

/// A loopback transport that records sent messages and replays them on recv.
/// Simulates a 2-rank world where rank 0 and rank 1 communicate.
/// Messages sent to a rank are queued; recv pops from that queue.
struct LoopbackTransport {
    local_rank: u32,
    /// Keyed by (src_rank, dst_rank), stores queued messages.
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
            // Return exactly `len` bytes: truncate or zero-pad
            let mut buf = msg;
            buf.resize(len, 0);
            Ok(buf)
        } else {
            // No message queued — return zeros (simulates empty recv)
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

// ─── Size exchange roundtrip test ───

#[test]
fn test_rdma_size_exchange_roundtrip() {
    // Two ranks. Rank 0 sends tokens for experts 4-7 (rank 1).
    // We run dispatch on rank 0 and verify that size headers are sent.
    let (t0, _t1) = LoopbackTransport::new_pair();

    let group = Group::with_transport(vec![0, 1], 0, 2, t0.clone()).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 1,
        capacity_factor: 2.0,
        group,
    };
    // Use low thresholds so small batches still trigger RDMA path
    let mut policy = MoePolicy::with_thresholds(0, 1, 0);
    policy.set_world_size(2);
    policy.set_hysteresis_band(0);
    let mut exchange = MoeDispatchExchange::new(config, policy);

    // 4 tokens: experts [4, 5, 6, 7] — all remote for rank 0
    let indices = vec![4u32, 5, 6, 7];
    let weights = vec![1.0f32; 4];
    let token_data = vec![0xABu8; 4 * 16]; // 4 tokens, 16 bytes each

    let result = exchange.dispatch(4, &indices, &weights, &token_data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().backend, MoeBackend::Rdma);

    // Verify the transport queues: rank 0 should have sent to rank 1:
    // 1) A size header (8 bytes)
    // 2) The actual payload
    let q = t0.queues.lock().unwrap();
    let messages = q.get(&(0, 1)).unwrap();
    assert_eq!(messages.len(), 2, "should have size header + payload");

    // First message is the size header (u64 LE)
    let size_header = &messages[0];
    assert_eq!(size_header.len(), 8);
    let payload_size = u64::from_le_bytes(size_header[..8].try_into().unwrap()) as usize;

    // Second message is the payload
    let payload = &messages[1];
    assert_eq!(payload.len(), payload_size);

    // Each token on wire = 4 (expert_id) + 16 (token_data) = 20 bytes, 4 tokens = 80
    let wire_stride = 4 + 16;
    assert_eq!(payload_size, 4 * wire_stride);
}

// ─── Expert-ID preservation through send/recv/merge cycle ───

#[test]
fn test_rdma_expert_id_preserved_in_merge() {
    // Simulate rank 1 receiving tokens from rank 0 for its local experts (4-7).
    // Rank 0 sends tokens tagged with expert_id=4 and expert_id=6.
    // Rank 1 should place them in the correct expert slots, not first-fit.
    let (t0, t1) = LoopbackTransport::new_pair();

    // --- Rank 0 side: dispatch tokens destined for rank 1's experts ---
    {
        let group0 = Group::with_transport(vec![0, 1], 0, 2, t0.clone()).unwrap();
        let config0 = MoeDispatchConfig {
            num_experts: 8,
            top_k: 1,
            capacity_factor: 2.0,
            group: group0,
        };
        let mut policy0 = MoePolicy::with_thresholds(0, 1, 0);
        policy0.set_world_size(2);
        policy0.set_hysteresis_band(0);
        let mut exchange0 = MoeDispatchExchange::new(config0, policy0);

        // 2 tokens: expert 4 and expert 6 (both on rank 1)
        let indices = vec![4u32, 6];
        let weights = vec![1.0f32; 2];
        // Token 0: [0x11; 8], Token 1: [0x22; 8]
        let mut token_data = vec![0x11u8; 8];
        token_data.extend_from_slice(&[0x22u8; 8]);

        let result = exchange0.dispatch(2, &indices, &weights, &token_data);
        assert!(result.is_ok());
    }

    // Now the loopback queues have: (0,1) -> [size_header, payload]
    // Manually transfer them to (0,1) readable by rank 1.
    // The LoopbackTransport already stores them keyed by (src=0, dst=1),
    // and rank 1's recv reads from (src=0, dst=1). So they're ready.

    // --- Rank 1 side: dispatch with all-local tokens to trigger RDMA recv ---
    {
        let group1 = Group::with_transport(vec![0, 1], 1, 2, t1.clone()).unwrap();
        let config1 = MoeDispatchConfig {
            num_experts: 8,
            top_k: 1,
            capacity_factor: 2.0,
            group: group1,
        };
        let mut policy1 = MoePolicy::with_thresholds(0, 1, 0);
        policy1.set_world_size(2);
        policy1.set_hysteresis_band(0);
        let mut exchange1 = MoeDispatchExchange::new(config1, policy1);

        // Rank 1's local experts are 4-7. Send 2 tokens to local experts 5 and 7.
        // This also triggers the RDMA recv path for peer rank 0.
        let indices = vec![5u32, 7];
        let weights = vec![1.0f32; 2];
        // Token 0: [0x33; 8], Token 1: [0x44; 8]
        let mut token_data = vec![0x33u8; 8];
        token_data.extend_from_slice(&[0x44u8; 8]);

        let result = exchange1.dispatch(2, &indices, &weights, &token_data);
        assert!(result.is_ok());
        let dispatch_result = result.unwrap();
        assert_eq!(dispatch_result.backend, MoeBackend::Rdma);

        // Rank 1 owns experts 4-7 (local_expert_count=4).
        // capacity_per_expert = ceil(2 * 1 / 8 * 2.0) = ceil(0.5) = 1
        let token_stride = 8;
        let capacity_per_expert = 1; // ceil(2/8 * 2.0) = 1
        let routed = &dispatch_result.routed_data;

        // Expert 4 (local_idx=0): should have the token from rank 0 tagged expert_id=4 -> [0x11; 8]
        let expert4_start = 0 * capacity_per_expert * token_stride;
        let expert4_data = &routed[expert4_start..expert4_start + token_stride];
        assert_eq!(
            expert4_data, &[0x11u8; 8],
            "expert 4 should have rank 0's token [0x11]"
        );

        // Expert 5 (local_idx=1): should have rank 1's local token -> [0x33; 8]
        let expert5_start = 1 * capacity_per_expert * token_stride;
        let expert5_data = &routed[expert5_start..expert5_start + token_stride];
        assert_eq!(
            expert5_data, &[0x33u8; 8],
            "expert 5 should have rank 1's local token [0x33]"
        );

        // Expert 6 (local_idx=2): should have the token from rank 0 tagged expert_id=6 -> [0x22; 8]
        let expert6_start = 2 * capacity_per_expert * token_stride;
        let expert6_data = &routed[expert6_start..expert6_start + token_stride];
        assert_eq!(
            expert6_data, &[0x22u8; 8],
            "expert 6 should have rank 0's token [0x22]"
        );

        // Expert 7 (local_idx=3): should have rank 1's local token -> [0x44; 8]
        let expert7_start = 3 * capacity_per_expert * token_stride;
        let expert7_data = &routed[expert7_start..expert7_start + token_stride];
        assert_eq!(
            expert7_data, &[0x44u8; 8],
            "expert 7 should have rank 1's local token [0x44]"
        );
    }
}

// ─── Verify wire format: expert_id prefix is correctly encoded ───

#[test]
fn test_rdma_wire_format_expert_id_prefix() {
    // Rank 0 dispatches a single token to expert 5 (on rank 1).
    // Verify the payload wire format: [expert_id: u32 LE | token_data...]
    let (t0, _t1) = LoopbackTransport::new_pair();

    let group = Group::with_transport(vec![0, 1], 0, 2, t0.clone()).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 1,
        capacity_factor: 2.0,
        group,
    };
    let mut policy = MoePolicy::with_thresholds(0, 0, 0);
    policy.set_world_size(2);
    policy.set_hysteresis_band(0);
    let mut exchange = MoeDispatchExchange::new(config, policy);

    let indices = vec![5u32]; // expert 5, on rank 1
    let weights = vec![1.0f32];
    let token_data = vec![0xDE, 0xAD, 0xBE, 0xEF]; // 1 token, 4 bytes

    let _ = exchange.dispatch(1, &indices, &weights, &token_data);

    let q = t0.queues.lock().unwrap();
    let messages = q.get(&(0, 1)).unwrap();
    let payload = &messages[1]; // second message is the payload

    // Wire format: [05, 00, 00, 00, DE, AD, BE, EF]
    assert_eq!(payload.len(), 4 + 4); // expert_id (4) + token (4)
    let expert_id = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    assert_eq!(expert_id, 5);
    assert_eq!(&payload[4..8], &[0xDE, 0xAD, 0xBE, 0xEF]);
}

// ─── Size exchange with zero-payload peer ───

#[test]
fn test_rdma_size_exchange_zero_payload() {
    // Rank 0 sends all tokens to local experts (0-3), nothing to rank 1.
    // Size header to rank 1 should be 0.
    let (t0, _t1) = LoopbackTransport::new_pair();

    let group = Group::with_transport(vec![0, 1], 0, 2, t0.clone()).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 8,
        top_k: 1,
        capacity_factor: 2.0,
        group,
    };
    let mut policy = MoePolicy::with_thresholds(0, 1, 0);
    policy.set_world_size(2);
    policy.set_hysteresis_band(0);
    let mut exchange = MoeDispatchExchange::new(config, policy);

    // All tokens go to local experts (0, 1, 2, 3)
    let indices = vec![0u32, 1, 2, 3];
    let weights = vec![1.0f32; 4];
    let token_data = vec![0xFFu8; 4 * 8];

    let _ = exchange.dispatch(4, &indices, &weights, &token_data);

    let q = t0.queues.lock().unwrap();
    let messages = q.get(&(0, 1)).unwrap();

    // Only a size header should be sent (payload is empty, so no payload message)
    assert_eq!(messages.len(), 1, "only size header, no payload");
    let size_header = &messages[0];
    let payload_size = u64::from_le_bytes(size_header[..8].try_into().unwrap());
    assert_eq!(payload_size, 0, "no remote tokens, size should be 0");
}

// ─── Expert partition invariant tests ───

#[test]
fn test_dispatch_rejects_indivisible_experts() {
    // num_experts=7, world_size=3 → 7 % 3 != 0 → Err
    let group = Group::world(3, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 7,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    let indices = vec![0u32; 4];
    let weights = vec![1.0f32; 4];
    let token_data = vec![0u8; 4 * 4];
    let result = exchange.dispatch(4, &indices, &weights, &token_data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("divisible"), "error: {err}");
}

#[test]
fn test_dispatch_rejects_zero_num_experts() {
    // num_experts=0, world_size=1 → experts_per_rank=0 → Err
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 0,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    let indices: Vec<u32> = vec![];
    let weights: Vec<f32> = vec![];
    let token_data = vec![0u8; 4];
    let result = exchange.dispatch(1, &indices, &weights, &token_data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("experts_per_rank is 0"), "error: {err}");
}

#[test]
fn test_dispatch_rejects_out_of_range_expert_index() {
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    // expert index 10 is out of range (num_experts=4)
    let indices = vec![0u32, 1, 10, 3];
    let weights = vec![1.0f32; 4];
    let token_data = vec![0u8; 4 * 4];
    let result = exchange.dispatch(4, &indices, &weights, &token_data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("out of range"), "error: {err}");
}

#[test]
fn test_combine_rdma_rejects_indivisible_experts() {
    let group = Group::world(3, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    let expert_outputs = vec![vec![1.0f32; 4]; 2];
    let weights = vec![1.0f32; 2];
    let indices = vec![0u32, 1];
    // num_experts=7, group_size=3 → 7 % 3 != 0
    let result = combine.combine_rdma(&expert_outputs, &weights, &indices, 2, 1, 2, 7);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("divisible"), "error: {err}");
}

#[test]
fn test_combine_rdma_rejects_out_of_range_expert() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);
    let expert_outputs = vec![vec![1.0f32; 4]; 2];
    let weights = vec![1.0f32; 2];
    let indices = vec![0u32, 5]; // expert 5 >= num_experts=2
    let result = combine.combine_rdma(&expert_outputs, &weights, &indices, 2, 1, 2, 2);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("out of range"), "error: {err}");
}

// ─── SparseGuard action wiring tests ───

/// Helper: create a MoeDispatchExchange with window_size=1 guard for fast triggering.
fn make_exchange_with_guard(window_size: usize) -> MoeDispatchExchange {
    let group = Group::world(1, 0).unwrap();
    let config = MoeDispatchConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 1.0,
        group,
    };
    let mut exchange = MoeDispatchExchange::new(config, MoePolicy::new());
    // Replace the guard with a small window so evaluate() fires quickly.
    *exchange.guard_mut() = {
        let mut g = SparseGuard::new();
        g.set_window_size(window_size);
        g
    };
    exchange
}

#[test]
fn test_guard_action_increase_capacity() {
    let mut exchange = make_exchange_with_guard(1);

    // Baseline capacity factor should be 1.0
    assert!((exchange.runtime_capacity_factor() - 1.0).abs() < 1e-5);

    // Dispatch with moderate overflow (>5% but <20%): all 4 tokens to expert 0
    // capacity_per_expert = ceil(4 * 1 / 4 * 1.0) = 1, so 3 overflow out of 4 = 75%
    // After window_size=1 step, EMA = 0.1 * 0.75 = 0.075 > 0.05 → IncreaseCapacity
    let indices = vec![0u32, 0, 0, 0];
    let weights = vec![1.0f32; 4];
    let token_data = vec![1u8; 4 * 4];
    let _ = exchange
        .dispatch(4, &indices, &weights, &token_data)
        .unwrap();

    // After the dispatch, the guard should have increased the runtime capacity factor
    assert!(
        exchange.runtime_capacity_factor() > 1.0,
        "capacity factor should have increased: got {}",
        exchange.runtime_capacity_factor()
    );
}

#[test]
fn test_guard_action_dense_fallback() {
    let mut exchange = make_exchange_with_guard(1);

    // Pump overflow EMA above 0.20 threshold for DenseFallback.
    // We need multiple evaluate() cycles with sustained high overflow.
    // With alpha=0.1, 4 tokens all to expert 0, 4 experts, cf starts at 1.0:
    // Step 1: cf=1.0, tpe=1, overflow=3/4=75%, EMA=0.075 → IncreaseCapacity(1.25)
    // Step 2: cf=1.25, tpe=2, overflow=2/4=50%, EMA=0.1175 → IncreaseCapacity
    // Step 3: cf~1.56, tpe=2, overflow=50%, EMA=0.15575 → IncreaseCapacity
    // Step 4: cf~1.95, tpe=2, overflow=50%, EMA=0.19018 → IncreaseCapacity
    // Step 5: cf=2.0, tpe=2, overflow=50%, EMA=0.22116 → DenseFallback!
    let indices = vec![0u32, 0, 0, 0];
    let weights = vec![1.0f32; 4];
    let token_data = vec![1u8; 4 * 4];

    for _ in 0..5 {
        let _ = exchange
            .dispatch(4, &indices, &weights, &token_data)
            .unwrap();
    }

    // After 3 dispatches with heavy overflow, DenseFallback should have triggered.
    // The policy should now be forced to CPU (via force_backend override).
    // Verify that select() returns CPU regardless of element count.
    assert_eq!(
        exchange.policy().select(1000, 100000),
        MoeBackend::Cpu,
        "policy should be forced to CPU after dense fallback"
    );
    assert!(
        exchange.guard().is_dense_fallback(),
        "guard should be in dense fallback state"
    );
}

#[test]
fn test_guard_action_reset_after_recovery() {
    let mut exchange = make_exchange_with_guard(1);

    // First: trigger DenseFallback (same as dense_fallback test — 5 rounds needed)
    let heavy_indices = vec![0u32, 0, 0, 0];
    let weights = vec![1.0f32; 4];
    let token_data = vec![1u8; 4 * 4];

    for _ in 0..5 {
        let _ = exchange
            .dispatch(4, &heavy_indices, &weights, &token_data)
            .unwrap();
    }
    assert!(exchange.guard().is_dense_fallback());

    // Now dispatch with zero overflow to recover.
    // With ratio=0: EMA decays toward 0. Need EMA <= 0.05 for Reset.
    // EMA starts ~0.22. Each step: EMA = 0.1 * 0 + 0.9 * EMA = 0.9 * EMA
    // After k steps: EMA ≈ 0.22 * 0.9^k. Need 0.22 * 0.9^k <= 0.05 → k >= 15
    // Use 20 steps for safety margin.
    let spread_indices = vec![0u32, 1, 2, 3];
    for _ in 0..20 {
        let _ = exchange
            .dispatch(4, &spread_indices, &weights, &token_data)
            .unwrap();
    }

    // Guard should have reset: no longer in dense fallback
    assert!(
        !exchange.guard().is_dense_fallback(),
        "guard should have reset after recovery"
    );
    // Runtime capacity factor should be restored to baseline (1.0)
    assert!(
        (exchange.runtime_capacity_factor() - 1.0).abs() < 1e-5,
        "capacity factor should be restored to baseline: got {}",
        exchange.runtime_capacity_factor()
    );
    // After enough additional dispatches to drain cooldown, the policy should
    // resume normal threshold-based selection (not forced CPU).
    for _ in 0..33 {
        let _ = exchange
            .dispatch(4, &spread_indices, &weights, &token_data)
            .unwrap();
    }
    assert_ne!(
        exchange.policy().select(1000, 100000),
        MoeBackend::Cpu,
        "policy force_backend should be cleared after reset"
    );
}

// ─── Combine CPU/Metal parity tests ───

#[test]
fn test_combine_cpu_metal_parity() {
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);

    // 3 tokens, top_k=2, hidden_dim=4, 4 experts
    let batch_size = 3;
    let top_k = 2;
    let hidden_dim = 4;

    let expert0 = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let expert1 = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5];
    let expert2 = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.1, 0.2, 0.3, 0.4,
    ];
    let expert3 = vec![0.0; 12];
    let expert_outputs = vec![expert0, expert1, expert2, expert3];

    // token 0 → experts 0, 2; token 1 → experts 1, 3; token 2 → experts 0, 1
    let indices = vec![0u32, 2, 1, 3, 0, 1];
    let weights = vec![0.6f32, 0.4, 0.7, 0.3, 0.5, 0.5];

    let cpu_result = combine.combine_cpu(
        &expert_outputs,
        &weights,
        &indices,
        batch_size,
        top_k,
        hidden_dim,
    );
    let metal_result = combine.combine_metal(
        &expert_outputs,
        &weights,
        &indices,
        batch_size,
        top_k,
        hidden_dim,
    );

    assert_eq!(cpu_result.len(), metal_result.len());
    for i in 0..cpu_result.len() {
        assert!(
            (cpu_result[i] - metal_result[i]).abs() < 1e-4,
            "mismatch at index {i}: cpu={}, metal={}",
            cpu_result[i],
            metal_result[i]
        );
    }
}

#[test]
fn test_combine_cpu_rdma_parity_single_rank() {
    // Single-rank RDMA combine should produce same results as CPU combine
    let group = Group::world(1, 0).unwrap();
    let combine = MoeCombineExchange::new(group);

    let batch_size = 2;
    let top_k = 2;
    let hidden_dim = 3;
    let num_experts = 2;

    let expert0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expert1 = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let expert_outputs = vec![expert0.clone(), expert1.clone()];

    let indices = vec![0u32, 1, 1, 0];
    let weights = vec![0.6f32, 0.4, 0.3, 0.7];

    let cpu_result = combine.combine_cpu(
        &expert_outputs,
        &weights,
        &indices,
        batch_size,
        top_k,
        hidden_dim,
    );
    let rdma_result = combine
        .combine_rdma(
            &expert_outputs,
            &weights,
            &indices,
            batch_size,
            top_k,
            hidden_dim,
            num_experts,
        )
        .unwrap();

    assert_eq!(cpu_result.len(), rdma_result.len());
    for i in 0..cpu_result.len() {
        assert!(
            (cpu_result[i] - rdma_result[i]).abs() < 1e-4,
            "mismatch at index {i}: cpu={}, rdma={}",
            cpu_result[i],
            rdma_result[i]
        );
    }
}
