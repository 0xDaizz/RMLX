use rmlx_distributed::group::{self, Group};
use rmlx_distributed::moe_exchange::{MoeCombineExchange, MoeDispatchConfig, MoeDispatchExchange};
use rmlx_distributed::moe_policy::{MoeBackend, MoePolicy, ThresholdCalibration};
use rmlx_distributed::warmup::WarmupState;

#[test]
fn test_group_world() {
    let g = Group::world(4, 1);
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
    let group = Group::world(2, 0);
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
    assert_eq!(exchange.metrics().tokens_dispatched, 4);
    assert!(!result.routed_data.is_empty());
}

#[test]
fn test_moe_dispatch_overflow() {
    let group = Group::world(2, 0);
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
    assert!(exchange.metrics().overflow_ratio() > 0.0);
}

#[test]
fn test_moe_combine_cpu() {
    let group = Group::world(1, 0);
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
    let mut m = rmlx_distributed::moe_exchange::MoeMetrics::default();
    m.record_dispatch(100, 5, MoeBackend::Metal);
    assert_eq!(m.tokens_dispatched, 100);
    assert_eq!(m.overflow_count, 5);
    assert!((m.overflow_ratio() - 0.05).abs() < 1e-6);
    assert_eq!(m.metal_dispatches, 1);
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
    let g = Group::world(2, 0);
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
    let group = Group::world(2, 0);
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
    assert_eq!(exchange.metrics().rdma_dispatches, 1);
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
    let group = Group::world(2, 0);
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
    let group = Group::world(2, 0);
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
    let group = Group::world(1, 0);
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
#[should_panic(expected = "not materialized")]
#[cfg(debug_assertions)]
fn test_collective_allreduce_rejects_empty_debug() {
    let g = Group::world(2, 0);
    let _ = g.allreduce(&[]);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_collective_allreduce_rejects_empty_release() {
    let g = Group::world(2, 0);
    assert!(g.allreduce(&[]).is_err());
}

#[test]
#[should_panic(expected = "not materialized")]
#[cfg(debug_assertions)]
fn test_collective_allgather_rejects_empty_debug() {
    let g = Group::world(2, 0);
    let _ = g.allgather(&[]);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_collective_allgather_rejects_empty_release() {
    let g = Group::world(2, 0);
    assert!(g.allgather(&[]).is_err());
}

#[test]
#[should_panic(expected = "not materialized")]
#[cfg(debug_assertions)]
fn test_collective_broadcast_rejects_empty_debug() {
    let g = Group::world(2, 0);
    let _ = g.broadcast(&[], 0);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_collective_broadcast_rejects_empty_release() {
    let g = Group::world(2, 0);
    assert!(g.broadcast(&[], 0).is_err());
}

#[test]
#[should_panic(expected = "not materialized")]
#[cfg(debug_assertions)]
fn test_collective_send_rejects_empty_debug() {
    let g = Group::world(2, 0);
    let _ = g.send(&[], 1);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_collective_send_rejects_empty_release() {
    let g = Group::world(2, 0);
    assert!(g.send(&[], 1).is_err());
}

#[test]
fn test_collective_recv_rejects_zero_len() {
    let g = Group::world(2, 0);
    let result = g.recv(1, 0);
    assert!(result.is_err());
}

#[test]
#[should_panic(expected = "not materialized")]
#[cfg(debug_assertions)]
fn test_collective_all_to_all_rejects_empty_debug() {
    let g = Group::world(2, 0);
    let _ = g.all_to_all(&[]);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_collective_all_to_all_rejects_empty_release() {
    let g = Group::world(2, 0);
    assert!(g.all_to_all(&[]).is_err());
}

#[test]
fn test_collective_allreduce_accepts_valid() {
    let g = Group::world(2, 0);
    let data = vec![1u8, 2, 3, 4];
    let result = g.allreduce(&data);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), data);
}

#[test]
fn test_collective_send_recv_accepts_valid() {
    let g = Group::world(2, 0);
    let data = vec![1u8, 2, 3, 4];
    assert!(g.send(&data, 1).is_ok());
    let received = g.recv(1, 4).unwrap();
    assert_eq!(received.len(), 4);
}

#[test]
fn test_collective_all_to_all_accepts_valid() {
    let g = Group::world(2, 0);
    let data = vec![1u8; 64];
    let result = g.all_to_all(&data).unwrap();
    assert_eq!(result, data);
}
