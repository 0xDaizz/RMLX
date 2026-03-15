//! Expert Parallelism (EP) Benchmark — **production path** optimized.
//!
//! Uses single-CB encoding throughout:
//! - Single expert FFN: `forward_into_cb` + `fused_silu_mul_into_cb` in one command buffer
//! - MoE grouped forward: `ExpertGroup::grouped_forward()` — single CB for all experts
//! - EP-2 simulation: same grouped forward for local expert partition
//!
//! Measures MoE layer components at Qwen 3.5 MoE A22B configuration:
//! - 8 experts, hidden=3584, intermediate=18944, top_k=2
//! - Router (gate) latency [requires CPU sync for topk]
//! - Expert FFN: gate_up + SiLU + down per expert [single-CB]
//! - Total MoE layer (Grouped vs GatherMM strategies)
//! - EP-2 simulation: 4 experts per rank (half experts, half compute)
//! - Real RDMA token exchange: actual 2-node all_to_all over TB5
//!
//! Token counts per expert: M = [1, 4, 8, 16, 32]
//! MoE/EP benchmarks also run prefill sizes: M = [128, 512]
//!
//! Environment variables:
//!   RMLX_BENCH_WARMUP  — warmup iterations (default: 5)
//!   RMLX_BENCH_ITERS   — measurement iterations (default: 30)
//!
//! Run (single-node):
//!   cargo bench -p rmlx-nn --bench ep_bench
//! Run (2-node with RDMA):
//!   cargo bench -p rmlx-nn --bench ep_bench --features distributed
//! Run with custom iteration counts:
//!   RMLX_BENCH_WARMUP=10 RMLX_BENCH_ITERS=50 cargo bench -p rmlx-nn --bench ep_bench

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_nn::{Expert, ExpertGroup, Linear, LinearConfig, MoeConfig, MoeLayer, MoeStrategy};

// ─── Qwen 3.5 MoE A22B config ───
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const HIDDEN_DIM: usize = 3584;
const INTERMEDIATE_DIM: usize = 18944;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 30;

// ─── Token counts to benchmark (tokens routed per expert) ───
const M_VALUES: &[usize] = &[1, 4, 8, 16, 32];
// Extended M values including prefill sizes for MoE/EP benchmarks (fair comparison with MLX EP bench)
const M_VALUES_WITH_PREFILL: &[usize] = &[1, 4, 8, 16, 32, 128, 512];

// ---------------------------------------------------------------------------
// Stats helper (same as distributed_bench)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    min: f64,
    max: f64,
    count: usize,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = micros.iter().sum();
        let mean = sum / n as f64;
        let variance: f64 = micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let p50 = percentile(&micros, 50.0);
        let p95 = percentile(&micros, 95.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            min: micros[0],
            max: micros[n - 1],
            count: n,
        }
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - (rank - lower as f64)) + sorted[upper] * (rank - lower as f64)
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:8.1}us std={:7.1}us p50={:8.1}us p95={:8.1}us min={:8.1}us max={:8.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Random array generation (f16 — realistic inference dtype)
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign as u16
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | frac) as u16
    }
}

fn rand_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
    let numel: usize = shape.iter().product();
    let mut f16_bytes = Vec::with_capacity(numel * 2);
    let mut state = seed;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        let h = f32_to_f16_bits(val as f32);
        f16_bytes.extend_from_slice(&h.to_le_bytes());
    }
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// Layer construction helpers
// ---------------------------------------------------------------------------

fn make_linear(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    in_f: usize,
    out_f: usize,
    seed: u64,
) -> Linear {
    let weight = rand_array(device, &[out_f, in_f], seed);
    Linear::from_arrays(
        LinearConfig {
            in_features: in_f,
            out_features: out_f,
            has_bias: false,
        },
        weight,
        None,
    )
    .expect("linear from_arrays")
}

fn make_expert(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hidden: usize,
    inter: usize,
    seed_base: u64,
) -> Expert {
    Expert {
        gate_proj: make_linear(device, hidden, inter, seed_base),
        up_proj: make_linear(device, hidden, inter, seed_base + 1),
        down_proj: make_linear(device, inter, hidden, seed_base + 2),
    }
}

fn make_gate(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hidden: usize,
    num_experts: usize,
    seed: u64,
) -> Linear {
    make_linear(device, hidden, num_experts, seed)
}

fn build_moe_layer(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    num_experts: usize,
    hidden: usize,
    inter: usize,
    strategy: MoeStrategy,
) -> MoeLayer {
    let config = MoeConfig {
        num_experts,
        num_experts_per_token: TOP_K,
        hidden_dim: hidden,
        intermediate_dim: inter,
        capacity_factor: 1.0,
        enable_fp8: false,
    };
    let gate = make_gate(device, hidden, num_experts, 1000);
    let experts: Vec<Expert> = (0..num_experts)
        .map(|i| make_expert(device, hidden, inter, 2000 + i as u64 * 10))
        .collect();
    MoeLayer::from_layers(config, gate, experts)
        .expect("MoeLayer")
        .with_strategy(strategy)
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

fn run_bench<F>(label: &str, warmup: usize, iters: usize, mut f: F) -> Stats
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    let mut latencies = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        latencies.push(start.elapsed());
    }
    let stats = Stats::from_durations(&latencies);
    println!("  {:50} {}", label, stats);
    stats
}

// ---------------------------------------------------------------------------
// Benchmark: Router (gate) latency [requires CPU sync]
// ---------------------------------------------------------------------------

fn bench_router(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== Router (gate) latency [requires CPU sync] ===");

    let gate = make_gate(device, HIDDEN_DIM, NUM_EXPERTS, 500);

    for &m in M_VALUES {
        let seq_len = m * NUM_EXPERTS / TOP_K; // total tokens to produce m per expert on average
        let input = rand_array(device, &[seq_len, HIDDEN_DIM], 42 + m as u64);
        let label = format!("gate [{}x{}->{}] + topk", seq_len, HIDDEN_DIM, NUM_EXPERTS);
        run_bench(&label, warmup, iters, || {
            let gate_logits = gate.forward(&input, registry, queue).expect("gate forward");
            // topk_route requires f32 logits
            let gate_logits_f32 =
                ops::copy::copy_cast(registry, &gate_logits, DType::Float32, queue)
                    .expect("cast to f32");
            let _ = ops::topk_route::gpu_topk_route(registry, &gate_logits_f32, TOP_K, None, queue)
                .expect("topk_route");
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Single expert FFN [single-CB] (SwiGLU)
// ---------------------------------------------------------------------------

fn bench_single_expert_ffn(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== Single expert FFN [single-CB] (SwiGLU: gate*up -> silu -> down) ===");

    let mut expert = make_expert(device, HIDDEN_DIM, INTERMEDIATE_DIM, 100);
    expert
        .gate_proj
        .prepare_weight_t(registry, queue)
        .expect("prepare gate_proj");
    expert
        .up_proj
        .prepare_weight_t(registry, queue)
        .expect("prepare up_proj");
    expert
        .down_proj
        .prepare_weight_t(registry, queue)
        .expect("prepare down_proj");

    for &m in M_VALUES_WITH_PREFILL {
        let input = rand_array(device, &[m, HIDDEN_DIM], 42 + m as u64);
        let label = format!("expert_ffn [M={}] 3584->18944->3584", m);
        run_bench(&label, warmup, iters, || {
            let cb = queue.commandBufferWithUnretainedReferences().unwrap();
            let gate_out = expert
                .gate_proj
                .forward_into_cb(&input, registry, &cb)
                .unwrap();
            let up_out = expert
                .up_proj
                .forward_into_cb(&input, registry, &cb)
                .unwrap();
            let hidden =
                ops::fused::fused_silu_mul_into_cb(registry, &gate_out, &up_out, &cb).unwrap();
            let _ = expert
                .down_proj
                .forward_into_cb(&hidden, registry, &cb)
                .unwrap();
            cb.commit();
            cb.waitUntilCompleted();
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: MoE grouped forward [production path] (single-CB, all experts)
// ---------------------------------------------------------------------------

fn bench_moe_per_expert(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== MoE grouped forward [production path] (single-CB, 8 experts) ===");

    let experts: Vec<Expert> = (0..NUM_EXPERTS)
        .map(|i| make_expert(device, HIDDEN_DIM, INTERMEDIATE_DIM, 2000 + i as u64 * 10))
        .collect();
    let expert_group =
        ExpertGroup::from_experts(&experts, registry, queue).expect("ExpertGroup::from_experts");

    for &m in M_VALUES_WITH_PREFILL {
        let seq_len = m * NUM_EXPERTS / TOP_K;
        // Create per-expert input batches (simulating router dispatch)
        let expert_inputs: Vec<Array> = (0..NUM_EXPERTS)
            .map(|i| rand_array(device, &[m, HIDDEN_DIM], 42 + m as u64 + i as u64 * 100))
            .collect();
        let expert_input_refs: Vec<(usize, &Array)> = expert_inputs.iter().enumerate().collect();

        let label = format!("moe_grouped [seq={}, ~{} tok/expert]", seq_len, m);
        run_bench(&label, warmup, iters, || {
            let _ = expert_group
                .grouped_forward(&expert_input_refs, registry, queue)
                .expect("grouped fwd");
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: GatherMM strategy [multi-CB] (batched GEMM across experts)
// ---------------------------------------------------------------------------

fn bench_moe_gather_mm(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== MoE GatherMM strategy [multi-CB] (batched GEMM, 8 experts) ===");

    let moe = build_moe_layer(
        device,
        NUM_EXPERTS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        MoeStrategy::GatherMM,
    );

    for &m in M_VALUES_WITH_PREFILL {
        let seq_len = m * NUM_EXPERTS / TOP_K;
        let input = rand_array(device, &[seq_len, HIDDEN_DIM], 42 + m as u64);
        let label = format!("moe_gather_mm  [seq={}, ~{} tok/expert]", seq_len, m);
        run_bench(&label, warmup, iters, || {
            let _ = moe.forward(&input, registry, queue).expect("moe fwd");
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: gather_mm kernel (raw, no MoE routing)
// ---------------------------------------------------------------------------

fn bench_gather_mm_raw(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== Raw gather_mm kernel (no routing overhead) ===");

    // Stacked expert weights: [8, K, N]
    // gate_proj: K=3584, N=18944
    let gate_weights = rand_array(device, &[NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM], 800);

    for &m in M_VALUES {
        let batch = m * NUM_EXPERTS / TOP_K; // total (token, expert) assignments
                                             // x: [batch, 1, K] - one token per batch element
        let x = rand_array(device, &[batch, 1, HIDDEN_DIM], 42 + m as u64);
        // indices: [batch] - uniformly distributed across experts
        let indices_data: Vec<u32> = (0..batch).map(|i| (i % NUM_EXPERTS) as u32).collect();
        let indices = Array::from_slice(device, &indices_data, vec![batch]);

        let label = format!("gather_mm [batch={}, 3584->18944]", batch);
        run_bench(&label, warmup, iters, || {
            let _ = ops::gather_mm::gather_mm(registry, &x, &gate_weights, &indices, queue)
                .expect("gather_mm");
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: EP-2 simulation [production path] (4 experts per rank)
// ---------------------------------------------------------------------------

fn bench_ep2_simulation(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== EP-2 simulation [production path]: 4 experts per rank (half compute) ===");
    println!("  Simulates rank 0 of EP=2: only experts 0..4 are local.");
    println!("  Uses ExpertGroup::grouped_forward() — single CB for all local experts.");

    let ep_experts = NUM_EXPERTS / 2; // 4 experts on this rank

    let experts: Vec<Expert> = (0..ep_experts)
        .map(|i| make_expert(device, HIDDEN_DIM, INTERMEDIATE_DIM, 2000 + i as u64 * 10))
        .collect();
    let expert_group =
        ExpertGroup::from_experts(&experts, registry, queue).expect("ExpertGroup::from_experts");

    for &m in M_VALUES_WITH_PREFILL {
        let seq_len = m * ep_experts / TOP_K;
        let expert_inputs: Vec<Array> = (0..ep_experts)
            .map(|i| rand_array(device, &[m, HIDDEN_DIM], 42 + m as u64 + i as u64 * 100))
            .collect();
        let expert_input_refs: Vec<(usize, &Array)> = expert_inputs.iter().enumerate().collect();

        let label = format!("ep2_grouped [seq={}, 4 experts]", seq_len);
        run_bench(&label, warmup, iters, || {
            let _ = expert_group
                .grouped_forward(&expert_input_refs, registry, queue)
                .expect("ep2 grouped fwd");
        });
    }

    // Also benchmark the GatherMM strategy for EP-2
    println!("\n=== EP-2 simulation with GatherMM [multi-CB] (4 experts per rank) ===");
    let moe_ep_gmm = build_moe_layer(
        device,
        ep_experts,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        MoeStrategy::GatherMM,
    );

    for &m in M_VALUES_WITH_PREFILL {
        let seq_len = m * ep_experts / TOP_K;
        let input = rand_array(device, &[seq_len, HIDDEN_DIM], 42 + m as u64);
        let label = format!("ep2_gather_mm  [seq={}, 4 experts]", seq_len);
        run_bench(&label, warmup, iters, || {
            let _ = moe_ep_gmm
                .forward(&input, registry, queue)
                .expect("ep2 gmm fwd");
        });
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Real RDMA EP token exchange (requires distributed feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
fn bench_ep2_real_exchange(
    group: &rmlx_distributed::group::Group,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    warmup: usize,
    iters: usize,
) {
    println!("\n=== EP-2 real RDMA token exchange (all_to_all) ===");
    println!(
        "  rank={}, world_size={}, has_transport={}",
        group.local_rank(),
        group.world_size(),
        group.has_transport()
    );

    // Measure raw all_to_all latency at different token counts
    for &m in M_VALUES {
        let total_tokens = m * NUM_EXPERTS / TOP_K;
        // Each rank holds total_tokens worth of hidden states
        // all_to_all exchanges half with the peer (EP=2)
        let payload_bytes = total_tokens * HIDDEN_DIM * 2; // f16

        // Create a byte buffer for all_to_all
        let data = vec![0u8; payload_bytes];
        let label = format!(
            "all_to_all [{}tok x {}B = {:.1}KB]",
            total_tokens,
            HIDDEN_DIM * 2,
            payload_bytes as f64 / 1024.0
        );
        run_bench(&label, warmup, iters, || {
            let _ = group.all_to_all(&data).expect("all_to_all");
        });
    }

    // EP-2 end-to-end: local compute + RDMA exchange
    println!(
        "\n=== EP-2 end-to-end [production path]: local compute + real RDMA token exchange ==="
    );

    let ep_experts = NUM_EXPERTS / 2;
    let experts: Vec<Expert> = (0..ep_experts)
        .map(|i| make_expert(device, HIDDEN_DIM, INTERMEDIATE_DIM, 2000 + i as u64 * 10))
        .collect();
    let expert_group =
        ExpertGroup::from_experts(&experts, registry, queue).expect("ExpertGroup::from_experts");

    for &m in M_VALUES {
        let seq_len = m * ep_experts / TOP_K;
        let expert_inputs: Vec<Array> = (0..ep_experts)
            .map(|i| rand_array(device, &[m, HIDDEN_DIM], 42 + m as u64 + i as u64 * 100))
            .collect();
        let expert_input_refs: Vec<(usize, &Array)> = expert_inputs.iter().enumerate().collect();
        let payload_bytes = seq_len * HIDDEN_DIM * 2;
        let dispatch_data = vec![0u8; payload_bytes];

        let label = format!("ep2_e2e [seq={}, 4 local experts + RDMA]", seq_len);
        run_bench(&label, warmup, iters, || {
            // 1. Dispatch: exchange tokens between nodes
            let _ = group.all_to_all(&dispatch_data).expect("dispatch");

            // 2. Local expert compute (single CB via grouped_forward)
            let _ = expert_group
                .grouped_forward(&expert_input_refs, registry, queue)
                .expect("local grouped fwd");

            // 3. Combine: exchange results back
            let _ = group.all_to_all(&dispatch_data).expect("combine");
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    println!(
        "Device: {} (unified_memory={})",
        gpu.name(),
        gpu.has_unified_memory()
    );

    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("kernel registration failed");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    // ── Env var overrides ──
    let warmup: usize = std::env::var("RMLX_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(WARMUP_ITERS);
    let iters: usize = std::env::var("RMLX_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BENCH_ITERS);

    // ── Distributed init ──
    #[cfg(feature = "distributed")]
    let dist_ctx = {
        let config = rmlx_distributed::init::InitConfig::default();
        match rmlx_distributed::init::init(config) {
            Ok(ctx) => {
                println!(
                    "\nDistributed: rank={}/{}, backend={:?}, transport={}",
                    ctx.rank,
                    ctx.world_size,
                    ctx.backend,
                    ctx.group.has_transport()
                );
                Some(ctx)
            }
            Err(e) => {
                println!("\nDistributed init failed (running single-node): {e}");
                None
            }
        }
    };

    println!("\nConfig: Qwen 3.5 MoE A22B MoE layer");
    println!(
        "  experts={}, top_k={}, hidden={}, intermediate={}",
        NUM_EXPERTS, TOP_K, HIDDEN_DIM, INTERMEDIATE_DIM
    );
    println!("  dtype: float16, warmup={}, bench_iters={}", warmup, iters);
    if warmup != WARMUP_ITERS || iters != BENCH_ITERS {
        println!("  (overridden via RMLX_BENCH_WARMUP / RMLX_BENCH_ITERS)");
    }
    println!("  M values (tokens per expert): {:?}", M_VALUES);
    println!(
        "  M values with prefill (MoE/EP):  {:?}",
        M_VALUES_WITH_PREFILL
    );

    // ── 1. Router latency [requires CPU sync] ──
    bench_router(device, &registry, &queue, warmup, iters);

    // ── 2. Single expert FFN [single-CB] ──
    bench_single_expert_ffn(device, &registry, &queue, warmup, iters);

    // ── 3. Full MoE: Grouped forward [production path] ──
    bench_moe_per_expert(device, &registry, &queue, warmup, iters);

    // ── 4. Full MoE: GatherMM strategy [multi-CB] ──
    bench_moe_gather_mm(device, &registry, &queue, warmup, iters);

    // ── 5. Raw gather_mm kernel ──
    bench_gather_mm_raw(device, &registry, &queue, warmup, iters);

    // ── 6. EP-2 simulation [production path] ──
    bench_ep2_simulation(device, &registry, &queue, warmup, iters);

    // ── 7. Real RDMA EP exchange (requires distributed + RDMA transport) ──
    #[cfg(feature = "distributed")]
    if let Some(ref ctx) = dist_ctx {
        if ctx.group.has_transport() && ctx.world_size > 1 {
            bench_ep2_real_exchange(&ctx.group, device, &registry, &queue, warmup, iters);
        }
    }

    // ── Summary ──
    println!("\n========================================================================");
    println!("SUMMARY");
    println!("========================================================================");
    println!("  Grouped:   ExpertGroup::grouped_forward() — single CB for all experts");
    println!("  Single-CB: forward_into_cb + fused_silu_mul_into_cb in one command buffer");
    println!("  GatherMM:  single batched GPU call per projection (gate, up, down)");
    println!("  EP-2 sim:  half the experts (4 of 8) on a single rank");
    #[cfg(feature = "distributed")]
    if let Some(ref ctx) = dist_ctx {
        if ctx.group.has_transport() && ctx.world_size > 1 {
            println!("  EP-2 real: local compute + RDMA all_to_all token exchange");
        }
    }
    println!();
    println!("  Key comparison: Single-CB encodes all expert GEMMs + SwiGLU into one");
    println!("  command buffer (1 commit). Old PerExpert had O(E) individual CB commits.");
    println!("  EP-2 halves the compute but adds network dispatch/combine overhead.");
    #[cfg(feature = "distributed")]
    {
        if dist_ctx
            .as_ref()
            .is_some_and(|c| c.group.has_transport() && c.world_size > 1)
        {
            println!("  RDMA exchange cost is measured above with real TB5 transport.");
        } else {
            println!("  (RDMA transport not available -- run with RMLX_RANK/RMLX_WORLD_SIZE/RMLX_COORDINATOR)");
        }
    }
    #[cfg(not(feature = "distributed"))]
    {
        println!("  (Enable `distributed` feature for real RDMA EP benchmarks)");
    }
    println!();
}
