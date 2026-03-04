//! GPU Pipeline Performance Benchmark
//!
//! Apples-to-apples comparison of single-layer transformer forward pass:
//! - **Baseline**: `forward()` — per-op dispatch, each op commits+waits its own CB
//! - **ExecGraph**: `forward_graph()` — same compute, batched into 6 CBs with
//!   GPU-side event chaining (single CPU sync at the end)
//!
//! Both paths execute identical operations (norm, Q/K/V matmul, RoPE, SDPA,
//! O_proj, residual, FFN). The only difference is dispatch strategy.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench pipeline_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::batcher::reset_counters;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::{ExecGraph, ExecGraphStats};
use rmlx_nn::{Attention, AttentionConfig, FeedForward, Linear, LinearConfig, TransformerBlock};

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 11008;
const SEQ_LEN: usize = 1;
const RMS_NORM_EPS: f32 = 1e-5;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 50;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    p99: f64,
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
        let p99 = percentile(&micros, 99.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            p99,
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
            "mean={:8.1}us std={:7.1}us p50={:8.1}us p95={:8.1}us p99={:8.1}us min={:8.1}us max={:8.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.p99, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Random array generation (deterministic PRNG)
// ---------------------------------------------------------------------------

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut state = seed;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        data.push(val as f32);
    }
    Array::from_slice(device, &data, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Layer construction helpers
// ---------------------------------------------------------------------------

fn make_linear(device: &metal::Device, in_f: usize, out_f: usize, seed: u64) -> Linear {
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

fn build_transformer_block(device: &metal::Device) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, 2);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, 3);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 4);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 5);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 6);
    let down_proj = make_linear(device, INTERMEDIATE_DIM, HIDDEN_SIZE, 7);
    let ffn = FeedForward::Dense {
        gate_proj,
        up_proj,
        down_proj,
    };

    let norm1_weight = Array::ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = Array::ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Benchmark entry point
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
    let queue = device.new_command_queue();

    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}, seq_len={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM, SEQ_LEN
    );

    let mut block = build_transformer_block(device);
    let input = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 42);

    // ---- Warmup baseline ----
    println!("\nWarming up baseline ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        let _ = block.forward(&input, None, None, None, None, &registry, &queue);
    }

    // ---- Benchmark baseline: forward() with per-op commit+wait ----
    println!("Benchmarking baseline ({} iterations)...", BENCH_ITERS);
    let mut baseline_latencies = Vec::with_capacity(BENCH_ITERS);
    let mut baseline_cbs = 0u64;
    for i in 0..BENCH_ITERS {
        reset_counters();
        ops::reset_op_cbs();
        let start = Instant::now();
        let _ = block
            .forward(&input, None, None, None, None, &registry, &queue)
            .expect("forward failed");
        baseline_latencies.push(start.elapsed());
        if i == BENCH_ITERS - 1 {
            baseline_cbs = ops::total_op_cbs();
        }
    }

    // ---- Prepare weights for ExecGraph path ----
    println!("\nPreparing weights for ExecGraph (pre-transposing)...");
    block
        .prepare_weights_for_graph(&registry, &queue)
        .expect("prepare_weights_for_graph failed");

    // ---- Warmup ExecGraph ----
    println!("Warming up ExecGraph ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);
        let _ = block.forward_graph(&input, None, None, None, None, &registry, &mut graph, &queue);
        let _ = graph.sync_and_reset();
    }

    // ---- Benchmark ExecGraph: forward_graph() with batched CBs ----
    println!("Benchmarking ExecGraph ({} iterations)...", BENCH_ITERS);
    let mut graph_latencies = Vec::with_capacity(BENCH_ITERS);
    let mut graph_total_batches = 0usize;
    let mut graph_total_cbs = 0usize;
    let mut graph_total_encoders = 0usize;
    for i in 0..BENCH_ITERS {
        reset_counters();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);

        let start = Instant::now();
        let _ = block
            .forward_graph(&input, None, None, None, None, &registry, &mut graph, &queue)
            .expect("forward_graph failed");
        let _ = graph.sync_and_reset().expect("sync failed");
        graph_latencies.push(start.elapsed());

        if i == BENCH_ITERS - 1 {
            let stats = ExecGraphStats::from_graph(&graph);
            graph_total_batches = stats.total_batches;
            graph_total_cbs = stats.total_cbs;
            graph_total_encoders = stats.total_encoders;
            // Re-run once more to capture stats before reset
            let event2 = GpuEvent::new(device);
            let mut graph2 = ExecGraph::new(&queue, &event2, 32);
            let _ = block.forward_graph(
                &input, None, None, None, None, &registry, &mut graph2, &queue,
            );
            let stats2 = ExecGraphStats::from_graph(&graph2);
            graph_total_batches = stats2.total_batches;
            graph_total_cbs = stats2.total_cbs;
            graph_total_encoders = stats2.total_encoders;
            let _ = graph2.sync_and_reset();
        }
    }

    // ---- Results ----
    let baseline_stats = Stats::from_durations(&baseline_latencies);
    let graph_stats = Stats::from_durations(&graph_latencies);
    let speedup = if graph_stats.mean > 0.0 {
        baseline_stats.mean / graph_stats.mean
    } else {
        0.0
    };

    println!("\n========== Results ==========");
    println!("Baseline (per-op forward):");
    println!("  {}", baseline_stats);
    println!("  Command buffers per forward: {}", baseline_cbs);
    println!();
    println!("ExecGraph (forward_graph):");
    println!("  {}", graph_stats);
    println!(
        "  Batches: {}, CBs: {}, Encoders: {}",
        graph_total_batches, graph_total_cbs, graph_total_encoders
    );
    println!();
    println!("Speedup:        {:.2}x", speedup);
    println!(
        "CB reduction:   {} -> {} ({:.1}% fewer)",
        baseline_cbs,
        graph_total_cbs,
        if baseline_cbs > 0 {
            (1.0 - graph_total_cbs as f64 / baseline_cbs as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("=================================");
}
