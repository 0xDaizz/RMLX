//! GPU Pipeline Performance Benchmark
//!
//! Measures the impact of command buffer batching and async execution
//! on single-layer transformer forward pass performance.
//!
//! Metrics tracked:
//! - Command buffer count (baseline vs pipelined)
//! - Sync point count (baseline vs pipelined)
//! - Per-token wall-clock latency
//! - Statistical summary: mean, std, p50, p95, p99
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench pipeline_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::batcher::{reset_counters, total_cbs_created, total_encoders_created};
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_nn::{Attention, AttentionConfig, FeedForward, Linear, LinearConfig, TransformerBlock};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 11008; // Llama-style 2.7x hidden
const SEQ_LEN: usize = 1; // Single-token decode (latency-sensitive)
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
const MAX_SEQ_LEN: usize = 2048;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 50;

// ---------------------------------------------------------------------------
// Statistics
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
        assert!(n > 0, "need at least one sample");

        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = micros.iter().sum();
        let mean = sum / n as f64;

        let variance: f64 = micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let p50 = percentile(&micros, 50.0);
        let p95 = percentile(&micros, 95.0);
        let p99 = percentile(&micros, 99.0);
        let min = micros[0];
        let max = micros[n - 1];

        Stats {
            mean,
            std_dev,
            p50,
            p95,
            p99,
            min,
            max,
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
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:8.1}us  std={:7.1}us  p50={:8.1}us  p95={:8.1}us  p99={:8.1}us  \
             min={:8.1}us  max={:8.1}us  (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.p99, self.min, self.max, self.count,
        )
    }
}

// ---------------------------------------------------------------------------
// CB tracking helper
// ---------------------------------------------------------------------------

struct CbSnapshot {
    cbs: u64,
    encoders: u64,
}

fn snapshot_counters() -> CbSnapshot {
    CbSnapshot {
        cbs: total_cbs_created(),
        encoders: total_encoders_created(),
    }
}

fn delta_counters(before: &CbSnapshot) -> (u64, u64) {
    let cbs = total_cbs_created() - before.cbs;
    let encoders = total_encoders_created() - before.encoders;
    (cbs, encoders)
}

// ---------------------------------------------------------------------------
// Weight initialisation helpers
// ---------------------------------------------------------------------------

/// Create a random-ish f32 array (deterministic, not cryptographic).
/// Uses a simple LCG seeded by shape to produce varied but reproducible data.
fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut state = seed;
    for _ in 0..numel {
        // LCG: fast, deterministic, good enough for benchmark weights
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to small float range [-0.02, 0.02] (typical init scale)
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        data.push(val as f32);
    }
    Array::from_slice(device, &data, shape.to_vec())
}

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

// ---------------------------------------------------------------------------
// Build a single TransformerBlock with real weights
// ---------------------------------------------------------------------------

fn build_transformer_block(device: &metal::Device) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    // Attention projections
    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, 2);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, 3);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 4);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    // Dense FFN (SwiGLU)
    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 5);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 6);
    let down_proj = make_linear(device, INTERMEDIATE_DIM, HIDDEN_SIZE, 7);
    let ffn = FeedForward::Dense {
        gate_proj,
        up_proj,
        down_proj,
    };

    // Norm weights (ones -- standard init)
    let norm1_weight = Array::ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = Array::ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Benchmark runners
// ---------------------------------------------------------------------------

fn bench_baseline(
    block: &TransformerBlock,
    input: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    iters: usize,
) -> (Vec<Duration>, Vec<(u64, u64)>) {
    let mut latencies = Vec::with_capacity(iters);
    let mut cb_counts = Vec::with_capacity(iters);

    for _ in 0..iters {
        reset_counters();
        let snap = snapshot_counters();

        let start = Instant::now();
        let _out = block
            .forward(
                input, None, // cos_freqs
                None, // sin_freqs
                None, // mask
                None, // cache
                registry, queue,
            )
            .expect("baseline forward failed");
        let elapsed = start.elapsed();

        let (cbs, encs) = delta_counters(&snap);
        latencies.push(elapsed);
        cb_counts.push((cbs, encs));
    }
    (latencies, cb_counts)
}

fn bench_pipelined(
    block: &TransformerBlock,
    input: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    event: &GpuEvent,
    iters: usize,
) -> (Vec<Duration>, Vec<(u64, u64)>) {
    let mut latencies = Vec::with_capacity(iters);
    let mut cb_counts = Vec::with_capacity(iters);

    for _ in 0..iters {
        reset_counters();
        let snap = snapshot_counters();

        let start = Instant::now();
        let _out = block
            .forward_pipelined(
                input, None, // cos_freqs
                None, // sin_freqs
                None, // mask
                None, // cache
                registry, queue, event,
            )
            .expect("pipelined forward failed");
        let elapsed = start.elapsed();

        let (cbs, encs) = delta_counters(&snap);
        latencies.push(elapsed);
        cb_counts.push((cbs, encs));

        // Reset event for next iteration
        event.reset();
    }
    (latencies, cb_counts)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_separator(width: usize) {
    println!("{}", "-".repeat(width));
}

fn print_header() {
    let width = 120;
    println!();
    print_separator(width);
    println!(
        "  RMLX GPU Pipeline Benchmark  --  seq_len={}  hidden={}  heads={}/{}  head_dim={}",
        SEQ_LEN, HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );
    print_separator(width);
}

fn print_cb_comparison(baseline_cbs: &[(u64, u64)], pipelined_cbs: &[(u64, u64)]) {
    // Average CB and encoder counts
    let avg = |data: &[(u64, u64)], idx: usize| -> f64 {
        let sum: u64 = data.iter().map(|x| if idx == 0 { x.0 } else { x.1 }).sum();
        sum as f64 / data.len() as f64
    };

    let base_cbs = avg(baseline_cbs, 0);
    let base_encs = avg(baseline_cbs, 1);
    let pipe_cbs = avg(pipelined_cbs, 0);
    let pipe_encs = avg(pipelined_cbs, 1);

    let cb_reduction = if base_cbs > 0.0 {
        (1.0 - pipe_cbs / base_cbs) * 100.0
    } else {
        0.0
    };
    let enc_reduction = if base_encs > 0.0 {
        (1.0 - pipe_encs / base_encs) * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "  [1] Command Buffer Count (per forward pass, averaged over {} iters)",
        baseline_cbs.len()
    );
    println!();
    println!(
        "      {:>30}  {:>10}  {:>10}  {:>12}",
        "", "CBs", "Encoders", "Enc/CB"
    );
    println!(
        "      {:>30}  {:>10.1}  {:>10.1}  {:>12.1}",
        "Baseline (forward)",
        base_cbs,
        base_encs,
        if base_cbs > 0.0 {
            base_encs / base_cbs
        } else {
            0.0
        }
    );
    println!(
        "      {:>30}  {:>10.1}  {:>10.1}  {:>12.1}",
        "Pipelined (forward_pipelined)",
        pipe_cbs,
        pipe_encs,
        if pipe_cbs > 0.0 {
            pipe_encs / pipe_cbs
        } else {
            0.0
        }
    );
    println!(
        "      {:>30}  {:>9.1}%  {:>9.1}%",
        "Reduction", cb_reduction, enc_reduction
    );
}

fn print_sync_comparison(baseline_cbs: &[(u64, u64)], pipelined_cbs: &[(u64, u64)]) {
    // In the baseline path, each op creates and commits its own CB with
    // waitUntilCompleted(), so the number of sync points == number of CBs.
    // In the pipelined path, sync points are reduced (ideally 1 final sync).
    let avg_base_syncs: f64 =
        baseline_cbs.iter().map(|(cbs, _)| *cbs as f64).sum::<f64>() / baseline_cbs.len() as f64;
    let avg_pipe_syncs: f64 = pipelined_cbs
        .iter()
        .map(|(cbs, _)| *cbs as f64)
        .sum::<f64>()
        / pipelined_cbs.len() as f64;

    let reduction = if avg_base_syncs > 0.0 {
        (1.0 - avg_pipe_syncs / avg_base_syncs) * 100.0
    } else {
        0.0
    };

    println!();
    println!("  [2] Sync Point Count (CPU-GPU round-trips per forward pass)");
    println!();
    println!(
        "      {:>30}  {:>10.1}",
        "Baseline (per-op commit+wait)", avg_base_syncs
    );
    println!(
        "      {:>30}  {:>10.1}",
        "Pipelined (batched CBs)", avg_pipe_syncs
    );
    println!("      {:>30}  {:>9.1}%", "Reduction", reduction);
}

fn print_latency_comparison(baseline_stats: &Stats, pipelined_stats: &Stats) {
    let speedup_mean = if pipelined_stats.mean > 0.0 {
        baseline_stats.mean / pipelined_stats.mean
    } else {
        0.0
    };
    let speedup_p50 = if pipelined_stats.p50 > 0.0 {
        baseline_stats.p50 / pipelined_stats.p50
    } else {
        0.0
    };
    let latency_reduction = if baseline_stats.mean > 0.0 {
        (1.0 - pipelined_stats.mean / baseline_stats.mean) * 100.0
    } else {
        0.0
    };

    println!();
    println!("  [3] Per-Token Latency (us)");
    println!();
    println!("      Baseline:   {}", baseline_stats);
    println!("      Pipelined:  {}", pipelined_stats);
    println!();
    println!(
        "      Speedup (mean): {:.2}x    Latency reduction: {:.1}%",
        speedup_mean, latency_reduction
    );
    println!("      Speedup (p50):  {:.2}x", speedup_p50);
}

fn print_statistical_summary(baseline_stats: &Stats, pipelined_stats: &Stats) {
    println!();
    println!("  [4] Statistical Reliability");
    println!();
    println!(
        "      {:>20}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "", "Mean (us)", "Std (us)", "P50 (us)", "P95 (us)", "P99 (us)"
    );
    println!(
        "      {:>20}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}",
        "Baseline",
        baseline_stats.mean,
        baseline_stats.std_dev,
        baseline_stats.p50,
        baseline_stats.p95,
        baseline_stats.p99
    );
    println!(
        "      {:>20}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}",
        "Pipelined",
        pipelined_stats.mean,
        pipelined_stats.std_dev,
        pipelined_stats.p50,
        pipelined_stats.p95,
        pipelined_stats.p99
    );

    // Coefficient of variation (lower is more stable)
    let cv_base = if baseline_stats.mean > 0.0 {
        baseline_stats.std_dev / baseline_stats.mean * 100.0
    } else {
        0.0
    };
    let cv_pipe = if pipelined_stats.mean > 0.0 {
        pipelined_stats.std_dev / pipelined_stats.mean * 100.0
    } else {
        0.0
    };
    println!();
    println!(
        "      Coefficient of variation:  Baseline={:.1}%  Pipelined={:.1}%",
        cv_base, cv_pipe
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // --- Device setup ---
    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    let device_name = gpu.name().to_string();
    let unified_mem = gpu.has_unified_memory();
    println!("Device: {} (unified_memory={})", device_name, unified_mem);

    let registry = KernelRegistry::new(gpu);

    // Register all JIT kernels
    ops::register_all(&registry).expect("kernel registration failed");

    // Get device and queue from registry (registry now owns the GpuDevice)
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    // --- Build model components ---
    println!(
        "Building transformer block (hidden={}, heads={}/{}, head_dim={})...",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );

    let block = build_transformer_block(device);

    // --- Create input tensor ---
    let input = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 42);

    // --- GpuEvent for pipelined path ---
    let event = GpuEvent::new(device);

    // --- Warmup phase ---
    println!("Warming up ({} iterations each)...", WARMUP_ITERS);

    let _ = bench_baseline(&block, &input, &registry, &queue, WARMUP_ITERS);
    let _ = bench_pipelined(&block, &input, &registry, &queue, &event, WARMUP_ITERS);

    // --- Measurement phase ---
    println!("Benchmarking ({} iterations each)...", BENCH_ITERS);

    let (baseline_latencies, baseline_cbs) =
        bench_baseline(&block, &input, &registry, &queue, BENCH_ITERS);
    let (pipelined_latencies, pipelined_cbs) =
        bench_pipelined(&block, &input, &registry, &queue, &event, BENCH_ITERS);

    let baseline_stats = Stats::from_durations(&baseline_latencies);
    let pipelined_stats = Stats::from_durations(&pipelined_latencies);

    // --- Report ---
    print_header();
    print_cb_comparison(&baseline_cbs, &pipelined_cbs);
    print_sync_comparison(&baseline_cbs, &pipelined_cbs);
    print_latency_comparison(&baseline_stats, &pipelined_stats);
    print_statistical_summary(&baseline_stats, &pipelined_stats);

    println!();
    print_separator(120);
    println!("  Benchmark complete.");
    print_separator(120);
    println!();
}
