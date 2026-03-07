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
use rmlx_nn::{
    Attention, AttentionConfig, FeedForward, LayerKvCache, Linear, LinearConfig, TransformerBlock,
};

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
// f16 random array generation
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        // zero or subnormal
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        // inf or nan
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // overflow → inf
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // underflow → zero
        return (sign << 15) as u16;
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

fn rand_array_f16(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    use rmlx_core::dtype::DType;
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
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
    };

    let norm1_weight = Array::ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = Array::ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

fn make_linear_f16(device: &metal::Device, in_f: usize, out_f: usize, seed: u64) -> Linear {
    let weight = rand_array_f16(device, &[out_f, in_f], seed);
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

fn build_transformer_block_f16(device: &metal::Device) -> TransformerBlock {
    use rmlx_core::dtype::DType;

    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    let q_proj = make_linear_f16(device, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    let k_proj = make_linear_f16(device, HIDDEN_SIZE, kv_size, 2);
    let v_proj = make_linear_f16(device, HIDDEN_SIZE, kv_size, 3);
    let o_proj = make_linear_f16(device, HIDDEN_SIZE, HIDDEN_SIZE, 4);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear_f16(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 5);
    let up_proj = make_linear_f16(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 6);
    let down_proj = make_linear_f16(device, INTERMEDIATE_DIM, HIDDEN_SIZE, 7);
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
    };

    let norm1_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE]; // f16 value 1.0
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };
    let norm2_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE]; // f16 value 1.0
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Per-dispatch profiling helper
// ---------------------------------------------------------------------------

/// Time a single operation: warm up `WARMUP_ITERS` times, then measure
/// `BENCH_ITERS` iterations, each with its own command buffer commit+wait
/// to capture accurate GPU timing per dispatch.
fn profile_op<F>(label: &str, queue: &metal::CommandQueue, mut op: F) -> Stats
where
    F: FnMut(&metal::CommandBufferRef),
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        op(cb);
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        op(cb);
        cb.commit();
        cb.wait_until_completed();
        latencies.push(start.elapsed());
    }

    let stats = Stats::from_durations(&latencies);
    println!("  {:40} {}", label, stats);
    stats
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
        let _ = block.forward_graph(
            &input, None, None, None, None, &registry, &mut graph, &queue,
        );
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
            .forward_graph(
                &input, None, None, None, None, &registry, &mut graph, &queue,
            )
            .expect("forward_graph failed");
        let _ = graph.sync_and_reset().expect("sync failed");
        graph_latencies.push(start.elapsed());

        if i == BENCH_ITERS - 1 {
            // Re-run once more to capture stats before reset
            let event2 = GpuEvent::new(device);
            let mut graph2 = ExecGraph::new(&queue, &event2, 32);
            let _ = block.forward_graph(
                &input,
                None,
                None,
                None,
                None,
                &registry,
                &mut graph2,
                &queue,
            );
            let stats2 = ExecGraphStats::from_graph(&graph2);
            graph_total_batches = stats2.total_batches;
            graph_total_cbs = stats2.total_cbs;
            graph_total_encoders = stats2.total_encoders;
            let _ = graph2.sync_and_reset();
        }
    }

    // ---- Benchmark Single-CB: forward_single_cb() ----
    println!("\nWarming up Single-CB ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_single_cb(&input, None, None, None, None, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    println!("Benchmarking Single-CB ({} iterations)...", BENCH_ITERS);
    let mut single_cb_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_single_cb(&input, None, None, None, None, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
        single_cb_latencies.push(start.elapsed());
    }

    // ---- Prepare weights for 9-dispatch ----
    println!("\nPreparing weights for 9-dispatch (merging QKV and gate+up)...");
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed");
    // Convert weights to StorageModePrivate (GPU-only) for optimal performance
    block.prepare_weights_private(device, &queue);

    // ---- Warmup 9-dispatch ----
    println!("Warming up 9-dispatch ({} iterations)...", WARMUP_ITERS);
    let mut cache_9d = LayerKvCache::preallocated(
        device,
        NUM_KV_HEADS,
        HEAD_DIM,
        2048,
        rmlx_core::dtype::DType::Float32,
    );
    for _ in 0..WARMUP_ITERS {
        cache_9d.seq_len = 0;
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_single_cb_9dispatch(&input, None, None, None, &mut cache_9d, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- Benchmark 9-dispatch ----
    println!("Benchmarking 9-dispatch ({} iterations)...", BENCH_ITERS);
    let mut nine_dispatch_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        cache_9d.seq_len = 0; // Reset position instead of reallocating
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_single_cb_9dispatch(&input, None, None, None, &mut cache_9d, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
        nine_dispatch_latencies.push(start.elapsed());
    }

    // ---- Warmup Concurrent-9-dispatch ----
    println!(
        "\nWarming up Concurrent-9-dispatch ({} iterations)...",
        WARMUP_ITERS
    );
    let mut cache_c9d = LayerKvCache::preallocated(
        device,
        NUM_KV_HEADS,
        HEAD_DIM,
        2048,
        rmlx_core::dtype::DType::Float32,
    );
    for _ in 0..WARMUP_ITERS {
        cache_c9d.seq_len = 0;
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_concurrent_9dispatch(&input, None, None, None, &mut cache_c9d, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- Benchmark Concurrent-9-dispatch ----
    println!(
        "Benchmarking Concurrent-9-dispatch ({} iterations)...",
        BENCH_ITERS
    );
    let mut concurrent_9d_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        cache_c9d.seq_len = 0;
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_concurrent_9dispatch(&input, None, None, None, &mut cache_c9d, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
        concurrent_9d_latencies.push(start.elapsed());
    }

    // ---- Results ----
    let baseline_stats = Stats::from_durations(&baseline_latencies);
    let graph_stats = Stats::from_durations(&graph_latencies);
    let single_cb_stats = Stats::from_durations(&single_cb_latencies);
    let nine_dispatch_stats = Stats::from_durations(&nine_dispatch_latencies);
    let concurrent_9d_stats = Stats::from_durations(&concurrent_9d_latencies);
    let speedup = if graph_stats.mean > 0.0 {
        baseline_stats.mean / graph_stats.mean
    } else {
        0.0
    };
    let single_cb_speedup = if single_cb_stats.mean > 0.0 {
        baseline_stats.mean / single_cb_stats.mean
    } else {
        0.0
    };
    let nine_dispatch_speedup = if nine_dispatch_stats.mean > 0.0 {
        baseline_stats.mean / nine_dispatch_stats.mean
    } else {
        0.0
    };
    let concurrent_9d_speedup = if concurrent_9d_stats.mean > 0.0 {
        baseline_stats.mean / concurrent_9d_stats.mean
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
    println!("Single-CB (forward_single_cb):");
    println!("  {}", single_cb_stats);
    println!("  Command buffers per forward: 1");
    println!();
    println!("9-Dispatch (forward_single_cb_9dispatch):");
    println!("  {}", nine_dispatch_stats);
    println!("  Dispatches per forward: 9");
    println!();
    println!("Concurrent-9-Dispatch (forward_concurrent_9dispatch):");
    println!("  {}", concurrent_9d_stats);
    println!("  Dispatches per forward: 9 (concurrent encoders)");
    println!();
    println!("ExecGraph speedup:  {:.2}x", speedup);
    println!("Single-CB speedup:  {:.2}x", single_cb_speedup);
    println!("9-Dispatch speedup: {:.2}x", nine_dispatch_speedup);
    println!(
        "Concurrent-9-Dispatch speedup: {:.2}x",
        concurrent_9d_speedup
    );
    println!(
        "CB reduction:   {} -> {} -> 1",
        baseline_cbs, graph_total_cbs
    );
    println!("=================================");

    // =====================================================================
    // 9-Dispatch Profiling: per-dispatch GPU timing
    // =====================================================================
    // Each of the 9 dispatches is run in isolation with its own CB to
    // measure individual kernel latencies and identify the bottleneck.
    println!("\n========== 9-Dispatch Profiling ==========");
    println!(
        "Each dispatch: {} warmup + {} measured iterations, separate CB per iteration",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!();

    // KV cache seq_len for SDPA decode — realistic decode-time value
    let kv_seq_len: usize = 128;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Pre-allocate all arrays needed for profiling ---

    // Inputs
    let prof_hidden = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 100); // [1, 4096]
    let prof_hidden_1d = rand_array(device, &[HIDDEN_SIZE], 101); // [4096]
    let norm_weight = rand_array(device, &[HIDDEN_SIZE], 102); // [4096]

    // Merged QKV weight: [6144, 4096] where 6144 = 4096 + 1024 + 1024
    let qkv_dim = HIDDEN_SIZE + NUM_KV_HEADS * HEAD_DIM * 2; // 4096 + 2048 = 6144
    let qkv_weight = rand_array(device, &[qkv_dim, HIDDEN_SIZE], 110);

    // RoPE inputs: [n_batch, seq_len=1, head_dim=128]
    // We apply rope to Q (32 heads) and K (8 heads) = 40 heads total
    let rope_n_batch = NUM_HEADS + NUM_KV_HEADS; // 40
    let rope_input = rand_array(device, &[rope_n_batch, SEQ_LEN, HEAD_DIM], 120);
    // cos/sin freq tables: [max_seq_len, head_dim/2] -- 2D
    let rope_cos = rand_array(device, &[2048, HEAD_DIM / 2], 121);
    let rope_sin = rand_array(device, &[2048, HEAD_DIM / 2], 122);

    // SDPA decode inputs (flat slab layout)
    // q_slab: [NUM_HEADS * HEAD_DIM] = [4096]
    let sdpa_q = rand_array(device, &[NUM_HEADS * HEAD_DIM], 130);
    // k_slab: [NUM_KV_HEADS * kv_seq_len * HEAD_DIM]
    let sdpa_k = rand_array(device, &[NUM_KV_HEADS * kv_seq_len * HEAD_DIM], 131);
    // v_slab: same shape as k_slab
    let sdpa_v = rand_array(device, &[NUM_KV_HEADS * kv_seq_len * HEAD_DIM], 132);

    // O_proj weight [4096, 4096] and bias [4096] (for residual add)
    let oproj_weight = rand_array(device, &[HIDDEN_SIZE, HIDDEN_SIZE], 140);
    let oproj_bias = rand_array(device, &[HIDDEN_SIZE], 141); // residual

    // Merged gate+up weight: [22016, 4096] where 22016 = 11008 * 2
    let gate_up_dim = INTERMEDIATE_DIM * 2; // 22016
    let gate_up_weight = rand_array(device, &[gate_up_dim, HIDDEN_SIZE], 150);

    // SiLU*mul inputs: gate_out [1, 11008] and up_out [1, 11008]
    let silu_gate = rand_array(device, &[SEQ_LEN, INTERMEDIATE_DIM], 160);
    let silu_up = rand_array(device, &[SEQ_LEN, INTERMEDIATE_DIM], 161);

    // Down projection: [4096, 11008] weight, [4096] bias (for residual add)
    let down_weight = rand_array(device, &[HIDDEN_SIZE, INTERMEDIATE_DIM], 170);
    let down_bias = rand_array(device, &[HIDDEN_SIZE], 171); // residual

    // --- Profile each dispatch ---
    let mut all_stats: Vec<(&str, Stats)> = Vec::new();

    // 1. RMSNorm (pre-attention)
    let s = profile_op("1. rms_norm [1,4096]", &queue, |cb| {
        let _ = ops::rms_norm::rms_norm_into_cb(
            &registry,
            &prof_hidden,
            Some(&norm_weight),
            RMS_NORM_EPS,
            cb,
        )
        .expect("rms_norm failed");
    });
    all_stats.push(("1. rms_norm (pre-attn)", s));

    // 2. GEMV merged QKV: [6144, 4096] * [4096]
    let s = profile_op("2. gemv QKV [6144,4096]*[4096]", &queue, |cb| {
        let _ = ops::gemv::gemv_into_cb(&registry, &qkv_weight, &prof_hidden_1d, cb)
            .expect("gemv QKV failed");
    });
    all_stats.push(("2. gemv QKV", s));

    // 3. RoPE: [40, 1, 128]
    let s = profile_op("3. rope [40,1,128]", &queue, |cb| {
        let _ = ops::rope::rope_ext_into_cb(
            &registry,
            &rope_input,
            &rope_cos,
            &rope_sin,
            0,     // offset
            1.0,   // scale
            false, // traditional
            true,  // forward
            cb,
        )
        .expect("rope failed");
    });
    all_stats.push(("3. rope", s));

    // 4. SDPA decode batched: 32 heads, kv_seq_len, head_dim=128
    let s = profile_op(
        &format!("4. sdpa_decode 32h seq={}", kv_seq_len),
        &queue,
        |cb| {
            let _ = ops::sdpa::sdpa_decode_batched_slab_into_cb(
                &registry,
                &sdpa_q,
                &sdpa_k,
                &sdpa_v,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                kv_seq_len,
                None, // no mask
                scale,
                cb,
            )
            .expect("sdpa_decode failed");
        },
    );
    all_stats.push(("4. sdpa_decode", s));

    // 5. GEMV + bias (O_proj + residual): [4096, 4096] * [4096] + [4096]
    let s = profile_op("5. gemv_bias O_proj [4096,4096]*[4096]", &queue, |cb| {
        let _ = ops::gemv::gemv_bias_into_cb(
            &registry,
            &oproj_weight,
            &prof_hidden_1d,
            &oproj_bias,
            cb,
        )
        .expect("gemv_bias O_proj failed");
    });
    all_stats.push(("5. gemv_bias O_proj+res", s));

    // 6. RMSNorm (pre-FFN)
    let s = profile_op("6. rms_norm [1,4096]", &queue, |cb| {
        let _ = ops::rms_norm::rms_norm_into_cb(
            &registry,
            &prof_hidden,
            Some(&norm_weight),
            RMS_NORM_EPS,
            cb,
        )
        .expect("rms_norm failed");
    });
    all_stats.push(("6. rms_norm (pre-FFN)", s));

    // 7. GEMV merged gate+up: [22016, 4096] * [4096]
    let s = profile_op("7. gemv gate+up [22016,4096]*[4096]", &queue, |cb| {
        let _ = ops::gemv::gemv_into_cb(&registry, &gate_up_weight, &prof_hidden_1d, cb)
            .expect("gemv gate+up failed");
    });
    all_stats.push(("7. gemv gate+up", s));

    // 8. Fused SiLU*mul: [1, 11008] * [1, 11008]
    let s = profile_op("8. fused_silu_mul [1,11008]", &queue, |cb| {
        let _ = ops::fused::fused_silu_mul_into_cb(&registry, &silu_gate, &silu_up, cb)
            .expect("fused_silu_mul failed");
    });
    all_stats.push(("8. fused_silu_mul", s));

    // 9. GEMV + bias (down_proj + residual): [4096, 11008] * [11008] + [4096]
    let down_vec = rand_array(device, &[INTERMEDIATE_DIM], 180); // [11008]
    let s = profile_op("9. gemv_bias down [4096,11008]*[11008]", &queue, |cb| {
        let _ = ops::gemv::gemv_bias_into_cb(&registry, &down_weight, &down_vec, &down_bias, cb)
            .expect("gemv_bias down failed");
    });
    all_stats.push(("9. gemv_bias down+res", s));

    // --- Summary ---
    println!();
    println!("---------- Per-Dispatch Summary ----------");
    let total_mean: f64 = all_stats.iter().map(|(_, s)| s.mean).sum();
    for (label, s) in &all_stats {
        let pct = if total_mean > 0.0 {
            s.mean / total_mean * 100.0
        } else {
            0.0
        };
        println!("  {:30} {:8.1}us  ({:5.1}%)", label, s.mean, pct);
    }
    println!(
        "  {:30} {:8.1}us  (100.0%)",
        "TOTAL (sum of means)", total_mean
    );

    // Identify bottleneck
    if let Some((label, s)) = all_stats
        .iter()
        .max_by(|a, b| a.1.mean.partial_cmp(&b.1.mean).unwrap())
    {
        let pct = s.mean / total_mean * 100.0;
        println!();
        println!(
            "  BOTTLENECK: {} at {:.1}us ({:.1}% of total)",
            label, s.mean, pct
        );
    }

    // Compare sum-of-parts to end-to-end 9-dispatch
    println!();
    println!("  Sum of 9 individual dispatches: {:8.1}us", total_mean);
    println!(
        "  End-to-end 9-dispatch (single CB): {:8.1}us",
        nine_dispatch_stats.mean
    );
    let overhead = total_mean - nine_dispatch_stats.mean;
    println!(
        "  CB overhead (9 CBs vs 1 CB):   {:8.1}us ({:.1}%)",
        overhead,
        if nine_dispatch_stats.mean > 0.0 {
            overhead / nine_dispatch_stats.mean * 100.0
        } else {
            0.0
        }
    );
    println!("==========================================");

    // ========================================================================
    // f16 Weight Benchmark
    // ========================================================================
    println!("\n========== f16 Weight Benchmark ==========");
    let mut block_f16 = build_transformer_block_f16(device);
    block_f16
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch f16 failed");
    block_f16.prepare_weights_private(device, &queue);

    let x_f16 = rand_array_f16(device, &[SEQ_LEN, HIDDEN_SIZE], 200);
    let mut cache_f16 = LayerKvCache::preallocated(
        device,
        NUM_KV_HEADS,
        HEAD_DIM,
        2048,
        rmlx_core::dtype::DType::Float16,
    );

    // ---- Warmup f16 9-dispatch ----
    println!(
        "Warming up f16 9-dispatch ({} iterations)...",
        WARMUP_ITERS
    );
    for _ in 0..WARMUP_ITERS {
        cache_f16.seq_len = 0;
        let cb = queue.new_command_buffer();
        let _ = block_f16
            .forward_single_cb_9dispatch(&x_f16, None, None, None, &mut cache_f16, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- Benchmark f16 9-dispatch ----
    println!(
        "Benchmarking f16 9-dispatch ({} iterations)...",
        BENCH_ITERS
    );
    let mut f16_9d_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        cache_f16.seq_len = 0;
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = block_f16
            .forward_single_cb_9dispatch(&x_f16, None, None, None, &mut cache_f16, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
        f16_9d_latencies.push(start.elapsed());
    }
    let f16_9d_stats = Stats::from_durations(&f16_9d_latencies);
    let f16_9d_speedup = if f16_9d_stats.mean > 0.0 {
        baseline_stats.mean / f16_9d_stats.mean
    } else {
        0.0
    };
    let f16_vs_f32_speedup = if f16_9d_stats.mean > 0.0 {
        nine_dispatch_stats.mean / f16_9d_stats.mean
    } else {
        0.0
    };
    println!("  f16 9-dispatch: {}", f16_9d_stats);
    println!(
        "  vs baseline: {:.2}x, vs f32 9-dispatch: {:.2}x",
        f16_9d_speedup, f16_vs_f32_speedup
    );

    // ========================================================================
    // Allocation overhead measurement
    // ========================================================================
    println!("\n========== Allocation Overhead ==========");

    // 1. Measure KV cache allocation cost
    {
        let mut cache_alloc_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                2048,
                rmlx_core::dtype::DType::Float32,
            );
            cache_alloc_times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&cache_alloc_times);
        println!("  KV cache alloc (2x8MB slab):             {}", stats);
    }

    // 2. Measure intermediate buffer allocation cost (mimics 9-dispatch)
    {
        let mut buf_alloc_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _b1 = Array::uninit(device, &[1, HIDDEN_SIZE], rmlx_core::dtype::DType::Float32);
            let _b2 = Array::uninit(
                device,
                &[NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM * 2],
                rmlx_core::dtype::DType::Float32,
            );
            let _b3 = Array::uninit(
                device,
                &[(NUM_HEADS + NUM_KV_HEADS), 1, HEAD_DIM],
                rmlx_core::dtype::DType::Float32,
            );
            let _b4 = Array::uninit(
                device,
                &[NUM_HEADS * HEAD_DIM],
                rmlx_core::dtype::DType::Float32,
            );
            let _b5 = Array::uninit(device, &[HIDDEN_SIZE], rmlx_core::dtype::DType::Float32);
            let _b6 = Array::uninit(device, &[1, HIDDEN_SIZE], rmlx_core::dtype::DType::Float32);
            let _b7 = Array::uninit(
                device,
                &[INTERMEDIATE_DIM * 2],
                rmlx_core::dtype::DType::Float32,
            );
            let _b8 = Array::uninit(
                device,
                &[1, INTERMEDIATE_DIM],
                rmlx_core::dtype::DType::Float32,
            );
            let _b9 = Array::uninit(device, &[HIDDEN_SIZE], rmlx_core::dtype::DType::Float32);
            buf_alloc_times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&buf_alloc_times);
        println!("  Intermediate buf alloc (9 buffers):      {}", stats);
    }

    // 3. 9-dispatch with pre-allocated cache (amortized)
    {
        let mut cache_pre = LayerKvCache::preallocated(
            device,
            NUM_KV_HEADS,
            HEAD_DIM,
            2048,
            rmlx_core::dtype::DType::Float32,
        );
        // Warmup
        for _ in 0..WARMUP_ITERS {
            cache_pre.seq_len = 0; // reset position
            let cb = queue.new_command_buffer();
            let _ = block
                .forward_single_cb_9dispatch(
                    &input,
                    None,
                    None,
                    None,
                    &mut cache_pre,
                    &registry,
                    cb,
                )
                .unwrap();
            cb.commit();
            cb.wait_until_completed();
        }
        // Bench
        let mut pre_alloc_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            cache_pre.seq_len = 0; // reset position
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let _ = block
                .forward_single_cb_9dispatch(
                    &input,
                    None,
                    None,
                    None,
                    &mut cache_pre,
                    &registry,
                    cb,
                )
                .unwrap();
            cb.commit();
            cb.wait_until_completed();
            pre_alloc_times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&pre_alloc_times);
        println!("  9-dispatch (pre-alloc cache, reset seq):  {}", stats);
        println!(
            "  vs original 9-dispatch:                   {}",
            nine_dispatch_stats
        );
        let diff = nine_dispatch_stats.mean - stats.mean;
        println!("  Diff (cache alloc overhead):               {:.1}us", diff);
    }

    // ========================================================================
    // Multi-Layer Pipeline (4 layers)
    // ========================================================================
    println!("\n========== Multi-Layer Pipeline (4 layers) ==========");

    const NUM_LAYERS: usize = 4;

    // Build 4 transformer blocks, each with its own weights and KV cache
    let mut blocks: Vec<TransformerBlock> = Vec::with_capacity(NUM_LAYERS);
    for _ in 0..NUM_LAYERS {
        let mut blk = build_transformer_block(device);
        blk.prepare_weights_9dispatch(device)
            .expect("prepare_weights_9dispatch for multi-layer");
        blk.prepare_weights_private(device, &queue);
        blocks.push(blk);
    }

    let mut ml_kv_caches: Vec<LayerKvCache> = (0..NUM_LAYERS)
        .map(|_| {
            LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                2048,
                rmlx_core::dtype::DType::Float32,
            )
        })
        .collect();

    // ---- Warmup serial 9-dispatch (4 layers, single CB) ----
    println!(
        "Warming up serial 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, WARMUP_ITERS
    );
    for _ in 0..WARMUP_ITERS {
        for cache in ml_kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let cb = queue.new_command_buffer();
        let mut x = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (layer, cache) in blocks.iter().zip(ml_kv_caches.iter_mut()) {
            x = layer
                .forward_single_cb_9dispatch(&x, None, None, None, cache, &registry, cb)
                .expect("serial multi-layer warmup failed");
        }
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- Benchmark serial 9-dispatch (4 layers) ----
    println!(
        "Benchmarking serial 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, BENCH_ITERS
    );
    let mut ml_serial_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for cache in ml_kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let mut x = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (layer, cache) in blocks.iter().zip(ml_kv_caches.iter_mut()) {
            x = layer
                .forward_single_cb_9dispatch(&x, None, None, None, cache, &registry, cb)
                .expect("serial multi-layer bench failed");
        }
        cb.commit();
        cb.wait_until_completed();
        ml_serial_latencies.push(start.elapsed());
    }

    // ---- Warmup concurrent 9-dispatch (4 layers, single CB) ----
    println!(
        "\nWarming up concurrent 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, WARMUP_ITERS
    );
    for _ in 0..WARMUP_ITERS {
        for cache in ml_kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let cb = queue.new_command_buffer();
        let mut x = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (layer, cache) in blocks.iter().zip(ml_kv_caches.iter_mut()) {
            x = layer
                .forward_concurrent_9dispatch(&x, None, None, None, cache, &registry, cb)
                .expect("concurrent multi-layer warmup failed");
        }
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- Benchmark concurrent 9-dispatch (4 layers) ----
    println!(
        "Benchmarking concurrent 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, BENCH_ITERS
    );
    let mut ml_concurrent_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for cache in ml_kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let mut x = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (layer, cache) in blocks.iter().zip(ml_kv_caches.iter_mut()) {
            x = layer
                .forward_concurrent_9dispatch(&x, None, None, None, cache, &registry, cb)
                .expect("concurrent multi-layer bench failed");
        }
        cb.commit();
        cb.wait_until_completed();
        ml_concurrent_latencies.push(start.elapsed());
    }

    // ---- 2-encoder 9-dispatch (single layer) ----
    println!(
        "\nWarming up 2-encoder 9-dispatch ({} iterations)...",
        WARMUP_ITERS
    );
    let mut cache_2enc = LayerKvCache::preallocated(
        device,
        NUM_KV_HEADS,
        HEAD_DIM,
        2048,
        rmlx_core::dtype::DType::Float32,
    );
    for _ in 0..WARMUP_ITERS {
        cache_2enc.seq_len = 0;
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_2encoder_9dispatch(&input, None, None, None, &mut cache_2enc, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    println!(
        "Benchmarking 2-encoder 9-dispatch ({} iterations)...",
        BENCH_ITERS
    );
    let mut times_2enc = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        cache_2enc.seq_len = 0;
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = block
            .forward_2encoder_9dispatch(&input, None, None, None, &mut cache_2enc, &registry, cb)
            .unwrap();
        cb.commit();
        cb.wait_until_completed();
        times_2enc.push(start.elapsed());
    }
    let two_enc_stats = Stats::from_durations(&times_2enc);
    println!("  2-encoder 9-dispatch (single layer): {}", two_enc_stats);

    // ---- 2-encoder 9-dispatch x4 (multi-layer) ----
    println!(
        "\nWarming up 2-encoder 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, WARMUP_ITERS
    );
    for _ in 0..WARMUP_ITERS {
        for c in &mut ml_kv_caches {
            c.seq_len = 0;
        }
        let cb = queue.new_command_buffer();
        let mut h = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (i, c) in ml_kv_caches.iter_mut().enumerate() {
            h = blocks[i]
                .forward_2encoder_9dispatch(&h, None, None, None, c, &registry, cb)
                .unwrap();
        }
        cb.commit();
        cb.wait_until_completed();
    }
    println!(
        "Benchmarking 2-encoder 9-dispatch x{} ({} iterations)...",
        NUM_LAYERS, BENCH_ITERS
    );
    let mut times_2enc_4l = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for c in &mut ml_kv_caches {
            c.seq_len = 0;
        }
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let mut h = Array::ones(device, &[SEQ_LEN, HIDDEN_SIZE]);
        for (i, c) in ml_kv_caches.iter_mut().enumerate() {
            h = blocks[i]
                .forward_2encoder_9dispatch(&h, None, None, None, c, &registry, cb)
                .unwrap();
        }
        cb.commit();
        cb.wait_until_completed();
        times_2enc_4l.push(start.elapsed());
    }
    let stats_2enc_4l = Stats::from_durations(&times_2enc_4l);
    println!(
        "  2-encoder 9-dispatch ({} layers):  {}",
        NUM_LAYERS, stats_2enc_4l
    );

    // ---- Multi-layer results ----
    let ml_serial_stats = Stats::from_durations(&ml_serial_latencies);
    let ml_concurrent_stats = Stats::from_durations(&ml_concurrent_latencies);
    let ml_speedup = if ml_concurrent_stats.mean > 0.0 {
        ml_serial_stats.mean / ml_concurrent_stats.mean
    } else {
        0.0
    };
    let ml_2enc_speedup = if stats_2enc_4l.mean > 0.0 {
        ml_serial_stats.mean / stats_2enc_4l.mean
    } else {
        0.0
    };

    println!();
    println!(
        "  Serial 9-dispatch ({} layers):     {}",
        NUM_LAYERS, ml_serial_stats
    );
    println!(
        "  Concurrent 9-dispatch ({} layers): {}",
        NUM_LAYERS, ml_concurrent_stats
    );
    println!(
        "  2-encoder 9-dispatch ({} layers):  {}",
        NUM_LAYERS, stats_2enc_4l
    );
    println!("  Multi-layer concurrent speedup: {:.2}x", ml_speedup);
    println!("  Multi-layer 2-encoder speedup:  {:.2}x", ml_2enc_speedup);
    println!("=====================================================");

    // ---- Summary comparison ----
    let two_enc_speedup = if two_enc_stats.mean > 0.0 {
        baseline_stats.mean / two_enc_stats.mean
    } else {
        0.0
    };
    println!("\n  --- Summary ---");
    println!(
        "  {:40} mean={:8.1}us  (1.00x)",
        "Baseline (per-op forward)", baseline_stats.mean
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "ExecGraph", graph_stats.mean, speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "Single-CB", single_cb_stats.mean, single_cb_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "9-Dispatch", nine_dispatch_stats.mean, nine_dispatch_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "Concurrent-9-Dispatch", concurrent_9d_stats.mean, concurrent_9d_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "2-Encoder 9-Dispatch", two_enc_stats.mean, two_enc_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x) [{:.2}x vs f32 9d]",
        "f16 9-Dispatch", f16_9d_stats.mean, f16_9d_speedup, f16_vs_f32_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x)",
        "Serial x4", ml_serial_stats.mean, 1.0
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x vs serial x4)",
        "Concurrent x4", ml_concurrent_stats.mean, ml_speedup
    );
    println!(
        "  {:40} mean={:8.1}us  ({:.2}x vs serial x4)",
        "2-Encoder x4", stats_2enc_4l.mean, ml_2enc_speedup
    );
    println!("  -----------------");
}
