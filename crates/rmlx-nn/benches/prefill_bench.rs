//! GPU Prefill Benchmark (Llama-3 8B single layer, seq_len > 1)
//!
//! Measures single-layer TransformerBlock forward pass latency across
//! multiple sequence lengths to profile prefill performance.
//!
//! Only benchmarks single_cb and ExecGraph paths — the per-op forward()
//! baseline has been removed because it uses a different code path and
//! can silently poison the shared command queue, invalidating subsequent
//! measurements.
//!
//! Each seq_len gets a **fresh command queue** to prevent cross-contamination.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench prefill_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::ExecGraph;
use rmlx_metal::ScopedPool;
use rmlx_nn::{
    Attention, AttentionConfig, FeedForward, LayerKvCache, Linear, LinearConfig, TransformerBlock,
};

// ---------------------------------------------------------------------------
// Llama-3 8B config
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 14336;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
const MAX_SEQ_LEN: usize = 2048;

/// Total trainable parameters for a single Llama-3 8B layer (approximate).
/// Used to compute TFLOPS: 2 * PARAMS_PER_LAYER * seq_len = total FLOPs.
const PARAMS_PER_LAYER: f64 = 218_112_000.0;

const SEQ_LENS: &[usize] = &[256];
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper (same as pipeline_bench.rs)
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
            "mean={:10.1}us std={:8.1}us p50={:10.1}us p95={:10.1}us p99={:10.1}us min={:10.1}us max={:10.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.p99, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// f16 helpers (same as pipeline_bench.rs)
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
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
// Layer construction (f16, same as pipeline_bench.rs)
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
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
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
        gate_up_merged_weight_t: None,
    };

    let norm1_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE]; // f16 1.0
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };
    let norm2_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE]; // f16 1.0
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Causal mask builder (f16)
// ---------------------------------------------------------------------------

fn build_causal_mask(device: &metal::Device, seq_len: usize) -> Array {
    let neg_inf_f16: u16 = 0xFC00; // f16 -inf
    let mut mask_f16 = vec![0u16; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_f16[i * seq_len + j] = neg_inf_f16;
        }
    }
    let bytes: Vec<u8> = mask_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    Array::from_bytes(device, &bytes, vec![seq_len, seq_len], DType::Float16)
}

// ---------------------------------------------------------------------------
// CB status validation
// ---------------------------------------------------------------------------

fn assert_cb_ok(cb: &metal::CommandBufferRef, context: &str) {
    let status = cb.status();
    assert!(
        status != metal::MTLCommandBufferStatus::Error,
        "GPU command buffer error in {context}: status={status:?}"
    );
}

// ---------------------------------------------------------------------------
// Memory bandwidth estimate
// ---------------------------------------------------------------------------

/// Estimate total bytes read/written for a single-layer prefill pass.
///
/// Weights (read once):
///   Q_proj: hidden * hidden * 2  = 4096*4096*2
///   K_proj: hidden * kv_size * 2 = 4096*1024*2
///   V_proj: hidden * kv_size * 2 = 4096*1024*2
///   O_proj: hidden * hidden * 2  = 4096*4096*2
///   gate:   hidden * inter * 2   = 4096*14336*2
///   up:     hidden * inter * 2   = 4096*14336*2
///   down:   inter * hidden * 2   = 14336*4096*2
///   norms:  2 * hidden * 2       (negligible)
///
/// Activations (proportional to seq_len, relatively small for typical sizes).
fn estimate_bytes(seq_len: usize) -> f64 {
    let elem = 2.0; // f16

    // Weight bytes
    let q_w = (HIDDEN_SIZE * HIDDEN_SIZE) as f64 * elem;
    let k_w = (HIDDEN_SIZE * NUM_KV_HEADS * HEAD_DIM) as f64 * elem;
    let v_w = k_w;
    let o_w = q_w;
    let gate_w = (HIDDEN_SIZE * INTERMEDIATE_DIM) as f64 * elem;
    let up_w = gate_w;
    let down_w = gate_w;
    let weight_bytes = q_w + k_w + v_w + o_w + gate_w + up_w + down_w;

    // Activation bytes (input/output per matmul, rough estimate)
    let act_per_matmul = (seq_len * HIDDEN_SIZE) as f64 * elem * 2.0; // read + write
    let act_bytes = act_per_matmul * 7.0; // 7 matmuls

    weight_bytes + act_bytes
}

// ---------------------------------------------------------------------------
// TFLOPS computation
// ---------------------------------------------------------------------------

fn compute_tflops(seq_len: usize, mean_us: f64) -> f64 {
    let total_flops = 2.0 * PARAMS_PER_LAYER * seq_len as f64;
    let seconds = mean_us / 1e6;
    total_flops / seconds / 1e12
}

// ---------------------------------------------------------------------------
// MLX reference numbers (from hwstudio1 benchmark run, 2026-03-09)
// ---------------------------------------------------------------------------

fn mlx_ref_us(seq_len: usize) -> Option<f64> {
    match seq_len {
        128 => Some(3443.0),
        256 => Some(6980.0),
        512 => Some(11369.0),
        1024 => Some(20356.0),
        2048 => Some(40979.0),
        _ => None,
    }
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

    // We need a temporary queue for prepare_weights_for_graph only.
    let setup_queue = device.new_command_queue();

    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!("dtype: float16");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!("PARAMS_PER_LAYER: {:.0}", PARAMS_PER_LAYER);

    // Build transformer blocks (f16) — one for single_cb, one for graph
    let mut block_cb = build_transformer_block(device);
    let mut block_graph = build_transformer_block(device);

    // Pre-transpose weights for both paths
    block_cb
        .prepare_weights_for_graph(&registry, &setup_queue)
        .expect("prepare_weights_for_graph failed (single_cb)");
    block_graph
        .prepare_weights_for_graph(&registry, &setup_queue)
        .expect("prepare_weights_for_graph failed (graph)");

    // Precompute RoPE cos/sin tables: shape [MAX_SEQ_LEN, HEAD_DIM/2]
    let (cos_vec, sin_vec) = ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
        .expect("precompute_freqs failed");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    // Collect results: (seq_len, single_cb_stats, graph_stats)
    let mut results: Vec<(usize, Stats, Stats)> = Vec::new();

    for &seq_len in SEQ_LENS {
        println!("\n--- seq_len={} ---", seq_len);
        // Scope ensures Metal command queues are dropped before next seq_len,
        // preventing GPU state contamination between iterations.
        let (stats_cb_out, stats_graph_out) = {
            // Slice RoPE tables to [seq_len, HEAD_DIM/2]
            let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
            let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");

            // Build causal mask [seq_len, seq_len] in f16
            let mask = build_causal_mask(device, seq_len);

            // Input: [seq_len, HIDDEN_SIZE]
            let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

            // ==== Benchmark 1: forward_prefill_single_cb() ====
            // Fresh queue for this seq_len to prevent cross-contamination
            let queue_cb = device.new_command_queue();
            let stats_cb = {
                let mut cache_cb = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );

                // Warmup
                let mut last_output = None;
                for _ in 0..WARMUP_ITERS {
                    let _pool = ScopedPool::new();
                    cache_cb.seq_len = 0;
                    let cb = queue_cb.new_command_buffer();
                    let out = block_cb
                        .forward_prefill_single_cb(
                            &input,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            Some(&mask),
                            &mut cache_cb,
                            &registry,
                            cb,
                        )
                        .expect("single_cb warmup failed");
                    cb.commit();
                    cb.wait_until_completed();
                    assert_cb_ok(cb, "single_cb warmup");
                    last_output = Some(out);
                }

                // Validate output after warmup
                if let Some(ref out) = last_output {
                    let out_shape = out.shape();
                    assert_eq!(
                        out_shape,
                        &[seq_len, HIDDEN_SIZE],
                        "wrong output shape for single_cb: expected [{}, {}], got {:?}",
                        seq_len,
                        HIDDEN_SIZE,
                        out_shape
                    );
                }

                // Benchmark
                let mut latencies = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let _pool = ScopedPool::new();
                    cache_cb.seq_len = 0;
                    let cb = queue_cb.new_command_buffer();
                    let start = Instant::now();
                    let _ = block_cb
                        .forward_prefill_single_cb(
                            &input,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            Some(&mask),
                            &mut cache_cb,
                            &registry,
                            cb,
                        )
                        .expect("forward_prefill_single_cb failed");
                    cb.commit();
                    cb.wait_until_completed();
                    assert_cb_ok(cb, "single_cb bench");
                    latencies.push(start.elapsed());
                }

                Stats::from_durations(&latencies)
            };

            let tflops_cb = compute_tflops(seq_len, stats_cb.mean);
            println!("  single_cb : {}", stats_cb);
            println!("    estimated TFLOPS: {:.2}", tflops_cb);
            assert!(
            tflops_cb < 80.0,
            "single_cb TFLOPS ({:.2}) exceeds hardware peak (65.54T)! Measurement is likely wrong",
            tflops_cb
        );

            // ==== Benchmark 2: forward_prefill_graph() (ExecGraph) ====
            // Fresh queue for graph path
            let queue_graph = device.new_command_queue();
            let stats_graph = {
                let mut cache_graph = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );

                // Warmup
                let mut last_output = None;
                for _ in 0..WARMUP_ITERS {
                    let _pool = ScopedPool::new();
                    cache_graph.seq_len = 0;
                    let event = GpuEvent::new(device);
                    let mut graph = ExecGraph::new(&queue_graph, &event, 64);
                    let cb = graph.command_buffer();
                    let out = block_graph
                        .forward_prefill_single_cb(
                            &input,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            Some(&mask),
                            &mut cache_graph,
                            &registry,
                            cb,
                        )
                        .expect("graph warmup failed");
                    let _t = graph.submit_batch();
                    graph.sync().expect("graph warmup sync failed");
                    last_output = Some(out);
                }

                // Validate output after warmup
                if let Some(ref out) = last_output {
                    let out_shape = out.shape();
                    assert_eq!(
                        out_shape,
                        &[seq_len, HIDDEN_SIZE],
                        "wrong output shape for graph: expected [{}, {}], got {:?}",
                        seq_len,
                        HIDDEN_SIZE,
                        out_shape
                    );
                }

                // Benchmark
                let mut latencies = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let _pool = ScopedPool::new();
                    cache_graph.seq_len = 0;
                    let event = GpuEvent::new(device);
                    let mut graph = ExecGraph::new(&queue_graph, &event, 64);
                    let start = Instant::now();
                    let cb = graph.command_buffer();
                    let _ = block_graph
                        .forward_prefill_single_cb(
                            &input,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            Some(&mask),
                            &mut cache_graph,
                            &registry,
                            cb,
                        )
                        .expect("forward_prefill_graph failed");
                    let _t = graph.submit_batch();
                    graph.sync().expect("graph sync failed");
                    latencies.push(start.elapsed());
                }

                Stats::from_durations(&latencies)
            };

            let tflops_graph = compute_tflops(seq_len, stats_graph.mean);
            let graph_speedup = stats_cb.mean / stats_graph.mean;
            println!(
                "  graph     : {}   speedup={:.2}x",
                stats_graph, graph_speedup
            );
            println!("    estimated TFLOPS: {:.2}", tflops_graph);
            assert!(
                tflops_graph < 80.0,
                "graph TFLOPS ({:.2}) exceeds hardware peak (65.54T)! Measurement is likely wrong",
                tflops_graph
            );

            (stats_cb, stats_graph)
        }; // drop queues, caches, and ExecGraph state

        // Let Metal driver fully drain GPU resources before next seq_len
        std::thread::sleep(std::time::Duration::from_millis(100));

        results.push((seq_len, stats_cb_out, stats_graph_out));
    }

    // ---- Comparison summary table ----
    println!("\n{}", "=".repeat(120));
    println!("========== Comparison ==========");
    println!(
        "{:>8} | {:>14} | {:>8} | {:>14} | {:>8} | {:>10} | {:>12} | {:>8}",
        "seq_len",
        "single_cb (us)",
        "TFLOPS",
        "graph (us)",
        "TFLOPS",
        "graph spd",
        "MLX ref (us)",
        "vs MLX"
    );
    println!("{}", "-".repeat(120));
    for (seq_len, cb_stats, graph_stats) in &results {
        let tflops_cb = compute_tflops(*seq_len, cb_stats.mean);
        let tflops_graph = compute_tflops(*seq_len, graph_stats.mean);
        let graph_speedup = cb_stats.mean / graph_stats.mean;
        let (mlx_str, vs_mlx_str) = match mlx_ref_us(*seq_len) {
            Some(mlx_us) => {
                let ratio = graph_stats.mean / mlx_us;
                (format!("{:.0}", mlx_us), format!("{:.2}x", ratio))
            }
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "{:>8} | {:>14.0} | {:>8.2} | {:>14.0} | {:>8.2} | {:>9.2}x | {:>12} | {:>8}",
            seq_len,
            cb_stats.mean,
            tflops_cb,
            graph_stats.mean,
            tflops_graph,
            graph_speedup,
            mlx_str,
            vs_mlx_str
        );
    }
    println!("{}", "=".repeat(120));

    // Weight size reference
    let weight_mb = estimate_bytes(0) / 1e6;
    println!("\nWeight size per layer: {:.1} MB (f16)", weight_mb);
}
