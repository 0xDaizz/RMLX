//! GPU Prefill Benchmark (Llama-3 8B single layer, seq_len > 1)
//!
//! Measures single-layer TransformerBlock forward pass latency across
//! multiple sequence lengths to profile prefill performance.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench prefill_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
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

const SEQ_LENS: &[usize] = &[128, 256, 512, 1024, 2048];
const WARMUP_ITERS: usize = 3;
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
// Causal mask builder
// ---------------------------------------------------------------------------

fn build_causal_mask(device: &metal::Device, seq_len: usize) -> Array {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Array::from_slice(device, &mask, vec![seq_len, seq_len])
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
// Benchmark entry point
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MLX reference numbers (hardcoded from earlier benchmark runs)
// ---------------------------------------------------------------------------

fn mlx_ref_us(seq_len: usize) -> Option<f64> {
    match seq_len {
        128 => Some(3466.0),
        256 => Some(6743.0),
        512 => Some(14872.0),
        1024 => Some(42618.0),
        2048 => Some(153721.0),
        _ => None,
    }
}

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
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!("dtype: float16");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );

    // Build transformer blocks (f16) — one for baseline, one for single_cb
    let block = build_transformer_block(device);
    let mut block_cb = build_transformer_block(device);

    // Pre-transpose weights for the single_cb path
    block_cb
        .prepare_weights_for_graph(&registry, &queue)
        .expect("prepare_weights_for_graph failed");

    // Precompute RoPE cos/sin tables: shape [MAX_SEQ_LEN, HEAD_DIM/2]
    let (cos_vec, sin_vec) =
        ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
            .expect("precompute_freqs failed");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    // Collect results for summary table: (seq_len, forward_stats, single_cb_stats)
    let mut results: Vec<(usize, Stats, Stats)> = Vec::new();

    for &seq_len in SEQ_LENS {
        println!("\nseq_len={}:", seq_len);

        // Slice RoPE tables to [seq_len, HEAD_DIM/2] (view into precomputed table)
        let cos_freqs = cos_full
            .slice(0, 0, seq_len)
            .expect("cos slice failed");
        let sin_freqs = sin_full
            .slice(0, 0, seq_len)
            .expect("sin slice failed");

        // Build causal mask [seq_len, seq_len]
        let mask = build_causal_mask(device, seq_len);

        // Input: [seq_len, HIDDEN_SIZE]
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

        // ---- Benchmark 1: forward() baseline ----
        {
            let mut cache = LayerKvCache::preallocated(
                device, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DType::Float16,
            );

            // Warmup
            for _ in 0..WARMUP_ITERS {
                cache.seq_len = 0;
                let _ = block.forward(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    Some(&mask),
                    Some(&mut cache),
                    &registry,
                    &queue,
                );
            }

            // Benchmark
            let mut latencies = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                cache.seq_len = 0;
                let start = Instant::now();
                let _ = block
                    .forward(
                        &input,
                        Some(&cos_freqs),
                        Some(&sin_freqs),
                        Some(&mask),
                        Some(&mut cache),
                        &registry,
                        &queue,
                    )
                    .expect("forward failed");
                latencies.push(start.elapsed());
            }

            let stats = Stats::from_durations(&latencies);
            println!("  forward()              : {}", stats);

            // ---- Benchmark 2: forward_prefill_single_cb() ----
            let mut cache_cb = LayerKvCache::preallocated(
                device, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DType::Float16,
            );

            // Warmup
            for _ in 0..WARMUP_ITERS {
                cache_cb.seq_len = 0;
                let cb = queue.new_command_buffer();
                let _ = block_cb.forward_prefill_single_cb(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    Some(&mask),
                    &mut cache_cb,
                    &registry,
                    cb,
                );
                cb.commit();
                cb.wait_until_completed();
            }

            // Benchmark
            let mut latencies_cb = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                cache_cb.seq_len = 0;
                let cb = queue.new_command_buffer();
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
                latencies_cb.push(start.elapsed());
            }

            let stats_cb = Stats::from_durations(&latencies_cb);
            let speedup = stats.mean / stats_cb.mean;
            println!(
                "  prefill_single_cb()    : {}   speedup={:.2}x",
                stats_cb, speedup
            );

            results.push((seq_len, stats, stats_cb));
        }
    }

    // Comparison summary table
    println!("\n{}", "=".repeat(80));
    println!("========== Comparison ==========");
    println!(
        "{:>8} | {:>14} | {:>14} | {:>8} | {:>12}",
        "seq_len", "forward (us)", "single_cb (us)", "speedup", "MLX ref (us)"
    );
    println!("{}", "-".repeat(80));
    for (seq_len, fwd_stats, cb_stats) in &results {
        let speedup = fwd_stats.mean / cb_stats.mean;
        let mlx_str = match mlx_ref_us(*seq_len) {
            Some(v) => format!("{:.0}", v),
            None => "-".to_string(),
        };
        println!(
            "{:>8} | {:>14.0} | {:>14.0} | {:>7.2}x | {:>12}",
            seq_len, fwd_stats.mean, cb_stats.mean, speedup, mlx_str
        );
    }
    println!("{}", "=".repeat(80));

    // Weight size reference
    let weight_mb = estimate_bytes(0) / 1e6;
    println!("\nWeight size per layer: {:.1} MB (f16)", weight_mb);
}
