//! DIAGNOSTIC — single-layer MoE expert prefill (Qwen 3.5-style dimensions).
//!
//! Tests `forward_prefill_into_cb` (production single-layer path) and
//! `forward_prefill_into_encoder` at the TransformerBlock level.
//! For full 32-layer production prefill, use `e2e_prefill_bench`.
//! For production decode, use `pipeline_bench`.
//!
//! Includes per-dispatch GPU timing breakdown (9-CB and cumulative subtraction).
//!
//! Each seq_len gets a **fresh command queue** to prevent cross-contamination.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench prefill_bench --features bench

use std::time::{Duration, Instant};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLDevice as _};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::ExecGraph;
use rmlx_metal::autoreleasepool;
use rmlx_nn::{
    Attention, AttentionConfig, FeedForward, LayerKvCache, Linear, LinearConfig, TransformerBlock,
};

// ---------------------------------------------------------------------------
// MoE Expert Layer config (Qwen 3.5-style dimensions)
//
// In MoE models, each expert FFN has a small intermediate_dim (e.g., 2560)
// and tokens are distributed across 64-128 experts, so M per expert is low.
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 3584;
const NUM_HEADS: usize = 28;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 2560;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 1000000.0;
const MAX_SEQ_LEN: usize = 2048;

/// Total trainable parameters for a single MoE expert layer (approximate).
/// Used to compute TFLOPS: 2 * PARAMS_PER_LAYER * seq_len = total FLOPs.
const PARAMS_PER_LAYER: f64 = 56_885_248.0;

const SEQ_LENS: &[usize] = &[128, 256, 512, 1024, 2048];
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

fn rand_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
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

fn make_linear(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, in_f: usize, out_f: usize, seed: u64) -> Linear {
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

fn build_transformer_block(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> TransformerBlock {
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

fn build_causal_mask(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, seq_len: usize) -> Array {
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

fn assert_cb_ok(cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>, context: &str) {
    let status = cb.status();
    assert!(
        status != objc2_metal::MTLCommandBufferStatus::Error,
        "GPU command buffer error in {context}: status={status:?}"
    );
}

// ---------------------------------------------------------------------------
// Memory bandwidth estimate
// ---------------------------------------------------------------------------

/// Estimate total bytes read/written for a single-layer prefill pass.
///
/// Weights (read once):
///   Q_proj: hidden * hidden * 2  = 3584*3584*2
///   K_proj: hidden * kv_size * 2 = 3584*512*2
///   V_proj: hidden * kv_size * 2 = 3584*512*2
///   O_proj: hidden * hidden * 2  = 3584*3584*2
///   gate:   hidden * inter * 2   = 3584*2560*2
///   up:     hidden * inter * 2   = 3584*2560*2
///   down:   inter * hidden * 2   = 2560*3584*2
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
// MLX reference numbers
// ---------------------------------------------------------------------------

fn mlx_ref_us(_seq_len: usize) -> Option<f64> {
    // MLX reference numbers not yet collected for MoE expert dimensions.
    // Run MLX benchmark with matching config and fill in here.
    None
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

    // Build transformer blocks (f16) — one per benchmark path
    let mut block_cb = build_transformer_block(device);
    let mut block_graph = build_transformer_block(device);
    let mut block_enc = build_transformer_block(device);

    // Merge Q/K/V into a single weight matrix (3 GEMMs → 1 GEMM per layer)
    block_cb
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed (single_cb)");
    block_graph
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed (graph)");
    block_enc
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed (single_encoder)");

    // Pre-transpose weights for all paths.
    // Scoped so the temporary queue is dropped (and fully drained) before benchmarks start.
    {
        let setup_queue = device.newCommandQueue().unwrap();
        block_cb
            .prepare_weights_for_graph(&registry, &setup_queue)
            .expect("prepare_weights_for_graph failed (single_cb)");
        block_graph
            .prepare_weights_for_graph(&registry, &setup_queue)
            .expect("prepare_weights_for_graph failed (graph)");
        block_enc
            .prepare_weights_for_graph(&registry, &setup_queue)
            .expect("prepare_weights_for_graph failed (single_encoder)");
    }
    // Let Metal driver fully drain GPU resources from weight preparation
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Precompute RoPE cos/sin tables: shape [MAX_SEQ_LEN, HEAD_DIM/2]
    let (cos_vec, sin_vec) = ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
        .expect("precompute_freqs failed");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    // Collect results: (seq_len, single_cb, graph, single_encoder)
    let mut results: Vec<(usize, Stats, Stats, Stats)> = Vec::new();

    for &seq_len in SEQ_LENS {
        println!("\n--- seq_len={} ---", seq_len);
        // Scope ensures Metal command queues are dropped before next seq_len,
        // preventing GPU state contamination between iterations.
        let (stats_cb_out, stats_graph_out, stats_enc_out) = {
            // Slice RoPE tables to [seq_len, HEAD_DIM/2]
            let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
            let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");

            // Causal mask no longer needed — kernels handle causal masking internally
            // via is_causal function constant. Kept for reference/fallback.
            let _mask = build_causal_mask(device, seq_len);

            // Input: [seq_len, HIDDEN_SIZE]
            let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

            // ==== Benchmark 1: forward_prefill_into_cb() ====
            // Fresh queue for this seq_len to prevent cross-contamination
            let queue_cb = device.newCommandQueue().unwrap();
            let stats_cb = {
                let mut cache_cb = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );

                // Warmup — single autorelease pool wraps the entire loop to prevent
                // premature release of Metal objects (command buffers, encoders) that
                // can leave the queue in an error state.
                let mut last_output = None;
                autoreleasepool(|_| {
                    for _ in 0..WARMUP_ITERS {
                        cache_cb.seq_len = 0;
                        let cb = queue_cb.commandBufferWithUnretainedReferences().unwrap();
                        let out = block_cb
                            .forward_prefill_into_cb(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None, // causal masking handled in-kernel via is_causal FC
                                &mut cache_cb,
                                &registry,
                                &cb,
                            )
                            .expect("single_cb warmup failed");
                        cb.commit();
                        cb.waitUntilCompleted();
                        assert_cb_ok(&cb, "single_cb warmup");
                        last_output = Some(out);
                    }
                });

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
                autoreleasepool(|_| {
                    for _ in 0..BENCH_ITERS {
                        cache_cb.seq_len = 0;
                        let cb = queue_cb.commandBufferWithUnretainedReferences().unwrap();
                        let start = Instant::now();
                        let _ = block_cb
                            .forward_prefill_into_cb(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None, // causal masking handled in-kernel via is_causal FC
                                &mut cache_cb,
                                &registry,
                                &cb,
                            )
                            .expect("forward_prefill_into_cb failed");
                        cb.commit();
                        cb.waitUntilCompleted();
                        assert_cb_ok(&cb, "single_cb bench");
                        latencies.push(start.elapsed());
                    }
                });

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

            // ==== Benchmark 2: forward_prefill_into_cb via ExecGraph ====
            // Fresh queue for graph path
            let queue_graph = device.newCommandQueue().unwrap();
            let stats_graph = {
                let mut cache_graph = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );

                // Warmup — single autorelease pool wraps the entire loop.
                // GpuEvent is created once outside the loop to avoid MTLSharedEvent
                // alloc/dealloc overhead (~200-400us) polluting measurements.
                let event = GpuEvent::new(device);
                let mut last_output = None;
                autoreleasepool(|_| {
                    for _ in 0..WARMUP_ITERS {
                        cache_graph.seq_len = 0;
                        let mut graph = ExecGraph::new(&queue_graph, &event, 64);
                        let cb = graph.command_buffer();
                        let out = block_graph
                            .forward_prefill_into_cb(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                &mut cache_graph,
                                &registry,
                                cb,
                            )
                            .expect("graph warmup failed");
                        graph.submit_batch();
                        graph.sync().expect("graph warmup sync failed");
                        graph.reset();
                        last_output = Some(out);
                    }
                });

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
                autoreleasepool(|_| {
                    for _ in 0..BENCH_ITERS {
                        cache_graph.seq_len = 0;
                        let mut graph = ExecGraph::new(&queue_graph, &event, 64);
                        let start = Instant::now();
                        let cb = graph.command_buffer();
                        let _ = block_graph
                            .forward_prefill_into_cb(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                &mut cache_graph,
                                &registry,
                                cb,
                            )
                            .expect("forward_prefill_into_cb failed");
                        graph.submit_batch();
                        graph.sync().expect("graph sync failed");
                        latencies.push(start.elapsed());
                        graph.reset();
                    }
                });

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

            // ==== Benchmark 3: forward_prefill_into_encoder() ====
            // All dispatches share ONE compute command encoder per CB,
            // eliminating per-op encoder create/destroy overhead.
            let queue_enc = device.newCommandQueue().unwrap();
            let stats_enc = {
                let mut cache_enc = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );

                // Warmup
                let mut last_output = None;
                autoreleasepool(|_| {
                    for _ in 0..WARMUP_ITERS {
                        cache_enc.seq_len = 0;
                        let cb = queue_enc.commandBufferWithUnretainedReferences().unwrap();
                        let encoder = cb.computeCommandEncoder().unwrap();
                        let out = block_enc
                            .forward_prefill_into_encoder(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                &mut cache_enc,
                                &registry,
                                &encoder,
                            )
                            .expect("single_encoder warmup failed");
                        encoder.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        assert_cb_ok(&cb, "single_encoder warmup");
                        last_output = Some(out);
                    }
                });

                // Validate output after warmup
                if let Some(ref out) = last_output {
                    let out_shape = out.shape();
                    assert_eq!(
                        out_shape,
                        &[seq_len, HIDDEN_SIZE],
                        "wrong output shape for single_encoder: expected [{}, {}], got {:?}",
                        seq_len,
                        HIDDEN_SIZE,
                        out_shape
                    );
                }

                // Benchmark
                let mut latencies = Vec::with_capacity(BENCH_ITERS);
                autoreleasepool(|_| {
                    for _ in 0..BENCH_ITERS {
                        cache_enc.seq_len = 0;
                        let cb = queue_enc.commandBufferWithUnretainedReferences().unwrap();
                        let encoder = cb.computeCommandEncoder().unwrap();
                        let start = Instant::now();
                        let _ = block_enc
                            .forward_prefill_into_encoder(
                                &input,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                &mut cache_enc,
                                &registry,
                                &encoder,
                            )
                            .expect("forward_prefill_into_encoder failed");
                        encoder.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        assert_cb_ok(&cb, "single_encoder bench");
                        latencies.push(start.elapsed());
                    }
                });

                Stats::from_durations(&latencies)
            };

            let tflops_enc = compute_tflops(seq_len, stats_enc.mean);
            let enc_speedup = stats_cb.mean / stats_enc.mean;
            println!("  single_enc: {}   speedup={:.2}x", stats_enc, enc_speedup);
            println!("    estimated TFLOPS: {:.2}", tflops_enc);
            assert!(
                tflops_enc < 80.0,
                "single_encoder TFLOPS ({:.2}) exceeds hardware peak (65.54T)! Measurement is likely wrong",
                tflops_enc
            );

            (stats_cb, stats_graph, stats_enc)
        }; // drop queues, caches, and ExecGraph state

        // Let Metal driver fully drain GPU resources before next seq_len
        std::thread::sleep(std::time::Duration::from_millis(100));

        results.push((seq_len, stats_cb_out, stats_graph_out, stats_enc_out));
    }

    // ---- Comparison summary table ----
    println!("\n{}", "=".repeat(140));
    println!("========== Comparison ==========");
    println!(
        "{:>8} | {:>14} | {:>8} | {:>14} | {:>8} | {:>14} | {:>8} | {:>12} | {:>8}",
        "seq_len",
        "single_cb (us)",
        "TFLOPS",
        "graph (us)",
        "TFLOPS",
        "single_enc(us)",
        "TFLOPS",
        "MLX ref (us)",
        "vs MLX"
    );
    println!("{}", "-".repeat(140));
    for (seq_len, cb_stats, graph_stats, enc_stats) in &results {
        let tflops_cb = compute_tflops(*seq_len, cb_stats.mean);
        let tflops_graph = compute_tflops(*seq_len, graph_stats.mean);
        let tflops_enc = compute_tflops(*seq_len, enc_stats.mean);
        let (mlx_str, vs_mlx_str) = match mlx_ref_us(*seq_len) {
            Some(mlx_us) => {
                let ratio = graph_stats.mean / mlx_us;
                (format!("{:.0}", mlx_us), format!("{:.2}x", ratio))
            }
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "{:>8} | {:>14.0} | {:>8.2} | {:>14.0} | {:>8.2} | {:>14.0} | {:>8.2} | {:>12} | {:>8}",
            seq_len,
            cb_stats.mean,
            tflops_cb,
            graph_stats.mean,
            tflops_graph,
            enc_stats.mean,
            tflops_enc,
            mlx_str,
            vs_mlx_str
        );
    }
    println!("{}", "=".repeat(140));

    // Weight size reference
    let weight_mb = estimate_bytes(0) / 1e6;
    println!("\nWeight size per layer: {:.1} MB (f16)", weight_mb);

    // ---- Per-dispatch GPU timing breakdown (old: 9 separate CBs) ----
    bench_dispatch_breakdown(&registry, device, &cos_full, &sin_full);

    // ---- Per-dispatch GPU timing (cumulative subtraction method) ----
    bench_cumulative_breakdown(&registry, device, &cos_full, &sin_full);
}

// ---------------------------------------------------------------------------
// Per-dispatch GPU timing breakdown
// ---------------------------------------------------------------------------

const BREAKDOWN_SEQ_LENS: &[usize] = &[32, 128, 256];
const BREAKDOWN_WARMUP: usize = 5;
const BREAKDOWN_ITERS: usize = 20;

const CUMULATIVE_SEQ_LENS: &[usize] = &[32, 128, 256, 512, 1024];
const CUMULATIVE_WARMUP: usize = 5;
const CUMULATIVE_ITERS: usize = 20;

fn bench_dispatch_breakdown(
    registry: &KernelRegistry,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    cos_full: &Array,
    sin_full: &Array,
) {
    println!("\n{}", "=".repeat(90));
    println!("========== Per-Dispatch GPU Timing Breakdown ==========");
    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!(
        "Warmup: {} iters, Bench: {} iters per dispatch",
        BREAKDOWN_WARMUP, BREAKDOWN_ITERS
    );

    let mut block = build_transformer_block(device);
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed (breakdown)");
    {
        let setup_queue = device.newCommandQueue().unwrap();
        block
            .prepare_weights_for_graph(registry, &setup_queue)
            .expect("prepare_weights_for_graph failed (breakdown)");
    }
    std::thread::sleep(std::time::Duration::from_millis(50));

    for &seq_len in BREAKDOWN_SEQ_LENS {
        println!("\n--- seq_len={} ---", seq_len);
        autoreleasepool(|_| {

        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

        let queue = device.newCommandQueue().unwrap();

        // Warmup
        for _ in 0..BREAKDOWN_WARMUP {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let _ = block
                .forward_prefill_breakdown(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    &mut cache,
                    registry,
                    &queue,
                )
                .expect("breakdown warmup failed");
        }

        // Collect per-dispatch timings across iterations
        // timings_per_dispatch[dispatch_idx] = Vec<Duration> across iters
        let num_dispatches = 9;
        let mut dispatch_names: Vec<&str> = Vec::new();
        let mut timings_per_dispatch: Vec<Vec<Duration>> = (0..num_dispatches)
            .map(|_| Vec::with_capacity(BREAKDOWN_ITERS))
            .collect();

        for _ in 0..BREAKDOWN_ITERS {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let (_, timings) = block
                .forward_prefill_breakdown(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    &mut cache,
                    registry,
                    &queue,
                )
                .expect("breakdown bench failed");

            if dispatch_names.is_empty() {
                dispatch_names = timings.iter().map(|(name, _)| *name).collect();
            }
            for (i, (_, dur)) in timings.iter().enumerate() {
                if i < num_dispatches {
                    timings_per_dispatch[i].push(*dur);
                }
            }
        }

        // Also measure single_cb total for overhead comparison
        let mut single_cb_latencies = Vec::with_capacity(BREAKDOWN_ITERS);
        {
            for _ in 0..BREAKDOWN_ITERS {
                let mut cache = LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                );
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let start = Instant::now();
                let _ = block
                    .forward_prefill_into_cb(
                        &input,
                        Some(&cos_freqs),
                        Some(&sin_freqs),
                        None,
                        &mut cache,
                        registry,
                        &cb,
                    )
                    .expect("single_cb failed");
                cb.commit();
                cb.waitUntilCompleted();
                single_cb_latencies.push(start.elapsed());
            }
        }

        // Compute mean for each dispatch
        let mut means_us: Vec<f64> = Vec::with_capacity(num_dispatches);
        for durations in &timings_per_dispatch {
            if durations.is_empty() {
                means_us.push(0.0);
            } else {
                let sum: f64 = durations.iter().map(|d| d.as_secs_f64() * 1e6).sum();
                means_us.push(sum / durations.len() as f64);
            }
        }
        let total_breakdown_us: f64 = means_us.iter().sum();

        let single_cb_mean_us: f64 = {
            let sum: f64 = single_cb_latencies
                .iter()
                .map(|d| d.as_secs_f64() * 1e6)
                .sum();
            sum / single_cb_latencies.len() as f64
        };

        // Print table
        println!(
            "| {:>2} | {:<28} | {:>10} | {:>6} |",
            "#", "Dispatch", "Time (us)", "%"
        );
        println!("|{:-<4}|{:-<30}|{:-<12}|{:-<8}|", "", "", "", "");
        for (i, name) in dispatch_names.iter().enumerate() {
            let us = means_us[i];
            let pct = if total_breakdown_us > 0.0 {
                us / total_breakdown_us * 100.0
            } else {
                0.0
            };
            println!(
                "| {:>2} | {:<28} | {:>10.1} | {:>5.1}% |",
                i + 1,
                name,
                us,
                pct
            );
        }
        println!("|{:-<4}|{:-<30}|{:-<12}|{:-<8}|", "", "", "", "");
        println!(
            "|    | {:<28} | {:>10.1} | {:>5.1}% |",
            "TOTAL (breakdown)", total_breakdown_us, 100.0
        );
        println!(
            "|    | {:<28} | {:>10.1} |        |",
            "single_cb (reference)", single_cb_mean_us
        );
        let overhead_us = total_breakdown_us - single_cb_mean_us;
        let overhead_pct = if single_cb_mean_us > 0.0 {
            overhead_us / single_cb_mean_us * 100.0
        } else {
            0.0
        };
        println!(
            "|    | {:<28} | {:>10.1} | {:>5.1}% |",
            "CB overhead (9 CBs vs 1)", overhead_us, overhead_pct
        );

        // Let GPU cool between seq_lens
        std::thread::sleep(std::time::Duration::from_millis(100));
        }); // autoreleasepool
    }

    println!("\n{}", "=".repeat(90));
}

// ---------------------------------------------------------------------------
// Per-dispatch GPU timing via cumulative subtraction
// ---------------------------------------------------------------------------
//
// Instead of 9 separate CBs (each with ~350-450us overhead), this method
// measures cumulative groups: group N encodes dispatches 1..N into a single
// CB+encoder, then subtracts group N-1 to isolate per-kernel GPU time.
//
// This eliminates the N × CB-overhead problem — each group's overhead is
// amortized across all dispatches in that group.

fn bench_cumulative_breakdown(
    registry: &KernelRegistry,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    cos_full: &Array,
    sin_full: &Array,
) {
    println!("\n{}", "=".repeat(120));
    println!("========== Per-Dispatch GPU Timing (Cumulative Subtraction) ==========");
    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!("Method: group N = dispatches 1..N in single CB → per-op = group[N] - group[N-1]");
    println!(
        "Warmup: {} iters, Bench: {} iters per group",
        CUMULATIVE_WARMUP, CUMULATIVE_ITERS
    );

    let mut block = build_transformer_block(device);
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch failed (cumulative)");
    {
        let setup_queue = device.newCommandQueue().unwrap();
        block
            .prepare_weights_for_graph(registry, &setup_queue)
            .expect("prepare_weights_for_graph failed (cumulative)");
    }
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Collect results per seq_len: Vec<(seq_len, Vec<(name, us)>, single_enc_us)>
    let mut all_results: Vec<(usize, Vec<(&str, f64)>, f64)> = Vec::new();

    for &seq_len in CUMULATIVE_SEQ_LENS {
        println!("\n--- seq_len={} ---", seq_len);
        autoreleasepool(|_| {

        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);
        let queue = device.newCommandQueue().unwrap();

        let per_dispatch = block
            .forward_prefill_cumulative_breakdown(
                &input,
                Some(&cos_freqs),
                Some(&sin_freqs),
                registry,
                &queue,
                device,
                MAX_SEQ_LEN,
                CUMULATIVE_WARMUP,
                CUMULATIVE_ITERS,
            )
            .expect("cumulative breakdown failed");

        // Also measure single_encoder total for reference
        let mut single_enc_latencies = Vec::with_capacity(CUMULATIVE_ITERS);
        // Warmup
        for _ in 0..CUMULATIVE_WARMUP {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let cb = queue.commandBufferWithUnretainedReferences().unwrap();
            let encoder = cb.computeCommandEncoder().unwrap();
            let _ = block
                .forward_prefill_into_encoder(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    None,
                    &mut cache,
                    registry,
                    &encoder,
                )
                .expect("single_encoder warmup failed");
            encoder.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }
        // Bench
        for _ in 0..CUMULATIVE_ITERS {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let cb = queue.commandBufferWithUnretainedReferences().unwrap();
            let encoder = cb.computeCommandEncoder().unwrap();
            let start = std::time::Instant::now();
            let _ = block
                .forward_prefill_into_encoder(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    None,
                    &mut cache,
                    registry,
                    &encoder,
                )
                .expect("single_encoder bench failed");
            encoder.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            single_enc_latencies.push(start.elapsed());
        }
        let single_enc_mean_us: f64 = {
            let sum: f64 = single_enc_latencies
                .iter()
                .map(|d| d.as_secs_f64() * 1e6)
                .sum();
            sum / single_enc_latencies.len() as f64
        };

        // Print per-dispatch table
        let total_cum_us: f64 = per_dispatch.iter().map(|(_, us)| us).sum();
        println!(
            "| {:>2} | {:<28} | {:>10} | {:>6} |",
            "#", "Dispatch", "Time (us)", "%"
        );
        println!("|{:-<4}|{:-<30}|{:-<12}|{:-<8}|", "", "", "", "");
        for (i, (name, us)) in per_dispatch.iter().enumerate() {
            let pct = if total_cum_us > 0.0 {
                us / total_cum_us * 100.0
            } else {
                0.0
            };
            println!(
                "| {:>2} | {:<28} | {:>10.1} | {:>5.1}% |",
                i + 1,
                name,
                us,
                pct
            );
        }
        println!("|{:-<4}|{:-<30}|{:-<12}|{:-<8}|", "", "", "", "");
        println!(
            "|    | {:<28} | {:>10.1} | {:>5.1}% |",
            "TOTAL (cumulative)", total_cum_us, 100.0
        );
        println!(
            "|    | {:<28} | {:>10.1} |        |",
            "single_enc (reference)", single_enc_mean_us
        );
        let diff_us = total_cum_us - single_enc_mean_us;
        let diff_pct = if single_enc_mean_us > 0.0 {
            diff_us / single_enc_mean_us * 100.0
        } else {
            0.0
        };
        println!(
            "|    | {:<28} | {:>10.1} | {:>5.1}% |",
            "sum vs reference delta", diff_us, diff_pct
        );

        all_results.push((seq_len, per_dispatch, single_enc_mean_us));

        // Let GPU cool between seq_lens
        std::thread::sleep(std::time::Duration::from_millis(100));
        }); // autoreleasepool
    }

    // ---- Cross-seq_len comparison table (MLX-style) ----
    println!("\n{}", "=".repeat(120));
    println!("========== Per-Op Timing Across Sequence Lengths (us) ==========");
    println!("Method: cumulative subtraction (single CB+encoder per group)");

    // Header
    print!("| {:<28} |", "Dispatch");
    for &seq_len in CUMULATIVE_SEQ_LENS {
        print!(" {:>8} |", format!("S={}", seq_len));
    }
    println!();
    print!("|{:-<30}|", "");
    for _ in CUMULATIVE_SEQ_LENS {
        print!("{:-<10}|", "");
    }
    println!();

    // Dispatch names (from first result)
    if let Some((_, ref first_per_dispatch, _)) = all_results.first() {
        for (i, (name, _)) in first_per_dispatch.iter().enumerate() {
            print!("| {:<28} |", name);
            for (_, ref per_dispatch, _) in &all_results {
                if i < per_dispatch.len() {
                    print!(" {:>8.1} |", per_dispatch[i].1);
                } else {
                    print!(" {:>8} |", "-");
                }
            }
            println!();
        }
    }

    // Totals
    print!("|{:-<30}|", "");
    for _ in CUMULATIVE_SEQ_LENS {
        print!("{:-<10}|", "");
    }
    println!();
    print!("| {:<28} |", "TOTAL (cumulative sum)");
    for (_, ref per_dispatch, _) in &all_results {
        let total: f64 = per_dispatch.iter().map(|(_, us)| us).sum();
        print!(" {:>8.1} |", total);
    }
    println!();
    print!("| {:<28} |", "single_enc (1 CB actual)");
    for (_, _, enc_us) in &all_results {
        print!(" {:>8.1} |", enc_us);
    }
    println!();
    print!("| {:<28} |", "TFLOPS (single_enc)");
    for (seq_len, _, enc_us) in &all_results {
        let tflops = compute_tflops(*seq_len, *enc_us);
        print!(" {:>8.2} |", tflops);
    }
    println!();

    println!("{}", "=".repeat(120));
}
