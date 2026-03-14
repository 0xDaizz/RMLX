//! PRODUCTION PATH — E2E prefill benchmark for 32-layer TransformerModel.
//!
//! Measures full 32-layer Qwen 7B-style forward pass latency across multiple
//! sequence lengths. Three paths are compared:
//!
//!   1. `forward()` — per-op baseline (no graph batching)
//!   2. `forward_graph_unified(Prefill{layers_per_cb:4})` — production ExecGraph path
//!   3. `forward_graph_unified(CompiledPrefill)` — single CB + single encoder path
//!
//! Each seq_len gets a **fresh command queue** to prevent cross-contamination.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench e2e_prefill_bench

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::autoreleasepool;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_nn::{
    Attention, AttentionConfig, Embedding, EmbeddingConfig, FeedForward, FeedForwardType,
    ForwardMode, LayerKvCache, Linear, LinearConfig, TransformerBlock, TransformerConfig,
    TransformerModel,
};

// ---------------------------------------------------------------------------
// Qwen 7B-style config
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 14336;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 1000000.0;
const MAX_SEQ_LEN: usize = 2048;
const VOCAB_SIZE: usize = 152064;
const NUM_LAYERS: usize = 32;

/// Approximate FLOPs per layer for TFLOPS computation.
///
/// Dominant GEMMs per layer (2*M*N*K each):
///   Q_proj: 2 * M * hidden * hidden          = 2*M*4096*4096
///   K_proj: 2 * M * hidden * kv_size          = 2*M*4096*1024
///   V_proj: 2 * M * hidden * kv_size          = 2*M*4096*1024
///   O_proj: 2 * M * hidden * hidden           = 2*M*4096*4096
///   gate:   2 * M * hidden * intermediate     = 2*M*4096*14336
///   up:     2 * M * hidden * intermediate     = 2*M*4096*14336
///   down:   2 * M * intermediate * hidden     = 2*M*14336*4096
///
/// Total per layer = 2*M * (4096*4096 + 1024*4096 + 1024*4096 + 4096*4096
///                          + 4096*14336 + 4096*14336 + 14336*4096)
///                 = 2*M * (16M + 4M + 4M + 16M + 58M + 58M + 58M)
///                 = 2*M * 214_695_936
const FLOPS_PER_TOKEN_PER_LAYER: f64 = 2.0 * 214_695_936.0;

const SEQ_LENS: &[usize] = &[32, 64, 128, 256, 512, 1024];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

// ---------------------------------------------------------------------------
// Stats helper (same pattern as prefill_bench.rs)
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
// f16 helpers (same as prefill_bench.rs)
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

fn ones_f16(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, size: usize) -> Array {
    let ones: Vec<u16> = vec![0x3C00u16; size]; // f16 1.0
    let bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    Array::from_bytes(device, &bytes, vec![size], DType::Float16)
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

fn build_transformer_block(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    layer_idx: usize,
    seed_base: u64,
) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;
    let s = seed_base;

    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, s);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, s + 1);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, s + 2);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, s + 3);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, s + 4);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, s + 5);
    let down_proj = make_linear(device, INTERMEDIATE_DIM, HIDDEN_SIZE, s + 6);
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
        gate_proj_quantized: None,
        up_proj_quantized: None,
        down_proj_quantized: None,
    };

    let norm1_weight = ones_f16(device, HIDDEN_SIZE);
    let norm2_weight = ones_f16(device, HIDDEN_SIZE);

    TransformerBlock::from_parts(
        layer_idx,
        attention,
        ffn,
        norm1_weight,
        norm2_weight,
        RMS_NORM_EPS,
    )
}

fn build_model(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> TransformerModel {
    println!("Building 32-layer TransformerModel with random weights...");
    let mut layers = Vec::with_capacity(NUM_LAYERS);
    for i in 0..NUM_LAYERS {
        if i % 8 == 0 {
            println!("  building layer {}/{}...", i, NUM_LAYERS);
        }
        layers.push(build_transformer_block(device, i, (i as u64 + 1) * 100));
    }

    // Embedding: [vocab_size, hidden_size]
    let embed_weight = rand_array(device, &[VOCAB_SIZE, HIDDEN_SIZE], 9999);
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            embed_dim: HIDDEN_SIZE,
        },
        embed_weight,
    )
    .expect("embedding");

    // Final norm weight
    let final_norm = ones_f16(device, HIDDEN_SIZE);

    // LM head: [vocab_size, hidden_size]
    let lm_head = make_linear(device, HIDDEN_SIZE, VOCAB_SIZE, 8888);

    let config = TransformerConfig {
        hidden_size: HIDDEN_SIZE,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        num_layers: NUM_LAYERS,
        vocab_size: VOCAB_SIZE,
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
        rms_norm_eps: RMS_NORM_EPS,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: INTERMEDIATE_DIM,
        },
    };

    TransformerModel::from_parts(config, embedding, layers, final_norm, lm_head)
        .expect("TransformerModel::from_parts")
}

fn make_caches(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> Vec<LayerKvCache> {
    (0..NUM_LAYERS)
        .map(|_| {
            LayerKvCache::preallocated(device, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DType::Float16)
        })
        .collect()
}

fn reset_caches(caches: &mut Vec<LayerKvCache>) {
    for cache in caches.iter_mut() {
        cache.seq_len = 0;
    }
}

// ---------------------------------------------------------------------------
// TFLOPS computation
// ---------------------------------------------------------------------------

fn compute_tflops(seq_len: usize, mean_us: f64) -> f64 {
    let total_flops = FLOPS_PER_TOKEN_PER_LAYER * NUM_LAYERS as f64 * seq_len as f64;
    let seconds = mean_us / 1e6;
    total_flops / seconds / 1e12
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
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}, layers={}, vocab={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM, NUM_LAYERS, VOCAB_SIZE
    );
    println!("dtype: float16");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!(
        "FLOPs/token/layer: {:.0}, total FLOPs/token: {:.0}",
        FLOPS_PER_TOKEN_PER_LAYER,
        FLOPS_PER_TOKEN_PER_LAYER * NUM_LAYERS as f64
    );

    // Enable thread-local buffer pool to eliminate Metal allocation overhead
    rmlx_core::array::enable_array_pool();
    println!("Array buffer pool: ENABLED");

    // Build model with random weights
    let mut model = build_model(device);
    println!("Model built. Preparing weights...");

    // Pre-transpose weights for GEMM paths
    {
        let setup_queue = device.newCommandQueue().unwrap();
        model
            .prepare_weights_for_graph(&registry, &setup_queue)
            .expect("prepare_weights_for_graph failed");
    }
    // Merge QKV and gate+up weights for CompiledPrefill (into_encoder) path
    for block in model.layers_mut() {
        block
            .prepare_weights_9dispatch(device)
            .expect("prepare_weights_9dispatch failed");
    }

    // Let Metal driver fully drain GPU resources from weight preparation
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("Weights prepared.");

    // Precompute RoPE cos/sin tables: shape [MAX_SEQ_LEN, HEAD_DIM/2]
    let (cos_vec, sin_vec) = ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
        .expect("precompute_freqs failed");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    // Generate token_ids (reused across seq_lens, sliced as needed)
    let max_tokens: usize = *SEQ_LENS.last().unwrap();
    let mut token_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut state: u64 = 12345;
    for _ in 0..max_tokens {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        token_ids.push((state >> 33) as u32 % VOCAB_SIZE as u32);
    }

    // Collect results: (seq_len, forward_stats, tflops, prefill_graph_stats, tflops, compiled_stats, tflops)
    let mut results: Vec<(usize, Stats, f64, Stats, f64, Stats, f64)> = Vec::new();

    for &seq_len in SEQ_LENS {
        println!("\n--- seq_len={} ---", seq_len);

        // Slice RoPE tables to [seq_len, HEAD_DIM/2]
        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");

        let tids = &token_ids[..seq_len];

        // ==== Benchmark 1: forward() ====
        let queue_fwd = device.newCommandQueue().unwrap();
        let stats_fwd = {
            let mut caches = make_caches(device);

            // Warmup
            autoreleasepool(|_| {
                for _ in 0..WARMUP_ITERS {
                    reset_caches(&mut caches);
                    let _out = model
                        .forward(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches),
                            &registry,
                            &queue_fwd,
                        )
                        .expect("forward warmup failed");
                }
            });

            // Benchmark
            let mut latencies = Vec::with_capacity(BENCH_ITERS);
            autoreleasepool(|_| {
                for _ in 0..BENCH_ITERS {
                    reset_caches(&mut caches);
                    let start = Instant::now();
                    let _out = model
                        .forward(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches),
                            &registry,
                            &queue_fwd,
                        )
                        .expect("forward failed");
                    latencies.push(start.elapsed());
                }
            });

            Stats::from_durations(&latencies)
        };

        let tflops_fwd = compute_tflops(seq_len, stats_fwd.mean);
        println!("  forward          : {}", stats_fwd);
        println!("    estimated TFLOPS: {:.2}", tflops_fwd);

        // ==== Benchmark 2: forward_graph_unified(Prefill) (single CB per layer, production path) ====
        let queue_graph = device.newCommandQueue().unwrap();
        let event = GpuEvent::new(device);
        let stats_graph = {
            let mut caches = make_caches(device);

            // Warmup
            autoreleasepool(|_| {
                for _ in 0..WARMUP_ITERS {
                    reset_caches(&mut caches);
                    let _out = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::Prefill { layers_per_cb: 4 },
                            &registry,
                            &queue_graph,
                            &event,
                        )
                        .expect("forward_graph_unified warmup failed");
                    // Force GPU completion by submitting an empty CB and waiting
                    let sync_cb = queue_graph.commandBuffer().unwrap();
                    sync_cb.commit();
                    sync_cb.waitUntilCompleted();
                }
            });

            // Debug: single timed call with output validation
            autoreleasepool(|_| {
                reset_caches(&mut caches);
                let debug_start = Instant::now();
                let debug_out = model
                    .forward_graph_unified(
                        tids,
                        Some(&cos_freqs),
                        Some(&sin_freqs),
                        None,
                        Some(&mut caches[..]),
                        ForwardMode::Prefill { layers_per_cb: 4 },
                        &registry,
                        &queue_graph,
                        &event,
                    )
                    .expect("debug forward failed");
                // Force GPU sync
                let sync_cb = queue_graph.commandBuffer().unwrap();
                sync_cb.commit();
                sync_cb.waitUntilCompleted();
                let debug_elapsed = debug_start.elapsed();
                println!(
                    "  [DEBUG] prefill_graph single call: {:.1}ms",
                    debug_elapsed.as_secs_f64() * 1000.0
                );
                println!("  [DEBUG] output shape: {:?}", debug_out.shape());
                // Read first bytes to verify non-zero computation
                let first_bytes = debug_out.to_bytes();
                let show_len = 8.min(first_bytes.len());
                println!(
                    "  [DEBUG] output bytes[0..{}]: {:?}",
                    show_len,
                    &first_bytes[..show_len]
                );
            });

            // Benchmark
            let mut latencies = Vec::with_capacity(BENCH_ITERS);
            autoreleasepool(|_| {
                for _ in 0..BENCH_ITERS {
                    reset_caches(&mut caches);
                    let start = Instant::now();
                    let _out = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::Prefill { layers_per_cb: 4 },
                            &registry,
                            &queue_graph,
                            &event,
                        )
                        .expect("forward_graph_unified failed");
                    // Force GPU completion by submitting an empty CB and waiting
                    let sync_cb = queue_graph.commandBuffer().unwrap();
                    sync_cb.commit();
                    sync_cb.waitUntilCompleted();
                    latencies.push(start.elapsed());
                }
            });

            Stats::from_durations(&latencies)
        };

        let tflops_graph = compute_tflops(seq_len, stats_graph.mean);
        println!("  prefill_graph    : {}", stats_graph);
        println!("    estimated TFLOPS: {:.2}", tflops_graph);

        // ==== Benchmark 3: forward_graph_unified(CompiledPrefill) (single CB + single encoder) ====
        let queue_compiled = device.newCommandQueue().unwrap();
        let event_compiled = GpuEvent::new(device);
        let (stats_compiled, alloc_count, alloc_nanos, alloc_bytes) = {
            let mut caches = make_caches(device);

            // Warmup
            autoreleasepool(|_| {
                for _ in 0..WARMUP_ITERS {
                    reset_caches(&mut caches);
                    let _out = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::CompiledPrefill,
                            &registry,
                            &queue_compiled,
                            &event_compiled,
                        )
                        .expect("compiled_prefill warmup failed");
                    let sync_cb = queue_compiled.commandBuffer().unwrap();
                    sync_cb.commit();
                    sync_cb.waitUntilCompleted();
                }
            });

            // Benchmark
            let mut latencies = Vec::with_capacity(BENCH_ITERS);
            rmlx_core::array::reset_alloc_stats();
            autoreleasepool(|_| {
                for _ in 0..BENCH_ITERS {
                    reset_caches(&mut caches);
                    let start = Instant::now();
                    let _out = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::CompiledPrefill,
                            &registry,
                            &queue_compiled,
                            &event_compiled,
                        )
                        .expect("compiled_prefill failed");
                    let sync_cb = queue_compiled.commandBuffer().unwrap();
                    sync_cb.commit();
                    sync_cb.waitUntilCompleted();
                    latencies.push(start.elapsed());
                }
            });
            let (alloc_count, alloc_nanos, alloc_bytes) = rmlx_core::array::get_alloc_stats();

            (
                Stats::from_durations(&latencies),
                alloc_count,
                alloc_nanos,
                alloc_bytes,
            )
        };

        let tflops_compiled = compute_tflops(seq_len, stats_compiled.mean);
        println!("  compiled_prefill : {}", stats_compiled);
        println!("    estimated TFLOPS: {:.2}", tflops_compiled);
        {
            let alloc_us_total = alloc_nanos as f64 / 1000.0;
            let alloc_us_per_iter = alloc_us_total / BENCH_ITERS as f64;
            let alloc_count_per_iter = alloc_count / BENCH_ITERS as u64;
            let alloc_mb_per_iter = (alloc_bytes / BENCH_ITERS as u64) as f64 / 1024.0 / 1024.0;
            println!("  alloc stats (compiled_prefill, per iter):");
            println!("    count: {} allocations", alloc_count_per_iter);
            println!(
                "    time:  {:.0}us ({:.1}% of total)",
                alloc_us_per_iter,
                alloc_us_per_iter / stats_compiled.mean * 100.0
            );
            println!("    bytes: {:.1} MB", alloc_mb_per_iter);
        }

        // ==== Benchmark 4: CB overhead analysis ====
        // Compare different layers_per_cb values to measure CB creation overhead.
        // Only run for small seq_lens to keep total bench time manageable.
        if seq_len == 32 || seq_len == 64 || seq_len == 128 {
            let queue_oh = device.newCommandQueue().unwrap();
            let event_oh = GpuEvent::new(device);
            let n_runs = 10;

            // Test different layers_per_cb to measure CB creation overhead
            let cb_configs = vec![1, 2, 4, 8, 16, 32];
            println!("\n  --- CB overhead analysis (seq_len={}) ---", seq_len);
            println!(
                "  | {:>12} | {:>12} | {:>12} |",
                "layers/CB", "mean (us)", "vs compiled"
            );
            println!("  |--------------|--------------|--------------|");

            for &layers_per_cb in &cb_configs {
                let mut times = Vec::with_capacity(n_runs);
                let mut caches = make_caches(device);

                // Warmup
                autoreleasepool(|_| {
                    for _ in 0..3 {
                        reset_caches(&mut caches);
                        let _ = model
                            .forward_graph_unified(
                                tids,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                Some(&mut caches[..]),
                                ForwardMode::Prefill { layers_per_cb },
                                &registry,
                                &queue_oh,
                                &event_oh,
                            )
                            .expect("overhead warmup failed");
                        let sync = queue_oh.commandBuffer().unwrap();
                        sync.commit();
                        sync.waitUntilCompleted();
                    }
                });

                // Measure
                autoreleasepool(|_| {
                    for _ in 0..n_runs {
                        reset_caches(&mut caches);
                        let start = Instant::now();
                        let _ = model
                            .forward_graph_unified(
                                tids,
                                Some(&cos_freqs),
                                Some(&sin_freqs),
                                None,
                                Some(&mut caches[..]),
                                ForwardMode::Prefill { layers_per_cb },
                                &registry,
                                &queue_oh,
                                &event_oh,
                            )
                            .expect("overhead bench failed");
                        let sync = queue_oh.commandBuffer().unwrap();
                        sync.commit();
                        sync.waitUntilCompleted();
                        times.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                });

                let mean: f64 = times.iter().sum::<f64>() / n_runs as f64;
                let diff = mean - stats_compiled.mean;
                println!(
                    "  | {:>12} | {:>12.0} | {:>+12.0} |",
                    layers_per_cb, mean, diff
                );
            }

            // Also print CompiledPrefill reference
            println!(
                "  | {:>12} | {:>12.0} | {:>12} |",
                "compiled", stats_compiled.mean, "ref"
            );
        }

        results.push((
            seq_len,
            stats_fwd,
            tflops_fwd,
            stats_graph,
            tflops_graph,
            stats_compiled,
            tflops_compiled,
        ));
    }

    // ---------------------------------------------------------------------------
    // Summary table
    // ---------------------------------------------------------------------------

    println!("\n\n========== E2E Prefill Summary (32-layer, Qwen 7B-style) ==========");
    println!(
        "| {:>7} | {:>12} | {:>8} | {:>15} | {:>8} | {:>8} | {:>15} | {:>8} | {:>8} |",
        "seq_len",
        "forward (us)",
        "TFLOPS",
        "prefill_graph",
        "TFLOPS",
        "speedup",
        "compiled_pfill",
        "TFLOPS",
        "speedup"
    );
    println!(
        "|---------|--------------|----------|-----------------|----------|----------|-----------------|----------|----------|"
    );
    for &(seq_len, ref s_fwd, t_fwd, ref s_graph, t_graph, ref s_compiled, t_compiled) in &results {
        let sp_graph = s_fwd.mean / s_graph.mean;
        let sp_compiled = s_fwd.mean / s_compiled.mean;
        println!(
            "| {:>7} | {:>12.1} | {:>8.2} | {:>15.1} | {:>8.2} | {:>6.2}x | {:>15.1} | {:>8.2} | {:>6.2}x |",
            seq_len, s_fwd.mean, t_fwd, s_graph.mean, t_graph, sp_graph, s_compiled.mean, t_compiled, sp_compiled
        );
    }
    println!();

    // ---------------------------------------------------------------------------
    // Fused Norm Threshold Sweep (CompiledPrefill only)
    // ---------------------------------------------------------------------------

    println!("\n\n========== Fused Norm Threshold Sweep ==========");
    println!(
        "| {:>7} | {:>12} | {:>12} | {:>12} | {:>12} |",
        "seq_len", "fused_all", "m<=32", "m<=64", "m<=128"
    );
    println!("|---------|--------------|--------------|--------------|--------------|");

    let thresholds = [0usize, 32, 64, 128];

    for &seq_len in SEQ_LENS {
        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice failed");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice failed");
        let tids = &token_ids[..seq_len];
        let queue_sweep = device.newCommandQueue().unwrap();
        let event_sweep = GpuEvent::new(device);

        let mut row_values: Vec<f64> = Vec::new();

        for &threshold in &thresholds {
            rmlx_nn::set_fused_norm_threshold(threshold);

            let mut caches = make_caches(device);

            // Warmup
            autoreleasepool(|_| {
                for _ in 0..3 {
                    reset_caches(&mut caches);
                    let _ = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::CompiledPrefill,
                            &registry,
                            &queue_sweep,
                            &event_sweep,
                        )
                        .expect("sweep warmup failed");
                    let sync = queue_sweep.commandBuffer().unwrap();
                    sync.commit();
                    sync.waitUntilCompleted();
                }
            });

            // Measure
            let mut times = Vec::with_capacity(5);
            autoreleasepool(|_| {
                for _ in 0..5 {
                    reset_caches(&mut caches);
                    let start = Instant::now();
                    let _ = model
                        .forward_graph_unified(
                            tids,
                            Some(&cos_freqs),
                            Some(&sin_freqs),
                            None,
                            Some(&mut caches[..]),
                            ForwardMode::CompiledPrefill,
                            &registry,
                            &queue_sweep,
                            &event_sweep,
                        )
                        .expect("sweep failed");
                    let sync = queue_sweep.commandBuffer().unwrap();
                    sync.commit();
                    sync.waitUntilCompleted();
                    times.push(start.elapsed().as_secs_f64() * 1e6);
                }
            });

            let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
            row_values.push(mean);
        }

        // Find best
        let best_idx = row_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        print!("| {:>7} |", seq_len);
        for (i, val) in row_values.iter().enumerate() {
            if i == best_idx {
                print!(" {:>10.0}* |", val);
            } else {
                print!(" {:>11.0} |", val);
            }
        }
        println!();
    }

    // Reset to default
    rmlx_nn::set_fused_norm_threshold(0);
    println!("(* = best for this seq_len)");

    // Print buffer pool stats
    let (hits, misses, cached) = rmlx_core::array::array_pool_stats();
    println!("=== Array Buffer Pool Stats ===");
    println!("  hits: {}, misses: {}, cached: {}", hits, misses, cached);
    println!(
        "  hit rate: {:.1}%",
        if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64 * 100.0
        } else {
            0.0
        }
    );
    rmlx_core::array::disable_array_pool();
}
