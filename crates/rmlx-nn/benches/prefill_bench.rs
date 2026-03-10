//! RMLX Prefill Benchmark Suite
//!
//! Comprehensive benchmarks for all Wave 1-3 prefill optimizations and full
//! pipeline performance (Llama-3 8B).
//!
//! Sections:
//!   1-A: SDPA NAX vs MMA (prefill attention kernels)
//!   1-C: GEMM align_K (function constant optimization)
//!   2-A: Split-K GEMM (medium M, auto-dispatch)
//!   2-B: QMV Split-K (Q4, M=1)
//!   2-C: BatchQMV Limits (Q4, M=17-32 on Ultra)
//!   Full: Llama-3 8B 32-layer prefill pipeline + single-layer profiling
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench prefill_bench
//!
//! Environment variables (set to "0" to skip):
//!   BENCH_SDPA=0        skip SDPA section
//!   BENCH_ALIGNK=0      skip align_K section
//!   BENCH_SPLITK=0      skip Split-K section
//!   BENCH_QMV=0          skip QMV Split-K section
//!   BENCH_BATCHQMV=0     skip BatchQMV section
//!   BENCH_PIPELINE=0     skip full pipeline section
//!   BENCH_LAYERS=4       override layer count for pipeline (default: 32)

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::ScopedPool;
use rmlx_nn::{
    Attention, AttentionConfig, Embedding, EmbeddingConfig, FeedForward, FeedForwardType,
    LayerKvCache, Linear, LinearConfig, TransformerBlock, TransformerConfig, TransformerModel,
};

// ---------------------------------------------------------------------------
// Llama-3 8B parameters
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 14336;
const VOCAB_SIZE: usize = 128256;
const NUM_LAYERS: usize = 32;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 500000.0;
const MAX_SEQ_LEN: usize = 4096;

/// Approximate trainable parameters per Llama-3 8B layer (for TFLOPS calc).
const PARAMS_PER_LAYER: f64 = 218_112_000.0;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    min: f64,
    max: f64,
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
            "p50={:10.1}us mean={:10.1}us std={:8.1}us p95={:10.1}us min={:10.1}us max={:10.1}us",
            self.p50, self.mean, self.std_dev, self.p95, self.min, self.max,
        )
    }
}

// ---------------------------------------------------------------------------
// f16 helpers
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

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
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

fn rand_f16_ones(device: &metal::Device, shape: &[usize]) -> Array {
    let numel: usize = shape.iter().product();
    let ones: Vec<u16> = vec![0x3C00u16; numel]; // f16 value 1.0
    let bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    Array::from_bytes(device, &bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// Quantized weight helpers
// ---------------------------------------------------------------------------

fn make_quantized_weight(
    device: &metal::Device,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: u32,
    seed: u64,
) -> QuantizedWeight {
    let mut state = seed;
    let opts = metal::MTLResourceOptions::StorageModeShared;

    let elems_per_u32 = 32 / bits as usize;
    let total_elements = out_features * in_features;
    let num_u32s = total_elements.div_ceil(elems_per_u32);
    let w_data: Vec<u32> = (0..num_u32s).map(|_| lcg_next(&mut state) as u32).collect();
    let weights_buf =
        device.new_buffer_with_data(w_data.as_ptr() as *const _, (num_u32s * 4) as u64, opts);

    let num_groups = total_elements / group_size as usize;
    let scales_data: Vec<f32> = (0..num_groups)
        .map(|_| ((lcg_next(&mut state) >> 33) as f64 / (1u64 << 31) as f64) as f32 * 0.02 + 0.001)
        .collect();
    let biases_data: Vec<f32> = (0..num_groups)
        .map(|_| ((lcg_next(&mut state) >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.01)
        .collect();

    let scales_buf = device.new_buffer_with_data(
        scales_data.as_ptr() as *const _,
        (num_groups * 4) as u64,
        opts,
    );
    let biases_buf = device.new_buffer_with_data(
        biases_data.as_ptr() as *const _,
        (num_groups * 4) as u64,
        opts,
    );

    QuantizedWeight::new(
        weights_buf,
        scales_buf,
        biases_buf,
        group_size,
        bits,
        out_features,
        in_features,
    )
    .expect("Failed to create QuantizedWeight")
}

// ---------------------------------------------------------------------------
// Layer construction helpers (f16)
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

fn build_transformer_block(device: &metal::Device, layer_idx: usize) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 1 + layer_idx as u64 * 10);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, 2 + layer_idx as u64 * 10);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, 3 + layer_idx as u64 * 10);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 4 + layer_idx as u64 * 10);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear(
        device,
        HIDDEN_SIZE,
        INTERMEDIATE_DIM,
        5 + layer_idx as u64 * 10,
    );
    let up_proj = make_linear(
        device,
        HIDDEN_SIZE,
        INTERMEDIATE_DIM,
        6 + layer_idx as u64 * 10,
    );
    let down_proj = make_linear(
        device,
        INTERMEDIATE_DIM,
        HIDDEN_SIZE,
        7 + layer_idx as u64 * 10,
    );
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
    };

    let norm1_weight = rand_f16_ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = rand_f16_ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(
        layer_idx,
        attention,
        ffn,
        norm1_weight,
        norm2_weight,
        RMS_NORM_EPS,
    )
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
// Env helpers
// ---------------------------------------------------------------------------

fn env_enabled(name: &str) -> bool {
    std::env::var(name).unwrap_or_else(|_| "1".to_string()) != "0"
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
// MLX reference numbers (from M3-Ultra-80c benchmark run, 2026-03-09)
// (single-layer prefill, f16)
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

fn compute_tflops(seq_len: usize, mean_us: f64) -> f64 {
    let total_flops = 2.0 * PARAMS_PER_LAYER * seq_len as f64;
    total_flops / (mean_us / 1e6) / 1e12
}

// =========================================================================
// Section 1-A: SDPA NAX vs MMA
// =========================================================================

fn bench_sdpa(registry: &KernelRegistry, queue: &metal::CommandQueue, device: &metal::Device) {
    println!(
        "\n--- 1-A: SDPA NAX vs MMA (prefill, causal, head_dim={}) ---",
        HEAD_DIM
    );
    println!(
        "{:<10} {:<10} {:>12} {:>12} {:>10}",
        "seq_len", "kv_len", "MMA (us)", "NAX (us)", "speedup"
    );

    let supports_nax = registry.device().tuning().supports_nax;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let configs: Vec<(usize, usize)> = vec![(128, 128), (512, 512), (2048, 2048)];

    for (seq_len, kv_len) in &configs {
        let seq_len = *seq_len;
        let kv_len = *kv_len;

        // Slab layout: contiguous [num_heads * seq_len * head_dim]
        let q_slab = rand_array(device, &[NUM_HEADS * seq_len * HEAD_DIM], 100);
        let k_slab = rand_array(device, &[NUM_KV_HEADS * kv_len * HEAD_DIM], 101);
        let v_slab = rand_array(device, &[NUM_KV_HEADS * kv_len * HEAD_DIM], 102);

        // --- MMA variant ---
        let mma_stats = {
            for _ in 0..WARMUP_ITERS {
                let _pool = ScopedPool::new();
                let cb = queue.new_command_buffer();
                let _ = ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                    registry,
                    &q_slab,
                    &k_slab,
                    &v_slab,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    kv_len,
                    None,
                    None,
                    scale,
                    true,
                    cb,
                );
                cb.commit();
                cb.wait_until_completed();
            }
            let mut times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let _pool = ScopedPool::new();
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let _ = ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                    registry,
                    &q_slab,
                    &k_slab,
                    &v_slab,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    kv_len,
                    None,
                    None,
                    scale,
                    true,
                    cb,
                );
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }
            Stats::from_durations(&times)
        };

        // --- NAX variant (M3+ only) ---
        let nax_str;
        let speedup_str;
        if supports_nax {
            let nax_stats = {
                for _ in 0..WARMUP_ITERS {
                    let _pool = ScopedPool::new();
                    let cb = queue.new_command_buffer();
                    let _ = ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                        registry,
                        &q_slab,
                        &k_slab,
                        &v_slab,
                        NUM_HEADS,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        seq_len,
                        kv_len,
                        None,
                        scale,
                        true,
                        cb,
                    );
                    cb.commit();
                    cb.wait_until_completed();
                }
                let mut times = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let _pool = ScopedPool::new();
                    let start = Instant::now();
                    let cb = queue.new_command_buffer();
                    let _ = ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                        registry,
                        &q_slab,
                        &k_slab,
                        &v_slab,
                        NUM_HEADS,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        seq_len,
                        kv_len,
                        None,
                        scale,
                        true,
                        cb,
                    );
                    cb.commit();
                    cb.wait_until_completed();
                    times.push(start.elapsed());
                }
                Stats::from_durations(&times)
            };
            let spd = mma_stats.p50 / nax_stats.p50;
            nax_str = format!("{:.1}", nax_stats.p50);
            speedup_str = format!("{:.2}x", spd);
        } else {
            nax_str = "N/A (no NAX)".to_string();
            speedup_str = "N/A".to_string();
        }

        println!(
            "{:<10} {:<10} {:>12.1} {:>12} {:>10}",
            seq_len, kv_len, mma_stats.p50, nax_str, speedup_str,
        );
    }
}

// =========================================================================
// Section 1-C: GEMM align_K
// =========================================================================

fn bench_align_k(registry: &KernelRegistry, queue: &metal::CommandQueue, device: &metal::Device) {
    println!("\n--- 1-C: GEMM align_K (function constant optimization) ---");
    println!(
        "{:<8} {:<8} {:<8} {:>10} {:>10} {:>12}",
        "M", "N", "K", "p50 (us)", "TFLOPS", "note"
    );

    let m = 512;
    let n = 4096;

    for &(k, note) in &[(4096usize, "aligned"), (4097, "unaligned")] {
        let a = rand_array(device, &[m, k], 42);
        let b = rand_array(device, &[k, n], 44);

        for _ in 0..WARMUP_ITERS {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb);
            cb.commit();
            cb.wait_until_completed();
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let _pool = ScopedPool::new();
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb);
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&times);
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tflops = flops / (stats.p50 * 1e-6) / 1e12;

        println!(
            "{:<8} {:<8} {:<8} {:>10.1} {:>10.2} {:>12}",
            m, n, k, stats.p50, tflops, note,
        );
    }
}

// =========================================================================
// Section 2-A: Split-K GEMM (auto-dispatch for medium M)
// =========================================================================

fn bench_splitk_gemm(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
) {
    println!("\n--- 2-A: Split-K GEMM (auto-dispatch, K=14336) ---");
    println!(
        "{:<8} {:<8} {:<8} {:>10} {:>10}",
        "M", "N", "K", "p50 (us)", "TFLOPS"
    );

    let n = 4096;
    let k = 14336;

    for &m in &[32usize, 64, 128] {
        let a = rand_array(device, &[m, k], 42);
        let b = rand_array(device, &[k, n], 44);

        for _ in 0..WARMUP_ITERS {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb);
            cb.commit();
            cb.wait_until_completed();
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let _pool = ScopedPool::new();
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb);
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&times);
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tflops = flops / (stats.p50 * 1e-6) / 1e12;

        println!(
            "{:<8} {:<8} {:<8} {:>10.1} {:>10.2}",
            m, n, k, stats.p50, tflops,
        );
    }
}

// =========================================================================
// Section 2-B: QMV Split-K (Q4, M=1)
// =========================================================================

fn bench_qmv_splitk(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
) {
    println!("\n--- 2-B: QMV Split-K (Q4, M=1, group_size=32) ---");
    println!(
        "{:<6} {:<8} {:<8} {:>10} {:>10} {:>22}",
        "M", "K", "N", "p50 (us)", "TFLOPS", "note"
    );

    let scenarios: Vec<(usize, usize, usize, &str)> = vec![
        (1, 14336, 2048, "K large, should split-K"),
        (1, 14336, 4096, "N large, may not split"),
        (1, 4096, 2048, "K small, may not split"),
    ];

    for (m, k, n, note) in &scenarios {
        let x = rand_array(device, &[*m, *k], 42);
        let qw = make_quantized_weight(device, *n, *k, 4, 32, 100);

        for _ in 0..WARMUP_ITERS {
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue);
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue);
            times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&times);
        let flops = 2.0 * *m as f64 * *k as f64 * *n as f64;
        let tflops = flops / (stats.p50 * 1e-6) / 1e12;

        println!(
            "{:<6} {:<8} {:<8} {:>10.1} {:>10.3} {:>22}",
            m, k, n, stats.p50, tflops, note,
        );
    }
}

// =========================================================================
// Section 2-C: BatchQMV Limits (Q4, M=17-32 on Ultra)
// =========================================================================

fn bench_batch_qmv(registry: &KernelRegistry, queue: &metal::CommandQueue, device: &metal::Device) {
    println!("\n--- 2-C: BatchQMV Limits (Q4, group_size=32) ---");
    println!(
        "{:<6} {:<8} {:<8} {:>10} {:>10}",
        "M", "K", "N", "p50 (us)", "TFLOPS"
    );

    let scenarios: Vec<(usize, usize, usize)> =
        vec![(17, 4096, 4096), (24, 4096, 4096), (32, 2048, 2048)];

    for (m, k, n) in &scenarios {
        let x = rand_array(device, &[*m, *k], 42);
        let qw = make_quantized_weight(device, *n, *k, 4, 32, 100);

        for _ in 0..WARMUP_ITERS {
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue);
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue);
            times.push(start.elapsed());
        }
        let stats = Stats::from_durations(&times);
        let flops = 2.0 * *m as f64 * *k as f64 * *n as f64;
        let tflops = flops / (stats.p50 * 1e-6) / 1e12;

        println!(
            "{:<6} {:<8} {:<8} {:>10.1} {:>10.2}",
            m, k, n, stats.p50, tflops,
        );
    }
}

// =========================================================================
// Full Pipeline: Llama-3 8B prefill (N-layer + single-layer profiling)
// =========================================================================

fn bench_pipeline(registry: &KernelRegistry, queue: &metal::CommandQueue, device: &metal::Device) {
    let num_layers = env_usize("BENCH_LAYERS", NUM_LAYERS);

    // ---- Full-model prefill ----
    println!(
        "\n--- Full Prefill Pipeline (Llama-3 8B, {} layers, f16) ---",
        num_layers
    );
    println!("Building model ({} layers)...", num_layers);

    let config = TransformerConfig {
        hidden_size: HIDDEN_SIZE,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        num_layers,
        vocab_size: VOCAB_SIZE,
        max_seq_len: MAX_SEQ_LEN,
        rope_theta: ROPE_THETA,
        rms_norm_eps: RMS_NORM_EPS,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: INTERMEDIATE_DIM,
        },
    };

    let layers: Vec<TransformerBlock> = (0..num_layers)
        .map(|i| build_transformer_block(device, i))
        .collect();

    let embed_weight = rand_array(device, &[VOCAB_SIZE, HIDDEN_SIZE], 900);
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: VOCAB_SIZE,
            embed_dim: HIDDEN_SIZE,
        },
        embed_weight,
    )
    .expect("embedding");

    let final_norm_weight = rand_f16_ones(device, &[HIDDEN_SIZE]);
    let lm_head = make_linear(device, HIDDEN_SIZE, VOCAB_SIZE, 950);

    let model = TransformerModel::from_parts(config, embedding, layers, final_norm_weight, lm_head)
        .expect("model");

    println!(
        "\n{:<10} {:>12} {:>14} {:>12} {:>10}",
        "seq_len", "total (ms)", "per-layer (us)", "tok/s", "std (ms)"
    );

    let seq_lens = [128usize, 512, 2048];

    for &seq_len in &seq_lens {
        let token_ids: Vec<u32> = (0..seq_len).map(|i| (i % VOCAB_SIZE) as u32).collect();

        let mut caches: Vec<LayerKvCache> = (0..num_layers)
            .map(|_| {
                LayerKvCache::preallocated(
                    device,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    MAX_SEQ_LEN,
                    DType::Float16,
                )
            })
            .collect();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _pool = ScopedPool::new();
            for c in caches.iter_mut() {
                c.seq_len = 0;
            }
            let event = GpuEvent::new(device);
            let _ = model.forward_prefill_graph(
                &token_ids,
                None,
                None,
                None,
                &mut caches,
                registry,
                queue,
                &event,
            );
        }

        // Benchmark
        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let _pool = ScopedPool::new();
            for c in caches.iter_mut() {
                c.seq_len = 0;
            }
            let event = GpuEvent::new(device);
            let start = Instant::now();
            let _ = model.forward_prefill_graph(
                &token_ids,
                None,
                None,
                None,
                &mut caches,
                registry,
                queue,
                &event,
            );
            times.push(start.elapsed());
        }

        let stats = Stats::from_durations(&times);
        let total_ms = stats.p50 / 1000.0;
        let per_layer_us = stats.p50 / num_layers as f64;
        let tok_per_sec = seq_len as f64 / (stats.p50 * 1e-6);

        println!(
            "{:<10} {:>12.2} {:>14.1} {:>12.0} {:>10.2}",
            seq_len,
            total_ms,
            per_layer_us,
            tok_per_sec,
            stats.std_dev / 1000.0,
        );
    }

    // ---- Single-layer prefill profiling (with RoPE + mask) ----
    println!("\n--- Single-Layer Prefill Profiling (layer 0, with RoPE + causal mask) ---");

    let mut block = build_transformer_block(device, 0);
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch");
    let prep_queue = device.new_command_queue();
    block
        .prepare_weights_for_graph(registry, &prep_queue)
        .expect("prepare_weights_for_graph");

    // Precompute RoPE cos/sin tables
    let (cos_vec, sin_vec) = ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
        .expect("precompute_freqs");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8}",
        "seq_len", "p50 (us)", "mean (us)", "min (us)", "TFLOPS", "MLX ref", "vs MLX"
    );

    for &seq_len in &seq_lens {
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);
        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice");
        let mask = build_causal_mask(device, seq_len);

        let mut cache =
            LayerKvCache::preallocated(device, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DType::Float16);

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _pool = ScopedPool::new();
            cache.seq_len = 0;
            let cb = queue.new_command_buffer();
            let _ = block.forward_prefill_single_cb(
                &input,
                Some(&cos_freqs),
                Some(&sin_freqs),
                Some(&mask),
                &mut cache,
                registry,
                cb,
            );
            cb.commit();
            cb.wait_until_completed();
            assert_cb_ok(cb, "single-layer warmup");
        }

        // Bench
        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let _pool = ScopedPool::new();
            cache.seq_len = 0;
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let _ = block.forward_prefill_single_cb(
                &input,
                Some(&cos_freqs),
                Some(&sin_freqs),
                Some(&mask),
                &mut cache,
                registry,
                cb,
            );
            cb.commit();
            cb.wait_until_completed();
            assert_cb_ok(cb, "single-layer bench");
            times.push(start.elapsed());
        }

        let stats = Stats::from_durations(&times);
        let tflops = compute_tflops(seq_len, stats.p50);

        let (mlx_str, vs_mlx_str) = match mlx_ref_us(seq_len) {
            Some(mlx_us) => {
                let ratio = stats.p50 / mlx_us;
                (format!("{:.0}", mlx_us), format!("{:.2}x", ratio))
            }
            None => ("-".to_string(), "-".to_string()),
        };

        println!(
            "{:<10} {:>10.1} {:>10.1} {:>10.1} {:>10.2} {:>12} {:>8}",
            seq_len, stats.p50, stats.mean, stats.min, tflops, mlx_str, vs_mlx_str,
        );
    }

    // Weight size reference
    let elem = 2.0_f64; // f16
    let q_w = (HIDDEN_SIZE * HIDDEN_SIZE) as f64 * elem;
    let kv_w = (HIDDEN_SIZE * NUM_KV_HEADS * HEAD_DIM) as f64 * elem * 2.0;
    let o_w = q_w;
    let ffn_w = (HIDDEN_SIZE * INTERMEDIATE_DIM) as f64 * elem * 3.0;
    let weight_mb = (q_w + kv_w + o_w + ffn_w) / 1e6;
    println!("\nWeight size per layer: {:.1} MB (f16)", weight_mb);
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    println!("=== RMLX Prefill Benchmark Suite ===");
    println!(
        "Device: {} (cores={}, nax={})",
        gpu.name(),
        gpu.tuning().gpu_cores,
        gpu.tuning().supports_nax,
    );
    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}, vocab={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM, VOCAB_SIZE,
    );
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );

    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("kernel registration failed");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    // --- Section 1-A: SDPA ---
    if env_enabled("BENCH_SDPA") {
        bench_sdpa(&registry, &queue, device);
    }

    // --- Section 1-C: align_K ---
    if env_enabled("BENCH_ALIGNK") {
        bench_align_k(&registry, &queue, device);
    }

    // --- Section 2-A: Split-K GEMM ---
    if env_enabled("BENCH_SPLITK") {
        bench_splitk_gemm(&registry, &queue, device);
    }

    // --- Section 2-B: QMV Split-K ---
    if env_enabled("BENCH_QMV") {
        bench_qmv_splitk(&registry, &queue, device);
    }

    // --- Section 2-C: BatchQMV ---
    if env_enabled("BENCH_BATCHQMV") {
        bench_batch_qmv(&registry, &queue, device);
    }

    // --- Full pipeline ---
    if env_enabled("BENCH_PIPELINE") {
        bench_pipeline(&registry, &queue, device);
    }

    println!("\n=== Done ===");
}
