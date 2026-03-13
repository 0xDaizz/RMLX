//! ⚠️ NON-PRODUCTION PATH — GPU timestamp per-op profiling. Diagnostic only;
//! measures pure GPU execution time per op, not production TransformerModel forward path.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! GPU-timestamp-based per-operation profiling benchmark for RMLX prefill.
//!
//! Unlike `op_profile_bench.rs` which uses `Instant::now()` (wall-clock),
//! this benchmark uses Metal's `GPUStartTime`/`GPUEndTime` properties on
//! completed CommandBuffers to measure **pure GPU execution time**, excluding
//! CB creation, commit, and CPU-side wait overhead.
//!
//! This enables accurate per-op breakdown where Sum(GPU times) can be
//! meaningfully compared against a full-pipeline GPU time to measure
//! true GPU pipelining overlap.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench gpu_profile_bench

#![allow(unexpected_cfgs)]

#[macro_use]
extern crate objc;

use std::time::Instant;

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::ScopedPool;
use rmlx_nn::{
    Attention, AttentionConfig, FeedForward, LayerKvCache, Linear, LinearConfig, TransformerBlock,
};

// ---------------------------------------------------------------------------
// MoE Expert Layer config (Qwen 3.5-style dimensions)
// ---------------------------------------------------------------------------

const HIDDEN_SIZE: usize = 3584;
const NUM_HEADS: usize = 28;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 2560;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 1000000.0;
const MAX_SEQ_LEN: usize = 2048;

const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 3584
const K_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512
const V_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512
const TOTAL_QKV: usize = Q_DIM + K_DIM + V_DIM; // 4608

const SEQ_LENS: &[usize] = &[128, 256, 512, 1024, 2048];
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

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

fn rand_f16_bytes(numel: usize, seed: u64) -> Vec<u8> {
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
    f16_bytes
}

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let f16_bytes = rand_f16_bytes(numel, seed);
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

fn ones_f16(device: &metal::Device, size: usize) -> Array {
    let ones: Vec<u16> = vec![0x3C00u16; size];
    let bytes: Vec<u8> = ones.iter().flat_map(|h| h.to_le_bytes()).collect();
    Array::from_bytes(device, &bytes, vec![size], DType::Float16)
}

// ---------------------------------------------------------------------------
// Layer construction helpers (for full-pipeline baseline only)
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

    let norm1_weight = ones_f16(device, HIDDEN_SIZE);
    let norm2_weight = ones_f16(device, HIDDEN_SIZE);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Build merged transposed weight on CPU
// ---------------------------------------------------------------------------

fn build_merged_weight_t(
    device: &metal::Device,
    weights_row_major: &[(&[u8], usize, usize)],
) -> Array {
    let in_dim = weights_row_major[0].2;
    let total_out: usize = weights_row_major.iter().map(|w| w.1).sum();

    let mut result_bytes = vec![0u8; in_dim * total_out * 2];

    let mut col_offset = 0;
    for &(bytes, rows, cols) in weights_row_major {
        assert_eq!(cols, in_dim);
        for r in 0..rows {
            for c in 0..cols {
                let src_idx = (r * cols + c) * 2;
                let dst_idx = (c * total_out + col_offset + r) * 2;
                result_bytes[dst_idx] = bytes[src_idx];
                result_bytes[dst_idx + 1] = bytes[src_idx + 1];
            }
        }
        col_offset += rows;
    }

    Array::from_bytes(
        device,
        &result_bytes,
        vec![in_dim, total_out],
        DType::Float16,
    )
}

fn transpose_weight_cpu(device: &metal::Device, bytes: &[u8], rows: usize, cols: usize) -> Array {
    let mut result = vec![0u8; rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            let src_idx = (r * cols + c) * 2;
            let dst_idx = (c * rows + r) * 2;
            result[dst_idx] = bytes[src_idx];
            result[dst_idx + 1] = bytes[src_idx + 1];
        }
    }
    Array::from_bytes(device, &result, vec![cols, rows], DType::Float16)
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
// GPU timestamp helpers
// ---------------------------------------------------------------------------

/// Extract pure GPU execution time from a completed CommandBuffer.
/// Uses Metal's GPUStartTime/GPUEndTime properties (seconds, f64).
/// Only valid after wait_until_completed().
fn gpu_time_us(cb: &metal::CommandBufferRef) -> f64 {
    unsafe {
        let obj: *const objc::runtime::Object = cb as *const metal::CommandBufferRef as *const _;
        let start: f64 = msg_send![obj, GPUStartTime];
        let end: f64 = msg_send![obj, GPUEndTime];
        (end - start) * 1_000_000.0
    }
}

/// Measure a single op with both GPU timestamp and wall-clock.
/// Returns (gpu_us, wall_us).
fn time_op_gpu<F>(queue: &metal::CommandQueue, f: F) -> (f64, f64)
where
    F: FnOnce(&metal::CommandBufferRef),
{
    let _pool = ScopedPool::new();
    let cb = queue.new_command_buffer_with_unretained_references();
    let wall_start = Instant::now();
    f(cb);
    cb.commit();
    cb.wait_until_completed();
    let wall_us = wall_start.elapsed().as_secs_f64() * 1_000_000.0;
    assert_cb_ok(cb, "time_op_gpu");
    let gpu_us = gpu_time_us(cb);
    (gpu_us, wall_us)
}

// ---------------------------------------------------------------------------
// Op profiling result (GPU + Wall)
// ---------------------------------------------------------------------------

struct OpResult {
    name: &'static str,
    gpu_mean_us: f64,
    gpu_p50_us: f64,
    gpu_min_us: f64,
    wall_mean_us: f64,
}

fn compute_stats(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    let mean = sum / sorted.len() as f64;
    let min = sorted[0];
    let p50 = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    (mean, p50, min)
}

fn bench_op_gpu<F>(name: &'static str, queue: &metal::CommandQueue, mut f: F) -> OpResult
where
    F: FnMut(&metal::CommandBufferRef),
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        time_op_gpu(queue, &mut f);
    }
    // Bench
    let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
    let mut wall_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let (gpu_us, wall_us) = time_op_gpu(queue, &mut f);
        gpu_times.push(gpu_us);
        wall_times.push(wall_us);
    }
    let (gpu_mean, gpu_p50, gpu_min) = compute_stats(&gpu_times);
    let (wall_mean, _, _) = compute_stats(&wall_times);
    OpResult {
        name,
        gpu_mean_us: gpu_mean,
        gpu_p50_us: gpu_p50,
        gpu_min_us: gpu_min,
        wall_mean_us: wall_mean,
    }
}

// ---------------------------------------------------------------------------
// SDPA variant name for display
// ---------------------------------------------------------------------------

fn sdpa_variant_name(supports_nax: bool, total_seq: usize) -> &'static str {
    if supports_nax {
        "NAX (is_causal)"
    } else if total_seq >= 256 {
        "MMA BK=32 (is_causal)"
    } else {
        "MMA BK=16 (is_causal)"
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
    let supports_nax = registry.device().tuning().supports_nax;

    let setup_queue = device.new_command_queue();

    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!("dtype: float16, supports_nax: {}", supports_nax);
    println!(
        "Warmup: {} iters, Bench: {} iters per op",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!("Timing: GPU timestamps (GPUStartTime/GPUEndTime) + wall-clock");

    // ---- Build TransformerBlock for full pipeline baseline ----
    let mut block = build_transformer_block(device);
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch");
    block
        .prepare_weights_for_graph(&registry, &setup_queue)
        .expect("prepare_weights_for_graph");

    // ---- Build raw weight matrices for per-op profiling ----
    let w_q_bytes = rand_f16_bytes(Q_DIM * HIDDEN_SIZE, 1);
    let w_k_bytes = rand_f16_bytes(K_DIM * HIDDEN_SIZE, 2);
    let w_v_bytes = rand_f16_bytes(V_DIM * HIDDEN_SIZE, 3);
    let w_o_bytes = rand_f16_bytes(HIDDEN_SIZE * HIDDEN_SIZE, 4);
    let w_gate_bytes = rand_f16_bytes(INTERMEDIATE_DIM * HIDDEN_SIZE, 5);
    let w_up_bytes = rand_f16_bytes(INTERMEDIATE_DIM * HIDDEN_SIZE, 6);
    let w_down_bytes = rand_f16_bytes(HIDDEN_SIZE * INTERMEDIATE_DIM, 7);

    // Merged transposed QKV weight: [HIDDEN, TOTAL_QKV]
    let qkv_wt = build_merged_weight_t(
        device,
        &[
            (&w_q_bytes, Q_DIM, HIDDEN_SIZE),
            (&w_k_bytes, K_DIM, HIDDEN_SIZE),
            (&w_v_bytes, V_DIM, HIDDEN_SIZE),
        ],
    );

    // Merged transposed gate+up weight: [HIDDEN, 2*INTERMEDIATE]
    let gate_up_wt = build_merged_weight_t(
        device,
        &[
            (&w_gate_bytes, INTERMEDIATE_DIM, HIDDEN_SIZE),
            (&w_up_bytes, INTERMEDIATE_DIM, HIDDEN_SIZE),
        ],
    );

    // O projection transposed: [HIDDEN, HIDDEN]
    let w_o_t = transpose_weight_cpu(device, &w_o_bytes, HIDDEN_SIZE, HIDDEN_SIZE);

    // Down projection transposed: [INTERMEDIATE, HIDDEN]
    let w_down_t = transpose_weight_cpu(device, &w_down_bytes, HIDDEN_SIZE, INTERMEDIATE_DIM);

    // Norm weights (ones)
    let norm1_w = ones_f16(device, HIDDEN_SIZE);
    let norm2_w = ones_f16(device, HIDDEN_SIZE);

    // Precompute RoPE cos/sin tables
    let (cos_vec, sin_vec) = ops::rope::precompute_freqs(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA, 1.0)
        .expect("precompute_freqs");
    let cos_full = Array::from_slice(device, &cos_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);
    let sin_full = Array::from_slice(device, &sin_vec, vec![MAX_SEQ_LEN, HEAD_DIM / 2]);

    // Let Metal driver drain setup work
    std::thread::sleep(std::time::Duration::from_millis(50));

    for &seq_len in SEQ_LENS {
        println!("\n{}", "=".repeat(110));
        println!(
            "seq_len={}  |  SDPA variant: {}",
            seq_len,
            sdpa_variant_name(supports_nax, seq_len)
        );
        println!("{}", "=".repeat(110));

        let queue = device.new_command_queue();

        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice");
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

        // --- Full pipeline baseline: single CB (GPU timestamp) ---
        let baseline = bench_op_gpu("TOTAL (single-CB)", &queue, |cb| {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let _ = block
                .forward_prefill_single_cb(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    None,
                    &mut cache,
                    &registry,
                    cb,
                )
                .expect("full pipeline");
        });

        // --- Full pipeline: single encoder (GPU timestamp) ---
        // All ops share ONE compute command encoder, eliminating per-op
        // encoder create/destroy overhead (~17 encoders → 1 per layer).
        let single_encoder = bench_op_gpu("TOTAL (single-encoder)", &queue, |cb| {
            let mut cache = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            let encoder = cb.new_compute_command_encoder();
            let _ = block
                .forward_prefill_single_encoder(
                    &input,
                    Some(&cos_freqs),
                    Some(&sin_freqs),
                    None,
                    &mut cache,
                    &registry,
                    encoder,
                )
                .expect("single-encoder pipeline");
            encoder.end_encoding();
        });

        // --- Materialize intermediates for downstream ops ---

        // 1. RMSNorm -> normed
        let normed = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::rms_norm::rms_norm_into_cb(
                &registry,
                &input,
                Some(&norm1_w),
                RMS_NORM_EPS,
                cb,
            )
            .expect("rms_norm");
            cb.commit();
            cb.wait_until_completed();
            r
        };
        let normed_2d = normed.reshape(vec![seq_len, HIDDEN_SIZE]).expect("reshape");

        // 2. QKV GEMM -> qkv
        let qkv = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::matmul::matmul_into_cb(&registry, &normed_2d, &qkv_wt, cb)
                .expect("qkv matmul");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 3. Q/K/V views from merged QKV
        let elem_size = qkv.dtype().size_of();
        let q_view = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, Q_DIM],
            vec![TOTAL_QKV, 1],
            qkv.dtype(),
            qkv.offset(),
        );
        let k_view = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, K_DIM],
            vec![TOTAL_QKV, 1],
            qkv.dtype(),
            qkv.offset() + Q_DIM * elem_size,
        );
        let v_view = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, V_DIM],
            vec![TOTAL_QKV, 1],
            qkv.dtype(),
            qkv.offset() + (Q_DIM + K_DIM) * elem_size,
        );

        // 4. RoPE + Deinterleave
        let (q_batched, k_batched, v_batched) = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let qb = ops::rope::rope_multihead_into_cb(
                &registry,
                &q_view,
                &cos_freqs,
                &sin_freqs,
                NUM_HEADS,
                0,
                q_view.strides()[0],
                cb,
            )
            .expect("rope q");
            let kb = ops::rope::rope_multihead_into_cb(
                &registry,
                &k_view,
                &cos_freqs,
                &sin_freqs,
                NUM_KV_HEADS,
                0,
                k_view.strides()[0],
                cb,
            )
            .expect("rope k");
            let vb = ops::rope::deinterleave_heads_into_cb(
                &registry,
                &v_view,
                NUM_KV_HEADS,
                v_view.strides()[0],
                cb,
            )
            .expect("deinterleave v");
            cb.commit();
            cb.wait_until_completed();
            (qb, kb, vb)
        };

        // 5. SDPA
        let total_seq = seq_len;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let is_f16_d128 = true;
        let use_nax = is_f16_d128 && supports_nax;
        let use_mma_bk32 = is_f16_d128 && !use_nax && total_seq >= 256;
        let seq_major_output = is_f16_d128;

        let attn_slab = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = if use_nax {
                ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa nax")
            } else if use_mma_bk32 {
                ops::sdpa::sdpa_prefill_mma_bk32_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa mma bk32")
            } else {
                ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa mma bk16")
            };
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 6. Head concat
        let attn_concat = if seq_major_output {
            attn_slab.view(
                vec![seq_len, HIDDEN_SIZE],
                vec![HIDDEN_SIZE, 1],
                attn_slab.offset(),
            )
        } else {
            let packed = attn_slab.view(
                vec![NUM_HEADS * seq_len, HEAD_DIM],
                vec![HEAD_DIM, 1],
                attn_slab.offset(),
            );
            let interleaved = {
                let _pool = ScopedPool::new();
                let cb = queue.new_command_buffer_with_unretained_references();
                let r =
                    ops::rope::interleave_heads_into_cb(&registry, &packed, NUM_HEADS, seq_len, cb)
                        .expect("interleave_heads");
                cb.commit();
                cb.wait_until_completed();
                r
            };
            interleaved
        };

        // 7. O projection
        let o_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r =
                ops::matmul::matmul_into_cb(&registry, &attn_concat, &w_o_t, cb).expect("o_proj");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 8. Fused residual + RMSNorm
        let (normed2, h) = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::rms_norm::rms_norm_residual_add_into_cb(
                &registry,
                &o_out,
                &input,
                &norm2_w,
                RMS_NORM_EPS,
                cb,
            )
            .expect("fused residual+norm");
            cb.commit();
            cb.wait_until_completed();
            r
        };
        let normed2_2d = normed2
            .reshape(vec![seq_len, HIDDEN_SIZE])
            .expect("reshape");

        // 9. Gate+Up GEMM
        let gate_up_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::matmul::matmul_into_cb(&registry, &normed2_2d, &gate_up_wt, cb)
                .expect("gate_up");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 10. SiLU*mul (strided)
        let hidden_act = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::fused::fused_silu_mul_strided_into_cb(
                &registry,
                &gate_up_out,
                INTERMEDIATE_DIM,
                cb,
            )
            .expect("silu_mul");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 11. Down proj
        let ffn_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer_with_unretained_references();
            let r = ops::matmul::matmul_into_cb(&registry, &hidden_act, &w_down_t, cb)
                .expect("down_proj");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // ====================================================================
        // Benchmark each op in isolation (GPU timestamps)
        // ====================================================================
        let mut results: Vec<OpResult> = Vec::new();

        // 1. RMSNorm (pre-attention)
        results.push(bench_op_gpu("RMSNorm (pre-attn)", &queue, |cb| {
            let _ = ops::rms_norm::rms_norm_into_cb(
                &registry,
                &input,
                Some(&norm1_w),
                RMS_NORM_EPS,
                cb,
            )
            .expect("rms_norm");
        }));

        // 2. Merged QKV GEMM (4096 -> 6144)
        results.push(bench_op_gpu("QKV GEMM (3584->4608)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &normed_2d, &qkv_wt, cb).expect("qkv");
        }));

        // 3. RoPE Q + RoPE K + Deinterleave V
        results.push(bench_op_gpu("RoPE Q+K + Deint V", &queue, |cb| {
            let _ = ops::rope::rope_multihead_into_cb(
                &registry,
                &q_view,
                &cos_freqs,
                &sin_freqs,
                NUM_HEADS,
                0,
                q_view.strides()[0],
                cb,
            )
            .expect("rope q");
            let _ = ops::rope::rope_multihead_into_cb(
                &registry,
                &k_view,
                &cos_freqs,
                &sin_freqs,
                NUM_KV_HEADS,
                0,
                k_view.strides()[0],
                cb,
            )
            .expect("rope k");
            let _ = ops::rope::deinterleave_heads_into_cb(
                &registry,
                &v_view,
                NUM_KV_HEADS,
                v_view.strides()[0],
                cb,
            )
            .expect("deinterleave v");
        }));

        // 4. SDPA (matching production dispatch)
        let sdpa_name: &'static str = if use_nax {
            "SDPA NAX"
        } else if use_mma_bk32 {
            "SDPA MMA BK=32"
        } else {
            "SDPA MMA BK=16"
        };
        results.push(bench_op_gpu(sdpa_name, &queue, |cb| {
            if use_nax {
                let _ = ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa nax");
            } else if use_mma_bk32 {
                let _ = ops::sdpa::sdpa_prefill_mma_bk32_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa mma bk32");
            } else {
                let _ = ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                    &registry,
                    &q_batched,
                    &k_batched,
                    &v_batched,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    seq_len,
                    total_seq,
                    None,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    cb,
                )
                .expect("sdpa mma bk16");
            }
        }));

        // 5. Head Interleave (only for NAX)
        if !seq_major_output {
            let packed_for_bench = attn_slab.view(
                vec![NUM_HEADS * seq_len, HEAD_DIM],
                vec![HEAD_DIM, 1],
                attn_slab.offset(),
            );
            results.push(bench_op_gpu("Head Interleave", &queue, |cb| {
                let _ = ops::rope::interleave_heads_into_cb(
                    &registry,
                    &packed_for_bench,
                    NUM_HEADS,
                    seq_len,
                    cb,
                )
                .expect("interleave");
            }));
        }

        // 6. O Projection GEMM (4096 -> 4096)
        results.push(bench_op_gpu("O Proj GEMM (3584->3584)", &queue, |cb| {
            let _ =
                ops::matmul::matmul_into_cb(&registry, &attn_concat, &w_o_t, cb).expect("o_proj");
        }));

        // 7. Fused Residual + RMSNorm (pre-FFN)
        results.push(bench_op_gpu("Res+RMSNorm (fused)", &queue, |cb| {
            let _ = ops::rms_norm::rms_norm_residual_add_into_cb(
                &registry,
                &o_out,
                &input,
                &norm2_w,
                RMS_NORM_EPS,
                cb,
            )
            .expect("fused residual+norm");
        }));

        // 8. Merged Gate+Up GEMM (4096 -> 28672)
        results.push(bench_op_gpu("Gate+Up GEMM (3584->5120)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &normed2_2d, &gate_up_wt, cb)
                .expect("gate_up");
        }));

        // 9. Fused SiLU*mul (strided)
        results.push(bench_op_gpu("Fused SiLU*mul", &queue, |cb| {
            let _ = ops::fused::fused_silu_mul_strided_into_cb(
                &registry,
                &gate_up_out,
                INTERMEDIATE_DIM,
                cb,
            )
            .expect("silu_mul");
        }));

        // 10. Down Projection GEMM (2560 -> 3584)
        println!(
            "\n  [DIAG] hidden_act: shape={:?} strides={:?} offset={} buf_len={}",
            hidden_act.shape(),
            hidden_act.strides(),
            hidden_act.offset(),
            hidden_act.metal_buffer().length()
        );
        println!(
            "  [DIAG] w_down_t:   shape={:?} strides={:?} offset={} buf_len={}",
            w_down_t.shape(),
            w_down_t.strides(),
            w_down_t.offset(),
            w_down_t.metal_buffer().length()
        );
        results.push(bench_op_gpu("Down Proj GEMM (2560->3584)", &queue, |cb| {
            let _ =
                ops::matmul::matmul_into_cb(&registry, &hidden_act, &w_down_t, cb).expect("down");
        }));

        // 10b-d. Down Projection 4-way diagnostic
        {
            let fresh_a = rand_array(device, &[seq_len, INTERMEDIATE_DIM], 999);
            println!(
                "  [DIAG] fresh_a:    shape={:?} strides={:?} offset={} buf_len={}",
                fresh_a.shape(),
                fresh_a.strides(),
                fresh_a.offset(),
                fresh_a.metal_buffer().length()
            );
            results.push(bench_op_gpu("Down (fresh A, same W)", &queue, |cb| {
                let _ = ops::matmul::matmul_into_cb(&registry, &fresh_a, &w_down_t, cb)
                    .expect("down fresh_a");
            }));
            let fresh_b = rand_array(device, &[INTERMEDIATE_DIM, HIDDEN_SIZE], 998);
            results.push(bench_op_gpu("Down (same A, fresh W)", &queue, |cb| {
                let _ = ops::matmul::matmul_into_cb(&registry, &hidden_act, &fresh_b, cb)
                    .expect("down fresh_b");
            }));
            results.push(bench_op_gpu("Down (both fresh)", &queue, |cb| {
                let _ = ops::matmul::matmul_into_cb(&registry, &fresh_a, &fresh_b, cb)
                    .expect("down both_fresh");
            }));
        }

        // 11. Residual Add (final: h + ffn_out)
        results.push(bench_op_gpu("Residual Add (final)", &queue, |cb| {
            let _ = ops::binary::add_into_cb(&registry, &h, &ffn_out, cb).expect("add final");
        }));

        // ====================================================================
        // Print results table
        // ====================================================================
        let sum_gpu: f64 = results.iter().map(|r| r.gpu_mean_us).sum();
        let sum_wall: f64 = results.iter().map(|r| r.wall_mean_us).sum();

        println!(
            "\n{:<32} {:>12} {:>10} {:>10} {:>12} {:>8}",
            "Operation", "GPU Mean(us)", "GPU P50", "GPU Min", "Wall Mean", "CB OH(%)"
        );
        println!("{}", "-".repeat(88));

        for r in &results {
            let cb_oh = if r.gpu_mean_us > 0.0 {
                (r.wall_mean_us - r.gpu_mean_us) / r.gpu_mean_us * 100.0
            } else {
                0.0
            };
            println!(
                "{:<32} {:>12.1} {:>10.1} {:>10.1} {:>12.1} {:>7.1}%",
                r.name, r.gpu_mean_us, r.gpu_p50_us, r.gpu_min_us, r.wall_mean_us, cb_oh
            );
        }

        println!("{}", "-".repeat(88));

        // GPU times summary
        println!("{:<32} {:>12.1}", "Sum of GPU times", sum_gpu);
        println!(
            "{:<32} {:>12.1}",
            "Single-CB (GPU time)", baseline.gpu_mean_us
        );
        println!(
            "{:<32} {:>12.1}",
            "Single-encoder (GPU time)", single_encoder.gpu_mean_us
        );
        if baseline.gpu_mean_us > 0.0 {
            let gpu_overlap = (sum_gpu - baseline.gpu_mean_us) / baseline.gpu_mean_us * 100.0;
            println!("{:<32} {:>12.1}%", "GPU pipelining overlap", gpu_overlap);
        }
        if baseline.gpu_mean_us > 0.0 && single_encoder.gpu_mean_us > 0.0 {
            let encoder_savings =
                (baseline.gpu_mean_us - single_encoder.gpu_mean_us) / baseline.gpu_mean_us * 100.0;
            println!(
                "{:<32} {:>12.1}%",
                "Encoder reuse GPU savings", encoder_savings
            );
        }

        println!();

        // Wall times summary
        println!("{:<32} {:>12.1}", "Sum of Wall times", sum_wall);
        println!(
            "{:<32} {:>12.1}  (p50={:.1}, min={:.1})",
            "Single-CB (Wall time)",
            baseline.wall_mean_us,
            baseline.gpu_p50_us,
            baseline.gpu_min_us
        );
        println!(
            "{:<32} {:>12.1}  (p50={:.1}, min={:.1})",
            "Single-encoder (Wall time)",
            single_encoder.wall_mean_us,
            single_encoder.gpu_p50_us,
            single_encoder.gpu_min_us
        );
        if baseline.wall_mean_us > 0.0 {
            let cb_overhead = (sum_wall - baseline.wall_mean_us) / baseline.wall_mean_us * 100.0;
            println!(
                "{:<32} {:>12.1}%",
                "CB overhead (wall sum vs pipe)", cb_overhead
            );
        }
        if baseline.wall_mean_us > 0.0 && single_encoder.wall_mean_us > 0.0 {
            let wall_savings = (baseline.wall_mean_us - single_encoder.wall_mean_us)
                / baseline.wall_mean_us
                * 100.0;
            println!(
                "{:<32} {:>12.1}%",
                "Encoder reuse wall savings", wall_savings
            );
        }

        // Let GPU drain before next seq_len
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
