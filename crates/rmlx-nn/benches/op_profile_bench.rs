//! Per-operation profiling benchmark for RMLX single-layer prefill pipeline.
//!
//! Breaks down the Llama-3 8B TransformerBlock `forward_prefill_single_cb`
//! into individual operations, each measured with its own CommandBuffer +
//! commit + wait_until_completed() for accurate per-op timing.
//!
//! Uses low-level `ops::*_into_cb` functions directly with raw weight
//! matrices, avoiding private struct field access.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench op_profile_bench

use std::time::{Duration, Instant};

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

const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 4096
const K_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
const V_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
const TOTAL_QKV: usize = Q_DIM + K_DIM + V_DIM; // 6144

const SEQ_LENS: &[usize] = &[256, 512, 1024];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

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

/// Build a transposed merged weight matrix on CPU.
///
/// Given weight matrices W1=[out1, in], W2=[out2, in], ...,
/// produces the transpose of the row-concatenation:
///   merged = [W1; W2; ...] has shape [sum(out_i), in]
///   merged^T has shape [in, sum(out_i)]
///
/// We directly produce the transposed layout.
fn build_merged_weight_t(
    device: &metal::Device,
    weights_row_major: &[(&[u8], usize, usize)], // (bytes, rows=out, cols=in)
) -> Array {
    let in_dim = weights_row_major[0].2;
    let total_out: usize = weights_row_major.iter().map(|w| w.1).sum();

    // Result: [in_dim, total_out] in row-major (i.e., transposed merged)
    let mut result_bytes = vec![0u8; in_dim * total_out * 2];

    let mut col_offset = 0;
    for &(bytes, rows, cols) in weights_row_major {
        assert_eq!(cols, in_dim);
        // Source: [rows, cols] row-major
        // Dest:   [cols, total_out] row-major, writing at column col_offset..col_offset+rows
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

/// Build a single transposed weight: [out, in] -> [in, out]
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
// Causal mask builder
// ---------------------------------------------------------------------------

fn build_causal_mask(device: &metal::Device, seq_len: usize) -> Array {
    let neg_inf_f16: u16 = 0xFC00;
    let mut mask_f16 = vec![0u16; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_f16[i * seq_len + j] = neg_inf_f16;
        }
    }
    let bytes: Vec<u8> = mask_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    Array::from_bytes(device, &bytes, vec![seq_len, seq_len], DType::Float16)
}

fn assert_cb_ok(cb: &metal::CommandBufferRef, context: &str) {
    let status = cb.status();
    assert!(
        status != metal::MTLCommandBufferStatus::Error,
        "GPU command buffer error in {context}: status={status:?}"
    );
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

fn time_op<F>(queue: &metal::CommandQueue, f: F) -> Duration
where
    F: FnOnce(&metal::CommandBufferRef),
{
    let _pool = ScopedPool::new();
    let cb = queue.new_command_buffer();
    let start = Instant::now();
    f(cb);
    cb.commit();
    cb.wait_until_completed();
    let elapsed = start.elapsed();
    assert_cb_ok(cb, "time_op");
    elapsed
}

// ---------------------------------------------------------------------------
// Op profiling result
// ---------------------------------------------------------------------------

struct OpResult {
    name: &'static str,
    mean_us: f64,
    std_us: f64,
    min_us: f64,
    max_us: f64,
}

fn bench_op<F>(name: &'static str, queue: &metal::CommandQueue, mut f: F) -> OpResult
where
    F: FnMut(&metal::CommandBufferRef),
{
    for _ in 0..WARMUP_ITERS {
        time_op(queue, &mut f);
    }
    let mut durations = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        durations.push(time_op(queue, &mut f));
    }
    let micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    let sum: f64 = micros.iter().sum();
    let mean = sum / micros.len() as f64;
    let variance: f64 =
        micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / micros.len() as f64;
    let std_dev = variance.sqrt();
    let min = micros.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = micros.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    OpResult {
        name,
        mean_us: mean,
        std_us: std_dev,
        min_us: min,
        max_us: max,
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

    let setup_queue = device.new_command_queue();

    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM
    );
    println!("dtype: float16");
    println!(
        "Warmup: {} iters, Bench: {} iters per op",
        WARMUP_ITERS, BENCH_ITERS
    );

    // ---- Build TransformerBlock for full pipeline baseline ----
    let mut block = build_transformer_block(device);
    block
        .prepare_weights_9dispatch(device)
        .expect("prepare_weights_9dispatch");
    block
        .prepare_weights_for_graph(&registry, &setup_queue)
        .expect("prepare_weights_for_graph");

    // ---- Build raw weight matrices for per-op profiling ----
    // Generate the same random bytes used by make_linear (seeds 1-7)
    let w_q_bytes = rand_f16_bytes(Q_DIM * HIDDEN_SIZE, 1);
    let w_k_bytes = rand_f16_bytes(K_DIM * HIDDEN_SIZE, 2);
    let w_v_bytes = rand_f16_bytes(V_DIM * HIDDEN_SIZE, 3);
    let w_o_bytes = rand_f16_bytes(HIDDEN_SIZE * HIDDEN_SIZE, 4);
    let w_gate_bytes = rand_f16_bytes(INTERMEDIATE_DIM * HIDDEN_SIZE, 5);
    let w_up_bytes = rand_f16_bytes(INTERMEDIATE_DIM * HIDDEN_SIZE, 6);
    let w_down_bytes = rand_f16_bytes(HIDDEN_SIZE * INTERMEDIATE_DIM, 7);

    // Build merged transposed QKV weight: [HIDDEN, TOTAL_QKV]
    let qkv_wt = build_merged_weight_t(
        device,
        &[
            (&w_q_bytes, Q_DIM, HIDDEN_SIZE),
            (&w_k_bytes, K_DIM, HIDDEN_SIZE),
            (&w_v_bytes, V_DIM, HIDDEN_SIZE),
        ],
    );

    // Build merged transposed gate+up weight: [HIDDEN, 2*INTERMEDIATE]
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

    for &seq_len in SEQ_LENS {
        println!("\n{}", "=".repeat(100));
        println!("seq_len={}", seq_len);
        println!("{}", "=".repeat(100));

        let queue = device.new_command_queue();

        let cos_freqs = cos_full.slice(0, 0, seq_len).expect("cos slice");
        let sin_freqs = sin_full.slice(0, 0, seq_len).expect("sin slice");
        let mask = build_causal_mask(device, seq_len);
        let input = rand_array(device, &[seq_len, HIDDEN_SIZE], 42);

        // --- Full pipeline baseline ---
        let baseline = bench_op("TOTAL (full pipeline)", &queue, |cb| {
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
                    Some(&mask),
                    &mut cache,
                    &registry,
                    cb,
                )
                .expect("full pipeline");
        });

        // --- Materialize intermediates for downstream ops ---

        // 1. RMSNorm -> normed
        let normed = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
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
            let cb = queue.new_command_buffer();
            let r = ops::matmul::matmul_into_cb(&registry, &normed_2d, &qkv_wt, cb)
                .expect("qkv matmul");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 3. Q/K/V views + copy
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

        // 4. RoPE + Deinterleave (strided — no copy needed)
        let (q_batched, k_batched, v_batched) = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
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

        // 5. KV Cache
        let mut cache_for_sdpa =
            LayerKvCache::preallocated(device, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DType::Float16);
        {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            cache_for_sdpa
                .append_batched_into_cb(&k_batched, &v_batched, seq_len, &registry, cb)
                .expect("kv cache append");
            cb.commit();
            cb.wait_until_completed();
        }
        let total_seq = cache_for_sdpa.seq_len;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let k_slab = cache_for_sdpa.keys_slab_view().expect("k slab");
        let v_slab = cache_for_sdpa.values_slab_view().expect("v slab");
        let kv_stride = if cache_for_sdpa.max_seq_len() != total_seq {
            Some(cache_for_sdpa.max_seq_len())
        } else {
            None
        };

        // 6. SDPA
        let attn_slab = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r = ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                &registry,
                &q_batched,
                &k_slab,
                &v_slab,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                seq_len,
                total_seq,
                kv_stride,
                Some(&mask),
                scale,
                false,
                cb,
            )
            .expect("sdpa");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 7. SDPA output is already seq-major [seq_len, num_heads * head_dim]
        let attn_out = attn_slab
            .reshape(vec![seq_len, HIDDEN_SIZE])
            .expect("reshape attn");

        // 8. O proj
        let o_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r = ops::matmul::matmul_into_cb(&registry, &attn_out, &w_o_t, cb).expect("o_proj");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 9. Residual add
        let h = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r = ops::binary::add_into_cb(&registry, &input, &o_out, cb).expect("add");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 10. RMSNorm (pre-FFN)
        let normed2 = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r =
                ops::rms_norm::rms_norm_into_cb(&registry, &h, Some(&norm2_w), RMS_NORM_EPS, cb)
                    .expect("rms_norm 2");
            cb.commit();
            cb.wait_until_completed();
            r
        };
        let normed2_2d = normed2
            .reshape(vec![seq_len, HIDDEN_SIZE])
            .expect("reshape");

        // 11. Gate+Up GEMM
        let gate_up_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r = ops::matmul::matmul_into_cb(&registry, &normed2_2d, &gate_up_wt, cb)
                .expect("gate_up");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // 12. SiLU*mul (strided — no copy needed)
        let hidden_act = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
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

        // 13. Down proj
        let ffn_out = {
            let _pool = ScopedPool::new();
            let cb = queue.new_command_buffer();
            let r = ops::matmul::matmul_into_cb(&registry, &hidden_act, &w_down_t, cb)
                .expect("down_proj");
            cb.commit();
            cb.wait_until_completed();
            r
        };

        // ====================================================================
        // Benchmark each op in isolation
        // ====================================================================
        let mut results: Vec<OpResult> = Vec::new();

        // 1. RMSNorm (pre-attention)
        results.push(bench_op("RMSNorm (pre-attn)", &queue, |cb| {
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
        results.push(bench_op("QKV GEMM (4096->6144)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &normed_2d, &qkv_wt, cb).expect("qkv");
        }));

        // 3. RoPE Q + RoPE K + Deinterleave V (strided input)
        results.push(bench_op("RoPE Q+K + Deinterleave V", &queue, |cb| {
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

        // 5. KV Cache Append
        results.push(bench_op("KV Cache Append", &queue, |cb| {
            let mut cache_tmp = LayerKvCache::preallocated(
                device,
                NUM_KV_HEADS,
                HEAD_DIM,
                MAX_SEQ_LEN,
                DType::Float16,
            );
            cache_tmp
                .append_batched_into_cb(&k_batched, &v_batched, seq_len, &registry, cb)
                .expect("cache");
        }));

        // 6. SDPA MMA
        results.push(bench_op("SDPA MMA", &queue, |cb| {
            let _ = ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                &registry,
                &q_batched,
                &k_slab,
                &v_slab,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                seq_len,
                total_seq,
                kv_stride,
                Some(&mask),
                scale,
                false,
                cb,
            )
            .expect("sdpa");
        }));

        // 7. O Projection GEMM (4096 -> 4096)
        results.push(bench_op("O Proj GEMM (4096->4096)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &attn_out, &w_o_t, cb).expect("o_proj");
        }));

        // 8. Residual Add (post-attention)
        results.push(bench_op("Residual Add (post-attn)", &queue, |cb| {
            let _ = ops::binary::add_into_cb(&registry, &input, &o_out, cb).expect("add");
        }));

        // 9. RMSNorm (pre-FFN)
        results.push(bench_op("RMSNorm (pre-FFN)", &queue, |cb| {
            let _ =
                ops::rms_norm::rms_norm_into_cb(&registry, &h, Some(&norm2_w), RMS_NORM_EPS, cb)
                    .expect("rms_norm 2");
        }));

        // 10. Merged Gate+Up GEMM (4096 -> 28672)
        results.push(bench_op("Gate+Up GEMM (4096->28672)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &normed2_2d, &gate_up_wt, cb)
                .expect("gate_up");
        }));

        // 11. Fused SiLU*mul (strided)
        results.push(bench_op("Fused SiLU*mul (strided)", &queue, |cb| {
            let _ = ops::fused::fused_silu_mul_strided_into_cb(
                &registry,
                &gate_up_out,
                INTERMEDIATE_DIM,
                cb,
            )
            .expect("silu_mul");
        }));

        // 12. Down Projection GEMM (14336 -> 4096)
        results.push(bench_op("Down Proj GEMM (14336->4096)", &queue, |cb| {
            let _ = ops::matmul::matmul_into_cb(&registry, &hidden_act, &w_down_t, cb)
                .expect("down_proj");
        }));

        // 13. Residual Add (final)
        results.push(bench_op("Residual Add (final)", &queue, |cb| {
            let _ = ops::binary::add_into_cb(&registry, &h, &ffn_out, cb).expect("add final");
        }));

        // --- Print table ---
        let sum_ops: f64 = results.iter().map(|r| r.mean_us).sum();

        println!(
            "\n{:<32} {:>10} {:>8} {:>10} {:>10} {:>8}",
            "Operation", "Mean (us)", "Std", "Min (us)", "Max (us)", "% Total"
        );
        println!("{}", "-".repeat(82));

        for r in &results {
            let pct = r.mean_us / sum_ops * 100.0;
            println!(
                "{:<32} {:>10.1} {:>8.1} {:>10.1} {:>10.1} {:>7.1}%",
                r.name, r.mean_us, r.std_us, r.min_us, r.max_us, pct
            );
        }

        println!("{}", "-".repeat(82));
        println!("{:<32} {:>10.1}", "Sum of individual ops", sum_ops);
        println!(
            "{:<32} {:>10.1}",
            "Full pipeline (single CB)", baseline.mean_us
        );
        let overhead = (sum_ops - baseline.mean_us) / baseline.mean_us * 100.0;
        println!("{:<32} {:>10.1}%", "CB overhead (sum - pipeline)", overhead);

        // Let GPU drain before next seq_len
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
