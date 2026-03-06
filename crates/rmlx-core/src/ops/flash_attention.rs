//! FlashAttention-2 Metal kernel for efficient long-sequence attention.
//!
//! Implements the FlashAttention-2 algorithm via a JIT-compiled Metal compute shader.
//! Uses tiling with online softmax to achieve O(N) memory instead of O(N^2) for the
//! attention score matrix.
//!
//! **v1 scope:**
//! - f32 only
//! - head_dim = 128 only
//! - Causal mask support (upper triangular)
//! - No dropout, alibi, or sliding window
//! - Falls back to naive SDPA for unsupported configurations

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Tiling constants — must match the Metal shader.
// ---------------------------------------------------------------------------

/// Query block size (rows of Q processed per threadgroup).
const BR: usize = 32;

/// Key/Value block size (columns of K/V iterated per inner loop step).
const _BC: usize = 32;

/// Head dimension supported by v1 kernel.
const HEAD_DIM: usize = 128;

/// Threads per threadgroup for the flash attention kernel.
const THREADS_PER_TG: u64 = 128;

/// Minimum sequence length to prefer flash attention over naive SDPA.
pub const FLASH_ATTN_SEQ_THRESHOLD: usize = 128;

// ---------------------------------------------------------------------------
// Metal shader source — FlashAttention-2, f32, head_dim=128
// ---------------------------------------------------------------------------

/// Metal shader implementing FlashAttention-2 for f32 with head_dim=128.
///
/// Algorithm:
/// 1. Each threadgroup owns a block of BR rows of Q.
/// 2. For each Q block, iterate over K/V in blocks of BC columns.
/// 3. Compute S = Q_block @ K_block^T * scale in threadgroup memory.
/// 4. Apply causal mask (if enabled) and online softmax (running max + sum).
/// 5. Accumulate O = softmax(S) @ V_block with rescaling for new max.
/// 6. Final normalization by the softmax denominator.
///
/// Buffers:
///   0: Q      [N, 128]
///   1: K      [S, 128]
///   2: V      [S, 128]
///   3: O      [N, 128]
///   4: params [4 x uint32]: { N, S, is_causal, _pad }
///   5: scale  [float]
///
/// Grid: (ceil(N / BR), 1, 1) threadgroups, 128 threads each.
pub const FLASH_ATTN_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BR = 32;    // Q block rows
constant constexpr uint BC = 32;    // K/V block columns
constant constexpr uint HD = 128;   // head_dim
constant constexpr uint N_THREADS = 128;

// Each thread "owns" HD / N_THREADS = 1 element per row for the O accumulator,
// but we process tiles cooperatively. We use shared memory for Q tile, S tile,
// and per-row softmax stats, with O accumulator and V reads from global memory.
//
// Shared memory budget (floats):
//   Q_tile:  BR * HD  = 32 * 128 = 4096  -> 16384 bytes
//   S_tile:  BR * BC  = 32 * 32  = 1024  ->  4096 bytes
//   O_acc:   BR * HD  = 32 * 128 = 4096  -> 16384 bytes
//   Total so far: 36864 bytes > 32KB limit
//
// To fit in 32KB, we keep O_acc in global memory (per-threadgroup scratch)
// or reduce tile sizes. Actually on Apple Silicon the threadgroup memory
// limit is 32KB for most chips.
//
// Strategy: Keep Q_tile and S_tile in shared memory. O_acc in shared memory
// but we reduce BR or split. Actually let's just use BR=32, BC=32, HD=128
// with Q_tile in shared memory and O_acc in shared memory, total = 32KB + overhead.
//
// Better approach: Don't store full Q_tile. Each thread computes its assigned
// score elements by reading Q from global memory. Only S_tile and O_acc need
// shared memory.
//
// S_tile: BR * BC = 1024 floats = 4096 bytes
// O_acc:  BR * HD = 4096 floats = 16384 bytes
// m_row:  BR = 32 floats = 128 bytes
// l_row:  BR = 32 floats = 128 bytes
// Total: ~20736 bytes < 32KB. Good.

kernel void flash_attn_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device       float* O         [[buffer(3)]],
    constant     uint*  params    [[buffer(4)]],
    constant     float& scale     [[buffer(5)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]])
{
    const uint N         = params[0];
    const uint S         = params[1];
    const uint is_causal = params[2];

    // This threadgroup handles Q rows [q_start .. q_start + BR)
    const uint q_start = tg_id * BR;
    if (q_start >= N) return;
    const uint q_end   = min(q_start + BR, N);
    const uint q_count = q_end - q_start;

    // Shared memory
    threadgroup float S_tile[BR * BC];     // Score block [BR, BC]
    threadgroup float O_acc[BR * HD];      // Output accumulator [BR, HD]
    threadgroup float m_row[BR];           // Running max per row
    threadgroup float l_row[BR];           // Running sum(exp) per row

    // Initialize O_acc to zero, m_row to -inf, l_row to 0
    for (uint idx = tid; idx < BR * HD; idx += N_THREADS) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < BR; idx += N_THREADS) {
        m_row[idx] = -INFINITY;
        l_row[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Number of K/V blocks
    const uint n_kv_blocks = (S + BC - 1) / BC;

    // For causal masking: last relevant KV block
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / BC) + 1)
        : n_kv_blocks;

    // Iterate over K/V blocks
    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * BC;
        const uint kv_end   = min(kv_start + BC, S);
        const uint kv_count = kv_end - kv_start;

        // --- Compute S_tile[q_count, kv_count] = Q @ K^T * scale ---
        // Each thread computes multiple (i, j) entries
        for (uint idx = tid; idx < q_count * kv_count; idx += N_THREADS) {
            uint i = idx / kv_count;  // row in Q block
            uint j = idx % kv_count;  // col in KV block

            float dot = 0.0f;
            // Dot product over head_dim=128
            for (uint d = 0; d < HD; d++) {
                dot += Q[(q_start + i) * HD + d] * K[(kv_start + j) * HD + d];
            }
            dot *= scale;

            // Apply causal mask
            if (is_causal) {
                uint q_idx  = q_start + i;
                uint kv_idx = kv_start + j;
                if (kv_idx > q_idx) {
                    dot = -INFINITY;
                }
            }

            S_tile[i * BC + j] = dot;
        }

        // Fill out-of-bounds entries with -inf
        for (uint idx = tid; idx < q_count * BC; idx += N_THREADS) {
            uint j = idx % BC;
            if (j >= kv_count) {
                uint i = idx / BC;
                S_tile[i * BC + j] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Online softmax: compute new max, rescale, exponentiate ---
        // Each thread handles a subset of rows
        for (uint i = tid; i < q_count; i += N_THREADS) {
            // Find row max of S_tile[i, :]
            float row_max = -INFINITY;
            for (uint j = 0; j < BC; j++) {
                row_max = max(row_max, S_tile[i * BC + j]);
            }

            float m_old = m_row[i];
            float m_new = max(m_old, row_max);

            // Correction factor for previous accumulator
            float correction = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;

            // Exponentiate scores with new max
            float block_sum = 0.0f;
            for (uint j = 0; j < BC; j++) {
                float val = (S_tile[i * BC + j] > -INFINITY)
                    ? exp(S_tile[i * BC + j] - m_new)
                    : 0.0f;
                S_tile[i * BC + j] = val;
                block_sum += val;
            }

            // Update running stats
            float l_new = l_row[i] * correction + block_sum;

            // Rescale existing O_acc rows
            for (uint d = 0; d < HD; d++) {
                O_acc[i * HD + d] *= correction;
            }

            // Accumulate: O_acc[i, :] += S_tile[i, :] @ V[kv_block, :]
            for (uint j = 0; j < kv_count; j++) {
                float s_val = S_tile[i * BC + j];
                if (s_val > 0.0f) {
                    for (uint d = 0; d < HD; d++) {
                        O_acc[i * HD + d] += s_val * V[(kv_start + j) * HD + d];
                    }
                }
            }

            m_row[i] = m_new;
            l_row[i] = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Final normalization: O = O_acc / l_row ---
    for (uint idx = tid; idx < q_count * HD; idx += N_THREADS) {
        uint i = idx / HD;
        float l = l_row[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + (idx / HD)) * HD + (idx % HD)] = O_acc[idx] * inv_l;
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register FlashAttention-2 Metal kernels with the given registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("flash_attention", FLASH_ATTN_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether flash attention can handle the given configuration.
///
/// Returns `true` if head_dim == 128, dtype is f32, and seq_len >= threshold.
pub fn supports_flash_attention(head_dim: usize, dtype: DType, seq_len: usize) -> bool {
    head_dim == HEAD_DIM && dtype == DType::Float32 && seq_len >= FLASH_ATTN_SEQ_THRESHOLD
}

/// FlashAttention-2 forward pass.
///
/// Computes `softmax(Q @ K^T * scale [+ causal_mask]) @ V` using tiled online
/// softmax, avoiding materialization of the full N x S attention matrix.
///
/// # Arguments
/// - `query`: `[batch, num_heads, seq_len, 128]` or `[seq_len, 128]` (2D)
/// - `key`:   `[batch, num_kv_heads, kv_len, 128]` or `[kv_len, 128]` (2D)
/// - `value`: `[batch, num_kv_heads, kv_len, 128]` or `[kv_len, 128]` (2D)
/// - `scale`: typically `1.0 / sqrt(128)`
/// - `causal`: if true, applies upper-triangular causal mask
///
/// # Restrictions (v1)
/// - f32 only
/// - head_dim must be 128
/// - No additive mask (use causal flag instead)
///
/// # Fallback
/// Returns `Err(KernelError::InvalidShape)` for unsupported configs so the
/// caller can fall back to naive SDPA.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_forward(
    registry: &KernelRegistry,
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
    causal: bool,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate dtype ---
    if query.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "flash_attention: only f32 supported, got {:?}",
            query.dtype()
        )));
    }
    if query.dtype() != key.dtype() || query.dtype() != value.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "flash_attention: Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
            query.dtype(),
            key.dtype(),
            value.dtype()
        )));
    }

    // --- Determine layout: 2D [N, D] or 4D [B, H, N, D] ---
    let (batch_heads, n, d, kv_len) = match query.ndim() {
        2 => {
            let n = query.shape()[0];
            let d = query.shape()[1];
            let s = key.shape()[0];
            if key.ndim() != 2 || value.ndim() != 2 {
                return Err(KernelError::InvalidShape(
                    "flash_attention: Q is 2D but K/V are not 2D".into(),
                ));
            }
            (1, n, d, s)
        }
        4 => {
            let b = query.shape()[0];
            let h = query.shape()[1];
            let n = query.shape()[2];
            let d = query.shape()[3];
            if key.ndim() != 4 || value.ndim() != 4 {
                return Err(KernelError::InvalidShape(
                    "flash_attention: Q is 4D but K/V are not 4D".into(),
                ));
            }
            let kv_len = key.shape()[2];
            // For GQA: num_kv_heads may differ from num_heads
            let kv_h = key.shape()[1];
            if b != key.shape()[0] || b != value.shape()[0] {
                return Err(KernelError::InvalidShape(
                    "flash_attention: batch size mismatch".into(),
                ));
            }
            if h % kv_h != 0 {
                return Err(KernelError::InvalidShape(format!(
                    "flash_attention: num_heads ({h}) not divisible by num_kv_heads ({kv_h})"
                )));
            }
            if key.shape()[3] != d || value.shape()[3] != d {
                return Err(KernelError::InvalidShape(
                    "flash_attention: head_dim mismatch in K or V".into(),
                ));
            }
            if value.shape()[2] != kv_len {
                return Err(KernelError::InvalidShape(
                    "flash_attention: K and V seq_len mismatch".into(),
                ));
            }
            (b * h, n, d, kv_len)
        }
        _ => {
            return Err(KernelError::InvalidShape(format!(
                "flash_attention: Q must be 2D or 4D, got {}D",
                query.ndim()
            )));
        }
    };

    // --- Validate head_dim ---
    if d != HEAD_DIM {
        return Err(KernelError::InvalidShape(format!(
            "flash_attention: head_dim must be {HEAD_DIM}, got {d}"
        )));
    }

    // --- Make contiguous ---
    let q_c = super::make_contiguous(query, registry, queue)?;
    let query = q_c.as_ref().unwrap_or(query);
    let k_c = super::make_contiguous(key, registry, queue)?;
    let key = k_c.as_ref().unwrap_or(key);
    let v_c = super::make_contiguous(value, registry, queue)?;
    let value = v_c.as_ref().unwrap_or(value);

    // --- For 2D inputs, dispatch single head ---
    if batch_heads == 1 {
        return dispatch_single_head(registry, query, key, value, n, kv_len, scale, causal, queue);
    }

    // --- For 4D inputs, dispatch per batch*head ---
    let b = query.shape()[0];
    let h = query.shape()[1];
    let kv_h = key.shape()[1];
    let repeats = h / kv_h;
    let dev = registry.device().raw();

    // Output: same shape as query
    let out = Array::zeros(dev, query.shape(), DType::Float32);

    let head_q_bytes = n * HEAD_DIM * 4; // f32
    let head_kv_bytes = kv_len * HEAD_DIM * 4;
    let head_o_bytes = n * HEAD_DIM * 4;

    let pipeline = registry.get_pipeline("flash_attn_f32", DType::Float32)?;
    let n_threadgroups = n.div_ceil(BR) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());

    let params: [u32; 4] = [n as u32, kv_len as u32, if causal { 1 } else { 0 }, 0];
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        16,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = dev.new_buffer_with_data(
        &scale as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();

    for bi in 0..b {
        for hi in 0..h {
            let kv_hi = hi / repeats;
            let q_offset = (bi * h + hi) * head_q_bytes;
            let k_offset = (bi * kv_h + kv_hi) * head_kv_bytes;
            let v_offset = (bi * kv_h + kv_hi) * head_kv_bytes;
            let o_offset = (bi * h + hi) * head_o_bytes;

            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(
                0,
                Some(query.metal_buffer()),
                (query.offset() + q_offset) as u64,
            );
            encoder.set_buffer(
                1,
                Some(key.metal_buffer()),
                (key.offset() + k_offset) as u64,
            );
            encoder.set_buffer(
                2,
                Some(value.metal_buffer()),
                (value.offset() + v_offset) as u64,
            );
            encoder.set_buffer(3, Some(out.metal_buffer()), o_offset as u64);
            encoder.set_buffer(4, Some(&params_buf), 0);
            encoder.set_buffer(5, Some(&scale_buf), 0);

            encoder.dispatch_thread_groups(
                MTLSize::new(n_threadgroups, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            encoder.end_encoding();
        }
    }

    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

/// Dispatch a single 2D head through the flash attention kernel.
#[allow(clippy::too_many_arguments)]
fn dispatch_single_head(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    n: usize,
    s: usize,
    scale: f32,
    causal: bool,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let pipeline = registry.get_pipeline("flash_attn_f32", DType::Float32)?;
    let dev = registry.device().raw();

    let out = Array::zeros(dev, &[n, HEAD_DIM], DType::Float32);

    let params: [u32; 4] = [n as u32, s as u32, if causal { 1 } else { 0 }, 0];
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        16,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = dev.new_buffer_with_data(
        &scale as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(1, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(2, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(&params_buf), 0);
    encoder.set_buffer(5, Some(&scale_buf), 0);

    let n_threadgroups = n.div_ceil(BR) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());

    encoder.dispatch_thread_groups(
        MTLSize::new(n_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    fn setup() -> (KernelRegistry, metal::CommandQueue) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu_dev.raw().new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        ops::sdpa::register(&registry).unwrap();
        ops::copy::register(&registry).unwrap();
        (registry, queue)
    }

    /// Reference naive attention in f32 on CPU.
    /// Q: [N, D], K: [S, D], V: [S, D] -> O: [N, D]
    #[allow(clippy::too_many_arguments)]
    fn naive_attention_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n: usize,
        s: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            // Compute scores[j] = dot(Q[i], K[j]) * scale
            let mut scores = vec![0.0f32; s];
            for j in 0..s {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[j] = dot * scale;
            }

            // Apply causal mask
            if causal {
                for (j, score) in scores.iter_mut().enumerate() {
                    if j > i {
                        *score = f32::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exp_scores = vec![0.0f32; s];
            for (j, exp_s) in exp_scores.iter_mut().enumerate() {
                if scores[j] > f32::NEG_INFINITY {
                    *exp_s = (scores[j] - max_score).exp();
                }
                sum_exp += *exp_s;
            }
            if sum_exp > 0.0 {
                for exp_s in exp_scores.iter_mut() {
                    *exp_s /= sum_exp;
                }
            }

            // Weighted sum of V
            for dd in 0..d {
                let mut val = 0.0f32;
                for j in 0..s {
                    val += exp_scores[j] * v[j * d + dd];
                }
                out[i * d + dd] = val;
            }
        }
        out
    }

    /// Generate deterministic pseudo-random f32 data in [-1, 1].
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut data = Vec::with_capacity(len);
        let mut state = seed;
        for _ in 0..len {
            // Simple xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let val = ((state & 0xFFFF) as f32 / 32768.0) - 1.0;
            data.push(val);
        }
        data
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_flash_attn_matches_naive_seq64() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();
        let n = 64;
        let s = 64;
        let d = HEAD_DIM;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data = pseudo_random(n * d, 42);
        let k_data = pseudo_random(s * d, 137);
        let v_data = pseudo_random(s * d, 256);

        let q = Array::from_slice(dev, &q_data, vec![n, d]);
        let k = Array::from_slice(dev, &k_data, vec![s, d]);
        let v = Array::from_slice(dev, &v_data, vec![s, d]);

        let result = flash_attention_forward(&registry, &q, &k, &v, scale, false, &queue).unwrap();
        let got: Vec<f32> = result.to_vec_checked();
        let expected = naive_attention_cpu(&q_data, &k_data, &v_data, n, s, d, scale, false);

        let diff = max_abs_diff(&got, &expected);
        assert!(
            diff < 1e-3,
            "flash_attn seq=64 non-causal: max_abs_diff={diff} (expected < 1e-3)"
        );
    }

    #[test]
    fn test_flash_attn_matches_naive_seq128() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();
        let n = 128;
        let s = 128;
        let d = HEAD_DIM;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data = pseudo_random(n * d, 7);
        let k_data = pseudo_random(s * d, 13);
        let v_data = pseudo_random(s * d, 19);

        let q = Array::from_slice(dev, &q_data, vec![n, d]);
        let k = Array::from_slice(dev, &k_data, vec![s, d]);
        let v = Array::from_slice(dev, &v_data, vec![s, d]);

        let result = flash_attention_forward(&registry, &q, &k, &v, scale, false, &queue).unwrap();
        let got: Vec<f32> = result.to_vec_checked();
        let expected = naive_attention_cpu(&q_data, &k_data, &v_data, n, s, d, scale, false);

        let diff = max_abs_diff(&got, &expected);
        assert!(
            diff < 1e-3,
            "flash_attn seq=128 non-causal: max_abs_diff={diff} (expected < 1e-3)"
        );
    }

    #[test]
    fn test_flash_attn_matches_naive_seq512() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();
        let n = 512;
        let s = 512;
        let d = HEAD_DIM;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data = pseudo_random(n * d, 99);
        let k_data = pseudo_random(s * d, 101);
        let v_data = pseudo_random(s * d, 103);

        let q = Array::from_slice(dev, &q_data, vec![n, d]);
        let k = Array::from_slice(dev, &k_data, vec![s, d]);
        let v = Array::from_slice(dev, &v_data, vec![s, d]);

        let result = flash_attention_forward(&registry, &q, &k, &v, scale, false, &queue).unwrap();
        let got: Vec<f32> = result.to_vec_checked();
        let expected = naive_attention_cpu(&q_data, &k_data, &v_data, n, s, d, scale, false);

        let diff = max_abs_diff(&got, &expected);
        assert!(
            diff < 1e-2,
            "flash_attn seq=512 non-causal: max_abs_diff={diff} (expected < 1e-2)"
        );
    }

    #[test]
    fn test_flash_attn_causal_matches_naive() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();
        let n = 128;
        let s = 128;
        let d = HEAD_DIM;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data = pseudo_random(n * d, 55);
        let k_data = pseudo_random(s * d, 66);
        let v_data = pseudo_random(s * d, 77);

        let q = Array::from_slice(dev, &q_data, vec![n, d]);
        let k = Array::from_slice(dev, &k_data, vec![s, d]);
        let v = Array::from_slice(dev, &v_data, vec![s, d]);

        let result = flash_attention_forward(&registry, &q, &k, &v, scale, true, &queue).unwrap();
        let got: Vec<f32> = result.to_vec_checked();
        let expected = naive_attention_cpu(&q_data, &k_data, &v_data, n, s, d, scale, true);

        let diff = max_abs_diff(&got, &expected);
        assert!(
            diff < 1e-3,
            "flash_attn seq=128 causal: max_abs_diff={diff} (expected < 1e-3)"
        );
    }

    #[test]
    fn test_flash_attn_non_128_head_dim_errors() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[32, 64], DType::Float32);
        let k = Array::zeros(dev, &[32, 64], DType::Float32);
        let v = Array::zeros(dev, &[32, 64], DType::Float32);

        let result = flash_attention_forward(&registry, &q, &k, &v, 0.125, false, &queue);
        assert!(result.is_err(), "head_dim=64 should return error");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("head_dim must be 128"),
            "error should mention head_dim, got: {msg}"
        );
    }

    #[test]
    fn test_flash_attn_f16_unsupported() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[32, 128], DType::Float16);
        let k = Array::zeros(dev, &[32, 128], DType::Float16);
        let v = Array::zeros(dev, &[32, 128], DType::Float16);

        let result = flash_attention_forward(&registry, &q, &k, &v, 0.088, false, &queue);
        assert!(result.is_err(), "f16 should return error in v1");
    }

    #[test]
    fn test_flash_attn_supports_check() {
        assert!(supports_flash_attention(128, DType::Float32, 256));
        assert!(supports_flash_attention(128, DType::Float32, 128));
        assert!(!supports_flash_attention(64, DType::Float32, 256));
        assert!(!supports_flash_attention(128, DType::Float16, 256));
        assert!(!supports_flash_attention(128, DType::Float32, 64));
    }

    #[test]
    fn test_flash_attn_batch_4d() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();
        let b = 2;
        let h = 2;
        let n = 64;
        let s = 64;
        let d = HEAD_DIM;
        let scale = 1.0 / (d as f32).sqrt();

        // Create 4D tensors [B, H, N, D]
        let total_q = b * h * n * d;
        let total_kv = b * h * s * d;
        let q_data = pseudo_random(total_q, 1001);
        let k_data = pseudo_random(total_kv, 1002);
        let v_data = pseudo_random(total_kv, 1003);

        let q = Array::from_slice(dev, &q_data, vec![b, h, n, d]);
        let k = Array::from_slice(dev, &k_data, vec![b, h, s, d]);
        let v = Array::from_slice(dev, &v_data, vec![b, h, s, d]);

        let result = flash_attention_forward(&registry, &q, &k, &v, scale, false, &queue).unwrap();
        assert_eq!(result.shape(), &[b, h, n, d]);

        let got: Vec<f32> = result.to_vec_checked();

        // Verify each (batch, head) slice independently
        let head_q_size = n * d;
        let head_kv_size = s * d;
        let head_o_size = n * d;

        for bi in 0..b {
            for hi in 0..h {
                let q_off = (bi * h + hi) * head_q_size;
                let k_off = (bi * h + hi) * head_kv_size;
                let v_off = (bi * h + hi) * head_kv_size;
                let o_off = (bi * h + hi) * head_o_size;

                let q_slice = &q_data[q_off..q_off + head_q_size];
                let k_slice = &k_data[k_off..k_off + head_kv_size];
                let v_slice = &v_data[v_off..v_off + head_kv_size];
                let o_slice = &got[o_off..o_off + head_o_size];

                let expected =
                    naive_attention_cpu(q_slice, k_slice, v_slice, n, s, d, scale, false);
                let diff = max_abs_diff(o_slice, &expected);
                assert!(
                    diff < 1e-3,
                    "flash_attn 4D batch={bi} head={hi}: max_abs_diff={diff}"
                );
            }
        }
    }
}
