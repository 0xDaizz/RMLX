//! Fused Scaled Dot-Product Attention (SDPA) — Flash Attention style.
//!
//! Single-kernel computation of `softmax(Q @ K^T / sqrt(d) + mask) @ V`.
//!
//! Key optimisations over the unfused path:
//! - **No intermediate materialisation**: The `[seq, total_seq]` score matrix never
//!   exists in device memory — it lives in threadgroup shared memory only.
//! - **Online softmax**: Running max/normaliser across K blocks with O(1) correction.
//! - **Single command buffer**: One encode for the entire attention head.
//!
//! Tiling: Br (query block) x Bc (key block) tiles, iterating K/V blocks in the
//! inner loop. Each threadgroup owns one Br-row chunk of the output.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// Tiling parameters — must match the Metal shader constants.
const BR: usize = 16; // Query block rows
const _BC: usize = 16; // Key/Value block columns (used in shader only)
const THREADS_PER_TG: u64 = 128; // Threads per threadgroup

/// Metal shader for fused SDPA.
///
/// Each threadgroup processes one Br-row block of Q and iterates over all Bc-column
/// blocks of K/V, maintaining a running online-softmax state.
pub const SDPA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Tile sizes — keep in sync with Rust constants.
constant constexpr uint Br = 16;
constant constexpr uint Bc = 16;
constant constexpr uint SIMD_SIZE = 32;

// ─── SIMD reduction helpers ─────────────────────────────────────────────────

inline float simdgroup_max(float val) {
    return simd_max(val);
}

inline float simdgroup_sum(float val) {
    return simd_sum(val);
}

// ─── sdpa_f32 ───────────────────────────────────────────────────────────────
//
// Buffers:
//   0: Q     [N, D]     — query matrix
//   1: K     [S, D]     — key matrix
//   2: V     [S, D]     — value matrix
//   3: O     [N, D]     — output matrix
//   4: mask  [N, S]     — additive attention mask (or nullptr sentinel)
//   5: params [4 x uint32]: { N, S, D, has_mask }
//   6: scale  [float]   — 1/sqrt(D)
//
// Grid:  (ceil(N / Br), 1, 1)  threadgroups
// Threads per threadgroup: 256

kernel void sdpa_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device       float* O         [[buffer(3)]],
    device const float* mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint N        = params[0];
    const uint S        = params[1];
    const uint D        = params[2];
    const uint has_mask = params[3];

    // This threadgroup handles Q rows [q_start .. q_start + Br)
    const uint q_start = tg_id * Br;
    if (q_start >= N) return;
    const uint q_end = min(q_start + Br, N);
    const uint q_count = q_end - q_start;  // actual rows this TG handles

    // Shared memory
    threadgroup float Q_tile[Br * 128];   // Q block [Br, D] — D <= 128
    threadgroup float S_tile[Br * Bc];    // Score block [Br, Bc]
    threadgroup float V_tile[Bc * 128];   // V block [Bc, D] — D <= 128

    // Per-row online softmax state (in registers, one per thread that "owns" a row)
    // We assign threads to rows: thread tid handles row (tid % Br) for row-level ops
    // For matrix ops, all threads cooperate.

    // Output accumulator in shared memory
    threadgroup float O_acc[Br * 128];    // [Br, D]
    threadgroup float m_prev[Br];          // running max per row
    threadgroup float l_prev[Br];          // running sum(exp) per row

    const uint n_threads = 256;

    // Initialise accumulators
    for (uint idx = tid; idx < Br * D; idx += n_threads) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br; idx += n_threads) {
        m_prev[idx] = -INFINITY;
        l_prev[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q tile [q_count, D] into shared memory
    for (uint idx = tid; idx < q_count * D; idx += n_threads) {
        uint r = idx / D;
        uint d = idx % D;
        Q_tile[r * D + d] = Q[(q_start + r) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over K/V blocks
    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        // Load K block [kv_count, D] — we need K^T, so we'll read K row-major
        // and compute Q @ K^T by dot products.
        // Actually, load K into a temporary (reuse V_tile temporarily or compute on the fly).
        // For simplicity and shared memory efficiency, compute S = Q @ K^T directly:
        // S[i,j] = sum_d Q_tile[i,d] * K[(kv_start+j)*D + d]

        // Compute score tile S[Br, Bc] = Q_tile @ K_block^T * scale
        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;  // Q row in tile
            uint j = idx % kv_count;  // K column in tile
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += Q_tile[i * D + d] * K[(kv_start + j) * D + d];
            }
            dot *= scale;

            // Apply additive mask if present
            if (has_mask) {
                dot += mask[(q_start + i) * S + (kv_start + j)];
            }

            S_tile[i * Bc + j] = dot;
        }
        // Fill out-of-bounds score entries with -inf
        for (uint idx = tid; idx < q_count * Bc; idx += n_threads) {
            uint j = idx % Bc;
            if (j >= kv_count) {
                uint i = idx / Bc;
                S_tile[i * Bc + j] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V block [kv_count, D] into shared memory
        for (uint idx = tid; idx < kv_count * D; idx += n_threads) {
            uint r = idx / D;
            uint d = idx % D;
            V_tile[r * D + d] = V[(kv_start + r) * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update per row
        // For each row i in [0, q_count):
        //   m_new = max(m_prev[i], max_j S_tile[i,j])
        //   correction = exp(m_prev[i] - m_new)
        //   l_new = l_prev[i] * correction + sum_j exp(S_tile[i,j] - m_new)
        //   O_acc[i,:] = O_acc[i,:] * correction + sum_j exp(S_tile[i,j] - m_new) * V_tile[j,:]
        //   m_prev[i] = m_new, l_prev[i] = l_new

        // Step 1: Compute per-row max of S_tile
        // Each thread handles a subset of (row, col) pairs
        // We'll do this row by row for clarity, with threads cooperating on columns.

        for (uint i = 0; i < q_count; i++) {
            // Find max of this row's scores
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc + j]);
            }
            // Reduce across threads using simd + shared memory
            float sg_max = simd_max(local_max);
            threadgroup float tg_max_buf[SIMD_SIZE];
            if (lane_id == 0) tg_max_buf[sg_id] = sg_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float row_max;
            if (sg_id == 0) {
                float v = (lane_id < (n_threads / SIMD_SIZE)) ? tg_max_buf[lane_id] : -INFINITY;
                row_max = simd_max(v);
            }
            if (sg_id == 0 && lane_id == 0) {
                tg_max_buf[0] = row_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_new = max(m_prev[i], tg_max_buf[0]);

            // Compute exp(S - m_new) and sum
            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc + j] - m_new);
                S_tile[i * Bc + j] = e;  // overwrite scores with exp values
                local_sum += e;
            }
            float sg_sum = simd_sum(local_sum);
            threadgroup float tg_sum_buf[SIMD_SIZE];
            if (lane_id == 0) tg_sum_buf[sg_id] = sg_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float block_sum;
            if (sg_id == 0) {
                float v = (lane_id < (n_threads / SIMD_SIZE)) ? tg_sum_buf[lane_id] : 0.0f;
                block_sum = simd_sum(v);
            }
            if (sg_id == 0 && lane_id == 0) {
                tg_sum_buf[0] = block_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum_exp = tg_sum_buf[0];

            // Online correction
            float correction = fast::exp(m_prev[i] - m_new);
            float l_new = l_prev[i] * correction + sum_exp;

            // Update O_acc: rescale old accumulator and add new contribution
            // O_acc[i,:] = O_acc[i,:] * correction + sum_j exp_scores[j] * V_tile[j,:]
            for (uint d = tid; d < D; d += n_threads) {
                float o_val = O_acc[i * D + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * V_tile[j * D + d];
                }
                O_acc[i * D + d] = o_val + v_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Update state
            if (tid == 0) {
                m_prev[i] = m_new;
                l_prev[i] = l_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Final normalisation: O[i,:] = O_acc[i,:] / l_prev[i]
    for (uint idx = tid; idx < q_count * D; idx += n_threads) {
        uint i = idx / D;
        uint d = idx % D;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + d] = O_acc[idx] * inv_l;
    }
}

// ─── sdpa_f16 ───────────────────────────────────────────────────────────────
// Same algorithm, half storage, float accumulation.

kernel void sdpa_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint N        = params[0];
    const uint S        = params[1];
    const uint D        = params[2];
    const uint has_mask = params[3];

    const uint q_start = tg_id * Br;
    if (q_start >= N) return;
    const uint q_end = min(q_start + Br, N);
    const uint q_count = q_end - q_start;

    threadgroup float Q_tile[Br * 128];
    threadgroup float S_tile[Br * Bc];
    threadgroup float V_tile[Bc * 128];
    threadgroup float O_acc[Br * 128];
    threadgroup float m_prev[Br];
    threadgroup float l_prev[Br];

    const uint n_threads = 256;

    for (uint idx = tid; idx < Br * D; idx += n_threads) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br; idx += n_threads) {
        m_prev[idx] = -INFINITY;
        l_prev[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q as float
    for (uint idx = tid; idx < q_count * D; idx += n_threads) {
        uint r = idx / D;
        uint d = idx % D;
        Q_tile[r * D + d] = float(Q[(q_start + r) * D + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        // Compute S = Q @ K^T * scale (reading K as float)
        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;
            uint j = idx % kv_count;
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += Q_tile[i * D + d] * float(K[(kv_start + j) * D + d]);
            }
            dot *= scale;
            if (has_mask) {
                dot += float(mask[(q_start + i) * S + (kv_start + j)]);
            }
            S_tile[i * Bc + j] = dot;
        }
        for (uint idx = tid; idx < q_count * Bc; idx += n_threads) {
            uint j = idx % Bc;
            if (j >= kv_count) {
                uint i = idx / Bc;
                S_tile[i * Bc + j] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V as float
        for (uint idx = tid; idx < kv_count * D; idx += n_threads) {
            uint r = idx / D;
            uint d = idx % D;
            V_tile[r * D + d] = float(V[(kv_start + r) * D + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + accumulate (same as f32)
        for (uint i = 0; i < q_count; i++) {
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc + j]);
            }
            float sg_max = simd_max(local_max);
            threadgroup float tg_max_buf[SIMD_SIZE];
            if (lane_id == 0) tg_max_buf[sg_id] = sg_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0) {
                float v = (lane_id < (n_threads / SIMD_SIZE)) ? tg_max_buf[lane_id] : -INFINITY;
                float rm = simd_max(v);
                if (lane_id == 0) tg_max_buf[0] = rm;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_new = max(m_prev[i], tg_max_buf[0]);

            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc + j] - m_new);
                S_tile[i * Bc + j] = e;
                local_sum += e;
            }
            float sg_sum = simd_sum(local_sum);
            threadgroup float tg_sum_buf[SIMD_SIZE];
            if (lane_id == 0) tg_sum_buf[sg_id] = sg_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0) {
                float v = (lane_id < (n_threads / SIMD_SIZE)) ? tg_sum_buf[lane_id] : 0.0f;
                float s = simd_sum(v);
                if (lane_id == 0) tg_sum_buf[0] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum_exp = tg_sum_buf[0];

            float correction = fast::exp(m_prev[i] - m_new);
            float l_new = l_prev[i] * correction + sum_exp;

            for (uint d = tid; d < D; d += n_threads) {
                float o_val = O_acc[i * D + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * V_tile[j * D + d];
                }
                O_acc[i * D + d] = o_val + v_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                m_prev[i] = m_new;
                l_prev[i] = l_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output as half
    for (uint idx = tid; idx < q_count * D; idx += n_threads) {
        uint i = idx / D;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + idx % D] = half(O_acc[idx] * inv_l);
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("sdpa", SDPA_SHADER_SOURCE)
}

/// Fused Scaled Dot-Product Attention.
///
/// Computes `softmax(Q @ K^T / sqrt(D) + mask) @ V` in a single GPU kernel,
/// avoiding materialisation of the full `[N, S]` score matrix.
///
/// # Arguments
/// - `q`: Query matrix `[N, D]` (N = number of query tokens, D = head dimension)
/// - `k`: Key matrix `[S, D]` (S = number of key/value tokens)
/// - `v`: Value matrix `[S, D]`
/// - `mask`: Optional additive mask `[N, S]` (e.g. causal mask with -inf)
/// - `scale`: Scale factor, typically `1.0 / sqrt(D)`
///
/// # Returns
/// Output matrix `[N, D]`.
pub fn sdpa(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate shapes
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "sdpa: Q, K, V must be 2D [tokens, head_dim]".into(),
        ));
    }
    let n = q.shape()[0]; // query tokens
    let d = q.shape()[1]; // head dim
    let s = k.shape()[0]; // key/value tokens

    if k.shape()[1] != d {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: K head_dim {} != Q head_dim {d}",
            k.shape()[1]
        )));
    }
    if v.shape() != [s, d] {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: V shape {:?}, expected [{s}, {d}]",
            v.shape()
        )));
    }
    if d > 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: head_dim {d} > 128 not supported by fused kernel"
        )));
    }
    if let Some(m) = mask {
        if m.shape() != [n, s] {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask shape {:?}, expected [{n}, {s}]",
                m.shape()
            )));
        }
    }

    // Make inputs contiguous
    let q_c = super::make_contiguous(q, registry, queue)?;
    let q = q_c.as_ref().unwrap_or(q);
    let k_c = super::make_contiguous(k, registry, queue)?;
    let k = k_c.as_ref().unwrap_or(k);
    let v_c = super::make_contiguous(v, registry, queue)?;
    let v = v_c.as_ref().unwrap_or(v);

    let kernel_name = match q.dtype() {
        DType::Float32 => "sdpa_f32",
        DType::Float16 => "sdpa_f16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa: unsupported dtype {:?}",
                q.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, q.dtype())?;
    let dev = registry.device().raw();

    // Output array
    let out = Array::zeros(dev, &[n, d], q.dtype());

    // Params buffer: [N, S, D, has_mask]
    let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
    let params: [u32; 4] = [n as u32, s as u32, d as u32, has_mask];
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        16,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Scale buffer
    let scale_buf = dev.new_buffer_with_data(
        &scale as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dummy mask buffer if no mask
    let dummy_buf;
    let mask_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(1, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(2, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(mask_buf), mask_offset);
    encoder.set_buffer(5, Some(&params_buf), 0);
    encoder.set_buffer(6, Some(&scale_buf), 0);

    let n_threadgroups = n.div_ceil(BR) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());

    encoder.dispatch_thread_groups(
        MTLSize::new(n_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

/// Encode fused SDPA into an existing command buffer (no commit/wait).
///
/// **Caller must ensure Q, K, V are contiguous 2D `[tokens, head_dim]`.**
/// Shapes must be pre-validated. head_dim <= 128.
pub fn sdpa_into_cb(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let n = q.shape()[0]; // query tokens
    let d = q.shape()[1]; // head dim
    let s = k.shape()[0]; // key/value tokens

    let kernel_name = match q.dtype() {
        DType::Float32 => "sdpa_f32",
        DType::Float16 => "sdpa_f16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa: unsupported dtype {:?}",
                q.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, q.dtype())?;
    let dev = registry.device().raw();

    // Output array
    let out = Array::zeros(dev, &[n, d], q.dtype());

    // Params buffer: [N, S, D, has_mask]
    let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
    let params: [u32; 4] = [n as u32, s as u32, d as u32, has_mask];
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        16,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Scale buffer
    let scale_buf = dev.new_buffer_with_data(
        &scale as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dummy mask buffer if no mask
    let dummy_buf;
    let mask_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(1, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(2, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(mask_buf), mask_offset);
    encoder.set_buffer(5, Some(&params_buf), 0);
    encoder.set_buffer(6, Some(&scale_buf), 0);

    let n_threadgroups = n.div_ceil(BR) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());

    encoder.dispatch_thread_groups(
        MTLSize::new(n_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// Batched fused SDPA — encode all heads into an existing command buffer.
///
/// **Caller must ensure all head arrays are contiguous 2D.**
pub fn sdpa_batched_into_cb(
    registry: &KernelRegistry,
    q_heads: &[Array],
    k_heads: &[Array],
    v_heads: &[Array],
    mask: Option<&Array>,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Vec<Array>, KernelError> {
    let num_heads = q_heads.len();
    let num_kv_heads = k_heads.len();
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_batched_into_cb: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }
    let repeats = num_heads / num_kv_heads;

    let mut outputs = Vec::with_capacity(num_heads);
    for (h, q_h) in q_heads.iter().enumerate() {
        let kv_idx = h / repeats;
        let out = sdpa_into_cb(
            registry,
            q_h,
            &k_heads[kv_idx],
            &v_heads[kv_idx],
            mask,
            scale,
            cb,
        )?;
        outputs.push(out);
    }
    Ok(outputs)
}

/// Batched fused SDPA — runs one fused kernel per head.
///
/// This is the entry point for attention modules that have already split Q/K/V
/// into per-head arrays.
///
/// # Arguments
/// - `q_heads`: Vec of `[N, D]` query arrays, one per head
/// - `k_heads`: Vec of `[S, D]` key arrays, one per KV head
/// - `v_heads`: Vec of `[S, D]` value arrays, one per KV head
/// - `mask`: Optional additive mask `[N, S]` (shared across heads)
/// - `scale`: Scale factor
/// - `num_heads`: Number of query heads
/// - `num_kv_heads`: Number of KV heads (for GQA)
///
/// # Returns
/// Vec of `[N, D]` output arrays, one per query head.
pub fn sdpa_batched(
    registry: &KernelRegistry,
    q_heads: &[Array],
    k_heads: &[Array],
    v_heads: &[Array],
    mask: Option<&Array>,
    scale: f32,
    queue: &metal::CommandQueue,
) -> Result<Vec<Array>, KernelError> {
    let num_heads = q_heads.len();
    let num_kv_heads = k_heads.len();
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_batched: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }
    let repeats = num_heads / num_kv_heads;

    let mut outputs = Vec::with_capacity(num_heads);
    for (h, q_h) in q_heads.iter().enumerate() {
        let kv_idx = h / repeats;
        let out = sdpa(
            registry,
            q_h,
            &k_heads[kv_idx],
            &v_heads[kv_idx],
            mask,
            scale,
            queue,
        )?;
        outputs.push(out);
    }
    Ok(outputs)
}
