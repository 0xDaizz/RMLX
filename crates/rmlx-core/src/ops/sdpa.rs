//! Fused Scaled Dot-Product Attention (SDPA) — Flash Attention 2 style.
//!
//! Single-kernel computation of `softmax(Q @ K^T / sqrt(d) + mask) @ V`.
//!
//! Key optimisations (Flash Attention 2):
//! - **No intermediate materialisation**: The `[seq, total_seq]` score matrix never
//!   exists in device memory — it lives in threadgroup shared memory only.
//! - **Online softmax**: Running max/normaliser across K/V blocks with O(1) correction.
//! - **K/V outer loop, Q inner loop**: FA2 reorganisation for better GPU utilisation
//!   by reducing shared memory reads of K/V blocks.
//! - **Causal mask skip**: When `is_causal` is set, K/V blocks entirely above the
//!   diagonal are skipped, saving up to ~50% compute for autoregressive decoding.
//! - **Decode fast path**: When N==1 (single query token), a specialised kernel avoids
//!   shared memory overhead for the Q tile and score tile.
//! - **D up to 256**: For D <= 128, shared memory tiles are used. For 128 < D <= 256,
//!   K/V are read directly from global memory while Q and O remain in shared memory.
//!
//! Tiling: Br (query block) x Bc (key block) tiles. Each threadgroup owns one
//! Br-row chunk of the output and iterates over all K/V blocks.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use crate::ops::buffer_slots::{
    sdpa as sdpa_slot, sdpa_diag_qkt, sdpa_diag_single_mma, sdpa_mma, sdpa_nax,
};
use metal::MTLSize;

// Tiling parameters — must match the Metal shader constants.
const BR: usize = 16; // Query block rows
const _BC: usize = 64; // Key/Value block columns (used in shader only)
const THREADS_PER_TG: u64 = 128; // Threads per threadgroup
const DECODE_THREADS: u64 = 256; // Threads for decode kernel

/// Metal shader for fused SDPA — Flash Attention 2 implementation.
///
/// Contains four kernels:
/// - `sdpa_f32` / `sdpa_f16`: General FA2 kernel with causal mask optimisation
/// - `sdpa_decode_f32` / `sdpa_decode_f16`: Fast path for single-query decoding (N==1)
///
/// Each general kernel threadgroup processes one Br-row block of Q against all K/V
/// blocks, with online softmax and optional causal block skipping.
pub const SDPA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Tile sizes — keep in sync with Rust constants.
constant constexpr uint Br = 16;   // Query block rows
constant constexpr uint Bc = 64;   // Key/Value block columns
constant constexpr uint SIMD_SIZE = 32;

// Function constants for specialization (set via MTLFunctionConstantValues)
constant uint FC_HEAD_DIM [[function_constant(200)]];
constant bool FC_HAS_HEAD_DIM = is_function_constant_defined(FC_HEAD_DIM);

// ─── Threadgroup-wide reduction helpers ────────────────────────────────────

// Reduce max across all 256 threads in a threadgroup.
// Uses simd_max within each simdgroup, then cross-simdgroup reduction via
// shared memory.
inline float tg_reduce_max(float val, uint tid, uint lane_id, uint sg_id,
                           uint n_threads, threadgroup float* buf) {
    float sg_val = simd_max(val);
    if (lane_id == 0) buf[sg_id] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result;
    if (sg_id == 0) {
        float v = (lane_id < (n_threads / SIMD_SIZE)) ? buf[lane_id] : -INFINITY;
        result = simd_max(v);
    }
    if (sg_id == 0 && lane_id == 0) buf[0] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// Reduce sum across all 256 threads in a threadgroup.
inline float tg_reduce_sum(float val, uint tid, uint lane_id, uint sg_id,
                           uint n_threads, threadgroup float* buf) {
    float sg_val = simd_sum(val);
    if (lane_id == 0) buf[sg_id] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result;
    if (sg_id == 0) {
        float v = (lane_id < (n_threads / SIMD_SIZE)) ? buf[lane_id] : 0.0f;
        result = simd_sum(v);
    }
    if (sg_id == 0 && lane_id == 0) buf[0] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// ─── MLX sdpa_vector constants ────────────────────────────────────────────
//
// BN_DECODE = number of simdgroups per threadgroup (8 for 256 threads)
// BD_DECODE = SIMD width (always 32 on Apple Silicon)
// All threads parallel across D: each thread handles D/BD elements.
constant constexpr uint BN_DECODE = 8;
constant constexpr uint BD_DECODE = 32;

// ─── sdpa_decode_f32 ──────────────────────────────────────────────────────
//
// MLX sdpa_vector pattern: parallelize across head dimension D, not keys.
// All 256 threads (8 simdgroups x 32 lanes) stay active throughout.
//
// Each thread owns D/32 elements of Q, K, V, and O in registers.
// Each simdgroup processes keys at stride BN_DECODE.
// Cross-simdgroup reduction via transposed shared memory + simd_sum.
//
// Buffers:
//   0: Q      [1, D]     — single query vector
//   1: K      [S, D]     — key matrix
//   2: V      [S, D]     — value matrix
//   3: O      [1, D]     — output vector
//   4: mask   [1, S]     — additive mask (or dummy)
//   5: params [5 x uint32]: { N=1, S, D, has_mask, is_causal }
//   6: scale  [float]    — 1/sqrt(D)
//
// Grid: (1, 1, 1) — single threadgroup
// Threads per threadgroup: 256

kernel void sdpa_decode_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device       float* O         [[buffer(3)]],
    device const float* mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];

    const uint elems_per_thread = D / BD_DECODE;

    // Load Q into registers (each thread owns its D-slice, pre-scaled)
    float q_reg[8];  // max D=256 -> 256/32 = 8
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = Q[simd_lid * elems_per_thread + i] * scale;
    }

    // Output accumulator in registers
    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Per-simdgroup online softmax state
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    // Each simdgroup handles keys at stride BN_DECODE
    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        // Load K[i] slice into registers
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = K[i * D + simd_lid * elems_per_thread + j];
        }

        // Q·K dot product: partial per thread, then simd_sum across lanes
        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        // Apply mask
        if (has_mask) {
            score += mask[i];
        }

        // Online softmax update
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Accumulate V[i] weighted by exp_score, with correction
        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * V[i * D + simd_lid * elems_per_thread + j];
        }
    }

    // Cross-simdgroup reduction (MLX pattern)
    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute global max using SIMD reduction
    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);

    // Per-simdgroup correction factor (each lane holds factor for one simdgroup)
    float factor = fast::exp(max_score - g_max);

    // Compute global sum
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    // Aggregate outputs across simdgroups via transpose + simd_sum
    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output — only lane 0 of each simdgroup writes its chunk
    if (simd_lid == 0) {
        for (uint j = 0; j < elems_per_thread; j++) {
            O[simd_gid * elems_per_thread + j] = o_reg[j];
        }
    }
}

// ─── sdpa_decode_f16 ──────────────────────────────────────────────────────
// MLX sdpa_vector pattern for f16. Reads/writes half, accumulates in float.

kernel void sdpa_decode_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];

    const uint elems_per_thread = D / BD_DECODE;

    float q_reg[8];
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = float(Q[simd_lid * elems_per_thread + i]) * scale;
    }

    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = float(K[i * D + simd_lid * elems_per_thread + j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        if (has_mask) {
            score += float(mask[i]);
        }

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * float(V[i * D + simd_lid * elems_per_thread + j]);
        }
    }

    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);
    float factor = fast::exp(max_score - g_max);
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint j = 0; j < elems_per_thread; j++) {
            O[simd_gid * elems_per_thread + j] = half(o_reg[j]);
        }
    }
}

// ─── sdpa_f32 ─────────────────────────────────────────────────────────────
//
// Flash Attention 2: General kernel with causal mask optimisation.
//
// Each threadgroup owns one Br-row block of Q and iterates over K/V blocks.
// The FA2 improvement: causal block skipping and improved inner loop.
//
// For D <= 128: Q_tile, O_acc, S_tile, V_tile all in shared memory.
// For 128 < D <= 256: Q_tile and O_acc in shared memory, K/V read from
// global memory directly.
//
// Buffers:
//   0: Q      [N, D]
//   1: K      [S, D]
//   2: V      [S, D]
//   3: O      [N, D]
//   4: mask   [N, S] or dummy
//   5: params [5 x uint32]: { N, S, D, has_mask, is_causal }
//   6: scale  [float]
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
    const uint N         = params[0];
    const uint S         = params[1];
    const uint D         = params[2];
    const uint has_mask  = params[3];
    const uint is_causal = params[4];

    // This threadgroup handles Q rows [q_start .. q_start + Br)
    const uint q_start = tg_id * Br;
    if (q_start >= N) return;
    const uint q_end   = min(q_start + Br, N);
    const uint q_count = q_end - q_start;

    // Shared memory layout:
    // For D <= 128: Q_tile[Br*128] + O_acc[Br*128] + S_tile[Br*Bc] + V_tile[Bc*128]
    //              + m_prev[Br] + l_prev[Br] + reduce_buf[SIMD_SIZE]
    //   = 4096 + 4096 + 1024 + 4096 + 32 + 32 + 32 = ~13312 floats = ~52KB
    //   This is tight but within Metal's 32KB threadgroup limit when Br=Bc=32, D=128:
    //   Actually 13312 * 4 = 53248 bytes > 32KB. We need to be smarter.
    //
    // Optimisation: Don't store V_tile in shared memory — read V from global
    // memory directly in the accumulation loop. This saves Bc*128 = 4096 floats.
    // Similarly for D > 128, read K from global memory too.
    //
    // With V from global memory:
    //   Q_tile[Br*D] + O_acc[Br*D] + S_tile[Br*Bc] + m_prev[Br] + l_prev[Br]
    //   For D=128: 4096 + 4096 + 1024 + 32 + 32 = 9280 floats = 37120 bytes
    //   Still over 32KB. Reduce further by using D directly:
    //   For D=64: 2048 + 2048 + 1024 + 32 + 32 = 5184 floats = 20736 bytes -> OK
    //   For D=128: 4096 + 4096 + 1024 + 32 + 32 = 9280 floats = 37120 bytes -> over
    //
    // Solution: Use a smaller Br tile or split D into passes.
    // Actually Metal's threadgroup memory limit on Apple Silicon is typically 32KB.
    // But we can use less shared memory by not storing O_acc — write partial results
    // to device memory.
    //
    // Practical approach: Q_tile + S_tile only in shared memory.
    // O_acc, m_prev, l_prev in device memory (per-threadgroup scratch).
    //
    // Even simpler: just Q_tile and S_tile in shared memory, accumulate O
    // in shared memory but cap tile sizes.
    //
    // Let's use the proven approach: Q_tile[Br * D] + S_tile[Br * Bc] in shared
    // memory. O_acc stored per-thread in registers (each thread owns D/n_threads
    // dimensions). m_prev and l_prev are small (Br floats each).
    //
    // With D=128, Br=32:
    //   Q_tile: 32 * 128 = 4096 floats = 16384 bytes
    //   S_tile: 32 * 32  = 1024 floats =  4096 bytes
    //   m_prev: 32 floats = 128 bytes
    //   l_prev: 32 floats = 128 bytes
    //   reduce_buf: 32 floats = 128 bytes
    //   Total: ~20864 bytes -> OK within 32KB
    //
    // With D=256, Br=32:
    //   Q_tile: 32 * 256 = 8192 floats = 32768 bytes -> exactly 32KB, too tight
    //   Need smaller Br for D>128. Use Br=16 for D>128:
    //   Q_tile: 16 * 256 = 4096 floats = 16384 bytes
    //   S_tile: 16 * 32  = 512 floats  =  2048 bytes
    //   Total: ~18560 bytes -> OK
    //
    // For simplicity in this kernel, we use Br=32 and cap Q_tile at 128 columns.
    // For D > 128, we process D in two passes (0..128 and 128..D) for the Q*K dot
    // product, and read K/V directly from global memory.

    // Shared memory — sized for worst case D=128 in Q_tile
    threadgroup float Q_tile[Br * 128];   // [Br, min(D,128)] for dot products
    threadgroup float S_tile[Br * Bc];    // Score block [Br, Bc]
    threadgroup float m_prev[Br];          // running max per row
    threadgroup float l_prev[Br];          // running sum(exp) per row
    threadgroup float reduce_buf[SIMD_SIZE]; // for reductions

    // O_acc in shared memory — we need to read/write across iterations
    // For D<=128 this fits. For D>128 we use a separate second-half array.
    threadgroup float O_acc[Br * 128];     // [Br, min(D,128)]
    threadgroup float O_acc2[Br * 128];    // [Br, D-128] for D>128 (max 128 cols)

    const uint n_threads = 128;
    const uint D_lo = min(D, 128u);  // first chunk of D
    const uint D_hi = (D > 128u) ? (D - 128u) : 0u;  // second chunk (0 if D<=128)

    // Initialise accumulators
    for (uint idx = tid; idx < Br * D_lo; idx += n_threads) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br * D_hi; idx += n_threads) {
        O_acc2[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br; idx += n_threads) {
        m_prev[idx] = -INFINITY;
        l_prev[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q tile [q_count, D_lo] into shared memory (first 128 dims)
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint r = idx / D_lo;
        uint d = idx % D_lo;
        Q_tile[r * D_lo + d] = Q[(q_start + r) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine K/V block range based on causal masking
    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    // For causal masking: the last K/V block this Q block can attend to.
    // Q row q_start+i can attend to K indices 0..q_start+i (inclusive).
    // So the last relevant K/V block contains index (q_start + q_count - 1).
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc) + 1)
        : n_kv_blocks;

    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        // Compute score tile S[q_count, kv_count] = Q @ K^T * scale
        // For D <= 128: dot product uses Q_tile (shared) and K (global)
        // For D > 128: first 128 dims from Q_tile, remaining from Q global
        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;  // Q row in tile
            uint j = idx % kv_count;  // K column in tile
            float dot = 0.0f;

            // First D_lo dimensions from shared memory Q_tile
            for (uint d = 0; d < D_lo; d++) {
                dot += Q_tile[i * D_lo + d] * K[(kv_start + j) * D + d];
            }
            // Remaining dimensions (D > 128) from global memory
            for (uint d = D_lo; d < D; d++) {
                dot += Q[(q_start + i) * D + d] * K[(kv_start + j) * D + d];
            }
            dot *= scale;

            // Apply additive mask if present
            if (has_mask) {
                dot += mask[(q_start + i) * S + (kv_start + j)];
            }

            // Apply causal mask: positions where kv_idx > q_idx get -inf
            if (is_causal) {
                uint q_idx = q_start + i;
                uint kv_idx = kv_start + j;
                if (kv_idx > q_idx) {
                    dot = -INFINITY;
                }
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

        // Online softmax update + O accumulation per row
        for (uint i = 0; i < q_count; i++) {
            // Row max
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc + j]);
            }
            float m_new = tg_reduce_max(local_max, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
            m_new = max(m_prev[i], m_new);

            // Exp scores and sum
            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc + j] - m_new);
                S_tile[i * Bc + j] = e;
                local_sum += e;
            }
            float sum_exp = tg_reduce_sum(local_sum, tid, lane_id, sg_id,
                                          n_threads, reduce_buf);

            // Online correction
            float correction = fast::exp(m_prev[i] - m_new);
            float l_new = l_prev[i] * correction + sum_exp;

            // Update O_acc: rescale old accumulator and add new contribution
            // O_acc[i,:] = O_acc[i,:] * correction + sum_j exp_scores[j] * V[j,:]
            // V is read directly from global memory.

            // First D_lo dimensions
            for (uint d = tid; d < D_lo; d += n_threads) {
                float o_val = O_acc[i * D_lo + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * V[(kv_start + j) * D + d];
                }
                O_acc[i * D_lo + d] = o_val + v_sum;
            }

            // Remaining D_hi dimensions (only when D > 128)
            for (uint d = tid; d < D_hi; d += n_threads) {
                float o_val = O_acc2[i * D_hi + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * V[(kv_start + j) * D + (D_lo + d)];
                }
                O_acc2[i * D_hi + d] = o_val + v_sum;
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
    // First D_lo dimensions
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint i = idx / D_lo;
        uint d = idx % D_lo;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + d] = O_acc[idx] * inv_l;
    }
    // Remaining D_hi dimensions
    for (uint idx = tid; idx < q_count * D_hi; idx += n_threads) {
        uint i = idx / D_hi;
        uint d = idx % D_hi;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + (D_lo + d)] = O_acc2[idx] * inv_l;
    }
}

// ─── sdpa_f16 ─────────────────────────────────────────────────────────────
// Flash Attention 2 for f16. Q_tile stored as half (2x memory savings),
// enabling BC=128 (doubled from 64). S_tile, O_acc, softmax stats remain f32.
//
// Shared memory budget (D=128, Br=16, Bc_f16=128):
//   Q_tile:  16*128*2 =  4096 bytes (half)
//   O_acc:   16*128*4 =  8192 bytes (float)
//   O_acc2:  16*128*4 =  8192 bytes (float, for D>128)
//   S_tile:  16*128*4 =  8192 bytes (float, for softmax numerics)
//   m/l/red: ~288 bytes
//   Total:   ~28960 bytes < 32KB

constant constexpr uint Bc_f16 = 128;  // Doubled KV block size for f16

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
    const uint N         = params[0];
    const uint S         = params[1];
    const uint D         = params[2];
    const uint has_mask  = params[3];
    const uint is_causal = params[4];

    const uint q_start = tg_id * Br;
    if (q_start >= N) return;
    const uint q_end   = min(q_start + Br, N);
    const uint q_count = q_end - q_start;

    // Q_tile in half — 2x memory savings vs float
    threadgroup half  Q_tile[Br * 128];
    threadgroup float S_tile[Br * Bc_f16];   // f32 for softmax numerics
    threadgroup float m_prev[Br];
    threadgroup float l_prev[Br];
    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_acc[Br * 128];
    threadgroup float O_acc2[Br * 128];

    const uint n_threads = 128;
    const uint D_lo = min(D, 128u);
    const uint D_hi = (D > 128u) ? (D - 128u) : 0u;

    for (uint idx = tid; idx < Br * D_lo; idx += n_threads) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br * D_hi; idx += n_threads) {
        O_acc2[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br; idx += n_threads) {
        m_prev[idx] = -INFINITY;
        l_prev[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q as half (first D_lo dims) — stays in half precision
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint r = idx / D_lo;
        uint d = idx % D_lo;
        Q_tile[r * D_lo + d] = Q[(q_start + r) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (S + Bc_f16 - 1) / Bc_f16;
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc_f16) + 1)
        : n_kv_blocks;

    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * Bc_f16;
        const uint kv_end   = min(kv_start + Bc_f16, S);
        const uint kv_count = kv_end - kv_start;

        // Compute S = Q @ K^T * scale (Q from half shmem, K from global half)
        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;
            uint j = idx % kv_count;
            float dot = 0.0f;

            // First D_lo dims from half shared memory
            for (uint d = 0; d < D_lo; d++) {
                dot += float(Q_tile[i * D_lo + d]) * float(K[(kv_start + j) * D + d]);
            }
            // Remaining dims (D > 128) from global memory
            for (uint d = D_lo; d < D; d++) {
                dot += float(Q[(q_start + i) * D + d]) * float(K[(kv_start + j) * D + d]);
            }
            dot *= scale;

            if (has_mask) {
                dot += float(mask[(q_start + i) * S + (kv_start + j)]);
            }
            if (is_causal) {
                uint q_idx = q_start + i;
                uint kv_idx = kv_start + j;
                if (kv_idx > q_idx) {
                    dot = -INFINITY;
                }
            }

            S_tile[i * Bc_f16 + j] = dot;
        }
        for (uint idx = tid; idx < q_count * Bc_f16; idx += n_threads) {
            uint j = idx % Bc_f16;
            if (j >= kv_count) {
                uint i = idx / Bc_f16;
                S_tile[i * Bc_f16 + j] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + accumulate
        for (uint i = 0; i < q_count; i++) {
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc_f16 + j]);
            }
            float m_new = tg_reduce_max(local_max, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
            m_new = max(m_prev[i], m_new);

            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc_f16 + j] - m_new);
                S_tile[i * Bc_f16 + j] = e;
                local_sum += e;
            }
            float sum_exp = tg_reduce_sum(local_sum, tid, lane_id, sg_id,
                                          n_threads, reduce_buf);

            float correction = fast::exp(m_prev[i] - m_new);
            float l_new = l_prev[i] * correction + sum_exp;

            for (uint d = tid; d < D_lo; d += n_threads) {
                float o_val = O_acc[i * D_lo + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc_f16 + j] * float(V[(kv_start + j) * D + d]);
                }
                O_acc[i * D_lo + d] = o_val + v_sum;
            }
            for (uint d = tid; d < D_hi; d += n_threads) {
                float o_val = O_acc2[i * D_hi + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc_f16 + j] * float(V[(kv_start + j) * D + (D_lo + d)]);
                }
                O_acc2[i * D_hi + d] = o_val + v_sum;
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
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint i = idx / D_lo;
        uint d = idx % D_lo;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + d] = half(O_acc[idx] * inv_l);
    }
    for (uint idx = tid; idx < q_count * D_hi; idx += n_threads) {
        uint i = idx / D_hi;
        uint d = idx % D_hi;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + (D_lo + d)] = half(O_acc2[idx] * inv_l);
    }
}

// ─── sdpa_decode_batched_f32 ──────────────────────────────────────────────
//
// MLX sdpa_vector pattern — batched multi-head decode.
// Grid.x = num_heads, one threadgroup per head.
// All 256 threads parallel across D for full GPU utilization.
//
// Buffers:
//   0: Q      [num_heads * D]              — flat query slab
//   1: K      [num_kv_heads * S * D]       — flat key slab
//   2: V      [num_kv_heads * S * D]       — flat value slab
//   3: O      [num_heads * D]              — flat output slab
//   4: mask   [S]                          — additive mask (shared, or dummy)
//   5: params [6 x uint32]: { num_heads, num_kv_heads, S, D, has_mask, stride_S }
//   6: scale  [float]
//
// Grid: (num_heads, 1, 1)
// Threads per threadgroup: 256

kernel void sdpa_decode_batched_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device const float* V         [[buffer(2)]],
    device       float* O         [[buffer(3)]],
    device const float* mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  tg_pos    [[threadgroup_position_in_grid]])
{
    const uint head_id      = tg_pos;
    const uint num_heads    = params[0];
    const uint num_kv_heads = params[1];
    const uint S            = params[2];
    const uint D            = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[3];
    const uint has_mask     = params[4];
    const uint stride_S     = (params[5] > 0) ? params[5] : S;

    if (head_id >= num_heads) return;

    const uint kv_head = head_id * num_kv_heads / num_heads;

    device const float* q = Q + head_id * D;
    device const float* k = K + kv_head * stride_S * D;
    device const float* v = V + kv_head * stride_S * D;

    const uint elems_per_thread = D / BD_DECODE;

    // Load Q into registers (pre-scaled)
    float q_reg[8];
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = q[simd_lid * elems_per_thread + i] * scale;
    }

    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = k[i * D + simd_lid * elems_per_thread + j];
        }

        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        if (has_mask) {
            score += mask[i];
        }

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * v[i * D + simd_lid * elems_per_thread + j];
        }
    }

    // Cross-simdgroup reduction
    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);
    float factor = fast::exp(max_score - g_max);
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device float* o = O + head_id * D;
        for (uint j = 0; j < elems_per_thread; j++) {
            o[simd_gid * elems_per_thread + j] = o_reg[j];
        }
    }
}

// ─── sdpa_decode_batched_f16 ──────────────────────────────────────────────
// MLX sdpa_vector pattern for batched f16 decode.

kernel void sdpa_decode_batched_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  tg_pos    [[threadgroup_position_in_grid]])
{
    const uint head_id      = tg_pos;
    const uint num_heads    = params[0];
    const uint num_kv_heads = params[1];
    const uint S            = params[2];
    const uint D            = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[3];
    const uint has_mask     = params[4];
    const uint stride_S     = (params[5] > 0) ? params[5] : S;

    if (head_id >= num_heads) return;

    const uint kv_head = head_id * num_kv_heads / num_heads;

    device const half* q = Q + head_id * D;
    device const half* k = K + kv_head * stride_S * D;
    device const half* v = V + kv_head * stride_S * D;

    const uint elems_per_thread = D / BD_DECODE;

    float q_reg[8];
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = float(q[simd_lid * elems_per_thread + i]) * scale;
    }

    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = float(k[i * D + simd_lid * elems_per_thread + j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        if (has_mask) {
            score += float(mask[i]);
        }

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * float(v[i * D + simd_lid * elems_per_thread + j]);
        }
    }

    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);
    float factor = fast::exp(max_score - g_max);
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device half* o = O + head_id * D;
        for (uint j = 0; j < elems_per_thread; j++) {
            o[simd_gid * elems_per_thread + j] = half(o_reg[j]);
        }
    }
}

// ─── sdpa_prefill_gqa_f16 ─────────────────────────────────────────────────
//
// GQA-optimized prefill: processes multiple Q heads sharing the same K/V head
// in a single threadgroup, loading K/V once for all Q heads in the group.
//
// For GQA with ratio R (e.g., 32 Q heads / 8 KV heads = R=4), this kernel
// loads each K/V block once and computes attention for R Q heads, saving
// R-1 redundant K/V global memory reads per block.
//
// Grid: (ceil(N/Br), num_kv_heads, 1)
// Each threadgroup iterates over R Q heads internally.
//
// Buffers:
//   0: Q      [num_heads * N * D]     — all Q heads contiguous
//   1: K      [num_kv_heads * S * D]  — all KV heads contiguous
//   2: V      [num_kv_heads * S * D]  — all KV heads contiguous
//   3: O      [num_heads * N * D]     — output, same layout as Q
//   4: mask   [N * S] or dummy        — additive mask (shared across heads)
//   5: params [10 x uint32]: { N, S, D, has_mask, is_causal, num_heads, num_kv_heads, gqa_ratio, n_q_blocks, kv_stride_S }
//   6: scale  [float]
//
// Threads per threadgroup: 128

// Grid is dispatched as 1D: grid.x = n_q_blocks * num_kv_heads
// We compute q_block_id and kv_head_id from the flat threadgroup index.
// params[8] = n_q_blocks (grid stride for kv_head decomposition)
// params[9] = kv_stride_S (inter-head stride in KV slab, may be > S for pre-alloc cache)

kernel void sdpa_prefill_gqa_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  tg_flat   [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint N          = params[0];
    const uint S          = params[1];
    const uint D          = params[2];
    const uint has_mask   = params[3];
    const uint is_causal  = params[4];
    const uint num_heads  = params[5];
    const uint num_kv_heads = params[6];
    const uint gqa_ratio  = params[7];  // num_heads / num_kv_heads
    const uint n_q_blocks = params[8];  // ceil(N / Br)
    const uint kv_stride_S = params[9]; // inter-head stride (may be > S for pre-alloc cache)

    const uint q_block_id = tg_flat % n_q_blocks;
    const uint kv_head_id = tg_flat / n_q_blocks;

    const uint q_start = q_block_id * Br;
    if (q_start >= N) return;
    const uint q_end   = min(q_start + Br, N);
    const uint q_count = q_end - q_start;

    const uint n_threads = 128;
    const uint D_lo = min(D, 128u);

    // KV pointers for this KV head (use kv_stride_S for inter-head offset)
    device const half* K_head = K + kv_head_id * kv_stride_S * D;
    device const half* V_head = V + kv_head_id * kv_stride_S * D;

    const uint n_kv_blocks = (S + Bc_f16 - 1) / Bc_f16;

    // Threadgroup memory — declared OUTSIDE the GQA loop so the Metal
    // compiler allocates a single set (not per-iteration).  All arrays
    // are re-initialised at the top of each iteration so this is safe.
    threadgroup half  Q_tile[Br * 128];
    threadgroup float S_tile[Br * Bc_f16];
    threadgroup float m_prev[Br];
    threadgroup float l_prev[Br];
    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_acc[Br * 128];

    // Process each Q head in this GQA group
    for (uint qh = 0; qh < gqa_ratio; qh++) {
        const uint head_id = kv_head_id * gqa_ratio + qh;
        if (head_id >= num_heads) break;

        device const half* Q_head = Q + head_id * N * D;
        device       half* O_head = O + head_id * N * D;

        // Initialize accumulators
        for (uint idx = tid; idx < Br * D_lo; idx += n_threads) {
            O_acc[idx] = 0.0f;
        }
        for (uint idx = tid; idx < Br; idx += n_threads) {
            m_prev[idx] = -INFINITY;
            l_prev[idx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load Q tile for this head
        for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
            uint r = idx / D_lo;
            uint d = idx % D_lo;
            Q_tile[r * D_lo + d] = Q_head[(q_start + r) * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint max_kv_block = is_causal
            ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc_f16) + 1)
            : n_kv_blocks;

        for (uint kb = 0; kb < max_kv_block; kb++) {
            const uint kv_start = kb * Bc_f16;
            const uint kv_end   = min(kv_start + Bc_f16, S);
            const uint kv_count = kv_end - kv_start;

            // Compute S = Q @ K^T * scale
            for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
                uint i = idx / kv_count;
                uint j = idx % kv_count;
                float dot = 0.0f;

                for (uint d = 0; d < D_lo; d++) {
                    dot += float(Q_tile[i * D_lo + d]) * float(K_head[(kv_start + j) * D + d]);
                }
                dot *= scale;

                if (has_mask) {
                    dot += float(mask[(q_start + i) * S + (kv_start + j)]);
                }
                if (is_causal) {
                    if ((kv_start + j) > (q_start + i)) {
                        dot = -INFINITY;
                    }
                }
                S_tile[i * Bc_f16 + j] = dot;
            }
            for (uint idx = tid; idx < q_count * Bc_f16; idx += n_threads) {
                uint j = idx % Bc_f16;
                if (j >= kv_count) {
                    uint i = idx / Bc_f16;
                    S_tile[i * Bc_f16 + j] = -INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Online softmax + accumulate
            for (uint i = 0; i < q_count; i++) {
                float local_max = -INFINITY;
                for (uint j = tid; j < kv_count; j += n_threads) {
                    local_max = max(local_max, S_tile[i * Bc_f16 + j]);
                }
                float m_new = tg_reduce_max(local_max, tid, lane_id, sg_id,
                                            n_threads, reduce_buf);
                m_new = max(m_prev[i], m_new);

                float local_sum = 0.0f;
                for (uint j = tid; j < kv_count; j += n_threads) {
                    float e = fast::exp(S_tile[i * Bc_f16 + j] - m_new);
                    S_tile[i * Bc_f16 + j] = e;
                    local_sum += e;
                }
                float sum_exp = tg_reduce_sum(local_sum, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);

                float correction = fast::exp(m_prev[i] - m_new);
                float l_new = l_prev[i] * correction + sum_exp;

                for (uint d = tid; d < D_lo; d += n_threads) {
                    float o_val = O_acc[i * D_lo + d] * correction;
                    float v_sum = 0.0f;
                    for (uint j = 0; j < kv_count; j++) {
                        v_sum += S_tile[i * Bc_f16 + j] * float(V_head[(kv_start + j) * D + d]);
                    }
                    O_acc[i * D_lo + d] = o_val + v_sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0) {
                    m_prev[i] = m_new;
                    l_prev[i] = l_new;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Write output
        for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
            uint i = idx / D_lo;
            uint d = idx % D_lo;
            float l = l_prev[i];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            O_head[(q_start + i) * D + d] = half(O_acc[idx] * inv_l);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

// ---------------------------------------------------------------------------
// bf16 SDPA shader — same FA2 algorithm, reads/writes bfloat, accumulates f32
// (C8: SDPA missing bf16 support)
// ---------------------------------------------------------------------------

pub const SDPA_BF16_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint Br = 16;
constant constexpr uint Bc = 64;
constant constexpr uint SIMD_SIZE = 32;

// Function constants for specialization (set via MTLFunctionConstantValues)
constant uint FC_HEAD_DIM [[function_constant(200)]];
constant bool FC_HAS_HEAD_DIM = is_function_constant_defined(FC_HEAD_DIM);

inline float tg_reduce_max_bf16(float val, uint tid, uint lane_id, uint sg_id,
                                 uint n_threads, threadgroup float* buf) {
    float sg_val = simd_max(val);
    if (lane_id == 0) buf[sg_id] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result;
    if (sg_id == 0) {
        float v = (lane_id < (n_threads / SIMD_SIZE)) ? buf[lane_id] : -INFINITY;
        result = simd_max(v);
    }
    if (sg_id == 0 && lane_id == 0) buf[0] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

inline float tg_reduce_sum_bf16(float val, uint tid, uint lane_id, uint sg_id,
                                 uint n_threads, threadgroup float* buf) {
    float sg_val = simd_sum(val);
    if (lane_id == 0) buf[sg_id] = sg_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result;
    if (sg_id == 0) {
        float v = (lane_id < (n_threads / SIMD_SIZE)) ? buf[lane_id] : 0.0f;
        result = simd_sum(v);
    }
    if (sg_id == 0 && lane_id == 0) buf[0] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// ─── MLX sdpa_vector constants for bf16 shader ────────────────────────────
constant constexpr uint BN_DECODE = 8;
constant constexpr uint BD_DECODE = 32;

// ─── sdpa_decode_bf16 ─────────────────────────────────────────────────────
// MLX sdpa_vector pattern for bf16. Reads/writes bfloat, accumulates in float.

kernel void sdpa_decode_bf16(
    device const bfloat* Q         [[buffer(0)]],
    device const bfloat* K         [[buffer(1)]],
    device const bfloat* V         [[buffer(2)]],
    device       bfloat* O         [[buffer(3)]],
    device const bfloat* mask      [[buffer(4)]],
    constant     uint*   params    [[buffer(5)]],
    constant     float&  scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];

    const uint elems_per_thread = D / BD_DECODE;

    float q_reg[8];
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = float(Q[simd_lid * elems_per_thread + i]) * scale;
    }

    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = float(K[i * D + simd_lid * elems_per_thread + j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        if (has_mask) {
            score += float(mask[i]);
        }

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * float(V[i * D + simd_lid * elems_per_thread + j]);
        }
    }

    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);
    float factor = fast::exp(max_score - g_max);
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint j = 0; j < elems_per_thread; j++) {
            O[simd_gid * elems_per_thread + j] = bfloat(o_reg[j]);
        }
    }
}

// ─── sdpa_bf16 ────────────────────────────────────────────────────────────
// Q_tile stored as bfloat (2x memory savings), BC doubled to 128.

constant constexpr uint Bc_bf16 = 128;  // Doubled KV block size for bf16

kernel void sdpa_bf16(
    device const bfloat* Q         [[buffer(0)]],
    device const bfloat* K         [[buffer(1)]],
    device const bfloat* V         [[buffer(2)]],
    device       bfloat* O         [[buffer(3)]],
    device const bfloat* mask      [[buffer(4)]],
    constant     uint*   params    [[buffer(5)]],
    constant     float&  scale     [[buffer(6)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint N         = params[0];
    const uint S         = params[1];
    const uint D         = params[2];
    const uint has_mask  = params[3];
    const uint is_causal = params[4];

    const uint q_start = tg_id * Br;
    if (q_start >= N) return;
    const uint q_end   = min(q_start + Br, N);
    const uint q_count = q_end - q_start;

    threadgroup bfloat Q_tile[Br * 128];    // bfloat — 2x savings
    threadgroup float  S_tile[Br * Bc_bf16]; // f32 for softmax numerics
    threadgroup float  m_prev[Br];
    threadgroup float  l_prev[Br];
    threadgroup float  reduce_buf[SIMD_SIZE];
    threadgroup float  O_acc[Br * 128];
    threadgroup float  O_acc2[Br * 128];

    const uint n_threads = 128;
    const uint D_lo = min(D, 128u);
    const uint D_hi = (D > 128u) ? (D - 128u) : 0u;

    for (uint idx = tid; idx < Br * D_lo; idx += n_threads) {
        O_acc[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br * D_hi; idx += n_threads) {
        O_acc2[idx] = 0.0f;
    }
    for (uint idx = tid; idx < Br; idx += n_threads) {
        m_prev[idx] = -INFINITY;
        l_prev[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load Q as bfloat (first D_lo dims) — stays in bfloat precision
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint r = idx / D_lo;
        uint d = idx % D_lo;
        Q_tile[r * D_lo + d] = Q[(q_start + r) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (S + Bc_bf16 - 1) / Bc_bf16;
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc_bf16) + 1)
        : n_kv_blocks;

    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * Bc_bf16;
        const uint kv_end   = min(kv_start + Bc_bf16, S);
        const uint kv_count = kv_end - kv_start;

        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;
            uint j = idx % kv_count;
            float dot = 0.0f;

            for (uint d = 0; d < D_lo; d++) {
                dot += float(Q_tile[i * D_lo + d]) * float(K[(kv_start + j) * D + d]);
            }
            for (uint d = D_lo; d < D; d++) {
                dot += float(Q[(q_start + i) * D + d]) * float(K[(kv_start + j) * D + d]);
            }
            dot *= scale;

            if (has_mask) {
                dot += float(mask[(q_start + i) * S + (kv_start + j)]);
            }
            if (is_causal) {
                uint q_idx = q_start + i;
                uint kv_idx = kv_start + j;
                if (kv_idx > q_idx) {
                    dot = -INFINITY;
                }
            }

            S_tile[i * Bc_bf16 + j] = dot;
        }
        for (uint idx = tid; idx < q_count * Bc_bf16; idx += n_threads) {
            uint j = idx % Bc_bf16;
            if (j >= kv_count) {
                uint i = idx / Bc_bf16;
                S_tile[i * Bc_bf16 + j] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < q_count; i++) {
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc_bf16 + j]);
            }
            float m_new = tg_reduce_max_bf16(local_max, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
            m_new = max(m_prev[i], m_new);

            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc_bf16 + j] - m_new);
                S_tile[i * Bc_bf16 + j] = e;
                local_sum += e;
            }
            float sum_exp = tg_reduce_sum_bf16(local_sum, tid, lane_id, sg_id,
                                                n_threads, reduce_buf);

            float correction = fast::exp(m_prev[i] - m_new);
            float l_new = l_prev[i] * correction + sum_exp;

            for (uint d = tid; d < D_lo; d += n_threads) {
                float o_val = O_acc[i * D_lo + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc_bf16 + j] * float(V[(kv_start + j) * D + d]);
                }
                O_acc[i * D_lo + d] = o_val + v_sum;
            }
            for (uint d = tid; d < D_hi; d += n_threads) {
                float o_val = O_acc2[i * D_hi + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc_bf16 + j] * float(V[(kv_start + j) * D + (D_lo + d)]);
                }
                O_acc2[i * D_hi + d] = o_val + v_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                m_prev[i] = m_new;
                l_prev[i] = l_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output as bfloat
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint i = idx / D_lo;
        uint d = idx % D_lo;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + d] = bfloat(O_acc[idx] * inv_l);
    }
    for (uint idx = tid; idx < q_count * D_hi; idx += n_threads) {
        uint i = idx / D_hi;
        uint d = idx % D_hi;
        float l = l_prev[i];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        O[(q_start + i) * D + (D_lo + d)] = bfloat(O_acc2[idx] * inv_l);
    }
}

// ─── sdpa_decode_batched_bf16 ─────────────────────────────────────────────
// MLX sdpa_vector pattern for batched bf16 decode.

kernel void sdpa_decode_batched_bf16(
    device const bfloat* Q         [[buffer(0)]],
    device const bfloat* K         [[buffer(1)]],
    device const bfloat* V         [[buffer(2)]],
    device       bfloat* O         [[buffer(3)]],
    device const bfloat* mask      [[buffer(4)]],
    constant     uint*   params    [[buffer(5)]],
    constant     float&  scale     [[buffer(6)]],
    uint  simd_lid  [[thread_index_in_simdgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  tg_pos    [[threadgroup_position_in_grid]])
{
    const uint head_id      = tg_pos;
    const uint num_heads    = params[0];
    const uint num_kv_heads = params[1];
    const uint S            = params[2];
    const uint D            = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[3];
    const uint has_mask     = params[4];
    const uint stride_S     = (params[5] > 0) ? params[5] : S;

    if (head_id >= num_heads) return;

    const uint kv_head = head_id * num_kv_heads / num_heads;

    device const bfloat* q = Q + head_id * D;
    device const bfloat* k = K + kv_head * stride_S * D;
    device const bfloat* v = V + kv_head * stride_S * D;

    const uint elems_per_thread = D / BD_DECODE;

    float q_reg[8];
    for (uint i = 0; i < elems_per_thread; i++) {
        q_reg[i] = float(q[simd_lid * elems_per_thread + i]) * scale;
    }

    float o_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint i = simd_gid; i < S; i += BN_DECODE) {
        float k_reg[8];
        for (uint j = 0; j < elems_per_thread; j++) {
            k_reg[j] = float(k[i * D + simd_lid * elems_per_thread + j]);
        }

        float score = 0.0f;
        for (uint j = 0; j < elems_per_thread; j++) {
            score += q_reg[j] * k_reg[j];
        }
        score = simd_sum(score);

        if (has_mask) {
            score += float(mask[i]);
        }

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < elems_per_thread; j++) {
            o_reg[j] = o_reg[j] * factor + exp_score * float(v[i * D + simd_lid * elems_per_thread + j]);
        }
    }

    threadgroup float tg_max[BN_DECODE];
    threadgroup float tg_sum[BN_DECODE];
    threadgroup float tg_out[BN_DECODE * BD_DECODE];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float loaded_max = (simd_lid < BN_DECODE) ? tg_max[simd_lid] : -INFINITY;
    float g_max = simd_max(loaded_max);
    float factor = fast::exp(max_score - g_max);
    float loaded_sum = (simd_lid < BN_DECODE)
        ? tg_sum[simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
        : 0.0f;
    float g_sum = simd_sum(loaded_sum);

    for (uint j = 0; j < elems_per_thread; j++) {
        tg_out[simd_lid * BN_DECODE + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = (simd_lid < BN_DECODE)
            ? tg_out[simd_gid * BN_DECODE + simd_lid] * fast::exp(tg_max[simd_lid] - g_max)
            : 0.0f;
        o_reg[j] = simd_sum(val);
        o_reg[j] = (g_sum > 0.0f) ? (o_reg[j] / g_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device bfloat* o = O + head_id * D;
        for (uint j = 0; j < elems_per_thread; j++) {
            o[simd_gid * elems_per_thread + j] = bfloat(o_reg[j]);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// MMA-based SDPA prefill kernel — simdgroup 8×8 matrix multiply for Q@K^T
// and P@V, matching the MLX steel attention architecture.
//
// Design: BQ=32, BK=16, BD=128, 4 simdgroups × 32 lanes = 128 threads
// Online softmax with exp2 (not exp) and simd_shuffle_xor reductions.
// Function constants: align_Q, align_K, is_causal, has_mask
// ---------------------------------------------------------------------------

/// MMA tile sizes — must match the Metal shader constants.
const MMA_BQ: usize = 32;
const MMA_BK: usize = 16;
const MMA_THREADS: u64 = 128;

pub const SDPA_MMA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ─── Tile sizes ────────────────────────────────────────────────────────────
constant constexpr int BQ = 32;    // Query block rows
constant constexpr int BK = 16;    // K/V block rows
constant constexpr int BD = 128;   // Head dimension (fixed for this kernel)
constant constexpr int WM = 4;     // Simdgroups along Q dimension
constant constexpr int WN = 1;     // Simdgroups along K dimension (always 1)
constant constexpr int kFragSize = 8;
constant constexpr int kNWarps = WM * WN;

// MMA tile dimensions:
//   TQ = BQ / (WM * 8) = 32 / (4*8) = 1 fragment per SG in Q dim
//   TK = BK / 8 = 2 fragments in K dim
//   TD = BD / 8 = 16 fragments in head dim
constant constexpr int TQ = BQ / (kNWarps * kFragSize);  // 1
constant constexpr int TK = BK / kFragSize;               // 2
constant constexpr int TD = BD / kFragSize;                // 16

// Shared memory padding to avoid bank conflicts (8 halfs = 16 bytes)
constant constexpr int padQ = 8;
constant constexpr int padKV = 8;
constant constexpr int LDQ = BD + padQ;     // Q: [BQ, BD+pad] row-major
constant constexpr int LDK = BK + padKV;    // K transposed: [BD, BK+pad]
constant constexpr int LDV = BD + padKV;    // V: [BK, BD+pad] row-major

// ─── Function constants ────────────────────────────────────────────────────
constant bool align_Q [[function_constant(200)]];   // N divisible by BQ
constant bool align_K [[function_constant(201)]];   // S divisible by BK
constant bool is_causal [[function_constant(300)]]; // Causal masking
constant bool has_mask [[function_constant(301)]];  // Explicit additive mask

// ─── Helper ops ────────────────────────────────────────────────────────────
struct MaxOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};

struct SumOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};

// Row-wise reduction across a simdgroup_matrix 8x8 fragment.
// Each thread holds 2 elements (kElemRows=1, kElemCols=2).
// The 2 elements belong to the same row, so we reduce cols first,
// then use simd_shuffle_xor to reduce across threads in the same row.
template <typename Op>
METAL_FUNC void frag_row_reduce(thread const float2& frag, thread float* out) {
    float thr = Op::apply(frag.x, frag.y);
    float xor1 = simd_shuffle_xor(thr, ushort(1));
    xor1 = Op::apply(thr, xor1);
    float xor8 = simd_shuffle_xor(xor1, ushort(8));
    xor8 = Op::apply(xor1, xor8);
    out[0] = Op::apply(out[0], xor8);
}

// Apply a scalar row-value to both elements of a fragment.
METAL_FUNC void frag_row_mul(thread float2& frag, float val) {
    frag.x *= val;
    frag.y *= val;
}

METAL_FUNC void frag_row_sub_exp2(thread float2& frag, float row_max) {
    frag.x = fast::exp2(frag.x - row_max);
    frag.y = fast::exp2(frag.y - row_max);
}

// ─── Wide load type for aligned 4×half vectorized reads ─────────────────────
struct alignas(8) ReadVec4 { half v[4]; };

// ─── Cooperative block loader (row-major, no transpose) ────────────────────
// Loads [BROWS, BCOLS] from device memory into threadgroup memory.
// dst_ld is the threadgroup leading dimension (includes padding).
// Uses ReadVec4 wide loads when n_reads is divisible by 4.
template <int BROWS, int BCOLS, int dst_ld, int tgp_size>
struct RowLoader {
    static constant constexpr int n_reads = (BCOLS * BROWS) / tgp_size;
    static constant constexpr int TCOLS = BCOLS / n_reads;
    static constant constexpr int TROWS = tgp_size / TCOLS;
    static constant constexpr int n_reads_v = n_reads / 4;  // vec4 groups

    const int src_ld;
    const short bi;
    const short bj;
    threadgroup half* dst;
    const device half* src;

    METAL_FUNC RowLoader(
        const device half* src_, int src_ld_,
        threadgroup half* dst_,
        ushort simd_group_id, ushort simd_lane_id)
        : src_ld(src_ld_),
          bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
          bj(n_reads * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
          dst(dst_ + bi * dst_ld + bj),
          src(src_ + bi * src_ld_ + bj) {}

    METAL_FUNC void load_unsafe() const {
        for (short i = 0; i < BROWS; i += TROWS) {
            #pragma clang loop unroll(full)
            for (short j = 0; j < n_reads_v; j++) {
                *((threadgroup ReadVec4*)(dst + i * dst_ld + j * 4)) =
                    *((const device ReadVec4*)(src + i * src_ld + j * 4));
            }
        }
    }

    METAL_FUNC void load_safe(short2 tile_dim) const {
        // tile_dim = (cols_valid, rows_valid)
        short2 adj = tile_dim - short2(bj, bi);
        if (adj.x <= 0 || adj.y <= 0) {
            for (short i = 0; i < BROWS; i += TROWS) {
                for (short j = 0; j < n_reads; j++) {
                    dst[i * dst_ld + j] = half(0);
                }
            }
            return;
        }
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                bool valid = (i < adj.y) && (j < adj.x);
                dst[i * dst_ld + j] = valid ? src[i * src_ld + j] : half(0);
            }
        }
    }

    METAL_FUNC void next() {
        src += BROWS * src_ld;
    }
};

// Transposed loader: loads [BROWS, BCOLS] from device (row-major, ld=src_ld)
// and stores transposed as [BCOLS, BROWS+pad] in threadgroup memory.
// dst layout: dst[col * dst_col_stride + row]
template <int BROWS, int BCOLS, int dst_row_stride, int dst_col_stride, int tgp_size>
struct TransposeLoader {
    static constant constexpr int n_reads = (BCOLS * BROWS) / tgp_size;
    static constant constexpr int TCOLS = BCOLS / n_reads;
    static constant constexpr int TROWS = tgp_size / TCOLS;

    const int src_ld;
    const short bi;
    const short bj;
    threadgroup half* dst;
    const device half* src;

    METAL_FUNC TransposeLoader(
        const device half* src_, int src_ld_,
        threadgroup half* dst_,
        ushort simd_group_id, ushort simd_lane_id)
        : src_ld(src_ld_),
          bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
          bj(n_reads * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
          dst(dst_ + bi * dst_row_stride + bj * dst_col_stride),
          src(src_ + bi * src_ld_ + bj) {}

    METAL_FUNC void load_unsafe() const {
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                dst[i * dst_row_stride + j * dst_col_stride] = src[i * src_ld + j];
            }
        }
    }

    METAL_FUNC void load_safe(short2 tile_dim) const {
        short2 adj = tile_dim - short2(bj, bi);
        if (adj.x <= 0 || adj.y <= 0) {
            for (short i = 0; i < BROWS; i += TROWS) {
                for (short j = 0; j < n_reads; j++) {
                    dst[i * dst_row_stride + j * dst_col_stride] = half(0);
                }
            }
            return;
        }
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                bool valid = (i < adj.y) && (j < adj.x);
                dst[i * dst_row_stride + j * dst_col_stride] =
                    valid ? src[i * src_ld + j] : half(0);
            }
        }
    }

    METAL_FUNC void next() {
        src += BROWS * src_ld;
    }
};

// ─── Main MMA attention kernel ─────────────────────────────────────────────
//
// Grid: (ceildiv(N, BQ), num_q_heads, 1)
// Threadgroup: (128, 1, 1)
//
// Buffers:
//   0: Q      [num_q_heads, N, D]
//   1: K      [num_kv_heads, S, D]
//   2: V      [num_kv_heads, S, D]
//   3: O      [num_q_heads, N, D]
//   4: mask   [N, S] or nullptr
//   5: N      (query sequence length)
//   6: S      (key/value sequence length)
//   7: D_val  (head dimension, must be 128)
//   8: gqa_factor (num_q_heads / num_kv_heads)
//   9: scale  (1/sqrt(D))
//  10: kv_stride_S
//  11: num_q_heads (for seq-major output indexing)

kernel void sdpa_prefill_mma_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O       [[buffer(3)]],
    device const half* mask_buf [[buffer(4)]],
    constant uint& N     [[buffer(5)]],
    constant uint& S     [[buffer(6)]],
    constant uint& D_val [[buffer(7)]],
    constant uint& gqa_factor [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& kv_stride_S [[buffer(10)]],
    constant uint& num_q_heads [[buffer(11)]],
    constant uint& v_head_stride [[buffer(12)]],
    constant uint& v_row_stride  [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    // tid.x = q_block index, tid.y = q_head index
    const uint q_block = tid.x;
    const uint q_head = tid.y;
    const uint kv_head = q_head / gqa_factor;

    const uint q_start = q_block * BQ;
    if (q_start >= N) return;

    // Pointer offsets for this head (use kv_stride_S for K, v_head_stride for V)
    const device half* Q_head = Q + q_head * N * BD;
    const device half* K_head = K + kv_head * kv_stride_S * BD;
    const device half* V_head = V + kv_head * v_head_stride;
    // Seq-major output: O[seq_pos * num_q_heads * BD + q_head * BD + d]
    const uint o_row_stride = num_q_heads * BD;

    // Pre-multiply scale by log2(e) for exp2-based softmax
    const float scale_log2e = scale * M_LOG2E_F;

    // ─── Threadgroup memory ────────────────────────────────────────────
    threadgroup half Q_smem[BQ * LDQ];          // [BQ, BD+pad]
    // KV_smem shared between K (transposed) and V (row-major)
    // K transposed: [BD, LDK] = [128, 24] — but we only need BK cols at a time
    // V row-major: [BK, LDV] = [16, 136]
    // Size needed: max(BD * LDK, BK * LDV) = max(128*24, 16*136) = max(3072, 2176) = 3072
    threadgroup half KV_smem[BD * LDK];  // large enough for both K^T and V tiles

    threadgroup half* Qs = Q_smem;
    threadgroup half* Ks = KV_smem;
    threadgroup half* Vs = KV_smem;

    // ─── Load Q tile ───────────────────────────────────────────────────
    // Q: [BQ, BD] row-major → Q_smem: [BQ, BD+pad]
    using QLoader = RowLoader<BQ, BD, LDQ, kNWarps * 32>;
    QLoader loader_q(Q_head + q_start * BD, BD, Qs, simd_gid, simd_lid);

    if (!align_Q && q_block == ((N - 1) / BQ)) {
        loader_q.load_safe(short2(BD, N - q_start));
    } else {
        loader_q.load_unsafe();
    }

    // ─── Prepare K and V loaders ───────────────────────────────────────
    // K is loaded transposed: [BK, BD] from device → [BD, BK+pad] in TG mem
    // So: BROWS=BK, BCOLS=BD, but stored transposed with dst_row_stride=1, dst_col_stride=LDK
    using KLoader = TransposeLoader<BK, BD, 1, LDK, kNWarps * 32>;
    KLoader loader_k(K_head, BD, Ks, simd_gid, simd_lid);

    // V is loaded row-major: [BK, BD] → [BK, BD+pad]
    // v_row_stride allows strided V access (e.g. from merged QKV buffer)
    using VLoader = RowLoader<BK, BD, LDV, kNWarps * 32>;
    VLoader loader_v(V_head, int(v_row_stride), Vs, simd_gid, simd_lid);

    // ─── MMA fragment setup ────────────────────────────────────────────
    // Each simdgroup owns TQ=1 rows of 8 query positions
    // Position within the BQ tile: row offset = simd_gid * 8
    const short tm = kFragSize * TQ * simd_gid;  // row offset in Q tile

    // simd_lane coordinate within 8x8 fragment
    const short qid = simd_lid / 4;
    const short fm = (qid & 4) + ((simd_lid / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;

    // simdgroup_load base addresses use simdgroup-level offsets only (no fm/fn)
    // — the hardware distributes elements across threads internally.

    // ─── Output and softmax accumulators ───────────────────────────────
    // O tile: TQ × TD = 1 × 16 fragments per SG
    float2 O_frags[TQ * TD];  // 16 fragments, 2 elems each = 32 floats
    for (int i = 0; i < TQ * TD; i++) O_frags[i] = float2(0.0f);

    // S tile: TQ × TK = 1 × 2 fragments (score)
    float2 S_frags[TQ * TK];

    // Running softmax state (1 row per TQ)
    float max_score[TQ];
    float sum_score[TQ];
    for (int i = 0; i < TQ; i++) {
        max_score[i] = -INFINITY;
        sum_score[i] = 0.0f;
    }

    // ─── Compute KV block limit ────────────────────────────────────────
    int NK = (int(S) + BK - 1) / BK;
    int kb_lim = NK;

    if (is_causal) {
        int q_max = int(q_start) + BQ;
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(NK, kb_lim);
    }

    const int NK_aligned = align_K ? NK : (NK - 1);

    // ─── Main loop over KV blocks ──────────────────────────────────────
    for (int kb = 0; kb < kb_lim; kb++) {
        // ── Load K^T tile ──────────────────────────────────────────────
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!align_K && kb == NK_aligned) {
            loader_k.load_safe(short2(BD, int(S) - kb * BK));
        } else {
            loader_k.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute S = Q @ K^T ───────────────────────────────────────
        // S_frags[tq][tk]: TQ=1 × TK=2
        for (int i = 0; i < TQ * TK; i++) S_frags[i] = float2(0.0f);

        // Iterate over head dimension in blocks of kFragSize
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load Q fragment: [8, 8] from Qs at row offset, col = dd*8
            simdgroup_matrix<half, 8, 8> Q_mat;
            // Q_smem layout: [BQ, LDQ], row-major
            // For MMA Q @ K^T: Q is [M, K] = [8, 8] loaded from [row, d]
            simdgroup_load(Q_mat, Qs + tm * LDQ + dd * kFragSize, LDQ);

            // Load K^T fragment: K is stored as [BD, LDK] (transposed)
            // For MMA: K^T is [K, N] = [8, TK*8]
            // Each TK fragment is [8, 8] at K_smem[dd*8][tk*8]
            #pragma clang loop unroll(full)
            for (short tk = 0; tk < TK; tk++) {
                simdgroup_matrix<half, 8, 8> K_mat;
                simdgroup_load(K_mat, Ks + dd * kFragSize * LDK + tk * kFragSize, LDK);

                simdgroup_matrix<float, 8, 8> S_mat;
                // Load current accumulator
                { thread auto& se = S_mat.thread_elements(); se[0] = S_frags[0 * TK + tk].x; se[1] = S_frags[0 * TK + tk].y; }
                simdgroup_multiply_accumulate(S_mat, Q_mat, K_mat, S_mat);
                { thread auto& se = S_mat.thread_elements(); S_frags[0 * TK + tk] = float2(se[0], se[1]); }
            }

            simdgroup_barrier(mem_flags::mem_none);
        }

        // ── Apply scale ────────────────────────────────────────────────
        for (int i = 0; i < TQ * TK; i++) {
            S_frags[i] *= scale_log2e;
        }

        // ── Mask out-of-bounds K positions ─────────────────────────────
        if (!align_K && kb == NK_aligned) {
            int kL_rem = int(S) - kb * BK;
            for (short tk = 0; tk < TK; tk++) {
                short col_pos = fn + tk * kFragSize;
                if (col_pos >= kL_rem) S_frags[0 * TK + tk] = float2(-INFINITY);
                else if (col_pos + 1 >= kL_rem) S_frags[0 * TK + tk].y = -INFINITY;
            }
        }

        // ── Causal masking ─────────────────────────────────────────────
        if (is_causal) {
            // Check if we're near the diagonal
            int kb_start = kb * BK;
            int q_end_pos = int(q_start) + tm + fm;  // row in Q
            for (short tk = 0; tk < TK; tk++) {
                short col_base = kb_start + fn + tk * kFragSize;
                if (col_base > q_end_pos) {
                    S_frags[0 * TK + tk] = float2(-INFINITY);
                } else if (col_base + 1 > q_end_pos) {
                    S_frags[0 * TK + tk].y = -INFINITY;
                }
            }
        }

        // ── Additive mask ──────────────────────────────────────────────
        if (has_mask) {
            int row_pos = int(q_start) + tm + fm;
            for (short tk = 0; tk < TK; tk++) {
                short col_base = kb * BK + fn + tk * kFragSize;
                float m0 = (row_pos < int(N) && col_base < int(S))
                    ? float(mask_buf[row_pos * int(S) + col_base]) * M_LOG2E_F : 0.0f;
                float m1 = (row_pos < int(N) && col_base + 1 < int(S))
                    ? float(mask_buf[row_pos * int(S) + col_base + 1]) * M_LOG2E_F : 0.0f;
                S_frags[0 * TK + tk].x += m0;
                S_frags[0 * TK + tk].y += m1;
            }
        }

        // ── Online softmax ─────────────────────────────────────────────
        // 1. Row max over S_frags
        float new_max[TQ];
        for (int i = 0; i < TQ; i++) new_max[i] = max_score[i];

        for (short tk = 0; tk < TK; tk++) {
            frag_row_reduce<MaxOp>(S_frags[0 * TK + tk], &new_max[0]);
        }

        // 2. exp2(S - max) in-place
        for (short tk = 0; tk < TK; tk++) {
            frag_row_sub_exp2(S_frags[0 * TK + tk], new_max[0]);
        }

        // 3. Correction factor for old accumulator
        float factor[TQ];
        for (int i = 0; i < TQ; i++) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
        }

        // 4. Row sum of exp scores
        float sum_tmp[TQ] = {0.0f};
        for (short tk = 0; tk < TK; tk++) {
            frag_row_reduce<SumOp>(S_frags[0 * TK + tk], &sum_tmp[0]);
        }
        for (int i = 0; i < TQ; i++) {
            sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
        }

        // 5. Rescale existing O accumulator
        for (int id = 0; id < TD; id++) {
            frag_row_mul(O_frags[0 * TD + id], factor[0]);
        }

        // ── Load V and compute O += S @ V ──────────────────────────────
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!align_K && kb == NK_aligned) {
            loader_v.load_safe(short2(BD, int(S) - kb * BK));
        } else {
            loader_v.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P @ V: S_frags[1, TK] @ V_smem[TK, TD] → O_frags[1, TD]
        for (short id = 0; id < TD; id++) {
            #pragma clang loop unroll(full)
            for (short tk = 0; tk < TK; tk++) {
                simdgroup_barrier(mem_flags::mem_none);

                // Load V fragment: V_smem[BK, BD+pad], row=k, col=d
                simdgroup_matrix<half, 8, 8> V_mat;
                simdgroup_load(V_mat, Vs + tk * kFragSize * LDV + id * kFragSize, LDV);

                // Convert S scores to half for MMA input — use half2 bulk conversion
                simdgroup_matrix<half, 8, 8> S_half;
                { float2 sv = S_frags[0 * TK + tk]; thread auto& sh = S_half.thread_elements(); half2 h2 = half2(sv); sh[0] = h2.x; sh[1] = h2.y; }

                simdgroup_matrix<float, 8, 8> O_mat;
                { thread auto& oe = O_mat.thread_elements(); oe[0] = O_frags[0 * TD + id].x; oe[1] = O_frags[0 * TD + id].y; }
                simdgroup_multiply_accumulate(O_mat, S_half, V_mat, O_mat);
                { thread auto& oe = O_mat.thread_elements(); O_frags[0 * TD + id] = float2(oe[0], oe[1]); }

                simdgroup_barrier(mem_flags::mem_none);
            }
        }

        // Advance K/V loaders
        loader_k.next();
        loader_v.next();
    }

    // ─── Final normalization: O /= sum_score ───────────────────────────
    for (int id = 0; id < TD; id++) {
        float inv_sum = (sum_score[0] > 0.0f) ? (1.0f / sum_score[0]) : 0.0f;
        O_frags[0 * TD + id] *= inv_sum;
    }

    // ─── Store output (seq-major) ─────────────────────────────────────
    // O layout: [N, num_q_heads * BD], seq-major
    // O[seq_pos * o_row_stride + q_head * BD + d]
    uint seq_pos = q_start + tm + fm;
    device half* O_row = O + seq_pos * o_row_stride + q_head * BD + fn;

    if (!align_Q && q_block == ((N - 1) / BQ)) {
        int qL_rem = int(N) - int(q_start);
        short row_in_tile = tm + fm;
        if (row_in_tile >= qL_rem) return;

        for (short id = 0; id < TD; id++) {
            short col = fn + id * kFragSize;
            if (col < BD) {
                O_row[id * kFragSize] = half(O_frags[0 * TD + id].x);
            }
            if (col + 1 < BD) {
                O_row[id * kFragSize + 1] = half(O_frags[0 * TD + id].y);
            }
        }
    } else {
        for (short id = 0; id < TD; id++) {
            O_row[id * kFragSize] = half(O_frags[0 * TD + id].x);
            O_row[id * kFragSize + 1] = half(O_frags[0 * TD + id].y);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// MMA SDPA kernel variant: BQ=32, BK=32, BD=128
// Halves the number of KV loop iterations vs BK=16 — benefits longer sequences.
// ---------------------------------------------------------------------------
const MMA_BK32: usize = 32;

pub const SDPA_MMA_BK32_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ─── Tile sizes ────────────────────────────────────────────────────────────
constant constexpr int BQ = 32;    // Query block rows
constant constexpr int BK = 32;    // K/V block rows (doubled vs BK=16 variant)
constant constexpr int BD = 128;   // Head dimension (fixed for this kernel)
constant constexpr int WM = 4;     // Simdgroups along Q dimension
constant constexpr int WN = 1;     // Simdgroups along K dimension (always 1)
constant constexpr int kFragSize = 8;
constant constexpr int kNWarps = WM * WN;

// MMA tile dimensions:
//   TQ = BQ / (WM * 8) = 32 / (4*8) = 1 fragment per SG in Q dim
//   TK = BK / 8 = 4 fragments in K dim (was 2 for BK=16)
//   TD = BD / 8 = 16 fragments in head dim
constant constexpr int TQ = BQ / (kNWarps * kFragSize);  // 1
constant constexpr int TK = BK / kFragSize;               // 4
constant constexpr int TD = BD / kFragSize;                // 16

// Shared memory padding to avoid bank conflicts (8 halfs = 16 bytes)
constant constexpr int padQ = 8;
constant constexpr int padKV = 8;
constant constexpr int LDQ = BD + padQ;     // Q: [BQ, BD+pad] row-major = 136
constant constexpr int LDK = BK + padKV;    // K transposed: [BD, BK+pad] = 40
constant constexpr int LDV = BD + padKV;    // V: [BK, BD+pad] row-major = 136

// ─── Function constants ────────────────────────────────────────────────────
constant bool align_Q [[function_constant(200)]];   // N divisible by BQ
constant bool align_K [[function_constant(201)]];   // S divisible by BK
constant bool is_causal [[function_constant(300)]]; // Causal masking
constant bool has_mask [[function_constant(301)]];  // Explicit additive mask

// ─── Helper ops ────────────────────────────────────────────────────────────
struct MaxOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};

struct SumOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};

// Row-wise reduction across a simdgroup_matrix 8x8 fragment.
template <typename Op>
METAL_FUNC void frag_row_reduce(thread const float2& frag, thread float* out) {
    float thr = Op::apply(frag.x, frag.y);
    float xor1 = simd_shuffle_xor(thr, ushort(1));
    xor1 = Op::apply(thr, xor1);
    float xor8 = simd_shuffle_xor(xor1, ushort(8));
    xor8 = Op::apply(xor1, xor8);
    out[0] = Op::apply(out[0], xor8);
}

METAL_FUNC void frag_row_mul(thread float2& frag, float val) {
    frag.x *= val;
    frag.y *= val;
}

METAL_FUNC void frag_row_sub_exp2(thread float2& frag, float row_max) {
    frag.x = fast::exp2(frag.x - row_max);
    frag.y = fast::exp2(frag.y - row_max);
}

// ─── Wide load type for aligned 4×half vectorized reads ─────────────────────
struct alignas(8) ReadVec4 { half v[4]; };

// ─── Cooperative block loader (row-major, no transpose) ────────────────────
// Uses ReadVec4 wide loads when n_reads is divisible by 4.
template <int BROWS, int BCOLS, int dst_ld, int tgp_size>
struct RowLoader {
    static constant constexpr int n_reads = (BCOLS * BROWS) / tgp_size;
    static constant constexpr int TCOLS = BCOLS / n_reads;
    static constant constexpr int TROWS = tgp_size / TCOLS;
    static constant constexpr int n_reads_v = n_reads / 4;  // vec4 groups

    const int src_ld;
    const short bi;
    const short bj;
    threadgroup half* dst;
    const device half* src;

    METAL_FUNC RowLoader(
        const device half* src_, int src_ld_,
        threadgroup half* dst_,
        ushort simd_group_id, ushort simd_lane_id)
        : src_ld(src_ld_),
          bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
          bj(n_reads * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
          dst(dst_ + bi * dst_ld + bj),
          src(src_ + bi * src_ld_ + bj) {}

    METAL_FUNC void load_unsafe() const {
        for (short i = 0; i < BROWS; i += TROWS) {
            #pragma clang loop unroll(full)
            for (short j = 0; j < n_reads_v; j++) {
                *((threadgroup ReadVec4*)(dst + i * dst_ld + j * 4)) =
                    *((const device ReadVec4*)(src + i * src_ld + j * 4));
            }
        }
    }

    METAL_FUNC void load_safe(short2 tile_dim) const {
        short2 adj = tile_dim - short2(bj, bi);
        if (adj.x <= 0 || adj.y <= 0) {
            for (short i = 0; i < BROWS; i += TROWS) {
                for (short j = 0; j < n_reads; j++) {
                    dst[i * dst_ld + j] = half(0);
                }
            }
            return;
        }
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                bool valid = (i < adj.y) && (j < adj.x);
                dst[i * dst_ld + j] = valid ? src[i * src_ld + j] : half(0);
            }
        }
    }

    METAL_FUNC void next() {
        src += BROWS * src_ld;
    }
};

// Transposed loader: loads [BROWS, BCOLS] from device (row-major, ld=src_ld)
// and stores transposed as [BCOLS, BROWS+pad] in threadgroup memory.
template <int BROWS, int BCOLS, int dst_row_stride, int dst_col_stride, int tgp_size>
struct TransposeLoader {
    static constant constexpr int n_reads = (BCOLS * BROWS) / tgp_size;
    static constant constexpr int TCOLS = BCOLS / n_reads;
    static constant constexpr int TROWS = tgp_size / TCOLS;

    const int src_ld;
    const short bi;
    const short bj;
    threadgroup half* dst;
    const device half* src;

    METAL_FUNC TransposeLoader(
        const device half* src_, int src_ld_,
        threadgroup half* dst_,
        ushort simd_group_id, ushort simd_lane_id)
        : src_ld(src_ld_),
          bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
          bj(n_reads * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
          dst(dst_ + bi * dst_row_stride + bj * dst_col_stride),
          src(src_ + bi * src_ld_ + bj) {}

    METAL_FUNC void load_unsafe() const {
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                dst[i * dst_row_stride + j * dst_col_stride] = src[i * src_ld + j];
            }
        }
    }

    METAL_FUNC void load_safe(short2 tile_dim) const {
        short2 adj = tile_dim - short2(bj, bi);
        if (adj.x <= 0 || adj.y <= 0) {
            for (short i = 0; i < BROWS; i += TROWS) {
                for (short j = 0; j < n_reads; j++) {
                    dst[i * dst_row_stride + j * dst_col_stride] = half(0);
                }
            }
            return;
        }
        for (short i = 0; i < BROWS; i += TROWS) {
            for (short j = 0; j < n_reads; j++) {
                bool valid = (i < adj.y) && (j < adj.x);
                dst[i * dst_row_stride + j * dst_col_stride] =
                    valid ? src[i * src_ld + j] : half(0);
            }
        }
    }

    METAL_FUNC void next() {
        src += BROWS * src_ld;
    }
};

// ─── Main MMA attention kernel (BK=32) ─────────────────────────────────────
//
// Grid: (ceildiv(N, BQ), num_q_heads, 1)
// Threadgroup: (128, 1, 1)
//
// Buffers: same as BK=16 variant
//   0: Q      [num_q_heads, N, D]
//   1: K      [num_kv_heads, S, D]
//   2: V      [num_kv_heads, S, D]
//   3: O      [num_q_heads, N, D]
//   4: mask   [N, S] or nullptr
//   5: N      (query sequence length)
//   6: S      (key/value sequence length)
//   7: D_val  (head dimension, must be 128)
//   8: gqa_factor (num_q_heads / num_kv_heads)
//   9: scale  (1/sqrt(D))

[[max_total_threads_per_threadgroup(128)]]
kernel void sdpa_prefill_mma_bk32_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O       [[buffer(3)]],
    device const half* mask_buf [[buffer(4)]],
    constant uint& N     [[buffer(5)]],
    constant uint& S     [[buffer(6)]],
    constant uint& D_val [[buffer(7)]],
    constant uint& gqa_factor [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& kv_stride_S [[buffer(10)]],
    constant uint& num_q_heads [[buffer(11)]],
    constant uint& v_head_stride [[buffer(12)]],
    constant uint& v_row_stride  [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    const uint q_block = tid.x;
    const uint q_head = tid.y;
    const uint kv_head = q_head / gqa_factor;

    const uint q_start = q_block * BQ;
    if (q_start >= N) return;

    const device half* Q_head = Q + q_head * N * BD;
    const device half* K_head = K + kv_head * kv_stride_S * BD;
    const device half* V_head = V + kv_head * v_head_stride;
    // Seq-major output: O[seq_pos * num_q_heads * BD + q_head * BD + d]
    const uint o_row_stride = num_q_heads * BD;

    const float scale_log2e = scale * M_LOG2E_F;

    // ─── Threadgroup memory ────────────────────────────────────────────
    threadgroup half Q_smem[BQ * LDQ];          // [32, 136] = 4352
    // KV_smem shared between K (transposed) and V (row-major)
    // K transposed: [BD, LDK] = [128, 40] = 5120
    // V row-major: [BK, LDV] = [32, 136] = 4352
    // Size needed: max(5120, 4352) = 5120
    threadgroup half KV_smem[BD * LDK];  // 5120

    threadgroup half* Qs = Q_smem;
    threadgroup half* Ks = KV_smem;
    threadgroup half* Vs = KV_smem;

    // ─── Load Q tile ───────────────────────────────────────────────────
    using QLoader = RowLoader<BQ, BD, LDQ, kNWarps * 32>;
    QLoader loader_q(Q_head + q_start * BD, BD, Qs, simd_gid, simd_lid);

    if (!align_Q && q_block == ((N - 1) / BQ)) {
        loader_q.load_safe(short2(BD, N - q_start));
    } else {
        loader_q.load_unsafe();
    }

    // ─── Prepare K and V loaders ───────────────────────────────────────
    using KLoader = TransposeLoader<BK, BD, 1, LDK, kNWarps * 32>;
    KLoader loader_k(K_head, BD, Ks, simd_gid, simd_lid);

    // v_row_stride allows strided V access (e.g. from merged QKV buffer)
    using VLoader = RowLoader<BK, BD, LDV, kNWarps * 32>;
    VLoader loader_v(V_head, int(v_row_stride), Vs, simd_gid, simd_lid);

    // ─── MMA fragment setup ────────────────────────────────────────────
    const short tm = kFragSize * TQ * simd_gid;
    const short qid = simd_lid / 4;
    const short fm = (qid & 4) + ((simd_lid / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;

    // ─── Output and softmax accumulators ───────────────────────────────
    float2 O_frags[TQ * TD];  // 1 × 16 = 16 fragments
    for (int i = 0; i < TQ * TD; i++) O_frags[i] = float2(0.0f);

    // S tile: TQ × TK = 1 × 4 fragments (score) — doubled vs BK=16
    float2 S_frags[TQ * TK];

    float max_score[TQ];
    float sum_score[TQ];
    for (int i = 0; i < TQ; i++) {
        max_score[i] = -INFINITY;
        sum_score[i] = 0.0f;
    }

    // ─── Compute KV block limit ────────────────────────────────────────
    int NK = (int(S) + BK - 1) / BK;
    int kb_lim = NK;

    if (is_causal) {
        int q_max = int(q_start) + BQ;
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(NK, kb_lim);
    }

    const int NK_aligned = align_K ? NK : (NK - 1);

    // ─── Main loop over KV blocks ──────────────────────────────────────
    for (int kb = 0; kb < kb_lim; kb++) {
        // ── Load K^T tile ──────────────────────────────────────────────
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!align_K && kb == NK_aligned) {
            loader_k.load_safe(short2(BD, int(S) - kb * BK));
        } else {
            loader_k.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute S = Q @ K^T ───────────────────────────────────────
        for (int i = 0; i < TQ * TK; i++) S_frags[i] = float2(0.0f);

        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);

            simdgroup_matrix<half, 8, 8> Q_mat;
            simdgroup_load(Q_mat, Qs + tm * LDQ + dd * kFragSize, LDQ);

            #pragma clang loop unroll(full)
            for (short tk = 0; tk < TK; tk++) {
                simdgroup_matrix<half, 8, 8> K_mat;
                simdgroup_load(K_mat, Ks + dd * kFragSize * LDK + tk * kFragSize, LDK);

                simdgroup_matrix<float, 8, 8> S_mat;
                { thread auto& se = S_mat.thread_elements(); se[0] = S_frags[0 * TK + tk].x; se[1] = S_frags[0 * TK + tk].y; }
                simdgroup_multiply_accumulate(S_mat, Q_mat, K_mat, S_mat);
                { thread auto& se = S_mat.thread_elements(); S_frags[0 * TK + tk] = float2(se[0], se[1]); }
            }

            simdgroup_barrier(mem_flags::mem_none);
        }

        // ── Apply scale ────────────────────────────────────────────────
        for (int i = 0; i < TQ * TK; i++) {
            S_frags[i] *= scale_log2e;
        }

        // ── Mask out-of-bounds K positions ─────────────────────────────
        if (!align_K && kb == NK_aligned) {
            int kL_rem = int(S) - kb * BK;
            for (short tk = 0; tk < TK; tk++) {
                short col_pos = fn + tk * kFragSize;
                if (col_pos >= kL_rem) S_frags[0 * TK + tk] = float2(-INFINITY);
                else if (col_pos + 1 >= kL_rem) S_frags[0 * TK + tk].y = -INFINITY;
            }
        }

        // ── Causal masking ─────────────────────────────────────────────
        if (is_causal) {
            int kb_start = kb * BK;
            int q_end_pos = int(q_start) + tm + fm;
            for (short tk = 0; tk < TK; tk++) {
                short col_base = kb_start + fn + tk * kFragSize;
                if (col_base > q_end_pos) {
                    S_frags[0 * TK + tk] = float2(-INFINITY);
                } else if (col_base + 1 > q_end_pos) {
                    S_frags[0 * TK + tk].y = -INFINITY;
                }
            }
        }

        // ── Additive mask ──────────────────────────────────────────────
        if (has_mask) {
            int row_pos = int(q_start) + tm + fm;
            for (short tk = 0; tk < TK; tk++) {
                short col_base = kb * BK + fn + tk * kFragSize;
                float m0 = (row_pos < int(N) && col_base < int(S))
                    ? float(mask_buf[row_pos * int(S) + col_base]) * M_LOG2E_F : 0.0f;
                float m1 = (row_pos < int(N) && col_base + 1 < int(S))
                    ? float(mask_buf[row_pos * int(S) + col_base + 1]) * M_LOG2E_F : 0.0f;
                S_frags[0 * TK + tk].x += m0;
                S_frags[0 * TK + tk].y += m1;
            }
        }

        // ── Online softmax ─────────────────────────────────────────────
        float new_max[TQ];
        for (int i = 0; i < TQ; i++) new_max[i] = max_score[i];

        for (short tk = 0; tk < TK; tk++) {
            frag_row_reduce<MaxOp>(S_frags[0 * TK + tk], &new_max[0]);
        }

        for (short tk = 0; tk < TK; tk++) {
            frag_row_sub_exp2(S_frags[0 * TK + tk], new_max[0]);
        }

        float factor[TQ];
        for (int i = 0; i < TQ; i++) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
        }

        float sum_tmp[TQ] = {0.0f};
        for (short tk = 0; tk < TK; tk++) {
            frag_row_reduce<SumOp>(S_frags[0 * TK + tk], &sum_tmp[0]);
        }
        for (int i = 0; i < TQ; i++) {
            sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
        }

        for (int id = 0; id < TD; id++) {
            frag_row_mul(O_frags[0 * TD + id], factor[0]);
        }

        // ── Load V and compute O += S @ V ──────────────────────────────
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!align_K && kb == NK_aligned) {
            loader_v.load_safe(short2(BD, int(S) - kb * BK));
        } else {
            loader_v.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P @ V: S_frags[1, TK] @ V_smem[TK, TD] → O_frags[1, TD]
        for (short id = 0; id < TD; id++) {
            #pragma clang loop unroll(full)
            for (short tk = 0; tk < TK; tk++) {
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_matrix<half, 8, 8> V_mat;
                simdgroup_load(V_mat, Vs + tk * kFragSize * LDV + id * kFragSize, LDV);

                // Convert S scores to half for MMA input — use half2 bulk conversion
                simdgroup_matrix<half, 8, 8> S_half;
                { float2 sv = S_frags[0 * TK + tk]; thread auto& sh = S_half.thread_elements(); half2 h2 = half2(sv); sh[0] = h2.x; sh[1] = h2.y; }

                simdgroup_matrix<float, 8, 8> O_mat;
                { thread auto& oe = O_mat.thread_elements(); oe[0] = O_frags[0 * TD + id].x; oe[1] = O_frags[0 * TD + id].y; }
                simdgroup_multiply_accumulate(O_mat, S_half, V_mat, O_mat);
                { thread auto& oe = O_mat.thread_elements(); O_frags[0 * TD + id] = float2(oe[0], oe[1]); }

                simdgroup_barrier(mem_flags::mem_none);
            }
        }

        loader_k.next();
        loader_v.next();
    }

    // ─── Final normalization: O /= sum_score ───────────────────────────
    for (int id = 0; id < TD; id++) {
        float inv_sum = (sum_score[0] > 0.0f) ? (1.0f / sum_score[0]) : 0.0f;
        O_frags[0 * TD + id] *= inv_sum;
    }

    // ─── Store output (seq-major) ─────────────────────────────────────
    // O layout: [N, num_q_heads * BD], seq-major
    uint seq_pos = q_start + tm + fm;
    device half* O_row = O + seq_pos * o_row_stride + q_head * BD + fn;

    if (!align_Q && q_block == ((N - 1) / BQ)) {
        int qL_rem = int(N) - int(q_start);
        short row_in_tile = tm + fm;
        if (row_in_tile >= qL_rem) return;

        for (short id = 0; id < TD; id++) {
            short col = fn + id * kFragSize;
            if (col < BD) {
                O_row[id * kFragSize] = half(O_frags[0 * TD + id].x);
            }
            if (col + 1 < BD) {
                O_row[id * kFragSize + 1] = half(O_frags[0 * TD + id].y);
            }
        }
    } else {
        for (short id = 0; id < TD; id++) {
            O_row[id * kFragSize] = half(O_frags[0 * TD + id].x);
            O_row[id * kFragSize + 1] = half(O_frags[0 * TD + id].y);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// NAX SDPA kernel: BQ=64, BK=32, BD=128, 4 SGs, 128 threads
// Uses MetalPerformancePrimitives matmul2d (16×16 NAX fragments).
// NO threadgroup memory — all loads go directly from device to registers.
// ---------------------------------------------------------------------------
const NAX_BQ: usize = 64;
const NAX_BK: usize = 32;
const NAX_THREADS: u64 = 128;

pub const SDPA_NAX_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// ─── Tile sizes ────────────────────────────────────────────────────────────
constant constexpr int BQ  = 64;   // Query block rows
constant constexpr int BK  = 32;   // K/V block rows
constant constexpr int BD  = 128;  // Head dimension (fixed)
constant constexpr int WM  = 4;    // Simdgroups along Q dimension
constant constexpr int WN  = 1;    // Simdgroups along K dimension

// NAX subtile dimensions (16×16 fragments):
//   UQ  = 16  (Q subtile M-dim per SG)
//   UK  = 16  (P@V K-dim fragment)
//   UD  = 32  (output D-dim subtile = 1×2 fragments)
//   UKs = 32  (S subtile K-dim = 1×2 fragments)
//   UDs = 16  (Q@K^T D-dim fragment)
constant constexpr int UQ  = 16;
constant constexpr int UD  = 32;
constant constexpr int UK  = 16;
constant constexpr int UKs = 32;
constant constexpr int UDs = 16;

// Derived tile counts:
constant constexpr int TQ  = BQ / (WM * UQ);    // 64 / (4*16) = 1
constant constexpr int TD  = BD / UD;            // 128 / 32 = 4
constant constexpr int TK  = BK / UK;            // 32 / 16 = 2  (P@V subtiles)
constant constexpr int TKs = BK / UKs;           // 32 / 32 = 1  (S subtile)
constant constexpr int TDs = BD / UDs;           // 128 / 16 = 8  (D-dim chunks for Q@K^T)

// ─── Function constants ────────────────────────────────────────────────────
constant bool align_Q [[function_constant(200)]];   // seq_len divisible by BQ
constant bool align_K [[function_constant(201)]];   // kv_len divisible by BK
constant bool is_causal [[function_constant(300)]]; // Causal masking

// ─── NAX fragment coordinate helpers ────────────────────────────────────────
// Each lane in a 16×16 fragment holds 8 elements: 2 rows × 4 cols
// Row indices: fm, fm+8
// Col indices: fn, fn+1, fn+2, fn+3
//
// Element layout per lane (8 elements):
//   [0..3] = row fm,   cols fn..fn+3
//   [4..7] = row fm+8, cols fn..fn+3

inline short nax_fm(uint slid) {
    short qid = short(slid) >> 2;
    return ((qid & 4) | ((short(slid) >> 1) & 3));
}

inline short nax_fn(uint slid) {
    short qid = short(slid) >> 2;
    return ((qid & 2) | (short(slid) & 1)) * 4;
}

// ─── Row reduction for a 16×16 fragment (8 elements/lane) ──────────────────
// Reduces across 4 cols within lane, then across lanes sharing the same row.
// ri = 0 → row fm, ri = 1 → row fm+8.
// Returns the same value in all lanes that share the same row.

template <typename Op>
METAL_FUNC float nax_frag_row_reduce(thread const float* frag, int ri) {
    // Reduce 4 cols within this lane
    float v = Op::apply(Op::apply(frag[ri*4+0], frag[ri*4+1]),
                        Op::apply(frag[ri*4+2], frag[ri*4+3]));
    // Reduce across quads (lanes with same row but different fn)
    float xor1 = simd_shuffle_xor(v, ushort(1));
    v = Op::apply(v, xor1);
    // Reduce across row groups (lanes offset by 8 share fm vs fm+8)
    float xor8 = simd_shuffle_xor(v, ushort(8));
    v = Op::apply(v, xor8);
    return v;
}

// Row reduction for a 16×32 tile (two 16×16 sub-fragments, 16 elements/lane).
// frag0[8] = first 16×16 (cols 0..15), frag1[8] = second 16×16 (cols 16..31).
template <typename Op>
METAL_FUNC float nax_wide_row_reduce(thread const float* frag0,
                                      thread const float* frag1, int ri) {
    float v0 = Op::apply(Op::apply(frag0[ri*4+0], frag0[ri*4+1]),
                         Op::apply(frag0[ri*4+2], frag0[ri*4+3]));
    float v1 = Op::apply(Op::apply(frag1[ri*4+0], frag1[ri*4+1]),
                         Op::apply(frag1[ri*4+2], frag1[ri*4+3]));
    float v = Op::apply(v0, v1);
    float xor1 = simd_shuffle_xor(v, ushort(1));
    v = Op::apply(v, xor1);
    float xor8 = simd_shuffle_xor(v, ushort(8));
    v = Op::apply(v, xor8);
    return v;
}

struct MaxOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};

struct SumOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};

// ─── Device memory loader helpers ──────────────────────────────────────────
// Load a 16×16 block from device memory into a NAX fragment register array.
// src points to the top-left corner, src_ld is the leading dimension.
// The fragment stores 8 elements per lane: rows [fm, fm+8], cols [fn..fn+3].

METAL_FUNC void load_frag_16x16(thread half* frag,
                                  const device half* src, int src_ld,
                                  short fm, short fn,
                                  int valid_rows, int valid_cols) {
    // Row fm
    for (short c = 0; c < 4; c++) {
        bool ok = (fm < valid_rows) && (fn + c < valid_cols);
        frag[c] = ok ? src[fm * src_ld + fn + c] : half(0);
    }
    // Row fm+8
    for (short c = 0; c < 4; c++) {
        bool ok = (fm + 8 < valid_rows) && (fn + c < valid_cols);
        frag[4 + c] = ok ? src[(fm + 8) * src_ld + fn + c] : half(0);
    }
}

METAL_FUNC void load_frag_16x16_unsafe(thread half* frag,
                                         const device half* src, int src_ld,
                                         short fm, short fn) {
    for (short c = 0; c < 4; c++) {
        frag[c] = src[fm * src_ld + fn + c];
    }
    for (short c = 0; c < 4; c++) {
        frag[4 + c] = src[(fm + 8) * src_ld + fn + c];
    }
}

// Store a 16×16 block from float accumulator to device half memory.
METAL_FUNC void store_frag_16x16(device half* dst, int dst_ld,
                                   thread const float* frag,
                                   short fm, short fn,
                                   int valid_rows, int valid_cols) {
    for (short c = 0; c < 4; c++) {
        if (fm < valid_rows && fn + c < valid_cols) {
            dst[fm * dst_ld + fn + c] = half(frag[c]);
        }
    }
    for (short c = 0; c < 4; c++) {
        if (fm + 8 < valid_rows && fn + c < valid_cols) {
            dst[(fm + 8) * dst_ld + fn + c] = half(frag[4 + c]);
        }
    }
}

METAL_FUNC void store_frag_16x16_unsafe(device half* dst, int dst_ld,
                                          thread const float* frag,
                                          short fm, short fn) {
    for (short c = 0; c < 4; c++) {
        dst[fm * dst_ld + fn + c] = half(frag[c]);
    }
    for (short c = 0; c < 4; c++) {
        dst[(fm + 8) * dst_ld + fn + c] = half(frag[4 + c]);
    }
}

// ─── NAX matmul2d descriptors ──────────────────────────────────────────────
// Q@K^T: MMA(16, 32, 16, trans_a=false, trans_b=true)
// The output S is 16×32 = one fragment with 16 elements/lane.
// P@V:   MMA(16, 32, 16, trans_a=false, trans_b=false)
// Left input 16×16 (8 elems), right input 16×32 (16 elems), output 16×32 (16 elems).

// ─── Main NAX SDPA prefill kernel ──────────────────────────────────────────
//
// Grid: (ceildiv(seq_len, BQ), num_q_heads, batch_size)
// Threadgroup: (128, 1, 1) — 4 simdgroups × 32 lanes
//
// Buffers:
//   0: Q      [batch, q_heads, seq_len, head_dim]
//   1: K      [batch, kv_heads, kv_len, head_dim]
//   2: V      [batch, kv_heads, kv_len, head_dim]
//   3: O      [batch, q_heads, seq_len, head_dim]
//   4: seq_len
//   5: kv_len
//   6: head_dim (must be 128)
//   7: gqa_factor
//   8: scale
//   9: kv_stride_S

kernel void sdpa_prefill_nax_f16(
    device const half*  Q      [[buffer(0)]],
    device const half*  K      [[buffer(1)]],
    device const half*  V      [[buffer(2)]],
    device half*        O      [[buffer(3)]],
    constant uint& N           [[buffer(4)]],
    constant uint& S           [[buffer(5)]],
    constant uint& D_val       [[buffer(6)]],
    constant uint& gqa_factor  [[buffer(7)]],
    constant float& scale      [[buffer(8)]],
    constant uint& kv_stride_S [[buffer(9)]],
    constant uint& num_q_heads [[buffer(10)]],
    constant uint& v_head_stride [[buffer(11)]],
    constant uint& v_row_stride  [[buffer(12)]],
    uint3 tid    [[threadgroup_position_in_grid]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  slid   [[thread_index_in_simdgroup]])
{
    // tid.x = q_block index, tid.y = q_head index
    const uint q_block = tid.x;
    const uint q_head  = tid.y;
    const uint kv_head = q_head / gqa_factor;

    const uint q_start = q_block * BQ;
    if (q_start >= N) return;

    // Head pointers (use kv_stride_S for K, v_head_stride/v_row_stride for V)
    const device half* Q_head = Q + q_head  * N * BD;
    const device half* K_head = K + kv_head * kv_stride_S * BD;
    const device half* V_head = V + kv_head * v_head_stride;
    // Seq-major output: O[seq_pos * num_q_heads * BD + q_head * BD + d]
    const uint o_row_stride = num_q_heads * BD;

    const float scale2 = scale * M_LOG2E_F;

    // SG-local row offset: each SG handles UQ=16 rows
    const short tm = short(UQ * TQ * sgid);  // 0, 16, 32, 48

    // Per-lane fragment coordinates within a 16×16 fragment
    const short fm = nax_fm(slid);
    const short fn = nax_fn(slid);

    // MPP matmul2d descriptors (must be inside function, not program scope)
    // FM=16, FN=32, FK=16 — at least one dim must be 32 for cooperative tensors
    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // ─── Output accumulators: TD=4 subtiles of 16×32 (16 elems/lane) ────
    // O_acc[id][0..7] = first 16 D-cols, O_acc[id][8..15] = next 16 D-cols
    float O_acc[TD][16];
    for (int id = 0; id < TD; id++)
        for (int e = 0; e < 16; e++)
            O_acc[id][e] = 0.0f;

    // ─── Softmax state: per-row (2 rows per lane: fm and fm+8) ──────────
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};

    // ─── KV block loop ──────────────────────────────────────────────────
    int NK = (int(S) + BK - 1) / BK;
    int kb_lim = NK;

    if (is_causal) {
        int q_max = int(q_start) + tm + UQ;  // max Q row for this SG
        kb_lim = (q_max + BK - 1) / BK;
        kb_lim = min(NK, kb_lim);
    }

    const int NK_aligned = align_K ? NK : (NK - 1);
    const bool is_last_q_block = !align_Q && (q_block == ((N - 1) / BQ));
    const int q_rows_valid = is_last_q_block ? int(N) - int(q_start) : BQ;

    for (int kb = 0; kb < kb_lim; kb++) {
        const int kv_start = kb * BK;
        const bool is_last_kv = (!align_K && kb == NK_aligned);
        const int kv_rows_valid = is_last_kv ? (int(S) - kv_start) : BK;

        // ── Step 1: Compute S = Q @ K^T ─────────────────────────────────
        // S is 16×32 = one MMA(16,32,16) fragment per SG (16 elems/lane).
        // S_frag[0..7] = first 16 K-cols, S_frag[8..15] = next 16 K-cols
        float S_frag[16];
        for (int e = 0; e < 16; e++)
            S_frag[e] = 0.0f;

        // Iterate over D in chunks of UDs=16
        for (int dd = 0; dd < TDs; dd++) {
            // Load Q fragment: 16×16 from Q_head[(q_start+tm).., dd*16..]
            half Q_frag[8];
            const device half* Q_ptr = Q_head + (q_start + tm) * BD + dd * UDs;
            int q_valid = min(int(UQ), q_rows_valid - int(tm));
            if (q_valid <= 0) {
                for (int e = 0; e < 8; e++) Q_frag[e] = half(0);
            } else if (is_last_q_block && q_valid < UQ) {
                load_frag_16x16(Q_frag, Q_ptr, BD, fm, fn, q_valid, UDs);
            } else {
                load_frag_16x16_unsafe(Q_frag, Q_ptr, BD, fm, fn);
            }

            // Load K fragment as 32×16: two 16×16 stacked vertically.
            // K is [kv_len, BD], trans_b=true → MMA transposes internally.
            // K_frag[0..7] = K[kv_start..kv_start+16, dd*16..dd*16+16]
            // K_frag[8..15] = K[kv_start+16..kv_start+32, dd*16..dd*16+16]
            half K_frag[16];
            {
                const device half* K_ptr0 = K_head + kv_start * BD + dd * UDs;
                const device half* K_ptr1 = K_head + (kv_start + 16) * BD + dd * UDs;
                int k_valid0 = min(16, kv_rows_valid);
                int k_valid1 = min(16, kv_rows_valid - 16);

                // First 16 rows
                if (k_valid0 <= 0) {
                    for (int e = 0; e < 8; e++) K_frag[e] = half(0);
                } else if (is_last_kv && k_valid0 < 16) {
                    load_frag_16x16(K_frag, K_ptr0, BD, fm, fn, k_valid0, UDs);
                } else {
                    load_frag_16x16_unsafe(K_frag, K_ptr0, BD, fm, fn);
                }

                // Next 16 rows
                if (k_valid1 <= 0) {
                    for (int e = 0; e < 8; e++) K_frag[8 + e] = half(0);
                } else if (is_last_kv && k_valid1 < 16) {
                    load_frag_16x16(K_frag + 8, K_ptr1, BD, fm, fn, k_valid1, UDs);
                } else {
                    load_frag_16x16_unsafe(K_frag + 8, K_ptr1, BD, fm, fn);
                }
            }

            // MMA(16,32,16): S_frag += Q @ K^T
            // ct_a: 8 elems (FM=16, FK=16), ct_b: 16 elems (FN=32, FK=16), ct_c: 16 elems
            {
                mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> qk_op;
                auto ct_a = qk_op.template get_left_input_cooperative_tensor<half, half, float>();
                auto ct_b = qk_op.template get_right_input_cooperative_tensor<half, half, float>();
                auto ct_c = qk_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
                for (int e = 0; e < 8; e++) ct_a[e] = Q_frag[e];
                for (int e = 0; e < 16; e++) ct_b[e] = K_frag[e];
                for (int e = 0; e < 16; e++) ct_c[e] = S_frag[e];
                qk_op.run(ct_a, ct_b, ct_c);
                for (int e = 0; e < 16; e++) S_frag[e] = ct_c[e];
            }
        }

        // ── Step 2: Scale S by scale * log2(e) ─────────────────────────
        for (int e = 0; e < 16; e++)
            S_frag[e] *= scale2;

        // ── Step 3: Mask out-of-bounds K positions ──────────────────────
        if (is_last_kv) {
            // S_frag[0..7] covers K cols 0..15, S_frag[8..15] covers K cols 16..31
            // Sub-frag sf: actual K col = sf*16 + fn + ci
            for (int sf = 0; sf < 2; sf++) {
                for (int ri = 0; ri < 2; ri++) {
                    for (int ci = 0; ci < 4; ci++) {
                        int col = sf * 16 + fn + ci;
                        if (col >= kv_rows_valid) {
                            S_frag[sf * 8 + ri * 4 + ci] = -INFINITY;
                        }
                    }
                }
            }
        }

        // ── Step 3b: Causal masking ─────────────────────────────────────
        if (is_causal) {
            for (int sf = 0; sf < 2; sf++) {
                for (int ri = 0; ri < 2; ri++) {
                    int q_row = int(q_start) + tm + fm + ri * 8;
                    for (int ci = 0; ci < 4; ci++) {
                        int k_col = kv_start + sf * 16 + fn + ci;
                        if (k_col > q_row) {
                            S_frag[sf * 8 + ri * 4 + ci] = -INFINITY;
                        }
                    }
                }
            }
        }

        // ── Step 3c: Mask out-of-bounds Q rows ──────────────────────────
        if (is_last_q_block) {
            int q_valid_local = q_rows_valid - int(tm);
            for (int sf = 0; sf < 2; sf++) {
                for (int ri = 0; ri < 2; ri++) {
                    int row = fm + ri * 8;
                    if (row >= q_valid_local) {
                        for (int ci = 0; ci < 4; ci++)
                            S_frag[sf * 8 + ri * 4 + ci] = -INFINITY;
                    }
                }
            }
        }

        // ── Step 4: Online softmax ──────────────────────────────────────
        // 4a. Row max across the 16×32 S tile
        // S_frag[0..7] = first 16 cols sub-frag, S_frag[8..15] = second
        float new_max[2];
        for (int ri = 0; ri < 2; ri++) {
            new_max[ri] = nax_wide_row_reduce<MaxOp>(S_frag, S_frag + 8, ri);
            new_max[ri] = max(new_max[ri], row_max[ri]);
        }

        // 4b. P = exp2(S - new_max)  (in-place)
        for (int sf = 0; sf < 2; sf++) {
            for (int ri = 0; ri < 2; ri++) {
                for (int ci = 0; ci < 4; ci++) {
                    S_frag[sf * 8 + ri * 4 + ci] = fast::exp2(S_frag[sf * 8 + ri * 4 + ci] - new_max[ri]);
                }
            }
        }

        // 4c. Correction factor for old accumulator
        float factor[2];
        for (int ri = 0; ri < 2; ri++) {
            factor[ri] = fast::exp2(row_max[ri] - new_max[ri]);
            row_max[ri] = new_max[ri];
        }

        // 4d. Row sum of exp scores
        float new_sum[2];
        for (int ri = 0; ri < 2; ri++) {
            new_sum[ri] = nax_wide_row_reduce<SumOp>(S_frag, S_frag + 8, ri);
            row_sum[ri] = row_sum[ri] * factor[ri] + new_sum[ri];
        }

        // 4e. Rescale existing O accumulator
        for (int id = 0; id < TD; id++) {
            for (int ri = 0; ri < 2; ri++) {
                for (int ci = 0; ci < 4; ci++) {
                    O_acc[id][ri * 4 + ci] *= factor[ri];
                    O_acc[id][8 + ri * 4 + ci] *= factor[ri];
                }
            }
        }

        // ── Step 5: S→P retile ──────────────────────────────────────────
        // S_frag is a 16×32 fragment. For P@V MMA(16,32,16), the left input
        // is 16×16 (8 elems). We split S into two halves: P_frag[0] = first
        // 16 K-cols, P_frag[1] = next 16 K-cols, and iterate ik=0,1.
        float P_frag[2][8];
        for (int ik = 0; ik < 2; ik++)
            for (int e = 0; e < 8; e++)
                P_frag[ik][e] = S_frag[ik * 8 + e];

        // ── Step 6: O += P @ V ──────────────────────────────────────────
        // For each of TD=4 output D-subtiles (each 16×32 = 16 elems):
        for (int id = 0; id < TD; id++) {
            // For each of TK=2 P subtiles along K dim:
            for (int ik = 0; ik < TK; ik++) {
                // Load V fragment as 16×32: V[kv_start+ik*16.., id*32..]
                // V is [kv_len, BD]. Two 16×16 sub-frags side by side.
                // V_frag[0..7] = cols id*32..id*32+16
                // V_frag[8..15] = cols id*32+16..id*32+32
                half V_frag[16];
                int v_row = kv_start + ik * 16;
                int v_valid = min(16, kv_rows_valid - ik * 16);
                {
                    int v_col0 = id * UD;
                    int v_col1 = id * UD + 16;
                    const device half* V_ptr0 = V_head + v_row * int(v_row_stride) + v_col0;
                    const device half* V_ptr1 = V_head + v_row * int(v_row_stride) + v_col1;

                    if (v_valid <= 0) {
                        for (int e = 0; e < 16; e++) V_frag[e] = half(0);
                    } else if (is_last_kv && v_valid < 16) {
                        load_frag_16x16(V_frag, V_ptr0, int(v_row_stride), fm, fn, v_valid, 16);
                        load_frag_16x16(V_frag + 8, V_ptr1, int(v_row_stride), fm, fn, v_valid, 16);
                    } else {
                        load_frag_16x16_unsafe(V_frag, V_ptr0, int(v_row_stride), fm, fn);
                        load_frag_16x16_unsafe(V_frag + 8, V_ptr1, int(v_row_stride), fm, fn);
                    }
                }

                // MMA(16,32,16): O_acc[id] += P_frag[ik] @ V_frag
                // ct_a: 8 elems (16×16 left), ct_b: 16 elems (16×32 right), ct_c: 16 elems
                {
                    mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> pv_op;
                    auto ct_a = pv_op.template get_left_input_cooperative_tensor<float, half, float>();
                    auto ct_b = pv_op.template get_right_input_cooperative_tensor<float, half, float>();
                    auto ct_c = pv_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
                    for (int e = 0; e < 8; e++) ct_a[e] = P_frag[ik][e];
                    for (int e = 0; e < 16; e++) ct_b[e] = V_frag[e];
                    for (int e = 0; e < 16; e++) ct_c[e] = O_acc[id][e];
                    pv_op.run(ct_a, ct_b, ct_c);
                    for (int e = 0; e < 16; e++) O_acc[id][e] = ct_c[e];
                }
            }
        }
    } // end KV block loop

    // ─── Final normalization: O /= row_sum ──────────────────────────────
    for (int id = 0; id < TD; id++) {
        for (int ri = 0; ri < 2; ri++) {
            float inv_sum = (row_sum[ri] > 0.0f) ? (1.0f / row_sum[ri]) : 0.0f;
            for (int ci = 0; ci < 4; ci++) {
                O_acc[id][ri * 4 + ci] *= inv_sum;
                O_acc[id][8 + ri * 4 + ci] *= inv_sum;
            }
        }
    }

    // ─── Store output (seq-major) ─────────────────────────────────────
    // O layout: [N, num_q_heads * BD], seq-major
    // O[seq_pos * o_row_stride + q_head * BD + d]
    device half* O_base = O + (q_start + tm) * o_row_stride + q_head * BD;

    if (is_last_q_block) {
        int q_valid_local = int(q_rows_valid) - int(tm);
        if (fm >= q_valid_local && fm + 8 >= q_valid_local) return;

        for (int id = 0; id < TD; id++) {
            for (int sf = 0; sf < 2; sf++) {
                int col_base = id * UD + sf * 16;
                store_frag_16x16(O_base, int(o_row_stride), O_acc[id] + sf * 8, fm, short(col_base + fn),
                                  q_valid_local, BD);
            }
        }
    } else {
        for (int id = 0; id < TD; id++) {
            for (int sf = 0; sf < 2; sf++) {
                int col_base = id * UD + sf * 16;
                store_frag_16x16_unsafe(O_base, int(o_row_stride), O_acc[id] + sf * 8, fm, short(col_base + fn));
            }
        }
    }
}

"#;

pub const SDPA_NAX_DIAG_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Minimal NAX coordinate helpers (copied, no function constants)
inline short nax_fm(uint slid) {
    short qid = short(slid) >> 2;
    return ((qid & 4) | ((short(slid) >> 1) & 3));
}
inline short nax_fn(uint slid) {
    short qid = short(slid) >> 2;
    return ((qid & 2) | (short(slid) & 1)) * 4;
}

// Minimal diagnostic: single 16x16 @ 16x32^T MMA, NO accumulation loop
// Tests whether cooperative tensor element mapping is correct
// Q: [16, 16] half, K: [32, 16] half, S_out: [16, 32] float
kernel void sdpa_nax_diag_single_mma(
    device const half*  Q      [[buffer(0)]],   // [16, 16]
    device const half*  K      [[buffer(1)]],   // [32, 16]
    device float*       S_out  [[buffer(2)]],   // [16, 32]
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  slid   [[thread_index_in_simdgroup]])
{
    if (sgid != 0) return;

    const short fm = nax_fm(slid);
    const short fn = nax_fn(slid);

    // MLX pattern: pre-offset pointers by per-lane coordinates
    const device half* Q_lane = Q + fm * 16 + fn;   // Q[fm, fn]
    const device half* K_lane = K + fm * 16 + fn;   // K[fm, fn]
    device float* S_lane = S_out + fm * 32 + fn;    // S[fm, fn]

    // Load Q: 16x16 fragment using MLX's sc={0,0} pattern
    // All lanes use same relative offsets, but pointers differ per lane
    half Q_frag[8];
    Q_frag[0] = Q_lane[0 * 16 + 0];
    Q_frag[1] = Q_lane[0 * 16 + 1];
    Q_frag[2] = Q_lane[0 * 16 + 2];
    Q_frag[3] = Q_lane[0 * 16 + 3];
    Q_frag[4] = Q_lane[8 * 16 + 0];
    Q_frag[5] = Q_lane[8 * 16 + 1];
    Q_frag[6] = Q_lane[8 * 16 + 2];
    Q_frag[7] = Q_lane[8 * 16 + 3];

    // Load K: 32x16 as two 16x16 blocks
    half K_frag[16];
    // Block 0: K[0..15, 0..15]
    K_frag[0] = K_lane[0 * 16 + 0];
    K_frag[1] = K_lane[0 * 16 + 1];
    K_frag[2] = K_lane[0 * 16 + 2];
    K_frag[3] = K_lane[0 * 16 + 3];
    K_frag[4] = K_lane[8 * 16 + 0];
    K_frag[5] = K_lane[8 * 16 + 1];
    K_frag[6] = K_lane[8 * 16 + 2];
    K_frag[7] = K_lane[8 * 16 + 3];
    // Block 1: K[16..31, 0..15]
    const device half* K_lane1 = K_lane + 16 * 16;
    K_frag[8]  = K_lane1[0 * 16 + 0];
    K_frag[9]  = K_lane1[0 * 16 + 1];
    K_frag[10] = K_lane1[0 * 16 + 2];
    K_frag[11] = K_lane1[0 * 16 + 3];
    K_frag[12] = K_lane1[8 * 16 + 0];
    K_frag[13] = K_lane1[8 * 16 + 1];
    K_frag[14] = K_lane1[8 * 16 + 2];
    K_frag[15] = K_lane1[8 * 16 + 3];

    // MMA
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    for (int e = 0; e < 8; e++) ct_a[e] = Q_frag[e];
    for (int e = 0; e < 16; e++) ct_b[e] = K_frag[e];
    for (int e = 0; e < 16; e++) ct_c[e] = 0.0f;

    op.run(ct_a, ct_b, ct_c);

    // Store using MLX pattern: pre-offset pointer + 0-offset
    // First 16x16 sub-frag (cols 0..15)
    S_lane[0 * 32 + 0] = ct_c[0];
    S_lane[0 * 32 + 1] = ct_c[1];
    S_lane[0 * 32 + 2] = ct_c[2];
    S_lane[0 * 32 + 3] = ct_c[3];
    S_lane[8 * 32 + 0] = ct_c[4];
    S_lane[8 * 32 + 1] = ct_c[5];
    S_lane[8 * 32 + 2] = ct_c[6];
    S_lane[8 * 32 + 3] = ct_c[7];
    // Second 16x16 sub-frag (cols 16..31)
    S_lane[0 * 32 + 16 + 0] = ct_c[8];
    S_lane[0 * 32 + 16 + 1] = ct_c[9];
    S_lane[0 * 32 + 16 + 2] = ct_c[10];
    S_lane[0 * 32 + 16 + 3] = ct_c[11];
    S_lane[8 * 32 + 16 + 0] = ct_c[12];
    S_lane[8 * 32 + 16 + 1] = ct_c[13];
    S_lane[8 * 32 + 16 + 2] = ct_c[14];
    S_lane[8 * 32 + 16 + 3] = ct_c[15];
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("sdpa", SDPA_SHADER_SOURCE)?;
    registry.register_jit_source("sdpa_bf16", SDPA_BF16_SHADER_SOURCE)?;
    registry.register_jit_source("sdpa_mma", SDPA_MMA_SHADER_SOURCE)?;
    registry.register_jit_source("sdpa_mma_bk32", SDPA_MMA_BK32_SHADER_SOURCE)?;
    // NAX kernels require MetalPerformancePrimitives (Metal 3.1+), gracefully skip if unavailable
    if let Err(e) = registry.register_jit_source("sdpa_nax", SDPA_NAX_SHADER_SOURCE) {
        eprintln!("warning: sdpa_nax registration skipped (MPP unavailable): {e}");
    }
    if let Err(e) = registry.register_jit_source("sdpa_nax_diag", SDPA_NAX_DIAG_SHADER_SOURCE) {
        eprintln!("warning: sdpa_nax_diag registration skipped (MPP unavailable): {e}");
    }
    Ok(())
}

/// Fused Scaled Dot-Product Attention — Flash Attention 2.
///
/// Computes `softmax(Q @ K^T / sqrt(D) + mask) @ V` in a single GPU kernel,
/// avoiding materialisation of the full `[N, S]` score matrix.
///
/// # Arguments
/// - `q`: Query matrix `[N, D]` (N = number of query tokens, D = head dimension)
/// - `k`: Key matrix `[S, D]` (S = number of key/value tokens)
/// - `v`: Value matrix `[S, D]`
/// - `mask`: Optional additive mask tensor. Accepted shapes:
///   - `[N, S]` — per-query-token mask (standard)
///   - `[1, S]` — broadcast across query tokens (e.g. padding mask)
///     The mask is added to attention scores before softmax:
///     `scores = Q @ K^T / sqrt(D) + mask`
/// - `scale`: Scale factor, typically `1.0 / sqrt(D)`
/// - `is_causal`: If true, applies causal masking and enables block skipping
///   optimisation (K/V blocks entirely above the diagonal are skipped).
///
/// Supports f32, f16, and bf16 dtypes. For bf16, all intermediate computation
/// uses f32 accumulation for numerical stability.
///
/// # Returns
/// Output matrix `[N, D]`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate dtypes: Q, K, V must all share the same dtype
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        )));
    }

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
    if d > 256 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: head_dim {d} > 256 not supported by fused kernel"
        )));
    }

    // Validate mask shape: accept [N, S] or broadcastable shapes like [1, S].
    // The kernel expects [N, S], so if mask is [1, S] we broadcast it via copy.
    let mask_expanded: Option<Array> = if let Some(m) = mask {
        if m.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask must be 2D, got {}D",
                m.ndim()
            )));
        }
        // Mask dtype must be f32, f16, or bf16 (matching Q or float)
        match m.dtype() {
            DType::Float32 | DType::Float16 | DType::Bfloat16 => {}
            _ => {
                return Err(KernelError::InvalidShape(format!(
                    "sdpa: mask dtype {:?} not supported, expected f32/f16/bf16",
                    m.dtype()
                )));
            }
        }
        // Reject non-contiguous masks — the kernel reads mask data linearly
        if !m.is_contiguous() {
            return Err(KernelError::InvalidShape(
                "sdpa: mask must be contiguous".into(),
            ));
        }
        if m.shape()[1] != s {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask columns {} != S={s}",
                m.shape()[1]
            )));
        }
        if m.shape()[0] == n {
            // Exact match — no expansion needed
            None
        } else if m.shape()[0] == 1 && n > 1 {
            // Broadcast [1, S] -> [N, S] by repeating the single row
            let m_c = super::make_contiguous(m, registry, queue)?;
            let m = m_c.as_ref().unwrap_or(m);
            let dev = registry.device().raw();
            let expanded = Array::uninit(dev, &[n, s], m.dtype());
            let src = m.metal_buffer().contents() as *const u8;
            let dst = expanded.metal_buffer().contents() as *mut u8;
            let row_bytes = m
                .dtype()
                .numel_to_bytes(s)
                .expect("numel must be block-aligned");
            // SAFETY: SharedMode buffers are CPU-accessible; bounds checked by
            // Array::uninit allocation and row_bytes computation.
            unsafe {
                let src_ptr = src.add(m.offset());
                for row in 0..n {
                    let dst_ptr = dst.add(row * row_bytes);
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_bytes);
                }
            }
            Some(expanded)
        } else {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask shape {:?}, expected [{n}, {s}] or [1, {s}]",
                m.shape()
            )));
        }
    } else {
        None
    };
    // Resolve the mask reference: use expanded if we created one, otherwise the original.
    let mask_ref: Option<&Array> = match (&mask_expanded, mask) {
        (Some(expanded), _) => Some(expanded),
        (None, Some(m)) => Some(m),
        _ => None,
    };

    // Make inputs contiguous
    let q_c = super::make_contiguous(q, registry, queue)?;
    let q = q_c.as_ref().unwrap_or(q);
    let k_c = super::make_contiguous(k, registry, queue)?;
    let k = k_c.as_ref().unwrap_or(k);
    let v_c = super::make_contiguous(v, registry, queue)?;
    let v = v_c.as_ref().unwrap_or(v);

    // Select kernel: decode fast path when N == 1, general kernel otherwise
    let use_decode = n == 1;

    let kernel_name = match (q.dtype(), use_decode) {
        (DType::Float32, true) => "sdpa_decode_f32",
        (DType::Float32, false) => "sdpa_f32",
        (DType::Float16, true) => "sdpa_decode_f16",
        (DType::Float16, false) => "sdpa_f16",
        (DType::Bfloat16, true) => "sdpa_decode_bf16",
        (DType::Bfloat16, false) => "sdpa_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa: unsupported dtype {:?}",
                q.dtype()
            )));
        }
    };

    let pipeline = if use_decode {
        let constants = sdpa_decode_constants(d, false);
        registry.get_pipeline_with_constants(kernel_name, q.dtype(), &constants)?
    } else {
        registry.get_pipeline(kernel_name, q.dtype())?
    };
    let dev = registry.device().raw();

    // Output array
    let out = Array::uninit(dev, &[n, d], q.dtype());

    // Params buffer: [N, S, D, has_mask, is_causal]
    let has_mask: u32 = if mask_ref.is_some() { 1 } else { 0 };
    let is_causal_u32: u32 = if is_causal { 1 } else { 0 };
    let params: [u32; 5] = [n as u32, s as u32, d as u32, has_mask, is_causal_u32];
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        20, // 5 * 4 bytes
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
    let mask_buf = if let Some(m) = mask_ref {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask_ref.map_or(0, |m| m.offset()) as u64;

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(sdpa_slot::Q, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(sdpa_slot::K, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(sdpa_slot::V, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_buffer(sdpa_slot::PARAMS, Some(&params_buf), 0);
    encoder.set_buffer(sdpa_slot::SCALE, Some(&scale_buf), 0);

    if use_decode {
        // Decode kernel: single threadgroup
        let tg_size = std::cmp::min(DECODE_THREADS, pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
    } else {
        // General kernel: one threadgroup per Q block
        let n_threadgroups = n.div_ceil(BR) as u64;
        let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_thread_groups(
            MTLSize::new(n_threadgroups, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }

    encoder.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
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
/// - `is_causal`: If true, applies causal masking with block skipping
///
/// # Returns
/// Vec of `[N, D]` output arrays, one per query head.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_batched(
    registry: &KernelRegistry,
    q_heads: &[Array],
    k_heads: &[Array],
    v_heads: &[Array],
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
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
            is_causal,
            queue,
        )?;
        outputs.push(out);
    }
    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Into-CB variants (encode into existing command buffer, no commit/wait)
// ---------------------------------------------------------------------------

/// Private: encode SDPA into an existing compute encoder (does NOT call end_encoding).
fn sdpa_encode_impl(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    // Validate dtypes: Q, K, V must all share the same dtype
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        )));
    }

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
    if d > 256 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa: head_dim {d} > 256 not supported by fused kernel"
        )));
    }

    // Validate mask if provided
    if let Some(m) = mask {
        if m.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask must be 2D, got {}D",
                m.ndim()
            )));
        }
        match m.dtype() {
            DType::Float32 | DType::Float16 | DType::Bfloat16 => {}
            _ => {
                return Err(KernelError::InvalidShape(format!(
                    "sdpa: mask dtype {:?} not supported, expected f32/f16/bf16",
                    m.dtype()
                )));
            }
        }
        if !m.is_contiguous() {
            return Err(KernelError::InvalidShape(
                "sdpa: mask must be contiguous".into(),
            ));
        }
        if m.shape()[1] != s {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask columns {} != S={s}",
                m.shape()[1]
            )));
        }
        if m.shape()[0] != n && !(m.shape()[0] == 1 && n > 1) {
            return Err(KernelError::InvalidShape(format!(
                "sdpa: mask shape {:?}, expected [{n}, {s}] or [1, {s}]",
                m.shape()
            )));
        }
    }

    // Select kernel: decode fast path when N == 1, general kernel otherwise
    let use_decode = n == 1;

    let kernel_name = match (q.dtype(), use_decode) {
        (DType::Float32, true) => "sdpa_decode_f32",
        (DType::Float32, false) => "sdpa_f32",
        (DType::Float16, true) => "sdpa_decode_f16",
        (DType::Float16, false) => "sdpa_f16",
        (DType::Bfloat16, true) => "sdpa_decode_bf16",
        (DType::Bfloat16, false) => "sdpa_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa: unsupported dtype {:?}",
                q.dtype()
            )));
        }
    };

    let pipeline = if use_decode {
        let constants = sdpa_decode_constants(d, false);
        registry.get_pipeline_with_constants(kernel_name, q.dtype(), &constants)?
    } else {
        registry.get_pipeline(kernel_name, q.dtype())?
    };
    let dev = registry.device().raw();

    let out = Array::uninit(dev, &[n, d], q.dtype());

    let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
    let is_causal_u32: u32 = 0; // encode path does not support causal flag
    let params: [u32; 5] = [n as u32, s as u32, d as u32, has_mask, is_causal_u32];

    let dummy_buf;
    let mask_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(sdpa_slot::Q, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(sdpa_slot::K, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(sdpa_slot::V, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        20,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    if use_decode {
        // Decode kernel: single threadgroup
        let tg_size = std::cmp::min(DECODE_THREADS, pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
    } else {
        // General kernel: one threadgroup per Q block
        let n_threadgroups = n.div_ceil(BR) as u64;
        let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_thread_groups(
            MTLSize::new(n_threadgroups, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }

    Ok(out)
}

/// Encode SDPA into an existing command buffer (no commit/wait).
pub fn sdpa_into_cb(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let encoder = cb.new_compute_command_encoder();
    let result = sdpa_encode_impl(registry, q, k, v, mask, scale, encoder);
    if result.is_err() {
        encoder.end_encoding();
        return result;
    }
    encoder.end_encoding();
    result
}

/// Encode SDPA into an existing compute encoder (no encoder create/end).
///
/// Caller is responsible for creating and ending the encoder.
pub fn sdpa_encode(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    sdpa_encode_impl(registry, q, k, v, mask, scale, encoder)
}

/// Encode batched SDPA (multi-head) into an existing command buffer (no commit/wait).
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

/// GQA-optimized batched prefill SDPA — single kernel dispatch for all heads.
///
/// For GQA models (e.g., num_heads=32, num_kv_heads=8), processes R=4 Q heads
/// per KV head in a single threadgroup, loading K/V once for all Q heads in
/// the group. This saves (R-1)/R of K/V global memory reads.
///
/// Currently f16-only. Falls back to per-head `sdpa_into_cb` for f32/bf16.
///
/// Inputs are contiguous slabs:
/// - `q_slab`: `[num_heads, N, D]` (contiguous Q for all heads)
/// - `k_slab`: `[num_kv_heads, S, D]` (contiguous K for all KV heads)
/// - `v_slab`: `[num_kv_heads, S, D]` (contiguous V for all KV heads)
/// - `kv_seq_stride`: optional inter-head stride for KV slabs. When `Some(max_seq_len)`,
///   the kernel uses `max_seq_len * D` for KV head offsets instead of `kv_len * D`.
///   Use this when KV comes from a pre-allocated cache with `max_seq_len` stride.
///   When `None`, stride equals `kv_len` (backward compatible).
/// - `mask`: optional `[N, S]` (shared across heads)
///
/// Returns: `[num_heads, N, D]` output slab.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_gqa_slab_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let dtype = q_slab.dtype();

    // Currently only f16 has the GQA prefill kernel
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_gqa: only f16 supported, got {:?}",
            dtype
        )));
    }

    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }
    if head_dim > 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: head_dim {head_dim} > 128 not supported (D>128 needs O_acc2)"
        )));
    }

    let gqa_ratio = num_heads / num_kv_heads;
    let stride_s = kv_seq_stride.unwrap_or(kv_len);
    if stride_s < kv_len {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: kv_seq_stride ({stride_s}) must be >= kv_len ({kv_len})"
        )));
    }

    let pipeline = registry.get_pipeline("sdpa_prefill_gqa_f16", dtype)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let has_mask_u32: u32 = if mask.is_some() { 1 } else { 0 };
    let is_causal_u32: u32 = if is_causal { 1 } else { 0 };
    let n_q_blocks = seq_len.div_ceil(BR);
    let params: [u32; 10] = [
        seq_len as u32,
        kv_len as u32,
        head_dim as u32,
        has_mask_u32,
        is_causal_u32,
        num_heads as u32,
        num_kv_heads as u32,
        gqa_ratio as u32,
        n_q_blocks as u32,
        stride_s as u32,
    ];

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
    encoder.set_buffer(
        sdpa_slot::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        std::mem::size_of::<[u32; 10]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    // Grid: 1D, n_q_blocks * num_kv_heads threadgroups
    let total_tgs = (n_q_blocks * num_kv_heads) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(total_tgs, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();

    Ok(out)
}

/// MMA-based SDPA prefill for f16, D=128 — simdgroup 8×8 matrix multiply.
///
/// Uses the same slab layout as [`sdpa_prefill_gqa_slab_into_cb`]:
/// - `q_slab`: `[num_heads * seq_len * head_dim]`
/// - `k_slab`: `[num_kv_heads * S * head_dim]`
/// - `v_slab`: `[num_kv_heads * S * head_dim]`
///
/// Returns: `[num_heads * seq_len * head_dim]` output slab.
///
/// Output is in seq-major layout `[seq_len, num_heads * head_dim]` (not head-major).
///
/// Grid: (ceildiv(seq_len, 32), num_heads, 1) — each TG processes 32 Q rows.
/// Function constants control alignment and masking specialisation.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_mma_f16_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_mma: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    // Function constants for pipeline specialization
    let align_q = seq_len % MMA_BQ == 0;
    let align_k = kv_len % MMA_BK == 0;
    let has_mask_val = mask.is_some();

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
        (301, FunctionConstantValue::Bool(has_mask_val)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_mma_f16", dtype, &constants)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let dummy_buf;
    let mask_metal_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_mma::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_mma::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_mma::MASK, Some(mask_metal_buf), mask_offset);
    encoder.set_bytes(
        sdpa_mma::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_mma::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_mma::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_mma::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    // Grid: (ceildiv(seq_len, BQ), num_heads, 1)
    let n_q_blocks = seq_len.div_ceil(MMA_BQ) as u64;
    let tg_size = std::cmp::min(MMA_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// MMA SDPA prefill kernel with BK=32 — halves KV loop iterations.
///
/// Output is in seq-major layout `[seq_len, num_heads * head_dim]` (not head-major).
///
/// Same interface as [`sdpa_prefill_mma_f16_into_cb`] but uses a larger K/V
/// block size (32 instead of 16), reducing the number of outer-loop iterations
/// by 2×. Benefits longer sequences (seq_len >= 256) where the loop overhead
/// is more significant.
///
/// Grid: (ceildiv(seq_len, 32), num_heads, 1) — each TG processes 32 Q rows.
/// TG memory: ~19 KB (vs ~10 KB for BK=16).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_mma_bk32_f16_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_mma_bk32: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma_bk32: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma_bk32: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    let align_q = seq_len % MMA_BQ == 0;
    let align_k = kv_len % MMA_BK32 == 0;
    let has_mask_val = mask.is_some();

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
        (301, FunctionConstantValue::Bool(has_mask_val)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_mma_bk32_f16", dtype, &constants)?;

    // BK=32 cooperative loaders require exactly 128 threads.
    // If Metal compiler reduced max_threads (register pressure), fall back to BK=16.
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    if max_threads < MMA_THREADS {
        eprintln!(
            "sdpa_prefill_mma_bk32: max_threads={max_threads} < {MMA_THREADS}, falling back to BK=16"
        );
        return sdpa_prefill_mma_f16_into_cb(
            registry,
            q_slab,
            k_slab,
            v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            kv_len,
            kv_seq_stride,
            mask,
            scale,
            is_causal,
            v_head_stride,
            v_row_stride,
            cb,
        );
    }

    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let dummy_buf;
    let mask_metal_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_mma::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_mma::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_mma::MASK, Some(mask_metal_buf), mask_offset);
    encoder.set_bytes(
        sdpa_mma::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_mma::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_mma::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_mma::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    let n_q_blocks = seq_len.div_ceil(MMA_BQ) as u64;
    let tg_size = MMA_THREADS;
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// NAX SDPA prefill kernel — MetalPerformancePrimitives based.
///
/// Uses BQ=64, BK=32, BD=128 with 4 simdgroups (128 threads).
/// No threadgroup memory — all loads go directly from device to registers.
///
/// Grid: (ceildiv(seq_len, 64), num_heads, 1) — each TG processes 64 Q rows.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_nax_f16_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_nax: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_nax: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_nax: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    let align_q = seq_len % NAX_BQ == 0;
    let align_k = kv_len % NAX_BK == 0;

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_nax_f16", dtype, &constants)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_nax::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_nax::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_nax::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_nax::OUT, Some(out.metal_buffer()), 0);
    encoder.set_bytes(
        sdpa_nax::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_nax::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_nax::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_nax::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    // Grid: (ceildiv(seq_len, BQ=64), num_heads, 1)
    let n_q_blocks = seq_len.div_ceil(NAX_BQ) as u64;
    let tg_size = std::cmp::min(NAX_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

// ---------------------------------------------------------------------------
// _encode variants — accept &ComputeCommandEncoderRef instead of &CommandBufferRef
// ---------------------------------------------------------------------------

/// Encode GQA slab SDPA prefill into an existing compute command encoder.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_gqa_slab_encode(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    let dtype = q_slab.dtype();

    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_gqa: only f16 supported, got {:?}",
            dtype
        )));
    }

    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }
    if head_dim > 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: head_dim {head_dim} > 128 not supported (D>128 needs O_acc2)"
        )));
    }

    let gqa_ratio = num_heads / num_kv_heads;
    let stride_s = kv_seq_stride.unwrap_or(kv_len);
    if stride_s < kv_len {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_gqa: kv_seq_stride ({stride_s}) must be >= kv_len ({kv_len})"
        )));
    }

    let pipeline = registry.get_pipeline("sdpa_prefill_gqa_f16", dtype)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let has_mask_u32: u32 = if mask.is_some() { 1 } else { 0 };
    let is_causal_u32: u32 = if is_causal { 1 } else { 0 };
    let n_q_blocks = seq_len.div_ceil(BR);
    let params: [u32; 10] = [
        seq_len as u32,
        kv_len as u32,
        head_dim as u32,
        has_mask_u32,
        is_causal_u32,
        num_heads as u32,
        num_kv_heads as u32,
        gqa_ratio as u32,
        n_q_blocks as u32,
        stride_s as u32,
    ];

    let dummy_buf;
    let mask_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_slot::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        std::mem::size_of::<[u32; 10]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    let total_tgs = (n_q_blocks * num_kv_heads) as u64;
    let tg_size = std::cmp::min(THREADS_PER_TG, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(total_tgs, 1, 1), MTLSize::new(tg_size, 1, 1));

    Ok(out)
}

/// Encode MMA-based SDPA prefill (f16, D=128) into an existing compute command encoder.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_mma_f16_encode(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_mma: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    let align_q = seq_len % MMA_BQ == 0;
    let align_k = kv_len % MMA_BK == 0;
    let has_mask_val = mask.is_some();

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
        (301, FunctionConstantValue::Bool(has_mask_val)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_mma_f16", dtype, &constants)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let dummy_buf;
    let mask_metal_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_mma::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_mma::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_mma::MASK, Some(mask_metal_buf), mask_offset);
    encoder.set_bytes(
        sdpa_mma::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_mma::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_mma::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_mma::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    let n_q_blocks = seq_len.div_ceil(MMA_BQ) as u64;
    let tg_size = std::cmp::min(MMA_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

/// Encode MMA SDPA prefill with BK=32 into an existing compute command encoder.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_mma_bk32_f16_encode(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_mma_bk32: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma_bk32: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_mma_bk32: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    let align_q = seq_len % MMA_BQ == 0;
    let align_k = kv_len % MMA_BK32 == 0;
    let has_mask_val = mask.is_some();

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
        (301, FunctionConstantValue::Bool(has_mask_val)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_mma_bk32_f16", dtype, &constants)?;

    let max_threads = pipeline.max_total_threads_per_threadgroup();
    if max_threads < MMA_THREADS {
        eprintln!(
            "sdpa_prefill_mma_bk32: max_threads={max_threads} < {MMA_THREADS}, falling back to BK=16"
        );
        return sdpa_prefill_mma_f16_encode(
            registry,
            q_slab,
            k_slab,
            v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            kv_len,
            kv_seq_stride,
            mask,
            scale,
            is_causal,
            v_head_stride,
            v_row_stride,
            encoder,
        );
    }

    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let dummy_buf;
    let mask_metal_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_mma::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_mma::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_mma::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_mma::MASK, Some(mask_metal_buf), mask_offset);
    encoder.set_bytes(
        sdpa_mma::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_mma::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_mma::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_mma::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_mma::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    let n_q_blocks = seq_len.div_ceil(MMA_BQ) as u64;
    let tg_size = MMA_THREADS;
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

/// Encode NAX SDPA prefill into an existing compute command encoder.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill_nax_f16_encode(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    kv_seq_stride: Option<usize>,
    scale: f32,
    is_causal: bool,
    v_head_stride: Option<usize>,
    v_row_stride: Option<usize>,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let dtype = q_slab.dtype();
    if dtype != DType::Float16 {
        return Err(KernelError::NotFound(format!(
            "sdpa_prefill_nax: only f16 supported, got {:?}",
            dtype
        )));
    }
    if head_dim != 128 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_nax: only head_dim=128 supported, got {head_dim}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_prefill_nax: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let gqa_factor = (num_heads / num_kv_heads) as u32;
    let align_q = seq_len % NAX_BQ == 0;
    let align_k = kv_len % NAX_BK == 0;

    let constants = vec![
        (200u32, FunctionConstantValue::Bool(align_q)),
        (201, FunctionConstantValue::Bool(align_k)),
        (300, FunctionConstantValue::Bool(is_causal)),
    ];

    let pipeline =
        registry.get_pipeline_with_constants("sdpa_prefill_nax_f16", dtype, &constants)?;
    let dev = registry.device().raw();

    let out_numel = num_heads * seq_len * head_dim;
    let out = Array::uninit(dev, &[out_numel], dtype);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_nax::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_nax::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_nax::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_nax::OUT, Some(out.metal_buffer()), 0);
    encoder.set_bytes(
        sdpa_nax::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::GQA_FACTOR,
        4,
        &gqa_factor as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );
    let stride_s_val = kv_seq_stride.unwrap_or(kv_len) as u32;
    encoder.set_bytes(
        sdpa_nax::STRIDE_S,
        4,
        &stride_s_val as *const u32 as *const std::ffi::c_void,
    );
    let num_q_heads_val = num_heads as u32;
    encoder.set_bytes(
        sdpa_nax::NUM_Q_HEADS,
        4,
        &num_q_heads_val as *const u32 as *const std::ffi::c_void,
    );
    let v_hs = v_head_stride.unwrap_or(stride_s_val as usize * head_dim) as u32;
    let v_rs = v_row_stride.unwrap_or(head_dim) as u32;
    encoder.set_bytes(
        sdpa_nax::V_HEAD_STRIDE,
        4,
        &v_hs as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_nax::V_ROW_STRIDE,
        4,
        &v_rs as *const u32 as *const std::ffi::c_void,
    );

    let n_q_blocks = seq_len.div_ceil(NAX_BQ) as u64;
    let tg_size = std::cmp::min(NAX_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(n_q_blocks, num_heads as u64, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

/// Diagnostic: compute only Q@K^T using NAX MMA, output float S matrix.
/// For debugging cooperative tensor element mapping.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_nax_diag_qkt_into_cb(
    registry: &KernelRegistry,
    q: &Array, // [N, D] f16
    k: &Array, // [S, D] f16
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let pipeline = registry.get_pipeline("sdpa_nax_diag_qkt_f16", DType::Float16)?;
    let dev = registry.device().raw();

    let out = Array::uninit(dev, &[seq_len * kv_len], DType::Float32);

    let n_val = seq_len as u32;
    let s_val = kv_len as u32;
    let d_val = head_dim as u32;

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(sdpa_diag_qkt::Q, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(sdpa_diag_qkt::K, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(sdpa_diag_qkt::OUT, Some(out.metal_buffer()), 0);
    encoder.set_bytes(
        sdpa_diag_qkt::N,
        4,
        &n_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_diag_qkt::S,
        4,
        &s_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_diag_qkt::D,
        4,
        &d_val as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_diag_qkt::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    // Single threadgroup: 128 threads (4 SG), processes up to 64 Q rows
    encoder.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(128, 1, 1));
    encoder.end_encoding();

    Ok(out)
}

/// Diagnostic: single MMA Q@K^T (no accumulation loop).
/// Q: [16, 16] f16, K: [32, 16] f16 → S: [16, 32] f32
pub fn sdpa_nax_diag_single_mma_into_cb(
    registry: &KernelRegistry,
    q: &Array, // [16*16] f16
    k: &Array, // [32*16] f16
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let pipeline = registry.get_pipeline("sdpa_nax_diag_single_mma", DType::Float16)?;
    let dev = registry.device().raw();

    let out = Array::uninit(dev, &[16 * 32], DType::Float32);

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_diag_single_mma::Q,
        Some(q.metal_buffer()),
        q.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_diag_single_mma::K,
        Some(k.metal_buffer()),
        k.offset() as u64,
    );
    encoder.set_buffer(sdpa_diag_single_mma::OUT, Some(out.metal_buffer()), 0);

    encoder.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(128, 1, 1));
    encoder.end_encoding();

    Ok(out)
}

/// Batched multi-head SDPA decode — single GPU dispatch for ALL heads.
///
/// Replaces the per-head loop in `sdpa_batched_into_cb` with a single
/// kernel launch where `grid.x = num_heads`, processing every head in
/// parallel on the GPU.
///
/// Inputs are contiguous slabs:
/// - `q_slab`: `[num_heads * head_dim]` (flat query vector per head)
/// - `k_slab`: `[num_kv_heads * seq_len * head_dim]` (contiguous KV cache)
/// - `v_slab`: `[num_kv_heads * seq_len * head_dim]` (contiguous KV cache)
/// - `mask`: optional `[seq_len]` (shared across heads for decode)
///
/// Returns: `[num_heads * head_dim]` (flat output slab).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_batched_slab_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    mask: Option<&Array>,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    sdpa_decode_batched_slab_stride_into_cb(
        registry,
        q_slab,
        k_slab,
        v_slab,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        None,
        mask,
        scale,
        cb,
    )
}

/// Like [`sdpa_decode_batched_slab_into_cb`] but accepts an explicit `stride_seq_len`
/// for KV slabs where the inter-head stride (max_seq_len) differs from actual `seq_len`.
///
/// When `stride_seq_len` is `Some(max_seq_len)`, the kernel uses `max_seq_len` for the
/// inter-head pointer offset in the KV slab, but only iterates over `seq_len` keys.
/// When `None`, stride equals `seq_len` (backward compatible).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_batched_slab_stride_into_cb(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    stride_seq_len: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    // Validate dtypes
    let dtype = q_slab.dtype();
    if k_slab.dtype() != dtype || v_slab.dtype() != dtype {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
            dtype,
            k_slab.dtype(),
            v_slab.dtype()
        )));
    }

    // Validate head counts
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    // Validate head_dim
    if head_dim > 256 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: head_dim {head_dim} > 256 not supported"
        )));
    }

    // Validate slab sizes
    let stride_s = stride_seq_len.unwrap_or(seq_len);
    if stride_s < seq_len {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: stride_seq_len ({stride_s}) must be >= seq_len ({seq_len})"
        )));
    }
    let q_expected = num_heads * head_dim;
    // KV slab validation uses stride (max_seq_len) for total size, not seq_len
    let kv_expected_stride = num_kv_heads * stride_s * head_dim;
    if q_slab.numel() < q_expected {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: q_slab length {} < expected {q_expected}",
            q_slab.numel()
        )));
    }
    if k_slab.numel() < kv_expected_stride {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: k_slab length {} < expected {kv_expected_stride}",
            k_slab.numel()
        )));
    }
    if v_slab.numel() < kv_expected_stride {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: v_slab length {} < expected {kv_expected_stride}",
            v_slab.numel()
        )));
    }

    // Validate mask if provided
    if let Some(m) = mask {
        if m.dtype() != dtype {
            return Err(KernelError::InvalidShape(format!(
                "sdpa_decode_batched: mask dtype {:?} != Q dtype {:?}",
                m.dtype(),
                dtype
            )));
        }
        if !m.is_contiguous() {
            return Err(KernelError::InvalidShape(
                "sdpa_decode_batched: mask must be contiguous".into(),
            ));
        }
        if m.numel() < seq_len {
            return Err(KernelError::InvalidShape(format!(
                "sdpa_decode_batched: mask length {} < seq_len {seq_len}",
                m.numel()
            )));
        }
    }

    // Select kernel by dtype
    let kernel_name = match dtype {
        DType::Float32 => "sdpa_decode_batched_f32",
        DType::Float16 => "sdpa_decode_batched_f16",
        DType::Bfloat16 => "sdpa_decode_batched_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa_decode_batched: unsupported dtype {:?}",
                dtype
            )));
        }
    };

    let constants = sdpa_decode_constants(head_dim, false);
    let pipeline = registry.get_pipeline_with_constants(kernel_name, dtype, &constants)?;
    let dev = registry.device().raw();

    let out = Array::uninit(dev, &[q_expected], dtype);

    let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
    let params: [u32; 6] = [
        num_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        head_dim as u32,
        has_mask,
        stride_s as u32,
    ];

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
    encoder.set_buffer(
        sdpa_slot::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        std::mem::size_of::<[u32; 6]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    // One threadgroup per Q head, 256 threads each
    let tg_size = std::cmp::min(DECODE_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// Encode batched SDPA decode into an existing compute command encoder (no encoder create/end).
/// Caller is responsible for creating and ending the encoder.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_batched_slab_stride_into_encoder(
    registry: &KernelRegistry,
    q_slab: &Array,
    k_slab: &Array,
    v_slab: &Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    stride_seq_len: Option<usize>,
    mask: Option<&Array>,
    scale: f32,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    // Validate dtypes
    let dtype = q_slab.dtype();
    if k_slab.dtype() != dtype || v_slab.dtype() != dtype {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
            dtype,
            k_slab.dtype(),
            v_slab.dtype()
        )));
    }

    // Validate head counts
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    // Validate head_dim
    if head_dim > 256 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: head_dim {head_dim} > 256 not supported"
        )));
    }

    // Validate slab sizes
    let stride_s = stride_seq_len.unwrap_or(seq_len);
    if stride_s < seq_len {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: stride_seq_len ({stride_s}) must be >= seq_len ({seq_len})"
        )));
    }
    let q_expected = num_heads * head_dim;
    // KV slab validation uses stride (max_seq_len) for total size, not seq_len
    let kv_expected_stride = num_kv_heads * stride_s * head_dim;
    if q_slab.numel() < q_expected {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: q_slab length {} < expected {q_expected}",
            q_slab.numel()
        )));
    }
    if k_slab.numel() < kv_expected_stride {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: k_slab length {} < expected {kv_expected_stride}",
            k_slab.numel()
        )));
    }
    if v_slab.numel() < kv_expected_stride {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_decode_batched: v_slab length {} < expected {kv_expected_stride}",
            v_slab.numel()
        )));
    }

    // Validate mask if provided
    if let Some(m) = mask {
        if m.dtype() != dtype {
            return Err(KernelError::InvalidShape(format!(
                "sdpa_decode_batched: mask dtype {:?} != Q dtype {:?}",
                m.dtype(),
                dtype
            )));
        }
        if !m.is_contiguous() {
            return Err(KernelError::InvalidShape(
                "sdpa_decode_batched: mask must be contiguous".into(),
            ));
        }
        if m.numel() < seq_len {
            return Err(KernelError::InvalidShape(format!(
                "sdpa_decode_batched: mask length {} < seq_len {seq_len}",
                m.numel()
            )));
        }
    }

    // Select kernel by dtype
    let kernel_name = match dtype {
        DType::Float32 => "sdpa_decode_batched_f32",
        DType::Float16 => "sdpa_decode_batched_f16",
        DType::Bfloat16 => "sdpa_decode_batched_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "sdpa_decode_batched: unsupported dtype {:?}",
                dtype
            )));
        }
    };

    let constants = sdpa_decode_constants(head_dim, false);
    let pipeline = registry.get_pipeline_with_constants(kernel_name, dtype, &constants)?;
    let dev = registry.device().raw();

    let out = Array::uninit(dev, &[q_expected], dtype);

    let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
    let params: [u32; 6] = [
        num_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        head_dim as u32,
        has_mask,
        stride_s as u32,
    ];

    let dummy_buf;
    let mask_buf = if let Some(m) = mask {
        m.metal_buffer()
    } else {
        dummy_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        &dummy_buf
    };
    let mask_offset = mask.map_or(0, |m| m.offset()) as u64;

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        sdpa_slot::Q,
        Some(q_slab.metal_buffer()),
        q_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::K,
        Some(k_slab.metal_buffer()),
        k_slab.offset() as u64,
    );
    encoder.set_buffer(
        sdpa_slot::V,
        Some(v_slab.metal_buffer()),
        v_slab.offset() as u64,
    );
    encoder.set_buffer(sdpa_slot::OUT, Some(out.metal_buffer()), 0);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        std::mem::size_of::<[u32; 6]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    // One threadgroup per Q head, 256 threads each
    let tg_size = std::cmp::min(DECODE_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode SDPA decode using a pre-resolved PSO and pre-allocated output buffer.
/// Skips all validation — caller must ensure correctness.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    q_buf: &metal::BufferRef,
    q_offset: u64,
    k_buf: &metal::BufferRef,
    k_offset: u64,
    v_buf: &metal::BufferRef,
    v_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    mask_buf: &metal::BufferRef,
    mask_offset: u64,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    has_mask: u32,
    stride_s: u32,
    scale: f32,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    let params: [u32; 6] = [
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        has_mask,
        stride_s,
    ];

    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(sdpa_slot::Q, Some(q_buf), q_offset);
    encoder.set_buffer(sdpa_slot::K, Some(k_buf), k_offset);
    encoder.set_buffer(sdpa_slot::V, Some(v_buf), v_offset);
    encoder.set_buffer(sdpa_slot::OUT, Some(out_buf), out_offset);
    encoder.set_buffer(sdpa_slot::MASK, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        sdpa_slot::PARAMS,
        std::mem::size_of::<[u32; 6]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        sdpa_slot::SCALE,
        4,
        &scale as *const f32 as *const std::ffi::c_void,
    );

    let tg_size = std::cmp::min(DECODE_THREADS, pso.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(num_heads as u64, 1, 1),
        metal::MTLSize::new(tg_size, 1, 1),
    );
}

/// Get the SDPA decode kernel name for a dtype.
pub fn sdpa_decode_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("sdpa_decode_batched_f32"),
        DType::Float16 => Ok("sdpa_decode_batched_f16"),
        DType::Bfloat16 => Ok("sdpa_decode_batched_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "sdpa_decode: unsupported dtype {:?}",
            dtype
        ))),
    }
}

/// Build function constants for SDPA decode specialization.
///
/// When `head_dim` is a known common value (64, 128, 256), we specialize
/// the kernel for that value, enabling compile-time loop unrolling.
pub fn sdpa_decode_constants(
    head_dim: usize,
    is_causal: bool,
) -> Vec<(u32, crate::kernels::FunctionConstantValue)> {
    use crate::kernels::FunctionConstantValue;
    let mut constants = Vec::new();
    // Only specialize for common head dims to limit PSO cache bloat
    if matches!(head_dim, 64 | 128 | 256) {
        constants.push((200, FunctionConstantValue::U32(head_dim as u32)));
    }
    constants.push((201, FunctionConstantValue::Bool(is_causal)));
    constants
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, metal::CommandQueue) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu_dev.raw().new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();
        (registry, queue)
    }

    #[test]
    fn test_sdpa_dtype_mismatch_qk_returns_error() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[4, 8], DType::Float32);
        let k = Array::zeros(dev, &[6, 8], DType::Float16);
        let v = Array::zeros(dev, &[6, 8], DType::Float32);

        let result = sdpa(&registry, &q, &k, &v, None, 0.125, false, &queue);
        let err = result.expect_err("expected error");
        let msg = format!("{err}");
        assert!(
            msg.contains("dtype mismatch"),
            "expected dtype mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_sdpa_dtype_mismatch_qv_returns_error() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[4, 8], DType::Float16);
        let k = Array::zeros(dev, &[6, 8], DType::Float16);
        let v = Array::zeros(dev, &[6, 8], DType::Float32);

        let result = sdpa(&registry, &q, &k, &v, None, 0.125, false, &queue);
        let err = result.expect_err("expected error");
        let msg = format!("{err}");
        assert!(
            msg.contains("dtype mismatch"),
            "expected dtype mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_sdpa_non_contiguous_mask_returns_error() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[4, 8], DType::Float32);
        let k = Array::zeros(dev, &[6, 8], DType::Float32);
        let v = Array::zeros(dev, &[6, 8], DType::Float32);

        // Create a non-contiguous mask via view with stride [12, 2] over a [4,12] buffer.
        let big_mask = Array::zeros(dev, &[4, 12], DType::Float32);
        let nc_mask = big_mask.view(vec![4, 6], vec![12, 2], 0);
        assert!(!nc_mask.is_contiguous(), "mask should be non-contiguous");

        let result = sdpa(&registry, &q, &k, &v, Some(&nc_mask), 0.125, false, &queue);
        let err = result.expect_err("expected error");
        let msg = format!("{err}");
        assert!(
            msg.contains("contiguous"),
            "expected contiguous error, got: {msg}"
        );
    }

    #[test]
    fn test_sdpa_into_cb_dtype_mismatch_returns_error() {
        let (registry, _queue) = setup();
        let dev = registry.device().raw();

        let q = Array::zeros(dev, &[4, 8], DType::Float32);
        let k = Array::zeros(dev, &[6, 8], DType::Float16);
        let v = Array::zeros(dev, &[6, 8], DType::Float32);

        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let result = sdpa_into_cb(&registry, &q, &k, &v, None, 0.125, cb);
        let err = result.expect_err("expected error");
        let msg = format!("{err}");
        assert!(
            msg.contains("dtype mismatch"),
            "expected dtype mismatch error, got: {msg}"
        );
    }

    // --- GQA prefill kernel tests -----------------------------------------------

    /// Generate deterministic pseudo-random f32 data in [-0.5, 0.5].
    /// Smaller range to avoid f16 overflow in attention scores.
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut data = Vec::with_capacity(len);
        let mut state = seed;
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let val = ((state & 0xFFFF) as f32 / 65536.0) - 0.5;
            data.push(val);
        }
        data
    }

    /// Create an f16 array from f32 data using GPU copy_cast.
    fn make_f16_array(
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Array {
        let dev = registry.device().raw();
        let f32_arr = Array::from_slice(dev, data, shape);
        crate::ops::copy::copy_cast(registry, &f32_arr, DType::Float16, queue).unwrap()
    }

    /// Read f16 array back as f32 via GPU copy_cast.
    fn read_f16_as_f32(
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        arr: &Array,
    ) -> Vec<f32> {
        let f32_arr = crate::ops::copy::copy_cast(registry, arr, DType::Float32, queue).unwrap();
        f32_arr.to_vec_checked()
    }

    fn setup_with_copy() -> (KernelRegistry, metal::CommandQueue) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu_dev.raw().new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();
        (registry, queue)
    }

    /// Helper: run GQA prefill and reference sdpa_batched, return (gqa_f32, ref_f32).
    #[allow(clippy::too_many_arguments)]
    fn run_gqa_vs_reference(
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        is_causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let _gqa_ratio = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Generate data
        let q_data = pseudo_random(num_heads * seq_len * head_dim, 42);
        let k_data = pseudo_random(num_kv_heads * kv_len * head_dim, 137);
        let v_data = pseudo_random(num_kv_heads * kv_len * head_dim, 256);

        // Create f16 slabs
        let q_slab = make_f16_array(
            registry,
            queue,
            &q_data,
            vec![num_heads * seq_len * head_dim],
        );
        let k_slab = make_f16_array(
            registry,
            queue,
            &k_data,
            vec![num_kv_heads * kv_len * head_dim],
        );
        let v_slab = make_f16_array(
            registry,
            queue,
            &v_data,
            vec![num_kv_heads * kv_len * head_dim],
        );

        // --- GQA prefill path ---
        let cb = queue.new_command_buffer();
        let gqa_out = sdpa_prefill_gqa_slab_into_cb(
            registry,
            &q_slab,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            kv_len,
            None, // kv_seq_stride = kv_len (contiguous)
            None,
            scale,
            is_causal,
            cb,
        )
        .unwrap();
        cb.commit();
        cb.wait_until_completed();
        let gqa_f32 = read_f16_as_f32(registry, queue, &gqa_out);

        // --- Reference: per-head sdpa path ---
        // Split slabs into per-head arrays via views
        let _dev = registry.device().raw();
        let head_q_size = seq_len * head_dim;
        let head_kv_size = kv_len * head_dim;
        let elem_bytes = 2usize; // f16

        let mut q_heads = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let offset = h * head_q_size * elem_bytes;
            let view = q_slab.view(vec![seq_len, head_dim], vec![head_dim, 1], offset);
            q_heads.push(view);
        }

        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for kh in 0..num_kv_heads {
            let k_offset = kh * head_kv_size * elem_bytes;
            let v_offset = kh * head_kv_size * elem_bytes;
            k_heads.push(k_slab.view(vec![kv_len, head_dim], vec![head_dim, 1], k_offset));
            v_heads.push(v_slab.view(vec![kv_len, head_dim], vec![head_dim, 1], v_offset));
        }

        let ref_outputs = sdpa_batched(
            registry, &q_heads, &k_heads, &v_heads, None, scale, is_causal, queue,
        )
        .unwrap();

        // Concatenate reference outputs into a flat f32 vec
        let mut ref_f32 = Vec::with_capacity(num_heads * head_q_size);
        for out in &ref_outputs {
            let f32_out = read_f16_as_f32(registry, queue, out);
            ref_f32.extend_from_slice(&f32_out);
        }

        (gqa_f32, ref_f32)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Reorder head-major [num_heads, seq_len, head_dim] → seq-major [seq_len, num_heads * head_dim]
    fn head_major_to_seq_major(
        data: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src = h * seq_len * head_dim + s * head_dim + d;
                    let dst = s * num_heads * head_dim + h * head_dim + d;
                    out[dst] = data[src];
                }
            }
        }
        out
    }

    fn read_f32_array(arr: &Array) -> Vec<f32> {
        let ptr = arr.metal_buffer().contents() as *const f32;
        let len = arr.numel();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn test_gqa_prefill_ratio4_seq64() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(
            &registry, &queue, 32,  // num_heads
            8,   // num_kv_heads (ratio=4)
            64,  // seq_len
            64,  // kv_len
            128, // head_dim
            false,
        );
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 1e-2,
            "GQA ratio=4 seq=64: max_abs_diff={diff} (expected < 1e-2)"
        );
    }

    #[test]
    fn test_gqa_prefill_ratio1_seq64() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(
            &registry, &queue, 8,   // num_heads
            8,   // num_kv_heads (ratio=1, non-GQA)
            64,  // seq_len
            64,  // kv_len
            128, // head_dim
            false,
        );
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 1e-2,
            "GQA ratio=1 seq=64: max_abs_diff={diff} (expected < 1e-2)"
        );
    }

    #[test]
    fn test_gqa_prefill_ratio4_seq33_unaligned() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(
            &registry, &queue, 32, 8, 33, // seq_len — not aligned to BR=16
            33, 128, false,
        );
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 1e-2,
            "GQA ratio=4 seq=33 (unaligned): max_abs_diff={diff} (expected < 1e-2)"
        );
    }

    #[test]
    fn test_gqa_prefill_ratio4_seq128() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(&registry, &queue, 32, 8, 128, 128, 128, false);
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 8e-2,
            "GQA ratio=4 seq=128: max_abs_diff={diff} (expected < 8e-2)"
        );
    }

    #[test]
    fn test_gqa_prefill_ratio4_seq257_unaligned() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(
            &registry, &queue, 32, 8, 257, // seq_len — not aligned to BR=16 or BC=128
            257, 128, false,
        );
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 1.5e-2,
            "GQA ratio=4 seq=257 (unaligned): max_abs_diff={diff} (expected < 1.5e-2)"
        );
    }

    #[test]
    fn test_gqa_prefill_ratio4_causal() {
        let (registry, queue) = setup_with_copy();
        let (gqa, reference) = run_gqa_vs_reference(
            &registry, &queue, 32, 8, 64, 64, 128, true, // causal
        );
        let diff = max_abs_diff(&gqa, &reference);
        assert!(
            diff < 1e-2,
            "GQA ratio=4 seq=64 causal: max_abs_diff={diff} (expected < 1e-2)"
        );
    }

    // --- MMA vs Scalar SDPA correctness tests ------------------------------------

    /// Run both scalar (GQA) and MMA prefill kernels with the same inputs,
    /// return (scalar_f32, mma_f32) for element-wise comparison.
    #[allow(clippy::too_many_arguments)]
    fn run_mma_vs_scalar(
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Generate deterministic pseudo-random data
        let q_data = pseudo_random(num_heads * seq_len * head_dim, 42);
        let k_data = pseudo_random(num_kv_heads * max_seq * head_dim, 137);
        let v_data = pseudo_random(num_kv_heads * max_seq * head_dim, 256);

        // Create f16 slabs
        let q_slab = make_f16_array(
            registry,
            queue,
            &q_data,
            vec![num_heads * seq_len * head_dim],
        );
        let k_slab = make_f16_array(
            registry,
            queue,
            &k_data,
            vec![num_kv_heads * max_seq * head_dim],
        );
        let v_slab = make_f16_array(
            registry,
            queue,
            &v_data,
            vec![num_kv_heads * max_seq * head_dim],
        );

        // --- Scalar kernel ---
        let cb_scalar = queue.new_command_buffer();
        let scalar_out = sdpa_prefill_gqa_slab_into_cb(
            registry,
            &q_slab,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            seq_len,       // kv_len = seq_len (causal prefill)
            Some(max_seq), // kv_seq_stride = max_seq (slab stride)
            None,          // mask
            scale,
            true, // is_causal
            cb_scalar,
        )
        .unwrap();
        cb_scalar.commit();
        cb_scalar.wait_until_completed();
        let scalar_f32_raw = read_f16_as_f32(registry, queue, &scalar_out);
        // Convert scalar head-major output to seq-major for comparison with MMA
        let scalar_f32 = head_major_to_seq_major(&scalar_f32_raw, num_heads, seq_len, head_dim);

        // --- MMA kernel (outputs seq-major) ---
        let cb_mma = queue.new_command_buffer();
        let mma_out = sdpa_prefill_mma_f16_into_cb(
            registry,
            &q_slab,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            seq_len,       // kv_len = seq_len
            Some(max_seq), // kv_seq_stride = max_seq
            None,          // mask
            scale,
            true, // is_causal
            None, // v_head_stride (default)
            None, // v_row_stride (default)
            cb_mma,
        )
        .unwrap();
        cb_mma.commit();
        cb_mma.wait_until_completed();
        let mma_f32 = read_f16_as_f32(registry, queue, &mma_out);

        (scalar_f32, mma_f32)
    }

    #[test]
    fn test_sdpa_mma_vs_scalar_correctness() {
        let (registry, queue) = setup_with_copy();

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 128;
        let max_seq = 1024;

        for seq_len in [32, 64, 128, 256, 512] {
            let (scalar, mma) = run_mma_vs_scalar(
                &registry,
                &queue,
                num_heads,
                num_kv_heads,
                seq_len,
                head_dim,
                max_seq,
            );
            let diff = max_abs_diff(&scalar, &mma);
            // MMA and scalar are both f16 — expect small numerical differences
            // from different accumulation order. Tolerance scales with seq_len
            // because longer sequences accumulate more softmax error.
            let tol = if seq_len <= 128 { 2e-2 } else { 5e-2 };
            assert!(
                diff < tol,
                "MMA vs Scalar seq_len={seq_len}: max_abs_diff={diff} (expected < {tol})"
            );
            eprintln!("  seq_len={seq_len:>4}: max_abs_diff={diff:.6} (tol={tol})",);
        }
    }

    /// Probe whether MetalPerformancePrimitives headers are available at JIT compile time.
    /// This test never panics — it prints the result for diagnostic purposes.
    #[test]
    fn test_mpp_jit_availability() {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let registry = KernelRegistry::new(gpu_dev);

        // --- Probe 1: MPP header inclusion ---
        let mpp_source = r#"
#include <metal_stdlib>
using namespace metal;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

kernel void mpp_probe(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = 1.0f;
}
"#;

        let mpp_result = registry.register_jit_source("mpp_probe", mpp_source);
        match &mpp_result {
            Ok(()) => eprintln!("[MPP probe] SUCCESS — MetalPerformancePrimitives header is available at JIT compile time"),
            Err(e) => eprintln!("[MPP probe] FAILED — MetalPerformancePrimitives header NOT available: {e}"),
        }

        // --- Probe 2: Baseline Metal 3.2 simdgroup_matrix (no MPP) ---
        let baseline_source = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void metal32_probe(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    simdgroup_float8x8 mat;
    simdgroup_load(mat, out, 8);
    simdgroup_store(mat, out, 8);
    out[tid] = 1.0f;
}
"#;

        let baseline_result = registry.register_jit_source("metal32_probe", baseline_source);
        match &baseline_result {
            Ok(()) => eprintln!("[Metal 3.2 baseline] SUCCESS — simdgroup_matrix compiles OK"),
            Err(e) => {
                eprintln!("[Metal 3.2 baseline] FAILED — simdgroup_matrix compilation error: {e}")
            }
        }

        // Summary
        eprintln!("---");
        eprintln!(
            "MPP available: {}  |  Metal 3.2 baseline: {}",
            mpp_result.is_ok(),
            baseline_result.is_ok()
        );
    }

    #[test]
    fn test_sdpa_mma_bk32_vs_scalar_correctness() {
        let (registry, queue) = setup_with_copy();

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 128;
        let max_seq = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for seq_len in [32, 64, 128, 256, 512] {
            let q_data = pseudo_random(num_heads * seq_len * head_dim, 42);
            let k_data = pseudo_random(num_kv_heads * max_seq * head_dim, 137);
            let v_data = pseudo_random(num_kv_heads * max_seq * head_dim, 256);

            let q_slab = make_f16_array(
                &registry,
                &queue,
                &q_data,
                vec![num_heads * seq_len * head_dim],
            );
            let k_slab = make_f16_array(
                &registry,
                &queue,
                &k_data,
                vec![num_kv_heads * max_seq * head_dim],
            );
            let v_slab = make_f16_array(
                &registry,
                &queue,
                &v_data,
                vec![num_kv_heads * max_seq * head_dim],
            );

            // --- Scalar kernel ---
            let cb_scalar = queue.new_command_buffer();
            let scalar_out = sdpa_prefill_gqa_slab_into_cb(
                &registry,
                &q_slab,
                &k_slab,
                &v_slab,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                seq_len,
                Some(max_seq),
                None,
                scale,
                true,
                cb_scalar,
            )
            .unwrap();
            cb_scalar.commit();
            cb_scalar.wait_until_completed();
            let scalar_f32_raw = read_f16_as_f32(&registry, &queue, &scalar_out);
            // Convert scalar head-major output to seq-major for comparison with MMA
            let scalar_f32 = head_major_to_seq_major(&scalar_f32_raw, num_heads, seq_len, head_dim);

            // --- MMA BK=32 kernel (outputs seq-major) ---
            let cb_mma = queue.new_command_buffer();
            let mma_out = sdpa_prefill_mma_bk32_f16_into_cb(
                &registry,
                &q_slab,
                &k_slab,
                &v_slab,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                seq_len,
                Some(max_seq),
                None,
                scale,
                true,
                None, // v_head_stride
                None, // v_row_stride
                cb_mma,
            )
            .unwrap();
            cb_mma.commit();
            cb_mma.wait_until_completed();
            let mma_f32 = read_f16_as_f32(&registry, &queue, &mma_out);

            let diff = max_abs_diff(&scalar_f32, &mma_f32);
            let tol = if seq_len <= 128 { 2e-2 } else { 5e-2 };
            assert!(
                diff < tol,
                "MMA BK32 vs Scalar seq_len={seq_len}: max_abs_diff={diff} (expected < {tol})"
            );
            eprintln!("  BK32 seq_len={seq_len:>4}: max_abs_diff={diff:.6} (tol={tol})");
        }
    }

    #[test]
    #[ignore] // NAX ct_c element mapping not yet resolved
    fn test_sdpa_nax_diag_qkt() {
        let (registry, queue) = setup_with_copy();

        // Minimal: Q=[16,16], K=[32,16], S=[16,32]
        let q_data = pseudo_random(16 * 16, 42);
        let k_data = pseudo_random(32 * 16, 137);

        let q = make_f16_array(&registry, &queue, &q_data, vec![16 * 16]);
        let k = make_f16_array(&registry, &queue, &k_data, vec![32 * 16]);

        // GPU
        let cb = queue.new_command_buffer();
        let s_out = sdpa_nax_diag_single_mma_into_cb(&registry, &q, &k, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        let s_gpu = read_f32_array(&s_out);

        // CPU reference: S = Q @ K^T where Q=[16,16], K=[32,16]
        let q_f16 = read_f16_as_f32(&registry, &queue, &q);
        let k_f16 = read_f16_as_f32(&registry, &queue, &k);

        let mut s_ref = vec![0.0f32; 16 * 32];
        for i in 0..16 {
            for j in 0..32 {
                let mut dot = 0.0f32;
                for d in 0..16 {
                    dot += q_f16[i * 16 + d] * k_f16[j * 16 + d];
                }
                s_ref[i * 32 + j] = dot;
            }
        }

        let diff = max_abs_diff(&s_ref, &s_gpu);
        eprintln!("NAX single MMA Q@K^T: max_abs_diff={diff:.6}");
        eprintln!("  GPU S[0,0..8]: {:?}", &s_gpu[0..8]);
        eprintln!("  REF S[0,0..8]: {:?}", &s_ref[0..8]);
        eprintln!("  GPU S[1,0..8]: {:?}", &s_gpu[32..40]);
        eprintln!("  REF S[1,0..8]: {:?}", &s_ref[32..40]);

        // Also print S[0, 16..24] to check second sub-frag
        eprintln!("  GPU S[0,16..24]: {:?}", &s_gpu[16..24]);
        eprintln!("  REF S[0,16..24]: {:?}", &s_ref[16..24]);

        // === All-ones test: Q=1, K=1, expect S=16 everywhere ===
        {
            let ones_q: Vec<f32> = vec![1.0; 16 * 16];
            let ones_k: Vec<f32> = vec![1.0; 32 * 16];

            let q_ones = make_f16_array(&registry, &queue, &ones_q, vec![16 * 16]);
            let k_ones = make_f16_array(&registry, &queue, &ones_k, vec![32 * 16]);

            let cb2 = queue.new_command_buffer();
            let s_ones =
                sdpa_nax_diag_single_mma_into_cb(&registry, &q_ones, &k_ones, cb2).unwrap();
            cb2.commit();
            cb2.wait_until_completed();
            let s_ones_gpu = read_f32_array(&s_ones);

            eprintln!("\n=== All-ones test (expect 16.0 everywhere) ===");
            eprintln!("  GPU S[0,0..8]: {:?}", &s_ones_gpu[0..8]);
            eprintln!("  GPU S[1,0..8]: {:?}", &s_ones_gpu[32..40]);
            eprintln!("  GPU S[0,16..24]: {:?}", &s_ones_gpu[16..24]);
            eprintln!(
                "  GPU S[15,24..32]: {:?}",
                &s_ones_gpu[15 * 32 + 24..15 * 32 + 32]
            );

            let expected_16: Vec<f32> = vec![16.0; 16 * 32];
            let ones_diff = max_abs_diff(&expected_16, &s_ones_gpu);
            eprintln!("  max_abs_diff from 16.0: {ones_diff:.6}");
        }

        // === Identity K test: K=I → S=Q@I^T → S[:,:16]=Q, S[:,16:]=0 ===
        {
            // Q[i,j] = (i * 16 + j) as f32 — each element encodes its position
            let mut pos_q = vec![0.0f32; 16 * 16];
            for i in 0..16 {
                for j in 0..16 {
                    pos_q[i * 16 + j] = (i * 16 + j) as f32;
                }
            }
            // K[j,d] = delta(j,d) for j<16, K[j,d]=0 for j>=16
            let mut id_k = vec![0.0f32; 32 * 16];
            for d in 0..16 {
                id_k[d * 16 + d] = 1.0;
            }

            let q_pos = make_f16_array(&registry, &queue, &pos_q, vec![16 * 16]);
            let k_id = make_f16_array(&registry, &queue, &id_k, vec![32 * 16]);

            let cb3 = queue.new_command_buffer();
            let s_id = sdpa_nax_diag_single_mma_into_cb(&registry, &q_pos, &k_id, cb3).unwrap();
            cb3.commit();
            cb3.wait_until_completed();
            let s_id_gpu = read_f32_array(&s_id);

            eprintln!("\n=== Identity K test (S[:,:16] should equal Q, S[:,16:]=0) ===");
            // Print first few rows of S[:,:16]
            eprintln!("  S[0,0..16]:  {:?}", &s_id_gpu[0..16]);
            eprintln!("  Expected:    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]");
            eprintln!("  S[1,0..16]:  {:?}", &s_id_gpu[32..48]);
            eprintln!("  Expected:    [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]");
            eprintln!("  S[2,0..16]:  {:?}", &s_id_gpu[64..80]);
            eprintln!("  Expected:    [32,33,...,47]");
            // Print S[:,16:] (should be 0)
            eprintln!("  S[0,16..32]: {:?}", &s_id_gpu[16..32]);
            eprintln!("  Expected:    all zeros");

            // Check first column: S[i,0] should be Q[i,0] = i*16
            eprintln!("\n  Column 0 check (S[i,0] should be i*16):");
            for i in 0..16 {
                let actual = s_id_gpu[i * 32];
                let expected = (i * 16) as f32;
                let marker = if (actual - expected).abs() < 0.1 {
                    "OK"
                } else {
                    "WRONG"
                };
                eprintln!("    S[{i},0] = {actual:.1} (expected {expected:.1}) {marker}");
            }
        }

        assert!(
            diff < 0.05,
            "NAX single MMA Q@K^T: max_abs_diff={diff} (expected < 0.05)"
        );
    }

    #[test]
    #[ignore] // NAX ct_c element mapping not yet resolved
    fn test_sdpa_nax_vs_scalar_correctness() {
        let (registry, queue) = setup_with_copy();

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 128;
        let max_seq = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for seq_len in [64, 128, 256, 512] {
            let q_data = pseudo_random(num_heads * seq_len * head_dim, 42);
            let k_data = pseudo_random(num_kv_heads * max_seq * head_dim, 137);
            let v_data = pseudo_random(num_kv_heads * max_seq * head_dim, 256);

            let q_slab = make_f16_array(
                &registry,
                &queue,
                &q_data,
                vec![num_heads * seq_len * head_dim],
            );
            let k_slab = make_f16_array(
                &registry,
                &queue,
                &k_data,
                vec![num_kv_heads * max_seq * head_dim],
            );
            let v_slab = make_f16_array(
                &registry,
                &queue,
                &v_data,
                vec![num_kv_heads * max_seq * head_dim],
            );

            // --- Scalar kernel ---
            let cb_scalar = queue.new_command_buffer();
            let scalar_out = sdpa_prefill_gqa_slab_into_cb(
                &registry,
                &q_slab,
                &k_slab,
                &v_slab,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                seq_len,
                Some(max_seq),
                None,
                scale,
                true,
                cb_scalar,
            )
            .unwrap();
            cb_scalar.commit();
            cb_scalar.wait_until_completed();
            let scalar_f32_raw = read_f16_as_f32(&registry, &queue, &scalar_out);
            // Convert scalar head-major output to seq-major for comparison with NAX
            let scalar_f32 = head_major_to_seq_major(&scalar_f32_raw, num_heads, seq_len, head_dim);

            // --- NAX kernel ---
            let cb_nax = queue.new_command_buffer();
            let nax_out = sdpa_prefill_nax_f16_into_cb(
                &registry,
                &q_slab,
                &k_slab,
                &v_slab,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                seq_len,
                Some(max_seq),
                scale,
                true,
                None, // v_head_stride
                None, // v_row_stride
                cb_nax,
            )
            .unwrap();
            cb_nax.commit();
            cb_nax.wait_until_completed();
            let nax_f32 = read_f16_as_f32(&registry, &queue, &nax_out);

            let diff = max_abs_diff(&scalar_f32, &nax_f32);
            let tol = if seq_len <= 128 { 2e-2 } else { 5e-2 };
            assert!(
                diff < tol,
                "NAX vs Scalar seq_len={seq_len}: max_abs_diff={diff} (expected < {tol})"
            );
            eprintln!("  NAX seq_len={seq_len:>4}: max_abs_diff={diff:.6} (tol={tol})");
        }
    }
}
