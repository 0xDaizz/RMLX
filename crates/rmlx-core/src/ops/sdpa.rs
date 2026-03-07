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
use metal::MTLSize;

// Tiling parameters — must match the Metal shader constants.
const BR: usize = 16; // Query block rows
const _BC: usize = 16; // Key/Value block columns (used in shader only)
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
constant constexpr uint Bc = 16;   // Key/Value block columns
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

// ─── sdpa_decode_f32 ──────────────────────────────────────────────────────
//
// Optimised single-query path (N == 1). No Q tile or score tile needed in
// shared memory — Q is loaded once into registers, and we iterate over K/V
// blocks with online softmax.
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
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];
    // is_causal not needed for N=1: the single query attends to all keys

    const uint n_threads = 256;

    // Shared memory for reductions and O accumulator
    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];  // D <= 256

    // Initialise output accumulator in shared memory
    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    // Process K/V in blocks of Bc
    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        // Each thread computes a subset of the Bc dot products
        // Thread tid handles key index (tid) if tid < kv_count
        // With 256 threads and Bc=32, we have plenty of threads per key.
        // Strategy: each thread computes one score by looping over D.
        // Only threads with tid < kv_count compute a valid score.

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += Q[d] * K[(kv_start + tid) * D + d];
            }
            dot *= scale;
            if (has_mask) {
                dot += mask[kv_start + tid];
            }
            score = dot;
        }

        // Find block max across all threads
        float block_max = tg_reduce_max(score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);

        // Online softmax update
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);

        // Compute exp(score - m_new) for this thread's key
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;

        // Sum exp scores across threads
        float block_sum = tg_reduce_sum(exp_score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);

        float l_new = l_running * correction + block_sum;

        // Update O: rescale old accumulator and add new contribution
        // Each thread handles a subset of D dimensions
        // We need each thread's exp_score, but we also need to sum over keys.
        // With Bc=32 and n_threads=256, threads 0..31 have valid exp_scores.
        // We need to broadcast V[j,:] * exp_score[j] across D.

        // Store exp_scores in shared memory for broadcasting
        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        // Need scores from threads 0..Bc-1 — they are already computed above.
        // But only threads tid < kv_count have valid scores. With Bc=32 and
        // n_threads=256, threads 0..31 hold the scores for this block.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Now update O_shared: each thread handles a stripe of D
        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * V[(kv_start + j) * D + d];
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    // Final normalisation
    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        O[d] = O_shared[d] * inv_l;
    }
}

// ─── sdpa_decode_f16 ──────────────────────────────────────────────────────
// Same as decode_f32 but reads/writes half, accumulates in float.

kernel void sdpa_decode_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];

    const uint n_threads = 256;

    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];

    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += float(Q[d]) * float(K[(kv_start + tid) * D + d]);
            }
            dot *= scale;
            if (has_mask) {
                dot += float(mask[kv_start + tid]);
            }
            score = dot;
        }

        float block_max = tg_reduce_max(score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;
        float block_sum = tg_reduce_sum(exp_score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float l_new = l_running * correction + block_sum;

        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * float(V[(kv_start + j) * D + d]);
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        O[d] = half(O_shared[d] * inv_l);
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
// Same FA2 algorithm as sdpa_f32, but reads/writes half, accumulates in float.

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

    threadgroup float Q_tile[Br * 128];
    threadgroup float S_tile[Br * Bc];
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

    // Load Q as float (first D_lo dims)
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint r = idx / D_lo;
        uint d = idx % D_lo;
        Q_tile[r * D_lo + d] = float(Q[(q_start + r) * D + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (S + Bc - 1) / Bc;
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc) + 1)
        : n_kv_blocks;

    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        // Compute S = Q @ K^T * scale (K read as float from global)
        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;
            uint j = idx % kv_count;
            float dot = 0.0f;

            for (uint d = 0; d < D_lo; d++) {
                dot += Q_tile[i * D_lo + d] * float(K[(kv_start + j) * D + d]);
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

        // Online softmax + accumulate
        for (uint i = 0; i < q_count; i++) {
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc + j]);
            }
            float m_new = tg_reduce_max(local_max, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
            m_new = max(m_prev[i], m_new);

            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc + j] - m_new);
                S_tile[i * Bc + j] = e;
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
                    v_sum += S_tile[i * Bc + j] * float(V[(kv_start + j) * D + d]);
                }
                O_acc[i * D_lo + d] = o_val + v_sum;
            }
            for (uint d = tid; d < D_hi; d += n_threads) {
                float o_val = O_acc2[i * D_hi + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * float(V[(kv_start + j) * D + (D_lo + d)]);
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
// Batched multi-head decode: processes ALL attention heads in a single
// GPU dispatch.  Grid.x = num_heads, one threadgroup per head.
// Supports GQA (grouped query attention).
//
// Buffers:
//   0: Q      [num_heads * D]              — flat query slab
//   1: K      [num_kv_heads * S * D]       — flat key slab
//   2: V      [num_kv_heads * S * D]       — flat value slab
//   3: O      [num_heads * D]              — flat output slab
//   4: mask   [S]                          — additive mask (shared, or dummy)
//   5: params [6 x uint32]: { num_heads, num_kv_heads, S, D, has_mask, stride_S }
//      stride_S: inter-head stride in KV slab (0 = same as S, >0 = max_seq_len for pre-allocated caches)
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
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]],
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

    // GQA: map Q head to KV head
    const uint kv_head = head_id * num_kv_heads / num_heads;

    // Per-head pointers — use stride_S for inter-head offset (slab layout)
    device const float* q = Q + head_id * D;
    device const float* k = K + kv_head * stride_S * D;
    device const float* v = V + kv_head * stride_S * D;
    device       float* o = O + head_id * D;

    const uint n_threads = 256;

    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];  // D <= 256

    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += q[d] * k[(kv_start + tid) * D + d];
            }
            dot *= scale;
            if (has_mask) {
                dot += mask[kv_start + tid];
            }
            score = dot;
        }

        float block_max = tg_reduce_max(score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;
        float block_sum = tg_reduce_sum(exp_score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float l_new = l_running * correction + block_sum;

        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * v[(kv_start + j) * D + d];
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        o[d] = O_shared[d] * inv_l;
    }
}

// ─── sdpa_decode_batched_f16 ──────────────────────────────────────────────
// Same as batched_f32 but reads/writes half, accumulates in float.

kernel void sdpa_decode_batched_f16(
    device const half*  Q         [[buffer(0)]],
    device const half*  K         [[buffer(1)]],
    device const half*  V         [[buffer(2)]],
    device       half*  O         [[buffer(3)]],
    device const half*  mask      [[buffer(4)]],
    constant     uint*  params    [[buffer(5)]],
    constant     float& scale     [[buffer(6)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]],
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
    device       half* o = O + head_id * D;

    const uint n_threads = 256;

    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];

    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += float(q[d]) * float(k[(kv_start + tid) * D + d]);
            }
            dot *= scale;
            if (has_mask) {
                dot += float(mask[kv_start + tid]);
            }
            score = dot;
        }

        float block_max = tg_reduce_max(score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;
        float block_sum = tg_reduce_sum(exp_score, tid, lane_id, sg_id,
                                        n_threads, reduce_buf);
        float l_new = l_running * correction + block_sum;

        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * float(v[(kv_start + j) * D + d]);
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        o[d] = half(O_shared[d] * inv_l);
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
constant constexpr uint Bc = 16;
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

// ─── sdpa_decode_bf16 ─────────────────────────────────────────────────────

kernel void sdpa_decode_bf16(
    device const bfloat* Q         [[buffer(0)]],
    device const bfloat* K         [[buffer(1)]],
    device const bfloat* V         [[buffer(2)]],
    device       bfloat* O         [[buffer(3)]],
    device const bfloat* mask      [[buffer(4)]],
    constant     uint*   params    [[buffer(5)]],
    constant     float&  scale     [[buffer(6)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint S        = params[1];
    const uint D        = FC_HAS_HEAD_DIM ? FC_HEAD_DIM : params[2];
    const uint has_mask = params[3];

    const uint n_threads = 256;

    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];

    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += float(Q[d]) * float(K[(kv_start + tid) * D + d]);
            }
            dot *= scale;
            if (has_mask) {
                dot += float(mask[kv_start + tid]);
            }
            score = dot;
        }

        float block_max = tg_reduce_max_bf16(score, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;
        float block_sum = tg_reduce_sum_bf16(exp_score, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
        float l_new = l_running * correction + block_sum;

        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * float(V[(kv_start + j) * D + d]);
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        O[d] = bfloat(O_shared[d] * inv_l);
    }
}

// ─── sdpa_bf16 ────────────────────────────────────────────────────────────

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

    threadgroup float Q_tile[Br * 128];
    threadgroup float S_tile[Br * Bc];
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

    // Load Q as float
    for (uint idx = tid; idx < q_count * D_lo; idx += n_threads) {
        uint r = idx / D_lo;
        uint d = idx % D_lo;
        Q_tile[r * D_lo + d] = float(Q[(q_start + r) * D + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (S + Bc - 1) / Bc;
    const uint max_kv_block = is_causal
        ? min(n_kv_blocks, ((q_start + q_count - 1) / Bc) + 1)
        : n_kv_blocks;

    for (uint kb = 0; kb < max_kv_block; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        for (uint idx = tid; idx < q_count * kv_count; idx += n_threads) {
            uint i = idx / kv_count;
            uint j = idx % kv_count;
            float dot = 0.0f;

            for (uint d = 0; d < D_lo; d++) {
                dot += Q_tile[i * D_lo + d] * float(K[(kv_start + j) * D + d]);
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

        for (uint i = 0; i < q_count; i++) {
            float local_max = -INFINITY;
            for (uint j = tid; j < kv_count; j += n_threads) {
                local_max = max(local_max, S_tile[i * Bc + j]);
            }
            float m_new = tg_reduce_max_bf16(local_max, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
            m_new = max(m_prev[i], m_new);

            float local_sum = 0.0f;
            for (uint j = tid; j < kv_count; j += n_threads) {
                float e = fast::exp(S_tile[i * Bc + j] - m_new);
                S_tile[i * Bc + j] = e;
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
                    v_sum += S_tile[i * Bc + j] * float(V[(kv_start + j) * D + d]);
                }
                O_acc[i * D_lo + d] = o_val + v_sum;
            }
            for (uint d = tid; d < D_hi; d += n_threads) {
                float o_val = O_acc2[i * D_hi + d] * correction;
                float v_sum = 0.0f;
                for (uint j = 0; j < kv_count; j++) {
                    v_sum += S_tile[i * Bc + j] * float(V[(kv_start + j) * D + (D_lo + d)]);
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
// Batched multi-head decode for bf16. Same as f32 variant but reads/writes
// bfloat, accumulates in float.

kernel void sdpa_decode_batched_bf16(
    device const bfloat* Q         [[buffer(0)]],
    device const bfloat* K         [[buffer(1)]],
    device const bfloat* V         [[buffer(2)]],
    device       bfloat* O         [[buffer(3)]],
    device const bfloat* mask      [[buffer(4)]],
    constant     uint*   params    [[buffer(5)]],
    constant     float&  scale     [[buffer(6)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  lane_id   [[thread_index_in_simdgroup]],
    uint  sg_id     [[simdgroup_index_in_threadgroup]],
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
    device       bfloat* o = O + head_id * D;

    const uint n_threads = 256;

    threadgroup float reduce_buf[SIMD_SIZE];
    threadgroup float O_shared[256];

    for (uint d = tid; d < D; d += n_threads) {
        O_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_running = -INFINITY;
    float l_running = 0.0f;

    const uint n_kv_blocks = (S + Bc - 1) / Bc;

    for (uint kb = 0; kb < n_kv_blocks; kb++) {
        const uint kv_start = kb * Bc;
        const uint kv_end   = min(kv_start + Bc, S);
        const uint kv_count = kv_end - kv_start;

        float score = -INFINITY;
        if (tid < kv_count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; d++) {
                dot += float(q[d]) * float(k[(kv_start + tid) * D + d]);
            }
            dot *= scale;
            if (has_mask) {
                dot += float(mask[kv_start + tid]);
            }
            score = dot;
        }

        float block_max = tg_reduce_max_bf16(score, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
        float m_new = max(m_running, block_max);
        float correction = fast::exp(m_running - m_new);
        float exp_score = (tid < kv_count) ? fast::exp(score - m_new) : 0.0f;
        float block_sum = tg_reduce_sum_bf16(exp_score, tid, lane_id, sg_id,
                                              n_threads, reduce_buf);
        float l_new = l_running * correction + block_sum;

        threadgroup float exp_scores_shared[Bc];
        if (tid < Bc) {
            exp_scores_shared[tid] = (tid < kv_count) ? exp_score : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < D; d += n_threads) {
            float o_val = O_shared[d] * correction;
            float v_sum = 0.0f;
            for (uint j = 0; j < kv_count; j++) {
                v_sum += exp_scores_shared[j] * float(v[(kv_start + j) * D + d]);
            }
            O_shared[d] = o_val + v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_running = m_new;
        l_running = l_new;
    }

    float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    for (uint d = tid; d < D; d += n_threads) {
        o[d] = bfloat(O_shared[d] * inv_l);
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("sdpa", SDPA_SHADER_SOURCE)?;
    registry.register_jit_source("sdpa_bf16", SDPA_BF16_SHADER_SOURCE)?;
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

    // Try FlashAttention-2 for eligible configurations:
    // f32, head_dim=128, no additive mask, seq_len >= threshold.
    if mask.is_none() && super::flash_attention::supports_flash_attention(d, q.dtype(), n) {
        match super::flash_attention::flash_attention_forward(
            registry, q, k, v, scale, is_causal, queue,
        ) {
            Ok(out) => return Ok(out),
            Err(_) => {
                // Fall through to naive SDPA on any flash attention error
            }
        }
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
            let expanded = Array::zeros(dev, &[n, s], m.dtype());
            let src = m.metal_buffer().contents() as *const u8;
            let dst = expanded.metal_buffer().contents() as *mut u8;
            let row_bytes = m
                .dtype()
                .numel_to_bytes(s)
                .expect("numel must be block-aligned");
            // SAFETY: SharedMode buffers are CPU-accessible; bounds checked by
            // Array::zeros allocation and row_bytes computation.
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
    let out = Array::zeros(dev, &[n, d], q.dtype());

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
    encoder.set_buffer(0, Some(q.metal_buffer()), q.offset() as u64);
    encoder.set_buffer(1, Some(k.metal_buffer()), k.offset() as u64);
    encoder.set_buffer(2, Some(v.metal_buffer()), v.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(mask_buf), mask_offset);
    encoder.set_buffer(5, Some(&params_buf), 0);
    encoder.set_buffer(6, Some(&scale_buf), 0);

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
    let is_causal_u32: u32 = 0; // into_cb path does not support causal flag
    let params: [u32; 5] = [n as u32, s as u32, d as u32, has_mask, is_causal_u32];

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
    encoder.set_bytes(5, 20, params.as_ptr() as *const std::ffi::c_void);
    encoder.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);

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

    Ok(out)
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

    let out = Array::zeros(dev, &[q_expected], dtype);

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
    encoder.set_buffer(0, Some(q_slab.metal_buffer()), q_slab.offset() as u64);
    encoder.set_buffer(1, Some(k_slab.metal_buffer()), k_slab.offset() as u64);
    encoder.set_buffer(2, Some(v_slab.metal_buffer()), v_slab.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        5,
        std::mem::size_of::<[u32; 6]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);

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

    let out = Array::zeros(dev, &[q_expected], dtype);

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
    encoder.set_buffer(0, Some(q_slab.metal_buffer()), q_slab.offset() as u64);
    encoder.set_buffer(1, Some(k_slab.metal_buffer()), k_slab.offset() as u64);
    encoder.set_buffer(2, Some(v_slab.metal_buffer()), v_slab.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(mask_buf), mask_offset);
    encoder.set_bytes(
        5,
        std::mem::size_of::<[u32; 6]>() as u64,
        params.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);

    // One threadgroup per Q head, 256 threads each
    let tg_size = std::cmp::min(DECODE_THREADS, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
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
}
