//! Matrix multiplication with tiled GEMM, GEMV auto-dispatch, and batch support.
//!
//! Uses tiled GEMM kernels with threadgroup (shared) memory and simdgroup matrix
//! operations on Apple Silicon.
//!
//! Kernel variants:
//! - **HiPerf GEMM v2** (BM=64, BN=64, BK=16): 128 threads (4 simdgroups),
//!   2×2 SG grid with 4×4 MMA per SG (16 accumulators), double-buffered 8KB shmem.
//!   Used for f16 when M >= 33, N >= 33. Combines deeper MMA pipeline with
//!   memory latency hiding via double buffering.
//! - **Full tile GEMM** (BM=64, BN=64, BK=32): 256 threads (8 simdgroups),
//!   half-precision shared memory, double buffering, simdgroup MMA.
//!   Used for f32/bf16 when M >= 33.
//! - **Skinny GEMM** (BM=32, BN=128, BK=32): optimized for M=5..32 (small
//!   batch prefill) with high N-direction parallelism.
//! - **Small tile GEMM** (BM=16, BN=16, BK=16): for tiny matrices (M,N < 33).
//! - **GEMV** auto-dispatch for M<=4 (single-token decode hot path).
//!
//! Split-K support: for K-dominated problems (K > 4*max(M,N)), a two-pass Split-K
//! kernel partitions K into chunks, computing partial sums in pass 1 and reducing
//! them in pass 2.
//!
//! Supports:
//! - f32, f16, bf16 (f16/bf16 accumulate in f32, read/write in native precision)
//! - Batched matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
//! - Auto GEMV dispatch for M<=4 or N=1 (single-token decode hot path)

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source: optimized GEMM with simdgroup MMA, f16 shmem,
// double buffering. BM=64, BN=64, BK=32.
// ---------------------------------------------------------------------------

pub const GEMM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===== Full-tile GEMM: BM=64, BN=64, BK=32 =====
// 256 threads = 8 simdgroups.
// Each simdgroup handles a 16x16 sub-tile of C (2x2 grid of 8x8 MMA tiles).
// Simdgroups laid out as 2 rows x 4 cols over the 64x64 output tile.
// Shared memory: half A[2][64*32] + half B[2][32*64] = 2 * 2 * (64*32*2) = 16KB
// (double-buffered, half precision) — fits in 32KB threadgroup memory.
//
// Double buffering: while computing on buffer[stage], the next tile is being
// loaded into buffer[1-stage]. This hides global memory latency.

constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 32;
constant constexpr uint N_SIMDGROUPS = 8;  // 256 / 32
constant constexpr uint SG_ROWS = 2;       // simdgroup grid rows
constant constexpr uint SG_COLS = 4;       // simdgroup grid cols
constant constexpr uint N_THREADS = 256;

// Uniform hint for Metal 3.1+ (M3 and later)
#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T as_uniform(T val) {
    return val;
}
#endif

inline uint2 swizzle_threadgroup(uint2 tid, uint swizzle_log) {
    if (swizzle_log == 0) return tid;
    return uint2(
        tid.x >> swizzle_log,
        (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))
    );
}

// ---------------------------------------------------------------------------
// f32 full-tile GEMM: BM=64, BN=64, BK=32 with simdgroup MMA
// ---------------------------------------------------------------------------
kernel void gemm_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    // Double-buffered shared memory (f32 for f32 path)
    threadgroup float As[2][BM * BK];  // 2 * 64 * 32 * 4 = 16KB
    threadgroup float Bs[2][BK * BN];  // 2 * 32 * 64 * 4 = 16KB

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(BM);
    const uint col_start = swizzled.x * as_uniform(BN);

    device const float* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const float* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device float*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    // Simdgroup 2D position: 2 rows x 4 cols
    const uint sg_row = sgid / SG_COLS;  // 0..1
    const uint sg_col = sgid % SG_COLS;  // 0..3

    // Each simdgroup computes 2x2 grid of 8x8 MMA tiles = 16x16 output
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    // Prefetch first tile (stage 0)
    {
        for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = row_start + r;
            uint gc = c;
            As[0][r * BK + c] = (gr < uM && gc < uK) ? A_batch[gr * uK + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += N_THREADS) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = r;
            uint gc = col_start + c;
            Bs[0][r * BN + c] = (gr < uK && gc < uN) ? B_batch[gr * uN + gc] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_tiles = (uK + BK - 1) / BK;
    for (uint tile = 0; tile < n_tiles; tile++) {
        uint stage = tile & 1;
        uint next_stage = 1 - stage;
        uint next_kb = (tile + 1) * BK;

        // Prefetch next tile into next_stage (if there is one)
        if (tile + 1 < n_tiles) {
            for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
                uint r = idx / BK;
                uint c = idx % BK;
                uint gr = row_start + r;
                uint gc = next_kb + c;
                As[next_stage][r * BK + c] = (gr < uM && gc < uK)
                    ? A_batch[gr * uK + gc] : 0.0f;
            }
            for (uint idx = tid_in_group; idx < BK * BN; idx += N_THREADS) {
                uint r = idx / BN;
                uint c = idx % BN;
                uint gr = next_kb + r;
                uint gc = col_start + c;
                Bs[next_stage][r * BN + c] = (gr < uK && gc < uN)
                    ? B_batch[gr * uN + gc] : 0.0f;
            }
        }

        // Compute on current stage
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a_frag[2];
            simdgroup_float8x8 b_frag[2];

            // Load A sub-tiles: sg_row * 16 + {0,8} rows, kk col
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_frag[i],
                    &As[stage][(sg_row * 16 + i * 8) * BK + kk], BK);
            }
            // Load B sub-tiles: kk row, sg_col * 16 + {0,8} cols
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j],
                    &Bs[stage][kk * BN + sg_col * 16 + j * 8], BN);
            }

            // 2x2 outer product
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        if (tile + 1 < n_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store 2x2 grid of 8x8 results
    // Use threadgroup memory for bounds-checked store
    threadgroup float result_buf[N_SIMDGROUPS * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + sg_row * 16 + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = result_buf[sgid * 64 + lr * 8 + lc];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// f16 full-tile GEMM: BM=64, BN=64, BK=32 — half shmem, f32 accumulators
// ---------------------------------------------------------------------------
kernel void gemm_tiled_f16(
    device const half* A  [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device half* C        [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    // Double-buffered shared memory in half precision
    // 2 * (64*32 + 32*64) * 2 bytes = 2 * 4096 * 2 = 16KB — fits in 32KB
    threadgroup half As[2][BM * BK];
    threadgroup half Bs[2][BK * BN];

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(BM);
    const uint col_start = swizzled.x * as_uniform(BN);

    device const half* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const half* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device half*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    const uint sg_row = sgid / SG_COLS;
    const uint sg_col = sgid % SG_COLS;

    // f32 accumulators for numerical stability
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    // Prefetch first tile (vectorized half4 loads)
    {
        for (uint idx = tid_in_group; idx < (BM * BK) / 4; idx += N_THREADS) {
            uint flat = idx * 4;
            uint r = flat / BK;
            uint c = flat % BK;
            uint gr = row_start + r;
            uint gc = c;
            half4 val;
            if (gr < uM && gc + 3 < uK) {
                val = *((device const half4*)(&A_batch[gr * uK + gc]));
            } else {
                val = half4(
                    (gr < uM && gc+0 < uK) ? A_batch[gr*uK + gc+0] : half(0),
                    (gr < uM && gc+1 < uK) ? A_batch[gr*uK + gc+1] : half(0),
                    (gr < uM && gc+2 < uK) ? A_batch[gr*uK + gc+2] : half(0),
                    (gr < uM && gc+3 < uK) ? A_batch[gr*uK + gc+3] : half(0)
                );
            }
            *((threadgroup half4*)(&As[0][r * BK + c])) = val;
        }
        for (uint idx = tid_in_group; idx < (BK * BN) / 4; idx += N_THREADS) {
            uint flat = idx * 4;
            uint r = flat / BN;
            uint c = flat % BN;
            uint gr = r;
            uint gc = col_start + c;
            half4 val;
            if (gr < uK && gc + 3 < uN) {
                val = *((device const half4*)(&B_batch[gr * uN + gc]));
            } else {
                val = half4(
                    (gr < uK && gc+0 < uN) ? B_batch[gr*uN + gc+0] : half(0),
                    (gr < uK && gc+1 < uN) ? B_batch[gr*uN + gc+1] : half(0),
                    (gr < uK && gc+2 < uN) ? B_batch[gr*uN + gc+2] : half(0),
                    (gr < uK && gc+3 < uN) ? B_batch[gr*uN + gc+3] : half(0)
                );
            }
            *((threadgroup half4*)(&Bs[0][r * BN + c])) = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_tiles = (uK + BK - 1) / BK;
    for (uint tile = 0; tile < n_tiles; tile++) {
        uint stage = tile & 1;
        uint next_stage = 1 - stage;
        uint next_kb = (tile + 1) * BK;

        if (tile + 1 < n_tiles) {
            for (uint idx = tid_in_group; idx < (BM * BK) / 4; idx += N_THREADS) {
                uint flat = idx * 4;
                uint r = flat / BK;
                uint c = flat % BK;
                uint gr = row_start + r;
                uint gc = next_kb + c;
                half4 val;
                if (gr < uM && gc + 3 < uK) {
                    val = *((device const half4*)(&A_batch[gr * uK + gc]));
                } else {
                    val = half4(
                        (gr < uM && gc+0 < uK) ? A_batch[gr*uK + gc+0] : half(0),
                        (gr < uM && gc+1 < uK) ? A_batch[gr*uK + gc+1] : half(0),
                        (gr < uM && gc+2 < uK) ? A_batch[gr*uK + gc+2] : half(0),
                        (gr < uM && gc+3 < uK) ? A_batch[gr*uK + gc+3] : half(0)
                    );
                }
                *((threadgroup half4*)(&As[next_stage][r * BK + c])) = val;
            }
            for (uint idx = tid_in_group; idx < (BK * BN) / 4; idx += N_THREADS) {
                uint flat = idx * 4;
                uint r = flat / BN;
                uint c = flat % BN;
                uint gr = next_kb + r;
                uint gc = col_start + c;
                half4 val;
                if (gr < uK && gc + 3 < uN) {
                    val = *((device const half4*)(&B_batch[gr * uN + gc]));
                } else {
                    val = half4(
                        (gr < uK && gc+0 < uN) ? B_batch[gr*uN + gc+0] : half(0),
                        (gr < uK && gc+1 < uN) ? B_batch[gr*uN + gc+1] : half(0),
                        (gr < uK && gc+2 < uN) ? B_batch[gr*uN + gc+2] : half(0),
                        (gr < uK && gc+3 < uN) ? B_batch[gr*uN + gc+3] : half(0)
                    );
                }
                *((threadgroup half4*)(&Bs[next_stage][r * BN + c])) = val;
            }
        }

        // Compute using half-precision simdgroup loads, f32 accumulation
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 a_frag[2];
            simdgroup_half8x8 b_frag[2];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_frag[i],
                    &As[stage][(sg_row * 16 + i * 8) * BK + kk], BK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j],
                    &Bs[stage][kk * BN + sg_col * 16 + j * 8], BN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        if (tile + 1 < n_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results — convert f32 acc to half output
    // Fast path: simdgroup_store directly to device memory when fully in-bounds
    // Slow path: bounds-checked store via threadgroup scratch
    threadgroup float result_buf[N_SIMDGROUPS * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            uint base_row = row_start + sg_row * 16 + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            if (base_row + 7 < uM && base_col + 7 < uN) {
                // Fast path: entire 8x8 block is in-bounds
                // Store f32 acc to threadgroup, convert to half, write to device
                simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint idx = lane_id; idx < 64; idx += 32) {
                    uint lr = idx / 8;
                    uint lc = idx % 8;
                    C_batch[(base_row + lr) * uN + (base_col + lc)] =
                        half(result_buf[sgid * 64 + lr * 8 + lc]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                // Slow path: boundary tile — per-element bounds check
                simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint idx = lane_id; idx < 64; idx += 32) {
                    uint lr = idx / 8;
                    uint lc = idx % 8;
                    uint gr = base_row + lr;
                    uint gc = base_col + lc;
                    if (gr < uM && gc < uN) {
                        C_batch[gr * uN + gc] = half(result_buf[sgid * 64 + lr * 8 + lc]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bf16 full-tile GEMM: BM=64, BN=64, BK=32 — bfloat shmem, f32 accumulators
// ---------------------------------------------------------------------------
kernel void gemm_tiled_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C       [[buffer(2)]],
    constant uint& M       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& K       [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id         [[threadgroup_position_in_grid]],
    uint  tid_in_group     [[thread_index_in_threadgroup]],
    uint  sgid             [[simdgroup_index_in_threadgroup]],
    uint  lane_id          [[thread_index_in_simdgroup]])
{
    threadgroup bfloat As[2][BM * BK];
    threadgroup bfloat Bs[2][BK * BN];

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(BM);
    const uint col_start = swizzled.x * as_uniform(BN);

    device const bfloat* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const bfloat* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device bfloat*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    const uint sg_row = sgid / SG_COLS;
    const uint sg_col = sgid % SG_COLS;

    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    // Prefetch first tile
    {
        for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint gr = row_start + r;
            uint gc = c;
            As[0][r * BK + c] = (gr < uM && gc < uK) ? A_batch[gr * uK + gc] : bfloat(0);
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += N_THREADS) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = r;
            uint gc = col_start + c;
            Bs[0][r * BN + c] = (gr < uK && gc < uN) ? B_batch[gr * uN + gc] : bfloat(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_tiles = (uK + BK - 1) / BK;
    for (uint tile = 0; tile < n_tiles; tile++) {
        uint stage = tile & 1;
        uint next_stage = 1 - stage;
        uint next_kb = (tile + 1) * BK;

        if (tile + 1 < n_tiles) {
            for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
                uint r = idx / BK;
                uint c = idx % BK;
                uint gr = row_start + r;
                uint gc = next_kb + c;
                As[next_stage][r * BK + c] = (gr < uM && gc < uK)
                    ? A_batch[gr * uK + gc] : bfloat(0);
            }
            for (uint idx = tid_in_group; idx < BK * BN; idx += N_THREADS) {
                uint r = idx / BN;
                uint c = idx % BN;
                uint gr = next_kb + r;
                uint gc = col_start + c;
                Bs[next_stage][r * BN + c] = (gr < uK && gc < uN)
                    ? B_batch[gr * uN + gc] : bfloat(0);
            }
        }

        for (uint kk = 0; kk < BK; kk += 8) {
            // Load as float from bfloat shmem (no native bfloat simdgroup_matrix)
            simdgroup_float8x8 a_frag[2];
            simdgroup_float8x8 b_frag[2];

            // Manual load: bfloat -> float for simdgroup_load
            // We use threadgroup float scratch for the conversion
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++) {
                // Load from bfloat shared memory and convert
                threadgroup float a_tmp[64];
                for (uint t = lane_id; t < 64; t += 32) {
                    uint tr = t / 8;
                    uint tc = t % 8;
                    a_tmp[t] = float(As[stage][(sg_row * 16 + i * 8 + tr) * BK + kk + tc]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a_frag[i], a_tmp, 8);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                threadgroup float b_tmp[64];
                for (uint t = lane_id; t < 64; t += 32) {
                    uint tr = t / 8;
                    uint tc = t % 8;
                    b_tmp[t] = float(Bs[stage][(kk + tr) * BN + sg_col * 16 + j * 8 + tc]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(b_frag[j], b_tmp, 8);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        if (tile + 1 < n_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup float result_buf[N_SIMDGROUPS * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + sg_row * 16 + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = bfloat(result_buf[sgid * 64 + lr * 8 + lc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---------------------------------------------------------------------------
// Simdgroup MMA variants (legacy 32x32 — kept as gemm_simd_*)
// These use BM=BN=32, BK=16, 1024 threads, single buffer.
// Used as fallback for medium-sized matrices or when 64x64 tiles
// exceed problem dimensions.
// ---------------------------------------------------------------------------

constant constexpr uint BM32 = 32;
constant constexpr uint BN32 = 32;
constant constexpr uint BK16 = 16;

kernel void gemm_simd_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    const uint sg_cols = BN32 / 8;
    const uint sg_row = sgid / sg_cols;
    const uint sg_col = sgid % sg_cols;

    if (sg_row >= BM32 / 8) return;

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM32;
    const uint col_start = group_id.x * BN32;

    device const float* A_batch = A + batch_idx * batch_stride_a;
    device const float* B_batch = B + batch_idx * batch_stride_b;
    device float*       C_batch = C + batch_idx * batch_stride_c;

    threadgroup float As32[BM32 * BK16];
    threadgroup float Bs32[BK16 * BN32];

    simdgroup_float8x8 acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_threads = BM32 * BN32;

    for (uint kb = 0; kb < K; kb += BK16) {
        for (uint idx = tid_in_group; idx < BM32 * BK16; idx += n_threads) {
            uint r = idx / BK16;
            uint c = idx % BK16;
            uint gr = row_start + r;
            uint gc = kb + c;
            As32[r * BK16 + c] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK16 * BN32; idx += n_threads) {
            uint r = idx / BN32;
            uint c = idx % BN32;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs32[r * BN32 + c] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK16; kk += 8) {
            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            simdgroup_load(a_tile, &As32[sg_row * 8 * BK16 + kk], BK16);
            simdgroup_load(b_tile, &Bs32[kk * BN32 + sg_col * 8], BN32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + sg_row * 8;
    uint out_col = col_start + sg_col * 8;

    threadgroup float result_tiles[16][64];
    simdgroup_store(acc, &result_tiles[sgid][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lane_id; idx < 64; idx += 32) {
        uint lr = idx / 8;
        uint lc = idx % 8;
        uint gr = out_row + lr;
        uint gc = out_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * N + gc] = result_tiles[sgid][lr * 8 + lc];
        }
    }
}

kernel void gemm_simd_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    const uint sg_cols = BN32 / 8;
    const uint sg_row = sgid / sg_cols;
    const uint sg_col = sgid % sg_cols;

    if (sg_row >= BM32 / 8) return;

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM32;
    const uint col_start = group_id.x * BN32;

    device const half* A_batch = A + batch_idx * batch_stride_a;
    device const half* B_batch = B + batch_idx * batch_stride_b;
    device half*       C_batch = C + batch_idx * batch_stride_c;

    threadgroup half As32[BM32 * BK16];
    threadgroup half Bs32[BK16 * BN32];

    simdgroup_float8x8 acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_threads = BM32 * BN32;

    for (uint kb = 0; kb < K; kb += BK16) {
        for (uint idx = tid_in_group; idx < BM32 * BK16; idx += n_threads) {
            uint r = idx / BK16;
            uint c = idx % BK16;
            uint gr = row_start + r;
            uint gc = kb + c;
            As32[r * BK16 + c] = (gr < M && gc < K) ? A_batch[gr * K + gc] : half(0);
        }
        for (uint idx = tid_in_group; idx < BK16 * BN32; idx += n_threads) {
            uint r = idx / BN32;
            uint c = idx % BN32;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs32[r * BN32 + c] = (gr < K && gc < N) ? B_batch[gr * N + gc] : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK16; kk += 8) {
            simdgroup_half8x8 a_tile;
            simdgroup_half8x8 b_tile;
            simdgroup_load(a_tile, &As32[sg_row * 8 * BK16 + kk], BK16);
            simdgroup_load(b_tile, &Bs32[kk * BN32 + sg_col * 8], BN32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + sg_row * 8;
    uint out_col = col_start + sg_col * 8;

    threadgroup float result_tiles[16][64];
    simdgroup_store(acc, &result_tiles[sgid][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lane_id; idx < 64; idx += 32) {
        uint lr = idx / 8;
        uint lc = idx % 8;
        uint gr = out_row + lr;
        uint gc = out_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * N + gc] = half(result_tiles[sgid][lr * 8 + lc]);
        }
    }
}

kernel void gemm_simd_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    const uint sg_cols = BN32 / 8;
    const uint sg_row = sgid / sg_cols;
    const uint sg_col = sgid % sg_cols;

    if (sg_row >= BM32 / 8) return;

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM32;
    const uint col_start = group_id.x * BN32;

    device const bfloat* A_batch = A + batch_idx * batch_stride_a;
    device const bfloat* B_batch = B + batch_idx * batch_stride_b;
    device bfloat*       C_batch = C + batch_idx * batch_stride_c;

    threadgroup float As32[BM32 * BK16];
    threadgroup float Bs32[BK16 * BN32];

    simdgroup_float8x8 acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_threads = BM32 * BN32;

    for (uint kb = 0; kb < K; kb += BK16) {
        for (uint idx = tid_in_group; idx < BM32 * BK16; idx += n_threads) {
            uint r = idx / BK16;
            uint c = idx % BK16;
            uint gr = row_start + r;
            uint gc = kb + c;
            As32[r * BK16 + c] = (gr < M && gc < K) ? float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK16 * BN32; idx += n_threads) {
            uint r = idx / BN32;
            uint c = idx % BN32;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs32[r * BN32 + c] = (gr < K && gc < N) ? float(B_batch[gr * N + gc]) : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK16; kk += 8) {
            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            simdgroup_load(a_tile, &As32[sg_row * 8 * BK16 + kk], BK16);
            simdgroup_load(b_tile, &Bs32[kk * BN32 + sg_col * 8], BN32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + sg_row * 8;
    uint out_col = col_start + sg_col * 8;

    threadgroup float result_tiles[16][64];
    simdgroup_store(acc, &result_tiles[sgid][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = lane_id; idx < 64; idx += 32) {
        uint lr = idx / 8;
        uint lc = idx % 8;
        uint gr = out_row + lr;
        uint gc = out_col + lc;
        if (gr < M && gc < N) {
            C_batch[gr * N + gc] = bfloat(result_tiles[sgid][lr * 8 + lc]);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// HiPerf f16 GEMM: BM=64, BN=64, BK=16 — 128 threads (4 simdgroups)
// Each SG computes 32×32 output = 4×4 grid of 8×8 MMA = 16 accumulators.
// Single-buffered shmem: (64*16 + 16*64) * 2 bytes = 4KB — minimal footprint.
// 16 MMA ops per SG per K-step (vs 4 in old kernel) = 4× pipeline utilization.
// ---------------------------------------------------------------------------

pub const GEMM_HIPERF_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===== HiPerf v2 f16 GEMM: BM=64, BN=64, BK=16, 4 simdgroups, DOUBLE-BUFFERED =====
// 128 threads = 4 simdgroups, 2×2 SG grid.
// Each simdgroup computes a 32×32 sub-tile = 4×4 grid of 8×8 MMA tiles.
// This gives 16 accumulators per SG, keeping the MMA pipeline fully occupied.
// Double-buffered shared memory: 2 × (64×16 + 16×64) × 2 bytes = 8KB total.
// While computing on buffer[stage], the next tile is prefetched into buffer[1-stage],
// hiding global memory latency.

constant constexpr uint HP_BM = 64;
constant constexpr uint HP_BN = 64;
constant constexpr uint HP_BK = 16;
constant constexpr uint HP_N_SG = 4;       // 4 simdgroups
constant constexpr uint HP_SG_ROWS = 2;
constant constexpr uint HP_SG_COLS = 2;
constant constexpr uint HP_N_THREADS = 128; // 4 * 32

// Uniform hint for Metal 3.1+ (M3 and later)
#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> hp_as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T hp_as_uniform(T val) {
    return val;
}
#endif

inline uint2 hp_swizzle_threadgroup(uint2 tid, uint swizzle_log) {
    if (swizzle_log == 0) return tid;
    return uint2(
        tid.x >> swizzle_log,
        (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))
    );
}

kernel void gemm_hiperf_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C       [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id       [[threadgroup_position_in_grid]],
    uint  tid_in_group   [[thread_index_in_threadgroup]],
    uint  sgid           [[simdgroup_index_in_threadgroup]],
    uint  lane_id        [[thread_index_in_simdgroup]])
{
    // Double-buffered shared memory — 8KB total
    threadgroup half As[2][HP_BM * HP_BK]; // 2 * 64*16 = 2048 halves = 4KB
    threadgroup half Bs[2][HP_BK * HP_BN]; // 2 * 16*64 = 2048 halves = 4KB

    const uint batch_idx = group_id.z;
    uint2 swizzled = hp_swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * hp_as_uniform(HP_BM);
    const uint col_start = swizzled.x * hp_as_uniform(HP_BN);

    device const half* A_batch = A + batch_idx * hp_as_uniform(batch_stride_a);
    device const half* B_batch = B + batch_idx * hp_as_uniform(batch_stride_b);
    device half*       C_batch = C + batch_idx * hp_as_uniform(batch_stride_c);

    // SG 2×2 grid: each SG handles a 32×32 sub-tile of the 64×64 output
    const uint sg_row = sgid / HP_SG_COLS; // 0..1
    const uint sg_col = sgid % HP_SG_COLS; // 0..1

    // 16 f32 accumulators per SG: 4×4 grid of 8×8 tiles
    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = hp_as_uniform(K);
    const uint uM = hp_as_uniform(M);
    const uint uN = hp_as_uniform(N);
    const uint n_tiles = (uK + HP_BK - 1) / HP_BK;

    // ── Prefetch first tile into buffer[0] ──
    for (uint idx = tid_in_group; idx < HP_BM * HP_BK; idx += HP_N_THREADS) {
        uint r = idx / HP_BK;
        uint c = idx % HP_BK;
        uint gr = row_start + r;
        uint gc = c;
        As[0][r * HP_BK + c] = (gr < uM && gc < uK) ? A_batch[gr * uK + gc] : half(0);
    }
    for (uint idx = tid_in_group; idx < HP_BK * HP_BN; idx += HP_N_THREADS) {
        uint r = idx / HP_BN;
        uint c = idx % HP_BN;
        uint gr = r;
        uint gc = col_start + c;
        Bs[0][r * HP_BN + c] = (gr < uK && gc < uN) ? B_batch[gr * uN + gc] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Main loop: double-buffered over K dimension ──
    for (uint tile = 0; tile < n_tiles; tile++) {
        uint stage = tile & 1;
        uint next_stage = 1 - stage;
        uint next_kb = (tile + 1) * HP_BK;

        // Prefetch next tile into buffer[next_stage] (if exists)
        if (tile + 1 < n_tiles) {
            for (uint idx = tid_in_group; idx < HP_BM * HP_BK; idx += HP_N_THREADS) {
                uint r = idx / HP_BK;
                uint c = idx % HP_BK;
                uint gr = row_start + r;
                uint gc = next_kb + c;
                As[next_stage][r * HP_BK + c] = (gr < uM && gc < uK)
                    ? A_batch[gr * uK + gc] : half(0);
            }
            for (uint idx = tid_in_group; idx < HP_BK * HP_BN; idx += HP_N_THREADS) {
                uint r = idx / HP_BN;
                uint c = idx % HP_BN;
                uint gr = next_kb + r;
                uint gc = col_start + c;
                Bs[next_stage][r * HP_BN + c] = (gr < uK && gc < uN)
                    ? B_batch[gr * uN + gc] : half(0);
            }
        }

        // ── Compute on buffer[stage]: 2 kk iterations (BK=16, step 8) ──
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < HP_BK; kk += 8) {
            simdgroup_half8x8 a_frag[4]; // 4 tiles in M direction (32 rows)
            simdgroup_half8x8 b_frag[4]; // 4 tiles in N direction (32 cols)

            // Load A fragments: sg_row*32 + {0,8,16,24} rows, kk col
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_frag[i],
                    &As[stage][(sg_row * 32 + i * 8) * HP_BK + kk], HP_BK);
            }

            // Load B fragments: kk row, sg_col*32 + {0,8,16,24} cols
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b_frag[j],
                    &Bs[stage][kk * HP_BN + sg_col * 32 + j * 8], HP_BN);
            }

            // 4×4 outer product = 16 MMA ops per kk iteration
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        if (tile + 1 < n_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Store results: 4×4 grid of 8×8 tiles per simdgroup ──
    // Each SG writes 32×32 = 1024 values at its sub-tile position.
    // Use threadgroup scratch for bounds-checked output (one 8×8 tile at a time).
    threadgroup float result_scratch[HP_N_SG * 64]; // 4 SGs * 64 floats = 1KB

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], &result_scratch[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + sg_row * 32 + i * 8;
            uint base_col = col_start + sg_col * 32 + j * 8;

            // Each lane writes 2 elements (64 elements / 32 lanes)
            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = half(result_scratch[sgid * 64 + lr * 8 + lc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Skinny GEMM shader (M=5..32): BM=32, BN=128, BK=32
// 256 threads = 8 simdgroups, laid out as 1 row x 8 cols.
// Each simdgroup handles one 8x16 sub-tile (2x1 grid of 8x8 MMA).
// For M<32, some rows are zero-padded (handled by bounds check).
// Optimized for prefill with small batch but large N.
// ---------------------------------------------------------------------------

pub const GEMM_SKINNY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SBM = 32;
constant constexpr uint SBN = 128;
constant constexpr uint SBK = 32;
constant constexpr uint SN_THREADS = 256;
constant constexpr uint SSG_COLS = 8;

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T as_uniform(T val) {
    return val;
}
#endif

inline uint2 swizzle_threadgroup(uint2 tid, uint swizzle_log) {
    if (swizzle_log == 0) return tid;
    return uint2(
        tid.x >> swizzle_log,
        (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))
    );
}

// f32 skinny GEMM
kernel void gemm_skinny_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    // Each simdgroup handles 8x16 of the output: row sg_row*8..(sg_row+1)*8,
    // but since SG layout is 1x8, sg_row=0 always.
    // We map: each simdgroup -> 16 columns of output (2 x 8x8 MMA in N dir)
    // With 8 simdgroups: 8 * 16 = 128 = SBN columns.
    // For rows, all simdgroups cover the same 32 rows via 4 x 8x8 tiles.

    threadgroup float As[SBM * SBK];   // 32 * 32 * 4 = 4KB
    threadgroup float Bs[SBK * SBN];   // 32 * 128 * 4 = 16KB

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(SBM);
    const uint col_start = swizzled.x * as_uniform(SBN);

    device const float* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const float* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device float*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    const uint sg_col = sgid;

    // Each simdgroup: 4 row blocks x 2 col blocks of 8x8 = 32x16
    simdgroup_float8x8 acc[4][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    for (uint kb = 0; kb < uK; kb += SBK) {
        for (uint idx = tid_in_group; idx < SBM * SBK; idx += SN_THREADS) {
            uint r = idx / SBK;
            uint c = idx % SBK;
            uint gr = row_start + r;
            uint gc = kb + c;
            As[r * SBK + c] = (gr < uM && gc < uK) ? A_batch[gr * uK + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < SBK * SBN; idx += SN_THREADS) {
            uint r = idx / SBN;
            uint c = idx % SBN;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs[r * SBN + c] = (gr < uK && gc < uN) ? B_batch[gr * uN + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SBK; kk += 8) {
            simdgroup_float8x8 a_frag[4];
            simdgroup_float8x8 b_frag[2];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_frag[i], &As[(i * 8) * SBK + kk], SBK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j], &Bs[kk * SBN + sg_col * 16 + j * 8], SBN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    threadgroup float result_buf[8 * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = result_buf[sgid * 64 + lr * 8 + lc];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// f16 skinny GEMM
kernel void gemm_skinny_f16(
    device const half* A  [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device half* C        [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    threadgroup half As[SBM * SBK];   // 32 * 32 * 2 = 2KB
    threadgroup half Bs[SBK * SBN];   // 32 * 128 * 2 = 8KB

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(SBM);
    const uint col_start = swizzled.x * as_uniform(SBN);

    device const half* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const half* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device half*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    const uint sg_col = sgid;

    simdgroup_float8x8 acc[4][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    for (uint kb = 0; kb < uK; kb += SBK) {
        for (uint idx = tid_in_group; idx < SBM * SBK; idx += SN_THREADS) {
            uint r = idx / SBK;
            uint c = idx % SBK;
            uint gr = row_start + r;
            uint gc = kb + c;
            As[r * SBK + c] = (gr < uM && gc < uK) ? A_batch[gr * uK + gc] : half(0);
        }
        for (uint idx = tid_in_group; idx < SBK * SBN; idx += SN_THREADS) {
            uint r = idx / SBN;
            uint c = idx % SBN;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs[r * SBN + c] = (gr < uK && gc < uN) ? B_batch[gr * uN + gc] : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SBK; kk += 8) {
            simdgroup_half8x8 a_frag[4];
            simdgroup_half8x8 b_frag[2];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_frag[i], &As[(i * 8) * SBK + kk], SBK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j], &Bs[kk * SBN + sg_col * 16 + j * 8], SBN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_buf[8 * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = half(result_buf[sgid * 64 + lr * 8 + lc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// bf16 skinny GEMM
kernel void gemm_skinny_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C       [[buffer(2)]],
    constant uint& M       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& K       [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    constant uint& swizzle_log    [[buffer(9)]],
    uint3 group_id         [[threadgroup_position_in_grid]],
    uint  tid_in_group     [[thread_index_in_threadgroup]],
    uint  sgid             [[simdgroup_index_in_threadgroup]],
    uint  lane_id          [[thread_index_in_simdgroup]])
{
    threadgroup float As[SBM * SBK];
    threadgroup float Bs[SBK * SBN];

    const uint batch_idx = group_id.z;
    uint2 swizzled = swizzle_threadgroup(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * as_uniform(SBM);
    const uint col_start = swizzled.x * as_uniform(SBN);

    device const bfloat* A_batch = A + batch_idx * as_uniform(batch_stride_a);
    device const bfloat* B_batch = B + batch_idx * as_uniform(batch_stride_b);
    device bfloat*       C_batch = C + batch_idx * as_uniform(batch_stride_c);

    const uint sg_col = sgid;

    simdgroup_float8x8 acc[4][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint uK = as_uniform(K);
    const uint uM = as_uniform(M);
    const uint uN = as_uniform(N);

    for (uint kb = 0; kb < uK; kb += SBK) {
        for (uint idx = tid_in_group; idx < SBM * SBK; idx += SN_THREADS) {
            uint r = idx / SBK;
            uint c = idx % SBK;
            uint gr = row_start + r;
            uint gc = kb + c;
            As[r * SBK + c] = (gr < uM && gc < uK) ? float(A_batch[gr * uK + gc]) : 0.0f;
        }
        for (uint idx = tid_in_group; idx < SBK * SBN; idx += SN_THREADS) {
            uint r = idx / SBN;
            uint c = idx % SBN;
            uint gr = kb + r;
            uint gc = col_start + c;
            Bs[r * SBN + c] = (gr < uK && gc < uN) ? float(B_batch[gr * uN + gc]) : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SBK; kk += 8) {
            simdgroup_float8x8 a_frag[4];
            simdgroup_float8x8 b_frag[2];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_frag[i], &As[(i * 8) * SBK + kk], SBK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j], &Bs[kk * SBN + sg_col * 16 + j * 8], SBN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++)
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float result_buf[8 * 64];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], &result_buf[sgid * 64], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint base_row = row_start + i * 8;
            uint base_col = col_start + sg_col * 16 + j * 8;

            for (uint idx = lane_id; idx < 64; idx += 32) {
                uint lr = idx / 8;
                uint lc = idx % 8;
                uint gr = base_row + lr;
                uint gc = base_col + lc;
                if (gr < uM && gc < uN) {
                    C_batch[gr * uN + gc] = bfloat(result_buf[sgid * 64 + lr * 8 + lc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Small-tile GEMM shader (16x16x16) for small matrices
// ---------------------------------------------------------------------------

pub const GEMM_SMALL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BM = 16;
constant constexpr uint BN = 16;
constant constexpr uint BK = 16;

kernel void gemm_small_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]])
{
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;
    const uint local_row = tid_in_group / BN;
    const uint local_col = tid_in_group % BN;

    device const float* A_batch = A + batch_idx * batch_stride_a;
    device const float* B_batch = B + batch_idx * batch_stride_b;
    device float*       C_batch = C + batch_idx * batch_stride_c;

    float acc = 0.0f;
    const uint n_threads = BM * BN;

    for (uint kb = 0; kb < K; kb += BK) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += n_threads) {
            uint r = idx / BK, c = idx % BK;
            uint gr = row_start + r, gc = kb + c;
            As[r * BK + c] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += n_threads) {
            uint r = idx / BN, c = idx % BN;
            uint gr = kb + r, gc = col_start + c;
            Bs[r * BN + c] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk++) {
            acc += As[local_row * BK + kk] * Bs[kk * BN + local_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + local_row;
    uint out_col = col_start + local_col;
    if (out_row < M && out_col < N) {
        C_batch[out_row * N + out_col] = acc;
    }
}

kernel void gemm_small_f16(
    device const half* A  [[buffer(0)]],
    device const half* B  [[buffer(1)]],
    device half* C        [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]])
{
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;
    const uint local_row = tid_in_group / BN;
    const uint local_col = tid_in_group % BN;

    device const half* A_batch = A + batch_idx * batch_stride_a;
    device const half* B_batch = B + batch_idx * batch_stride_b;
    device half*       C_batch = C + batch_idx * batch_stride_c;

    float acc = 0.0f;
    const uint n_threads = BM * BN;

    for (uint kb = 0; kb < K; kb += BK) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += n_threads) {
            uint r = idx / BK, c = idx % BK;
            uint gr = row_start + r, gc = kb + c;
            As[r * BK + c] = (gr < M && gc < K) ? float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += n_threads) {
            uint r = idx / BN, c = idx % BN;
            uint gr = kb + r, gc = col_start + c;
            Bs[r * BN + c] = (gr < K && gc < N) ? float(B_batch[gr * N + gc]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk++) {
            acc += As[local_row * BK + kk] * Bs[kk * BN + local_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + local_row;
    uint out_col = col_start + local_col;
    if (out_row < M && out_col < N) {
        C_batch[out_row * N + out_col] = half(acc);
    }
}

kernel void gemm_small_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C       [[buffer(2)]],
    constant uint& M       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& K       [[buffer(5)]],
    constant uint& batch_stride_a [[buffer(6)]],
    constant uint& batch_stride_b [[buffer(7)]],
    constant uint& batch_stride_c [[buffer(8)]],
    uint3 group_id         [[threadgroup_position_in_grid]],
    uint  tid_in_group     [[thread_index_in_threadgroup]])
{
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];

    const uint batch_idx = group_id.z;
    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;
    const uint local_row = tid_in_group / BN;
    const uint local_col = tid_in_group % BN;

    device const bfloat* A_batch = A + batch_idx * batch_stride_a;
    device const bfloat* B_batch = B + batch_idx * batch_stride_b;
    device bfloat*       C_batch = C + batch_idx * batch_stride_c;

    float acc = 0.0f;
    const uint n_threads = BM * BN;

    for (uint kb = 0; kb < K; kb += BK) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += n_threads) {
            uint r = idx / BK, c = idx % BK;
            uint gr = row_start + r, gc = kb + c;
            As[r * BK + c] = (gr < M && gc < K) ? float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += n_threads) {
            uint r = idx / BN, c = idx % BN;
            uint gr = kb + r, gc = col_start + c;
            Bs[r * BN + c] = (gr < K && gc < N) ? float(B_batch[gr * N + gc]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk++) {
            acc += As[local_row * BK + kk] * Bs[kk * BN + local_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + local_row;
    uint out_col = col_start + local_col;
    if (out_row < M && out_col < N) {
        C_batch[out_row * N + out_col] = bfloat(acc);
    }
}
"#;

// ---------------------------------------------------------------------------
// Split-K GEMM shader: pass 1 computes partial sums per K-chunk,
// pass 2 reduces them. Used when K >> max(M, N).
// ---------------------------------------------------------------------------

pub const SPLIT_K_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SK_BM = 32;
constant constexpr uint SK_BN = 32;
constant constexpr uint SK_BK = 16;

// Pass 1: each threadgroup processes one (tile_m, tile_n, k_split) chunk.
// Output: partial[k_split_idx * M * N + row * N + col] += partial C tile.
kernel void splitk_pass1_f32(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float* partial     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& n_splits   [[buffer(6)]],
    uint3 group_id            [[threadgroup_position_in_grid]],
    uint  tid_in_group        [[thread_index_in_threadgroup]])
{
    threadgroup float As[SK_BM * SK_BK];
    threadgroup float Bs[SK_BK * SK_BN];

    const uint split_idx = group_id.z;
    const uint row_start = group_id.y * SK_BM;
    const uint col_start = group_id.x * SK_BN;
    const uint local_row = tid_in_group / SK_BN;
    const uint local_col = tid_in_group % SK_BN;

    // K range for this split
    uint k_per_split = (K + n_splits - 1) / n_splits;
    uint k_start = split_idx * k_per_split;
    uint k_end = min(k_start + k_per_split, K);

    float acc = 0.0f;
    const uint n_threads = SK_BM * SK_BN;

    for (uint kb = k_start; kb < k_end; kb += SK_BK) {
        for (uint idx = tid_in_group; idx < SK_BM * SK_BK; idx += n_threads) {
            uint r = idx / SK_BK, c = idx % SK_BK;
            uint gr = row_start + r, gc = kb + c;
            As[r * SK_BK + c] = (gr < M && gc < k_end) ? A[gr * K + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < SK_BK * SK_BN; idx += n_threads) {
            uint r = idx / SK_BN, c = idx % SK_BN;
            uint gr = kb + r, gc = col_start + c;
            Bs[r * SK_BN + c] = (gr < k_end && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < SK_BK; kk++) {
            acc += As[local_row * SK_BK + kk] * Bs[kk * SK_BN + local_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + local_row;
    uint out_col = col_start + local_col;
    if (out_row < M && out_col < N) {
        partial[split_idx * M * N + out_row * N + out_col] = acc;
    }
}

// Pass 2: reduce partial sums across K splits.
// Each thread handles one output element.
kernel void splitk_reduce_f32(
    device const float* partial [[buffer(0)]],
    device float* C             [[buffer(1)]],
    constant uint& M            [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& n_splits     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint total = M * N;
    if (id >= total) return;
    float acc = 0.0f;
    for (uint s = 0; s < n_splits; s++) {
        acc += partial[s * total + id];
    }
    C[id] = acc;
}
"#;

// ---------------------------------------------------------------------------
// Tile configuration — adaptive tile sizing
// ---------------------------------------------------------------------------

/// Tile configuration for GEMM dispatch.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub bm: usize,
    pub bn: usize,
    pub variant: TileVariant,
}

/// Which GEMM kernel variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileVariant {
    /// 16x16x16 tiles for small matrices.
    Small,
    /// 32x32x16 tiles with simdgroup MMA (medium, one dim < 33).
    Medium,
    /// 32x32x16 tiles with simdgroup MMA (both dims >= 33 but < 65).
    Simd,
    /// 32x128x32 skinny GEMM for M=5..32 with large N.
    Skinny,
    /// 64x64x32 full tile GEMM with double buffering.
    Full,
}

/// Select the best tile configuration based on matrix dimensions.
///
/// Dispatch hierarchy:
/// - M <= 4: handled by GEMV (not reached here)
/// - M = 5..32, N >= 33: skinny GEMM (BM=32, BN=128)
/// - M,N < 33: small tiles (BM=16, BN=16)
/// - M = 33..64 or N = 33..64 (but not both >= 33): medium (BM=32, BN=32)
/// - M >= 33, N >= 33: full tile GEMM (BM=64, BN=64)
pub fn select_tile_config(m: usize, n: usize, _k: usize) -> TileConfig {
    // Skinny GEMM for small M with large N
    if (5..=32).contains(&m) && n >= 33 {
        return TileConfig {
            bm: 32,
            bn: 128,
            variant: TileVariant::Skinny,
        };
    }

    if m < 33 && n < 33 {
        TileConfig {
            bm: 16,
            bn: 16,
            variant: TileVariant::Small,
        }
    } else if m >= 33 && n >= 33 {
        TileConfig {
            bm: 64,
            bn: 64,
            variant: TileVariant::Full,
        }
    } else {
        // One dim >= 33, other < 33 (and not covered by skinny)
        TileConfig {
            bm: 32,
            bn: 32,
            variant: TileVariant::Simd,
        }
    }
}

/// Compute swizzle_log for threadblock swizzle.
pub fn compute_swizzle_log(m: usize, bm: usize) -> u32 {
    let tiles_m = m.div_ceil(bm);
    if tiles_m > 3 {
        1
    } else {
        0
    }
}

/// Returns true if Split-K should be used for the given dimensions.
pub fn should_use_split_k(m: usize, n: usize, k: usize, batch: usize) -> bool {
    batch == 1 && k > 4 * m.max(n) && m.max(n) > 0
}

/// Number of K-splits to use for Split-K GEMM.
fn split_k_count(m: usize, n: usize, k: usize) -> usize {
    let max_mn = m.max(n).max(1);
    let desired = k / (4 * max_mn);
    desired.clamp(2, 16)
}

// ---------------------------------------------------------------------------
// GEMM tile configuration — hardware-adaptive tile selection
// ---------------------------------------------------------------------------

/// GEMM tile configuration selected at runtime based on hardware and problem size.
#[derive(Debug, Clone, Copy)]
pub struct GemmTileConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    /// Whether to use NAX (M3+ hardware MMA) path.
    pub use_nax: bool,
}

impl GemmTileConfig {
    /// Select optimal tile configuration based on hardware and problem dimensions.
    pub fn select(
        chip: &rmlx_metal::device::ChipTuning,
        m: usize,
        n: usize,
        k: usize,
        is_half: bool,
    ) -> Self {
        // NAX path: M3+ non-phone, half precision, large enough problem
        if chip.supports_nax && is_half && m >= 64 && n >= 64 && k >= 64 {
            return Self {
                bm: 128,
                bn: 128,
                bk: 16,
                wm: 4,
                wn: 4,
                use_nax: true,
            }
            .clamp_to_device(chip);
        }

        // Large matrices: 64x64 tiles
        if m > 512 && n > 512 {
            return Self {
                bm: 64,
                bn: 64,
                bk: 32,
                wm: 2,
                wn: 2,
                use_nax: false,
            }
            .clamp_to_device(chip);
        }

        // Small matrices: 16x16 for less wasted work
        if m < 64 || n < 64 {
            return Self {
                bm: 16,
                bn: 16,
                bk: 16,
                wm: 1,
                wn: 1,
                use_nax: false,
            }
            .clamp_to_device(chip);
        }

        // Default: 32x32
        Self {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 1,
            wn: 1,
            use_nax: false,
        }
        .clamp_to_device(chip)
    }

    /// Clamp tile config to device limits.
    fn clamp_to_device(self, chip: &rmlx_metal::device::ChipTuning) -> Self {
        let tg_mem_needed = 2 * (self.bm * self.bk + self.bk * self.bn) * 4;
        let threads_needed = if self.use_nax {
            32 * self.wm * self.wn
        } else {
            (self.bm / self.wm) * (self.bn / self.wn)
        };

        if tg_mem_needed > chip.max_threadgroup_memory
            || threads_needed > chip.max_threads_per_threadgroup
        {
            Self {
                bm: 32,
                bn: 32,
                bk: 16,
                wm: 1,
                wn: 1,
                use_nax: false,
            }
        } else {
            self
        }
    }
}

// ---------------------------------------------------------------------------
// Constants matching the shader tile sizes
// ---------------------------------------------------------------------------

const BM: usize = 32;
const BN: usize = 32;

/// Register all GEMM kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gemm", GEMM_SHADER_SOURCE)?;
    registry.register_jit_source("gemm_hiperf", GEMM_HIPERF_SHADER_SOURCE)?;
    registry.register_jit_source("gemm_small", GEMM_SMALL_SHADER_SOURCE)?;
    registry.register_jit_source("gemm_skinny", GEMM_SKINNY_SHADER_SOURCE)?;
    registry.register_jit_source("gemm_splitk", SPLIT_K_SHADER_SOURCE)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ceiling division.
#[allow(clippy::manual_div_ceil)]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Create a u32 Metal constant buffer.
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
}

// ---------------------------------------------------------------------------
// Public API: matmul (2D) — backward-compatible entry point
// ---------------------------------------------------------------------------

/// Matrix multiply with auto GEMV/GEMM dispatch.
///
/// Supports 2D inputs:  A: [M, K], B: [K, N] -> C: [M, N]
/// Falls back to GEMV when M<=4 or N=1 (single-token decode hot path).
pub fn matmul(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if a.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "matmul requires 2D arrays, a is {}D",
            a.ndim()
        )));
    }
    if b.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "matmul requires 2D arrays, b is {}D",
            b.ndim()
        )));
    }
    if a.shape()[1] != b.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            a.shape()[1],
            b.shape()[0]
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtypes must match: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // -----------------------------------------------------------------------
    // Auto dispatch: use GEMV for M<=4 or N=1 (single-token decode hot path).
    // -----------------------------------------------------------------------

    // Case 1: M=1 -- [1,K] @ [K,N]
    // Transpose to: B^T[N,K] @ a_vec[K] -> [N], reshaped to [1,N].
    if m == 1 {
        let a_vec = Array::new(
            a.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            a.dtype(),
            a.offset(),
        );
        let b_t = b.view(vec![n, k], vec![1, n], b.offset());
        let b_t = super::copy::copy(registry, &b_t, queue)?;
        let result = super::gemv::gemv(registry, &b_t, &a_vec, queue)?;
        return Ok(Array::new(
            result.metal_buffer().to_owned(),
            vec![1, n],
            vec![n, 1],
            result.dtype(),
            result.offset(),
        ));
    }

    // Case 2: N=1 -- [M,K] @ [K,1]
    if n == 1 {
        let b_vec = Array::new(
            b.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            b.dtype(),
            b.offset(),
        );
        let result = super::gemv::gemv(registry, a, &b_vec, queue)?;
        return Ok(Array::new(
            result.metal_buffer().to_owned(),
            vec![m, 1],
            vec![1, 1],
            result.dtype(),
            result.offset(),
        ));
    }

    // Case 3: M=2..4 -- use GEMV per row (small enough that GEMV is faster)
    if m <= 4 {
        let b_t = b.view(vec![n, k], vec![1, n], b.offset());
        let b_t = super::copy::copy(registry, &b_t, queue)?;
        let a_vec = Array::new(
            a.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            a.dtype(),
            a.offset(),
        );
        // For M=2..4, use B^T @ a_rows approach:
        // Actually, let's just use the GEMM path for M=2..4 with small tiles.
        // The overhead of calling GEMV M times may not be worth it.
        // Fall through to GEMM dispatch.
        drop(b_t);
        drop(a_vec);
    }

    // -----------------------------------------------------------------------
    // Split-K dispatch for K-dominated problems
    // -----------------------------------------------------------------------
    if should_use_split_k(m, n, k, 1) && a.dtype() == DType::Float32 {
        return dispatch_split_k(registry, a, b, queue, m, n, k);
    }

    // -----------------------------------------------------------------------
    // Full tiled GEMM dispatch (batch=1)
    // -----------------------------------------------------------------------
    dispatch_tiled_gemm(
        registry,
        a,
        b,
        queue,
        m,
        n,
        k,
        1,       // batch
        m * k,   // batch_stride_a (unused for batch=1)
        k * n,   // batch_stride_b
        m * n,   // batch_stride_c
        &[m, n], // output shape
    )
}

// ---------------------------------------------------------------------------
// Public API: batched matmul (3D)
// ---------------------------------------------------------------------------

/// Batched matrix multiply: A[B, M, K] @ B[B, K, N] -> C[B, M, N].
pub fn batched_matmul(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if a.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "batched_matmul requires 3D arrays, a is {}D",
            a.ndim()
        )));
    }
    if b.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "batched_matmul requires 3D arrays, b is {}D",
            b.ndim()
        )));
    }
    if a.shape()[0] != b.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "batch dimensions must match: {} vs {}",
            a.shape()[0],
            b.shape()[0]
        )));
    }
    if a.shape()[2] != b.shape()[1] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            a.shape()[2],
            b.shape()[1]
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtypes must match: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

    let batch_stride_a = m * k;
    let batch_stride_b = k * n;
    let batch_stride_c = m * n;

    dispatch_tiled_gemm(
        registry,
        a,
        b,
        queue,
        m,
        n,
        k,
        batch,
        batch_stride_a,
        batch_stride_b,
        batch_stride_c,
        &[batch, m, n],
    )
}

// ---------------------------------------------------------------------------
// Internal: tiled GEMM dispatch
// ---------------------------------------------------------------------------

/// Dispatch the tiled GEMM kernel for the given parameters.
#[allow(clippy::too_many_arguments)]
fn dispatch_tiled_gemm(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    batch_stride_a: usize,
    batch_stride_b: usize,
    batch_stride_c: usize,
    output_shape: &[usize],
) -> Result<Array, KernelError> {
    let tile = select_tile_config(m, n, k);

    let kernel_name = match (tile.variant, a.dtype()) {
        (TileVariant::Small, DType::Float32) => "gemm_small_f32",
        (TileVariant::Small, DType::Float16) => "gemm_small_f16",
        (TileVariant::Small, DType::Bfloat16) => "gemm_small_bf16",
        (TileVariant::Simd, DType::Float32) => "gemm_simd_f32",
        (TileVariant::Simd, DType::Float16) => "gemm_simd_f16",
        (TileVariant::Simd, DType::Bfloat16) => "gemm_simd_bf16",
        (TileVariant::Medium, DType::Float32) => "gemm_simd_f32",
        (TileVariant::Medium, DType::Float16) => "gemm_simd_f16",
        (TileVariant::Medium, DType::Bfloat16) => "gemm_simd_bf16",
        (TileVariant::Skinny, DType::Float32) => "gemm_skinny_f32",
        (TileVariant::Skinny, DType::Float16) => "gemm_skinny_f16",
        (TileVariant::Skinny, DType::Bfloat16) => "gemm_skinny_bf16",
        (TileVariant::Full, DType::Float32) => "gemm_tiled_f32",
        (TileVariant::Full, DType::Float16) => "gemm_hiperf_f16",
        (TileVariant::Full, DType::Bfloat16) => "gemm_tiled_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "matmul not supported for {:?}",
                a.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let out = Array::zeros(registry.device().raw(), output_shape, a.dtype());

    let dev = registry.device().raw();

    let m_buf = make_u32_buf(dev, super::checked_u32(m, "M")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let bsa_buf = make_u32_buf(dev, super::checked_u32(batch_stride_a, "batch_stride_a")?);
    let bsb_buf = make_u32_buf(dev, super::checked_u32(batch_stride_b, "batch_stride_b")?);
    let bsc_buf = make_u32_buf(dev, super::checked_u32(batch_stride_c, "batch_stride_c")?);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&m_buf), 0);
    enc.set_buffer(4, Some(&n_buf), 0);
    enc.set_buffer(5, Some(&k_buf), 0);
    enc.set_buffer(6, Some(&bsa_buf), 0);
    enc.set_buffer(7, Some(&bsb_buf), 0);
    enc.set_buffer(8, Some(&bsc_buf), 0);

    // Pass swizzle_log for Full and Skinny variants (buffer 9)
    let swizzle_log_buf = match tile.variant {
        TileVariant::Full | TileVariant::Skinny => {
            let swizzle_log = compute_swizzle_log(m, tile.bm);
            let buf = make_u32_buf(dev, swizzle_log);
            enc.set_buffer(9, Some(&buf), 0);
            Some(buf)
        }
        _ => None,
    };

    let grid_x = ceil_div(n, tile.bn) as u64;
    let grid_y = ceil_div(m, tile.bm) as u64;
    let grid_z = batch as u64;
    let grid = MTLSize::new(grid_x, grid_y, grid_z);

    // Thread count per threadgroup depends on variant
    // Full+f16 uses gemm_hiperf_f16 (4 SG = 128 threads), others use 256
    let tg_threads = match (tile.variant, a.dtype()) {
        (TileVariant::Small, _) => 256_u64,
        (TileVariant::Medium, _) | (TileVariant::Simd, _) => 1024_u64,
        (TileVariant::Skinny, _) => 256_u64,
        (TileVariant::Full, DType::Float16) => 128_u64, // gemm_hiperf_f16: 4 simdgroups
        (TileVariant::Full, _) => 256_u64,
    };
    let tg = MTLSize::new(tg_threads, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    // Keep swizzle_log_buf alive until after encoding
    drop(swizzle_log_buf);
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Split-K dispatch
// ---------------------------------------------------------------------------

/// Dispatch a two-pass Split-K GEMM for K-dominated problems.
fn dispatch_split_k(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Array, KernelError> {
    let n_splits = split_k_count(m, n, k);
    let dev = registry.device().raw();

    let partial = Array::zeros(dev, &[n_splits * m * n], DType::Float32);
    let out = Array::zeros(dev, &[m, n], DType::Float32);

    let pass1_pipeline = registry.get_pipeline("splitk_pass1_f32", DType::Float32)?;

    let m_buf = make_u32_buf(dev, super::checked_u32(m, "M")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let splits_buf = make_u32_buf(dev, super::checked_u32(n_splits, "n_splits")?);

    let cb = queue.new_command_buffer();

    // Pass 1 encoder
    {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pass1_pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
        enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
        enc.set_buffer(2, Some(partial.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&splits_buf), 0);

        let grid = MTLSize::new(
            ceil_div(n, BN) as u64,
            ceil_div(m, BM) as u64,
            n_splits as u64,
        );
        let tg = MTLSize::new((BM * BN) as u64, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    // Pass 2: reduce
    {
        let pass2_pipeline = registry.get_pipeline("splitk_reduce_f32", DType::Float32)?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pass2_pipeline);
        enc.set_buffer(0, Some(partial.metal_buffer()), 0);
        enc.set_buffer(1, Some(out.metal_buffer()), 0);
        enc.set_buffer(2, Some(&m_buf), 0);
        enc.set_buffer(3, Some(&n_buf), 0);
        enc.set_buffer(4, Some(&splits_buf), 0);

        let total = m * n;
        let tg_size = 256u64;
        let n_groups = ceil_div(total, tg_size as usize) as u64;
        enc.dispatch_thread_groups(MTLSize::new(n_groups, 1, 1), MTLSize::new(tg_size, 1, 1));
        enc.end_encoding();
    }

    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

/// Build function constants for matmul alignment specialization.
pub fn matmul_align_constants(
    m: usize,
    n: usize,
    bm: usize,
    bn: usize,
) -> Vec<(u32, crate::kernels::FunctionConstantValue)> {
    use crate::kernels::FunctionConstantValue;
    vec![
        (200, FunctionConstantValue::Bool(m % bm == 0)),
        (201, FunctionConstantValue::Bool(n % bn == 0)),
    ]
}

// ---------------------------------------------------------------------------
// Public API: matmul_into_cb — encode into an existing command buffer
// ---------------------------------------------------------------------------

/// Matrix multiply encoded into an existing command buffer (no commit/wait).
///
/// Like [`matmul`] but does not create its own command buffer. The caller is
/// responsible for committing and waiting on `cb`.
///
/// **Inputs must be contiguous.** If either `a` or `b` is non-contiguous this
/// function returns an error. The prefill path should pre-transpose weights
/// before calling this.
///
/// Supports the same dispatch hierarchy as `matmul`:
/// - M=1 → GEMV via `gemv_into_cb` (B^T @ a_vec)
/// - N=1 → GEMV via `gemv_into_cb` (A @ b_vec)
/// - M>=5 → tiled GEMM (Small / Skinny / Simd / Medium / Full)
///
/// Split-K is **not** supported in this path (it requires a two-pass
/// encode with an intermediate buffer; the caller should use `matmul()` for
/// K-dominated problems).
pub fn matmul_into_cb(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    // --- Validation ---
    if a.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "matmul_into_cb requires 2D arrays, a is {}D",
            a.ndim()
        )));
    }
    if b.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "matmul_into_cb requires 2D arrays, b is {}D",
            b.ndim()
        )));
    }
    if a.shape()[1] != b.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            a.shape()[1],
            b.shape()[0]
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtypes must match: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    if !a.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "matmul_into_cb: input `a` must be contiguous".to_string(),
        ));
    }
    if !b.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "matmul_into_cb: input `b` must be contiguous".to_string(),
        ));
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // -------------------------------------------------------------------
    // GEMV fast-paths (same logic as matmul)
    // -------------------------------------------------------------------

    // Case 1: M=1 — [1,K] @ [K,N] → transpose B to [N,K], GEMV → [N], reshape [1,N]
    if m == 1 {
        let a_vec = Array::new(
            a.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            a.dtype(),
            a.offset(),
        );
        // B is [K,N] row-major. We need B^T = [N,K] contiguous for GEMV.
        let b_t_view = b.view(vec![n, k], vec![1, n], b.offset());
        let b_t = super::copy::copy_into_cb(registry, &b_t_view, cb)?;
        let result = super::gemv::gemv_into_cb(registry, &b_t, &a_vec, cb)?;
        return Ok(Array::new(
            result.metal_buffer().to_owned(),
            vec![1, n],
            vec![n, 1],
            result.dtype(),
            result.offset(),
        ));
    }

    // Case 2: N=1 — [M,K] @ [K,1] → GEMV A @ b_vec → [M], reshape [M,1]
    if n == 1 {
        let b_vec = Array::new(
            b.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            b.dtype(),
            b.offset(),
        );
        let result = super::gemv::gemv_into_cb(registry, a, &b_vec, cb)?;
        return Ok(Array::new(
            result.metal_buffer().to_owned(),
            vec![m, 1],
            vec![1, 1],
            result.dtype(),
            result.offset(),
        ));
    }

    // -------------------------------------------------------------------
    // Tiled GEMM dispatch (batch=1)
    // -------------------------------------------------------------------
    let tile = select_tile_config(m, n, k);

    let kernel_name = match (tile.variant, a.dtype()) {
        (TileVariant::Small, DType::Float32) => "gemm_small_f32",
        (TileVariant::Small, DType::Float16) => "gemm_small_f16",
        (TileVariant::Small, DType::Bfloat16) => "gemm_small_bf16",
        (TileVariant::Simd, DType::Float32) | (TileVariant::Medium, DType::Float32) => {
            "gemm_simd_f32"
        }
        (TileVariant::Simd, DType::Float16) | (TileVariant::Medium, DType::Float16) => {
            "gemm_simd_f16"
        }
        (TileVariant::Simd, DType::Bfloat16) | (TileVariant::Medium, DType::Bfloat16) => {
            "gemm_simd_bf16"
        }
        (TileVariant::Skinny, DType::Float32) => "gemm_skinny_f32",
        (TileVariant::Skinny, DType::Float16) => "gemm_skinny_f16",
        (TileVariant::Skinny, DType::Bfloat16) => "gemm_skinny_bf16",
        (TileVariant::Full, DType::Float32) => "gemm_tiled_f32",
        (TileVariant::Full, DType::Float16) => "gemm_hiperf_f16",
        (TileVariant::Full, DType::Bfloat16) => "gemm_tiled_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "matmul_into_cb: unsupported dtype {:?}",
                a.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &[m, n], a.dtype());

    let m_buf = make_u32_buf(dev, super::checked_u32(m, "M")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let bsa_buf = make_u32_buf(dev, super::checked_u32(m * k, "batch_stride_a")?);
    let bsb_buf = make_u32_buf(dev, super::checked_u32(k * n, "batch_stride_b")?);
    let bsc_buf = make_u32_buf(dev, super::checked_u32(m * n, "batch_stride_c")?);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&m_buf), 0);
    enc.set_buffer(4, Some(&n_buf), 0);
    enc.set_buffer(5, Some(&k_buf), 0);
    enc.set_buffer(6, Some(&bsa_buf), 0);
    enc.set_buffer(7, Some(&bsb_buf), 0);
    enc.set_buffer(8, Some(&bsc_buf), 0);

    // Pass swizzle_log for Full and Skinny variants (buffer 9)
    let swizzle_log_buf = match tile.variant {
        TileVariant::Full | TileVariant::Skinny => {
            let swizzle_log = compute_swizzle_log(m, tile.bm);
            let buf = make_u32_buf(dev, swizzle_log);
            enc.set_buffer(9, Some(&buf), 0);
            Some(buf)
        }
        _ => None,
    };

    let grid_x = ceil_div(n, tile.bn) as u64;
    let grid_y = ceil_div(m, tile.bm) as u64;
    let grid = MTLSize::new(grid_x, grid_y, 1); // batch=1

    // Thread count per threadgroup depends on variant
    // Full+f16 uses gemm_hiperf_f16 (4 SG = 128 threads), others use 256
    let tg_threads = match (tile.variant, a.dtype()) {
        (TileVariant::Small, _) => 256_u64,
        (TileVariant::Medium, _) | (TileVariant::Simd, _) => 1024_u64,
        (TileVariant::Skinny, _) => 256_u64,
        (TileVariant::Full, DType::Float16) => 128_u64,  // gemm_hiperf_f16: 4 simdgroups
        (TileVariant::Full, _) => 256_u64,
    };
    let tg = MTLSize::new(tg_threads, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    // Keep swizzle_log_buf alive until after encoding
    drop(swizzle_log_buf);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Adaptive tile config tests ──

    #[test]
    fn test_select_tile_config_small() {
        let cfg = select_tile_config(32, 32, 64);
        assert_eq!(cfg.bm, 16);
        assert_eq!(cfg.bn, 16);
        assert_eq!(cfg.variant, TileVariant::Small);
    }

    #[test]
    fn test_select_tile_config_medium() {
        // Both >= 33: Full tile
        let cfg = select_tile_config(128, 128, 64);
        assert_eq!(cfg.bm, 64);
        assert_eq!(cfg.bn, 64);
        assert_eq!(cfg.variant, TileVariant::Full);
    }

    #[test]
    fn test_select_tile_config_boundary() {
        // M=64 and N=64 -> Full
        let cfg = select_tile_config(64, 64, 64);
        assert_eq!(cfg.variant, TileVariant::Full);

        // M=32, N=32 -> Small (both < 33)
        let cfg = select_tile_config(32, 32, 64);
        assert_eq!(cfg.variant, TileVariant::Small);
    }

    #[test]
    fn test_select_tile_config_mixed() {
        // M < 33 but N >= 33, M >= 5 -> Skinny
        let cfg = select_tile_config(16, 128, 64);
        assert_eq!(cfg.variant, TileVariant::Skinny);
    }

    #[test]
    fn test_select_tile_config_simd() {
        // Both M >= 33 and N >= 33 -> Full
        let cfg = select_tile_config(128, 128, 128);
        assert_eq!(cfg.bm, 64);
        assert_eq!(cfg.bn, 64);
        assert_eq!(cfg.variant, TileVariant::Full);

        // M >= 33, N < 33 -> Simd
        let cfg = select_tile_config(128, 32, 128);
        assert_eq!(cfg.variant, TileVariant::Simd);
    }

    #[test]
    fn test_select_tile_config_skinny() {
        // M=5..32, N >= 33 -> Skinny
        let cfg = select_tile_config(8, 256, 128);
        assert_eq!(cfg.variant, TileVariant::Skinny);
        assert_eq!(cfg.bm, 32);
        assert_eq!(cfg.bn, 128);

        let cfg = select_tile_config(32, 64, 128);
        assert_eq!(cfg.variant, TileVariant::Skinny);

        // M=4 is below skinny threshold, M < 33 && N < 33 -> Small
        let cfg = select_tile_config(4, 32, 128);
        assert_eq!(cfg.variant, TileVariant::Small);

        // M=4, N >= 33 -> not skinny (M < 5), goes to Simd
        let cfg = select_tile_config(4, 64, 128);
        assert_eq!(cfg.variant, TileVariant::Simd);
    }

    #[test]
    fn test_select_tile_config_full() {
        let cfg = select_tile_config(64, 64, 128);
        assert_eq!(cfg.variant, TileVariant::Full);
        assert_eq!(cfg.bm, 64);
        assert_eq!(cfg.bn, 64);

        let cfg = select_tile_config(512, 512, 1024);
        assert_eq!(cfg.variant, TileVariant::Full);
    }

    // ── Split-K heuristic tests ──

    #[test]
    fn test_should_use_split_k() {
        assert!(should_use_split_k(16, 16, 256, 1));
        assert!(!should_use_split_k(128, 128, 256, 1));
        assert!(!should_use_split_k(16, 16, 256, 4));
    }

    #[test]
    fn test_split_k_count_bounds() {
        let count = split_k_count(16, 16, 256);
        assert!(count >= 2);
        assert!(count <= 16);
    }

    #[test]
    fn test_split_k_count_large_k() {
        let count = split_k_count(16, 16, 4096);
        assert!(count >= 2);
        assert!(count <= 16);
    }

    // ── GemmTileConfig tests ──

    #[test]
    fn test_gemm_tile_config_small() {
        let chip = chip_m2_max();
        let config = GemmTileConfig::select(&chip, 16, 32, 128, false);
        assert_eq!(config.bm, 16);
        assert_eq!(config.bn, 16);
        assert!(!config.use_nax);
    }

    #[test]
    fn test_gemm_tile_config_large() {
        let chip = chip_m2_max();
        let config = GemmTileConfig::select(&chip, 1024, 1024, 1024, false);
        assert_eq!(config.bm, 64);
        assert!(!config.use_nax);
    }

    #[test]
    fn test_gemm_tile_config_nax() {
        let chip = chip_m3_max();
        let config = GemmTileConfig::select(&chip, 512, 512, 512, true);
        assert!(config.use_nax);
        assert_eq!(config.bm, 128);
    }

    #[test]
    fn test_gemm_tile_config_clamp_fallback() {
        let chip = chip_unknown();
        let config = GemmTileConfig::select(&chip, 1024, 1024, 1024, true);
        assert!(!config.use_nax, "unknown device should not use NAX");
    }

    // ── Threadblock swizzle tests ──

    #[test]
    fn test_compute_swizzle_log() {
        assert_eq!(compute_swizzle_log(32, 32), 0); // 1 tile
        assert_eq!(compute_swizzle_log(96, 32), 0); // 3 tiles
        assert_eq!(compute_swizzle_log(128, 32), 1); // 4 tiles
        assert_eq!(compute_swizzle_log(4096, 32), 1);
    }

    // ── ChipTuning test helpers ──

    fn chip_m2_max() -> rmlx_metal::device::ChipTuning {
        rmlx_metal::device::ChipTuning {
            max_threadgroup_memory: 32 * 1024,
            max_threads_per_threadgroup: 1024,
            preferred_simd_width: 32,
            supports_unretained_refs: true,
            arch_gen: 16,
            arch_class: rmlx_metal::device::ArchClass::Max,
            supports_nax: false,
            max_ops_per_batch: 50,
            max_mb_per_batch: 50,
            supports_concurrent_dispatch: true,
        }
    }

    fn chip_m3_max() -> rmlx_metal::device::ChipTuning {
        rmlx_metal::device::ChipTuning {
            max_threadgroup_memory: 32 * 1024,
            max_threads_per_threadgroup: 1024,
            preferred_simd_width: 32,
            supports_unretained_refs: true,
            arch_gen: 17,
            arch_class: rmlx_metal::device::ArchClass::Max,
            supports_nax: true,
            max_ops_per_batch: 50,
            max_mb_per_batch: 50,
            supports_concurrent_dispatch: true,
        }
    }

    fn chip_unknown() -> rmlx_metal::device::ChipTuning {
        rmlx_metal::device::ChipTuning {
            max_threadgroup_memory: 16 * 1024,
            max_threads_per_threadgroup: 512,
            preferred_simd_width: 32,
            supports_unretained_refs: false,
            arch_gen: 0,
            arch_class: rmlx_metal::device::ArchClass::Unknown,
            supports_nax: false,
            max_ops_per_batch: 32,
            max_mb_per_batch: 32,
            supports_concurrent_dispatch: false,
        }
    }
}
