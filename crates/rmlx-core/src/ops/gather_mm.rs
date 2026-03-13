//! Index-based batched GEMM for Mixture-of-Experts (MoE) expert dispatch.
//!
//! `gather_mm(x, weights, indices)` performs a batched matrix multiply where each
//! batch element selects which weight matrix to use via an index tensor.
//!
//! This is the key primitive for MoE architectures where a router assigns each
//! token to one (or more) experts, and each expert has its own weight matrix.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLResourceOptions;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source for GatherMM
// ---------------------------------------------------------------------------

/// Metal shader for index-based batched GEMM using simdgroup MMA.
///
/// Each threadgroup processes one (batch_element, tile) pair. The batch element's
/// index selects which weight matrix slice to read from.
///
/// Layout:
/// - x:       [batch, M_per_batch, K]   (input activations)
/// - weights: [n_experts, K, N]         (expert weight matrices)
/// - indices: [batch]                    (uint32 expert index per batch element)
/// - output:  [batch, M_per_batch, N]
///
/// Architecture: MLX-style simdgroup MMA
/// - BM=32, BN=32, BK=16, 64 threads (2 simdgroups)
/// - SG layout 1×2: each SG covers 32×16 output
/// - TM=4, TN=2 per SG (8 accumulator fragments)
/// - Serpentine MMA ordering for register locality
/// - thread_elements() direct store (Metal 3.1+)
///
/// Provides three kernel variants:
/// - `gather_mm_f32`: float32 I/O, float32 accumulator
/// - `gather_mm_f16`: float16 I/O, float32 accumulator, half threadgroup mem
/// - `gather_mm_bf16`: bfloat16 I/O, float32 accumulator, per-SG f32 B buffer
pub const GATHER_MM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 16;
constant constexpr uint N_SG = 2;
constant constexpr uint N_THREADS = 64;
constant constexpr uint TM = 4;   // BM / 8
constant constexpr uint TN = 2;   // (BN / N_SG) / 8

// ---- f32 kernel ----

kernel void gather_mm_f32(
    device const float*  x       [[buffer(0)]],
    device const float*  weights [[buffer(1)]],
    device const uint*   indices [[buffer(2)]],
    device       float*  output  [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& M_per_batch   [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& n_experts     [[buffer(8)]],
    uint3 group_id               [[threadgroup_position_in_grid]],
    uint  tid_in_group           [[thread_index_in_threadgroup]],
    uint  sgid                   [[simdgroup_index_in_threadgroup]],
    uint  lane_id                [[thread_index_in_simdgroup]])
{
    threadgroup float As[BM * BK];   // 32×16 = 512 floats = 2KB
    threadgroup float Bs[BK * BN];   // 16×32 = 512 floats = 2KB

    const uint batch_idx = group_id.z;
    if (batch_idx >= batch) return;

    const uint expert_idx = indices[batch_idx];
    if (expert_idx >= n_experts) return;

    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;

    device const float* X_batch  = x + batch_idx * M_per_batch * K;
    device const float* W_expert = weights + expert_idx * K * N;
    device float*       O_batch  = output + batch_idx * M_per_batch * N;

    // SG layout 1×2: sg_col = sgid (0 or 1), each covers 16 cols
    const uint base_n = sgid * 16;

    // Accumulators: TM×TN = 4×2 = 8 fragments per SG
    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint kb = 0; kb < K; kb += BK) {
        // Load A tile: 64 threads, each loads 8 elements (32×16 / 64 = 8)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // Each thread loads a row-chunk: tid maps to (row, col_base)
            // 64 threads, 512 elements → 8 elements per thread
            for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
                uint r = idx / BK;
                uint c = idx % BK;
                uint gr = row_start + r;
                uint gc = kb + c;
                As[r * BK + c] = (gr < M_per_batch && gc < K)
                    ? X_batch[gr * K + gc] : 0.0f;
            }
        }
        // Load B tile: 64 threads, 512 elements → 8 per thread
        {
            for (uint idx = tid_in_group; idx < BK * BN; idx += N_THREADS) {
                uint r = idx / BN;
                uint c = idx % BN;
                uint gr = kb + r;
                uint gc = col_start + c;
                Bs[r * BN + c] = (gr < K && gc < N)
                    ? W_expert[gr * N + gc] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA compute with serpentine ordering
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 2; kk++) {  // BK/8 = 2
            simdgroup_float8x8 a_frag[TM];
            simdgroup_float8x8 b_frag[TN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                simdgroup_load(a_frag[i], &As[i * 8 * BK + kk * 8], BK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; j++) {
                simdgroup_load(b_frag[j], &Bs[kk * 8 * BN + base_n + j * 8], BN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; j++) {
                    uint n_serp = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // Store via thread_elements() direct store
    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++) {
            uint gr = row_start + i * 8 + fm;
            uint gc0 = col_start + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (gr < M_per_batch && gc0 < N) {
                O_batch[gr * N + gc0] = elems[0];
            }
            if (gr < M_per_batch && gc1 < N) {
                O_batch[gr * N + gc1] = elems[1];
            }
        }
    }
}

// ---- f16 kernel ----

kernel void gather_mm_f16(
    device const half*   x       [[buffer(0)]],
    device const half*   weights [[buffer(1)]],
    device const uint*   indices [[buffer(2)]],
    device       half*   output  [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& M_per_batch   [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& n_experts     [[buffer(8)]],
    uint3 group_id               [[threadgroup_position_in_grid]],
    uint  tid_in_group           [[thread_index_in_threadgroup]],
    uint  sgid                   [[simdgroup_index_in_threadgroup]],
    uint  lane_id                [[thread_index_in_simdgroup]])
{
    threadgroup half As[BM * BK];   // 32×16 = 512 halves = 1KB
    threadgroup half Bs[BK * BN];   // 16×32 = 512 halves = 1KB

    const uint batch_idx = group_id.z;
    if (batch_idx >= batch) return;

    const uint expert_idx = indices[batch_idx];
    if (expert_idx >= n_experts) return;

    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;

    device const half* X_batch  = x + batch_idx * M_per_batch * K;
    device const half* W_expert = weights + expert_idx * K * N;
    device half*       O_batch  = output + batch_idx * M_per_batch * N;

    const uint base_n = sgid * 16;

    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint kb = 0; kb < K; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile with half4 wide loads where possible
        {
            // 32 rows × 16 cols = 512 halves, 64 threads → 8 per thread
            // Each thread loads 4 elements at a time via half4 when aligned
            uint a_row = tid_in_group / 4;   // 0..15 (maps to rows)
            uint a_col = (tid_in_group % 4) * 4;  // 0, 4, 8, 12
            // Two passes: rows 0..15, then 16..31
            #pragma clang loop unroll(full)
            for (uint pass = 0; pass < 2; pass++) {
                uint r = a_row + pass * 16;
                uint gr = row_start + r;
                uint gc = kb + a_col;
                if (gr < M_per_batch && gc + 3 < K) {
                    *reinterpret_cast<threadgroup half4*>(&As[r * BK + a_col]) =
                        *reinterpret_cast<device const half4*>(&X_batch[gr * K + gc]);
                } else {
                    for (uint d = 0; d < 4; d++) {
                        As[r * BK + a_col + d] = (gr < M_per_batch && gc + d < K)
                            ? X_batch[gr * K + gc + d] : half(0);
                    }
                }
            }
        }

        // Load B tile with half4 wide loads
        {
            // 16 rows × 32 cols = 512 halves, 64 threads → 8 per thread
            uint b_row = tid_in_group / 8;          // 0..7
            uint b_col = (tid_in_group % 8) * 4;    // 0, 4, 8, ..., 28
            #pragma clang loop unroll(full)
            for (uint pass = 0; pass < 2; pass++) {
                uint r = b_row + pass * 8;
                uint gr = kb + r;
                uint gc = col_start + b_col;
                if (gr < K && gc + 3 < N) {
                    *reinterpret_cast<threadgroup half4*>(&Bs[r * BN + b_col]) =
                        *reinterpret_cast<device const half4*>(&W_expert[gr * N + gc]);
                } else {
                    for (uint d = 0; d < 4; d++) {
                        Bs[r * BN + b_col + d] = (gr < K && gc + d < N)
                            ? W_expert[gr * N + gc + d] : half(0);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA compute with serpentine ordering
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 2; kk++) {
            simdgroup_half8x8 a_frag[TM];
            simdgroup_half8x8 b_frag[TN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                simdgroup_load(a_frag[i], &As[i * 8 * BK + kk * 8], BK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; j++) {
                simdgroup_load(b_frag[j], &Bs[kk * 8 * BN + base_n + j * 8], BN);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; j++) {
                    uint n_serp = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // Store via thread_elements() direct store
    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++) {
            uint gr = row_start + i * 8 + fm;
            uint gc0 = col_start + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (gr < M_per_batch && gc0 < N) {
                O_batch[gr * N + gc0] = half(elems[0]);
            }
            if (gr < M_per_batch && gc1 < N) {
                O_batch[gr * N + gc1] = half(elems[1]);
            }
        }
    }
}

// ---- bf16 kernel ----
// Uses per-SG f32 buffers for B tile to avoid race condition on bfloat→float
// conversion. A tile is loaded as bfloat, B tile is bulk-converted to f32
// in a per-SG region before MMA.

kernel void gather_mm_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device const uint*   indices [[buffer(2)]],
    device       bfloat* output  [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& M_per_batch   [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& n_experts     [[buffer(8)]],
    uint3 group_id               [[threadgroup_position_in_grid]],
    uint  tid_in_group           [[thread_index_in_threadgroup]],
    uint  sgid                   [[simdgroup_index_in_threadgroup]],
    uint  lane_id                [[thread_index_in_simdgroup]])
{
    // A stored as f32 (bfloat has no simdgroup_matrix support, convert on load)
    threadgroup float As[BM * BK];        // 32×16 = 512 floats = 2KB
    // B stored as f32, per-SG region to avoid race condition
    threadgroup float Bs[N_SG][BK * 16];  // 2 × (16×16) = 512 floats = 2KB each

    const uint batch_idx = group_id.z;
    if (batch_idx >= batch) return;

    const uint expert_idx = indices[batch_idx];
    if (expert_idx >= n_experts) return;

    const uint row_start = group_id.y * BM;
    const uint col_start = group_id.x * BN;

    device const bfloat* X_batch  = x + batch_idx * M_per_batch * K;
    device const bfloat* W_expert = weights + expert_idx * K * N;
    device bfloat*       O_batch  = output + batch_idx * M_per_batch * N;

    const uint base_n = sgid * 16;

    simdgroup_float8x8 acc[TM][TN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint kb = 0; kb < K; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile: bfloat → f32 on load
        // 64 threads, 512 elements → 8 per thread
        {
            for (uint idx = tid_in_group; idx < BM * BK; idx += N_THREADS) {
                uint r = idx / BK;
                uint c = idx % BK;
                uint gr = row_start + r;
                uint gc = kb + c;
                As[r * BK + c] = (gr < M_per_batch && gc < K)
                    ? float(X_batch[gr * K + gc]) : 0.0f;
            }
        }

        // Load B tile: bfloat → f32, each SG loads its own 16-col slice
        // Each SG has 32 threads, loads 16×16 = 256 elements → 8 per thread
        {
            for (uint idx = lane_id; idx < BK * 16; idx += 32) {
                uint r = idx / 16;
                uint c = idx % 16;
                uint gr = kb + r;
                uint gc = col_start + base_n + c;
                Bs[sgid][r * 16 + c] = (gr < K && gc < N)
                    ? float(W_expert[gr * N + gc]) : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA compute with serpentine ordering (f32 fragments)
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 2; kk++) {
            simdgroup_float8x8 a_frag[TM];
            simdgroup_float8x8 b_frag[TN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                simdgroup_load(a_frag[i], &As[i * 8 * BK + kk * 8], BK);
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; j++) {
                simdgroup_load(b_frag[j], &Bs[sgid][kk * 8 * 16 + j * 8], 16);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; j++) {
                    uint n_serp = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // Store via thread_elements() direct store
    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; j++) {
            uint gr = row_start + i * 8 + fm;
            uint gc0 = col_start + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (gr < M_per_batch && gc0 < N) {
                O_batch[gr * N + gc0] = bfloat(elems[0]);
            }
            if (gr < M_per_batch && gc1 < N) {
                O_batch[gr * N + gc1] = bfloat(elems[1]);
            }
        }
    }
}
"#;

/// Register GatherMM kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gather_mm", GATHER_MM_SHADER_SOURCE)
}

/// Ceiling division.
#[allow(clippy::manual_div_ceil)]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Create a u32 Metal constant buffer.
fn make_u32_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: u32,
) -> rmlx_metal::MtlBuffer {
    let opts = MTLResourceOptions::StorageModeShared;
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(&val as *const u32 as *const _ as *mut std::ffi::c_void)
                    .unwrap(),
                4_usize,
                opts,
            )
            .unwrap()
    }
}

/// Index-based batched GEMM for MoE expert dispatch.
///
/// Computes `output[b] = x[b] @ weights[indices[b]]` for each batch element.
///
/// # Arguments
/// - `x`: Input activations `[batch, m_per_batch, k]`
/// - `weights`: Expert weight matrices `[n_experts, k, n]`
/// - `indices`: Expert index per batch element `[batch]` (UInt32)
/// - `m_per_batch`: Rows per batch element
/// - `n`: Output columns
/// - `k`: Inner dimension
///
/// # Returns
/// Output matrix `[batch, m_per_batch, n]`.
#[allow(clippy::too_many_arguments)]
pub fn gather_mm(
    registry: &KernelRegistry,
    x: &Array,
    weights: &Array,
    indices: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    // Validate shapes
    if x.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: x must be 3D [batch, m, k], got {}D",
            x.ndim()
        )));
    }
    if weights.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: weights must be 3D [n_experts, k, n], got {}D",
            weights.ndim()
        )));
    }
    if indices.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices must be 1D [batch], got {}D",
            indices.ndim()
        )));
    }
    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }

    let batch = x.shape()[0];
    let m_per_batch = x.shape()[1];
    let k = x.shape()[2];
    let n_experts = weights.shape()[0];
    let n = weights.shape()[2];

    if indices.shape()[0] != batch {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices length {} != batch {}",
            indices.shape()[0],
            batch
        )));
    }
    if weights.shape()[1] != k {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: weights K dim {} != x K dim {}",
            weights.shape()[1],
            k
        )));
    }

    // Rust-side validation: all expert indices must be in [0, n_experts)
    if batch > 0 {
        let idx_vec = indices.to_vec_checked::<u32>();
        for (i, &idx) in idx_vec.iter().enumerate() {
            if (idx as usize) >= n_experts {
                return Err(KernelError::InvalidShape(format!(
                    "gather_mm: index[{i}]={idx} out of range [0, {n_experts})"
                )));
            }
        }
    }

    // Make inputs contiguous
    let x_c = super::make_contiguous(x, registry, queue)?;
    let x = x_c.as_ref().unwrap_or(x);
    let w_c = super::make_contiguous(weights, registry, queue)?;
    let weights = w_c.as_ref().unwrap_or(weights);

    let kernel_name = match x.dtype() {
        DType::Float32 => "gather_mm_f32",
        DType::Float16 => "gather_mm_f16",
        DType::Bfloat16 => "gather_mm_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gather_mm not supported for {:?}",
                x.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, x.dtype())?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &[batch, m_per_batch, n], x.dtype());

    let batch_buf = make_u32_buf(dev, super::checked_u32(batch, "batch")?);
    let m_buf = make_u32_buf(dev, super::checked_u32(m_per_batch, "M_per_batch")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let ne_buf = make_u32_buf(dev, super::checked_u32(n_experts, "n_experts")?);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset());
    enc.set_buffer(1, Some(weights.metal_buffer()), weights.offset());
    enc.set_buffer(2, Some(indices.metal_buffer()), indices.offset());
    enc.set_buffer(3, Some(out.metal_buffer()), 0);
    enc.set_buffer(4, Some(&batch_buf), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&ne_buf), 0);
    const BM: usize = 32;
    const BN: usize = 32;
    const N_THREADS: usize = 64; // 2 simdgroups × 32 threads

    let grid = MTLSize {
        width: ceil_div(n, BN),
        height: ceil_div(m_per_batch, BM),
        depth: batch,
    };
    let tg = MTLSize {
        width: N_THREADS,
        height: 1,
        depth: 1,
    };

    enc.dispatch_threadgroups(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Encode gather_mm into an existing command buffer (no commit/wait).
///
/// This variant encodes the gather_mm compute pass into the caller-provided
/// command buffer without committing or waiting. This is useful for pipelining
/// multiple GPU operations into a single command buffer to avoid per-op
/// commit overhead.
///
/// # Note on index validation
/// Unlike `gather_mm()`, this function does **not** perform Rust-side index
/// validation (checking that all expert indices are in `[0, n_experts)`).
/// Because the command buffer has not yet been committed, the index buffer
/// contents may not be available on the CPU (e.g., if indices were produced
/// by a prior GPU operation in the same command buffer). The GPU kernel
/// itself performs a bounds check and returns early for out-of-range indices.
///
/// # Arguments
/// - `registry`: Kernel registry for pipeline lookup
/// - `x`: Input activations `[batch, m_per_batch, k]` (f32, f16, or bf16)
/// - `weights`: Expert weight matrices `[n_experts, k, n]` (same dtype as x)
/// - `indices`: Expert index per batch element `[batch]` (UInt32)
/// - `cb`: The command buffer to encode into
///
/// # Returns
/// Output array `[batch, m_per_batch, n]`. The caller must commit and wait
/// on the command buffer before reading the output data.
pub fn gather_mm_into_cb(
    registry: &KernelRegistry,
    x: &Array,
    weights: &Array,
    indices: &Array,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    // Validate shapes (same as gather_mm, except no Rust-side index bounds check)
    if x.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: x must be 3D [batch, m, k], got {}D",
            x.ndim()
        )));
    }
    if weights.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: weights must be 3D [n_experts, k, n], got {}D",
            weights.ndim()
        )));
    }
    if indices.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices must be 1D [batch], got {}D",
            indices.ndim()
        )));
    }
    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }

    let batch = x.shape()[0];
    let m_per_batch = x.shape()[1];
    let k = x.shape()[2];
    let n_experts = weights.shape()[0];
    let n = weights.shape()[2];

    if indices.shape()[0] != batch {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: indices length {} != batch {}",
            indices.shape()[0],
            batch
        )));
    }
    if weights.shape()[1] != k {
        return Err(KernelError::InvalidShape(format!(
            "gather_mm: weights K dim {} != x K dim {}",
            weights.shape()[1],
            k
        )));
    }

    // Skip Rust-side index validation: indices may have been produced by a
    // prior GPU op in the same command buffer and are not yet readable on CPU.
    // The GPU kernel itself checks `if (expert_idx >= n_experts) return;`.

    let kernel_name = match x.dtype() {
        DType::Float32 => "gather_mm_f32",
        DType::Float16 => "gather_mm_f16",
        DType::Bfloat16 => "gather_mm_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gather_mm not supported for {:?}",
                x.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, x.dtype())?;
    let dev = registry.device().raw();
    let out = Array::uninit(dev, &[batch, m_per_batch, n], x.dtype());

    let batch_buf = make_u32_buf(dev, super::checked_u32(batch, "batch")?);
    let m_buf = make_u32_buf(dev, super::checked_u32(m_per_batch, "M_per_batch")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let ne_buf = make_u32_buf(dev, super::checked_u32(n_experts, "n_experts")?);

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset());
    enc.set_buffer(1, Some(weights.metal_buffer()), weights.offset());
    enc.set_buffer(2, Some(indices.metal_buffer()), indices.offset());
    enc.set_buffer(3, Some(out.metal_buffer()), 0);
    enc.set_buffer(4, Some(&batch_buf), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&ne_buf), 0);
    const BM: usize = 32;
    const BN: usize = 32;
    const N_THREADS: usize = 64; // 2 simdgroups × 32 threads

    let grid = MTLSize {
        width: ceil_div(n, BN),
        height: ceil_div(m_per_batch, BM),
        depth: batch,
    };
    let tg = MTLSize {
        width: N_THREADS,
        height: 1,
        depth: 1,
    };

    enc.dispatch_threadgroups(grid, tg);
    enc.end();

    Ok(out)
}
