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
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source for GatherMM
// ---------------------------------------------------------------------------

/// Metal shader for index-based batched GEMM.
///
/// Each threadgroup processes one (batch_element, tile) pair. The batch element's
/// index selects which weight matrix slice to read from.
///
/// Layout:
/// - x:       [batch, M_per_batch, K]   (input activations)
/// - weights: [n_experts, K, N]         (expert weight matrices)
/// - indices: [batch]                    (uint32 expert index per batch element)
/// - output:  [batch, M_per_batch, N]
pub const GATHER_MM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 16;

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
    uint  tid_in_group           [[thread_index_in_threadgroup]])
{
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];

    const uint batch_idx = group_id.z;
    if (batch_idx >= batch) return;

    const uint expert_idx = indices[batch_idx];
    if (expert_idx >= n_experts) return;
    const uint row_start  = group_id.y * BM;
    const uint col_start  = group_id.x * BN;
    const uint local_row  = tid_in_group / BN;
    const uint local_col  = tid_in_group % BN;

    // Pointers for this batch element
    device const float* X_batch = x + batch_idx * M_per_batch * K;
    device const float* W_expert = weights + expert_idx * K * N;
    device float* O_batch = output + batch_idx * M_per_batch * N;

    float acc = 0.0f;
    const uint n_threads = BM * BN;

    for (uint kb = 0; kb < K; kb += BK) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += n_threads) {
            uint r = idx / BK, c = idx % BK;
            uint gr = row_start + r, gc = kb + c;
            As[r * BK + c] = (gr < M_per_batch && gc < K) ? X_batch[gr * K + gc] : 0.0f;
        }
        for (uint idx = tid_in_group; idx < BK * BN; idx += n_threads) {
            uint r = idx / BN, c = idx % BN;
            uint gr = kb + r, gc = col_start + c;
            Bs[r * BN + c] = (gr < K && gc < N) ? W_expert[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk++) {
            acc += As[local_row * BK + kk] * Bs[kk * BN + local_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint out_row = row_start + local_row;
    uint out_col = col_start + local_col;
    if (out_row < M_per_batch && out_col < N) {
        O_batch[out_row * N + out_col] = acc;
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
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
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
    queue: &metal::CommandQueue,
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

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(weights.metal_buffer()), weights.offset() as u64);
    enc.set_buffer(2, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(3, Some(out.metal_buffer()), 0);
    enc.set_buffer(4, Some(&batch_buf), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&ne_buf), 0);

    const BM: usize = 32;
    const BN: usize = 32;

    let grid = MTLSize::new(
        ceil_div(n, BN) as u64,
        ceil_div(m_per_batch, BM) as u64,
        batch as u64,
    );
    let tg = MTLSize::new((BM * BN) as u64, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}
