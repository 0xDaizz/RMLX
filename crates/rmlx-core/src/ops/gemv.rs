//! General Matrix-Vector multiplication: y = A * x

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const GEMV_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// GEMV: y[row] = dot(A[row,:], x[:])
// A shape: [M, K], x shape: [K], y shape: [M]
kernel void gemv_f32(
    device const float* mat [[buffer(0)]],
    device const float* vec [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    if (row >= M) return;

    threadgroup float shared_sum[1024];

    float sum = 0.0;
    uint base = row * K;
    for (uint i = tid; i < K; i += tgsize) {
        sum += mat[base + i] * vec[i];
    }
    shared_sum[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gemv", GEMV_SHADER_SOURCE)
}

/// Matrix-vector multiply: y = A * x
/// - mat: [M, K]
/// - vec: [K]
/// - output: [M]
pub fn gemv(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }

    let mat_contig = super::make_contiguous(mat, registry, queue)?;
    let mat = mat_contig.as_ref().unwrap_or(mat);
    let vec_contig = super::make_contiguous(vec, registry, queue)?;
    let vec = vec_contig.as_ref().unwrap_or(vec);

    let kernel_name = match mat.dtype() {
        DType::Float32 => "gemv_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let m = super::checked_u32(mat.shape()[0], "M")?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::zeros(registry.device().raw(), &[m as usize], mat.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let m_buf = dev.new_buffer_with_data(&m as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&m_buf), 0);
    encoder.set_buffer(4, Some(&k_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}
