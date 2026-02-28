//! Matrix multiplication with automatic GEMV/GEMM dispatch.
//! Uses GEMV for vector-matrix (M=1 or N=1) and GEMM for full matrix multiply.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

pub const GEMM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Naive GEMM (tiled version for production would use Steel)
// C[M,N] = A[M,K] * B[K,N]
kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gemm", GEMM_SHADER_SOURCE)
}

/// Matrix multiply with auto GEMV/GEMM dispatch.
/// A: [M, K], B: [K, N] -> C: [M, N]
/// Falls back to GEMV when M=1 (using gemv module).
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

    // Auto dispatch: use GEMV for M=1 or N=1 (single-token decode hot path).
    // GEMV computes mat[M,K] @ vec[K] -> [M].

    // Case 1: M=1 — [1,K] @ [K,N]
    // Transpose to: B^T[N,K] @ a_vec[K] -> [N], reshaped to [1,N].
    if m == 1 {
        let a_vec = Array::new(
            a.metal_buffer().to_owned(),
            vec![k],
            vec![1],
            a.dtype(),
            a.offset(),
        );
        // B^T: view B[K,N] as [N,K] by swapping strides (column-major read)
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

    // Case 2: N=1 — [M,K] @ [K,1]
    // Squeeze b to 1D [K], GEMV: A[M,K] @ b[K] -> [M], reshaped to [M,1].
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

    let kernel_name = match a.dtype() {
        DType::Float32 => "gemm_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "matmul not supported for {:?}",
                a.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let out = Array::zeros(registry.device().raw(), &[m, n], a.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let m_u32 = super::checked_u32(m, "M")?;
    let n_u32 = super::checked_u32(n, "N")?;
    let k_u32 = super::checked_u32(k, "K")?;
    let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
    let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&m_buf), 0);
    enc.set_buffer(4, Some(&n_buf), 0);
    enc.set_buffer(5, Some(&k_buf), 0);

    let grid = metal::MTLSize::new(n as u64, m as u64, 1);
    let tg = metal::MTLSize::new(std::cmp::min(16, n as u64), std::cmp::min(16, m as u64), 1);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}
