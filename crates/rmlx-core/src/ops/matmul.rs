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
    assert_eq!(a.ndim(), 2, "matmul requires 2D arrays");
    assert_eq!(b.ndim(), 2, "matmul requires 2D arrays");
    assert_eq!(a.shape()[1], b.shape()[0], "inner dimensions must match");
    assert_eq!(a.dtype(), b.dtype(), "dtypes must match");

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // Auto dispatch: use GEMV for vector-matrix cases
    if m == 1 {
        // A is a row vector — reshape and use GEMV with transposed B
        // For now, fall through to GEMM
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
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
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
