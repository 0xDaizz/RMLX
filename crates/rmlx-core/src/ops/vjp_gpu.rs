//! GPU-accelerated VJP (backward pass dispatch).
//!
//! Moves gradient computation to Metal GPU kernels instead of CPU.
//! Provides GPU-based backward passes for elementwise add, mul, and matmul.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLResourceOptions;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source for VJP backward kernels
// ---------------------------------------------------------------------------

/// Metal shaders for backward pass computations.
///
/// Kernels:
/// - `vjp_add_f32`: Backward for elementwise add (passthrough)
/// - `vjp_mul_f32`: Backward for elementwise mul (product rule)
/// - `vjp_matmul_grad_a_f32`: dA = dC @ B^T for matmul backward
/// - `vjp_matmul_grad_b_f32`: dB = A^T @ dC for matmul backward
pub const VJP_GPU_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── Elementwise Add backward ──────────────────────────────────────────
// grad_a[i] = grad_output[i], grad_b[i] = grad_output[i]
kernel void vjp_add_f32(
    device const float* grad_output [[buffer(0)]],
    device float*       grad_a      [[buffer(1)]],
    device float*       grad_b      [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float g = grad_output[id];
    grad_a[id] = g;
    grad_b[id] = g;
}

// ─── Elementwise Mul backward ──────────────────────────────────────────
// grad_a[i] = grad_output[i] * b[i], grad_b[i] = grad_output[i] * a[i]
kernel void vjp_mul_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* a           [[buffer(1)]],
    device const float* b           [[buffer(2)]],
    device float*       grad_a      [[buffer(3)]],
    device float*       grad_b      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    float g = grad_output[id];
    grad_a[id] = g * b[id];
    grad_b[id] = g * a[id];
}

// ─── MatMul backward: dA = dC @ B^T ────────────────────────────────────
// A: [M, K], B: [K, N], C: [M, N]
// dA[i, j] = sum_p dC[i, p] * B[j, p]   (B^T[p, j] = B[j, p])
//
// Grid: (K, M) -- each thread computes one element of grad_A
kernel void vjp_matmul_grad_a_f32(
    device const float* grad_c  [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       grad_a  [[buffer(2)]],
    constant uint& M            [[buffer(3)]],
    constant uint& K            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;  // K dimension
    uint row = gid.y;  // M dimension
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (uint p = 0; p < N; p++) {
        sum += grad_c[row * N + p] * b[col * N + p];
    }
    grad_a[row * K + col] = sum;
}

// ─── MatMul backward: dB = A^T @ dC ────────────────────────────────────
// dB[i, j] = sum_p A[p, i] * dC[p, j]
//
// Grid: (N, K) -- each thread computes one element of grad_B
kernel void vjp_matmul_grad_b_f32(
    device const float* a       [[buffer(0)]],
    device const float* grad_c  [[buffer(1)]],
    device float*       grad_b  [[buffer(2)]],
    constant uint& M            [[buffer(3)]],
    constant uint& K            [[buffer(4)]],
    constant uint& N            [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;  // N dimension
    uint row = gid.y;  // K dimension
    if (row >= K || col >= N) return;

    float sum = 0.0f;
    for (uint p = 0; p < M; p++) {
        sum += a[p * K + row] * grad_c[p * N + col];
    }
    grad_b[row * N + col] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register VJP GPU kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("vjp_gpu", VJP_GPU_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a u32 Metal constant buffer.
fn make_u32_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: u32,
) -> rmlx_metal::MtlBuffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(&val as *const u32 as *const _ as *mut std::ffi::c_void)
                    .unwrap(),
                std::mem::size_of::<u32>() as u64 as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

// --------------------------------------------------------------------------- // Public API // ---------------------------------------------------------------------------
/// GPU backward for elementwise addition. /// /// Given `grad_output`, produces `grad_a = grad_output` and `grad_b = grad_output`. /// Returns `(grad_a, grad_b)`. pub
pub fn vjp_add(
    registry: &KernelRegistry,
    grad_output: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Array), KernelError> {
    if grad_output.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "vjp_add: requires Float32, got {:?}",
            grad_output.dtype()
        )));
    }

    let numel = grad_output.numel();
    if numel == 0 {
        let dev = registry.device().raw();
        let ga = Array::zeros(dev, grad_output.shape(), DType::Float32);
        let gb = Array::zeros(dev, grad_output.shape(), DType::Float32);
        return Ok((ga, gb));
    }

    let g_c = super::make_contiguous(grad_output, registry, queue)?;
    let grad_output = g_c.as_ref().unwrap_or(grad_output);

    let pipeline = registry.get_pipeline("vjp_add_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let grad_a = Array::zeros(dev, grad_output.shape(), DType::Float32);
    let grad_b = Array::zeros(dev, grad_output.shape(), DType::Float32);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(grad_output.metal_buffer()), grad_output.offset());
    enc.set_buffer(1, Some(grad_a.metal_buffer()), 0);
    enc.set_buffer(2, Some(grad_b.metal_buffer()), 0);
    let grid = MTLSize {
        width: numel,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok((grad_a, grad_b))
}

/// GPU backward for elementwise multiplication (product rule).
///
/// Given `grad_output`, `a`, `b`, produces:
/// - `grad_a[i] = grad_output[i] * b[i]`
/// - `grad_b[i] = grad_output[i] * a[i]`
///
/// Returns `(grad_a, grad_b)`.
pub fn vjp_mul(
    registry: &KernelRegistry,
    grad_output: &Array,
    a: &Array,
    b: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Array), KernelError> {
    if grad_output.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "vjp_mul: requires Float32, got {:?}",
            grad_output.dtype()
        )));
    }

    let numel = grad_output.numel();

    let g_c = super::make_contiguous(grad_output, registry, queue)?;
    let grad_output = g_c.as_ref().unwrap_or(grad_output);
    let a_c = super::make_contiguous(a, registry, queue)?;
    let a = a_c.as_ref().unwrap_or(a);
    let b_c = super::make_contiguous(b, registry, queue)?;
    let b = b_c.as_ref().unwrap_or(b);

    let pipeline = registry.get_pipeline("vjp_mul_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let grad_a = Array::zeros(dev, grad_output.shape(), DType::Float32);
    let grad_b = Array::zeros(dev, grad_output.shape(), DType::Float32);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(grad_output.metal_buffer()), grad_output.offset());
    enc.set_buffer(1, Some(a.metal_buffer()), a.offset());
    enc.set_buffer(2, Some(b.metal_buffer()), b.offset());
    enc.set_buffer(3, Some(grad_a.metal_buffer()), 0);
    enc.set_buffer(4, Some(grad_b.metal_buffer()), 0);
    let grid = MTLSize {
        width: numel,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok((grad_a, grad_b))
}

/// GPU backward for matrix multiplication.
///
/// Given `grad_c` (gradient of output C = A @ B), produces:
/// - `grad_a = grad_c @ B^T` (shape [M, K])
/// - `grad_b = A^T @ grad_c` (shape [K, N])
///
/// # Arguments
/// - `grad_c`: gradient of output, shape `[M, N]`
/// - `a`: original input A, shape `[M, K]`
/// - `b`: original input B, shape `[K, N]`
///
/// Returns `(grad_a, grad_b)`.
pub fn vjp_matmul(
    registry: &KernelRegistry,
    grad_c: &Array,
    a: &Array,
    b: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Array), KernelError> {
    if grad_c.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "vjp_matmul: requires Float32, got {:?}",
            grad_c.dtype()
        )));
    }
    if a.ndim() != 2 || b.ndim() != 2 || grad_c.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "vjp_matmul: all inputs must be 2D".into(),
        ));
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // Ensure contiguous
    let gc_c = super::make_contiguous(grad_c, registry, queue)?;
    let grad_c = gc_c.as_ref().unwrap_or(grad_c);
    let a_c = super::make_contiguous(a, registry, queue)?;
    let a = a_c.as_ref().unwrap_or(a);
    let b_c = super::make_contiguous(b, registry, queue)?;
    let b = b_c.as_ref().unwrap_or(b);

    let dev = registry.device().raw();
    let grad_a = Array::zeros(dev, &[m, k], DType::Float32);
    let grad_b = Array::zeros(dev, &[k, n], DType::Float32);

    let m_buf = make_u32_buf(dev, super::checked_u32(m, "M")?);
    let k_buf = make_u32_buf(dev, super::checked_u32(k, "K")?);
    let n_buf = make_u32_buf(dev, super::checked_u32(n, "N")?);

    // --- grad_a = grad_c @ B^T ---
    {
        let pipeline = registry.get_pipeline("vjp_matmul_grad_a_f32", DType::Float32)?;
        let cb = queue.commandBuffer().unwrap();
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(grad_c.metal_buffer()), grad_c.offset());
        enc.set_buffer(1, Some(b.metal_buffer()), b.offset());
        enc.set_buffer(2, Some(grad_a.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&k_buf), 0);
        enc.set_buffer(5, Some(&n_buf), 0);
        let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
        let tw = pipeline.threadExecutionWidth();
        let tg_x = tw.min(k);
        let tg_y = (max_tg / tg_x).max(1).min(m);

        let grid = MTLSize {
            width: k,
            height: m,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_x,
            height: tg_y,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end();
        super::commit_with_mode(&cb, super::ExecMode::Sync);
    }

    // --- grad_b = A^T @ grad_c ---
    {
        let pipeline = registry.get_pipeline("vjp_matmul_grad_b_f32", DType::Float32)?;
        let cb = queue.commandBuffer().unwrap();
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), a.offset());
        enc.set_buffer(1, Some(grad_c.metal_buffer()), grad_c.offset());
        enc.set_buffer(2, Some(grad_b.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&k_buf), 0);
        enc.set_buffer(5, Some(&n_buf), 0);
        let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
        let tw = pipeline.threadExecutionWidth();
        let tg_x = tw.min(n);
        let tg_y = (max_tg / tg_x).max(1).min(k);

        let grid = MTLSize {
            width: n,
            height: k,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_x,
            height: tg_y,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end();
        super::commit_with_mode(&cb, super::ExecMode::Sync);
    }

    Ok((grad_a, grad_b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(KernelRegistry, rmlx_metal::MtlQueue)> {
        let gpu_dev = crate::test_utils::test_gpu()?;
        let queue = gpu_dev.new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();
        Some((registry, queue))
    }

    #[test]
    fn test_vjp_add_gpu() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        let grad = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let (ga, gb) = vjp_add(&registry, &grad, &queue).unwrap();

        let ga_vec = ga.to_vec_checked::<f32>();
        let gb_vec = gb.to_vec_checked::<f32>();
        assert_eq!(ga_vec, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gb_vec, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vjp_mul_gpu() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        let grad = Array::from_slice(dev, &[1.0f32, 1.0, 1.0, 1.0], vec![4]);
        let a = Array::from_slice(dev, &[2.0f32, 3.0, 4.0, 5.0], vec![4]);
        let b = Array::from_slice(dev, &[10.0f32, 20.0, 30.0, 40.0], vec![4]);

        let (ga, gb) = vjp_mul(&registry, &grad, &a, &b, &queue).unwrap();

        let ga_vec = ga.to_vec_checked::<f32>();
        let gb_vec = gb.to_vec_checked::<f32>();
        // grad_a = grad * b, grad_b = grad * a
        assert_eq!(ga_vec, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(gb_vec, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vjp_matmul_gpu() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        // A: [2, 3], B: [3, 2], C: [2, 2]
        let a = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Array::from_slice(dev, &[1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0], vec![3, 2]);
        // grad_c = identity-ish: all ones [2, 2]
        let grad_c = Array::from_slice(dev, &[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]);

        let (ga, gb) = vjp_matmul(&registry, &grad_c, &a, &b, &queue).unwrap();

        assert_eq!(ga.shape(), &[2, 3]);
        assert_eq!(gb.shape(), &[3, 2]);

        // Verify grad_a = grad_c @ B^T
        // B^T = [[1,0,1],[0,1,0]]
        // grad_c @ B^T = [[1,1],[1,1]] @ [[1,0,1],[0,1,0]] = [[1,1,1],[1,1,1]]
        let ga_vec = ga.to_vec_checked::<f32>();
        assert_eq!(ga_vec, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
