//! Fused kernel dispatch patterns for common operation sequences.
//!
//! Reduces dispatch count and intermediate memory traffic by combining
//! commonly co-occurring operations into single command buffer dispatches.
//!
//! # Fusion patterns
//!
//! | Pattern | Current Ops | Fused | Savings |
//! |---------|------------|-------|---------|
//! | Q/K/V Projection | 3 matmuls | Batched GEMM | 3 CBs -> 1 |
//! | SiLU + Mul (SwiGLU) | 2 CBs | 1 CB | 1 less sync |

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Batched Q/K/V projection: single command buffer for 3 matmuls.
///
/// Instead of 3 separate `matmul()` calls (each creating its own CB),
/// this encodes all 3 projections into one command buffer.
///
/// `input`: [seq_len, hidden_size]
/// `wq`, `wk`, `wv`: projection weight matrices (transposed views)
///
/// Returns (Q, K, V) arrays.
pub fn batched_qkv_proj(
    registry: &KernelRegistry,
    input: &Array,
    wq_t: &Array,
    wk_t: &Array,
    wv_t: &Array,
    queue: &metal::CommandQueue,
) -> Result<(Array, Array, Array), KernelError> {
    let input = ensure_contiguous(input, registry, queue)?;

    let m = input.shape()[0] as u32;
    let k = input.shape()[1] as u32;
    let nq = wq_t.shape()[1] as u32;
    let nk = wk_t.shape()[1] as u32;
    let nv = wv_t.shape()[1] as u32;

    let dev = registry.device().raw();
    let q_out = Array::zeros(dev, &[m as usize, nq as usize], input.dtype());
    let k_out = Array::zeros(dev, &[m as usize, nk as usize], input.dtype());
    let v_out = Array::zeros(dev, &[m as usize, nv as usize], input.dtype());

    let kernel_name = gemm_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    // Single command buffer for all 3 projections
    let cb = queue.new_command_buffer();

    // Encode Q projection
    encode_gemm(cb, &pipeline, &input, wq_t, &q_out, m, nq, k, registry)?;

    // Encode K projection
    encode_gemm(cb, &pipeline, &input, wk_t, &k_out, m, nk, k, registry)?;

    // Encode V projection
    encode_gemm(cb, &pipeline, &input, wv_t, &v_out, m, nv, k, registry)?;

    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok((q_out, k_out, v_out))
}

/// Fused SwiGLU: silu(gate(x)) * up(x), single command buffer.
///
/// Combines gate projection, SiLU activation, up projection, and element-wise
/// multiply into a minimal number of encoders on one command buffer.
///
/// `gate_out`: output of gate_proj.forward()
/// `up_out`: output of up_proj.forward()
///
/// Returns: silu(gate_out) * up_out
pub fn fused_silu_mul(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::zeros(dev, gate_out.shape(), gate_out.dtype());
    let numel_buf = make_u32_buf(dev, numel as u32);

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    enc.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    enc.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    enc.set_buffer(3, Some(&numel_buf), 0);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    enc.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(output)
}

/// Encode fused SwiGLU (silu(gate) * up) into an existing command buffer (no commit/wait).
///
/// **Caller must ensure gate_out and up_out are contiguous with matching shapes/dtypes.**
pub fn fused_silu_mul_into_cb(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_cb: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_cb: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul_into_cb: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::uninit(dev, gate_out.shape(), gate_out.dtype());
    let numel_u32 = numel as u32;

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    enc.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    enc.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    enc.set_bytes(3, 4, &numel_u32 as *const u32 as *const std::ffi::c_void);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    enc.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();

    Ok(output)
}

/// Encode fused SiLU * mul into an existing compute command encoder (no encoder create/end).
/// Caller is responsible for creating and ending the encoder.
pub fn fused_silu_mul_into_encoder(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_encoder: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_encoder: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul_into_encoder: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::uninit(dev, gate_out.shape(), gate_out.dtype());
    let numel_u32 = numel as u32;

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    encoder.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    encoder.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    encoder.set_bytes(3, 4, &numel_u32 as *const u32 as *const std::ffi::c_void);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    encoder.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));

    Ok(output)
}

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode fused SiLU * mul using a pre-resolved PSO and pre-allocated output buffer.
#[allow(clippy::too_many_arguments)]
pub fn fused_silu_mul_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    gate_buf: &metal::BufferRef,
    gate_offset: u64,
    up_buf: &metal::BufferRef,
    up_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    numel: u32,
    elems_per_thread: u64,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(gate_buf), gate_offset);
    encoder.set_buffer(1, Some(up_buf), up_offset);
    encoder.set_buffer(2, Some(out_buf), out_offset);
    encoder.set_bytes(3, 4, &numel as *const u32 as *const std::ffi::c_void);
    let tg = std::cmp::min(pso.max_total_threads_per_threadgroup(), grid_threads);
    encoder.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
}

/// Get the silu_gate kernel name for a dtype.
pub fn silu_gate_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("silu_gate_f32"),
        DType::Float16 => Ok("silu_gate_f16"),
        DType::Bfloat16 => Ok("silu_gate_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "silu_gate: unsupported dtype {:?}",
            dtype
        ))),
    }
}

/// Get the number of elements per thread for fused silu_mul.
pub fn silu_gate_elems_per_thread(dtype: DType) -> u64 {
    match dtype {
        DType::Float32 => 2,
        _ => 4,
    }
}

/// Batched Q/K/V projection using the CommandBatcher (no commit/wait).
///
/// Same as `batched_qkv_proj` but encodes into a provided command buffer.
pub fn batched_qkv_proj_into(
    registry: &KernelRegistry,
    input: &Array,
    wq_t: &Array,
    wk_t: &Array,
    wv_t: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<(Array, Array, Array), KernelError> {
    // Ensure all inputs are contiguous for the GEMM kernel
    let input = if input.is_contiguous() {
        input.view(
            input.shape().to_vec(),
            input.strides().to_vec(),
            input.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, input, cb)?
    };
    let wq_t = if wq_t.is_contiguous() {
        wq_t.view(
            wq_t.shape().to_vec(),
            wq_t.strides().to_vec(),
            wq_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wq_t, cb)?
    };
    let wk_t = if wk_t.is_contiguous() {
        wk_t.view(
            wk_t.shape().to_vec(),
            wk_t.strides().to_vec(),
            wk_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wk_t, cb)?
    };
    let wv_t = if wv_t.is_contiguous() {
        wv_t.view(
            wv_t.shape().to_vec(),
            wv_t.strides().to_vec(),
            wv_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wv_t, cb)?
    };

    let m = input.shape()[0] as u32;
    let k = input.shape()[1] as u32;
    let nq = wq_t.shape()[1] as u32;
    let nk = wk_t.shape()[1] as u32;
    let nv = wv_t.shape()[1] as u32;

    let dev = registry.device().raw();
    let q_out = Array::uninit(dev, &[m as usize, nq as usize], input.dtype());
    let k_out = Array::uninit(dev, &[m as usize, nk as usize], input.dtype());
    let v_out = Array::uninit(dev, &[m as usize, nv as usize], input.dtype());

    let kernel_name = gemm_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    encode_gemm(cb, &pipeline, &input, &wq_t, &q_out, m, nq, k, registry)?;
    encode_gemm(cb, &pipeline, &input, &wk_t, &k_out, m, nk, k, registry)?;
    encode_gemm(cb, &pipeline, &input, &wv_t, &v_out, m, nv, k, registry)?;

    Ok((q_out, k_out, v_out))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn gemm_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("gemm_tiled_f32"),
        DType::Float16 => Ok("gemm_tiled_f16"),
        DType::Bfloat16 => Ok("gemm_tiled_bf16"),
        other => Err(KernelError::InvalidShape(format!(
            "fused: unsupported dtype for GEMM: {:?}",
            other
        ))),
    }
}

/// Encode a single GEMM dispatch into a command buffer.
///
/// C[M, N] = A[M, K] @ B[K, N]
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    cb: &metal::CommandBufferRef,
    pipeline: &metal::ComputePipelineState,
    a: &Array,
    b: &Array,
    c: &Array,
    m: u32,
    n: u32,
    k: u32,
    _registry: &KernelRegistry,
) -> Result<(), KernelError> {
    let batch_stride_a = m * k;
    let batch_stride_b = k * n;
    let batch_stride_c = m * n;

    const BM: u64 = 32;
    const BN: u64 = 32;
    let grid_x = (n as u64).div_ceil(BN);
    let grid_y = (m as u64).div_ceil(BM);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    enc.set_buffer(2, Some(c.metal_buffer()), c.offset() as u64);
    enc.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(
        6,
        4,
        &batch_stride_a as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        7,
        4,
        &batch_stride_b as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        8,
        4,
        &batch_stride_c as *const u32 as *const std::ffi::c_void,
    );

    let grid = MTLSize::new(grid_x, grid_y, 1);
    let tg = MTLSize::new(BM * BN, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();

    Ok(())
}

fn make_u32_buf(device: &metal::Device, value: u32) -> metal::Buffer {
    let data = value.to_ne_bytes();
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn ensure_contiguous(
    array: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if array.is_contiguous() {
        Ok(array.view(
            array.shape().to_vec(),
            array.strides().to_vec(),
            array.offset(),
        ))
    } else {
        super::copy::copy(registry, array, queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_silu_mul_shape_validation() {
        let device = metal::Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(
            rmlx_metal::device::GpuDevice::system_default().expect("GPU device"),
        );
        super::super::silu::register(&registry).ok();

        let a = Array::from_slice(&device, &[1.0f32, 2.0, 3.0], vec![3]);
        let b = Array::from_slice(&device, &[1.0f32, 2.0], vec![2]);

        let result = fused_silu_mul(&registry, &a, &b, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_kernel_name() {
        assert_eq!(gemm_kernel_name(DType::Float32).unwrap(), "gemm_tiled_f32");
        assert_eq!(gemm_kernel_name(DType::Float16).unwrap(), "gemm_tiled_f16");
        assert!(gemm_kernel_name(DType::Q4_0).is_err());
    }
}
