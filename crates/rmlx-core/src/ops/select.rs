//! Select/Where (ternary) operation: `out[i] = cond[i] ? a[i] : b[i]`.
//!
//! Supports f32, f16, bf16 for value operands. The condition tensor must be
//! UInt32 (0 = false, nonzero = true), matching the output of comparison ops.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for select/where kernels.
pub const SELECT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define SELECT_KERNEL(NAME, TYPE)                                          \
kernel void NAME(                                                          \
    device const uint* cond [[buffer(0)]],                                 \
    device const TYPE* a    [[buffer(1)]],                                 \
    device const TYPE* b    [[buffer(2)]],                                 \
    device TYPE* out        [[buffer(3)]],                                 \
    uint id [[thread_position_in_grid]])                                   \
{                                                                          \
    out[id] = cond[id] ? a[id] : b[id];                                   \
}

SELECT_KERNEL(select_f32,  float)
SELECT_KERNEL(select_f16,  half)
SELECT_KERNEL(select_bf16, bfloat)
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register select/where kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("select", SELECT_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn select_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("select_f32"),
        DType::Float16 => Ok("select_f16"),
        DType::Bfloat16 => Ok("select_bf16"),
        _ => Err(KernelError::InvalidShape(format!(
            "select: unsupported value dtype {:?}; expected f32/f16/bf16",
            dtype
        ))),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Element-wise ternary select: `out[i] = cond[i] ? a[i] : b[i]`.
///
/// # Arguments
/// - `cond`: Condition array, UInt32 (0 = false, nonzero = true).
/// - `a`: Values selected where condition is true.
/// - `b`: Values selected where condition is false.
///
/// All three arrays must have the same shape. `a` and `b` must share the same
/// dtype (f32/f16/bf16). The output dtype matches `a`/`b`.
pub fn select(
    registry: &KernelRegistry,
    cond: &Array,
    a: &Array,
    b: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    // Validate dtypes
    if cond.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "select: cond must be UInt32, got {:?}",
            cond.dtype()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "select: a and b must have matching dtypes: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    // Validate shapes match
    if cond.shape() != a.shape() || a.shape() != b.shape() {
        return Err(KernelError::InvalidShape(format!(
            "select: all inputs must have the same shape: cond={:?}, a={:?}, b={:?}",
            cond.shape(),
            a.shape(),
            b.shape()
        )));
    }

    let numel = a.numel();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), a.shape(), a.dtype()));
    }

    // Ensure contiguous
    let cond_c = super::make_contiguous(cond, registry, queue)?;
    let cond = cond_c.as_ref().unwrap_or(cond);
    let a_c = super::make_contiguous(a, registry, queue)?;
    let a = a_c.as_ref().unwrap_or(a);
    let b_c = super::make_contiguous(b, registry, queue)?;
    let b = b_c.as_ref().unwrap_or(b);

    let kernel_name = select_kernel_name(a.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let out = Array::zeros(registry.device().raw(), a.shape(), a.dtype());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(cond.metal_buffer()), cond.offset());
    encoder.set_buffer(1, Some(a.metal_buffer()), a.offset());
    encoder.set_buffer(2, Some(b.metal_buffer()), b.offset());
    encoder.set_buffer(3, Some(out.metal_buffer()), out.offset());
    let grid_size = MTLSize {
        width: numel,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();
    super::commit_with_mode(&command_buffer, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_kernel_name() {
        assert_eq!(select_kernel_name(DType::Float32).unwrap(), "select_f32");
        assert_eq!(select_kernel_name(DType::Float16).unwrap(), "select_f16");
        assert_eq!(select_kernel_name(DType::Bfloat16).unwrap(), "select_bf16");
        assert!(select_kernel_name(DType::UInt32).is_err());
    }

    fn setup() -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu_dev.new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();
        (registry, queue)
    }

    #[test]
    fn test_select_dtype_validation() {
        let (registry, queue) = setup();
        let device = registry.device().raw();

        // Mismatched a/b dtypes should fail
        let cond = Array::zeros(device, &[4], DType::UInt32);
        let a = Array::zeros(device, &[4], DType::Float32);
        let b = Array::zeros(device, &[4], DType::Float32);

        // Valid case should work
        let result = select(&registry, &cond, &a, &b, &queue);
        assert!(result.is_ok());

        // Wrong cond dtype
        let bad_cond = Array::zeros(device, &[4], DType::Float32);
        let result = select(&registry, &bad_cond, &a, &b, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_basic() {
        let (registry, queue) = setup();
        let device = registry.device().raw();

        // cond = [1, 0, 1, 0]
        let cond = Array::from_slice(device, &[1u32, 0, 1, 0], vec![4]);
        // a = [10, 20, 30, 40]
        let a = Array::from_slice(device, &[10.0f32, 20.0, 30.0, 40.0], vec![4]);
        // b = [100, 200, 300, 400]
        let b = Array::from_slice(device, &[100.0f32, 200.0, 300.0, 400.0], vec![4]);

        let out = select(&registry, &cond, &a, &b, &queue).unwrap();
        let result = out.to_vec_checked::<f32>();
        assert_eq!(result, vec![10.0, 200.0, 30.0, 400.0]);
    }
}
