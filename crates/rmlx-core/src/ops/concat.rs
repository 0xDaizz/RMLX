//! Concatenate arrays along a specified axis via strided copy.
//!
//! Supports f32, f16, bf16. All inputs must share the same dtype and shape
//! on all axes except the concatenation axis.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader for concatenation along an arbitrary axis.
///
/// Each thread copies one element from the correct input array into the output.
/// Parameters:
///   - `out`: output buffer
///   - `n_inputs`: number of input arrays
///   - `out_numel`: total elements in output
///   - `axis`: concatenation axis
///   - `ndim`: number of dimensions
///   - `out_shape`: output shape [ndim]
///   - `input_axis_offsets`: cumulative offsets along axis per input [n_inputs + 1]
///   - `input_ptrs_*`: up to 8 input buffers (we encode them as separate buffer bindings)
///
/// For simplicity and Metal buffer slot limits, we use a single-kernel approach:
/// the host copies input data into the output buffer directly using Metal blit
/// or a flat copy kernel. We use the copy_v kernels to do strided copies.
///
/// However, for a proper GPU-only concat, we provide a kernel that given
/// a flat output index, determines which input it came from and copies.
pub const CONCAT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Concat kernel: copies from two input arrays into one output along an axis.
// For more inputs, the host dispatches this kernel repeatedly or chains.
//
// Flat copy approach: each thread handles one element.
// We split the output into two regions along the concat axis.

kernel void concat2_flat_f32(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint& numel_a   [[buffer(3)]],
    constant uint& numel_total [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id < numel_a) {
        out[id] = a[id];
    } else if (id < numel_total) {
        out[id] = b[id - numel_a];
    }
}

kernel void concat2_flat_f16(
    device const half* a    [[buffer(0)]],
    device const half* b    [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint& numel_a   [[buffer(3)]],
    constant uint& numel_total [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id < numel_a) {
        out[id] = a[id];
    } else if (id < numel_total) {
        out[id] = b[id - numel_a];
    }
}

kernel void concat2_flat_bf16(
    device const bfloat* a    [[buffer(0)]],
    device const bfloat* b    [[buffer(1)]],
    device bfloat*       out  [[buffer(2)]],
    constant uint& numel_a    [[buffer(3)]],
    constant uint& numel_total [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id < numel_a) {
        out[id] = a[id];
    } else if (id < numel_total) {
        out[id] = b[id - numel_a];
    }
}

// General axis-aware concat for 2 inputs. Each thread computes one output
// element, determining whether it comes from a or b based on the coordinate
// along the concat axis.
kernel void concat2_axis_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint& axis_dim_a   [[buffer(3)]],
    constant uint& axis_dim_out [[buffer(4)]],
    constant uint& outer_size   [[buffer(5)]],
    constant uint& inner_size   [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = outer_size * axis_dim_out * inner_size;
    if (id >= total) return;

    uint inner_idx = id % inner_size;
    uint tmp = id / inner_size;
    uint axis_idx = tmp % axis_dim_out;
    uint outer_idx = tmp / axis_dim_out;

    if (axis_idx < axis_dim_a) {
        uint a_idx = outer_idx * axis_dim_a * inner_size + axis_idx * inner_size + inner_idx;
        out[id] = a[a_idx];
    } else {
        uint b_axis = axis_idx - axis_dim_a;
        uint axis_dim_b = axis_dim_out - axis_dim_a;
        uint b_idx = outer_idx * axis_dim_b * inner_size + b_axis * inner_size + inner_idx;
        out[id] = b[b_idx];
    }
}

kernel void concat2_axis_f16(
    device const half* a        [[buffer(0)]],
    device const half* b        [[buffer(1)]],
    device half*       out      [[buffer(2)]],
    constant uint& axis_dim_a   [[buffer(3)]],
    constant uint& axis_dim_out [[buffer(4)]],
    constant uint& outer_size   [[buffer(5)]],
    constant uint& inner_size   [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = outer_size * axis_dim_out * inner_size;
    if (id >= total) return;

    uint inner_idx = id % inner_size;
    uint tmp = id / inner_size;
    uint axis_idx = tmp % axis_dim_out;
    uint outer_idx = tmp / axis_dim_out;

    if (axis_idx < axis_dim_a) {
        uint a_idx = outer_idx * axis_dim_a * inner_size + axis_idx * inner_size + inner_idx;
        out[id] = a[a_idx];
    } else {
        uint b_axis = axis_idx - axis_dim_a;
        uint axis_dim_b = axis_dim_out - axis_dim_a;
        uint b_idx = outer_idx * axis_dim_b * inner_size + b_axis * inner_size + inner_idx;
        out[id] = b[b_idx];
    }
}

kernel void concat2_axis_bf16(
    device const bfloat* a      [[buffer(0)]],
    device const bfloat* b      [[buffer(1)]],
    device bfloat*       out    [[buffer(2)]],
    constant uint& axis_dim_a   [[buffer(3)]],
    constant uint& axis_dim_out [[buffer(4)]],
    constant uint& outer_size   [[buffer(5)]],
    constant uint& inner_size   [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = outer_size * axis_dim_out * inner_size;
    if (id >= total) return;

    uint inner_idx = id % inner_size;
    uint tmp = id / inner_size;
    uint axis_idx = tmp % axis_dim_out;
    uint outer_idx = tmp / axis_dim_out;

    if (axis_idx < axis_dim_a) {
        uint a_idx = outer_idx * axis_dim_a * inner_size + axis_idx * inner_size + inner_idx;
        out[id] = a[a_idx];
    } else {
        uint b_axis = axis_idx - axis_dim_a;
        uint axis_dim_b = axis_dim_out - axis_dim_a;
        uint b_idx = outer_idx * axis_dim_b * inner_size + b_axis * inner_size + inner_idx;
        out[id] = b[b_idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register concat kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("concat", CONCAT_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn axis_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("concat2_axis_f32"),
        DType::Float16 => Ok("concat2_axis_f16"),
        DType::Bfloat16 => Ok("concat2_axis_bf16"),
        _ => Err(KernelError::InvalidShape(format!(
            "concat: unsupported dtype {:?}; expected f32/f16/bf16",
            dtype
        ))),
    }
}

/// Create a Metal buffer from a single u32 value.
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Concatenate two arrays along the specified axis.
///
/// Both arrays must have the same dtype and matching shapes on all axes
/// except `axis`.
///
/// # Arguments
/// - `a`, `b`: Input arrays.
/// - `axis`: The axis along which to concatenate.
///
/// # Returns
/// A new array with `shape[axis] = a.shape[axis] + b.shape[axis]`.
pub fn concat(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    axis: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate dtypes match
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "concat: dtype mismatch: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    // Validate ndim match
    if a.ndim() != b.ndim() {
        return Err(KernelError::InvalidShape(format!(
            "concat: ndim mismatch: {} vs {}",
            a.ndim(),
            b.ndim()
        )));
    }

    let ndim = a.ndim();
    if axis >= ndim {
        return Err(KernelError::InvalidShape(format!(
            "concat: axis {} out of range for {}D arrays",
            axis, ndim
        )));
    }

    // Validate all dimensions match except axis
    for d in 0..ndim {
        if d != axis && a.shape()[d] != b.shape()[d] {
            return Err(KernelError::InvalidShape(format!(
                "concat: shape mismatch on dim {}: {} vs {} (shapes {:?} vs {:?})",
                d,
                a.shape()[d],
                b.shape()[d],
                a.shape(),
                b.shape()
            )));
        }
    }

    let axis_dim_a = a.shape()[axis];
    let axis_dim_b = b.shape()[axis];
    let axis_dim_out = axis_dim_a + axis_dim_b;

    // Build output shape
    let mut out_shape = a.shape().to_vec();
    out_shape[axis] = axis_dim_out;

    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok(Array::zeros(
            registry.device().raw(),
            &out_shape,
            a.dtype(),
        ));
    }

    // Ensure contiguous
    let a_c = super::make_contiguous(a, registry, queue)?;
    let a = a_c.as_ref().unwrap_or(a);
    let b_c = super::make_contiguous(b, registry, queue)?;
    let b = b_c.as_ref().unwrap_or(b);

    // Compute outer_size (product of dims before axis) and inner_size (product of dims after axis)
    let outer_size: usize = a.shape()[..axis].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size: usize = a.shape()[axis + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let kernel_name = axis_kernel_name(a.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let out = Array::zeros(registry.device().raw(), &out_shape, a.dtype());

    let device = registry.device().raw();

    let axis_dim_a_buf = make_u32_buf(device, super::checked_u32(axis_dim_a, "axis_dim_a")?);
    let axis_dim_out_buf = make_u32_buf(device, super::checked_u32(axis_dim_out, "axis_dim_out")?);
    let outer_buf = make_u32_buf(device, super::checked_u32(outer_size, "outer_size")?);
    let inner_buf = make_u32_buf(device, super::checked_u32(inner_size, "inner_size")?);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    encoder.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(3, Some(&axis_dim_a_buf), 0);
    encoder.set_buffer(4, Some(&axis_dim_out_buf), 0);
    encoder.set_buffer(5, Some(&outer_buf), 0);
    encoder.set_buffer(6, Some(&inner_buf), 0);

    let grid_size = MTLSize::new(out_numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(
            pipeline.max_total_threads_per_threadgroup(),
            out_numel as u64,
        ),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Concatenate multiple arrays along the specified axis.
///
/// All arrays must have the same dtype and matching shapes on all axes
/// except `axis`. Implemented by pairwise reduction using [`concat`].
pub fn concat_many(
    registry: &KernelRegistry,
    arrays: &[&Array],
    axis: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if arrays.is_empty() {
        return Err(KernelError::InvalidShape(
            "concat_many: no arrays provided".into(),
        ));
    }
    if arrays.len() == 1 {
        return super::copy::copy(registry, arrays[0], queue);
    }

    // Pairwise reduction
    let mut result = concat(registry, arrays[0], arrays[1], axis, queue)?;
    for arr in &arrays[2..] {
        result = concat(registry, &result, arr, axis, queue)?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_axis0_1d() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let gpu_dev = rmlx_metal::device::GpuDevice::new(device.clone());
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();

        let a = Array::from_slice(&device, &[1.0f32, 2.0, 3.0], vec![3]);
        let b = Array::from_slice(&device, &[4.0f32, 5.0], vec![2]);

        let out = concat(&registry, &a, &b, 0, &queue).unwrap();
        assert_eq!(out.shape(), &[5]);
        let result = out.to_vec_checked::<f32>();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concat_axis1_2d() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let gpu_dev = rmlx_metal::device::GpuDevice::new(device.clone());
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();

        // a: [[1,2],[3,4]] shape [2, 2]
        let a = Array::from_slice(&device, &[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        // b: [[5],[6]] shape [2, 1]
        let b = Array::from_slice(&device, &[5.0f32, 6.0], vec![2, 1]);

        let out = concat(&registry, &a, &b, 1, &queue).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let result = out.to_vec_checked::<f32>();
        assert_eq!(result, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_concat_shape_mismatch() {
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let gpu_dev = rmlx_metal::device::GpuDevice::new(device.clone());
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();

        let a = Array::from_slice(&device, &[1.0f32, 2.0], vec![2]);
        let b = Array::from_slice(&device, &[3.0f32, 4.0, 5.0, 6.0], vec![2, 2]);

        let result = concat(&registry, &a, &b, 0, &queue);
        assert!(result.is_err());
    }
}
