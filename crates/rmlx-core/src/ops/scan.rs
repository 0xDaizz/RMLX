//! Prefix scan operations: cumsum and cumprod.
//!
//! Uses an inclusive Hillis-Steele parallel prefix scan on Metal.
//! Each threadgroup handles one row of the 2D work array. For N-D inputs,
//! the target axis is moved to the last position via permutation.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for prefix scan kernels (Hillis-Steele, inclusive).
///
/// Each threadgroup processes one row. Two threadgroup buffers are ping-ponged
/// for the scan passes. Maximum row length is 2048 (threadgroup memory limit).
pub const SCAN_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Inclusive prefix sum (Hillis-Steele) for one row.
kernel void cumsum_f32(
    device const float*  input   [[buffer(0)]],
    device       float*  output  [[buffer(1)]],
    constant     uint&   cols    [[buffer(2)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float buf_a[2048];
    threadgroup float buf_b[2048];

    uint row = tgid;
    uint row_offset = row * cols;

    // Load row into shared memory.
    for (uint i = tid; i < cols; i += tg_size) {
        buf_a[i] = input[row_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan.
    // ping-pong between buf_a and buf_b.
    bool src_is_a = true;
    for (uint offset = 1; offset < cols; offset *= 2) {
        for (uint i = tid; i < cols; i += tg_size) {
            float val;
            if (src_is_a) {
                val = buf_a[i];
                if (i >= offset) val += buf_a[i - offset];
                buf_b[i] = val;
            } else {
                val = buf_b[i];
                if (i >= offset) val += buf_b[i - offset];
                buf_a[i] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        src_is_a = !src_is_a;
    }

    // Write output.
    for (uint i = tid; i < cols; i += tg_size) {
        output[row_offset + i] = src_is_a ? buf_a[i] : buf_b[i];
    }
}

// Inclusive prefix product (Hillis-Steele) for one row.
kernel void cumprod_f32(
    device const float*  input   [[buffer(0)]],
    device       float*  output  [[buffer(1)]],
    constant     uint&   cols    [[buffer(2)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float buf_a[2048];
    threadgroup float buf_b[2048];

    uint row = tgid;
    uint row_offset = row * cols;

    for (uint i = tid; i < cols; i += tg_size) {
        buf_a[i] = input[row_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool src_is_a = true;
    for (uint offset = 1; offset < cols; offset *= 2) {
        for (uint i = tid; i < cols; i += tg_size) {
            float val;
            if (src_is_a) {
                val = buf_a[i];
                if (i >= offset) val *= buf_a[i - offset];
                buf_b[i] = val;
            } else {
                val = buf_b[i];
                if (i >= offset) val *= buf_b[i - offset];
                buf_a[i] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        src_is_a = !src_is_a;
    }

    for (uint i = tid; i < cols; i += tg_size) {
        output[row_offset + i] = src_is_a ? buf_a[i] : buf_b[i];
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register scan kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("scan_ops", SCAN_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_u32_buf(device: &metal::Device, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ---------------------------------------------------------------------------
// Internal: dispatch a scan kernel along the last axis of a 2D array.
// ---------------------------------------------------------------------------

fn dispatch_scan_2d(
    registry: &KernelRegistry,
    input_2d: &Array,
    kernel_name: &str,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let rows = input_2d.shape()[0];
    let cols = input_2d.shape()[1];

    let pipeline = registry.get_pipeline(kernel_name, DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, input_2d.shape(), DType::Float32);

    let cols_buf = make_u32_buf(dev, cols as u32);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(input_2d.metal_buffer()), input_2d.offset() as u64);
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&cols_buf), 0);

    let tg_size = std::cmp::min(cols as u64, pipeline.max_total_threads_per_threadgroup());
    let tg_size = std::cmp::min(tg_size, 1024);
    enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Internal: reshape to 2D with target axis last.
// ---------------------------------------------------------------------------

fn prepare_2d(
    input: &Array,
    axis: usize,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<(Array, Vec<usize>), KernelError> {
    let ndim = input.ndim();
    let shape = input.shape();
    let cols = shape[axis];
    let outer: usize = shape.iter().product::<usize>() / cols;

    if axis == ndim - 1 {
        let work = input.reshape(vec![outer, cols])?;
        Ok((work, shape.to_vec()))
    } else {
        // Move target axis to the end.
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(axis);
        perm.push(axis);

        let src_strides = input.strides();
        let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
        let view = input.view(new_shape, new_strides, input.offset());

        let contiguous = super::copy::copy(registry, &view, queue)?;
        let work = contiguous.reshape(vec![outer, cols])?;
        Ok((work, shape.to_vec()))
    }
}

fn restore_shape(
    result_2d: Array,
    axis: usize,
    original_shape: &[usize],
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let ndim = original_shape.len();
    if axis == ndim - 1 {
        return result_2d.reshape(original_shape.to_vec());
    }

    // The 2D result is in permuted order (axis moved to end).
    // Rebuild the permuted N-D shape, then un-permute via view + copy.
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.remove(axis);
    perm.push(axis);
    let perm_shape: Vec<usize> = perm.iter().map(|&p| original_shape[p]).collect();
    let perm_nd = result_2d.reshape(perm_shape)?;

    // Inverse permutation.
    let mut inv_perm = vec![0usize; ndim];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        inv_perm[old_pos] = new_pos;
    }

    let cur_shape = perm_nd.shape();
    let cur_strides = perm_nd.strides();
    let new_shape: Vec<usize> = inv_perm.iter().map(|&p| cur_shape[p]).collect();
    let new_strides: Vec<usize> = inv_perm.iter().map(|&p| cur_strides[p]).collect();
    let view = perm_nd.view(new_shape, new_strides, perm_nd.offset());

    let contiguous = super::copy::copy(registry, &view, queue)?;
    contiguous.reshape(original_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Cumulative sum along `axis`.
///
/// `output[..., i, ...] = sum(input[..., 0:i+1, ...])` along the given axis.
///
/// Currently supports Float32 only. Axis length must be <= 2048.
pub fn cumsum(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    validate_scan_input(input, axis)?;

    let input_c = super::make_contiguous(input, registry, queue)?;
    let input_ref = input_c.as_ref().unwrap_or(input);

    let (work_2d, orig_shape) = prepare_2d(input_ref, axis, registry, queue)?;
    let result_2d = dispatch_scan_2d(registry, &work_2d, "cumsum_f32", queue)?;
    restore_shape(result_2d, axis, &orig_shape, registry, queue)
}

/// Cumulative product along `axis`.
///
/// `output[..., i, ...] = prod(input[..., 0:i+1, ...])` along the given axis.
///
/// Currently supports Float32 only. Axis length must be <= 2048.
pub fn cumprod(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    validate_scan_input(input, axis)?;

    let input_c = super::make_contiguous(input, registry, queue)?;
    let input_ref = input_c.as_ref().unwrap_or(input);

    let (work_2d, orig_shape) = prepare_2d(input_ref, axis, registry, queue)?;
    let result_2d = dispatch_scan_2d(registry, &work_2d, "cumprod_f32", queue)?;
    restore_shape(result_2d, axis, &orig_shape, registry, queue)
}

fn validate_scan_input(input: &Array, axis: usize) -> Result<(), KernelError> {
    if input.dtype() != DType::Float32 {
        return Err(KernelError::NotFound(format!(
            "scan ops only support Float32, got {:?}",
            input.dtype()
        )));
    }
    if axis >= input.ndim() {
        return Err(KernelError::InvalidShape(format!(
            "scan: axis {} out of range for {}D array",
            axis,
            input.ndim()
        )));
    }
    let cols = input.shape()[axis];
    if cols > 2048 {
        return Err(KernelError::InvalidShape(format!(
            "scan: axis size {} exceeds max 2048",
            cols
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, metal::CommandQueue) {
        let device = rmlx_metal::device::GpuDevice::system_default().expect("Metal device");
        let queue = device.raw().new_command_queue();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register scan kernels");
        crate::ops::copy::register(&registry).expect("register copy kernels");
        (registry, queue)
    }

    #[test]
    fn test_cumsum_1d() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let out = cumsum(&reg, &input, 0, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_cumprod_1d() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let out = cumprod(&reg, &input, 0, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cumsum_2d_axis1() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        // [[1, 2, 3], [4, 5, 6]]
        let input = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = cumsum(&reg, &input, 1, &q).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }
}
