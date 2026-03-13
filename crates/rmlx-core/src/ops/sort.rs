//! Sort and argsort operations using bitonic sort on Metal.
//!
//! Bitonic sort is well-suited for GPU execution: it has a fixed comparison
//! pattern that maps naturally to parallel threads. We sort within threadgroups
//! and handle arbitrary lengths by padding to the next power of 2.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use rmlx_metal::MTLSize;
use rmlx_metal::ComputePass;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::MTLResourceOptions;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for bitonic sort and argsort kernels.
///
/// The kernel sorts along the last dimension of a 2D `[rows, cols]` array.
/// Higher-dimensional inputs should be reshaped to 2D before dispatch.
///
/// `padded_len` is the next power of 2 >= cols. Unused slots are filled with
/// +INF (ascending) or -INF (descending) so they sort to the end.
pub const SORT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Bitonic sort — sorts one row of length `cols`, padded to `padded_len`.
// Each threadgroup handles one row.
kernel void sort_f32(
    device const float*  input       [[buffer(0)]],
    device       float*  output      [[buffer(1)]],
    constant     uint&   cols        [[buffer(2)]],
    constant     uint&   padded_len  [[buffer(3)]],
    constant     uint&   descending  [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory for one row (padded).
    threadgroup float local[2048];

    uint row = tgid;
    uint row_offset = row * cols;

    // Load into shared memory with padding.
    float pad_val = (descending != 0) ? -INFINITY : INFINITY;
    for (uint i = tid; i < padded_len; i += tg_size) {
        local[i] = (i < cols) ? input[row_offset + i] : pad_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort.
    for (uint size = 2; size <= padded_len; size *= 2) {
        for (uint stride = size / 2; stride > 0; stride /= 2) {
            for (uint i = tid; i < padded_len / 2; i += tg_size) {
                uint pos = 2 * i - (i & (stride - 1));
                bool asc_block = ((pos & size) == 0);
                bool do_asc = (descending != 0) ? !asc_block : asc_block;

                float a = local[pos];
                float b = local[pos + stride];
                if (do_asc ? (a > b) : (a < b)) {
                    local[pos] = b;
                    local[pos + stride] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back (only the real elements).
    for (uint i = tid; i < cols; i += tg_size) {
        output[row_offset + i] = local[i];
    }
}

// Argsort — same as sort but tracks original indices.
kernel void argsort_f32(
    device const float*  input       [[buffer(0)]],
    device       uint*   output      [[buffer(1)]],
    constant     uint&   cols        [[buffer(2)]],
    constant     uint&   padded_len  [[buffer(3)]],
    constant     uint&   descending  [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float local_vals[2048];
    threadgroup uint  local_idxs[2048];

    uint row = tgid;
    uint row_offset = row * cols;

    float pad_val = (descending != 0) ? -INFINITY : INFINITY;
    for (uint i = tid; i < padded_len; i += tg_size) {
        local_vals[i] = (i < cols) ? input[row_offset + i] : pad_val;
        local_idxs[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort on (value, index) pairs.
    for (uint size = 2; size <= padded_len; size *= 2) {
        for (uint stride = size / 2; stride > 0; stride /= 2) {
            for (uint i = tid; i < padded_len / 2; i += tg_size) {
                uint pos = 2 * i - (i & (stride - 1));
                bool asc_block = ((pos & size) == 0);
                bool do_asc = (descending != 0) ? !asc_block : asc_block;

                float a = local_vals[pos];
                float b = local_vals[pos + stride];
                if (do_asc ? (a > b) : (a < b)) {
                    local_vals[pos] = b;
                    local_vals[pos + stride] = a;
                    uint tmp = local_idxs[pos];
                    local_idxs[pos] = local_idxs[pos + stride];
                    local_idxs[pos + stride] = tmp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back indices (only the real elements).
    for (uint i = tid; i < cols; i += tg_size) {
        output[row_offset + i] = local_idxs[i];
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register sort kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("sort_ops", SORT_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_u32_buf(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, val: u32) -> rmlx_metal::MtlBuffer {
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new(&val as *const u32 as *const std::ffi::c_void as *mut std::ffi::c_void).unwrap(), 4_usize, MTLResourceOptions::StorageModeShared).unwrap() }
}

/// Next power of 2 >= n (capped at 2048 for threadgroup memory).
fn next_pow2(n: usize) -> usize {
    let mut v = 1;
    while v < n {
        v <<= 1;
    }
    v
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sort `input` along `axis`, returning a new array with sorted values.
///
/// * `axis`       - dimension to sort along.
/// * `descending` - if true, sort in descending order.
///
/// Currently supports sorting dimensions up to 2048 elements (threadgroup
/// memory limit for bitonic sort).
pub fn sort(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    descending: bool,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float32 {
        return Err(KernelError::NotFound(format!(
            "sort only supports Float32, got {:?}",
            input.dtype()
        )));
    }
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(KernelError::InvalidShape(format!(
            "sort: axis {} out of range for {}D array",
            axis, ndim
        )));
    }

    let shape = input.shape();
    let cols = shape[axis];
    if cols > 2048 {
        return Err(KernelError::InvalidShape(format!(
            "sort: axis size {} exceeds max 2048 for bitonic sort",
            cols
        )));
    }
    if cols == 0 {
        return Ok(Array::zeros(registry.device().raw(), shape, input.dtype()));
    }

    // Make contiguous, then reshape so the sort axis is last.
    let input_c = super::make_contiguous(input, registry, queue)?;
    let input_ref = input_c.as_ref().unwrap_or(input);

    let (work_2d, out_shape) = reshape_for_axis_last(input_ref, axis, registry, queue)?;

    let rows: usize = work_2d.shape()[0];
    let padded_len = next_pow2(cols);

    let pipeline = registry.get_pipeline("sort_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, work_2d.shape(), DType::Float32);

    let cols_buf = make_u32_buf(dev, cols as u32);
    let padded_buf = make_u32_buf(dev, padded_len as u32);
    let desc_buf = make_u32_buf(dev, if descending { 1 } else { 0 });

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(work_2d.metal_buffer()), work_2d.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&cols_buf), 0);
    enc.set_buffer(3, Some(&padded_buf), 0);
    enc.set_buffer(4, Some(&desc_buf), 0);
    let tg_size = std::cmp::min(
        padded_len,
        pipeline.maxTotalThreadsPerThreadgroup(),
    );
    enc.dispatch_threadgroups(MTLSize { width: rows, height: 1, depth: 1 }, MTLSize { width: tg_size, height: 1, depth: 1 });
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    // Reshape back to original shape if needed.
    if axis == ndim - 1 {
        out.reshape(out_shape)
    } else {
        // We transposed axis to the end; need to transpose back.
        let transposed_2d = out.reshape(permuted_shape(shape, axis))?;
        let view = unpermute_view(&transposed_2d, axis, shape);
        let result = super::copy::copy(registry, &view, queue)?;
        result.reshape(out_shape)
    }
}

/// Return the indices that would sort `input` along `axis`.
///
/// Output dtype is UInt32.
pub fn argsort(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    descending: bool,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float32 {
        return Err(KernelError::NotFound(format!(
            "argsort only supports Float32, got {:?}",
            input.dtype()
        )));
    }
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(KernelError::InvalidShape(format!(
            "argsort: axis {} out of range for {}D array",
            axis, ndim
        )));
    }

    let shape = input.shape();
    let cols = shape[axis];
    if cols > 2048 {
        return Err(KernelError::InvalidShape(format!(
            "argsort: axis size {} exceeds max 2048 for bitonic sort",
            cols
        )));
    }
    if cols == 0 {
        let out_shape = shape.to_vec();
        return Ok(Array::zeros(
            registry.device().raw(),
            &out_shape,
            DType::UInt32,
        ));
    }

    let input_c = super::make_contiguous(input, registry, queue)?;
    let input_ref = input_c.as_ref().unwrap_or(input);

    let (work_2d, _out_shape) = reshape_for_axis_last(input_ref, axis, registry, queue)?;

    let rows = work_2d.shape()[0];
    let padded_len = next_pow2(cols);

    let pipeline = registry.get_pipeline("argsort_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let out_2d = Array::zeros(dev, &[rows, cols], DType::UInt32);

    let cols_buf = make_u32_buf(dev, cols as u32);
    let padded_buf = make_u32_buf(dev, padded_len as u32);
    let desc_buf = make_u32_buf(dev, if descending { 1 } else { 0 });

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(work_2d.metal_buffer()), work_2d.offset());
    enc.set_buffer(1, Some(out_2d.metal_buffer()), 0);
    enc.set_buffer(2, Some(&cols_buf), 0);
    enc.set_buffer(3, Some(&padded_buf), 0);
    enc.set_buffer(4, Some(&desc_buf), 0);
    let tg_size = std::cmp::min(
        padded_len,
        pipeline.maxTotalThreadsPerThreadgroup(),
    );
    enc.dispatch_threadgroups(MTLSize { width: rows, height: 1, depth: 1 }, MTLSize { width: tg_size, height: 1, depth: 1 });
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    out_2d.reshape(shape.to_vec())
}

// ---------------------------------------------------------------------------
// Internal helpers for axis permutation
// ---------------------------------------------------------------------------

/// Reshape input so the target axis is last, returning the 2D work array
/// and the original output shape.
fn reshape_for_axis_last(
    input: &Array,
    axis: usize,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Vec<usize>), KernelError> {
    let shape = input.shape();
    let ndim = shape.len();
    let out_shape = shape.to_vec();
    let cols = shape[axis];
    let outer: usize = shape.iter().product::<usize>() / cols;

    if axis == ndim - 1 {
        let work = input.reshape(vec![outer, cols])?;
        Ok((work, out_shape))
    } else {
        // Move target axis to the end via permuted view + copy.
        let _perm_shape = permuted_shape(shape, axis);
        let view = permute_view(input, axis);
        let contiguous = super::copy::copy(registry, &view, queue)?;
        let work = contiguous.reshape(vec![outer, cols])?;
        Ok((work, out_shape))
    }
}

/// Build the shape after moving `axis` to the end.
fn permuted_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..shape.len()).collect();
    perm.remove(axis);
    perm.push(axis);
    perm.iter().map(|&p| shape[p]).collect()
}

/// Create a view with `axis` moved to the end (non-contiguous).
fn permute_view(input: &Array, axis: usize) -> Array {
    let ndim = input.ndim();
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.remove(axis);
    perm.push(axis);

    let src_strides = input.strides();
    let shape = input.shape();
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
    input.view(new_shape, new_strides, input.offset())
}

/// Create a view that reverses the permutation (moves last axis back to `axis`).
fn unpermute_view(input: &Array, axis: usize, original_shape: &[usize]) -> Array {
    let ndim = original_shape.len();
    // The current order is: [0..axis-1, axis+1..ndim-1, axis]
    // We need the inverse permutation.
    let mut inv_perm = vec![0usize; ndim];
    let mut src_pos = 0;
    for (i, inv) in inv_perm.iter_mut().enumerate() {
        if i == axis {
            continue;
        }
        *inv = src_pos;
        src_pos += 1;
    }
    inv_perm[axis] = ndim - 1;

    let cur_shape = input.shape();
    let cur_strides = input.strides();
    let new_shape: Vec<usize> = inv_perm.iter().map(|&p| cur_shape[p]).collect();
    let new_strides: Vec<usize> = inv_perm.iter().map(|&p| cur_strides[p]).collect();
    input.view(new_shape, new_strides, input.offset())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let device = rmlx_metal::device::GpuDevice::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register sort kernels");
        crate::ops::copy::register(&registry).expect("register copy kernels");
        (registry, queue)
    }

    #[test]
    fn test_sort_ascending() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], vec![8]);
        let out = sort(&reg, &input, 0, false, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
    }

    #[test]
    fn test_sort_descending() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[3.0f32, 1.0, 4.0, 1.0, 5.0], vec![5]);
        let out = sort(&reg, &input, 0, true, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![5.0, 4.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_argsort_ascending() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[30.0f32, 10.0, 20.0], vec![3]);
        let out = argsort(&reg, &input, 0, false, &q).unwrap();
        let vals: Vec<u32> = out.to_vec_checked();
        assert_eq!(vals, vec![1, 2, 0]);
    }
}
