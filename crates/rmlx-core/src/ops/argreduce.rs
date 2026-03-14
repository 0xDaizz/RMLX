//! Argmin and argmax reduction operations.
//!
//! These reduce along a specified axis and return the index (UInt32) of the
//! minimum or maximum element. Each threadgroup processes one row of the
//! 2D work array using SIMD reductions for efficiency.

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
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for argmin/argmax reduction kernels.
///
/// Each threadgroup handles one row. Threads cooperate via SIMD reductions
/// to find the min/max value and its index. Cross-simdgroup reduction uses
/// threadgroup memory.
pub const ARGREDUCE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint SIMD_SIZE = 32;

// Per-row argmax: output[row] = index of max element in row.
kernel void argmax_row_f32(
    device const float*  input   [[buffer(0)]],
    device       uint*   output  [[buffer(1)]],
    constant     uint&   cols    [[buffer(2)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];
    threadgroup uint  local_idxs[SIMD_SIZE];

    uint row = tgid;
    uint row_offset = row * cols;

    // Each thread finds its local max across its assigned elements.
    float best_val = -INFINITY;
    uint  best_idx = 0;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[row_offset + i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // SIMD reduction: find max within simdgroup.
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Cross-simdgroup reduction via threadgroup memory.
    if (simd_group_id == 0) {
        local_vals[simd_lane_id] = -INFINITY;
        local_idxs[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        local_vals[simd_group_id] = best_val;
        local_idxs[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        best_val = local_vals[simd_lane_id];
        best_idx = local_idxs[simd_lane_id];

        for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (simd_lane_id == 0) {
            output[row] = best_idx;
        }
    }
}

// Per-row argmin.
kernel void argmin_row_f32(
    device const float*  input   [[buffer(0)]],
    device       uint*   output  [[buffer(1)]],
    constant     uint&   cols    [[buffer(2)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];
    threadgroup uint  local_idxs[SIMD_SIZE];

    uint row = tgid;
    uint row_offset = row * cols;

    float best_val = INFINITY;
    uint  best_idx = 0;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[row_offset + i];
        if (v < best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val < best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    if (simd_group_id == 0) {
        local_vals[simd_lane_id] = INFINITY;
        local_idxs[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        local_vals[simd_group_id] = best_val;
        local_idxs[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        best_val = local_vals[simd_lane_id];
        best_idx = local_idxs[simd_lane_id];

        for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val < best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (simd_lane_id == 0) {
            output[row] = best_idx;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register argreduce kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("argreduce_ops", ARGREDUCE_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_u32_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: u32,
) -> rmlx_metal::MtlBuffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(
                    &val as *const u32 as *const std::ffi::c_void as *mut std::ffi::c_void,
                )
                .unwrap(),
                4_usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

fn dispatch_argreduce_2d(
    registry: &KernelRegistry,
    input_2d: &Array,
    kernel_name: &str,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let rows = input_2d.shape()[0];

    let cols = input_2d.shape()[1];

    let pipeline = registry.get_pipeline(kernel_name, DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &[rows], DType::UInt32);

    let cols_buf = make_u32_buf(dev, cols as u32);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(input_2d.metal_buffer()), input_2d.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&cols_buf), 0);
    let tg_size = std::cmp::min(cols, pipeline.maxTotalThreadsPerThreadgroup());
    // Must be at least 32 (SIMD width) so simd_shuffle_down reads valid lanes.
    let tg_size = std::cmp::min(tg_size, 1024).max(32);
    enc.dispatch_threadgroups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

fn prepare_and_dispatch(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    kernel_name: &str,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float32 {
        return Err(KernelError::NotFound(format!(
            "argreduce ops only support Float32, got {:?}",
            input.dtype()
        )));
    }
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(KernelError::InvalidShape(format!(
            "argreduce: axis {} out of range for {}D array",
            axis, ndim
        )));
    }
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot argreduce empty array".into(),
        ));
    }

    let input_c = super::make_contiguous(input, registry, queue)?;
    let input_ref = input_c.as_ref().unwrap_or(input);

    let shape = input_ref.shape();
    let cols = shape[axis];
    let outer: usize = shape.iter().product::<usize>() / cols;

    let work_2d = if axis == ndim - 1 {
        input_ref.reshape(vec![outer, cols])?
    } else {
        // Move target axis to the end.
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(axis);
        perm.push(axis);

        let src_strides = input_ref.strides();
        let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
        let view = input_ref.view(new_shape, new_strides, input_ref.offset());

        let contiguous = super::copy::copy(registry, &view, queue)?;
        contiguous.reshape(vec![outer, cols])?
    };

    let result_1d = dispatch_argreduce_2d(registry, &work_2d, kernel_name, queue)?;

    // Build output shape: original shape with reduced axis removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    result_1d.reshape(out_shape)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the indices of the minimum values along `axis`.
///
/// Output dtype is UInt32. Currently supports Float32 input only.
pub fn argmin(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    prepare_and_dispatch(registry, input, axis, "argmin_row_f32", queue)
}

/// Return the indices of the maximum values along `axis`.
///
/// Output dtype is UInt32. Currently supports Float32 input only.
pub fn argmax(
    registry: &KernelRegistry,
    input: &Array,
    axis: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    prepare_and_dispatch(registry, input, axis, "argmax_row_f32", queue)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(KernelRegistry, rmlx_metal::MtlQueue)> {
        let device = crate::test_utils::test_gpu()?;
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register argreduce kernels");
        crate::ops::copy::register(&registry).expect("register copy kernels");
        Some((registry, queue))
    }

    #[test]
    fn test_argmax_1d() {
        let Some((reg, q)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[3.0f32, 1.0, 5.0, 2.0, 4.0], vec![5]);
        let out = argmax(&reg, &input, 0, &q).unwrap();
        let vals: Vec<u32> = out.to_vec_checked();
        assert_eq!(vals, vec![2]); // index of 5.0
    }

    #[test]
    fn test_argmin_1d() {
        let Some((reg, q)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[3.0f32, 1.0, 5.0, 2.0, 4.0], vec![5]);
        let out = argmin(&reg, &input, 0, &q).unwrap();
        let vals: Vec<u32> = out.to_vec_checked();
        assert_eq!(vals, vec![1]); // index of 1.0
    }

    #[test]
    fn test_argmax_2d_axis1() {
        let Some((reg, q)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = reg.device().raw();
        // [[1, 5, 3], [9, 2, 7]]
        let input = Array::from_slice(dev, &[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]);
        let out = argmax(&reg, &input, 1, &q).unwrap();
        assert_eq!(out.shape(), &[2]);
        let vals: Vec<u32> = out.to_vec_checked();
        assert_eq!(vals, vec![1, 0]); // max in row 0 is at col 1 (5.0), row 1 at col 0 (9.0)
    }

    #[test]
    fn test_argmin_2d_axis0() {
        let Some((reg, q)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = reg.device().raw();
        // [[10, 20], [5, 25]]
        let input = Array::from_slice(dev, &[10.0f32, 20.0, 5.0, 25.0], vec![2, 2]);
        let out = argmin(&reg, &input, 0, &q).unwrap();
        assert_eq!(out.shape(), &[2]);
        let vals: Vec<u32> = out.to_vec_checked();
        assert_eq!(vals, vec![1, 0]); // col 0: min at row 1 (5.0), col 1: min at row 0 (20.0)
    }
}
