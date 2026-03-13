//! Slice operation: extract a sub-tensor with start/end/stride per dimension.
//!
//! The Metal kernel reads from strided positions in the source array and writes
//! contiguous output. Supports arbitrary dimensionality (up to 8D) with
//! per-dimension start, end, and stride parameters.

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

/// Metal shader source for the slice kernel.
///
/// The kernel takes flattened output indices, converts them to N-D coordinates
/// in the output space, maps them back to source coordinates using
/// `start + coord * stride`, and reads from the source.
///
/// Supports up to 8 dimensions. Shape metadata is passed as constant buffers.
pub const SLICE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_DIMS = 8;

kernel void slice_f32(
    device const float*  src          [[buffer(0)]],
    device       float*  output       [[buffer(1)]],
    constant     uint&   ndim         [[buffer(2)]],
    constant     uint*   out_shape    [[buffer(3)]],
    constant     uint*   out_strides  [[buffer(4)]],
    constant     uint*   src_strides  [[buffer(5)]],
    constant     uint*   starts       [[buffer(6)]],
    constant     uint*   slice_strides [[buffer(7)]],
    constant     uint&   numel        [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numel) return;

    // Convert flat output index to N-D output coordinate, then to source offset.
    uint remaining = id;
    uint src_offset = 0;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        uint src_coord = starts[d] + coord * slice_strides[d];
        src_offset += src_coord * src_strides[d];
    }

    output[id] = src[src_offset];
}

kernel void slice_f16(
    device const half*   src          [[buffer(0)]],
    device       half*   output       [[buffer(1)]],
    constant     uint&   ndim         [[buffer(2)]],
    constant     uint*   out_shape    [[buffer(3)]],
    constant     uint*   out_strides  [[buffer(4)]],
    constant     uint*   src_strides  [[buffer(5)]],
    constant     uint*   starts       [[buffer(6)]],
    constant     uint*   slice_strides [[buffer(7)]],
    constant     uint&   numel        [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numel) return;

    uint remaining = id;
    uint src_offset = 0;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        uint src_coord = starts[d] + coord * slice_strides[d];
        src_offset += src_coord * src_strides[d];
    }

    output[id] = src[src_offset];
}

kernel void slice_bf16(
    device const bfloat* src          [[buffer(0)]],
    device       bfloat* output       [[buffer(1)]],
    constant     uint&   ndim         [[buffer(2)]],
    constant     uint*   out_shape    [[buffer(3)]],
    constant     uint*   out_strides  [[buffer(4)]],
    constant     uint*   src_strides  [[buffer(5)]],
    constant     uint*   starts       [[buffer(6)]],
    constant     uint*   slice_strides [[buffer(7)]],
    constant     uint&   numel        [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= numel) return;

    uint remaining = id;
    uint src_offset = 0;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        uint src_coord = starts[d] + coord * slice_strides[d];
        src_offset += src_coord * src_strides[d];
    }

    output[id] = src[src_offset];
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register slice kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("slice_ops", SLICE_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_u32_buf(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, val: u32) -> rmlx_metal::MtlBuffer {
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new(&val as *const u32 as *const std::ffi::c_void as *mut std::ffi::c_void).unwrap(), 4_usize, MTLResourceOptions::StorageModeShared).unwrap() }
}

fn make_u32_vec_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vals: &[u32],
) -> rmlx_metal::MtlBuffer {
    if vals.is_empty() { // Metal needs at least 1 byte. return device.newBufferWithLength_options(4 as usize, MTLResourceOptions::StorageModeShared).unwrap();

    }
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new(vals.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void).unwrap(), (vals.len() * 4) as u64 as usize, MTLResourceOptions::StorageModeShared).unwrap() }
}

/// Compute contiguous strides for a shape (in elements).
fn contiguous_strides(
    shape: &[usize],
) -> Vec<usize> {
    let ndim = shape.len();

    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract a slice from `input` with per-dimension `starts`, `ends`, and `strides`.
///
/// * `starts[d]`  - starting index along dimension `d` (inclusive).
/// * `ends[d]`    - ending index along dimension `d` (exclusive).
/// * `strides[d]` - step along dimension `d` (must be >= 1).
///
/// The output shape along dimension `d` is `ceil((ends[d] - starts[d]) / strides[d])`.
///
/// All three slices must have the same length as `input.ndim()`.
pub fn slice(
    registry: &KernelRegistry,
    input: &Array,
    starts: &[usize],
    ends: &[usize],
    strides: &[usize],
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let ndim = input.ndim();
    if starts.len() != ndim || ends.len() != ndim || strides.len() != ndim {
        return Err(KernelError::InvalidShape(format!(
            "slice: starts/ends/strides length must match ndim={ndim}, got {}/{}/{}",
            starts.len(),
            ends.len(),
            strides.len()
        )));
    }
    if ndim > 8 {
        return Err(KernelError::InvalidShape(format!(
            "slice: ndim={ndim} exceeds maximum 8"
        )));
    }

    let src_shape = input.shape();
    let mut out_shape = Vec::with_capacity(ndim);
    for d in 0..ndim {
        if strides[d] == 0 {
            return Err(KernelError::InvalidShape(format!(
                "slice: stride for dim {d} must be >= 1"
            )));
        }
        if starts[d] >= ends[d] || ends[d] > src_shape[d] {
            return Err(KernelError::InvalidShape(format!(
                "slice: invalid range [{}..{}) for dim {} with size {}",
                starts[d], ends[d], d, src_shape[d]
            )));
        }
        let extent = ends[d] - starts[d];
        out_shape.push(extent.div_ceil(strides[d]));
    }

    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok(Array::zeros(
            registry.device().raw(),
            &out_shape,
            input.dtype(),
        ));
    }

    // Make input contiguous if needed.
    let input_c = super::make_contiguous(input, registry, queue)?;
    let input = input_c.as_ref().unwrap_or(input);

    let kernel_name = match input.dtype() {
        DType::Float32 => "slice_f32",
        DType::Float16 => "slice_f16",
        DType::Bfloat16 => "slice_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "slice not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &out_shape, input.dtype());

    // Compute source strides (contiguous, in elements).
    let src_strides = contiguous_strides(src_shape);

    // Build constant buffers.
    let ndim_u32 = super::checked_u32(ndim, "ndim")?;
    let numel_u32 = super::checked_u32(out_numel, "out_numel")?;
    let out_strides_elem = contiguous_strides(&out_shape);

    let out_shape_u32: Vec<u32> = out_shape.iter().map(|&v| v as u32).collect();
    let out_strides_u32: Vec<u32> = out_strides_elem.iter().map(|&v| v as u32).collect();
    let src_strides_u32: Vec<u32> = src_strides.iter().map(|&v| v as u32).collect();
    let starts_u32: Vec<u32> = starts.iter().map(|&v| v as u32).collect();
    let slice_strides_u32: Vec<u32> = strides.iter().map(|&v| v as u32).collect();

    let ndim_buf = make_u32_buf(dev, ndim_u32);
    let out_shape_buf = make_u32_vec_buf(dev, &out_shape_u32);
    let out_strides_buf = make_u32_vec_buf(dev, &out_strides_u32);
    let src_strides_buf = make_u32_vec_buf(dev, &src_strides_u32);
    let starts_buf = make_u32_vec_buf(dev, &starts_u32);
    let slice_strides_buf = make_u32_vec_buf(dev, &slice_strides_u32);
    let numel_buf = make_u32_buf(dev, numel_u32);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), input.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&ndim_buf), 0);
    enc.set_buffer(3, Some(&out_shape_buf), 0);
    enc.set_buffer(4, Some(&out_strides_buf), 0);
    enc.set_buffer(5, Some(&src_strides_buf), 0);
    enc.set_buffer(6, Some(&starts_buf), 0);
    enc.set_buffer(7, Some(&slice_strides_buf), 0);
    enc.set_buffer(8, Some(&numel_buf), 0);
    let grid = MTLSize { width: out_numel, height: 1, depth: 1 };
    let tg = MTLSize { width: std::cmp::min(256usize, pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let device = rmlx_metal::device::GpuDevice::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register slice kernels");
        (registry, queue)
    }

    #[test]
    fn test_slice_1d_basic() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5]);
        let out = slice(&reg, &input, &[1], &[4], &[1], &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_slice_1d_strided() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let input = Array::from_slice(dev, &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6]);
        let out = slice(&reg, &input, &[0], &[6], &[2], &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_2d() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Array::from_slice(dev, &data, vec![3, 4]);
        // Slice rows [0..2), cols [1..3)
        let out = slice(&reg, &input, &[0, 1], &[2, 3], &[1, 1], &q).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 2.0, 5.0, 6.0]);
    }
}
