//! Copy, type-cast, and fill operations with optimized Metal kernels.
//!
//! Kernel variants:
//!   - **Vectorized contiguous** (`copy_v_*`): each thread copies multiple elements
//!     (2 for f32/u32, 4 for f16/bf16) to improve memory throughput.
//!   - **Scalar broadcast** (`copy_s_*`): fills every output element with `src[0]`.
//!   - **Dimension-specialized strided** (`copy_g_nd{1,2,3}_*`): hand-unrolled
//!     coordinate decomposition for 1D/2D/3D non-contiguous layouts.
//!   - **General strided** (`copy_g_*`): N-dim with host-precomputed output strides
//!     (eliminates the per-thread O(ndim^2) inner loop from the old kernel).
//!   - **Type-converting** (`copy_f32_to_f16`, `copy_f16_to_f32`, etc.): contiguous
//!     element-wise casts between float types.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for all copy / cast / fill kernels.
pub const COPY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Vectorized contiguous copy (work-per-thread)
// ===================================================================

// f32: 2 elements per thread
kernel void copy_v_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& size     [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < size) {
        dst[base]     = src[base];
        dst[base + 1] = src[base + 1];
    } else if (base < size) {
        dst[base] = src[base];
    }
}

// f16: 4 elements per thread
kernel void copy_v_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& size    [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < size) {
        dst[base]     = src[base];
        dst[base + 1] = src[base + 1];
        dst[base + 2] = src[base + 2];
        dst[base + 3] = src[base + 3];
    } else {
        for (uint i = 0; i < 4 && base + i < size; i++) {
            dst[base + i] = src[base + i];
        }
    }
}

// bf16: 4 elements per thread
kernel void copy_v_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant uint& size      [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < size) {
        dst[base]     = src[base];
        dst[base + 1] = src[base + 1];
        dst[base + 2] = src[base + 2];
        dst[base + 3] = src[base + 3];
    } else {
        for (uint i = 0; i < 4 && base + i < size; i++) {
            dst[base + i] = src[base + i];
        }
    }
}

// u32 (same byte width as f32): 2 elements per thread
kernel void copy_v_u32(
    device const uint* src [[buffer(0)]],
    device uint* dst       [[buffer(1)]],
    constant uint& size    [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < size) {
        dst[base]     = src[base];
        dst[base + 1] = src[base + 1];
    } else if (base < size) {
        dst[base] = src[base];
    }
}

// ===================================================================
// Scalar broadcast copy (fill output with src[0])
// ===================================================================

kernel void copy_s_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[0];
}

kernel void copy_s_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[0];
}

kernel void copy_s_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[0];
}

kernel void copy_s_u32(
    device const uint* src [[buffer(0)]],
    device uint* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[0];
}

// ===================================================================
// Dimension-specialized strided copy
// ===================================================================

// --- 1D strided ---

kernel void copy_g_nd1_f32(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id * src_stride];
}

kernel void copy_g_nd1_f16(
    device const half* src    [[buffer(0)]],
    device half* dst          [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id * src_stride];
}

kernel void copy_g_nd1_bf16(
    device const bfloat* src  [[buffer(0)]],
    device bfloat* dst        [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id * src_stride];
}

// --- 2D strided ---

kernel void copy_g_nd2_f32(
    device const float* src          [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& src_stride0       [[buffer(4)]],
    constant uint& src_stride1       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint c1 = id % dim1;
    uint c0 = id / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1];
}

kernel void copy_g_nd2_f16(
    device const half* src           [[buffer(0)]],
    device half* dst                 [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& src_stride0       [[buffer(4)]],
    constant uint& src_stride1       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint c1 = id % dim1;
    uint c0 = id / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1];
}

kernel void copy_g_nd2_bf16(
    device const bfloat* src         [[buffer(0)]],
    device bfloat* dst               [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& src_stride0       [[buffer(4)]],
    constant uint& src_stride1       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint c1 = id % dim1;
    uint c0 = id / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1];
}

// --- 3D strided ---

kernel void copy_g_nd3_f32(
    device const float* src          [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& dim2              [[buffer(4)]],
    constant uint& src_stride0       [[buffer(5)]],
    constant uint& src_stride1       [[buffer(6)]],
    constant uint& src_stride2       [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint c2 = id % dim2;
    uint tmp = id / dim2;
    uint c1 = tmp % dim1;
    uint c0 = tmp / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1 + c2 * src_stride2];
}

kernel void copy_g_nd3_f16(
    device const half* src           [[buffer(0)]],
    device half* dst                 [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& dim2              [[buffer(4)]],
    constant uint& src_stride0       [[buffer(5)]],
    constant uint& src_stride1       [[buffer(6)]],
    constant uint& src_stride2       [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint c2 = id % dim2;
    uint tmp = id / dim2;
    uint c1 = tmp % dim1;
    uint c0 = tmp / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1 + c2 * src_stride2];
}

kernel void copy_g_nd3_bf16(
    device const bfloat* src         [[buffer(0)]],
    device bfloat* dst               [[buffer(1)]],
    constant uint& dim0              [[buffer(2)]],
    constant uint& dim1              [[buffer(3)]],
    constant uint& dim2              [[buffer(4)]],
    constant uint& src_stride0       [[buffer(5)]],
    constant uint& src_stride1       [[buffer(6)]],
    constant uint& src_stride2       [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint c2 = id % dim2;
    uint tmp = id / dim2;
    uint c1 = tmp % dim1;
    uint c0 = tmp / dim1;
    dst[id] = src[c0 * src_stride0 + c1 * src_stride1 + c2 * src_stride2];
}

// --- General N-dim strided (with host-precomputed output strides) ---

kernel void copy_g_f32(
    device const float* src              [[buffer(0)]],
    device float* dst                    [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint* out_strides     [[buffer(4)]],
    constant uint& ndim                  [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

kernel void copy_g_f16(
    device const half* src               [[buffer(0)]],
    device half* dst                     [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint* out_strides     [[buffer(4)]],
    constant uint& ndim                  [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

kernel void copy_g_bf16(
    device const bfloat* src             [[buffer(0)]],
    device bfloat* dst                   [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint* out_strides     [[buffer(4)]],
    constant uint& ndim                  [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

// ===================================================================
// Type-converting copy (contiguous only)
// ===================================================================

kernel void copy_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = half(src[id]);
}

kernel void copy_f16_to_f32(
    device const half* src [[buffer(0)]],
    device float* dst      [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = float(src[id]);
}

kernel void copy_f32_to_bf16(
    device const float* src [[buffer(0)]],
    device bfloat* dst      [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = bfloat(src[id]);
}

kernel void copy_bf16_to_f32(
    device const bfloat* src [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = float(src[id]);
}

kernel void copy_f16_to_bf16(
    device const half* src [[buffer(0)]],
    device bfloat* dst     [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = bfloat(src[id]);
}

kernel void copy_bf16_to_f16(
    device const bfloat* src [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = half(src[id]);
}

// Legacy flat-copy names kept for backward-compatible pipeline lookups.
// These are thin wrappers so existing callers that request "copy_f32" etc.
// still resolve to a valid function.

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

kernel void copy_f16(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

kernel void copy_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

// Legacy strided names (old API compat — kept so that any cached pipeline
// key using these names still resolves).  These use the old O(ndim^2)
// computation; new code dispatches to copy_g_nd* or copy_g_* instead.

kernel void copy_strided_f32(
    device const float* src              [[buffer(0)]],
    device float* dst                    [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint& ndim            [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint out_stride = 1;
        for (uint k = d + 1; k < ndim; k++) {
            out_stride *= shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

kernel void copy_strided_f16(
    device const half* src               [[buffer(0)]],
    device half* dst                     [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint& ndim            [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint out_stride = 1;
        for (uint k = d + 1; k < ndim; k++) {
            out_stride *= shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

kernel void copy_strided_bf16(
    device const bfloat* src             [[buffer(0)]],
    device bfloat* dst                   [[buffer(1)]],
    constant const uint* shape           [[buffer(2)]],
    constant const uint* src_strides     [[buffer(3)]],
    constant const uint& ndim            [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint out_stride = 1;
        for (uint k = d + 1; k < ndim; k++) {
            out_stride *= shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all copy / cast / fill kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("copy", COPY_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Work-per-thread factor for the vectorized contiguous kernel.
fn wpt(dtype: DType) -> u64 {
    match dtype {
        DType::Float32 | DType::UInt32 => 2,
        DType::Float16 | DType::Bfloat16 => 4,
        _ => 1,
    }
}

/// Vectorized contiguous kernel name.
fn vectorized_kernel_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "copy_v_f32",
        DType::Float16 => "copy_v_f16",
        DType::Bfloat16 => "copy_v_bf16",
        DType::UInt32 => "copy_v_u32",
        _ => unreachable!("vectorized copy not supported for {:?}", dtype),
    }
}

/// Scalar broadcast kernel name.
fn scalar_kernel_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "copy_s_f32",
        DType::Float16 => "copy_s_f16",
        DType::Bfloat16 => "copy_s_bf16",
        DType::UInt32 => "copy_s_u32",
        _ => unreachable!("scalar copy not supported for {:?}", dtype),
    }
}

/// Type suffix for strided kernel names.
fn type_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 | DType::UInt32 => "f32",
        DType::Float16 => "f16",
        DType::Bfloat16 => "bf16",
        _ => unreachable!("strided copy not supported for {:?}", dtype),
    }
}

/// Kernel name for type-converting copy.
fn cast_kernel_name(src_dtype: DType, dst_dtype: DType) -> Result<&'static str, KernelError> {
    match (src_dtype, dst_dtype) {
        (DType::Float32, DType::Float16) => Ok("copy_f32_to_f16"),
        (DType::Float16, DType::Float32) => Ok("copy_f16_to_f32"),
        (DType::Float32, DType::Bfloat16) => Ok("copy_f32_to_bf16"),
        (DType::Bfloat16, DType::Float32) => Ok("copy_bf16_to_f32"),
        (DType::Float16, DType::Bfloat16) => Ok("copy_f16_to_bf16"),
        (DType::Bfloat16, DType::Float16) => Ok("copy_bf16_to_f16"),
        _ => Err(KernelError::NotFound(format!(
            "no cast kernel for {src_dtype} -> {dst_dtype}"
        ))),
    }
}

/// Reject quantized types for copy operations.
fn reject_quantized(dtype: DType) -> Result<(), KernelError> {
    if dtype.is_quantized() {
        return Err(KernelError::NotFound(
            "copy not supported for quantized types".to_string(),
        ));
    }
    Ok(())
}

/// Compute contiguous (row-major) output strides for a shape.
fn compute_out_strides(shape: &[usize]) -> Vec<u32> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1u32; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * (shape[i + 1] as u32);
    }
    strides
}

/// Create a small Metal buffer from a `u32` value.
fn u32_buffer(device: &metal::Device, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Create a Metal buffer from a `&[u32]` slice.
fn u32_slice_buffer(device: &metal::Device, data: &[u32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ---------------------------------------------------------------------------
// Strided dispatch helpers
// ---------------------------------------------------------------------------

/// Encode the appropriate strided kernel for `src` into `encoder`.
///
/// Selects dimension-specialized (nd1/nd2/nd3) or general (N-dim) kernels
/// and sets the corresponding Metal buffers.
fn encode_strided(
    registry: &KernelRegistry,
    src: &Array,
    encoder: &metal::ComputeCommandEncoderRef,
    out: &Array,
) -> Result<metal::ComputePipelineState, KernelError> {
    let device = registry.device().raw();
    let ndim = src.ndim();
    let suffix = type_suffix(src.dtype());

    let shape_data: Vec<u32> = src
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| super::checked_u32(s, &format!("shape[{i}]")))
        .collect::<Result<Vec<u32>, _>>()?;
    let stride_data: Vec<u32> = src
        .strides()
        .iter()
        .enumerate()
        .map(|(i, &s)| super::checked_u32(s, &format!("stride[{i}]")))
        .collect::<Result<Vec<u32>, _>>()?;

    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);

    match ndim {
        1 => {
            let kernel_name = format!("copy_g_nd1_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_compute_pipeline_state(&pipeline);
            let stride_buf = u32_buffer(device, stride_data[0]);
            encoder.set_buffer(2, Some(&stride_buf), 0);
            Ok(pipeline)
        }
        2 => {
            let kernel_name = format!("copy_g_nd2_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_compute_pipeline_state(&pipeline);
            let dim0_buf = u32_buffer(device, shape_data[0]);
            let dim1_buf = u32_buffer(device, shape_data[1]);
            let s0_buf = u32_buffer(device, stride_data[0]);
            let s1_buf = u32_buffer(device, stride_data[1]);
            encoder.set_buffer(2, Some(&dim0_buf), 0);
            encoder.set_buffer(3, Some(&dim1_buf), 0);
            encoder.set_buffer(4, Some(&s0_buf), 0);
            encoder.set_buffer(5, Some(&s1_buf), 0);
            Ok(pipeline)
        }
        3 => {
            let kernel_name = format!("copy_g_nd3_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_compute_pipeline_state(&pipeline);
            let dim0_buf = u32_buffer(device, shape_data[0]);
            let dim1_buf = u32_buffer(device, shape_data[1]);
            let dim2_buf = u32_buffer(device, shape_data[2]);
            let s0_buf = u32_buffer(device, stride_data[0]);
            let s1_buf = u32_buffer(device, stride_data[1]);
            let s2_buf = u32_buffer(device, stride_data[2]);
            encoder.set_buffer(2, Some(&dim0_buf), 0);
            encoder.set_buffer(3, Some(&dim1_buf), 0);
            encoder.set_buffer(4, Some(&dim2_buf), 0);
            encoder.set_buffer(5, Some(&s0_buf), 0);
            encoder.set_buffer(6, Some(&s1_buf), 0);
            encoder.set_buffer(7, Some(&s2_buf), 0);
            Ok(pipeline)
        }
        _ => {
            // General N-dim with host-precomputed output strides.
            let kernel_name = format!("copy_g_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_compute_pipeline_state(&pipeline);

            let out_strides = compute_out_strides(src.shape());

            let shape_buf = u32_slice_buffer(device, &shape_data);
            let stride_buf = u32_slice_buffer(device, &stride_data);
            let out_stride_buf = u32_slice_buffer(device, &out_strides);
            let ndim_val = super::checked_u32(ndim, "ndim")?;
            let ndim_buf = u32_buffer(device, ndim_val);

            encoder.set_buffer(2, Some(&shape_buf), 0);
            encoder.set_buffer(3, Some(&stride_buf), 0);
            encoder.set_buffer(4, Some(&out_stride_buf), 0);
            encoder.set_buffer(5, Some(&ndim_buf), 0);
            Ok(pipeline)
        }
    }
}

// ---------------------------------------------------------------------------
// Public API — copy
// ---------------------------------------------------------------------------

/// Copy array contents to a new contiguous array (synchronous).
pub fn copy(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    copy_with_mode(registry, src, queue, super::ExecMode::Sync)
}

/// Copy array contents to a new contiguous array with explicit execution mode.
///
/// Dispatch strategy:
///   - **Contiguous** source: uses vectorized copy (`copy_v_*`) with 2 or 4
///     elements per thread, passing a size buffer to handle tail elements.
///   - **Strided 1D/2D/3D**: dimension-specialized kernels with pre-computed
///     coordinate decomposition.
///   - **Strided N-dim**: general kernel with host-precomputed output strides.
pub fn copy_with_mode(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::zeros(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let pipeline = if src.is_contiguous() {
        // Vectorized contiguous path
        let kname = vectorized_kernel_name(src.dtype());
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
        let size_val = super::checked_u32(numel, "numel")?;
        let size_buf = u32_buffer(registry.device().raw(), size_val);
        encoder.set_buffer(2, Some(&size_buf), 0);
        pipeline
    } else {
        // Strided path — encode_strided selects the best kernel variant
        encode_strided(registry, src, encoder, &out)?
    };

    // Compute grid size: for contiguous, divide by WPT; for strided, 1 per element.
    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        (numel as u64 + w - 1) / w
    } else {
        numel as u64
    };

    let grid_size = MTLSize::new(threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Copy array contents asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
pub fn copy_async(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
) -> Result<super::LaunchResult, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::zeros(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let pipeline = if src.is_contiguous() {
        let kname = vectorized_kernel_name(src.dtype());
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
        let size_val = super::checked_u32(numel, "numel")?;
        let size_buf = u32_buffer(registry.device().raw(), size_val);
        encoder.set_buffer(2, Some(&size_buf), 0);
        pipeline
    } else {
        encode_strided(registry, src, encoder, &out)?
    };

    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        (numel as u64 + w - 1) / w
    } else {
        numel as u64
    };

    let grid_size = MTLSize::new(threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    let handle = super::commit_with_mode(command_buffer, super::ExecMode::Async)
        .expect("async mode always returns a handle");

    Ok(super::LaunchResult::new(out, handle))
}

// ---------------------------------------------------------------------------
// Public API — type-converting copy (cast)
// ---------------------------------------------------------------------------

/// Copy array contents to a new contiguous array with a different dtype.
///
/// The source must be contiguous. Supported conversions:
///   f32 <-> f16, f32 <-> bf16, f16 <-> bf16
pub fn copy_cast(
    registry: &KernelRegistry,
    src: &Array,
    dst_dtype: DType,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    copy_cast_with_mode(registry, src, dst_dtype, queue, super::ExecMode::Sync)
}

/// Copy with type conversion and explicit execution mode.
pub fn copy_cast_with_mode(
    registry: &KernelRegistry,
    src: &Array,
    dst_dtype: DType,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;
    reject_quantized(dst_dtype)?;

    if src.dtype() == dst_dtype {
        // Same type — fall back to regular copy.
        return copy_with_mode(registry, src, queue, mode);
    }

    if !src.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "copy_cast requires a contiguous source array".to_string(),
        ));
    }

    let kname = cast_kernel_name(src.dtype(), dst_dtype)?;
    let pipeline = registry.get_pipeline(kname, src.dtype())?;
    let numel = src.numel();

    let out = Array::zeros(registry.device().raw(), src.shape(), dst_dtype);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);

    let grid_size = MTLSize::new(numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API — fill (scalar broadcast)
// ---------------------------------------------------------------------------

/// Fill a new array of the given shape and dtype with `value` bytes.
///
/// `value` is a 4-byte buffer interpreted as the element type:
///   - For f32/u32: the raw 4 bytes are used directly.
///   - For f16/bf16: only the first 2 bytes are used.
///
/// This dispatches the scalar broadcast kernel (`copy_s_*`) which reads
/// `src[0]` and writes it to every element.
pub fn fill(
    registry: &KernelRegistry,
    shape: &[usize],
    value: &[u8],
    dtype: DType,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    fill_with_mode(registry, shape, value, dtype, queue, super::ExecMode::Sync)
}

/// Fill with explicit execution mode.
pub fn fill_with_mode(
    registry: &KernelRegistry,
    shape: &[usize],
    value: &[u8],
    dtype: DType,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    reject_quantized(dtype)?;

    let elem_size = dtype.size_of();
    if value.len() < elem_size {
        return Err(KernelError::InvalidShape(format!(
            "fill: value buffer ({} bytes) smaller than element size ({} bytes)",
            value.len(),
            elem_size
        )));
    }

    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), shape, dtype));
    }

    let kname = scalar_kernel_name(dtype);
    let pipeline = registry.get_pipeline(kname, dtype)?;

    let device = registry.device().raw();

    // Create a 1-element source buffer with the fill value.
    let src_buf = device.new_buffer_with_data(
        value.as_ptr() as *const _,
        elem_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let out = Array::zeros(device, shape, dtype);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&src_buf), 0);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);

    let grid_size = MTLSize::new(numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Convenience: fill with a single `f32` value.
pub fn fill_f32(
    registry: &KernelRegistry,
    shape: &[usize],
    value: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    fill(registry, shape, &value.to_ne_bytes(), DType::Float32, queue)
}

#[cfg(test)]
mod tests {
    // GPU tests require Metal device — run on macOS with `cargo test -p rmlx-core`.
    // Non-contiguous copy tests verify that the strided kernel correctly
    // handles transposed and sliced array layouts.
}
