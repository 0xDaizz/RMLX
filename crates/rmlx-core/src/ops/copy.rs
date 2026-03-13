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

// ===================================================================
// Interleave heads (copy one head into columns of concatenated output)
// ===================================================================

kernel void interleave_heads_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_idx [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint row = id / head_dim;
    uint col = id % head_dim;
    if (row >= seq_len) return;
    uint src_idx = row * head_dim + col;
    uint dst_idx = row * (num_heads * head_dim) + head_idx * head_dim + col;
    dst[dst_idx] = src[src_idx];
}

kernel void interleave_heads_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_idx [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint row = id / head_dim;
    uint col = id % head_dim;
    if (row >= seq_len) return;
    uint src_idx = row * head_dim + col;
    uint dst_idx = row * (num_heads * head_dim) + head_idx * head_dim + col;
    dst[dst_idx] = src[src_idx];
}

kernel void interleave_heads_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_idx [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint row = id / head_dim;
    uint col = id % head_dim;
    if (row >= seq_len) return;
    uint src_idx = row * head_dim + col;
    uint dst_idx = row * (num_heads * head_dim) + head_idx * head_dim + col;
    dst[dst_idx] = src[src_idx];
}
"#;

/// Metal shader source for batched KV cache copy (all heads in one dispatch).
pub const KV_CACHE_COPY_BATCHED_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Copy K and V for all heads in a single dispatch.
// src layout:  [num_kv_heads, new_tokens, head_dim] (contiguous, head-major)
// dst layout:  [num_kv_heads, max_seq_len, head_dim] (slab with max_seq_len stride)
// Writes into dst at rows [offset .. offset+new_tokens] per head.
kernel void kv_cache_copy_batched_f16(
    device const half* src_k [[buffer(0)]],
    device const half* src_v [[buffer(1)]],
    device half* dst_k       [[buffer(2)]],
    device half* dst_v       [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& new_tokens   [[buffer(5)]],
    constant uint& head_dim     [[buffer(6)]],
    constant uint& max_seq_len  [[buffer(7)]],
    constant uint& offset       [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint tok  = tid.y;
    uint head = tid.z;
    if (elem >= head_dim || tok >= new_tokens || head >= num_kv_heads) return;

    uint src_idx = head * new_tokens * head_dim + tok * head_dim + elem;
    uint dst_idx = head * max_seq_len * head_dim + (offset + tok) * head_dim + elem;
    dst_k[dst_idx] = src_k[src_idx];
    dst_v[dst_idx] = src_v[src_idx];
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all copy / cast / fill kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("copy", COPY_SHADER_SOURCE)?;
    registry.register_jit_source("kv_cache_copy_batched", KV_CACHE_COPY_BATCHED_F16)?;
    Ok(())
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
fn vectorized_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("copy_v_f32"),
        DType::Float16 => Ok("copy_v_f16"),
        DType::Bfloat16 => Ok("copy_v_bf16"),
        DType::UInt32 => Ok("copy_v_u32"),
        _ => Err(KernelError::NotFound(format!(
            "vectorized copy not supported for {dtype}"
        ))),
    }
}

/// Scalar broadcast kernel name.
fn scalar_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("copy_s_f32"),
        DType::Float16 => Ok("copy_s_f16"),
        DType::Bfloat16 => Ok("copy_s_bf16"),
        DType::UInt32 => Ok("copy_s_u32"),
        _ => Err(KernelError::NotFound(format!(
            "scalar copy not supported for {dtype}"
        ))),
    }
}

/// Type suffix for strided kernel names.
fn type_suffix(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 | DType::UInt32 => Ok("f32"),
        DType::Float16 => Ok("f16"),
        DType::Bfloat16 => Ok("bf16"),
        _ => Err(KernelError::NotFound(format!(
            "strided copy not supported for {dtype}"
        ))),
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

/// Reject quantized and FP8 types for copy operations.
///
/// FP8 types are stored as uint8 and have no dedicated copy kernels.
/// Callers should dequantize FP8 to f16 first via `fp8::dequant_*`.
fn reject_quantized(dtype: DType) -> Result<(), KernelError> {
    if dtype.is_quantized() {
        return Err(KernelError::NotFound(
            "copy not supported for quantized types".to_string(),
        ));
    }
    if matches!(dtype, DType::Float8E4M3 | DType::Float8E5M2) {
        return Err(KernelError::NotFound(
            "copy not supported for FP8 types; dequantize to f16 first".to_string(),
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
fn u32_buffer(
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

/// Create a Metal buffer from a `&[u32]` slice.
fn u32_slice_buffer(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    data: &[u32],
) -> rmlx_metal::MtlBuffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(data.as_ptr() as *const _ as *mut std::ffi::c_void).unwrap(),
                std::mem::size_of_val(data) as u64 as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
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
    encoder: ComputePass<'_>,
    out: &Array,
) -> Result<rmlx_metal::MtlPipeline, KernelError> {
    let device = registry.device().raw();

    let ndim = src.ndim();
    let suffix = type_suffix(src.dtype())?;

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

    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
    match ndim {
        1 => {
            let kernel_name = format!("copy_g_nd1_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_pipeline(&pipeline);
            let stride_buf = u32_buffer(device, stride_data[0]);
            encoder.set_buffer(2, Some(&stride_buf), 0);
            Ok(pipeline)
        }
        2 => {
            let kernel_name = format!("copy_g_nd2_{suffix}");
            let pipeline = registry.get_pipeline(&kernel_name, src.dtype())?;
            encoder.set_pipeline(&pipeline);
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
            encoder.set_pipeline(&pipeline);
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
            encoder.set_pipeline(&pipeline);

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
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
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
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::uninit(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);

    let pipeline = if src.is_contiguous() {
        // Vectorized contiguous path
        let kname = vectorized_kernel_name(src.dtype())?;
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_pipeline(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
        let size_val = super::checked_u32(numel, "numel")?;
        encoder.set_val(2, &size_val);
        pipeline
    } else {
        // Strided path — encode_strided selects the best kernel variant
        encode_strided(registry, src, encoder, &out)?
    };

    // Compute grid size: for contiguous, divide by WPT; for strided, 1 per element.
    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        numel.div_ceil(w as usize)
    } else {
        numel
    };

    let grid_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), threads),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();
    super::commit_with_mode(&command_buffer, mode);

    Ok(out)
}

/// Copy array contents asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
pub fn copy_async(
    registry: &KernelRegistry,
    src: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<super::LaunchResult, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::uninit(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);

    let pipeline = if src.is_contiguous() {
        let kname = vectorized_kernel_name(src.dtype())?;
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_pipeline(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
        let size_val = super::checked_u32(numel, "numel")?;
        encoder.set_val(2, &size_val);
        pipeline
    } else {
        encode_strided(registry, src, encoder, &out)?
    };

    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        numel.div_ceil(w as usize)
    } else {
        numel
    };

    let grid_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), threads),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();

    let handle = super::commit_with_mode(&command_buffer, super::ExecMode::Async)
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
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    copy_cast_with_mode(registry, src, dst_dtype, queue, super::ExecMode::Sync)
}

/// Copy with type conversion and explicit execution mode.
pub fn copy_cast_with_mode(
    registry: &KernelRegistry,
    src: &Array,
    dst_dtype: DType,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
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

    let out = Array::uninit(registry.device().raw(), src.shape(), dst_dtype);

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
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
    super::commit_with_mode(&command_buffer, mode);

    Ok(out)
}

/// Copy with type conversion, encoded into an existing command buffer.
///
/// Like [`copy_cast`] but does not create or commit a command buffer.
/// The caller is responsible for committing the CB.
/// Source must be contiguous.
pub fn copy_cast_into_cb(
    registry: &KernelRegistry,
    src: &Array,
    dst_dtype: DType,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;
    reject_quantized(dst_dtype)?;

    if src.dtype() == dst_dtype {
        return copy_into_cb(registry, src, cb);
    }

    if !src.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "copy_cast_into_cb requires a contiguous source array".to_string(),
        ));
    }

    let kname = cast_kernel_name(src.dtype(), dst_dtype)?;
    let pipeline = registry.get_pipeline(kname, src.dtype())?;
    let numel = src.numel();

    let out = Array::uninit(registry.device().raw(), src.shape(), dst_dtype);

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
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
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    fill_with_mode(registry, shape, value, dtype, queue, super::ExecMode::Sync)
}

/// Fill with explicit execution mode.
pub fn fill_with_mode(
    registry: &KernelRegistry,
    shape: &[usize],
    value: &[u8],
    dtype: DType,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
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

    let kname = scalar_kernel_name(dtype)?;
    let pipeline = registry.get_pipeline(kname, dtype)?;

    let device = registry.device().raw();

    // Create a 1-element source buffer with the fill value.
    let src_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(value.as_ptr() as *const _ as *mut std::ffi::c_void)
                    .unwrap(),
                elem_size as u64 as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    };

    let out = Array::uninit(device, shape, dtype);

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(&src_buf), 0);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
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
    super::commit_with_mode(&command_buffer, mode);

    Ok(out)
}

/// Convenience: fill with a single `f32` value.
pub fn fill_f32(
    registry: &KernelRegistry,
    shape: &[usize],
    value: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    fill(registry, shape, &value.to_ne_bytes(), DType::Float32, queue)
}

// ---------------------------------------------------------------------------
// Into-CB variants (encode into existing command buffer, no commit/wait)
// ---------------------------------------------------------------------------

/// Encode a copy into an existing command buffer (no commit/wait).
pub fn copy_into_cb(
    registry: &KernelRegistry,
    src: &Array,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::uninit(registry.device().raw(), src.shape(), src.dtype());

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);

    let pipeline = if src.is_contiguous() {
        let kname = vectorized_kernel_name(src.dtype())?;
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_pipeline(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
        let size_val = super::checked_u32(numel, "numel")?;
        encoder.set_val(2, &size_val);
        pipeline
    } else {
        encode_strided(registry, src, encoder, &out)?
    };

    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        numel.div_ceil(w as usize)
    } else {
        numel
    };

    let grid_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), threads),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();

    Ok(out)
}

/// Encode head interleaving into an existing command buffer (no commit/wait).
///
/// Copies one attention head's output [seq_len, head_dim] into the correct
/// columns of the output [seq_len, num_heads * head_dim] for head concatenation.
#[allow(clippy::too_many_arguments)]
pub fn interleave_heads_into_cb(
    registry: &KernelRegistry,
    src: &Array,
    dst: &Array,
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    head_idx: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<(), KernelError> {
    let kernel_name = match src.dtype() {
        DType::Float32 => "interleave_heads_f32",
        DType::Float16 => "interleave_heads_f16",
        DType::Bfloat16 => "interleave_heads_bf16",
        other => {
            return Err(KernelError::NotFound(format!(
                "interleave_heads not supported for {other}"
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;

    let seq_len_u32 = seq_len as u32;
    let head_dim_u32 = head_dim as u32;
    let num_heads_u32 = num_heads as u32;
    let head_idx_u32 = head_idx as u32;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
    encoder.set_buffer(1, Some(dst.metal_buffer()), dst.offset());
    encoder.set_val(2, &seq_len_u32);
    encoder.set_val(3, &head_dim_u32);
    encoder.set_val(4, &num_heads_u32);
    encoder.set_val(5, &head_idx_u32);
    let total_threads = seq_len * head_dim;
    let grid_size = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), total_threads),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();

    Ok(())
}

// ---------------------------------------------------------------------------
// Batched KV cache copy (single dispatch for all heads)
// ---------------------------------------------------------------------------

/// Encode a batched KV cache copy into an existing command buffer.
///
/// Copies K and V for all heads in a single GPU dispatch, replacing per-head
/// copy loops (2 * num_kv_heads dispatches → 1 dispatch).
///
/// Source layout: `[num_kv_heads, new_tokens, head_dim]` (contiguous, head-major).
/// Destination layout: `[num_kv_heads, max_seq_len, head_dim]` (slab with stride).
/// Writes into destination rows `[offset .. offset + new_tokens]` per head.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_copy_batched_f16_into_cb(
    src_k: &Array,
    src_v: &Array,
    dst_k: &Array,
    dst_v: &Array,
    num_kv_heads: usize,
    new_tokens: usize,
    head_dim: usize,
    max_seq_len: usize,
    offset: usize,
    registry: &KernelRegistry,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<(), KernelError> {
    if new_tokens == 0 {
        return Ok(());
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batched_f16", DType::Float16)?;

    let num_kv_heads_u32 = super::checked_u32(num_kv_heads, "num_kv_heads")?;
    let new_tokens_u32 = super::checked_u32(new_tokens, "new_tokens")?;
    let head_dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let max_seq_len_u32 = super::checked_u32(max_seq_len, "max_seq_len")?;
    let offset_u32 = super::checked_u32(offset, "offset")?;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(src_k.metal_buffer()), src_k.offset());
    enc.set_buffer(1, Some(src_v.metal_buffer()), src_v.offset());
    enc.set_buffer(2, Some(dst_k.metal_buffer()), dst_k.offset());
    enc.set_buffer(3, Some(dst_v.metal_buffer()), dst_v.offset());
    enc.set_val(4, &num_kv_heads_u32);
    enc.set_val(5, &new_tokens_u32);
    enc.set_val(6, &head_dim_u32);
    enc.set_val(7, &max_seq_len_u32);
    enc.set_val(8, &offset_u32);
    let grid = MTLSize {
        width: head_dim,
        height: new_tokens,
        depth: num_kv_heads,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, head_dim),
        height: std::cmp::min(4, new_tokens),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end();

    Ok(())
}

// ---------------------------------------------------------------------------
// _encode variants — accept &ComputeCommandEncoderRef instead of &CommandBufferRef
// ---------------------------------------------------------------------------

/// Encode a copy into an existing compute command encoder (no encoder create/end).
pub fn copy_encode(
    registry: &KernelRegistry,
    src: &Array,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;

    let numel = src.numel();
    let out = Array::uninit(registry.device().raw(), src.shape(), src.dtype());

    let pipeline = if src.is_contiguous() {
        let kname = vectorized_kernel_name(src.dtype())?;
        let pipeline = registry.get_pipeline(kname, src.dtype())?;
        encoder.set_pipeline(&pipeline);
        encoder.set_buffer(0, Some(src.metal_buffer()), src.offset());
        encoder.set_buffer(1, Some(out.metal_buffer()), out.offset());
        let size_val = super::checked_u32(numel, "numel")?;
        encoder.set_val(2, &size_val);
        pipeline
    } else {
        encode_strided(registry, src, encoder, &out)?
    };

    let threads = if src.is_contiguous() {
        let w = wpt(src.dtype());
        numel.div_ceil(w as usize)
    } else {
        numel
    };

    let grid_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), threads),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_size, threadgroup_size);

    Ok(out)
}

/// Encode batched KV cache copy into an existing compute command encoder (no encoder create/end).
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_copy_batched_f16_encode(
    src_k: &Array,
    src_v: &Array,
    dst_k: &Array,
    dst_v: &Array,
    num_kv_heads: usize,
    new_tokens: usize,
    head_dim: usize,
    max_seq_len: usize,
    offset: usize,
    registry: &KernelRegistry,
    encoder: ComputePass<'_>,
) -> Result<(), KernelError> {
    if new_tokens == 0 {
        return Ok(());
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batched_f16", DType::Float16)?;

    let num_kv_heads_u32 = super::checked_u32(num_kv_heads, "num_kv_heads")?;
    let new_tokens_u32 = super::checked_u32(new_tokens, "new_tokens")?;
    let head_dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let max_seq_len_u32 = super::checked_u32(max_seq_len, "max_seq_len")?;
    let offset_u32 = super::checked_u32(offset, "offset")?;

    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(src_k.metal_buffer()), src_k.offset());
    encoder.set_buffer(1, Some(src_v.metal_buffer()), src_v.offset());
    encoder.set_buffer(2, Some(dst_k.metal_buffer()), dst_k.offset());
    encoder.set_buffer(3, Some(dst_v.metal_buffer()), dst_v.offset());
    encoder.set_val(4, &num_kv_heads_u32);
    encoder.set_val(5, &new_tokens_u32);
    encoder.set_val(6, &head_dim_u32);
    encoder.set_val(7, &max_seq_len_u32);
    encoder.set_val(8, &offset_u32);
    let grid = MTLSize {
        width: head_dim,
        height: new_tokens,
        depth: num_kv_heads,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, head_dim),
        height: std::cmp::min(4, new_tokens),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);

    Ok(())
}

// ---------------------------------------------------------------------------
// Metal 4 blit-integrated paths (ComputePass4)
// ---------------------------------------------------------------------------
//
// Metal 4 unifies compute and blit operations in a single encoder. These
// variants use `ComputePass4::copy_buffer()` for contiguous same-type copies
// and `ComputePass4::fill_buffer()` for zero-fills, eliminating the need to
// switch between compute and blit encoders.
//
// Note: Metal 4 uses argument tables for compute buffer bindings (rather than
// the Metal 3 `setBuffer_offset_atIndex` API), so strided copies that require
// compute kernels are not handled here. Use the Metal 3 `copy_encode` for
// strided data.

#[cfg(feature = "metal4")]
use rmlx_metal::metal4::compute::ComputePass4;

/// Encode a contiguous same-type copy as a hardware blit in a Metal 4 compute pass.
///
/// Uses `ComputePass4::copy_buffer()` which is a DMA operation — no compute
/// pipeline setup, no shader invocation. This is the optimal path for
/// contiguous same-type copies when already in a Metal 4 encoder.
///
/// Returns `Err` if the source is not contiguous (use `copy_encode` for strided data).
///
/// The caller is responsible for encoder lifecycle (`end()` / command buffer commit).
#[cfg(feature = "metal4")]
pub fn copy_blit_m4(
    registry: &KernelRegistry,
    src: &Array,
    encoder: ComputePass4<'_>,
) -> Result<Array, KernelError> {
    reject_quantized(src.dtype())?;

    if !src.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "copy_blit_m4 requires a contiguous source array; use copy_encode for strided data"
                .to_string(),
        ));
    }

    let numel = src.numel();
    let out = Array::uninit(registry.device().raw(), src.shape(), src.dtype());

    let byte_size = numel * src.dtype().size_of();
    encoder.copy_buffer(
        src.metal_buffer(),
        src.offset(),
        out.metal_buffer(),
        out.offset(),
        byte_size,
    );

    Ok(out)
}

/// Encode a contiguous buffer-to-buffer copy as a hardware blit in a Metal 4
/// compute pass, writing into a pre-allocated destination.
///
/// Unlike [`copy_blit_m4`] which allocates a new output, this writes into
/// `dst` at `dst.offset()`. Both `src` and `dst` must be contiguous and have
/// the same byte size.
#[cfg(feature = "metal4")]
pub fn copy_blit_into_m4(
    src: &Array,
    dst: &Array,
    encoder: ComputePass4<'_>,
) -> Result<(), KernelError> {
    reject_quantized(src.dtype())?;

    if !src.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "copy_blit_into_m4 requires a contiguous source array".to_string(),
        ));
    }

    let byte_size = src.numel() * src.dtype().size_of();
    encoder.copy_buffer(
        src.metal_buffer(),
        src.offset(),
        dst.metal_buffer(),
        dst.offset(),
        byte_size,
    );

    Ok(())
}

/// Fill a buffer region with zeros using the Metal 4 blit path.
///
/// Uses `ComputePass4::fill_buffer()` which is a hardware DMA zero-fill —
/// no compute pipeline, no shader invocation. Significantly faster than
/// dispatching the `copy_s_*` kernel for zero initialization.
///
/// Returns `Err` if `value` is non-zero (use the compute `fill_*` functions
/// for non-zero fills).
#[cfg(feature = "metal4")]
pub fn fill_zero_blit_m4(
    shape: &[usize],
    dtype: DType,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    encoder: ComputePass4<'_>,
) -> Result<Array, KernelError> {
    reject_quantized(dtype)?;

    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(Array::zeros(device, shape, dtype));
    }

    let out = Array::uninit(device, shape, dtype);
    let byte_size = numel * dtype.size_of();

    encoder.fill_buffer(
        out.metal_buffer(),
        out.offset()..out.offset() + byte_size,
        0,
    );

    Ok(out)
}

/// Fill a buffer with a constant byte pattern using the Metal 4 blit path.
///
/// Uses `ComputePass4::fill_buffer()` to set every byte in the output to
/// `byte_value`. This is suitable for memset-style fills (e.g., zero-fill
/// with `byte_value = 0`, or 0xFF-fill for sentinel values).
///
/// Note: this fills at byte granularity, not element granularity. For
/// element-wise fills with arbitrary values, use the compute `fill_*` functions.
#[cfg(feature = "metal4")]
pub fn fill_bytes_blit_m4(
    shape: &[usize],
    dtype: DType,
    byte_value: u8,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    encoder: ComputePass4<'_>,
) -> Result<Array, KernelError> {
    reject_quantized(dtype)?;

    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(Array::zeros(device, shape, dtype));
    }

    let out = Array::uninit(device, shape, dtype);
    let byte_size = numel * dtype.size_of();

    encoder.fill_buffer(
        out.metal_buffer(),
        out.offset()..out.offset() + byte_size,
        byte_value,
    );

    Ok(out)
}

/// Encode batched KV cache copy as hardware blits in a Metal 4 compute pass.
///
/// Uses per-head `copy_buffer()` blit operations instead of dispatching a
/// compute kernel. Since the source is contiguous
/// `[num_kv_heads, new_tokens, head_dim]` and the destination is a slab
/// `[num_kv_heads, max_seq_len, head_dim]`, each head's data is a contiguous
/// region that can be copied via DMA.
///
/// This replaces `2 * num_kv_heads` potential encoder switches (Metal 3 blit)
/// or one compute kernel dispatch with inline blit operations in the same
/// compute pass.
#[cfg(feature = "metal4")]
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_copy_batched_f16_blit_m4(
    src_k: &Array,
    src_v: &Array,
    dst_k: &Array,
    dst_v: &Array,
    num_kv_heads: usize,
    new_tokens: usize,
    head_dim: usize,
    max_seq_len: usize,
    offset: usize,
    encoder: ComputePass4<'_>,
) -> Result<(), KernelError> {
    if new_tokens == 0 {
        return Ok(());
    }

    let bytes_per_elem = DType::Float16.size_of();
    let head_copy_bytes = new_tokens * head_dim * bytes_per_elem;
    let src_head_stride = new_tokens * head_dim * bytes_per_elem;
    let dst_head_stride = max_seq_len * head_dim * bytes_per_elem;
    let dst_row_offset = offset * head_dim * bytes_per_elem;

    for h in 0..num_kv_heads {
        let src_k_off = src_k.offset() + h * src_head_stride;
        let dst_k_off = dst_k.offset() + h * dst_head_stride + dst_row_offset;
        encoder.copy_buffer(
            src_k.metal_buffer(),
            src_k_off,
            dst_k.metal_buffer(),
            dst_k_off,
            head_copy_bytes,
        );

        let src_v_off = src_v.offset() + h * src_head_stride;
        let dst_v_off = dst_v.offset() + h * dst_head_stride + dst_row_offset;
        encoder.copy_buffer(
            src_v.metal_buffer(),
            src_v_off,
            dst_v.metal_buffer(),
            dst_v_off,
            head_copy_bytes,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // GPU tests require Metal device — run on macOS with `cargo test -p rmlx-core`.
    // Non-contiguous copy tests verify that the strided kernel correctly
    // handles transposed and sliced array layouts.

    use super::*;

    #[test]
    fn test_vectorized_kernel_name_unsupported_dtype_returns_error() {
        let result = vectorized_kernel_name(DType::Float8E4M3);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            KernelError::NotFound(msg) => {
                assert!(
                    msg.contains("vectorized copy not supported"),
                    "unexpected message: {msg}"
                );
            }
            _ => panic!("expected NotFound, got {err:?}"),
        }
    }

    #[test]
    fn test_scalar_kernel_name_unsupported_dtype_returns_error() {
        let result = scalar_kernel_name(DType::Q4_0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            KernelError::NotFound(msg) => {
                assert!(
                    msg.contains("scalar copy not supported"),
                    "unexpected message: {msg}"
                );
            }
            _ => panic!("expected NotFound, got {err:?}"),
        }
    }

    #[test]
    fn test_type_suffix_unsupported_dtype_returns_error() {
        let result = type_suffix(DType::Float8E5M2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            KernelError::NotFound(msg) => {
                assert!(
                    msg.contains("strided copy not supported"),
                    "unexpected message: {msg}"
                );
            }
            _ => panic!("expected NotFound, got {err:?}"),
        }
    }

    #[test]
    fn test_supported_dtypes_return_ok() {
        // vectorized
        assert!(vectorized_kernel_name(DType::Float32).is_ok());
        assert!(vectorized_kernel_name(DType::Float16).is_ok());
        assert!(vectorized_kernel_name(DType::Bfloat16).is_ok());
        assert!(vectorized_kernel_name(DType::UInt32).is_ok());

        // scalar
        assert!(scalar_kernel_name(DType::Float32).is_ok());
        assert!(scalar_kernel_name(DType::Float16).is_ok());
        assert!(scalar_kernel_name(DType::Bfloat16).is_ok());
        assert!(scalar_kernel_name(DType::UInt32).is_ok());

        // type suffix
        assert!(type_suffix(DType::Float32).is_ok());
        assert!(type_suffix(DType::Float16).is_ok());
        assert!(type_suffix(DType::Bfloat16).is_ok());
        assert!(type_suffix(DType::UInt32).is_ok());
    }
}
