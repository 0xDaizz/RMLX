//! Element-wise binary operations with NumPy-style broadcasting.
//!
//! Supports arithmetic ops (add, mul, sub, div) and comparison ops
//! (eq, ne, lt, le, gt, ge). Comparison ops output UInt32 (0 or 1).
//!
//! Broadcasting dispatch strategy:
//! - Both scalars (numel=1) -> `{op}_ss_{dtype}` kernel
//! - One scalar, one vector  -> `{op}_sv_{dtype}` or `{op}_vs_{dtype}`
//! - Same shape (fast path)  -> flat element-wise `{op}_{dtype}` kernel
//! - Otherwise               -> general N-dim broadcast `{op}_g_{dtype}` with stride buffers

use crate::array::{broadcast_shape, Array};
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for all binary kernels.
///
/// Includes flat (element-wise), scalar-scalar, scalar-vector, vector-scalar,
/// and general broadcast variants for arithmetic and comparison ops.
pub const BINARY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// =====================================================================
// Macros for generating kernel variants
// =====================================================================

// --- Flat element-wise (same-shape fast path) ---
#define BINARY_FLAT(NAME, TYPE, OP)                                  \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device TYPE* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = a[id] OP b[id];                                       \
}

// --- Scalar-scalar ---
#define BINARY_SS(NAME, TYPE, OP)                                    \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device TYPE* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[0] = a[0] OP b[0];                                          \
}

// --- Scalar-vector (a is scalar, b is vector) ---
#define BINARY_SV(NAME, TYPE, OP)                                    \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device TYPE* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = a[0] OP b[id];                                        \
}

// --- Vector-scalar (a is vector, b is scalar) ---
#define BINARY_VS(NAME, TYPE, OP)                                    \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device TYPE* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = a[id] OP b[0];                                        \
}

// --- General N-dim broadcast with strides ---
#define BINARY_GENERAL(NAME, TYPE, OP)                               \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device TYPE* out [[buffer(2)]],                                  \
    constant const uint* out_shape [[buffer(3)]],                    \
    constant const uint* a_strides [[buffer(4)]],                    \
    constant const uint* b_strides [[buffer(5)]],                    \
    constant uint& ndim [[buffer(6)]],                               \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    uint a_offset = 0;                                               \
    uint b_offset = 0;                                               \
    uint remaining = id;                                             \
    for (uint d = 0; d < ndim; d++) {                                \
        uint out_stride = 1;                                         \
        for (uint k = d + 1; k < ndim; k++) out_stride *= out_shape[k]; \
        uint coord = remaining / out_stride;                         \
        remaining %= out_stride;                                     \
        a_offset += coord * a_strides[d];                            \
        b_offset += coord * b_strides[d];                            \
    }                                                                \
    out[id] = a[a_offset] OP b[b_offset];                           \
}

// --- Comparison: flat ---
#define CMP_FLAT(NAME, TYPE, OP)                                     \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device uint* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = (a[id] OP b[id]) ? 1u : 0u;                           \
}

// --- Comparison: scalar-scalar ---
#define CMP_SS(NAME, TYPE, OP)                                       \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device uint* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[0] = (a[0] OP b[0]) ? 1u : 0u;                              \
}

// --- Comparison: scalar-vector ---
#define CMP_SV(NAME, TYPE, OP)                                       \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device uint* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = (a[0] OP b[id]) ? 1u : 0u;                            \
}

// --- Comparison: vector-scalar ---
#define CMP_VS(NAME, TYPE, OP)                                       \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device uint* out [[buffer(2)]],                                  \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    out[id] = (a[id] OP b[0]) ? 1u : 0u;                            \
}

// --- Comparison: general N-dim broadcast ---
#define CMP_GENERAL(NAME, TYPE, OP)                                  \
kernel void NAME(                                                    \
    device const TYPE* a [[buffer(0)]],                              \
    device const TYPE* b [[buffer(1)]],                              \
    device uint* out [[buffer(2)]],                                  \
    constant const uint* out_shape [[buffer(3)]],                    \
    constant const uint* a_strides [[buffer(4)]],                    \
    constant const uint* b_strides [[buffer(5)]],                    \
    constant uint& ndim [[buffer(6)]],                               \
    uint id [[thread_position_in_grid]])                             \
{                                                                    \
    uint a_offset = 0;                                               \
    uint b_offset = 0;                                               \
    uint remaining = id;                                             \
    for (uint d = 0; d < ndim; d++) {                                \
        uint out_stride = 1;                                         \
        for (uint k = d + 1; k < ndim; k++) out_stride *= out_shape[k]; \
        uint coord = remaining / out_stride;                         \
        remaining %= out_stride;                                     \
        a_offset += coord * a_strides[d];                            \
        b_offset += coord * b_strides[d];                            \
    }                                                                \
    out[id] = (a[a_offset] OP b[b_offset]) ? 1u : 0u;               \
}

// Expand all variants for one arithmetic op + type
#define ARITH_ALL(OP_NAME, TYPE, TYPE_SUFFIX, OP)      \
    BINARY_FLAT(OP_NAME ## _ ## TYPE_SUFFIX, TYPE, OP)  \
    BINARY_SS(OP_NAME ## _ss_ ## TYPE_SUFFIX, TYPE, OP) \
    BINARY_SV(OP_NAME ## _sv_ ## TYPE_SUFFIX, TYPE, OP) \
    BINARY_VS(OP_NAME ## _vs_ ## TYPE_SUFFIX, TYPE, OP) \
    BINARY_GENERAL(OP_NAME ## _g_ ## TYPE_SUFFIX, TYPE, OP)

// Expand all variants for one comparison op + type
#define CMP_ALL(OP_NAME, TYPE, TYPE_SUFFIX, OP)        \
    CMP_FLAT(OP_NAME ## _ ## TYPE_SUFFIX, TYPE, OP)     \
    CMP_SS(OP_NAME ## _ss_ ## TYPE_SUFFIX, TYPE, OP)    \
    CMP_SV(OP_NAME ## _sv_ ## TYPE_SUFFIX, TYPE, OP)    \
    CMP_VS(OP_NAME ## _vs_ ## TYPE_SUFFIX, TYPE, OP)    \
    CMP_GENERAL(OP_NAME ## _g_ ## TYPE_SUFFIX, TYPE, OP)

// =====================================================================
// Arithmetic ops: add, sub, mul, div  x  f32, f16, bf16
// =====================================================================

ARITH_ALL(add, float, f32, +)
ARITH_ALL(sub, float, f32, -)
ARITH_ALL(mul, float, f32, *)
ARITH_ALL(div, float, f32, /)

ARITH_ALL(add, half, f16, +)
ARITH_ALL(sub, half, f16, -)
ARITH_ALL(mul, half, f16, *)
ARITH_ALL(div, half, f16, /)

ARITH_ALL(add, bfloat, bf16, +)
ARITH_ALL(sub, bfloat, bf16, -)
ARITH_ALL(mul, bfloat, bf16, *)
ARITH_ALL(div, bfloat, bf16, /)

// =====================================================================
// Comparison ops: eq, ne, lt, le, gt, ge  x  f32, f16, bf16
// =====================================================================

CMP_ALL(eq, float, f32, ==)
CMP_ALL(ne, float, f32, !=)
CMP_ALL(lt, float, f32, <)
CMP_ALL(le, float, f32, <=)
CMP_ALL(gt, float, f32, >)
CMP_ALL(ge, float, f32, >=)

CMP_ALL(eq, half, f16, ==)
CMP_ALL(ne, half, f16, !=)
CMP_ALL(lt, half, f16, <)
CMP_ALL(le, half, f16, <=)
CMP_ALL(gt, half, f16, >)
CMP_ALL(ge, half, f16, >=)

CMP_ALL(eq, bfloat, bf16, ==)
CMP_ALL(ne, bfloat, bf16, !=)
CMP_ALL(lt, bfloat, bf16, <)
CMP_ALL(le, bfloat, bf16, <=)
CMP_ALL(gt, bfloat, bf16, >)
CMP_ALL(ge, bfloat, bf16, >=)
"#;

// ---------------------------------------------------------------------------
// BinaryOp enum
// ---------------------------------------------------------------------------

/// Binary operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    // Comparison ops (output UInt32: 0 or 1)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    /// Whether this is a comparison op (output is UInt32).
    fn is_comparison(&self) -> bool {
        matches!(self, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge)
    }

    /// Base name without dtype suffix.
    fn base_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Mul => "mul",
            BinaryOp::Sub => "sub",
            BinaryOp::Div => "div",
            BinaryOp::Eq => "eq",
            BinaryOp::Ne => "ne",
            BinaryOp::Lt => "lt",
            BinaryOp::Le => "le",
            BinaryOp::Gt => "gt",
            BinaryOp::Ge => "ge",
        }
    }

    /// Metal dtype suffix.
    fn dtype_suffix(dtype: DType) -> Result<&'static str, KernelError> {
        match dtype {
            DType::Float32 => Ok("f32"),
            DType::Float16 => Ok("f16"),
            DType::Bfloat16 => Ok("bf16"),
            DType::UInt32 => Err(KernelError::InvalidShape(
                "binary ops not supported for UInt32; cast to float first".into(),
            )),
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => Err(KernelError::InvalidShape(
                "binary ops not supported for quantized types; dequantize first".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Broadcast dispatch variant
// ---------------------------------------------------------------------------

/// Dispatch variant chosen based on input shapes.
#[derive(Debug, Clone, Copy)]
enum DispatchVariant {
    /// Both inputs are scalars (numel=1).
    ScalarScalar,
    /// `a` is a scalar, `b` is a vector/tensor.
    ScalarVector,
    /// `a` is a vector/tensor, `b` is a scalar.
    VectorScalar,
    /// Same shape -- flat element-wise (original fast path).
    Flat,
    /// General N-dim broadcast with stride buffers.
    General,
}

/// Build the kernel function name for a given op, variant, and dtype.
fn kernel_name_for(op: BinaryOp, variant: DispatchVariant, dtype: DType) -> Result<String, KernelError> {
    let base = op.base_name();
    let suffix = BinaryOp::dtype_suffix(dtype)?;
    let variant_tag = match variant {
        DispatchVariant::ScalarScalar => "_ss_",
        DispatchVariant::ScalarVector => "_sv_",
        DispatchVariant::VectorScalar => "_vs_",
        DispatchVariant::Flat => "_",
        DispatchVariant::General => "_g_",
    };
    Ok(format!("{base}{variant_tag}{suffix}"))
}

/// Choose the dispatch variant based on input shapes.
fn choose_variant(a_shape: &[usize], b_shape: &[usize]) -> DispatchVariant {
    let a_numel: usize = a_shape.iter().product();
    let b_numel: usize = b_shape.iter().product();

    if a_numel == 1 && b_numel == 1 {
        DispatchVariant::ScalarScalar
    } else if a_numel == 1 {
        DispatchVariant::ScalarVector
    } else if b_numel == 1 {
        DispatchVariant::VectorScalar
    } else if a_shape == b_shape {
        DispatchVariant::Flat
    } else {
        DispatchVariant::General
    }
}

// ---------------------------------------------------------------------------
// Broadcast stride computation
// ---------------------------------------------------------------------------

/// Compute broadcast strides for an input shape relative to the output shape.
///
/// For each output dimension, if the corresponding input dimension is 1
/// (broadcast), the stride is set to 0 so the index stays at 0 regardless
/// of the coordinate. Otherwise, the stride is the normal contiguous stride
/// for that dimension in the input.
fn broadcast_strides(input_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let out_ndim = out_shape.len();
    let in_ndim = input_shape.len();
    let mut strides = vec![0usize; out_ndim];

    // Compute contiguous strides for the input.
    let mut in_contiguous = vec![1usize; in_ndim];
    for i in (0..in_ndim.saturating_sub(1)).rev() {
        in_contiguous[i] = in_contiguous[i + 1] * input_shape[i + 1];
    }

    // Map input dims to the trailing dims of the output.
    // Leading dimensions where the input has no corresponding dim get stride 0.
    let offset = out_ndim - in_ndim;
    for i in 0..out_ndim {
        if i < offset {
            // Input doesn't have this dimension -- implicitly size 1.
            strides[i] = 0;
        } else {
            let in_idx = i - offset;
            if input_shape[in_idx] == 1 {
                // Broadcast dimension: stride 0.
                strides[i] = 0;
            } else {
                strides[i] = in_contiguous[in_idx];
            }
        }
    }

    strides
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register binary kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("binary", BINARY_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Core dispatch
// ---------------------------------------------------------------------------

/// Execute a binary operation with broadcasting support.
pub fn binary_op(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op_with_mode(registry, a, b, op, queue, super::ExecMode::Sync)
}

/// Execute a binary operation with broadcasting and explicit execution mode.
pub fn binary_op_with_mode(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    // Validate dtypes match.
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching dtypes: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    // Compute broadcast output shape.
    let out_shape = broadcast_shape(a.shape(), b.shape())?;

    // Ensure contiguous layout for inputs.
    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let variant = choose_variant(a.shape(), b.shape());
    let kernel_name = kernel_name_for(op, variant, a.dtype())?;
    let pipeline = registry.get_pipeline(&kernel_name, a.dtype())?;

    let out_numel: usize = out_shape.iter().product();
    // Handle empty tensors.
    if out_numel == 0 {
        let out_dtype = if op.is_comparison() { DType::UInt32 } else { a.dtype() };
        return Ok(Array::zeros(registry.device().raw(), &out_shape, out_dtype));
    }

    let out_dtype = if op.is_comparison() { DType::UInt32 } else { a.dtype() };
    let out = Array::zeros(registry.device().raw(), &out_shape, out_dtype);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    encoder.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);

    match variant {
        DispatchVariant::General => {
            // Pass out_shape, a_strides, b_strides, ndim as Metal buffers.
            let device = registry.device().raw();
            let ndim = out_shape.len();

            let shape_u32: Vec<u32> = out_shape
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("out_shape[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;

            let a_bcast_strides = broadcast_strides(a.shape(), &out_shape);
            let b_bcast_strides = broadcast_strides(b.shape(), &out_shape);

            let a_strides_u32: Vec<u32> = a_bcast_strides
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("a_strides[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;
            let b_strides_u32: Vec<u32> = b_bcast_strides
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("b_strides[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;

            let buf_size = |v: &[u32]| (v.len() * std::mem::size_of::<u32>()) as u64;

            let shape_buf = device.new_buffer_with_data(
                shape_u32.as_ptr() as *const _,
                buf_size(&shape_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let a_stride_buf = device.new_buffer_with_data(
                a_strides_u32.as_ptr() as *const _,
                buf_size(&a_strides_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let b_stride_buf = device.new_buffer_with_data(
                b_strides_u32.as_ptr() as *const _,
                buf_size(&b_strides_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let ndim_val = super::checked_u32(ndim, "ndim")?;
            let ndim_buf = device.new_buffer_with_data(
                &ndim_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(3, Some(&shape_buf), 0);
            encoder.set_buffer(4, Some(&a_stride_buf), 0);
            encoder.set_buffer(5, Some(&b_stride_buf), 0);
            encoder.set_buffer(6, Some(&ndim_buf), 0);
        }
        _ => {
            // Flat, SS, SV, VS: no extra buffers needed.
        }
    }

    let grid_size = MTLSize::new(out_numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), out_numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Execute a binary operation asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
pub fn binary_op_async(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
) -> Result<super::LaunchResult, KernelError> {
    // Validate dtypes match.
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching dtypes: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    // Compute broadcast output shape.
    let out_shape = broadcast_shape(a.shape(), b.shape())?;

    // Ensure contiguous layout for inputs.
    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let variant = choose_variant(a.shape(), b.shape());
    let kernel_name = kernel_name_for(op, variant, a.dtype())?;
    let pipeline = registry.get_pipeline(&kernel_name, a.dtype())?;

    let out_numel: usize = out_shape.iter().product();

    let out_dtype = if op.is_comparison() { DType::UInt32 } else { a.dtype() };
    let out = Array::zeros(registry.device().raw(), &out_shape, out_dtype);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    encoder.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);

    match variant {
        DispatchVariant::General => {
            let device = registry.device().raw();
            let ndim = out_shape.len();

            let shape_u32: Vec<u32> = out_shape
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("out_shape[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;

            let a_bcast_strides = broadcast_strides(a.shape(), &out_shape);
            let b_bcast_strides = broadcast_strides(b.shape(), &out_shape);

            let a_strides_u32: Vec<u32> = a_bcast_strides
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("a_strides[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;
            let b_strides_u32: Vec<u32> = b_bcast_strides
                .iter()
                .enumerate()
                .map(|(i, &s)| super::checked_u32(s, &format!("b_strides[{i}]")))
                .collect::<Result<Vec<u32>, _>>()?;

            let buf_size = |v: &[u32]| (v.len() * std::mem::size_of::<u32>()) as u64;

            let shape_buf = device.new_buffer_with_data(
                shape_u32.as_ptr() as *const _,
                buf_size(&shape_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let a_stride_buf = device.new_buffer_with_data(
                a_strides_u32.as_ptr() as *const _,
                buf_size(&a_strides_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let b_stride_buf = device.new_buffer_with_data(
                b_strides_u32.as_ptr() as *const _,
                buf_size(&b_strides_u32),
                metal::MTLResourceOptions::StorageModeShared,
            );
            let ndim_val = super::checked_u32(ndim, "ndim")?;
            let ndim_buf = device.new_buffer_with_data(
                &ndim_val as *const u32 as *const _,
                std::mem::size_of::<u32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(3, Some(&shape_buf), 0);
            encoder.set_buffer(4, Some(&a_stride_buf), 0);
            encoder.set_buffer(5, Some(&b_stride_buf), 0);
            encoder.set_buffer(6, Some(&ndim_buf), 0);
        }
        _ => {}
    }

    let grid_numel = if out_numel == 0 { 1 } else { out_numel };
    let grid_size = MTLSize::new(grid_numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_numel as u64),
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
// Convenience functions (backward compatible -- now support broadcasting)
// ---------------------------------------------------------------------------

/// Element-wise addition with broadcasting.
pub fn add(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Add, queue)
}

/// Element-wise multiplication with broadcasting.
pub fn mul(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Mul, queue)
}

/// Element-wise subtraction with broadcasting.
pub fn sub(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Sub, queue)
}

/// Element-wise division with broadcasting.
pub fn div(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Div, queue)
}

/// Element-wise equality comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn eq(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Eq, queue)
}

/// Element-wise not-equal comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn ne(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Ne, queue)
}

/// Element-wise less-than comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn lt(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Lt, queue)
}

/// Element-wise less-equal comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn le(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Le, queue)
}

/// Element-wise greater-than comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn gt(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Gt, queue)
}

/// Element-wise greater-equal comparison with broadcasting. Output is UInt32 (0 or 1).
pub fn ge(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Ge, queue)
}

// ---------------------------------------------------------------------------
// Unit tests (CPU-only, no Metal device required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_strides_same_shape() {
        // [3, 4] broadcast to [3, 4] -> normal contiguous strides [4, 1]
        let strides = broadcast_strides(&[3, 4], &[3, 4]);
        assert_eq!(strides, vec![4, 1]);
    }

    #[test]
    fn test_broadcast_strides_scalar_to_nd() {
        // [1] broadcast to [3, 4] -> [0, 0]
        let strides = broadcast_strides(&[1], &[3, 4]);
        assert_eq!(strides, vec![0, 0]);
    }

    #[test]
    fn test_broadcast_strides_row_vector() {
        // [1, 4] broadcast to [3, 4] -> [0, 1]
        let strides = broadcast_strides(&[1, 4], &[3, 4]);
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn test_broadcast_strides_col_vector() {
        // [3, 1] broadcast to [3, 4] -> [1, 0]
        let strides = broadcast_strides(&[3, 1], &[3, 4]);
        assert_eq!(strides, vec![1, 0]);
    }

    #[test]
    fn test_broadcast_strides_lower_ndim() {
        // [4] broadcast to [3, 4] -> [0, 1]
        let strides = broadcast_strides(&[4], &[3, 4]);
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn test_broadcast_strides_complex() {
        // [8, 1, 6, 1] broadcast to [8, 7, 6, 5]
        // input contiguous strides: [6, 6, 1, 1]
        // broadcast: dim1 (1->7): stride 0, dim3 (1->5): stride 0
        // result: [6, 0, 1, 0]
        let strides = broadcast_strides(&[8, 1, 6, 1], &[8, 7, 6, 5]);
        assert_eq!(strides, vec![6, 0, 1, 0]);
    }

    #[test]
    fn test_choose_variant_scalar_scalar() {
        assert!(matches!(choose_variant(&[1], &[1]), DispatchVariant::ScalarScalar));
    }

    #[test]
    fn test_choose_variant_scalar_vector() {
        assert!(matches!(choose_variant(&[1], &[4]), DispatchVariant::ScalarVector));
    }

    #[test]
    fn test_choose_variant_vector_scalar() {
        assert!(matches!(choose_variant(&[4], &[1]), DispatchVariant::VectorScalar));
    }

    #[test]
    fn test_choose_variant_same_shape() {
        assert!(matches!(choose_variant(&[3, 4], &[3, 4]), DispatchVariant::Flat));
    }

    #[test]
    fn test_choose_variant_broadcast() {
        assert!(matches!(choose_variant(&[3, 1], &[1, 4]), DispatchVariant::General));
    }

    #[test]
    fn test_kernel_name_flat() {
        let name = kernel_name_for(BinaryOp::Add, DispatchVariant::Flat, DType::Float32).unwrap();
        assert_eq!(name, "add_f32");
    }

    #[test]
    fn test_kernel_name_ss() {
        let name = kernel_name_for(BinaryOp::Mul, DispatchVariant::ScalarScalar, DType::Float16).unwrap();
        assert_eq!(name, "mul_ss_f16");
    }

    #[test]
    fn test_kernel_name_sv() {
        let name = kernel_name_for(BinaryOp::Sub, DispatchVariant::ScalarVector, DType::Bfloat16).unwrap();
        assert_eq!(name, "sub_sv_bf16");
    }

    #[test]
    fn test_kernel_name_vs() {
        let name = kernel_name_for(BinaryOp::Div, DispatchVariant::VectorScalar, DType::Float32).unwrap();
        assert_eq!(name, "div_vs_f32");
    }

    #[test]
    fn test_kernel_name_general() {
        let name = kernel_name_for(BinaryOp::Add, DispatchVariant::General, DType::Float32).unwrap();
        assert_eq!(name, "add_g_f32");
    }

    #[test]
    fn test_kernel_name_cmp() {
        let name = kernel_name_for(BinaryOp::Eq, DispatchVariant::Flat, DType::Float32).unwrap();
        assert_eq!(name, "eq_f32");
        let name = kernel_name_for(BinaryOp::Lt, DispatchVariant::General, DType::Float16).unwrap();
        assert_eq!(name, "lt_g_f16");
    }

    #[test]
    fn test_kernel_name_quantized_error() {
        let result = kernel_name_for(BinaryOp::Add, DispatchVariant::Flat, DType::Q8_0);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_name_uint32_error() {
        let result = kernel_name_for(BinaryOp::Add, DispatchVariant::Flat, DType::UInt32);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_comparison() {
        assert!(!BinaryOp::Add.is_comparison());
        assert!(!BinaryOp::Mul.is_comparison());
        assert!(!BinaryOp::Sub.is_comparison());
        assert!(!BinaryOp::Div.is_comparison());
        assert!(BinaryOp::Eq.is_comparison());
        assert!(BinaryOp::Ne.is_comparison());
        assert!(BinaryOp::Lt.is_comparison());
        assert!(BinaryOp::Le.is_comparison());
        assert!(BinaryOp::Gt.is_comparison());
        assert!(BinaryOp::Ge.is_comparison());
    }
}
