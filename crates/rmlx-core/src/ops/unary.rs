//! Standalone unary operations: exp, log, sqrt, rsqrt, sigmoid, abs, neg, tanh.
//!
//! Each operation takes an `Array` input and returns an `Array` output of the
//! same shape and dtype.
//!
//! ## Vectorisation
//!
//! - **f32**: 2 elements per thread.
//! - **f16 / bf16**: 4 elements per thread with f32 accumulation for precision.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader source for all unary kernels.
///
/// Includes f32 (2-element vectorization) and f16/bf16 (4-element vectorization)
/// variants for: exp, log, sqrt, rsqrt, sigmoid, abs, neg, tanh.
pub const UNARY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── Macro for f32 unary ops (2 elements per thread) ────────────────────

#define UNARY_F32(NAME, EXPR)                                           \
kernel void NAME(                                                       \
    device const float* input  [[buffer(0)]],                           \
    device       float* output [[buffer(1)]],                           \
    constant     uint&  numel  [[buffer(2)]],                           \
    uint id [[thread_position_in_grid]])                                \
{                                                                       \
    uint base = id * 2;                                                 \
    if (base + 1 < numel) {                                             \
        float x0 = input[base];                                         \
        float x1 = input[base + 1];                                     \
        output[base]     = EXPR(x0);                                    \
        output[base + 1] = EXPR(x1);                                    \
    } else if (base < numel) {                                          \
        float x0 = input[base];                                         \
        output[base] = EXPR(x0);                                        \
    }                                                                   \
}

// ─── Macro for f16 unary ops (4 elements per thread) ────────────────────

#define UNARY_F16(NAME, EXPR)                                           \
kernel void NAME(                                                       \
    device const half*  input  [[buffer(0)]],                           \
    device       half*  output [[buffer(1)]],                           \
    constant     uint&  numel  [[buffer(2)]],                           \
    uint id [[thread_position_in_grid]])                                \
{                                                                       \
    uint base = id * 4;                                                 \
    if (base + 3 < numel) {                                             \
        float x0 = float(input[base]);                                  \
        float x1 = float(input[base + 1]);                              \
        float x2 = float(input[base + 2]);                              \
        float x3 = float(input[base + 3]);                              \
        output[base]     = half(EXPR(x0));                              \
        output[base + 1] = half(EXPR(x1));                              \
        output[base + 2] = half(EXPR(x2));                              \
        output[base + 3] = half(EXPR(x3));                              \
    } else {                                                            \
        for (uint i = base; i < min(base + 4, numel); i++) {           \
            float x = float(input[i]);                                  \
            output[i] = half(EXPR(x));                                  \
        }                                                               \
    }                                                                   \
}

// ─── Macro for bf16 unary ops (4 elements per thread) ───────────────────

#define UNARY_BF16(NAME, EXPR)                                          \
kernel void NAME(                                                       \
    device const bfloat* input  [[buffer(0)]],                          \
    device       bfloat* output [[buffer(1)]],                          \
    constant     uint&   numel  [[buffer(2)]],                          \
    uint id [[thread_position_in_grid]])                                \
{                                                                       \
    uint base = id * 4;                                                 \
    if (base + 3 < numel) {                                             \
        float x0 = float(input[base]);                                  \
        float x1 = float(input[base + 1]);                              \
        float x2 = float(input[base + 2]);                              \
        float x3 = float(input[base + 3]);                              \
        output[base]     = bfloat(EXPR(x0));                            \
        output[base + 1] = bfloat(EXPR(x1));                            \
        output[base + 2] = bfloat(EXPR(x2));                            \
        output[base + 3] = bfloat(EXPR(x3));                            \
    } else {                                                            \
        for (uint i = base; i < min(base + 4, numel); i++) {           \
            float x = float(input[i]);                                  \
            output[i] = bfloat(EXPR(x));                                \
        }                                                               \
    }                                                                   \
}

// ─── Helper expressions ─────────────────────────────────────────────────

inline float op_exp(float x) { return exp(x); }
inline float op_log(float x) { return log(x); }
inline float op_sqrt(float x) { return sqrt(x); }
inline float op_rsqrt(float x) { return rsqrt(x); }
inline float op_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float op_abs(float x) { return abs(x); }
inline float op_neg(float x) { return -x; }
inline float op_tanh(float x) { return tanh(x); }

// ─── Instantiate all kernels ────────────────────────────────────────────

UNARY_F32(unary_exp_f32, op_exp)
UNARY_F32(unary_log_f32, op_log)
UNARY_F32(unary_sqrt_f32, op_sqrt)
UNARY_F32(unary_rsqrt_f32, op_rsqrt)
UNARY_F32(unary_sigmoid_f32, op_sigmoid)
UNARY_F32(unary_abs_f32, op_abs)
UNARY_F32(unary_neg_f32, op_neg)
UNARY_F32(unary_tanh_f32, op_tanh)

UNARY_F16(unary_exp_f16, op_exp)
UNARY_F16(unary_log_f16, op_log)
UNARY_F16(unary_sqrt_f16, op_sqrt)
UNARY_F16(unary_rsqrt_f16, op_rsqrt)
UNARY_F16(unary_sigmoid_f16, op_sigmoid)
UNARY_F16(unary_abs_f16, op_abs)
UNARY_F16(unary_neg_f16, op_neg)
UNARY_F16(unary_tanh_f16, op_tanh)

UNARY_BF16(unary_exp_bf16, op_exp)
UNARY_BF16(unary_log_bf16, op_log)
UNARY_BF16(unary_sqrt_bf16, op_sqrt)
UNARY_BF16(unary_rsqrt_bf16, op_rsqrt)
UNARY_BF16(unary_sigmoid_bf16, op_sigmoid)
UNARY_BF16(unary_abs_bf16, op_abs)
UNARY_BF16(unary_neg_bf16, op_neg)
UNARY_BF16(unary_tanh_bf16, op_tanh)
"#;

/// Register all unary kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("unary", UNARY_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Internal dispatch
// ---------------------------------------------------------------------------

/// Dispatch a unary operation on the GPU.
fn dispatch_unary(
    registry: &KernelRegistry,
    input: &Array,
    op_name: &str,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let input_c = super::make_contiguous(input, registry, queue)?;
    let input = input_c.as_ref().unwrap_or(input);

    let (kernel_name, elems_per_thread) = match input.dtype() {
        DType::Float32 => (format!("unary_{op_name}_f32"), 2usize),
        DType::Float16 => (format!("unary_{op_name}_f16"), 4usize),
        DType::Bfloat16 => (format!("unary_{op_name}_bf16"), 4usize),
        _ => {
            return Err(KernelError::NotFound(format!(
                "unary_{op_name} not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(&kernel_name, input.dtype())?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, input.shape(), input.dtype());

    let numel = input.numel();
    let numel_buf = dev.new_buffer_with_data(
        &(numel as u32) as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&numel_buf), 0);

    let n_threads = numel.div_ceil(elems_per_thread);
    let tg_size = std::cmp::min(256u64, pipeline.max_total_threads_per_threadgroup());
    let n_groups = (n_threads as u64).div_ceil(tg_size);

    enc.dispatch_thread_groups(MTLSize::new(n_groups, 1, 1), MTLSize::new(tg_size, 1, 1));
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Element-wise exponential: `output[i] = exp(input[i])`.
pub fn exp(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "exp", queue)
}

/// Element-wise natural logarithm: `output[i] = log(input[i])`.
pub fn log(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "log", queue)
}

/// Element-wise square root: `output[i] = sqrt(input[i])`.
pub fn sqrt(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "sqrt", queue)
}

/// Element-wise reciprocal square root: `output[i] = 1/sqrt(input[i])`.
pub fn rsqrt(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "rsqrt", queue)
}

/// Element-wise sigmoid: `output[i] = 1/(1 + exp(-input[i]))`.
pub fn sigmoid(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "sigmoid", queue)
}

/// Element-wise absolute value: `output[i] = |input[i]|`.
pub fn abs(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "abs", queue)
}

/// Element-wise negation: `output[i] = -input[i]`.
pub fn neg(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "neg", queue)
}

/// Element-wise hyperbolic tangent: `output[i] = tanh(input[i])`.
pub fn tanh_op(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    dispatch_unary(registry, input, "tanh", queue)
}
