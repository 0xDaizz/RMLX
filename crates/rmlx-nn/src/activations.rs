//! Standalone activation function modules (N10).
//!
//! Provides nn-module wrappers around the element-wise activation ops
//! in `rmlx_core::ops`. Each module is a zero-parameter layer that
//! can be composed in model definitions.
//!
//! Supported activations:
//! - [`SiLU`] — x * sigmoid(x)  (also known as Swish)
//! - [`GELU`] — tanh approximation (GPT-2, BERT)
//! - [`GELUFast`] — sigmoid approximation (faster, slightly less precise)
//! - [`Sigmoid`] — 1 / (1 + exp(-x))
//! - [`Tanh`] — hyperbolic tangent
//! - [`Swish`] — alias for SiLU
//! - [`ReLU`] — max(0, x)
//! - [`LeakyReLU`] — x if x > 0, alpha*x otherwise (default alpha=0.01)
//! - [`ELU`] — x if x > 0, alpha*(exp(x)-1) otherwise (default alpha=1.0)
//! - [`SELU`] — scale * elu(x, alpha) with Lecun constants
//! - [`Mish`] — x * tanh(softplus(x))
//! - [`QuickGELU`] — x * sigmoid(1.702 * x)
//! - [`HardSwish`] — x * clamp(x/6 + 0.5, 0, 1)
//! - [`HardSigmoid`] — clamp(x/6 + 0.5, 0, 1)
//! - [`Softplus`] — log(1 + exp(beta*x)) / beta (default beta=1.0)
//! - [`Softsign`] — x / (1 + |x|)
//! - [`GLU`] — splits input in half along last axis, applies sigmoid gate

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLResourceOptions};
use rmlx_metal::{ComputePass, MTLSize, MtlBuffer};
/// Trait for activation function modules.
///
/// All activations take an input array and return an output of the same shape.
pub trait Activation {
    /// Apply the activation function element-wise.
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError>;

    /// Human-readable name for debugging/logging.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// JIT Metal shader for activations that cannot be composed from existing ops
// ---------------------------------------------------------------------------

/// Metal shader source for custom activation kernels (ReLU, LeakyReLU, ELU,
/// SELU, Mish, HardSwish, HardSigmoid, Softplus, Softsign).
///
/// All kernels use the standard unary dispatch pattern: 2 elements/thread for
/// f32, 4 elements/thread for f16/bf16.
const ACTIVATION_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── ReLU: max(0, x) ───────────────────────────────────────────────────

kernel void relu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        output[base]     = max(0.0f, input[base]);
        output[base + 1] = max(0.0f, input[base + 1]);
    } else if (base < numel) {
        output[base] = max(0.0f, input[base]);
    }
}

kernel void relu_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) output[base+i] = max(half(0), input[base+i]);
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) output[i] = max(half(0), input[i]);
    }
}

kernel void relu_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(max(0.0f, x));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(max(0.0f, x));
        }
    }
}

// ─── LeakyReLU: x > 0 ? x : alpha*x ───────────────────────────────────

kernel void leaky_relu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = x0 > 0.0f ? x0 : alpha * x0;
        output[base+1] = x1 > 0.0f ? x1 : alpha * x1;
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 > 0.0f ? x0 : alpha * x0;
    }
}

kernel void leaky_relu_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(x > 0.0f ? x : alpha * x);
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(x > 0.0f ? x : alpha * x);
        }
    }
}

kernel void leaky_relu_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    constant     float&  alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(x > 0.0f ? x : alpha * x);
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(x > 0.0f ? x : alpha * x);
        }
    }
}

// ─── ELU: x > 0 ? x : alpha*(exp(x)-1) ────────────────────────────────

kernel void elu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = x0 > 0.0f ? x0 : alpha * (exp(x0) - 1.0f);
        output[base+1] = x1 > 0.0f ? x1 : alpha * (exp(x1) - 1.0f);
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 > 0.0f ? x0 : alpha * (exp(x0) - 1.0f);
    }
}

kernel void elu_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(x > 0.0f ? x : alpha * (exp(x) - 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(x > 0.0f ? x : alpha * (exp(x) - 1.0f));
        }
    }
}

kernel void elu_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    constant     float&  alpha  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(x > 0.0f ? x : alpha * (exp(x) - 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(x > 0.0f ? x : alpha * (exp(x) - 1.0f));
        }
    }
}

// ─── SELU: scale * (x > 0 ? x : alpha*(exp(x)-1)) ────────────────────
// Lecun constants: alpha = 1.6732632423543772, scale = 1.0507009873554805

kernel void selu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    constant float SELU_ALPHA = 1.6732632423543772f;
    constant float SELU_SCALE = 1.0507009873554805f;
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = SELU_SCALE * (x0 > 0.0f ? x0 : SELU_ALPHA * (exp(x0) - 1.0f));
        output[base+1] = SELU_SCALE * (x1 > 0.0f ? x1 : SELU_ALPHA * (exp(x1) - 1.0f));
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = SELU_SCALE * (x0 > 0.0f ? x0 : SELU_ALPHA * (exp(x0) - 1.0f));
    }
}

kernel void selu_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    constant float SELU_ALPHA = 1.6732632423543772f;
    constant float SELU_SCALE = 1.0507009873554805f;
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(SELU_SCALE * (x > 0.0f ? x : SELU_ALPHA * (exp(x) - 1.0f)));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(SELU_SCALE * (x > 0.0f ? x : SELU_ALPHA * (exp(x) - 1.0f)));
        }
    }
}

kernel void selu_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    constant float SELU_ALPHA = 1.6732632423543772f;
    constant float SELU_SCALE = 1.0507009873554805f;
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(SELU_SCALE * (x > 0.0f ? x : SELU_ALPHA * (exp(x) - 1.0f)));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(SELU_SCALE * (x > 0.0f ? x : SELU_ALPHA * (exp(x) - 1.0f)));
        }
    }
}

// ─── Mish: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x)) ─

kernel void mish_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        float sp0 = log(1.0f + exp(x0));
        float sp1 = log(1.0f + exp(x1));
        output[base]   = x0 * tanh(sp0);
        output[base+1] = x1 * tanh(sp1);
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 * tanh(log(1.0f + exp(x0)));
    }
}

kernel void mish_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(x * tanh(log(1.0f + exp(x))));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(x * tanh(log(1.0f + exp(x))));
        }
    }
}

kernel void mish_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(x * tanh(log(1.0f + exp(x))));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(x * tanh(log(1.0f + exp(x))));
        }
    }
}

// ─── HardSwish: x * clamp(x/6 + 0.5, 0, 1) ───────────────────────────

kernel void hard_swish_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = x0 * clamp(x0 / 6.0f + 0.5f, 0.0f, 1.0f);
        output[base+1] = x1 * clamp(x1 / 6.0f + 0.5f, 0.0f, 1.0f);
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 * clamp(x0 / 6.0f + 0.5f, 0.0f, 1.0f);
    }
}

kernel void hard_swish_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    }
}

kernel void hard_swish_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    }
}

// ─── HardSigmoid: clamp(x/6 + 0.5, 0, 1) ─────────────────────────────

kernel void hard_sigmoid_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        output[base]   = clamp(input[base]   / 6.0f + 0.5f, 0.0f, 1.0f);
        output[base+1] = clamp(input[base+1] / 6.0f + 0.5f, 0.0f, 1.0f);
    } else if (base < numel) {
        output[base] = clamp(input[base] / 6.0f + 0.5f, 0.0f, 1.0f);
    }
}

kernel void hard_sigmoid_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    }
}

kernel void hard_sigmoid_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(clamp(x / 6.0f + 0.5f, 0.0f, 1.0f));
        }
    }
}

// ─── Softplus: log(1 + exp(beta*x)) / beta ────────────────────────────

kernel void softplus_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& beta   [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    float inv_beta = 1.0f / beta;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = log(1.0f + exp(beta * x0)) * inv_beta;
        output[base+1] = log(1.0f + exp(beta * x1)) * inv_beta;
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = log(1.0f + exp(beta * x0)) * inv_beta;
    }
}

kernel void softplus_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    constant     float& beta   [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    float inv_beta = 1.0f / beta;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(log(1.0f + exp(beta * x)) * inv_beta);
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(log(1.0f + exp(beta * x)) * inv_beta);
        }
    }
}

kernel void softplus_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    constant     float&  beta   [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    float inv_beta = 1.0f / beta;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(log(1.0f + exp(beta * x)) * inv_beta);
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(log(1.0f + exp(beta * x)) * inv_beta);
        }
    }
}

// ─── Softsign: x / (1 + |x|) ──────────────────────────────────────────

kernel void softsign_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base], x1 = input[base+1];
        output[base]   = x0 / (1.0f + abs(x0));
        output[base+1] = x1 / (1.0f + abs(x1));
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 / (1.0f + abs(x0));
    }
}

kernel void softsign_f16(
    device const half*  input  [[buffer(0)]],
    device       half*  output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = half(x / (1.0f + abs(x)));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = half(x / (1.0f + abs(x)));
        }
    }
}

kernel void softsign_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        for (uint i = 0; i < 4; i++) {
            float x = float(input[base+i]);
            output[base+i] = bfloat(x / (1.0f + abs(x)));
        }
    } else {
        for (uint i = base; i < min(base + 4, numel); i++) {
            float x = float(input[i]);
            output[i] = bfloat(x / (1.0f + abs(x)));
        }
    }
}
"#;

/// Register custom activation kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("activations", ACTIVATION_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Internal dispatch helpers
// ---------------------------------------------------------------------------

/// If the array is non-contiguous, return a contiguous copy. Otherwise `None`.
fn ensure_contiguous(
    array: &Array,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Option<Array>, KernelError> {
    if array.is_contiguous() {
        Ok(None)
    } else {
        Ok(Some(ops::copy::copy(registry, array, queue)?))
    }
}

/// Dispatch a custom activation kernel that takes only (input, output, numel).
fn dispatch_activation_simple(
    registry: &KernelRegistry,
    input: &Array,
    kernel_base: &str,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let input_c = ensure_contiguous(input, registry, queue)?;
    let input = input_c.as_ref().unwrap_or(input);

    let (suffix, elems_per_thread) = dtype_info(input.dtype())?;
    let kernel_name = format!("{kernel_base}_{suffix}");
    let pipeline = registry.get_pipeline(&kernel_name, input.dtype())?;

    let numel = input.numel();
    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), input.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&numel_buf), 0);

    let n_threads = numel.div_ceil(elems_per_thread);
    let tg_size = std::cmp::min(256usize, pipeline.maxTotalThreadsPerThreadgroup());
    enc.dispatch_threads(
        MTLSize { width: n_threads, height: 1, depth: 1 },
        MTLSize { width: tg_size, height: 1, depth: 1 },
    );
    enc.end();
    cb.commit();
    cb.waitUntilCompleted();

    Ok(out)
}

/// Dispatch a custom activation kernel with an extra f32 parameter
/// (input, output, numel, param).
fn dispatch_activation_param(
    registry: &KernelRegistry,
    input: &Array,
    kernel_base: &str,
    param: f32,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let input_c = ensure_contiguous(input, registry, queue)?;
    let input = input_c.as_ref().unwrap_or(input);

    let (suffix, elems_per_thread) = dtype_info(input.dtype())?;
    let kernel_name = format!("{kernel_base}_{suffix}");
    let pipeline = registry.get_pipeline(&kernel_name, input.dtype())?;

    let numel = input.numel();
    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);
    let param_buf = make_f32_buf(registry.device().raw(), param);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), input.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(2, Some(&numel_buf), 0);
    enc.set_buffer(3, Some(&param_buf), 0);

    let n_threads = numel.div_ceil(elems_per_thread);
    let tg_size = std::cmp::min(256usize, pipeline.maxTotalThreadsPerThreadgroup());
    enc.dispatch_threads(
        MTLSize { width: n_threads, height: 1, depth: 1 },
        MTLSize { width: tg_size, height: 1, depth: 1 },
    );
    enc.end();
    cb.commit();
    cb.waitUntilCompleted();

    Ok(out)
}

/// Return (dtype_suffix, elements_per_thread) for a given dtype.
fn dtype_info(dtype: DType) -> Result<(&'static str, usize), KernelError> {
    match dtype {
        DType::Float32 => Ok(("f32", 2)),
        DType::Float16 => Ok(("f16", 4)),
        DType::Bfloat16 => Ok(("bf16", 4)),
        _ => Err(KernelError::NotFound(format!(
            "activation not supported for {:?}",
            dtype
        ))),
    }
}

/// Create a constant `uint` buffer on the device.
fn make_u32_buf(device: &ProtocolObject<dyn MTLDevice>, val: u32) -> MtlBuffer {
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new_unchecked(&val as *const u32 as *mut std::ffi::c_void), 4, MTLResourceOptions::StorageModeShared) }.unwrap()
}

/// Create a constant `float` buffer on the device.
fn make_f32_buf(device: &ProtocolObject<dyn MTLDevice>, val: f32) -> MtlBuffer {
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new_unchecked(&val as *const f32 as *mut std::ffi::c_void), 4, MTLResourceOptions::StorageModeShared) }.unwrap()
}

// ===========================================================================
// Existing activations (composing from rmlx_core::ops)
// ===========================================================================

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
///
/// Also known as Swish-1. Used in LLaMA, Mistral, and most modern LLMs.
pub struct SiLU;

impl Activation for SiLU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::silu::silu(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "SiLU"
    }
}

/// GELU activation using the tanh approximation (GPT-2, BERT, Gemma):
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub struct GELU;

impl Activation for GELU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::gelu::gelu(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "GELU"
    }
}

/// GELU activation using the fast sigmoid approximation:
/// `gelu_fast(x) = x * sigmoid(1.702 * x)`.
///
/// Slightly less precise than the tanh approximation but faster to compute.
pub struct GELUFast;

impl Activation for GELUFast {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::gelu::gelu_fast(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "GELUFast"
    }
}

/// Swish activation: alias for [`SiLU`] (x * sigmoid(x)).
///
/// Swish-1 is mathematically identical to SiLU. This alias is provided
/// for code that uses the "Swish" naming convention.
pub type Swish = SiLU;

/// Sigmoid activation: 1 / (1 + exp(-x)).
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::unary::sigmoid(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
pub struct Tanh;

impl Activation for Tanh {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::unary::tanh_op(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "Tanh"
    }
}

// ===========================================================================
// New activations (JIT Metal shaders)
// ===========================================================================

/// ReLU activation: max(0, x).
pub struct ReLU;

impl Activation for ReLU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "relu", queue)
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }
}

/// LeakyReLU activation: x if x > 0, alpha*x otherwise.
///
/// Default alpha = 0.01.
pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    /// Create a LeakyReLU with the default alpha of 0.01.
    pub fn new() -> Self {
        Self { alpha: 0.01 }
    }

    /// Create a LeakyReLU with a custom alpha.
    pub fn with_alpha(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for LeakyReLU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_param(registry, input, "leaky_relu", self.alpha, queue)
    }

    fn name(&self) -> &'static str {
        "LeakyReLU"
    }
}

/// ELU activation: x if x > 0, alpha*(exp(x)-1) otherwise.
///
/// Default alpha = 1.0.
pub struct ELU {
    pub alpha: f32,
}

impl ELU {
    /// Create an ELU with the default alpha of 1.0.
    pub fn new() -> Self {
        Self { alpha: 1.0 }
    }

    /// Create an ELU with a custom alpha.
    pub fn with_alpha(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for ELU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_param(registry, input, "elu", self.alpha, queue)
    }

    fn name(&self) -> &'static str {
        "ELU"
    }
}

/// SELU activation: scale * (x if x > 0, alpha*(exp(x)-1) otherwise).
///
/// Uses the Lecun constants:
/// - alpha = 1.6732632423543772
/// - scale = 1.0507009873554805
pub struct SELU;

impl Activation for SELU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "selu", queue)
    }

    fn name(&self) -> &'static str {
        "SELU"
    }
}

/// Mish activation: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x)).
pub struct Mish;

impl Activation for Mish {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "mish", queue)
    }

    fn name(&self) -> &'static str {
        "Mish"
    }
}

/// QuickGELU activation: x * sigmoid(1.702 * x).
///
/// Mathematically identical to GELUFast, delegates to gelu_fast.
pub struct QuickGELU;

impl Activation for QuickGELU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::gelu::gelu_fast(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "QuickGELU"
    }
}

/// HardSwish activation: x * clamp(x/6 + 0.5, 0, 1).
pub struct HardSwish;

impl Activation for HardSwish {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "hard_swish", queue)
    }

    fn name(&self) -> &'static str {
        "HardSwish"
    }
}

/// HardSigmoid activation: clamp(x/6 + 0.5, 0, 1).
pub struct HardSigmoid;

impl Activation for HardSigmoid {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "hard_sigmoid", queue)
    }

    fn name(&self) -> &'static str {
        "HardSigmoid"
    }
}

/// Softplus activation: log(1 + exp(beta*x)) / beta.
///
/// Default beta = 1.0.
pub struct Softplus {
    pub beta: f32,
}

impl Softplus {
    /// Create a Softplus with the default beta of 1.0.
    pub fn new() -> Self {
        Self { beta: 1.0 }
    }

    /// Create a Softplus with a custom beta.
    pub fn with_beta(beta: f32) -> Self {
        Self { beta }
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Softplus {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_param(registry, input, "softplus", self.beta, queue)
    }

    fn name(&self) -> &'static str {
        "Softplus"
    }
}

/// Softsign activation: x / (1 + |x|).
pub struct Softsign;

impl Activation for Softsign {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        dispatch_activation_simple(registry, input, "softsign", queue)
    }

    fn name(&self) -> &'static str {
        "Softsign"
    }
}

/// GLU (Gated Linear Unit) activation.
///
/// Splits the input tensor in half along the last axis, then computes:
/// `output = a * sigmoid(b)` where `[a, b] = split(input)`.
///
/// The last dimension of the input must be even.
pub struct GLU;

impl Activation for GLU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Err(KernelError::InvalidShape(
                "GLU requires at least 1D input".into(),
            ));
        }
        let last_dim = input.shape()[ndim - 1];
        if last_dim % 2 != 0 {
            return Err(KernelError::InvalidShape(format!(
                "GLU requires even last dimension, got {}",
                last_dim
            )));
        }
        let half = last_dim / 2;
        let last_axis = ndim - 1;

        // Split input along last axis
        let a = input.slice(last_axis, 0, half)?;
        let b = input.slice(last_axis, half, last_dim)?;

        // Ensure contiguous for ops
        let a_c = ensure_contiguous(&a, registry, queue)?;
        let a_ref = a_c.as_ref().unwrap_or(&a);
        let b_c = ensure_contiguous(&b, registry, queue)?;
        let b_ref = b_c.as_ref().unwrap_or(&b);

        // sigmoid(b)
        let sig_b = ops::unary::sigmoid(registry, b_ref, queue)?;
        // a * sigmoid(b)
        ops::binary::mul(registry, a_ref, &sig_b, queue)
    }

    fn name(&self) -> &'static str {
        "GLU"
    }
}

// ===========================================================================
// ActivationType enum (dynamic dispatch)
// ===========================================================================

/// Enumeration of all supported activation types for dynamic dispatch.
///
/// Useful in configuration-driven model construction where the activation
/// type is specified as a string or config parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    SiLU,
    GELU,
    GELUFast,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    ELU,
    SELU,
    Mish,
    QuickGELU,
    HardSwish,
    HardSigmoid,
    Softplus,
    Softsign,
    GLU,
}

impl ActivationType {
    /// Apply the activation function element-wise (dynamic dispatch).
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        match self {
            ActivationType::SiLU => SiLU.forward(input, registry, queue),
            ActivationType::GELU => GELU.forward(input, registry, queue),
            ActivationType::GELUFast => GELUFast.forward(input, registry, queue),
            ActivationType::Sigmoid => Sigmoid.forward(input, registry, queue),
            ActivationType::Tanh => Tanh.forward(input, registry, queue),
            ActivationType::ReLU => ReLU.forward(input, registry, queue),
            ActivationType::LeakyReLU => LeakyReLU::new().forward(input, registry, queue),
            ActivationType::ELU => ELU::new().forward(input, registry, queue),
            ActivationType::SELU => SELU.forward(input, registry, queue),
            ActivationType::Mish => Mish.forward(input, registry, queue),
            ActivationType::QuickGELU => QuickGELU.forward(input, registry, queue),
            ActivationType::HardSwish => HardSwish.forward(input, registry, queue),
            ActivationType::HardSigmoid => HardSigmoid.forward(input, registry, queue),
            ActivationType::Softplus => Softplus::new().forward(input, registry, queue),
            ActivationType::Softsign => Softsign.forward(input, registry, queue),
            ActivationType::GLU => GLU.forward(input, registry, queue),
        }
    }

    /// Parse from a string name (case-insensitive).
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "silu" | "swish" => Some(Self::SiLU),
            "gelu" => Some(Self::GELU),
            "gelu_fast" | "gelufast" => Some(Self::GELUFast),
            "sigmoid" => Some(Self::Sigmoid),
            "tanh" => Some(Self::Tanh),
            "relu" => Some(Self::ReLU),
            "leaky_relu" | "leakyrelu" => Some(Self::LeakyReLU),
            "elu" => Some(Self::ELU),
            "selu" => Some(Self::SELU),
            "mish" => Some(Self::Mish),
            "quick_gelu" | "quickgelu" => Some(Self::QuickGELU),
            "hard_swish" | "hardswish" => Some(Self::HardSwish),
            "hard_sigmoid" | "hardsigmoid" => Some(Self::HardSigmoid),
            "softplus" => Some(Self::Softplus),
            "softsign" => Some(Self::Softsign),
            "glu" => Some(Self::GLU),
            _ => None,
        }
    }
}
