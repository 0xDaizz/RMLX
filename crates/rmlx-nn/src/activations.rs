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

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

/// Trait for activation function modules.
///
/// All activations take an input array and return an output of the same shape.
pub trait Activation {
    /// Apply the activation function element-wise.
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError>;

    /// Human-readable name for debugging/logging.
    fn name(&self) -> &'static str;
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
///
/// Also known as Swish-1. Used in LLaMA, Mistral, and most modern LLMs.
pub struct SiLU;

impl Activation for SiLU {
    fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        ops::unary::tanh_op(registry, input, queue)
    }

    fn name(&self) -> &'static str {
        "Tanh"
    }
}

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
}

impl ActivationType {
    /// Apply the activation function element-wise (dynamic dispatch).
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        match self {
            ActivationType::SiLU => ops::silu::silu(registry, input, queue),
            ActivationType::GELU => ops::gelu::gelu(registry, input, queue),
            ActivationType::GELUFast => ops::gelu::gelu_fast(registry, input, queue),
            ActivationType::Sigmoid => ops::unary::sigmoid(registry, input, queue),
            ActivationType::Tanh => ops::unary::tanh_op(registry, input, queue),
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
            _ => None,
        }
    }
}
