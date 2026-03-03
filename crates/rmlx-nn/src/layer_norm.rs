//! LayerNorm neural network module (N9).
//!
//! Wraps `rmlx_core::ops::layer_norm` as an nn module with learnable
//! weight and bias parameters.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

/// Layer Normalization configuration.
pub struct LayerNormConfig {
    /// Number of features (last dimension of input).
    pub normalized_shape: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
    /// Whether to learn an affine scale (weight) parameter.
    pub elementwise_affine: bool,
    /// Whether to learn a bias parameter.
    pub has_bias: bool,
}

impl LayerNormConfig {
    /// Create a LayerNormConfig with common defaults (eps=1e-5, affine=true, bias=true).
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
            has_bias: true,
        }
    }
}

/// Layer Normalization module.
///
/// Applies `y = (x - mean) / sqrt(var + eps) * weight + bias` per row.
///
/// Weight shape: `[normalized_shape]`
/// Bias shape: `[normalized_shape]` (if present)
///
/// Forward input shape: `[*, normalized_shape]` (2D: `[batch, features]`)
/// Forward output shape: same as input
pub struct LayerNorm {
    config: LayerNormConfig,
    weight: Option<Array>,
    bias: Option<Array>,
}

impl LayerNorm {
    /// Create a config-only LayerNorm (weights loaded later).
    pub fn new(config: LayerNormConfig) -> Self {
        Self {
            config,
            weight: None,
            bias: None,
        }
    }

    /// Create a LayerNorm with pre-loaded weight and optional bias.
    pub fn from_arrays(
        config: LayerNormConfig,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<Self, KernelError> {
        // Validate weight shape
        if weight.ndim() != 1 || weight.shape()[0] != config.normalized_shape {
            return Err(KernelError::InvalidShape(format!(
                "LayerNorm: weight must be [{}], got {:?}",
                config.normalized_shape,
                weight.shape()
            )));
        }

        // Validate bias shape
        if let Some(ref b) = bias {
            if b.ndim() != 1 || b.shape()[0] != config.normalized_shape {
                return Err(KernelError::InvalidShape(format!(
                    "LayerNorm: bias must be [{}], got {:?}",
                    config.normalized_shape,
                    b.shape()
                )));
            }
        }

        Ok(Self {
            config,
            weight: Some(weight),
            bias,
        })
    }

    /// Forward pass: apply layer normalization.
    ///
    /// `input` shape: `[batch, normalized_shape]`
    /// Returns: same shape as input
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        if input.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "LayerNorm: input must be 2D [batch, features], got {}D",
                input.ndim()
            )));
        }

        if input.shape()[1] != self.config.normalized_shape {
            return Err(KernelError::InvalidShape(format!(
                "LayerNorm: input features {} != normalized_shape {}",
                input.shape()[1],
                self.config.normalized_shape
            )));
        }

        let weight_ref =
            if self.config.elementwise_affine {
                Some(self.weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape("LayerNorm: weight not loaded".into())
                })?)
            } else {
                None
            };

        let bias_ref = if self.config.has_bias {
            self.bias.as_ref()
        } else {
            None
        };

        ops::layer_norm::layer_norm(
            registry,
            input,
            weight_ref,
            bias_ref,
            self.config.eps,
            queue,
        )
    }

    /// Reference to the layer configuration.
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }

    /// Normalized shape (last dimension size).
    pub fn normalized_shape(&self) -> usize {
        self.config.normalized_shape
    }

    /// Epsilon value for numerical stability.
    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    /// Whether weights have been loaded.
    pub fn has_weights(&self) -> bool {
        self.weight.is_some()
    }

    /// Reference to weight array, if loaded.
    pub fn weight(&self) -> Option<&Array> {
        self.weight.as_ref()
    }

    /// Reference to bias array, if loaded.
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }
}
