//! Linear (fully connected) layer: y = x @ W^T + bias

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

/// Linear layer configuration.
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
}

/// Linear layer with optional bias.
///
/// Supports two modes:
/// 1. Config-only: for deferred weight loading (e.g., from safetensors)
/// 2. With arrays: weight and optional bias loaded via `from_arrays()`
pub struct Linear {
    config: LinearConfig,
    weight: Option<Array>,
    bias: Option<Array>,
}

impl Linear {
    /// Create a config-only linear layer (weights loaded later).
    pub fn new(config: LinearConfig) -> Self {
        Self {
            config,
            weight: None,
            bias: None,
        }
    }

    /// Create a linear layer with pre-loaded weight and optional bias.
    ///
    /// `weight` shape: [out_features, in_features]
    /// `bias` shape: [out_features] (if present)
    pub fn from_arrays(config: LinearConfig, weight: Array, bias: Option<Array>) -> Self {
        assert_eq!(
            weight.ndim(),
            2,
            "weight must be 2D [out_features, in_features]"
        );
        assert_eq!(weight.shape()[0], config.out_features);
        assert_eq!(weight.shape()[1], config.in_features);
        if let Some(ref b) = bias {
            assert_eq!(b.ndim(), 1, "bias must be 1D [out_features]");
            assert_eq!(b.shape()[0], config.out_features);
        }
        Self {
            config,
            weight: Some(weight),
            bias,
        }
    }

    /// Forward pass: output = input @ weight^T + bias
    ///
    /// `input` shape: [batch, in_features] or [in_features] (treated as [1, in_features])
    /// Returns: [batch, out_features]
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Linear: weights not loaded".to_string()))?;

        // Ensure input is 2D
        let input_2d = if input.ndim() == 1 {
            input.reshape(vec![1, input.shape()[0]])
        } else {
            assert_eq!(input.ndim(), 2, "input must be 1D or 2D");
            input.reshape(vec![input.shape()[0], input.shape()[1]])
        };

        assert_eq!(
            input_2d.shape()[1],
            self.config.in_features,
            "input features mismatch: {} vs {}",
            input_2d.shape()[1],
            self.config.in_features
        );

        // Transpose weight: [out, in] -> [in, out]
        // For now, use a view with swapped strides
        let w_t = weight.view(
            vec![self.config.in_features, self.config.out_features],
            vec![1, self.config.in_features],
            weight.offset(),
        );

        // matmul: [batch, in] @ [in, out] -> [batch, out]
        let mut output = ops::matmul::matmul(registry, &input_2d, &w_t, queue)?;

        // Add bias if present (batch=1 only; multi-batch broadcast not yet supported)
        if let Some(ref bias) = self.bias {
            let batch = output.shape()[0];
            if batch == 1 {
                let reshaped = output.reshape(vec![self.config.out_features]);
                let added = ops::binary::add(registry, &reshaped, bias, queue)?;
                output = added.reshape(vec![1, self.config.out_features]);
            }
        }

        Ok(output)
    }

    pub fn in_features(&self) -> usize {
        self.config.in_features
    }

    pub fn out_features(&self) -> usize {
        self.config.out_features
    }

    pub fn has_bias(&self) -> bool {
        self.config.has_bias
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
