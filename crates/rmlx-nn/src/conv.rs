//! Convolutional layers (Conv1d, Conv2d) wrapping GPU conv kernels.
//!
//! Supports config-only construction for deferred weight loading (e.g., from safetensors)
//! as well as direct construction via `from_arrays()`.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue};

// ============================================================================
// Conv1d
// ============================================================================

/// Configuration for a 1D convolution layer.
pub struct Conv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub has_bias: bool,
}

impl Conv1dConfig {
    /// Create a Conv1dConfig with common defaults (stride=1, padding=0, dilation=1, groups=1, no bias).
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        }
    }
}

/// 1D convolution layer with optional bias.
///
/// Weight shape: `[C_out, C_in/groups, kernel_size]`
/// Bias shape: `[C_out]` (if present)
///
/// Forward input shape: `[B, C_in, W]`
/// Forward output shape: `[B, C_out, W_out]`
pub struct Conv1d {
    config: Conv1dConfig,
    weight: Option<Array>,
    bias: Option<Array>,
}

impl Conv1d {
    /// Create a config-only Conv1d layer (weights loaded later via `load_weights`).
    pub fn new(config: Conv1dConfig) -> Self {
        Self {
            config,
            weight: None,
            bias: None,
        }
    }

    /// Create a Conv1d layer with pre-loaded weight and optional bias.
    ///
    /// `weight` shape: `[C_out, C_in/groups, kernel_size]`
    /// `bias` shape: `[C_out]` (if present)
    pub fn from_arrays(
        config: Conv1dConfig,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<Self, KernelError> {
        let mut layer = Self::new(config);
        layer.load_weights(weight, bias)?;
        Ok(layer)
    }

    /// Load weight and optional bias tensors into this layer.
    ///
    /// Validates shapes against the config.
    pub fn load_weights(&mut self, weight: Array, bias: Option<Array>) -> Result<(), KernelError> {
        if weight.ndim() != 3 {
            return Err(KernelError::InvalidShape(format!(
                "Conv1d: weight must be 3D [C_out, C_in/groups, K], got {}D",
                weight.ndim()
            )));
        }
        if weight.shape()[0] != self.config.out_channels {
            return Err(KernelError::InvalidShape(format!(
                "Conv1d: weight shape[0]={} != out_channels={}",
                weight.shape()[0],
                self.config.out_channels
            )));
        }
        let expected_c_in_per_group = self.config.in_channels / self.config.groups;
        if weight.shape()[1] != expected_c_in_per_group {
            return Err(KernelError::InvalidShape(format!(
                "Conv1d: weight shape[1]={} != in_channels/groups={}",
                weight.shape()[1],
                expected_c_in_per_group
            )));
        }
        if weight.shape()[2] != self.config.kernel_size {
            return Err(KernelError::InvalidShape(format!(
                "Conv1d: weight shape[2]={} != kernel_size={}",
                weight.shape()[2],
                self.config.kernel_size
            )));
        }
        if let Some(ref b) = bias {
            if b.ndim() != 1 {
                return Err(KernelError::InvalidShape(format!(
                    "Conv1d: bias must be 1D [C_out], got {}D",
                    b.ndim()
                )));
            }
            if b.shape()[0] != self.config.out_channels {
                return Err(KernelError::InvalidShape(format!(
                    "Conv1d: bias shape[0]={} != out_channels={}",
                    b.shape()[0],
                    self.config.out_channels
                )));
            }
        }
        self.weight = Some(weight);
        self.bias = bias;
        Ok(())
    }

    /// Forward pass: 1D convolution.
    ///
    /// `input` shape: `[B, C_in, W]`
    /// Returns: `[B, C_out, W_out]`
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Conv1d: weights not loaded".to_string()))?;

        ops::conv::conv1d(
            registry,
            input,
            weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.config.groups,
            queue,
        )
    }

    /// Reference to the layer configuration.
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
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

// ============================================================================
// Conv2d
// ============================================================================

/// Configuration for a 2D convolution layer.
pub struct Conv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub has_bias: bool,
}

impl Conv2dConfig {
    /// Create a Conv2dConfig with common defaults (stride=1, padding=0, dilation=1, groups=1, no bias).
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            has_bias: false,
        }
    }
}

/// 2D convolution layer with optional bias.
///
/// Weight shape: `[C_out, C_in/groups, kH, kW]`
/// Bias shape: `[C_out]` (if present)
///
/// Forward input shape: `[B, C_in, H, W]`
/// Forward output shape: `[B, C_out, H_out, W_out]`
pub struct Conv2d {
    config: Conv2dConfig,
    weight: Option<Array>,
    bias: Option<Array>,
}

impl Conv2d {
    /// Create a config-only Conv2d layer (weights loaded later via `load_weights`).
    pub fn new(config: Conv2dConfig) -> Self {
        Self {
            config,
            weight: None,
            bias: None,
        }
    }

    /// Create a Conv2d layer with pre-loaded weight and optional bias.
    ///
    /// `weight` shape: `[C_out, C_in/groups, kH, kW]`
    /// `bias` shape: `[C_out]` (if present)
    pub fn from_arrays(
        config: Conv2dConfig,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<Self, KernelError> {
        let mut layer = Self::new(config);
        layer.load_weights(weight, bias)?;
        Ok(layer)
    }

    /// Load weight and optional bias tensors into this layer.
    ///
    /// Validates shapes against the config.
    pub fn load_weights(&mut self, weight: Array, bias: Option<Array>) -> Result<(), KernelError> {
        if weight.ndim() != 4 {
            return Err(KernelError::InvalidShape(format!(
                "Conv2d: weight must be 4D [C_out, C_in/groups, kH, kW], got {}D",
                weight.ndim()
            )));
        }
        if weight.shape()[0] != self.config.out_channels {
            return Err(KernelError::InvalidShape(format!(
                "Conv2d: weight shape[0]={} != out_channels={}",
                weight.shape()[0],
                self.config.out_channels
            )));
        }
        let expected_c_in_per_group = self.config.in_channels / self.config.groups;
        if weight.shape()[1] != expected_c_in_per_group {
            return Err(KernelError::InvalidShape(format!(
                "Conv2d: weight shape[1]={} != in_channels/groups={}",
                weight.shape()[1],
                expected_c_in_per_group
            )));
        }
        if weight.shape()[2] != self.config.kernel_size.0 {
            return Err(KernelError::InvalidShape(format!(
                "Conv2d: weight shape[2]={} != kernel_size.0={}",
                weight.shape()[2],
                self.config.kernel_size.0
            )));
        }
        if weight.shape()[3] != self.config.kernel_size.1 {
            return Err(KernelError::InvalidShape(format!(
                "Conv2d: weight shape[3]={} != kernel_size.1={}",
                weight.shape()[3],
                self.config.kernel_size.1
            )));
        }
        if let Some(ref b) = bias {
            if b.ndim() != 1 {
                return Err(KernelError::InvalidShape(format!(
                    "Conv2d: bias must be 1D [C_out], got {}D",
                    b.ndim()
                )));
            }
            if b.shape()[0] != self.config.out_channels {
                return Err(KernelError::InvalidShape(format!(
                    "Conv2d: bias shape[0]={} != out_channels={}",
                    b.shape()[0],
                    self.config.out_channels
                )));
            }
        }
        self.weight = Some(weight);
        self.bias = bias;
        Ok(())
    }

    /// Forward pass: 2D convolution.
    ///
    /// `input` shape: `[B, C_in, H, W]`
    /// Returns: `[B, C_out, H_out, W_out]`
    pub fn forward(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Conv2d: weights not loaded".to_string()))?;

        ops::conv::conv2d(
            registry,
            input,
            weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.config.groups,
            queue,
        )
    }

    /// Reference to the layer configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
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
