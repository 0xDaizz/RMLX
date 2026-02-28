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
    pub fn from_arrays(
        config: LinearConfig,
        weight: Array,
        bias: Option<Array>,
    ) -> Result<Self, KernelError> {
        if weight.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "weight must be 2D [out_features, in_features], got {}D",
                weight.ndim()
            )));
        }
        if weight.shape()[0] != config.out_features {
            return Err(KernelError::InvalidShape(format!(
                "weight shape[0]={} != out_features={}",
                weight.shape()[0],
                config.out_features
            )));
        }
        if weight.shape()[1] != config.in_features {
            return Err(KernelError::InvalidShape(format!(
                "weight shape[1]={} != in_features={}",
                weight.shape()[1],
                config.in_features
            )));
        }
        if let Some(ref b) = bias {
            if b.ndim() != 1 {
                return Err(KernelError::InvalidShape(format!(
                    "bias must be 1D [out_features], got {}D",
                    b.ndim()
                )));
            }
            if b.shape()[0] != config.out_features {
                return Err(KernelError::InvalidShape(format!(
                    "bias shape[0]={} != out_features={}",
                    b.shape()[0],
                    config.out_features
                )));
            }
        }
        Ok(Self {
            config,
            weight: Some(weight),
            bias,
        })
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
            input.reshape(vec![1, input.shape()[0]])?
        } else if input.ndim() == 2 {
            input.reshape(vec![input.shape()[0], input.shape()[1]])?
        } else {
            return Err(KernelError::InvalidShape(format!(
                "input must be 1D or 2D, got {}D",
                input.ndim()
            )));
        };

        if input_2d.shape()[1] != self.config.in_features {
            return Err(KernelError::InvalidShape(format!(
                "input features mismatch: {} vs {}",
                input_2d.shape()[1],
                self.config.in_features
            )));
        }

        // Transpose weight: [out, in] -> [in, out] via stride swap (zero-copy)
        let w_t = weight.view(
            vec![self.config.in_features, self.config.out_features],
            vec![1, self.config.in_features],
            weight.offset(),
        );

        // matmul: [batch, in] @ [in, out] -> [batch, out]
        let mut output = ops::matmul::matmul(registry, &input_2d, &w_t, queue)?;

        // Add bias: broadcast bias [out_f] across each row of [batch, out_f].
        // Fused: expand bias to [batch, out_f] then single add, avoiding per-row dispatch.
        if let Some(ref bias) = self.bias {
            let batch = output.shape()[0];
            let out_f = self.config.out_features;
            let elem_size = output.dtype().size_of();

            // Build a [batch, out_f] bias tile by repeating the 1D bias for each row
            // in a single command buffer dispatch.
            let bias_tiled = Array::zeros(registry.device().raw(), &[batch, out_f], output.dtype());
            let copy_kernel = match output.dtype() {
                rmlx_core::dtype::DType::Float32 => "copy_f32",
                rmlx_core::dtype::DType::Float16 => "copy_f16",
                rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
                _ => {
                    return Err(KernelError::InvalidShape(format!(
                        "linear bias: unsupported dtype {:?}",
                        output.dtype()
                    )));
                }
            };
            let pipeline = registry.get_pipeline(copy_kernel, output.dtype())?;
            let cb = queue.new_command_buffer();
            for row in 0..batch {
                let dst_offset = row * out_f * elem_size;
                let dst_view = bias_tiled.view(vec![out_f], vec![1], dst_offset);
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(bias.metal_buffer()), bias.offset() as u64);
                enc.set_buffer(1, Some(dst_view.metal_buffer()), dst_view.offset() as u64);
                let grid = metal::MTLSize::new(out_f as u64, 1, 1);
                let tg = metal::MTLSize::new(
                    std::cmp::min(pipeline.max_total_threads_per_threadgroup(), out_f as u64),
                    1,
                    1,
                );
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();

            // Single fused add: [batch, out_f] + [batch, out_f]
            output = ops::binary::add(registry, &output, &bias_tiled, queue)?;
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
