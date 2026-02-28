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

        // Add bias to each row
        if let Some(ref bias) = self.bias {
            let batch = output.shape()[0];
            let out_f = self.config.out_features;
            let elem_size = output.dtype().size_of();
            let mut rows: Vec<Array> = Vec::with_capacity(batch);
            for row in 0..batch {
                let row_offset = output.offset() + row * out_f * elem_size;
                let row_view = output.view(vec![out_f], vec![1], row_offset);
                let added = ops::binary::add(registry, &row_view, bias, queue)?;
                rows.push(added);
            }
            // Build final output by copying rows into a single contiguous buffer
            let final_out = Array::zeros(registry.device().raw(), &[batch, out_f], output.dtype());
            for (row, src) in rows.iter().enumerate() {
                let dst_offset = row * out_f * elem_size;
                let dst_view = final_out.view(vec![out_f], vec![1], dst_offset);
                // Use copy kernel to write src into dst_view's region
                // Since both are contiguous 1D with same shape, we dispatch a flat copy
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                let copy_kernel = match output.dtype() {
                    rmlx_core::dtype::DType::Float32 => "copy_f32",
                    rmlx_core::dtype::DType::Float16 => "copy_f16",
                    rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
                    _ => unreachable!("linear output cannot be quantized"),
                };
                let pipeline = registry.get_pipeline(copy_kernel, output.dtype())?;
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
                enc.set_buffer(1, Some(dst_view.metal_buffer()), dst_view.offset() as u64);
                let grid = metal::MTLSize::new(out_f as u64, 1, 1);
                let tg = metal::MTLSize::new(
                    std::cmp::min(pipeline.max_total_threads_per_threadgroup(), out_f as u64),
                    1,
                    1,
                );
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            output = final_out;
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
