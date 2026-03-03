//! Linear (fully connected) layer: y = x @ W^T + bias
//!
//! Supports both standard per-op dispatch and pipelined dispatch via
//! `forward_into_cb()` which encodes into an existing command buffer
//! for batched execution.

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
    weight_t_cached: Option<Array>,
}

impl Linear {
    /// Create a config-only linear layer (weights loaded later).
    pub fn new(config: LinearConfig) -> Self {
        Self {
            config,
            weight: None,
            bias: None,
            weight_t_cached: None,
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
            weight_t_cached: None,
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

    /// Get the transposed weight view for use with batched operations.
    ///
    /// Returns [in_features, out_features] view with swapped strides (zero-copy).
    pub fn weight_transposed(&self) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Linear: weights not loaded".to_string()))?;
        Ok(weight.view(
            vec![self.config.in_features, self.config.out_features],
            vec![1, self.config.in_features],
            weight.offset(),
        ))
    }

    /// Return the contiguous transposed weight (cached if available, else creates view).
    pub fn weight_transposed_contiguous(&self) -> Result<Array, KernelError> {
        if let Some(ref cached) = self.weight_t_cached {
            return Ok(cached.view(
                cached.shape().to_vec(),
                cached.strides().to_vec(),
                cached.offset(),
            ));
        }
        self.weight_transposed()
    }

    /// Pre-compute and cache the contiguous transposed weight for `forward_into_cb`.
    ///
    /// Call once after weight loading. Eliminates per-pass weight copies in the
    /// ExecGraph path. Uses one command buffer internally.
    pub fn prepare_weight_t(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("Linear: weights not loaded".to_string())
        })?;
        let w_t = weight.view(
            vec![self.config.in_features, self.config.out_features],
            vec![1, self.config.in_features],
            weight.offset(),
        );
        // Use the standard copy op (creates its own CB, blocks)
        let w_t_contig = ops::copy::copy(registry, &w_t, queue)?;
        self.weight_t_cached = Some(w_t_contig);
        Ok(())
    }

    /// Encode the linear forward pass into an existing command buffer.
    ///
    /// This does NOT commit or wait — the caller manages the CB lifecycle.
    /// Used by `CommandBatcher` and `ExecGraph` for batched execution.
    ///
    /// Note: bias is not supported in the batched path (bias-free projections
    /// are the common case for Q/K/V/O projections).
    pub fn forward_into_cb(
        &self,
        input: &Array,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Linear: weights not loaded".to_string()))?;

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

        // Use pre-cached contiguous transposed weight if available
        let w_t = if let Some(ref cached) = self.weight_t_cached {
            cached.view(cached.shape().to_vec(), cached.strides().to_vec(), cached.offset())
        } else {
            // Fallback: create transposed view and copy to contiguous in-CB
            let w_t_view = weight.view(
                vec![self.config.in_features, self.config.out_features],
                vec![1, self.config.in_features],
                weight.offset(),
            );
            if w_t_view.is_contiguous() {
                w_t_view
            } else {
                ops::copy::copy_into_cb(registry, &w_t_view, cb)?
            }
        };

        // Ensure input is contiguous for GEMM
        let input_2d = if input_2d.is_contiguous() {
            input_2d
        } else {
            ops::copy::copy_into_cb(registry, &input_2d, cb)?
        };

        // Encode GEMM into the provided CB
        let m = input_2d.shape()[0] as u32;
        let n = self.config.out_features as u32;
        let k = self.config.in_features as u32;

        let kernel_name = match input_2d.dtype() {
            rmlx_core::dtype::DType::Float32 => "gemm_tiled_f32",
            rmlx_core::dtype::DType::Float16 => "gemm_tiled_f16",
            rmlx_core::dtype::DType::Bfloat16 => "gemm_tiled_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "linear: unsupported dtype {:?}",
                    other
                )))
            }
        };

        let pipeline = registry.get_pipeline(kernel_name, input_2d.dtype())?;
        let dev = registry.device().raw();
        let output = Array::zeros(dev, &[m as usize, n as usize], input_2d.dtype());

        let m_buf = make_u32_buf(dev, m);
        let n_buf = make_u32_buf(dev, n);
        let k_buf = make_u32_buf(dev, k);
        let bsa = make_u32_buf(dev, m * k);
        let bsb = make_u32_buf(dev, k * n);
        let bsc = make_u32_buf(dev, m * n);

        const BM: u64 = 32;
        const BN: u64 = 32;
        let grid_x = (n as u64).div_ceil(BN);
        let grid_y = (m as u64).div_ceil(BM);

        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(input_2d.metal_buffer()), input_2d.offset() as u64);
        enc.set_buffer(1, Some(w_t.metal_buffer()), w_t.offset() as u64);
        enc.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa), 0);
        enc.set_buffer(7, Some(&bsb), 0);
        enc.set_buffer(8, Some(&bsc), 0);

        let grid = metal::MTLSize::new(grid_x, grid_y, 1);
        let tg = metal::MTLSize::new(BM * BN, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();

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

fn make_u32_buf(device: &metal::Device, value: u32) -> metal::Buffer {
    let data = value.to_ne_bytes();
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}
