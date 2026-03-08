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
    /// Pre-computed contiguous transposed weight for the ExecGraph path.
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

    /// Convert all weight buffers to `StorageModePrivate` (GPU-only).
    ///
    /// Call after loading weights and before the inference loop. Weights
    /// become inaccessible to the CPU but may benefit from reduced memory
    /// pressure and faster GPU access.
    pub fn convert_weights_private(&mut self, device: &metal::Device, queue: &metal::CommandQueue) {
        if let Some(w) = self.weight.take() {
            self.weight = Some(w.to_private(device, queue));
        }
        if let Some(b) = self.bias.take() {
            self.bias = Some(b.to_private(device, queue));
        }
        if let Some(wt) = self.weight_t_cached.take() {
            self.weight_t_cached = Some(wt.to_private(device, queue));
        }
    }

    // -------------------------------------------------------------------
    // ExecGraph path — encode into existing command buffers
    // -------------------------------------------------------------------

    /// Return a (non-contiguous) transposed view of the weight matrix.
    ///
    /// Weight shape: `[out_features, in_features]` -> `[in_features, out_features]`
    /// via stride swap (zero-copy).
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

    /// Pre-compute and cache the contiguous transposed weight for `forward_into_cb`.
    ///
    /// Call once after weight loading. Eliminates per-pass weight copies in the
    /// ExecGraph path.
    pub fn prepare_weight_t(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Linear: weights not loaded".to_string()))?;
        let w_t = weight.view(
            vec![self.config.in_features, self.config.out_features],
            vec![1, self.config.in_features],
            weight.offset(),
        );
        let w_t_contig = ops::copy::copy(registry, &w_t, queue)?;
        self.weight_t_cached = Some(w_t_contig);
        Ok(())
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

    /// Encode the linear forward pass into an existing command buffer.
    ///
    /// This does NOT commit or wait — the caller manages the CB lifecycle.
    /// Note: bias is not supported in the batched path.
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
            cached.view(
                cached.shape().to_vec(),
                cached.strides().to_vec(),
                cached.offset(),
            )
        } else {
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
        let m = input_2d.shape()[0];
        let n = self.config.out_features;
        let k = self.config.in_features;

        // GEMV fast path for M=1 (single-token decode)
        // Use gemv on original weight [N, K] directly — no transposition needed.
        // gemv computes y[i] = sum_j W[i,j] * x[j], output is [N].
        // Row-major reads on W give better memory access (float4 vectorized loads).
        if m == 1 {
            let input_vec = Array::new(
                input_2d.metal_buffer().to_owned(),
                vec![k],
                vec![1],
                input_2d.dtype(),
                input_2d.offset(),
            );
            let weight = self.weight.as_ref().ok_or_else(|| {
                KernelError::InvalidShape("Linear: weights not loaded".to_string())
            })?;
            // Ensure weight is contiguous for gemv
            let weight_ref = if weight.is_contiguous() {
                weight.view(
                    weight.shape().to_vec(),
                    weight.strides().to_vec(),
                    weight.offset(),
                )
            } else {
                ops::copy::copy_into_cb(registry, weight, cb)?
            };
            let result = ops::gemv::gemv_into_cb(registry, &weight_ref, &input_vec, cb)?;
            // Reshape [N] to [1, N]
            return Ok(Array::new(
                result.metal_buffer().to_owned(),
                vec![1, n],
                vec![n, 1],
                result.dtype(),
                result.offset(),
            ));
        }

        let tile = ops::matmul::select_tile_config_with_dtype(m, n, k, input_2d.dtype());
        let kernel_name = match (tile.variant, input_2d.dtype()) {
            (ops::matmul::TileVariant::Simd, rmlx_core::dtype::DType::Float32)
            | (ops::matmul::TileVariant::Medium, rmlx_core::dtype::DType::Float32) => {
                "gemm_simd_f32"
            }
            (ops::matmul::TileVariant::Simd, rmlx_core::dtype::DType::Float16)
            | (ops::matmul::TileVariant::Medium, rmlx_core::dtype::DType::Float16) => {
                "gemm_simd_f16"
            }
            (ops::matmul::TileVariant::Simd, rmlx_core::dtype::DType::Bfloat16)
            | (ops::matmul::TileVariant::Medium, rmlx_core::dtype::DType::Bfloat16) => {
                "gemm_simd_bf16"
            }
            (ops::matmul::TileVariant::Small, rmlx_core::dtype::DType::Float32) => "gemm_small_f32",
            (ops::matmul::TileVariant::Small, rmlx_core::dtype::DType::Float16) => "gemm_small_f16",
            (ops::matmul::TileVariant::Small, rmlx_core::dtype::DType::Bfloat16) => {
                "gemm_small_bf16"
            }
            (ops::matmul::TileVariant::Skinny, rmlx_core::dtype::DType::Float32) => {
                "gemm_skinny_f32"
            }
            (ops::matmul::TileVariant::Skinny, rmlx_core::dtype::DType::Float16) => {
                "gemm_skinny_f16"
            }
            (ops::matmul::TileVariant::Skinny, rmlx_core::dtype::DType::Bfloat16) => {
                "gemm_skinny_bf16"
            }
            (ops::matmul::TileVariant::Full, rmlx_core::dtype::DType::Float32) => "gemm_tiled_f32",
            (ops::matmul::TileVariant::Full, rmlx_core::dtype::DType::Float16) => "gemm_tiled_f16",
            (ops::matmul::TileVariant::Full, rmlx_core::dtype::DType::Bfloat16) => {
                "gemm_tiled_bf16"
            }
            (ops::matmul::TileVariant::MlxArch, rmlx_core::dtype::DType::Float16) => "gemm_mlx_f16",
            (_, other) => {
                return Err(KernelError::InvalidShape(format!(
                    "linear: unsupported dtype {:?}",
                    other
                )))
            }
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        // MlxArch uses function constants for alignment specialization
        let pipeline = if tile.variant == ops::matmul::TileVariant::MlxArch {
            let constants = ops::matmul::matmul_align_constants(m, n, tile.bm, tile.bn);
            registry.get_pipeline_with_constants(kernel_name, input_2d.dtype(), &constants)?
        } else {
            registry.get_pipeline(kernel_name, input_2d.dtype())?
        };
        let dev = registry.device().raw();
        let output = Array::zeros(dev, &[m, n], input_2d.dtype());

        let m_buf = make_u32_buf(dev, m_u32);
        let n_buf = make_u32_buf(dev, n_u32);
        let k_buf = make_u32_buf(dev, k_u32);
        let bsa = make_u32_buf(dev, m_u32 * k_u32);
        let bsb = make_u32_buf(dev, k_u32 * n_u32);
        let bsc = make_u32_buf(dev, m_u32 * n_u32);

        let bm = tile.bm as u64;
        let bn = tile.bn as u64;
        let grid_x = (n_u32 as u64).div_ceil(bn);
        let grid_y = (m_u32 as u64).div_ceil(bm);

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

        // Steel, Full/Skinny/MlxArch kernels require swizzle_log (buffer 9)
        let swizzle_log_buf = if matches!(
            tile.variant,
            ops::matmul::TileVariant::Full
                | ops::matmul::TileVariant::Skinny
                | ops::matmul::TileVariant::MlxArch
        ) {
            let swizzle_log = ops::matmul::compute_swizzle_log(m, tile.bm);
            let buf = make_u32_buf(dev, swizzle_log);
            enc.set_buffer(9, Some(&buf), 0);
            Some(buf)
        } else {
            None
        };

        let tg_threads = match tile.variant {
            ops::matmul::TileVariant::Small => 256_u64,
            ops::matmul::TileVariant::Medium | ops::matmul::TileVariant::Simd => 1024_u64,
            ops::matmul::TileVariant::Skinny | ops::matmul::TileVariant::Full => 256_u64,
            ops::matmul::TileVariant::MlxArch => 64_u64,
        };

        let grid = metal::MTLSize::new(grid_x, grid_y, 1);
        let tg = metal::MTLSize::new(tg_threads, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        drop(swizzle_log_buf);

        Ok(output)
    }
}

/// Create a constant `u32` Metal buffer.
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}
