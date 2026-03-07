//! Transformer block: attention + MLP (or MoE).

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::ExecGraph;

use crate::attention::{Attention, LayerKvCache};
use crate::embedding::Embedding;
use crate::linear::Linear;
use crate::moe::MoeLayer;

/// Cast a BufferRef to a ResourceRef for use with memory_barrier_with_resources.
/// Safe because MTLBuffer inherits from MTLResource in ObjC.
fn buf_as_resource(buf: &metal::BufferRef) -> &metal::ResourceRef {
    unsafe { &*(buf as *const metal::BufferRef as *const metal::ResourceRef) }
}

pub enum FeedForwardType {
    /// Simple dense FFN: linear1 -> activation -> linear2
    Dense {
        intermediate_dim: usize,
        activation: crate::activations::ActivationType,
    },
    /// Gated FFN (SwiGLU): gate_proj * up_proj -> down_proj (Llama-style)
    Gated { intermediate_dim: usize },
    /// Mixture of Experts
    MoE { config: super::moe::MoeConfig },
}

pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub ff_type: FeedForwardType,
}

impl TransformerConfig {
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.hidden_size == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: hidden_size must be > 0".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: num_heads must be > 0".into(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: num_kv_heads must be > 0".into(),
            ));
        }
        if self.num_kv_heads > self.num_heads {
            return Err(KernelError::InvalidShape(format!(
                "TransformerConfig: num_kv_heads ({}) > num_heads ({})",
                self.num_kv_heads, self.num_heads
            )));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(KernelError::InvalidShape(format!(
                "TransformerConfig: num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.head_dim == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: head_dim must be > 0".into(),
            ));
        }
        if self.num_layers == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: num_layers must be > 0".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(KernelError::InvalidShape(
                "TransformerConfig: vocab_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Feed-forward network: dense, gated (SwiGLU), or MoE.
#[allow(clippy::large_enum_variant)]
pub enum FeedForward {
    /// Simple dense FFN: linear1 -> activation -> linear2
    Dense {
        linear1: Linear,
        linear2: Linear,
        activation: crate::activations::ActivationType,
    },
    /// Gated FFN (SwiGLU): silu(gate(x)) * up(x) -> down(x)
    Gated {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
        /// Merged gate+up weight [gate_dim + up_dim, in_features] for 9-dispatch path.
        gate_up_merged_weight: Option<Array>,
    },
    /// Mixture of Experts
    MoE(MoeLayer),
}

impl FeedForward {
    /// Forward pass for the FFN.
    ///
    /// `x`: [seq_len, hidden_size]
    /// Returns: [seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Dense {
                linear1,
                linear2,
                activation,
            } => linear2.forward(
                &activation.forward(&linear1.forward(x, registry, queue)?, registry, queue)?,
                registry,
                queue,
            ),
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                // SwiGLU: down(silu(gate(x)) * up(x))
                let gate_out = gate_proj.forward(x, registry, queue)?;
                let up_out = up_proj.forward(x, registry, queue)?;
                let gate_activated = ops::silu::silu(registry, &gate_out, queue)?;
                let hidden = ops::binary::mul(registry, &gate_activated, &up_out, queue)?;
                down_proj.forward(&hidden, registry, queue)
            }
            FeedForward::MoE(moe) => moe.forward(x, registry, queue),
        }
    }

    // -------------------------------------------------------------------
    // ExecGraph path
    // -------------------------------------------------------------------

    /// Pre-cache transposed weights for all dense FFN projections.
    ///
    /// No-op for MoE layers (expert weights are not pre-transposed).
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        match self {
            FeedForward::Dense {
                linear1, linear2, ..
            } => {
                linear1.prepare_weight_t(registry, queue)?;
                linear2.prepare_weight_t(registry, queue)?;
                Ok(())
            }
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                gate_proj.prepare_weight_t(registry, queue)?;
                up_proj.prepare_weight_t(registry, queue)?;
                down_proj.prepare_weight_t(registry, queue)?;
                Ok(())
            }
            FeedForward::MoE(_) => Ok(()),
        }
    }

    /// ExecGraph-based FFN forward using 2 command buffers.
    ///
    /// For gated SwiGLU:
    /// - CB5 (current): gate + up + fused silu*mul
    /// - CB6: down_proj + residual add
    ///
    /// For dense and MoE: falls back to sync path (graph sync + reset).
    pub fn forward_graph(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Dense { .. } => {
                graph
                    .sync()
                    .map_err(|e| KernelError::InvalidShape(format!("Dense graph sync: {e}")))?;
                graph.reset();
                let ffn_out = self.forward(normed, registry, queue)?;
                ops::binary::add(registry, residual, &ffn_out, queue)
            }
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                // CB5 (current): gate + up + fused silu*mul
                let cb5 = graph.command_buffer();
                let gate_out = gate_proj.forward_into_cb(normed, registry, cb5)?;
                let up_out = up_proj.forward_into_cb(normed, registry, cb5)?;
                let hidden = ops::fused::fused_silu_mul_into_cb(registry, &gate_out, &up_out, cb5)?;
                let t5 = graph.submit_batch();

                // CB6: down_proj + residual
                graph.wait_for(t5);
                let cb6 = graph.command_buffer();
                let ffn_out = down_proj.forward_into_cb(&hidden, registry, cb6)?;
                ops::binary::add_into_cb(registry, residual, &ffn_out, cb6)
            }
            FeedForward::MoE(moe) => {
                // MoE: sync, reset, run synchronously
                graph
                    .sync()
                    .map_err(|e| KernelError::InvalidShape(format!("MoE graph sync: {e}")))?;
                graph.reset();
                let ffn_out = moe.forward(normed, registry, queue)?;
                ops::binary::add(registry, residual, &ffn_out, queue)
            }
        }
    }

    /// Single-CB FFN forward: entire SwiGLU + residual in a provided CB.
    ///
    /// Does NOT commit or wait — the caller manages the CB lifecycle.
    pub fn forward_single_cb(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                let gate_out = gate_proj.forward_into_cb(normed, registry, cb)?;
                let up_out = up_proj.forward_into_cb(normed, registry, cb)?;
                let hidden = ops::fused::fused_silu_mul_into_cb(registry, &gate_out, &up_out, cb)?;
                let ffn_out = down_proj.forward_into_cb(&hidden, registry, cb)?;
                ops::binary::add_into_cb(registry, residual, &ffn_out, cb)
            }
            _ => Err(KernelError::InvalidShape(
                "forward_single_cb only supports Gated FFN".into(),
            )),
        }
    }

    /// Fused FFN: entire SwiGLU in 1 CB (gate + up + silu_mul + down + residual).
    pub fn forward_graph_fused(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Dense { .. } => {
                graph
                    .sync()
                    .map_err(|e| KernelError::InvalidShape(format!("Dense graph sync: {e}")))?;
                graph.reset();
                let ffn_out = self.forward(normed, registry, queue)?;
                ops::binary::add(registry, residual, &ffn_out, queue)
            }
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                let cb = graph.command_buffer();
                let gate_out = gate_proj.forward_into_cb(normed, registry, cb)?;
                let up_out = up_proj.forward_into_cb(normed, registry, cb)?;
                let hidden = ops::fused::fused_silu_mul_into_cb(registry, &gate_out, &up_out, cb)?;
                let ffn_out = down_proj.forward_into_cb(&hidden, registry, cb)?;
                ops::binary::add_into_cb(registry, residual, &ffn_out, cb)
            }
            FeedForward::MoE(moe) => {
                graph
                    .sync()
                    .map_err(|e| KernelError::InvalidShape(format!("MoE graph sync: {e}")))?;
                graph.reset();
                let ffn_out = moe.forward(normed, registry, queue)?;
                ops::binary::add(registry, residual, &ffn_out, queue)
            }
        }
    }

    // -------------------------------------------------------------------
    // 9-dispatch path: merged gate+up weight + fused residual
    // -------------------------------------------------------------------

    /// Merge gate and up projection weights into a single [gate_dim+up_dim, in_features] matrix.
    ///
    /// Must be called once after weights are loaded and before `forward_single_cb_9dispatch`.
    pub fn prepare_merged_gate_up(&mut self, device: &metal::Device) -> Result<(), KernelError> {
        match self {
            FeedForward::Gated {
                gate_proj,
                up_proj,
                gate_up_merged_weight,
                ..
            } => {
                let gate_w = gate_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "prepare_merged_gate_up: gate_proj weight not loaded".into(),
                    )
                })?;
                let up_w = up_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "prepare_merged_gate_up: up_proj weight not loaded".into(),
                    )
                })?;

                // Validate contiguity
                if !gate_w.is_contiguous() || !up_w.is_contiguous() {
                    return Err(KernelError::InvalidShape(
                        "prepare_merged_gate_up: all weights must be contiguous".into(),
                    ));
                }
                // Validate matching dtype
                if gate_w.dtype() != up_w.dtype() {
                    return Err(KernelError::InvalidShape(format!(
                        "prepare_merged_gate_up: dtype mismatch: gate={:?}, up={:?}",
                        gate_w.dtype(),
                        up_w.dtype()
                    )));
                }
                // Validate matching cols (in_features)
                if gate_w.shape()[1] != up_w.shape()[1] {
                    return Err(KernelError::InvalidShape(format!(
                        "prepare_merged_gate_up: in_features mismatch: gate={}, up={}",
                        gate_w.shape()[1],
                        up_w.shape()[1]
                    )));
                }
                // Validate matching rows (out_features)
                if gate_w.shape()[0] != up_w.shape()[0] {
                    return Err(KernelError::InvalidShape(format!(
                        "prepare_merged_gate_up: gate rows ({}) != up rows ({})",
                        gate_w.shape()[0],
                        up_w.shape()[0]
                    )));
                }

                let gate_rows = gate_w.shape()[0];
                let up_rows = up_w.shape()[0];
                let cols = gate_w.shape()[1];
                let total_rows = gate_rows + up_rows;
                let elem_size = gate_w.dtype().size_of();
                let total_bytes = total_rows * cols * elem_size;

                let buf = device.new_buffer(
                    total_bytes as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                unsafe {
                    let dst = buf.contents() as *mut u8;
                    let gate_src =
                        (gate_w.metal_buffer().contents() as *const u8).add(gate_w.offset());
                    std::ptr::copy_nonoverlapping(gate_src, dst, gate_rows * cols * elem_size);
                    let up_src = (up_w.metal_buffer().contents() as *const u8).add(up_w.offset());
                    std::ptr::copy_nonoverlapping(
                        up_src,
                        dst.add(gate_rows * cols * elem_size),
                        up_rows * cols * elem_size,
                    );
                }

                *gate_up_merged_weight = Some(Array::new(
                    buf,
                    vec![total_rows, cols],
                    vec![cols, 1],
                    gate_w.dtype(),
                    0,
                ));
                Ok(())
            }
            _ => Err(KernelError::InvalidShape(
                "prepare_merged_gate_up: only supported for Gated FFN".into(),
            )),
        }
    }

    /// Convert all static weights to `StorageModePrivate` (GPU-only).
    ///
    /// Call after loading weights and before the inference loop.
    pub fn prepare_weights_private(&mut self, device: &metal::Device, queue: &metal::CommandQueue) {
        match self {
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                gate_up_merged_weight,
            } => {
                gate_proj.convert_weights_private(device, queue);
                up_proj.convert_weights_private(device, queue);
                down_proj.convert_weights_private(device, queue);
                if let Some(w) = gate_up_merged_weight.take() {
                    *gate_up_merged_weight = Some(w.to_private(device, queue));
                }
            }
            FeedForward::Dense {
                linear1, linear2, ..
            } => {
                linear1.convert_weights_private(device, queue);
                linear2.convert_weights_private(device, queue);
            }
            FeedForward::MoE(_) => {
                // MoE expert weights are managed separately
            }
        }
    }

    /// 9-dispatch FFN forward: dispatches 7-9 of the 9-dispatch path.
    ///
    /// Dispatches:
    ///   7. merged gate+up gemv
    ///   8. silu_mul
    ///   9. gemv_bias(W_down, hidden, residual) — down_proj + residual fused
    pub fn forward_single_cb_9dispatch(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                // Guard: decode path requires seq_len=1
                let normed_seq = if normed.ndim() >= 2 {
                    normed.shape()[0]
                } else {
                    1
                };
                if normed_seq != 1 {
                    return Err(KernelError::InvalidShape(format!(
                        "forward_single_cb_9dispatch: requires seq_len=1, got {}",
                        normed_seq
                    )));
                }

                let hidden_size = if normed.ndim() == 2 {
                    normed.shape()[1]
                } else {
                    normed.shape()[0]
                };
                let normed_vec = Array::new(
                    normed.metal_buffer().to_owned(),
                    vec![hidden_size],
                    vec![1],
                    normed.dtype(),
                    normed.offset(),
                );

                // --- Single encoder for dispatches 7-9 (gate+up, silu_mul, down+residual) ---
                let encoder = cb.new_compute_command_encoder();

                // Dispatch 7: merged gate+up gemv
                let gate_up_w = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_single_cb_9dispatch: call prepare_merged_gate_up() first".into(),
                    )
                })?;
                let gate_up =
                    ops::gemv::gemv_into_encoder(registry, gate_up_w, &normed_vec, encoder)?;
                // gate_up is [gate_dim + up_dim] flat
                let gate_dim = gate_up_w.shape()[0] / 2;
                let elem_size = gate_up.dtype().size_of();

                // Memory barrier: ensure gate_up is visible to dispatch 8
                encoder.memory_barrier_with_resources(&[buf_as_resource(gate_up.metal_buffer())]);

                // Split into gate and up views
                let gate = Array::new(
                    gate_up.metal_buffer().to_owned(),
                    vec![1, gate_dim],
                    vec![gate_dim, 1],
                    gate_up.dtype(),
                    gate_up.offset(),
                );
                let up = Array::new(
                    gate_up.metal_buffer().to_owned(),
                    vec![1, gate_dim],
                    vec![gate_dim, 1],
                    gate_up.dtype(),
                    gate_up.offset() + gate_dim * elem_size,
                );

                // Dispatch 8: silu_mul
                let hidden =
                    ops::fused::fused_silu_mul_into_encoder(registry, &gate, &up, encoder)?;

                // Memory barrier: ensure hidden is visible to dispatch 9
                encoder.memory_barrier_with_resources(&[buf_as_resource(hidden.metal_buffer())]);

                // Dispatch 9: gemv_bias(W_down, hidden, residual) — down + residual fused
                let down_w = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("9dispatch: down_proj weight not loaded".into())
                })?;
                let hidden_vec = Array::new(
                    hidden.metal_buffer().to_owned(),
                    vec![hidden.numel()],
                    vec![1],
                    hidden.dtype(),
                    hidden.offset(),
                );
                let res_vec = Array::new(
                    residual.metal_buffer().to_owned(),
                    vec![hidden_size],
                    vec![1],
                    residual.dtype(),
                    residual.offset(),
                );
                let out = ops::gemv::gemv_bias_into_encoder(
                    registry,
                    down_w,
                    &hidden_vec,
                    &res_vec,
                    encoder,
                )?;

                encoder.end_encoding();

                Ok(Array::new(
                    out.metal_buffer().to_owned(),
                    vec![1, hidden_size],
                    vec![hidden_size, 1],
                    out.dtype(),
                    out.offset(),
                ))
            }
            _ => Err(KernelError::InvalidShape(
                "forward_single_cb_9dispatch only supports Gated FFN".into(),
            )),
        }
    }

    /// 9-dispatch FFN forward using concurrent encoder.
    pub fn forward_concurrent_9dispatch(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                // Guard: decode path requires seq_len=1
                let normed_seq = if normed.ndim() >= 2 {
                    normed.shape()[0]
                } else {
                    1
                };
                if normed_seq != 1 {
                    return Err(KernelError::InvalidShape(format!(
                        "forward_concurrent_9dispatch: requires seq_len=1, got {}",
                        normed_seq
                    )));
                }

                let hidden_size = if normed.ndim() == 2 {
                    normed.shape()[1]
                } else {
                    normed.shape()[0]
                };
                let normed_vec = Array::new(
                    normed.metal_buffer().to_owned(),
                    vec![hidden_size],
                    vec![1],
                    normed.dtype(),
                    normed.offset(),
                );

                // --- Single encoder for dispatches 7-9 (gate+up, silu_mul, down+residual) ---
                let encoder = rmlx_metal::new_concurrent_encoder(cb);

                // Dispatch 7: merged gate+up gemv
                let gate_up_w = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_single_cb_9dispatch: call prepare_merged_gate_up() first".into(),
                    )
                })?;
                let gate_up =
                    ops::gemv::gemv_into_encoder(registry, gate_up_w, &normed_vec, encoder)?;
                // gate_up is [gate_dim + up_dim] flat
                let gate_dim = gate_up_w.shape()[0] / 2;
                let elem_size = gate_up.dtype().size_of();

                // Memory barrier: ensure gate_up is visible to dispatch 8
                rmlx_metal::memory_barrier_scope_buffers(encoder);

                // Split into gate and up views
                let gate = Array::new(
                    gate_up.metal_buffer().to_owned(),
                    vec![1, gate_dim],
                    vec![gate_dim, 1],
                    gate_up.dtype(),
                    gate_up.offset(),
                );
                let up = Array::new(
                    gate_up.metal_buffer().to_owned(),
                    vec![1, gate_dim],
                    vec![gate_dim, 1],
                    gate_up.dtype(),
                    gate_up.offset() + gate_dim * elem_size,
                );

                // Dispatch 8: silu_mul
                let hidden =
                    ops::fused::fused_silu_mul_into_encoder(registry, &gate, &up, encoder)?;

                // Memory barrier: ensure hidden is visible to dispatch 9
                rmlx_metal::memory_barrier_scope_buffers(encoder);

                // Dispatch 9: gemv_bias(W_down, hidden, residual) — down + residual fused
                let down_w = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("9dispatch: down_proj weight not loaded".into())
                })?;
                let hidden_vec = Array::new(
                    hidden.metal_buffer().to_owned(),
                    vec![hidden.numel()],
                    vec![1],
                    hidden.dtype(),
                    hidden.offset(),
                );
                let res_vec = Array::new(
                    residual.metal_buffer().to_owned(),
                    vec![hidden_size],
                    vec![1],
                    residual.dtype(),
                    residual.offset(),
                );
                let out = ops::gemv::gemv_bias_into_encoder(
                    registry,
                    down_w,
                    &hidden_vec,
                    &res_vec,
                    encoder,
                )?;

                encoder.end_encoding();

                Ok(Array::new(
                    out.metal_buffer().to_owned(),
                    vec![1, hidden_size],
                    vec![hidden_size, 1],
                    out.dtype(),
                    out.offset(),
                ))
            }
            _ => Err(KernelError::InvalidShape(
                "forward_concurrent_9dispatch only supports Gated FFN".into(),
            )),
        }
    }
}

pub struct TransformerBlock {
    layer_idx: usize,
    attention: Attention,
    pub(crate) ffn: FeedForward,
    norm1_weight: Option<Array>,
    norm2_weight: Option<Array>,
    rms_norm_eps: f32,
}

impl TransformerBlock {
    /// Config-only constructor (no weights).
    pub fn new(layer_idx: usize, config: TransformerConfig) -> Result<Self, KernelError> {
        config.validate()?;
        let hidden_size = config.hidden_size;
        let rms_norm_eps = config.rms_norm_eps;
        let attn_config = crate::attention::AttentionConfig {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
        };
        let ffn = match config.ff_type {
            FeedForwardType::Dense {
                intermediate_dim,
                activation,
            } => FeedForward::Dense {
                linear1: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: intermediate_dim,
                    has_bias: false,
                }),
                linear2: Linear::new(crate::linear::LinearConfig {
                    in_features: intermediate_dim,
                    out_features: hidden_size,
                    has_bias: false,
                }),
                activation,
            },
            FeedForwardType::Gated { intermediate_dim } => FeedForward::Gated {
                gate_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: intermediate_dim,
                    has_bias: false,
                }),
                up_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: intermediate_dim,
                    has_bias: false,
                }),
                down_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: intermediate_dim,
                    out_features: hidden_size,
                    has_bias: false,
                }),
                gate_up_merged_weight: None,
            },
            FeedForwardType::MoE { .. } => {
                return Err(KernelError::InvalidShape(
                    "TransformerBlock::new(): MoE feed-forward cannot be constructed from config alone; \
                     use TransformerBlock::from_parts() with a pre-built MoeLayer instead"
                        .into(),
                ));
            }
        };
        // Create a dummy device-less norm weight — will be replaced by from_parts
        Ok(Self {
            layer_idx,
            attention: Attention::new(attn_config)?,
            ffn,
            norm1_weight: None,
            norm2_weight: None,
            rms_norm_eps,
        })
    }

    /// Create a transformer block with pre-loaded weights.
    pub fn from_parts(
        layer_idx: usize,
        attention: Attention,
        ffn: FeedForward,
        norm1_weight: Array,
        norm2_weight: Array,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            layer_idx,
            attention,
            ffn,
            norm1_weight: Some(norm1_weight),
            norm2_weight: Some(norm2_weight),
            rms_norm_eps,
        }
    }

    /// Forward pass for one transformer block.
    ///
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables
    /// `mask`: causal attention mask
    /// `cache`: optional per-layer KV cache for incremental decoding
    /// Returns: [seq_len, hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // Pre-attention norm
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // Attention
        let attn_out = self
            .attention
            .forward(&normed, cos_freqs, sin_freqs, mask, cache, registry, queue)?;

        // Residual connection: x + attn_out
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // Pre-FFN norm
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        // FFN
        let ffn_out = self.ffn.forward(&normed2, registry, queue)?;

        // Residual connection: h + ffn_out
        ops::binary::add(registry, &h, &ffn_out, queue)
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn hidden_size(&self) -> Option<usize> {
        self.norm1_weight.as_ref().map(|w| w.shape()[0])
    }

    // -------------------------------------------------------------------
    // ExecGraph path
    // -------------------------------------------------------------------

    /// Pre-cache transposed weights for attention and FFN projections.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        self.attention.prepare_weights_for_graph(registry, queue)?;
        self.ffn.prepare_weights_for_graph(registry, queue)?;
        Ok(())
    }

    /// Prepare merged weights for the 9-dispatch path.
    ///
    /// Merges Q/K/V weights and gate/up weights into single matrices.
    /// Must be called once after weights are loaded.
    pub fn prepare_weights_9dispatch(&mut self, device: &metal::Device) -> Result<(), KernelError> {
        self.attention.prepare_merged_qkv(device)?;
        self.ffn.prepare_merged_gate_up(device)?;
        Ok(())
    }

    /// Convert all static weights (projection matrices, norm weights) to
    /// `StorageModePrivate` (GPU-only memory).
    ///
    /// Weights cannot be read by CPU after this call. Call after loading
    /// weights (and after `prepare_weights_9dispatch` if using that path)
    /// and before the inference loop.
    pub fn prepare_weights_private(&mut self, device: &metal::Device, queue: &metal::CommandQueue) {
        self.attention.prepare_weights_private(device, queue);
        self.ffn.prepare_weights_private(device, queue);
        if let Some(w) = self.norm1_weight.take() {
            self.norm1_weight = Some(w.to_private(device, queue));
        }
        if let Some(w) = self.norm2_weight.take() {
            self.norm2_weight = Some(w.to_private(device, queue));
        }
    }

    /// 9-dispatch forward decode: entire transformer block in 9 GPU dispatches.
    ///
    /// Dispatches:
    ///   1. rms_norm(x, w1)
    ///   2. gemv(W_qkv_merged, normed)
    ///   3. rope(qk_concat, all heads)
    ///   4. sdpa_decode_batched(all heads)
    ///   5. gemv_bias(W_o, attn, x) — O_proj + residual fused
    ///   6. rms_norm(h, w2)
    ///   7. gemv(W_gate_up_merged, normed2)
    ///   8. silu_mul(gate, up)
    ///   9. gemv_bias(W_down, hidden, h) — down_proj + residual fused
    ///
    /// Requires `prepare_weights_9dispatch()` to have been called first.
    /// Requires a pre-allocated `LayerKvCache`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_single_cb_9dispatch(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // Guard: decode path requires seq_len=1
        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_single_cb_9dispatch: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // Dispatches 1-5: attention (norm + QKV + rope + SDPA + O_proj+residual)
        let h = self.attention.forward_decode_9dispatch(
            x,
            norm1_w,
            self.rms_norm_eps,
            cos_freqs,
            sin_freqs,
            mask,
            cache,
            registry,
            cb,
        )?;

        // Dispatch 6: rms_norm (single encoder)
        let enc6 = cb.new_compute_command_encoder();
        let normed2 = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &h,
            Some(norm2_w),
            self.rms_norm_eps,
            enc6,
        )?;
        enc6.end_encoding();

        // Dispatches 7-9: FFN (gate_up + silu_mul + down+residual)
        self.ffn
            .forward_single_cb_9dispatch(&normed2, &h, registry, cb)
    }

    /// 9-dispatch forward using concurrent encoders for better GPU scheduling.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_concurrent_9dispatch(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // Guard: decode path requires seq_len=1
        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_concurrent_9dispatch: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // Dispatches 1-5: attention (norm + QKV + rope + SDPA + O_proj+residual)
        let h = self.attention.forward_decode_concurrent(
            x,
            norm1_w,
            self.rms_norm_eps,
            cos_freqs,
            sin_freqs,
            mask,
            cache,
            registry,
            cb,
        )?;

        // Dispatch 6: rms_norm (single encoder)
        let enc6 = rmlx_metal::new_concurrent_encoder(cb);
        let normed2 = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &h,
            Some(norm2_w),
            self.rms_norm_eps,
            enc6,
        )?;
        enc6.end_encoding();

        // Dispatches 7-9: FFN (gate_up + silu_mul + down+residual)
        self.ffn
            .forward_concurrent_9dispatch(&normed2, &h, registry, cb)
    }

    /// Pipelined forward pass using fused SwiGLU for the FFN.
    ///
    /// Same structure as `forward` but uses `ops::fused::fused_silu_mul`
    /// instead of separate silu + mul.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pipelined(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // Pre-attention norm
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // Attention
        let attn_out = self
            .attention
            .forward(&normed, cos_freqs, sin_freqs, mask, cache, registry, queue)?;

        // Residual connection: x + attn_out
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // Pre-FFN norm
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        // FFN: dense uses generic path; gated uses fused SwiGLU
        let ffn_out = match &self.ffn {
            FeedForward::Dense { .. } => self.ffn.forward(&normed2, registry, queue)?,
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                let gate_out = gate_proj.forward(&normed2, registry, queue)?;
                let up_out = up_proj.forward(&normed2, registry, queue)?;
                let hidden = ops::fused::fused_silu_mul(registry, &gate_out, &up_out, queue)?;
                down_proj.forward(&hidden, registry, queue)?
            }
            FeedForward::MoE(moe) => moe.forward(&normed2, registry, queue)?,
        };

        // Residual connection: h + ffn_out
        ops::binary::add(registry, &h, &ffn_out, queue)
    }

    /// Single-CB forward pass for seq_len=1 decode.
    ///
    /// Encodes the entire transformer block (attention + FFN) into one
    /// command buffer with minimal dispatches. Does NOT commit or wait.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_single_cb(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // Attention (all in same CB — norm + Q/K/V + head split + RoPE + SDPA + concat + O_proj)
        let attn_out = self.attention.forward_decode_single_cb(
            x,
            norm1_w,
            self.rms_norm_eps,
            cos_freqs,
            sin_freqs,
            mask,
            cache,
            registry,
            cb,
        )?;

        // Residual + pre-FFN norm
        let h = ops::binary::add_into_cb(registry, x, &attn_out, cb)?;
        let normed2 =
            ops::rms_norm::rms_norm_into_cb(registry, &h, Some(norm2_w), self.rms_norm_eps, cb)?;

        // FFN (all in same CB)
        self.ffn.forward_single_cb(&normed2, &h, registry, cb)
    }

    /// Full ExecGraph forward pass (5 CBs total).
    ///
    /// CB1: RMS norm + Q/K/V projections (fused)
    /// CB2: head split + RoPE + cache append
    /// CB3: SDPA + head concat + O_proj
    /// CB4: residual + pre-FFN norm
    /// CB5: gate + up + silu_mul + down + residual (entire FFN)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // ---- CB1: norm + Q/K/V projections ----
        // Fuse pre-attention norm into the projection CB to save a submit.
        let (attn_out, t_attn) = self.attention.forward_graph_fused(
            x,
            norm1_w,
            self.rms_norm_eps,
            cos_freqs,
            sin_freqs,
            mask,
            cache,
            registry,
            graph,
        )?;

        // ---- CB4 (from attention): concat + O_proj + residual + pre-FFN norm ----
        // Fuse residual add + pre-FFN norm into attention's last CB.
        graph.wait_for(t_attn);
        let cb4 = graph.command_buffer();
        let h = ops::binary::add_into_cb(registry, x, &attn_out, cb4)?;
        let normed2 =
            ops::rms_norm::rms_norm_into_cb(registry, &h, Some(norm2_w), self.rms_norm_eps, cb4)?;
        let t4 = graph.submit_batch();
        graph.wait_for(t4);

        // ---- CB5: entire FFN (gate + up + silu_mul + down + residual) ----
        self.ffn
            .forward_graph_fused(&normed2, &h, registry, graph, queue)
    }
}

pub struct TransformerModel {
    config: TransformerConfig,
    embedding: Option<Embedding>,
    layers: Vec<TransformerBlock>,
    final_norm_weight: Option<Array>,
    lm_head: Option<Linear>,
    num_layers: usize,
}

impl TransformerModel {
    /// Config-only constructor (no weights loaded).
    pub fn new(config: TransformerConfig) -> Result<Self, KernelError> {
        config.validate()?;
        let num_layers = config.num_layers;
        Ok(Self {
            config,
            embedding: None,
            layers: Vec::new(),
            final_norm_weight: None,
            lm_head: None,
            num_layers,
        })
    }

    /// Create a model with all components pre-loaded.
    pub fn from_parts(
        config: TransformerConfig,
        embedding: Embedding,
        layers: Vec<TransformerBlock>,
        final_norm_weight: Array,
        lm_head: Linear,
    ) -> Result<Self, KernelError> {
        config.validate()?;
        let num_layers = layers.len();
        Ok(Self {
            config,
            embedding: Some(embedding),
            layers,
            final_norm_weight: Some(final_norm_weight),
            lm_head: Some(lm_head),
            num_layers,
        })
    }

    /// Forward pass: token IDs -> logits.
    ///
    /// `token_ids`: input token indices
    /// `cache`: optional per-layer KV caches for incremental decoding.
    ///          Must have exactly `num_layers` entries if provided.
    /// Returns: [seq_len, vocab_size] logits
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let embedding = self.embedding.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: embedding not loaded".into())
        })?;
        let final_norm = self.final_norm_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: final_norm not loaded".into())
        })?;
        let lm_head = self.lm_head.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: lm_head not loaded".into())
        })?;

        // Embedding lookup
        let mut x = embedding.forward(token_ids, registry, queue)?;

        // Validate cache vector length matches number of layers
        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "TransformerModel: cache has {} entries but model has {} layers",
                    c.len(),
                    self.layers.len()
                )));
            }
        }

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            x = layer.forward(&x, cos_freqs, sin_freqs, mask, layer_cache, registry, queue)?;
        }

        // Final norm
        x = ops::rms_norm::rms_norm(registry, &x, final_norm, self.config.rms_norm_eps, queue)?;

        // LM head
        lm_head.forward(&x, registry, queue)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    // -------------------------------------------------------------------
    // ExecGraph path
    // -------------------------------------------------------------------

    /// Pre-cache transposed weights for all layers and the LM head.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        for layer in &mut self.layers {
            layer.prepare_weights_for_graph(registry, queue)?;
        }
        if let Some(ref mut lm) = self.lm_head {
            lm.prepare_weight_t(registry, queue)?;
        }
        Ok(())
    }

    /// Pipelined model forward: uses fused SwiGLU within each block.
    ///
    /// Same as `forward` but each block uses `forward_pipelined`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pipelined(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let embedding = self.embedding.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: embedding not loaded".into())
        })?;
        let final_norm = self.final_norm_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: final_norm not loaded".into())
        })?;
        let lm_head = self.lm_head.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: lm_head not loaded".into())
        })?;

        let mut x = embedding.forward(token_ids, registry, queue)?;

        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "TransformerModel: cache has {} entries but model has {} layers",
                    c.len(),
                    self.layers.len()
                )));
            }
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            x = layer.forward_pipelined(
                &x,
                cos_freqs,
                sin_freqs,
                mask,
                layer_cache,
                registry,
                queue,
            )?;
        }

        x = ops::rms_norm::rms_norm(registry, &x, final_norm, self.config.rms_norm_eps, queue)?;
        lm_head.forward(&x, registry, queue)
    }

    /// Full ExecGraph model forward: token IDs -> logits.
    ///
    /// Creates an ExecGraph per forward pass, running each transformer block
    /// through `forward_graph` (6 CBs per block), then a final norm + LM head.
    /// The CPU blocks only once at the very end.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
        event: &GpuEvent,
    ) -> Result<Array, KernelError> {
        let embedding = self.embedding.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: embedding not loaded".into())
        })?;
        let final_norm = self.final_norm_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: final_norm not loaded".into())
        })?;
        let lm_head = self.lm_head.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerModel: lm_head not loaded".into())
        })?;

        // Embedding lookup (sync — typically fast)
        let mut x = embedding.forward(token_ids, registry, queue)?;

        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "TransformerModel: cache has {} entries but model has {} layers",
                    c.len(),
                    self.layers.len()
                )));
            }
        }

        // Create graph for the full forward pass
        let mut graph = ExecGraph::new(queue, event, 32);

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            x = layer.forward_graph(
                &x,
                cos_freqs,
                sin_freqs,
                mask,
                layer_cache,
                registry,
                &mut graph,
                queue,
            )?;
            // Submit remaining work and wait between layers to ensure
            // the output is ready for the next layer's input
            let t = graph.submit_batch();
            graph.wait_for(t);
        }

        // Final norm + LM head (encode into graph)
        let cb_final = graph.command_buffer();
        x = ops::rms_norm::rms_norm_into_cb(
            registry,
            &x,
            Some(final_norm),
            self.config.rms_norm_eps,
            cb_final,
        )?;
        x = lm_head.forward_into_cb(&x, registry, cb_final)?;

        // Single CPU sync at the end
        graph
            .sync()
            .map_err(|e| KernelError::InvalidShape(format!("TransformerModel graph sync: {e}")))?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_type_variants() {
        let _dense = FeedForwardType::Dense {
            intermediate_dim: 256,
            activation: crate::activations::ActivationType::GELU,
        };
        let _gated = FeedForwardType::Gated {
            intermediate_dim: 256,
        };
    }

    #[test]
    fn test_transformer_config_with_gated_ffn() {
        let config = TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            vocab_size: 100,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            ff_type: FeedForwardType::Gated {
                intermediate_dim: 128,
            },
        };
        assert!(config.validate().is_ok());
        let block = TransformerBlock::new(0, config).unwrap();
        assert_eq!(block.layer_idx(), 0);
        match &block.ffn {
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => {
                assert_eq!(gate_proj.in_features(), 64);
                assert_eq!(gate_proj.out_features(), 128);
                assert_eq!(up_proj.in_features(), 64);
                assert_eq!(up_proj.out_features(), 128);
                assert_eq!(down_proj.in_features(), 128);
                assert_eq!(down_proj.out_features(), 64);
            }
            _ => panic!("Expected Gated FFN variant"),
        }
    }

    #[test]
    fn test_transformer_config_with_dense_ffn() {
        let config = TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            vocab_size: 100,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            ff_type: FeedForwardType::Dense {
                intermediate_dim: 256,
                activation: crate::activations::ActivationType::GELU,
            },
        };
        assert!(config.validate().is_ok());
        let block = TransformerBlock::new(0, config).unwrap();
        match &block.ffn {
            FeedForward::Dense {
                linear1,
                linear2,
                activation,
            } => {
                assert_eq!(linear1.in_features(), 64);
                assert_eq!(linear1.out_features(), 256);
                assert_eq!(linear2.in_features(), 256);
                assert_eq!(linear2.out_features(), 64);
                assert_eq!(*activation, crate::activations::ActivationType::GELU);
            }
            _ => panic!("Expected Dense FFN variant"),
        }
    }
}
