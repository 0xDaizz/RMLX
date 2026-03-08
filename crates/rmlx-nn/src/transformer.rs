//! Transformer block: attention + MLP (or MoE).

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
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
        /// Transposed merged gate+up weight [in_features, gate_dim + up_dim] for prefill GEMM.
        gate_up_merged_weight_t: Option<Array>,
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
                gate_up_merged_weight_t,
                ..
            } => {
                let normed_2d = if normed.ndim() == 1 {
                    normed.reshape(vec![1, normed.shape()[0]])?
                } else {
                    normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
                };
                let (gate_out, up_out) = if let Some(ref guw_t) = gate_up_merged_weight_t {
                    // Single merged GEMM: [seq, hidden] @ [hidden, gate+up] = [seq, gate+up]
                    let merged = ops::matmul::matmul_into_cb(registry, &normed_2d, guw_t, cb)?;
                    let gate_dim = guw_t.shape()[1] / 2;
                    let total_out = guw_t.shape()[1];
                    let seq_len = normed_2d.shape()[0];
                    let elem_size = merged.dtype().size_of();
                    let gate_view = Array::new(
                        merged.metal_buffer().to_owned(),
                        vec![seq_len, gate_dim],
                        vec![total_out, 1],
                        merged.dtype(),
                        merged.offset(),
                    );
                    let up_view = Array::new(
                        merged.metal_buffer().to_owned(),
                        vec![seq_len, gate_dim],
                        vec![total_out, 1],
                        merged.dtype(),
                        merged.offset() + gate_dim * elem_size,
                    );
                    let g = ops::copy::copy_into_cb(registry, &gate_view, cb)?;
                    let u = ops::copy::copy_into_cb(registry, &up_view, cb)?;
                    (g, u)
                } else {
                    // Fallback: 2 separate GEMMs
                    let wgate_t = gate_proj.weight_transposed_contiguous()?;
                    let wup_t = up_proj.weight_transposed_contiguous()?;
                    ops::fused::batched_gate_up_into_cb(registry, &normed_2d, &wgate_t, &wup_t, cb)?
                };
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
                gate_up_merged_weight_t,
                ..
            } => {
                let cb = graph.command_buffer();
                let normed_2d = if normed.ndim() == 1 {
                    normed.reshape(vec![1, normed.shape()[0]])?
                } else {
                    normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
                };
                let (gate_out, up_out) = if let Some(ref guw_t) = gate_up_merged_weight_t {
                    let merged = ops::matmul::matmul_into_cb(registry, &normed_2d, guw_t, cb)?;
                    let gate_dim = guw_t.shape()[1] / 2;
                    let total_out = guw_t.shape()[1];
                    let seq_len = normed_2d.shape()[0];
                    let elem_size = merged.dtype().size_of();
                    let gate_view = Array::new(
                        merged.metal_buffer().to_owned(),
                        vec![seq_len, gate_dim],
                        vec![total_out, 1],
                        merged.dtype(),
                        merged.offset(),
                    );
                    let up_view = Array::new(
                        merged.metal_buffer().to_owned(),
                        vec![seq_len, gate_dim],
                        vec![total_out, 1],
                        merged.dtype(),
                        merged.offset() + gate_dim * elem_size,
                    );
                    let g = ops::copy::copy_into_cb(registry, &gate_view, cb)?;
                    let u = ops::copy::copy_into_cb(registry, &up_view, cb)?;
                    (g, u)
                } else {
                    let wgate_t = gate_proj.weight_transposed_contiguous()?;
                    let wup_t = up_proj.weight_transposed_contiguous()?;
                    ops::fused::batched_gate_up_into_cb(registry, &normed_2d, &wgate_t, &wup_t, cb)?
                };
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
                gate_up_merged_weight_t,
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
                    buf.clone(),
                    vec![total_rows, cols],
                    vec![cols, 1],
                    gate_w.dtype(),
                    0,
                ));

                // Also create transposed merged weight [cols, total_rows] for prefill GEMM.
                let buf_t = device.new_buffer(
                    total_bytes as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                unsafe {
                    let src = buf.contents() as *const u8;
                    let dst = buf_t.contents() as *mut u8;
                    for r in 0..total_rows {
                        for c in 0..cols {
                            let src_idx = (r * cols + c) * elem_size;
                            let dst_idx = (c * total_rows + r) * elem_size;
                            std::ptr::copy_nonoverlapping(
                                src.add(src_idx),
                                dst.add(dst_idx),
                                elem_size,
                            );
                        }
                    }
                }
                *gate_up_merged_weight_t = Some(Array::new(
                    buf_t,
                    vec![cols, total_rows],
                    vec![total_rows, 1],
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
                gate_up_merged_weight_t,
            } => {
                gate_proj.convert_weights_private(device, queue);
                up_proj.convert_weights_private(device, queue);
                down_proj.convert_weights_private(device, queue);
                if let Some(w) = gate_up_merged_weight.take() {
                    *gate_up_merged_weight = Some(w.to_private(device, queue));
                }
                if let Some(w) = gate_up_merged_weight_t.take() {
                    *gate_up_merged_weight_t = Some(w.to_private(device, queue));
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

    /// Shard FFN weights for Tensor Parallelism.
    ///
    /// - Gated: gate/up → column-parallel (shard output rows), down → row-parallel (shard input cols)
    /// - Dense: linear1 → column-parallel, linear2 → row-parallel
    /// - MoE: not supported (returns error)
    #[cfg(feature = "distributed")]
    pub(crate) fn shard_for_tp(
        &mut self,
        rank: u32,
        world_size: u32,
    ) -> Result<(), KernelError> {
        use crate::parallel::{ColumnParallelLinear, RowParallelLinear};

        if world_size <= 1 {
            return Ok(());
        }
        match self {
            FeedForward::Gated {
                gate_proj,
                up_proj,
                down_proj,
                gate_up_merged_weight,
                gate_up_merged_weight_t,
            } => {
                let gate_w = gate_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("shard_for_tp: gate_proj weight not loaded".into())
                })?;
                let up_w = up_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("shard_for_tp: up_proj weight not loaded".into())
                })?;
                let down_w = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("shard_for_tp: down_proj weight not loaded".into())
                })?;

                let inter_dim = gate_w.shape()[0];
                let hidden_size = gate_w.shape()[1];
                let local_inter = inter_dim / (world_size as usize);

                // Gate: column-parallel (shard output)
                let gate_shard = ColumnParallelLinear::shard_weight(gate_w, rank, world_size);
                *gate_proj = Linear::from_arrays(
                    crate::linear::LinearConfig {
                        in_features: hidden_size,
                        out_features: local_inter,
                        has_bias: false,
                    },
                    gate_shard,
                    None,
                )?;

                // Up: column-parallel
                let up_shard = ColumnParallelLinear::shard_weight(up_w, rank, world_size);
                *up_proj = Linear::from_arrays(
                    crate::linear::LinearConfig {
                        in_features: hidden_size,
                        out_features: local_inter,
                        has_bias: false,
                    },
                    up_shard,
                    None,
                )?;

                // Down: row-parallel (shard input columns)
                let down_shard = RowParallelLinear::shard_weight(down_w, rank, world_size);
                *down_proj = Linear::from_arrays(
                    crate::linear::LinearConfig {
                        in_features: local_inter,
                        out_features: hidden_size,
                        has_bias: false,
                    },
                    down_shard,
                    None,
                )?;

                // Invalidate merged weights
                *gate_up_merged_weight = None;
                *gate_up_merged_weight_t = None;

                Ok(())
            }
            FeedForward::Dense {
                linear1, linear2, ..
            } => {
                let w1 = linear1.weight().ok_or_else(|| {
                    KernelError::InvalidShape("shard_for_tp: linear1 weight not loaded".into())
                })?;
                let w2 = linear2.weight().ok_or_else(|| {
                    KernelError::InvalidShape("shard_for_tp: linear2 weight not loaded".into())
                })?;

                let inter_dim = w1.shape()[0];
                let hidden_size = w1.shape()[1];
                let local_inter = inter_dim / (world_size as usize);

                // linear1: column-parallel
                let w1_shard = ColumnParallelLinear::shard_weight(w1, rank, world_size);
                *linear1 = Linear::from_arrays(
                    crate::linear::LinearConfig {
                        in_features: hidden_size,
                        out_features: local_inter,
                        has_bias: false,
                    },
                    w1_shard,
                    None,
                )?;

                // linear2: row-parallel
                let w2_shard = RowParallelLinear::shard_weight(w2, rank, world_size);
                *linear2 = Linear::from_arrays(
                    crate::linear::LinearConfig {
                        in_features: local_inter,
                        out_features: hidden_size,
                        has_bias: false,
                    },
                    w2_shard,
                    None,
                )?;

                Ok(())
            }
            FeedForward::MoE(_) => Err(KernelError::InvalidShape(
                "shard_for_tp: MoE not supported for TP".into(),
            )),
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

    /// 9-dispatch FFN forward into a caller-supplied encoder (D7-D9).
    ///
    /// Same logic as `forward_single_cb_9dispatch` but does NOT create or end
    /// the encoder — the caller manages encoder lifetime. This enables merging
    /// D6-D9 (or D4-D9) into a single encoder to eliminate GPU barriers.
    pub fn forward_into_encoder_9dispatch(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                let normed_seq = if normed.ndim() >= 2 {
                    normed.shape()[0]
                } else {
                    1
                };
                if normed_seq != 1 {
                    return Err(KernelError::InvalidShape(format!(
                        "forward_into_encoder_9dispatch: requires seq_len=1, got {}",
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

                // Dispatch 7: merged gate+up gemv
                let gate_up_w = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_into_encoder_9dispatch: call prepare_merged_gate_up() first"
                            .into(),
                    )
                })?;
                let gate_up =
                    ops::gemv::gemv_into_encoder(registry, gate_up_w, &normed_vec, encoder)?;
                let gate_dim = gate_up_w.shape()[0] / 2;
                let elem_size = gate_up.dtype().size_of();

                encoder.memory_barrier_with_resources(&[buf_as_resource(gate_up.metal_buffer())]);

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

                encoder.memory_barrier_with_resources(&[buf_as_resource(hidden.metal_buffer())]);

                // Dispatch 9: gemv_bias(W_down, hidden, residual)
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

                Ok(Array::new(
                    out.metal_buffer().to_owned(),
                    vec![1, hidden_size],
                    vec![hidden_size, 1],
                    out.dtype(),
                    out.offset(),
                ))
            }
            _ => Err(KernelError::InvalidShape(
                "forward_into_encoder_9dispatch only supports Gated FFN".into(),
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
                gate_up_merged_weight_t: None,
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

    /// Shard this block's weights for Tensor Parallelism.
    ///
    /// Shards attention (QKV column-parallel, O row-parallel) and FFN
    /// (gate/up column-parallel, down row-parallel) weights, and updates
    /// the attention config to reflect local head counts.
    ///
    /// Must be called after weights are loaded and before `forward_with_group()`.
    #[cfg(feature = "distributed")]
    pub(crate) fn shard_for_tp(
        &mut self,
        rank: u32,
        world_size: u32,
    ) -> Result<(), KernelError> {
        self.attention.shard_for_tp(rank, world_size)?;
        self.ffn.shard_for_tp(rank, world_size)?;
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

    /// 2-encoder 9-dispatch forward: merges D4-D9 into a single compute encoder.
    ///
    /// Encoder layout:
    ///   Encoder A (compute): D1(rms_norm), D2(QKV gemv), D3(rope)
    ///   [KV cache blit encoders]
    ///   Encoder B (compute): D4(sdpa), D5(O_proj+res), D6(rms_norm),
    ///                        D7(gate+up gemv), D8(silu_mul), D9(down+res)
    ///
    /// This eliminates 2 full GPU barriers compared to `forward_single_cb_9dispatch`
    /// (which uses 4 compute encoders: A, B, 6, D).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_2encoder_9dispatch(
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

        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_2encoder_9dispatch: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // Phase 1: D1-D3 in encoder A + KV blit (creates/ends encoder A internally)
        let phase1 = self.attention.forward_decode_phase1(
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

        // --- Single encoder B for D4-D9 ---
        let encoder_b = cb.new_compute_command_encoder();

        // D4-D5: SDPA + O_proj+residual
        let h = self
            .attention
            .encode_phase2_into(&phase1, cache, registry, encoder_b)?;

        // D5→D6 barrier: ensure h is visible before rms_norm reads it
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(h.metal_buffer())]);

        // D6: rms_norm
        let normed2 = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &h,
            Some(norm2_w),
            self.rms_norm_eps,
            encoder_b,
        )?;

        // D6→D7 barrier: ensure normed2 is visible before FFN reads it
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(normed2.metal_buffer())]);

        // D7-D9: FFN (gate+up, silu_mul, down+residual)
        let out = self
            .ffn
            .forward_into_encoder_9dispatch(&normed2, &h, registry, encoder_b)?;

        encoder_b.end_encoding();

        Ok(out)
    }

    /// Prepare a `CachedDecode` for this layer.
    ///
    /// Pre-resolves all PSOs and allocates all scratch buffers. Call once at
    /// init time; reuse the returned `CachedDecode` for every decode step.
    pub fn prepare_cached_decode(
        &self,
        registry: &KernelRegistry,
    ) -> Result<CachedDecode, KernelError> {
        CachedDecode::new(self, registry)
    }

    /// Zero-overhead decode forward using pre-resolved PSOs and pre-allocated
    /// scratch buffers. Encodes all 9 dispatches + KV copy into 2 encoders
    /// with no dynamic allocation or PSO lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_2encoder_9dispatch(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        cached: &CachedDecode,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let hidden_size = cached.hidden_size;
        let num_heads = cached.num_heads;
        let num_kv_heads = cached.num_kv_heads;
        let head_dim = cached.head_dim;
        let elem_size = cached.dtype.size_of();

        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm2_weight not loaded".into())
        })?;
        let qkv_weight = self.attention.qkv_merged_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: qkv_merged_weight not loaded".into())
        })?;
        let o_weight = self.attention.o_proj_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: o_proj weight not loaded".into())
        })?;
        let (gate_up_w, down_w) = match &self.ffn {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                let guw = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_cached: gate_up_merged_weight not loaded".into(),
                    )
                })?;
                let dw = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("forward_cached: down_proj weight not loaded".into())
                })?;
                (guw, dw)
            }
            _ => {
                return Err(KernelError::InvalidShape(
                    "forward_cached only supports Gated FFN".into(),
                ))
            }
        };

        // =====================================================================
        // Encoder A: D1 (rms_norm), D2 (QKV gemv), D3 (rope), KV copy
        // =====================================================================
        let encoder_a = cb.new_compute_command_encoder();

        // D1: rms_norm → normed_buf
        ops::rms_norm::rms_norm_preresolved_into_encoder(
            &cached.rms_norm_pso,
            x.metal_buffer(),
            x.offset() as u64,
            norm1_w.metal_buffer(),
            norm1_w.offset() as u64,
            &cached.normed_buf,
            0,
            cached.rms_axis_size,
            cached.rms_norm_eps,
            cached.norm1_w_stride, // w_stride
            1,                     // has_w
            1,                     // rows
            encoder_a,
        );
        encoder_a.memory_barrier_with_resources(&[buf_as_resource(&cached.normed_buf)]);

        // D2: QKV gemv → qkv_buf
        ops::gemv::gemv_preresolved_into_encoder(
            &cached.gemv_qkv_pso,
            qkv_weight.metal_buffer(),
            qkv_weight.offset() as u64,
            &cached.normed_buf,
            0,
            &cached.qkv_buf,
            0,
            cached.gemv_qkv_m,
            cached.gemv_qkv_k,
            cached.gemv_qkv_grid,
            cached.gemv_qkv_tg,
            encoder_a,
        );

        // D3: rope → rope_buf (or use qkv_buf directly if no RoPE)
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let total_rope_heads = num_heads + num_kv_heads;
        let rope_offset = cache.seq_len as u32;

        let (qk_buf_ref, qk_offset, v_buf_ref, v_offset): (
            &metal::BufferRef,
            u64,
            &metal::BufferRef,
            u64,
        );

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            encoder_a.memory_barrier_with_resources(&[buf_as_resource(&cached.qkv_buf)]);
            ops::rope::rope_ext_preresolved_into_encoder(
                &cached.rope_pso,
                &cached.qkv_buf,
                0,
                cos.metal_buffer(),
                cos.offset() as u64,
                sin.metal_buffer(),
                sin.offset() as u64,
                &cached.rope_buf,
                0,
                1, // seq_len = 1
                head_dim as u32,
                rope_offset,
                1.0,
                0,                       // traditional = false
                1,                       // forward = true
                total_rope_heads as u64, // n_batch
                encoder_a,
            );
            qk_buf_ref = &cached.rope_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        } else {
            qk_buf_ref = &cached.qkv_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        }

        // Memory barrier: ensure rope/qkv output visible to KV copy
        encoder_a.memory_barrier_with_resources(&[buf_as_resource(qk_buf_ref)]);

        // KV cache append using pre-resolved copy PSO
        let k_roped_flat_offset = qk_offset + (q_dim * elem_size) as u64;
        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            k_heads.push(Array::new(
                qk_buf_ref.to_owned(),
                vec![1, head_dim],
                vec![head_dim, 1],
                cached.dtype,
                k_roped_flat_offset as usize + h * head_dim * elem_size,
            ));
            v_heads.push(Array::new(
                v_buf_ref.to_owned(),
                vec![1, head_dim],
                vec![head_dim, 1],
                cached.dtype,
                v_offset as usize + h * head_dim * elem_size,
            ));
        }
        cache.append_preresolved_into_encoder(k_heads, v_heads, 1, &cached.copy_pso, encoder_a)?;

        encoder_a.end_encoding();

        // =====================================================================
        // Encoder B: D4 (SDPA), D5 (O_proj+res), D6 (rms_norm2), D7 (gate_up),
        //            D8 (silu_mul), D9 (down_proj+res)
        // =====================================================================
        let encoder_b = cb.new_compute_command_encoder();

        // D4: SDPA decode → attn_out_buf
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no values slab after append".into())
        })?;
        let seq_len = cache.seq_len;
        let max_seq = cache.max_seq_len();

        ops::sdpa::sdpa_decode_preresolved_into_encoder(
            &cached.sdpa_pso,
            qk_buf_ref, // Q from rope output
            qk_offset,
            k_slab.metal_buffer(),
            k_slab.offset() as u64,
            v_slab.metal_buffer(),
            v_slab.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.dummy_mask_buf,
            0,
            num_heads as u32,
            num_kv_heads as u32,
            seq_len as u32,
            head_dim as u32,
            0, // has_mask = false
            max_seq as u32,
            cached.scale,
            encoder_b,
        );
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(&cached.attn_out_buf)]);

        // D5: gemv_bias(W_o, attn, x) → h_buf (O_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_oproj_pso,
            o_weight.metal_buffer(),
            o_weight.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.h_buf,
            0,
            cached.gemv_bias_oproj_m,
            cached.gemv_bias_oproj_k,
            x.metal_buffer(), // bias = input x (residual)
            x.offset() as u64,
            cached.gemv_bias_oproj_grid,
            cached.gemv_bias_oproj_tg,
            encoder_b,
        );
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(&cached.h_buf)]);

        // D6: rms_norm → normed2_buf
        ops::rms_norm::rms_norm_preresolved_into_encoder(
            &cached.rms_norm2_pso,
            &cached.h_buf,
            0,
            norm2_w.metal_buffer(),
            norm2_w.offset() as u64,
            &cached.normed2_buf,
            0,
            cached.rms_axis_size,
            cached.rms_norm_eps,
            cached.norm2_w_stride, // w_stride
            1,                     // has_w
            1,                     // rows
            encoder_b,
        );
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(&cached.normed2_buf)]);

        // D7: gate_up gemv → gate_up_buf
        ops::gemv::gemv_preresolved_into_encoder(
            &cached.gemv_gate_up_pso,
            gate_up_w.metal_buffer(),
            gate_up_w.offset() as u64,
            &cached.normed2_buf,
            0,
            &cached.gate_up_buf,
            0,
            cached.gemv_gate_up_m,
            cached.gemv_gate_up_k,
            cached.gemv_gate_up_grid,
            cached.gemv_gate_up_tg,
            encoder_b,
        );
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(&cached.gate_up_buf)]);

        // D8: silu_mul → silu_buf
        let gate_dim = cached.intermediate_dim;
        ops::fused::fused_silu_mul_preresolved_into_encoder(
            &cached.silu_mul_pso,
            &cached.gate_up_buf,
            0,
            &cached.gate_up_buf,
            (gate_dim * elem_size) as u64,
            &cached.silu_buf,
            0,
            cached.silu_numel,
            cached.silu_elems_per_thread,
            encoder_b,
        );
        encoder_b.memory_barrier_with_resources(&[buf_as_resource(&cached.silu_buf)]);

        // D9: gemv_bias(W_down, silu, h) → out_buf (down_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_down_pso,
            down_w.metal_buffer(),
            down_w.offset() as u64,
            &cached.silu_buf,
            0,
            &cached.out_buf,
            0,
            cached.gemv_bias_down_m,
            cached.gemv_bias_down_k,
            &cached.h_buf, // bias = h (residual from D5)
            0,
            cached.gemv_bias_down_grid,
            cached.gemv_bias_down_tg,
            encoder_b,
        );

        encoder_b.end_encoding();

        // Return as [1, hidden_size]
        Ok(Array::new(
            cached.out_buf.clone(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            cached.dtype,
            0,
        ))
    }

    /// Single-encoder 9-dispatch forward pass (Optimization A).
    ///
    /// Merges both encoders from `forward_cached_2encoder_9dispatch` into one,
    /// using memory barriers on KV slab buffers instead of an encoder boundary
    /// to synchronize the KV copy → SDPA dependency.
    ///
    /// Also uses `append_direct_into_encoder` (Optimization B) to avoid
    /// Vec<Array> allocation for KV copies, and pre-cached threadgroup sizes
    /// (Optimization C).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_1encoder_9dispatch(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        cached: &CachedDecode,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        let hidden_size = cached.hidden_size;
        let num_heads = cached.num_heads;
        let num_kv_heads = cached.num_kv_heads;
        let head_dim = cached.head_dim;
        let elem_size = cached.dtype.size_of();

        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm2_weight not loaded".into())
        })?;
        let qkv_weight = self.attention.qkv_merged_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: qkv_merged_weight not loaded".into())
        })?;
        let o_weight = self.attention.o_proj_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: o_proj weight not loaded".into())
        })?;
        let (gate_up_w, down_w) = match &self.ffn {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                let guw = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_cached: gate_up_merged_weight not loaded".into(),
                    )
                })?;
                let dw = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("forward_cached: down_proj weight not loaded".into())
                })?;
                (guw, dw)
            }
            _ => {
                return Err(KernelError::InvalidShape(
                    "forward_cached only supports Gated FFN".into(),
                ))
            }
        };

        // =====================================================================
        // Single Encoder: all 9 dispatches + KV copy
        // =====================================================================
        let encoder = cb.new_compute_command_encoder();

        // D1: rms_norm -> normed_buf
        ops::rms_norm::rms_norm_preresolved_into_encoder(
            &cached.rms_norm_pso,
            x.metal_buffer(),
            x.offset() as u64,
            norm1_w.metal_buffer(),
            norm1_w.offset() as u64,
            &cached.normed_buf,
            0,
            cached.rms_axis_size,
            cached.rms_norm_eps,
            cached.norm1_w_stride,
            1,
            1,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed_buf)]);

        // D2: QKV gemv -> qkv_buf
        ops::gemv::gemv_preresolved_into_encoder(
            &cached.gemv_qkv_pso,
            qkv_weight.metal_buffer(),
            qkv_weight.offset() as u64,
            &cached.normed_buf,
            0,
            &cached.qkv_buf,
            0,
            cached.gemv_qkv_m,
            cached.gemv_qkv_k,
            cached.gemv_qkv_grid,
            cached.gemv_qkv_tg,
            encoder,
        );

        // D3: rope -> rope_buf (or use qkv_buf directly if no RoPE)
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let total_rope_heads = num_heads + num_kv_heads;
        let rope_offset = cache.seq_len as u32;

        let (qk_buf_ref, qk_offset, v_buf_ref, v_offset): (
            &metal::BufferRef,
            u64,
            &metal::BufferRef,
            u64,
        );

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.qkv_buf)]);
            ops::rope::rope_ext_preresolved_into_encoder(
                &cached.rope_pso,
                &cached.qkv_buf,
                0,
                cos.metal_buffer(),
                cos.offset() as u64,
                sin.metal_buffer(),
                sin.offset() as u64,
                &cached.rope_buf,
                0,
                1,
                head_dim as u32,
                rope_offset,
                1.0,
                0,
                1,
                total_rope_heads as u64,
                encoder,
            );
            qk_buf_ref = &cached.rope_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        } else {
            qk_buf_ref = &cached.qkv_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        }

        // Memory barrier: ensure rope/qkv output visible to KV copy
        encoder.memory_barrier_with_resources(&[buf_as_resource(qk_buf_ref)]);

        // KV cache append using direct buffer refs (Optimization B)
        let k_roped_flat_offset = qk_offset + (q_dim * elem_size) as u64;
        cache.append_direct_into_encoder(
            qk_buf_ref,
            k_roped_flat_offset,
            v_buf_ref,
            v_offset,
            1,
            &cached.copy_pso,
            cached.copy_max_tg,
            encoder,
        )?;

        // Memory barrier on KV slab buffers: KV copy -> SDPA read dependency
        // (replaces encoder boundary from the 2-encoder path)
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no values slab after append".into())
        })?;
        encoder.memory_barrier_with_resources(&[
            buf_as_resource(k_slab.metal_buffer()),
            buf_as_resource(v_slab.metal_buffer()),
        ]);

        let seq_len = cache.seq_len;
        let max_seq = cache.max_seq_len();

        // D4: SDPA decode -> attn_out_buf
        ops::sdpa::sdpa_decode_preresolved_into_encoder(
            &cached.sdpa_pso,
            qk_buf_ref,
            qk_offset,
            k_slab.metal_buffer(),
            k_slab.offset() as u64,
            v_slab.metal_buffer(),
            v_slab.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.dummy_mask_buf,
            0,
            num_heads as u32,
            num_kv_heads as u32,
            seq_len as u32,
            head_dim as u32,
            0,
            max_seq as u32,
            cached.scale,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.attn_out_buf)]);

        // D5: gemv_bias(W_o, attn, x) -> h_buf (O_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_oproj_pso,
            o_weight.metal_buffer(),
            o_weight.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.h_buf,
            0,
            cached.gemv_bias_oproj_m,
            cached.gemv_bias_oproj_k,
            x.metal_buffer(),
            x.offset() as u64,
            cached.gemv_bias_oproj_grid,
            cached.gemv_bias_oproj_tg,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.h_buf)]);

        // D6: rms_norm -> normed2_buf
        ops::rms_norm::rms_norm_preresolved_into_encoder(
            &cached.rms_norm2_pso,
            &cached.h_buf,
            0,
            norm2_w.metal_buffer(),
            norm2_w.offset() as u64,
            &cached.normed2_buf,
            0,
            cached.rms_axis_size,
            cached.rms_norm_eps,
            cached.norm2_w_stride,
            1,
            1,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed2_buf)]);

        // D7: gate_up gemv -> gate_up_buf
        ops::gemv::gemv_preresolved_into_encoder(
            &cached.gemv_gate_up_pso,
            gate_up_w.metal_buffer(),
            gate_up_w.offset() as u64,
            &cached.normed2_buf,
            0,
            &cached.gate_up_buf,
            0,
            cached.gemv_gate_up_m,
            cached.gemv_gate_up_k,
            cached.gemv_gate_up_grid,
            cached.gemv_gate_up_tg,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.gate_up_buf)]);

        // D8: silu_mul -> silu_buf
        let gate_dim = cached.intermediate_dim;
        ops::fused::fused_silu_mul_preresolved_into_encoder(
            &cached.silu_mul_pso,
            &cached.gate_up_buf,
            0,
            &cached.gate_up_buf,
            (gate_dim * elem_size) as u64,
            &cached.silu_buf,
            0,
            cached.silu_numel,
            cached.silu_elems_per_thread,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.silu_buf)]);

        // D9: gemv_bias(W_down, silu, h) -> out_buf (down_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_down_pso,
            down_w.metal_buffer(),
            down_w.offset() as u64,
            &cached.silu_buf,
            0,
            &cached.out_buf,
            0,
            cached.gemv_bias_down_m,
            cached.gemv_bias_down_k,
            &cached.h_buf,
            0,
            cached.gemv_bias_down_grid,
            cached.gemv_bias_down_tg,
            encoder,
        );

        encoder.end_encoding();

        // Return as [1, hidden_size]
        Ok(Array::new(
            cached.out_buf.clone(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            cached.dtype,
            0,
        ))
    }

    /// Fused 7-dispatch forward using kernel fusion (Phase 10).
    ///
    /// Replaces the 9-dispatch path by fusing:
    /// - D1+D2 → fused_rms_gemv (rms_norm + QKV GEMV)
    /// - D6+D7 → fused_rms_gemv (rms_norm + gate_up GEMV)
    /// - D8+D9 → fused_swiglu_down (SwiGLU + down_proj GEMV + residual)
    ///
    /// Falls back to 9-dispatch if any fused PSO is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_fused_7dispatch(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        cached: &CachedDecode,
        cb: &metal::CommandBufferRef,
    ) -> Result<Array, KernelError> {
        // Check fused PSOs available — fallback to 9-dispatch if not
        let (rms_gemv_qkv_pso, rms_gemv_gate_up_pso, swiglu_down_pso) = match (
            &cached.fused_rms_gemv_qkv_pso,
            &cached.fused_rms_gemv_gate_up_pso,
            &cached.fused_swiglu_down_pso,
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                return self
                    .forward_cached_1encoder_9dispatch(x, cos_freqs, sin_freqs, cache, cached, cb)
            }
        };

        let hidden_size = cached.hidden_size;
        let num_heads = cached.num_heads;
        let num_kv_heads = cached.num_kv_heads;
        let head_dim = cached.head_dim;
        let elem_size = cached.dtype.size_of();

        // Get weights (same as 9-dispatch)
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm2_weight not loaded".into())
        })?;
        let qkv_weight = self.attention.qkv_merged_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: qkv_merged_weight not loaded".into())
        })?;
        let o_weight = self.attention.o_proj_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: o_proj weight not loaded".into())
        })?;
        let (gate_up_w, down_w) = match &self.ffn {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                let guw = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_cached: gate_up_merged_weight not loaded".into(),
                    )
                })?;
                let dw = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("forward_cached: down_proj weight not loaded".into())
                })?;
                (guw, dw)
            }
            _ => {
                return Err(KernelError::InvalidShape(
                    "forward_cached only supports Gated FFN".into(),
                ))
            }
        };

        let encoder = cb.new_compute_command_encoder();

        // D1: fused_rms_gemv(x, norm1_w, qkv_w) → qkv_buf
        // Replaces D1 (rms_norm) + D2 (gemv QKV) from 9-dispatch
        ops::fused::fused_rms_gemv_preresolved_into_encoder(
            rms_gemv_qkv_pso,
            x.metal_buffer(),
            x.offset() as u64,
            norm1_w.metal_buffer(),
            norm1_w.offset() as u64,
            qkv_weight.metal_buffer(),
            qkv_weight.offset() as u64,
            &cached.qkv_buf,
            0,
            cached.gemv_qkv_m,
            cached.gemv_qkv_k,
            cached.rms_norm_eps,
            cached.norm1_w_stride,
            cached.fused_rms_gemv_qkv_grid,
            cached.fused_rms_gemv_qkv_tg,
            encoder,
        );

        // D2: rope → rope_buf
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let total_rope_heads = num_heads + num_kv_heads;
        let rope_offset = cache.seq_len as u32;

        let (qk_buf_ref, qk_offset, v_buf_ref, v_offset): (
            &metal::BufferRef,
            u64,
            &metal::BufferRef,
            u64,
        );

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.qkv_buf)]);
            ops::rope::rope_ext_preresolved_into_encoder(
                &cached.rope_pso,
                &cached.qkv_buf,
                0,
                cos.metal_buffer(),
                cos.offset() as u64,
                sin.metal_buffer(),
                sin.offset() as u64,
                &cached.rope_buf,
                0,
                1,
                head_dim as u32,
                rope_offset,
                1.0,
                0,
                1,
                total_rope_heads as u64,
                encoder,
            );
            qk_buf_ref = &cached.rope_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        } else {
            qk_buf_ref = &cached.qkv_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        }

        encoder.memory_barrier_with_resources(&[buf_as_resource(qk_buf_ref)]);

        // KV cache append
        let k_roped_flat_offset = qk_offset + (q_dim * elem_size) as u64;
        cache.append_direct_into_encoder(
            qk_buf_ref,
            k_roped_flat_offset,
            v_buf_ref,
            v_offset,
            1,
            &cached.copy_pso,
            cached.copy_max_tg,
            encoder,
        )?;

        // Memory barrier on KV slab buffers
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no values slab after append".into())
        })?;
        encoder.memory_barrier_with_resources(&[
            buf_as_resource(k_slab.metal_buffer()),
            buf_as_resource(v_slab.metal_buffer()),
        ]);

        let seq_len = cache.seq_len;
        let max_seq = cache.max_seq_len();

        // D3: SDPA decode → attn_out_buf
        ops::sdpa::sdpa_decode_preresolved_into_encoder(
            &cached.sdpa_pso,
            qk_buf_ref,
            qk_offset,
            k_slab.metal_buffer(),
            k_slab.offset() as u64,
            v_slab.metal_buffer(),
            v_slab.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.dummy_mask_buf,
            0,
            num_heads as u32,
            num_kv_heads as u32,
            seq_len as u32,
            head_dim as u32,
            0,
            max_seq as u32,
            cached.scale,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.attn_out_buf)]);

        // D4: gemv_bias(W_o, attn, x) → h_buf (O_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_oproj_pso,
            o_weight.metal_buffer(),
            o_weight.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.h_buf,
            0,
            cached.gemv_bias_oproj_m,
            cached.gemv_bias_oproj_k,
            x.metal_buffer(),
            x.offset() as u64,
            cached.gemv_bias_oproj_grid,
            cached.gemv_bias_oproj_tg,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.h_buf)]);

        // D5: fused_rms_gemv(h, norm2_w, gate_up_w) → gate_up_buf
        // Replaces D6 (rms_norm) + D7 (gemv gate_up) from 9-dispatch
        ops::fused::fused_rms_gemv_preresolved_into_encoder(
            rms_gemv_gate_up_pso,
            &cached.h_buf,
            0,
            norm2_w.metal_buffer(),
            norm2_w.offset() as u64,
            gate_up_w.metal_buffer(),
            gate_up_w.offset() as u64,
            &cached.gate_up_buf,
            0,
            cached.gemv_gate_up_m,
            cached.gemv_gate_up_k,
            cached.rms_norm_eps,
            cached.norm2_w_stride,
            cached.fused_rms_gemv_gate_up_grid,
            cached.fused_rms_gemv_gate_up_tg,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.gate_up_buf)]);

        // D6: fused_swiglu_down(down_w, gate_up, h) → out_buf
        // Replaces D8 (silu_mul) + D9 (gemv_bias down) from 9-dispatch
        ops::fused::fused_swiglu_down_preresolved_into_encoder(
            swiglu_down_pso,
            down_w.metal_buffer(),
            down_w.offset() as u64,
            &cached.gate_up_buf,
            0,
            &cached.out_buf,
            0,
            cached.gemv_bias_down_m,
            cached.gemv_bias_down_k,
            &cached.h_buf,
            0,
            cached.fused_swiglu_down_grid,
            cached.fused_swiglu_down_tg,
            encoder,
        );

        encoder.end_encoding();

        Ok(Array::new(
            cached.out_buf.clone(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            cached.dtype,
            0,
        ))
    }

    /// Selective-fusion forward: individually toggle each fusion for A/B testing.
    ///
    /// - `fuse_rms_qkv`: use fused_rms_gemv for D1+D2 (rms_norm + QKV GEMV)
    /// - `fuse_rms_gate_up`: use fused_rms_gemv for D6+D7 (rms_norm + gate_up GEMV)
    /// - `fuse_swiglu_down`: use fused_swiglu_down for D8+D9 (SwiGLU + down GEMV)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_selective_fusion(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        cached: &CachedDecode,
        cb: &metal::CommandBufferRef,
        fuse_rms_qkv: bool,
        fuse_rms_gate_up: bool,
        fuse_swiglu_down: bool,
    ) -> Result<Array, KernelError> {
        let hidden_size = cached.hidden_size;
        let num_heads = cached.num_heads;
        let num_kv_heads = cached.num_kv_heads;
        let head_dim = cached.head_dim;
        let elem_size = cached.dtype.size_of();

        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: norm2_weight not loaded".into())
        })?;
        let qkv_weight = self.attention.qkv_merged_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: qkv_merged_weight not loaded".into())
        })?;
        let o_weight = self.attention.o_proj_weight().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: o_proj weight not loaded".into())
        })?;
        let (gate_up_w, down_w) = match &self.ffn {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                ..
            } => {
                let guw = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "forward_cached: gate_up_merged_weight not loaded".into(),
                    )
                })?;
                let dw = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("forward_cached: down_proj weight not loaded".into())
                })?;
                (guw, dw)
            }
            _ => {
                return Err(KernelError::InvalidShape(
                    "forward_cached only supports Gated FFN".into(),
                ))
            }
        };

        // =====================================================================
        // Single Encoder: all 9 dispatches + KV copy
        // =====================================================================
        let encoder = cb.new_compute_command_encoder();

        // D1+D2: rms_norm + QKV GEMV (conditionally fused)
        if fuse_rms_qkv {
            if let Some(pso) = &cached.fused_rms_gemv_qkv_pso {
                ops::fused::fused_rms_gemv_preresolved_into_encoder(
                    pso,
                    x.metal_buffer(),
                    x.offset() as u64,
                    norm1_w.metal_buffer(),
                    norm1_w.offset() as u64,
                    qkv_weight.metal_buffer(),
                    qkv_weight.offset() as u64,
                    &cached.qkv_buf,
                    0,
                    cached.gemv_qkv_m,
                    cached.gemv_qkv_k,
                    cached.rms_norm_eps,
                    cached.norm1_w_stride,
                    cached.fused_rms_gemv_qkv_grid,
                    cached.fused_rms_gemv_qkv_tg,
                    encoder,
                );
            } else {
                // Fallback: unfused
                ops::rms_norm::rms_norm_preresolved_into_encoder(
                    &cached.rms_norm_pso,
                    x.metal_buffer(),
                    x.offset() as u64,
                    norm1_w.metal_buffer(),
                    norm1_w.offset() as u64,
                    &cached.normed_buf,
                    0,
                    cached.rms_axis_size,
                    cached.rms_norm_eps,
                    cached.norm1_w_stride,
                    1,
                    1,
                    encoder,
                );
                encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed_buf)]);
                ops::gemv::gemv_preresolved_into_encoder(
                    &cached.gemv_qkv_pso,
                    qkv_weight.metal_buffer(),
                    qkv_weight.offset() as u64,
                    &cached.normed_buf,
                    0,
                    &cached.qkv_buf,
                    0,
                    cached.gemv_qkv_m,
                    cached.gemv_qkv_k,
                    cached.gemv_qkv_grid,
                    cached.gemv_qkv_tg,
                    encoder,
                );
            }
        } else {
            // Unfused D1 + D2
            ops::rms_norm::rms_norm_preresolved_into_encoder(
                &cached.rms_norm_pso,
                x.metal_buffer(),
                x.offset() as u64,
                norm1_w.metal_buffer(),
                norm1_w.offset() as u64,
                &cached.normed_buf,
                0,
                cached.rms_axis_size,
                cached.rms_norm_eps,
                cached.norm1_w_stride,
                1,
                1,
                encoder,
            );
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed_buf)]);
            ops::gemv::gemv_preresolved_into_encoder(
                &cached.gemv_qkv_pso,
                qkv_weight.metal_buffer(),
                qkv_weight.offset() as u64,
                &cached.normed_buf,
                0,
                &cached.qkv_buf,
                0,
                cached.gemv_qkv_m,
                cached.gemv_qkv_k,
                cached.gemv_qkv_grid,
                cached.gemv_qkv_tg,
                encoder,
            );
        }

        // D3: rope -> rope_buf (or use qkv_buf directly if no RoPE)
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let total_rope_heads = num_heads + num_kv_heads;
        let rope_offset = cache.seq_len as u32;

        let (qk_buf_ref, qk_offset, v_buf_ref, v_offset): (
            &metal::BufferRef,
            u64,
            &metal::BufferRef,
            u64,
        );

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.qkv_buf)]);
            ops::rope::rope_ext_preresolved_into_encoder(
                &cached.rope_pso,
                &cached.qkv_buf,
                0,
                cos.metal_buffer(),
                cos.offset() as u64,
                sin.metal_buffer(),
                sin.offset() as u64,
                &cached.rope_buf,
                0,
                1,
                head_dim as u32,
                rope_offset,
                1.0,
                0,
                1,
                total_rope_heads as u64,
                encoder,
            );
            qk_buf_ref = &cached.rope_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        } else {
            qk_buf_ref = &cached.qkv_buf;
            qk_offset = 0;
            v_buf_ref = &cached.qkv_buf;
            v_offset = ((q_dim + k_dim) * elem_size) as u64;
        }

        // Memory barrier: ensure rope/qkv output visible to KV copy
        encoder.memory_barrier_with_resources(&[buf_as_resource(qk_buf_ref)]);

        // KV cache append using direct buffer refs (Optimization B)
        let k_roped_flat_offset = qk_offset + (q_dim * elem_size) as u64;
        cache.append_direct_into_encoder(
            qk_buf_ref,
            k_roped_flat_offset,
            v_buf_ref,
            v_offset,
            1,
            &cached.copy_pso,
            cached.copy_max_tg,
            encoder,
        )?;

        // Memory barrier on KV slab buffers: KV copy -> SDPA read dependency
        // (replaces encoder boundary from the 2-encoder path)
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("forward_cached: no values slab after append".into())
        })?;
        encoder.memory_barrier_with_resources(&[
            buf_as_resource(k_slab.metal_buffer()),
            buf_as_resource(v_slab.metal_buffer()),
        ]);

        let seq_len = cache.seq_len;
        let max_seq = cache.max_seq_len();

        // D4: SDPA decode -> attn_out_buf
        ops::sdpa::sdpa_decode_preresolved_into_encoder(
            &cached.sdpa_pso,
            qk_buf_ref,
            qk_offset,
            k_slab.metal_buffer(),
            k_slab.offset() as u64,
            v_slab.metal_buffer(),
            v_slab.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.dummy_mask_buf,
            0,
            num_heads as u32,
            num_kv_heads as u32,
            seq_len as u32,
            head_dim as u32,
            0,
            max_seq as u32,
            cached.scale,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.attn_out_buf)]);

        // D5: gemv_bias(W_o, attn, x) -> h_buf (O_proj + residual add)
        ops::gemv::gemv_bias_preresolved_into_encoder(
            &cached.gemv_bias_oproj_pso,
            o_weight.metal_buffer(),
            o_weight.offset() as u64,
            &cached.attn_out_buf,
            0,
            &cached.h_buf,
            0,
            cached.gemv_bias_oproj_m,
            cached.gemv_bias_oproj_k,
            x.metal_buffer(),
            x.offset() as u64,
            cached.gemv_bias_oproj_grid,
            cached.gemv_bias_oproj_tg,
            encoder,
        );
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.h_buf)]);

        // D6+D7: rms_norm + gate_up GEMV (conditionally fused)
        if fuse_rms_gate_up {
            if let Some(pso) = &cached.fused_rms_gemv_gate_up_pso {
                ops::fused::fused_rms_gemv_preresolved_into_encoder(
                    pso,
                    &cached.h_buf,
                    0,
                    norm2_w.metal_buffer(),
                    norm2_w.offset() as u64,
                    gate_up_w.metal_buffer(),
                    gate_up_w.offset() as u64,
                    &cached.gate_up_buf,
                    0,
                    cached.gemv_gate_up_m,
                    cached.gemv_gate_up_k,
                    cached.rms_norm_eps,
                    cached.norm2_w_stride,
                    cached.fused_rms_gemv_gate_up_grid,
                    cached.fused_rms_gemv_gate_up_tg,
                    encoder,
                );
            } else {
                // Fallback: unfused
                ops::rms_norm::rms_norm_preresolved_into_encoder(
                    &cached.rms_norm2_pso,
                    &cached.h_buf,
                    0,
                    norm2_w.metal_buffer(),
                    norm2_w.offset() as u64,
                    &cached.normed2_buf,
                    0,
                    cached.rms_axis_size,
                    cached.rms_norm_eps,
                    cached.norm2_w_stride,
                    1,
                    1,
                    encoder,
                );
                encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed2_buf)]);
                ops::gemv::gemv_preresolved_into_encoder(
                    &cached.gemv_gate_up_pso,
                    gate_up_w.metal_buffer(),
                    gate_up_w.offset() as u64,
                    &cached.normed2_buf,
                    0,
                    &cached.gate_up_buf,
                    0,
                    cached.gemv_gate_up_m,
                    cached.gemv_gate_up_k,
                    cached.gemv_gate_up_grid,
                    cached.gemv_gate_up_tg,
                    encoder,
                );
            }
        } else {
            // Unfused D6 + D7
            ops::rms_norm::rms_norm_preresolved_into_encoder(
                &cached.rms_norm2_pso,
                &cached.h_buf,
                0,
                norm2_w.metal_buffer(),
                norm2_w.offset() as u64,
                &cached.normed2_buf,
                0,
                cached.rms_axis_size,
                cached.rms_norm_eps,
                cached.norm2_w_stride,
                1,
                1,
                encoder,
            );
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.normed2_buf)]);
            ops::gemv::gemv_preresolved_into_encoder(
                &cached.gemv_gate_up_pso,
                gate_up_w.metal_buffer(),
                gate_up_w.offset() as u64,
                &cached.normed2_buf,
                0,
                &cached.gate_up_buf,
                0,
                cached.gemv_gate_up_m,
                cached.gemv_gate_up_k,
                cached.gemv_gate_up_grid,
                cached.gemv_gate_up_tg,
                encoder,
            );
        }
        encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.gate_up_buf)]);

        // D8+D9: SwiGLU + down GEMV (conditionally fused)
        if fuse_swiglu_down {
            if let Some(pso) = &cached.fused_swiglu_down_pso {
                ops::fused::fused_swiglu_down_preresolved_into_encoder(
                    pso,
                    down_w.metal_buffer(),
                    down_w.offset() as u64,
                    &cached.gate_up_buf,
                    0,
                    &cached.out_buf,
                    0,
                    cached.gemv_bias_down_m,
                    cached.gemv_bias_down_k,
                    &cached.h_buf,
                    0,
                    cached.fused_swiglu_down_grid,
                    cached.fused_swiglu_down_tg,
                    encoder,
                );
            } else {
                // Fallback: unfused
                let gate_dim = cached.intermediate_dim;
                ops::fused::fused_silu_mul_preresolved_into_encoder(
                    &cached.silu_mul_pso,
                    &cached.gate_up_buf,
                    0,
                    &cached.gate_up_buf,
                    (gate_dim * elem_size) as u64,
                    &cached.silu_buf,
                    0,
                    cached.silu_numel,
                    cached.silu_elems_per_thread,
                    encoder,
                );
                encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.silu_buf)]);
                ops::gemv::gemv_bias_preresolved_into_encoder(
                    &cached.gemv_bias_down_pso,
                    down_w.metal_buffer(),
                    down_w.offset() as u64,
                    &cached.silu_buf,
                    0,
                    &cached.out_buf,
                    0,
                    cached.gemv_bias_down_m,
                    cached.gemv_bias_down_k,
                    &cached.h_buf,
                    0,
                    cached.gemv_bias_down_grid,
                    cached.gemv_bias_down_tg,
                    encoder,
                );
            }
        } else {
            // Unfused D8 + D9
            let gate_dim = cached.intermediate_dim;
            ops::fused::fused_silu_mul_preresolved_into_encoder(
                &cached.silu_mul_pso,
                &cached.gate_up_buf,
                0,
                &cached.gate_up_buf,
                (gate_dim * elem_size) as u64,
                &cached.silu_buf,
                0,
                cached.silu_numel,
                cached.silu_elems_per_thread,
                encoder,
            );
            encoder.memory_barrier_with_resources(&[buf_as_resource(&cached.silu_buf)]);
            ops::gemv::gemv_bias_preresolved_into_encoder(
                &cached.gemv_bias_down_pso,
                down_w.metal_buffer(),
                down_w.offset() as u64,
                &cached.silu_buf,
                0,
                &cached.out_buf,
                0,
                cached.gemv_bias_down_m,
                cached.gemv_bias_down_k,
                &cached.h_buf,
                0,
                cached.gemv_bias_down_grid,
                cached.gemv_bias_down_tg,
                encoder,
            );
        }

        encoder.end_encoding();

        Ok(Array::new(
            cached.out_buf.clone(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            cached.dtype,
            0,
        ))
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

    /// Single-CB forward pass for prefill (seq_len >= 1).
    ///
    /// Encodes the entire transformer block (attention + FFN) into one command
    /// buffer with no intermediate commit/wait. This eliminates ~54 sync points
    /// per layer that the standard `forward()` path incurs.
    ///
    /// **Prerequisites:**
    /// - `prepare_weights_for_graph()` must have been called so that all
    ///   projection weights (Q/K/V/O, gate/up/down) are pre-transposed.
    /// - The KV cache must be pre-allocated (`max_seq_len > 0`).
    ///
    /// Does NOT commit or wait on the CB — the caller manages its lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_single_cb(
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

        // Attention (all in same CB — norm + Q/K/V + deinterleave/RoPE
        //   + cache append + SDPA + head concat + O_proj)
        let attn_out = self.attention.forward_prefill_single_cb(
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

    /// ExecGraph-based prefill forward: 1 CB per block, GPU-side chaining.
    ///
    /// Like `forward_prefill_single_cb` but uses ExecGraph to submit
    /// the CB with GPU-side event signaling, enabling inter-layer
    /// overlap when the GPU has spare execution bandwidth.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_graph(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
    ) -> Result<Array, KernelError> {
        let cb = graph.command_buffer();
        let result =
            self.forward_prefill_single_cb(x, cos_freqs, sin_freqs, mask, cache, registry, cb)?;
        let t = graph.submit_batch();
        graph.wait_for(t);
        Ok(result)
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

// ---------------------------------------------------------------------------
// CachedDecode — pre-resolved PSOs and pre-allocated scratch buffers
// ---------------------------------------------------------------------------

/// Pre-resolved pipeline states and pre-allocated scratch buffers for
/// zero-overhead decode (seq_len=1) forward passes.
///
/// Each `TransformerBlock` layer needs its own `CachedDecode` because scratch
/// buffers are used across encoder boundaries and cannot be shared.
/// # Safety
///
/// This struct is NOT reentrant: scratch buffers are reused across calls.
/// The returned output Array is a view of an internal buffer — it must be
/// consumed (passed to the next layer) before calling `forward_cached_2encoder_9dispatch`
/// again on the same `CachedDecode`. This is always the case in serial decode.
pub struct CachedDecode {
    // --- Pre-resolved PSOs ---
    pub rms_norm_pso: metal::ComputePipelineState,
    pub gemv_qkv_pso: metal::ComputePipelineState,
    pub rope_pso: metal::ComputePipelineState,
    pub copy_pso: metal::ComputePipelineState,
    pub sdpa_pso: metal::ComputePipelineState,
    pub gemv_bias_oproj_pso: metal::ComputePipelineState,
    pub rms_norm2_pso: metal::ComputePipelineState,
    pub gemv_gate_up_pso: metal::ComputePipelineState,
    pub silu_mul_pso: metal::ComputePipelineState,
    pub gemv_bias_down_pso: metal::ComputePipelineState,

    // --- Pre-allocated scratch buffers ---
    /// D1 output: rms_norm result [1, hidden_size]
    pub normed_buf: metal::Buffer,
    /// D2 output: QKV result [qkv_dim]
    pub qkv_buf: metal::Buffer,
    /// D3 output: rope result [total_rope_heads * head_dim]
    pub rope_buf: metal::Buffer,
    /// D4 output: SDPA result [num_heads * head_dim]
    pub attn_out_buf: metal::Buffer,
    /// D5 output: O_proj+residual result [hidden_size]
    pub h_buf: metal::Buffer,
    /// D6 output: rms_norm2 result [1, hidden_size]
    pub normed2_buf: metal::Buffer,
    /// D7 output: gate_up result [2 * intermediate_dim]
    pub gate_up_buf: metal::Buffer,
    /// D8 output: silu_mul result [intermediate_dim]
    pub silu_buf: metal::Buffer,
    /// D9 output: down_proj+residual result [hidden_size]
    pub out_buf: metal::Buffer,
    /// Dummy buffer for SDPA when no mask is provided
    pub dummy_mask_buf: metal::Buffer,

    // --- Cached dimensions ---
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub dtype: DType,
    pub rms_norm_eps: f32,

    // --- Pre-computed dispatch geometries ---
    pub gemv_qkv_grid: metal::MTLSize,
    pub gemv_qkv_tg: metal::MTLSize,
    pub gemv_qkv_m: u32,
    pub gemv_qkv_k: u32,
    pub gemv_bias_oproj_grid: metal::MTLSize,
    pub gemv_bias_oproj_tg: metal::MTLSize,
    pub gemv_bias_oproj_m: u32,
    pub gemv_bias_oproj_k: u32,
    pub gemv_gate_up_grid: metal::MTLSize,
    pub gemv_gate_up_tg: metal::MTLSize,
    pub gemv_gate_up_m: u32,
    pub gemv_gate_up_k: u32,
    pub gemv_bias_down_grid: metal::MTLSize,
    pub gemv_bias_down_tg: metal::MTLSize,
    pub gemv_bias_down_m: u32,
    pub gemv_bias_down_k: u32,
    pub silu_numel: u32,
    pub silu_elems_per_thread: u64,
    pub rms_axis_size: u32,
    pub scale: f32,
    pub norm1_w_stride: u32,
    pub norm2_w_stride: u32,

    // --- Pre-cached threadgroup sizes (Optimization C) ---
    /// Pre-computed SDPA threadgroup size: min(DECODE_THREADS, pso.max_total_threads)
    pub sdpa_tg_size: u64,
    /// Pre-computed RMS norm threadgroup size: min(1024, pso.max_total_threads)
    pub rms_tg_size: u64,
    /// Pre-computed RMS norm2 threadgroup size: min(1024, pso.max_total_threads)
    pub rms2_tg_size: u64,
    /// Pre-computed copy threadgroup size: min(max_threads, count)
    pub copy_max_tg: u64,

    // --- Fused PSOs (Phase 10) ---
    pub fused_rms_gemv_qkv_pso: Option<metal::ComputePipelineState>,
    pub fused_rms_gemv_gate_up_pso: Option<metal::ComputePipelineState>,
    pub fused_swiglu_down_pso: Option<metal::ComputePipelineState>,

    // --- Fused dispatch geometries ---
    pub fused_rms_gemv_qkv_grid: metal::MTLSize,
    pub fused_rms_gemv_qkv_tg: metal::MTLSize,
    pub fused_rms_gemv_gate_up_grid: metal::MTLSize,
    pub fused_rms_gemv_gate_up_tg: metal::MTLSize,
    pub fused_swiglu_down_grid: metal::MTLSize,
    pub fused_swiglu_down_tg: metal::MTLSize,
}

impl CachedDecode {
    /// Build a CachedDecode for a single transformer layer.
    ///
    /// Resolves all PSOs and allocates all scratch buffers up-front.
    /// `block` must have all weights loaded and merged (prepare_merged_qkv, prepare_merged_gate_up).
    #[allow(clippy::too_many_arguments)]
    pub fn new(block: &TransformerBlock, registry: &KernelRegistry) -> Result<Self, KernelError> {
        let attn = &block.attention;
        let num_heads = attn.num_heads();
        let num_kv_heads = attn.num_kv_heads();
        let head_dim = attn.head_dim();
        let hidden_size = attn.hidden_size();
        let dtype = block
            .norm1_weight
            .as_ref()
            .ok_or_else(|| {
                KernelError::InvalidShape("CachedDecode: norm1_weight not loaded".into())
            })?
            .dtype();
        let rms_norm_eps = block.rms_norm_eps;
        let norm1_w_stride = block.norm1_weight.as_ref().unwrap().strides()[0] as u32;
        let norm2_w_stride = block
            .norm2_weight
            .as_ref()
            .ok_or_else(|| {
                KernelError::InvalidShape("CachedDecode: norm2_weight not loaded".into())
            })?
            .strides()[0] as u32;

        let intermediate_dim = match &block.ffn {
            FeedForward::Gated {
                gate_up_merged_weight,
                down_proj,
                gate_proj,
                ..
            } => {
                let guw = gate_up_merged_weight.as_ref().ok_or_else(|| {
                    KernelError::InvalidShape(
                        "CachedDecode: gate_up_merged_weight not loaded".into(),
                    )
                })?;
                let _dw = down_proj.weight().ok_or_else(|| {
                    KernelError::InvalidShape("CachedDecode: down_proj weight not loaded".into())
                })?;
                gate_proj
                    .weight()
                    .map(|w| w.shape()[0])
                    .unwrap_or(guw.shape()[0] / 2)
            }
            _ => {
                return Err(KernelError::InvalidShape(
                    "CachedDecode only supports Gated FFN".into(),
                ))
            }
        };

        let dev = registry.device().raw();
        let elem_size = dtype.size_of();

        // --- Resolve PSOs ---
        let rms_norm_pso = {
            let kname = ops::rms_norm::rms_norm_kernel_name_for(dtype, hidden_size)?;
            registry.get_pipeline(kname, dtype)?
        };

        let qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim;
        let gemv_qkv_pso = {
            let kname = ops::gemv::gemv_kernel_name(dtype, qkv_dim as u32)?;
            registry.get_pipeline(kname, dtype)?
        };

        let rope_pso = {
            let kname = ops::rope::rope_table_kernel_name(dtype)?;
            registry.get_pipeline(kname, dtype)?
        };

        let copy_pso = {
            let kname = match dtype {
                DType::Float32 => "copy_f32",
                DType::Float16 => "copy_f16",
                DType::Bfloat16 => "copy_bf16",
                _ => {
                    return Err(KernelError::NotFound(format!(
                        "copy: unsupported dtype {:?}",
                        dtype
                    )))
                }
            };
            registry.get_pipeline(kname, dtype)?
        };

        let sdpa_pso = {
            let kname = ops::sdpa::sdpa_decode_kernel_name(dtype)?;
            let constants = ops::sdpa::sdpa_decode_constants(head_dim, false);
            registry.get_pipeline_with_constants(kname, dtype, &constants)?
        };

        let gemv_bias_oproj_pso = {
            let kname = ops::gemv::gemv_bias_kernel_name(dtype, hidden_size as u32)?;
            registry.get_pipeline(kname, dtype)?
        };

        let rms_norm2_pso = {
            let kname = ops::rms_norm::rms_norm_kernel_name_for(dtype, hidden_size)?;
            registry.get_pipeline(kname, dtype)?
        };

        let gate_up_total = 2 * intermediate_dim;
        let gemv_gate_up_pso = {
            let kname = ops::gemv::gemv_kernel_name(dtype, gate_up_total as u32)?;
            registry.get_pipeline(kname, dtype)?
        };

        let silu_mul_pso = {
            let kname = ops::fused::silu_gate_kernel_name(dtype)?;
            registry.get_pipeline(kname, dtype)?
        };

        let gemv_bias_down_pso = {
            let kname = ops::gemv::gemv_bias_kernel_name(dtype, hidden_size as u32)?;
            registry.get_pipeline(kname, dtype)?
        };

        // --- Pre-compute dispatch geometries ---
        let (gemv_qkv_grid, gemv_qkv_tg) =
            ops::gemv::gemv_dispatch_sizes(qkv_dim as u32, &gemv_qkv_pso);
        let (gemv_bias_oproj_grid, gemv_bias_oproj_tg) =
            ops::gemv::gemv_dispatch_sizes(hidden_size as u32, &gemv_bias_oproj_pso);
        let (gemv_gate_up_grid, gemv_gate_up_tg) =
            ops::gemv::gemv_dispatch_sizes(gate_up_total as u32, &gemv_gate_up_pso);
        let (gemv_bias_down_grid, gemv_bias_down_tg) =
            ops::gemv::gemv_dispatch_sizes(hidden_size as u32, &gemv_bias_down_pso);

        let silu_elems_per_thread = ops::fused::silu_gate_elems_per_thread(dtype);

        // --- Fused PSOs (Phase 10) ---
        let fused_rms_gemv_qkv_pso = {
            let kname = ops::fused::fused_rms_gemv_kernel_name(dtype, qkv_dim as u32);
            kname
                .ok()
                .and_then(|k| registry.get_pipeline(k, dtype).ok())
        };
        let fused_rms_gemv_gate_up_pso = {
            let kname = ops::fused::fused_rms_gemv_kernel_name(dtype, gate_up_total as u32);
            kname
                .ok()
                .and_then(|k| registry.get_pipeline(k, dtype).ok())
        };
        let fused_swiglu_down_pso = {
            let kname = ops::fused::fused_swiglu_down_kernel_name(dtype, hidden_size as u32);
            kname
                .ok()
                .and_then(|k| registry.get_pipeline(k, dtype).ok())
        };

        let (fused_rms_gemv_qkv_grid, fused_rms_gemv_qkv_tg) = fused_rms_gemv_qkv_pso
            .as_ref()
            .map(|pso| ops::fused::fused_rms_gemv_dispatch_sizes(qkv_dim as u32, pso))
            .unwrap_or((metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1)));
        let (fused_rms_gemv_gate_up_grid, fused_rms_gemv_gate_up_tg) = fused_rms_gemv_gate_up_pso
            .as_ref()
            .map(|pso| ops::fused::fused_rms_gemv_dispatch_sizes(gate_up_total as u32, pso))
            .unwrap_or((metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1)));
        let (fused_swiglu_down_grid, fused_swiglu_down_tg) = fused_swiglu_down_pso
            .as_ref()
            .map(|pso| ops::fused::fused_swiglu_down_dispatch_sizes(hidden_size as u32, pso))
            .unwrap_or((metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1)));

        // --- Pre-allocate scratch buffers ---
        let normed_buf = dev.new_buffer(
            (hidden_size * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let qkv_buf = dev.new_buffer(
            (qkv_dim * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let total_rope_heads = num_heads + num_kv_heads;
        let rope_buf = dev.new_buffer(
            (total_rope_heads * head_dim * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let attn_out_buf = dev.new_buffer(
            (num_heads * head_dim * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let h_buf = dev.new_buffer(
            (hidden_size * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let normed2_buf = dev.new_buffer(
            (hidden_size * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let gate_up_buf = dev.new_buffer(
            (gate_up_total * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let silu_buf = dev.new_buffer(
            (intermediate_dim * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let out_buf = dev.new_buffer(
            (hidden_size * elem_size) as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let dummy_mask_buf = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Pre-cache threadgroup sizes (Optimization C)
        let sdpa_tg_size = std::cmp::min(256u64, sdpa_pso.max_total_threads_per_threadgroup());
        let rms_tg_size = std::cmp::min(1024u64, rms_norm_pso.max_total_threads_per_threadgroup());
        let rms2_tg_size =
            std::cmp::min(1024u64, rms_norm2_pso.max_total_threads_per_threadgroup());
        let copy_max_tg = copy_pso.max_total_threads_per_threadgroup();

        Ok(Self {
            rms_norm_pso,
            gemv_qkv_pso,
            rope_pso,
            copy_pso,
            sdpa_pso,
            gemv_bias_oproj_pso,
            rms_norm2_pso,
            gemv_gate_up_pso,
            silu_mul_pso,
            gemv_bias_down_pso,
            normed_buf,
            qkv_buf,
            rope_buf,
            attn_out_buf,
            h_buf,
            normed2_buf,
            gate_up_buf,
            silu_buf,
            out_buf,
            dummy_mask_buf,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            dtype,
            rms_norm_eps,
            gemv_qkv_grid,
            gemv_qkv_tg,
            gemv_qkv_m: qkv_dim as u32,
            gemv_qkv_k: hidden_size as u32,
            gemv_bias_oproj_grid,
            gemv_bias_oproj_tg,
            gemv_bias_oproj_m: hidden_size as u32,
            gemv_bias_oproj_k: hidden_size as u32,
            gemv_gate_up_grid,
            gemv_gate_up_tg,
            gemv_gate_up_m: gate_up_total as u32,
            gemv_gate_up_k: hidden_size as u32,
            gemv_bias_down_grid,
            gemv_bias_down_tg,
            gemv_bias_down_m: hidden_size as u32,
            gemv_bias_down_k: intermediate_dim as u32,
            silu_numel: intermediate_dim as u32,
            silu_elems_per_thread,
            rms_axis_size: hidden_size as u32,
            scale,
            norm1_w_stride,
            norm2_w_stride,
            sdpa_tg_size,
            rms_tg_size,
            rms2_tg_size,
            copy_max_tg,
            fused_rms_gemv_qkv_pso,
            fused_rms_gemv_gate_up_pso,
            fused_swiglu_down_pso,
            fused_rms_gemv_qkv_grid,
            fused_rms_gemv_qkv_tg,
            fused_rms_gemv_gate_up_grid,
            fused_rms_gemv_gate_up_tg,
            fused_swiglu_down_grid,
            fused_swiglu_down_tg,
        })
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

    /// ExecGraph prefill forward: token IDs -> logits.
    ///
    /// Uses 1 CB per layer via `forward_prefill_single_cb`, chained
    /// with GPU-side events through ExecGraph. CPU blocks only once
    /// at the end.
    ///
    /// **Requires** `prepare_weights_for_graph()` to have been called.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_graph(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: &mut [LayerKvCache],
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

        // Validate cache vector length matches number of layers
        if cache.len() != self.layers.len() {
            return Err(KernelError::InvalidShape(format!(
                "TransformerModel: cache has {} entries but model has {} layers",
                cache.len(),
                self.layers.len()
            )));
        }

        // High encoder limit since each layer uses only 1 CB
        let mut graph = ExecGraph::new(queue, event, 64);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_prefill_graph(
                &x,
                cos_freqs,
                sin_freqs,
                mask,
                &mut cache[i],
                registry,
                &mut graph,
            )?;
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
            .map_err(|e| KernelError::InvalidShape(format!("prefill graph sync: {e}")))?;

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Distributed (Tensor Parallel) forward paths
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
impl TransformerBlock {
    /// TP-aware forward pass for one transformer block.
    ///
    /// Assumes weights are **pre-sharded** across ranks (Megatron-LM pattern):
    /// - Attention QKV projections: column-parallel (local matmul, each rank owns a head shard)
    /// - Attention O projection: row-parallel (partial output → allreduce)
    /// - FFN gate/up: column-parallel (local matmul)
    /// - FFN down: row-parallel (partial output → allreduce)
    ///
    /// Two allreduce operations per block:
    /// 1. After attention (o_proj partial sum)
    /// 2. After FFN (down_proj partial sum)
    ///
    /// `x`: `[seq_len, hidden_size]`
    /// Returns: `[seq_len, hidden_size]`
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_group(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        group: &rmlx_distributed::group::Group,
        device: &metal::Device,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // ── Pre-attention RMS norm (local — full norm weights on every rank) ──
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // ── Attention (with sharded Q/K/V/O weights) ──
        // o_proj output is a partial sum when weight is row-sharded.
        let attn_partial = self
            .attention
            .forward(&normed, cos_freqs, sin_freqs, mask, cache, registry, queue)?;

        // Allreduce attention output across TP ranks
        let attn_out = group.allreduce_sum(&attn_partial, device).map_err(|e| {
            KernelError::InvalidShape(format!("TP allreduce after attention failed: {e}"))
        })?;

        // ── Residual connection: x + attn_out ──
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // ── Pre-FFN RMS norm (local) ──
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        // ── FFN (with sharded gate/up/down weights) ──
        // down_proj output is a partial sum when weight is row-sharded.
        let ffn_partial = self.ffn.forward(&normed2, registry, queue)?;

        // Allreduce FFN output across TP ranks
        let ffn_out = group.allreduce_sum(&ffn_partial, device).map_err(|e| {
            KernelError::InvalidShape(format!("TP allreduce after FFN failed: {e}"))
        })?;

        // ── Residual connection: h + ffn_out ──
        ops::binary::add(registry, &h, &ffn_out, queue)
    }
}

#[cfg(feature = "distributed")]
impl TransformerModel {
    /// Shard all layer weights for Tensor Parallelism.
    ///
    /// Must be called once after weights are loaded and before `forward_with_group()`.
    ///
    /// Shards:
    /// - Attention QKV: column-parallel (each rank gets `num_heads / world_size` heads)
    /// - Attention O: row-parallel (input cols sharded)
    /// - FFN gate/up: column-parallel (output sharded)
    /// - FFN down: row-parallel (input cols sharded)
    ///
    /// Norm weights and embedding weights are NOT sharded (replicated on all ranks).
    /// LM head is NOT sharded (each rank computes full logits).
    pub fn shard_weights_for_tp(
        &mut self,
        group: &rmlx_distributed::group::Group,
    ) -> Result<(), KernelError> {
        let world_size = group.size() as u32;
        let rank = group.local_rank();

        if world_size <= 1 {
            return Ok(());
        }

        for layer in &mut self.layers {
            layer.shard_for_tp(rank, world_size)?;
        }

        Ok(())
    }

    /// Distributed forward pass: token IDs → logits with Tensor Parallelism.
    ///
    /// Each rank holds pre-sharded weights (column-parallel for QKV/gate/up,
    /// row-parallel for O/down). Allreduce is inserted after attention and FFN
    /// in each transformer block.
    ///
    /// **Embedding**: each rank performs the full embedding lookup (embedding
    /// weights are replicated, not sharded).
    ///
    /// **LM head**: each rank computes logits locally. If the LM head weight is
    /// column-sharded, the caller is responsible for gathering the output.
    /// For tied embeddings (LM head = embedding weight), weights are replicated
    /// and the output is complete on every rank.
    ///
    /// `token_ids`: input token indices
    /// `cache`: optional per-layer KV caches for incremental decoding
    /// Returns: `[seq_len, vocab_size]` logits
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_group(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut Vec<LayerKvCache>>,
        group: &rmlx_distributed::group::Group,
        device: &metal::Device,
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

        // ── Embedding lookup (replicated on every rank) ──
        let mut x = embedding.forward(token_ids, registry, queue)?;

        // ── Validate cache length ──
        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "TransformerModel: cache has {} entries but model has {} layers",
                    c.len(),
                    self.layers.len()
                )));
            }
        }

        // ── Transformer layers with TP allreduce ──
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            x = layer.forward_with_group(
                &x,
                cos_freqs,
                sin_freqs,
                mask,
                layer_cache,
                group,
                device,
                registry,
                queue,
            )?;
        }

        // ── Final RMS norm (local — replicated weights) ──
        x = ops::rms_norm::rms_norm(registry, &x, final_norm, self.config.rms_norm_eps, queue)?;

        // ── LM head (local) ──
        lm_head.forward(&x, registry, queue)
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
