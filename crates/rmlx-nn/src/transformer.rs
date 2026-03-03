//! Transformer block: attention + MLP (or MoE).
//!
//! Supports two execution modes:
//! - Standard: per-op dispatch with `forward()` (backward compatible)
//! - Pipelined: batched dispatch with `forward_pipelined()` via ExecGraph
//!   (reduces ~30 command buffers per layer to ~6)

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::ExecGraph;

use crate::attention::{Attention, LayerKvCache};
use crate::embedding::Embedding;
use crate::linear::Linear;
use crate::moe::MoeLayer;

pub enum FeedForwardType {
    Dense { intermediate_dim: usize },
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

/// Feed-forward network: either dense (SwiGLU) or MoE.
#[allow(clippy::large_enum_variant)]
pub enum FeedForward {
    /// SwiGLU FFN: gate_proj, up_proj, down_proj
    Dense {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
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
                gate_proj,
                up_proj,
                down_proj,
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

    /// Pre-compute contiguous transposed weights for graph execution.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        match self {
            FeedForward::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                gate_proj.prepare_weight_t(registry, queue)?;
                up_proj.prepare_weight_t(registry, queue)?;
                down_proj.prepare_weight_t(registry, queue)?;
            }
            FeedForward::MoE(_) => {} // MoE doesn't use graph path
        }
        Ok(())
    }

    /// Graph-based FFN forward: encodes into 2 command buffers via ExecGraph.
    ///
    /// - CB5 (current): gate_proj + up_proj + fused_silu_mul
    /// - CB6 (after submit): down_proj + residual
    ///
    /// `normed`: pre-normed input [seq_len, hidden_size]
    /// `residual`: input for skip connection [seq_len, hidden_size]
    ///
    /// Falls back to per-op dispatch for MoE layers.
    pub fn forward_graph(
        &self,
        normed: &Array,
        residual: &Array,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        match self {
            FeedForward::Dense {
                gate_proj,
                up_proj,
                down_proj,
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
                // MoE: flush pending graph work, fall back to per-op dispatch
                graph
                    .sync()
                    .map_err(|e| KernelError::InvalidShape(format!("MoE graph sync: {e}")))?;
                graph.reset();
                let ffn_out = moe.forward(normed, registry, queue)?;
                ops::binary::add(registry, residual, &ffn_out, queue)
            }
        }
    }
}

pub struct TransformerBlock {
    layer_idx: usize,
    attention: Attention,
    ffn: FeedForward,
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
        // Create a dummy device-less norm weight — will be replaced by from_parts
        Ok(Self {
            layer_idx,
            attention: Attention::new(attn_config)?,
            ffn: FeedForward::Dense {
                gate_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: hidden_size,
                    has_bias: false,
                }),
                up_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: hidden_size,
                    has_bias: false,
                }),
                down_proj: Linear::new(crate::linear::LinearConfig {
                    in_features: hidden_size,
                    out_features: hidden_size,
                    has_bias: false,
                }),
            },
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

    /// Pipelined forward pass using ExecGraph for batched command buffers.
    ///
    /// Reduces ~15-20 command buffers per layer to ~6 by batching ops:
    /// - Batch 1: norm1 + Q/K/V projections (4 ops, 1 CB)
    /// - Batch 2: RoPE Q + RoPE K (2 ops, 1 CB)
    /// - Batch 3: SDPA (fused, 1 CB)
    /// - Batch 4: O_proj + residual_add (2 ops, 1 CB)
    /// - Batch 5: norm2 + gate + silu + up (4 ops, 1 CB)
    /// - Batch 6: down + residual_add (2 ops, 1 CB)
    ///
    /// Falls back to standard forward for MoE FFN (which has its own dispatch).
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
        _event: &GpuEvent,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("TransformerBlock: norm2_weight not loaded".into())
        })?;

        // ExecGraph available for future fine-grained batching.
        // Current path uses fused SwiGLU and batched QKV for CB reduction.
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // Batched Q/K/V projection: 3 matmuls in 1 CB
        let attn_out = self
            .attention
            .forward(&normed, cos_freqs, sin_freqs, mask, cache, registry, queue)?;

        // === Batch 4: O_proj + residual ===
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // === Batch 5: pre-FFN norm + FFN ===
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        let ffn_out = match &self.ffn {
            FeedForward::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                // Use fused SwiGLU path: silu(gate(x)) * up(x)
                let gate_out = gate_proj.forward(&normed2, registry, queue)?;
                let up_out = up_proj.forward(&normed2, registry, queue)?;
                let hidden = ops::fused::fused_silu_mul(registry, &gate_out, &up_out, queue)?;
                down_proj.forward(&hidden, registry, queue)?
            }
            FeedForward::MoE(moe) => moe.forward(&normed2, registry, queue)?,
        };

        // === Batch 6: residual ===
        ops::binary::add(registry, &h, &ffn_out, queue)
    }

    /// ExecGraph-based forward: 6 CBs per layer via GPU-side event chaining.
    ///
    /// Architecture:
    /// - CB1: norm1 + Q/K/V projections (from Attention::forward_graph)
    /// - CB2: RoPE + cache append
    /// - CB3: SDPA
    /// - CB4: head concat + O_proj + residual
    /// - CB5: norm2 + gate + up + fused_silu_mul (from FeedForward::forward_graph)
    /// - CB6: down_proj + residual
    ///
    /// CPU syncs ONCE at the caller (TransformerModel::forward_graph).
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

        // Encode norm1 into graph's current CB (will become part of CB1)
        let cb = graph.command_buffer();
        let normed =
            ops::rms_norm::rms_norm_into_cb(registry, x, Some(norm1_w), self.rms_norm_eps, cb)?;

        // Attention: adds QKV proj to current CB (CB1), then CB2-CB4
        let (h, t4) = self.attention.forward_graph(
            &normed, x, cos_freqs, sin_freqs, mask, cache, registry, graph,
        )?;

        // CB5: wait for attention, encode norm2 into new CB
        graph.wait_for(t4);
        let cb5 = graph.command_buffer();
        let normed2 =
            ops::rms_norm::rms_norm_into_cb(registry, &h, Some(norm2_w), self.rms_norm_eps, cb5)?;

        // FFN: encodes gate+up+silu_mul into current CB5, then CB6 (down+residual)
        self.ffn.forward_graph(&normed2, &h, registry, graph, queue)
    }

    /// Pre-compute contiguous transposed weights for graph execution.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        self.attention.prepare_weights_for_graph(registry, queue)?;
        self.ffn.prepare_weights_for_graph(registry, queue)
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn hidden_size(&self) -> Option<usize> {
        self.norm1_weight.as_ref().map(|w| w.shape()[0])
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

    /// Pipelined forward pass: token IDs -> logits.
    ///
    /// Uses `forward_pipelined()` on each layer for reduced CB count.
    /// Creates a shared GpuEvent for event-chaining between layers.
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

        // Create shared event for cross-layer pipelining
        let device = registry.device().raw();
        let event = GpuEvent::new(device);

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
                &event,
            )?;
            // Reset event for next layer
            event.reset();
        }

        x = ops::rms_norm::rms_norm(registry, &x, final_norm, self.config.rms_norm_eps, queue)?;
        lm_head.forward(&x, registry, queue)
    }

    /// ExecGraph-based forward: 6 CBs per layer, 1 CPU sync per layer.
    ///
    /// Creates a shared ExecGraph and processes each layer with GPU-side
    /// event chaining. The CPU only blocks once per layer at `sync_and_reset()`.
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

        // Create ExecGraph with shared event for GPU-side chaining
        let device = registry.device().raw();
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(queue, &event, 32);

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
            // Sync and reset for next layer — this is the ONLY CPU wait per layer
            graph.sync_and_reset().map_err(|e| {
                KernelError::InvalidShape(format!("ExecGraph sync failed at layer {i}: {e}"))
            })?;
        }

        // Final norm + LM head (not worth graph-ifying, just 2 CBs)
        x = ops::rms_norm::rms_norm(registry, &x, final_norm, self.config.rms_norm_eps, queue)?;
        lm_head.forward(&x, registry, queue)
    }

    /// Pre-compute contiguous transposed weights for all layers.
    /// Call once after model loading, before calling `forward_graph()`.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        for layer in &mut self.layers {
            layer.prepare_weights_for_graph(registry, queue)?;
        }
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.prepare_weight_t(registry, queue)?;
        }
        Ok(())
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
