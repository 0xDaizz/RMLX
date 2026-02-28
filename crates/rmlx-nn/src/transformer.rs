//! Transformer block: attention + MLP (or MoE).

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::attention::Attention;
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
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".into());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if self.num_kv_heads > self.num_heads {
            return Err("num_kv_heads > num_heads".into());
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err("num_heads must be divisible by num_kv_heads".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        Ok(())
    }
}

/// Feed-forward network: either dense (SwiGLU) or MoE.
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
                // SwiGLU: down(gate(x) * up(x))
                // (silu omitted here; proper silu kernel to be added to rmlx-core)
                let gate_out = gate_proj.forward(x, registry, queue)?;
                let up_out = up_proj.forward(x, registry, queue)?;
                let hidden = ops::binary::mul(registry, &gate_out, &up_out, queue)?;
                down_proj.forward(&hidden, registry, queue)
            }
            FeedForward::MoE(moe) => moe.forward(x, registry, queue),
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
    pub fn new(layer_idx: usize, config: TransformerConfig) -> Self {
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
        Self {
            layer_idx,
            attention: Attention::new(attn_config),
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
        }
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
    /// Returns: [seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
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
            .forward(&normed, cos_freqs, sin_freqs, mask, registry, queue)?;

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
    pub fn new(config: TransformerConfig) -> Self {
        let num_layers = config.num_layers;
        Self {
            config,
            embedding: None,
            layers: Vec::new(),
            final_norm_weight: None,
            lm_head: None,
            num_layers,
        }
    }

    /// Create a model with all components pre-loaded.
    pub fn from_parts(
        config: TransformerConfig,
        embedding: Embedding,
        layers: Vec<TransformerBlock>,
        final_norm_weight: Array,
        lm_head: Linear,
    ) -> Self {
        let num_layers = layers.len();
        Self {
            config,
            embedding: Some(embedding),
            layers,
            final_norm_weight: Some(final_norm_weight),
            lm_head: Some(lm_head),
            num_layers,
        }
    }

    /// Forward pass: token IDs -> logits.
    ///
    /// `token_ids`: input token indices
    /// Returns: [seq_len, vocab_size] logits
    pub fn forward(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
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

        // Transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, cos_freqs, sin_freqs, mask, registry, queue)?;
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
}
