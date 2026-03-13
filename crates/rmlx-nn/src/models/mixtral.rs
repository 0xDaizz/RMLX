//! Mixtral model architecture.
//!
//! Mixtral uses sliding window attention (GQA) + MoE FFN.
//! Each layer has `SlidingWindowAttention` for the attention mechanism
//! and a `MoeLayer` for the feed-forward network.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::attention::{AttentionConfig, LayerKvCache};
use crate::embedding::Embedding;
use crate::linear::Linear;
use crate::moe::{MoeConfig, MoeLayer};
use crate::sliding_window::{SlidingWindowAttention, SlidingWindowAttentionConfig};
use crate::transformer::{FeedForward, FeedForwardType, TransformerConfig};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue};

/// Mixtral 8x7B configuration preset.
pub fn mixtral_8x7b() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        num_layers: 32,
        vocab_size: 32000,
        max_seq_len: 32768,
        rope_theta: 1000000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::MoE {
            config: MoeConfig {
                num_experts: 8,
                num_experts_per_token: 2,
                hidden_dim: 4096,
                intermediate_dim: 14336,
                capacity_factor: 1.0,
            },
        },
    }
}

/// Mixtral-specific configuration extending `TransformerConfig` with
/// a sliding window size parameter.
pub struct MixtralConfig {
    /// Base transformer configuration (must have MoE ff_type).
    pub transformer: TransformerConfig,
    /// Sliding window size for attention. Mixtral 8x7B uses 4096.
    pub sliding_window_size: usize,
}

impl MixtralConfig {
    /// Validate the Mixtral configuration.
    pub fn validate(&self) -> Result<(), KernelError> {
        self.transformer.validate()?;
        if self.sliding_window_size == 0 {
            return Err(KernelError::InvalidShape(
                "MixtralConfig: sliding_window_size must be > 0".into(),
            ));
        }
        match &self.transformer.ff_type {
            FeedForwardType::MoE { .. } => Ok(()),
            _ => Err(KernelError::InvalidShape(
                "MixtralConfig: ff_type must be MoE for Mixtral architecture".into(),
            )),
        }
    }
}

/// Default Mixtral 8x7B full configuration.
pub fn mixtral_8x7b_full() -> MixtralConfig {
    MixtralConfig {
        transformer: mixtral_8x7b(),
        sliding_window_size: 4096,
    }
}

// ---------------------------------------------------------------------------
// Mixtral layer block
// ---------------------------------------------------------------------------

/// A single Mixtral transformer layer: sliding window attention + MoE FFN.
pub struct MixtralBlock {
    layer_idx: usize,
    attention: SlidingWindowAttention,
    ffn: FeedForward,
    norm1_weight: Option<Array>,
    norm2_weight: Option<Array>,
    rms_norm_eps: f32,
}

impl MixtralBlock {
    /// Create a block with config-only attention and FFN (no weights loaded).
    pub fn new(
        layer_idx: usize,
        sw_config: SlidingWindowAttentionConfig,
        ffn: FeedForward,
        rms_norm_eps: f32,
    ) -> Result<Self, KernelError> {
        let attention = SlidingWindowAttention::new(sw_config)?;
        Ok(Self {
            layer_idx,
            attention,
            ffn,
            norm1_weight: None,
            norm2_weight: None,
            rms_norm_eps,
        })
    }

    /// Create a block with pre-loaded components.
    pub fn from_parts(
        layer_idx: usize,
        attention: SlidingWindowAttention,
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

    /// Forward pass for one Mixtral block.
    ///
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables
    /// `cache`: optional per-layer KV cache
    /// Returns: [seq_len, hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("MixtralBlock: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("MixtralBlock: norm2_weight not loaded".into())
        })?;

        // Pre-attention RMS norm
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // Sliding window attention (mask is built internally by SlidingWindowAttention)
        let attn_out = self
            .attention
            .forward(&normed, cos_freqs, sin_freqs, cache, registry, queue)?;

        // Residual
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // Pre-FFN RMS norm
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        // MoE FFN
        let ffn_out = self.ffn.forward(&normed2, registry, queue)?;

        // Residual
        ops::binary::add(registry, &h, &ffn_out, queue)
    }

    /// Layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Access the sliding window attention layer.
    pub fn attention(&self) -> &SlidingWindowAttention {
        &self.attention
    }
}

// ---------------------------------------------------------------------------
// MixtralModel
// ---------------------------------------------------------------------------

/// Mixtral model: sliding window attention + MoE FFN.
///
/// Architecture:
/// - Token embedding
/// - N layers of: SlidingWindowAttention + MoE FFN (with RMS norms + residuals)
/// - Final RMS norm + LM head
pub struct MixtralModel {
    config: MixtralConfig,
    embedding: Option<Embedding>,
    layers: Vec<MixtralBlock>,
    final_norm_weight: Option<Array>,
    lm_head: Option<Linear>,
}

impl MixtralModel {
    /// Build a `MixtralModel` from config (no weights loaded).
    pub fn from_config(config: MixtralConfig) -> Result<Self, KernelError> {
        config.validate()?;

        let num_layers = config.transformer.num_layers;
        let rms_norm_eps = config.transformer.rms_norm_eps;

        let moe_config = match &config.transformer.ff_type {
            FeedForwardType::MoE { config: mc } => mc.clone(),
            _ => unreachable!("validate() ensures MoE"),
        };

        let base_attn = AttentionConfig {
            num_heads: config.transformer.num_heads,
            num_kv_heads: config.transformer.num_kv_heads,
            head_dim: config.transformer.head_dim,
            max_seq_len: config.transformer.max_seq_len,
            rope_theta: config.transformer.rope_theta,
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let sw_config = SlidingWindowAttentionConfig {
                base: AttentionConfig {
                    num_heads: base_attn.num_heads,
                    num_kv_heads: base_attn.num_kv_heads,
                    head_dim: base_attn.head_dim,
                    max_seq_len: base_attn.max_seq_len,
                    rope_theta: base_attn.rope_theta,
                },
                window_size: config.sliding_window_size,
            };

            let moe_layer = MoeLayer::new(moe_config.clone())?;
            let ffn = FeedForward::MoE(moe_layer);

            let block = MixtralBlock::new(i, sw_config, ffn, rms_norm_eps)?;
            layers.push(block);
        }

        Ok(Self {
            config,
            embedding: None,
            layers,
            final_norm_weight: None,
            lm_head: None,
        })
    }

    /// Forward pass: token IDs -> logits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        mut cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let embedding = self.embedding.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("MixtralModel: embedding not loaded".into())
        })?;
        let final_norm = self.final_norm_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("MixtralModel: final_norm not loaded".into())
        })?;
        let lm_head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("MixtralModel: lm_head not loaded".into()))?;

        // Validate cache length
        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "MixtralModel: cache has {} entries but model has {} layers",
                    c.len(),
                    self.layers.len()
                )));
            }
        }

        // Embedding lookup
        let mut x = embedding.forward(token_ids, registry, queue)?;

        // Run through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c[i]);
            x = layer.forward(&x, cos_freqs, sin_freqs, layer_cache, registry, queue)?;
        }

        // Final RMS norm
        x = ops::rms_norm::rms_norm(
            registry,
            &x,
            final_norm,
            self.config.transformer.rms_norm_eps,
            queue,
        )?;

        // LM head
        lm_head.forward(&x, registry, queue)
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Sliding window size.
    pub fn sliding_window_size(&self) -> usize {
        self.config.sliding_window_size
    }

    /// Access the full config.
    pub fn config(&self) -> &MixtralConfig {
        &self.config
    }

    /// Access a specific layer block.
    pub fn layer(&self, idx: usize) -> Option<&MixtralBlock> {
        self.layers.get(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_8x7b_config_valid() {
        let config = mixtral_8x7b();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_mixtral_8x7b_full_config_valid() {
        let config = mixtral_8x7b_full();
        assert!(config.validate().is_ok());
        assert_eq!(config.sliding_window_size, 4096);
    }

    #[test]
    fn test_mixtral_config_validation_errors() {
        // sliding_window_size == 0
        let bad = MixtralConfig {
            transformer: mixtral_8x7b(),
            sliding_window_size: 0,
        };
        assert!(bad.validate().is_err());

        // non-MoE ff_type
        let bad = MixtralConfig {
            transformer: TransformerConfig {
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
            },
            sliding_window_size: 256,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_mixtral_model_construction() {
        let small_config = MixtralConfig {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                num_layers: 2,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                ff_type: FeedForwardType::MoE {
                    config: MoeConfig {
                        num_experts: 4,
                        num_experts_per_token: 2,
                        hidden_dim: 128,
                        intermediate_dim: 256,
                        capacity_factor: 1.0,
                    },
                },
            },
            sliding_window_size: 64,
        };

        let model = MixtralModel::from_config(small_config).unwrap();
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.sliding_window_size(), 64);
    }

    #[test]
    fn test_mixtral_layer_structure() {
        let small_config = MixtralConfig {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                num_layers: 2,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                ff_type: FeedForwardType::MoE {
                    config: MoeConfig {
                        num_experts: 4,
                        num_experts_per_token: 2,
                        hidden_dim: 128,
                        intermediate_dim: 256,
                        capacity_factor: 1.0,
                    },
                },
            },
            sliding_window_size: 64,
        };

        let model = MixtralModel::from_config(small_config).unwrap();

        // All layers should have MoE FFN and sliding window attention
        for i in 0..2 {
            let layer = model.layer(i).unwrap();
            assert_eq!(layer.layer_idx(), i);
            assert!(matches!(layer.ffn, FeedForward::MoE(_)));
            assert_eq!(layer.attention().window_size(), 64);
        }
    }

    #[test]
    fn test_mixtral_model_rejects_non_moe() {
        let bad_config = MixtralConfig {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                num_layers: 2,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                ff_type: FeedForwardType::Gated {
                    intermediate_dim: 256,
                },
            },
            sliding_window_size: 64,
        };

        let result = MixtralModel::from_config(bad_config);
        assert!(result.is_err());
    }
}
