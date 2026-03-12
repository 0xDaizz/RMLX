//! Qwen2 model architecture.
//!
//! Qwen2 uses standard GQA attention with SwiGLU (gated) FFN, following the
//! standard decoder-only transformer pattern. This wraps `TransformerModel`
//! with Qwen-specific config presets.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};

use crate::attention::LayerKvCache;
use crate::transformer::{FeedForwardType, TransformerConfig, TransformerModel};

/// Qwen2-7B configuration preset.
pub fn qwen2_7b() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 3584,
        num_heads: 28,
        num_kv_heads: 4,
        head_dim: 128,
        num_layers: 28,
        vocab_size: 152064,
        max_seq_len: 32768,
        rope_theta: 1000000.0,
        rms_norm_eps: 1e-6,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: 18944,
        },
    }
}

// ---------------------------------------------------------------------------
// Qwen2Model — thin wrapper over TransformerModel
// ---------------------------------------------------------------------------

/// Qwen2 model: standard GQA + SwiGLU wrapped as `TransformerModel`.
///
/// Standard decoder-only transformer with Qwen2-specific config presets
/// (hidden size, head counts, vocab size, RoPE theta, etc.).
pub struct Qwen2Model {
    inner: TransformerModel,
}

impl Qwen2Model {
    /// Build a `Qwen2Model` from a `TransformerConfig`.
    ///
    /// Returns an error if the config uses a non-Gated FFN type (Qwen2 always
    /// uses SwiGLU gated FFN) or if the base config is invalid.
    pub fn from_config(config: TransformerConfig) -> Result<Self, KernelError> {
        match &config.ff_type {
            FeedForwardType::Gated { .. } => {}
            _ => {
                return Err(KernelError::InvalidShape(
                    "Qwen2Model: ff_type must be Gated (SwiGLU) for Qwen2 architecture".into(),
                ));
            }
        }
        let inner = TransformerModel::new(config)?;
        Ok(Self { inner })
    }

    /// Forward pass: token IDs -> logits.
    ///
    /// Delegates entirely to `TransformerModel::forward()`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut Vec<LayerKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        self.inner.forward(
            token_ids, cos_freqs, sin_freqs, mask, cache, registry, queue,
        )
    }

    /// Access the underlying `TransformerModel`.
    pub fn inner(&self) -> &TransformerModel {
        &self.inner
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Access the transformer config.
    pub fn config(&self) -> &TransformerConfig {
        self.inner.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_7b_config_valid() {
        let config = qwen2_7b();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_heads, 28);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.vocab_size, 152064);
    }

    #[test]
    fn test_qwen2_model_from_config() {
        let config = qwen2_7b();
        let model = Qwen2Model::from_config(config).unwrap();
        assert_eq!(model.num_layers(), 28);
    }

    #[test]
    fn test_qwen2_model_rejects_non_gated_ffn() {
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
        let result = Qwen2Model::from_config(config);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("Gated"),
            "Error should mention Gated FFN: {msg}"
        );
    }

    #[test]
    fn test_qwen2_model_rejects_moe_ffn() {
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
            ff_type: FeedForwardType::MoE {
                config: crate::moe::MoeConfig {
                    num_experts: 8,
                    num_experts_per_token: 2,
                    hidden_dim: 64,
                    intermediate_dim: 128,
                    capacity_factor: 1.0,
                },
            },
        };
        let result = Qwen2Model::from_config(config);
        assert!(result.is_err());
    }
}
