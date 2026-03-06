//! LLaMA model architecture.
//!
//! LLaMA uses standard GQA attention with SwiGLU (gated) FFN.
//! This wraps `TransformerModel` directly since the generic transformer
//! already implements the exact Llama architecture.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};

use crate::attention::LayerKvCache;
use crate::transformer::{FeedForwardType, TransformerConfig, TransformerModel};

/// LLaMA 7B configuration preset.
pub fn llama_7b() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 128,
        num_layers: 32,
        vocab_size: 32000,
        max_seq_len: 4096,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: 11008,
        },
    }
}

pub fn llama_3_8b() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        head_dim: 128,
        num_layers: 32,
        vocab_size: 128256,
        max_seq_len: 8192,
        rope_theta: 500000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: 14336,
        },
    }
}

// ---------------------------------------------------------------------------
// LlamaModel — thin wrapper over TransformerModel
// ---------------------------------------------------------------------------

/// LLaMA model: standard GQA + SwiGLU wrapped as `TransformerModel`.
///
/// This is a thin, type-safe wrapper that enforces Llama-specific constraints
/// (gated FFN) while delegating all computation to the generic transformer.
pub struct LlamaModel {
    inner: TransformerModel,
}

impl LlamaModel {
    /// Build a `LlamaModel` from a `TransformerConfig`.
    ///
    /// Returns an error if the config uses a non-Gated FFN type (Llama always
    /// uses SwiGLU gated FFN) or if the base config is invalid.
    pub fn from_config(config: TransformerConfig) -> Result<Self, KernelError> {
        // Llama requires gated (SwiGLU) FFN
        match &config.ff_type {
            FeedForwardType::Gated { .. } => {}
            _ => {
                return Err(KernelError::InvalidShape(
                    "LlamaModel: ff_type must be Gated (SwiGLU) for Llama architecture".into(),
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
        &self,
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
    fn test_llama_7b_config_valid() {
        let config = llama_7b();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_llama_3_8b_config_valid() {
        let config = llama_3_8b();
        assert!(config.validate().is_ok());
        assert_eq!(config.num_kv_heads, 8); // GQA
        assert_eq!(config.vocab_size, 128256);
    }

    #[test]
    fn test_llama_model_from_config() {
        let config = llama_7b();
        let model = LlamaModel::from_config(config).unwrap();
        assert_eq!(model.num_layers(), 32);
    }

    #[test]
    fn test_llama_model_rejects_non_gated_ffn() {
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
        let result = LlamaModel::from_config(config);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("Gated"),
            "Error should mention Gated FFN: {msg}"
        );
    }

    #[test]
    fn test_llama_model_rejects_moe_ffn() {
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
        let result = LlamaModel::from_config(config);
        assert!(result.is_err());
    }
}
