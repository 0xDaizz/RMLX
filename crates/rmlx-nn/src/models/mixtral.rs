//! Mixtral model configuration.

use crate::moe::MoeConfig;
use crate::transformer::{FeedForwardType, TransformerConfig};

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
            },
        },
    }
}
