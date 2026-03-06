//! Qwen model configuration.

use crate::transformer::{FeedForwardType, TransformerConfig};

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
