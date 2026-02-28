//! LLaMA model configuration.

use crate::transformer::{FeedForwardType, TransformerConfig};

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
        ff_type: FeedForwardType::Dense {
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
        ff_type: FeedForwardType::Dense {
            intermediate_dim: 14336,
        },
    }
}
