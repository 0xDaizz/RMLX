//! DeepSeek-V3 model configuration.

use crate::moe::MoeConfig;
use crate::transformer::{FeedForwardType, TransformerConfig};

pub fn deepseek_v3() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 7168,
        num_heads: 128,
        num_kv_heads: 1, // MLA
        head_dim: 128,
        num_layers: 61,
        vocab_size: 129280,
        max_seq_len: 16384,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        ff_type: FeedForwardType::MoE {
            config: MoeConfig {
                num_experts: 256,
                num_experts_per_token: 8,
                hidden_dim: 7168,
                intermediate_dim: 2048,
                capacity_factor: 1.25,
            },
        },
    }
}
