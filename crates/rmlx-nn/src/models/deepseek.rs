//! DeepSeek-V3 model configuration.
//!
//! DeepSeek-V3 uses Multi-head Latent Attention (MLA) which compresses KV
//! into a low-rank latent space. The effective `num_kv_heads` is 1 per head
//! group because Q/K/V share a compressed latent representation per group.
//!
//! Key architecture details:
//! - 256 routed experts with top-8 routing
//! - 1 shared expert processed on every token (DeepSeek-V3 specific)
//! - MLA with kv_lora_rank=512, q_lora_rank=1536
//! - Max sequence length: 163840 (with YaRN)

use crate::moe::MoeConfig;
use crate::transformer::{FeedForwardType, TransformerConfig};

/// DeepSeek-V3 specific configuration fields not captured by `TransformerConfig`.
///
/// These are needed for full model instantiation but go beyond the generic
/// transformer config (which is shared across LLaMA, Mixtral, etc.).
pub struct DeepSeekV3Config {
    /// Base transformer configuration.
    pub transformer: TransformerConfig,

    // ── MLA (Multi-head Latent Attention) ──
    /// KV compression rank for MLA. DeepSeek-V3 uses 512.
    pub kv_lora_rank: usize,
    /// Query compression rank for MLA. DeepSeek-V3 uses 1536.
    pub q_lora_rank: usize,
    /// Decoupled RoPE head dimension. DeepSeek-V3 uses 64.
    pub rope_head_dim: usize,
    /// Non-RoPE (value) head dimension in MLA. DeepSeek-V3 uses 128.
    pub v_head_dim: usize,

    // ── Shared expert configuration (N7/N8) ──
    /// Intermediate size for the shared expert FFN. DeepSeek-V3: 2048.
    pub shared_expert_intermediate_size: usize,
    /// Number of shared experts (always-active). DeepSeek-V3: 1.
    pub num_shared_experts: usize,

    // ── First-k dense layers ──
    /// Number of initial dense (non-MoE) layers. DeepSeek-V3: 1.
    pub first_k_dense_replace: usize,
}

/// Create the base `TransformerConfig` for DeepSeek-V3.
///
/// For full model configuration including MLA and shared expert details,
/// use [`deepseek_v3_full`].
pub fn deepseek_v3() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 7168,
        num_heads: 128,
        // MLA: KV is compressed into a shared latent space. The effective
        // number of KV heads is 1 per head group (128 heads / 128 = 1).
        // This is correct for GQA-style KV cache allocation, though the
        // actual MLA computation uses low-rank projections rather than
        // separate KV heads.
        num_kv_heads: 1, // MLA compressed KV
        head_dim: 128,
        num_layers: 61,
        vocab_size: 129280,
        max_seq_len: 163_840, // N8 fix: DeepSeek-V3 supports 163840 with YaRN
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

/// Full DeepSeek-V3 configuration including MLA and shared expert parameters.
pub fn deepseek_v3_full() -> DeepSeekV3Config {
    DeepSeekV3Config {
        transformer: deepseek_v3(),

        // MLA parameters
        kv_lora_rank: 512,
        q_lora_rank: 1536,
        rope_head_dim: 64,
        v_head_dim: 128,

        // Shared expert (N7/N8)
        shared_expert_intermediate_size: 2048,
        num_shared_experts: 1,

        // Dense layers before MoE kicks in
        first_k_dense_replace: 1,
    }
}
