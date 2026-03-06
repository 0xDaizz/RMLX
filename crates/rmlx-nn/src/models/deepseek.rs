//! DeepSeek-V3 model architecture.
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
//! - First `first_k_dense_replace` layers use dense (gated) FFN instead of MoE

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::embedding::Embedding;
use crate::linear::{Linear, LinearConfig};
use crate::mla::{Mla, MlaConfig, MlaKvCache};
use crate::moe::{MoeConfig, MoeLayer};
use crate::transformer::{FeedForward, FeedForwardType, TransformerConfig};

/// DeepSeek-V3 specific configuration fields not captured by `TransformerConfig`.
///
/// These are needed for full model instantiation but go beyond the generic
/// transformer config (which is shared across LLaMA, Mixtral, etc.).
pub struct DeepSeekV3Config {
    /// Base transformer configuration.
    pub transformer: TransformerConfig,

    // -- MLA (Multi-head Latent Attention) --
    /// KV compression rank for MLA. DeepSeek-V3 uses 512.
    pub kv_lora_rank: usize,
    /// Query compression rank for MLA. DeepSeek-V3 uses 1536.
    pub q_lora_rank: usize,
    /// Decoupled RoPE head dimension. DeepSeek-V3 uses 64.
    pub rope_head_dim: usize,
    /// Non-RoPE (value) head dimension in MLA. DeepSeek-V3 uses 128.
    pub v_head_dim: usize,

    // -- Shared expert configuration (N7/N8) --
    /// Intermediate size for the shared expert FFN. DeepSeek-V3: 2048.
    pub shared_expert_intermediate_size: usize,
    /// Number of shared experts (always-active). DeepSeek-V3: 1.
    pub num_shared_experts: usize,

    // -- First-k dense layers --
    /// Number of initial dense (non-MoE) layers. DeepSeek-V3: 1.
    pub first_k_dense_replace: usize,
}

impl DeepSeekV3Config {
    /// Validate the DeepSeek-V3 configuration.
    pub fn validate(&self) -> Result<(), KernelError> {
        self.transformer.validate()?;
        if self.kv_lora_rank == 0 {
            return Err(KernelError::InvalidShape(
                "DeepSeekV3Config: kv_lora_rank must be > 0".into(),
            ));
        }
        if self.q_lora_rank == 0 {
            return Err(KernelError::InvalidShape(
                "DeepSeekV3Config: q_lora_rank must be > 0".into(),
            ));
        }
        if self.rope_head_dim > self.transformer.head_dim {
            return Err(KernelError::InvalidShape(format!(
                "DeepSeekV3Config: rope_head_dim ({}) must be <= head_dim ({})",
                self.rope_head_dim, self.transformer.head_dim
            )));
        }
        if self.v_head_dim == 0 {
            return Err(KernelError::InvalidShape(
                "DeepSeekV3Config: v_head_dim must be > 0".into(),
            ));
        }
        if self.first_k_dense_replace > self.transformer.num_layers {
            return Err(KernelError::InvalidShape(format!(
                "DeepSeekV3Config: first_k_dense_replace ({}) > num_layers ({})",
                self.first_k_dense_replace, self.transformer.num_layers
            )));
        }
        Ok(())
    }

    /// Build the MLA config from this DeepSeek-V3 config.
    pub fn mla_config(&self) -> MlaConfig {
        MlaConfig {
            num_heads: self.transformer.num_heads,
            head_dim: self.transformer.head_dim,
            v_head_dim: self.v_head_dim,
            hidden_size: self.transformer.hidden_size,
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank,
            rope_head_dim: self.rope_head_dim,
            rope_theta: self.transformer.rope_theta,
            max_seq_len: self.transformer.max_seq_len,
        }
    }
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
        num_kv_heads: 1, // MLA compressed KV
        head_dim: 128,
        num_layers: 61,
        vocab_size: 129280,
        max_seq_len: 163_840, // DeepSeek-V3 supports 163840 with YaRN
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

// ---------------------------------------------------------------------------
// DeepSeek-V3 layer block
// ---------------------------------------------------------------------------

/// A single DeepSeek-V3 transformer layer: MLA attention + FFN (dense or MoE).
///
/// Unlike the generic `TransformerBlock` which uses standard `Attention`,
/// this block uses `Mla` for the attention mechanism.
pub struct DeepSeekV3Block {
    layer_idx: usize,
    mla: Mla,
    ffn: FeedForward,
    norm1_weight: Option<Array>,
    norm2_weight: Option<Array>,
    rms_norm_eps: f32,
}

impl DeepSeekV3Block {
    /// Create a block with config-only MLA and FFN (no weights loaded).
    pub fn new(
        layer_idx: usize,
        mla_config: MlaConfig,
        ffn: FeedForward,
        rms_norm_eps: f32,
    ) -> Result<Self, KernelError> {
        let mla = Mla::new(mla_config)?;
        Ok(Self {
            layer_idx,
            mla,
            ffn,
            norm1_weight: None,
            norm2_weight: None,
            rms_norm_eps,
        })
    }

    /// Create a block with pre-loaded components.
    pub fn from_parts(
        layer_idx: usize,
        mla: Mla,
        ffn: FeedForward,
        norm1_weight: Array,
        norm2_weight: Array,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            layer_idx,
            mla,
            ffn,
            norm1_weight: Some(norm1_weight),
            norm2_weight: Some(norm2_weight),
            rms_norm_eps,
        }
    }

    /// Forward pass for one DeepSeek-V3 block.
    ///
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables
    /// `cache`: optional MLA KV cache
    /// Returns: [seq_len, hidden_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: Option<&mut MlaKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let norm1_w = self.norm1_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("DeepSeekV3Block: norm1_weight not loaded".into())
        })?;
        let norm2_w = self.norm2_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("DeepSeekV3Block: norm2_weight not loaded".into())
        })?;

        // Pre-attention RMS norm
        let normed = ops::rms_norm::rms_norm(registry, x, norm1_w, self.rms_norm_eps, queue)?;

        // MLA attention (no causal mask param — MLA handles masking internally)
        let attn_out = self
            .mla
            .forward(&normed, cos_freqs, sin_freqs, cache, registry, queue)?;

        // Residual
        let h = ops::binary::add(registry, x, &attn_out, queue)?;

        // Pre-FFN RMS norm
        let normed2 = ops::rms_norm::rms_norm(registry, &h, norm2_w, self.rms_norm_eps, queue)?;

        // FFN (dense or MoE depending on layer index)
        let ffn_out = self.ffn.forward(&normed2, registry, queue)?;

        // Residual
        ops::binary::add(registry, &h, &ffn_out, queue)
    }

    /// Layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV3Model
// ---------------------------------------------------------------------------

/// DeepSeek-V3 model: MLA attention + mixed dense/MoE FFN layers.
///
/// Architecture:
/// - Token embedding
/// - First `first_k_dense_replace` layers: MLA + dense (gated) FFN
/// - Remaining layers: MLA + MoE FFN
/// - Final RMS norm + LM head
pub struct DeepSeekV3Model {
    config: DeepSeekV3Config,
    embedding: Option<Embedding>,
    layers: Vec<DeepSeekV3Block>,
    final_norm_weight: Option<Array>,
    lm_head: Option<Linear>,
}

impl DeepSeekV3Model {
    /// Build a `DeepSeekV3Model` from config (no weights loaded).
    ///
    /// Creates MLA + FFN blocks for all layers. The first `first_k_dense_replace`
    /// layers get gated (SwiGLU) FFN; remaining layers get config-only MoE FFN
    /// (experts must be loaded separately via weight loading).
    pub fn from_config(config: DeepSeekV3Config) -> Result<Self, KernelError> {
        config.validate()?;

        let num_layers = config.transformer.num_layers;
        let hidden_size = config.transformer.hidden_size;
        let rms_norm_eps = config.transformer.rms_norm_eps;
        let mla_cfg = config.mla_config();

        // Extract MoE config from the transformer ff_type
        let moe_config = match &config.transformer.ff_type {
            FeedForwardType::MoE { config: mc } => mc.clone(),
            _ => {
                return Err(KernelError::InvalidShape(
                    "DeepSeekV3Model: transformer.ff_type must be MoE".into(),
                ));
            }
        };

        // Determine the intermediate dim for dense layers.
        // DeepSeek-V3 uses the shared_expert_intermediate_size for the first dense layers.
        let dense_intermediate = config.shared_expert_intermediate_size;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let mla_config_i = MlaConfig {
                num_heads: mla_cfg.num_heads,
                head_dim: mla_cfg.head_dim,
                v_head_dim: mla_cfg.v_head_dim,
                hidden_size: mla_cfg.hidden_size,
                kv_lora_rank: mla_cfg.kv_lora_rank,
                q_lora_rank: mla_cfg.q_lora_rank,
                rope_head_dim: mla_cfg.rope_head_dim,
                rope_theta: mla_cfg.rope_theta,
                max_seq_len: mla_cfg.max_seq_len,
            };

            let ffn = if i < config.first_k_dense_replace {
                // Dense (gated/SwiGLU) FFN for first-k layers
                FeedForward::Gated {
                    gate_proj: Linear::new(LinearConfig {
                        in_features: hidden_size,
                        out_features: dense_intermediate,
                        has_bias: false,
                    }),
                    up_proj: Linear::new(LinearConfig {
                        in_features: hidden_size,
                        out_features: dense_intermediate,
                        has_bias: false,
                    }),
                    down_proj: Linear::new(LinearConfig {
                        in_features: dense_intermediate,
                        out_features: hidden_size,
                        has_bias: false,
                    }),
                }
            } else {
                // MoE FFN for remaining layers (config-only, experts loaded later)
                let moe_layer = MoeLayer::new(moe_config.clone())?;
                FeedForward::MoE(moe_layer)
            };

            let block = DeepSeekV3Block::new(i, mla_config_i, ffn, rms_norm_eps)?;
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
    ///
    /// Runs through embedding, all MLA+FFN layers, final norm, and LM head.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        token_ids: &[u32],
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        mut cache: Option<&mut Vec<MlaKvCache>>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let embedding = self.embedding.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("DeepSeekV3Model: embedding not loaded".into())
        })?;
        let final_norm = self.final_norm_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("DeepSeekV3Model: final_norm not loaded".into())
        })?;
        let lm_head = self.lm_head.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("DeepSeekV3Model: lm_head not loaded".into())
        })?;

        // Validate cache length
        if let Some(ref c) = cache {
            if c.len() != self.layers.len() {
                return Err(KernelError::InvalidShape(format!(
                    "DeepSeekV3Model: cache has {} entries but model has {} layers",
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

    /// Number of dense (non-MoE) layers at the start.
    pub fn num_dense_layers(&self) -> usize {
        self.config.first_k_dense_replace
    }

    /// Number of MoE layers.
    pub fn num_moe_layers(&self) -> usize {
        self.layers.len() - self.config.first_k_dense_replace
    }

    /// Access the full config.
    pub fn config(&self) -> &DeepSeekV3Config {
        &self.config
    }

    /// Access a specific layer block.
    pub fn layer(&self, idx: usize) -> Option<&DeepSeekV3Block> {
        self.layers.get(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_v3_config_valid() {
        let config = deepseek_v3();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.num_heads, 128);
        assert_eq!(config.num_layers, 61);
    }

    #[test]
    fn test_deepseek_v3_full_config_valid() {
        let config = deepseek_v3_full();
        assert!(config.validate().is_ok());
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.q_lora_rank, 1536);
        assert_eq!(config.rope_head_dim, 64);
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.first_k_dense_replace, 1);
    }

    #[test]
    fn test_deepseek_v3_full_mla_config() {
        let config = deepseek_v3_full();
        let mla_cfg = config.mla_config();
        assert_eq!(mla_cfg.num_heads, 128);
        assert_eq!(mla_cfg.head_dim, 128);
        assert_eq!(mla_cfg.v_head_dim, 128);
        assert_eq!(mla_cfg.kv_lora_rank, 512);
        assert_eq!(mla_cfg.q_lora_rank, 1536);
        assert_eq!(mla_cfg.rope_head_dim, 64);
        assert_eq!(mla_cfg.nope_head_dim(), 64);
    }

    #[test]
    fn test_deepseek_v3_model_construction() {
        // Use a small config for unit testing (not the full 61-layer model)
        let small_config = DeepSeekV3Config {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 1,
                head_dim: 32,
                num_layers: 3,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-6,
                ff_type: FeedForwardType::MoE {
                    config: MoeConfig {
                        num_experts: 4,
                        num_experts_per_token: 2,
                        hidden_dim: 128,
                        intermediate_dim: 64,
                        capacity_factor: 1.0,
                    },
                },
            },
            kv_lora_rank: 32,
            q_lora_rank: 64,
            rope_head_dim: 8,
            v_head_dim: 32,
            shared_expert_intermediate_size: 64,
            num_shared_experts: 1,
            first_k_dense_replace: 1,
        };

        let model = DeepSeekV3Model::from_config(small_config).unwrap();
        assert_eq!(model.num_layers(), 3);
        assert_eq!(model.num_dense_layers(), 1);
        assert_eq!(model.num_moe_layers(), 2);
    }

    #[test]
    fn test_deepseek_v3_layer_structure() {
        let small_config = DeepSeekV3Config {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 1,
                head_dim: 32,
                num_layers: 3,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-6,
                ff_type: FeedForwardType::MoE {
                    config: MoeConfig {
                        num_experts: 4,
                        num_experts_per_token: 2,
                        hidden_dim: 128,
                        intermediate_dim: 64,
                        capacity_factor: 1.0,
                    },
                },
            },
            kv_lora_rank: 32,
            q_lora_rank: 64,
            rope_head_dim: 8,
            v_head_dim: 32,
            shared_expert_intermediate_size: 64,
            num_shared_experts: 1,
            first_k_dense_replace: 1,
        };

        let model = DeepSeekV3Model::from_config(small_config).unwrap();

        // Layer 0 should be dense (Gated FFN)
        let layer0 = model.layer(0).unwrap();
        assert_eq!(layer0.layer_idx(), 0);
        assert!(matches!(layer0.ffn, FeedForward::Gated { .. }));

        // Layers 1,2 should be MoE
        let layer1 = model.layer(1).unwrap();
        assert_eq!(layer1.layer_idx(), 1);
        assert!(matches!(layer1.ffn, FeedForward::MoE(_)));

        let layer2 = model.layer(2).unwrap();
        assert_eq!(layer2.layer_idx(), 2);
        assert!(matches!(layer2.ffn, FeedForward::MoE(_)));
    }

    #[test]
    fn test_deepseek_v3_config_validation_errors() {
        // kv_lora_rank == 0
        let mut config = deepseek_v3_full();
        config.kv_lora_rank = 0;
        assert!(config.validate().is_err());

        // q_lora_rank == 0
        let mut config = deepseek_v3_full();
        config.q_lora_rank = 0;
        assert!(config.validate().is_err());

        // rope_head_dim > head_dim
        let mut config = deepseek_v3_full();
        config.rope_head_dim = 256; // > 128
        assert!(config.validate().is_err());

        // v_head_dim == 0
        let mut config = deepseek_v3_full();
        config.v_head_dim = 0;
        assert!(config.validate().is_err());

        // first_k_dense_replace > num_layers
        let mut config = deepseek_v3_full();
        config.first_k_dense_replace = 100;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deepseek_v3_model_rejects_non_moe_transformer() {
        let bad_config = DeepSeekV3Config {
            transformer: TransformerConfig {
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 1,
                head_dim: 32,
                num_layers: 2,
                vocab_size: 1000,
                max_seq_len: 256,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-6,
                ff_type: FeedForwardType::Gated {
                    intermediate_dim: 64,
                },
            },
            kv_lora_rank: 32,
            q_lora_rank: 64,
            rope_head_dim: 8,
            v_head_dim: 32,
            shared_expert_intermediate_size: 64,
            num_shared_experts: 1,
            first_k_dense_replace: 1,
        };

        let result = DeepSeekV3Model::from_config(bad_config);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("MoE"), "Error should mention MoE: {msg}");
    }
}
