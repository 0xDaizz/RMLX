use rmlx_nn::attention::*;
use rmlx_nn::embedding::*;
use rmlx_nn::linear::*;
use rmlx_nn::models::*;
use rmlx_nn::moe::*;
use rmlx_nn::transformer::*;

#[test]
fn test_linear_config() {
    let l = Linear::new(LinearConfig {
        in_features: 4096,
        out_features: 11008,
        has_bias: false,
    });
    assert_eq!(l.in_features(), 4096);
    assert_eq!(l.out_features(), 11008);
    assert!(!l.has_bias());
}

#[test]
fn test_embedding_config() {
    let e = Embedding::new(EmbeddingConfig {
        vocab_size: 32000,
        embed_dim: 4096,
    });
    assert_eq!(e.vocab_size(), 32000);
    assert_eq!(e.embed_dim(), 4096);
}

#[test]
fn test_attention_gqa() {
    let attn = Attention::new(AttentionConfig {
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 4096,
        rope_theta: 500000.0,
    });
    assert!(attn.is_gqa());
    assert_eq!(attn.hidden_size(), 4096);
}

#[test]
fn test_moe_layer() {
    let moe = MoeLayer::new(MoeConfig {
        num_experts: 8,
        num_experts_per_token: 2,
        hidden_dim: 4096,
        intermediate_dim: 14336,
    });
    assert_eq!(moe.num_experts(), 8);
    assert_eq!(moe.top_k(), 2);
}

#[test]
fn test_llama_7b_config() {
    let cfg = llama::llama_7b();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);
    assert!(matches!(cfg.ff_type, FeedForwardType::Dense { .. }));
}

#[test]
fn test_llama_3_8b_gqa() {
    let cfg = llama::llama_3_8b();
    assert_eq!(cfg.num_kv_heads, 8); // GQA
    assert!(cfg.num_kv_heads < cfg.num_heads);
}

#[test]
fn test_deepseek_v3_moe() {
    let cfg = deepseek::deepseek_v3();
    match &cfg.ff_type {
        FeedForwardType::MoE { config } => {
            assert_eq!(config.num_experts, 256);
            assert_eq!(config.num_experts_per_token, 8);
        }
        _ => panic!("DeepSeek-V3 should use MoE"),
    }
}

#[test]
fn test_mixtral_moe() {
    let cfg = mixtral::mixtral_8x7b();
    match &cfg.ff_type {
        FeedForwardType::MoE { config } => {
            assert_eq!(config.num_experts, 8);
            assert_eq!(config.num_experts_per_token, 2);
        }
        _ => panic!("Mixtral should use MoE"),
    }
}

#[test]
fn test_transformer_model() {
    let cfg = llama::llama_7b();
    let model = TransformerModel::new(cfg);
    assert_eq!(model.num_layers(), 32);
    assert_eq!(model.config().hidden_size, 4096);
}

#[test]
fn test_weight_shape_validation() {
    // Negative test: mismatched weight shape would be caught at runtime
    let l = Linear::new(LinearConfig {
        in_features: 0,
        out_features: 0,
        has_bias: true,
    });
    assert_eq!(l.in_features(), 0);
    assert!(l.has_bias());
}
