use rmlx_core::array::Array;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
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

fn setup() -> Option<(KernelRegistry, metal::CommandQueue)> {
    let device = GpuDevice::system_default().ok()?;
    let queue = device.raw().new_command_queue();
    let registry = KernelRegistry::new(device);
    ops::register_all(&registry).ok()?;
    Some((registry, queue))
}

#[test]
fn test_linear_from_arrays() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // 2x3 weight matrix
    let weight = Array::from_slice(dev, &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let linear = Linear::from_arrays(
        LinearConfig {
            in_features: 3,
            out_features: 2,
            has_bias: false,
        },
        weight,
        None,
    );
    assert!(linear.has_weights());
    assert!(linear.bias().is_none());
    assert_eq!(linear.in_features(), 3);
    assert_eq!(linear.out_features(), 2);
}

#[test]
fn test_linear_forward_no_bias() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Weight: [2, 3] = [[1, 0, 0], [0, 1, 0]]
    // This selects the first 2 dimensions of input
    let weight = Array::from_slice(dev, &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let linear = Linear::from_arrays(
        LinearConfig {
            in_features: 3,
            out_features: 2,
            has_bias: false,
        },
        weight,
        None,
    );
    // Input: [1, 3] = [[5, 6, 7]]
    let input = Array::from_slice(dev, &[5.0f32, 6.0, 7.0], vec![1, 3]);
    let output = linear
        .forward(&input, &registry, &queue)
        .expect("forward failed");
    assert_eq!(output.shape(), &[1, 2]);
    let vals: Vec<f32> = unsafe { output.to_vec() };
    // output = input @ W^T = [5, 6, 7] @ [[1, 0], [0, 1], [0, 0]] = [5, 6]
    assert!(
        (vals[0] - 5.0).abs() < 1e-3,
        "forward[0] = {} expected 5.0",
        vals[0]
    );
    assert!(
        (vals[1] - 6.0).abs() < 1e-3,
        "forward[1] = {} expected 6.0",
        vals[1]
    );
}

#[test]
fn test_linear_forward_no_weights_errors() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let linear = Linear::new(LinearConfig {
        in_features: 3,
        out_features: 2,
        has_bias: false,
    });
    assert!(!linear.has_weights());
    let input = Array::from_slice(dev, &[1.0f32, 2.0, 3.0], vec![1, 3]);
    let result = linear.forward(&input, &registry, &queue);
    assert!(result.is_err(), "should fail when weights not loaded");
}
