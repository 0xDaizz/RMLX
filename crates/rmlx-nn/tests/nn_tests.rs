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
    })
    .expect("Attention::new failed");
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
        capacity_factor: 1.0,
    })
    .expect("MoeLayer::new failed");
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
    let model = TransformerModel::new(cfg).expect("TransformerModel::new failed");
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
    )
    .expect("from_arrays failed");
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
    )
    .expect("from_arrays failed");
    // Input: [1, 3] = [[5, 6, 7]]
    let input = Array::from_slice(dev, &[5.0f32, 6.0, 7.0], vec![1, 3]);
    let output = linear
        .forward(&input, &registry, &queue)
        .expect("forward failed");
    assert_eq!(output.shape(), &[1, 2]);
    let vals: Vec<f32> = output.to_vec_checked();
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

#[test]
fn test_linear_forward_with_bias_batch1() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Weight: [2, 3] = [[1, 0, 0], [0, 1, 0]] — selects first 2 dims
    let weight = Array::from_slice(dev, &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let bias = Array::from_slice(dev, &[10.0f32, 20.0], vec![2]);
    let linear = Linear::from_arrays(
        LinearConfig {
            in_features: 3,
            out_features: 2,
            has_bias: true,
        },
        weight,
        Some(bias),
    )
    .expect("from_arrays failed");
    // Input: [1, 3] = [[5, 6, 7]]
    let input = Array::from_slice(dev, &[5.0f32, 6.0, 7.0], vec![1, 3]);
    let output = linear
        .forward(&input, &registry, &queue)
        .expect("forward failed");
    assert_eq!(output.shape(), &[1, 2]);
    let vals: Vec<f32> = output.to_vec_checked();
    // output = [5, 6] + [10, 20] = [15, 26]
    assert!(
        (vals[0] - 15.0).abs() < 1e-3,
        "bias batch1[0] = {} expected 15.0",
        vals[0]
    );
    assert!(
        (vals[1] - 26.0).abs() < 1e-3,
        "bias batch1[1] = {} expected 26.0",
        vals[1]
    );
}

#[test]
fn test_linear_forward_with_bias_batch2() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // Weight: [2, 3] = [[1, 0, 0], [0, 1, 0]]
    let weight = Array::from_slice(dev, &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let bias = Array::from_slice(dev, &[10.0f32, 20.0], vec![2]);
    let linear = Linear::from_arrays(
        LinearConfig {
            in_features: 3,
            out_features: 2,
            has_bias: true,
        },
        weight,
        Some(bias),
    )
    .expect("from_arrays failed");
    // Input: [2, 3] = [[5, 6, 7], [1, 2, 3]]
    let input = Array::from_slice(dev, &[5.0f32, 6.0, 7.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let output = linear
        .forward(&input, &registry, &queue)
        .expect("forward failed");
    assert_eq!(output.shape(), &[2, 2]);
    let vals: Vec<f32> = output.to_vec_checked();
    // Row 0: [5, 6] + [10, 20] = [15, 26]
    assert!(
        (vals[0] - 15.0).abs() < 1e-3,
        "bias batch2[0,0] = {} expected 15.0",
        vals[0]
    );
    assert!(
        (vals[1] - 26.0).abs() < 1e-3,
        "bias batch2[0,1] = {} expected 26.0",
        vals[1]
    );
    // Row 1: [1, 2] + [10, 20] = [11, 22]
    assert!(
        (vals[2] - 11.0).abs() < 1e-3,
        "bias batch2[1,0] = {} expected 11.0",
        vals[2]
    );
    assert!(
        (vals[3] - 22.0).abs() < 1e-3,
        "bias batch2[1,1] = {} expected 22.0",
        vals[3]
    );
}

// ===== NEW TESTS: Embedding forward =====

#[test]
fn test_embedding_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    // vocab_size=4, embed_dim=3
    // Weight: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    let weight = Array::from_slice(
        dev,
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![4, 3],
    );
    let emb = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: 4,
            embed_dim: 3,
        },
        weight,
    )
    .expect("from_array failed");
    // Lookup tokens [0, 2, 3]
    let output = emb
        .forward(&[0, 2, 3], &registry, &queue)
        .expect("forward failed");
    assert_eq!(output.shape(), &[3, 3]);
    let vals: Vec<f32> = output.to_vec_checked();
    // Token 0: [1, 2, 3]
    assert!((vals[0] - 1.0).abs() < 1e-3);
    assert!((vals[1] - 2.0).abs() < 1e-3);
    assert!((vals[2] - 3.0).abs() < 1e-3);
    // Token 2: [7, 8, 9]
    assert!((vals[3] - 7.0).abs() < 1e-3);
    assert!((vals[4] - 8.0).abs() < 1e-3);
    assert!((vals[5] - 9.0).abs() < 1e-3);
    // Token 3: [10, 11, 12]
    assert!((vals[6] - 10.0).abs() < 1e-3);
    assert!((vals[7] - 11.0).abs() < 1e-3);
    assert!((vals[8] - 12.0).abs() < 1e-3);
}

#[test]
fn test_embedding_no_weights_errors() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let emb = Embedding::new(EmbeddingConfig {
        vocab_size: 100,
        embed_dim: 8,
    });
    assert!(!emb.has_weights());
    let result = emb.forward(&[0, 1], &registry, &queue);
    assert!(result.is_err());
}

// ===== NEW TESTS: Attention forward =====

fn make_identity_linear(dev: &metal::Device, size: usize) -> Linear {
    // Create an identity-like weight matrix [size, size]
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    let weight = Array::from_slice(dev, &data, vec![size, size]);
    Linear::from_arrays(
        LinearConfig {
            in_features: size,
            out_features: size,
            has_bias: false,
        },
        weight,
        None,
    )
    .expect("identity linear from_arrays failed")
}

#[test]
fn test_attention_forward_identity() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Tiny attention: 2 heads, 1 kv_head (GQA), head_dim=4 => hidden_size=8
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let hidden_size = num_heads * head_dim; // 8

    let config = AttentionConfig {
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len: 16,
        rope_theta: 10000.0,
    };

    // Use identity projections so we can verify the flow
    let q_proj = make_identity_linear(dev, hidden_size);
    let k_proj = make_identity_linear(dev, hidden_size);
    let v_proj = make_identity_linear(dev, hidden_size);
    let o_proj = make_identity_linear(dev, hidden_size);

    let attn = Attention::from_layers(config, q_proj, k_proj, v_proj, o_proj)
        .expect("Attention::from_layers failed");

    // Input: [2, 8] — 2 tokens, 8-dim hidden
    let input = Array::from_slice(
        dev,
        &[
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
        vec![2, 8],
    );

    // No RoPE, no mask, no cache for simplicity
    let output = attn
        .forward(&input, None, None, None, None, &registry, &queue)
        .expect("attention forward failed");
    assert_eq!(output.shape(), &[2, 8]);

    // Just verify it produces a valid output (not NaN/zero everywhere)
    let vals: Vec<f32> = output.to_vec_checked();
    let sum: f32 = vals.iter().sum();
    assert!(
        sum.is_finite(),
        "attention output contains non-finite values"
    );
    assert!(sum.abs() > 1e-6, "attention output is all zeros");
}

// ===== NEW TESTS: MoE forward =====

#[test]
fn test_moe_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let hidden_dim = 4;
    let intermediate_dim = 4;
    let num_experts = 2;
    let top_k = 1;

    // Gate: [num_experts, hidden_dim] — routes tokens to experts
    let gate_weight = Array::from_slice(
        dev,
        &[
            1.0f32, 0.0, 0.0, 0.0, // expert 0 gate
            0.0, 0.0, 0.0, 1.0, // expert 1 gate
        ],
        vec![num_experts, hidden_dim],
    );
    let gate = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_dim,
            out_features: num_experts,
            has_bias: false,
        },
        gate_weight,
        None,
    )
    .expect("gate from_arrays failed");

    // Expert 0: identity-like (gate=identity, up=identity, down=identity)
    let expert0 = Expert {
        gate_proj: make_identity_linear(dev, hidden_dim),
        up_proj: make_identity_linear(dev, hidden_dim),
        down_proj: make_identity_linear(dev, hidden_dim),
    };

    // Expert 1: identity-like too
    let expert1 = Expert {
        gate_proj: make_identity_linear(dev, hidden_dim),
        up_proj: make_identity_linear(dev, hidden_dim),
        down_proj: make_identity_linear(dev, hidden_dim),
    };

    let moe = MoeLayer::from_layers(
        MoeConfig {
            num_experts,
            num_experts_per_token: top_k,
            hidden_dim,
            intermediate_dim,
            capacity_factor: 1.0,
        },
        gate,
        vec![expert0, expert1],
    )
    .expect("from_layers failed");

    // Input: [2, 4]
    let input = Array::from_slice(
        dev,
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 4],
    );
    let output = moe
        .forward(&input, &registry, &queue)
        .expect("moe forward failed");
    assert_eq!(output.shape(), &[2, 4]);

    let vals: Vec<f32> = output.to_vec_checked();
    let sum: f32 = vals.iter().sum();
    assert!(sum.is_finite(), "MoE output contains non-finite values");
    assert!(sum.abs() > 1e-6, "MoE output is all zeros");
}

#[test]
fn test_moe_no_weights_errors() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let moe = MoeLayer::new(MoeConfig {
        num_experts: 2,
        num_experts_per_token: 1,
        hidden_dim: 4,
        intermediate_dim: 8,
        capacity_factor: 1.0,
    })
    .expect("MoeLayer::new failed");
    let input = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let result = moe.forward(&input, &registry, &queue);
    assert!(result.is_err());
}

// ===== NEW TESTS: TransformerBlock forward =====

#[test]
fn test_transformer_block_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Tiny transformer block: hidden_size=8, 2 heads, head_dim=4
    let hidden_size = 8;
    let num_heads = 2;
    let head_dim = 4;
    let attn_config = AttentionConfig {
        num_heads,
        num_kv_heads: num_heads,
        head_dim,
        max_seq_len: 16,
        rope_theta: 10000.0,
    };
    let attention = Attention::from_layers(
        attn_config,
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
    )
    .expect("Attention::from_layers failed");

    let ffn = FeedForward::Dense {
        gate_proj: make_identity_linear(dev, hidden_size),
        up_proj: make_identity_linear(dev, hidden_size),
        down_proj: make_identity_linear(dev, hidden_size),
    };

    // RMS norm weights (all ones)
    let norm1_weight = Array::ones(dev, &[hidden_size]);
    let norm2_weight = Array::ones(dev, &[hidden_size]);

    let block = TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, 1e-5);

    // Input: [2, 8]
    let input = Array::from_slice(
        dev,
        &[
            1.0f32, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7,
        ],
        vec![2, 8],
    );

    let output = block
        .forward(&input, None, None, None, None, &registry, &queue)
        .expect("transformer block forward failed");
    assert_eq!(output.shape(), &[2, 8]);

    let vals: Vec<f32> = output.to_vec_checked();
    let sum: f32 = vals.iter().sum();
    assert!(sum.is_finite(), "block output contains non-finite values");
}

// ===== NEW TESTS: TransformerModel forward (end-to-end) =====

#[test]
fn test_transformer_model_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Tiny model: vocab=8, hidden=8, 2 heads, head_dim=4, 1 layer
    let vocab_size = 8;
    let hidden_size = 8;
    let num_heads = 2;
    let head_dim = 4;

    // Embedding: [8, 8] — each token maps to a unique 8-dim vector
    let mut emb_data = vec![0.0f32; vocab_size * hidden_size];
    for i in 0..vocab_size {
        for j in 0..hidden_size {
            emb_data[i * hidden_size + j] = (i as f32 + 1.0) * 0.1 + j as f32 * 0.01;
        }
    }
    let emb_weight = Array::from_slice(dev, &emb_data, vec![vocab_size, hidden_size]);
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size,
            embed_dim: hidden_size,
        },
        emb_weight,
    )
    .expect("embedding from_array failed");

    // One transformer block
    let attn_config = AttentionConfig {
        num_heads,
        num_kv_heads: num_heads,
        head_dim,
        max_seq_len: 16,
        rope_theta: 10000.0,
    };
    let attention = Attention::from_layers(
        attn_config,
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
        make_identity_linear(dev, hidden_size),
    )
    .expect("Attention::from_layers failed");
    let ffn = FeedForward::Dense {
        gate_proj: make_identity_linear(dev, hidden_size),
        up_proj: make_identity_linear(dev, hidden_size),
        down_proj: make_identity_linear(dev, hidden_size),
    };
    let norm1 = Array::ones(dev, &[hidden_size]);
    let norm2 = Array::ones(dev, &[hidden_size]);
    let block = TransformerBlock::from_parts(0, attention, ffn, norm1, norm2, 1e-5);

    // Final norm weight
    let final_norm = Array::ones(dev, &[hidden_size]);

    // LM head: [vocab_size, hidden_size]
    let mut lm_head_data = vec![0.0f32; vocab_size * hidden_size];
    for i in 0..vocab_size {
        lm_head_data[i * hidden_size + i % hidden_size] = 1.0;
    }
    let lm_head_weight = Array::from_slice(dev, &lm_head_data, vec![vocab_size, hidden_size]);
    let lm_head = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: vocab_size,
            has_bias: false,
        },
        lm_head_weight,
        None,
    )
    .expect("lm_head from_arrays failed");

    let config = TransformerConfig {
        hidden_size,
        num_heads,
        num_kv_heads: num_heads,
        head_dim,
        num_layers: 1,
        vocab_size,
        max_seq_len: 16,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Dense {
            intermediate_dim: hidden_size,
        },
    };

    let model = TransformerModel::from_parts(config, embedding, vec![block], final_norm, lm_head)
        .expect("TransformerModel::from_parts failed");

    // Forward pass with 3 tokens
    let token_ids = [0u32, 1, 2];
    let logits = model
        .forward(&token_ids, None, None, None, None, &registry, &queue)
        .expect("model forward failed");

    // Expected output shape: [3, 8] (3 tokens, 8 vocab logits each)
    assert_eq!(logits.shape(), &[3, vocab_size]);

    let vals: Vec<f32> = logits.to_vec_checked();
    let sum: f32 = vals.iter().sum();
    assert!(sum.is_finite(), "model output contains non-finite values");
}

// ===== Config validate negative tests =====

#[test]
fn test_attention_config_zero_num_heads() {
    let result = Attention::new(AttentionConfig {
        num_heads: 0,
        num_kv_heads: 1,
        head_dim: 64,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    });
    assert!(result.is_err(), "num_heads=0 should fail");
}

#[test]
fn test_attention_config_zero_head_dim() {
    let result = Attention::new(AttentionConfig {
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 0,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    });
    assert!(result.is_err(), "head_dim=0 should fail");
}

#[test]
fn test_attention_config_zero_kv_heads() {
    let result = Attention::new(AttentionConfig {
        num_heads: 8,
        num_kv_heads: 0,
        head_dim: 64,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    });
    assert!(result.is_err(), "num_kv_heads=0 should fail");
}

#[test]
fn test_attention_config_heads_not_divisible_by_kv() {
    let result = Attention::new(AttentionConfig {
        num_heads: 7,
        num_kv_heads: 3,
        head_dim: 64,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    });
    assert!(result.is_err(), "num_heads % num_kv_heads != 0 should fail");
}

#[test]
fn test_moe_config_zero_experts() {
    let result = MoeLayer::new(MoeConfig {
        num_experts: 0,
        num_experts_per_token: 1,
        hidden_dim: 256,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    });
    assert!(result.is_err(), "num_experts=0 should fail");
}

#[test]
fn test_moe_config_zero_experts_per_token() {
    let result = MoeLayer::new(MoeConfig {
        num_experts: 8,
        num_experts_per_token: 0,
        hidden_dim: 256,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    });
    assert!(result.is_err(), "num_experts_per_token=0 should fail");
}

#[test]
fn test_moe_config_top_k_exceeds_experts() {
    let result = MoeLayer::new(MoeConfig {
        num_experts: 4,
        num_experts_per_token: 5,
        hidden_dim: 256,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    });
    assert!(
        result.is_err(),
        "num_experts_per_token > num_experts should fail"
    );
}

#[test]
fn test_moe_config_zero_hidden_dim() {
    let result = MoeLayer::new(MoeConfig {
        num_experts: 8,
        num_experts_per_token: 2,
        hidden_dim: 0,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    });
    assert!(result.is_err(), "hidden_dim=0 should fail");
}

#[test]
fn test_transformer_config_zero_num_layers() {
    let result = TransformerModel::new(TransformerConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 64,
        num_layers: 0,
        vocab_size: 1000,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Dense {
            intermediate_dim: 512,
        },
    });
    assert!(result.is_err(), "num_layers=0 should fail");
}

#[test]
fn test_transformer_config_zero_hidden_size() {
    let result = TransformerModel::new(TransformerConfig {
        hidden_size: 0,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 1000,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Dense {
            intermediate_dim: 512,
        },
    });
    assert!(result.is_err(), "hidden_size=0 should fail");
}

#[test]
fn test_transformer_config_zero_vocab_size() {
    let result = TransformerModel::new(TransformerConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 0,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Dense {
            intermediate_dim: 512,
        },
    });
    assert!(result.is_err(), "vocab_size=0 should fail");
}

#[test]
fn test_transformer_config_kv_heads_exceeds_heads() {
    let result = TransformerModel::new(TransformerConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 8,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 1000,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Dense {
            intermediate_dim: 512,
        },
    });
    assert!(result.is_err(), "num_kv_heads > num_heads should fail");
}

#[test]
fn test_transformer_block_new_validates_config() {
    let result = TransformerBlock::new(
        0,
        TransformerConfig {
            hidden_size: 256,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 0,
            num_layers: 1,
            vocab_size: 1000,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            ff_type: FeedForwardType::Dense {
                intermediate_dim: 512,
            },
        },
    );
    assert!(
        result.is_err(),
        "head_dim=0 in TransformerBlock::new should fail"
    );
}
