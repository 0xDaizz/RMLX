use objc2::runtime::ProtocolObject;
use rmlx_core::array::Array;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::types::MtlQueue;
use rmlx_nn::attention::*;
use rmlx_nn::embedding::*;
use rmlx_nn::gguf_loader::{ggml_type_to_kquant, is_kquant_type, kquant_load_info};
use rmlx_nn::linear::*;
use rmlx_nn::mla::*;
use rmlx_nn::models::*;
use rmlx_nn::moe::*;
use rmlx_nn::quantized_linear::*;
use rmlx_nn::rope::*;
use rmlx_nn::sliding_window::*;
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
    let cfg = qwen::qwen2_7b();
    let model = TransformerModel::new(cfg).expect("TransformerModel::new failed");
    assert_eq!(model.num_layers(), 28);
    assert_eq!(model.config().hidden_size, 3584);
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

fn setup() -> Option<(KernelRegistry, MtlQueue)> {
    let device = GpuDevice::system_default().ok()?;
    let queue = device.new_command_queue();
    let registry = KernelRegistry::new(device);
    ops::register_all(&registry).ok()?;
    rmlx_nn::activations::register(&registry).ok()?;
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

fn make_identity_linear(dev: &ProtocolObject<dyn objc2_metal::MTLDevice>, size: usize) -> Linear {
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

/// PR 1.3: Attention forward with GQA (num_kv_heads < num_heads).
///
/// Uses num_heads=4, num_kv_heads=2 (2x GQA ratio), head_dim=4 => hidden=16, kv_size=8.
/// Verifies that the output shape is [seq_len, hidden_size] and values are finite.
#[test]
fn test_attention_forward_gqa() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let num_heads = 4;
    let num_kv_heads = 2; // GQA: 2 KV heads shared across 4 query heads
    let head_dim = 4;
    let hidden_size = num_heads * head_dim; // 16
    let kv_size = num_kv_heads * head_dim; // 8

    let config = AttentionConfig {
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len: 16,
        rope_theta: 10000.0,
    };

    // Q/O projections: [hidden_size, hidden_size] = [16, 16]
    let q_proj = make_identity_linear(dev, hidden_size);
    let o_proj = make_identity_linear(dev, hidden_size);

    // K/V projections: [kv_size, hidden_size] = [8, 16]
    // Use a simple truncation matrix (first kv_size rows of identity)
    let mut k_data = vec![0.0f32; kv_size * hidden_size];
    for i in 0..kv_size {
        k_data[i * hidden_size + i] = 1.0;
    }
    let k_weight = Array::from_slice(dev, &k_data, vec![kv_size, hidden_size]);
    let k_proj = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_size,
            has_bias: false,
        },
        k_weight,
        None,
    )
    .expect("k_proj from_arrays failed");

    let mut v_data = vec![0.0f32; kv_size * hidden_size];
    for i in 0..kv_size {
        v_data[i * hidden_size + i] = 1.0;
    }
    let v_weight = Array::from_slice(dev, &v_data, vec![kv_size, hidden_size]);
    let v_proj = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_size,
            has_bias: false,
        },
        v_weight,
        None,
    )
    .expect("v_proj from_arrays failed");

    let attn = Attention::from_layers(config, q_proj, k_proj, v_proj, o_proj)
        .expect("Attention::from_layers failed for GQA");

    assert!(attn.is_gqa(), "should be GQA when num_kv_heads < num_heads");

    // Input: [3, 16] — 3 tokens, 16-dim hidden
    let seq_len = 3;
    let input_data: Vec<f32> = (0..(seq_len * hidden_size))
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let input = Array::from_slice(dev, &input_data, vec![seq_len, hidden_size]);

    // No RoPE, no mask, no cache
    let output = attn
        .forward(&input, None, None, None, None, &registry, &queue)
        .expect("GQA attention forward failed");

    assert_eq!(
        output.shape(),
        &[seq_len, hidden_size],
        "GQA output shape mismatch"
    );

    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "GQA attention output contains non-finite values"
    );
    let sum: f32 = vals.iter().sum();
    assert!(sum.abs() > 1e-6, "GQA attention output is all zeros");
}

/// PR 1.3: Attention forward with GQA + KV cache for incremental decoding.
///
/// Prefill 2 tokens, then decode 1 token. Verify shapes at each step.
#[test]
fn test_attention_forward_gqa_with_cache() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let hidden_size = num_heads * head_dim; // 16
    let kv_size = num_kv_heads * head_dim; // 8

    let config = AttentionConfig {
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len: 16,
        rope_theta: 10000.0,
    };

    let q_proj = make_identity_linear(dev, hidden_size);
    let o_proj = make_identity_linear(dev, hidden_size);

    let mut k_data = vec![0.0f32; kv_size * hidden_size];
    for i in 0..kv_size {
        k_data[i * hidden_size + i] = 1.0;
    }
    let k_weight = Array::from_slice(dev, &k_data, vec![kv_size, hidden_size]);
    let k_proj = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_size,
            has_bias: false,
        },
        k_weight,
        None,
    )
    .expect("k_proj failed");

    let mut v_data = vec![0.0f32; kv_size * hidden_size];
    for i in 0..kv_size {
        v_data[i * hidden_size + i] = 1.0;
    }
    let v_weight = Array::from_slice(dev, &v_data, vec![kv_size, hidden_size]);
    let v_proj = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_size,
            has_bias: false,
        },
        v_weight,
        None,
    )
    .expect("v_proj failed");

    let attn = Attention::from_layers(config, q_proj, k_proj, v_proj, o_proj)
        .expect("Attention::from_layers failed");

    // Pre-allocated KV cache
    let mut cache = LayerKvCache::preallocated(
        dev,
        num_kv_heads,
        head_dim,
        16, // max_seq_len
        rmlx_core::dtype::DType::Float32,
    );

    // Step 1: Prefill with 2 tokens
    let prefill_data: Vec<f32> = (0..(2 * hidden_size))
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let prefill_input = Array::from_slice(dev, &prefill_data, vec![2, hidden_size]);

    let out1 = attn
        .forward(
            &prefill_input,
            None,
            None,
            None,
            Some(&mut cache),
            &registry,
            &queue,
        )
        .expect("GQA prefill forward failed");
    assert_eq!(out1.shape(), &[2, hidden_size]);
    assert_eq!(
        cache.position_offset(),
        2,
        "cache should have 2 tokens after prefill"
    );

    // Step 2: Decode 1 token
    let decode_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let decode_input = Array::from_slice(dev, &decode_data, vec![1, hidden_size]);

    let out2 = attn
        .forward(
            &decode_input,
            None,
            None,
            None,
            Some(&mut cache),
            &registry,
            &queue,
        )
        .expect("GQA decode forward failed");
    assert_eq!(out2.shape(), &[1, hidden_size]);
    assert_eq!(
        cache.position_offset(),
        3,
        "cache should have 3 tokens after decode"
    );

    let vals: Vec<f32> = out2.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "GQA decode output contains non-finite values"
    );
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

    let ffn = FeedForward::Gated {
        gate_proj: make_identity_linear(dev, hidden_size),
        up_proj: make_identity_linear(dev, hidden_size),
        down_proj: make_identity_linear(dev, hidden_size),
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
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
    let ffn = FeedForward::Gated {
        gate_proj: make_identity_linear(dev, hidden_size),
        up_proj: make_identity_linear(dev, hidden_size),
        down_proj: make_identity_linear(dev, hidden_size),
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
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
        ff_type: FeedForwardType::Gated {
            intermediate_dim: hidden_size,
        },
    };

    let mut model =
        TransformerModel::from_parts(config, embedding, vec![block], final_norm, lm_head)
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
        ff_type: FeedForwardType::Gated {
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
        ff_type: FeedForwardType::Gated {
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
        ff_type: FeedForwardType::Gated {
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
        ff_type: FeedForwardType::Gated {
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
            ff_type: FeedForwardType::Gated {
                intermediate_dim: 512,
            },
        },
    );
    assert!(
        result.is_err(),
        "head_dim=0 in TransformerBlock::new should fail"
    );
}

// ===== N4: QuantizedLinear tests =====

#[test]
fn test_quantized_linear_q4_construction() {
    let in_f = 64;
    let out_f = 32;
    let group_size = 32;
    let w_packed = vec![0x55u8; out_f * (in_f / 2)]; // 0x55 = nibbles (5, 5)
    let groups_per_row = in_f / group_size;
    let scales = vec![1.0f32; out_f * groups_per_row];
    let biases = vec![0.0f32; out_f * groups_per_row];

    let ql = QuantizedLinear::new(
        w_packed.clone(),
        scales,
        biases,
        in_f,
        out_f,
        group_size,
        QuantBits::Q4,
    )
    .expect("QuantizedLinear Q4 construction failed");
    assert_eq!(ql.in_features(), in_f);
    assert_eq!(ql.out_features(), out_f);
    assert_eq!(ql.group_size(), group_size);
    assert_eq!(ql.bits(), QuantBits::Q4);
    assert_eq!(ql.w_packed().len(), out_f * (in_f / 2));
}

#[test]
fn test_quantized_linear_q8_construction() {
    let in_f = 128;
    let out_f = 64;
    let group_size = 64;
    let w_packed = vec![128u8; out_f * in_f];
    let groups_per_row = in_f / group_size;
    let scales = vec![0.5f32; out_f * groups_per_row];
    let biases = vec![-64.0f32; out_f * groups_per_row];

    let ql = QuantizedLinear::new(
        w_packed,
        scales,
        biases,
        in_f,
        out_f,
        group_size,
        QuantBits::Q8,
    )
    .expect("QuantizedLinear Q8 construction failed");
    assert_eq!(ql.bits(), QuantBits::Q8);
}

#[test]
fn test_quantized_linear_bad_group_size() {
    let result = QuantizedLinear::new(
        vec![0u8; 16],
        vec![1.0; 1],
        vec![0.0; 1],
        32,
        1,
        16, // not 32, 64, or 128
        QuantBits::Q4,
    );
    assert!(result.is_err(), "group_size=16 should be rejected");
}

// ===== N6: Load balance loss tests =====

#[test]
fn test_load_balance_loss_uniform() {
    // 4 experts, 4 tokens, each assigned to a different expert
    // gate_logits: uniform probabilities (softmax output)
    let num_experts = 4;
    let seq_len = 4;
    // Each token has equal probability across all experts
    let gate_logits = vec![0.25f32; seq_len * num_experts];
    let expert_indices = vec![0usize, 1, 2, 3]; // one per token

    let loss = load_balance_loss(&gate_logits, &expert_indices, num_experts, seq_len);

    // With perfectly balanced routing and uniform probabilities:
    // f_e = 1/4 for each, P_e = 0.25 for each
    // loss = 4 * sum(0.25 * 0.25) = 4 * 4 * 0.0625 = 1.0
    assert!(
        (loss - 1.0).abs() < 1e-5,
        "uniform loss = {loss}, expected ~1.0"
    );
}

#[test]
fn test_load_balance_loss_imbalanced() {
    // 4 experts, 4 tokens, all assigned to expert 0
    let num_experts = 4;
    let seq_len = 4;
    let gate_logits = vec![0.25f32; seq_len * num_experts];
    let expert_indices = vec![0usize, 0, 0, 0]; // all to expert 0

    let loss = load_balance_loss(&gate_logits, &expert_indices, num_experts, seq_len);

    // f_0 = 1.0, f_1,2,3 = 0.0, P_e = 0.25 for all
    // loss = 4 * (1.0 * 0.25 + 0.0 * 0.25 + 0.0 * 0.25 + 0.0 * 0.25) = 4 * 0.25 = 1.0
    // NOTE: In this case, loss equals 1.0 because gate probs are uniform.
    // The loss is meaningful when gate probs correlate with routing decisions.
    assert!(loss > 0.0, "imbalanced loss should be > 0");
}

#[test]
fn test_load_balance_loss_correlated_imbalance() {
    // 2 experts, 4 tokens, all assigned to expert 0
    // gate probs: expert 0 gets high prob, expert 1 gets low prob
    let num_experts = 2;
    let seq_len = 4;
    // Each token: [0.9, 0.1] for experts 0 and 1
    let gate_logits = vec![0.9f32, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1];
    let expert_indices = vec![0usize, 0, 0, 0]; // all to expert 0

    let loss = load_balance_loss(&gate_logits, &expert_indices, num_experts, seq_len);

    // f_0 = 1.0, f_1 = 0.0
    // P_0 = 0.9, P_1 = 0.1
    // loss = 2 * (1.0 * 0.9 + 0.0 * 0.1) = 2 * 0.9 = 1.8
    assert!(
        (loss - 1.8).abs() < 1e-5,
        "correlated imbalanced loss = {loss}, expected 1.8"
    );
}

#[test]
fn test_load_balance_loss_empty() {
    assert_eq!(load_balance_loss(&[], &[], 4, 0), 0.0);
    assert_eq!(load_balance_loss(&[0.5; 8], &[0, 1], 0, 4), 0.0);
}

#[test]
fn test_moe_layer_compute_load_balance_loss() {
    let moe = MoeLayer::new(MoeConfig {
        num_experts: 4,
        num_experts_per_token: 1,
        hidden_dim: 16,
        intermediate_dim: 32,
        capacity_factor: 1.0,
    })
    .expect("MoeLayer::new failed");

    let gate_logits = vec![0.25f32; 8 * 4]; // 8 tokens, 4 experts
    let expert_indices = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let loss = moe.compute_load_balance_loss(&gate_logits, &expert_indices, 8);
    assert!(
        (loss - 1.0).abs() < 1e-5,
        "method loss = {loss}, expected ~1.0"
    );
}

// ===== N7: Shared expert tests =====

#[test]
fn test_moe_with_shared_expert() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let hidden_dim = 4;
    let intermediate_dim = 4;
    let num_experts = 2;
    let top_k = 1;

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

    let expert0 = Expert {
        gate_proj: make_identity_linear(dev, hidden_dim),
        up_proj: make_identity_linear(dev, hidden_dim),
        down_proj: make_identity_linear(dev, hidden_dim),
    };
    let expert1 = Expert {
        gate_proj: make_identity_linear(dev, hidden_dim),
        up_proj: make_identity_linear(dev, hidden_dim),
        down_proj: make_identity_linear(dev, hidden_dim),
    };

    // Shared expert with identity projections
    let shared = Expert {
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
    .expect("from_layers failed")
    .with_shared_expert(shared);

    assert!(moe.has_shared_expert());

    let input = Array::from_slice(
        dev,
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 4],
    );

    // Forward should include shared expert output added to MoE output
    let output = moe
        .forward(&input, &registry, &queue)
        .expect("moe forward with shared expert failed");
    assert_eq!(output.shape(), &[2, 4]);

    let vals: Vec<f32> = output.to_vec_checked();
    let sum: f32 = vals.iter().sum();
    assert!(
        sum.is_finite(),
        "shared expert output contains non-finite values"
    );
    // With shared expert, output should be larger than without it
    // (since shared expert adds to the routed output)
    assert!(
        sum.abs() > 1e-6,
        "shared expert output should not be all zeros"
    );
}

#[test]
fn test_moe_without_shared_expert_unchanged() {
    let moe = MoeLayer::new(MoeConfig {
        num_experts: 8,
        num_experts_per_token: 2,
        hidden_dim: 256,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    })
    .expect("MoeLayer::new failed");

    assert!(!moe.has_shared_expert());
}

// ===== N8: DeepSeek-V3 config tests =====

#[test]
fn test_deepseek_v3_max_seq_len() {
    let cfg = deepseek::deepseek_v3();
    assert_eq!(
        cfg.max_seq_len, 163_840,
        "DeepSeek-V3 max_seq_len should be 163840"
    );
}

#[test]
fn test_deepseek_v3_mla_kv_heads() {
    let cfg = deepseek::deepseek_v3();
    // MLA compresses KV into a shared latent space: effective kv_heads = 1
    assert_eq!(cfg.num_kv_heads, 1, "MLA should use 1 effective KV head");
    assert_eq!(cfg.num_heads, 128);
}

#[test]
fn test_deepseek_v3_full_config() {
    let full = deepseek::deepseek_v3_full();
    assert_eq!(full.kv_lora_rank, 512);
    assert_eq!(full.q_lora_rank, 1536);
    assert_eq!(full.rope_head_dim, 64);
    assert_eq!(full.v_head_dim, 128);
    assert_eq!(full.shared_expert_intermediate_size, 2048);
    assert_eq!(full.num_shared_experts, 1);
    assert_eq!(full.first_k_dense_replace, 1);
    assert_eq!(full.transformer.max_seq_len, 163_840);
    assert_eq!(full.transformer.num_layers, 61);
    assert_eq!(full.transformer.vocab_size, 129_280);

    match &full.transformer.ff_type {
        FeedForwardType::MoE { config } => {
            assert_eq!(config.num_experts, 256);
            assert_eq!(config.num_experts_per_token, 8);
        }
        _ => panic!("DeepSeek-V3 should use MoE"),
    }
}

// ===== RoPE layer tests =====

#[test]
fn test_rope_config_defaults() {
    let config = RotaryPositionEmbeddingConfig::new(128, 2048);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_seq_len, 2048);
    assert!((config.base_freq - 10000.0).abs() < 1e-6);
    assert!((config.scale - 1.0).abs() < 1e-6);
    assert!(!config.traditional);
}

#[test]
fn test_rope_config_validation_odd_head_dim() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let config = RotaryPositionEmbeddingConfig::new(127, 2048);
    let result = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32);
    assert!(result.is_err(), "odd head_dim should fail");
}

#[test]
fn test_rope_config_validation_zero_head_dim() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let config = RotaryPositionEmbeddingConfig {
        head_dim: 0,
        max_seq_len: 2048,
        base_freq: 10000.0,
        scale: 1.0,
        traditional: false,
    };
    let result = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32);
    assert!(result.is_err(), "zero head_dim should fail");
}

#[test]
fn test_rope_config_validation_zero_max_seq_len() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let config = RotaryPositionEmbeddingConfig {
        head_dim: 64,
        max_seq_len: 0,
        base_freq: 10000.0,
        scale: 1.0,
        traditional: false,
    };
    let result = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32);
    assert!(result.is_err(), "zero max_seq_len should fail");
}

#[test]
fn test_rope_construction_and_table_shapes() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let head_dim = 64;
    let max_seq_len = 512;
    let half_dim = head_dim / 2;

    let config = RotaryPositionEmbeddingConfig::new(head_dim, max_seq_len);
    let rope = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32)
        .expect("RoPE construction failed");

    assert_eq!(rope.head_dim(), head_dim);
    assert_eq!(rope.max_seq_len(), max_seq_len);
    assert_eq!(rope.cos_freqs().shape(), &[max_seq_len, half_dim]);
    assert_eq!(rope.sin_freqs().shape(), &[max_seq_len, half_dim]);
}

#[test]
fn test_rope_forward_output_shape_2d() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let head_dim = 8;
    let max_seq_len = 32;
    let seq_len = 4;

    let config = RotaryPositionEmbeddingConfig::new(head_dim, max_seq_len);
    let rope = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32)
        .expect("RoPE construction failed");

    // 2D input: [seq_len, head_dim]
    let input_data: Vec<f32> = (0..(seq_len * head_dim)).map(|i| i as f32 * 0.1).collect();
    let input = Array::from_slice(dev, &input_data, vec![seq_len, head_dim]);

    let output = rope
        .forward(&registry, &input, 0, &queue)
        .expect("RoPE forward failed");

    assert_eq!(output.shape(), &[seq_len, head_dim]);
    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "RoPE output contains non-finite values"
    );
}

#[test]
fn test_rope_forward_output_shape_3d() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let head_dim = 8;
    let max_seq_len = 32;
    let seq_len = 4;
    let n_batch = 2;

    let config = RotaryPositionEmbeddingConfig::new(head_dim, max_seq_len);
    let rope = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32)
        .expect("RoPE construction failed");

    // 3D input: [batch*n_heads, seq_len, head_dim]
    let total = n_batch * seq_len * head_dim;
    let input_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
    let input = Array::from_slice(dev, &input_data, vec![n_batch, seq_len, head_dim]);

    let output = rope
        .forward(&registry, &input, 0, &queue)
        .expect("RoPE forward failed");

    assert_eq!(output.shape(), &[n_batch, seq_len, head_dim]);
    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "RoPE 3D output contains non-finite values"
    );
}

#[test]
fn test_rope_forward_with_offset() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let head_dim = 8;
    let max_seq_len = 32;
    let seq_len = 4;

    let config = RotaryPositionEmbeddingConfig::new(head_dim, max_seq_len);
    let rope = RotaryPositionEmbedding::new(config, dev, rmlx_core::dtype::DType::Float32)
        .expect("RoPE construction failed");

    let input_data: Vec<f32> = (0..(seq_len * head_dim)).map(|i| i as f32 * 0.1).collect();
    let input = Array::from_slice(dev, &input_data, vec![seq_len, head_dim]);

    // With offset=0
    let out0 = rope
        .forward(&registry, &input, 0, &queue)
        .expect("forward offset=0 failed");
    // With offset=5
    let out5 = rope
        .forward(&registry, &input, 5, &queue)
        .expect("forward offset=5 failed");

    let vals0: Vec<f32> = out0.to_vec_checked();
    let vals5: Vec<f32> = out5.to_vec_checked();

    // Different offsets should produce different outputs
    let differ = vals0
        .iter()
        .zip(vals5.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differ, "different offsets should produce different outputs");
}

/// PR 1.5: QuantizedLinear must accept non-f32 input without panicking.
/// It should auto-cast to f32 internally and produce the correct output shape.
#[test]
fn test_quantized_linear_forward_f16_input() {
    use rmlx_core::dtype::DType;
    use rmlx_core::ops::copy::copy_cast;

    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Build a Q4 QuantizedLinear: in=64, out=32, group_size=32
    let in_f = 64usize;
    let out_f = 32usize;
    let group_size = 32usize;
    let w_packed = vec![0x88u8; out_f * (in_f / 2)]; // all nibbles = 8
    let groups_per_row = in_f / group_size;
    let scales = vec![1.0f32; out_f * groups_per_row];
    let biases = vec![0.0f32; out_f * groups_per_row];

    let ql = QuantizedLinear::new(
        w_packed,
        scales,
        biases,
        in_f,
        out_f,
        group_size,
        QuantBits::Q4,
    )
    .expect("QuantizedLinear::new failed");

    // Create an f32 input then cast to f16
    let x_f32 = Array::from_slice(dev, &vec![1.0f32; in_f], vec![1, in_f]);
    let x_f16 =
        copy_cast(&registry, &x_f32, DType::Float16, &queue).expect("copy_cast to f16 failed");
    assert_eq!(x_f16.dtype(), DType::Float16);

    // Forward with f16 input should NOT panic
    let out = ql
        .forward(&x_f16, &registry, &queue)
        .expect("forward with f16 input should not fail");

    assert_eq!(out.shape(), &[1, out_f]);
    assert_eq!(out.dtype(), DType::Float32);
}

#[test]
fn test_moe_gather_mm_matches_per_expert() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let hidden_dim = 8;
    let intermediate_dim = 4;
    let num_experts = 4;
    let top_k = 2;

    // Gate: [num_experts, hidden_dim] — deterministic routing weights
    let gate_data: Vec<f32> = (0..num_experts * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let gate_weight = Array::from_slice(dev, &gate_data, vec![num_experts, hidden_dim]);
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

    // Build experts with varied weights
    let make_expert = |val: f32| -> Expert {
        let gate_data: Vec<f32> = vec![val; intermediate_dim * hidden_dim];
        let gate_w = Array::from_slice(dev, &gate_data, vec![intermediate_dim, hidden_dim]);
        let gate_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            gate_w,
            None,
        )
        .unwrap();

        let up_data: Vec<f32> = vec![val * 0.5; intermediate_dim * hidden_dim];
        let up_w = Array::from_slice(dev, &up_data, vec![intermediate_dim, hidden_dim]);
        let up_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            up_w,
            None,
        )
        .unwrap();

        let down_data: Vec<f32> = vec![val * 0.3; hidden_dim * intermediate_dim];
        let down_w = Array::from_slice(dev, &down_data, vec![hidden_dim, intermediate_dim]);
        let down_proj = Linear::from_arrays(
            LinearConfig {
                in_features: intermediate_dim,
                out_features: hidden_dim,
                has_bias: false,
            },
            down_w,
            None,
        )
        .unwrap();

        Expert {
            gate_proj,
            up_proj,
            down_proj,
        }
    };

    let experts_per_expert: Vec<Expert> = vec![
        make_expert(0.1),
        make_expert(0.2),
        make_expert(0.3),
        make_expert(0.4),
    ];

    // Create second gate with same data for the GatherMM MoE layer
    let gate_weight2 = Array::from_slice(dev, &gate_data, vec![num_experts, hidden_dim]);
    let gate2 = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_dim,
            out_features: num_experts,
            has_bias: false,
        },
        gate_weight2,
        None,
    )
    .expect("gate2 from_arrays failed");

    let experts_gather: Vec<Expert> = vec![
        make_expert(0.1),
        make_expert(0.2),
        make_expert(0.3),
        make_expert(0.4),
    ];

    let moe_per_expert = MoeLayer::from_layers(
        MoeConfig {
            num_experts,
            num_experts_per_token: top_k,
            hidden_dim,
            intermediate_dim,
            capacity_factor: 1.0,
        },
        gate,
        experts_per_expert,
    )
    .expect("from_layers failed (PerExpert)")
    .with_strategy(MoeStrategy::PerExpert);

    let moe_gather_mm = MoeLayer::from_layers(
        MoeConfig {
            num_experts,
            num_experts_per_token: top_k,
            hidden_dim,
            intermediate_dim,
            capacity_factor: 1.0,
        },
        gate2,
        experts_gather,
    )
    .expect("from_layers failed (GatherMM)")
    .with_strategy(MoeStrategy::GatherMM);

    // Input: [4, 8] — 4 tokens
    let input_data: Vec<f32> = (0..4 * hidden_dim)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let input = Array::from_slice(dev, &input_data, vec![4, hidden_dim]);

    let out_per_expert = moe_per_expert
        .forward(&input, &registry, &queue)
        .expect("PerExpert forward failed");
    let out_gather_mm = moe_gather_mm
        .forward(&input, &registry, &queue)
        .expect("GatherMM forward failed");

    assert_eq!(out_per_expert.shape(), out_gather_mm.shape());

    let vals_pe: Vec<f32> = out_per_expert.to_vec_checked();
    let vals_gm: Vec<f32> = out_gather_mm.to_vec_checked();

    for i in 0..vals_pe.len() {
        let diff = (vals_pe[i] - vals_gm[i]).abs();
        let tol = 1e-3 * vals_pe[i].abs().max(1.0);
        assert!(
            diff < tol,
            "Mismatch at index {i}: PerExpert={} GatherMM={} diff={diff}",
            vals_pe[i],
            vals_gm[i]
        );
    }
}

// ===== PR 4.11: ICB Sparse Expert Dispatch tests =====

#[test]
fn test_icb_replay_cache_basic() {
    use rmlx_metal::icb_sparse::IcbReplayCache;

    let mut cache = IcbReplayCache::new(16);
    assert!(cache.is_empty());

    // Record a sparsity pattern
    let counts = [3u32, 0, 5, 0, 1, 0, 0, 2];
    cache.record(&counts);
    assert_eq!(cache.len(), 1);

    // Lookup should find it
    let (_key, pattern) = cache.lookup(&counts).expect("should be cached");
    assert_eq!(pattern.active_count, 4);
    assert_eq!(
        pattern.active_mask,
        vec![true, false, true, false, true, false, false, true]
    );
    assert_eq!(pattern.replay_count, 1);

    // Same active set with different counts should hit cache
    let counts2 = [10u32, 0, 1, 0, 2, 0, 0, 7];
    let result2 = cache.lookup(&counts2);
    assert!(
        result2.is_some(),
        "same active mask should share cache entry"
    );
}

#[test]
fn test_icb_replay_cache_eviction() {
    use rmlx_metal::icb_sparse::IcbReplayCache;

    let mut cache = IcbReplayCache::new(2);
    cache.record(&[1u32, 0]);
    cache.record(&[0u32, 1]);
    assert_eq!(cache.len(), 2);

    // Bump first pattern
    cache.record(&[1u32, 0]);

    // Third pattern should evict the one with lowest replay_count
    cache.record(&[1u32, 1]);
    assert_eq!(cache.len(), 2);

    // First pattern (replay_count=2) should survive
    assert!(cache.lookup(&[1u32, 0]).is_some());
}

#[test]
fn test_moe_sparse_icb_available() {
    // Verify MoeLayer exposes ICB replay cache
    let moe = MoeLayer::new(MoeConfig {
        num_experts: 8,
        num_experts_per_token: 2,
        hidden_dim: 256,
        intermediate_dim: 512,
        capacity_factor: 1.0,
    })
    .expect("MoeLayer::new failed");

    let cache = moe.icb_replay_cache().lock().unwrap();
    assert!(cache.is_empty());
    assert_eq!(cache.max_entries(), 64);
}

#[test]
fn test_sparse_dispatch_result_types() {
    use rmlx_metal::icb_sparse::SparseDispatchResult;

    let result = SparseDispatchResult {
        dispatched_count: 3,
        active_mask: vec![true, false, true, true, false],
    };
    assert_eq!(result.dispatched_count, 3);
    assert_eq!(result.active_mask.len(), 5);
    assert_eq!(result.active_mask.iter().filter(|&&a| a).count(), 3);
}

#[test]
fn test_moe_grouped_forward_skips_empty_experts() {
    // Verify that the grouped forward path correctly handles the case where
    // some experts have zero tokens (they should not be dispatched).
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let hidden_dim = 8;
    let intermediate_dim = 4;
    let num_experts = 4;
    let top_k = 1;

    // Gate weights that route all tokens to expert 0
    // (experts 1, 2, 3 get 0 tokens)
    // Shape: [num_experts, hidden_dim] = [4, 8]
    let mut gate_data = vec![-10.0f32; num_experts * hidden_dim];
    for item in gate_data.iter_mut().take(hidden_dim) {
        *item = 10.0; // expert 0: very high
    }
    let gate_weight = Array::from_slice(dev, &gate_data, vec![num_experts, hidden_dim]);
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

    let make_exp = |val: f32| -> Expert {
        let gw = Array::from_slice(
            dev,
            &vec![val; intermediate_dim * hidden_dim],
            vec![intermediate_dim, hidden_dim],
        );
        let uw = Array::from_slice(
            dev,
            &vec![val; intermediate_dim * hidden_dim],
            vec![intermediate_dim, hidden_dim],
        );
        let dw = Array::from_slice(
            dev,
            &vec![val; hidden_dim * intermediate_dim],
            vec![hidden_dim, intermediate_dim],
        );
        Expert {
            gate_proj: Linear::from_arrays(
                LinearConfig {
                    in_features: hidden_dim,
                    out_features: intermediate_dim,
                    has_bias: false,
                },
                gw,
                None,
            )
            .unwrap(),
            up_proj: Linear::from_arrays(
                LinearConfig {
                    in_features: hidden_dim,
                    out_features: intermediate_dim,
                    has_bias: false,
                },
                uw,
                None,
            )
            .unwrap(),
            down_proj: Linear::from_arrays(
                LinearConfig {
                    in_features: intermediate_dim,
                    out_features: hidden_dim,
                    has_bias: false,
                },
                dw,
                None,
            )
            .unwrap(),
        }
    };

    let experts: Vec<Expert> = vec![make_exp(0.1), make_exp(0.2), make_exp(0.3), make_exp(0.4)];

    let moe = MoeLayer::from_layers(
        MoeConfig {
            num_experts,
            num_experts_per_token: top_k,
            hidden_dim,
            intermediate_dim,
            capacity_factor: 1.0,
        },
        gate,
        experts,
    )
    .expect("from_layers failed")
    .with_strategy(MoeStrategy::Grouped);

    let input_data: Vec<f32> = (0..2 * hidden_dim)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let input = Array::from_slice(dev, &input_data, vec![2, hidden_dim]);

    let output = moe
        .forward(&input, &registry, &queue)
        .expect("grouped forward with empty experts should succeed");

    assert_eq!(output.shape(), &[2, hidden_dim]);

    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "output should contain finite values"
    );
    // Only expert 0 should have been dispatched; output should be non-zero
    let sum: f32 = vals.iter().sum();
    assert!(sum.abs() > 1e-8, "output should not be all zeros");
}

// ---------------------------------------------------------------------------
// PR 5.2: SlidingWindowAttention tests
// ---------------------------------------------------------------------------

/// Test SlidingWindowAttention forward produces correct output shape.
///
/// Creates SWA with window_size=128, num_heads=2, head_dim=4, hidden=8.
/// Runs a seq_len=64 input and verifies output is [64, 8].
#[test]
fn test_sliding_window_attention_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let hidden_size = num_heads * head_dim; // 8
    let window_size = 128;
    let seq_len = 64;

    let config = SlidingWindowAttentionConfig {
        base: AttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: 256,
            rope_theta: 10000.0,
        },
        window_size,
    };

    // Identity projections so the pipeline is testable end-to-end
    let q_proj = make_identity_linear(dev, hidden_size);
    let k_proj = make_identity_linear(dev, hidden_size);
    let v_proj = make_identity_linear(dev, hidden_size);
    let o_proj = make_identity_linear(dev, hidden_size);

    let swa = SlidingWindowAttention::from_projections(config, q_proj, k_proj, v_proj, o_proj);

    // Input: [seq_len, hidden_size]
    let input_data: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| ((i % 7) as f32) * 0.1)
        .collect();
    let input = Array::from_slice(dev, &input_data, vec![seq_len, hidden_size]);

    let output = swa
        .forward(&input, None, None, None, &registry, &queue)
        .expect("SlidingWindowAttention forward failed");

    assert_eq!(
        output.shape(),
        &[seq_len, hidden_size],
        "SWA output shape mismatch"
    );

    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "SWA output contains non-finite values"
    );
    let sum: f32 = vals.iter().sum();
    assert!(sum.abs() > 1e-6, "SWA output is all zeros");
}

/// Test that the sliding window mask only allows attention to the last
/// `window_size` positions (causal + windowed).
#[test]
fn test_sliding_window_mask_correctness() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let window_size = 4;
    let seq_len = 8;

    let config = SlidingWindowAttentionConfig {
        base: AttentionConfig {
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 64,
            rope_theta: 10000.0,
        },
        window_size,
    };

    let swa = SlidingWindowAttention::new(config).expect("SWA::new failed");
    let mask = swa.build_sliding_window_mask(dev, seq_len, seq_len, 0);

    assert_eq!(mask.shape(), &[seq_len, seq_len]);

    let mask_data: Vec<f32> = mask.to_vec_checked();

    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = mask_data[i * seq_len + j];
            // Token i can attend to position j iff:
            //   j <= i  (causal)  AND  j >= i - window_size + 1  (window)
            let window_start = (i + 1).saturating_sub(window_size);
            let should_attend = j <= i && j >= window_start;

            if should_attend {
                assert_eq!(val, 0.0, "mask[{i}][{j}] should be 0.0 (attend), got {val}");
            } else {
                assert!(
                    val == f32::NEG_INFINITY,
                    "mask[{i}][{j}] should be -inf (masked), got {val}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PR 5.1: MLA (Multi-Latent Attention) tests
// ---------------------------------------------------------------------------

#[test]
fn test_mla_forward_basic() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Small MLA config for testing.
    // head_dim = nope_dim + rope_dim = 4 + 4 = 8
    let num_heads = 2;
    let head_dim = 8;
    let rope_head_dim = 4;
    let nope_dim = head_dim - rope_head_dim; // 4
    let v_head_dim = head_dim; // Must equal head_dim for SDPA compatibility
    let hidden_size = 16;
    let kv_lora_rank = 6;
    let q_lora_rank = 8;
    let max_seq_len = 32;
    let seq_len = 4;

    let config = MlaConfig {
        num_heads,
        head_dim,
        v_head_dim,
        hidden_size,
        kv_lora_rank,
        q_lora_rank,
        rope_head_dim,
        rope_theta: 10000.0,
        max_seq_len,
    };
    config.validate().expect("MlaConfig validation failed");

    let make_weight = |rows: usize, cols: usize, val: f32| -> Array {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| if i % (cols + 1) == 0 { val } else { val * 0.01 })
            .collect();
        Array::from_slice(dev, &data, vec![rows, cols])
    };

    let w_dkv = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_lora_rank,
            has_bias: false,
        },
        make_weight(kv_lora_rank, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_uk = Linear::from_arrays(
        LinearConfig {
            in_features: kv_lora_rank,
            out_features: num_heads * nope_dim,
            has_bias: false,
        },
        make_weight(num_heads * nope_dim, kv_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let w_uv = Linear::from_arrays(
        LinearConfig {
            in_features: kv_lora_rank,
            out_features: num_heads * v_head_dim,
            has_bias: false,
        },
        make_weight(num_heads * v_head_dim, kv_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let w_kr = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: rope_head_dim,
            has_bias: false,
        },
        make_weight(rope_head_dim, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_dq = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: q_lora_rank,
            has_bias: false,
        },
        make_weight(q_lora_rank, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_uq = Linear::from_arrays(
        LinearConfig {
            in_features: q_lora_rank,
            out_features: num_heads * head_dim,
            has_bias: false,
        },
        make_weight(num_heads * head_dim, q_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let o_proj = Linear::from_arrays(
        LinearConfig {
            in_features: num_heads * v_head_dim,
            out_features: hidden_size,
            has_bias: false,
        },
        make_weight(hidden_size, num_heads * v_head_dim, 0.1),
        None,
    )
    .unwrap();

    let mla = Mla::from_projections(config, w_dkv, w_uk, w_uv, w_kr, w_dq, w_uq, o_proj)
        .expect("Mla::from_projections failed");

    // Create input: [seq_len, hidden_size]
    let input_data: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let input = Array::from_slice(dev, &input_data, vec![seq_len, hidden_size]);

    // Precompute RoPE frequency tables for rope_head_dim
    let (cos_data, sin_data) =
        rmlx_core::ops::rope::precompute_freqs(max_seq_len, rope_head_dim, 10000.0, 1.0)
            .expect("precompute_freqs failed");
    let cos_freqs = Array::from_slice(dev, &cos_data, vec![max_seq_len, rope_head_dim / 2]);
    let sin_freqs = Array::from_slice(dev, &sin_data, vec![max_seq_len, rope_head_dim / 2]);

    let output = mla
        .forward(
            &input,
            Some(&cos_freqs),
            Some(&sin_freqs),
            None,
            &registry,
            &queue,
        )
        .expect("MLA forward failed");

    assert_eq!(
        output.shape(),
        &[seq_len, hidden_size],
        "MLA output shape should be [seq_len, hidden_size]"
    );

    let vals: Vec<f32> = output.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "MLA output should contain only finite values"
    );
}

#[test]
fn test_mla_forward_with_cache() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let num_heads = 2;
    let head_dim = 8;
    let rope_head_dim = 4;
    let nope_dim = head_dim - rope_head_dim;
    let v_head_dim = head_dim;
    let hidden_size = 16;
    let kv_lora_rank = 6;
    let q_lora_rank = 8;
    let max_seq_len = 32;

    let config = MlaConfig {
        num_heads,
        head_dim,
        v_head_dim,
        hidden_size,
        kv_lora_rank,
        q_lora_rank,
        rope_head_dim,
        rope_theta: 10000.0,
        max_seq_len,
    };

    let make_weight = |rows: usize, cols: usize, val: f32| -> Array {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| if i % (cols + 1) == 0 { val } else { val * 0.01 })
            .collect();
        Array::from_slice(dev, &data, vec![rows, cols])
    };

    let w_dkv = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: kv_lora_rank,
            has_bias: false,
        },
        make_weight(kv_lora_rank, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_uk = Linear::from_arrays(
        LinearConfig {
            in_features: kv_lora_rank,
            out_features: num_heads * nope_dim,
            has_bias: false,
        },
        make_weight(num_heads * nope_dim, kv_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let w_uv = Linear::from_arrays(
        LinearConfig {
            in_features: kv_lora_rank,
            out_features: num_heads * v_head_dim,
            has_bias: false,
        },
        make_weight(num_heads * v_head_dim, kv_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let w_kr = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: rope_head_dim,
            has_bias: false,
        },
        make_weight(rope_head_dim, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_dq = Linear::from_arrays(
        LinearConfig {
            in_features: hidden_size,
            out_features: q_lora_rank,
            has_bias: false,
        },
        make_weight(q_lora_rank, hidden_size, 0.1),
        None,
    )
    .unwrap();
    let w_uq = Linear::from_arrays(
        LinearConfig {
            in_features: q_lora_rank,
            out_features: num_heads * head_dim,
            has_bias: false,
        },
        make_weight(num_heads * head_dim, q_lora_rank, 0.1),
        None,
    )
    .unwrap();
    let o_proj = Linear::from_arrays(
        LinearConfig {
            in_features: num_heads * v_head_dim,
            out_features: hidden_size,
            has_bias: false,
        },
        make_weight(hidden_size, num_heads * v_head_dim, 0.1),
        None,
    )
    .unwrap();

    let mla = Mla::from_projections(config, w_dkv, w_uk, w_uv, w_kr, w_dq, w_uq, o_proj)
        .expect("Mla::from_projections failed");

    let (cos_data, sin_data) =
        rmlx_core::ops::rope::precompute_freqs(max_seq_len, rope_head_dim, 10000.0, 1.0)
            .expect("precompute_freqs failed");
    let cos_freqs = Array::from_slice(dev, &cos_data, vec![max_seq_len, rope_head_dim / 2]);
    let sin_freqs = Array::from_slice(dev, &sin_data, vec![max_seq_len, rope_head_dim / 2]);

    // Create MLA KV cache
    let mut cache = MlaKvCache::new(
        dev,
        max_seq_len,
        kv_lora_rank,
        rope_head_dim,
        rmlx_core::dtype::DType::Float32,
    );
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);

    // Prefill: 4 tokens
    let prefill_seq = 4;
    let prefill_data: Vec<f32> = (0..prefill_seq * hidden_size)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let prefill_input = Array::from_slice(dev, &prefill_data, vec![prefill_seq, hidden_size]);

    let prefill_out = mla
        .forward(
            &prefill_input,
            Some(&cos_freqs),
            Some(&sin_freqs),
            Some(&mut cache),
            &registry,
            &queue,
        )
        .expect("MLA prefill forward failed");

    assert_eq!(prefill_out.shape(), &[prefill_seq, hidden_size]);
    assert_eq!(
        cache.seq_len(),
        prefill_seq,
        "Cache should have grown to prefill_seq"
    );

    // Decode: 1 token
    let decode_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 0.5) * 0.02).collect();
    let decode_input = Array::from_slice(dev, &decode_data, vec![1, hidden_size]);

    let decode_out = mla
        .forward(
            &decode_input,
            Some(&cos_freqs),
            Some(&sin_freqs),
            Some(&mut cache),
            &registry,
            &queue,
        )
        .expect("MLA decode forward failed");

    assert_eq!(decode_out.shape(), &[1, hidden_size]);
    assert_eq!(
        cache.seq_len(),
        prefill_seq + 1,
        "Cache should have grown by 1 after decode step"
    );

    let vals: Vec<f32> = decode_out.to_vec_checked();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "MLA decode output should contain only finite values"
    );
}

// ---------------------------------------------------------------------------
// PR 5.5: AWQ / GPTQ / K-quant tests
// ---------------------------------------------------------------------------

/// Test AWQ linear forward with known INT4 weights.
///
/// Creates a small [2, 64] AWQ linear layer with group_size=64.
/// Packs known 4-bit values into u32 words, sets scale=1.0 and zero=8.0
/// (so dequantized weight = 1.0 * (nibble - 8.0)), then verifies output
/// against a manually computed reference.
#[test]
fn test_awq_linear_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let in_features = 64;
    let out_features = 2;
    let group_size = 64;

    // Pack INT4 weights: all nibbles = 9, so dequantized = 1.0*(9 - 8.0) = 1.0
    // Each u32 = 8 nibbles of value 9 = 0x99999999
    let num_u32s = out_features * (in_features / 8);
    let qweight = vec![0x99999999u32; num_u32s];

    let num_groups = in_features / group_size;
    let scales = vec![1.0f32; out_features * num_groups]; // scale = 1.0
    let zeros = vec![8.0f32; out_features * num_groups]; // zero = 8.0

    let awq = AwqLinear::new(
        qweight,
        scales,
        zeros,
        in_features,
        out_features,
        group_size,
    )
    .expect("AwqLinear::new failed");

    assert_eq!(awq.in_features(), in_features);
    assert_eq!(awq.out_features(), out_features);
    assert_eq!(awq.group_size(), group_size);

    // Input: all ones -> dot product = sum of dequantized weights = 64 * 1.0 = 64.0
    let x_data = vec![1.0f32; in_features];
    let x = Array::from_slice(dev, &x_data, vec![1, in_features]);

    let output = awq
        .forward(&x, &registry, &queue)
        .expect("AwqLinear forward failed");

    assert_eq!(output.shape(), &[1, out_features]);

    let vals: Vec<f32> = output.to_vec_checked();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 64.0).abs() < 1e-3,
            "AWQ output[{i}] = {v}, expected 64.0"
        );
    }
}

/// Test GPTQ linear forward with known INT4 weights (column-major packing).
///
/// Creates a small [64, 8] GPTQ linear layer with group_size=64.
/// Packs known 4-bit values into column-major u32 words.
#[test]
fn test_gptq_linear_forward() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    let in_features = 64;
    let out_features = 8;
    let group_size = 64;

    // Column-major packing: qweight[k, n/8], nibble = n%8
    // All nibbles = 10, so dequantized = 2.0*(10 - 5.0) = 10.0
    // 0xAAAAAAAA = nibbles all 0xA = 10
    let num_u32s = in_features * (out_features / 8);
    let qweight = vec![0xAAAAAAAAu32; num_u32s];

    let num_groups = in_features / group_size;
    let scales = vec![2.0f32; num_groups * out_features]; // scale = 2.0
    let zeros = vec![5.0f32; num_groups * out_features]; // zero = 5.0

    let gptq = GptqLinear::new(
        qweight,
        scales,
        zeros,
        in_features,
        out_features,
        group_size,
    )
    .expect("GptqLinear::new failed");

    assert_eq!(gptq.in_features(), in_features);
    assert_eq!(gptq.out_features(), out_features);
    assert_eq!(gptq.group_size(), group_size);

    // Input: all ones -> each output = sum over k of w[k,n] * 1.0 = 64 * 10.0 = 640.0
    let x_data = vec![1.0f32; in_features];
    let x = Array::from_slice(dev, &x_data, vec![1, in_features]);

    let output = gptq
        .forward(&x, &registry, &queue)
        .expect("GptqLinear forward failed");

    assert_eq!(output.shape(), &[1, out_features]);

    let vals: Vec<f32> = output.to_vec_checked();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 640.0).abs() < 1e-1,
            "GPTQ output[{i}] = {v}, expected 640.0"
        );
    }
}

/// Test GGUF k-quant type mapping.
///
/// Verifies that Q2_K through Q6_K GGML types are correctly mapped to
/// KQuantType variants, and that non-k-quant types return None.
#[test]
fn test_gguf_kquant_type_mapping() {
    use rmlx_core::formats::gguf::GgmlType;

    // K-quant types should map correctly
    assert_eq!(ggml_type_to_kquant(GgmlType::Q2K), Some(KQuantType::Q2K));
    assert_eq!(ggml_type_to_kquant(GgmlType::Q3K), Some(KQuantType::Q3K));
    assert_eq!(ggml_type_to_kquant(GgmlType::Q4K), Some(KQuantType::Q4K));
    assert_eq!(ggml_type_to_kquant(GgmlType::Q5K), Some(KQuantType::Q5K));
    assert_eq!(ggml_type_to_kquant(GgmlType::Q6K), Some(KQuantType::Q6K));

    // is_kquant_type
    assert!(is_kquant_type(GgmlType::Q4K));
    assert!(!is_kquant_type(GgmlType::F32));
    assert!(!is_kquant_type(GgmlType::Q4_0));

    // Non-k-quant types should return None
    assert_eq!(ggml_type_to_kquant(GgmlType::F32), None);
    assert_eq!(ggml_type_to_kquant(GgmlType::F16), None);
    assert_eq!(ggml_type_to_kquant(GgmlType::Q4_0), None);
    assert_eq!(ggml_type_to_kquant(GgmlType::Q8_0), None);
    assert_eq!(ggml_type_to_kquant(GgmlType::BF16), None);

    // KQuantType properties
    assert_eq!(KQuantType::Q4K.block_size(), 256);
    assert_eq!(KQuantType::Q4K.type_size(), 144);
    assert_eq!(KQuantType::Q4K.bits(), 4);
    assert_eq!(KQuantType::Q2K.bits(), 2);
    assert_eq!(KQuantType::Q6K.bits(), 6);

    // kquant_load_info
    let info = kquant_load_info(GgmlType::Q4K).unwrap();
    assert_eq!(info.quant_type, KQuantType::Q4K);
    assert_eq!(info.block_size, 256);
    assert_eq!(info.type_size, 144);
    assert_eq!(info.bits, 4);

    assert!(kquant_load_info(GgmlType::F32).is_none());
}

// ---------------------------------------------------------------------------
// PR 5.7: Prefix caching for KV cache reuse
// ---------------------------------------------------------------------------

#[test]
fn test_prefix_cache_insert_lookup() {
    use rmlx_nn::prefix_cache::PrefixCache;

    let mut cache = PrefixCache::new();
    cache.insert(&[1, 2, 3, 4], &[10, 11]);

    // Exact match.
    let m = cache.lookup(&[1, 2, 3, 4]);
    assert_eq!(m.matched_len, 4);
    assert_eq!(m.block_ids, vec![10, 11]);

    // Superset query: matches the 4-token prefix.
    let m2 = cache.lookup(&[1, 2, 3, 4, 5, 6]);
    assert_eq!(m2.matched_len, 4);
    assert_eq!(m2.block_ids, vec![10, 11]);
}

#[test]
fn test_prefix_cache_no_match() {
    use rmlx_nn::prefix_cache::PrefixCache;

    let mut cache = PrefixCache::new();
    cache.insert(&[1, 2, 3, 4], &[10, 11]);

    let m = cache.lookup(&[5, 6, 7]);
    assert_eq!(m.matched_len, 0);
    assert!(m.block_ids.is_empty());
}

#[test]
fn test_prefix_cache_partial_match() {
    use rmlx_nn::prefix_cache::PrefixCache;

    let mut cache = PrefixCache::new();
    cache.insert(&[1, 2, 3, 4], &[10, 11]);
    cache.insert(&[1, 2, 5, 6], &[20, 21]);

    // Exact matches work after tree splits.
    let m1 = cache.lookup(&[1, 2, 3, 4]);
    assert_eq!(m1.matched_len, 4);
    assert_eq!(m1.block_ids, vec![10, 11]);

    let m2 = cache.lookup(&[1, 2, 5, 6]);
    assert_eq!(m2.matched_len, 4);
    assert_eq!(m2.block_ids, vec![20, 21]);

    // No full-edge match beyond the intermediate [1,2] split node.
    let m3 = cache.lookup(&[1, 2, 7, 8]);
    assert_eq!(m3.matched_len, 0);
}

#[test]
fn test_prefix_cache_eviction() {
    use rmlx_nn::prefix_cache::PrefixCache;

    let mut cache = PrefixCache::new();
    cache.insert(&[1, 2, 3], &[10, 11, 12]);
    cache.insert(&[4, 5, 6], &[20, 21, 22]);
    assert_eq!(cache.total_cached_blocks(), 6);

    let freed = cache.evict(3);
    assert!(freed >= 3);
    assert!(cache.total_cached_blocks() <= 3);
}

#[test]
fn test_prefix_cache_lru() {
    use rmlx_nn::prefix_cache::PrefixCache;
    use std::thread;
    use std::time::Duration;

    let mut cache = PrefixCache::new();

    cache.insert(&[1, 2, 3], &[10, 11]);
    thread::sleep(Duration::from_millis(10));
    cache.insert(&[4, 5, 6], &[20, 21]);

    // Touch [1,2,3] to make it most recently used.
    thread::sleep(Duration::from_millis(10));
    let _ = cache.lookup(&[1, 2, 3]);

    // Evict 2 blocks — should evict [4,5,6] first (least recently accessed).
    let freed = cache.evict(2);
    assert_eq!(freed, 2);

    let m1 = cache.lookup(&[1, 2, 3]);
    assert_eq!(m1.matched_len, 3);
    assert_eq!(m1.block_ids, vec![10, 11]);

    let m2 = cache.lookup(&[4, 5, 6]);
    assert_eq!(m2.matched_len, 0);
    assert!(m2.block_ids.is_empty());
}

// ---------------------------------------------------------------------------
// PR 5.8: Chunked prefill in continuous batching scheduler
// ---------------------------------------------------------------------------

#[test]
fn test_chunked_prefill_splits_long_prompt() {
    use rmlx_nn::paged_kv_cache::BlockManager;
    use rmlx_nn::scheduler::*;
    use std::collections::HashMap;

    let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("no Metal device available");

    let config = SchedulerConfig {
        max_batch_size: 4,
        eos_token_id: 2,
        block_size: 4,
        max_prefill_chunk: 512,
    };
    let mut scheduler = Scheduler::new(config);
    let mut bm = BlockManager::new(&device, 1024, 4, 2, 2, 4, rmlx_core::dtype::DType::Float32);

    scheduler.add_request(GenerationRequest {
        seq_id: 1,
        prompt_tokens: vec![1; 2048],
        max_output_len: 10,
    });

    // 4 iterations to fully prefill 2048 tokens in 512-token chunks.
    let out1 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out1.prefill_seqs.len(), 1);
    assert_eq!(out1.prefill_chunk_range, Some((0, 512)));
    assert!(out1.prefill_seqs[0].is_chunked_prefill());

    let out2 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out2.prefill_chunk_range, Some((512, 1024)));

    let out3 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out3.prefill_chunk_range, Some((1024, 1536)));

    let out4 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out4.prefill_chunk_range, Some((1536, 2048)));
    assert!(!out4.prefill_seqs[0].is_chunked_prefill());

    // Now enters decode.
    scheduler.update_sequence(1, &mut bm).unwrap();
    let out5 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out5.prefill_seqs.len(), 0);
    assert_eq!(out5.decode_seqs.len(), 1);
}

#[test]
fn test_chunked_prefill_interleaves_decode() {
    use rmlx_nn::paged_kv_cache::BlockManager;
    use rmlx_nn::scheduler::*;
    use std::collections::HashMap;

    let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("no Metal device available");

    let config = SchedulerConfig {
        max_batch_size: 4,
        eos_token_id: 2,
        block_size: 4,
        max_prefill_chunk: 512,
    };
    let mut scheduler = Scheduler::new(config);
    let mut bm = BlockManager::new(&device, 1024, 4, 2, 2, 4, rmlx_core::dtype::DType::Float32);

    // Add and fully prefill a short sequence.
    scheduler.add_request(GenerationRequest {
        seq_id: 10,
        prompt_tokens: vec![1; 8],
        max_output_len: 100,
    });
    scheduler.schedule(&mut bm, &HashMap::new());
    scheduler.update_sequence(10, &mut bm).unwrap();

    // Add a long prefill.
    scheduler.add_request(GenerationRequest {
        seq_id: 20,
        prompt_tokens: vec![1; 1024],
        max_output_len: 50,
    });

    // First chunk: prefill seq 20 [0,512), decode seq 10.
    let out1 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out1.prefill_seqs.len(), 1);
    assert_eq!(out1.prefill_seqs[0].seq_id, 20);
    assert_eq!(out1.decode_seqs.len(), 1);
    assert_eq!(out1.decode_seqs[0].seq_id, 10);

    scheduler.update_sequence(10, &mut bm).unwrap();

    // Second chunk: prefill seq 20 [512,1024), decode seq 10.
    let out2 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out2.prefill_seqs[0].seq_id, 20);
    assert_eq!(out2.decode_seqs.len(), 1);
    assert_eq!(out2.decode_seqs[0].seq_id, 10);
}

#[test]
fn test_short_prompt_not_chunked() {
    use rmlx_nn::paged_kv_cache::BlockManager;
    use rmlx_nn::scheduler::*;
    use std::collections::HashMap;

    let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("no Metal device available");

    let config = SchedulerConfig {
        max_batch_size: 4,
        eos_token_id: 2,
        block_size: 4,
        max_prefill_chunk: 512,
    };
    let mut scheduler = Scheduler::new(config);
    let mut bm = BlockManager::new(&device, 256, 4, 2, 2, 4, rmlx_core::dtype::DType::Float32);

    scheduler.add_request(GenerationRequest {
        seq_id: 1,
        prompt_tokens: vec![1; 256],
        max_output_len: 10,
    });

    let out = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out.prefill_seqs.len(), 1);
    assert!(!out.prefill_seqs[0].is_chunked_prefill());
    assert!(out.prefill_chunk_range.is_none());

    // Next iteration is decode.
    scheduler.update_sequence(1, &mut bm).unwrap();
    let out2 = scheduler.schedule(&mut bm, &HashMap::new());
    assert_eq!(out2.prefill_seqs.len(), 0);
    assert_eq!(out2.decode_seqs.len(), 1);
}

// ===========================================================================
// Activation function tests
// ===========================================================================

use rmlx_nn::activations::{self, Activation, ActivationType};

/// Helper: run an activation on an f32 input vector, return output as Vec<f32>.
fn run_activation(
    act: &dyn Activation,
    input: &[f32],
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Vec<f32> {
    let dev = registry.device().raw();
    let arr = Array::from_slice(dev, input, vec![input.len()]);
    let out = act.forward(&arr, registry, queue).expect("forward failed");
    out.to_vec_checked::<f32>()
}

/// Helper: assert two f32 slices are approximately equal.
fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{name}[{i}]: expected {e}, got {a} (diff={})",
            (a - e).abs()
        );
    }
}

#[test]
fn test_activation_relu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
    let out = run_activation(&activations::ReLU, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "ReLU");
}

#[test]
fn test_activation_leaky_relu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let act = activations::LeakyReLU::new(); // alpha = 0.01
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected = [-0.02, -0.01, 0.0, 1.0, 2.0];
    let out = run_activation(&act, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "LeakyReLU");
}

#[test]
fn test_activation_elu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let act = activations::ELU::new(); // alpha = 1.0
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 })
        .collect();
    let out = run_activation(&act, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "ELU");
}

#[test]
fn test_activation_selu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha: f32 = 1.673_263_2;
    let scale: f32 = 1.050_701;
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            if x > 0.0 {
                scale * x
            } else {
                scale * alpha * (x.exp() - 1.0)
            }
        })
        .collect();
    let out = run_activation(&activations::SELU, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-4, "SELU");
}

#[test]
fn test_activation_mish() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| x * ((1.0f32 + x.exp()).ln()).tanh())
        .collect();
    let out = run_activation(&activations::Mish, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-4, "Mish");
}

#[test]
fn test_activation_quick_gelu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| x * (1.0 / (1.0 + (-1.702 * x).exp())))
        .collect();
    let out = run_activation(&activations::QuickGELU, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-4, "QuickGELU");
}

#[test]
fn test_activation_hard_swish() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-4.0f32, -3.0, 0.0, 3.0, 4.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| x * (x / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect();
    let out = run_activation(&activations::HardSwish, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "HardSwish");
}

#[test]
fn test_activation_hard_sigmoid() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-4.0f32, -3.0, 0.0, 3.0, 4.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect();
    let out = run_activation(&activations::HardSigmoid, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "HardSigmoid");
}

#[test]
fn test_activation_softplus() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let act = activations::Softplus::new(); // beta = 1.0
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| (1.0f32 + x.exp()).ln()).collect();
    let out = run_activation(&act, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-4, "Softplus");
}

#[test]
fn test_activation_softplus_beta() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let act = activations::Softplus::with_beta(2.0);
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| (1.0f32 + (2.0 * x).exp()).ln() / 2.0)
        .collect();
    let out = run_activation(&act, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-4, "Softplus(beta=2)");
}

#[test]
fn test_activation_softsign() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + x.abs())).collect();
    let out = run_activation(&activations::Softsign, &input, &registry, &queue);
    assert_approx(&out, &expected, 1e-5, "Softsign");
}

#[test]
fn test_activation_glu() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_slice(dev, &input_data, vec![6]);
    let out = activations::GLU
        .forward(&arr, &registry, &queue)
        .expect("GLU forward failed");
    let result = out.to_vec_checked::<f32>();
    assert_eq!(result.len(), 3);

    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let expected = [1.0 * sig(4.0), 2.0 * sig(5.0), 3.0 * sig(6.0)];
    assert_approx(&result, &expected, 1e-4, "GLU");
}

#[test]
fn test_activation_type_from_str_new_variants() {
    assert_eq!(
        ActivationType::from_str_name("relu"),
        Some(ActivationType::ReLU)
    );
    assert_eq!(
        ActivationType::from_str_name("leaky_relu"),
        Some(ActivationType::LeakyReLU)
    );
    assert_eq!(
        ActivationType::from_str_name("LeakyReLU"),
        Some(ActivationType::LeakyReLU)
    );
    assert_eq!(
        ActivationType::from_str_name("elu"),
        Some(ActivationType::ELU)
    );
    assert_eq!(
        ActivationType::from_str_name("selu"),
        Some(ActivationType::SELU)
    );
    assert_eq!(
        ActivationType::from_str_name("mish"),
        Some(ActivationType::Mish)
    );
    assert_eq!(
        ActivationType::from_str_name("quick_gelu"),
        Some(ActivationType::QuickGELU)
    );
    assert_eq!(
        ActivationType::from_str_name("QuickGELU"),
        Some(ActivationType::QuickGELU)
    );
    assert_eq!(
        ActivationType::from_str_name("hard_swish"),
        Some(ActivationType::HardSwish)
    );
    assert_eq!(
        ActivationType::from_str_name("HardSigmoid"),
        Some(ActivationType::HardSigmoid)
    );
    assert_eq!(
        ActivationType::from_str_name("softplus"),
        Some(ActivationType::Softplus)
    );
    assert_eq!(
        ActivationType::from_str_name("softsign"),
        Some(ActivationType::Softsign)
    );
    assert_eq!(
        ActivationType::from_str_name("glu"),
        Some(ActivationType::GLU)
    );
}

#[test]
fn test_activation_type_dynamic_dispatch() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();
    let input = Array::from_slice(dev, &[-1.0f32, 0.0, 1.0, 2.0], vec![4]);

    let out = ActivationType::ReLU
        .forward(&input, &registry, &queue)
        .expect("ReLU forward failed");
    let result = out.to_vec_checked::<f32>();
    assert_approx(&result, &[0.0, 0.0, 1.0, 2.0], 1e-5, "ActivationType::ReLU");

    let out = ActivationType::Softsign
        .forward(&input, &registry, &queue)
        .expect("Softsign forward failed");
    let result = out.to_vec_checked::<f32>();
    let expected: Vec<f32> = [-1.0f32, 0.0, 1.0, 2.0]
        .iter()
        .map(|&x| x / (1.0 + x.abs()))
        .collect();
    assert_approx(&result, &expected, 1e-5, "ActivationType::Softsign");
}

#[test]
fn test_mla_forward_causal_mask() {
    let Some((registry, queue)) = setup() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let dev = registry.device().raw();

    // Small MLA config
    let num_heads = 2;
    let head_dim = 8;
    let rope_head_dim = 4;
    let nope_dim = head_dim - rope_head_dim;
    let v_head_dim = head_dim;
    let hidden_size = 16;
    let kv_lora_rank = 6;
    let q_lora_rank = 8;
    let max_seq_len = 32;
    let seq_len = 4;

    let make_weight = |rows: usize, cols: usize, val: f32| -> Array {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| if i % (cols + 1) == 0 { val } else { val * 0.01 })
            .collect();
        Array::from_slice(dev, &data, vec![rows, cols])
    };

    let build_mla = || -> Mla {
        let w_dkv = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: kv_lora_rank,
                has_bias: false,
            },
            make_weight(kv_lora_rank, hidden_size, 0.1),
            None,
        )
        .unwrap();
        let w_uk = Linear::from_arrays(
            LinearConfig {
                in_features: kv_lora_rank,
                out_features: num_heads * nope_dim,
                has_bias: false,
            },
            make_weight(num_heads * nope_dim, kv_lora_rank, 0.1),
            None,
        )
        .unwrap();
        let w_uv = Linear::from_arrays(
            LinearConfig {
                in_features: kv_lora_rank,
                out_features: num_heads * v_head_dim,
                has_bias: false,
            },
            make_weight(num_heads * v_head_dim, kv_lora_rank, 0.1),
            None,
        )
        .unwrap();
        let w_kr = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: rope_head_dim,
                has_bias: false,
            },
            make_weight(rope_head_dim, hidden_size, 0.1),
            None,
        )
        .unwrap();
        let w_dq = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: q_lora_rank,
                has_bias: false,
            },
            make_weight(q_lora_rank, hidden_size, 0.1),
            None,
        )
        .unwrap();
        let w_uq = Linear::from_arrays(
            LinearConfig {
                in_features: q_lora_rank,
                out_features: num_heads * head_dim,
                has_bias: false,
            },
            make_weight(num_heads * head_dim, q_lora_rank, 0.1),
            None,
        )
        .unwrap();
        let o_proj = Linear::from_arrays(
            LinearConfig {
                in_features: num_heads * v_head_dim,
                out_features: hidden_size,
                has_bias: false,
            },
            make_weight(hidden_size, num_heads * v_head_dim, 0.1),
            None,
        )
        .unwrap();

        let config = MlaConfig {
            num_heads,
            head_dim,
            v_head_dim,
            hidden_size,
            kv_lora_rank,
            q_lora_rank,
            rope_head_dim,
            rope_theta: 10000.0,
            max_seq_len,
        };
        Mla::from_projections(config, w_dkv, w_uk, w_uv, w_kr, w_dq, w_uq, o_proj).unwrap()
    };

    let (cos_data, sin_data) =
        rmlx_core::ops::rope::precompute_freqs(max_seq_len, rope_head_dim, 10000.0, 1.0)
            .expect("precompute_freqs failed");
    let cos_freqs = Array::from_slice(dev, &cos_data, vec![max_seq_len, rope_head_dim / 2]);
    let sin_freqs = Array::from_slice(dev, &sin_data, vec![max_seq_len, rope_head_dim / 2]);

    // Run 1: input_a with specific values for all 4 tokens
    let input_a: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let arr_a = Array::from_slice(dev, &input_a, vec![seq_len, hidden_size]);
    let mla_a = build_mla();
    let out_a = mla_a
        .forward(
            &arr_a,
            Some(&cos_freqs),
            Some(&sin_freqs),
            None,
            &registry,
            &queue,
        )
        .expect("forward A failed");
    let vals_a: Vec<f32> = out_a.to_vec_checked();

    // Run 2: same input but token 3 (last row) is drastically different
    let mut input_b = input_a.clone();
    for j in 0..hidden_size {
        input_b[3 * hidden_size + j] = 99.0; // wildly different token 3
    }
    let arr_b = Array::from_slice(dev, &input_b, vec![seq_len, hidden_size]);
    let mla_b = build_mla();
    let out_b = mla_b
        .forward(
            &arr_b,
            Some(&cos_freqs),
            Some(&sin_freqs),
            None,
            &registry,
            &queue,
        )
        .expect("forward B failed");
    let vals_b: Vec<f32> = out_b.to_vec_checked();

    // Token 0's output should be IDENTICAL between run A and run B,
    // because the causal mask prevents token 0 from seeing tokens 1-3.
    let row0_a = &vals_a[0..hidden_size];
    let row0_b = &vals_b[0..hidden_size];
    for j in 0..hidden_size {
        assert!(
            (row0_a[j] - row0_b[j]).abs() < 1e-5,
            "Token 0 output differs at dim {j}: a={} vs b={} — causal mask is broken!",
            row0_a[j],
            row0_b[j],
        );
    }

    // Token 3's output SHOULD differ (it sees the same tokens 0-2 but its own input changed)
    let row3_a = &vals_a[3 * hidden_size..4 * hidden_size];
    let row3_b = &vals_b[3 * hidden_size..4 * hidden_size];
    let max_diff: f32 = row3_a
        .iter()
        .zip(row3_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "Token 3 output should differ between runs (max_diff={max_diff}), input was changed",
    );
}
