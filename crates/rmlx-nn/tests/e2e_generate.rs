//! End-to-end smoke test: build a tiny transformer model with random weights,
//! run a forward pass, apply sampler, and verify the pipeline completes
//! without panics and produces valid token IDs.
//!
//! This does NOT check for coherent text — random weights produce random
//! outputs. The goal is to verify the full wiring:
//!   Embedding -> N x TransformerBlock -> RMSNorm -> LM Head -> Sampler

use objc2::runtime::ProtocolObject;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::types::MtlQueue;
use rmlx_nn::attention::{Attention, AttentionConfig};
use rmlx_nn::embedding::{Embedding, EmbeddingConfig};
use rmlx_nn::linear::{Linear, LinearConfig};
use rmlx_nn::sampler::{Sampler, SamplerConfig};
use rmlx_nn::transformer::{
    FeedForward, FeedForwardType, TransformerBlock, TransformerConfig, TransformerModel,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Tiny model dimensions for fast tests.
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 16; // HIDDEN / HEADS
const LAYERS: usize = 2;
const VOCAB: usize = 256;
const INTERMEDIATE: usize = 128;
const MAX_SEQ: usize = 64;

fn setup() -> Option<(KernelRegistry, MtlQueue)> {
    let gpu = GpuDevice::system_default().ok()?;
    let queue = gpu.new_command_queue();
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("register ops");
    Some((registry, queue))
}

/// Create a random-ish f32 array on the Metal device.
///
/// Uses a deterministic pattern (not truly random) so the test is reproducible.
fn pseudo_random_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u32,
) -> Array {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| {
            // Simple hash-based pseudo-random in [-0.1, 0.1]
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.2 - 0.1
        })
        .collect();
    Array::from_slice(device, &data, shape.to_vec())
}

/// Build a tiny TransformerModel with random weights suitable for smoke testing.
fn build_tiny_model(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> TransformerModel {
    let config = TransformerConfig {
        hidden_size: HIDDEN,
        num_heads: HEADS,
        num_kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        num_layers: LAYERS,
        vocab_size: VOCAB,
        max_seq_len: MAX_SEQ,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: INTERMEDIATE,
        },
    };

    // Embedding: [VOCAB, HIDDEN]
    let embed_weight = pseudo_random_array(device, &[VOCAB, HIDDEN], 42);
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: VOCAB,
            embed_dim: HIDDEN,
        },
        embed_weight,
    )
    .expect("embedding");

    // Build transformer blocks
    let mut layers = Vec::with_capacity(LAYERS);
    for layer_idx in 0..LAYERS {
        let seed_base = (layer_idx as u32 + 1) * 1000;

        // Attention projections
        let attn_config = AttentionConfig {
            num_heads: HEADS,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            max_seq_len: MAX_SEQ,
            rope_theta: 10000.0,
        };
        let q_weight = pseudo_random_array(device, &[HEADS * HEAD_DIM, HIDDEN], seed_base);
        let k_weight = pseudo_random_array(device, &[KV_HEADS * HEAD_DIM, HIDDEN], seed_base + 1);
        let v_weight = pseudo_random_array(device, &[KV_HEADS * HEAD_DIM, HIDDEN], seed_base + 2);
        let o_weight = pseudo_random_array(device, &[HIDDEN, HEADS * HEAD_DIM], seed_base + 3);

        let q_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HIDDEN,
                out_features: HEADS * HEAD_DIM,
                has_bias: false,
            },
            q_weight,
            None,
        )
        .unwrap();
        let k_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HIDDEN,
                out_features: KV_HEADS * HEAD_DIM,
                has_bias: false,
            },
            k_weight,
            None,
        )
        .unwrap();
        let v_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HIDDEN,
                out_features: KV_HEADS * HEAD_DIM,
                has_bias: false,
            },
            v_weight,
            None,
        )
        .unwrap();
        let o_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HEADS * HEAD_DIM,
                out_features: HIDDEN,
                has_bias: false,
            },
            o_weight,
            None,
        )
        .unwrap();

        let attention =
            Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

        // FFN (Gated / SwiGLU)
        let gate_w = pseudo_random_array(device, &[INTERMEDIATE, HIDDEN], seed_base + 10);
        let up_w = pseudo_random_array(device, &[INTERMEDIATE, HIDDEN], seed_base + 11);
        let down_w = pseudo_random_array(device, &[HIDDEN, INTERMEDIATE], seed_base + 12);

        let gate_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HIDDEN,
                out_features: INTERMEDIATE,
                has_bias: false,
            },
            gate_w,
            None,
        )
        .unwrap();
        let up_proj = Linear::from_arrays(
            LinearConfig {
                in_features: HIDDEN,
                out_features: INTERMEDIATE,
                has_bias: false,
            },
            up_w,
            None,
        )
        .unwrap();
        let down_proj = Linear::from_arrays(
            LinearConfig {
                in_features: INTERMEDIATE,
                out_features: HIDDEN,
                has_bias: false,
            },
            down_w,
            None,
        )
        .unwrap();

        let ffn = FeedForward::Gated {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_merged_weight: None,
            gate_up_merged_weight_t: None,
            gate_proj_quantized: None,
            up_proj_quantized: None,
            down_proj_quantized: None,
        };

        // Norm weights: [HIDDEN], initialized to 1.0 (like a real RMSNorm init)
        let ones: Vec<f32> = vec![1.0; HIDDEN];
        let norm1_weight = Array::from_slice(device, &ones, vec![HIDDEN]);
        let norm2_weight = Array::from_slice(device, &ones, vec![HIDDEN]);

        let block = TransformerBlock::from_parts(
            layer_idx,
            attention,
            ffn,
            norm1_weight,
            norm2_weight,
            1e-5,
        );
        layers.push(block);
    }

    // Final norm
    let final_norm_data: Vec<f32> = vec![1.0; HIDDEN];
    let final_norm = Array::from_slice(device, &final_norm_data, vec![HIDDEN]);

    // LM head: [VOCAB, HIDDEN]
    let lm_head_weight = pseudo_random_array(device, &[VOCAB, HIDDEN], 9999);
    let lm_head = Linear::from_arrays(
        LinearConfig {
            in_features: HIDDEN,
            out_features: VOCAB,
            has_bias: false,
        },
        lm_head_weight,
        None,
    )
    .unwrap();

    TransformerModel::from_parts(config, embedding, layers, final_norm, lm_head)
        .expect("model construction")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn e2e_forward_produces_valid_logits_shape() {
    let Some((registry, queue)) = setup() else {
        eprintln!("Skipping: no Metal GPU");
        return;
    };
    let device = registry.device().raw();
    let mut model = build_tiny_model(device);

    let input_tokens: Vec<u32> = vec![1, 42, 100, 7]; // 4 tokens
    let seq_len = input_tokens.len();

    let logits = model
        .forward(&input_tokens, None, None, None, None, &registry, &queue)
        .expect("forward pass");

    // Output shape: [seq_len, vocab_size]
    assert_eq!(logits.ndim(), 2, "logits should be 2D");
    assert_eq!(logits.shape()[0], seq_len, "logits batch dim = seq_len");
    assert_eq!(logits.shape()[1], VOCAB, "logits vocab dim = VOCAB");
    assert_eq!(logits.dtype(), DType::Float32);
}

#[test]
fn e2e_forward_then_sample_produces_valid_token() {
    let Some((registry, queue)) = setup() else {
        eprintln!("Skipping: no Metal GPU");
        return;
    };
    let device = registry.device().raw();
    let mut model = build_tiny_model(device);

    let input_tokens: Vec<u32> = vec![1, 42, 100];

    let logits = model
        .forward(&input_tokens, None, None, None, None, &registry, &queue)
        .expect("forward pass");

    // Extract the last token's logits for next-token prediction: [vocab_size]
    let seq_len = logits.shape()[0];
    let last_row_offset = (seq_len - 1) * VOCAB * DType::Float32.size_of();
    let last_logits = logits.view(vec![VOCAB], vec![1], last_row_offset);

    // Sample with greedy decoding (temperature=0)
    let config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let token = Sampler::sample_token(&last_logits, &input_tokens, &config, &registry, &queue)
        .expect("sample_token");

    assert!(
        (token as usize) < VOCAB,
        "sampled token {token} must be in [0, {VOCAB})"
    );
}

#[test]
fn e2e_greedy_decode_multi_step() {
    let Some((registry, queue)) = setup() else {
        eprintln!("Skipping: no Metal GPU");
        return;
    };
    let device = registry.device().raw();
    let mut model = build_tiny_model(device);

    let config = SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    };

    let mut tokens: Vec<u32> = vec![1]; // start token
    let num_generate = 5;

    for _ in 0..num_generate {
        let logits = model
            .forward(&tokens, None, None, None, None, &registry, &queue)
            .expect("forward");

        let seq_len = logits.shape()[0];
        let last_row_offset = (seq_len - 1) * VOCAB * DType::Float32.size_of();
        let last_logits = logits.view(vec![VOCAB], vec![1], last_row_offset);

        let next_token = Sampler::sample_token(&last_logits, &tokens, &config, &registry, &queue)
            .expect("sample");

        assert!(
            (next_token as usize) < VOCAB,
            "generated token {next_token} out of range"
        );
        tokens.push(next_token);
    }

    assert_eq!(
        tokens.len(),
        1 + num_generate,
        "should have start + generated tokens"
    );

    // With greedy decoding and deterministic pseudo-random weights,
    // the same input should produce the same output every time.
    // Run again and verify determinism.
    let mut tokens2: Vec<u32> = vec![1];
    for _ in 0..num_generate {
        let logits = model
            .forward(&tokens2, None, None, None, None, &registry, &queue)
            .expect("forward 2");

        let seq_len = logits.shape()[0];
        let last_row_offset = (seq_len - 1) * VOCAB * DType::Float32.size_of();
        let last_logits = logits.view(vec![VOCAB], vec![1], last_row_offset);

        let next_token = Sampler::sample_token(&last_logits, &tokens2, &config, &registry, &queue)
            .expect("sample 2");
        tokens2.push(next_token);
    }

    assert_eq!(tokens, tokens2, "greedy decode should be deterministic");
}

#[test]
fn e2e_stochastic_sample_tokens_in_range() {
    let Some((registry, queue)) = setup() else {
        eprintln!("Skipping: no Metal GPU");
        return;
    };
    let device = registry.device().raw();
    let mut model = build_tiny_model(device);

    let input_tokens: Vec<u32> = vec![5, 10, 20];

    let logits = model
        .forward(&input_tokens, None, None, None, None, &registry, &queue)
        .expect("forward");

    let seq_len = logits.shape()[0];
    let last_row_offset = (seq_len - 1) * VOCAB * DType::Float32.size_of();
    let last_logits = logits.view(vec![VOCAB], vec![1], last_row_offset);

    // Sample with temperature > 0 (stochastic)
    let config = SamplerConfig {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    for _ in 0..10 {
        let token = Sampler::sample_token(&last_logits, &input_tokens, &config, &registry, &queue)
            .expect("stochastic sample");
        assert!(
            (token as usize) < VOCAB,
            "stochastic token {token} must be in [0, {VOCAB})"
        );
    }
}

#[test]
fn e2e_model_config_accessible() {
    let Some((registry, _queue)) = setup() else {
        eprintln!("Skipping: no Metal GPU");
        return;
    };
    let device = registry.device().raw();
    let model = build_tiny_model(device);

    assert_eq!(model.config().hidden_size, HIDDEN);
    assert_eq!(model.config().num_heads, HEADS);
    assert_eq!(model.config().vocab_size, VOCAB);
    assert_eq!(model.num_layers(), LAYERS);
}
