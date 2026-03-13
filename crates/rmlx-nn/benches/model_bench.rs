//! ✅ PRODUCTION PATH — tests forward_graph() and forward_prefill_graph() with real
//! safetensors weights on a full 32-layer TransformerModel. Results directly reflect
//! production decode and prefill throughput.
//!
//! 32-Layer Full Model Benchmark: RMLX vs MLX
//!
//! End-to-end TransformerModel forward latency with real safetensors weights.
//!
//! **RMLX paths measured:**
//! - `forward()` — baseline per-layer submit+wait
//! - `forward_graph()` — ExecGraph, submit-only + 1 sync (J-3)
//! - `forward_prefill_graph()` — prefill ExecGraph variant
//!
//! **MLX reference:**
//! Run the companion script `benchmarks/mlx_model_bench.py` on the same machine
//! with the same model to get MLX numbers, then compare side-by-side.
//!
//! ## Usage
//!
//! ```bash
//! # RMLX (Rust):
//! RMLX_MODEL_DIR=~/models/YourModel \
//!   cargo bench -p rmlx-nn --bench model_bench
//!
//! # MLX (Python) — run separately for comparison:
//! python benchmarks/mlx_model_bench.py --model-dir ~/models/YourModel
//! ```
//!
//! `RMLX_MODEL_DIR` should point to a HuggingFace model directory containing:
//! - `*.safetensors` weight files
//! - `config.json` with model architecture parameters
//!
//! Optionally set `MLX_DECODE_US` and `MLX_PREFILL_US` env vars to embed
//! MLX reference numbers in the output for direct comparison.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_nn::attention::{AttentionConfig, LayerKvCache};
use rmlx_nn::embedding::{Embedding, EmbeddingConfig};
use rmlx_nn::linear::{Linear, LinearConfig};
use rmlx_nn::safetensors_loader::{
    load_safetensors_sharded, load_safetensors_weights, parse_quantization_config,
};
use rmlx_nn::transformer::{
    FeedForward, FeedForwardType, TransformerBlock, TransformerConfig, TransformerModel,
};
use rmlx_nn::Attention;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    min: f64,
    max: f64,
    count: usize,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = micros.iter().sum();
        let mean = sum / n as f64;
        let variance: f64 = micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let p50 = percentile(&micros, 50.0);
        let p95 = percentile(&micros, 95.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            min: micros[0],
            max: micros[n - 1],
            count: n,
        }
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - (rank - lower as f64)) + sorted[upper] * (rank - lower as f64)
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:10.1}us std={:8.1}us p50={:10.1}us p95={:10.1}us min={:10.1}us max={:10.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Model config from config.json
// ---------------------------------------------------------------------------

struct ModelInfo {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    model_type: String,
}

fn parse_model_config(config_path: &Path) -> Result<ModelInfo, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let hidden_size = json["hidden_size"].as_u64().unwrap_or(3584) as usize;
    let num_heads = json["num_attention_heads"].as_u64().unwrap_or(28) as usize;
    let num_kv_heads = json["num_key_value_heads"]
        .as_u64()
        .unwrap_or(num_heads as u64) as usize;
    let num_layers = json["num_hidden_layers"].as_u64().unwrap_or(64) as usize;
    let vocab_size = json["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let intermediate_dim = json["intermediate_size"].as_u64().unwrap_or(2560) as usize;
    let head_dim = hidden_size / num_heads;
    let rope_theta = json["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let rms_norm_eps = json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
    let model_type = json["model_type"].as_str().unwrap_or("qwen2").to_string();

    Ok(ModelInfo {
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        vocab_size,
        intermediate_dim,
        rope_theta,
        rms_norm_eps,
        model_type,
    })
}

// ---------------------------------------------------------------------------
// Safetensors → TransformerModel loader
// ---------------------------------------------------------------------------

fn load_model_from_safetensors(
    device: &metal::Device,
    model_dir: &Path,
) -> Result<(TransformerModel, ModelInfo), Box<dyn std::error::Error>> {
    let load_start = Instant::now();

    // Parse config.json
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("config.json not found in {}", model_dir.display()).into());
    }
    let info = parse_model_config(&config_path)?;
    println!("Model: {} ({})", model_dir.display(), info.model_type);
    println!(
        "  Config: hidden={}, heads={}/{}, head_dim={}, layers={}, vocab={}, intermediate={}",
        info.hidden_size,
        info.num_heads,
        info.num_kv_heads,
        info.head_dim,
        info.num_layers,
        info.vocab_size,
        info.intermediate_dim
    );

    // Check for quantization config
    let quant_config = parse_quantization_config(&config_path)?;
    if let Some(ref qc) = quant_config {
        println!(
            "  Quantization: {}bit, group_size={}",
            qc.bits, qc.group_size
        );
    } else {
        println!("  Quantization: none (f16/bf16/f32)");
    }

    // Find safetensors files
    let mut st_files: Vec<PathBuf> = fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(format!("No .safetensors files in {}", model_dir.display()).into());
    }
    println!("  Safetensors files: {}", st_files.len());

    // Load weights
    let mut weights = if st_files.len() == 1 {
        load_safetensors_weights(device, &st_files[0], quant_config.as_ref())?
    } else {
        let paths: Vec<&Path> = st_files.iter().map(|p| p.as_path()).collect();
        load_safetensors_sharded(device, &paths, quant_config.as_ref())?
    };
    println!(
        "  Loaded {} tensors + {} quantized layers in {:.1}s",
        weights.len(),
        weights.num_quantized(),
        load_start.elapsed().as_secs_f64()
    );

    // Build TransformerModel from HF-named tensors
    let transformer_config = TransformerConfig {
        hidden_size: info.hidden_size,
        num_heads: info.num_heads,
        num_kv_heads: info.num_kv_heads,
        head_dim: info.head_dim,
        num_layers: info.num_layers,
        vocab_size: info.vocab_size,
        max_seq_len: 8192,
        rope_theta: info.rope_theta,
        rms_norm_eps: info.rms_norm_eps,
        ff_type: FeedForwardType::Gated {
            intermediate_dim: info.intermediate_dim,
        },
    };

    // Embedding
    let embed_weight = weights.take("model.embed_tokens.weight")?;
    let embedding = Embedding::from_array(
        EmbeddingConfig {
            vocab_size: info.vocab_size,
            embed_dim: info.hidden_size,
        },
        embed_weight,
    )?;

    // Transformer layers
    let kv_size = info.num_kv_heads * info.head_dim;
    let mut layers = Vec::with_capacity(info.num_layers);

    for i in 0..info.num_layers {
        let pfx = format!("model.layers.{i}");

        // Attention projections
        let q_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: info.hidden_size,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.self_attn.q_proj.weight"))?,
            None,
        )?;
        let k_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: kv_size,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.self_attn.k_proj.weight"))?,
            None,
        )?;
        let v_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: kv_size,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.self_attn.v_proj.weight"))?,
            None,
        )?;
        let o_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: info.hidden_size,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.self_attn.o_proj.weight"))?,
            None,
        )?;

        let attn_config = AttentionConfig {
            num_heads: info.num_heads,
            num_kv_heads: info.num_kv_heads,
            head_dim: info.head_dim,
            max_seq_len: 8192,
            rope_theta: info.rope_theta,
        };
        let attention = Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj)?;

        // FFN projections (SwiGLU)
        let gate_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: info.intermediate_dim,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.mlp.gate_proj.weight"))?,
            None,
        )?;
        let up_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.hidden_size,
                out_features: info.intermediate_dim,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.mlp.up_proj.weight"))?,
            None,
        )?;
        let down_proj = Linear::from_arrays(
            LinearConfig {
                in_features: info.intermediate_dim,
                out_features: info.hidden_size,
                has_bias: false,
            },
            weights.take(&format!("{pfx}.mlp.down_proj.weight"))?,
            None,
        )?;
        let ffn = FeedForward::Gated {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_merged_weight: None,
            gate_up_merged_weight_t: None,
        };

        // Norms
        let norm1 = weights.take(&format!("{pfx}.input_layernorm.weight"))?;
        let norm2 = weights.take(&format!("{pfx}.post_attention_layernorm.weight"))?;

        let block =
            TransformerBlock::from_parts(i, attention, ffn, norm1, norm2, info.rms_norm_eps);
        layers.push(block);

        if (i + 1) % 8 == 0 {
            println!("  Built layer {}/{}", i + 1, info.num_layers);
        }
    }

    // Final norm + LM head
    let final_norm = weights.take("model.norm.weight")?;
    let lm_head_weight = if weights.contains("lm_head.weight") {
        weights.take("lm_head.weight")?
    } else {
        return Err("lm_head.weight not found (tied embeddings not supported in bench)".into());
    };
    let lm_head = Linear::from_arrays(
        LinearConfig {
            in_features: info.hidden_size,
            out_features: info.vocab_size,
            has_bias: false,
        },
        lm_head_weight,
        None,
    )?;

    let mut model =
        TransformerModel::from_parts(transformer_config, embedding, layers, final_norm, lm_head)?;
    println!(
        "  Model ready: {} layers, total load time {:.1}s",
        info.num_layers,
        load_start.elapsed().as_secs_f64()
    );

    Ok((model, info))
}

// ---------------------------------------------------------------------------
// Benchmark entry point
// ---------------------------------------------------------------------------

fn main() {
    let model_dir = env::var("RMLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{home}/models/model")
    });
    let model_dir = PathBuf::from(&model_dir);

    // Optional MLX reference values (microseconds) for comparison
    let mlx_decode_us: Option<f64> = env::var("MLX_DECODE_US").ok().and_then(|s| s.parse().ok());
    let mlx_prefill_us: Option<f64> = env::var("MLX_PREFILL_US").ok().and_then(|s| s.parse().ok());

    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    println!(
        "Device: {} (unified_memory={})",
        gpu.name(),
        gpu.has_unified_memory()
    );

    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("kernel registration failed");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    // Load model from safetensors
    let (mut model, info) = load_model_from_safetensors(device, &model_dir)
        .expect("Failed to load model from safetensors");

    let num_layers = info.num_layers;
    let num_kv_heads = info.num_kv_heads;
    let head_dim = info.head_dim;

    // Dummy token IDs
    let decode_tokens: Vec<u32> = vec![1]; // seq_len=1 (decode)
    let prefill_tokens: Vec<u32> = (0..512).map(|i| (i % info.vocab_size) as u32).collect(); // seq_len=512

    // =====================================================================
    // DECODE BENCHMARK (seq_len=1)
    // =====================================================================
    println!(
        "\n========== DECODE BENCHMARK (seq_len=1, {} layers) ==========",
        num_layers
    );

    // --- Baseline: forward() ---
    println!("\n--- RMLX Baseline: forward() [per-layer submit+wait] ---");
    let mut cache_baseline: Vec<LayerKvCache> = (0..num_layers)
        .map(|_| LayerKvCache::preallocated(device, num_kv_heads, head_dim, 2048, DType::Float16))
        .collect();

    println!("Warming up ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        for c in cache_baseline.iter_mut() {
            c.seq_len = 0;
        }
        let _ = model
            .forward(
                &decode_tokens,
                None,
                None,
                None,
                Some(&mut cache_baseline),
                &registry,
                &queue,
            )
            .expect("baseline forward failed");
    }

    println!("Benchmarking ({} iterations)...", BENCH_ITERS);
    let mut baseline_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for c in cache_baseline.iter_mut() {
            c.seq_len = 0;
        }
        let start = Instant::now();
        let _ = model
            .forward(
                &decode_tokens,
                None,
                None,
                None,
                Some(&mut cache_baseline),
                &registry,
                &queue,
            )
            .expect("baseline forward failed");
        baseline_latencies.push(start.elapsed());
    }
    let baseline_stats = Stats::from_durations(&baseline_latencies);
    println!("  Baseline: {}", baseline_stats);

    // --- ExecGraph: forward_graph() ---
    println!("\n--- RMLX ExecGraph: forward_graph() [submit-only, 1 sync] ---");

    println!("Preparing weights for ExecGraph (pre-transposing)...");
    model
        .prepare_weights_for_graph(&registry, &queue)
        .expect("prepare_weights_for_graph failed");

    let mut cache_graph: Vec<LayerKvCache> = (0..num_layers)
        .map(|_| LayerKvCache::preallocated(device, num_kv_heads, head_dim, 2048, DType::Float16))
        .collect();

    println!("Warming up ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        for c in cache_graph.iter_mut() {
            c.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let _ = model
            .forward_graph(
                &decode_tokens,
                None,
                None,
                None,
                Some(&mut cache_graph),
                &registry,
                &queue,
                &event,
            )
            .expect("graph forward failed");
    }

    println!("Benchmarking ({} iterations)...", BENCH_ITERS);
    let mut graph_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for c in cache_graph.iter_mut() {
            c.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let start = Instant::now();
        let _ = model
            .forward_graph(
                &decode_tokens,
                None,
                None,
                None,
                Some(&mut cache_graph),
                &registry,
                &queue,
                &event,
            )
            .expect("graph forward failed");
        graph_latencies.push(start.elapsed());
    }
    let graph_stats = Stats::from_durations(&graph_latencies);
    println!("  ExecGraph: {}", graph_stats);

    // =====================================================================
    // PREFILL BENCHMARK (seq_len=512)
    // =====================================================================
    println!(
        "\n========== PREFILL BENCHMARK (seq_len=512, {} layers) ==========",
        num_layers
    );

    // --- Baseline: forward() ---
    println!("\n--- RMLX Baseline: forward() [per-layer submit+wait] ---");
    let mut cache_prefill_baseline: Vec<LayerKvCache> = (0..num_layers)
        .map(|_| LayerKvCache::preallocated(device, num_kv_heads, head_dim, 2048, DType::Float16))
        .collect();

    println!("Warming up ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        for c in cache_prefill_baseline.iter_mut() {
            c.seq_len = 0;
        }
        let _ = model
            .forward(
                &prefill_tokens,
                None,
                None,
                None,
                Some(&mut cache_prefill_baseline),
                &registry,
                &queue,
            )
            .expect("prefill baseline forward failed");
    }

    println!("Benchmarking ({} iterations)...", BENCH_ITERS);
    let mut prefill_baseline_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for c in cache_prefill_baseline.iter_mut() {
            c.seq_len = 0;
        }
        let start = Instant::now();
        let _ = model
            .forward(
                &prefill_tokens,
                None,
                None,
                None,
                Some(&mut cache_prefill_baseline),
                &registry,
                &queue,
            )
            .expect("prefill baseline forward failed");
        prefill_baseline_latencies.push(start.elapsed());
    }
    let prefill_baseline_stats = Stats::from_durations(&prefill_baseline_latencies);
    println!("  Prefill Baseline: {}", prefill_baseline_stats);

    // --- ExecGraph: forward_prefill_graph() ---
    println!("\n--- RMLX ExecGraph: forward_prefill_graph() [submit-only, 1 sync] ---");
    let mut cache_prefill_graph: Vec<LayerKvCache> = (0..num_layers)
        .map(|_| LayerKvCache::preallocated(device, num_kv_heads, head_dim, 2048, DType::Float16))
        .collect();

    println!("Warming up ({} iterations)...", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        for c in cache_prefill_graph.iter_mut() {
            c.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let _ = model
            .forward_prefill_graph(
                &prefill_tokens,
                None,
                None,
                None,
                &mut cache_prefill_graph,
                &registry,
                &queue,
                &event,
            )
            .expect("prefill graph forward failed");
    }

    println!("Benchmarking ({} iterations)...", BENCH_ITERS);
    let mut prefill_graph_latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        for c in cache_prefill_graph.iter_mut() {
            c.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let start = Instant::now();
        let _ = model
            .forward_prefill_graph(
                &prefill_tokens,
                None,
                None,
                None,
                &mut cache_prefill_graph,
                &registry,
                &queue,
                &event,
            )
            .expect("prefill graph forward failed");
        prefill_graph_latencies.push(start.elapsed());
    }
    let prefill_graph_stats = Stats::from_durations(&prefill_graph_latencies);
    println!("  Prefill ExecGraph: {}", prefill_graph_stats);

    // =====================================================================
    // COMPARISON SUMMARY
    // =====================================================================
    println!("\n========== RMLX vs MLX COMPARISON ==========");
    println!("Model: {} ({})", model_dir.display(), info.model_type);
    println!(
        "Config: hidden={}, heads={}/{}, head_dim={}, layers={}, intermediate={}",
        info.hidden_size,
        info.num_heads,
        info.num_kv_heads,
        info.head_dim,
        info.num_layers,
        info.intermediate_dim,
    );
    println!();

    // Decode comparison
    println!("DECODE (seq_len=1):");
    println!(
        "  RMLX baseline:   {:10.1} us (p50={:.1}us)",
        baseline_stats.mean, baseline_stats.p50
    );
    println!(
        "  RMLX ExecGraph:  {:10.1} us (p50={:.1}us)",
        graph_stats.mean, graph_stats.p50
    );
    let decode_speedup = if graph_stats.mean > 0.0 {
        baseline_stats.mean / graph_stats.mean
    } else {
        0.0
    };
    println!("  RMLX speedup (baseline->graph): {:.2}x", decode_speedup);

    if let Some(mlx_us) = mlx_decode_us {
        println!("  MLX reference:   {:10.1} us", mlx_us);
        let vs_mlx_baseline = if mlx_us > 0.0 {
            baseline_stats.mean / mlx_us
        } else {
            0.0
        };
        let vs_mlx_graph = if mlx_us > 0.0 {
            graph_stats.mean / mlx_us
        } else {
            0.0
        };
        println!(
            "  RMLX/MLX ratio:  baseline={:.2}x  graph={:.2}x",
            vs_mlx_baseline, vs_mlx_graph
        );
        if vs_mlx_graph < 1.0 {
            println!(
                "  --> RMLX ExecGraph is {:.1}% FASTER than MLX",
                (1.0 - vs_mlx_graph) * 100.0
            );
        } else {
            println!(
                "  --> RMLX ExecGraph is {:.1}% slower than MLX",
                (vs_mlx_graph - 1.0) * 100.0
            );
        }
    } else {
        println!("  MLX reference:   (set MLX_DECODE_US env var)");
    }
    println!();

    // Prefill comparison
    println!("PREFILL (seq_len=512):");
    println!(
        "  RMLX baseline:   {:10.1} us (p50={:.1}us)",
        prefill_baseline_stats.mean, prefill_baseline_stats.p50
    );
    println!(
        "  RMLX ExecGraph:  {:10.1} us (p50={:.1}us)",
        prefill_graph_stats.mean, prefill_graph_stats.p50
    );
    let prefill_speedup = if prefill_graph_stats.mean > 0.0 {
        prefill_baseline_stats.mean / prefill_graph_stats.mean
    } else {
        0.0
    };
    println!("  RMLX speedup (baseline->graph): {:.2}x", prefill_speedup);

    if let Some(mlx_us) = mlx_prefill_us {
        println!("  MLX reference:   {:10.1} us", mlx_us);
        let vs_mlx_baseline = if mlx_us > 0.0 {
            prefill_baseline_stats.mean / mlx_us
        } else {
            0.0
        };
        let vs_mlx_graph = if mlx_us > 0.0 {
            prefill_graph_stats.mean / mlx_us
        } else {
            0.0
        };
        println!(
            "  RMLX/MLX ratio:  baseline={:.2}x  graph={:.2}x",
            vs_mlx_baseline, vs_mlx_graph
        );
        if vs_mlx_graph < 1.0 {
            println!(
                "  --> RMLX ExecGraph is {:.1}% FASTER than MLX",
                (1.0 - vs_mlx_graph) * 100.0
            );
        } else {
            println!(
                "  --> RMLX ExecGraph is {:.1}% slower than MLX",
                (vs_mlx_graph - 1.0) * 100.0
            );
        }
    } else {
        println!("  MLX reference:   (set MLX_PREFILL_US env var)");
    }

    println!();
    println!("To get MLX reference numbers, run on the same machine:");
    println!(
        "  python benchmarks/mlx_model_bench.py --model-dir {}",
        model_dir.display()
    );
    println!("Then re-run with:");
    println!(
        "  MLX_DECODE_US=<value> MLX_PREFILL_US=<value> RMLX_MODEL_DIR={} \\",
        model_dir.display()
    );
    println!("    cargo bench -p rmlx-nn --bench model_bench");
    println!("=================================");
}
