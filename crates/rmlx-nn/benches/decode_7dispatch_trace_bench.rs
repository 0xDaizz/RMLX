//! Decode 7-dispatch trace bench — runs ONLY the fused 7-dispatch path
//! for 200 iterations across 60 layers, designed for xctrace Metal System Trace capture.
//!
//! Usage:
//!   cargo bench --bench decode_7dispatch_trace_bench --no-run
//!   # then run the binary directly while xctrace attaches

use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice as _;

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;
use rmlx_metal::exec_graph::ExecGraph;
use rmlx_nn::{
    Attention, AttentionConfig, CachedDecode, FeedForward, LayerKvCache, Linear, LinearConfig,
    TransformerBlock,
};

const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 14336;
const SEQ_LEN: usize = 1;
const RMS_NORM_EPS: f32 = 1e-5;

const NUM_LAYERS: usize = 60;
const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 200;

// ---------------------------------------------------------------------------
// Stats helper (from pipeline_bench.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    p99: f64,
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
        let p99 = percentile(&micros, 99.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            p99,
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
            "mean={:8.1}us std={:7.1}us p50={:8.1}us p95={:8.1}us p99={:8.1}us min={:8.1}us max={:8.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.p99, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers (copied from pipeline_bench.rs — exact same API)
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

fn rand_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
    let numel: usize = shape.iter().product();
    let mut f16_bytes = Vec::with_capacity(numel * 2);
    let mut state = seed;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        let h = f32_to_f16_bits(val as f32);
        f16_bytes.extend_from_slice(&h.to_le_bytes());
    }
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

fn make_linear(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    in_f: usize,
    out_f: usize,
    seed: u64,
) -> Linear {
    let weight = rand_array(device, &[out_f, in_f], seed);
    Linear::from_arrays(
        LinearConfig {
            in_features: in_f,
            out_features: out_f,
            has_bias: false,
        },
        weight,
        None,
    )
    .expect("linear from_arrays")
}

fn build_transformer_block(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, 2);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, 3);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 4);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 5);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 6);
    let down_proj = make_linear(device, INTERMEDIATE_DIM, HIDDEN_SIZE, 7);
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
    };

    let norm1_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE];
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };
    let norm2_weight = {
        let ones_f16: Vec<u16> = vec![0x3C00u16; HIDDEN_SIZE];
        let bytes: Vec<u8> = ones_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
        Array::from_bytes(device, &bytes, vec![HIDDEN_SIZE], DType::Float16)
    };

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Main — 7-dispatch only, 60 layers, 200 iterations (xctrace attach window)
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    eprintln!(
        "Device: {} (unified_memory={})",
        gpu.name(),
        gpu.has_unified_memory()
    );

    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("kernel registration failed");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    eprintln!(
        "Config: hidden={}, heads={}/{}, head_dim={}, intermediate={}, seq_len={}, layers={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM, SEQ_LEN, NUM_LAYERS
    );
    eprintln!("dtype: float16");

    // Build layers
    eprintln!("Building {} transformer layers...", NUM_LAYERS);
    let mut blocks: Vec<TransformerBlock> = Vec::with_capacity(NUM_LAYERS);
    for _ in 0..NUM_LAYERS {
        let mut blk = build_transformer_block(device);
        blk.prepare_weights_9dispatch(device)
            .expect("prepare_weights_9dispatch");
        blk.prepare_weights_private(device, &queue);
        blocks.push(blk);
    }

    // KV caches
    let mut caches: Vec<LayerKvCache> = (0..NUM_LAYERS)
        .map(|_| LayerKvCache::preallocated(device, NUM_KV_HEADS, HEAD_DIM, 2048, DType::Float16))
        .collect();

    // Pre-resolve CachedDecode for each layer
    let cached: Vec<CachedDecode> = blocks
        .iter()
        .map(|blk| {
            blk.prepare_cached_decode(&registry)
                .expect("prepare_cached_decode")
        })
        .collect();

    // ---- Warmup ----
    eprintln!(
        "Warming up 7-dispatch x{} ({} iterations)...",
        NUM_LAYERS, WARMUP_ITERS
    );
    for _ in 0..WARMUP_ITERS {
        for cache in caches.iter_mut() {
            cache.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);
        let mut x = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 200);
        for ((layer, cache), cd) in blocks.iter().zip(caches.iter_mut()).zip(cached.iter()) {
            let cb = graph.command_buffer();
            x = layer
                .forward_cached_fused_7dispatch(&x, None, None, cache, cd, cb)
                .expect("warmup failed");
            let _t = graph.submit_batch();
        }
        graph.sync().expect("warmup sync failed");
    }

    // ---- Bench ----
    eprintln!(
        "Benchmarking 7-dispatch x{} ({} iterations)...",
        NUM_LAYERS, BENCH_ITERS
    );
    let mut latencies = Vec::with_capacity(BENCH_ITERS);

    for i in 0..BENCH_ITERS {
        for cache in caches.iter_mut() {
            cache.seq_len = 0;
        }
        let event = GpuEvent::new(device);
        let mut graph = ExecGraph::new(&queue, &event, 32);
        let start = Instant::now();
        let mut x = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 200);
        for ((layer, cache), cd) in blocks.iter().zip(caches.iter_mut()).zip(cached.iter()) {
            let cb = graph.command_buffer();
            x = layer
                .forward_cached_fused_7dispatch(&x, None, None, cache, cd, cb)
                .expect("bench failed");
            let _t = graph.submit_batch();
        }
        let _ = graph.sync().expect("sync failed");
        let elapsed = start.elapsed();
        latencies.push(elapsed);

        if (i + 1) % 20 == 0 {
            eprintln!(
                "  iter {}/{}: {:.1} us total ({:.1} us/layer)",
                i + 1,
                BENCH_ITERS,
                elapsed.as_secs_f64() * 1e6,
                elapsed.as_secs_f64() * 1e6 / NUM_LAYERS as f64
            );
        }
    }

    // Stats
    let stats = Stats::from_durations(&latencies);
    let per_layer_mean = stats.mean / NUM_LAYERS as f64;
    let per_layer_p50 = stats.p50 / NUM_LAYERS as f64;

    eprintln!(
        "\n========== 7-Dispatch Decode Results ({} layers x {} iters) ==========",
        NUM_LAYERS, BENCH_ITERS
    );
    eprintln!("  Total:     {}", stats);
    eprintln!(
        "  Per-layer: mean={:.1}us p50={:.1}us p95={:.1}us min={:.1}us max={:.1}us",
        per_layer_mean,
        per_layer_p50,
        stats.p95 / NUM_LAYERS as f64,
        stats.min / NUM_LAYERS as f64,
        stats.max / NUM_LAYERS as f64,
    );
    eprintln!("=====================================================================");
}
