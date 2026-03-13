//! ⚠️ NON-PRODUCTION PATH — pipelined QMM encoding (96 QMMs in 1 CB) measures kernel
//! throughput with amortized CB overhead, but bypasses TransformerModel dispatch logic.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Q4 Pipeline Benchmark — per-op vs pipeline (96 QMMs, 1 sync) comparison.
//!
//! Measures the dispatch overhead amortization benefit of encoding 32 layers × 3 QMMs
//! (gate, up, down) into a single command buffer with one sync, versus the standard
//! per-op dispatch path.
//!
//! Models tested (real MoE architectures, ordered smallest→largest for thermal):
//!   - Qwen3.5-MoE:    K=2048, N=512   (tiny expert, 256E top-8)
//!   - DeepSeek-V3:     K=7168, N=2048  (mainstream MoE, 256E top-8)
//!   - MiniMax-Text-01: K=6144, N=9216  (large expert, 32E top-2)
//!   - Mixtral-8x22B:   K=6144, N=16384 (largest expert, 8E top-2)
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench qmm_pipeline_bench

use std::time::{Duration, Instant};

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::MTLResourceOptions;
use std::ptr::NonNull;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

const M_VALUES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const N_LAYERS: usize = 32;

/// (name, K_hidden, N_intermediate)
const MODEL_DIMS: &[(&str, usize, usize)] = &[
    ("Qwen3.5-MoE", 2048, 512), // tiny expert, 256E top-8 — smallest, run first
    ("DeepSeek-V3", 7168, 2048), // mainstream MoE, 256E top-8
    ("MiniMax-Text-01", 6144, 9216), // large expert, 32E top-2
    ("Mixtral-8x22B", 6144, 16384), // largest expert, 8E top-2 — run last
];

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    #[allow(dead_code)]
    mean: f64,
    #[allow(dead_code)]
    std_dev: f64,
    p50: f64,
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
        Stats { mean, std_dev, p50 }
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

// ---------------------------------------------------------------------------
// Random data generation (LCG — deterministic, no deps)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn rand_f16_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
    let numel: usize = shape.iter().product();
    let mut state = seed;
    let mut f16_bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        let v = lcg_next(&mut state);
        let val = ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.1;
        let h = f16::from_f32(val);
        f16_bytes.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

fn make_quantized_weight(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: u32,
    seed: u64,
) -> QuantizedWeight {
    let mut state = seed;
    let opts = MTLResourceOptions::StorageModeShared;

    let elems_per_u32 = 32 / bits as usize;
    let total_elements = out_features * in_features;
    let num_u32s = total_elements.div_ceil(elems_per_u32);
    let w_data: Vec<u32> = (0..num_u32s)
        .map(|_| {
            let v = lcg_next(&mut state);
            v as u32
        })
        .collect();
    let weights_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(w_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_u32s * 4) as u64 as usize,
                opts,
            )
            .unwrap()
    };

    let num_groups = total_elements / group_size as usize;
    let scales_f32: Vec<f32> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64) as f32 * 0.02 + 0.001
        })
        .collect();
    let biases_f32: Vec<f32> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.01
        })
        .collect();

    // Convert to f16 (native QuantizedWeight format)
    let scales_data: Vec<u16> = scales_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();
    let biases_data: Vec<u16> = biases_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();

    let scales_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(scales_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_groups * 2) as u64 as usize,
                opts,
            )
            .unwrap()
    };
    let biases_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(biases_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_groups * 2) as u64 as usize,
                opts,
            )
            .unwrap()
    };

    QuantizedWeight::new(
        weights_buf,
        scales_buf,
        biases_buf,
        group_size,
        bits,
        out_features,
        in_features,
    )
    .expect("Failed to create QuantizedWeight")
}

// ---------------------------------------------------------------------------
// TFLOPS calculation
// ---------------------------------------------------------------------------

fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Mode A: Per-op benchmark
// ---------------------------------------------------------------------------

fn bench_per_op(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    k: usize,
    n: usize,
) -> Stats {
    let x = rand_f16_array(device, &[m, k], 42);
    let qw = make_quantized_weight(device, n, k, 4, 64, 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Per-op warmup failed");
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Per-op bench failed");
        times.push(start.elapsed());
    }
    Stats::from_durations(&times)
}

// ---------------------------------------------------------------------------
// Mode B: Pipeline benchmark (96 QMMs, 1 sync)
// ---------------------------------------------------------------------------

fn bench_pipeline(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    k_hidden: usize,
    n_inter: usize,
) -> Stats {
    // Pre-allocate inputs and weights for gate/up/down per layer
    let x = rand_f16_array(device, &[m, k_hidden], 42);
    let qw_gate = make_quantized_weight(device, n_inter, k_hidden, 4, 64, 100);
    let qw_up = make_quantized_weight(device, n_inter, k_hidden, 4, 64, 200);
    let qw_down = make_quantized_weight(device, k_hidden, n_inter, 4, 64, 300);

    // We also need an x_inter for the down projection
    let x_inter = rand_f16_array(device, &[m, n_inter], 43);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        for _ in 0..N_LAYERS {
            let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(
                registry, &x, &qw_gate, &cb,
            )
            .expect("Pipeline warmup gate failed");
            let _ =
                ops::quantized::affine_quantized_matmul_batched_into_cb(registry, &x, &qw_up, &cb)
                    .expect("Pipeline warmup up failed");
            let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(
                registry, &x_inter, &qw_down, &cb,
            )
            .expect("Pipeline warmup down failed");
        }
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let start = Instant::now();
        for _ in 0..N_LAYERS {
            let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(
                registry, &x, &qw_gate, &cb,
            )
            .expect("Pipeline bench gate failed");
            let _ =
                ops::quantized::affine_quantized_matmul_batched_into_cb(registry, &x, &qw_up, &cb)
                    .expect("Pipeline bench up failed");
            let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(
                registry, &x_inter, &qw_down, &cb,
            )
            .expect("Pipeline bench down failed");
        }
        cb.commit();
        cb.waitUntilCompleted();
        times.push(start.elapsed());
    }
    Stats::from_durations(&times)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    let total_ops = N_LAYERS * 3; // 32 layers × 3 QMMs = 96

    println!("=========================================================");
    println!("  Q4 Pipeline Benchmark (group_size=64)");
    println!("=========================================================");
    println!(
        "  Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS,
    );
    println!(
        "  Pipeline: {} layers x 3 QMMs = {} ops per CB",
        N_LAYERS, total_ops
    );
    println!();

    for &(model_name, k_hidden, n_inter) in MODEL_DIMS {
        println!("--- {} (K={}, N={}) ---", model_name, k_hidden, n_inter);
        println!();

        // --- Per-op: gate/up (K_hidden → N_inter) ---
        println!("  Per-op (gate/up K={}->N={}):", k_hidden, n_inter);
        for &m in M_VALUES {
            let stats = bench_per_op(&registry, &queue, device, m, k_hidden, n_inter);
            let tf = tflops(m, n_inter, k_hidden, stats.p50);
            println!("  M={:<5} p50={:>8.1}us   {:.4}T", m, stats.p50, tf,);
        }
        println!();

        // --- Per-op: down (N_inter → K_hidden) ---
        println!("  Per-op (down K={}->N={}):", n_inter, k_hidden);
        for &m in M_VALUES {
            let stats = bench_per_op(&registry, &queue, device, m, n_inter, k_hidden);
            let tf = tflops(m, k_hidden, n_inter, stats.p50);
            println!("  M={:<5} p50={:>8.1}us   {:.4}T", m, stats.p50, tf,);
        }
        println!();

        // --- Pipeline (96 QMMs, 1 sync) ---
        println!(
            "  Pipeline ({}L x 3 QMMs = {} ops, gate/up K={}, down K={}):",
            N_LAYERS, total_ops, k_hidden, n_inter,
        );
        for &m in M_VALUES {
            let stats = bench_pipeline(&registry, &queue, device, m, k_hidden, n_inter);

            let total_us = stats.p50;
            let per_layer_us = total_us / N_LAYERS as f64;
            let per_op_us = total_us / total_ops as f64;

            // Amortized TFLOPS: use average FLOP count across gate/up/down
            // gate: 2*M*K_hidden*N_inter, up: same, down: 2*M*N_inter*K_hidden
            // All three have the same FLOP count: 2*M*K_hidden*N_inter
            let flops_per_op = 2.0 * m as f64 * k_hidden as f64 * n_inter as f64;
            let amortized_tflops = flops_per_op / (per_op_us * 1e-6) / 1e12;

            println!(
                "  M={:<5} total={:>8.0}us  per_layer={:>7.1}us  per_op={:>7.1}us  amortized={:.4}T",
                m, total_us, per_layer_us, per_op_us, amortized_tflops,
            );
        }

        println!();
    }

    println!("Done.");
}
