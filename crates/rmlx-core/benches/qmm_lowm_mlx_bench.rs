//! QMM Low-M MLX Parity Benchmark
//!
//! Fair comparison between rmlx and MLX at MoE-critical low-M dimensions.
//! All tests use **f16 input** and **group_size=64** to match MLX conditions exactly.
//!
//! Two timing modes per (M, K, N):
//!   - "dispatch": affine_quantized_matmul_batched (includes CB creation overhead)
//!   - "kernel":   affine_quantized_matmul_batched_into_cb (shared CB, kernel-only)
//!
//! Dispatch paths exercised:
//!   - M=1:     QMV fast f16 (single-row, simdgroup reduction)
//!   - M=2-16:  QMV batched f16 (row-parallel, simdgroup reduction)
//!   - M>=32:   NAX MMA (4SG, BK=64) — all K values here are K%64==0
//!
//! MLX reference numbers (M3 Ultra 80-core, group_size=64, f16, mlx-lm 0.24):
//!   K=2560 N=2560: M=1:196.7us/0.067T  M=4:213.8us/0.245T  M=8:226.3us/0.463T  M=16:236.7us/0.886T  M=32:264.0us/1.589T  M=64:299.2us/2.804T
//!   K=3584 N=2560: M=1:191.1us/0.096T  M=4:206.9us/0.355T  M=8:218.7us/0.671T  M=16:240.9us/1.219T  M=32:280.5us/2.094T  M=64:291.2us/4.033T
//!   K=3584 N=3584: M=1:194.9us/0.132T  M=4:209.2us/0.491T  M=8:227.8us/0.902T  M=16:262.0us/1.569T  M=32:276.9us/2.969T  M=64:295.7us/5.560T
//!   K=4096 N=4096: M=1:209.4us/0.160T  M=4:224.4us/0.598T  M=8:247.1us/1.086T  M=16:289.5us/1.855T  M=32:298.8us/3.594T  M=64:327.9us/6.550T
//!   K=3584 N=4096: M=1:186.8us/0.157T  M=4:209.3us/0.561T  M=8:233.8us/1.005T  M=16:268.8us/1.748T  M=32:271.5us/3.460T  M=64:294.5us/6.380T
//!   K=4096 N=2560: M=1:200.2us/0.105T  M=4:218.8us/0.383T  M=8:231.7us/0.724T  M=16:247.6us/1.355T  M=32:293.3us/2.288T  M=64:302.3us/4.439T
//!
//! Usage: cargo bench -p rmlx-core --bench qmm_lowm_mlx_bench

use std::time::{Duration, Instant};

use half::f16;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const GROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// MLX reference data: (K, N, M) -> (latency_us, tflops)
// ---------------------------------------------------------------------------

struct MlxRef {
    k: usize,
    n: usize,
    m: usize,
    us: f64,
    tflops: f64,
}

fn mlx_reference() -> Vec<MlxRef> {
    vec![
        // K=2560 N=2560
        MlxRef { k: 2560, n: 2560, m: 1,  us: 196.7, tflops: 0.067 },
        MlxRef { k: 2560, n: 2560, m: 4,  us: 213.8, tflops: 0.245 },
        MlxRef { k: 2560, n: 2560, m: 8,  us: 226.3, tflops: 0.463 },
        MlxRef { k: 2560, n: 2560, m: 16, us: 236.7, tflops: 0.886 },
        MlxRef { k: 2560, n: 2560, m: 32, us: 264.0, tflops: 1.589 },
        MlxRef { k: 2560, n: 2560, m: 64, us: 299.2, tflops: 2.804 },
        // K=3584 N=2560
        MlxRef { k: 3584, n: 2560, m: 1,  us: 191.1, tflops: 0.096 },
        MlxRef { k: 3584, n: 2560, m: 4,  us: 206.9, tflops: 0.355 },
        MlxRef { k: 3584, n: 2560, m: 8,  us: 218.7, tflops: 0.671 },
        MlxRef { k: 3584, n: 2560, m: 16, us: 240.9, tflops: 1.219 },
        MlxRef { k: 3584, n: 2560, m: 32, us: 280.5, tflops: 2.094 },
        MlxRef { k: 3584, n: 2560, m: 64, us: 291.2, tflops: 4.033 },
        // K=3584 N=3584
        MlxRef { k: 3584, n: 3584, m: 1,  us: 194.9, tflops: 0.132 },
        MlxRef { k: 3584, n: 3584, m: 4,  us: 209.2, tflops: 0.491 },
        MlxRef { k: 3584, n: 3584, m: 8,  us: 227.8, tflops: 0.902 },
        MlxRef { k: 3584, n: 3584, m: 16, us: 262.0, tflops: 1.569 },
        MlxRef { k: 3584, n: 3584, m: 32, us: 276.9, tflops: 2.969 },
        MlxRef { k: 3584, n: 3584, m: 64, us: 295.7, tflops: 5.560 },
        // K=4096 N=4096
        MlxRef { k: 4096, n: 4096, m: 1,  us: 209.4, tflops: 0.160 },
        MlxRef { k: 4096, n: 4096, m: 4,  us: 224.4, tflops: 0.598 },
        MlxRef { k: 4096, n: 4096, m: 8,  us: 247.1, tflops: 1.086 },
        MlxRef { k: 4096, n: 4096, m: 16, us: 289.5, tflops: 1.855 },
        MlxRef { k: 4096, n: 4096, m: 32, us: 298.8, tflops: 3.594 },
        MlxRef { k: 4096, n: 4096, m: 64, us: 327.9, tflops: 6.550 },
        // K=3584 N=4096
        MlxRef { k: 3584, n: 4096, m: 1,  us: 186.8, tflops: 0.157 },
        MlxRef { k: 3584, n: 4096, m: 4,  us: 209.3, tflops: 0.561 },
        MlxRef { k: 3584, n: 4096, m: 8,  us: 233.8, tflops: 1.005 },
        MlxRef { k: 3584, n: 4096, m: 16, us: 268.8, tflops: 1.748 },
        MlxRef { k: 3584, n: 4096, m: 32, us: 271.5, tflops: 3.460 },
        MlxRef { k: 3584, n: 4096, m: 64, us: 294.5, tflops: 6.380 },
        // K=4096 N=2560
        MlxRef { k: 4096, n: 2560, m: 1,  us: 200.2, tflops: 0.105 },
        MlxRef { k: 4096, n: 2560, m: 4,  us: 218.8, tflops: 0.383 },
        MlxRef { k: 4096, n: 2560, m: 8,  us: 231.7, tflops: 0.724 },
        MlxRef { k: 4096, n: 2560, m: 16, us: 247.6, tflops: 1.355 },
        MlxRef { k: 4096, n: 2560, m: 32, us: 293.3, tflops: 2.288 },
        MlxRef { k: 4096, n: 2560, m: 64, us: 302.3, tflops: 4.439 },
    ]
}

fn find_mlx_ref(refs: &[MlxRef], k: usize, n: usize, m: usize) -> Option<&MlxRef> {
    refs.iter().find(|r| r.k == k && r.n == n && r.m == m)
}

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
// Random data generation
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn rand_f16_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
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
    device: &metal::Device,
    out_features: usize,
    in_features: usize,
    seed: u64,
) -> QuantizedWeight {
    let bits: u32 = 4;
    let mut state = seed;
    let opts = metal::MTLResourceOptions::StorageModeShared;

    let elems_per_u32 = 32 / bits as usize;
    let total_elements = out_features * in_features;
    let num_u32s = total_elements.div_ceil(elems_per_u32);
    let w_data: Vec<u32> = (0..num_u32s)
        .map(|_| {
            let v = lcg_next(&mut state);
            v as u32
        })
        .collect();
    let weights_buf =
        device.new_buffer_with_data(w_data.as_ptr() as *const _, (num_u32s * 4) as u64, opts);

    let num_groups = total_elements / GROUP_SIZE as usize;
    let scales_data: Vec<u16> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            let val = ((v >> 33) as f64 / (1u64 << 31) as f64) as f32 * 0.02 + 0.001;
            f16::from_f32(val).to_bits()
        })
        .collect();
    let biases_data: Vec<u16> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            let val = ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.01;
            f16::from_f32(val).to_bits()
        })
        .collect();

    let scales_buf = device.new_buffer_with_data(
        scales_data.as_ptr() as *const _,
        (num_groups * 2) as u64,
        opts,
    );
    let biases_buf = device.new_buffer_with_data(
        biases_data.as_ptr() as *const _,
        (num_groups * 2) as u64,
        opts,
    );

    QuantizedWeight::new(
        weights_buf,
        scales_buf,
        biases_buf,
        GROUP_SIZE,
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
// Benchmark runners
// ---------------------------------------------------------------------------

/// Dispatch path: affine_quantized_matmul_batched (includes CB creation overhead).
/// f16 input, Q4, group_size=64.
fn bench_dispatch(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("dispatch warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("dispatch bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// Kernel-only path: affine_quantized_matmul_batched_into_cb (shared CB, no CB overhead).
/// f16 input, Q4, group_size=64.
fn bench_kernel(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(registry, &x, &qw, cb)
            .expect("kernel warmup failed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer();
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched_into_cb(registry, &x, &qw, cb)
            .expect("kernel bench failed");
        cb.commit();
        cb.wait_until_completed();
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
    let queue = device.new_command_queue();

    let mlx_refs = mlx_reference();

    let k_values: &[usize] = &[2560, 3584, 4096];
    let n_values: &[usize] = &[2560, 3584, 4096];
    let m_values: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

    println!("=== QMM Low-M MLX Parity Benchmark ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, Q4, group_size={}, f16 input",
        WARMUP_ITERS, BENCH_ITERS, GROUP_SIZE,
    );
    println!("Dispatch paths: M=1 QMV fast, M=2-16 QMV batched, M=17-64 Skinny Split-K, M>=128 NAX");
    println!("MLX reference: M3 Ultra 80-core, mlx-lm 0.24, group_size=64, f16");
    println!();

    // Header
    println!(
        "{:>3}  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>14}",
        "M", "K", "N",
        "disp_us", "kern_us",
        "disp_T", "kern_T",
        "mlx_us", "mlx_T",
        "parity%", "path",
    );
    println!("{}", "-".repeat(120));

    // Collect results for per-M-range summary: (m, parity%)
    let mut results: Vec<(usize, f64)> = Vec::new();

    for &k in k_values {
        for &n in n_values {
            for &m in m_values {
                // Dispatch path
                let disp = bench_dispatch(&registry, &queue, device, m, n, k);
                let disp_tf = tflops(m, n, k, disp.p50);

                // Kernel-only path
                let kern = bench_kernel(&registry, &queue, device, m, n, k);
                let kern_tf = tflops(m, n, k, kern.p50);

                // Dispatch path label
                let path = if m == 1 {
                    "QMV-fast"
                } else if m <= 16 {
                    "QMV-batch"
                } else {
                    "Skinny-SK"
                };

                // MLX reference
                let (mlx_us, mlx_tf, parity_str) = match find_mlx_ref(&mlx_refs, k, n, m) {
                    Some(r) => {
                        // Parity = rmlx kernel TFLOPS / MLX TFLOPS * 100
                        let parity = kern_tf / r.tflops * 100.0;
                        results.push((m, parity));
                        (
                            format!("{:10.1}", r.us),
                            format!("{:10.3}", r.tflops),
                            format!("{:7.1}%", parity),
                        )
                    }
                    None => (
                        format!("{:>10}", "-"),
                        format!("{:>10}", "-"),
                        format!("{:>8}", "-"),
                    ),
                };

                println!(
                    "{:>3}  {:>4}  {:>4}  {:>10.1}  {:>10.1}  {:>10.3}  {:>10.3}  {}  {}  {}  {:>14}",
                    m, k, n,
                    disp.p50, kern.p50,
                    disp_tf, kern_tf,
                    mlx_us, mlx_tf,
                    parity_str, path,
                );
            }
            println!();
        }
    }

    // Summary
    println!("{}", "=".repeat(120));
    if !results.is_empty() {
        let avg_parity: f64 = results.iter().map(|(_, p)| p).sum::<f64>() / results.len() as f64;
        println!(
            "Average parity across {} data points: {:.1}% of MLX throughput",
            results.len(), avg_parity,
        );
    }

    // Per-M-range summary
    println!();
    println!("Per-M-range parity summary:");
    for &(label, m_lo, m_hi) in &[
        ("QMV (M=1)",         1usize, 1usize),
        ("QMV (M=2-16)",      2, 16),
        ("Skinny (M=32-64)", 32, 64),
    ] {
        let filtered: Vec<f64> = results
            .iter()
            .filter(|(m, _)| *m >= m_lo && *m <= m_hi)
            .map(|(_, p)| *p)
            .collect();
        if !filtered.is_empty() {
            let avg = filtered.iter().sum::<f64>() / filtered.len() as f64;
            println!("  {:<20} avg parity: {:6.1}% ({} points)", label, avg, filtered.len());
        }
    }

    println!();
    println!("Done.");
}
