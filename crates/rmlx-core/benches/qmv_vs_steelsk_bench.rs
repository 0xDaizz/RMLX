//! QMV-batch vs Steel Split-K diagnostic benchmark
//!
//! Compares the two dispatch paths at MoE-critical low-M dimensions:
//!   - QMV-batch: `affine_quantized_matmul_batched` (auto-dispatches to QMV-batch for M<=16)
//!   - Steel Split-K: `affine_qmm_steel_splitk_q4` (forces Steel Split-K regardless of M)
//!
//! f16 input, Q4, group_size=64
//! 5 warmup, 20 bench iters, p50 latency
//!
//! Usage: cargo bench -p rmlx-core --bench qmv_vs_steelsk_bench

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
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    p50: f64,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&micros, 50.0);
        Stats { p50 }
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

/// QMV-batch path: affine_quantized_matmul_batched (auto-dispatches to QMV-batch for M<=16).
fn bench_qmv_batch(
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
            .expect("qmv-batch warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("qmv-batch bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// Steel Split-K path: affine_qmm_steel_splitk_q4 (forces Steel Split-K regardless of M).
fn bench_steel_splitk(
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
        let _ = ops::quantized::affine_qmm_steel_splitk_q4(registry, &x, &qw, queue)
            .expect("steel-sk warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_steel_splitk_q4(registry, &x, &qw, queue)
            .expect("steel-sk bench failed");
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

    // --- GPU warmup: trigger Metal shader JIT compilation before timing ---
    println!("GPU warmup (triggering Metal shader compilation)...");
    {
        let warmup_qw = make_quantized_weight(device, 2560, 2560, 42);
        // Warmup both paths: QMV-batch (M=4) and Steel Split-K (M=4)
        let wx = rand_f16_array(device, &[4, 2560], 99);
        for _ in 0..5 {
            let _ =
                ops::quantized::affine_quantized_matmul_batched(&registry, &wx, &warmup_qw, &queue)
                    .expect("GPU warmup (qmv-batch) failed");
            let _ = ops::quantized::affine_qmm_steel_splitk_q4(&registry, &wx, &warmup_qw, &queue)
                .expect("GPU warmup (steel-sk) failed");
        }
    }
    println!("GPU warmup done.\n");

    let m_values: &[usize] = &[2, 4, 8, 16];
    let k_values: &[usize] = &[2560, 3584, 4096];
    let n_values: &[usize] = &[2560, 3584, 4096];

    println!("=== QMV-batch vs Steel Split-K Diagnostic Benchmark ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, Q4, group_size={}, f16 input",
        WARMUP_ITERS, BENCH_ITERS, GROUP_SIZE,
    );
    println!("QMV-batch: auto-dispatch path (affine_quantized_matmul_batched)");
    println!("Steel-SK:  forced Steel Split-K (affine_qmm_steel_splitk_q4)");
    println!();

    // Header
    println!(
        "{:>3}  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "M", "K", "N", "qmv_us", "steel_us", "qmv_T", "steel_T", "speedup", "winner",
    );
    println!("{}", "-".repeat(80));

    // Collect results for summary
    let mut qmv_wins = 0usize;
    let mut steel_wins = 0usize;
    let mut speedups: Vec<(usize, f64)> = Vec::new(); // (m, speedup where >1 means QMV faster)

    for &m in m_values {
        for &k in k_values {
            for &n in n_values {
                let qmv = bench_qmv_batch(&registry, &queue, device, m, n, k);
                let steel = bench_steel_splitk(&registry, &queue, device, m, n, k);

                let qmv_tf = tflops(m, n, k, qmv.p50);
                let steel_tf = tflops(m, n, k, steel.p50);

                // speedup > 1 means QMV-batch is faster (lower latency)
                let speedup = steel.p50 / qmv.p50;
                let winner = if speedup > 1.02 {
                    qmv_wins += 1;
                    "QMV"
                } else if speedup < 0.98 {
                    steel_wins += 1;
                    "Steel"
                } else {
                    "~tie"
                };
                speedups.push((m, speedup));

                println!(
                    "{:>3}  {:>4}  {:>4}  {:>10.1}  {:>10.1}  {:>10.3}  {:>10.3}  {:>7.2}x  {:>8}",
                    m, k, n, qmv.p50, steel.p50, qmv_tf, steel_tf, speedup, winner,
                );
            }
        }
        println!();
    }

    // Summary
    println!("{}", "=".repeat(80));
    let total = speedups.len();
    let ties = total - qmv_wins - steel_wins;
    println!(
        "Total: {} cases | QMV wins: {} | Steel wins: {} | Ties: {}",
        total, qmv_wins, steel_wins, ties,
    );
    println!();

    // Per-M summary
    println!("Per-M summary (speedup = steel_us / qmv_us, >1 means QMV faster):");
    for &m in m_values {
        let m_speedups: Vec<f64> = speedups
            .iter()
            .filter(|(mm, _)| *mm == m)
            .map(|(_, s)| *s)
            .collect();
        if !m_speedups.is_empty() {
            let avg = m_speedups.iter().sum::<f64>() / m_speedups.len() as f64;
            let min = m_speedups.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = m_speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let winner = if avg > 1.02 {
                "QMV"
            } else if avg < 0.98 {
                "Steel"
            } else {
                "~tie"
            };
            println!(
                "  M={:<3}  avg={:.2}x  min={:.2}x  max={:.2}x  => {}",
                m, avg, min, max, winner,
            );
        }
    }

    println!();
    println!("Done.");
}
