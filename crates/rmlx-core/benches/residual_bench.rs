//! GEMM + residual epilogue fusion benchmark (Phase H-2).
//!
//! Compares two approaches:
//!   1. `matmul_into_cb` + `add_into_cb` (two dispatches)
//!   2. `matmul_add_residual_into_cb`    (fused, single dispatch)
//!
//! MoE-focused scenarios (Mixtral 8x7B, DeepSeek-V2).
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench residual_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper (same as gemm_bench)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
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
// f16 random array generation (deterministic PRNG)
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

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
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

// ---------------------------------------------------------------------------
// Benchmark scenarios
// ---------------------------------------------------------------------------

struct Scenario {
    label: &'static str,
    k: usize,
    n: usize,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        // Mixtral 8x7B: gate/up projection
        Scenario {
            label: "Mixtral gate/up",
            k: 4096,
            n: 14336,
        },
        // Mixtral 8x7B: down projection
        Scenario {
            label: "Mixtral down",
            k: 14336,
            n: 4096,
        },
        // DeepSeek-V2: expert intermediate
        Scenario {
            label: "DS-V2 expert",
            k: 5120,
            n: 1536,
        },
    ]
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Benchmark the separate (matmul + add) path for a given M.
fn bench_separate(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    scenario: &Scenario,
    m: usize,
) -> Stats {
    let k = scenario.k;
    let n = scenario.n;

    let a = rand_array(device, &[m, k], 42);
    let b = rand_array(device, &[k, n], 44);
    let residual = rand_array(device, &[m, n], 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let c = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
        let _ = ops::binary::add_into_cb(registry, &c, &residual, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let c = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
        let _ = ops::binary::add_into_cb(registry, &c, &residual, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    let stats = Stats::from_durations(&times);
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let tflops = flops / (stats.p50 * 1e-6) / 1e12;
    println!(
        "  M={:5}  separate  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS={:.2}",
        m, stats.p50, stats.mean, stats.std_dev, tflops,
    );
    stats
}

/// Benchmark the fused (matmul_add_residual) path for a given M, with error handling.
///
/// Returns `None` if the fused kernel is not supported for this config (e.g. non-MlxArch tile).
fn bench_fused(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    scenario: &Scenario,
    m: usize,
    sep_stats: &Stats,
) -> Option<Stats> {
    let k = scenario.k;
    let n = scenario.n;

    let a = rand_array(device, &[m, k], 42);
    let b = rand_array(device, &[k, n], 44);
    let residual = rand_array(device, &[m, n], 99);

    // Probe: try one fused dispatch; if unsupported, skip gracefully.
    {
        let cb = queue.new_command_buffer_with_unretained_references();
        match ops::matmul::matmul_add_residual_into_cb(registry, &a, &b, &residual, cb) {
            Ok(_) => {
                cb.commit();
                cb.wait_until_completed();
            }
            Err(e) => {
                // Don't commit a partially-encoded CB — just drop it.
                eprintln!("  M={:5}  fused     SKIPPED — unsupported config: {}", m, e);
                return None;
            }
        }
    }

    // Warmup (first probe already served as one warmup iter)
    for _ in 1..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::matmul::matmul_add_residual_into_cb(registry, &a, &b, &residual, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::matmul::matmul_add_residual_into_cb(registry, &a, &b, &residual, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    let stats = Stats::from_durations(&times);
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let tflops = flops / (stats.p50 * 1e-6) / 1e12;
    let speedup = sep_stats.p50 / stats.p50;
    println!(
        "  M={:5}  fused     p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS={:.2}  speedup={:.2}x",
        m, stats.p50, stats.mean, stats.std_dev, tflops, speedup,
    );
    Some(stats)
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    // M values for the separate (matmul + add) baseline — includes M=1 for context.
    let m_values_separate = [1, 8, 32, 128, 256, 512];

    // M values for the fused path — M=1 is excluded because the fused kernel
    // (matmul_add_residual_into_cb) only supports MlxArch tiles (M >= 33, N >= 33).
    // Small-M dispatches to GEMV/Simd/Skinny paths which lack the residual epilogue.
    // We start at M=64 (first full MlxArch tile boundary); M=32 falls in Skinny range.
    let m_values_fused = [64, 128, 256, 512];

    println!("=== GEMM + Residual Epilogue Fusion Benchmark (Phase H-2, f16) ===");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!("Comparing: matmul+add (2 dispatches) vs matmul_add_residual (fused)");
    println!("Note: fused path requires MlxArch (M>=33, N>=33); small M values are separate-only.");
    println!();

    for scenario in &scenarios() {
        println!("[{}: K={}, N={}]", scenario.label, scenario.k, scenario.n);

        // Run separate baseline for all M values (including M=1 for context)
        let mut sep_stats_map = std::collections::HashMap::new();
        for &m in &m_values_separate {
            let stats = bench_separate(&registry, &queue, device, scenario, m);
            sep_stats_map.insert(m, stats);
        }

        // Run fused path only for M values that support MlxArch
        for &m in &m_values_fused {
            // Get or compute separate stats for comparison
            let sep_stats = sep_stats_map
                .entry(m)
                .or_insert_with(|| bench_separate(&registry, &queue, device, scenario, m));
            bench_fused(&registry, &queue, device, scenario, m, sep_stats);
        }
        println!();
    }

    println!("Done.");
}
