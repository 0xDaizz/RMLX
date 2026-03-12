//! Low-M Q4 QMM Benchmark — comprehensive kernel comparison for MoE expert inference.
//!
//! Tests all available Q4 QMM kernel approaches at low-M (M=1,4,8,16,32)
//! using DeepSeek V3 expert dimensions:
//!   - gate/up: K=7168, N=2048
//!   - down:    K=2048, N=7168
//!
//! Kernel variants tested:
//!   1. Auto dispatch   — `affine_quantized_matmul_batched` (auto-selects Skinny/NAX/Steel/MMA)
//!   2. Steel v2        — `affine_quantized_matmul_steel` (forced Steel path)
//!   3. NAX             — `affine_qmm_nax_q4_into_cb` (M>=32, K%64==0 only)
//!   4. QMV (M=1 only)  — `affine_quantized_matmul` (MLX qmv_fast pattern)
//!   5. Scalar           — `affine_quantized_matmul_batched_scalar`
//!   6. Batched QMV      — `affine_qmv_batched_q4` (M-batched qdot pattern)
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench qmm_lowm_bench

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

// ---------------------------------------------------------------------------
// Stats helper
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
// Random data generation (LCG — deterministic, no deps)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

#[allow(dead_code)]
fn rand_f32_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut state = seed;
    let data: Vec<f32> = (0..numel)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.1
        })
        .collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, numel * 4) };
    Array::from_bytes(device, bytes, shape.to_vec(), DType::Float32)
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
    bits: u32,
    group_size: u32,
    seed: u64,
) -> QuantizedWeight {
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
        group_size,
        bits,
        out_features,
        in_features,
    )
    .expect("Failed to create QuantizedWeight")
}

// make_quantized_weight_f16 removed — make_quantized_weight now natively creates f16 scales/biases.

// ---------------------------------------------------------------------------
// TFLOPS calculation
// ---------------------------------------------------------------------------

fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Result row for pretty printing
// ---------------------------------------------------------------------------

struct BenchResult {
    kernel: &'static str,
    m: usize,
    n: usize,
    k: usize,
    stats: Option<Stats>,
    skip_reason: Option<&'static str>,
}

impl BenchResult {
    fn ok(kernel: &'static str, m: usize, n: usize, k: usize, stats: Stats) -> Self {
        Self {
            kernel,
            m,
            n,
            k,
            stats: Some(stats),
            skip_reason: None,
        }
    }

    fn skipped(kernel: &'static str, m: usize, n: usize, k: usize, reason: &'static str) -> Self {
        Self {
            kernel,
            m,
            n,
            k,
            stats: None,
            skip_reason: Some(reason),
        }
    }

    fn print(&self) {
        if let Some(ref s) = self.stats {
            let tf = tflops(self.m, self.n, self.k, s.p50);
            let tf_mean = tflops(self.m, self.n, self.k, s.mean);
            println!(
                "  {:<12} M={:<3} K={:<5} N={:<5}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.4}  TFLOPS(mean)={:.4}",
                self.kernel, self.m, self.k, self.n,
                s.p50, s.mean, s.std_dev, tf, tf_mean,
            );
        } else {
            println!(
                "  {:<12} M={:<3} K={:<5} N={:<5}  N/A ({})",
                self.kernel,
                self.m,
                self.k,
                self.n,
                self.skip_reason.unwrap_or("unsupported"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Individual kernel bench runners
// ---------------------------------------------------------------------------

/// Auto dispatch — calls `affine_quantized_matmul_batched` which auto-selects
/// Skinny (M<=32), NAX (M>=32 + K%64==0), Steel, or standard MMA.
fn bench_auto(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Auto warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Auto bench failed");
        times.push(start.elapsed());
    }
    BenchResult::ok("Auto", m, n, k, Stats::from_durations(&times))
}

/// Steel v2 — forced `affine_quantized_matmul_steel`.
fn bench_steel(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("Steel warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("Steel bench failed");
        times.push(start.elapsed());
    }
    BenchResult::ok("Steel-v2", m, n, k, Stats::from_durations(&times))
}

/// NAX — `affine_qmm_nax_q4_into_cb` (M>=32, K%64==0 only, f16 input).
fn bench_nax(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    if m < 32 {
        return BenchResult::skipped("NAX", m, n, k, "requires M>=32");
    }
    if k % 64 != 0 {
        return BenchResult::skipped("NAX", m, n, k, "requires K%64==0");
    }

    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x_f16 = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw, cb)
            .expect("NAX warmup failed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw, cb)
            .expect("NAX bench failed");
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    BenchResult::ok("NAX", m, n, k, Stats::from_durations(&times))
}

/// QMV — `affine_quantized_matmul` (M=1 only, 1D vector, MLX qmv_fast pattern).
/// For M>1, we loop M times calling QMV once per row (simulates batched QMV).
fn bench_qmv(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);

    // Create M separate 1D vectors
    let vecs: Vec<Array> = (0..m)
        .map(|i| rand_f16_array(device, &[k], 99 + i as u64))
        .collect();

    // Warmup: call QMV for each row
    for _ in 0..WARMUP_ITERS {
        for v in &vecs {
            let _ = ops::quantized::affine_quantized_matmul(registry, &qw, v, queue)
                .expect("QMV warmup failed");
        }
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for v in &vecs {
            let _ = ops::quantized::affine_quantized_matmul(registry, &qw, v, queue)
                .expect("QMV bench failed");
        }
        times.push(start.elapsed());
    }
    let label = if m == 1 { "QMV" } else { "QMV-batched" };
    BenchResult::ok(label, m, n, k, Stats::from_durations(&times))
}

/// Scalar — `affine_quantized_matmul_batched_scalar`.
fn bench_scalar(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched_scalar(registry, &x, &qw, queue)
            .expect("Scalar warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched_scalar(registry, &x, &qw, queue)
            .expect("Scalar bench failed");
        times.push(start.elapsed());
    }
    BenchResult::ok("Scalar", m, n, k, Stats::from_durations(&times))
}

/// Batched QMV — `affine_qmv_batched_q4` (M-batched qdot pattern).
fn bench_batched_qmv(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_qmv_batched_q4(registry, &qw, &x, queue)
            .expect("BatchQMV warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_qmv_batched_q4(registry, &qw, &x, queue)
            .expect("BatchQMV bench failed");
        times.push(start.elapsed());
    }
    BenchResult::ok("BatchQMV", m, n, k, Stats::from_durations(&times))
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

    let gs: u32 = 32;
    let m_values: &[usize] = &[1, 4, 8, 16, 32];

    // DeepSeek V3 expert dimensions
    let dims: &[(&str, usize, usize)] =
        &[("DS-V3 gate/up", 7168, 2048), ("DS-V3 down", 2048, 7168)];

    println!("=== Low-M Q4 QMM Kernel Comparison ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, group_size={}",
        WARMUP_ITERS, BENCH_ITERS, gs,
    );
    println!("All kernels: f16 input | NAX: M>=32, K%64==0 only");
    println!("QMV: M=1 native; M>1 calls QMV M times sequentially");
    println!();

    for &(label, k, n) in dims {
        println!("=========================================================");
        println!("  {} (K={}, N={})", label, k, n);
        println!("=========================================================");
        println!();

        for &m in m_values {
            println!("--- M={} ---", m);

            // Run each kernel variant once
            let results: Vec<BenchResult> = vec![
                bench_auto(&registry, &queue, device, m, n, k, gs),
                bench_steel(&registry, &queue, device, m, n, k, gs),
                bench_nax(&registry, &queue, device, m, n, k, gs),
                bench_qmv(&registry, &queue, device, m, n, k, gs),
                bench_scalar(&registry, &queue, device, m, n, k, gs),
                bench_batched_qmv(&registry, &queue, device, m, n, k, gs),
            ];

            // Print each result
            for r in &results {
                r.print();
            }

            // Find and print best kernel by p50
            let mut best_label = "";
            let mut best_p50 = f64::MAX;
            for r in &results {
                if let Some(ref s) = r.stats {
                    if s.p50 < best_p50 {
                        best_p50 = s.p50;
                        best_label = r.kernel;
                    }
                }
            }
            if !best_label.is_empty() {
                let best_tf = tflops(m, n, k, best_p50);
                println!(
                    "  >> BEST: {} ({:.1}us, {:.4} TFLOPS)",
                    best_label, best_p50, best_tf,
                );
            }

            println!();
        }
    }

    // === f16 vs f32 A/B comparison ===
    println!("=========================================================");
    println!("  f16 vs f32 A/B Comparison (same thermal state)");
    println!("=========================================================");
    println!();

    // For each (M, K, N) combo, run f32 then f16 immediately after
    for &(label, k, n) in dims {
        for &m in &[1usize, 4, 8, 16] {
            if k % 512 != 0 {
                continue;
            }

            let qw = make_quantized_weight(device, n, k, 4, gs, 42);
            let x_f32 = rand_f32_array(device, &[m, k], 99);
            let x_f16 = rand_f16_array(device, &[m, k], 99);

            // Warmup both
            for _ in 0..WARMUP_ITERS {
                let _ =
                    ops::quantized::affine_quantized_matmul_batched(&registry, &x_f32, &qw, &queue);
                let _ =
                    ops::quantized::affine_quantized_matmul_batched(&registry, &x_f16, &qw, &queue);
            }

            // Bench f32
            let mut f32_times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let _ =
                    ops::quantized::affine_quantized_matmul_batched(&registry, &x_f32, &qw, &queue)
                        .expect("f32 bench failed");
                f32_times.push(start.elapsed());
            }
            let f32_stats = Stats::from_durations(&f32_times);

            // Bench f16 (immediately after, same thermal state)
            let mut f16_times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let _ =
                    ops::quantized::affine_quantized_matmul_batched(&registry, &x_f16, &qw, &queue)
                        .expect("f16 bench failed");
                f16_times.push(start.elapsed());
            }
            let f16_stats = Stats::from_durations(&f16_times);

            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let f32_tflops = flops / (f32_stats.p50 * 1e-6) / 1e12;
            let f16_tflops = flops / (f16_stats.p50 * 1e-6) / 1e12;
            let speedup = f32_stats.p50 / f16_stats.p50;

            println!(
                "  {label:12} M={m:<4} K={k:<6} N={n:<6}  f32={:>8.1}us ({:.4}T)  f16={:>8.1}us ({:.4}T)  f16/f32={:.2}x",
                f32_stats.p50,
                f32_tflops,
                f16_stats.p50,
                f16_tflops,
                speedup,
            );
        }
    }

    println!("\nDone.");
}
