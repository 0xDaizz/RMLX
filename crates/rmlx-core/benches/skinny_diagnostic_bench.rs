//! Skinny Split-K Diagnostic Benchmark
//!
//! Isolates Skinny Split-K performance components at MoE-critical dimensions.
//! For each (M, K, N), compares three dispatch paths:
//!   - Skinny-SK: affine_quantized_matmul_batched (auto-routes through Skinny Split-K)
//!   - Steel:     affine_quantized_matmul_steel (double-buffered GEMM, no Split-K)
//!   - NAX:       affine_qmm_nax_q4_into_cb (4SG MMA kernel)
//!
//! Also prints theoretical compute/BW bounds and MLX reference for context.
//!
//! MLX reference: M3 Ultra 80-core, mlx-lm 0.24, group_size=64, f16
//!
//! Usage: cargo bench -p rmlx-core --bench skinny_diagnostic_bench

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

// Peak specs: M3 Ultra 80-core
const PEAK_TFLOPS: f64 = 65.54; // FP16 peak TFLOPS
const MEM_BW_BYTES: f64 = 819e9; // memory bandwidth in bytes/s

// ---------------------------------------------------------------------------
// MLX reference data: (M, K, N) -> latency_us
// ---------------------------------------------------------------------------

struct MlxRef {
    m: usize,
    k: usize,
    n: usize,
    us: f64,
}

fn mlx_reference() -> Vec<MlxRef> {
    vec![
        MlxRef { m: 32, k: 3584, n: 2560, us: 280.5 },
        MlxRef { m: 64, k: 3584, n: 2560, us: 291.2 },
        MlxRef { m: 32, k: 3584, n: 4096, us: 271.5 },
        MlxRef { m: 64, k: 3584, n: 4096, us: 294.5 },
        MlxRef { m: 32, k: 4096, n: 2560, us: 293.3 },
        MlxRef { m: 64, k: 4096, n: 2560, us: 302.3 },
        MlxRef { m: 32, k: 4096, n: 4096, us: 298.8 },
        MlxRef { m: 64, k: 4096, n: 4096, us: 327.9 },
    ]
}

fn find_mlx_ref(refs: &[MlxRef], m: usize, k: usize, n: usize) -> Option<&MlxRef> {
    refs.iter().find(|r| r.m == m && r.k == k && r.n == n)
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
// Theoretical bounds
// ---------------------------------------------------------------------------

/// Compute bound: 2*M*N*K / peak_tflops (in us)
fn compute_bound_us(m: usize, n: usize, k: usize) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (PEAK_TFLOPS * 1e12) * 1e6
}

/// BW bound for GEMM: (N*K/2 + M*K*2 + M*N*2) / mem_bw (in us)
/// Q4 weights (N*K/2 bytes) + f16 A (M*K*2 bytes) + f16 output (M*N*2 bytes)
fn bw_bound_gemm_us(m: usize, n: usize, k: usize) -> f64 {
    let bytes = (n * k) as f64 / 2.0 + (m * k * 2) as f64 + (m * n * 2) as f64;
    bytes / MEM_BW_BYTES * 1e6
}

/// BW bound for QMV-batch: (M * N*K/2 + M*K*2 + M*N*2) / mem_bw (in us)
/// M redundant B loads of the full weight matrix
fn bw_bound_qmv_us(m: usize, n: usize, k: usize) -> f64 {
    let bytes = (m as f64 * (n * k) as f64 / 2.0) + (m * k * 2) as f64 + (m * n * 2) as f64;
    bytes / MEM_BW_BYTES * 1e6
}

// ---------------------------------------------------------------------------
// Benchmark runners
// ---------------------------------------------------------------------------

/// Full dispatch path: affine_quantized_matmul_batched (routes through Skinny Split-K).
fn bench_skinny_sk(
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
            .expect("skinny-sk warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("skinny-sk bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// Steel direct: affine_quantized_matmul_steel (double-buffered GEMM, no Split-K).
fn bench_steel(
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
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("steel warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("steel bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// NAX direct: affine_qmm_nax_q4_into_cb (4SG MMA kernel).
fn bench_nax(
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
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, cb)
            .expect("nax warmup failed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, cb)
            .expect("nax bench failed");
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

    let k_values: &[usize] = &[3584, 4096];
    let n_values: &[usize] = &[2560, 4096];
    let m_values: &[usize] = &[32, 64];

    println!("=== Skinny Split-K Diagnostic Benchmark ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, Q4, group_size={}, f16 input",
        WARMUP_ITERS, BENCH_ITERS, GROUP_SIZE,
    );
    println!(
        "Peak: {:.2} TFLOPS FP16, {:.0} GB/s memory BW",
        PEAK_TFLOPS,
        MEM_BW_BYTES / 1e9,
    );
    println!();

    for &k in k_values {
        for &n in n_values {
            for &m in m_values {
                let mlx = find_mlx_ref(&mlx_refs, m, k, n);
                let mlx_us = mlx.map(|r| r.us).unwrap_or(0.0);
                let mlx_tf = tflops(m, n, k, mlx_us);

                let comp_us = compute_bound_us(m, n, k);
                let bw_gemm_us = bw_bound_gemm_us(m, n, k);
                let bw_qmv_us = bw_bound_qmv_us(m, n, k);

                // Run benchmarks
                let sk = bench_skinny_sk(&registry, &queue, device, m, n, k);
                let steel = bench_steel(&registry, &queue, device, m, n, k);
                let nax = bench_nax(&registry, &queue, device, m, n, k);

                let sk_tf = tflops(m, n, k, sk.p50);
                let steel_tf = tflops(m, n, k, steel.p50);
                let nax_tf = tflops(m, n, k, nax.p50);

                let sk_parity = if mlx_us > 0.0 { mlx_us / sk.p50 * 100.0 } else { 0.0 };
                let steel_parity = if mlx_us > 0.0 { mlx_us / steel.p50 * 100.0 } else { 0.0 };
                let nax_parity = if mlx_us > 0.0 { mlx_us / nax.p50 * 100.0 } else { 0.0 };

                // Find best
                let (best_name, best_parity) = [
                    ("Skinny-SK", sk_parity),
                    ("Steel", steel_parity),
                    ("NAX", nax_parity),
                ]
                .iter()
                .copied()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

                let gap_vs_mlx = best_parity - 100.0;

                println!("M={:<3} K={:<4} N={:<4}", m, k, n);
                println!(
                    "  MLX ref:     {:6.1} us ({:.3} TFLOPS)",
                    mlx_us, mlx_tf,
                );
                println!(
                    "  Compute min: {:5.1} us  |  BW min (GEMM): {:5.1} us  |  BW min (QMV): {:5.1} us",
                    comp_us, bw_gemm_us, bw_qmv_us,
                );
                println!(
                    "  Skinny-SK: {:7.1} us ({:.3} T)  parity {:5.1}%",
                    sk.p50, sk_tf, sk_parity,
                );
                println!(
                    "  Steel:     {:7.1} us ({:.3} T)  parity {:5.1}%",
                    steel.p50, steel_tf, steel_parity,
                );
                println!(
                    "  NAX:       {:7.1} us ({:.3} T)  parity {:5.1}%",
                    nax.p50, nax_tf, nax_parity,
                );
                println!(
                    "  Best:        [{}] {:+.1}% vs MLX gap",
                    best_name, gap_vs_mlx,
                );
                println!();
            }
        }
    }

    println!("Done.");
}
