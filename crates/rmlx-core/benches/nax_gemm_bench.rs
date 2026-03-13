//! ⚠️ NON-PRODUCTION PATH — direct kernel encoding for NAX GEMM variant comparison.
//! Bypasses matmul() dispatch. Development/tuning only.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! NAX GEMM v1/v2/v3 A/B test benchmark.
//!
//! Compares three NAX kernel variants for f16 GEMM:
//!   - gemm_nax_f16    (v1): scalar loads, frag->ct copy, unroll_count(2)
//!   - gemm_nax_v2_f16 (v2): half4 vectorized loads, direct ct, unroll_count(2)
//!   - gemm_nax_v3_f16 (v3): half4 vectorized loads, direct ct, no unroll
//!
//! Sizes: M={128,256,512,1024}, N=3584, K=3584 (Qwen QKV dimensions)
//!
//! Usage: cargo bench -p rmlx-core --bench nax_gemm_bench

use std::time::{Duration, Instant};

use half::f16;
use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 20;
const BENCH_ITERS: usize = 100;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Stats {
    mean: f64,
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
        let p50 = percentile(&micros, 50.0);
        Stats { mean, p50 }
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
// Helpers
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

fn make_u32_buf(device: &metal::Device, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
}

fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Generic NAX kernel benchmarker
// ---------------------------------------------------------------------------

/// Benchmark a specific NAX kernel by name.
/// All NAX variants share the same buffer layout & grid config (BM=128, BN=128, 512 threads).
#[allow(clippy::too_many_arguments)]
fn bench_nax_kernel(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    kernel_name: &str,
    a: &Array,
    b: &Array,
    c: &Array,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let bm = 128usize;
    let bn = 128usize;
    let bk = 32usize;

    let constants = ops::matmul::matmul_align_constants(m, n, k, bm, bn, bk);
    let pipeline = registry
        .get_pipeline_with_constants(kernel_name, DType::Float16, &constants)
        .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"));

    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_log = ops::matmul::compute_swizzle_log(m, n, bm, bn);
    let swizzle_buf = make_u32_buf(device, swizzle_log);

    // Dummy residual buffer (has_residual=false via function constant)
    let residual_buf = make_u32_buf(device, 0);

    let grid_x = n.div_ceil(bn);
    let grid_y = m.div_ceil(bm);
    let grid = MTLSize::new(grid_x as u64, grid_y as u64, 1);
    let tg = MTLSize::new(512, 1, 1);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(c.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa_buf), 0);
        enc.set_buffer(7, Some(&bsb_buf), 0);
        enc.set_buffer(8, Some(&bsc_buf), 0);
        enc.set_buffer(9, Some(&swizzle_buf), 0);
        enc.set_buffer(10, Some(&residual_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), 0);
        enc.set_buffer(1, Some(b.metal_buffer()), 0);
        enc.set_buffer(2, Some(c.metal_buffer()), 0);
        enc.set_buffer(3, Some(&m_buf), 0);
        enc.set_buffer(4, Some(&n_buf), 0);
        enc.set_buffer(5, Some(&k_buf), 0);
        enc.set_buffer(6, Some(&bsa_buf), 0);
        enc.set_buffer(7, Some(&bsb_buf), 0);
        enc.set_buffer(8, Some(&bsc_buf), 0);
        enc.set_buffer(9, Some(&swizzle_buf), 0);
        enc.set_buffer(10, Some(&residual_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
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

    let n = 3584usize;
    let k = 3584usize;
    let m_values: &[usize] = &[128, 256, 512, 1024];

    let kernels: &[(&str, &str)] = &[
        ("gemm_nax_f16", "v1(scalar)"),
        ("gemm_nax_v2_f16", "v2(h4+unrl)"),
        ("gemm_nax_v3_f16", "v3(h4+auto)"),
    ];

    println!("=== NAX GEMM v1/v2/v3 A/B Test ===");
    println!(
        "N={}, K={}, Warmup={}, Bench={} iters",
        n, k, WARMUP_ITERS, BENCH_ITERS,
    );
    println!("BM=128, BN=128, BK=32, 16 SGs, 512 threads");
    println!();

    // Collect all results first (run largest M first to avoid thermal throttling)
    let mut all_results: Vec<(usize, Vec<Stats>)> = Vec::new();

    for &m in m_values.iter().rev() {
        print!("Benchmarking M={}...", m);
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);

        let mut row_stats: Vec<Stats> = Vec::new();
        for &(kernel_name, label) in kernels {
            let stats =
                bench_nax_kernel(&registry, &queue, device, kernel_name, &a, &b, &c, m, n, k);
            print!(" {}={:.1}us", label, stats.p50);
            row_stats.push(stats);
        }
        println!();
        all_results.push((m, row_stats));
    }

    // Sort by M ascending for display
    all_results.sort_by_key(|(m, _)| *m);

    println!();
    println!("=== Results ===");
    println!();
    println!(
        "| {:>5} | {:>12} | {:>10} | {:>12} | {:>10} | {:>12} | {:>10} | {:>7} | {:>7} |",
        "M",
        "v1 p50(us)",
        "v1 TFLOPS",
        "v2 p50(us)",
        "v2 TFLOPS",
        "v3 p50(us)",
        "v3 TFLOPS",
        "v2/v1",
        "v3/v1",
    );
    println!(
        "|{:-<7}|{:-<14}|{:-<12}|{:-<14}|{:-<12}|{:-<14}|{:-<12}|{:-<9}|{:-<9}|",
        "", "", "", "", "", "", "", "", "",
    );

    for (m, stats) in &all_results {
        let v1 = &stats[0];
        let v2 = &stats[1];
        let v3 = &stats[2];
        let tf1 = tflops(*m, n, k, v1.p50);
        let tf2 = tflops(*m, n, k, v2.p50);
        let tf3 = tflops(*m, n, k, v3.p50);
        let speedup_v2 = v1.p50 / v2.p50;
        let speedup_v3 = v1.p50 / v3.p50;

        println!(
            "| {:>5} | {:>12.1} | {:>10.2} | {:>12.1} | {:>10.2} | {:>12.1} | {:>10.2} | {:>6.3}x | {:>6.3}x |",
            m,
            v1.p50, tf1,
            v2.p50, tf2,
            v3.p50, tf3,
            speedup_v2, speedup_v3,
        );
    }

    println!();
    println!("Done.");
}
