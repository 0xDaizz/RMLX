//! ⚠️ NON-PRODUCTION PATH — direct kernel encoding for NAX tile parameter sweep.
//! Development/tuning only. Bypasses matmul() dispatch.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! NAX GEMM tile variant A/B benchmark.
//!
//! Compares four GEMM kernel variants across multiple M sizes:
//!   - gemm_nax_f16         (BM=128, BN=128, 16 SGs, 512 threads)
//!   - gemm_nax_64x128_f16  (BM=64, BN=128, 8 SGs, 256 threads)
//!   - gemm_nax_64x64_f16   (BM=64, BN=64, 4 SGs, 128 threads — NAX)
//!   - gemm_mlx_f16         (BM=64, BN=64, 4 SGs, 128 threads — MlxArch fallback)
//!
//! Sizes: M={32,64,96,128,192,256,384,512,768,1024}, N=3584, K=3584
//!
//! Usage: cargo bench -p rmlx-core --bench nax_tile_bench

use std::time::{Duration, Instant};

use half::f16;
use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    p50: f64,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
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
// Kernel config
// ---------------------------------------------------------------------------

struct KernelConfig {
    name: &'static str,
    label: &'static str,
    bm: usize,
    bn: usize,
    bk: usize,
    threads: u64,
    min_m: usize,
}

const KERNELS: &[KernelConfig] = &[
    KernelConfig {
        name: "gemm_nax_f16",
        label: "NAX 128x128",
        bm: 128,
        bn: 128,
        bk: 32,
        threads: 512,
        min_m: 128,
    },
    KernelConfig {
        name: "gemm_nax_64x128_f16",
        label: "NAX 64x128",
        bm: 64,
        bn: 128,
        bk: 32,
        threads: 256,
        min_m: 64,
    },
    KernelConfig {
        name: "gemm_nax_64x64_f16",
        label: "NAX 64x64",
        bm: 64,
        bn: 64,
        bk: 32,
        threads: 128,
        min_m: 64,
    },
    KernelConfig {
        name: "gemm_mlx_f16",
        label: "MLX 64x64",
        bm: 64,
        bn: 64,
        bk: 16,
        threads: 128,
        min_m: 33,
    },
];

// ---------------------------------------------------------------------------
// Generic kernel benchmarker
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_kernel(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    cfg: &KernelConfig,
    a: &Array,
    b: &Array,
    c: &Array,
    m: usize,
    n: usize,
    k: usize,
) -> Option<Stats> {
    if m < cfg.min_m {
        return None;
    }

    let constants = ops::matmul::matmul_align_constants(m, n, k, cfg.bm, cfg.bn, cfg.bk);
    let pipeline = match registry.get_pipeline_with_constants(cfg.name, DType::Float16, &constants)
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping {}: {e}", cfg.name);
            return None;
        }
    };

    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_log = ops::matmul::compute_swizzle_log(m, n, cfg.bm, cfg.bn);
    let swizzle_buf = make_u32_buf(device, swizzle_log);

    // Dummy residual buffer (has_residual=false via function constant)
    let residual_buf = make_u32_buf(device, 0);

    let grid_x = n.div_ceil(cfg.bn);
    let grid_y = m.div_ceil(cfg.bm);
    let grid = MTLSize::new(grid_x as u64, grid_y as u64, 1);
    let tg = MTLSize::new(cfg.threads, 1, 1);

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

    Some(Stats::from_durations(&times))
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
    let m_values: &[usize] = &[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

    println!("=== NAX Tile Variant A/B Test ===");
    println!(
        "N={}, K={}, Warmup={}, Bench={} iters",
        n, k, WARMUP_ITERS, BENCH_ITERS,
    );
    println!();

    // Collect results (run largest M first to avoid thermal throttling)
    let mut all_results: Vec<(usize, Vec<Option<Stats>>)> = Vec::new();

    for &m in m_values.iter().rev() {
        print!("Benchmarking M={}...", m);
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);

        let mut row: Vec<Option<Stats>> = Vec::new();
        for cfg in KERNELS {
            let stats = bench_kernel(&registry, &queue, device, cfg, &a, &b, &c, m, n, k);
            if let Some(ref s) = stats {
                print!(" {}={:.1}us", cfg.label, s.p50);
            } else {
                print!(" {}=N/A", cfg.label);
            }
            row.push(stats);
        }
        println!();
        all_results.push((m, row));
    }

    // Sort ascending for display
    all_results.sort_by_key(|(m, _)| *m);

    println!();
    println!("=== Results ===");
    println!();

    // Header
    print!("| {:>5} |", "M");
    for cfg in KERNELS {
        print!(" {:>12} | {:>8} |", format!("{} us", cfg.label), "TFLOPS");
    }
    println!(" {:>8} |", "64x128/128x128");

    // Separator
    print!("|{:-<7}|", "");
    for _ in KERNELS {
        print!("{:-<14}|{:-<10}|", "", "");
    }
    println!("{:-<10}|", "");

    // Data rows
    for (m, row) in &all_results {
        print!("| {:>5} |", m);
        let mut tflops_vals: Vec<Option<f64>> = Vec::new();
        for (i, _cfg) in KERNELS.iter().enumerate() {
            if let Some(ref stats) = row[i] {
                let tf = tflops(*m, n, k, stats.p50);
                print!(" {:>12.1} | {:>8.2} |", stats.p50, tf);
                tflops_vals.push(Some(tf));
            } else {
                print!(" {:>12} | {:>8} |", "N/A", "N/A");
                tflops_vals.push(None);
            }
        }
        // Speedup: NAX 64x128 vs NAX 128x128
        match (
            tflops_vals.first().and_then(|v| *v),
            tflops_vals.get(1).and_then(|v| *v),
        ) {
            (Some(t128), Some(t64)) => print!(" {:>7.3}x |", t64 / t128),
            _ => print!(" {:>8} |", "N/A"),
        }
        println!();
    }

    // =========================================================================
    // Low-M Benchmark (M=4..64)
    // =========================================================================
    println!();
    println!("=== Low-M Benchmark (M=4..64) ===");
    println!(
        "N={}, K={}, Warmup={}, Bench={} iters",
        n, k, WARMUP_ITERS, BENCH_ITERS,
    );
    println!();

    let low_m_kernels: &[KernelConfig] = &[
        KernelConfig {
            name: "gemm_nax_64x64_f16",
            label: "NAX 64x64",
            bm: 64,
            bn: 64,
            bk: 32,
            threads: 128,
            min_m: 1, // NAX works for all M (partial tiles via align_M=false)
        },
        KernelConfig {
            name: "gemm_mlx_f16",
            label: "MLX 64x64",
            bm: 64,
            bn: 64,
            bk: 16,
            threads: 128,
            min_m: 33,
        },
        KernelConfig {
            name: "gemm_mlx_small_f16",
            label: "MLX Small 32x32",
            bm: 32,
            bn: 32,
            bk: 16,
            threads: 64,
            min_m: 17,
        },
        KernelConfig {
            name: "gemm_mlx_m16_f16",
            label: "MLX Micro 16x32",
            bm: 16,
            bn: 32,
            bk: 16,
            threads: 64,
            min_m: 5,
        },
    ];

    let low_m_values: &[usize] = &[4, 8, 16, 32, 48, 64];

    // Collect results: Vec<(m, kernel_label, Stats)>
    let mut low_m_results: Vec<(usize, &str, Stats)> = Vec::new();

    // Run largest M first to avoid thermal throttling
    for &m in low_m_values.iter().rev() {
        print!("Benchmarking Low-M={}...", m);
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);

        for cfg in low_m_kernels {
            if m < cfg.min_m {
                continue;
            }
            // MLX Small: only for M=17..32 with N > 4096
            if cfg.name == "gemm_mlx_small_f16" && (m > 32 || n <= 4096) {
                continue;
            }
            // MLX Micro: only for M=5..16 with N <= 4096
            if cfg.name == "gemm_mlx_m16_f16" && (m > 16 || n > 4096) {
                continue;
            }

            if let Some(stats) = bench_kernel(&registry, &queue, device, cfg, &a, &b, &c, m, n, k) {
                print!(" {}={:.1}us", cfg.label, stats.p50);
                low_m_results.push((m, cfg.label, stats));
            } else {
                print!(" {}=N/A", cfg.label);
            }
        }
        println!();
    }

    // Sort ascending by M for display
    low_m_results.sort_by_key(|(m, _, _)| *m);

    println!();

    // Header
    println!(
        "| {:>5} | {:<18} | {:>10} | {:>8} |",
        "M", "Kernel", "Time (us)", "TFLOPS"
    );
    println!("|{:-<7}|{:-<20}|{:-<12}|{:-<10}|", "", "", "", "");

    // Data rows
    for (m, label, stats) in &low_m_results {
        let tf = tflops(*m, n, k, stats.p50);
        println!(
            "| {:>5} | {:<18} | {:>10.1} | {:>8.2} |",
            m, label, stats.p50, tf
        );
    }
}
