//! Standalone GEMM kernel benchmark — per-op latency and TFLOPS.
//!
//! Tests matmul [M, K] @ [K, N] for various M, K, N combinations using f16 dtype.
//! Includes MoE grouped GEMM benchmarks (set BENCH_GROUPED=0 to skip).
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench gemm_bench

use std::time::{Duration, Instant};

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
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    min: f64,
    max: f64,
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
// Helpers
// ---------------------------------------------------------------------------

fn make_u32_buf(device: &metal::Device, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_gemm(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    a: &Array,
    b: &Array,
    m: usize,
    k: usize,
    n: usize,
    label: &str,
) {
    // Select tile config and kernel
    let tile = ops::matmul::select_tile_config_with_dtype(m, n, k, a.dtype());
    let kernel_name = match (tile.variant, a.dtype()) {
        (ops::matmul::TileVariant::Full, DType::Float16) => "gemm_tiled_f16",
        (ops::matmul::TileVariant::Full, DType::Float32) => "gemm_tiled_f32",
        (ops::matmul::TileVariant::MlxArch, DType::Float16) => "gemm_mlx_f16",
        _ => {
            // Fallback to matmul_into_cb for non-full variants
            // (not the focus of this benchmark)
            for _ in 0..WARMUP_ITERS {
                let cb = queue.new_command_buffer();
                let _ = ops::matmul::matmul_into_cb(registry, a, b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
            }
            let mut times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let _ = ops::matmul::matmul_into_cb(registry, a, b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }
            let stats = Stats::from_durations(&times);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let tflops_p50 = flops / (stats.p50 * 1e-6) / 1e12;
            let tflops_mean = flops / (stats.mean * 1e-6) / 1e12;
            println!(
                "  M={:5}  {:<12}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  p95={:8.1}us  min={:8.1}us  max={:8.1}us  TFLOPS(p50)={:.2}  TFLOPS(mean)={:.2}",
                m, label, stats.p50, stats.mean, stats.std_dev, stats.p95, stats.min, stats.max, tflops_p50, tflops_mean,
            );
            return;
        }
    };

    let pipeline = if tile.variant == ops::matmul::TileVariant::MlxArch {
        let constants = ops::matmul::matmul_align_constants(m, n, k, tile.bm, tile.bn, tile.bk);
        registry
            .get_pipeline_with_constants(kernel_name, a.dtype(), &constants)
            .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"))
    } else {
        registry
            .get_pipeline(kernel_name, a.dtype())
            .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"))
    };

    // Pre-allocate output and constant buffers ONCE
    let c = Array::zeros(device, &[m, n], a.dtype());
    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_buf = make_u32_buf(
        device,
        ops::matmul::compute_swizzle_log(m, n, tile.bm, tile.bn),
    );

    let grid_x = ceil_div(n, tile.bn) as u64;
    let grid_y = ceil_div(m, tile.bm) as u64;
    let grid = MTLSize::new(grid_x, grid_y, 1);
    let tg_threads = match tile.variant {
        ops::matmul::TileVariant::MlxArch => 64_u64,
        _ => 256_u64,
    };
    let tg = MTLSize::new(tg_threads, 1, 1);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
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
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
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
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let tflops_p50 = flops / (stats.p50 * 1e-6) / 1e12;
    let tflops_mean = flops / (stats.mean * 1e-6) / 1e12;

    println!(
        "  M={:5}  {:<12}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  p95={:8.1}us  min={:8.1}us  max={:8.1}us  TFLOPS(p50)={:.2}  TFLOPS(mean)={:.2}",
        m, label, stats.p50, stats.mean, stats.std_dev, stats.p95, stats.min, stats.max, tflops_p50, tflops_mean,
    );
}

fn bench_grouped_gemm(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
) {
    let k = 4096;

    // MoE scenarios: (label, expert_ms, N)
    let scenarios: Vec<(&str, Vec<usize>, usize)> = vec![
        ("8E×M=4 N=2048", vec![4; 8], 2048),
        ("8E×variable N=2048", vec![3, 5, 8, 2, 6, 4, 7, 1], 2048),
        ("8E×M=8 N=1536", vec![8; 8], 1536),
        ("8E×M=4 N=768", vec![4; 8], 768),
        ("64E×M=1 N=2048", vec![1; 64], 2048),
    ];

    for (label, expert_ms, n) in &scenarios {
        let total_m: usize = expert_ms.iter().sum();
        let num_experts = expert_ms.len();
        let total_flops = 2.0 * total_m as f64 * k as f64 * *n as f64;

        // Create stacked inputs
        let a_stacked = rand_array(device, &[total_m, k], 42);
        let b_stacked = rand_array(device, &[num_experts, k * *n], 44);

        // Baseline: individual dispatches
        {
            // Pre-allocate expert arrays ONCE
            let experts: Vec<_> = expert_ms
                .iter()
                .map(|&m_i| {
                    (
                        rand_array(device, &[m_i.max(1), k], 42),
                        rand_array(device, &[k, *n], 44),
                    )
                })
                .collect();

            // Warmup
            for _ in 0..WARMUP_ITERS {
                for (a_exp, b_exp) in &experts {
                    let cb = queue.new_command_buffer();
                    let _ = ops::matmul::matmul_into_cb(registry, a_exp, b_exp, cb);
                    cb.commit();
                    cb.wait_until_completed();
                }
            }
            let mut times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                for (a_exp, b_exp) in &experts {
                    let cb = queue.new_command_buffer();
                    let _ = ops::matmul::matmul_into_cb(registry, a_exp, b_exp, cb);
                    cb.commit();
                    cb.wait_until_completed();
                }
                times.push(start.elapsed());
            }
            let stats = Stats::from_durations(&times);
            let tflops_p50 = total_flops / (stats.p50 * 1e-6) / 1e12;
            println!(
                "  {:<25}  baseline   p50={:8.1}us  mean={:8.1}us  TFLOPS(p50)={:.2}",
                label, stats.p50, stats.mean, tflops_p50,
            );
        }

        // Grouped: single dispatch via dispatch_grouped_gemm
        {
            let b_3d = Array::from_bytes(
                device,
                unsafe {
                    std::slice::from_raw_parts(
                        b_stacked.metal_buffer().contents() as *const u8,
                        num_experts * k * *n * 2, // f16
                    )
                },
                vec![num_experts, k, *n],
                DType::Float16,
            );

            // Warmup
            for _ in 0..WARMUP_ITERS {
                let _ = ops::matmul::dispatch_grouped_gemm(
                    registry, &a_stacked, &b_3d, queue, expert_ms, k, *n,
                );
            }
            let mut times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let _ = ops::matmul::dispatch_grouped_gemm(
                    registry, &a_stacked, &b_3d, queue, expert_ms, k, *n,
                );
                times.push(start.elapsed());
            }
            let stats = Stats::from_durations(&times);
            let tflops_p50 = total_flops / (stats.p50 * 1e-6) / 1e12;
            println!(
                "  {:<25}  grouped    p50={:8.1}us  mean={:8.1}us  TFLOPS(p50)={:.2}",
                label, stats.p50, stats.mean, tflops_p50,
            );
        }

        // CB-batched: all expert GEMMs in ONE command buffer
        {
            let experts: Vec<_> = expert_ms
                .iter()
                .map(|&m_i| {
                    (
                        rand_array(device, &[m_i.max(1), k], 42),
                        rand_array(device, &[k, *n], 44),
                    )
                })
                .collect();

            // Warmup
            for _ in 0..WARMUP_ITERS {
                let cb = queue.new_command_buffer();
                for (a_exp, b_exp) in &experts {
                    let _ = ops::matmul::matmul_into_cb(registry, a_exp, b_exp, cb);
                }
                cb.commit();
                cb.wait_until_completed();
            }
            let mut times = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                for (a_exp, b_exp) in &experts {
                    let _ = ops::matmul::matmul_into_cb(registry, a_exp, b_exp, cb);
                }
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }
            let stats = Stats::from_durations(&times);
            let tflops_p50 = total_flops / (stats.p50 * 1e-6) / 1e12;
            println!(
                "  {:<25}  cb-batch   p50={:8.1}us  mean={:8.1}us  TFLOPS(p50)={:.2}",
                label, stats.p50, stats.mean, tflops_p50,
            );
        }

        println!();
    }
}

/// Benchmark a specific tile configuration, returning p50 latency in microseconds.
#[allow(clippy::too_many_arguments)]
fn bench_with_tile(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    a: &Array,
    b: &Array,
    m: usize,
    k: usize,
    n: usize,
    tile: &ops::matmul::TileConfig,
) -> f64 {
    let kernel_name = match (tile.variant, a.dtype()) {
        (ops::matmul::TileVariant::MlxArchMicro, DType::Float16) => "gemm_mlx_m16_f16",
        (ops::matmul::TileVariant::MlxArchSmall, DType::Float16) => "gemm_mlx_small_f16",
        (ops::matmul::TileVariant::MlxArch, DType::Float16) => "gemm_mlx_f16",
        _ => panic!("Unsupported tile variant for boundary bench"),
    };

    let constants = ops::matmul::matmul_align_constants(m, n, k, tile.bm, tile.bn, tile.bk);
    let pipeline = registry
        .get_pipeline_with_constants(kernel_name, a.dtype(), &constants)
        .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"));

    let c = Array::zeros(device, &[m, n], a.dtype());
    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_buf = make_u32_buf(
        device,
        ops::matmul::compute_swizzle_log(m, n, tile.bm, tile.bn),
    );

    let grid_x = ceil_div(n, tile.bn) as u64;
    let grid_y = ceil_div(m, tile.bm) as u64;
    let grid = MTLSize::new(grid_x, grid_y, 1);
    let tg = MTLSize::new(64, 1, 1);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
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
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
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
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    stats.p50
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    let k = 4096;
    let n_values = [14336, 4096, 2048, 1536, 768];
    let m_values = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

    println!("=== GEMM Benchmark (f16) ===");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!();

    for &n in &n_values {
        println!("[M, {}] @ [{}, {}]:", k, k, n);
        for &m in &m_values {
            let a = rand_array(device, &[m, k], 42);
            let b = rand_array(device, &[k, n], 44);
            bench_gemm(
                &registry,
                &queue,
                device,
                &a,
                &b,
                m,
                k,
                n,
                &format!("{}x{}", k, n),
            );
        }
        println!();
    }

    // Grouped GEMM benchmark (if BENCH_GROUPED != "0")
    let bench_grouped = std::env::var("BENCH_GROUPED").unwrap_or_else(|_| "1".to_string());
    if bench_grouped != "0" {
        println!("=== Grouped GEMM Benchmark (MoE simulation) ===");
        println!();
        bench_grouped_gemm(&registry, &queue, device);
    }

    // Grouped Split-K benchmark (same gate as grouped GEMM)
    if bench_grouped != "0" {
        println!("=== Grouped Split-K Benchmark (low-occupancy MoE) ===");
        println!();

        let gpu_cores = registry.device().tuning().gpu_cores;
        let splitk_k = 4096;

        // Scenarios where per-expert occupancy is very low (few tiles)
        let splitk_scenarios: Vec<(&str, Vec<usize>, usize)> = vec![
            ("4E×M=2 N=768", vec![2; 4], 768),
            ("4E×M=4 N=512", vec![4; 4], 512),
            ("8E×M=1 N=1024", vec![1; 8], 1024),
            ("8E×M=2 N=2048", vec![2; 8], 2048),
        ];

        for (label, expert_ms, n) in &splitk_scenarios {
            let total_m: usize = expert_ms.iter().sum();
            let num_experts = expert_ms.len();
            let total_flops = 2.0 * total_m as f64 * splitk_k as f64 * *n as f64;

            let a_stacked = rand_array(device, &[total_m, splitk_k], 42);
            let b_3d = {
                let b_flat = rand_array(device, &[num_experts, splitk_k * *n], 44);
                Array::from_bytes(
                    device,
                    unsafe {
                        std::slice::from_raw_parts(
                            b_flat.metal_buffer().contents() as *const u8,
                            num_experts * splitk_k * *n * 2,
                        )
                    },
                    vec![num_experts, splitk_k, *n],
                    DType::Float16,
                )
            };

            // Grouped (no split-K)
            {
                for _ in 0..WARMUP_ITERS {
                    let _ = ops::matmul::dispatch_grouped_gemm(
                        &registry, &a_stacked, &b_3d, &queue, expert_ms, splitk_k, *n,
                    );
                }
                let mut times = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let start = Instant::now();
                    let _ = ops::matmul::dispatch_grouped_gemm(
                        &registry, &a_stacked, &b_3d, &queue, expert_ms, splitk_k, *n,
                    );
                    times.push(start.elapsed());
                }
                let stats = Stats::from_durations(&times);
                let tflops_p50 = total_flops / (stats.p50 * 1e-6) / 1e12;
                println!(
                    "  {:<25}  grouped    p50={:8.1}us  mean={:8.1}us  TFLOPS(p50)={:.2}",
                    label, stats.p50, stats.mean, tflops_p50,
                );
            }

            // Grouped Split-K
            {
                for _ in 0..WARMUP_ITERS {
                    let _ = ops::matmul::dispatch_grouped_splitk(
                        &registry, &queue, &a_stacked, &b_3d, expert_ms, splitk_k, *n, gpu_cores,
                    );
                }
                let mut times = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let start = Instant::now();
                    let _ = ops::matmul::dispatch_grouped_splitk(
                        &registry, &queue, &a_stacked, &b_3d, expert_ms, splitk_k, *n, gpu_cores,
                    );
                    times.push(start.elapsed());
                }
                let stats = Stats::from_durations(&times);
                let tflops_p50 = total_flops / (stats.p50 * 1e-6) / 1e12;
                println!(
                    "  {:<25}  grp-splitk p50={:8.1}us  mean={:8.1}us  TFLOPS(p50)={:.2}",
                    label, stats.p50, stats.mean, tflops_p50,
                );
            }

            println!();
        }
    }

    // GEMV vs GEMM comparison for low-M (set BENCH_GEMV=0 to skip)
    let bench_gemv = std::env::var("BENCH_GEMV").unwrap_or_else(|_| "1".to_string());
    if bench_gemv != "0" {
        println!("=== GEMV vs GEMM Comparison (f16) ===");
        println!();

        let gemv_k = 4096;
        for &gemv_n in &[2048usize, 1536, 768] {
            println!(
                "[M, {}] @ [{}, {}] — GEMV(M×row) vs GEMM:",
                gemv_k, gemv_k, gemv_n
            );
            for &gemv_m in &[2usize, 4, 8, 16] {
                // GEMM timing (this is what currently happens for M>=5)
                let a = rand_array(device, &[gemv_m, gemv_k], 42);
                let b = rand_array(device, &[gemv_k, gemv_n], 44);

                // GEMM
                for _ in 0..WARMUP_ITERS {
                    let cb = queue.new_command_buffer();
                    let _ = ops::matmul::matmul_into_cb(&registry, &a, &b, cb);
                    cb.commit();
                    cb.wait_until_completed();
                }
                let mut gemm_times = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let start = Instant::now();
                    let cb = queue.new_command_buffer();
                    let _ = ops::matmul::matmul_into_cb(&registry, &a, &b, cb);
                    cb.commit();
                    cb.wait_until_completed();
                    gemm_times.push(start.elapsed());
                }
                let gemm_stats = Stats::from_durations(&gemm_times);

                // GEMV: M individual row×matrix multiplications in one CB
                let a_rows: Vec<_> = (0..gemv_m)
                    .map(|i| rand_array(device, &[1, gemv_k], 42 + i as u64))
                    .collect();

                for _ in 0..WARMUP_ITERS {
                    let cb = queue.new_command_buffer();
                    for a_row in &a_rows {
                        let _ = ops::matmul::matmul_into_cb(&registry, a_row, &b, cb);
                    }
                    cb.commit();
                    cb.wait_until_completed();
                }
                let mut gemv_times = Vec::with_capacity(BENCH_ITERS);
                for _ in 0..BENCH_ITERS {
                    let start = Instant::now();
                    let cb = queue.new_command_buffer();
                    for a_row in &a_rows {
                        let _ = ops::matmul::matmul_into_cb(&registry, a_row, &b, cb);
                    }
                    cb.commit();
                    cb.wait_until_completed();
                    gemv_times.push(start.elapsed());
                }
                let gemv_stats = Stats::from_durations(&gemv_times);

                let flops = 2.0 * gemv_m as f64 * gemv_k as f64 * gemv_n as f64;
                let gemm_tflops = flops / (gemm_stats.p50 * 1e-6) / 1e12;
                let gemv_tflops = flops / (gemv_stats.p50 * 1e-6) / 1e12;
                let winner = if gemv_stats.p50 < gemm_stats.p50 {
                    "GEMV"
                } else {
                    "GEMM"
                };

                println!(
                    "  M={:3}  GEMM p50={:8.1}us ({:.2}T)  GEMV×{} p50={:8.1}us ({:.2}T)  winner={}",
                    gemv_m, gemm_stats.p50, gemm_tflops, gemv_m, gemv_stats.p50, gemv_tflops, winner,
                );
            }
            println!();
        }
    }

    // BM Tile Boundary Sweep (set BENCH_BOUNDARY=0 to skip)
    let bench_boundary = std::env::var("BENCH_BOUNDARY").unwrap_or_else(|_| "1".to_string());
    if bench_boundary != "0" {
        println!("=== BM Tile Boundary Sweep (f16) ===");
        println!("Comparing BM=16/BN=32 vs BM=32/BN=32 for each M");
        println!();

        let boundary_k = 4096;
        for &n in &[14336usize, 4096, 2048, 1536, 768] {
            println!("N={}:", n);
            for &m in &[5usize, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32] {
                let a = rand_array(device, &[m, boundary_k], 42);
                let b = rand_array(device, &[boundary_k, n], 44);

                // BM=16, BN=32 (MlxArchMicro)
                let tile_micro = ops::matmul::TileConfig {
                    bm: 16,
                    bn: 32,
                    bk: 16,
                    variant: ops::matmul::TileVariant::MlxArchMicro,
                };
                let t_micro = bench_with_tile(
                    &registry,
                    &queue,
                    device,
                    &a,
                    &b,
                    m,
                    boundary_k,
                    n,
                    &tile_micro,
                );

                // BM=32, BN=32 (MlxArchSmall)
                let tile_small = ops::matmul::TileConfig {
                    bm: 32,
                    bn: 32,
                    bk: 16,
                    variant: ops::matmul::TileVariant::MlxArchSmall,
                };
                let t_small = bench_with_tile(
                    &registry,
                    &queue,
                    device,
                    &a,
                    &b,
                    m,
                    boundary_k,
                    n,
                    &tile_small,
                );

                let flops = 2.0 * m as f64 * boundary_k as f64 * n as f64;
                let tf_micro = flops / (t_micro * 1e-6) / 1e12;
                let tf_small = flops / (t_small * 1e-6) / 1e12;
                let winner = if t_micro < t_small { "BM16" } else { "BM32" };
                let pct = ((t_small as f64 / t_micro as f64) - 1.0) * 100.0;

                println!(
                    "  M={:3}  BM16={:8.1}us ({:.2}T)  BM32={:8.1}us ({:.2}T)  winner={:<4}  diff={:+.1}%",
                    m, t_micro, tf_micro, t_small, tf_small, winner, pct,
                );
            }
            println!();
        }
    }

    println!("Done.");
}
