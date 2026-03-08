//! Standalone GEMM kernel benchmark — per-op latency and TFLOPS.
//!
//! Tests matmul [M, 4096] @ [4096, K] for K in {4096, 14336}
//! and M in {128, 256, 512, 1024, 2048}, using f16 dtype.
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
    let tile = ops::matmul::select_tile_config(m, n, k);
    let kernel_name = match (tile.variant, a.dtype()) {
        (ops::matmul::TileVariant::Full, DType::Float16) => "gemm_tiled_f16",
        (ops::matmul::TileVariant::Full, DType::Float32) => "gemm_tiled_f32",
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

    let pipeline = registry
        .get_pipeline(kernel_name, a.dtype())
        .unwrap_or_else(|e| panic!("Failed to get pipeline for {kernel_name}: {e}"));

    // Pre-allocate output and constant buffers ONCE
    let c = Array::zeros(device, &[m, n], a.dtype());
    let m_buf = make_u32_buf(device, m as u32);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (m * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (m * n) as u32);
    let swizzle_buf = make_u32_buf(device, ops::matmul::compute_swizzle_log(m, tile.bm));

    let grid_x = ceil_div(n, tile.bn) as u64;
    let grid_y = ceil_div(m, tile.bm) as u64;
    let grid = MTLSize::new(grid_x, grid_y, 1);
    let tg = MTLSize::new(256, 1, 1);

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

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    println!("=== GEMM Benchmark (f16) ===");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!();

    // --- [M, 4096] @ [4096, 4096] ---
    println!("[M, 4096] @ [4096, 4096]:");
    for m in [128, 256, 512, 1024, 2048] {
        let a = rand_array(device, &[m, 4096], 42);
        let b = rand_array(device, &[4096, 4096], 43);
        bench_gemm(&registry, &queue, device, &a, &b, m, 4096, 4096, "4096x4096");
    }
    println!();

    // --- [M, 4096] @ [4096, 14336] ---
    println!("[M, 4096] @ [4096, 14336]:");
    for m in [128, 256, 512, 1024, 2048] {
        let a = rand_array(device, &[m, 4096], 42);
        let b = rand_array(device, &[4096, 14336], 44);
        bench_gemm(&registry, &queue, device, &a, &b, m, 4096, 14336, "4096x14336");
    }
    println!();
    println!("Done.");
}
