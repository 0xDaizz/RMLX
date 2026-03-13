//! ⚠️ NON-PRODUCTION PATH — single kernel isolation for dispatch overhead investigation.
//! Direct kernel encoding, development only. Bypasses matmul() dispatch.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Single-kernel isolation benchmark.
//! Tests only MLX Micro to find where the ~320us overhead in nax_lowm_bench comes from.
//!
//! Hypothesis: pipeline switching between 10 different kernels in nax_lowm_bench
//! causes GPU state flush, inflating per-dispatch latency.
//!
//! Three modes per M value:
//!   A: Standard (5 warmup + 20 bench, new CB each iteration)
//!   B: Pre-warmed (100 warmup + 20 bench, new CB each iteration)
//!   C: Tight loop (5 warmup + 200 rapid-fire dispatches, p50)
//!
//! Usage: cargo bench -p rmlx-core --bench single_kernel_bench

use std::time::Instant;

use half::f16;
use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

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

fn p50(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(times, 50.0)
}

// ---------------------------------------------------------------------------
// All buffers needed for a single GEMM dispatch
// ---------------------------------------------------------------------------

struct GemmBuffers {
    a: Array,
    b: Array,
    c: Array,
    m_buf: metal::Buffer,
    n_buf: metal::Buffer,
    k_buf: metal::Buffer,
    bsa_buf: metal::Buffer,
    bsb_buf: metal::Buffer,
    bsc_buf: metal::Buffer,
    swizzle_buf: metal::Buffer,
    residual_buf: metal::Buffer,
}

impl GemmBuffers {
    fn new(device: &metal::Device, m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Self {
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);
        let m_buf = make_u32_buf(device, m as u32);
        let n_buf = make_u32_buf(device, n as u32);
        let k_buf = make_u32_buf(device, k as u32);
        let bsa_buf = make_u32_buf(device, (m * k) as u32);
        let bsb_buf = make_u32_buf(device, (k * n) as u32);
        let bsc_buf = make_u32_buf(device, (m * n) as u32);
        let swizzle_log = ops::matmul::compute_swizzle_log(m, n, bm, bn);
        let swizzle_buf = make_u32_buf(device, swizzle_log);
        let residual_buf = make_u32_buf(device, 0u32);
        Self {
            a,
            b,
            c,
            m_buf,
            n_buf,
            k_buf,
            bsa_buf,
            bsb_buf,
            bsc_buf,
            swizzle_buf,
            residual_buf,
        }
    }

    /// Encode one GEMM dispatch: sets all 11 buffers (index 0-10) exactly matching
    /// nax_lowm_bench.
    fn encode(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        pipeline: &metal::ComputePipelineState,
        grid: MTLSize,
        tg: MTLSize,
    ) {
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(self.a.metal_buffer()), 0);
        enc.set_buffer(1, Some(self.b.metal_buffer()), 0);
        enc.set_buffer(2, Some(self.c.metal_buffer()), 0);
        enc.set_buffer(3, Some(&self.m_buf), 0);
        enc.set_buffer(4, Some(&self.n_buf), 0);
        enc.set_buffer(5, Some(&self.k_buf), 0);
        enc.set_buffer(6, Some(&self.bsa_buf), 0);
        enc.set_buffer(7, Some(&self.bsb_buf), 0);
        enc.set_buffer(8, Some(&self.bsc_buf), 0);
        enc.set_buffer(9, Some(&self.swizzle_buf), 0);
        enc.set_buffer(10, Some(&self.residual_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Run `warmup` warmup iterations then `iters` measured iterations.
/// Each iteration creates a new CB, encodes one dispatch, commits, and waits.
/// Returns p50 latency in microseconds.
fn bench_standard(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    bufs: &GemmBuffers,
    grid: MTLSize,
    tg: MTLSize,
    warmup: usize,
    iters: usize,
) -> f64 {
    // Warmup
    for _ in 0..warmup {
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        bufs.encode(enc, pipeline, grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        bufs.encode(enc, pipeline, grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }

    p50(&mut times)
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
    let bm = 16usize;
    let bn = 32usize;
    let bk = 16usize;
    let threads = 64u64;
    let kernel_name = "gemm_mlx_m16_f16";
    let m_values: &[usize] = &[1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512];

    println!("=== Single-Kernel Isolation Benchmark ===");
    println!(
        "Kernel: {} only (BM={}, BN={}, BK={}, {} threads)",
        kernel_name, bm, bn, bk, threads
    );
    println!("N={}, K={}", n, k);
    println!("Mode A: 5 warmup + 20 bench (standard, matches nax_lowm_bench)");
    println!("Mode B: 100 warmup + 20 bench (pre-warmed)");
    println!("Mode C: 5 warmup + 200 bench (tight loop, p50)");
    println!();

    println!(
        "| {:>5} | {:>12} | {:>12} | {:>12} | {:>8} |",
        "M", "Mode A (us)", "Mode B (us)", "Mode C (us)", "TFLOPS-A"
    );
    println!("|-------|--------------|--------------|--------------|----------|");

    // Collect results (may be out of order since we run largest M first)
    let mut results: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    // Run largest M first to avoid thermal throttling on important cases
    for &m in m_values.iter().rev() {
        let constants = ops::matmul::matmul_align_constants(m, n, k, bm, bn, bk);
        let pipeline = registry
            .get_pipeline_with_constants(kernel_name, DType::Float16, &constants)
            .expect("Failed to create pipeline");

        let bufs = GemmBuffers::new(device, m, n, k, bm, bn);

        let grid_x = n.div_ceil(bn);
        let grid_y = m.div_ceil(bm);
        let grid = MTLSize::new(grid_x as u64, grid_y as u64, 1);
        let tg = MTLSize::new(threads, 1, 1);

        // Mode A: standard (5 warmup + 20 bench) — matches nax_lowm_bench
        let mode_a = bench_standard(&queue, &pipeline, &bufs, grid, tg, 5, 20);

        // Mode B: pre-warmed (100 warmup + 20 bench)
        let mode_b = bench_standard(&queue, &pipeline, &bufs, grid, tg, 100, 20);

        // Mode C: tight loop (5 warmup + 200 bench)
        let mode_c = bench_standard(&queue, &pipeline, &bufs, grid, tg, 5, 200);

        let tflops_a = 2.0 * m as f64 * n as f64 * k as f64 / (mode_a * 1e-6) / 1e12;

        println!(
            "  M={}: A={:.1}us B={:.1}us C={:.1}us ({:.2}T)",
            m, mode_a, mode_b, mode_c, tflops_a
        );

        results.push((m, mode_a, mode_b, mode_c, tflops_a));
    }

    // Sort ascending by M for final table
    results.sort_by_key(|r| r.0);

    println!();
    println!("=== Results (sorted by M) ===");
    println!();
    println!(
        "| {:>5} | {:>12} | {:>12} | {:>12} | {:>8} |",
        "M", "Mode A (us)", "Mode B (us)", "Mode C (us)", "TFLOPS-A"
    );
    println!("|-------|--------------|--------------|--------------|----------|");

    for &(m, a, b, c, tf) in &results {
        println!(
            "| {:>5} | {:>12.1} | {:>12.1} | {:>12.1} | {:>8.2} |",
            m, a, b, c, tf
        );
    }

    // Summary analysis
    println!();
    println!("=== Analysis ===");
    if let (Some(m1), Some(m128)) = (
        results.iter().find(|r| r.0 == 1),
        results.iter().find(|r| r.0 == 128),
    ) {
        println!("M=1 vs M=128 ratio (Mode A): {:.1}x", m1.1 / m128.1);
        println!(
            "M=1 Mode A vs Mode C: {:.1}us vs {:.1}us (delta={:.1}us)",
            m1.1,
            m1.3,
            m1.1 - m1.3
        );
        println!(
            "M=128 Mode A vs Mode C: {:.1}us vs {:.1}us (delta={:.1}us)",
            m128.1,
            m128.3,
            m128.1 - m128.3
        );
    }
    println!();
    println!("If Mode A >> Mode C at M=1: pipeline cold-start or CB overhead dominates.");
    println!(
        "If Mode A == Mode C at M=1: the ~335us in nax_lowm_bench is from pipeline switching."
    );
    println!("If Mode B < Mode A: warmup count matters (GPU caches / TLB).");
}
