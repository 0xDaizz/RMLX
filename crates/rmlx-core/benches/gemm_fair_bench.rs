//! RMLX vs MLX fair GEMM comparison benchmark.
//!
//! Two dispatch modes:
//!   1. **Sync**: 1 dispatch per CB, `wait_until_completed` every time (unfair overhead)
//!   2. **Pipelined**: 32 separate CBs each with 1 dispatch to its own output
//!      buffer (avoids WAW hazards), committed without waiting (simulating
//!      32-layer transformer), `wait_until_completed` on last CB only,
//!      amortized time = total/32
//!
//! Kernel selection per M (matching optimal dispatch):
//!   - M=1..128:  gemm_mlx_m16_f16    (BM=16, BN=32, BK=16, 64 threads)
//!   - M=256:     gemm_mlx_small_f16  (BM=32, BN=32, BK=16, 64 threads)
//!   - M=512:     gemm_nax_64x128_f16 (BM=64, BN=128, BK=32, 256 threads)
//!
//! N=3584, K=3584, f16, 5 warmup + 20 bench, largest M first (thermal).
//!
//! Usage: cargo bench -p rmlx-core --bench gemm_fair_bench

use std::time::Instant;

use half::f16;
use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const PIPELINE_N: usize = 32;
const N: usize = 3584;
const K: usize = 3584;
const M_VALUES: &[usize] = &[1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512];

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

fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Kernel selection
// ---------------------------------------------------------------------------

struct KernelSpec {
    name: &'static str,
    bm: usize,
    bn: usize,
    bk: usize,
    threads: u64,
}

fn select_kernel_for_m(m: usize) -> KernelSpec {
    if m >= 512 {
        KernelSpec {
            name: "gemm_nax_64x128_f16",
            bm: 64,
            bn: 128,
            bk: 32,
            threads: 256,
        }
    } else if m >= 256 {
        KernelSpec {
            name: "gemm_mlx_small_f16",
            bm: 32,
            bn: 32,
            bk: 16,
            threads: 64,
        }
    } else {
        KernelSpec {
            name: "gemm_mlx_m16_f16",
            bm: 16,
            bn: 32,
            bk: 16,
            threads: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM buffers
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
    fn new(device: &metal::Device, m: usize, n: usize, k: usize) -> Self {
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);
        let m_buf = make_u32_buf(device, m as u32);
        let n_buf = make_u32_buf(device, n as u32);
        let k_buf = make_u32_buf(device, k as u32);
        let bsa_buf = make_u32_buf(device, (m * k) as u32);
        let bsb_buf = make_u32_buf(device, (k * n) as u32);
        let bsc_buf = make_u32_buf(device, (m * n) as u32);
        // Force swizzle_log=0 to avoid OOB writes from mlx_swizzle_tg().
        // See: compute_swizzle_log returns values that expand new_y beyond tiles_m,
        // causing GPU PageFault with separate output buffers.
        let swizzle_buf = make_u32_buf(device, 0u32);
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
// Sync benchmark: 1 dispatch per CB
// ---------------------------------------------------------------------------

fn bench_sync(
    device: &metal::Device,
    pipeline: &metal::ComputePipelineState,
    bufs: &GemmBuffers,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    // Fresh queue to avoid poisoning from prior errors
    let queue = device.new_command_queue();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        bufs.encode(enc, pipeline, grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
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
// Pipelined benchmark: 32 separate CBs committed without waiting, sync on last
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_pipelined(
    device: &metal::Device,
    pipeline: &metal::ComputePipelineState,
    bufs: &GemmBuffers,
    m: usize,
    n: usize,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    // Fresh queue to avoid poisoning from prior errors
    let queue = device.new_command_queue();
    // Allocate separate output buffers to avoid WAW hazards across CBs
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64; // f16 = 2 bytes
    let out_bufs: Vec<metal::Buffer> = (0..PIPELINE_N)
        .map(|_| device.new_buffer(out_size, opts))
        .collect();

    // Warmup: 32 separate CBs, each with 1 dispatch to its own output buffer
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(bufs.a.metal_buffer()), 0);
                enc.set_buffer(1, Some(bufs.b.metal_buffer()), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_buffer(3, Some(&bufs.m_buf), 0);
                enc.set_buffer(4, Some(&bufs.n_buf), 0);
                enc.set_buffer(5, Some(&bufs.k_buf), 0);
                enc.set_buffer(6, Some(&bufs.bsa_buf), 0);
                enc.set_buffer(7, Some(&bufs.bsb_buf), 0);
                enc.set_buffer(8, Some(&bufs.bsc_buf), 0);
                enc.set_buffer(9, Some(&bufs.swizzle_buf), 0);
                enc.set_buffer(10, Some(&bufs.residual_buf), 0);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().wait_until_completed();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(bufs.a.metal_buffer()), 0);
                enc.set_buffer(1, Some(bufs.b.metal_buffer()), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_buffer(3, Some(&bufs.m_buf), 0);
                enc.set_buffer(4, Some(&bufs.n_buf), 0);
                enc.set_buffer(5, Some(&bufs.k_buf), 0);
                enc.set_buffer(6, Some(&bufs.bsa_buf), 0);
                enc.set_buffer(7, Some(&bufs.bsb_buf), 0);
                enc.set_buffer(8, Some(&bufs.bsc_buf), 0);
                enc.set_buffer(9, Some(&bufs.swizzle_buf), 0);
                enc.set_buffer(10, Some(&bufs.residual_buf), 0);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().wait_until_completed();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64); // amortized per dispatch
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

    println!("=== RMLX GEMM Fair Benchmark (Sync vs Pipelined) ===");
    println!(
        "N={}, K={}, dtype=f16, Warmup={}, Bench={} iters",
        N, K, WARMUP_ITERS, BENCH_ITERS
    );
    println!(
        "Pipeline: {} dispatches per CB (simulating 32-layer transformer)",
        PIPELINE_N
    );
    println!();

    // Collect results (run largest M first for thermal fairness)
    let mut results: Vec<(usize, &str, f64, f64, f64, f64, f64)> = Vec::new();

    for &m in M_VALUES.iter().rev() {
        let spec = select_kernel_for_m(m);
        let constants = ops::matmul::matmul_align_constants(m, N, K, spec.bm, spec.bn, spec.bk);
        let pipeline = registry
            .get_pipeline_with_constants(spec.name, DType::Float16, &constants)
            .unwrap_or_else(|e| panic!("Failed to get pipeline for {}: {e}", spec.name));

        let bufs = GemmBuffers::new(device, m, N, K);

        let grid_x = N.div_ceil(spec.bn);
        let grid_y = m.div_ceil(spec.bm);
        let grid = MTLSize::new(grid_x as u64, grid_y as u64, 1);
        let tg = MTLSize::new(spec.threads, 1, 1);

        println!(
            "  [DEBUG] M={} kernel={} grid=({},{},{}) tg=({},{},{}) bm={} bn={} bk={}",
            m,
            spec.name,
            grid.width,
            grid.height,
            grid.depth,
            tg.width,
            tg.height,
            tg.depth,
            spec.bm,
            spec.bn,
            spec.bk
        );

        let sync_us = bench_sync(device, &pipeline, &bufs, grid, tg);
        let pipe_us = bench_pipelined(device, &pipeline, &bufs, m, N, grid, tg);

        let sync_t = tflops(m, N, K, sync_us);
        let pipe_t = tflops(m, N, K, pipe_us);
        let speedup = if pipe_us > 0.0 {
            sync_us / pipe_us
        } else {
            0.0
        };

        println!(
            "  M={:>4}: {} sync={:.1}us ({:.2}T) pipe={:.1}us ({:.2}T) speedup={:.2}x",
            m, spec.name, sync_us, sync_t, pipe_us, pipe_t, speedup
        );

        results.push((m, spec.name, sync_us, sync_t, pipe_us, pipe_t, speedup));
    }

    // Sort ascending by M for display
    results.sort_by_key(|r| r.0);

    println!();
    println!("=== Results ===");
    println!();
    println!(
        "| {:>5} | {:>22} | {:>10} | {:>8} | {:>10} | {:>8} | {:>7} |",
        "M", "Kernel", "Sync (us)", "Sync T", "Pipe (us)", "Pipe T", "Speedup"
    );
    println!(
        "|{:-<7}|{:-<24}|{:-<12}|{:-<10}|{:-<12}|{:-<10}|{:-<9}|",
        "", "", "", "", "", "", ""
    );

    for &(m, kernel, sync_us, sync_t, pipe_us, pipe_t, speedup) in &results {
        println!(
            "| {:>5} | {:>22} | {:>10.1} | {:>8.2} | {:>10.1} | {:>8.2} | {:>6.2}x |",
            m, kernel, sync_us, sync_t, pipe_us, pipe_t, speedup
        );
    }

    println!();
    println!("* vs MLX column: fill in after running mlx_gemm_fair_bench.py");
    println!();

    // Summary
    if let (Some(first), Some(last)) = (
        results.iter().find(|r| r.0 == 1),
        results.iter().find(|r| r.0 == 512),
    ) {
        println!("=== Summary ===");
        println!(
            "M=1   sync={:.1}us pipe={:.1}us speedup={:.1}x",
            first.2, first.4, first.6
        );
        println!(
            "M=512 sync={:.1}us pipe={:.1}us speedup={:.1}x",
            last.2, last.4, last.6
        );
    }

    println!();
    println!("Done.");
}
