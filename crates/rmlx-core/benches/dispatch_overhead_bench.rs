//! ⚠️ NON-PRODUCTION PATH — dispatch overhead measurement via batch amortization.
//! Diagnostic only; measures CB creation/commit/wait costs, not production throughput.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Dispatch overhead analysis benchmark.
//! Separates CPU dispatch overhead from GPU kernel execution time using
//! batch amortization (no GPU timestamps needed).
//!
//! Method:
//!   1. Single dispatch in one CB → wall time = CB_overhead + GPU_time
//!   2. 10 dispatches in one CB  → wall time = CB_overhead + 10 × GPU_time
//!   3. GPU_time_per_dispatch = (time_10 - time_1) / 9
//!   4. CB_overhead = time_1 - GPU_time_per_dispatch
//!
//! Usage: cargo bench -p rmlx-core --bench dispatch_overhead_bench

use std::time::Instant;

use half::f16;
use metal::MTLSize;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP: usize = 5;
const ITERS: usize = 20;
const BATCH_N: usize = 10;

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

fn p50(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(times, 50.0)
}

/// Encode one GEMM dispatch into the given compute command encoder.
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    a: &Array,
    b: &Array,
    c: &Array,
    m_buf: &metal::Buffer,
    n_buf: &metal::Buffer,
    k_buf: &metal::Buffer,
    bsa_buf: &metal::Buffer,
    bsb_buf: &metal::Buffer,
    bsc_buf: &metal::Buffer,
    swizzle_buf: &metal::Buffer,
    grid: MTLSize,
    tg: MTLSize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), 0);
    enc.set_buffer(1, Some(b.metal_buffer()), 0);
    enc.set_buffer(2, Some(c.metal_buffer()), 0);
    enc.set_buffer(3, Some(m_buf), 0);
    enc.set_buffer(4, Some(n_buf), 0);
    enc.set_buffer(5, Some(k_buf), 0);
    enc.set_buffer(6, Some(bsa_buf), 0);
    enc.set_buffer(7, Some(bsb_buf), 0);
    enc.set_buffer(8, Some(bsc_buf), 0);
    enc.set_buffer(9, Some(swizzle_buf), 0);
    enc.dispatch_thread_groups(grid, tg);
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    let n = 3584usize;
    let k = 3584usize;
    let m_values: &[usize] = &[1, 4, 8, 16, 32, 64, 128, 256, 512];

    // Kernel config: MLX Micro (BM=16, BN=32, BK=16, 64 threads)
    let bm = 16usize;
    let bn = 32usize;
    let bk = 16usize;
    let threads = 64u64;
    let kernel_name = "gemm_mlx_m16_f16";

    println!("=== Dispatch Overhead Analysis (Batch Amortization) ===");
    println!(
        "Kernel: {} (BM={}, BN={}, BK={}, {} threads)",
        kernel_name, bm, bn, bk, threads
    );
    println!(
        "N={}, K={}, Warmup={}, Iters={}, Batch={}",
        n, k, WARMUP, ITERS, BATCH_N
    );
    println!();
    println!(
        "Method: GPU_time = (wall_{}cb - wall_1cb) / {}",
        BATCH_N,
        BATCH_N - 1
    );
    println!("        CB_overhead = wall_1cb - GPU_time");
    println!();

    // ---- Step 1: Bare CB overhead (empty command buffer) ----
    println!("--- Bare Command Buffer Overhead ---");
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer_with_unretained_references();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut bare_times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        cb.commit();
        cb.wait_until_completed();
        bare_times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let bare_p50 = p50(&mut bare_times);
    println!("Bare CB p50: {:.1} us", bare_p50);
    println!();

    // ---- Step 2: Per-M analysis ----
    println!(
        "| {:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>10} |",
        "M", "Wall 1 (us)", "Wall 10(us)", "GPU (us)", "CB OH(us)", "CB OH %", "TFLOPS"
    );
    println!(
        "|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<10}|{:-<12}|",
        "", "", "", "", "", "", ""
    );

    // Run largest M first to avoid thermal bias on important cases
    let mut results: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for &m in m_values.iter().rev() {
        let constants = ops::matmul::matmul_align_constants(m, n, k, bm, bn, bk);
        let pipeline = registry
            .get_pipeline_with_constants(kernel_name, DType::Float16, &constants)
            .expect("Pipeline failed");

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

        let grid_x = n.div_ceil(bn);
        let grid_y = m.div_ceil(bm);
        let grid = MTLSize::new(grid_x as u64, grid_y as u64, 1);
        let tg = MTLSize::new(threads, 1, 1);

        // --- Warmup (single dispatch) ---
        for _ in 0..WARMUP {
            let cb = queue.new_command_buffer_with_unretained_references();
            let enc = cb.new_compute_command_encoder();
            encode_gemm(
                enc,
                &pipeline,
                &a,
                &b,
                &c,
                &m_buf,
                &n_buf,
                &k_buf,
                &bsa_buf,
                &bsb_buf,
                &bsc_buf,
                &swizzle_buf,
                grid,
                tg,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // --- Measure: single dispatch (1 CB, 1 kernel) ---
        let mut wall_1 = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let enc = cb.new_compute_command_encoder();
            encode_gemm(
                enc,
                &pipeline,
                &a,
                &b,
                &c,
                &m_buf,
                &n_buf,
                &k_buf,
                &bsa_buf,
                &bsb_buf,
                &bsc_buf,
                &swizzle_buf,
                grid,
                tg,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            wall_1.push(start.elapsed().as_secs_f64() * 1e6);
        }

        // --- Warmup (batch dispatch) ---
        for _ in 0..WARMUP {
            let cb = queue.new_command_buffer_with_unretained_references();
            for _ in 0..BATCH_N {
                let enc = cb.new_compute_command_encoder();
                encode_gemm(
                    enc,
                    &pipeline,
                    &a,
                    &b,
                    &c,
                    &m_buf,
                    &n_buf,
                    &k_buf,
                    &bsa_buf,
                    &bsb_buf,
                    &bsc_buf,
                    &swizzle_buf,
                    grid,
                    tg,
                );
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
        }

        // --- Measure: batch dispatch (1 CB, BATCH_N kernels) ---
        let mut wall_batch = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            for _ in 0..BATCH_N {
                let enc = cb.new_compute_command_encoder();
                encode_gemm(
                    enc,
                    &pipeline,
                    &a,
                    &b,
                    &c,
                    &m_buf,
                    &n_buf,
                    &k_buf,
                    &bsa_buf,
                    &bsb_buf,
                    &bsc_buf,
                    &swizzle_buf,
                    grid,
                    tg,
                );
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
            wall_batch.push(start.elapsed().as_secs_f64() * 1e6);
        }

        let w1_p50 = p50(&mut wall_1);
        let wb_p50 = p50(&mut wall_batch);

        // GPU_time = (wall_batch - wall_1) / (BATCH_N - 1)
        let gpu_time = (wb_p50 - w1_p50) / (BATCH_N - 1) as f64;
        // CB_overhead = wall_1 - GPU_time
        let cb_overhead = w1_p50 - gpu_time;

        results.push((m, w1_p50, wb_p50, gpu_time, cb_overhead));
    }

    // Sort ascending by M for display
    results.sort_by_key(|(m, _, _, _, _)| *m);

    for &(m, w1, wb, gpu, cb_oh) in &results {
        let cb_pct = if w1 > 0.0 { cb_oh / w1 * 100.0 } else { 0.0 };
        let tflops = if gpu > 0.0 {
            2.0 * m as f64 * n as f64 * k as f64 / (gpu * 1e-6) / 1e12
        } else {
            0.0
        };
        println!(
            "| {:>5} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>7.1}% | {:>10.2} |",
            m, w1, wb, gpu, cb_oh, cb_pct, tflops
        );
    }

    // ---- Step 3: Encoding-only overhead ----
    println!();
    println!("--- Encoding-only Overhead (encode + end, no GPU work) ---");

    let constants = ops::matmul::matmul_align_constants(128, n, k, bm, bn, bk);
    let pipeline = registry
        .get_pipeline_with_constants(kernel_name, DType::Float16, &constants)
        .expect("Pipeline failed");
    let a = rand_f16_array(device, &[128, k], 42);
    let b = rand_f16_array(device, &[k, n], 99);
    let c = Array::zeros(device, &[128, n], DType::Float16);
    let m_buf = make_u32_buf(device, 128);
    let n_buf = make_u32_buf(device, n as u32);
    let k_buf = make_u32_buf(device, k as u32);
    let bsa_buf = make_u32_buf(device, (128 * k) as u32);
    let bsb_buf = make_u32_buf(device, (k * n) as u32);
    let bsc_buf = make_u32_buf(device, (128 * n) as u32);
    let swizzle_buf = make_u32_buf(device, 0);
    let grid = MTLSize::new(n.div_ceil(bn) as u64, 128usize.div_ceil(bm) as u64, 1);
    let tg = MTLSize::new(threads, 1, 1);

    // Measure encoding time separately: create CB + encode + end, then commit+wait
    // We time only the encoding portion.
    let mut encode_times = Vec::with_capacity(100);
    for _ in 0..100 {
        let cb = queue.new_command_buffer_with_unretained_references();
        let start = Instant::now();
        let enc = cb.new_compute_command_encoder();
        encode_gemm(
            enc,
            &pipeline,
            &a,
            &b,
            &c,
            &m_buf,
            &n_buf,
            &k_buf,
            &bsa_buf,
            &bsb_buf,
            &bsc_buf,
            &swizzle_buf,
            grid,
            tg,
        );
        enc.end_encoding();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        encode_times.push(elapsed);
        // Must commit to avoid leaking the CB
        cb.commit();
        cb.wait_until_completed();
    }
    let encode_p50 = p50(&mut encode_times);
    println!(
        "Encode-only p50 (10 set_buffer + dispatch + end_encoding): {:.1} us",
        encode_p50
    );

    // ---- Step 4: Summary ----
    println!();
    println!("--- Summary ---");
    println!(
        "Bare CB overhead (empty commit+wait):     {:.1} us",
        bare_p50
    );
    println!(
        "Encode overhead (pipeline+buffers+dispatch): {:.1} us",
        encode_p50
    );
    if let Some(&(_, _, _, _, cb_oh_m1)) = results.iter().find(|r| r.0 == 1) {
        println!(
            "Total CB overhead at M=1 (via amortization): {:.1} us",
            cb_oh_m1
        );
    }
    if let Some(&(_, w1, _, gpu, cb_oh)) = results.iter().find(|r| r.0 == 128) {
        println!(
            "M=128: wall={:.1}us, gpu={:.1}us, overhead={:.1}us ({:.1}%)",
            w1,
            gpu,
            cb_oh,
            cb_oh / w1 * 100.0
        );
    }
}
