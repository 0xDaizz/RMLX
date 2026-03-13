//! ⚠️ NON-PRODUCTION PATH — GPU timestamp profiling for dispatch overhead analysis.
//! Diagnostic only; measures CB creation/commit/wait costs via GPUStartTime/GPUEndTime.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! GEMM dispatch overhead benchmark using Metal GPU timestamps.
//!
//! Separates wall time into GPU time (actual kernel execution) and
//! host overhead (CB create + encode + commit + wait roundtrip).
//!
//! Uses GPUStartTime/GPUEndTime for accurate GPU-side measurement,
//! unlike dispatch_overhead_bench which uses batch amortization.
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench dispatch_gpu_time_bench

#![allow(unexpected_cfgs)]

use std::time::Instant;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
    MTLComputeCommandEncoder as _, MTLDevice as _,
};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::types::{MtlBuffer, MtlPipeline};
use rmlx_metal::{MTLResourceOptions, MTLSize, autoreleasepool};
use std::ptr::NonNull;

const N: usize = 3584;
const K: usize = 3584;
const BM: usize = 16;
const BN: usize = 32;
const BK: usize = 16;
const THREADS: u64 = 64;
const KERNEL_NAME: &str = "gemm_mlx_m16_f16";

const M_VALUES: &[usize] = &[512, 256, 128, 64, 32, 16, 8, 4, 1]; // largest first
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// f16 helpers
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

fn rand_f16_bytes(numel: usize, seed: u64) -> Vec<u8> {
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
    f16_bytes
}

fn rand_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
    let numel: usize = shape.iter().product();
    let f16_bytes = rand_f16_bytes(numel, seed);
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

fn make_u32_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: u32,
) -> MtlBuffer {
    let opts = MTLResourceOptions::StorageModeShared;
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(&val as *const u32 as *mut _).unwrap(),
                4_usize,
                opts,
            )
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// GPU timestamp helpers
// ---------------------------------------------------------------------------

/// Extract pure GPU execution time from a completed CommandBuffer.
/// Uses Metal's GPUStartTime/GPUEndTime properties (seconds, f64).
/// Only valid after waitUntilCompleted().
fn gpu_time_us(cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>) -> f64 {
    let start = cb.GPUStartTime();
    let end = cb.GPUEndTime();
    (end - start) * 1_000_000.0
}

fn assert_cb_ok(cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>, context: &str) {
    let status = cb.status();
    assert!(
        status != objc2_metal::MTLCommandBufferStatus::Error,
        "GPU command buffer error in {context}: status={status:?}"
    );
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
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
    percentile_sorted(times, 50.0)
}

// ---------------------------------------------------------------------------
// GEMM encoding helper (all 11 buffers: 0-10 including residual)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    pipeline: &MtlPipeline,
    a: &Array,
    b: &Array,
    c: &Array,
    m_buf: &MtlBuffer,
    n_buf: &MtlBuffer,
    k_buf: &MtlBuffer,
    bsa_buf: &MtlBuffer,
    bsb_buf: &MtlBuffer,
    bsc_buf: &MtlBuffer,
    swizzle_buf: &MtlBuffer,
    residual: &Array,
    grid: MTLSize,
    tg: MTLSize,
) {
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), a.offset(), 0);
        enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), b.offset(), 1);
        enc.setBuffer_offset_atIndex(Some(c.metal_buffer()), c.offset(), 2);
        enc.setBuffer_offset_atIndex(Some(m_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(n_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(k_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(bsa_buf), 0, 6);
        enc.setBuffer_offset_atIndex(Some(bsb_buf), 0, 7);
        enc.setBuffer_offset_atIndex(Some(bsc_buf), 0, 8);
        enc.setBuffer_offset_atIndex(Some(swizzle_buf), 0, 9);
        enc.setBuffer_offset_atIndex(Some(residual.metal_buffer()), residual.offset(), 10);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("Metal GPU device required");
    println!(
        "Device: {} (unified_memory={})",
        gpu.name(),
        gpu.has_unified_memory()
    );

    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("kernel registration failed");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    println!(
        "Kernel: {} (BM={}, BN={}, BK={}, {} threads)",
        KERNEL_NAME, BM, BN, BK, THREADS
    );
    println!(
        "N={}, K={}, Warmup={}, Bench={}",
        N, K, WARMUP_ITERS, BENCH_ITERS
    );
    println!("Timing: GPU timestamps (GPUStartTime/GPUEndTime) + wall-clock Instant");
    println!();

    // ===================================================================
    // Step 1: Bare CB overhead (empty command buffer, no kernel)
    // ===================================================================
    println!("--- Bare Command Buffer Overhead (empty CB, GPU timestamps) ---");
    {
        // Warmup
        for _ in 0..WARMUP_ITERS {
            autoreleasepool(|_| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                cb.commit();
                cb.waitUntilCompleted();
            });
        }

        let mut bare_wall = Vec::with_capacity(BENCH_ITERS);
        let mut bare_gpu = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            autoreleasepool(|_| {
                let start = Instant::now();
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                cb.commit();
                cb.waitUntilCompleted();
                let wall_us = start.elapsed().as_secs_f64() * 1_000_000.0;
                assert_cb_ok(&cb, "bare CB");
                let gpu_us = gpu_time_us(&cb);
                bare_wall.push(wall_us);
                bare_gpu.push(gpu_us);
            });
        }

        let wall_p50 = p50(&mut bare_wall);
        let gpu_p50 = p50(&mut bare_gpu);
        let host_oh_p50 = wall_p50 - gpu_p50;
        println!(
            "Bare CB: wall_p50={:.1} us, gpu_p50={:.1} us, host_oh={:.1} us",
            wall_p50, gpu_p50, host_oh_p50
        );
        println!();
    }

    // ===================================================================
    // Step 2: Per-M GEMM dispatch analysis
    // ===================================================================
    println!(
        "| {:>5} | {:>10} | {:>10} | {:>12} | {:>10} | {:>10} |",
        "M", "Wall (us)", "GPU (us)", "Host OH (us)", "Host OH %", "GPU TFLOPS"
    );
    println!(
        "|{:-<7}|{:-<12}|{:-<12}|{:-<14}|{:-<12}|{:-<12}|",
        "", "", "", "", "", ""
    );

    for &m in M_VALUES {
        let constants = ops::matmul::matmul_align_constants(m, N, K, BM, BN, BK);
        let pipeline = registry
            .get_pipeline_with_constants(KERNEL_NAME, DType::Float16, &constants)
            .expect("Pipeline creation failed");

        let a = rand_array(device, &[m, K], 42);
        let b = rand_array(device, &[K, N], 99);
        let c = Array::zeros(device, &[m, N], DType::Float16);
        let residual = Array::zeros(device, &[m, N], DType::Float16);

        let m_buf = make_u32_buf(device, m as u32);
        let n_buf = make_u32_buf(device, N as u32);
        let k_buf = make_u32_buf(device, K as u32);
        let bsa_buf = make_u32_buf(device, (m * K) as u32);
        let bsb_buf = make_u32_buf(device, (K * N) as u32);
        let bsc_buf = make_u32_buf(device, (m * N) as u32);
        let swizzle_log = ops::matmul::compute_swizzle_log(m, N, BM, BN);
        let swizzle_buf = make_u32_buf(device, swizzle_log);

        let grid_x = N.div_ceil(BN);
        let grid_y = m.div_ceil(BM);
        let grid = MTLSize {
            width: grid_x,
            height: grid_y,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS as usize,
            height: 1,
            depth: 1,
        };

        // Warmup
        for _ in 0..WARMUP_ITERS {
            autoreleasepool(|_| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                encode_gemm(
                    &enc,
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
                    &residual,
                    grid,
                    tg,
                );
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                assert_cb_ok(&cb, "warmup");
            });
        }

        // Bench
        let mut wall_times = Vec::with_capacity(BENCH_ITERS);
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            autoreleasepool(|_| {
                let start = Instant::now();
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                encode_gemm(
                    &enc,
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
                    &residual,
                    grid,
                    tg,
                );
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                let wall_us = start.elapsed().as_secs_f64() * 1_000_000.0;
                assert_cb_ok(&cb, &format!("bench M={m}"));
                let gpu_us = gpu_time_us(&cb);
                wall_times.push(wall_us);
                gpu_times.push(gpu_us);
            });
        }

        let wall_p50 = p50(&mut wall_times);
        let gpu_p50 = p50(&mut gpu_times);
        let host_oh = wall_p50 - gpu_p50;
        let host_oh_pct = if wall_p50 > 0.0 {
            host_oh / wall_p50 * 100.0
        } else {
            0.0
        };
        let tflops = if gpu_p50 > 0.0 {
            2.0 * m as f64 * N as f64 * K as f64 / (gpu_p50 * 1e-6) / 1e12
        } else {
            0.0
        };

        println!(
            "| {:>5} | {:>10.1} | {:>10.1} | {:>12.1} | {:>9.1}% | {:>10.2} |",
            m, wall_p50, gpu_p50, host_oh, host_oh_pct, tflops
        );

        // Brief pause between M values to let GPU drain
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
