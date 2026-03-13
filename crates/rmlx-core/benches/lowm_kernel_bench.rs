//! ⚠️ NON-PRODUCTION PATH — low-M kernel comparison with direct kernel encoding.
//! Development only; pipelined mode simulates 32-layer but bypasses TransformerModel.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Low-M kernel comparison benchmark (M=2..8).
//!
//! Compares all viable kernel dispatch strategies for small M values
//! in pipelined mode (32 CBs), the same pattern as a 32-layer transformer.
//!
//! Kernel candidates:
//!   1. **MlxArchMicro** (`gemm_mlx_m16_f16`): BM=16, BN=32, BK=16, 64 threads
//!   2. **Skinny** (`gemm_skinny_f16`): BM=32, BN=128, BK=32, 256 threads
//!   3. **Split-K small** (`splitk_small_pass1_f16` + `splitk_reduce_f16`): 2-pass
//!   4. **Split-K MLX** (`splitk_pass1_mlx_f16` + `splitk_reduce_f16`): 2-pass
//!   5. **Auto-dispatch** (`matmul_into_cb`): production dispatch path
//!
//! N=3584, K=3584, f16, 5 warmup + 20 bench, largest M first (thermal).
//!
//! Usage: cargo bench -p rmlx-core --bench lowm_kernel_bench

use std::time::Instant;
use std::ptr::NonNull;

use half::f16;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _, MTLCommandQueue as _, MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLCommandEncoder as _};
use rmlx_metal::{MTLSize, MTLResourceOptions};
use rmlx_metal::types::{MtlBuffer, MtlPipeline};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const PIPELINE_N: usize = 32;
const N: usize = 3584;
const K: usize = 3584;
const M_VALUES: &[usize] = &[2, 4, 6, 8];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn rand_f16_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
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

#[inline(always)]
fn set_u32(enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, index: usize, val: u32) {
    unsafe { enc.setBytes_length_atIndex(NonNull::new(&val as *const u32 as *const std::ffi::c_void as *mut _).unwrap(), 4_usize, index) };
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
// Kernel specs
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct KernelSpec {
    label: &'static str,
    name: &'static str,
    bm: usize,
    bn: usize,
    bk: usize,
    threads: u64,
    uses_function_constants: bool,
    has_swizzle: bool,
}

const KERNEL_MLX_MICRO: KernelSpec = KernelSpec {
    label: "MlxMicro",
    name: "gemm_mlx_m16_f16",
    bm: 16,
    bn: 32,
    bk: 16,
    threads: 64,
    uses_function_constants: true,
    has_swizzle: true,
};

const KERNEL_SKINNY: KernelSpec = KernelSpec {
    label: "Skinny",
    name: "gemm_skinny_f16",
    bm: 32,
    bn: 128,
    bk: 32,
    threads: 256,
    uses_function_constants: false,
    has_swizzle: true,
};

// ---------------------------------------------------------------------------
// Pipelined GEMM benchmark (single-pass kernels)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_pipelined_gemm(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    pipeline: &MtlPipeline,
    a: &Array,
    b: &Array,
    m: usize,
    n: usize,
    k: usize,
    spec: &KernelSpec,
) -> f64 {
    let queue = device.newCommandQueue().unwrap();
    let opts = MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64; // f16 = 2 bytes
    let out_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(out_size as usize, opts).unwrap())
        .collect();

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let bsa = (m * k) as u32;
    let bsb = (k * n) as u32;
    let bsc = (m * n) as u32;

    let swizzle_log = if spec.has_swizzle {
        ops::matmul::compute_swizzle_log(m, n, spec.bm, spec.bn)
    } else {
        0
    };

    let grid = MTLSize { width: (n.div_ceil(spec.bn) << swizzle_log) as usize, height: (m.div_ceil(spec.bm) >> swizzle_log) as usize, depth: 1_usize };
    let tg = MTLSize { width: spec.threads as usize, height: 1_usize, depth: 1_usize };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pipeline);
                unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
                set_u32(&enc, 3, m_u32);
                set_u32(&enc, 4, n_u32);
                set_u32(&enc, 5, k_u32);
                set_u32(&enc, 6, bsa);
                set_u32(&enc, 7, bsb);
                set_u32(&enc, 8, bsc);
                set_u32(&enc, 9, swizzle_log);
                set_u32(&enc, 10, 0u32); // residual flag
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pipeline);
                unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
                set_u32(&enc, 3, m_u32);
                set_u32(&enc, 4, n_u32);
                set_u32(&enc, 5, k_u32);
                set_u32(&enc, 6, bsa);
                set_u32(&enc, 7, bsb);
                set_u32(&enc, 8, bsc);
                set_u32(&enc, 9, swizzle_log);
                set_u32(&enc, 10, 0u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Pipelined Split-K benchmark (2-pass: pass1 + reduce per CB)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_pipelined_splitk(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    m: usize,
    n: usize,
    k: usize,
    splitk_bm: usize,
    splitk_bn: usize,
    pass1_kernel: &str,
    n_splits: usize,
) -> f64 {
    let queue = device.newCommandQueue().unwrap();
    let opts = MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64; // f16
    let partial_size = (n_splits * m * n * 4) as u64; // f32

    let out_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(out_size as usize, opts).unwrap())
        .collect();
    let partial_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(partial_size as usize, opts).unwrap())
        .collect();

    let bk = 16usize;
    let constants = ops::matmul::matmul_align_constants(m, n, k, splitk_bm, splitk_bn, bk);
    let pass1_pipeline = registry
        .get_pipeline_with_constants(pass1_kernel, DType::Float16, &constants)
        .unwrap_or_else(|e| panic!("Failed to get pipeline for {}: {e}", pass1_kernel));
    let pass2_pipeline = registry
        .get_pipeline("splitk_reduce_f16", DType::Float16)
        .expect("Failed to get splitk_reduce_f16 pipeline");

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let splits_u32 = n_splits as u32;
    let swizzle_log = ops::matmul::compute_swizzle_log(m, n, splitk_bm, splitk_bn);

    let pass1_grid = MTLSize { width: (n.div_ceil(splitk_bn) << swizzle_log) as usize, height: (m.div_ceil(splitk_bm) >> swizzle_log) as usize, depth: n_splits };
    let pass1_tg = MTLSize { width: 64_usize, height: 1_usize, depth: 1_usize };

    let total_elems = m * n;
    let reduce_tg_size = 256u64;
    let reduce_groups = total_elems.div_ceil(reduce_tg_size as usize) as u64;
    let reduce_grid = MTLSize { width: reduce_groups as usize, height: 1_usize, depth: 1_usize };
    let reduce_tg = MTLSize { width: reduce_tg_size as usize, height: 1_usize, depth: 1_usize };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .zip(partial_bufs.iter())
            .map(|(out_buf, partial_buf)| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                // Pass 1
                {
                    let enc = cb.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pass1_pipeline);
                    unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(partial_buf), 0_usize, 2_usize) };
                    set_u32(&enc, 3, m_u32);
                    set_u32(&enc, 4, n_u32);
                    set_u32(&enc, 5, k_u32);
                    set_u32(&enc, 6, splits_u32);
                    set_u32(&enc, 7, swizzle_log);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(pass1_grid, pass1_tg);
                    enc.endEncoding();
                }
                // Pass 2 (reduce)
                {
                    let enc = cb.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pass2_pipeline);
                    unsafe { enc.setBuffer_offset_atIndex(Some(partial_buf), 0_usize, 0_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 1_usize) };
                    set_u32(&enc, 2, m_u32);
                    set_u32(&enc, 3, n_u32);
                    set_u32(&enc, 4, splits_u32);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(reduce_grid, reduce_tg);
                    enc.endEncoding();
                }
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cbs: Vec<_> = out_bufs
            .iter()
            .zip(partial_bufs.iter())
            .map(|(out_buf, partial_buf)| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                // Pass 1
                {
                    let enc = cb.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pass1_pipeline);
                    unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(partial_buf), 0_usize, 2_usize) };
                    set_u32(&enc, 3, m_u32);
                    set_u32(&enc, 4, n_u32);
                    set_u32(&enc, 5, k_u32);
                    set_u32(&enc, 6, splits_u32);
                    set_u32(&enc, 7, swizzle_log);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(pass1_grid, pass1_tg);
                    enc.endEncoding();
                }
                // Pass 2 (reduce)
                {
                    let enc = cb.computeCommandEncoder().unwrap();
                    enc.setComputePipelineState(&pass2_pipeline);
                    unsafe { enc.setBuffer_offset_atIndex(Some(partial_buf), 0_usize, 0_usize) };
                    unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 1_usize) };
                    set_u32(&enc, 2, m_u32);
                    set_u32(&enc, 3, n_u32);
                    set_u32(&enc, 4, splits_u32);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(reduce_grid, reduce_tg);
                    enc.endEncoding();
                }
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Pipelined auto-dispatch benchmark (uses matmul_into_cb)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn bench_pipelined_auto(registry: &KernelRegistry, a: &Array, b: &Array) -> f64 {
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = (0..PIPELINE_N)
            .map(|_| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let _out =
                    ops::matmul::matmul_into_cb(registry, a, b, &cb).expect("matmul_into_cb failed");
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cbs: Vec<_> = (0..PIPELINE_N)
            .map(|_| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let _out =
                    ops::matmul::matmul_into_cb(registry, a, b, &cb).expect("matmul_into_cb failed");
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct BenchRow {
    m: usize,
    label: &'static str,
    pipe_us: f64,
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();

    // Compute split-K parameters
    // For M=2-8, N=3584, K=3584: tile config would be MlxArchMicro (BM=16, BN=32)
    // non_splitk_tgs = ceil(3584/16) * ceil(3584/32) = 224 * 112 = 25088
    // gpu_cores ~80, target = 80*4=320. 25088 >> 320, so Split-K won't auto-trigger.
    // We force n_splits for the manual Split-K benchmarks.
    let gpu_cores = registry.device().tuning().gpu_cores;
    let splitk_small_bm = 32usize;
    let splitk_small_bn = 32usize;
    let splitk_mlx_bm = 64usize;
    let splitk_mlx_bn = 64usize;

    println!("=== RMLX Low-M Kernel Comparison Benchmark ===");
    println!(
        "N={}, K={}, dtype=f16, Pipeline={} CBs, Warmup={}, Bench={} iters",
        N, K, PIPELINE_N, WARMUP_ITERS, BENCH_ITERS
    );
    println!("GPU cores: {}", gpu_cores);
    println!();

    let mut results: Vec<BenchRow> = Vec::new();

    // Run largest M first (thermal fairness)
    for &m in M_VALUES.iter().rev() {
        println!("--- M={} ---", m);

        let a = rand_f16_array(device, &[m, K], 42);
        let b = rand_f16_array(device, &[K, N], 99);

        // 1. MlxArchMicro
        {
            let spec = &KERNEL_MLX_MICRO;
            let constants = ops::matmul::matmul_align_constants(m, N, K, spec.bm, spec.bn, spec.bk);
            let pipeline = registry
                .get_pipeline_with_constants(spec.name, DType::Float16, &constants)
                .unwrap_or_else(|e| panic!("Failed to get pipeline for {}: {e}", spec.name));
            let us = bench_pipelined_gemm(device, &pipeline, &a, &b, m, N, K, spec);
            let t = tflops(m, N, K, us);
            println!("  {:<16} {:>7.1} us  {:>6.3} TFLOPS", spec.label, us, t);
            results.push(BenchRow {
                m,
                label: spec.label,
                pipe_us: us,
            });
        }

        // 2. Skinny
        {
            let spec = &KERNEL_SKINNY;
            let pipeline = registry
                .get_pipeline(spec.name, DType::Float16)
                .unwrap_or_else(|e| panic!("Failed to get pipeline for {}: {e}", spec.name));
            let us = bench_pipelined_gemm(device, &pipeline, &a, &b, m, N, K, spec);
            let t = tflops(m, N, K, us);
            println!("  {:<16} {:>7.1} us  {:>6.3} TFLOPS", spec.label, us, t);
            results.push(BenchRow {
                m,
                label: spec.label,
                pipe_us: us,
            });
        }

        // 3. Split-K small (BM=32, BN=32)
        {
            let splitk_tgs = m.div_ceil(splitk_small_bm) * N.div_ceil(splitk_small_bn);
            let target_tgs = gpu_cores * 4;
            let k_tiles = K / 16;
            let n_splits = (target_tgs / splitk_tgs.max(1)).clamp(2, k_tiles.min(32));
            let us = bench_pipelined_splitk(
                device,
                &registry,
                &a,
                &b,
                m,
                N,
                K,
                splitk_small_bm,
                splitk_small_bn,
                "splitk_small_pass1_f16",
                n_splits,
            );
            let t = tflops(m, N, K, us);
            println!(
                "  {:<16} {:>7.1} us  {:>6.3} TFLOPS  (splits={})",
                "SplitK-Small", us, t, n_splits
            );
            results.push(BenchRow {
                m,
                label: "SplitK-Small",
                pipe_us: us,
            });
        }

        // 4. Split-K MLX (BM=64, BN=64)
        {
            let splitk_tgs = m.div_ceil(splitk_mlx_bm) * N.div_ceil(splitk_mlx_bn);
            let target_tgs = gpu_cores * 4;
            let k_tiles = K / 16;
            let n_splits = (target_tgs / splitk_tgs.max(1)).clamp(2, k_tiles.min(32));
            let us = bench_pipelined_splitk(
                device,
                &registry,
                &a,
                &b,
                m,
                N,
                K,
                splitk_mlx_bm,
                splitk_mlx_bn,
                "splitk_pass1_mlx_f16",
                n_splits,
            );
            let t = tflops(m, N, K, us);
            println!(
                "  {:<16} {:>7.1} us  {:>6.3} TFLOPS  (splits={})",
                "SplitK-MLX", us, t, n_splits
            );
            results.push(BenchRow {
                m,
                label: "SplitK-MLX",
                pipe_us: us,
            });
        }

        /* // 5. Auto-dispatch (production path)
        {
            let us = bench_pipelined_auto(&registry, &a, &b);
            let t = tflops(m, N, K, us);
            println!("  {:<16} {:>7.1} us  {:>6.3} TFLOPS", "Auto", us, t);
            results.push(BenchRow {
                m,
                label: "Auto",
                pipe_us: us,
            });
        } */

        println!();
    }

    // --- Markdown results table ---
    println!("=== Results (Pipelined, amortized per dispatch) ===");
    println!();
    println!(
        "| {:>3} | {:<16} | {:>10} | {:>10} |",
        "M", "Kernel", "Latency us", "TFLOPS"
    );
    println!("|{:-<5}|{:-<18}|{:-<12}|{:-<12}|", "", "", "", "");

    // Sort by M then latency
    results.sort_by(|a, b| {
        a.m.cmp(&b.m)
            .then(a.pipe_us.partial_cmp(&b.pipe_us).unwrap())
    });

    let mut current_m = 0;
    for r in &results {
        if r.m != current_m {
            if current_m != 0 {
                println!("|{:-<5}|{:-<18}|{:-<12}|{:-<12}|", "", "", "", "");
            }
            current_m = r.m;
        }
        let t = tflops(r.m, N, K, r.pipe_us);
        let best_for_m = results
            .iter()
            .filter(|x| x.m == r.m)
            .map(|x| x.pipe_us)
            .fold(f64::INFINITY, f64::min);
        let marker = if (r.pipe_us - best_for_m).abs() < 0.01 {
            " *"
        } else {
            ""
        };
        println!(
            "| {:>3} | {:<16} | {:>8.1} us | {:>8.3} T{}|",
            r.m, r.label, r.pipe_us, t, marker
        );
    }

    // --- Winner summary ---
    println!();
    println!("=== Winner per M ===");
    println!();
    println!(
        "| {:>3} | {:<16} | {:>10} | {:>10} |",
        "M", "Best Kernel", "Latency us", "TFLOPS"
    );
    println!("|{:-<5}|{:-<18}|{:-<12}|{:-<12}|", "", "", "", "");

    for &m in M_VALUES {
        if let Some(best) = results
            .iter()
            .filter(|r| r.m == m)
            .min_by(|a, b| a.pipe_us.partial_cmp(&b.pipe_us).unwrap())
        {
            let t = tflops(m, N, K, best.pipe_us);
            println!(
                "| {:>3} | {:<16} | {:>8.1} us | {:>8.3} T |",
                m, best.label, best.pipe_us, t
            );
        }
    }

    println!();
    println!("Done. (* = fastest for that M value)");
}
