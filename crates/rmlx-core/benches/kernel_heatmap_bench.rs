//! ⚠️ NON-PRODUCTION PATH — direct kernel encoding for kernel selection tuning.
//! Bypasses matmul() dispatch (Split-K, GEMV routing, contiguity checks).
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Comprehensive f16 GEMM kernel heatmap benchmark.
//!
//! Tests ALL available f16 GEMM kernel implementations across a wide range of
//! M values with fixed representative N,K pairs. Output is a heatmap of
//! "best kernel per (M, N×K) cell".
//!
//! Kernels tested:
//!   1. **MlxMicro**    — `gemm_mlx_m16_f16`    (BM=16, BN=32, BK=16, 64 threads)
//!   2. **MlxSmall**    — `gemm_mlx_small_f16`   (BM=32, BN=32, BK=16, 64 threads)
//!   3. **MlxArch**     — `gemm_mlx_f16`         (BM=64, BN=64, BK=16, 128 threads)
//!   4. **Skinny**      — `gemm_skinny_f16`       (BM=32, BN=128, BK=32, 256 threads)
//!   5. **SplitK-Small** — `splitk_small_pass1_f16` + `splitk_reduce_f16` (BM=32, BN=32, 2-split)
//!   6. **SplitK-MLX**  — `splitk_pass1_mlx_f16`  + `splitk_reduce_f16` (BM=64, BN=64)
//!   7. **NAX 64x64**   — `gemm_nax_64x64_f16`   (BM=64, BN=64, BK=32, 128 threads, MMA)
//!   8. **NAX 64x128**  — `gemm_nax_64x128_f16`  (BM=64, BN=128, BK=32, 256 threads, MMA)
//!   9. **NAX 128x128** — `gemm_nax_f16`         (BM=128, BN=128, BK=32, 512 threads, MMA)
//!  10. **NAX V2**      — `gemm_nax_v2_f16`      (BM=128, BN=128, BK=32, 512 threads, MMA+half4)
//!
//! M = [2, 4, 8, 16, 32, 64, 128, 256, 512]
//! Shapes: 3584×3584, 4096×4096, 14336×4096, 4096×14336
//!
//! Usage: cargo bench -p rmlx-core --bench kernel_heatmap_bench

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

// M values tested (run largest-first for thermal fairness)
const M_VALUES: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256, 512];

// (N, K) shape pairs
const SHAPES: &[(usize, usize)] = &[
    (3584, 3584),  // Qwen attention projection
    (4096, 4096),  // standard transformer
    (14336, 4096), // FFN up-projection
    (4096, 14336), // FFN down-projection
];

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

#[derive(Clone)]
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

const KERNEL_MLX_SMALL: KernelSpec = KernelSpec {
    label: "MlxSmall",
    name: "gemm_mlx_small_f16",
    bm: 32,
    bn: 32,
    bk: 16,
    threads: 64,
    uses_function_constants: true,
    has_swizzle: true,
};

const KERNEL_MLX_ARCH: KernelSpec = KernelSpec {
    label: "MlxArch",
    name: "gemm_mlx_f16",
    bm: 64,
    bn: 64,
    bk: 16,
    threads: 128,
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

const KERNEL_NAX_64X64: KernelSpec = KernelSpec {
    label: "NAX-64x64",
    name: "gemm_nax_64x64_f16",
    bm: 64,
    bn: 64,
    bk: 32,
    threads: 128,
    uses_function_constants: true,
    has_swizzle: true,
};

const KERNEL_NAX_64X128: KernelSpec = KernelSpec {
    label: "NAX-64x128",
    name: "gemm_nax_64x128_f16",
    bm: 64,
    bn: 128,
    bk: 32,
    threads: 256,
    uses_function_constants: true,
    has_swizzle: true,
};

const KERNEL_NAX_128X128: KernelSpec = KernelSpec {
    label: "NAX-128x128",
    name: "gemm_nax_f16",
    bm: 128,
    bn: 128,
    bk: 32,
    threads: 512,
    uses_function_constants: true,
    has_swizzle: true,
};

const KERNEL_NAX_V2: KernelSpec = KernelSpec {
    label: "NAX-V2",
    name: "gemm_nax_v2_f16",
    bm: 128,
    bn: 128,
    bk: 32,
    threads: 512,
    uses_function_constants: true,
    has_swizzle: true,
};

// ---------------------------------------------------------------------------
// Single-pass pipelined GEMM benchmark
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

    // Dummy buffer for function-constant kernel epilogue bindings (buffer 10-13).
    // When has_residual=false, the kernel never reads these, but Metal validation
    // requires all declared buffer arguments to be bound.
    let dummy_buf = if spec.uses_function_constants {
        Some(device.newBufferWithLength_options(out_size.max(16) as usize, opts).unwrap())
    } else {
        None
    };

    // Helper closure to encode one dispatch
    let encode_dispatch = |enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, out_buf: &MtlBuffer| {
        enc.setComputePipelineState(pipeline);
        unsafe { enc.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0_usize, 1_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
        set_u32(enc, 3, m_u32);
        set_u32(enc, 4, n_u32);
        set_u32(enc, 5, k_u32);
        set_u32(enc, 6, bsa);
        set_u32(enc, 7, bsb);
        set_u32(enc, 8, bsc);
        set_u32(enc, 9, swizzle_log);
        if let Some(ref dummy) = dummy_buf {
            unsafe { enc.setBuffer_offset_atIndex(Some(dummy), 0_usize, 10_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(dummy), 0_usize, 11_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(dummy), 0_usize, 12_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(dummy), 0_usize, 13_usize) };
        } else {
            set_u32(enc, 10, 0u32); // residual flag (non-FC kernels)
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                encode_dispatch(&enc, out_buf);
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
                encode_dispatch(&enc, out_buf);
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
// Split-K pipelined benchmark (2-pass: pass1 + reduce per CB)
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
    splitk_bk: usize,
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

    let constants = ops::matmul::matmul_align_constants(m, n, k, splitk_bm, splitk_bn, splitk_bk);
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
// Result types
// ---------------------------------------------------------------------------

struct BenchRow {
    m: usize,
    n: usize,
    k: usize,
    label: &'static str,
    latency_us: f64,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let gpu_cores = registry.device().tuning().gpu_cores;

    println!("=== RMLX Kernel Heatmap Benchmark ===");
    println!(
        "dtype=f16, Pipeline={} CBs, Warmup={}, Bench={} iters",
        PIPELINE_N, WARMUP_ITERS, BENCH_ITERS
    );
    println!("GPU cores: {}", gpu_cores);
    println!("M values: {:?}", M_VALUES);
    println!(
        "Shapes: {}",
        SHAPES
            .iter()
            .map(|(n, k)| format!("{}x{}", n, k))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    // Pre-compile all single-pass kernel pipelines per shape
    // (function constants depend on M, so we compile on-the-fly per M)
    let single_pass_specs: Vec<KernelSpec> = vec![
        KERNEL_MLX_MICRO.clone(),
        KERNEL_MLX_SMALL.clone(),
        KERNEL_MLX_ARCH.clone(),
        KERNEL_SKINNY.clone(),
        KERNEL_NAX_64X64.clone(),
        KERNEL_NAX_64X128.clone(),
        KERNEL_NAX_128X128.clone(),
        KERNEL_NAX_V2.clone(),
    ];

    let mut results: Vec<BenchRow> = Vec::new();
    let total_combos = M_VALUES.len() * SHAPES.len() * (single_pass_specs.len() + 2); // +2 for split-k
    let mut done = 0usize;

    // Run shapes with largest M first for thermal fairness
    for &(n, k) in SHAPES {
        println!("=== Shape N={}, K={} ===", n, k);

        // Pre-allocate B matrix (shared across M values for same shape)
        // A matrix depends on M, allocated per-M

        for &m in M_VALUES.iter().rev() {
            println!("  --- M={} N={} K={} ---", m, n, k);

            let a = rand_f16_array(device, &[m, k], 42);
            let b = rand_f16_array(device, &[k, n], 99);

            // 1-4: Single-pass kernels
            for spec in &single_pass_specs {
                let pipeline_result = if spec.uses_function_constants {
                    let constants =
                        ops::matmul::matmul_align_constants(m, n, k, spec.bm, spec.bn, spec.bk);
                    registry.get_pipeline_with_constants(spec.name, DType::Float16, &constants)
                } else {
                    registry.get_pipeline(spec.name, DType::Float16)
                };

                match pipeline_result {
                    Ok(pipeline) => {
                        let us = bench_pipelined_gemm(device, &pipeline, &a, &b, m, n, k, spec);
                        let t = tflops(m, n, k, us);
                        println!("    {:<14} {:>8.1} us  {:>6.3} TFLOPS", spec.label, us, t);
                        results.push(BenchRow {
                            m,
                            n,
                            k,
                            label: spec.label,
                            latency_us: us,
                        });
                    }
                    Err(e) => {
                        println!("    {:<14} SKIPPED ({})", spec.label, e);
                    }
                }
                done += 1;
            }

            // 5: Split-K Small (BM=32, BN=32, BK=16)
            {
                let splitk_bm = 32usize;
                let splitk_bn = 32usize;
                let splitk_bk = 16usize;
                let splitk_tgs = m.div_ceil(splitk_bm) * n.div_ceil(splitk_bn);
                let target_tgs = gpu_cores * 4;
                let k_tiles = k / splitk_bk;
                let n_splits = (target_tgs / splitk_tgs.max(1)).clamp(2, k_tiles.min(32));

                let us = bench_pipelined_splitk(
                    device,
                    &registry,
                    &a,
                    &b,
                    m,
                    n,
                    k,
                    splitk_bm,
                    splitk_bn,
                    splitk_bk,
                    "splitk_small_pass1_f16",
                    n_splits,
                );
                let t = tflops(m, n, k, us);
                println!(
                    "    {:<14} {:>8.1} us  {:>6.3} TFLOPS  (splits={})",
                    "SplitK-Small", us, t, n_splits
                );
                results.push(BenchRow {
                    m,
                    n,
                    k,
                    label: "SplitK-Small",
                    latency_us: us,
                });
                done += 1;
            }

            // 6: Split-K MLX (BM=64, BN=64, BK=16)
            {
                let splitk_bm = 64usize;
                let splitk_bn = 64usize;
                let splitk_bk = 16usize;
                let splitk_tgs = m.div_ceil(splitk_bm) * n.div_ceil(splitk_bn);
                let target_tgs = gpu_cores * 4;
                let k_tiles = k / splitk_bk;
                let n_splits = (target_tgs / splitk_tgs.max(1)).clamp(2, k_tiles.min(32));

                let us = bench_pipelined_splitk(
                    device,
                    &registry,
                    &a,
                    &b,
                    m,
                    n,
                    k,
                    splitk_bm,
                    splitk_bn,
                    splitk_bk,
                    "splitk_pass1_mlx_f16",
                    n_splits,
                );
                let t = tflops(m, n, k, us);
                println!(
                    "    {:<14} {:>8.1} us  {:>6.3} TFLOPS  (splits={})",
                    "SplitK-MLX", us, t, n_splits
                );
                results.push(BenchRow {
                    m,
                    n,
                    k,
                    label: "SplitK-MLX",
                    latency_us: us,
                });
                done += 1;
            }

            println!("    [{}/{} combos done]", done, total_combos);
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Full results table
    // -----------------------------------------------------------------------
    println!();
    println!("=== Full Results (Pipelined, amortized per dispatch) ===");
    println!();
    println!(
        "| {:>3} | {:>5} | {:>5} | {:<14} | {:>12} | {:>8} |",
        "M", "N", "K", "Kernel", "Latency (us)", "TFLOPS"
    );
    println!(
        "|{:-<5}|{:-<7}|{:-<7}|{:-<16}|{:-<14}|{:-<10}|",
        "", "", "", "", "", ""
    );

    // Sort by (N, K, M, latency) for display
    results.sort_by(|a, b| {
        a.n.cmp(&b.n)
            .then(a.k.cmp(&b.k))
            .then(a.m.cmp(&b.m))
            .then(a.latency_us.partial_cmp(&b.latency_us).unwrap())
    });

    for r in &results {
        let t = tflops(r.m, r.n, r.k, r.latency_us);
        let best_for_cell = results
            .iter()
            .filter(|x| x.m == r.m && x.n == r.n && x.k == r.k)
            .map(|x| x.latency_us)
            .fold(f64::INFINITY, f64::min);
        let marker = if (r.latency_us - best_for_cell).abs() < 0.01 {
            " *"
        } else {
            ""
        };
        println!(
            "| {:>3} | {:>5} | {:>5} | {:<14} | {:>9.1} us | {:>6.3}{}|",
            r.m, r.n, r.k, r.label, r.latency_us, t, marker
        );
    }

    // -----------------------------------------------------------------------
    // Winner summary table
    // -----------------------------------------------------------------------
    println!();
    println!("=== Winner per (M, Shape) ===");
    println!();
    println!(
        "| {:>3} | {:>10} | {:<14} | {:>12} | {:>8} | {:<14} | {:>6} |",
        "M", "NxK", "Best Kernel", "Latency (us)", "TFLOPS", "2nd Best", "Gap %"
    );
    println!(
        "|{:-<5}|{:-<12}|{:-<16}|{:-<14}|{:-<10}|{:-<16}|{:-<8}|",
        "", "", "", "", "", "", ""
    );

    for &m in M_VALUES {
        for &(n, k) in SHAPES {
            let mut cell_results: Vec<&BenchRow> = results
                .iter()
                .filter(|r| r.m == m && r.n == n && r.k == k)
                .collect();
            cell_results.sort_by(|a, b| a.latency_us.partial_cmp(&b.latency_us).unwrap());

            if let Some(best) = cell_results.first() {
                let t = tflops(m, n, k, best.latency_us);
                let (second_label, gap_pct) = if cell_results.len() >= 2 {
                    let second = cell_results[1];
                    let gap = (second.latency_us / best.latency_us - 1.0) * 100.0;
                    (second.label, gap)
                } else {
                    ("-", 0.0)
                };
                println!(
                    "| {:>3} | {:>5}x{:<4} | {:<14} | {:>9.1} us | {:>6.3} | {:<14} | {:>5.1}% |",
                    m, n, k, best.label, best.latency_us, t, second_label, gap_pct,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Heatmap grid (best kernel per cell)
    // -----------------------------------------------------------------------
    println!();
    println!("=== Heatmap: Best Kernel per (M, Shape) ===");
    println!();

    // Header
    print!("| {:>5} |", "M");
    for &(n, k) in SHAPES {
        print!(" {:>14} |", format!("{}x{}", n, k));
    }
    println!();

    print!("|{:-<7}|", "");
    for _ in SHAPES {
        print!("{:-<16}|", "");
    }
    println!();

    // Rows
    for &m in M_VALUES {
        print!("| {:>5} |", m);
        for &(n, k) in SHAPES {
            let cell_best = results
                .iter()
                .filter(|r| r.m == m && r.n == n && r.k == k)
                .min_by(|a, b| a.latency_us.partial_cmp(&b.latency_us).unwrap());
            if let Some(best) = cell_best {
                let t = tflops(m, n, k, best.latency_us);
                print!(" {:>7} {:.1}T |", best.label, t);
            } else {
                print!(" {:>14} |", "N/A");
            }
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Heatmap grid (TFLOPS only, compact)
    // -----------------------------------------------------------------------
    println!();
    println!("=== Heatmap: Peak TFLOPS per (M, Shape) ===");
    println!();

    print!("| {:>5} |", "M");
    for &(n, k) in SHAPES {
        print!(" {:>10} |", format!("{}x{}", n, k));
    }
    println!();

    print!("|{:-<7}|", "");
    for _ in SHAPES {
        print!("{:-<12}|", "");
    }
    println!();

    for &m in M_VALUES {
        print!("| {:>5} |", m);
        for &(n, k) in SHAPES {
            let cell_best = results
                .iter()
                .filter(|r| r.m == m && r.n == n && r.k == k)
                .min_by(|a, b| a.latency_us.partial_cmp(&b.latency_us).unwrap());
            if let Some(best) = cell_best {
                let t = tflops(m, n, k, best.latency_us);
                print!(" {:>8.2} T |", t);
            } else {
                print!(" {:>10} |", "N/A");
            }
        }
        println!();
    }

    println!();
    println!(
        "Done. {} combos benchmarked. (* = fastest for that M/shape cell)",
        results.len()
    );
}
