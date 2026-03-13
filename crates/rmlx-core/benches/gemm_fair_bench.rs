//! RMLX vs MLX fair GEMM comparison benchmark.
//!
//! Includes both direct kernel encoding (for GPU-level comparison) and a **production
//! dispatch mode** that calls `ops::matmul::matmul()` with full Split-K routing, GEMV
//! dispatch, kernel selection, and per-op CB overhead — matching what the real inference
//! path executes.
//!
//! Five dispatch modes:
//!   1. **Sync**: 1 dispatch per CB, `wait_until_completed` every time (unfair overhead)
//!   2. **Pipelined**: 32 separate CBs each with 1 dispatch to its own output
//!      buffer (avoids WAW hazards), committed without waiting (simulating
//!      32-layer transformer), `wait_until_completed` on last CB only,
//!      amortized time = total/32
//!   3. **Multi-encoder**: 1 CB, 32 encoders (each with 1 dispatch to its own
//!      output buffer), commit once, wait once — isolates CB creation overhead
//!   4. **Single-encoder**: 1 CB, 1 encoder, 32 dispatches (each to its own
//!      output buffer via set_buffer + dispatch_thread_groups), commit once,
//!      wait once — measures absolute minimum dispatch overhead
//!   5. **Production**: 1 CB, 32 × `matmul_into_cb()` encodes, commit once, wait once
//!      — matches `forward_graph_unified(Prefill)` ExecGraph batch pattern.
//!      Tests production kernel selection (tile config, GEMV, function constants).
//!
//! Kernel selection per M (matching optimal dispatch):
//!   - M=1..128:  gemm_mlx_m16_f16    (BM=16, BN=32, BK=16, 64 threads)
//!   - M=256:     gemm_mlx_small_f16  (BM=32, BN=32, BK=16, 64 threads)
//!   - M=512:     gemm_nax_64x128_f16 (BM=64, BN=128, BK=32, 256 threads)
//!
//! All 4 NK shapes (Qwen 7B), f16, 5 warmup + 20 bench, largest M first (thermal).
//!
//! Usage: cargo bench -p rmlx-core --bench gemm_fair_bench

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
const M_VALUES: &[usize] = &[1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512];

/// (N, K, label) shapes for Qwen 7B benchmark.
const NK_SHAPES: &[(usize, usize, &str)] = &[
    (3584, 3584, "attn_proj"),
    (4096, 4096, "standard"),
    (14336, 4096, "ffn_up"),
    (4096, 14336, "ffn_down"),
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

/// Bind a u32 scalar to a compute encoder argument slot via `set_bytes`.
///
/// Replaces the old `make_u32_buf` approach which allocated a 4-byte Metal
/// buffer per call. `set_bytes` copies the value into the encoder's argument
/// buffer inline, avoiding per-dispatch buffer allocation overhead.
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
// Kernel selection
// ---------------------------------------------------------------------------

struct KernelSpec {
    name: &'static str,
    bm: usize,
    bn: usize,
    bk: usize,
    threads: u64,
}

fn select_kernel_for_m(m: usize, _n: usize, _k: usize) -> KernelSpec {
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
    m_val: u32,
    n_val: u32,
    k_val: u32,
    bsa_val: u32,
    bsb_val: u32,
    bsc_val: u32,
    swizzle_val: u32,
}

impl GemmBuffers {
    fn new(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, m: usize, n: usize, k: usize, bm: usize, bn: usize) -> Self {
        let a = rand_f16_array(device, &[m, k], 42);
        let b = rand_f16_array(device, &[k, n], 99);
        let c = Array::zeros(device, &[m, n], DType::Float16);
        let swizzle_val = ops::matmul::compute_swizzle_log(m, n, bm, bn);
        Self {
            a,
            b,
            c,
            m_val: m as u32,
            n_val: n as u32,
            k_val: k as u32,
            bsa_val: (m * k) as u32,
            bsb_val: (k * n) as u32,
            bsc_val: (m * n) as u32,
            swizzle_val,
        }
    }

    fn encode(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        pipeline: &MtlPipeline,
        grid: MTLSize,
        tg: MTLSize,
    ) {
        enc.setComputePipelineState(pipeline);
        unsafe { enc.setBuffer_offset_atIndex(Some(self.a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(self.b.metal_buffer()), 0_usize, 1_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(self.c.metal_buffer()), 0_usize, 2_usize) };
        set_u32(enc, 3, self.m_val);
        set_u32(enc, 4, self.n_val);
        set_u32(enc, 5, self.k_val);
        set_u32(enc, 6, self.bsa_val);
        set_u32(enc, 7, self.bsb_val);
        set_u32(enc, 8, self.bsc_val);
        set_u32(enc, 9, self.swizzle_val);
        set_u32(enc, 10, 0u32); // residual flag
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
}

// ---------------------------------------------------------------------------
// Sync benchmark: 1 dispatch per CB
// ---------------------------------------------------------------------------

fn bench_sync(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    pipeline: &MtlPipeline,
    bufs: &GemmBuffers,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    // Fresh queue to avoid poisoning from prior errors
    let queue = device.newCommandQueue().unwrap();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        bufs.encode(&enc, pipeline, grid, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        bufs.encode(&enc, pipeline, grid, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Pipelined benchmark: 32 separate CBs committed without waiting, sync on last
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_pipelined(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    pipeline: &MtlPipeline,
    bufs: &GemmBuffers,
    m: usize,
    n: usize,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    // Fresh queue to avoid poisoning from prior errors
    let queue = device.newCommandQueue().unwrap();
    // Allocate separate output buffers to avoid WAW hazards across CBs
    let opts = MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64; // f16 = 2 bytes
    let out_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(out_size as usize, opts).unwrap())
        .collect();

    // Warmup: 32 separate CBs, each with 1 dispatch to its own output buffer
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .map(|out_buf| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pipeline);
                unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
                set_u32(&enc, 3, bufs.m_val);
                set_u32(&enc, 4, bufs.n_val);
                set_u32(&enc, 5, bufs.k_val);
                set_u32(&enc, 6, bufs.bsa_val);
                set_u32(&enc, 7, bufs.bsb_val);
                set_u32(&enc, 8, bufs.bsc_val);
                set_u32(&enc, 9, bufs.swizzle_val);
                set_u32(&enc, 10, 0u32);
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
                unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
                unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
                set_u32(&enc, 3, bufs.m_val);
                set_u32(&enc, 4, bufs.n_val);
                set_u32(&enc, 5, bufs.k_val);
                set_u32(&enc, 6, bufs.bsa_val);
                set_u32(&enc, 7, bufs.bsb_val);
                set_u32(&enc, 8, bufs.bsc_val);
                set_u32(&enc, 9, bufs.swizzle_val);
                set_u32(&enc, 10, 0u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64); // amortized per dispatch
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Multi-encoder benchmark: 1 CB, 32 encoders (each with 1 dispatch)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_multi_encoder(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    pipeline: &MtlPipeline,
    bufs: &GemmBuffers,
    m: usize,
    n: usize,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    let queue = device.newCommandQueue().unwrap();
    let opts = MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64;
    let out_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(out_size as usize, opts).unwrap())
        .collect();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        for out_buf in &out_bufs {
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pipeline);
            unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
            set_u32(&enc, 3, bufs.m_val);
            set_u32(&enc, 4, bufs.n_val);
            set_u32(&enc, 5, bufs.k_val);
            set_u32(&enc, 6, bufs.bsa_val);
            set_u32(&enc, 7, bufs.bsb_val);
            set_u32(&enc, 8, bufs.bsc_val);
            set_u32(&enc, 9, bufs.swizzle_val);
            set_u32(&enc, 10, 0u32);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        for out_buf in &out_bufs {
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pipeline);
            unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
            unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
            set_u32(&enc, 3, bufs.m_val);
            set_u32(&enc, 4, bufs.n_val);
            set_u32(&enc, 5, bufs.k_val);
            set_u32(&enc, 6, bufs.bsa_val);
            set_u32(&enc, 7, bufs.bsb_val);
            set_u32(&enc, 8, bufs.bsc_val);
            set_u32(&enc, 9, bufs.swizzle_val);
            set_u32(&enc, 10, 0u32);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Single-encoder benchmark: 1 CB, 1 encoder, 32 dispatches
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_single_encoder(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    pipeline: &MtlPipeline,
    bufs: &GemmBuffers,
    m: usize,
    n: usize,
    grid: MTLSize,
    tg: MTLSize,
) -> f64 {
    let queue = device.newCommandQueue().unwrap();
    let opts = MTLResourceOptions::StorageModeShared;
    let out_size = (m * n * 2) as u64;
    let out_bufs: Vec<MtlBuffer> = (0..PIPELINE_N)
        .map(|_| device.newBufferWithLength_options(out_size as usize, opts).unwrap())
        .collect();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pipeline);
        // Set shared buffers once
        unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
        set_u32(&enc, 3, bufs.m_val);
        set_u32(&enc, 4, bufs.n_val);
        set_u32(&enc, 5, bufs.k_val);
        set_u32(&enc, 6, bufs.bsa_val);
        set_u32(&enc, 7, bufs.bsb_val);
        set_u32(&enc, 8, bufs.bsc_val);
        set_u32(&enc, 9, bufs.swizzle_val);
        set_u32(&enc, 10, 0u32);
        for out_buf in &out_bufs {
            unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pipeline);
        unsafe { enc.setBuffer_offset_atIndex(Some(bufs.a.metal_buffer()), 0_usize, 0_usize) };
        unsafe { enc.setBuffer_offset_atIndex(Some(bufs.b.metal_buffer()), 0_usize, 1_usize) };
        set_u32(&enc, 3, bufs.m_val);
        set_u32(&enc, 4, bufs.n_val);
        set_u32(&enc, 5, bufs.k_val);
        set_u32(&enc, 6, bufs.bsa_val);
        set_u32(&enc, 7, bufs.bsb_val);
        set_u32(&enc, 8, bufs.bsc_val);
        set_u32(&enc, 9, bufs.swizzle_val);
        set_u32(&enc, 10, 0u32);
        for out_buf in &out_bufs {
            unsafe { enc.setBuffer_offset_atIndex(Some(out_buf), 0_usize, 2_usize) };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Production benchmark: matmul_into_cb() × 32 in 1 CB (ExecGraph pattern)
// ---------------------------------------------------------------------------

/// Production dispatch benchmark: matmul_into_cb() × 32 in 1 CB.
/// Matches forward_graph_unified(Prefill) ExecGraph pattern where multiple
/// matmuls share a single command buffer.
fn bench_production(
    registry: &KernelRegistry,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    a: &Array,
    b: &Array,
) -> f64 {
    let queue = device.newCommandQueue().unwrap();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBuffer().unwrap();
        for _ in 0..PIPELINE_N {
            let _ = ops::matmul::matmul_into_cb(registry, a, b, &cb).unwrap();
        }
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Measure
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.commandBuffer().unwrap();
        for _ in 0..PIPELINE_N {
            let _ = ops::matmul::matmul_into_cb(registry, a, b, &cb).unwrap();
        }
        cb.commit();
        cb.waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }
    p50(&mut times)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct BenchResult {
    m: usize,
    n: usize,
    k: usize,
    shape_name: &'static str,
    kernel: &'static str,
    sync_us: f64,
    pipe_us: f64,
    multi_enc_us: f64,
    single_enc_us: f64,
    prod_us: f64,
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();

    println!("=== RMLX GEMM Fair Benchmark (Dispatch Overhead Analysis) ===");
    println!(
        "dtype=f16, Warmup={}, Bench={} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!(
        "Pipeline depth: {} dispatches (simulating 32-layer transformer)",
        PIPELINE_N
    );
    println!("Modes: Sync (1 CB/wait), Pipe (32 CBs/1 wait), Multi-enc (1 CB/32 enc), Single-enc (1 CB/1 enc/32 disp), Prod (1 CB/32 matmul_into_cb)");

    // Collect results across all shapes
    let mut all_results: Vec<BenchResult> = Vec::new();

    for &(shape_n, shape_k, shape_name) in NK_SHAPES {
        println!();
        println!(
            "=== Shape: {} (N={}, K={}) ===",
            shape_name, shape_n, shape_k
        );

        // Run largest M first for thermal fairness
        let mut shape_results: Vec<BenchResult> = Vec::new();

        for &m in M_VALUES.iter().rev() {
            let spec = select_kernel_for_m(m, shape_n, shape_k);
            let constants =
                ops::matmul::matmul_align_constants(m, shape_n, shape_k, spec.bm, spec.bn, spec.bk);
            let pipeline = registry
                .get_pipeline_with_constants(spec.name, DType::Float16, &constants)
                .unwrap_or_else(|e| panic!("Failed to get pipeline for {}: {e}", spec.name));

            let bufs = GemmBuffers::new(device, m, shape_n, shape_k, spec.bm, spec.bn);

            let tiles_n = shape_n.div_ceil(spec.bn);
            let tiles_m = m.div_ceil(spec.bm);
            let swizzle = bufs.swizzle_val;
            let grid_x = (tiles_n << swizzle) as u64;
            let grid_y = (tiles_m >> swizzle) as u64;
            let grid = MTLSize { width: grid_x as usize, height: grid_y as usize, depth: 1_usize };
            let tg = MTLSize { width: spec.threads as usize, height: 1_usize, depth: 1_usize };

            println!(
                "  [DEBUG] M={} kernel={} grid=({},{},{}) tg=({},{},{}) bm={} bn={} bk={} swizzle={}",
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
                spec.bk,
                swizzle
            );

            let sync_us = bench_sync(device, &pipeline, &bufs, grid, tg);
            let pipe_us = bench_pipelined(device, &pipeline, &bufs, m, shape_n, grid, tg);
            let multi_enc_us = bench_multi_encoder(device, &pipeline, &bufs, m, shape_n, grid, tg);
            let single_enc_us =
                bench_single_encoder(device, &pipeline, &bufs, m, shape_n, grid, tg);

            // Production dispatch: uses ops::matmul::matmul() with full routing
            let prod_a = rand_f16_array(device, &[m, shape_k], 42);
            let prod_b = rand_f16_array(device, &[shape_k, shape_n], 99);
            let prod_us = bench_production(&registry, device, &prod_a, &prod_b);

            println!(
                "  M={:>4}: {} sync={:.1}us pipe={:.1}us multi_enc={:.1}us single_enc={:.1}us prod={:.1}us",
                m, spec.name, sync_us, pipe_us, multi_enc_us, single_enc_us, prod_us
            );

            shape_results.push(BenchResult {
                m,
                n: shape_n,
                k: shape_k,
                shape_name,
                kernel: spec.name,
                sync_us,
                pipe_us,
                multi_enc_us,
                single_enc_us,
                prod_us,
            });
        }

        // Sort ascending by M for display
        shape_results.sort_by_key(|r| r.m);

        // Per-shape results table
        println!();
        println!(
            "--- {} (N={}, K={}) Results ---",
            shape_name, shape_n, shape_k
        );
        println!(
            "| {:>5} | {:>22} | {:>12} | {:>12} | {:>14} | {:>16} | {:>12} |",
            "M",
            "Kernel",
            "Sync (us/T)",
            "Pipe (us/T)",
            "Multi-enc (us/T)",
            "Single-enc (us/T)",
            "Prod (us/T)"
        );
        println!(
            "|{:-<7}|{:-<24}|{:-<14}|{:-<14}|{:-<16}|{:-<18}|{:-<14}|",
            "", "", "", "", "", "", ""
        );

        for r in &shape_results {
            let sync_t = tflops(r.m, r.n, r.k, r.sync_us);
            let pipe_t = tflops(r.m, r.n, r.k, r.pipe_us);
            let menc_t = tflops(r.m, r.n, r.k, r.multi_enc_us);
            let senc_t = tflops(r.m, r.n, r.k, r.single_enc_us);
            let prod_t = tflops(r.m, r.n, r.k, r.prod_us);
            println!(
                "| {:>5} | {:>22} | {:>5.1}/{:<5.2} | {:>5.1}/{:<5.2} | {:>7.1}/{:<5.2} | {:>9.1}/{:<5.2} | {:>5.1}/{:<5.2} |",
                r.m, r.kernel,
                r.sync_us, sync_t,
                r.pipe_us, pipe_t,
                r.multi_enc_us, menc_t,
                r.single_enc_us, senc_t,
                r.prod_us, prod_t,
            );
        }

        all_results.extend(shape_results);
    }

    // -----------------------------------------------------------------------
    // Combined Pipelined Summary Table (all shapes × all M values)
    // -----------------------------------------------------------------------
    println!();
    println!("=== Pipelined Summary ({}x, p50) ===", PIPELINE_N);
    println!();
    print!("| {:>5} |", "M");
    for &(shape_n, shape_k, label) in NK_SHAPES {
        let col_label = format!("{} {}x{}", label, shape_n, shape_k);
        print!(" {:>18} |", col_label);
    }
    println!();
    print!("|-------|");
    for _ in NK_SHAPES {
        print!("--------------------|");
    }
    println!();

    for &m in M_VALUES {
        print!("| {:>5} |", m);
        for &(shape_n, shape_k, shape_name) in NK_SHAPES {
            if let Some(r) = all_results.iter().find(|r| {
                r.m == m && r.n == shape_n && r.k == shape_k && r.shape_name == shape_name
            }) {
                let tf = tflops(m, shape_n, shape_k, r.pipe_us);
                print!(" {:>11.1}us/{:.2}T |", r.pipe_us, tf);
            } else {
                print!(" {:>18} |", "N/A");
            }
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Combined Production Summary Table (all shapes × all M values)
    // -----------------------------------------------------------------------
    println!();
    println!(
        "=== Production Dispatch Summary ({}x matmul_into_cb() batched, p50) ===",
        PIPELINE_N
    );
    println!();
    print!("| {:>5} |", "M");
    for &(shape_n, shape_k, label) in NK_SHAPES {
        let col_label = format!("{} {}x{}", label, shape_n, shape_k);
        print!(" {:>18} |", col_label);
    }
    println!();
    print!("|-------|");
    for _ in NK_SHAPES {
        print!("--------------------|");
    }
    println!();

    for &m in M_VALUES {
        print!("| {:>5} |", m);
        for &(shape_n, shape_k, shape_name) in NK_SHAPES {
            if let Some(r) = all_results.iter().find(|r| {
                r.m == m && r.n == shape_n && r.k == shape_k && r.shape_name == shape_name
            }) {
                let tf = tflops(m, shape_n, shape_k, r.prod_us);
                print!(" {:>11.1}us/{:.2}T |", r.prod_us, tf);
            } else {
                print!(" {:>18} |", "N/A");
            }
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Dispatch Overhead Analysis (first shape only, representative)
    // -----------------------------------------------------------------------
    let first_shape = NK_SHAPES[0];
    let first_results: Vec<&BenchResult> = all_results
        .iter()
        .filter(|r| r.n == first_shape.0 && r.k == first_shape.1)
        .collect();

    if !first_results.is_empty() {
        println!();
        println!(
            "=== Dispatch Overhead Analysis — {} (N={}, K={}) ===",
            first_shape.2, first_shape.0, first_shape.1
        );
        println!("(us per dispatch, amortized over {})", PIPELINE_N);
        println!();
        println!(
            "| {:>5} | {:>22} | {:>12} | {:>12} | {:>12} | {:>12} |",
            "M", "Kernel", "CB overhead", "Enc overhead", "GPU pure", "Sync total"
        );
        println!(
            "|{:-<7}|{:-<24}|{:-<14}|{:-<14}|{:-<14}|{:-<14}|",
            "", "", "", "", "", ""
        );

        for r in &first_results {
            let cb_overhead = r.pipe_us - r.multi_enc_us;
            let enc_overhead = r.multi_enc_us - r.single_enc_us;
            let gpu_pure = r.single_enc_us;
            let sync_total = r.sync_us - r.single_enc_us;

            println!(
                "| {:>5} | {:>22} | {:>9.1} us | {:>9.1} us | {:>9.1} us | {:>9.1} us |",
                r.m, r.kernel, cb_overhead, enc_overhead, gpu_pure, sync_total,
            );
        }

        println!();
        println!("Legend:");
        println!("  CB overhead   = Pipe - Multi-enc   (per-CB creation + commit cost)");
        println!("  Enc overhead  = Multi-enc - Single-enc (per-encoder creation cost)");
        println!("  GPU pure      = Single-enc          (minimum achievable dispatch time)");
        println!("  Sync total    = Sync - Single-enc   (total CPU overhead in sync mode)");
    }

    println!();
    println!("Done.");
}
