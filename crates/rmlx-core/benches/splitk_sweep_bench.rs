//! ⚠️ NON-PRODUCTION PATH — tests Split-K through direct dispatch, not through matmul().
//! Useful for n_splits tuning only.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Split-K n_splits sweep benchmark.
//!
//! For each (M, N, K) combination, sweeps n_splits values for both
//! SplitK-Small (BM=32, BN=32, BK=16) and SplitK-MLX (BM=64, BN=64, BK=16)
//! kernels to find the optimal split count.
//!
//! Usage: cargo bench -p rmlx-core --bench splitk_sweep_bench

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
use rmlx_metal::types::{MtlBuffer};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const PIPELINE_N: usize = 32;

// M values (largest-first for thermal fairness)
const M_VALUES: &[usize] = &[128, 64, 32, 16, 8, 4, 2];

// (N, K) shape pairs
const SHAPES: &[(usize, usize)] = &[(3584, 3584), (4096, 4096), (14336, 4096), (4096, 14336)];

// n_splits for SplitK-Small (BM=32, BN=32, BK=16)
const SMALL_SPLITS: &[usize] = &[2, 3, 4, 6, 8, 12, 16];
// n_splits for SplitK-MLX (BM=64, BN=64, BK=16)
const MLX_SPLITS: &[usize] = &[2, 3, 4, 5, 6, 8, 12, 16];

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
// Pipelined Split-K benchmark (parameterized n_splits)
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
    pass1_threads: u64,
) -> Option<f64> {
    // Validate: n_splits must not exceed K/BK tiles
    let k_tiles = k / splitk_bk;
    if n_splits > k_tiles || n_splits < 2 {
        return None;
    }

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
    let pass1_pipeline =
        match registry.get_pipeline_with_constants(pass1_kernel, DType::Float16, &constants) {
            Ok(p) => p,
            Err(_) => return None,
        };
    let pass2_pipeline = match registry.get_pipeline("splitk_reduce_f16", DType::Float16) {
        Ok(p) => p,
        Err(_) => return None,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let splits_u32 = n_splits as u32;
    let swizzle_log = ops::matmul::compute_swizzle_log(m, n, splitk_bm, splitk_bn);

    let pass1_grid = MTLSize { width: (n.div_ceil(splitk_bn) << swizzle_log) as usize, height: (m.div_ceil(splitk_bm) >> swizzle_log) as usize, depth: n_splits };
    let pass1_tg = MTLSize { width: pass1_threads as usize, height: 1_usize, depth: 1_usize };

    let total_elems = m * n;
    let reduce_tg_size = 256u64;
    let reduce_groups = total_elems.div_ceil(reduce_tg_size as usize) as u64;
    let reduce_grid = MTLSize { width: reduce_groups as usize, height: 1_usize, depth: 1_usize };
    let reduce_tg = MTLSize { width: reduce_tg_size as usize, height: 1_usize, depth: 1_usize };

    let encode_dispatch =
        |cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>, out_buf: &MtlBuffer, partial_buf: &MtlBuffer| {
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
        };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cbs: Vec<_> = out_bufs
            .iter()
            .zip(partial_bufs.iter())
            .map(|(out_buf, partial_buf)| {
                let cb = queue.commandBufferWithUnretainedReferences().unwrap();
                encode_dispatch(&cb, out_buf, partial_buf);
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
                encode_dispatch(&cb, out_buf, partial_buf);
                cb.commit();
                cb
            })
            .collect();
        cbs.last().unwrap().waitUntilCompleted();
        let total_us = start.elapsed().as_secs_f64() * 1e6;
        times.push(total_us / PIPELINE_N as f64);
    }

    Some(p50(&mut times))
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

struct SweepResult {
    m: usize,
    n: usize,
    k: usize,
    kernel_label: &'static str,
    n_splits: usize,
    latency_us: f64,
}

struct BestResult {
    m: usize,
    n: usize,
    k: usize,
    kernel_label: &'static str,
    best_n: usize,
    best_us: f64,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();

    println!("=== RMLX Split-K n_splits Sweep Benchmark ===");
    println!(
        "dtype=f16, Pipeline={} CBs, Warmup={}, Bench={} iters",
        PIPELINE_N, WARMUP_ITERS, BENCH_ITERS
    );
    println!("M values: {:?}", M_VALUES);
    println!(
        "Shapes: {}",
        SHAPES
            .iter()
            .map(|(n, k)| format!("{}x{}", n, k))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("SplitK-Small splits: {:?}", SMALL_SPLITS);
    println!("SplitK-MLX splits:   {:?}", MLX_SPLITS);
    println!();

    let mut all_results: Vec<SweepResult> = Vec::new();
    let mut best_results: Vec<BestResult> = Vec::new();

    // Total combos for progress
    let total_combos = M_VALUES.len() * SHAPES.len() * (SMALL_SPLITS.len() + MLX_SPLITS.len());
    let mut done = 0usize;

    // Run largest M first (already ordered in M_VALUES)
    for &m in M_VALUES {
        for &(n, k) in SHAPES {
            println!("--- M={} N={} K={} ---", m, n, k);

            let a = rand_f16_array(device, &[m, k], 42);
            let b = rand_f16_array(device, &[k, n], 99);

            // SplitK-Small: BM=32, BN=32, BK=16, 64 threads
            {
                let bm = 32usize;
                let bn = 32usize;
                let bk = 16usize;
                let label = "SplitK-Small";
                let mut row_latencies: Vec<(usize, f64)> = Vec::new();

                for &ns in SMALL_SPLITS {
                    let result = bench_pipelined_splitk(
                        device,
                        &registry,
                        &a,
                        &b,
                        m,
                        n,
                        k,
                        bm,
                        bn,
                        bk,
                        "splitk_small_pass1_f16",
                        ns,
                        64,
                    );
                    if let Some(us) = result {
                        row_latencies.push((ns, us));
                        all_results.push(SweepResult {
                            m,
                            n,
                            k,
                            kernel_label: label,
                            n_splits: ns,
                            latency_us: us,
                        });
                    }
                    done += 1;
                }

                // Print row
                if !row_latencies.is_empty() {
                    print!("  {:<14}", label);
                    for &(ns, us) in &row_latencies {
                        print!("  n={}: {:.1}us", ns, us);
                    }
                    let best = row_latencies
                        .iter()
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();
                    println!("  -> BEST n={} {:.1}us", best.0, best.1);

                    best_results.push(BestResult {
                        m,
                        n,
                        k,
                        kernel_label: label,
                        best_n: best.0,
                        best_us: best.1,
                    });
                }
            }

            // SplitK-MLX: BM=64, BN=64, BK=16, 64 threads
            {
                let bm = 64usize;
                let bn = 64usize;
                let bk = 16usize;
                let label = "SplitK-MLX";
                let mut row_latencies: Vec<(usize, f64)> = Vec::new();

                for &ns in MLX_SPLITS {
                    let result = bench_pipelined_splitk(
                        device,
                        &registry,
                        &a,
                        &b,
                        m,
                        n,
                        k,
                        bm,
                        bn,
                        bk,
                        "splitk_pass1_mlx_f16",
                        ns,
                        64,
                    );
                    if let Some(us) = result {
                        row_latencies.push((ns, us));
                        all_results.push(SweepResult {
                            m,
                            n,
                            k,
                            kernel_label: label,
                            n_splits: ns,
                            latency_us: us,
                        });
                    }
                    done += 1;
                }

                // Print row
                if !row_latencies.is_empty() {
                    print!("  {:<14}", label);
                    for &(ns, us) in &row_latencies {
                        print!("  n={}: {:.1}us", ns, us);
                    }
                    let best = row_latencies
                        .iter()
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();
                    println!("  -> BEST n={} {:.1}us", best.0, best.1);

                    best_results.push(BestResult {
                        m,
                        n,
                        k,
                        kernel_label: label,
                        best_n: best.0,
                        best_us: best.1,
                    });
                }
            }

            println!("  [{}/{} combos done]", done, total_combos);
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Per-cell detailed table
    // -----------------------------------------------------------------------
    println!();
    println!("=== Detailed Results: SplitK-Small (BM=32, BN=32, BK=16) ===");
    println!();
    print!("| {:>3} | {:>10} |", "M", "NxK");
    for &ns in SMALL_SPLITS {
        print!(" {:>7} |", format!("n={}", ns));
    }
    println!(" {:>6} | {:>8} |", "Best n", "Best us");

    print!("|{:-<5}|{:-<12}|", "", "");
    for _ in SMALL_SPLITS {
        print!("{:-<9}|", "");
    }
    println!("{:-<8}|{:-<10}|", "", "");

    for &m in M_VALUES {
        for &(n, k) in SHAPES {
            print!("| {:>3} | {:>5}x{:<4} |", m, n, k);
            let mut best_ns = 0usize;
            let mut best_us = f64::INFINITY;
            for &ns in SMALL_SPLITS {
                let entry = all_results.iter().find(|r| {
                    r.m == m
                        && r.n == n
                        && r.k == k
                        && r.kernel_label == "SplitK-Small"
                        && r.n_splits == ns
                });
                if let Some(r) = entry {
                    print!(" {:>5.1}us |", r.latency_us);
                    if r.latency_us < best_us {
                        best_us = r.latency_us;
                        best_ns = ns;
                    }
                } else {
                    print!(" {:>7} |", "---");
                }
            }
            if best_ns > 0 {
                println!(" {:>6} | {:>6.1}us |", best_ns, best_us);
            } else {
                println!(" {:>6} | {:>8} |", "---", "---");
            }
        }
    }

    println!();
    println!("=== Detailed Results: SplitK-MLX (BM=64, BN=64, BK=16) ===");
    println!();
    print!("| {:>3} | {:>10} |", "M", "NxK");
    for &ns in MLX_SPLITS {
        print!(" {:>7} |", format!("n={}", ns));
    }
    println!(" {:>6} | {:>8} |", "Best n", "Best us");

    print!("|{:-<5}|{:-<12}|", "", "");
    for _ in MLX_SPLITS {
        print!("{:-<9}|", "");
    }
    println!("{:-<8}|{:-<10}|", "", "");

    for &m in M_VALUES {
        for &(n, k) in SHAPES {
            print!("| {:>3} | {:>5}x{:<4} |", m, n, k);
            let mut best_ns = 0usize;
            let mut best_us = f64::INFINITY;
            for &ns in MLX_SPLITS {
                let entry = all_results.iter().find(|r| {
                    r.m == m
                        && r.n == n
                        && r.k == k
                        && r.kernel_label == "SplitK-MLX"
                        && r.n_splits == ns
                });
                if let Some(r) = entry {
                    print!(" {:>5.1}us |", r.latency_us);
                    if r.latency_us < best_us {
                        best_us = r.latency_us;
                        best_ns = ns;
                    }
                } else {
                    print!(" {:>7} |", "---");
                }
            }
            if best_ns > 0 {
                println!(" {:>6} | {:>6.1}us |", best_ns, best_us);
            } else {
                println!(" {:>6} | {:>8} |", "---", "---");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Summary heatmap: best n_splits per (M, Shape) for each kernel
    // -----------------------------------------------------------------------
    println!();
    println!("=== Summary Heatmap: Best n_splits ===");
    println!();

    for kernel_label in &["SplitK-Small", "SplitK-MLX"] {
        println!("--- {} ---", kernel_label);
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

        for &m in M_VALUES {
            print!("| {:>5} |", m);
            for &(n, k) in SHAPES {
                let best = best_results
                    .iter()
                    .find(|r| r.m == m && r.n == n && r.k == k && r.kernel_label == *kernel_label);
                if let Some(b) = best {
                    let t = tflops(m, n, k, b.best_us);
                    print!(" n={:<2} {:>5.1}us {:>4.2}T |", b.best_n, b.best_us, t);
                } else {
                    print!(" {:>14} |", "---");
                }
            }
            println!();
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Cross-kernel best per cell
    // -----------------------------------------------------------------------
    println!("=== Overall Best Split-K per (M, Shape) ===");
    println!();
    print!("| {:>5} |", "M");
    for &(n, k) in SHAPES {
        print!(" {:>18} |", format!("{}x{}", n, k));
    }
    println!();

    print!("|{:-<7}|", "");
    for _ in SHAPES {
        print!("{:-<20}|", "");
    }
    println!();

    for &m in M_VALUES {
        print!("| {:>5} |", m);
        for &(n, k) in SHAPES {
            let best = best_results
                .iter()
                .filter(|r| r.m == m && r.n == n && r.k == k)
                .min_by(|a, b| a.best_us.partial_cmp(&b.best_us).unwrap());
            if let Some(b) = best {
                let t = tflops(m, n, k, b.best_us);
                let short_label = if b.kernel_label == "SplitK-Small" {
                    "Sm"
                } else {
                    "Mx"
                };
                print!(
                    " {} n={:<2} {:>5.1}us {:.2}T |",
                    short_label, b.best_n, b.best_us, t
                );
            } else {
                print!(" {:>18} |", "---");
            }
        }
        println!();
    }

    println!();
    println!(
        "Done. {} results across {} combos.",
        all_results.len(),
        total_combos
    );
}
