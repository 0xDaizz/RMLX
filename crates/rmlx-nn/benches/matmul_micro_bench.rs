//! ⚠️ NON-PRODUCTION PATH — matmul micro-benchmark with per-op sync CB via matmul_into_cb().
//! Isolates GEMM from other ops but includes CB overhead. Not production dispatch path.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Pure matmul microbenchmark for RMLX.
//!
//! Tests `ops::matmul::matmul_into_cb()` with Qwen 3.5 MoE shapes,
//! isolating GEMM performance from all other ops.
//!
//! Shapes (Qwen 3.5 MoE config):
//!   QKV:    [M, 3584] x [3584, 4608]
//!   O_proj: [M, 3584] x [3584, 3584]
//!   GateUp: [M, 3584] x [3584, 5120]
//!   Down:   [M, 2560] x [2560, 3584]
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench matmul_micro_bench

use std::time::{Duration, Instant};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::autoreleasepool;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

const M_VALUES: &[usize] = &[128, 256, 512, 1024];

struct GemmShape {
    name: &'static str,
    k: usize,
    n: usize,
}

const SHAPES: &[GemmShape] = &[
    GemmShape {
        name: "QKV",
        k: 3584,
        n: 4608,
    },
    GemmShape {
        name: "O_proj",
        k: 3584,
        n: 3584,
    },
    GemmShape {
        name: "GateUp",
        k: 3584,
        n: 5120,
    },
    GemmShape {
        name: "Down",
        k: 2560,
        n: 3584,
    },
];

// ---------------------------------------------------------------------------
// f16 helpers (same as op_profile_bench)
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

fn rand_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let f16_bytes = rand_f16_bytes(numel, seed);
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// CB status validation
// ---------------------------------------------------------------------------

fn assert_cb_ok(cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>, context: &str) {
    let status = cb.status();
    assert!(
        status != objc2_metal::MTLCommandBufferStatus::Error,
        "GPU command buffer error in {context}: status={status:?}"
    );
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

fn time_op<F>(queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>, f: F) -> Duration
where
    F: FnOnce(&ProtocolObject<dyn objc2_metal::MTLCommandBuffer>),
{
    autoreleasepool(|_| {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let start = Instant::now();
        f(&cb);
        cb.commit();
        cb.waitUntilCompleted();
        let elapsed = start.elapsed();
        assert_cb_ok(&cb, "time_op");
        elapsed
    })
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct BenchResult {
    name: String,
    m: usize,
    _k: usize,
    _n: usize,
    mean_us: f64,
    std_us: f64,
    min_us: f64,
    p50_us: f64,
    tflops: f64,
}

fn compute_stats(
    durations: &[Duration],
    m: usize,
    k: usize,
    n: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = micros.iter().sum();
    let mean = sum / micros.len() as f64;
    let min = micros[0];
    let p50 = if micros.len() % 2 == 0 {
        (micros[micros.len() / 2 - 1] + micros[micros.len() / 2]) / 2.0
    } else {
        micros[micros.len() / 2]
    };

    let variance: f64 =
        micros.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / micros.len() as f64;
    let std_dev = variance.sqrt();

    // TFLOPS = 2*M*N*K / (mean_us * 1e-6) / 1e12
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (mean * 1e-6) / 1e12;

    (mean, std_dev, min, p50, tflops)
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

    println!("dtype: Float16");
    println!(
        "Warmup: {} iters, Bench: {} iters per shape",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!();

    // ── JIT Warmup: pre-compile PSOs for all shapes ──
    autoreleasepool(|_| {
        println!("[JIT Warmup] Pre-compiling matmul kernels for all shape variants...");
        let warmup_queue = device.newCommandQueue().unwrap();

        for m in M_VALUES {
            for shape in SHAPES {
                let a = rand_array(device, &[*m, shape.k], 42);
                let b = rand_array(device, &[shape.k, shape.n], 43);
                let cb = warmup_queue.commandBufferWithUnretainedReferences().unwrap();
                let _ =
                    ops::matmul::matmul_into_cb(&registry, &a, &b, &cb).expect("jit warmup matmul");
                cb.commit();
                cb.waitUntilCompleted();
            }
        }
        println!("[JIT Warmup] Done.\n");
    });

    // Let Metal driver settle
    std::thread::sleep(std::time::Duration::from_millis(100));

    // ── Run benchmarks ──
    let mut results: Vec<BenchResult> = Vec::new();

    for &m in M_VALUES {
        let queue = device.newCommandQueue().unwrap();

        for shape in SHAPES {
            let a = rand_array(device, &[m, shape.k], 100 + m as u64);
            let b = rand_array(device, &[shape.k, shape.n], 200 + shape.n as u64);

            // Warmup iterations
            for _ in 0..WARMUP_ITERS {
                time_op(&queue, |cb| {
                    let _ =
                        ops::matmul::matmul_into_cb(&registry, &a, &b, cb).expect("warmup matmul");
                });
            }

            // Bench iterations
            let mut durations = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                durations.push(time_op(&queue, |cb| {
                    let _ =
                        ops::matmul::matmul_into_cb(&registry, &a, &b, cb).expect("bench matmul");
                }));
            }

            let (mean, std_dev, min, p50, tflops) = compute_stats(&durations, m, shape.k, shape.n);

            results.push(BenchResult {
                name: format!(
                    "{} [{}x{}]x[{}x{}]",
                    shape.name, m, shape.k, shape.k, shape.n
                ),
                m,
                _k: shape.k,
                _n: shape.n,
                mean_us: mean,
                std_us: std_dev,
                min_us: min,
                p50_us: p50,
                tflops,
            });
        }

        // Let GPU cool between M sizes
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // ── Print markdown table ──
    println!("## Matmul Microbenchmark Results (Float16)\n");
    println!(
        "| {:<42} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |",
        "Shape", "Mean(us)", "Std(us)", "Min(us)", "P50(us)", "TFLOPS"
    );
    println!(
        "|{:-<44}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<10}|",
        "", "", "", "", "", ""
    );

    let mut current_m = 0;
    for r in &results {
        if r.m != current_m {
            if current_m != 0 {
                // Separator between M groups
                println!(
                    "| {:<42} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |",
                    "", "", "", "", "", ""
                );
            }
            current_m = r.m;
        }
        println!(
            "| {:<42} | {:>10.1} | {:>10.2} | {:>10.1} | {:>10.1} | {:>8.2} |",
            r.name, r.mean_us, r.std_us, r.min_us, r.p50_us, r.tflops
        );
    }

    // ── Summary: per-M totals ──
    println!("\n## Per-M Summary\n");
    println!("| {:>6} | {:>12} | {:>8} |", "M", "Total(us)", "Avg TFLOPS");
    println!("|{:-<8}|{:-<14}|{:-<10}|", "", "", "");

    for &m in M_VALUES {
        let m_results: Vec<&BenchResult> = results.iter().filter(|r| r.m == m).collect();
        let total_us: f64 = m_results.iter().map(|r| r.mean_us).sum();
        let avg_tflops: f64 =
            m_results.iter().map(|r| r.tflops).sum::<f64>() / m_results.len() as f64;
        println!("| {:>6} | {:>12.1} | {:>8.2} |", m, total_us, avg_tflops);
    }
}
