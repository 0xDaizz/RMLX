//! ⚠️ NON-PRODUCTION PATH — NAX vs Steel kernel transition point analysis.
//! Direct kernel encoding, development only. Bypasses quantized_matmul() dispatch.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! NAX vs Steel Q4 QMM Transition Benchmark
//!
//! Systematically compares NAX (4SG, BK=64) vs Steel (2SG, BK=32)
//! across MoE-relevant dimensions to identify transition areas.
//!
//! For M >= 32: runs both NAX and Steel directly, reports speedup ratio.
//! For M < 32:  runs dispatch path only as baseline reference.
//!
//! Usage: cargo bench -p rmlx-core --bench nax_transition_bench

use std::time::{Duration, Instant};

use half::f16;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;
use std::ptr::NonNull;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _, MTLCommandQueue as _, MTLCommandBuffer as _};
use rmlx_metal::{MTLResourceOptions};

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
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
        Stats { mean, std_dev, p50 }
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
// Random data generation
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

fn rand_f32_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut state = seed;
    let data: Vec<f32> = (0..numel)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.1
        })
        .collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, numel * 4) };
    Array::from_bytes(device, bytes, shape.to_vec(), DType::Float32)
}

fn make_quantized_weight(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: u32,
    seed: u64,
) -> QuantizedWeight {
    let mut state = seed;
    let opts = MTLResourceOptions::StorageModeShared;

    let elems_per_u32 = 32 / bits as usize;
    let total_elements = out_features * in_features;
    let num_u32s = total_elements.div_ceil(elems_per_u32);
    let w_data: Vec<u32> = (0..num_u32s)
        .map(|_| {
            let v = lcg_next(&mut state);
            v as u32
        })
        .collect();
    let weights_buf =
        unsafe { device.newBufferWithBytes_length_options(NonNull::new(w_data.as_ptr() as *const _ as *mut _).unwrap(), (num_u32s * 4) as u64 as usize, opts).unwrap() };

    let num_groups = total_elements / group_size as usize;
    let scales_data: Vec<u16> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            let val = ((v >> 33) as f64 / (1u64 << 31) as f64) as f32 * 0.02 + 0.001;
            f16::from_f32(val).to_bits()
        })
        .collect();
    let biases_data: Vec<u16> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            let val = ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.01;
            f16::from_f32(val).to_bits()
        })
        .collect();

    let scales_buf = unsafe { device.newBufferWithBytes_length_options(NonNull::new(scales_data.as_ptr() as *const _ as *mut _).unwrap(), (num_groups * 2) as u64 as usize, opts).unwrap() };
    let biases_buf = unsafe { device.newBufferWithBytes_length_options(NonNull::new(biases_data.as_ptr() as *const _ as *mut _).unwrap(), (num_groups * 2) as u64 as usize, opts).unwrap() };

    QuantizedWeight::new(
        weights_buf,
        scales_buf,
        biases_buf,
        group_size,
        bits,
        out_features,
        in_features,
    )
    .expect("Failed to create QuantizedWeight")
}

// ---------------------------------------------------------------------------
// TFLOPS calculation
// ---------------------------------------------------------------------------

fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Benchmark runners
// ---------------------------------------------------------------------------

/// Benchmark NAX kernel (f16 input, group_size=64).
/// Returns Stats for the run.
fn bench_nax(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 4, 64, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, &cb)
            .expect("NAX warmup failed");
        cb.commit();
        cb.waitUntilCompleted();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, &cb)
            .expect("NAX bench failed");
        cb.commit();
        cb.waitUntilCompleted();
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// Benchmark Steel kernel (f16 input, group_size=32).
/// Returns Stats for the run.
fn bench_steel(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 4, 32, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("Steel warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("Steel bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

/// Benchmark dispatch path (for M < 32 baseline).
/// Returns Stats for the run.
fn bench_dispatch(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 4, 32, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Dispatch warmup failed");
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("Dispatch bench failed");
        times.push(start.elapsed());
    }

    Stats::from_durations(&times)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.newCommandQueue().unwrap();

    let k_values: &[usize] = &[2048, 2560, 3072, 3584, 4096, 5120];
    let n_values: &[usize] = &[1536, 2048, 2560, 3584, 4096];
    let m_values: &[usize] = &[4, 8, 16, 32, 64, 128, 256];

    println!("=== NAX vs Steel Q4 QMM Transition Benchmark ===");
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS,
    );
    println!("NAX: 4SG/128threads, BK=64, group_size=64, f16 input");
    println!("Steel: 2SG/64threads, BK=32, group_size=32, f16 input");
    println!("Dispatch: auto-selected kernel via affine_quantized_matmul_batched");
    println!();

    // Collect transition points for summary
    let mut transitions: Vec<(usize, usize, Option<usize>)> = Vec::new();

    for &k in k_values {
        for &n in n_values {
            println!("--- K={:<5} N={:<5} (K%64={}) ---", k, n, k % 64,);
            println!(
                "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
                "M",
                "NAX p50us",
                "NAX T",
                "Steel p50us",
                "Steel T",
                "Disp p50us",
                "Disp T",
                "NAX/Stl",
            );

            let mut transition_m: Option<usize> = None;

            for &m in m_values {
                if m < 32 {
                    // M < 32: dispatch path only (NAX not used at small M)
                    let disp = bench_dispatch(&registry, &queue, device, m, n, k);
                    let disp_tf = tflops(m, n, k, disp.p50);
                    println!(
                        "  {:>5}  {:>10}  {:>10.3}  {:>10}  {:>10}  {:>10.1}  {:>10.3}  {:>8}",
                        m, "-", "-", "-", "-", disp.p50, disp_tf, "-",
                    );
                } else {
                    // M >= 32: head-to-head NAX vs Steel
                    let nax = bench_nax(&registry, &queue, device, m, n, k);
                    let steel = bench_steel(&registry, &queue, device, m, n, k);
                    let disp = bench_dispatch(&registry, &queue, device, m, n, k);

                    let nax_tf = tflops(m, n, k, nax.p50);
                    let steel_tf = tflops(m, n, k, steel.p50);
                    let disp_tf = tflops(m, n, k, disp.p50);
                    let speedup = steel.p50 / nax.p50;

                    let marker = if speedup > 1.0 { " <NAX" } else { "" };

                    println!(
                        "  {:>5}  {:>10.1}  {:>10.3}  {:>10.1}  {:>10.3}  {:>10.1}  {:>10.3}  {:>7.2}x{}",
                        m, nax.p50, nax_tf, steel.p50, steel_tf, disp.p50, disp_tf, speedup, marker,
                    );

                    // Track transition: first M where NAX becomes faster
                    if transition_m.is_none() && speedup > 1.0 {
                        transition_m = Some(m);
                    }
                }
            }

            transitions.push((k, n, transition_m));
            println!();
        }
    }

    // =========================================================================
    // Summary: transition points
    // =========================================================================
    println!("=== Transition Summary (first M where NAX > Steel) ===");
    println!("  {:>5}  {:>5}  {:>12}", "K", "N", "Transition M");
    for &(k, n, t_m) in &transitions {
        let label = match t_m {
            Some(m) => format!("M={}", m),
            None => "Steel always".to_string(),
        };
        println!("  {:>5}  {:>5}  {:>12}", k, n, label);
    }
    println!();
    println!("Done.");
}
