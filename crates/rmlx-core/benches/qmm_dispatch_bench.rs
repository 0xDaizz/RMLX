//! ⚠️ NON-PRODUCTION PATH — Q4 QMM dispatch through quantized_matmul() sync path.
//! Per-op CB overhead included; throughput not representative of production pipelining.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! QMM Dispatch Benchmark — Steel vs NAX vs Auto at multiple M values.
//!
//! Tests Q4 QMM across M = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
//! with three dispatch variants:
//!   1. Auto dispatch — `affine_quantized_matmul_batched` (auto-selects kernel)
//!   2. Steel forced  — `affine_quantized_matmul_steel`
//!   3. NAX forced    — `affine_qmm_nax_q4_into_cb` (M>=32, K%64==0 only)
//!
//! Dimensions tested (realistic model shapes):
//!   - K=4096, N=4096   (attention projections)
//!   - K=4096, N=14336  (FFN gate/up)
//!   - K=7168, N=2048   (DeepSeek expert)
//!
//! MLX reference numbers can be injected via MLX_QMM_REF env var (JSON).
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench qmm_dispatch_bench

use std::time::{Duration, Instant};

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;
use rmlx_metal::MTLResourceOptions;
use std::ptr::NonNull;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
// Random data generation (LCG)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn rand_f16_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
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
    let weights_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(w_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_u32s * 4) as u64 as usize,
                opts,
            )
            .unwrap()
    };

    let num_groups = total_elements / group_size as usize;
    let scales_f32: Vec<f32> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64) as f32 * 0.02 + 0.001
        })
        .collect();
    let biases_f32: Vec<f32> = (0..num_groups)
        .map(|_| {
            let v = lcg_next(&mut state);
            ((v >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32 * 0.01
        })
        .collect();

    // Convert to f16 (native QuantizedWeight format)
    let scales_data: Vec<u16> = scales_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();
    let biases_data: Vec<u16> = biases_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();

    let scales_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(scales_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_groups * 2) as u64 as usize,
                opts,
            )
            .unwrap()
    };
    let biases_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(biases_data.as_ptr() as *const _ as *mut _).unwrap(),
                (num_groups * 2) as u64 as usize,
                opts,
            )
            .unwrap()
    };

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
// Result row
// ---------------------------------------------------------------------------

struct BenchResult {
    kernel: &'static str,
    m: usize,
    n: usize,
    k: usize,
    stats: Option<Stats>,
    skip_reason: Option<&'static str>,
}

impl BenchResult {
    fn ok(kernel: &'static str, m: usize, n: usize, k: usize, stats: Stats) -> Self {
        Self {
            kernel,
            m,
            n,
            k,
            stats: Some(stats),
            skip_reason: None,
        }
    }

    fn skipped(kernel: &'static str, m: usize, n: usize, k: usize, reason: &'static str) -> Self {
        Self {
            kernel,
            m,
            n,
            k,
            stats: None,
            skip_reason: Some(reason),
        }
    }

    fn tflops_p50(&self) -> Option<f64> {
        self.stats
            .as_ref()
            .map(|s| tflops(self.m, self.n, self.k, s.p50))
    }

    fn latency_p50(&self) -> Option<f64> {
        self.stats.as_ref().map(|s| s.p50)
    }
}

// ---------------------------------------------------------------------------
// Kernel bench runners
// ---------------------------------------------------------------------------

/// Auto dispatch — `affine_quantized_matmul_batched` for M>1,
/// `affine_quantized_matmul` (QMV) for M=1.
fn bench_auto(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);

    if m == 1 {
        // QMV path for M=1
        let vec = rand_f16_array(device, &[k], 99);

        for _ in 0..WARMUP_ITERS {
            let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
                .expect("Auto(QMV) warmup failed");
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
                .expect("Auto(QMV) bench failed");
            times.push(start.elapsed());
        }
        BenchResult::ok("Auto(QMV)", m, n, k, Stats::from_durations(&times))
    } else {
        let x = rand_f16_array(device, &[m, k], 99);

        for _ in 0..WARMUP_ITERS {
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
                .expect("Auto warmup failed");
        }

        let mut times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
                .expect("Auto bench failed");
            times.push(start.elapsed());
        }
        BenchResult::ok("Auto", m, n, k, Stats::from_durations(&times))
    }
}

/// Steel forced — `affine_quantized_matmul_steel`.
fn bench_steel(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    // Try warmup — Steel may not support all M values
    for _ in 0..WARMUP_ITERS {
        match ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue) {
            Ok(_) => {}
            Err(_) => return BenchResult::skipped("Steel", m, n, k, "unsupported config"),
        }
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("Steel bench failed");
        times.push(start.elapsed());
    }
    BenchResult::ok("Steel", m, n, k, Stats::from_durations(&times))
}

/// NAX forced — `affine_qmm_nax_q4_into_cb` (M>=32, K%64==0 only, f16 input, native f16 scales).
fn bench_nax(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> BenchResult {
    if m < 32 {
        return BenchResult::skipped("NAX", m, n, k, "requires M>=32");
    }
    if k % 64 != 0 {
        return BenchResult::skipped("NAX", m, n, k, "requires K%64==0");
    }

    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x_f16 = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw, &cb)
            .expect("NAX warmup failed");
        cb.commit();
        cb.waitUntilCompleted();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw, &cb)
            .expect("NAX bench failed");
        cb.commit();
        cb.waitUntilCompleted();
        times.push(start.elapsed());
    }
    BenchResult::ok("NAX", m, n, k, Stats::from_durations(&times))
}

// ---------------------------------------------------------------------------
// MLX reference parsing
// ---------------------------------------------------------------------------

/// Parsed MLX reference entry.
struct MlxRef {
    tflops: f64,
    latency_us: f64,
}

/// Parse MLX_QMM_REF env var as JSON.
/// Expected format: {"M_K_N": {"tflops": X, "latency_us": Y}, ...}
fn parse_mlx_ref() -> std::collections::HashMap<String, MlxRef> {
    let mut map = std::collections::HashMap::new();
    let json_str = match std::env::var("MLX_QMM_REF") {
        Ok(s) if !s.is_empty() => s,
        _ => return map,
    };

    // Minimal JSON parsing without serde dependency.
    // Parse top-level object entries like: "key": {"tflops": X, "latency_us": Y}
    let trimmed = json_str
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}');
    // Split on "}, " to get each entry
    for entry in trimmed.split("},") {
        let entry = entry.trim().trim_end_matches('}');
        // Find key
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = parts[0]
            .trim()
            .trim_matches('"')
            .trim_matches(',')
            .trim()
            .trim_matches('"')
            .to_string();
        let val_str = parts[1].trim().trim_start_matches('{');

        // Parse tflops and latency_us from inner object
        let mut tf = 0.0_f64;
        let mut lat = 0.0_f64;
        for field in val_str.split(',') {
            let field = field.trim();
            if field.contains("tflops") {
                if let Some(v) = field.split(':').nth(1) {
                    tf = v.trim().trim_matches('"').parse().unwrap_or(0.0);
                }
            } else if field.contains("latency_us") {
                if let Some(v) = field.split(':').nth(1) {
                    lat = v.trim().trim_matches('"').parse().unwrap_or(0.0);
                }
            }
        }
        if tf > 0.0 || lat > 0.0 {
            map.insert(
                key,
                MlxRef {
                    tflops: tf,
                    latency_us: lat,
                },
            );
        }
    }
    map
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

    let gs: u32 = 32;
    let m_values: &[usize] = &[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let dims: &[(&str, usize, usize)] = &[
        ("Attn proj", 4096, 4096),
        ("FFN gate/up", 4096, 14336),
        ("DS expert", 7168, 2048),
    ];

    let mlx_ref = parse_mlx_ref();
    let has_mlx = !mlx_ref.is_empty();

    println!("=== QMM Dispatch Benchmark: Steel vs NAX vs Auto ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, Q4, group_size={}",
        WARMUP_ITERS, BENCH_ITERS, gs,
    );
    println!("M values: {:?}", m_values,);
    if has_mlx {
        println!("MLX reference: loaded from MLX_QMM_REF env var");
    } else {
        println!("MLX reference: not available (set MLX_QMM_REF env var)");
    }
    println!();

    // Collect all results for summary table
    let mut all_results: Vec<(
        usize,
        usize,
        usize,
        &str,
        BenchResult,
        BenchResult,
        BenchResult,
    )> = Vec::new();

    for &(label, k, n) in dims {
        println!("=========================================================");
        println!("  {} (K={}, N={})", label, k, n);
        println!("=========================================================");
        println!();

        for &m in m_values {
            println!("--- M={} ---", m);

            let auto = bench_auto(&registry, &queue, device, m, n, k, gs);
            let steel = bench_steel(&registry, &queue, device, m, n, k, gs);
            let nax = bench_nax(&registry, &queue, device, m, n, k, gs);

            // Print individual results
            for r in [&auto, &steel, &nax] {
                if let Some(ref s) = r.stats {
                    let tf = tflops(r.m, r.n, r.k, s.p50);
                    let tf_mean = tflops(r.m, r.n, r.k, s.mean);
                    println!(
                        "  {:<12} p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.4}  TFLOPS(mean)={:.4}",
                        r.kernel, s.p50, s.mean, s.std_dev, tf, tf_mean,
                    );
                } else {
                    println!(
                        "  {:<12} N/A ({})",
                        r.kernel,
                        r.skip_reason.unwrap_or("unsupported"),
                    );
                }
            }

            // Print MLX reference if available
            let mlx_key = format!("{}_{}", m, k);
            let mlx_key_full = format!("{}_{}_{}", m, k, n);
            if let Some(mlx) = mlx_ref.get(&mlx_key_full).or(mlx_ref.get(&mlx_key)) {
                println!(
                    "  {:<12} p50={:8.1}us  TFLOPS={:.4}",
                    "MLX-ref", mlx.latency_us, mlx.tflops,
                );
            }

            // Find best
            let mut best_label = "";
            let mut best_p50 = f64::MAX;
            for r in [&auto, &steel, &nax] {
                if let Some(p) = r.latency_p50() {
                    if p < best_p50 {
                        best_p50 = p;
                        best_label = r.kernel;
                    }
                }
            }
            if !best_label.is_empty() {
                let best_tf = tflops(m, n, k, best_p50);
                println!(
                    "  >> BEST: {} ({:.1}us, {:.4} TFLOPS)",
                    best_label, best_p50, best_tf,
                );
            }

            println!();
            all_results.push((m, k, n, label, auto, steel, nax));
        }
    }

    // =========================================================================
    // Summary table
    // =========================================================================
    println!("=========================================================");
    println!("  SUMMARY TABLE");
    println!("=========================================================");
    println!();

    if has_mlx {
        println!(
            "{:<6} {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "M",
            "Dims",
            "Auto(us)",
            "Auto(T)",
            "Steel(us)",
            "Steel(T)",
            "NAX(us)",
            "NAX(T)",
            "MLX(us)",
            "MLX(T)",
        );
    } else {
        println!(
            "{:<6} {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "M", "Dims", "Auto(us)", "Auto(T)", "Steel(us)", "Steel(T)", "NAX(us)", "NAX(T)",
        );
    }
    println!("{}", "-".repeat(if has_mlx { 104 } else { 80 }));

    for (m, k, n, label, auto, steel, nax) in &all_results {
        let dims_str = format!("{}x{}", k, n);
        let auto_us = auto
            .latency_p50()
            .map_or("-".to_string(), |v| format!("{:.1}", v));
        let auto_tf = auto
            .tflops_p50()
            .map_or("-".to_string(), |v| format!("{:.4}", v));
        let steel_us = steel
            .latency_p50()
            .map_or("-".to_string(), |v| format!("{:.1}", v));
        let steel_tf = steel
            .tflops_p50()
            .map_or("-".to_string(), |v| format!("{:.4}", v));
        let nax_us = nax
            .latency_p50()
            .map_or("-".to_string(), |v| format!("{:.1}", v));
        let nax_tf = nax
            .tflops_p50()
            .map_or("-".to_string(), |v| format!("{:.4}", v));

        if has_mlx {
            let mlx_key = format!("{}_{}_{}", m, k, n);
            let (mlx_us, mlx_tf) = mlx_ref
                .get(&mlx_key)
                .map(|r| (format!("{:.1}", r.latency_us), format!("{:.4}", r.tflops)))
                .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
            println!(
                "{:<6} {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                m, dims_str, auto_us, auto_tf, steel_us, steel_tf, nax_us, nax_tf, mlx_us, mlx_tf,
            );
        } else {
            println!(
                "{:<6} {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                m, dims_str, auto_us, auto_tf, steel_us, steel_tf, nax_us, nax_tf,
            );
        }

        let _ = label; // suppress unused warning
    }

    println!("\nDone.");
}
