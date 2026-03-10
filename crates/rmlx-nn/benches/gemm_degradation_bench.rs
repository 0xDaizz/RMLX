//! GEMM pipeline degradation diagnostic benchmark.
//!
//! Isolates potential causes of 2-3x GEMM throughput drop when running inside
//! the prefill pipeline vs standalone:
//!
//!   Test 1: Baseline (isolated GEMM, only A/B/C in memory)
//!   Test 2: Memory pressure (~2GB of dummy tensors allocated)
//!   Test 3: Sequential CB overhead (10 prior ops, each in separate CB)
//!   Test 4: Single CB pipelined (norm + GEMM in one CB)
//!   Test 5: Fresh queue / PSO cache warm-up
//!   Test 6: Cache pollution (large buffer touched before GEMM)
//!   Test 7: Post-cleanup baseline (verify recovery)
//!
//! Run with:
//!   cargo bench -p rmlx-nn --bench gemm_degradation_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

const WARMUP: usize = 5;
const ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper (same as gemm_bench)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    p50: f64,
    min: f64,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = micros.iter().sum();
        let mean = sum / n as f64;
        let p50 = micros[n / 2];
        let min = micros[0];
        Stats { mean, p50, min }
    }
}

// ---------------------------------------------------------------------------
// f16 random array generation (deterministic PRNG, same as gemm_bench)
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

fn rand_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
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
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// GEMM timing helper
// ---------------------------------------------------------------------------

fn bench_gemm(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    a: &Array,
    b: &Array,
    warmup: usize,
    iters: usize,
) -> Stats {
    for _ in 0..warmup {
        let cb = queue.new_command_buffer();
        let _ = ops::matmul::matmul_into_cb(registry, a, b, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = ops::matmul::matmul_into_cb(registry, a, b, cb).unwrap();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    Stats::from_durations(&times)
}

// ---------------------------------------------------------------------------
// RMSNorm timing helper
// ---------------------------------------------------------------------------

fn bench_rms_norm(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    input: &Array,
    iters: usize,
) -> Stats {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let _ = ops::rms_norm::rms_norm_into_cb(registry, input, None, 1e-5, cb);
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    Stats::from_durations(&times)
}

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    let sizes: Vec<(&str, usize, usize, usize)> = vec![
        ("O_proj", 1024, 4096, 4096),
        ("Down", 1024, 14336, 4096),
    ];

    println!("=== GEMM Pipeline Degradation Diagnostic ===");
    println!("Warmup: {} iters, Bench: {} iters", WARMUP, ITERS);

    for (name, m, k, n) in &sizes {
        let a = rand_array(device, &[*m, *k], 42);
        let b = rand_array(device, &[*k, *n], 44);
        let flops = 2.0 * *m as f64 * *k as f64 * *n as f64;

        println!();
        println!("{}", "=".repeat(90));
        println!("{}: M={}, K={}, N={}", name, m, k, n);
        println!("{}", "=".repeat(90));

        // ---------------------------------------------------------------
        // Test 1: Baseline (isolated GEMM)
        // ---------------------------------------------------------------
        let baseline = bench_gemm(&registry, &queue, &a, &b, WARMUP, ITERS);
        let tflops = flops / (baseline.mean / 1e6) / 1e12;
        println!(
            "Test 1 - Baseline (isolated):     mean={:8.1}us  p50={:8.1}us  min={:8.1}us  {:.2}T",
            baseline.mean, baseline.p50, baseline.min, tflops
        );
        let base_mean = baseline.mean;

        // ---------------------------------------------------------------
        // Test 2: Memory pressure (~2GB of dummy tensors)
        // ---------------------------------------------------------------
        let _dummies: Vec<Array> = vec![
            rand_array(device, &[4096, 6144], 100),  // QKV weight ~48MB
            rand_array(device, &[4096, 4096], 101),  // O weight ~32MB
            rand_array(device, &[4096, 14336], 102), // gate weight ~112MB
            rand_array(device, &[4096, 14336], 103), // up weight ~112MB
            rand_array(device, &[14336, 4096], 104), // down weight ~112MB
            rand_array(device, &[4096], 105),        // norm1 weight
            rand_array(device, &[4096], 106),        // norm2 weight
            rand_array(device, &[1024, 4096], 107),  // intermediate 1
            rand_array(device, &[1024, 4096], 108),  // intermediate 2
            rand_array(device, &[1024, 14336], 109), // intermediate 3
            rand_array(device, &[1024, 28672], 110), // intermediate 4
            rand_array(device, &[32 * 1024, 128], 111), // KV cache 1
            rand_array(device, &[32 * 1024, 128], 112), // KV cache 2
        ];
        let stats = bench_gemm(&registry, &queue, &a, &b, WARMUP, ITERS);
        let tflops = flops / (stats.mean / 1e6) / 1e12;
        println!(
            "Test 2 - Memory pressure (~2GB):   mean={:8.1}us  p50={:8.1}us  min={:8.1}us  {:.2}T  ({:+.1}% vs baseline)",
            stats.mean, stats.p50, stats.min, tflops, (stats.mean / base_mean - 1.0) * 100.0
        );

        // ---------------------------------------------------------------
        // Test 3: Sequential CB overhead (10 prior small ops in separate CBs)
        // ---------------------------------------------------------------
        {
            let norm_input = rand_array(device, &[1024, 4096], 200);
            for _ in 0..10 {
                let cb = queue.new_command_buffer();
                let _ = ops::rms_norm::rms_norm_into_cb(&registry, &norm_input, None, 1e-5, cb);
                cb.commit();
                cb.wait_until_completed();
            }
        }
        let stats = bench_gemm(&registry, &queue, &a, &b, WARMUP, ITERS);
        let tflops = flops / (stats.mean / 1e6) / 1e12;
        println!(
            "Test 3 - After 10 prior CB ops:    mean={:8.1}us  p50={:8.1}us  min={:8.1}us  {:.2}T  ({:+.1}% vs baseline)",
            stats.mean, stats.p50, stats.min, tflops, (stats.mean / base_mean - 1.0) * 100.0
        );

        // ---------------------------------------------------------------
        // Test 4: Single CB pipelined (norm + GEMM in one command buffer)
        // ---------------------------------------------------------------
        {
            let norm_input = rand_array(device, &[*m, *k], 300);

            // Measure norm alone
            let norm_stats = bench_rms_norm(&registry, &queue, &norm_input, ITERS);

            // Warmup combo
            for _ in 0..WARMUP {
                let cb = queue.new_command_buffer();
                let _ = ops::rms_norm::rms_norm_into_cb(&registry, &norm_input, None, 1e-5, cb);
                let _ = ops::matmul::matmul_into_cb(&registry, &a, &b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
            }

            // Measure combo
            let mut combo_times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let _ = ops::rms_norm::rms_norm_into_cb(&registry, &norm_input, None, 1e-5, cb);
                let _ = ops::matmul::matmul_into_cb(&registry, &a, &b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
                combo_times.push(start.elapsed());
            }
            let combo_stats = Stats::from_durations(&combo_times);
            let gemm_est = combo_stats.mean - norm_stats.mean;
            let tflops = flops / (gemm_est / 1e6) / 1e12;
            println!(
                "Test 4 - Single CB (pipelined):    gemm_est={:8.1}us  (combo={:.1} - norm={:.1})  {:.2}T  ({:+.1}% vs baseline)",
                gemm_est, combo_stats.mean, norm_stats.mean, tflops, (gemm_est / base_mean - 1.0) * 100.0
            );
        }

        // ---------------------------------------------------------------
        // Test 5: Fresh queue / PSO cache warm-up test
        // ---------------------------------------------------------------
        {
            let fresh_queue = device.new_command_queue();
            // First call (cold for this queue — PSO is per-device so should be warm)
            let first = bench_gemm(&registry, &fresh_queue, &a, &b, 0, 1);
            let second = bench_gemm(&registry, &fresh_queue, &a, &b, 0, 1);
            let warm = bench_gemm(&registry, &fresh_queue, &a, &b, WARMUP, ITERS);
            let tflops = flops / (warm.mean / 1e6) / 1e12;
            println!(
                "Test 5 - Fresh queue:              first={:8.1}us  second={:8.1}us  warm={:8.1}us  {:.2}T  ({:+.1}% vs baseline)",
                first.mean, second.mean, warm.mean, tflops, (warm.mean / base_mean - 1.0) * 100.0
            );
        }

        // ---------------------------------------------------------------
        // Test 6: Cache pollution (large buffer touched before GEMM)
        // ---------------------------------------------------------------
        {
            let polluter = rand_array(device, &[8192, 8192], 400); // ~128MB f16
            // Run a dummy op on polluter to push GEMM data out of caches
            let cb = queue.new_command_buffer();
            let _ = ops::rms_norm::rms_norm_into_cb(&registry, &polluter, None, 1e-5, cb);
            cb.commit();
            cb.wait_until_completed();

            // Now measure GEMM (fewer warmup — we want to capture pollution effect)
            let stats = bench_gemm(&registry, &queue, &a, &b, 2, ITERS);
            let tflops = flops / (stats.mean / 1e6) / 1e12;
            println!(
                "Test 6 - After cache pollution:    mean={:8.1}us  p50={:8.1}us  min={:8.1}us  {:.2}T  ({:+.1}% vs baseline)",
                stats.mean, stats.p50, stats.min, tflops, (stats.mean / base_mean - 1.0) * 100.0
            );
            drop(polluter);
        }

        // Drop dummies to free GPU memory
        drop(_dummies);

        // ---------------------------------------------------------------
        // Test 7: Post-cleanup baseline (verify recovery)
        // ---------------------------------------------------------------
        let stats = bench_gemm(&registry, &queue, &a, &b, WARMUP, ITERS);
        let tflops = flops / (stats.mean / 1e6) / 1e12;
        println!(
            "Test 7 - Post-cleanup baseline:    mean={:8.1}us  p50={:8.1}us  min={:8.1}us  {:.2}T  ({:+.1}% vs baseline)",
            stats.mean, stats.p50, stats.min, tflops, (stats.mean / base_mean - 1.0) * 100.0
        );
    }

    println!();
    println!("=== Interpretation Guide ===");
    println!("  Test 2 regresses  -> memory pressure / wired page contention");
    println!("  Test 3 regresses  -> CB creation/commit overhead accumulation");
    println!("  Test 4 faster     -> single-CB batching avoids inter-CB overhead");
    println!("  Test 5 first slow -> PSO compilation / queue warm-up cost");
    println!("  Test 6 regresses  -> GPU cache thrashing from prior Metal ops");
    println!("  Test 7 recovers   -> degradation is transient (not permanent state)");
    println!();
    println!("Done.");
}
