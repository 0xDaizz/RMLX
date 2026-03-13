//! ⚠️ NON-PRODUCTION PATH — Rust-side optimization micro-benchmarks (P0-P4).
//! Measures pipeline lookup, PipelineKey creation, dispatch calculation overhead.
//! Diagnostic only; does not test full TransformerModel forward path.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Benchmark for Rust-side optimizations (P0-P4).
//!
//! Measures:
//! 1. P1-B: frozen vs RwLock pipeline lookup latency
//! 2. P3-A: PipelineKey creation cost (Cow + SmallVec vs old String + Vec)
//! 3. P4-A: dispatch calculation overhead (const method vs match)
//! 4. Combined: end-to-end matmul dispatch (frozen + SmallVec + const dispatch)
//!
//! Run: cargo bench -p rmlx-core --bench rust_opt_bench

use std::borrow::Cow;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{FunctionConstantValue, KernelRegistry, PipelineKey};
use rmlx_core::ops;
use rmlx_core::ops::matmul::{select_tile_config_with_nax, TileVariant};
use rmlx_metal::device::GpuDevice;
use smallvec::SmallVec;

const WARMUP_ITERS: usize = 20;
const BENCH_ITERS: usize = 200;

// ---------------------------------------------------------------------------
// Stats helper (same pattern as dispatch_overhead.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    min: f64,
    max: f64,
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
        let p95 = percentile(&micros, 95.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            min: micros[0],
            max: micros[n - 1],
        }
    }

    /// Variant that reports in nanoseconds (for micro-benchmarks).
    fn from_durations_ns(durations: &[Duration]) -> Self {
        let n = durations.len();
        assert!(n > 0);
        let mut nanos: Vec<f64> = durations.iter().map(|d| d.as_nanos() as f64).collect();
        nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sum: f64 = nanos.iter().sum();
        let mean = sum / n as f64;
        let variance: f64 = nanos.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let p50 = percentile(&nanos, 50.0);
        let p95 = percentile(&nanos, 95.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            min: nanos[0],
            max: nanos[n - 1],
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p50={:8.1}  mean={:8.1}  std={:6.1}  p95={:8.1}  min={:8.1}  max={:8.1}",
            self.p50, self.mean, self.std_dev, self.p95, self.min, self.max,
        )
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
// f16 random array generation (deterministic PRNG)
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

fn rand_f16_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
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

// ===========================================================================
// Bench 1: P1-B — Frozen vs Unfrozen Pipeline Lookup
// ===========================================================================

fn bench_pipeline_lookup(registry: &KernelRegistry) {
    println!("=== 1. P1-B: Frozen vs Unfrozen Pipeline Lookup ===");
    println!("  get_pipeline() latency for cached kernel (ns per call)");
    println!();

    // Ensure rms_norm_f16 is in the cache by calling get_pipeline once.
    registry
        .get_pipeline("rms_norm_f16", DType::Float16)
        .expect("rms_norm_f16 should be registered");

    // --- Bench A: Unfrozen (RwLock read path) ---
    let iters_per_sample = 1000;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            let _ = black_box(registry.get_pipeline("rms_norm_f16", DType::Float16));
        }
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            let _ = black_box(registry.get_pipeline("rms_norm_f16", DType::Float16));
        }
        let elapsed = start.elapsed();
        times.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let unfrozen_stats = Stats::from_durations_ns(&times);

    // --- Bench B: Frozen (atomic pointer fast path) ---
    registry.freeze();
    assert!(registry.is_frozen(), "freeze() should set frozen state");

    // Warmup
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            let _ = black_box(registry.get_pipeline("rms_norm_f16", DType::Float16));
        }
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            let _ = black_box(registry.get_pipeline("rms_norm_f16", DType::Float16));
        }
        let elapsed = start.elapsed();
        times.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let frozen_stats = Stats::from_durations_ns(&times);

    let speedup = unfrozen_stats.p50 / frozen_stats.p50;

    println!("| variant  | p50 (ns) | mean (ns) | p95 (ns) | min (ns) | speedup |");
    println!("|----------|----------|-----------|----------|----------|---------|");
    println!(
        "| unfrozen | {:8.1} | {:9.1} | {:8.1} | {:8.1} |    1.0x |",
        unfrozen_stats.p50, unfrozen_stats.mean, unfrozen_stats.p95, unfrozen_stats.min,
    );
    println!(
        "| frozen   | {:8.1} | {:9.1} | {:8.1} | {:8.1} | {:5.2}x |",
        frozen_stats.p50, frozen_stats.mean, frozen_stats.p95, frozen_stats.min, speedup,
    );
    println!();
}

// ===========================================================================
// Bench 2: P3-A — PipelineKey Creation Cost
// ===========================================================================

fn bench_pipeline_key_creation() {
    println!("=== 2. P3-A: PipelineKey Creation Cost ===");
    println!("  PipelineKey::new with 0, 2, 5 constants (ns per creation)");
    println!();

    let iters_per_sample = 10_000;

    // --- 0 constants (common case: Cow::Borrowed + empty SmallVec) ---
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("rms_norm_f16"),
                dtype: DType::Float16,
                constants: SmallVec::new(),
            };
            black_box(&key);
        }
    }

    let mut times_0 = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("rms_norm_f16"),
                dtype: DType::Float16,
                constants: SmallVec::new(),
            };
            black_box(&key);
        }
        let elapsed = start.elapsed();
        times_0.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let stats_0 = Stats::from_durations_ns(&times_0);

    // --- 2 constants (typical GEMM: align_M + align_N, fits SmallVec inline) ---
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("gemm_mlx_f16"),
                dtype: DType::Float16,
                constants: SmallVec::from_buf([
                    (200, FunctionConstantValue::Bool(true)),
                    (201, FunctionConstantValue::Bool(false)),
                    // padding slots unused
                    (0, FunctionConstantValue::Bool(false)),
                    (0, FunctionConstantValue::Bool(false)),
                ]),
            };
            black_box(&key);
        }
    }

    let mut times_2 = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            let constants: SmallVec<[(u32, FunctionConstantValue); 4]> = smallvec::smallvec![
                (200, FunctionConstantValue::Bool(true)),
                (201, FunctionConstantValue::Bool(false)),
            ];
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("gemm_mlx_f16"),
                dtype: DType::Float16,
                constants,
            };
            black_box(&key);
        }
        let elapsed = start.elapsed();
        times_2.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let stats_2 = Stats::from_durations_ns(&times_2);

    // --- 5 constants (above SmallVec capacity — spills to heap) ---
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            let constants: SmallVec<[(u32, FunctionConstantValue); 4]> = smallvec::smallvec![
                (200, FunctionConstantValue::Bool(true)),
                (201, FunctionConstantValue::Bool(false)),
                (202, FunctionConstantValue::Bool(true)),
                (203, FunctionConstantValue::Bool(false)),
                (204, FunctionConstantValue::Bool(false)),
            ];
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("gemm_mlx_f16"),
                dtype: DType::Float16,
                constants,
            };
            black_box(&key);
        }
    }

    let mut times_5 = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            let constants: SmallVec<[(u32, FunctionConstantValue); 4]> = smallvec::smallvec![
                (200, FunctionConstantValue::Bool(true)),
                (201, FunctionConstantValue::Bool(false)),
                (202, FunctionConstantValue::Bool(true)),
                (203, FunctionConstantValue::Bool(false)),
                (204, FunctionConstantValue::Bool(false)),
            ];
            let key = PipelineKey {
                kernel_name: Cow::Borrowed("gemm_mlx_f16"),
                dtype: DType::Float16,
                constants,
            };
            black_box(&key);
        }
        let elapsed = start.elapsed();
        times_5.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let stats_5 = Stats::from_durations_ns(&times_5);

    println!("| constants | p50 (ns) | mean (ns) | p95 (ns) | min (ns) | note            |");
    println!("|-----------|----------|-----------|----------|----------|-----------------|");
    println!(
        "| 0         | {:8.1} | {:9.1} | {:8.1} | {:8.1} | Cow::Borrowed   |",
        stats_0.p50, stats_0.mean, stats_0.p95, stats_0.min,
    );
    println!(
        "| 2         | {:8.1} | {:9.1} | {:8.1} | {:8.1} | SmallVec inline |",
        stats_2.p50, stats_2.mean, stats_2.p95, stats_2.min,
    );
    println!(
        "| 5         | {:8.1} | {:9.1} | {:8.1} | {:8.1} | SmallVec spill  |",
        stats_5.p50, stats_5.mean, stats_5.p95, stats_5.min,
    );
    println!();
}

// ===========================================================================
// Bench 3: P4-A — Dispatch Calculation Overhead
// ===========================================================================

fn bench_dispatch_calc() {
    println!("=== 3. P4-A: Dispatch Calculation (threads_per_tg / has_swizzle) ===");
    println!("  Const method on TileVariant vs simulated old match (ns per call)");
    println!();

    let iters_per_sample = 100_000;
    let variants = [
        TileVariant::Small,
        TileVariant::Medium,
        TileVariant::Simd,
        TileVariant::Full,
        TileVariant::Skinny,
        TileVariant::MlxArch,
        TileVariant::MlxArchSmall,
        TileVariant::MlxArchMicro,
        TileVariant::NaxArch,
        TileVariant::NaxArch64x128,
    ];

    // --- Bench A: Simulate old-style inline match (not const) ---
    #[inline(never)]
    fn old_threads_per_tg(variant: TileVariant) -> u64 {
        match variant {
            TileVariant::Small => 256,
            TileVariant::Medium | TileVariant::Simd => 1024,
            TileVariant::Full | TileVariant::Skinny => 1024,
            TileVariant::MlxArch => 128,
            TileVariant::MlxArchSmall => 128,
            TileVariant::MlxArchMicro => 128,
            TileVariant::NaxArch => 128,
            TileVariant::NaxArch64x128 => 128,
            TileVariant::NaxArch64x64 => 128,
        }
    }

    #[inline(never)]
    fn old_has_swizzle(variant: TileVariant) -> bool {
        match variant {
            TileVariant::Full
            | TileVariant::Skinny
            | TileVariant::MlxArch
            | TileVariant::MlxArchSmall
            | TileVariant::MlxArchMicro
            | TileVariant::NaxArch
            | TileVariant::NaxArch64x128
            | TileVariant::NaxArch64x64 => true,
            TileVariant::Small | TileVariant::Medium | TileVariant::Simd => false,
        }
    }

    // Warmup old path
    for _ in 0..WARMUP_ITERS {
        for v in &variants {
            for _ in 0..(iters_per_sample / variants.len()) {
                black_box(old_threads_per_tg(black_box(*v)));
                black_box(old_has_swizzle(black_box(*v)));
            }
        }
    }

    let mut times_old = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for v in &variants {
            for _ in 0..(iters_per_sample / variants.len()) {
                black_box(old_threads_per_tg(black_box(*v)));
                black_box(old_has_swizzle(black_box(*v)));
            }
        }
        let elapsed = start.elapsed();
        let total_calls = iters_per_sample * 2; // threads_per_tg + has_swizzle
        times_old.push(Duration::from_nanos(
            (elapsed.as_nanos() / total_calls as u128) as u64,
        ));
    }
    let stats_old = Stats::from_durations_ns(&times_old);

    // --- Bench B: New const method path ---
    for _ in 0..WARMUP_ITERS {
        for v in &variants {
            for _ in 0..(iters_per_sample / variants.len()) {
                black_box(black_box(*v).threads_per_tg());
                black_box(black_box(*v).has_swizzle());
            }
        }
    }

    let mut times_new = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for v in &variants {
            for _ in 0..(iters_per_sample / variants.len()) {
                black_box(black_box(*v).threads_per_tg());
                black_box(black_box(*v).has_swizzle());
            }
        }
        let elapsed = start.elapsed();
        let total_calls = iters_per_sample * 2;
        times_new.push(Duration::from_nanos(
            (elapsed.as_nanos() / total_calls as u128) as u64,
        ));
    }
    let stats_new = Stats::from_durations_ns(&times_new);

    // --- Bench C: select_tile_config_with_nax full dispatch calc ---
    for _ in 0..WARMUP_ITERS {
        for _ in 0..iters_per_sample {
            black_box(select_tile_config_with_nax(
                black_box(128),
                black_box(3584),
                black_box(3584),
                black_box(DType::Float16),
                black_box(true),
            ));
        }
    }

    let mut times_select = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            black_box(select_tile_config_with_nax(
                black_box(128),
                black_box(3584),
                black_box(3584),
                black_box(DType::Float16),
                black_box(true),
            ));
        }
        let elapsed = start.elapsed();
        times_select.push(Duration::from_nanos(
            (elapsed.as_nanos() / iters_per_sample as u128) as u64,
        ));
    }
    let stats_select = Stats::from_durations_ns(&times_select);

    println!("| method                     | p50 (ns) | mean (ns) | p95 (ns) | min (ns) |");
    println!("|----------------------------|----------|-----------|----------|----------|");
    println!(
        "| old inline match           | {:8.1} | {:9.1} | {:8.1} | {:8.1} |",
        stats_old.p50, stats_old.mean, stats_old.p95, stats_old.min,
    );
    println!(
        "| const fn (threads+swizzle) | {:8.1} | {:9.1} | {:8.1} | {:8.1} |",
        stats_new.p50, stats_new.mean, stats_new.p95, stats_new.min,
    );
    println!(
        "| select_tile_config_nax     | {:8.1} | {:9.1} | {:8.1} | {:8.1} |",
        stats_select.p50, stats_select.mean, stats_select.p95, stats_select.min,
    );
    println!();
}

// ===========================================================================
// Bench 4: End-to-End — Single matmul dispatch with all optimizations
// ===========================================================================

fn bench_e2e_matmul(registry: &KernelRegistry) {
    println!("=== 4. End-to-End: matmul dispatch (M=128, N=3584, K=3584) ===");
    println!("  Complete matmul pipeline: lookup + dispatch + encoding (us)");
    println!();

    let device = registry.device().raw();
    let queue = device.new_command_queue();

    let m = 128;
    let n = 3584;
    let k = 3584;
    let a = rand_f16_array(device, &[m, k], 42);
    let b = rand_f16_array(device, &[k, n], 123);

    // --- Bench A: Without freeze (clear frozen state first) ---
    registry.clear_pipeline_cache().ok();
    // Re-populate the cache with a single call
    let _ = ops::matmul::matmul(registry, &a, &b, &queue);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::matmul::matmul(registry, &a, &b, &queue);
    }

    let mut times_unfrozen = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = black_box(ops::matmul::matmul(registry, &a, &b, &queue));
        times_unfrozen.push(start.elapsed());
    }
    let stats_unfrozen = Stats::from_durations(&times_unfrozen);

    // --- Bench B: With freeze ---
    registry.freeze();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::matmul::matmul(registry, &a, &b, &queue);
    }

    let mut times_frozen = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = black_box(ops::matmul::matmul(registry, &a, &b, &queue));
        times_frozen.push(start.elapsed());
    }
    let stats_frozen = Stats::from_durations(&times_frozen);

    // --- Bench C: matmul_into_cb (encode only, no commit/wait) ---
    // Re-freeze after clear
    registry.freeze();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb);
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times_into_cb = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let start = Instant::now();
        let _ = black_box(ops::matmul::matmul_into_cb(registry, &a, &b, cb));
        let encode_time = start.elapsed();
        cb.commit();
        cb.wait_until_completed();
        times_into_cb.push(encode_time);
    }
    let stats_into_cb = Stats::from_durations(&times_into_cb);

    let speedup = stats_unfrozen.p50 / stats_frozen.p50;

    println!("| variant              | p50 (us) | mean (us) | p95 (us) | min (us) | speedup |");
    println!("|----------------------|----------|-----------|----------|----------|---------|");
    println!(
        "| matmul (unfrozen)    | {:8.1} | {:9.1} | {:8.1} | {:8.1} |    1.0x |",
        stats_unfrozen.p50, stats_unfrozen.mean, stats_unfrozen.p95, stats_unfrozen.min,
    );
    println!(
        "| matmul (frozen)      | {:8.1} | {:9.1} | {:8.1} | {:8.1} | {:5.2}x |",
        stats_frozen.p50, stats_frozen.mean, stats_frozen.p95, stats_frozen.min, speedup,
    );
    println!(
        "| matmul_into_cb (frz) | {:8.1} | {:9.1} | {:8.1} | {:8.1} |     n/a |",
        stats_into_cb.p50, stats_into_cb.mean, stats_into_cb.p95, stats_into_cb.min,
    );
    println!();
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");

    println!("Rust-side Optimization Benchmarks (P0-P4)");
    println!("  warmup={WARMUP_ITERS}  iters={BENCH_ITERS}");
    println!();

    // Pure CPU benchmarks first (no GPU state dependency)
    bench_pipeline_key_creation();
    bench_dispatch_calc();

    // GPU-dependent benchmarks
    bench_pipeline_lookup(&registry);
    bench_e2e_matmul(&registry);
}
