//! ⚠️ NON-PRODUCTION PATH — QMM NAX kernel development benchmark. Direct encoding,
//! bypasses quantized_matmul() dispatch. Development/tuning only.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! QMM NAX V1 vs V2 Benchmark — vectorized dequant comparison.
//!
//! Compares `affine_qmm_nax_q4` (scalar dequant) vs `affine_qmm_nax_v2_q4`
//! (uchar4 + half4 vectorized dequant + vectorized fragment loads).
//!
//! Dimensions: M={128,256,512}, N=14336 (Qwen FFN), K=3584, Q4, group_size=64.
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench qmm_nax_bench

use std::time::{Duration, Instant};

use half::f16;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_core::ops::quantized::QuantizedWeight;
use rmlx_metal::device::GpuDevice;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    p50: f64,
}

impl Stats {
    fn from_durations(durations: &[Duration]) -> Self {
        let mut micros: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&micros, 50.0);
        Stats { p50 }
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

fn rand_f16_array(device: &metal::Device, shape: &[usize], seed: u64) -> Array {
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
    device: &metal::Device,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: u32,
    seed: u64,
) -> QuantizedWeight {
    let mut state = seed;
    let opts = metal::MTLResourceOptions::StorageModeShared;

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
        device.new_buffer_with_data(w_data.as_ptr() as *const _, (num_u32s * 4) as u64, opts);

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

    let scales_data: Vec<u16> = scales_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();
    let biases_data: Vec<u16> = biases_f32
        .iter()
        .map(|&v| rmlx_core::ops::quantized::f32_to_f16_bits(v))
        .collect();

    let scales_buf = device.new_buffer_with_data(
        scales_data.as_ptr() as *const _,
        (num_groups * 2) as u64,
        opts,
    );
    let biases_buf = device.new_buffer_with_data(
        biases_data.as_ptr() as *const _,
        (num_groups * 2) as u64,
        opts,
    );

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
// Bench runners
// ---------------------------------------------------------------------------

fn bench_nax_v1(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, cb)
            .expect("NAX v1 warmup failed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x, &qw, cb)
            .expect("NAX v1 bench failed");
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    Stats::from_durations(&times)
}

fn bench_nax_v2(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Stats {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let _ = ops::quantized::affine_qmm_nax_v2_q4_into_cb(registry, &x, &qw, cb)
            .expect("NAX v2 warmup failed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_v2_q4_into_cb(registry, &x, &qw, cb)
            .expect("NAX v2 bench failed");
        cb.commit();
        cb.wait_until_completed();
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
    let queue = device.new_command_queue();

    let n: usize = 14336;
    let k: usize = 3584;
    let gs: u32 = 64;
    let m_values: &[usize] = &[128, 256, 512];

    println!("=== QMM NAX V1 vs V2 Benchmark ===");
    println!(
        "N={}, K={}, Q4, group_size={}, warmup={}, bench={}",
        n, k, gs, WARMUP_ITERS, BENCH_ITERS,
    );
    println!();

    // Collect results for table output
    struct Row {
        m: usize,
        v1_p50: f64,
        v1_tflops: f64,
        v2_p50: f64,
        v2_tflops: f64,
        speedup: f64,
    }
    let mut rows: Vec<Row> = Vec::new();

    for &m in m_values {
        println!("Benchmarking M={}...", m);

        let v1 = bench_nax_v1(&registry, &queue, device, m, n, k, gs);
        let v2 = bench_nax_v2(&registry, &queue, device, m, n, k, gs);

        let v1_tf = tflops(m, n, k, v1.p50);
        let v2_tf = tflops(m, n, k, v2.p50);
        let speedup = v1.p50 / v2.p50;

        rows.push(Row {
            m,
            v1_p50: v1.p50,
            v1_tflops: v1_tf,
            v2_p50: v2.p50,
            v2_tflops: v2_tf,
            speedup,
        });
    }

    // Print markdown table
    println!();
    println!("| M | Kernel | p50 (us) | TFLOPS | Speedup |");
    println!("|----:|--------|----------:|--------:|--------:|");
    for row in &rows {
        println!(
            "| {} | NAX v1 | {:.1} | {:.4} | - |",
            row.m, row.v1_p50, row.v1_tflops,
        );
        println!(
            "| {} | NAX v2 | {:.1} | {:.4} | {:.3}x |",
            row.m, row.v2_p50, row.v2_tflops, row.speedup,
        );
    }

    println!("\nDone.");
}
