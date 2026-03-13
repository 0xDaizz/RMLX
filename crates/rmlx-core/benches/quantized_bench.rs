//! ⚠️ NON-PRODUCTION PATH — per-op sync quantized_matmul(): creates+commits+waits a new
//! CB per call. Per-op CB overhead included. Useful for kernel comparison only.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Quantized matmul benchmark — QMV (M=1 decode) and QMM (batched prefill).
//!
//! MoE-focused dimensions (Mixtral-like, DeepSeek-V2):
//! - Expert FFN:        K=4096, N=2048  (per-expert intermediate)
//! - Expert FFN large:  K=5120, N=1536  (DeepSeek-V2 expert)
//! - Attention proj:    K=4096, N=4096
//! - Gate/Up proj:      K=4096, N=14336 (Mixtral)
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench quantized_bench

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
// Stats helper (identical to gemm_bench.rs)
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
// Random data generation
// ---------------------------------------------------------------------------

/// Simple LCG PRNG for deterministic benchmark data.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Create a random f32 Array on GPU with small values (suitable as activations).
fn rand_f32_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
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

/// Create a QuantizedWeight with random packed data.
///
/// For Q4: 8 values per u32. For Q8: 4 values per u32.
/// Scales and biases are f32 per group.
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

    // Packed weights buffer
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

    // Scales and biases as f16 (native QuantizedWeight format)
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

/// Create a random f16 Array on GPU with small values (suitable as activations).
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

// make_quantized_weight_f16 removed — make_quantized_weight now natively creates f16 scales/biases.

// ---------------------------------------------------------------------------
// TFLOPS calculation
// ---------------------------------------------------------------------------

/// Effective FLOPs for quantized matmul: 2*M*N*K (same as dense, since we
/// count the multiply-accumulate on dequantized values).
fn tflops(m: usize, n: usize, k: usize, latency_us: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / (latency_us * 1e-6) / 1e12
}

// ---------------------------------------------------------------------------
// Benchmark runners
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_qmv(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    n: usize,
    k: usize,
    bits: u32,
    group_size: u32,
    label: &str,
) {
    let qw = make_quantized_weight(device, n, k, bits, group_size, 42);
    let vec = rand_f32_array(device, &[k], 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
            .expect("QMV warmup failed");
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
            .expect("QMV bench failed");
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    let tf_p50 = tflops(1, n, k, stats.p50);
    let tf_mean = tflops(1, n, k, stats.mean);
    println!(
        "  QMV  Q{:<2}  {:<22}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.3}  TFLOPS(mean)={:.3}",
        bits, label, stats.p50, stats.mean, stats.std_dev, tf_p50, tf_mean,
    );
}

#[allow(clippy::too_many_arguments)]
fn bench_qmm(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    bits: u32,
    group_size: u32,
    label: &str,
) {
    let qw = make_quantized_weight(device, n, k, bits, group_size, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("QMM warmup failed");
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("QMM bench failed");
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    let tf_p50 = tflops(m, n, k, stats.p50);
    let tf_mean = tflops(m, n, k, stats.mean);
    println!(
        "  QMM  Q{:<2}  M={:<4}  {:<16}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.3}  TFLOPS(mean)={:.3}",
        bits, m, label, stats.p50, stats.mean, stats.std_dev, tf_p50, tf_mean,
    );
}

/// Benchmark the Steel (MLX-arch) QMM kernel. Q4 only.
#[allow(clippy::too_many_arguments)]
fn bench_qmm_steel(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    bits: u32,
    group_size: u32,
    label: &str,
) {
    let qw = make_quantized_weight(device, n, k, bits, group_size, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("QMM-Steel warmup failed");
    }

    // Bench
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("QMM-Steel bench failed");
        times.push(start.elapsed());
    }

    let stats = Stats::from_durations(&times);
    let tf_p50 = tflops(m, n, k, stats.p50);
    let tf_mean = tflops(m, n, k, stats.mean);
    println!(
        "  QMM-Steel  Q{:<2}  M={:<4}  {:<16}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.3}  TFLOPS(mean)={:.3}",
        bits, m, label, stats.p50, stats.mean, stats.std_dev, tf_p50, tf_mean,
    );
}

/// 3-way comparison: QMM v1 (batched) vs QMM v2 (Steel) side by side.
/// Prints both latencies and the speedup ratio (v2/v1).
#[allow(clippy::too_many_arguments)]
fn bench_qmm_3way(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    label: &str,
) {
    let bits: u32 = 4;
    let qw = make_quantized_weight(device, n, k, bits, group_size, 42);
    let x = rand_f32_array(device, &[m, k], 99);

    // --- v1 (batched) ---
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("QMM v1 warmup failed");
    }
    let mut v1_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x, &qw, queue)
            .expect("QMM v1 bench failed");
        v1_times.push(start.elapsed());
    }
    let v1_stats = Stats::from_durations(&v1_times);

    // --- v2 (Steel) ---
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("QMM v2 warmup failed");
    }
    let mut v2_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x, &qw, queue)
            .expect("QMM v2 bench failed");
        v2_times.push(start.elapsed());
    }
    let v2_stats = Stats::from_durations(&v2_times);

    let v1_tf = tflops(m, n, k, v1_stats.p50);
    let v2_tf = tflops(m, n, k, v2_stats.p50);
    let speedup = v1_stats.p50 / v2_stats.p50;

    println!(
        "  Q4  M={:<4}  {:<16}  v1 p50={:8.1}us ({:.3}T)  v2 p50={:8.1}us ({:.3}T)  speedup={:.2}x",
        m, label, v1_stats.p50, v1_tf, v2_stats.p50, v2_tf, speedup,
    );
}

/// Benchmark QMV vs QMM at M=1 to compare which path is faster for decode.
#[allow(clippy::too_many_arguments)]
fn bench_qmv_vs_qmm(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    n: usize,
    k: usize,
    bits: u32,
    group_size: u32,
    label: &str,
) {
    let qw = make_quantized_weight(device, n, k, bits, group_size, 42);
    let vec = rand_f32_array(device, &[k], 99);
    let x_2d = rand_f32_array(device, &[1, k], 99);

    // --- QMV path ---
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
            .expect("QMV warmup failed");
    }
    let mut qmv_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul(registry, &qw, &vec, queue)
            .expect("QMV bench failed");
        qmv_times.push(start.elapsed());
    }
    let qmv_stats = Stats::from_durations(&qmv_times);

    // --- QMM path (M=1) ---
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x_2d, &qw, queue)
            .expect("QMM warmup failed");
    }
    let mut qmm_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x_2d, &qw, queue)
            .expect("QMM bench failed");
        qmm_times.push(start.elapsed());
    }
    let qmm_stats = Stats::from_durations(&qmm_times);

    let qmv_tf = tflops(1, n, k, qmv_stats.p50);
    let qmm_tf = tflops(1, n, k, qmm_stats.p50);
    let winner = if qmv_stats.p50 < qmm_stats.p50 {
        "QMV"
    } else {
        "QMM"
    };
    let speedup = if qmv_stats.p50 < qmm_stats.p50 {
        qmm_stats.p50 / qmv_stats.p50
    } else {
        qmv_stats.p50 / qmm_stats.p50
    };

    println!(
        "  Q{:<2}  {:<22}  QMV p50={:8.1}us ({:.3}T)  QMM p50={:8.1}us ({:.3}T)  winner={} ({:.2}x)",
        bits, label, qmv_stats.p50, qmv_tf, qmm_stats.p50, qmm_tf, winner, speedup,
    );
}

/// Benchmark the NAX (MPP) QMM Q4 kernel with f16 input and f16 scales/biases.
/// Uses command buffer directly for precise timing.
#[allow(clippy::too_many_arguments)]
fn bench_qmm_nax(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    label: &str,
) {
    let qw = make_quantized_weight(device, n, k, 4, group_size, 42);
    let x_f16 = rand_f16_array(device, &[m, k], 99);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw, &cb)
            .expect("NAX warmup failed");
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Bench
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

    let stats = Stats::from_durations(&times);
    let tf_p50 = tflops(m, n, k, stats.p50);
    let tf_mean = tflops(m, n, k, stats.mean);
    println!(
        "  QMM-NAX  Q4  M={:<4}  {:<16}  p50={:8.1}us  mean={:8.1}us  std={:6.1}us  TFLOPS(p50)={:.3}  TFLOPS(mean)={:.3}",
        m, label, stats.p50, stats.mean, stats.std_dev, tf_p50, tf_mean,
    );
}

/// Head-to-head: QMM MMA v1 (f32) vs QMM Steel v2 (f32) vs QMM NAX (f16).
/// Prints all three latencies and speedup of NAX over each.
#[allow(clippy::too_many_arguments)]
fn bench_qmm_nax_vs_mma(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    m: usize,
    n: usize,
    k: usize,
    group_size_f32: u32,
    group_size_nax: u32,
    label: &str,
) {
    // --- MMA v1 (f32 scales, group_size_f32) ---
    let qw_f32 = make_quantized_weight(device, n, k, 4, group_size_f32, 42);
    let x_f32 = rand_f32_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x_f32, &qw_f32, queue)
            .expect("MMA v1 warmup failed");
    }
    let mut v1_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_batched(registry, &x_f32, &qw_f32, queue)
            .expect("MMA v1 bench failed");
        v1_times.push(start.elapsed());
    }
    let v1_stats = Stats::from_durations(&v1_times);

    // --- Steel v2 (f32 scales, group_size_f32) ---
    for _ in 0..WARMUP_ITERS {
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x_f32, &qw_f32, queue)
            .expect("Steel v2 warmup failed");
    }
    let mut v2_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = ops::quantized::affine_quantized_matmul_steel(registry, &x_f32, &qw_f32, queue)
            .expect("Steel v2 bench failed");
        v2_times.push(start.elapsed());
    }
    let v2_stats = Stats::from_durations(&v2_times);

    // --- NAX (f16 scales, group_size_nax) ---
    let qw_nax = make_quantized_weight(device, n, k, 4, group_size_nax, 42);
    let x_f16 = rand_f16_array(device, &[m, k], 99);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw_nax, &cb)
            .expect("NAX warmup failed");
        cb.commit();
        cb.waitUntilCompleted();
    }
    let mut nax_times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let start = Instant::now();
        let _ = ops::quantized::affine_qmm_nax_q4_into_cb(registry, &x_f16, &qw_nax, &cb)
            .expect("NAX bench failed");
        cb.commit();
        cb.waitUntilCompleted();
        nax_times.push(start.elapsed());
    }
    let nax_stats = Stats::from_durations(&nax_times);

    let v1_tf = tflops(m, n, k, v1_stats.p50);
    let v2_tf = tflops(m, n, k, v2_stats.p50);
    let nax_tf = tflops(m, n, k, nax_stats.p50);
    let nax_vs_v1 = v1_stats.p50 / nax_stats.p50;
    let nax_vs_v2 = v2_stats.p50 / nax_stats.p50;

    println!(
        "  Q4  M={:<4}  {:<16}  v1={:8.1}us ({:.3}T)  v2={:8.1}us ({:.3}T)  NAX={:8.1}us ({:.3}T)  NAX/v1={:.2}x  NAX/v2={:.2}x",
        m, label,
        v1_stats.p50, v1_tf,
        v2_stats.p50, v2_tf,
        nax_stats.p50, nax_tf,
        nax_vs_v1, nax_vs_v2,
    );
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

    let group_size: u32 = 32;

    // MoE-focused dimension sets: (label, K, N)
    let dimensions: Vec<(&str, usize, usize)> = vec![
        ("K=4096 N=14336", 4096, 14336), // Mixtral gate/up
        ("K=4096 N=4096", 4096, 4096),   // attention projections
        ("K=4096 N=2048", 4096, 2048),   // expert FFN (Mixtral 8x7B)
        ("K=5120 N=1536", 5120, 1536),   // expert FFN (DeepSeek-V2)
    ];

    // =========================================================================
    // 1. QMV M=1 (decode) — Q4 and Q8, all dimension combos
    // =========================================================================
    println!("=== QMV Decode Benchmark (M=1, f32 input) ===");
    println!(
        "Warmup: {} iters, Bench: {} iters, group_size: {}",
        WARMUP_ITERS, BENCH_ITERS, group_size,
    );
    println!();

    for &(label, k, n) in &dimensions {
        bench_qmv(&registry, &queue, device, n, k, 4, group_size, label);
        bench_qmv(&registry, &queue, device, n, k, 8, group_size, label);
    }
    println!();

    // =========================================================================
    // 2. QMM batched (prefill) — M={32, 128, 256}, Q4 and Q8
    //    Fixed K=4096, N=4096 (attention projection, representative)
    // =========================================================================
    println!("=== QMM Batched Benchmark (prefill, f32 input) ===");
    println!();

    let qmm_k = 4096;
    let qmm_n = 4096;
    let m_values = [32, 128, 256];

    for &m in &m_values {
        bench_qmm(
            &registry,
            &queue,
            device,
            m,
            qmm_n,
            qmm_k,
            4,
            group_size,
            "K=4096 N=4096",
        );
        bench_qmm(
            &registry,
            &queue,
            device,
            m,
            qmm_n,
            qmm_k,
            8,
            group_size,
            "K=4096 N=4096",
        );
    }
    println!();

    // =========================================================================
    // 3. QMV vs QMM at M=1 — which path is faster for decode?
    // =========================================================================
    println!("=== QMV vs QMM at M=1 (decode path comparison) ===");
    println!();

    for &(label, k, n) in &dimensions {
        bench_qmv_vs_qmm(&registry, &queue, device, n, k, 4, group_size, label);
        bench_qmv_vs_qmm(&registry, &queue, device, n, k, 8, group_size, label);
    }
    println!();

    // =========================================================================
    // 4. QMM Steel (MLX-arch) Benchmark — Q4 only
    // =========================================================================
    println!("=== QMM Steel (MLX-arch) Benchmark (Q4 only, f32 input) ===");
    println!();

    let steel_k = 4096;
    let steel_n = 4096;
    for &m in &m_values {
        bench_qmm_steel(
            &registry,
            &queue,
            device,
            m,
            steel_n,
            steel_k,
            4,
            group_size,
            "K=4096 N=4096",
        );
    }
    // Large dimension: K=4096 N=14336, M=256 — key comparison point
    bench_qmm_steel(
        &registry,
        &queue,
        device,
        256,
        14336,
        4096,
        4,
        group_size,
        "K=4096 N=14336",
    );
    println!();

    // =========================================================================
    // 5. QMM v1 vs v2 (Steel) Head-to-Head — Q4 only
    // =========================================================================
    println!("=== QMM v1 vs v2 (Steel) Head-to-Head (Q4 only) ===");
    println!();

    for &m in &m_values {
        bench_qmm_3way(
            &registry,
            &queue,
            device,
            m,
            steel_n,
            steel_k,
            group_size,
            "K=4096 N=4096",
        );
    }
    // Large dimension head-to-head
    bench_qmm_3way(
        &registry,
        &queue,
        device,
        256,
        14336,
        4096,
        group_size,
        "K=4096 N=14336",
    );
    println!();

    // =========================================================================
    // 6. QMM NAX (MPP) Benchmark — Q4 only, f16 input, f16 scales/biases
    //    group_size=64 (NAX prefers BK=64 aligned groups)
    // =========================================================================
    println!("=== QMM NAX (MPP) Benchmark (Q4 only, f16 input, group_size=64) ===");
    println!();

    let nax_group_size: u32 = 64;
    let nax_m_values = [32, 128, 256];

    for &m in &nax_m_values {
        bench_qmm_nax(
            &registry,
            &queue,
            device,
            m,
            4096,
            4096,
            nax_group_size,
            "K=4096 N=4096",
        );
    }
    // Large dimension: K=4096 N=14336, M=256
    bench_qmm_nax(
        &registry,
        &queue,
        device,
        256,
        14336,
        4096,
        nax_group_size,
        "K=4096 N=14336",
    );
    println!();

    // =========================================================================
    // 7. QMM MMA v1 vs Steel v2 vs NAX Head-to-Head — Q4 only
    //    Compares all three kernels at each (M,N,K) configuration.
    //    Note: v1/v2 use group_size=32, NAX uses group_size=64. All use native f16 scales.
    // =========================================================================
    println!("=== QMM v1 vs v2 (Steel) vs NAX Head-to-Head (Q4 only) ===");
    println!();

    for &m in &nax_m_values {
        bench_qmm_nax_vs_mma(
            &registry,
            &queue,
            device,
            m,
            4096,
            4096,
            group_size,     // v1/v2: group_size=32
            nax_group_size, // NAX: group_size=64
            "K=4096 N=4096",
        );
    }
    // Large dimension head-to-head
    bench_qmm_nax_vs_mma(
        &registry,
        &queue,
        device,
        256,
        14336,
        4096,
        group_size,
        nax_group_size,
        "K=4096 N=14336",
    );
    println!();

    println!("Done.");
}
