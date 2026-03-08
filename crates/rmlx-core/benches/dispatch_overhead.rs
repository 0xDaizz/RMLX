//! F-1: Metal dispatch overhead profiling benchmark.
//!
//! Quantifies CB (Command Buffer) creation/commit/wait costs:
//!
//! 1. Empty kernel single-dispatch latency
//! 2. 1CB-N encoders vs NCB-1encoder (N = 1, 4, 8, 16, 32)
//! 3. Llama-3 8B single-layer op breakdown: matmul, rms_norm, rope, fused_silu_mul
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench dispatch_overhead

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;

// ---------------------------------------------------------------------------
// Llama-3 8B parameters
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 4096;
const INTERMEDIATE_DIM: usize = 14336;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const RMS_NORM_EPS: f32 = 1e-5;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;

// ---------------------------------------------------------------------------
// Stats helper
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
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p50={:8.1}us  mean={:8.1}us  std={:6.1}us  p95={:8.1}us  min={:8.1}us  max={:8.1}us",
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
// Bench 1: Empty kernel single-dispatch latency
// ===========================================================================

fn bench_empty_dispatch(queue: &metal::CommandQueue) {
    println!("=== 1. Empty CB dispatch latency ===");
    println!("  CB create + commit + waitUntilCompleted (no compute encoder)");
    println!();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
    }

    // Bench: empty CB (no encoder at all)
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    let stats = Stats::from_durations(&times);
    println!("  empty CB (no encoder):       {stats}");

    // Bench: CB with empty compute encoder
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    let stats = Stats::from_durations(&times);
    println!("  empty CB (1 empty encoder):  {stats}");

    // Bench: CB with unretained references + empty encoder
    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let cb = queue.new_command_buffer_with_unretained_references();
        let enc = cb.new_compute_command_encoder();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    let stats = Stats::from_durations(&times);
    println!("  empty CB (unretained):       {stats}");

    println!();
}

// ===========================================================================
// Bench 2: 1CB-N encoders vs NCB-1encoder
// ===========================================================================

fn bench_cb_vs_encoder(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
) {
    println!("=== 2. 1CB-N encoders vs NCB-1encoder ===");
    println!("  Using matmul [1, 4096] @ [4096, 4096] as the compute work unit");
    println!();

    let a = rand_f16_array(device, &[1, HIDDEN_DIM], 42);
    let b = rand_f16_array(device, &[HIDDEN_DIM, HIDDEN_DIM], 43);

    let ns = [1, 4, 8, 16, 32];

    for &n in &ns {
        // --- NCB-1encoder: N separate command buffers, each with 1 matmul ---
        for _ in 0..WARMUP_ITERS {
            for _ in 0..n {
                let cb = queue.new_command_buffer();
                let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
            }
        }

        let mut times_ncb = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            for _ in 0..n {
                let cb = queue.new_command_buffer();
                let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
                cb.commit();
                cb.wait_until_completed();
            }
            times_ncb.push(start.elapsed());
        }
        let stats_ncb = Stats::from_durations(&times_ncb);

        // --- 1CB-N encoder: 1 command buffer, N matmuls encoded sequentially ---
        for _ in 0..WARMUP_ITERS {
            let cb = queue.new_command_buffer();
            for _ in 0..n {
                let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
            }
            cb.commit();
            cb.wait_until_completed();
        }

        let mut times_1cb = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            for _ in 0..n {
                let _ = ops::matmul::matmul_into_cb(registry, &a, &b, cb).unwrap();
            }
            cb.commit();
            cb.wait_until_completed();
            times_1cb.push(start.elapsed());
        }
        let stats_1cb = Stats::from_durations(&times_1cb);

        let overhead_us = stats_ncb.p50 - stats_1cb.p50;
        let overhead_per_cb = if n > 1 {
            overhead_us / (n - 1) as f64
        } else {
            0.0
        };
        let pct = if stats_1cb.p50 > 0.0 {
            (overhead_us / stats_1cb.p50) * 100.0
        } else {
            0.0
        };

        println!(
            "  N={:2}  {n}CB x 1enc: {}  1CB x {n}enc: {}  overhead={:+.1}us ({:.1}%)  per-CB={:.1}us",
            n, stats_ncb, stats_1cb, overhead_us, pct, overhead_per_cb,
        );
    }

    println!();
}

// ===========================================================================
// Bench 3: Llama-3 8B single-layer op breakdown
// ===========================================================================

fn bench_llama3_layer_ops(
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    device: &metal::Device,
) {
    println!("=== 3. Llama-3 8B single-layer op breakdown (decode M=1, f16) ===");
    println!(
        "  hidden_dim={HIDDEN_DIM}  intermediate={INTERMEDIATE_DIM}  \
         n_heads={N_HEADS}  n_kv_heads={N_KV_HEADS}  head_dim={HEAD_DIM}"
    );
    println!();

    // --- Prepare test arrays ---
    let x = rand_f16_array(device, &[1, HIDDEN_DIM], 42);
    let norm_w = rand_f16_array(device, &[HIDDEN_DIM], 100);
    let wq = rand_f16_array(device, &[HIDDEN_DIM, N_HEADS * HEAD_DIM], 101);
    let wk = rand_f16_array(device, &[HIDDEN_DIM, N_KV_HEADS * HEAD_DIM], 102);
    let wv = rand_f16_array(device, &[HIDDEN_DIM, N_KV_HEADS * HEAD_DIM], 103);
    let wo = rand_f16_array(device, &[N_HEADS * HEAD_DIM, HIDDEN_DIM], 104);
    let wgate = rand_f16_array(device, &[HIDDEN_DIM, INTERMEDIATE_DIM], 105);
    let wup = rand_f16_array(device, &[HIDDEN_DIM, INTERMEDIATE_DIM], 106);
    let wdown = rand_f16_array(device, &[INTERMEDIATE_DIM, HIDDEN_DIM], 107);

    // RoPE freqs
    let (cos_data, sin_data) =
        ops::rope::precompute_freqs(2048, HEAD_DIM, 500000.0, 1.0).unwrap();
    let cos_bytes: Vec<u8> = cos_data[..HEAD_DIM / 2]
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect();
    let sin_bytes: Vec<u8> = sin_data[..HEAD_DIM / 2]
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect();
    let cos_freqs =
        Array::from_bytes(device, &cos_bytes, vec![1, 1, HEAD_DIM / 2], DType::Float16);
    let sin_freqs =
        Array::from_bytes(device, &sin_bytes, vec![1, 1, HEAD_DIM / 2], DType::Float16);
    let rope_input = rand_f16_array(device, &[1, 1, HEAD_DIM], 150);

    // FFN intermediates
    let gate_out = rand_f16_array(device, &[1, INTERMEDIATE_DIM], 200);
    let up_out = rand_f16_array(device, &[1, INTERMEDIATE_DIM], 201);
    let ffn_mid = rand_f16_array(device, &[1, INTERMEDIATE_DIM], 210);

    // ---------------------------------------------------------------------------
    // Per-op measurement
    // ---------------------------------------------------------------------------

    struct OpBench {
        name: &'static str,
        description: &'static str,
        stats_individual: Stats, // per-op CB (total wall time)
        stats_batched: Stats,    // shared CB (pure GPU time estimate)
    }

    let mut results: Vec<OpBench> = Vec::new();

    // Helper: bench a single op both individually (own CB) and batched (shared CB)
    macro_rules! bench_op {
        ($name:expr, $desc:expr, $op:expr) => {{
            let op_fn = $op;

            // Individual CB per op
            for _ in 0..WARMUP_ITERS {
                let cb = queue.new_command_buffer();
                op_fn(registry, cb);
                cb.commit();
                cb.wait_until_completed();
            }
            let mut times_ind = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                op_fn(registry, cb);
                cb.commit();
                cb.wait_until_completed();
                times_ind.push(start.elapsed());
            }

            // Batched: 16 of same op in one CB, divide time by 16
            let batch_n: u32 = 16;
            for _ in 0..WARMUP_ITERS {
                let cb = queue.new_command_buffer();
                for _ in 0..batch_n {
                    op_fn(registry, cb);
                }
                cb.commit();
                cb.wait_until_completed();
            }
            let mut times_bat = Vec::with_capacity(BENCH_ITERS);
            for _ in 0..BENCH_ITERS {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                for _ in 0..batch_n {
                    op_fn(registry, cb);
                }
                cb.commit();
                cb.wait_until_completed();
                times_bat.push(start.elapsed() / batch_n);
            }

            results.push(OpBench {
                name: $name,
                description: $desc,
                stats_individual: Stats::from_durations(&times_ind),
                stats_batched: Stats::from_durations(&times_bat),
            });
        }};
    }

    // 1. RMS Norm (attention)
    bench_op!(
        "rms_norm",
        "[1,4096] eps=1e-5",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::rms_norm::rms_norm_into_cb(reg, &x, Some(&norm_w), RMS_NORM_EPS, cb);
        }
    );

    // 2. Q matmul
    bench_op!(
        "matmul_q",
        "[1,4096]@[4096,4096]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wq, cb);
        }
    );

    // 3. K matmul
    bench_op!(
        "matmul_k",
        "[1,4096]@[4096,1024]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wk, cb);
        }
    );

    // 4. V matmul
    bench_op!(
        "matmul_v",
        "[1,4096]@[4096,1024]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wv, cb);
        }
    );

    // 5. RoPE
    bench_op!(
        "rope",
        "[1,1,128] table",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::rope::rope_ext_into_cb(
                reg,
                &rope_input,
                &cos_freqs,
                &sin_freqs,
                0,
                1.0,
                false,
                true,
                cb,
            );
        }
    );

    // 6. O projection
    bench_op!(
        "matmul_o",
        "[1,4096]@[4096,4096]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wo, cb);
        }
    );

    // 7. RMS Norm (FFN)
    bench_op!(
        "rms_norm_ffn",
        "[1,4096] eps=1e-5",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::rms_norm::rms_norm_into_cb(reg, &x, Some(&norm_w), RMS_NORM_EPS, cb);
        }
    );

    // 8. Gate matmul
    bench_op!(
        "matmul_gate",
        "[1,4096]@[4096,14336]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wgate, cb);
        }
    );

    // 9. Up matmul
    bench_op!(
        "matmul_up",
        "[1,4096]@[4096,14336]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &x, &wup, cb);
        }
    );

    // 10. Fused SiLU * mul
    bench_op!(
        "fused_silu_mul",
        "[1,14336]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::fused::fused_silu_mul_into_cb(reg, &gate_out, &up_out, cb);
        }
    );

    // 11. Down matmul
    bench_op!(
        "matmul_down",
        "[1,14336]@[14336,4096]",
        |reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
            let _ = ops::matmul::matmul_into_cb(reg, &ffn_mid, &wdown, cb);
        }
    );

    // --- Print results ---
    println!(
        "  {:<18} {:<25} {:>10} {:>10} {:>10} {:>8}",
        "Op", "Shape", "Total(p50)", "GPU(p50)", "Overhead", "OH %",
    );
    println!("  {:-<93}", "");

    let mut total_individual = 0.0_f64;
    let mut total_gpu = 0.0_f64;

    for r in &results {
        let overhead = r.stats_individual.p50 - r.stats_batched.p50;
        let oh_pct = if r.stats_individual.p50 > 0.0 {
            (overhead / r.stats_individual.p50) * 100.0
        } else {
            0.0
        };
        total_individual += r.stats_individual.p50;
        total_gpu += r.stats_batched.p50;

        println!(
            "  {:<18} {:<25} {:>8.1}us {:>8.1}us {:>8.1}us {:>6.1}%",
            r.name, r.description, r.stats_individual.p50, r.stats_batched.p50, overhead, oh_pct,
        );
    }

    let total_overhead = total_individual - total_gpu;
    let total_oh_pct = if total_individual > 0.0 {
        (total_overhead / total_individual) * 100.0
    } else {
        0.0
    };
    println!("  {:-<93}", "");
    println!(
        "  {:<18} {:<25} {:>8.1}us {:>8.1}us {:>8.1}us {:>6.1}%",
        "TOTAL (1 layer)", "", total_individual, total_gpu, total_overhead, total_oh_pct,
    );
    println!(
        "  {:<18} {:<25} {:>8.1}us {:>8.1}us {:>8.1}us",
        "x 32 layers",
        "",
        total_individual * 32.0,
        total_gpu * 32.0,
        total_overhead * 32.0,
    );
    println!();

    // --- Full layer: all ops in one CB vs individual CBs ---
    println!("  --- Full layer comparison: 11 individual CBs vs 1 batched CB ---");

    let do_layer_individual = || {
        let ops_list: Vec<Box<dyn Fn(&KernelRegistry, &metal::CommandBufferRef)>> = vec![
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ =
                    ops::rms_norm::rms_norm_into_cb(reg, &x, Some(&norm_w), RMS_NORM_EPS, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wq, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wk, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wv, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::rope::rope_ext_into_cb(
                    reg,
                    &rope_input,
                    &cos_freqs,
                    &sin_freqs,
                    0,
                    1.0,
                    false,
                    true,
                    cb,
                );
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wo, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ =
                    ops::rms_norm::rms_norm_into_cb(reg, &x, Some(&norm_w), RMS_NORM_EPS, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wgate, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &x, &wup, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::fused::fused_silu_mul_into_cb(reg, &gate_out, &up_out, cb);
            }),
            Box::new(|reg: &KernelRegistry, cb: &metal::CommandBufferRef| {
                let _ = ops::matmul::matmul_into_cb(reg, &ffn_mid, &wdown, cb);
            }),
        ];
        for op_fn in &ops_list {
            let cb = queue.new_command_buffer();
            op_fn(registry, cb);
            cb.commit();
            cb.wait_until_completed();
        }
    };

    let do_layer_batched = || {
        let cb = queue.new_command_buffer();
        let _ = ops::rms_norm::rms_norm_into_cb(registry, &x, Some(&norm_w), RMS_NORM_EPS, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wq, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wk, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wv, cb);
        let _ = ops::rope::rope_ext_into_cb(
            registry,
            &rope_input,
            &cos_freqs,
            &sin_freqs,
            0,
            1.0,
            false,
            true,
            cb,
        );
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wo, cb);
        let _ = ops::rms_norm::rms_norm_into_cb(registry, &x, Some(&norm_w), RMS_NORM_EPS, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wgate, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &x, &wup, cb);
        let _ = ops::fused::fused_silu_mul_into_cb(registry, &gate_out, &up_out, cb);
        let _ = ops::matmul::matmul_into_cb(registry, &ffn_mid, &wdown, cb);
        cb.commit();
        cb.wait_until_completed();
    };

    for _ in 0..WARMUP_ITERS {
        do_layer_individual();
        do_layer_batched();
    }

    let mut times_ind = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        do_layer_individual();
        times_ind.push(start.elapsed());
    }

    let mut times_bat = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        do_layer_batched();
        times_bat.push(start.elapsed());
    }

    let stats_ind = Stats::from_durations(&times_ind);
    let stats_bat = Stats::from_durations(&times_bat);
    let layer_overhead = stats_ind.p50 - stats_bat.p50;
    let layer_oh_pct = if stats_ind.p50 > 0.0 {
        (layer_overhead / stats_ind.p50) * 100.0
    } else {
        0.0
    };

    println!("  11 CBs (individual): {stats_ind}");
    println!("  1 CB  (batched):     {stats_bat}");
    println!(
        "  dispatch overhead:   {:.1}us ({:.1}%)  x32 layers = {:.1}us",
        layer_overhead,
        layer_oh_pct,
        layer_overhead * 32.0,
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
    let device = registry.device().raw();
    let queue = device.new_command_queue();

    println!("F-1: Metal Dispatch Overhead Profiling");
    println!("  warmup={WARMUP_ITERS}  iters={BENCH_ITERS}");
    println!();

    bench_empty_dispatch(&queue);
    bench_cb_vs_encoder(&registry, &queue, device);
    bench_llama3_layer_ops(&registry, &queue, device);

    println!("Done.");
}
