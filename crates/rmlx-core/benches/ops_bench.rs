//! ⚠️ NON-PRODUCTION PATH — individual op benchmarks with per-op sync CB.
//! Each op creates+commits+waits its own CB; overhead not representative of production.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! GPU Ops Performance Benchmark
//!
//! Benchmarks all major RMLX GPU operations with sizes matching MLX benchmarks
//! for fair comparison.
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench ops_bench

use std::time::{Duration, Instant};

use rmlx_core::array::Array;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 100;

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Stats {
    mean: f64,
    std_dev: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
    count: usize,
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
        let p99 = percentile(&micros, 99.0);
        Stats {
            mean,
            std_dev,
            p50,
            p95,
            p99,
            min: micros[0],
            max: micros[n - 1],
            count: n,
        }
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

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:8.1}us std={:7.1}us p50={:8.1}us p95={:8.1}us p99={:8.1}us min={:8.1}us max={:8.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.p99, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Random array generation (deterministic PRNG)
// ---------------------------------------------------------------------------

fn rand_array(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize], seed: u64) -> Array {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut state = seed;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
        data.push(val as f32);
    }
    Array::from_slice(device, &data, shape.to_vec())
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn bench<F>(name: &str, gflops_fn: Option<f64>, mut f: F)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        f();
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        latencies.push(start.elapsed());
    }

    let stats = Stats::from_durations(&latencies);
    print!("  {:<40} {}", name, stats);
    if let Some(theoretical_flops) = gflops_fn {
        let gflops = theoretical_flops / (stats.mean * 1e-6) / 1e9;
        print!("  [{:.1} GFLOPS]", gflops);
    }
    println!();
}

// ---------------------------------------------------------------------------
// Benchmark entry point
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
    let queue = device.newCommandQueue().unwrap();

    println!(
        "Config: warmup={}, bench_iters={}\n",
        WARMUP_ITERS, BENCH_ITERS
    );

    // =====================================================================
    // matmul
    // =====================================================================
    println!("=== matmul ===");
    {
        let a = rand_array(device, &[1024, 1024], 1);
        let b = rand_array(device, &[1024, 1024], 2);
        let flops = 2.0 * 1024.0 * 1024.0 * 1024.0;
        bench("matmul [1024x1024] x [1024x1024]", Some(flops), || {
            let _ = ops::matmul::matmul(&registry, &a, &b, &queue).unwrap();
        });
    }
    {
        let a = rand_array(device, &[4096, 4096], 3);
        let b = rand_array(device, &[4096, 4096], 4);
        let flops = 2.0 * 4096.0 * 4096.0 * 4096.0;
        bench("matmul [4096x4096] x [4096x4096]", Some(flops), || {
            let _ = ops::matmul::matmul(&registry, &a, &b, &queue).unwrap();
        });
    }

    // =====================================================================
    // softmax
    // =====================================================================
    println!("\n=== softmax ===");
    {
        // [8, 1024, 4096] -- softmax over last axis (4096)
        let x = rand_array(device, &[8 * 1024, 4096], 10);
        // softmax flattens leading dims, so [8*1024, 4096] works for 3D
        bench("softmax [8, 1024, 4096] axis=-1", None, || {
            let _ = ops::softmax::softmax(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // rms_norm
    // =====================================================================
    println!("\n=== rms_norm ===");
    {
        // rms_norm requires 2D input [rows, axis_size]
        let x = rand_array(device, &[8 * 1024, 4096], 20);
        let w = rand_array(device, &[4096], 21);
        bench("rms_norm [8*1024, 4096] eps=1e-5", None, || {
            let _ = ops::rms_norm::rms_norm(&registry, &x, &w, 1e-5, &queue).unwrap();
        });
    }

    // =====================================================================
    // layer_norm
    // =====================================================================
    println!("\n=== layer_norm ===");
    {
        let x = rand_array(device, &[8 * 1024, 4096], 25);
        let w = rand_array(device, &[4096], 26);
        let b = rand_array(device, &[4096], 27);
        bench("layer_norm [8*1024, 4096] eps=1e-5", None, || {
            let _ = ops::layer_norm::layer_norm(&registry, &x, Some(&w), Some(&b), 1e-5, &queue)
                .unwrap();
        });
    }

    // =====================================================================
    // rope (on-the-fly variant -- no precomputed tables needed)
    // =====================================================================
    println!("\n=== rope ===");
    {
        // Vector: [1*32, 1, 128] = [32, 1, 128] (batch*n_heads=32, seq_len=1, head_dim=128)
        let x = rand_array(device, &[32, 1, 128], 30);
        bench("rope_otf [32, 1, 128] (decode)", None, || {
            let _ =
                ops::rope::rope_otf(&registry, &x, 0, 1.0, 10000.0, false, true, &queue).unwrap();
        });
    }
    {
        // Matrix: [1*32, 1024, 128] = [32, 1024, 128]
        let x = rand_array(device, &[32, 1024, 128], 31);
        bench("rope_otf [32, 1024, 128] (prefill)", None, || {
            let _ =
                ops::rope::rope_otf(&registry, &x, 0, 1.0, 10000.0, false, true, &queue).unwrap();
        });
    }

    // =====================================================================
    // sdpa (Scaled Dot-Product Attention)
    // =====================================================================
    // RMLX SDPA is per-head 2D [tokens, head_dim]. To match MLX's multi-head
    // benchmark (4D [1, 32, 1024, 128] with 8 KV heads for GQA), we loop
    // over all 32 query heads. GQA: 32 Q heads, 8 KV heads (ratio 4:1).
    println!("\n=== sdpa ===");
    {
        let n_q_heads: usize = 32;
        let n_kv_heads: usize = 8;
        let head_dim: usize = 128;
        let q_sl: usize = 1024;
        let k_sl: usize = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Per-head Q arrays (32 heads)
        let q_heads: Vec<Array> = (0..n_q_heads)
            .map(|i| rand_array(device, &[q_sl, head_dim], 40 + i as u64))
            .collect();
        // Per-head KV arrays (8 KV heads, each shared by 4 Q heads)
        let k_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 140 + i as u64))
            .collect();
        let v_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 240 + i as u64))
            .collect();

        // FLOPS for all heads: n_q_heads * (2*N*S*D for QK^T + 2*N*S*D for attn@V)
        let flops =
            n_q_heads as f64 * 2.0 * (q_sl as f64) * (k_sl as f64) * (head_dim as f64) * 2.0;
        let gqa_ratio = n_q_heads / n_kv_heads;
        #[allow(clippy::needless_range_loop)]
        bench("sdpa 32h GQA [1024,128] (32Q/8KV)", Some(flops), || {
            for h in 0..n_q_heads {
                let kv_idx = h / gqa_ratio;
                let _ = ops::sdpa::sdpa(
                    &registry,
                    &q_heads[h],
                    &k_heads[kv_idx],
                    &v_heads[kv_idx],
                    None,
                    scale,
                    false,
                    &queue,
                )
                .unwrap();
            }
        });
    }
    {
        // Causal SDPA — 32 heads, GQA 32Q/8KV
        let n_q_heads: usize = 32;
        let n_kv_heads: usize = 8;
        let head_dim: usize = 128;
        let q_sl: usize = 1024;
        let k_sl: usize = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_heads: Vec<Array> = (0..n_q_heads)
            .map(|i| rand_array(device, &[q_sl, head_dim], 43 + i as u64))
            .collect();
        let k_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 143 + i as u64))
            .collect();
        let v_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 243 + i as u64))
            .collect();

        let flops =
            n_q_heads as f64 * 2.0 * (q_sl as f64) * (k_sl as f64) * (head_dim as f64) * 2.0;
        let gqa_ratio = n_q_heads / n_kv_heads;
        #[allow(clippy::needless_range_loop)]
        bench("sdpa 32h GQA [1024,128] causal", Some(flops), || {
            for h in 0..n_q_heads {
                let kv_idx = h / gqa_ratio;
                let _ = ops::sdpa::sdpa(
                    &registry,
                    &q_heads[h],
                    &k_heads[kv_idx],
                    &v_heads[kv_idx],
                    None,
                    scale,
                    true,
                    &queue,
                )
                .unwrap();
            }
        });
    }
    {
        // Decode path: single query token, 32 heads GQA
        let n_q_heads: usize = 32;
        let n_kv_heads: usize = 8;
        let head_dim: usize = 128;
        let k_sl: usize = 1024;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_heads: Vec<Array> = (0..n_q_heads)
            .map(|i| rand_array(device, &[1, head_dim], 46 + i as u64))
            .collect();
        let k_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 146 + i as u64))
            .collect();
        let v_heads: Vec<Array> = (0..n_kv_heads)
            .map(|i| rand_array(device, &[k_sl, head_dim], 246 + i as u64))
            .collect();

        let gqa_ratio = n_q_heads / n_kv_heads;
        #[allow(clippy::needless_range_loop)]
        bench("sdpa 32h GQA [1,128]x[1024,128] decode", None, || {
            for h in 0..n_q_heads {
                let kv_idx = h / gqa_ratio;
                let _ = ops::sdpa::sdpa(
                    &registry,
                    &q_heads[h],
                    &k_heads[kv_idx],
                    &v_heads[kv_idx],
                    None,
                    scale,
                    false,
                    &queue,
                )
                .unwrap();
            }
        });
    }

    // =====================================================================
    // gelu
    // =====================================================================
    println!("\n=== gelu ===");
    {
        let x = rand_array(device, &[8 * 1024 * 4096], 50);
        bench("gelu [8*1024*4096]", None, || {
            let _ = ops::gelu::gelu(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // silu
    // =====================================================================
    println!("\n=== silu ===");
    {
        let x = rand_array(device, &[8 * 1024 * 4096], 60);
        bench("silu [8*1024*4096]", None, || {
            let _ = ops::silu::silu(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // binary::add
    // =====================================================================
    println!("\n=== binary::add ===");
    {
        let a = rand_array(device, &[32, 1024, 1024], 70);
        let b = rand_array(device, &[32, 1024, 1024], 71);
        bench("add [32, 1024, 1024]", None, || {
            let _ = ops::binary::add(&registry, &a, &b, &queue).unwrap();
        });
    }

    // =====================================================================
    // unary::neg
    // =====================================================================
    println!("\n=== unary::neg ===");
    {
        let x = rand_array(device, &[10000, 1000], 80);
        bench("neg [10000, 1000]", None, || {
            let _ = ops::unary::neg(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // unary::exp
    // =====================================================================
    println!("\n=== unary::exp ===");
    {
        let x = rand_array(device, &[10000, 1000], 81);
        bench("exp [10000, 1000]", None, || {
            let _ = ops::unary::exp(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // reduce::sum
    // =====================================================================
    // MLX benchmark uses axis=0 reduce on [32M] (1D), which for 1D is
    // equivalent to full reduce. For 2D we benchmark axis=0 (Col) to match
    // MLX's mx.sum(x, axis=0) semantics.
    println!("\n=== reduce ===");
    {
        // axis=0 reduce on 2D: [1024, 1024] -> [1024] (matches MLX axis=0)
        let x = rand_array(device, &[1024, 1024], 90);
        bench("sum [1024, 1024] axis=0 (col reduce)", None, || {
            let _ = ops::reduce::reduce(
                &registry,
                &x,
                ops::reduce::ReduceOp::Sum,
                ops::reduce::ReduceAxis::Col,
                &queue,
            )
            .unwrap();
        });
    }
    {
        // Row reduction: [1024, 1024] -> [1024] (axis=-1)
        let x = rand_array(device, &[1024, 1024], 91);
        bench("sum [1024, 1024] axis=-1 (row reduce)", None, || {
            let _ = ops::reduce::reduce(
                &registry,
                &x,
                ops::reduce::ReduceOp::Sum,
                ops::reduce::ReduceAxis::Row,
                &queue,
            )
            .unwrap();
        });
    }
    {
        // Full reduce for reference
        let x = rand_array(device, &[32 * 1024 * 1024], 92);
        bench("sum [32M] (full reduce, reference)", None, || {
            let _ = ops::reduce::sum(&registry, &x, &queue).unwrap();
        });
    }

    // =====================================================================
    // gemv
    // =====================================================================
    println!("\n=== gemv ===");
    {
        // gemv: [4096, 4096] * [4096] -> [4096]
        let mat = rand_array(device, &[4096, 4096], 100);
        let vec = rand_array(device, &[4096], 101);
        let flops = 2.0 * 4096.0 * 4096.0;
        bench("gemv [4096, 4096] * [4096]", Some(flops), || {
            let _ = ops::gemv::gemv(&registry, &mat, &vec, &queue).unwrap();
        });
    }

    // =====================================================================
    // sort
    // =====================================================================
    println!("\n=== sort ===");
    {
        let x = rand_array(device, &[1024, 1024], 110);
        bench("sort [1024, 1024] axis=1 ascending", None, || {
            let _ = ops::sort::sort(&registry, &x, 1, false, &queue).unwrap();
        });
    }

    println!("\n========== Benchmark complete ==========");
}
