//! ⚠️ NON-PRODUCTION PATH — dual-queue concurrency experiment. Not production dispatch;
//! tests whether two MTLCommandQueues provide wall-time overlap.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Dual-queue concurrent execution benchmark for Apple Silicon Metal.
//!
//! Tests whether running two independent GPU workloads on separate
//! `MTLCommandQueue`s provides wall-time overlap benefits compared
//! to sequential execution on a single queue.
//!
//! Profiles modeled after real MoE architectures:
//!   - DeepSeek V3:     E=256, top_k=8, expert_dim=2048, MLA attention
//!   - Llama 4 Maverick: E=128, top_k=1, expert_dim=8192, GQA attention
//!   - Qwen3-235B:      E=128, top_k=8, expert_dim=1536, GQA attention
//!
//! Run with:
//!   cargo bench -p rmlx-core --bench dual_queue_bench

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
use objc2_metal::MTLDevice as _;
use rmlx_metal::{MTLResourceOptions};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 25;

/// Metal objects are thread-safe at the API level (Apple guarantees this).
/// objc2-metal doesn't add Send/Sync, so we use an unsafe wrapper for bench threads.
struct UnsafeSendSync<T>(T);
unsafe impl<T> Send for UnsafeSendSync<T> {}
unsafe impl<T> Sync for UnsafeSendSync<T> {}

// ---------------------------------------------------------------------------
// Model profiles
// ---------------------------------------------------------------------------

struct ModelProfile {
    name: &'static str,
    // MoE expert GEMM (BW-bound) — per-expert shape at prefill M=512
    expert_m: usize, // ≈ (512 × top_k) / num_experts
    expert_k: usize, // hidden_dim
    expert_n: usize, // expert_intermediate_dim
    // Attention projection (Compute-bound) — micro-batch M=256
    attn_m: usize,
    attn_k: usize,
    attn_n: usize,
    description: &'static str,
}

const PROFILES: &[ModelProfile] = &[
    ModelProfile {
        name: "DeepSeek-V3",
        // 256 experts, top_k=8, prefill=512 → M≈16/expert
        expert_m: 16,
        expert_k: 7168,
        expert_n: 2048,
        // MLA Q down-proj: M=256, K=7168, N=24576 (q_lora_rank → heads)
        attn_m: 256,
        attn_k: 7168,
        attn_n: 24576,
        description: "E=256 top8, MLA Q-proj (q_lora→heads)",
    },
    ModelProfile {
        name: "Llama-4-Maverick",
        // 128 experts, top_k=1, prefill=512 → M≈4/expert
        expert_m: 4,
        expert_k: 5120,
        expert_n: 8192,
        // GQA Q proj: M=256, K=5120, N=5120
        attn_m: 256,
        attn_k: 5120,
        attn_n: 5120,
        description: "E=128 top1, GQA Q-proj",
    },
    ModelProfile {
        name: "Qwen3-235B",
        // 128 experts, top_k=8, prefill=512 → M≈32/expert
        expert_m: 32,
        expert_k: 4096,
        expert_n: 1536,
        // GQA Q proj: M=256, K=4096, N=8192
        attn_m: 256,
        attn_k: 4096,
        attn_n: 8192,
        description: "E=128 top8, GQA Q-proj",
    },
    // --- 2026 latest MoE models ---
    ModelProfile {
        name: "Kimi-K2",
        // 384 experts, top_k=8, prefill=512 → M≈11/expert
        expert_m: 11,
        expert_k: 7168,
        expert_n: 2048,
        // MLA Q up-proj: M=256, K=1536(q_lora_rank), N=64*(128+64)=12288
        attn_m: 256,
        attn_k: 7168,
        attn_n: 12288,
        description: "E=384 top8, MLA Q-proj (1T/32B active)",
    },
    ModelProfile {
        name: "GLM-5",
        // 256 experts, top_k=8, prefill=512 → M≈16/expert
        expert_m: 16,
        expert_k: 6144,
        expert_n: 2048,
        // MLA Q up-proj: M=256, K=2048(q_lora_rank), N=64*(192+64)=16384
        attn_m: 256,
        attn_k: 6144,
        attn_n: 16384,
        description: "E=256 top8, MLA+DSA (744B/40B active)",
    },
    ModelProfile {
        name: "MiniMax-M1",
        // 32 experts, top_k=2, prefill=512 → M≈32/expert
        expert_m: 32,
        expert_k: 6144,
        expert_n: 9216,
        // GQA Q proj: M=256, K=6144, N=64*128=8192
        attn_m: 256,
        attn_k: 6144,
        attn_n: 8192,
        description: "E=32 top2, Lightning+Softmax hybrid (456B/46B)",
    },
    ModelProfile {
        name: "MiniMax-M2.5",
        // 256 experts, top_k=8, prefill=512 → M≈16/expert
        expert_m: 16,
        expert_k: 3072,
        expert_n: 1536,
        // GQA Q proj: M=256, K=3072, N=48*128=6144
        attn_m: 256,
        attn_k: 3072,
        attn_n: 6144,
        description: "E=256 top8, Lightning+Softmax (229B/10B active)",
    },
];

// ---------------------------------------------------------------------------
// Stats helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Stats {
    median: f64,
    min: f64,
    mean: f64,
    std_dev: f64,
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
        let median = if n % 2 == 0 {
            (micros[n / 2 - 1] + micros[n / 2]) / 2.0
        } else {
            micros[n / 2]
        };
        let min = micros[0];
        Stats {
            median,
            min,
            mean,
            std_dev,
        }
    }
}

// ---------------------------------------------------------------------------
// Random data generation (LCG -- deterministic, no deps)
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
// Workload wrappers
// ---------------------------------------------------------------------------

fn run_bw_bound(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    qw: &QuantizedWeight,
    x: &Array,
) {
    let _ = ops::quantized::affine_quantized_matmul_batched(registry, x, qw, queue)
        .expect("QMM (BW-bound) failed");
}

fn run_compute_bound(registry: &KernelRegistry, queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>, a: &Array, b: &Array) {
    let _ = ops::matmul::matmul(registry, a, b, queue).expect("GEMM (compute-bound) failed");
}

// ---------------------------------------------------------------------------
// Benchmark modes
// ---------------------------------------------------------------------------

fn bench_sequential(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    qw: &QuantizedWeight,
    qmm_x: &Array,
    gemm_a: &Array,
    gemm_b: &Array,
) -> Duration {
    let start = Instant::now();
    run_bw_bound(registry, queue, qw, qmm_x);
    run_compute_bound(registry, queue, gemm_a, gemm_b);
    start.elapsed()
}

fn bench_sequential_reversed(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    qw: &QuantizedWeight,
    qmm_x: &Array,
    gemm_a: &Array,
    gemm_b: &Array,
) -> Duration {
    let start = Instant::now();
    run_compute_bound(registry, queue, gemm_a, gemm_b);
    run_bw_bound(registry, queue, qw, qmm_x);
    start.elapsed()
}

fn bench_concurrent(
    registry: &KernelRegistry,
    queue_bw: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    queue_compute: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    qw: &QuantizedWeight,
    qmm_x: &Array,
    gemm_a: &Array,
    gemm_b: &Array,
) -> Duration {
    let start = Instant::now();
    // Metal API is thread-safe; objc2-metal just doesn't declare Send/Sync.
    // Wrap all captured state in UnsafeSendSync to satisfy thread::scope Send bounds.
    // Safety: Metal objects are thread-safe, scoped threads ensure lifetimes.
    struct Args<'a> {
        registry: &'a KernelRegistry,
        queue_bw: &'a ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        queue_compute: &'a ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        qw: &'a QuantizedWeight,
        qmm_x: &'a Array,
        gemm_a: &'a Array,
        gemm_b: &'a Array,
    }
    let args = UnsafeSendSync(Args {
        registry, queue_bw, queue_compute, qw, qmm_x, gemm_a, gemm_b,
    });
    // UnsafeSendSync<Args> is Send+Sync, so &UnsafeSendSync<Args> is Send.
    // Force closures to capture `&args` (which is Send because UnsafeSendSync is Sync),
    // not individual fields from args.0 which would bypass the wrapper.
    let args_ref = &args;
    std::thread::scope(|s| {
        let bw_handle = s.spawn(|| {
            let a = &args_ref.0;
            run_bw_bound(a.registry, a.queue_bw, a.qw, a.qmm_x);
        });
        let compute_handle = s.spawn(|| {
            let a = &args_ref.0;
            run_compute_bound(a.registry, a.queue_compute, a.gemm_a, a.gemm_b);
        });
        bw_handle.join().expect("BW-bound thread panicked");
        compute_handle.join().expect("Compute-bound thread panicked");
    });
    start.elapsed()
}

fn bench_bw_only(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    qw: &QuantizedWeight,
    qmm_x: &Array,
) -> Duration {
    let start = Instant::now();
    run_bw_bound(registry, queue, qw, qmm_x);
    start.elapsed()
}

fn bench_compute_only(
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    gemm_a: &Array,
    gemm_b: &Array,
) -> Duration {
    let start = Instant::now();
    run_compute_bound(registry, queue, gemm_a, gemm_b);
    start.elapsed()
}

// ---------------------------------------------------------------------------
// Run one profile
// ---------------------------------------------------------------------------

fn run_profile(registry: &KernelRegistry, device: &ProtocolObject<dyn objc2_metal::MTLDevice>, profile: &ModelProfile) {
    let queue_shared = device.newCommandQueue().unwrap();
    let queue_bw = device.newCommandQueue().unwrap();
    let queue_compute = device.newCommandQueue().unwrap();

    // Prepare BW-bound data (Q4 expert GEMM)
    let group_size: u32 = 32;
    let qw = make_quantized_weight(
        device,
        profile.expert_n,
        profile.expert_k,
        4,
        group_size,
        42,
    );
    let qmm_x = rand_f32_array(device, &[profile.expert_m, profile.expert_k], 99);

    // Prepare compute-bound data (FP16 attention projection)
    let gemm_a = rand_f16_array(device, &[profile.attn_m, profile.attn_k], 77);
    let gemm_b = rand_f16_array(device, &[profile.attn_k, profile.attn_n], 88);

    println!(
        "  Expert (BW):  Q4 QMM  M={:>3}, K={:>5}, N={:>5}",
        profile.expert_m, profile.expert_k, profile.expert_n
    );
    println!(
        "  Attn (Comp):  FP16    M={:>3}, K={:>5}, N={:>5}",
        profile.attn_m, profile.attn_k, profile.attn_n
    );
    println!("  ({})", profile.description);
    println!();

    // --- Baselines ---
    for _ in 0..WARMUP_ITERS {
        run_bw_bound(registry, &queue_shared, &qw, &qmm_x);
    }
    let bw_times: Vec<Duration> = (0..BENCH_ITERS)
        .map(|_| bench_bw_only(registry, &queue_shared, &qw, &qmm_x))
        .collect();
    let bw_stats = Stats::from_durations(&bw_times);

    for _ in 0..WARMUP_ITERS {
        run_compute_bound(registry, &queue_shared, &gemm_a, &gemm_b);
    }
    let compute_times: Vec<Duration> = (0..BENCH_ITERS)
        .map(|_| bench_compute_only(registry, &queue_shared, &gemm_a, &gemm_b))
        .collect();
    let compute_stats = Stats::from_durations(&compute_times);

    // --- Sequential ---
    for _ in 0..WARMUP_ITERS {
        bench_sequential(registry, &queue_shared, &qw, &qmm_x, &gemm_a, &gemm_b);
    }
    let seq_times: Vec<Duration> = (0..BENCH_ITERS)
        .map(|_| bench_sequential(registry, &queue_shared, &qw, &qmm_x, &gemm_a, &gemm_b))
        .collect();
    let seq_stats = Stats::from_durations(&seq_times);

    // --- Sequential reversed ---
    for _ in 0..WARMUP_ITERS {
        bench_sequential_reversed(registry, &queue_shared, &qw, &qmm_x, &gemm_a, &gemm_b);
    }
    let seq_rev_times: Vec<Duration> = (0..BENCH_ITERS)
        .map(|_| bench_sequential_reversed(registry, &queue_shared, &qw, &qmm_x, &gemm_a, &gemm_b))
        .collect();
    let seq_rev_stats = Stats::from_durations(&seq_rev_times);

    // --- Concurrent ---
    for _ in 0..WARMUP_ITERS {
        bench_concurrent(
            registry,
            &queue_bw,
            &queue_compute,
            &qw,
            &qmm_x,
            &gemm_a,
            &gemm_b,
        );
    }
    let conc_times: Vec<Duration> = (0..BENCH_ITERS)
        .map(|_| {
            bench_concurrent(
                registry,
                &queue_bw,
                &queue_compute,
                &qw,
                &qmm_x,
                &gemm_a,
                &gemm_b,
            )
        })
        .collect();
    let conc_stats = Stats::from_durations(&conc_times);

    // --- Results ---
    let best_seq = seq_stats.median.min(seq_rev_stats.median);
    let overlap_ratio = conc_stats.median / best_seq;
    let min_component = bw_stats.median.min(compute_stats.median);
    let saved = best_seq - conc_stats.median;
    let overlap_pct = if min_component > 0.0 {
        (saved / min_component * 100.0).clamp(-100.0, 100.0)
    } else {
        0.0
    };

    println!("  {:32} {:>9} {:>9}", "", "Median", "Min");
    println!("  {:32} {:>9} {:>9}", "", "(us)", "(us)");
    println!(
        "  {:32} {:>9} {:>9}",
        "─".repeat(32),
        "─".repeat(9),
        "─".repeat(9)
    );
    println!(
        "  {:32} {:9.1} {:9.1}",
        "Expert only (Q4 QMM)", bw_stats.median, bw_stats.min
    );
    println!(
        "  {:32} {:9.1} {:9.1}",
        "Attn only (FP16 GEMM)", compute_stats.median, compute_stats.min
    );
    println!(
        "  {:32} {:9.1} {:9.1}",
        "Sequential (expert→attn)", seq_stats.median, seq_stats.min
    );
    println!(
        "  {:32} {:9.1} {:9.1}",
        "Sequential (attn→expert)", seq_rev_stats.median, seq_rev_stats.min
    );
    println!(
        "  {:32} {:9.1} {:9.1}",
        "▸ Concurrent (dual-queue)", conc_stats.median, conc_stats.min
    );
    println!();
    if overlap_ratio < 0.95 {
        println!(
            "  ✓ Overlap effective: {:.1}% wall-time reduction (ratio={:.3}x, {:.0}% of expert hidden)",
            (1.0 - overlap_ratio) * 100.0,
            overlap_ratio,
            overlap_pct,
        );
    } else if overlap_ratio < 1.05 {
        println!(
            "  ~ Marginal overlap (ratio={:.3}x, {:.0}% hidden)",
            overlap_ratio, overlap_pct,
        );
    } else {
        println!(
            "  ✗ No overlap: concurrent {:.1}% slower (contention)",
            (overlap_ratio - 1.0) * 100.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let gpu = GpuDevice::system_default().expect("No GPU device found");
    let device_name = gpu.name().to_string();
    let registry = KernelRegistry::new(gpu);
    ops::register_all(&registry).expect("Failed to register kernels");
    let device = registry.device().raw();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Metal Dual-Queue Overlap Benchmark — MoE Architectures ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Device: {}", device_name);
    println!(
        "Warmup: {} iters, Bench: {} iters",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!("Scenario: Prefill M=512, micro-batch M=256 for attention");
    println!();

    for (i, profile) in PROFILES.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Profile {}: {}", i + 1, profile.name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        run_profile(&registry, device, profile);
        println!();
    }

    println!("Done.");
}
