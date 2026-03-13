//! ⚠️ NON-PRODUCTION PATH — distributed TP benchmark with single-layer forward.
//! Tests TP sharding/allreduce overhead, not full 32-layer TransformerModel pipeline.
//! For production throughput, use e2e_prefill_bench (prefill) or pipeline_bench (decode).
//!
//! Distributed Tensor Parallel Benchmark (Phase I-1 / J-7)
//!
//! Measures single-layer transformer forward pass with and without TP overhead,
//! using Mixtral 8x7B-like config (hidden=4096, intermediate=14336, 32 heads, 8 KV heads).
//!
//! All benchmarks use **f16** dtype (realistic inference precision) and include:
//! 1. **Baseline sync forward**: `forward()` — per-op dispatch (slow, for reference only)
//! 2. **Optimized single-CB forward**: `forward_decode_into_cb()` — all ops in one command buffer
//! 3. **TP-sharded compute**: half-size weights (no communication)
//! 4. **Weight sharding**: time to shard 7 weight matrices for TP=2
//! 5. **Allreduce stub**: overhead of 2x allreduce_sum calls (single-rank identity)
//! 6. **ColumnParallelLinear / RowParallelLinear**: individual TP layer forward times
//!    (distributed feature required)
//! 7. **Real RDMA allreduce**: actual 2-node allreduce over TB5 RDMA
//!    (requires RMLX_RANK, RMLX_WORLD_SIZE, RMLX_COORDINATOR env vars)
//!
//! Run locally (single-node):
//!   cargo bench -p rmlx-nn --bench distributed_bench
//!   cargo bench -p rmlx-nn --bench distributed_bench --features distributed
//!
//! Run on two nodes via the launcher script:
//!   ./scripts/run_distributed_bench.sh

use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelRegistry;
use rmlx_core::ops;
use rmlx_metal::device::GpuDevice;
use rmlx_nn::{Attention, AttentionConfig, FeedForward, Linear, LinearConfig, TransformerBlock};

// ─── Mixtral 8x7B-like config (dense attention + gated FFN per expert) ───
const HIDDEN_SIZE: usize = 4096;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 14336;
const SEQ_LEN: usize = 1; // decode token
const RMS_NORM_EPS: f32 = 1e-5;

const WARMUP_ITERS: usize = 5;
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
        Stats {
            mean,
            std_dev,
            p50,
            p95,
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
            "mean={:8.1}us std={:7.1}us p50={:8.1}us p95={:8.1}us min={:8.1}us max={:8.1}us (n={})",
            self.mean, self.std_dev, self.p50, self.p95, self.min, self.max, self.count
        )
    }
}

// ---------------------------------------------------------------------------
// Random array generation (f16 deterministic PRNG — realistic inference dtype)
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        return (sign << 15) as u16; // zero / subnormal → zero
    }
    if exp == 0xFF {
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        return (sign << 15) as u16; // underflow → zero
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

fn rand_array(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    shape: &[usize],
    seed: u64,
) -> Array {
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

fn rand_array_ones(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize]) -> Array {
    let numel: usize = shape.iter().product();
    let one_f16 = f32_to_f16_bits(1.0);
    let mut f16_bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        f16_bytes.extend_from_slice(&one_f16.to_le_bytes());
    }
    Array::from_bytes(device, &f16_bytes, shape.to_vec(), DType::Float16)
}

// ---------------------------------------------------------------------------
// Layer construction helpers
// ---------------------------------------------------------------------------

fn make_linear(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    in_f: usize,
    out_f: usize,
    seed: u64,
) -> Linear {
    let weight = rand_array(device, &[out_f, in_f], seed);
    Linear::from_arrays(
        LinearConfig {
            in_features: in_f,
            out_features: out_f,
            has_bias: false,
        },
        weight,
        None,
    )
    .expect("linear from_arrays")
}

/// Build a full-size (unsharded) transformer block for baseline measurement.
fn build_transformer_block(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;

    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 1);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size, 2);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size, 3);
    let o_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE, 4);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 5);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM, 6);
    let down_proj = make_linear(device, INTERMEDIATE_DIM, HIDDEN_SIZE, 7);
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
    };

    let norm1_weight = rand_array_ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = rand_array_ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

/// Build a half-size (TP=2 sharded) transformer block for TP compute measurement.
///
/// Uses `rank` to vary the PRNG seed so each rank has different weights.
/// - Q/K/V: column-parallel (output rows halved)
/// - O: row-parallel (input cols halved)
/// - gate/up: column-parallel (output rows halved)
/// - down: row-parallel (input cols halved)
fn build_sharded_transformer_block(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    rank: u32,
) -> TransformerBlock {
    let kv_size = NUM_KV_HEADS * HEAD_DIM;
    let seed_offset = rank as u64 * 100;

    // Column-parallel: shard output features (rows) by 2
    let q_proj = make_linear(device, HIDDEN_SIZE, HIDDEN_SIZE / 2, 11 + seed_offset);
    let k_proj = make_linear(device, HIDDEN_SIZE, kv_size / 2, 12 + seed_offset);
    let v_proj = make_linear(device, HIDDEN_SIZE, kv_size / 2, 13 + seed_offset);
    // Row-parallel: shard input features (cols) by 2
    let o_proj = make_linear(device, HIDDEN_SIZE / 2, HIDDEN_SIZE, 14 + seed_offset);

    let attn_config = AttentionConfig {
        num_heads: NUM_HEADS / 2, // half the heads on this rank
        num_kv_heads: NUM_KV_HEADS / 2,
        head_dim: HEAD_DIM,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    };
    let attention =
        Attention::from_layers(attn_config, q_proj, k_proj, v_proj, o_proj).expect("attention");

    // Column-parallel: shard output features (rows) by 2
    let gate_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM / 2, 15 + seed_offset);
    let up_proj = make_linear(device, HIDDEN_SIZE, INTERMEDIATE_DIM / 2, 16 + seed_offset);
    // Row-parallel: shard input features (cols) by 2
    let down_proj = make_linear(device, INTERMEDIATE_DIM / 2, HIDDEN_SIZE, 17 + seed_offset);
    let ffn = FeedForward::Gated {
        gate_proj,
        up_proj,
        down_proj,
        gate_up_merged_weight: None,
        gate_up_merged_weight_t: None,
    };

    let norm1_weight = rand_array_ones(device, &[HIDDEN_SIZE]);
    let norm2_weight = rand_array_ones(device, &[HIDDEN_SIZE]);

    TransformerBlock::from_parts(0, attention, ffn, norm1_weight, norm2_weight, RMS_NORM_EPS)
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

fn run_bench<F>(label: &str, mut f: F) -> Stats
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
    println!("  {:40} {}", label, stats);
    stats
}

// ---------------------------------------------------------------------------
// Benchmark: single-node baseline forward pass
// ---------------------------------------------------------------------------

fn bench_baseline_sync(
    block: &TransformerBlock,
    input: &Array,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Stats {
    println!("\n=== Baseline sync: single-layer forward() — per-op dispatch (slow) ===");
    run_bench("baseline_sync_forward", || {
        let _ = block
            .forward(input, None, None, None, None, registry, queue)
            .expect("baseline forward");
    })
}

fn bench_optimized_single_cb(
    block: &TransformerBlock,
    input: &Array,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Stats {
    println!("\n=== Optimized: forward_decode_into_cb() — all ops in one command buffer ===");
    run_bench("optimized_single_cb", || {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = block
            .forward_decode_into_cb(input, None, None, None, None, registry, &cb)
            .expect("single_cb forward");
        cb.commit();
        cb.waitUntilCompleted();
    })
}

// ---------------------------------------------------------------------------
// Benchmark: TP-sharded compute (half-size weights, no communication)
// ---------------------------------------------------------------------------

fn bench_sharded_compute(
    block: &TransformerBlock,
    input: &Array,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Stats {
    println!("\n=== TP-sharded compute (half-size weights, single-CB, no allreduce) ===");
    println!("  Simulates rank 0 of TP=2: Q/K/V/gate/up halved, O/down halved.");
    run_bench("sharded_single_cb (TP=2 rank0)", || {
        let cb = queue.commandBufferWithUnretainedReferences().unwrap();
        let _ = block
            .forward_decode_into_cb(input, None, None, None, None, registry, &cb)
            .expect("sharded forward");
        cb.commit();
        cb.waitUntilCompleted();
    })
}

// ---------------------------------------------------------------------------
// Benchmark: weight sharding overhead
// ---------------------------------------------------------------------------

fn bench_weight_sharding(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> Stats {
    use rmlx_nn::{ColumnParallelLinear, RowParallelLinear};

    println!("\n=== Weight sharding overhead (Mixtral-like, TP=2) ===");

    let q_weight = rand_array(device, &[HIDDEN_SIZE, HIDDEN_SIZE], 100);
    let k_weight = rand_array(device, &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], 101);
    let v_weight = rand_array(device, &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], 102);
    let o_weight = rand_array(device, &[HIDDEN_SIZE, HIDDEN_SIZE], 103);
    let gate_weight = rand_array(device, &[INTERMEDIATE_DIM, HIDDEN_SIZE], 104);
    let up_weight = rand_array(device, &[INTERMEDIATE_DIM, HIDDEN_SIZE], 105);
    let down_weight = rand_array(device, &[HIDDEN_SIZE, INTERMEDIATE_DIM], 106);

    let stats = run_bench("shard_all_weights (7 matrices)", || {
        let _ = ColumnParallelLinear::shard_weight(&q_weight, 0, 2);
        let _ = ColumnParallelLinear::shard_weight(&k_weight, 0, 2);
        let _ = ColumnParallelLinear::shard_weight(&v_weight, 0, 2);
        let _ = RowParallelLinear::shard_weight(&o_weight, 0, 2);
        let _ = ColumnParallelLinear::shard_weight(&gate_weight, 0, 2);
        let _ = ColumnParallelLinear::shard_weight(&up_weight, 0, 2);
        let _ = RowParallelLinear::shard_weight(&down_weight, 0, 2);
    });

    let total_params = HIDDEN_SIZE * HIDDEN_SIZE
        + NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE
        + NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + INTERMEDIATE_DIM * HIDDEN_SIZE
        + INTERMEDIATE_DIM * HIDDEN_SIZE
        + HIDDEN_SIZE * INTERMEDIATE_DIM;
    let total_bytes = total_params * 2; // f16
    println!(
        "  Total weight params: {} ({:.1} MB f16)",
        total_params,
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    stats
}

// ---------------------------------------------------------------------------
// Distributed context initialization
// ---------------------------------------------------------------------------

/// Distributed context for the benchmark.
#[cfg(feature = "distributed")]
struct BenchDistCtx {
    group: rmlx_distributed::Group,
    rank: u32,
    world_size: u32,
    is_multi_rank: bool,
}

/// Initialize distributed context from env vars.
///
/// When RMLX_WORLD_SIZE > 1 and RMLX_COORDINATOR is set, uses `rmlx_distributed::init()`
/// to establish real RDMA connections via the coordinator-mediated all_gather protocol.
/// Otherwise falls back to single-rank stub mode.
#[cfg(feature = "distributed")]
fn init_distributed() -> BenchDistCtx {
    use rmlx_distributed::{BackendHint, InitConfig};

    let world_size: u32 = std::env::var("RMLX_WORLD_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    if world_size > 1 {
        let config = InitConfig {
            strict: true,
            backend: BackendHint::Auto,
            ..Default::default()
        };

        eprintln!(
            "[distributed_bench] Initializing RDMA (world_size={}, rank={})...",
            world_size,
            std::env::var("RMLX_RANK").unwrap_or_else(|_| "?".into()),
        );

        match rmlx_distributed::init(config) {
            Ok(ctx) => {
                eprintln!(
                    "[distributed_bench] RDMA init OK: rank={}, world_size={}, backend={:?}, transport={}",
                    ctx.rank, ctx.world_size, ctx.backend, ctx.group.has_transport(),
                );
                BenchDistCtx {
                    group: ctx.group,
                    rank: ctx.rank,
                    world_size: ctx.world_size,
                    is_multi_rank: true,
                }
            }
            Err(e) => {
                eprintln!(
                    "[distributed_bench] RDMA init FAILED: {}. Falling back to single-rank stub.",
                    e
                );
                BenchDistCtx {
                    group: rmlx_distributed::Group::world(1, 0).expect("single-rank group"),
                    rank: 0,
                    world_size: 1,
                    is_multi_rank: false,
                }
            }
        }
    } else {
        BenchDistCtx {
            group: rmlx_distributed::Group::world(1, 0).expect("single-rank group"),
            rank: 0,
            world_size: 1,
            is_multi_rank: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark: allreduce (real RDMA or single-rank stub)
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
fn bench_allreduce(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    dist: &BenchDistCtx,
) -> Stats {
    if dist.is_multi_rank {
        println!(
            "\n=== Real RDMA allreduce (rank={}, world_size={}) ===",
            dist.rank, dist.world_size
        );
    } else {
        println!("\n=== Simulated allreduce overhead (single-rank stub) ===");
    }

    let attn_output = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 200);
    let bytes_per_allreduce = SEQ_LEN * HIDDEN_SIZE * 2; // f16
    println!(
        "  Per allreduce payload: {} bytes ({:.1} KB)",
        bytes_per_allreduce,
        bytes_per_allreduce as f64 / 1024.0
    );

    let label = if dist.is_multi_rank {
        "2x allreduce_sum (RDMA)"
    } else {
        "2x allreduce_sum (stub, identity)"
    };

    let stats = run_bench(label, || {
        let _ = dist.group.allreduce_sum(&attn_output, device).unwrap();
        let _ = dist.group.allreduce_sum(&attn_output, device).unwrap();
    });

    if dist.is_multi_rank {
        let single_stats = run_bench("1x allreduce_sum (RDMA)", || {
            let _ = dist.group.allreduce_sum(&attn_output, device).unwrap();
        });
        println!(
            "  Single RDMA allreduce latency: {:.1} us",
            single_stats.mean
        );

        // Also measure with larger payload
        let large_output = rand_array(device, &[SEQ_LEN, INTERMEDIATE_DIM], 201);
        let large_bytes = SEQ_LEN * INTERMEDIATE_DIM * 2;
        println!(
            "  Large payload: {} bytes ({:.1} KB)",
            large_bytes,
            large_bytes as f64 / 1024.0
        );
        let _large_stats = run_bench("1x allreduce_sum (large, RDMA)", || {
            let _ = dist.group.allreduce_sum(&large_output, device).unwrap();
        });
    }

    stats
}

// ---------------------------------------------------------------------------
// Benchmark: ColumnParallelLinear + RowParallelLinear forward
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
fn bench_parallel_linear(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    dist: &BenchDistCtx,
) -> (Stats, Stats) {
    use rmlx_nn::{ColumnParallelLinear, RowParallelLinear};

    if dist.is_multi_rank {
        println!(
            "\n=== Parallel linear layer forward (RDMA, rank={}, world_size={}) ===",
            dist.rank, dist.world_size
        );

        let full_weight = rand_array(device, &[INTERMEDIATE_DIM, HIDDEN_SIZE], 300);
        let shard = ColumnParallelLinear::shard_weight(&full_weight, dist.rank, dist.world_size);
        let col_layer = ColumnParallelLinear::new(
            shard,
            None,
            INTERMEDIATE_DIM,
            HIDDEN_SIZE,
            dist.rank,
            dist.world_size,
        )
        .expect("ColumnParallelLinear");

        let input = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 42);

        let col_stats = run_bench("ColumnParallel [4096->7168] (RDMA)", || {
            let _ = col_layer
                .forward_with_group(&input, &dist.group, registry, queue)
                .expect("col forward");
        });

        let full_weight_row = rand_array(device, &[HIDDEN_SIZE, INTERMEDIATE_DIM], 301);
        let shard_row =
            RowParallelLinear::shard_weight(&full_weight_row, dist.rank, dist.world_size);
        let shard_in = INTERMEDIATE_DIM / dist.world_size as usize;
        let row_layer = RowParallelLinear::new(
            shard_row,
            None,
            HIDDEN_SIZE,
            INTERMEDIATE_DIM,
            dist.rank,
            dist.world_size,
        )
        .expect("RowParallelLinear");

        let row_input = rand_array(device, &[SEQ_LEN, shard_in], 43);

        let row_stats = run_bench("RowParallel    [7168->4096] (RDMA)", || {
            let _ = row_layer
                .forward_with_group(&row_input, &dist.group, registry, queue)
                .expect("row forward");
        });

        (col_stats, row_stats)
    } else {
        println!("\n=== Parallel linear layer forward (single-rank stub, half-size weights) ===");
        println!("  Uses world_size=1 group with pre-sharded (half-size) weights.");

        let full_weight = rand_array(device, &[INTERMEDIATE_DIM, HIDDEN_SIZE], 300);
        let shard = ColumnParallelLinear::shard_weight(&full_weight, 0, 2);
        let col_layer =
            ColumnParallelLinear::new(shard, None, INTERMEDIATE_DIM / 2, HIDDEN_SIZE, 0, 1)
                .expect("ColumnParallelLinear");

        let input = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 42);

        let col_stats = run_bench("ColumnParallel [4096->7168] (half)", || {
            let _ = col_layer
                .forward_with_group(&input, &dist.group, registry, queue)
                .expect("col forward");
        });

        let full_weight_row = rand_array(device, &[HIDDEN_SIZE, INTERMEDIATE_DIM], 301);
        let shard_row = RowParallelLinear::shard_weight(&full_weight_row, 0, 2);
        let row_layer =
            RowParallelLinear::new(shard_row, None, HIDDEN_SIZE, INTERMEDIATE_DIM / 2, 0, 1)
                .expect("RowParallelLinear");

        let row_input = rand_array(device, &[SEQ_LEN, INTERMEDIATE_DIM / 2], 43);

        let row_stats = run_bench("RowParallel    [7168->4096] (half)", || {
            let _ = row_layer
                .forward_with_group(&row_input, &dist.group, registry, queue)
                .expect("row forward");
        });

        (col_stats, row_stats)
    }
}

// ---------------------------------------------------------------------------
// Benchmark: TP forward_with_group (real allreduce, multi-rank only)
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
fn bench_tp_forward_with_group(
    block: &TransformerBlock,
    input: &Array,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    dist: &BenchDistCtx,
) -> Stats {
    println!(
        "\n=== TP forward_with_group (RDMA allreduce, rank={}, world_size={}) ===",
        dist.rank, dist.world_size
    );
    println!("  Full layer forward: compute (half weights) + 2x real allreduce");

    run_bench("tp_forward_with_group (RDMA)", || {
        let _ = block
            .forward_with_group(
                input,
                None,
                None,
                None,
                None,
                &dist.group,
                device,
                registry,
                queue,
            )
            .expect("tp forward_with_group");
    })
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
    let queue = device.newCommandQueue().unwrap();

    // ── Distributed init (env-based RDMA or single-node fallback) ──
    #[cfg(feature = "distributed")]
    let dist = init_distributed();

    println!("\nConfig: Mixtral 8x7B-like single expert layer");
    println!(
        "  hidden={}, heads={}/{}, head_dim={}, intermediate={}, seq_len={}",
        HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_DIM, SEQ_LEN
    );
    println!(
        "  dtype: float16, warmup={}, bench_iters={}",
        WARMUP_ITERS, BENCH_ITERS
    );

    // ── 1. Baseline sync: full-size single-layer forward (per-op dispatch) ──
    let block = build_transformer_block(device);
    let input = rand_array(device, &[SEQ_LEN, HIDDEN_SIZE], 42);
    let baseline_stats = bench_baseline_sync(&block, &input, &registry, &queue);

    // ── 2. Optimized: forward_decode_into_cb (all ops in one CB) ──
    let optimized_stats = bench_optimized_single_cb(&block, &input, &registry, &queue);

    // ── 3. TP-sharded compute (half-size weights, single-CB) ──
    #[cfg(feature = "distributed")]
    let rank_for_shard = dist.rank;
    #[cfg(not(feature = "distributed"))]
    let rank_for_shard = 0u32;
    let sharded_block = build_sharded_transformer_block(device, rank_for_shard);
    let sharded_stats = bench_sharded_compute(&sharded_block, &input, &registry, &queue);

    // ── 4. Weight sharding overhead ──
    let shard_stats = bench_weight_sharding(device);

    // ── 5. Allreduce (real RDMA or single-rank stub) ──
    #[cfg(feature = "distributed")]
    let allreduce_stats = bench_allreduce(device, &dist);

    // ── 6. Parallel linear layers (requires distributed feature) ──
    #[cfg(feature = "distributed")]
    let (_col_stats, _row_stats) = bench_parallel_linear(device, &registry, &queue, &dist);

    // ── 7. TP forward_with_group (multi-rank only) ──
    #[cfg(feature = "distributed")]
    let tp_forward_stats = if dist.is_multi_rank {
        Some(bench_tp_forward_with_group(
            &sharded_block,
            &input,
            device,
            &registry,
            &queue,
            &dist,
        ))
    } else {
        None
    };

    // ── Summary ──
    println!("\n========================================================================");
    println!("SUMMARY (f16)");
    println!("========================================================================");

    #[cfg(feature = "distributed")]
    println!(
        "  {:44} rank={}/{} (transport={})",
        "Distributed mode", dist.rank, dist.world_size, dist.is_multi_rank
    );

    println!(
        "  {:44} {:>8.1} us",
        "Baseline sync forward (per-op dispatch)", baseline_stats.mean
    );
    println!(
        "  {:44} {:>8.1} us",
        "Optimized single-CB forward (full weights)", optimized_stats.mean
    );

    let sync_vs_cb = if optimized_stats.mean > 0.0 {
        baseline_stats.mean / optimized_stats.mean
    } else {
        0.0
    };
    println!("  {:44} {:>8.2}x", "  Sync / single-CB ratio", sync_vs_cb);
    println!(
        "  {:44} {:>8.1} us",
        "Sharded single-CB (TP=2, half weights)", sharded_stats.mean
    );

    let compute_saving = optimized_stats.mean - sharded_stats.mean;
    let compute_pct = if optimized_stats.mean > 0.0 {
        compute_saving / optimized_stats.mean * 100.0
    } else {
        0.0
    };
    println!(
        "  {:44} {:>8.1} us ({:.1}% reduction)",
        "Compute saving from TP=2 (vs single-CB)", compute_saving, compute_pct
    );
    println!(
        "  {:44} {:>8.1} us",
        "Weight sharding (7 matrices, one-time)", shard_stats.mean
    );

    #[cfg(feature = "distributed")]
    {
        let ar_label = if dist.is_multi_rank {
            "Allreduce (2x, real RDMA)"
        } else {
            "Allreduce stub (2x per layer)"
        };
        println!("  {:44} {:>8.1} us", ar_label, allreduce_stats.mean);

        if let Some(ref tp_stats) = tp_forward_stats {
            println!();
            println!("  Real RDMA TP measurements (TB5):");
            println!(
                "  {:44} {:>8.1} us",
                "  TP forward_with_group (RDMA)", tp_stats.mean
            );
            let real_speedup = if tp_stats.mean > 0.0 {
                optimized_stats.mean / tp_stats.mean
            } else {
                0.0
            };
            println!(
                "  {:44} {:>8.2}x",
                "  Real TP=2 speedup vs single-CB", real_speedup
            );
            let comm_overhead = tp_stats.mean - sharded_stats.mean;
            let comm_pct = if tp_stats.mean > 0.0 {
                comm_overhead / tp_stats.mean * 100.0
            } else {
                0.0
            };
            println!(
                "  {:44} {:>8.1} us ({:.1}%)",
                "  Communication overhead", comm_overhead, comm_pct
            );
        } else {
            // Estimate real TP layer time
            let estimated_rdma_allreduce_us = 10.0;
            let estimated_tp_layer_time = sharded_stats.mean + 2.0 * estimated_rdma_allreduce_us;
            println!();
            println!("  Estimated real TP layer time (TB5 RDMA):");
            println!(
                "  {:44} {:>8.1} us",
                "  sharded compute + 2x RDMA allreduce", estimated_tp_layer_time
            );
            let speedup = if estimated_tp_layer_time > 0.0 {
                optimized_stats.mean / estimated_tp_layer_time
            } else {
                0.0
            };
            println!(
                "  {:44} {:>8.2}x",
                "  Estimated TP=2 speedup vs single-CB", speedup
            );
        }
    }

    #[cfg(not(feature = "distributed"))]
    {
        // Estimate real TP layer time
        let estimated_rdma_allreduce_us = 10.0;
        let estimated_tp_layer_time = sharded_stats.mean + 2.0 * estimated_rdma_allreduce_us;
        println!();
        println!("  Estimated real TP layer time (TB5 RDMA):");
        println!(
            "  {:44} {:>8.1} us",
            "  sharded compute + 2x RDMA allreduce", estimated_tp_layer_time
        );
        let speedup = if estimated_tp_layer_time > 0.0 {
            optimized_stats.mean / estimated_tp_layer_time
        } else {
            0.0
        };
        println!(
            "  {:44} {:>8.2}x",
            "  Estimated TP=2 speedup vs single-CB", speedup
        );
        println!();
        println!("  (Enable `distributed` feature for allreduce + ColumnParallel/RowParallel benchmarks)");
        println!("  cargo bench -p rmlx-nn --bench distributed_bench --features distributed");
    }

    println!();
}
