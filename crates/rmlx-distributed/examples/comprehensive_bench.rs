//! Comprehensive RDMA benchmark for fair comparison with MLX JACCL.
//!
//! Benchmarks send/recv, allreduce (f32/f16), allgather, broadcast,
//! reduce_scatter, all_to_all, and EP dispatch/combine patterns.
//!
//! Usage:
//!   # Rank 0 (coordinator):
//!   RMLX_RANK=0 RMLX_WORLD_SIZE=2 RMLX_COORDINATOR=<coordinator_ip> \
//!     RMLX_IBV_DEVICES=<device_file> \
//!     cargo run --release -p rmlx-distributed --example comprehensive_bench
//!
//!   # Rank 1:
//!   RMLX_RANK=1 RMLX_WORLD_SIZE=2 RMLX_COORDINATOR=<coordinator_ip> \
//!     RMLX_IBV_DEVICES=<device_file> \
//!     cargo run --release -p rmlx-distributed --example comprehensive_bench

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use rmlx_distributed::{init, Group, InitConfig, ReduceDtype, ReduceOp};

// ── Constants ──

const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 30;

const SIZES: &[usize] = &[4096, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216];

/// EP configs: (n_tokens, hidden_dim)
const EP_CONFIGS: &[(usize, usize)] = &[(16, 1024), (64, 1024), (256, 1024), (512, 1024)];

// ── Stats helpers ──

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    if n % 2 == 0 {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    } else {
        times[n / 2]
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bandwidth_gbps(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    let bits = bytes as f64 * 8.0;
    bits / (ms * 1e-3) / 1e9
}

fn compute_stats(times: &mut [f64], size_bytes: usize) -> serde_json::Value {
    let med = median(times);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = percentile(times, 95.0);
    serde_json::json!({
        "size_bytes": size_bytes,
        "median_ms": (med * 1e6).round() / 1e6,
        "p95_ms": (p95 * 1e6).round() / 1e6,
        "bandwidth_gbps": (bandwidth_gbps(size_bytes, med) * 1000.0).round() / 1000.0
    })
}

fn compute_stats_ep(
    times: &mut [f64],
    n_tokens: usize,
    hidden_dim: usize,
    size_bytes: usize,
) -> serde_json::Value {
    let med = median(times);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = percentile(times, 95.0);
    serde_json::json!({
        "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
        "size_bytes": size_bytes,
        "median_ms": (med * 1e6).round() / 1e6,
        "p95_ms": (p95 * 1e6).round() / 1e6,
        "bandwidth_gbps": (bandwidth_gbps(size_bytes, med) * 1000.0).round() / 1000.0
    })
}

// ── Benchmark functions ──

fn bench_send_recv(group: &Group) -> Vec<serde_json::Value> {
    let rank = group.local_rank();
    let peer = if rank == 0 { 1 } else { 0 };
    let mut results = Vec::new();

    for &size in SIZES {
        eprintln!("  send_recv: size={size}");
        let data = vec![0xABu8; size];

        // Warmup
        for _ in 0..WARMUP_ITERS {
            group.barrier().unwrap();
            if rank == 0 {
                group.send(&data, peer).unwrap();
            } else {
                let _ = group.recv(peer, size).unwrap();
            }
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            if rank == 0 {
                group.send(&data, peer).unwrap();
            } else {
                let _ = group.recv(peer, size).unwrap();
            }
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, size));
    }
    results
}

fn bench_allreduce_f32(group: &Group) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for &size in SIZES {
        // f32: size must be multiple of 4
        let aligned_size = (size + 3) & !3;
        eprintln!("  allreduce_f32: size={aligned_size}");
        let data = vec![0u8; aligned_size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group.allreduce(&data).unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group.allreduce(&data).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, aligned_size));
    }
    results
}

fn bench_allreduce_f16(group: &Group) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for &size in SIZES {
        // f16: size must be multiple of 2
        let aligned_size = (size + 1) & !1;
        eprintln!("  allreduce_f16: size={aligned_size}");
        let data = vec![0u8; aligned_size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group
                .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
                .unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group
                .allreduce_op(&data, ReduceOp::Sum, ReduceDtype::F16)
                .unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, aligned_size));
    }
    results
}

fn bench_allgather(group: &Group) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for &size in SIZES {
        eprintln!("  allgather: size={size}");
        let data = vec![0u8; size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group.allgather(&data).unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group.allgather(&data).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, size));
    }
    results
}

fn bench_broadcast(group: &Group) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for &size in SIZES {
        eprintln!("  broadcast: size={size}");
        let data = vec![0u8; size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group.broadcast(&data, 0).unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group.broadcast(&data, 0).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, size));
    }
    results
}

fn bench_reduce_scatter(group: &Group) -> Vec<serde_json::Value> {
    let world_size = group.ranks().len();
    let mut results = Vec::new();

    for &size in SIZES {
        // Must be divisible by world_size AND by 4 (f32 granularity)
        let lcm = lcm(world_size, 4);
        let aligned_size = size.div_ceil(lcm) * lcm;
        eprintln!("  reduce_scatter: size={aligned_size}");
        let data = vec![0u8; aligned_size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group.reduce_scatter(&data).unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group.reduce_scatter(&data).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, aligned_size));
    }
    results
}

fn bench_all_to_all(group: &Group) -> Vec<serde_json::Value> {
    let world_size = group.ranks().len();
    let mut results = Vec::new();

    for &size in SIZES {
        // Must be divisible by world_size
        let aligned_size = size.div_ceil(world_size) * world_size;
        eprintln!("  all_to_all: size={aligned_size}");
        let data = vec![0u8; aligned_size];

        // Warmup
        group.barrier().unwrap();
        for _ in 0..WARMUP_ITERS {
            let _ = group.all_to_all(&data).unwrap();
        }

        // Timed
        let mut times = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group.all_to_all(&data).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }

        results.push(compute_stats(&mut times, aligned_size));
    }
    results
}

fn bench_ep_exchange(group: &Group) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let rank = group.local_rank();
    let peer = if rank == 0 { 1 } else { 0 };
    let mut dispatch_results = Vec::new();
    let mut combine_results = Vec::new();

    for &(n_tokens, hidden_dim) in EP_CONFIGS {
        let payload_size = n_tokens * hidden_dim * 2; // f16
        eprintln!(
            "  ep_exchange: n_tokens={n_tokens}, hidden_dim={hidden_dim}, payload={payload_size}"
        );
        let send_data = vec![rank as u8; payload_size];

        // --- Dispatch phase ---
        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = group
                .sendrecv(&send_data, peer, payload_size, peer)
                .unwrap();
        }

        // Timed
        let mut times_dispatch = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group
                .sendrecv(&send_data, peer, payload_size, peer)
                .unwrap();
            times_dispatch.push(start.elapsed().as_secs_f64() * 1e3);
        }

        dispatch_results.push(compute_stats_ep(
            &mut times_dispatch,
            n_tokens,
            hidden_dim,
            payload_size,
        ));

        // --- Combine phase ---
        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = group
                .sendrecv(&send_data, peer, payload_size, peer)
                .unwrap();
        }

        // Timed
        let mut times_combine = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            group.barrier().unwrap();
            let start = Instant::now();
            let _ = group
                .sendrecv(&send_data, peer, payload_size, peer)
                .unwrap();
            times_combine.push(start.elapsed().as_secs_f64() * 1e3);
        }

        combine_results.push(compute_stats_ep(
            &mut times_combine,
            n_tokens,
            hidden_dim,
            payload_size,
        ));
    }

    (dispatch_results, combine_results)
}

// ── Utility ──

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

// ── Main ──

fn main() {
    eprintln!("[bench] Starting RMLX comprehensive RDMA benchmark...");

    let config = InitConfig::default();
    let ctx = match init(config) {
        Ok(ctx) => {
            eprintln!(
                "[bench] init OK: rank={}, world_size={}, transport={}",
                ctx.rank,
                ctx.world_size,
                if ctx.group.has_transport() {
                    "rdma"
                } else {
                    "loopback"
                }
            );
            ctx
        }
        Err(e) => {
            eprintln!("[bench] init FAILED: {e}");
            std::process::exit(1);
        }
    };

    let group = &ctx.group;
    let mut results: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    // 1. Send/Recv
    eprintln!("[bench] === send_recv ===");
    results.insert(
        "send_recv".into(),
        serde_json::json!(bench_send_recv(group)),
    );

    // 2. Allreduce f32
    eprintln!("[bench] === allreduce_f32 ===");
    results.insert(
        "allreduce_f32".into(),
        serde_json::json!(bench_allreduce_f32(group)),
    );

    // 3. Allreduce f16
    eprintln!("[bench] === allreduce_f16 ===");
    results.insert(
        "allreduce_f16".into(),
        serde_json::json!(bench_allreduce_f16(group)),
    );

    // 4. Allgather
    eprintln!("[bench] === allgather ===");
    results.insert(
        "allgather".into(),
        serde_json::json!(bench_allgather(group)),
    );

    // 5. Broadcast (RMLX only — no MLX equivalent)
    eprintln!("[bench] === broadcast (RMLX only) ===");
    results.insert(
        "broadcast_rmlx_only".into(),
        serde_json::json!(bench_broadcast(group)),
    );

    // 6. Reduce-scatter (RMLX only — no MLX equivalent)
    eprintln!("[bench] === reduce_scatter (RMLX only) ===");
    results.insert(
        "reduce_scatter_rmlx_only".into(),
        serde_json::json!(bench_reduce_scatter(group)),
    );

    // 7. All-to-all
    eprintln!("[bench] === all_to_all ===");
    results.insert(
        "all_to_all".into(),
        serde_json::json!(bench_all_to_all(group)),
    );

    // 8. EP exchange
    eprintln!("[bench] === ep_exchange ===");
    let (ep_dispatch, ep_combine) = bench_ep_exchange(group);
    results.insert("ep_dispatch".into(), serde_json::json!(ep_dispatch));
    results.insert("ep_combine".into(), serde_json::json!(ep_combine));

    // Build final JSON
    let output = serde_json::json!({
        "framework": "RMLX",
        "timestamp": chrono_timestamp(),
        "node_count": ctx.world_size,
        "rank": ctx.rank,
        "warmup": WARMUP_ITERS,
        "iters": TIMED_ITERS,
        "results": results
    });

    let json_str = serde_json::to_string_pretty(&output).unwrap();

    // Write to file
    let out_path = "/tmp/rmlx_bench_results.json";
    if let Ok(mut f) = std::fs::File::create(out_path) {
        let _ = f.write_all(json_str.as_bytes());
        eprintln!("[bench] Results written to {out_path}");
    }

    // Print to stdout
    println!("{json_str}");

    eprintln!("[bench] Done.");
}

/// Simple ISO-8601 timestamp without external crate dependency.
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate UTC breakdown (no leap seconds)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let months_days: &[i64] = if is_leap(y) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0usize;
    for (i, &md) in months_days.iter().enumerate() {
        if remaining_days < md {
            m = i + 1;
            break;
        }
        remaining_days -= md;
    }
    if m == 0 {
        m = 12;
    }
    let d = remaining_days + 1;

    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
