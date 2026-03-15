#!/usr/bin/env python3
"""Comprehensive MLX JACCL distributed benchmark for fair comparison with RMLX.

Benchmarks:
  1. send/recv raw throughput (2-node)
  2. allreduce (all_sum) for f16/f32 (2-node)
  3. allgather (2-node)
  4. all-to-all (2-node)
  5. EP transport — pure all_to_all with EP-sized payloads (fair comparison with RMLX)
  6. EP pipeline — full moe_dispatch/combine_exchange (MLX only, for reference)

Run:
    mlx.launch --backend jaccl --hostfile hosts.json -- \
        python3 benchmarks/mlx_vs_rmlx/bench_mlx.py [--warmup 10] [--iters 30]

Output: /tmp/mlx_bench_results.json
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

import mlx.core as mx


# ── Helpers ──────────────────────────────────────────────────────────────────


def barrier(group):
    """Synchronize all ranks."""
    s = mx.distributed.all_sum(mx.array(0.0), group=group, stream=mx.cpu)
    mx.eval(s)
    mx.synchronize()


def rdma_warmup(group, rank, n=10):
    """Warm up JACCL RDMA path to stabilize timings."""
    if rank == 0:
        print("  RDMA warmup ...", flush=True)
    buf = mx.ones(1024)
    for _ in range(n):
        buf = mx.distributed.all_sum(buf, group=group, stream=mx.cpu)
        mx.eval(buf)
        mx.synchronize()
    if rank == 0:
        print("  RDMA warmup done.", flush=True)


def compute_stats(times_ms):
    """Return dict with median, p95, mean, min, max in ms."""
    a = np.array(times_ms)
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(np.mean(a)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
    }


def bandwidth_gbps(size_bytes, median_ms):
    """Compute bandwidth in Gbps from size in bytes and time in ms."""
    if median_ms <= 0:
        return 0.0
    return (size_bytes * 8) / (median_ms * 1e-3) / 1e9


def dtype_size(dtype):
    if dtype == mx.float16 or dtype == mx.bfloat16:
        return 2
    elif dtype == mx.float32:
        return 4
    return 4


# ── Category 1: Send/Recv Raw Throughput ─────────────────────────────────────


def bench_send_recv(group, rank, warmup, iters, sizes):
    """Measure point-to-point send/recv latency and bandwidth."""
    results = []
    ws = group.size()
    if ws < 2:
        return results

    for size in sizes:
        nelems = size // 4  # float32 = 4 bytes
        data = mx.ones(nelems, dtype=mx.float32)
        mx.eval(data)
        barrier(group)

        # Warmup
        for _ in range(warmup):
            if rank == 0:
                out = mx.distributed.send(data, dst=1, group=group, stream=mx.cpu)
                mx.eval(out)
            else:
                out = mx.distributed.recv(
                    data.shape, data.dtype, src=0, group=group, stream=mx.cpu
                )
                mx.eval(out)
            mx.synchronize()
            barrier(group)

        # Timed
        times = []
        for _ in range(iters):
            barrier(group)
            t0 = time.perf_counter()
            if rank == 0:
                out = mx.distributed.send(data, dst=1, group=group, stream=mx.cpu)
                mx.eval(out)
            else:
                out = mx.distributed.recv(
                    data.shape, data.dtype, src=0, group=group, stream=mx.cpu
                )
                mx.eval(out)
            mx.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        stats = compute_stats(times)
        stats["size_bytes"] = size
        stats["bandwidth_gbps"] = bandwidth_gbps(size, stats["median_ms"])
        results.append(stats)

        if rank == 0:
            print(
                f"    send_recv {size:>10d}B: "
                f"med={stats['median_ms']:.3f}ms "
                f"p95={stats['p95_ms']:.3f}ms "
                f"bw={stats['bandwidth_gbps']:.2f}Gbps",
                flush=True,
            )

    return results


# ── Category 2: Collective Operations ────────────────────────────────────────


def bench_allreduce(group, rank, warmup, iters, sizes):
    """Allreduce (all_sum) latency for f16 and f32."""
    results_by_dtype = {}

    for dtype, dtype_name in [(mx.float16, "f16"), (mx.float32, "f32")]:
        results = []
        ds = dtype_size(dtype)

        for size in sizes:
            nelems = size // ds
            data = mx.random.normal((nelems,)).astype(dtype)
            mx.eval(data)
            barrier(group)

            # Warmup
            for _ in range(warmup):
                out = mx.distributed.all_sum(data, group=group, stream=mx.cpu)
                mx.eval(out)
                mx.synchronize()

            # Timed
            times = []
            for _ in range(iters):
                barrier(group)
                t0 = time.perf_counter()
                out = mx.distributed.all_sum(data, group=group, stream=mx.cpu)
                mx.eval(out)
                mx.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)

            stats = compute_stats(times)
            stats["size_bytes"] = size
            stats["dtype"] = dtype_name
            stats["bandwidth_gbps"] = bandwidth_gbps(size, stats["median_ms"])
            results.append(stats)

            if rank == 0:
                print(
                    f"    allreduce_{dtype_name} {size:>10d}B: "
                    f"med={stats['median_ms']:.3f}ms "
                    f"p95={stats['p95_ms']:.3f}ms "
                    f"bw={stats['bandwidth_gbps']:.2f}Gbps",
                    flush=True,
                )

        results_by_dtype[dtype_name] = results

    return results_by_dtype


def bench_allgather(group, rank, warmup, iters, sizes):
    """all_gather latency for various sizes."""
    results = []
    ws = group.size()

    for size in sizes:
        # Each rank contributes size/ws bytes, total gathered = size bytes
        nelems_per_rank = size // (4 * ws)  # float32
        if nelems_per_rank < 1:
            nelems_per_rank = 1
        data = mx.random.normal((nelems_per_rank,)).astype(mx.float32)
        mx.eval(data)
        barrier(group)

        # Warmup
        for _ in range(warmup):
            out = mx.distributed.all_gather(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()

        # Timed
        times = []
        for _ in range(iters):
            barrier(group)
            t0 = time.perf_counter()
            out = mx.distributed.all_gather(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        total_bytes = nelems_per_rank * 4 * ws
        stats = compute_stats(times)
        stats["size_bytes"] = total_bytes
        stats["bandwidth_gbps"] = bandwidth_gbps(total_bytes, stats["median_ms"])
        results.append(stats)

        if rank == 0:
            print(
                f"    allgather {total_bytes:>10d}B: "
                f"med={stats['median_ms']:.3f}ms "
                f"p95={stats['p95_ms']:.3f}ms "
                f"bw={stats['bandwidth_gbps']:.2f}Gbps",
                flush=True,
            )

    return results


def bench_all_to_all(group, rank, warmup, iters, sizes):
    """all_to_all exchange latency for various sizes."""
    results = []
    ws = group.size()

    for size in sizes:
        # x.shape[0] must be divisible by ws
        nelems = size // 4  # float32
        nelems = max(nelems, ws)
        nelems = (nelems // ws) * ws  # round down to divisible
        data = mx.random.normal((nelems,)).astype(mx.float32)
        mx.eval(data)
        barrier(group)

        # Warmup
        for _ in range(warmup):
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()

        # Timed
        times = []
        for _ in range(iters):
            barrier(group)
            t0 = time.perf_counter()
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        actual_bytes = nelems * 4
        stats = compute_stats(times)
        stats["size_bytes"] = actual_bytes
        stats["bandwidth_gbps"] = bandwidth_gbps(actual_bytes, stats["median_ms"])
        results.append(stats)

        if rank == 0:
            print(
                f"    all_to_all {actual_bytes:>10d}B: "
                f"med={stats['median_ms']:.3f}ms "
                f"p95={stats['p95_ms']:.3f}ms "
                f"bw={stats['bandwidth_gbps']:.2f}Gbps",
                flush=True,
            )

    return results


# ── Category 3a: EP Transport (pure all_to_all — matches RMLX sendrecv) ──────


# Aligned with RMLX EP_CONFIGS: (n_tokens, hidden_dim), payload = n_tokens * hidden_dim * 2 (f16)
EP_TRANSPORT_CONFIGS = [(16, 1024), (64, 1024), (256, 1024), (512, 1024)]


def bench_ep_transport(group, rank, warmup, iters):
    """Pure all_to_all transport with EP-sized payloads (fair comparison with RMLX sendrecv)."""
    ws = group.size()
    if ws < 2:
        return [], []

    dispatch_results = []
    combine_results = []

    for n_tokens, hidden_dim in EP_TRANSPORT_CONFIGS:
        payload_bytes = n_tokens * hidden_dim * 2  # f16
        nelems = payload_bytes // 4  # float32 elements
        nelems = max(nelems, ws)
        nelems = (nelems // ws) * ws  # divisible by ws

        data = mx.random.normal((nelems,)).astype(mx.float32)
        mx.eval(data)
        actual_bytes = nelems * 4
        barrier(group)

        # --- Dispatch phase (forward all_to_all) ---
        for _ in range(warmup):
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()

        times_dispatch = []
        for _ in range(iters):
            barrier(group)
            t0 = time.perf_counter()
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()
            times_dispatch.append((time.perf_counter() - t0) * 1e3)

        d_stats = compute_stats(times_dispatch)
        d_stats["n_tokens"] = n_tokens
        d_stats["hidden_dim"] = hidden_dim
        d_stats["size_bytes"] = actual_bytes
        d_stats["bandwidth_gbps"] = bandwidth_gbps(actual_bytes, d_stats["median_ms"])
        dispatch_results.append(d_stats)

        # --- Combine phase (reverse all_to_all) ---
        for _ in range(warmup):
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()

        times_combine = []
        for _ in range(iters):
            barrier(group)
            t0 = time.perf_counter()
            out = mx.distributed.all_to_all(data, group=group, stream=mx.cpu)
            mx.eval(out)
            mx.synchronize()
            times_combine.append((time.perf_counter() - t0) * 1e3)

        c_stats = compute_stats(times_combine)
        c_stats["n_tokens"] = n_tokens
        c_stats["hidden_dim"] = hidden_dim
        c_stats["size_bytes"] = actual_bytes
        c_stats["bandwidth_gbps"] = bandwidth_gbps(actual_bytes, c_stats["median_ms"])
        combine_results.append(c_stats)

        if rank == 0:
            print(
                f"    ep_transport n_tokens={n_tokens} hidden={hidden_dim}: "
                f"dispatch={d_stats['median_ms']:.3f}ms "
                f"combine={c_stats['median_ms']:.3f}ms "
                f"bw={d_stats['bandwidth_gbps']:.2f}Gbps",
                flush=True,
            )

    return dispatch_results, combine_results


# ── Category 3b: EP Pipeline (full MoE dispatch/combine — MLX only reference) ─


EP_PIPELINE_CONFIGS = [
    # (N, D, E_total, top_k) — representative MoE configurations
    (16, 1024, 16, 4),
    (64, 1024, 16, 4),
    (256, 1024, 16, 4),
    (512, 1024, 16, 4),
]


def bench_ep_pipeline(group, rank, warmup, iters, configs=None):
    """Full MoE dispatch + combine exchange (MLX only, for reference)."""
    if configs is None:
        configs = EP_PIPELINE_CONFIGS

    ws = group.size()
    if ws < 2:
        return [], []

    dispatch_results = []
    combine_results = []

    for N, D, E, top_k in configs:
        if E % ws != 0:
            if rank == 0:
                print(
                    f"    ep_pipeline N={N} D={D} E={E} top_k={top_k}: SKIPPED (E % ws != 0)",
                    flush=True,
                )
            continue

        capacity = max(1, math.ceil(N * top_k * 1.25 / E))

        tokens = mx.random.normal((N, D)).astype(mx.float16)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights = mx.softmax(
            mx.random.normal((N, top_k)), axis=-1
        ).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights)

        # Synchronize capacity
        cap_arr = mx.array(capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_sum(cap_arr, group=group, stream=mx.cpu)
        mx.eval(cap_arr)

        barrier(group)

        # ── Warmup dispatch+combine ──
        for _ in range(warmup):
            try:
                dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                    tokens, expert_indices,
                    num_experts=E, capacity=capacity,
                    group=group, backend="cpu",
                )
                route_idx = mx.stop_gradient(route_idx)
                mx.eval(dispatched, route_idx)
                mx.synchronize()

                combined = mx.distributed.moe_combine_exchange(
                    dispatched, route_idx, weights, tokens,
                    num_experts=E, capacity=capacity,
                    group=group, backend="cpu",
                )
                mx.eval(combined)
                mx.synchronize()
            except Exception:
                break

        # ── Timed runs ──
        dispatch_times = []
        combine_times = []
        total_times = []
        success = True

        for _ in range(iters):
            barrier(group)
            try:
                # Dispatch
                t0 = time.perf_counter()
                dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                    tokens, expert_indices,
                    num_experts=E, capacity=capacity,
                    group=group, backend="cpu",
                )
                route_idx = mx.stop_gradient(route_idx)
                mx.eval(dispatched, route_idx)
                mx.synchronize()
                dt_dispatch = (time.perf_counter() - t0) * 1e3

                # Combine
                t1 = time.perf_counter()
                combined = mx.distributed.moe_combine_exchange(
                    dispatched, route_idx, weights, tokens,
                    num_experts=E, capacity=capacity,
                    group=group, backend="cpu",
                )
                mx.eval(combined)
                mx.synchronize()
                dt_combine = (time.perf_counter() - t1) * 1e3

                dispatch_times.append(dt_dispatch)
                combine_times.append(dt_combine)
                total_times.append(dt_dispatch + dt_combine)
            except Exception as ex:
                if rank == 0:
                    print(
                        f"    ep_pipeline N={N} D={D} E={E} top_k={top_k}: ERROR {ex}",
                        flush=True,
                    )
                success = False
                break

        if not success or len(dispatch_times) == 0:
            continue

        config_label = f"N={N}_D={D}_E={E}_k={top_k}"

        d_stats = compute_stats(dispatch_times)
        d_stats["config"] = config_label
        d_stats["N"] = N
        d_stats["D"] = D
        d_stats["E"] = E
        d_stats["top_k"] = top_k
        d_stats["capacity"] = capacity
        d_stats["data_bytes"] = N * D * 2
        dispatch_results.append(d_stats)

        c_stats = compute_stats(combine_times)
        c_stats["config"] = config_label
        c_stats["N"] = N
        c_stats["D"] = D
        c_stats["E"] = E
        c_stats["top_k"] = top_k
        c_stats["capacity"] = capacity
        c_stats["data_bytes"] = N * D * 2
        combine_results.append(c_stats)

        t_stats = compute_stats(total_times)

        if rank == 0:
            print(
                f"    ep_pipeline {config_label}: "
                f"dispatch={d_stats['median_ms']:.3f}ms "
                f"combine={c_stats['median_ms']:.3f}ms "
                f"total={t_stats['median_ms']:.3f}ms",
                flush=True,
            )

    return dispatch_results, combine_results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MLX JACCL distributed benchmark for comparison with RMLX"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iters", type=int, default=30, help="Timed iterations (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/mlx_bench_results.json",
        help="Output JSON path (default: /tmp/mlx_bench_results.json)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["send_recv", "allreduce", "allgather", "all_to_all", "ep_transport", "ep_pipeline"],
        help="Categories to benchmark (default: all)",
    )
    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────────
    group = mx.distributed.init(strict=True)
    rank = group.rank()
    ws = group.size()

    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw)

    log(f"MLX JACCL Benchmark — rank={rank} world_size={ws}")
    log(f"  warmup={args.warmup}  iters={args.iters}")
    log(f"  categories={args.categories}")
    log()

    # ── RDMA warmup ──────────────────────────────────────────────────────
    rdma_warmup(group, rank)
    barrier(group)

    # ── Data sizes for raw ops ───────────────────────────────────────────
    sizes = [4096, 65536, 262144, 1048576, 4194304, 16777216]

    all_results = {}

    # ── Category 1: Send/Recv ────────────────────────────────────────────
    if "send_recv" in args.categories and ws >= 2:
        log("\n[1/5] Send/Recv raw throughput", flush=True)
        all_results["send_recv"] = bench_send_recv(
            group, rank, args.warmup, args.iters, sizes
        )

    # ── Category 2: Allreduce ────────────────────────────────────────────
    if "allreduce" in args.categories:
        log("\n[2/5] Allreduce (all_sum)", flush=True)
        ar = bench_allreduce(group, rank, args.warmup, args.iters, sizes)
        all_results["allreduce_f16"] = ar.get("f16", [])
        all_results["allreduce_f32"] = ar.get("f32", [])

    # ── Category 3: Allgather ────────────────────────────────────────────
    if "allgather" in args.categories:
        log("\n[3/5] Allgather", flush=True)
        all_results["allgather"] = bench_allgather(
            group, rank, args.warmup, args.iters, sizes
        )

    # ── Category 4: All-to-All ───────────────────────────────────────────
    if "all_to_all" in args.categories:
        log("\n[4/5] All-to-All", flush=True)
        all_results["all_to_all"] = bench_all_to_all(
            group, rank, args.warmup, args.iters, sizes
        )

    # ── Category 5: EP Transport (pure all_to_all, fair comparison) ─────
    if "ep_transport" in args.categories and ws >= 2:
        log("\n[5/6] EP Transport (pure all_to_all)", flush=True)
        try:
            ep_t_dispatch, ep_t_combine = bench_ep_transport(
                group, rank, args.warmup, args.iters,
            )
            all_results["ep_dispatch"] = ep_t_dispatch
            all_results["ep_combine"] = ep_t_combine
        except Exception as ex:
            log(f"  EP transport benchmark failed: {ex}", flush=True)
            all_results["ep_dispatch"] = []
            all_results["ep_combine"] = []

    # ── Category 6: EP Pipeline (full MoE, MLX only reference) ────────
    if "ep_pipeline" in args.categories and ws >= 2:
        log("\n[6/6] EP Pipeline (full MoE dispatch/combine — MLX only)", flush=True)
        try:
            ep_p_dispatch, ep_p_combine = bench_ep_pipeline(
                group, rank, args.warmup, args.iters,
            )
            all_results["ep_pipeline_dispatch_mlx_only"] = ep_p_dispatch
            all_results["ep_pipeline_combine_mlx_only"] = ep_p_combine
        except Exception as ex:
            log(f"  EP pipeline benchmark failed: {ex}", flush=True)
            all_results["ep_pipeline_dispatch_mlx_only"] = []
            all_results["ep_pipeline_combine_mlx_only"] = []

    # ── Write results (rank 0 only) ──────────────────────────────────────
    if rank == 0:
        output = {
            "framework": "MLX",
            "backend": "JACCL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": ws,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": all_results,
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults written to {args.output}")

        # ── Summary table ────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)

        for category, entries in all_results.items():
            if not entries:
                continue
            print(f"\n  {category}:")
            print(f"    {'Size/Config':<20s} {'Median(ms)':>10s} {'P95(ms)':>10s} {'BW(Gbps)':>10s}")
            print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
            for e in entries:
                label = e.get("config", f"{e.get('size_bytes', '?')}B")
                if "dtype" in e:
                    label = f"{label} ({e['dtype']})"
                med = e.get("median_ms", 0)
                p95 = e.get("p95_ms", 0)
                bw = e.get("bandwidth_gbps", 0)
                bw_str = f"{bw:.2f}" if bw > 0 else "N/A"
                print(f"    {label:<20s} {med:>10.3f} {p95:>10.3f} {bw_str:>10s}")

        print()

    barrier(group)


if __name__ == "__main__":
    main()
