#!/usr/bin/env python3
"""MLX JACCL Expert Parallelism layer benchmark — Qwen 3.5 MoE A22B config.

Matches RMLX ep_bench.rs workloads exactly:
  1. Single expert FFN (SwiGLU: gate*up -> silu -> down)
  2. EP-2 JACCL (real 2-node moe_dispatch/combine_exchange)

Config (Qwen 3.5 MoE A22B):
  - HIDDEN_DIM = 3584, INTERMEDIATE_DIM = 18944
  - NUM_EXPERTS = 8, TOP_K = 2, dtype = float16

Run:
    mlx.launch --backend jaccl --hostfile ../../rmlx-hosts.json -- \
        python3 benchmarks/mlx_vs_rmlx/bench_mlx_ep_layer.py [--warmup 10] [--iters 30]
"""

import argparse
import math
import time

import numpy as np

import mlx.core as mx


# ── Qwen 3.5 MoE A22B config (matches ep_bench.rs) ──────────────────────────

HIDDEN_DIM = 3584
INTERMEDIATE_DIM = 18944
NUM_EXPERTS = 8
TOP_K = 2

# Token counts per expert — matches RMLX M_VALUES / M_VALUES_WITH_PREFILL
M_DECODE = [1, 4, 8, 16, 32]
M_PREFILL = [128, 512]
M_ALL = M_DECODE + M_PREFILL


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


def compute_stats(times_us):
    """Return dict with mean, p50, p95 in microseconds."""
    a = np.array(times_us)
    return {
        "mean_us": float(np.mean(a)),
        "p50_us": float(np.median(a)),
        "p95_us": float(np.percentile(a, 95)),
        "min_us": float(np.min(a)),
        "max_us": float(np.max(a)),
        "count": len(a),
    }


def fmt_stats(stats):
    """Format stats dict for display."""
    return (
        f"mean={stats['mean_us']:8.1f}us "
        f"p50={stats['p50_us']:8.1f}us "
        f"p95={stats['p95_us']:8.1f}us "
        f"min={stats['min_us']:8.1f}us "
        f"max={stats['max_us']:8.1f}us "
        f"(n={stats['count']})"
    )


# ── Expert class ─────────────────────────────────────────────────────────────


class Expert:
    """Single MoE expert: SwiGLU FFN (gate_proj, up_proj, down_proj)."""

    def __init__(self, hidden, inter, seed=0):
        mx.random.seed(seed)
        self.w_gate = (mx.random.normal((hidden, inter)) * 0.02).astype(mx.float16)
        self.w_up = (mx.random.normal((hidden, inter)) * 0.02).astype(mx.float16)
        self.w_down = (mx.random.normal((inter, hidden)) * 0.02).astype(mx.float16)
        mx.eval(self.w_gate, self.w_up, self.w_down)

    def __call__(self, x):
        gate = x @ self.w_gate
        up = x @ self.w_up
        hidden = mx.sigmoid(gate) * gate * up  # SwiGLU
        return hidden @ self.w_down


# ── Benchmark: Single Expert FFN ─────────────────────────────────────────────


def _expert_forward(x, w_gate, w_up, w_down):
    """Standalone expert FFN for mx.compile (no class overhead)."""
    gate = x @ w_gate
    up = x @ w_up
    hidden = mx.sigmoid(gate) * gate * up  # SwiGLU
    return hidden @ w_down


_compiled_expert = mx.compile(_expert_forward)


def bench_single_expert_ffn(rank, warmup, iters):
    """Single expert FFN benchmark — runs only on rank 0 (local GPU compute)."""
    if rank != 0:
        return

    print("\n=== Single Expert FFN (SwiGLU: gate*up -> silu -> down) ===")
    print(f"  Config: hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, dtype=float16")

    expert = Expert(HIDDEN_DIM, INTERMEDIATE_DIM, seed=42)

    for m in M_ALL:
        x = (mx.random.normal((m, HIDDEN_DIM)) * 0.02).astype(mx.float16)
        mx.eval(x)

        # ── Uncompiled (reference) ──
        for _ in range(warmup):
            out = expert(x)
            mx.eval(out)
            mx.synchronize()

        times_uncompiled = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = expert(x)
            mx.eval(out)
            mx.synchronize()
            times_uncompiled.append((time.perf_counter() - t0) * 1e6)

        stats_uc = compute_stats(times_uncompiled)
        print(f"  expert_ffn M={m:<4d} (uncompiled)  {fmt_stats(stats_uc)}", flush=True)

        # ── Compiled (primary — fair comparison with RMLX single-CB) ──
        for _ in range(warmup):
            out = _compiled_expert(x, expert.w_gate, expert.w_up, expert.w_down)
            mx.eval(out)
            mx.synchronize()

        times_compiled = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = _compiled_expert(x, expert.w_gate, expert.w_up, expert.w_down)
            mx.eval(out)
            mx.synchronize()
            times_compiled.append((time.perf_counter() - t0) * 1e6)

        stats_c = compute_stats(times_compiled)
        print(f"  expert_ffn M={m:<4d} (mx.compile)  {fmt_stats(stats_c)}", flush=True)


# ── Benchmark: EP-2 JACCL (real 2-node) ─────────────────────────────────────


def bench_ep2_jaccl(group, rank, warmup, iters):
    """EP-2 benchmark using moe_dispatch_exchange / moe_combine_exchange."""
    ws = group.size()
    if ws < 2:
        if rank == 0:
            print("\n=== EP-2 JACCL: SKIPPED (need 2+ nodes) ===")
        return

    print(f"\n=== EP-2 JACCL (real 2-node, rank={rank}) ===")
    print(f"  Config: {NUM_EXPERTS} experts, top_k={TOP_K}, "
          f"{NUM_EXPERTS // ws} local experts/rank")
    print(f"  hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, dtype=float16")

    # Build local experts (4 per rank for EP=2)
    local_experts_count = NUM_EXPERTS // ws
    local_experts = []
    for i in range(local_experts_count):
        expert_id = rank * local_experts_count + i
        local_experts.append(Expert(HIDDEN_DIM, INTERMEDIATE_DIM, seed=2000 + expert_id * 10))

    for m in M_ALL:
        # total_tokens = m * NUM_EXPERTS / TOP_K (matches RMLX seq_len calculation)
        total_tokens = m * NUM_EXPERTS // TOP_K

        # Create input tokens and routing info
        mx.random.seed(42 + m)
        tokens = (mx.random.normal((total_tokens, HIDDEN_DIM)) * 0.02).astype(mx.float16)
        expert_indices = mx.random.randint(0, NUM_EXPERTS, shape=(total_tokens, TOP_K)).astype(
            mx.int32
        )
        weights = mx.softmax(mx.random.normal((total_tokens, TOP_K)), axis=-1).astype(mx.float32)
        mx.eval(tokens, expert_indices, weights)

        # Capacity: ceil(total_tokens * top_k * 1.25 / num_experts)
        capacity = max(1, math.ceil(total_tokens * TOP_K * 1.25 / NUM_EXPERTS))

        # Sync capacity across ranks
        cap_arr = mx.array(capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_sum(cap_arr, group=group, stream=mx.cpu)
        mx.eval(cap_arr)

        barrier(group)

        # ── Warmup ──
        for _ in range(warmup):
            try:
                dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                    tokens,
                    expert_indices,
                    num_experts=NUM_EXPERTS,
                    capacity=capacity,
                    group=group,
                    backend="cpu",
                )
                route_idx = mx.stop_gradient(route_idx)
                mx.eval(dispatched, route_idx)
                mx.synchronize()

                # Local expert compute (uncompiled — compile not applied here
                # because the loop over local_experts + dispatch/combine exchange
                # already forces sync; the overhead is amortised across experts)
                # dispatched shape: [local_experts_count, capacity, HIDDEN_DIM]
                expert_outputs = mx.zeros_like(dispatched)
                for ei in range(local_experts_count):
                    expert_input = dispatched[ei]  # [capacity, HIDDEN_DIM]
                    expert_outputs[ei] = local_experts[ei](expert_input)
                mx.eval(expert_outputs)
                mx.synchronize()

                combined = mx.distributed.moe_combine_exchange(
                    expert_outputs,
                    route_idx,
                    weights,
                    tokens,
                    num_experts=NUM_EXPERTS,
                    capacity=capacity,
                    group=group,
                    backend="cpu",
                )
                mx.eval(combined)
                mx.synchronize()
            except Exception as ex:
                if rank == 0:
                    print(f"  WARMUP ERROR M={m}: {ex}", flush=True)
                break

        # ── Timed runs ──
        dispatch_times = []
        compute_times = []
        combine_times = []
        total_times = []
        success = True

        for _ in range(iters):
            barrier(group)
            try:
                t_total_start = time.perf_counter()

                # 1. Dispatch
                t0 = time.perf_counter()
                dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                    tokens,
                    expert_indices,
                    num_experts=NUM_EXPERTS,
                    capacity=capacity,
                    group=group,
                    backend="cpu",
                )
                route_idx = mx.stop_gradient(route_idx)
                mx.eval(dispatched, route_idx)
                mx.synchronize()
                dispatch_times.append((time.perf_counter() - t0) * 1e6)

                # 2. Local expert compute
                t1 = time.perf_counter()
                expert_outputs = mx.zeros_like(dispatched)
                for ei in range(local_experts_count):
                    expert_input = dispatched[ei]
                    expert_outputs[ei] = local_experts[ei](expert_input)
                mx.eval(expert_outputs)
                mx.synchronize()
                compute_times.append((time.perf_counter() - t1) * 1e6)

                # 3. Combine
                t2 = time.perf_counter()
                combined = mx.distributed.moe_combine_exchange(
                    expert_outputs,
                    route_idx,
                    weights,
                    tokens,
                    num_experts=NUM_EXPERTS,
                    capacity=capacity,
                    group=group,
                    backend="cpu",
                )
                mx.eval(combined)
                mx.synchronize()
                combine_times.append((time.perf_counter() - t2) * 1e6)

                total_times.append((time.perf_counter() - t_total_start) * 1e6)

            except Exception as ex:
                if rank == 0:
                    print(f"  ERROR M={m}: {ex}", flush=True)
                success = False
                break

        if not success or len(total_times) == 0:
            if rank == 0:
                print(f"  ep2 M={m:<4d}  FAILED", flush=True)
            continue

        d_stats = compute_stats(dispatch_times)
        c_stats = compute_stats(compute_times)
        cb_stats = compute_stats(combine_times)
        t_stats = compute_stats(total_times)

        if rank == 0:
            label = "Decode" if m <= 32 else "Prefill"
            print(
                f"  {label} (total_tokens={total_tokens}, ~{m} tok/expert):",
                flush=True,
            )
            print(
                f"    dispatch:  {fmt_stats(d_stats)}",
                flush=True,
            )
            print(
                f"    compute:   {fmt_stats(c_stats)}",
                flush=True,
            )
            print(
                f"    combine:   {fmt_stats(cb_stats)}",
                flush=True,
            )
            print(
                f"    total:     {fmt_stats(t_stats)}",
                flush=True,
            )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MLX JACCL EP layer benchmark (Qwen 3.5 MoE A22B)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iters", type=int, default=30, help="Timed iterations (default: 30)"
    )
    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────────
    group = mx.distributed.init(strict=True)
    rank = group.rank()
    ws = group.size()

    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw)

    log("MLX JACCL EP Layer Benchmark")
    log(f"  Config: Qwen 3.5 MoE A22B")
    log(f"  experts={NUM_EXPERTS}, top_k={TOP_K}, "
        f"hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}")
    log(f"  dtype=float16, rank={rank}, world_size={ws}")
    log(f"  warmup={args.warmup}, iters={args.iters}")
    log(f"  M decode: {M_DECODE}")
    log(f"  M prefill: {M_PREFILL}")

    # ── RDMA warmup ──────────────────────────────────────────────────────
    rdma_warmup(group, rank)
    barrier(group)

    # ── 1. Single Expert FFN (rank 0 only) ───────────────────────────────
    bench_single_expert_ffn(rank, args.warmup, args.iters)
    barrier(group)

    # ── 2. EP-2 JACCL (real 2-node) ─────────────────────────────────────
    bench_ep2_jaccl(group, rank, args.warmup, args.iters)

    # ── Summary ──────────────────────────────────────────────────────────
    barrier(group)
    if rank == 0:
        print("\n" + "=" * 72)
        print("DONE")
        print("=" * 72)
        print(f"  Single Expert FFN: per-expert compute (rank 0 only)")
        print(f"  EP-2 JACCL: real 2-node moe_dispatch/combine_exchange")
        print(f"  Config: Qwen 3.5 MoE A22B — {NUM_EXPERTS} experts, top_k={TOP_K}")
        print(f"    hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, float16")
        print()


if __name__ == "__main__":
    main()
