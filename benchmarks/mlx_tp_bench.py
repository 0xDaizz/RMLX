#!/usr/bin/env python3
"""MLX Tensor Parallel comparison benchmark (production-optimized).

Uses mx.fast.* fused kernels (rms_norm, rope, scaled_dot_product_attention)
and mx.compile() to match real MLX inference performance. Includes KV cache
to mirror RMLX's production decode path.

Measures single-layer transformer forward pass latency at TP=1 and TP=2
(simulated via half-sized weights), matching the RMLX distributed_bench
configuration (Qwen 3.5 MoE A22B: hidden=3584, intermediate=18944, 28 heads, 4 KV heads).

For TP=2 simulation: weights are halved on the output/input dimension
(ColumnParallel/RowParallel), identical to RMLX's approach.

Usage:
    python mlx_tp_bench.py [--warmup N] [--iters N]
"""

import argparse
import time
import statistics

import mlx.core as mx

# ---------------------------------------------------------------------------
# Qwen 3.5 MoE A22B config (matches RMLX distributed_bench)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 3584
NUM_HEADS = 28
HEAD_DIM = 128
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 18944
SEQ_LEN = 1  # decode token
RMS_NORM_EPS = 1e-6
KV_CACHE_LEN = 128  # pre-filled cache tokens

WARMUP = 5
ITERS = 50


# ---------------------------------------------------------------------------
# Single layer forward (fused kernels, KV cache)
# ---------------------------------------------------------------------------
def layer_forward(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2,
                  num_heads, num_kv_heads, head_dim, hidden_size,
                  k_cache, v_cache, offset):
    """Single transformer decoder layer forward pass (decode mode, with KV cache).

    Args:
        x:          [1, hidden_size]
        w_q:        [hidden_size, num_heads * head_dim]
        w_k:        [hidden_size, num_kv_heads * head_dim]
        w_v:        [hidden_size, num_kv_heads * head_dim]
        w_o:        [num_heads * head_dim, hidden_size]
        w_gate:     [hidden_size, intermediate]
        w_up:       [hidden_size, intermediate]
        w_down:     [intermediate, hidden_size]
        rms_w1:     [hidden_size]
        rms_w2:     [hidden_size]
        k_cache:    [1, num_kv_heads, cache_len, head_dim]
        v_cache:    [1, num_kv_heads, cache_len, head_dim]
        offset:     int, position offset for RoPE
    """
    seq_len = x.shape[0]  # tokens in this forward call (1 for decode, N for prefill)

    # Pre-attention norm (fused)
    h = mx.fast.rms_norm(x, rms_w1, RMS_NORM_EPS)

    # QKV projections
    q = h @ w_q
    k = h @ w_k
    v = h @ w_v

    # Reshape: [batch=1, heads, seq_len, head_dim]
    q = q.reshape(1, num_heads, seq_len, head_dim)
    k = k.reshape(1, num_kv_heads, seq_len, head_dim)
    v = v.reshape(1, num_kv_heads, seq_len, head_dim)

    # RoPE (fused)
    q = mx.fast.rope(q, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)
    k = mx.fast.rope(k, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)

    # KV cache update
    k_cache_new = mx.concatenate([k_cache, k], axis=2)
    v_cache_new = mx.concatenate([v_cache, v], axis=2)

    # SDPA (fused) - handles GQA internally
    scale = head_dim ** -0.5
    attn_out = mx.fast.scaled_dot_product_attention(q, k_cache_new, v_cache_new, scale=scale)
    attn_out = attn_out.reshape(seq_len, num_heads * head_dim)

    # O projection + residual
    x = x + attn_out @ w_o

    # Pre-FFN norm (fused)
    h2 = mx.fast.rms_norm(x, rms_w2, RMS_NORM_EPS)

    # SwiGLU FFN
    gate = h2 @ w_gate
    up = h2 @ w_up
    hidden = mx.sigmoid(gate) * gate * up
    x = x + hidden @ w_down

    return x, k_cache_new, v_cache_new


# ---------------------------------------------------------------------------
# Build weights for a given TP configuration
# ---------------------------------------------------------------------------
def build_weights(hidden_size, num_heads, num_kv_heads, head_dim, intermediate, tp_size=1):
    """Build random weights for one transformer layer.

    For tp_size > 1, simulates column/row parallel by halving dims:
    - Q/K/V/gate/up: column parallel (output dim / tp_size)
    - O/down: row parallel (input dim / tp_size)
    """
    nh = num_heads // tp_size
    nkv = num_kv_heads // tp_size
    inter = intermediate // tp_size

    mx.random.seed(42)

    w_q = (mx.random.normal((hidden_size, nh * head_dim)) * 0.02).astype(mx.float16)
    w_k = (mx.random.normal((hidden_size, nkv * head_dim)) * 0.02).astype(mx.float16)
    w_v = (mx.random.normal((hidden_size, nkv * head_dim)) * 0.02).astype(mx.float16)
    w_o = (mx.random.normal((nh * head_dim, hidden_size)) * 0.02).astype(mx.float16)
    w_gate = (mx.random.normal((hidden_size, inter)) * 0.02).astype(mx.float16)
    w_up = (mx.random.normal((hidden_size, inter)) * 0.02).astype(mx.float16)
    w_down = (mx.random.normal((inter, hidden_size)) * 0.02).astype(mx.float16)
    rms_w1 = mx.ones((hidden_size,), dtype=mx.float16)
    rms_w2 = mx.ones((hidden_size,), dtype=mx.float16)

    # Pre-fill KV cache
    k_cache = mx.zeros((1, nkv, KV_CACHE_LEN, head_dim), dtype=mx.float16)
    v_cache = mx.zeros((1, nkv, KV_CACHE_LEN, head_dim), dtype=mx.float16)

    # Evaluate to materialize
    mx.eval(w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2, k_cache, v_cache)

    return {
        "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
        "w_gate": w_gate, "w_up": w_up, "w_down": w_down,
        "rms_w1": rms_w1, "rms_w2": rms_w2,
        "k_cache": k_cache, "v_cache": v_cache,
        "num_heads": nh, "num_kv_heads": nkv,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def bench_layer(label, weights, hidden_size, head_dim, warmup, iters, use_compile=False):
    """Benchmark a single layer forward pass."""
    x = (mx.random.normal((SEQ_LEN, hidden_size)) * 0.02).astype(mx.float16)
    mx.eval(x)

    nh = weights["num_heads"]
    nkv = weights["num_kv_heads"]
    offset = KV_CACHE_LEN  # position after pre-filled cache

    forward_fn = mx.compile(layer_forward) if use_compile else layer_forward

    def run_once():
        return forward_fn(
            x, weights["w_q"], weights["w_k"], weights["w_v"], weights["w_o"],
            weights["w_gate"], weights["w_up"], weights["w_down"],
            weights["rms_w1"], weights["rms_w2"],
            nh, nkv, head_dim, hidden_size,
            weights["k_cache"], weights["v_cache"], offset,
        )

    # Warmup
    for _ in range(warmup):
        out, _, _ = run_once()
        mx.eval(out)

    # Benchmark
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _, _ = run_once()
        mx.eval(out)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # us

    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    p50 = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p95_idx = int(0.95 * (len(sorted_lat) - 1))
    p95 = sorted_lat[p95_idx]

    print(f"  {label:50s} mean={mean:8.1f}us std={std:7.1f}us p50={p50:8.1f}us p95={p95:8.1f}us (n={iters})")
    return mean


def main():
    parser = argparse.ArgumentParser(description="MLX TP comparison benchmark (production-optimized)")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    print(f"MLX TP Benchmark (production-optimized)")
    print(f"  Config: Qwen 3.5 MoE A22B, hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_SIZE}")
    print(f"  seq_len={SEQ_LEN} (decode), dtype=float16, kv_cache={KV_CACHE_LEN} tokens")
    print(f"  Fused: mx.fast.rms_norm, mx.fast.rope, mx.fast.scaled_dot_product_attention")
    print(f"  warmup={args.warmup}, iters={args.iters}")

    # ── TP=1 (full weights, no parallelism) ──
    print(f"\n=== TP=1: Full weights (baseline) ===")
    w_tp1 = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE, tp_size=1)

    tp1_mean = bench_layer(
        "TP=1 single-layer (uncompiled)",
        w_tp1, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters, use_compile=False,
    )
    tp1_compiled = bench_layer(
        "TP=1 single-layer (mx.compile)",
        w_tp1, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters, use_compile=True,
    )

    # ── TP=2 (half-sized weights, simulating one rank) ──
    print(f"\n=== TP=2: Half-sized weights (simulating rank 0 of 2) ===")
    print(f"  Q/K/V/gate/up: column-parallel (output halved)")
    print(f"  O/down: row-parallel (input halved)")
    w_tp2 = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE, tp_size=2)

    tp2_mean = bench_layer(
        "TP=2 single-layer (uncompiled)",
        w_tp2, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters, use_compile=False,
    )
    tp2_compiled = bench_layer(
        "TP=2 single-layer (mx.compile)",
        w_tp2, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters, use_compile=True,
    )

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"  {'TP=1 baseline (uncompiled)':<44s} {tp1_mean:>8.1f} us")
    print(f"  {'TP=1 baseline (mx.compile)':<44s} {tp1_compiled:>8.1f} us")
    print(f"  {'TP=2 half-weight (uncompiled)':<44s} {tp2_mean:>8.1f} us")
    print(f"  {'TP=2 half-weight (mx.compile)':<44s} {tp2_compiled:>8.1f} us")

    # Use compiled results for the main comparison
    tp1_best = min(tp1_mean, tp1_compiled)
    tp2_best = min(tp2_mean, tp2_compiled)

    saving = tp1_best - tp2_best
    pct = saving / tp1_best * 100 if tp1_best > 0 else 0
    print(f"\n  {'Compute saving (best of each)':<44s} {saving:>8.1f} us ({pct:.1f}%)")

    compile_speedup_tp1 = tp1_mean / tp1_compiled if tp1_compiled > 0 else 0
    compile_speedup_tp2 = tp2_mean / tp2_compiled if tp2_compiled > 0 else 0
    print(f"  {'mx.compile speedup (TP=1)':<44s} {compile_speedup_tp1:>8.2f}x")
    print(f"  {'mx.compile speedup (TP=2)':<44s} {compile_speedup_tp2:>8.2f}x")

    # Estimate real TP layer time with RDMA allreduce
    rdma_allreduce_us = 15.8  # measured per-call latency for TB5
    tp2_real = tp2_best + 2 * rdma_allreduce_us
    print()
    print(f"  Estimated real TP=2 with 2x RDMA allreduce ({rdma_allreduce_us} us/call):")
    print(f"  {'  sharded compute + 2x RDMA':<44s} {tp2_real:>8.1f} us")
    speedup = tp1_best / tp2_real if tp2_real > 0 else 0
    print(f"  {'  Estimated speedup vs baseline':<44s} {speedup:>8.2f}x")
    print()


if __name__ == "__main__":
    main()
