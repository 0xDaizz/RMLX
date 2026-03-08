#!/usr/bin/env python3
"""MLX Tensor Parallel comparison benchmark.

Measures single-layer transformer forward pass latency at TP=1 and TP=2
(simulated via half-sized weights), matching the RMLX distributed_bench
configuration (Mixtral: hidden=4096, intermediate=14336, 32 heads, 8 KV heads).

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
# Mixtral 8x7B-like config (matches RMLX distributed_bench)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
NUM_KV_HEADS = 8
INTERMEDIATE_SIZE = 14336
SEQ_LEN = 1  # decode token
RMS_NORM_EPS = 1e-5

WARMUP = 5
ITERS = 50


# ---------------------------------------------------------------------------
# RMS Norm
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=RMS_NORM_EPS):
    """RMS normalization. x: [..., hidden], weight: [hidden]."""
    rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rms * weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def apply_rope(x, offset, head_dim):
    """Apply rotary position embeddings. x: [B, n_heads, seq, head_dim]."""
    half = head_dim // 2
    positions = mx.arange(offset, offset + x.shape[2])
    freqs = 1.0 / (10000.0 ** (mx.arange(0, half).astype(mx.float32) / half))
    theta = positions[:, None] * freqs[None, :]
    cos_t = mx.cos(theta)[None, None, :, :]
    sin_t = mx.sin(theta)[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], axis=-1)


# ---------------------------------------------------------------------------
# Single layer forward (configurable head/dim counts for TP)
# ---------------------------------------------------------------------------
def layer_forward(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2,
                  num_heads, num_kv_heads, head_dim, hidden_size):
    """Single transformer decoder layer forward pass (decode mode, no KV cache).

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
    """
    batch = x.shape[0]

    # Pre-attention norm
    h = rms_norm(x, rms_w1)

    # QKV projections
    q = h @ w_q  # [batch, num_heads * head_dim]
    k = h @ w_k  # [batch, num_kv_heads * head_dim]
    v = h @ w_v  # [batch, num_kv_heads * head_dim]

    # Reshape for attention
    q = q.reshape(batch, num_heads, 1, head_dim)
    k = k.reshape(batch, num_kv_heads, 1, head_dim)
    v = v.reshape(batch, num_kv_heads, 1, head_dim)

    # RoPE
    q = apply_rope(q, 0, head_dim)
    k = apply_rope(k, 0, head_dim)

    # GQA: repeat KV heads to match Q heads
    if num_kv_heads < num_heads:
        rep = num_heads // num_kv_heads
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)

    # Attention (no cache, single token)
    scale = head_dim ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [batch, heads, 1, 1]
    attn = mx.softmax(scores, axis=-1)
    attn_out = (attn @ v).reshape(batch, num_heads * head_dim)  # [batch, q_dim]

    # Output projection
    o = attn_out @ w_o  # [batch, hidden_size]

    # Residual
    x = x + o

    # Pre-FFN norm
    h2 = rms_norm(x, rms_w2)

    # SwiGLU FFN
    gate = h2 @ w_gate
    up = h2 @ w_up
    hidden = mx.sigmoid(gate) * gate * up  # SiLU(gate) * up
    down = hidden @ w_down  # [batch, hidden_size]

    # Residual
    x = x + down

    return x


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

    # Evaluate to materialize
    mx.eval(w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2)

    return {
        "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
        "w_gate": w_gate, "w_up": w_up, "w_down": w_down,
        "rms_w1": rms_w1, "rms_w2": rms_w2,
        "num_heads": nh, "num_kv_heads": nkv,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def bench_layer(label, weights, hidden_size, head_dim, warmup, iters):
    """Benchmark a single layer forward pass."""
    x = (mx.random.normal((SEQ_LEN, hidden_size)) * 0.02).astype(mx.float16)
    mx.eval(x)

    nh = weights["num_heads"]
    nkv = weights["num_kv_heads"]

    # Warmup
    for _ in range(warmup):
        out = layer_forward(
            x, weights["w_q"], weights["w_k"], weights["w_v"], weights["w_o"],
            weights["w_gate"], weights["w_up"], weights["w_down"],
            weights["rms_w1"], weights["rms_w2"],
            nh, nkv, head_dim, hidden_size,
        )
        mx.eval(out)

    # Benchmark
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = layer_forward(
            x, weights["w_q"], weights["w_k"], weights["w_v"], weights["w_o"],
            weights["w_gate"], weights["w_up"], weights["w_down"],
            weights["rms_w1"], weights["rms_w2"],
            nh, nkv, head_dim, hidden_size,
        )
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
    parser = argparse.ArgumentParser(description="MLX TP comparison benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    print(f"MLX TP Benchmark")
    print(f"  Config: Mixtral-like, hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
          f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_SIZE}")
    print(f"  seq_len={SEQ_LEN} (decode), dtype=float16")
    print(f"  warmup={args.warmup}, iters={args.iters}")

    # ── TP=1 (full weights, no parallelism) ──
    print(f"\n=== TP=1: Full weights (baseline) ===")
    w_tp1 = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE, tp_size=1)
    tp1_mean = bench_layer(
        "TP=1 single-layer forward",
        w_tp1, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters,
    )

    # ── TP=2 (half-sized weights, simulating one rank) ──
    print(f"\n=== TP=2: Half-sized weights (simulating rank 0 of 2) ===")
    print(f"  Q/K/V/gate/up: column-parallel (output halved)")
    print(f"  O/down: row-parallel (input halved)")
    w_tp2 = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE, tp_size=2)
    tp2_mean = bench_layer(
        "TP=2 single-layer forward (half weights)",
        w_tp2, HIDDEN_SIZE, HEAD_DIM, args.warmup, args.iters,
    )

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"  {'TP=1 baseline':<44s} {tp1_mean:>8.1f} us")
    print(f"  {'TP=2 half-weight compute':<44s} {tp2_mean:>8.1f} us")

    saving = tp1_mean - tp2_mean
    pct = saving / tp1_mean * 100 if tp1_mean > 0 else 0
    print(f"  {'Compute saving from TP=2':<44s} {saving:>8.1f} us ({pct:.1f}%)")

    # Estimate real TP layer time with RDMA allreduce
    rdma_allreduce_us = 10.0  # conservative per-call estimate for TB5
    tp2_real = tp2_mean + 2 * rdma_allreduce_us
    print()
    print(f"  Estimated real TP=2 with 2x RDMA allreduce:")
    print(f"  {'  sharded compute + 2x RDMA':<44s} {tp2_real:>8.1f} us")
    speedup = tp1_mean / tp2_real if tp2_real > 0 else 0
    print(f"  {'  Estimated speedup vs baseline':<44s} {speedup:>8.2f}x")
    print()


if __name__ == "__main__":
    main()
