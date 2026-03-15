#!/usr/bin/env python3
"""MLX JACCL TP layer forward benchmark — distributed Qwen 3.5 MoE A22B.

Measures real end-to-end per-layer latency with JACCL allreduce over
Thunderbolt 5, comparing TP=1 baseline vs TP=2 distributed.

Supports multiple seq_len values:
  - seq_len=1   → decode  (KV cache pre-filled with 128 tokens)
  - seq_len>1   → prefill (KV cache empty, processing initial prompt)

Benchmarks (per seq_len):
  1. TP=1 baseline  — full weights, no allreduce, mx.compile, rank 0 only
  2. TP=2 distributed — half weights, real JACCL allreduce (2x per layer)
  3. Allreduce-only — [seq_len, 3584] f16, 2x per layer

Launch:
    mlx.launch --backend jaccl --hostfile ../../rmlx-hosts.json -- \
        python3 bench_mlx_tp_layer.py [--warmup 10] [--iters 50]
"""

import argparse
import statistics
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Qwen 3.5 MoE A22B config
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 3584
NUM_HEADS = 28
HEAD_DIM = 128
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 18944
SEQ_LEN_VALUES = [1, 128, 512]
RMS_NORM_EPS = 1e-6
DECODE_KV_CACHE_LEN = 128  # pre-filled tokens for decode (seq_len=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def fmt_stats(latencies):
    """Return (mean, p50, p95) in microseconds from a list of us values."""
    mean = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p95_idx = int(0.95 * (len(sorted_lat) - 1))
    p95 = sorted_lat[p95_idx]
    return mean, p50, p95


def print_stats(label, latencies, n):
    mean, p50, p95 = fmt_stats(latencies)
    print(f"  {label:<20s} mean={mean:8.1f}us p50={p50:8.1f}us p95={p95:8.1f}us (n={n})")
    return mean


# ---------------------------------------------------------------------------
# Single layer forward (fused kernels, KV cache) — no allreduce
# ---------------------------------------------------------------------------
def layer_forward(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2,
                  num_heads, num_kv_heads, head_dim, hidden_size,
                  k_cache, v_cache, offset):
    """Single transformer decoder layer forward pass (decode mode, with KV cache)."""
    seq_len = x.shape[0]

    # Pre-attention norm
    h = mx.fast.rms_norm(x, rms_w1, RMS_NORM_EPS)

    # QKV projections
    q = h @ w_q
    k = h @ w_k
    v = h @ w_v

    # Reshape: [batch=1, heads, seq_len, head_dim]
    q = q.reshape(1, num_heads, seq_len, head_dim)
    k = k.reshape(1, num_kv_heads, seq_len, head_dim)
    v = v.reshape(1, num_kv_heads, seq_len, head_dim)

    # RoPE
    q = mx.fast.rope(q, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)
    k = mx.fast.rope(k, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)

    # KV cache update
    k_cache_new = mx.concatenate([k_cache, k], axis=2)
    v_cache_new = mx.concatenate([v_cache, v], axis=2)

    # SDPA (handles GQA internally)
    scale = head_dim ** -0.5
    attn_out = mx.fast.scaled_dot_product_attention(q, k_cache_new, v_cache_new, scale=scale)
    attn_out = attn_out.reshape(seq_len, num_heads * head_dim)

    # O projection + residual
    x = x + attn_out @ w_o

    # Pre-FFN norm
    h2 = mx.fast.rms_norm(x, rms_w2, RMS_NORM_EPS)

    # SwiGLU FFN
    gate = h2 @ w_gate
    up = h2 @ w_up
    hidden = mx.sigmoid(gate) * gate * up
    x = x + hidden @ w_down

    return x, k_cache_new, v_cache_new


# ---------------------------------------------------------------------------
# Build weights
# ---------------------------------------------------------------------------
def build_weights(hidden_size, num_heads, num_kv_heads, head_dim, intermediate,
                   tp_size=1, kv_cache_len=0):
    """Build random weights for one transformer layer.

    For tp_size > 1:
    - Q/K/V/gate/up: column parallel (output dim / tp_size)
    - O/down: row parallel (input dim / tp_size)

    kv_cache_len: number of pre-filled KV cache tokens.
      - decode (seq_len=1): 128 tokens pre-filled
      - prefill (seq_len>1): 0 (empty cache)
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

    # KV cache (empty for prefill, pre-filled for decode)
    if kv_cache_len > 0:
        k_cache = mx.zeros((1, nkv, kv_cache_len, head_dim), dtype=mx.float16)
        v_cache = mx.zeros((1, nkv, kv_cache_len, head_dim), dtype=mx.float16)
    else:
        k_cache = mx.zeros((1, nkv, 0, head_dim), dtype=mx.float16)
        v_cache = mx.zeros((1, nkv, 0, head_dim), dtype=mx.float16)

    mx.eval(w_q, w_k, w_v, w_o, w_gate, w_up, w_down, rms_w1, rms_w2, k_cache, v_cache)

    return {
        "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
        "w_gate": w_gate, "w_up": w_up, "w_down": w_down,
        "rms_w1": rms_w1, "rms_w2": rms_w2,
        "k_cache": k_cache, "v_cache": v_cache,
        "num_heads": nh, "num_kv_heads": nkv,
    }


# ---------------------------------------------------------------------------
# TP=1 baseline benchmark (rank 0 only, mx.compile)
# ---------------------------------------------------------------------------
def bench_tp1_baseline(warmup, iters, seq_len=1):
    """Full-weight layer forward on rank 0 only, with mx.compile."""
    kv_cache_len = DECODE_KV_CACHE_LEN if seq_len == 1 else 0
    offset = kv_cache_len  # decode: after cached tokens, prefill: from 0

    w = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE,
                      tp_size=1, kv_cache_len=kv_cache_len)
    x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
    mx.eval(x)

    nh = w["num_heads"]
    nkv = w["num_kv_heads"]

    forward_fn = mx.compile(layer_forward)

    def run_once():
        return forward_fn(
            x, w["w_q"], w["w_k"], w["w_v"], w["w_o"],
            w["w_gate"], w["w_up"], w["w_down"],
            w["rms_w1"], w["rms_w2"],
            nh, nkv, HEAD_DIM, HIDDEN_SIZE,
            w["k_cache"], w["v_cache"], offset,
        )

    # Warmup
    for _ in range(warmup):
        out, _, _ = run_once()
        mx.eval(out)
        mx.synchronize()

    # Timed
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _, _ = run_once()
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    return latencies


# ---------------------------------------------------------------------------
# TP=2 distributed forward (half weights + real JACCL allreduce)
# ---------------------------------------------------------------------------
def tp2_layer_forward_distributed(x, w, group, hidden_size, head_dim, offset):
    """TP=2 layer forward with real allreduce after O-proj and after down-proj.

    Compute is NOT compiled — we measure wall-clock including allreduce sync.
    """
    seq_len = x.shape[0]
    nh = w["num_heads"]
    nkv = w["num_kv_heads"]

    # Pre-attention norm
    h = mx.fast.rms_norm(x, w["rms_w1"], RMS_NORM_EPS)

    # QKV projections (column-parallel: each rank has half the heads)
    q = h @ w["w_q"]
    k = h @ w["w_k"]
    v = h @ w["w_v"]

    q = q.reshape(1, nh, seq_len, head_dim)
    k = k.reshape(1, nkv, seq_len, head_dim)
    v = v.reshape(1, nkv, seq_len, head_dim)

    q = mx.fast.rope(q, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)
    k = mx.fast.rope(k, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)

    k_cache_new = mx.concatenate([w["k_cache"], k], axis=2)
    v_cache_new = mx.concatenate([w["v_cache"], v], axis=2)

    scale = head_dim ** -0.5
    attn_out = mx.fast.scaled_dot_product_attention(q, k_cache_new, v_cache_new, scale=scale)
    attn_out = attn_out.reshape(seq_len, nh * head_dim)

    # O projection (row-parallel: partial sum)
    attn_out = attn_out @ w["w_o"]

    # ── Allreduce #1: attention output ──
    attn_out = mx.distributed.all_sum(attn_out, group=group, stream=mx.cpu)
    mx.eval(attn_out)
    mx.synchronize()

    # Residual
    x = x + attn_out

    # Pre-FFN norm
    h2 = mx.fast.rms_norm(x, w["rms_w2"], RMS_NORM_EPS)

    # SwiGLU FFN (column-parallel gate/up, row-parallel down)
    gate = h2 @ w["w_gate"]
    up = h2 @ w["w_up"]
    hidden = mx.sigmoid(gate) * gate * up
    ffn_out = hidden @ w["w_down"]

    # ── Allreduce #2: FFN output ──
    ffn_out = mx.distributed.all_sum(ffn_out, group=group, stream=mx.cpu)
    mx.eval(ffn_out)
    mx.synchronize()

    # Residual
    x = x + ffn_out

    return x, k_cache_new, v_cache_new


def bench_tp2_distributed(group, warmup, iters, seq_len=1):
    """TP=2 layer forward with real JACCL allreduce."""
    kv_cache_len = DECODE_KV_CACHE_LEN if seq_len == 1 else 0
    offset = kv_cache_len

    w = build_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE,
                      tp_size=2, kv_cache_len=kv_cache_len)
    x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        barrier(group)
        out, _, _ = tp2_layer_forward_distributed(x, w, group, HIDDEN_SIZE, HEAD_DIM, offset)
        mx.eval(out)

    # Timed
    latencies = []
    for _ in range(iters):
        barrier(group)
        t0 = time.perf_counter()
        out, _, _ = tp2_layer_forward_distributed(x, w, group, HIDDEN_SIZE, HEAD_DIM, offset)
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    return latencies


# ---------------------------------------------------------------------------
# Allreduce-only benchmark ([seq_len, 3584] f16, 2x per layer)
# ---------------------------------------------------------------------------
def bench_allreduce_only(group, warmup, iters, seq_len=1):
    """Measure 2x allreduce latency matching TP layer payload: [seq_len, 3584] f16."""
    payload = mx.zeros((seq_len, HIDDEN_SIZE), dtype=mx.float16)
    mx.eval(payload)

    # Warmup
    for _ in range(warmup):
        barrier(group)
        r1 = mx.distributed.all_sum(payload, group=group, stream=mx.cpu)
        mx.eval(r1)
        mx.synchronize()
        r2 = mx.distributed.all_sum(payload, group=group, stream=mx.cpu)
        mx.eval(r2)
        mx.synchronize()

    # Timed
    latencies = []
    for _ in range(iters):
        barrier(group)
        t0 = time.perf_counter()
        r1 = mx.distributed.all_sum(payload, group=group, stream=mx.cpu)
        mx.eval(r1)
        mx.synchronize()
        r2 = mx.distributed.all_sum(payload, group=group, stream=mx.cpu)
        mx.eval(r2)
        mx.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    return latencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX JACCL TP layer forward benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # ── Distributed init ─────────────────────────────────────────────────
    group = mx.distributed.init(strict=True)
    rank = group.rank()
    ws = group.size()

    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw, flush=True)

    log(f"MLX JACCL TP Layer Benchmark")
    log(f"  Config: Qwen 3.5 MoE A22B, hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_SIZE}")
    log(f"  seq_len values: {SEQ_LEN_VALUES}, dtype=float16")
    log(f"  Backend: JACCL, world_size={ws}")
    log(f"  warmup={args.warmup}, iters={args.iters}")

    assert ws == 2, f"This benchmark requires world_size=2, got {ws}"

    # ── RDMA warmup ──────────────────────────────────────────────────────
    rdma_warmup(group, rank)
    barrier(group)

    # Collect summary results for final table
    summaries = []

    for seq_len in SEQ_LEN_VALUES:
        is_decode = (seq_len == 1)
        kv_cache_len = DECODE_KV_CACHE_LEN if is_decode else 0
        mode_label = "Decode" if is_decode else "Prefill"
        payload_bytes = seq_len * HIDDEN_SIZE * 2  # f16 = 2 bytes

        log(f"\n{'=' * 60}")
        log(f"=== {mode_label} (seq_len={seq_len}, kv_cache={kv_cache_len}) ===")
        log(f"{'=' * 60}")

        # ==============================================================
        # 1. TP=1 Baseline (rank 0 only, no allreduce)
        # ==============================================================
        log(f"\n  --- TP=1 Baseline (rank 0 only, no allreduce) ---")

        if rank == 0:
            tp1_lats = bench_tp1_baseline(args.warmup, args.iters, seq_len=seq_len)
            tp1_mean = print_stats("tp1_compiled", tp1_lats, args.iters)
        else:
            tp1_mean = 0.0

        barrier(group)

        # ==============================================================
        # 2. TP=2 Distributed (real JACCL allreduce)
        # ==============================================================
        log(f"\n  --- TP=2 Distributed (real JACCL allreduce) ---")

        tp2_lats = bench_tp2_distributed(group, args.warmup, args.iters, seq_len=seq_len)
        if rank == 0:
            tp2_mean = print_stats("tp2_distributed", tp2_lats, args.iters)
        else:
            tp2_mean = 0.0

        barrier(group)

        # ==============================================================
        # 3. Allreduce only ([seq_len, 3584] f16, 2x per layer)
        # ==============================================================
        log(f"\n  --- Allreduce only ({payload_bytes} bytes f16, 2x per layer) ---")

        ar_lats = bench_allreduce_only(group, args.warmup, args.iters, seq_len=seq_len)
        if rank == 0:
            ar_mean = print_stats("2x_allreduce", ar_lats, args.iters)
        else:
            ar_mean = 0.0

        barrier(group)

        if rank == 0:
            summaries.append({
                "mode": mode_label,
                "seq_len": seq_len,
                "kv_cache": kv_cache_len,
                "tp1": tp1_mean,
                "tp2": tp2_mean,
                "allreduce": ar_mean,
            })

    # ==================================================================
    # Summary table (rank 0 only)
    # ==================================================================
    if rank == 0:
        print()
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)

        for s in summaries:
            print(f"\n  {s['mode']} (seq_len={s['seq_len']}, kv_cache={s['kv_cache']}):")
            print(f"    TP=1 baseline:     {s['tp1']:8.1f} us")
            print(f"    TP=2 JACCL:        {s['tp2']:8.1f} us")
            print(f"    2x allreduce:      {s['allreduce']:8.1f} us")

            if s['tp2'] > 0:
                speedup = s['tp1'] / s['tp2']
                print(f"    TP=2 speedup:      {speedup:8.2f}x")

            overhead = s['tp2'] - (s['tp1'] / 2.0) if s['tp1'] > 0 else 0
            print(f"    Comm overhead:     {overhead:8.1f} us (tp2 - tp1/2)")

        print()

    barrier(group)


if __name__ == "__main__":
    main()
