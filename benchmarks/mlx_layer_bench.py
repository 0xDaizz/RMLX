#!/usr/bin/env python3
"""MLX single transformer layer decode latency benchmark.

Measures raw-op latency for a Llama-2 7B style decoder layer in decode mode
(seq_len=1) for direct comparison with RMLX benchmarks.

All operations use raw MLX primitives (no mlx.nn layers) for fair comparison.

Usage:
    python mlx_layer_bench.py [--warmup N] [--iters N]
"""

import argparse
import time
import statistics

import mlx.core as mx

# ---------------------------------------------------------------------------
# Llama-2 7B shapes
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
INTERMEDIATE_SIZE = 11008
NUM_KV_HEADS = 32
SEQ_LEN = 1          # decode
KV_CACHE_LEN = 128   # existing cached tokens

WARMUP = 5
ITERS = 50


# ---------------------------------------------------------------------------
# RoPE (raw implementation)
# ---------------------------------------------------------------------------
def apply_rope(x, offset, head_dim):
    """Apply rotary position embeddings. x: [B, n_heads, seq, head_dim]."""
    half = head_dim // 2
    positions = mx.arange(offset, offset + x.shape[2])  # [seq]
    freqs = 1.0 / (10000.0 ** (mx.arange(0, half).astype(mx.float32) / half))
    theta = positions[:, None] * freqs[None, :]  # [seq, half]
    cos_t = mx.cos(theta)  # [seq, half]
    sin_t = mx.sin(theta)  # [seq, half]
    # Broadcast to [1, 1, seq, half]
    cos_t = cos_t[None, None, :, :]
    sin_t = sin_t[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], axis=-1)


# ---------------------------------------------------------------------------
# RMS Norm (raw implementation)
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=1e-5):
    """RMS normalization. x: [..., hidden], weight: [hidden]."""
    rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rms * weight


# ---------------------------------------------------------------------------
# Single layer forward (raw ops)
# ---------------------------------------------------------------------------
def layer_forward(x, w_qkv, w_o, w_gate_up, w_down, rms_w1, rms_w2,
                  k_cache, v_cache, rope_offset):
    """Single Llama-2 7B decoder layer forward pass (decode mode).

    Args:
        x:          [1, 1, 4096]  input hidden state
        w_qkv:      [4096, 12288] merged QKV projection  (32+32+32)*128 = 12288
        w_o:        [4096, 4096]  output projection
        w_gate_up:  [4096, 22016] merged gate+up projection (11008*2)
        w_down:     [11008, 4096] down projection
        rms_w1:     [4096]        pre-attention RMS norm weight
        rms_w2:     [4096]        pre-FFN RMS norm weight
        k_cache:    [1, 32, 128, 128]  key cache (KV_CACHE_LEN tokens)
        v_cache:    [1, 32, 128, 128]  value cache
        rope_offset: int  position offset for RoPE
    """
    B = 1

    # --- Pre-attention RMS norm ---
    normed = rms_norm(x, rms_w1)

    # --- QKV projection ---
    # normed: [1, 1, 4096] @ w_qkv: [4096, 12288] -> [1, 1, 12288]
    qkv = normed @ w_qkv

    # Split into Q, K, V
    q = qkv[..., :NUM_HEADS * HEAD_DIM]                                    # [1, 1, 4096]
    k = qkv[..., NUM_HEADS * HEAD_DIM:NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM]  # [1, 1, 4096]
    v = qkv[..., NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM:]          # [1, 1, 4096]

    # Reshape to [B, n_heads, seq, head_dim]
    q = q.reshape(B, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)     # [1, 32, 1, 128]
    k = k.reshape(B, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [1, 32, 1, 128]
    v = v.reshape(B, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [1, 32, 1, 128]

    # --- RoPE ---
    q = apply_rope(q, rope_offset, HEAD_DIM)
    k = apply_rope(k, rope_offset, HEAD_DIM)

    # --- Append to KV cache ---
    k_full = mx.concatenate([k_cache, k], axis=2)  # [1, 32, 129, 128]
    v_full = mx.concatenate([v_cache, v], axis=2)   # [1, 32, 129, 128]

    # --- Scaled dot-product attention ---
    scale = HEAD_DIM ** -0.5
    attn = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=scale)
    # attn: [1, 32, 1, 128]

    # --- O projection + residual ---
    attn_out = attn.transpose(0, 2, 1, 3).reshape(B, SEQ_LEN, NUM_HEADS * HEAD_DIM)  # [1, 1, 4096]
    o = attn_out @ w_o  # [1, 1, 4096]
    x = x + o

    # --- Pre-FFN RMS norm ---
    normed2 = rms_norm(x, rms_w2)

    # --- Gate+Up projection ---
    # normed2: [1, 1, 4096] @ w_gate_up: [4096, 22016] -> [1, 1, 22016]
    gate_up = normed2 @ w_gate_up
    gate = gate_up[..., :INTERMEDIATE_SIZE]   # [1, 1, 11008]
    up = gate_up[..., INTERMEDIATE_SIZE:]     # [1, 1, 11008]

    # --- SiLU * gate ---
    ffn = (gate * mx.sigmoid(gate)) * up  # SiLU(gate) * up

    # --- Down projection + residual ---
    down = ffn @ w_down  # [1, 1, 4096]
    x = x + down

    return x, k_full, v_full


# ---------------------------------------------------------------------------
# Per-operation breakdown
# ---------------------------------------------------------------------------
def bench_per_op(x, w_qkv, w_o, w_gate_up, w_down, rms_w1, rms_w2,
                 k_cache, v_cache, rope_offset, warmup, iters):
    """Benchmark each sub-operation independently."""
    B = 1

    def bench_single(name, fn, warmup_n=warmup, iters_n=iters):
        for _ in range(warmup_n):
            mx.eval(fn())
        times = []
        for _ in range(iters_n):
            t0 = time.perf_counter_ns()
            mx.eval(fn())
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)
        times.sort()
        mean = statistics.mean(times)
        p50 = times[len(times) // 2]
        print(f"  {name:42s}  mean={mean:8.1f}us  p50={p50:8.1f}us")
        return mean

    print("\n--- Per-Operation Breakdown ---")

    # Pre-attention RMS norm
    normed = rms_norm(x, rms_w1)
    mx.eval(normed)
    t_rms1 = bench_single("rms_norm (pre-attn)", lambda: rms_norm(x, rms_w1))

    # QKV projection
    t_qkv = bench_single("qkv_proj [4096x12288]", lambda: normed @ w_qkv)

    qkv = normed @ w_qkv
    mx.eval(qkv)
    q = qkv[..., :NUM_HEADS * HEAD_DIM].reshape(B, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_new = qkv[..., NUM_HEADS * HEAD_DIM:NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM].reshape(B, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    mx.eval(q, k_new)

    # RoPE
    t_rope = bench_single("rope", lambda: apply_rope(q, rope_offset, HEAD_DIM))

    # KV cache concat
    t_kv = bench_single("kv_cache_concat", lambda: mx.concatenate([k_cache, k_new], axis=2))

    # SDPA
    q_r = apply_rope(q, rope_offset, HEAD_DIM)
    k_r = apply_rope(k_new, rope_offset, HEAD_DIM)
    k_full = mx.concatenate([k_cache, k_r], axis=2)
    v_new = qkv[..., NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM:].reshape(B, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_full = mx.concatenate([v_cache, v_new], axis=2)
    mx.eval(q_r, k_full, v_full)
    scale = HEAD_DIM ** -0.5
    t_sdpa = bench_single("sdpa (decode, kv=129)",
                          lambda: mx.fast.scaled_dot_product_attention(q_r, k_full, v_full, scale=scale))

    # O projection
    attn = mx.fast.scaled_dot_product_attention(q_r, k_full, v_full, scale=scale)
    attn_out = attn.transpose(0, 2, 1, 3).reshape(B, SEQ_LEN, NUM_HEADS * HEAD_DIM)
    mx.eval(attn_out)
    t_oproj = bench_single("o_proj [4096x4096]", lambda: attn_out @ w_o)

    # Pre-FFN RMS norm
    o = attn_out @ w_o
    x2 = x + o
    mx.eval(x2)
    t_rms2 = bench_single("rms_norm (pre-ffn)", lambda: rms_norm(x2, rms_w2))

    # Gate+Up projection
    normed2 = rms_norm(x2, rms_w2)
    mx.eval(normed2)
    t_gateup = bench_single("gate_up_proj [4096x22016]", lambda: normed2 @ w_gate_up)

    # SiLU * gate
    gate_up = normed2 @ w_gate_up
    mx.eval(gate_up)
    gate = gate_up[..., :INTERMEDIATE_SIZE]
    up = gate_up[..., INTERMEDIATE_SIZE:]
    mx.eval(gate, up)
    t_silu = bench_single("silu_gate", lambda: (gate * mx.sigmoid(gate)) * up)

    # Down projection
    ffn = (gate * mx.sigmoid(gate)) * up
    mx.eval(ffn)
    t_down = bench_single("down_proj [11008x4096]", lambda: ffn @ w_down)

    total = t_rms1 + t_qkv + t_rope + t_kv + t_sdpa + t_oproj + t_rms2 + t_gateup + t_silu + t_down
    print(f"  {'sum of ops':42s}  total={total:8.1f}us")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_layer(fn, label, warmup, iters):
    """Benchmark a callable, forcing eval after each call."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    times.sort()
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    p50 = times[n // 2]
    p95 = times[int(n * 0.95)]
    p99 = times[int(n * 0.99)]
    lo = times[0]
    hi = times[-1]

    print(f"\n=== {label} ({iters} iters, {warmup} warmup) ===")
    print(f"  mean : {mean:10.1f} us")
    print(f"  std  : {std:10.1f} us")
    print(f"  p50  : {p50:10.1f} us")
    print(f"  p95  : {p95:10.1f} us")
    print(f"  p99  : {p99:10.1f} us")
    print(f"  min  : {lo:10.1f} us")
    print(f"  max  : {hi:10.1f} us")
    return mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX single-layer decode latency benchmark (raw ops)")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    print("=" * 72)
    print("MLX Single-Layer Decode Latency Benchmark (raw ops)")
    print(f"  mlx version : {mx.__version__}")
    print(f"  device      : {mx.default_device()}")
    print(f"  dtype       : float32")
    print(f"  hidden      : {HIDDEN_SIZE}")
    print(f"  heads       : {NUM_HEADS} (kv_heads={NUM_KV_HEADS})")
    print(f"  head_dim    : {HEAD_DIM}")
    print(f"  intermediate: {INTERMEDIATE_SIZE}")
    print(f"  seq_len     : {SEQ_LEN} (decode)")
    print(f"  kv_cache    : {KV_CACHE_LEN} tokens")
    print(f"  warmup      : {args.warmup}")
    print(f"  iters       : {args.iters}")
    print("=" * 72)

    # --- Allocate weights (f32) ---
    scale = 0.01
    w_qkv     = mx.random.normal((HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM)) * scale
    w_o       = mx.random.normal((NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)) * scale
    w_gate_up = mx.random.normal((HIDDEN_SIZE, INTERMEDIATE_SIZE * 2)) * scale
    w_down    = mx.random.normal((INTERMEDIATE_SIZE, HIDDEN_SIZE)) * scale
    rms_w1    = mx.ones((HIDDEN_SIZE,))
    rms_w2    = mx.ones((HIDDEN_SIZE,))

    # --- Allocate input and KV cache ---
    x = mx.random.normal((1, SEQ_LEN, HIDDEN_SIZE))
    k_cache = mx.random.normal((1, NUM_KV_HEADS, KV_CACHE_LEN, HEAD_DIM)) * scale
    v_cache = mx.random.normal((1, NUM_KV_HEADS, KV_CACHE_LEN, HEAD_DIM)) * scale
    rope_offset = KV_CACHE_LEN

    # Force materialization
    mx.eval(w_qkv, w_o, w_gate_up, w_down, rms_w1, rms_w2, x, k_cache, v_cache)

    # ---- 1. Single layer (uncompiled) ----
    def run_layer():
        return layer_forward(x, w_qkv, w_o, w_gate_up, w_down,
                             rms_w1, rms_w2, k_cache, v_cache, rope_offset)

    bench_layer(run_layer, "Single layer (uncompiled)", args.warmup, args.iters)

    # ---- 2. Single layer (compiled) ----
    compiled_forward = mx.compile(layer_forward)

    def run_compiled():
        return compiled_forward(x, w_qkv, w_o, w_gate_up, w_down,
                                rms_w1, rms_w2, k_cache, v_cache, rope_offset)

    bench_layer(run_compiled, "Single layer (mx.compile)", args.warmup, args.iters)

    # ---- 3. 4-layer sequential (compiled) ----
    # Create 4 sets of weights
    layers_w = []
    for _ in range(4):
        lw = {
            "w_qkv":     mx.random.normal((HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM)) * scale,
            "w_o":       mx.random.normal((NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)) * scale,
            "w_gate_up": mx.random.normal((HIDDEN_SIZE, INTERMEDIATE_SIZE * 2)) * scale,
            "w_down":    mx.random.normal((INTERMEDIATE_SIZE, HIDDEN_SIZE)) * scale,
            "rms_w1":    mx.ones((HIDDEN_SIZE,)),
            "rms_w2":    mx.ones((HIDDEN_SIZE,)),
            "k_cache":   mx.random.normal((1, NUM_KV_HEADS, KV_CACHE_LEN, HEAD_DIM)) * scale,
            "v_cache":   mx.random.normal((1, NUM_KV_HEADS, KV_CACHE_LEN, HEAD_DIM)) * scale,
        }
        mx.eval(*lw.values())
        layers_w.append(lw)

    def run_4_layers():
        h = x
        all_kv = []
        for lw in layers_w:
            h, k_out, v_out = layer_forward(
                h, lw["w_qkv"], lw["w_o"], lw["w_gate_up"], lw["w_down"],
                lw["rms_w1"], lw["rms_w2"], lw["k_cache"], lw["v_cache"],
                rope_offset,
            )
            all_kv.extend([k_out, v_out])
        return (h, *all_kv)

    compiled_4 = mx.compile(run_4_layers)

    bench_layer(compiled_4, "4-layer sequential (mx.compile)", args.warmup, args.iters)

    # ---- 4. Per-operation breakdown ----
    bench_per_op(x, w_qkv, w_o, w_gate_up, w_down, rms_w1, rms_w2,
                 k_cache, v_cache, rope_offset, args.warmup, args.iters)

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
