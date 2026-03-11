#!/usr/bin/env python3
"""MLX Prefill Benchmark — single Qwen 3.5 MoE expert layer.

Matches rmlx prefill_bench config for direct comparison.
Run each seq_len as a separate invocation to avoid GPU resource accumulation.

Usage:
    python mlx_prefill_bench.py              # all seq_lens
    python mlx_prefill_bench.py 128          # single seq_len
    python mlx_prefill_bench.py 16 32 48     # specific seq_lens
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

# Config
HIDDEN_SIZE = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_DIM = 2560
RMS_NORM_EPS = 1e-5
ROPE_THETA = 1000000.0
MAX_SEQ_LEN = 2048
PARAMS_PER_LAYER = 56_885_248.0

WARMUP_ITERS = 5
BENCH_ITERS = 20


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.rope = nn.RoPE(HEAD_DIM, traditional=True, base=ROPE_THETA)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x, mask=None, cache=None):
        # Add batch dim for MLX attention
        x_3d = x[None, :, :]  # [1, seq_len, hidden]

        q = self.q_proj(x_3d)
        k = self.k_proj(x_3d)
        v = self.v_proj(x_3d)

        # Reshape to [1, seq_len, num_heads, head_dim] then transpose to [1, num_heads, seq_len, head_dim]
        q = q.reshape(1, -1, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(1, -1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(1, -1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: repeat K/V heads
        if NUM_KV_HEADS < NUM_HEADS:
            n_rep = NUM_HEADS // NUM_KV_HEADS
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot product attention with causal mask
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            w = w + mask
        w = mx.softmax(w, axis=-1)
        o = w @ v  # [1, num_heads, seq_len, head_dim]

        o = o.transpose(0, 2, 1, 3).reshape(1, -1, NUM_HEADS * HEAD_DIM)
        o = self.o_proj(o)
        return o[0]  # Remove batch dim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_DIM, HIDDEN_SIZE, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

    def __call__(self, x, mask=None, cache=None):
        r = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask=mask, cache=cache)
        h = r + h
        r = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = r + h
        return h


def build_causal_mask(seq_len):
    """Build causal attention mask."""
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    return mask.astype(mx.float16)


def bench_seq_len(block, seq_len):
    """Benchmark a single seq_len."""
    x = mx.random.normal((seq_len, HIDDEN_SIZE)).astype(mx.float16)
    mask = build_causal_mask(seq_len)

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = block(x, mask=mask)
        mx.eval(out)

    # Bench
    latencies = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        out = block(x, mask=mask)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1e6)  # to microseconds

    latencies.sort()
    n = len(latencies)
    mean = sum(latencies) / n
    variance = sum((lat - mean) ** 2 for lat in latencies) / n
    std = variance ** 0.5
    p50 = latencies[n // 2]

    tflops = 2.0 * PARAMS_PER_LAYER * seq_len / (mean / 1e6) / 1e12

    return {
        "mean": mean,
        "std": std,
        "p50": p50,
        "min": latencies[0],
        "max": latencies[-1],
        "tflops": tflops,
    }


def main():
    if len(sys.argv) > 1:
        seq_lens = [int(s) for s in sys.argv[1:]]
    else:
        seq_lens = [16, 32, 48, 128, 256, 512, 1024, 2048]

    print("MLX Prefill Benchmark")
    print(
        f"Config: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_DIM}"
    )
    print("dtype: float16")
    print(f"Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS}")
    print(f"PARAMS_PER_LAYER: {PARAMS_PER_LAYER:.0f}")
    print()

    block = TransformerBlock()
    # Initialize with float16
    block.set_dtype(mx.float16)
    # Force parameter initialization
    x_init = mx.random.normal((1, HIDDEN_SIZE)).astype(mx.float16)
    mx.eval(block(x_init))

    results = []
    for seq_len in seq_lens:
        r = bench_seq_len(block, seq_len)
        print(
            f"seq_len={seq_len:>5d}  "
            f"mean={r['mean']:>10.1f}us  "
            f"std={r['std']:>8.1f}us  "
            f"p50={r['p50']:>10.1f}us  "
            f"min={r['min']:>10.1f}us  "
            f"max={r['max']:>10.1f}us  "
            f"TFLOPS={r['tflops']:.2f}"
        )
        results.append((seq_len, r))

    # Summary table
    print()
    print("=" * 80)
    print(
        f"{'seq_len':>8} | {'mean (us)':>12} | {'TFLOPS':>8} | "
        f"{'p50 (us)':>12} | {'std (us)':>10}"
    )
    print("-" * 80)
    for seq_len, r in results:
        print(
            f"{seq_len:>8} | {r['mean']:>12.0f} | {r['tflops']:>8.2f} | "
            f"{r['p50']:>12.0f} | {r['std']:>10.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
