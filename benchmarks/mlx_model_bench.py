"""
Benchmark a single transformer layer forward pass in MLX.

Configuration matches RMLX's pipeline_bench:
  - Llama-2 7B-style single decoder layer
  - Decode mode (seq_len=1)
"""

import time
import statistics

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Configuration (matches RMLX pipeline_bench)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_DIM = 11008
SEQ_LEN = 1
RMS_NORM_EPS = 1e-5

WARMUP = 5
ITERS = 50


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Norms
        self.attention_norm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.ffn_norm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

        # Attention projections
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

        # FFN projections
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_DIM, HIDDEN_SIZE, bias=False)

        # RoPE
        self.rope = nn.RoPE(HEAD_DIM, base=10000.0)

    def __call__(self, x):
        # x: [batch=1, seq_len=1, hidden=4096]

        # ---------- Self-Attention ----------
        h = self.attention_norm(x)

        q = self.q_proj(h).reshape(1, -1, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(1, -1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(1, -1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        # q: [1, 32, seq, 128],  k/v: [1, 8, seq, 128]

        q = self.rope(q)
        k = self.rope(k)

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=HEAD_DIM**-0.5
        )
        # attn: [1, 32, seq, 128]
        attn = attn.transpose(0, 2, 1, 3).reshape(1, -1, NUM_HEADS * HEAD_DIM)

        o = self.o_proj(attn)
        x = x + o

        # ---------- FFN ----------
        h = self.ffn_norm(x)
        gate = self.gate_proj(h)
        up = self.up_proj(h)
        ffn = nn.silu(gate) * up
        ffn = self.down_proj(ffn)
        x = x + ffn

        return x


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def bench(fn, x, label: str):
    """Warm up, then time `ITERS` iterations, printing statistics."""
    # Warmup
    for _ in range(WARMUP):
        y = fn(x)
        mx.eval(y)

    times_ns = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        y = fn(x)
        mx.eval(y)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    times_us = [t / 1000.0 for t in times_ns]

    mean = statistics.mean(times_us)
    std = statistics.stdev(times_us) if len(times_us) > 1 else 0.0
    sorted_t = sorted(times_us)
    p50 = sorted_t[len(sorted_t) // 2]
    p95 = sorted_t[int(len(sorted_t) * 0.95)]
    p99 = sorted_t[int(len(sorted_t) * 0.99)]
    lo = sorted_t[0]
    hi = sorted_t[-1]

    print(f"\n=== {label} ({ITERS} iters, {WARMUP} warmup) ===")
    print(f"  mean : {mean:10.1f} us")
    print(f"  std  : {std:10.1f} us")
    print(f"  p50  : {p50:10.1f} us")
    print(f"  p95  : {p95:10.1f} us")
    print(f"  p99  : {p99:10.1f} us")
    print(f"  min  : {lo:10.1f} us")
    print(f"  max  : {hi:10.1f} us")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("MLX single-layer transformer benchmark")
    print(f"  hidden={HIDDEN_SIZE}  heads={NUM_HEADS}  kv_heads={NUM_KV_HEADS}  "
          f"head_dim={HEAD_DIM}  intermediate={INTERMEDIATE_DIM}")
    print(f"  seq_len={SEQ_LEN}  dtype=float32  warmup={WARMUP}  iters={ITERS}")

    layer = TransformerLayer()

    x = mx.random.normal((1, SEQ_LEN, HIDDEN_SIZE))
    mx.eval(x)
    mx.eval(layer.parameters())

    # 1) Uncompiled forward
    bench(layer, x, "Uncompiled forward")

    # 2) Compiled forward
    compiled_fn = mx.compile(layer)
    bench(compiled_fn, x, "Compiled forward (mx.compile)")


if __name__ == "__main__":
    main()
