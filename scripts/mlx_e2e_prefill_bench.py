#!/usr/bin/env python3
"""MLX E2E Prefill Benchmark — 32-layer Qwen 7B-style transformer.

Matches rmlx e2e_prefill_bench.rs configuration exactly for direct comparison.

Usage:
    python mlx_e2e_prefill_bench.py              # all seq_lens
    python mlx_e2e_prefill_bench.py 128          # single seq_len
    python mlx_e2e_prefill_bench.py 128 256 512  # specific seq_lens
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Qwen 7B-style config (must match e2e_prefill_bench.rs exactly)
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_DIM = 14336
RMS_NORM_EPS = 1e-5
ROPE_THETA = 1000000.0
VOCAB_SIZE = 152064
NUM_LAYERS = 32

# FLOPs per token per layer (matching Rust):
#   Q_proj: 2*4096*4096, K_proj: 2*4096*1024, V_proj: 2*4096*1024,
#   O_proj: 2*4096*4096, gate: 2*4096*14336, up: 2*4096*14336, down: 2*14336*4096
#   = 2 * 214_695_936
FLOPS_PER_TOKEN_PER_LAYER = 2.0 * 214_695_936.0  # 429,391,872

SEQ_LENS = [32, 64, 128, 256, 512, 1024]
WARMUP_ITERS = 3
BENCH_ITERS = 10


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.rope = nn.RoPE(HEAD_DIM, traditional=True, base=ROPE_THETA)
        self.scale = HEAD_DIM**-0.5

    def __call__(self, x, mask=None):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        o = o.transpose(0, 2, 1, 3).reshape(B, L, NUM_HEADS * HEAD_DIM)
        return self.o_proj(o)


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

    def __call__(self, x, mask=None):
        r = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask=mask)
        h = r + h
        r = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = r + h
        return h


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = [TransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def __call__(self, token_ids):
        # token_ids: [1, seq_len] integer
        h = self.embed_tokens(token_ids)  # [1, seq_len, hidden]

        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)

        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.norm(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_stats(latencies_us):
    n = len(latencies_us)
    latencies_us = sorted(latencies_us)
    mean = sum(latencies_us) / n
    variance = sum((x - mean) ** 2 for x in latencies_us) / n
    std = variance**0.5
    p50 = latencies_us[n // 2]
    return {
        "mean": mean,
        "std": std,
        "p50": p50,
        "min": latencies_us[0],
        "max": latencies_us[-1],
    }


def compute_tflops(seq_len, mean_us):
    total_flops = FLOPS_PER_TOKEN_PER_LAYER * NUM_LAYERS * seq_len
    seconds = mean_us / 1e6
    return total_flops / seconds / 1e12


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_seq_len(model, seq_len):
    token_ids = mx.random.randint(0, VOCAB_SIZE, shape=(1, seq_len))

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = model(token_ids)
        mx.eval(out)

    # Benchmark
    latencies = []
    for _ in range(BENCH_ITERS):
        start = time.perf_counter()
        out = model(token_ids)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1e6)

    return compute_stats(latencies)


def main():
    if len(sys.argv) > 1:
        seq_lens = [int(s) for s in sys.argv[1:]]
    else:
        seq_lens = SEQ_LENS

    print(f"=== MLX E2E Prefill (32-layer, Qwen 7B-style) ===")
    print(
        f"Config: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_DIM}, layers={NUM_LAYERS}"
    )
    print(f"dtype: float16")
    print(f"Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS}")
    print(
        f"FLOPs/token/layer: {FLOPS_PER_TOKEN_PER_LAYER:.0f}, "
        f"total FLOPs/token: {FLOPS_PER_TOKEN_PER_LAYER * NUM_LAYERS:.0f}"
    )
    print()

    # Build model
    print("Building 32-layer TransformerModel...")
    model = TransformerModel()
    model.set_dtype(mx.float16)

    # Force parameter initialization
    dummy = mx.array([[0]], dtype=mx.int32)
    mx.eval(model(dummy))
    print("Model initialized.\n")

    results = []
    for seq_len in seq_lens:
        print(f"--- seq_len={seq_len} ---")
        r = bench_seq_len(model, seq_len)
        tflops = compute_tflops(seq_len, r["mean"])
        r["tflops"] = tflops

        print(
            f"  mean={r['mean']:>10.1f}us  "
            f"std={r['std']:>8.1f}us  "
            f"p50={r['p50']:>10.1f}us  "
            f"min={r['min']:>10.1f}us  "
            f"max={r['max']:>10.1f}us  "
            f"TFLOPS={tflops:.2f}"
        )
        results.append((seq_len, r))

    # Summary table
    print()
    print(f"| {'seq_len':>7} | {'MLX (us)':>10} | {'TFLOPS':>6} |")
    print(f"|{'-' * 9}|{'-' * 12}|{'-' * 8}|")
    for seq_len, r in results:
        print(f"| {seq_len:>7} | {r['mean']:>10.0f} | {r['tflops']:>6.2f} |")
    print()


if __name__ == "__main__":
    main()
