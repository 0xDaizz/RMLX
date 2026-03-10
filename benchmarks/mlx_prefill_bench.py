#!/usr/bin/env python3
"""MLX Prefill Benchmark (Reference) — Kernel-Level + Full Model

Measures the exact same operations and shapes as the RMLX prefill benchmark
for fair 1:1 comparison.

Sections:
  1-A: SDPA (scaled dot-product attention)
  1-C: GEMM align_K (K=4096 vs K=4097)
  2-A: Split-K GEMM (low-M prefill shapes)
  2-B: QMV (M=1, quantized matmul vector)
  2-C: BatchQMV (M=17/24/32, quantized matmul batch)
  Full: Llama-3 8B 32-layer prefill with random weights

Usage:
    python mlx_prefill_bench.py [--warmup N] [--iters N] [--section all|kernel|full]
"""

import argparse
import gc
import platform
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ===========================================================================
# Timing helpers
# ===========================================================================

def bench_fn(fn, warmup=3, iters=20):
    """Run fn() with warmup, return median latency in seconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times)


def fmt_us(seconds):
    """Format seconds as microseconds string."""
    us = seconds * 1e6
    if us >= 1000:
        return f"{us / 1000:.2f}ms"
    return f"{us:.1f}us"


def fmt_tflops(flops, seconds):
    """Compute and format TFLOPS."""
    if seconds <= 0:
        return "---"
    t = flops / seconds / 1e12
    return f"{t:.2f} TFLOPS"


# ===========================================================================
# Section 1-A: SDPA
# ===========================================================================

def bench_sdpa(seq_len, kv_len, num_heads=32, num_kv_heads=8, head_dim=128,
               warmup=3, iters=20):
    q = mx.random.normal((1, num_heads, seq_len, head_dim)).astype(mx.float16)
    k = mx.random.normal((1, num_kv_heads, kv_len, head_dim)).astype(mx.float16)
    v = mx.random.normal((1, num_kv_heads, kv_len, head_dim)).astype(mx.float16)
    scale = 1.0 / (head_dim ** 0.5)

    # Causal mask for prefill fairness
    if seq_len == kv_len:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(mx.float16)
    else:
        mask = None

    mx.eval(q, k, v)
    if mask is not None:
        mx.eval(mask)

    def run():
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        mx.eval(out)

    return bench_fn(run, warmup=warmup, iters=iters)


def run_sdpa_section(warmup, iters):
    print("--- 1-A: SDPA ---")
    cases = [128, 512, 2048]
    for sl in cases:
        lat = bench_sdpa(sl, sl, warmup=warmup, iters=iters)
        print(f"seq={sl:<5d} kv={sl:<5d}  latency: {fmt_us(lat)}")
    print()


# ===========================================================================
# Section 1-C: GEMM align_K
# ===========================================================================

def bench_gemm(m, n, k, warmup=3, iters=20):
    a = mx.random.normal((m, k)).astype(mx.float16)
    b = mx.random.normal((k, n)).astype(mx.float16)
    mx.eval(a, b)

    def run():
        out = a @ b
        mx.eval(out)

    lat = bench_fn(run, warmup=warmup, iters=iters)
    flops = 2.0 * m * n * k
    return lat, flops


def run_gemm_section(warmup, iters):
    print("--- 1-C: GEMM ---")
    cases = [
        (512, 4096, 4096),
        (512, 4096, 4097),
    ]
    for m, n, k in cases:
        lat, flops = bench_gemm(m, n, k, warmup=warmup, iters=iters)
        print(f"M={m} N={n} K={k}: {fmt_tflops(flops, lat)}")
    print()


# ===========================================================================
# Section 2-A: Split-K GEMM (low-M)
# ===========================================================================

def run_splitk_section(warmup, iters):
    print("--- 2-A: Split-K GEMM ---")
    m_values = [32, 64, 128]
    n, k = 4096, 14336
    for m in m_values:
        lat, flops = bench_gemm(m, n, k, warmup=warmup, iters=iters)
        print(f"M={m:<4d} N={n} K={k}: {fmt_tflops(flops, lat)}")
    print()


# ===========================================================================
# Section 2-B: QMV (M=1, quantized)
# ===========================================================================

def bench_qmv(m, n, k, bits=4, group_size=64, warmup=3, iters=20):
    x = mx.random.normal((m, k)).astype(mx.float16)
    w = mx.random.normal((n, k)).astype(mx.float16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(x, w_q, scales, biases)

    def run():
        out = mx.quantized_matmul(
            x, w_q, scales, biases,
            transpose=True, group_size=group_size, bits=bits,
        )
        mx.eval(out)

    lat = bench_fn(run, warmup=warmup, iters=iters)
    flops = 2.0 * m * n * k
    return lat, flops


def run_qmv_section(warmup, iters):
    print("--- 2-B: QMV (M=1, Q4) ---")
    cases = [
        (1, 2048,  14336),
        (1, 4096,  14336),
        (1, 2048,  4096),
    ]
    for m, n, k in cases:
        lat, flops = bench_qmv(m, n, k, warmup=warmup, iters=iters)
        print(f"M={m} K={k:<6d} N={n:<5d}: {fmt_tflops(flops, lat)}  latency: {fmt_us(lat)}")
    print()


# ===========================================================================
# Section 2-C: BatchQMV (M=17/24/32, Q4)
# ===========================================================================

def run_batch_qmv_section(warmup, iters):
    print("--- 2-C: BatchQMV (Q4) ---")
    m_values = [17, 24, 32]
    nk_cases = [
        (4096, 14336),
        (14336, 4096),
    ]
    for n, k in nk_cases:
        for m in m_values:
            lat, flops = bench_qmv(m, n, k, warmup=warmup, iters=iters)
            print(f"M={m:<3d} K={k:<6d} N={n:<5d}: {fmt_tflops(flops, lat)}  latency: {fmt_us(lat)}")
    print()


# ===========================================================================
# Section: Full Prefill (Llama-3 8B, 32 layers, f16)
# ===========================================================================

class LlamaConfig:
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8
    intermediate_size = 14336
    num_layers = 32
    vocab_size = 128256
    rms_norm_eps = 1e-5
    rope_theta = 500000.0


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.wq = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, mask=None):
        h = x + self.attention(self.attention_norm(x), mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, tokens):
        x = self.embed(tokens)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(mx.float16)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)


def run_full_prefill_section(warmup, iters):
    print("--- Full Prefill (Llama-3 8B, 32 layers, f16) ---")

    config = LlamaConfig()
    model = LlamaModel(config)

    # Cast all weights to float16 and materialize
    model.update(model.apply(lambda p: p.astype(mx.float16)))
    mx.eval(model.parameters())

    # Count parameters
    num_params = sum(p.size for p in model.parameters().values()
                     if hasattr(p, 'size'))
    # Flatten nested params for counting
    def count_params(tree):
        total = 0
        if isinstance(tree, mx.array):
            return tree.size
        elif isinstance(tree, dict):
            for v in tree.values():
                total += count_params(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                total += count_params(v)
        return total

    num_params = count_params(model.parameters())
    mem_gb = num_params * 2 / 1e9  # f16 = 2 bytes
    print(f"  Model params: {num_params / 1e9:.2f}B ({mem_gb:.2f} GB in f16)")
    print()

    seq_lens = [128, 512, 2048]

    for seq_len in seq_lens:
        tokens = mx.array([[1] * seq_len])
        mx.eval(tokens)

        # Warmup
        for _ in range(warmup):
            out = model(tokens)
            mx.eval(out)

        # Measure
        times = []
        for _ in range(iters):
            mx.synchronize()
            t0 = time.perf_counter()
            out = model(tokens)
            mx.eval(out)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median = np.median(times)
        tok_per_sec = seq_len / median
        per_layer = median / config.num_layers * 1000  # ms
        print(f"seq={seq_len:<5d}: {median * 1000:.1f}ms, "
              f"{tok_per_sec:.0f} tok/s, "
              f"{per_layer:.2f}ms/layer")

        # Force cleanup between sizes
        gc.collect()

    print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLX Prefill Benchmark (Reference)"
    )
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (default 3)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Benchmark iterations (default 20)")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "kernel", "full"],
                        help="Which sections to run (default: all)")
    args = parser.parse_args()

    warmup = args.warmup
    iters = args.iters

    print("=" * 60)
    print("  MLX Prefill Benchmark (Reference)")
    print("=" * 60)
    print(f"  MLX version : {mx.__version__}")
    print(f"  Device      : {mx.default_device()}")
    print(f"  Platform    : {platform.platform()}")
    print(f"  Python      : {platform.python_version()}")
    print(f"  Warmup      : {warmup}")
    print(f"  Iters       : {iters}")
    print("=" * 60)
    print()

    if args.section in ("all", "kernel"):
        run_sdpa_section(warmup, iters)
        run_gemm_section(warmup, iters)
        run_splitk_section(warmup, iters)
        run_qmv_section(warmup, iters)
        run_batch_qmv_section(warmup, iters)

    if args.section in ("all", "full"):
        run_full_prefill_section(warmup, iters)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
