#!/usr/bin/env python3
"""RMLX vs MLX Comparison Benchmark — MLX side

Benchmarks MLX operations with identical parameters to RMLX ops_bench
for fair, apples-to-apples comparison.

Usage:
    python mlx_comparison_bench.py [--warmup N] [--iters N]
"""

import argparse
import time
import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WARMUP = 5
ITERS = 100


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_op(name, fn, warmup=WARMUP, iters=ITERS):
    """Benchmark *fn* (a zero-arg callable that returns an mx array).

    Returns mean_us so callers can accumulate totals.
    """
    # Warmup
    for _ in range(warmup):
        mx.eval(fn())

    times = []
    for _ in range(iters):
        tic = time.perf_counter_ns()
        mx.eval(fn())
        toc = time.perf_counter_ns()
        times.append((toc - tic) / 1000.0)  # nanoseconds -> microseconds

    times.sort()
    n = len(times)
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    p50 = times[n // 2]
    p95 = times[int(n * 0.95)]

    print(f"{name:40s}  mean={mean:10.1f}us  std={std:8.1f}us  p50={p50:10.1f}us  p95={p95:10.1f}us")
    return mean


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_matmul(warmup, iters):
    print("\n===== matmul =====")
    a = mx.random.normal((1024, 1024))
    b = mx.random.normal((1024, 1024))
    mx.eval(a, b)
    bench_op("matmul [1024x1024] x [1024x1024]", lambda: a @ b, warmup, iters)

    a4 = mx.random.normal((4096, 4096))
    b4 = mx.random.normal((4096, 4096))
    mx.eval(a4, b4)
    bench_op("matmul [4096x4096] x [4096x4096]", lambda: a4 @ b4, warmup, iters)


def bench_softmax(warmup, iters):
    print("\n===== softmax =====")
    x = mx.random.normal((8, 1024, 4096))
    mx.eval(x)
    bench_op("softmax [8,1024,4096] axis=-1", lambda: mx.softmax(x, axis=-1), warmup, iters)


def bench_rms_norm(warmup, iters):
    print("\n===== rms_norm =====")
    x = mx.random.normal((8, 1024, 4096))
    w = mx.random.normal((4096,))
    mx.eval(x, w)
    bench_op("rms_norm [8,1024,4096]", lambda: mx.fast.rms_norm(x, w, eps=1e-5), warmup, iters)


def bench_rope(warmup, iters):
    print("\n===== rope =====")
    # Vector: single-token inference shape [B=1, n_heads=32, seq=1, head_dim=128]
    v = mx.random.normal((1, 32, 1, 128))
    mx.eval(v)
    bench_op("rope vector [1,32,1,128]", lambda: mx.fast.rope(v, dims=128, traditional=False, base=10000.0, scale=1.0, offset=0), warmup, iters)

    # Matrix: prefill shape [B=1, n_heads=32, seq=1024, head_dim=128]
    m = mx.random.normal((1, 32, 1024, 128))
    mx.eval(m)
    bench_op("rope matrix [1,32,1024,128]", lambda: mx.fast.rope(m, dims=128, traditional=False, base=10000.0, scale=1.0, offset=0), warmup, iters)


def bench_sdpa(warmup, iters):
    print("\n===== scaled_dot_product_attention =====")
    # B=1, n_qh=32, n_kvh=8, qsl=ksl=1024, head_dim=128
    # GQA: queries have 32 heads, KV have 8 heads (4x repeat)
    q = mx.random.normal((1, 32, 1024, 128))
    k = mx.random.normal((1, 8, 1024, 128))
    v = mx.random.normal((1, 8, 1024, 128))
    mx.eval(q, k, v)
    scale = 128 ** -0.5
    bench_op(
        "sdpa B=1 qsl=1024 ksl=1024 h=32/8 d=128",
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
        warmup, iters,
    )


def bench_gelu(warmup, iters):
    print("\n===== gelu =====")
    x = mx.random.normal((8, 1024, 4096))
    mx.eval(x)
    bench_op("gelu [8,1024,4096]", lambda: nn.gelu(x), warmup, iters)


def bench_silu(warmup, iters):
    print("\n===== silu =====")
    x = mx.random.normal((8, 1024, 4096))
    mx.eval(x)
    # SiLU = x * sigmoid(x), matches RMLX implementation
    bench_op("silu [8,1024,4096]", lambda: x * mx.sigmoid(x), warmup, iters)


def bench_layer_norm(warmup, iters):
    print("\n===== layer_norm =====")
    x = mx.random.normal((8, 1024, 4096))
    w = mx.random.normal((4096,))
    b = mx.random.normal((4096,))
    mx.eval(x, w, b)
    bench_op("layer_norm [8,1024,4096]", lambda: mx.fast.layer_norm(x, w, b, eps=1e-5), warmup, iters)


def bench_add(warmup, iters):
    print("\n===== add =====")
    a = mx.random.normal((32, 1024, 1024))
    b = mx.random.normal((32, 1024, 1024))
    mx.eval(a, b)
    bench_op("add [32,1024,1024]", lambda: mx.add(a, b), warmup, iters)


def bench_neg(warmup, iters):
    print("\n===== neg =====")
    x = mx.random.normal((10000, 1000))
    mx.eval(x)
    bench_op("neg [10000,1000]", lambda: -x, warmup, iters)


def bench_sum(warmup, iters):
    print("\n===== sum =====")
    x = mx.random.normal((32, 1024, 1024))
    mx.eval(x)
    bench_op("sum [32,1024,1024] axis=0", lambda: mx.sum(x, axis=0), warmup, iters)


def bench_sort(warmup, iters):
    print("\n===== sort =====")
    x = mx.random.normal((1024, 1024))
    mx.eval(x)
    bench_op("sort [1024,1024]", lambda: mx.sort(x), warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_BENCHMARKS = [
    bench_matmul,
    bench_softmax,
    bench_rms_norm,
    bench_rope,
    bench_sdpa,
    bench_gelu,
    bench_silu,
    bench_layer_norm,
    bench_add,
    bench_neg,
    bench_sum,
    bench_sort,
]


def main():
    parser = argparse.ArgumentParser(description="MLX comparison benchmark for RMLX")
    parser.add_argument("--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default {WARMUP})")
    parser.add_argument("--iters", type=int, default=ITERS, help=f"Benchmark iterations (default {ITERS})")
    args = parser.parse_args()

    print("=" * 80)
    print("MLX Comparison Benchmark")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"  mlx version: {mx.__version__}")
    print(f"  default device: {mx.default_device()}")
    print("=" * 80)

    for bench_fn in ALL_BENCHMARKS:
        bench_fn(args.warmup, args.iters)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
