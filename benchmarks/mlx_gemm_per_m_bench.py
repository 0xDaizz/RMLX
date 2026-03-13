#!/usr/bin/env python3
"""MLX GEMM per-M benchmark for comparison with RMLX nax_lowm_bench.

Tests f16 matmul A[M,K] @ B[K,N] across M values.
N=3584, K=3584, 20 iters, p50.

Usage: python mlx_gemm_per_m_bench.py
"""

import time
import mlx.core as mx

WARMUP = 5
ITERS = 20
N = 3584
K = 3584
M_VALUES = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]


def bench_gemm(m, n, k):
    a = mx.random.normal((m, k)).astype(mx.float16)
    b = mx.random.normal((k, n)).astype(mx.float16)
    mx.eval(a, b)

    # Warmup
    for _ in range(WARMUP):
        c = a @ b
        mx.eval(c)

    times = []
    for _ in range(ITERS):
        tic = time.perf_counter_ns()
        c = a @ b
        mx.eval(c)
        toc = time.perf_counter_ns()
        times.append((toc - tic) / 1000.0)

    times.sort()
    p50 = times[len(times) // 2]
    return p50


def tflops(m, n, k, latency_us):
    flops = 2.0 * m * n * k
    return flops / (latency_us * 1e-6) / 1e12


def main():
    print(f"=== MLX GEMM Per-M Benchmark ===")
    print(f"N={N}, K={K}, dtype=float16, Warmup={WARMUP}, Bench={ITERS} iters")
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print()

    results = []
    # Run largest M first to avoid thermal throttle
    for m in reversed(M_VALUES):
        p50 = bench_gemm(m, N, K)
        tf = tflops(m, N, K, p50)
        print(f"  M={m:>4d}: {p50:>8.1f} us, {tf:>6.2f} TFLOPS")
        results.append((m, p50, tf))

    # Sort ascending for display
    results.sort(key=lambda x: x[0])

    print()
    print("| M | MLX Time (us) | MLX TFLOPS |")
    print("|-----|---------------|------------|")
    for m, p50, tf in results:
        print(f"| {m:>3d} | {p50:>13.1f} | {tf:>10.2f} |")


if __name__ == "__main__":
    main()
