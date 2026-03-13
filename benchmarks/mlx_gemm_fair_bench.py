#!/usr/bin/env python3
"""MLX GEMM fair benchmark — matches RMLX gemm_fair_bench conditions.

Tests f16 matmul A[M,K] @ B[K,N] across M values.
N=3584, K=3584, 20 iters, p50.

Also tests pipelined mode: 32 matmuls in a single mx.eval() call
to match RMLX's 32-dispatch-per-CB pattern.

Usage: python mlx_gemm_fair_bench.py
"""

import time
import mlx.core as mx

WARMUP = 5
ITERS = 20
PIPELINE_N = 32
N = 3584
K = 3584
M_VALUES = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]


def bench_single(m, n, k):
    """Single matmul per mx.eval() — standard benchmark."""
    a = mx.random.normal((m, k)).astype(mx.float16)
    b = mx.random.normal((k, n)).astype(mx.float16)
    mx.eval(a, b)

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
    return times[len(times) // 2]


def bench_pipelined(m, n, k, pipeline_n=PIPELINE_N):
    """Pipeline_n matmuls in a single mx.eval() — amortized overhead."""
    a = mx.random.normal((m, k)).astype(mx.float16)
    # Use different B matrices to prevent MLX from caching/deduplicating
    bs = [mx.random.normal((k, n)).astype(mx.float16) for _ in range(pipeline_n)]
    mx.eval(a, *bs)

    for _ in range(WARMUP):
        results = [a @ b for b in bs]
        mx.eval(*results)

    times = []
    for _ in range(ITERS):
        tic = time.perf_counter_ns()
        results = [a @ b for b in bs]
        mx.eval(*results)
        toc = time.perf_counter_ns()
        times.append((toc - tic) / 1000.0)

    times.sort()
    total_p50 = times[len(times) // 2]
    return total_p50 / pipeline_n  # amortized per matmul


def tflops(m, n, k, latency_us):
    flops = 2.0 * m * n * k
    return flops / (latency_us * 1e-6) / 1e12


def main():
    print(f"=== MLX GEMM Fair Benchmark ===")
    print(f"N={N}, K={K}, dtype=float16, Warmup={WARMUP}, Bench={ITERS}")
    print(f"Pipeline: {PIPELINE_N} matmuls per mx.eval()")
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print()

    results = []
    # Run largest M first to avoid thermal throttling
    for m in reversed(M_VALUES):
        single = bench_single(m, N, K)
        pipelined = bench_pipelined(m, N, K)
        single_t = tflops(m, N, K, single)
        pipe_t = tflops(m, N, K, pipelined)
        speedup = single / pipelined if pipelined > 0 else 0
        print(
            f"  M={m:>4d}: single={single:>8.1f}us ({single_t:>6.2f}T)"
            f"  pipe={pipelined:>8.1f}us ({pipe_t:>6.2f}T)"
            f"  speedup={speedup:.2f}x"
        )
        results.append((m, single, single_t, pipelined, pipe_t, speedup))

    results.sort(key=lambda x: x[0])

    print()
    print(
        f"| {'M':>5} | {'Single (us)':>11} | {'Single T':>8} "
        f"| {'Pipe (us)':>9} | {'Pipe T':>8} | {'Speedup':>7} |"
    )
    print(
        f"|{'---':->7}|{'---':->13}|{'---':->10}"
        f"|{'---':->11}|{'---':->10}|{'---':->9}|"
    )
    for m, single, single_t, pipelined, pipe_t, speedup in results:
        print(
            f"| {m:>5} | {single:>11.1f} | {single_t:>8.2f} "
            f"| {pipelined:>9.1f} | {pipe_t:>8.2f} | {speedup:>6.2f}x |"
        )


if __name__ == "__main__":
    main()
