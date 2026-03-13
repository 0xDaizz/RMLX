#!/usr/bin/env python3
"""MLX GEMM benchmark for comparison with RMLX gemm_fair_bench auto-dispatch."""
import time
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: mlx not installed. pip install mlx")
    exit(1)

NK_SHAPES = [
    (3584, 3584, "attn_proj"),
    (4096, 4096, "standard"),
    (14336, 4096, "ffn_up"),
    (4096, 14336, "ffn_down"),
]
M_VALUES = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512]
PIPELINE_N = 32
WARMUP = 5
BENCH = 20

def bench_matmul(m, n, k):
    a = mx.random.normal(shape=(m, k)).astype(mx.float16)
    b = mx.random.normal(shape=(k, n)).astype(mx.float16)
    mx.eval(a, b)

    # warmup
    for _ in range(WARMUP):
        outs = []
        for _ in range(PIPELINE_N):
            outs.append(a @ b)
        mx.eval(*outs)

    # bench
    times = []
    for _ in range(BENCH):
        outs = []
        start = time.perf_counter()
        for _ in range(PIPELINE_N):
            outs.append(a @ b)
        mx.eval(*outs)
        elapsed = time.perf_counter() - start
        times.append(elapsed / PIPELINE_N)

    times.sort()
    p50 = times[len(times) // 2]
    return p50 * 1e6  # us

def tflops(m, n, k, us):
    return 2.0 * m * n * k / (us * 1e-6) / 1e12

def main():
    print(f"=== MLX GEMM Benchmark (pipelined {PIPELINE_N}x, f16) ===")
    print(f"Device: {mx.default_device()}")
    print(f"Warmup: {WARMUP}, Bench: {BENCH}\n")

    all_results = {}

    for n, k, name in NK_SHAPES:
        print(f"--- {name} (N={n}, K={k}) ---")
        print(f"| {'M':>5} | {'latency (us)':>12} | {'TFLOPS':>8} |")
        print(f"|-------|--------------|----------|")

        for m in reversed(M_VALUES):
            lat = bench_matmul(m, n, k)
            t = tflops(m, n, k, lat)
            print(f"| {m:>5} | {lat:>12.1f} | {t:>8.2f} |")
            all_results[(m, n, k)] = (lat, t)
        print()

    # Summary table
    print(f"\n=== MLX Summary (p50 latency, TFLOPS) ===")
    header = f"| {'M':>5} |"
    sep = f"|-------|"
    for n, k, name in NK_SHAPES:
        header += f" {name:>14} |"
        sep += f"----------------|"
    print(header)
    print(sep)

    for m in M_VALUES:
        row = f"| {m:>5} |"
        for n, k, name in NK_SHAPES:
            if (m, n, k) in all_results:
                lat, t = all_results[(m, n, k)]
                row += f" {lat:>6.0f}/{t:>5.2f}T |"
            else:
                row += f" {'N/A':>14} |"
        print(row)
    print()

if __name__ == "__main__":
    main()
