#!/usr/bin/env python3
"""MLX Pure Matmul Microbenchmark — isolated GEMM timing."""
import sys
import time
import mlx.core as mx

WARMUP = 10
BENCH = 30

# Qwen 3.5 MoE layer shapes: (M, K, N, name)
SHAPES = [
    (None, 3584, 4608, "QKV"),
    (None, 3584, 3584, "O_proj"),
    (None, 3584, 5120, "GateUp"),
    (None, 2560, 3584, "Down"),
]

def bench_matmul(M, K, N, name, transpose_b=False):
    a = mx.random.normal((M, K)).astype(mx.float16)
    if transpose_b:
        b = mx.random.normal((N, K)).astype(mx.float16)  # [N, K], will transpose
    else:
        b = mx.random.normal((K, N)).astype(mx.float16)  # [K, N], direct
    mx.eval(a, b)

    # Warmup
    for _ in range(WARMUP):
        if transpose_b:
            c = mx.matmul(a, b.T)
        else:
            c = mx.matmul(a, b)
        mx.eval(c)

    # Bench
    lats = []
    for _ in range(BENCH):
        t0 = time.perf_counter()
        if transpose_b:
            c = mx.matmul(a, b.T)
        else:
            c = mx.matmul(a, b)
        mx.eval(c)
        lats.append((time.perf_counter() - t0) * 1e6)

    lats.sort()
    n = len(lats)
    mean = sum(lats) / n
    std = (sum((x - mean)**2 for x in lats) / n) ** 0.5
    tflops = 2.0 * M * K * N / (mean / 1e6) / 1e12

    return {
        "name": name,
        "M": M,
        "K": K,
        "N": N,
        "transpose_b": transpose_b,
        "mean": mean,
        "std": std,
        "min": lats[0],
        "p50": lats[n // 2],
        "tflops": tflops,
    }

def main():
    m_values = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [128, 256, 512, 1024]

    print("# MLX Pure Matmul Microbenchmark")
    print(f"Warmup: {WARMUP}, Bench: {BENCH}, dtype: float16\n")

    for M in m_values:
        print(f"## M={M}")
        print(f"| Op | M×K×N | Mode | Mean (us) | Std | Min | P50 | TFLOPS |")
        print(f"|------|-------|------|-----------|-----|-----|-----|--------|")

        for _, K, N, name in SHAPES:
            # Direct (no transpose)
            r = bench_matmul(M, K, N, name, transpose_b=False)
            print(f"| {name} | {M}×{K}×{N} | direct | {r['mean']:.1f} | {r['std']:.1f} | {r['min']:.1f} | {r['p50']:.1f} | {r['tflops']:.2f} |")

            # Transposed B
            r_t = bench_matmul(M, K, N, name, transpose_b=True)
            print(f"| {name} | {M}×{K}×{N} | trans_b | {r_t['mean']:.1f} | {r_t['std']:.1f} | {r_t['min']:.1f} | {r_t['p50']:.1f} | {r_t['tflops']:.2f} |")

        print()

if __name__ == "__main__":
    main()
