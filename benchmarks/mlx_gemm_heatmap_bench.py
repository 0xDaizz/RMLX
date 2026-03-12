#!/usr/bin/env python3
"""MLX f16 GEMM Heatmap Benchmark — pipelined 32-layer pattern

Benchmarks MLX f16 matmul across all M values and shapes using a pipelined
pattern that simulates 32-layer transformer dispatch (32 matmuls → single eval).

Usage:
    python benchmarks/mlx_gemm_heatmap_bench.py [--warmup N] [--iters N] [--layers N]
"""

import argparse
import time
import mlx.core as mx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WARMUP = 5
ITERS = 20
N_LAYERS = 32

# Test grid — largest M first for thermal fairness
M_VALUES = [512, 256, 128, 64, 32, 16, 8, 4, 2]
SHAPES = [
    (3584, 3584),
    (4096, 4096),
    (14336, 4096),
    (4096, 14336),
]


# ---------------------------------------------------------------------------
# Pipelined benchmark (matches RMLX 32-CB pattern)
# ---------------------------------------------------------------------------
def pipelined_bench(a, b, n_layers=N_LAYERS, warmup=WARMUP, iters=ITERS):
    """Run n_layers back-to-back matmuls then eval once.

    Returns p50 per-dispatch latency in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        results = []
        for _ in range(n_layers):
            results.append(a @ b)
        mx.eval(*results)

    times = []
    for _ in range(iters):
        results = []
        tic = time.perf_counter_ns()
        for _ in range(n_layers):
            results.append(a @ b)
        mx.eval(*results)
        toc = time.perf_counter_ns()
        elapsed_us = (toc - tic) / 1000.0
        per_dispatch_us = elapsed_us / n_layers
        times.append(per_dispatch_us)

    times.sort()
    return times[len(times) // 2]  # p50


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX f16 GEMM heatmap benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    args = parser.parse_args()

    print("=" * 80)
    print("MLX f16 GEMM Heatmap Benchmark (pipelined, 32-layer)")
    print(f"  warmup={args.warmup}  iters={args.iters}  layers={args.layers}")
    print(f"  mlx version: {mx.__version__}")
    print(f"  default device: {mx.default_device()}")
    print(f"  M values (thermal order): {M_VALUES}")
    print("=" * 80)

    # Collect results: results[m][(n,k)] = (latency_us, tflops)
    results = {}

    # --- Detailed table ---
    print()
    print("| M | N | K | MLX Latency (us) | MLX TFLOPS |")
    print("|---|---|---|------------------|------------|")

    for m in M_VALUES:
        results[m] = {}
        for n, k in SHAPES:
            # Create f16 matrices
            a = mx.random.normal((m, k)).astype(mx.float16)
            b = mx.random.normal((k, n)).astype(mx.float16)
            mx.eval(a, b)

            lat_us = pipelined_bench(
                a, b,
                n_layers=args.layers,
                warmup=args.warmup,
                iters=args.iters,
            )
            tflops = 2.0 * m * n * k / (lat_us * 1e6) / 1e12

            results[m][(n, k)] = (lat_us, tflops)
            print(f"| {m:>3} | {n:>5} | {k:>5} | {lat_us:>16.1f} | {tflops:>10.2f} |")

    # --- Heatmap: latency ---
    print()
    print("### Latency Heatmap (us, p50 per dispatch)")
    print()
    header = "| M \\ Shape |"
    sep = "|-----------|"
    for n, k in SHAPES:
        header += f" {n}x{k} |"
        sep += "------------|"
    print(header)
    print(sep)

    for m in M_VALUES:
        row = f"| {m:>9} |"
        for n, k in SHAPES:
            lat_us, _ = results[m][(n, k)]
            row += f" {lat_us:>9.1f} |"
        print(row)

    # --- Heatmap: TFLOPS ---
    print()
    print("### TFLOPS Heatmap")
    print()
    header = "| M \\ Shape |"
    sep = "|-----------|"
    for n, k in SHAPES:
        header += f" {n}x{k} |"
        sep += "------------|"
    print(header)
    print(sep)

    for m in M_VALUES:
        row = f"| {m:>9} |"
        for n, k in SHAPES:
            _, tflops = results[m][(n, k)]
            row += f" {tflops:>9.2f} |"
        print(row)

    print()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
