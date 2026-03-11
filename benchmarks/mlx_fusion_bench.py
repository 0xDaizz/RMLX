#!/usr/bin/env python3
"""MLX RMSNorm + GEMM Fusion Benchmark — MLX reference side.

Measures the same operations as RMLX fusion_bench.rs:
  - RMSNorm + matmul (MLX lazy eval fuses internally)
  - Matmul only (for overhead isolation)

Parameters match Qwen 3.5 MoE expert: M=1/32/512, K=3584, N=3584/2560.

Outputs results to stdout (human-readable) AND writes a JSON file
for consumption by RMLX fusion_bench.rs via MLX_FUSION_REF env var.

Usage:
    python mlx_fusion_bench.py [--warmup N] [--iters N] [--dtype float16]
    python mlx_fusion_bench.py --out /tmp/mlx_fusion_ref.json

Then run RMLX side:
    MLX_FUSION_REF=/tmp/mlx_fusion_ref.json cargo bench -p rmlx-core --bench fusion_bench
"""

import argparse
import json
import time

import mlx.core as mx

WARMUP = 5
ITERS = 20


def bench_op(fn, warmup=WARMUP, iters=ITERS):
    """Benchmark fn. Returns (mean_us, std_us, p50_us)."""
    for _ in range(warmup):
        mx.eval(fn())

    times = []
    for _ in range(iters):
        tic = time.perf_counter_ns()
        mx.eval(fn())
        toc = time.perf_counter_ns()
        times.append((toc - tic) / 1000.0)

    times.sort()
    n = len(times)
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    p50 = times[n // 2]
    return mean, std, p50


def main():
    parser = argparse.ArgumentParser(description="MLX RMSNorm+GEMM fusion benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--out", default="mlx_fusion_ref.json",
                        help="Output JSON file path (default: mlx_fusion_ref.json)")
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.float32

    print("=" * 90)
    print("MLX RMSNorm + GEMM Fusion Benchmark (reference)")
    print(f"  warmup={args.warmup}  iters={args.iters}  dtype={args.dtype}")
    print(f"  mlx version: {mx.__version__}")
    print(f"  default device: {mx.default_device()}")
    print("=" * 90)

    scenarios = [
        ("Qwen3.5-MoE attn proj", 3584, 3584),
        ("Qwen3.5-MoE FFN up/gate", 3584, 2560),
    ]
    m_values = [1, 32, 512]
    eps = 1e-5

    # JSON results: list of {m, k, n, rms_mm_p50, mm_only_p50, ...}
    json_results = []

    for label, k, n in scenarios:
        print(f"\n[{label}: K={k}, N={n}]")

        for m in m_values:
            a = mx.random.normal((m, k)).astype(dtype)
            b = mx.random.normal((k, n)).astype(dtype)
            w = mx.random.normal((k,)).astype(dtype)
            mx.eval(a, b, w)

            # RMSNorm + matmul
            def rms_then_matmul(a=a, b=b, w=w):
                normed = mx.fast.rms_norm(a, w, eps=eps)
                return normed @ b

            mean, std, p50 = bench_op(rms_then_matmul, args.warmup, args.iters)
            flops = 2.0 * m * k * n
            tf = flops / (p50 * 1e-6) / 1e12
            print(
                f"  M={m:5d}  rms+mm    "
                f"p50={p50:8.1f}us  mean={mean:8.1f}us  std={std:6.1f}us  "
                f"TFLOPS={tf:.2f}"
            )

            # Matmul only
            def matmul_only(a=a, b=b):
                return a @ b

            mean2, std2, p502 = bench_op(matmul_only, args.warmup, args.iters)
            tf2 = flops / (p502 * 1e-6) / 1e12
            print(
                f"  M={m:5d}  mm_only   "
                f"p50={p502:8.1f}us  mean={mean2:8.1f}us  std={std2:6.1f}us  "
                f"TFLOPS={tf2:.2f}"
            )

            overhead = p50 - p502
            pct = overhead / p50 * 100 if p50 > 0 else 0
            print(
                f"  M={m:5d}  norm_overhead = {overhead:.1f}us ({pct:.1f}% of total)"
            )

            json_results.append({
                "m": m,
                "k": k,
                "n": n,
                "rms_mm_p50_us": round(p50, 2),
                "rms_mm_mean_us": round(mean, 2),
                "mm_only_p50_us": round(p502, 2),
                "mm_only_mean_us": round(mean2, 2),
            })

    # Write JSON
    out_path = args.out
    with open(out_path, "w") as f:
        json.dump({
            "mlx_version": mx.__version__,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": json_results,
        }, f, indent=2)
    print(f"\nJSON results written to: {out_path}")

    print(f"\n{'=' * 90}")
    print("Done.")
    print(f"\nTo use with RMLX fusion_bench:")
    print(f"  MLX_FUSION_REF={out_path} cargo bench -p rmlx-core --bench fusion_bench")


if __name__ == "__main__":
    main()
