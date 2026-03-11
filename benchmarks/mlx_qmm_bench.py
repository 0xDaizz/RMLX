#!/usr/bin/env python3
"""MLX QMM Dispatch Reference Benchmark

Benchmarks MLX's mx.quantized_matmul for Q4 across multiple M values,
matching the RMLX qmm_dispatch_bench.rs configurations.

Outputs JSON to stdout for capture as MLX_QMM_REF env var:
    MLX_QMM_REF=$(python3 benchmarks/mlx_qmm_bench.py)

Human-readable table is printed to stderr.

M values: 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024
Dimensions:
  - K=4096, N=4096   (attention projections)
  - K=4096, N=14336  (FFN gate/up)
  - K=7168, N=2048   (DeepSeek expert)
"""

import argparse
import json
import sys
import time
import statistics

import mlx.core as mx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WARMUP = 5
ITERS = 20
BITS = 4
GROUP_SIZE = 32

M_VALUES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DIMS = [
    ("Attn proj", 4096, 4096),
    ("FFN gate/up", 4096, 14336),
    ("DS expert", 7168, 2048),
]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_op(fn, warmup=WARMUP, iters=ITERS):
    """Benchmark fn (zero-arg callable). Returns (mean_us, std_us, p50_us)."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        else:
            mx.eval(result)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # us

    times.sort()
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    p50 = times[n // 2]

    return mean, std, p50


def compute_tflops(M, K, N, elapsed_us):
    """Compute effective TFLOPS."""
    flops = 2.0 * M * K * N
    seconds = elapsed_us / 1e6
    if seconds == 0:
        return 0.0
    return flops / seconds / 1e12


# ---------------------------------------------------------------------------
# Prepare quantized weights
# ---------------------------------------------------------------------------
def make_quantized_weight(K_dim, N_dim, bits=BITS, group_size=GROUP_SIZE):
    """Create quantized weight using mx.quantize().

    Returns (w_quant, scales, biases).
    """
    w_fp = mx.random.normal((N_dim, K_dim)).astype(mx.float16)
    mx.eval(w_fp)
    w_quant, scales, biases = mx.quantize(w_fp, bits=bits, group_size=group_size)
    mx.eval(w_quant, scales, biases)
    return w_quant, scales, biases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX QMM Dispatch Reference Benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    # Print info to stderr
    eprint = lambda *a, **kw: print(*a, file=sys.stderr, **kw)

    eprint("=" * 90)
    eprint("MLX QMM Dispatch Reference Benchmark")
    eprint(f"  mlx version      : {mx.__version__}")
    eprint(f"  device           : {mx.default_device()}")
    eprint(f"  input dtype      : float16")
    eprint(f"  bits             : {BITS}")
    eprint(f"  group_size       : {GROUP_SIZE}")
    eprint(f"  M values         : {M_VALUES}")
    eprint(f"  warmup           : {args.warmup}")
    eprint(f"  iters            : {args.iters}")
    eprint("=" * 90)
    eprint()

    results = {}

    eprint(f"{'M':<6} {'Dims':<14} {'mean(us)':>10} {'std(us)':>10} {'p50(us)':>10} {'TFLOPS':>8}")
    eprint("-" * 62)

    for label, K, N in DIMS:
        eprint(f"\n  {label} (K={K}, N={N})")
        eprint(f"  {'-' * 56}")

        w_quant, scales, biases = make_quantized_weight(K, N)

        for M in M_VALUES:
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)

            def run_qmm(x=x, w=w_quant, s=scales, b=biases):
                return mx.quantized_matmul(
                    x, w, scales=s, biases=b,
                    transpose=True, bits=BITS, group_size=GROUP_SIZE,
                )

            mean, std, p50 = bench_op(run_qmm, args.warmup, args.iters)
            tf = compute_tflops(M, K, N, p50)
            tf_mean = compute_tflops(M, K, N, mean)

            eprint(
                f"  M={M:<5} {K}x{N:<10} {mean:10.1f} {std:10.1f} {p50:10.1f} {tf:8.4f}"
            )

            key = f"{M}_{K}_{N}"
            results[key] = {
                "tflops": round(tf, 4),
                "latency_us": round(p50, 1),
                "tflops_mean": round(tf_mean, 4),
                "latency_mean_us": round(mean, 1),
            }

    eprint()
    eprint("=" * 90)
    eprint("Done.")

    # Output JSON to stdout for capture
    print(json.dumps(results, indent=None, separators=(",", ":")))


if __name__ == "__main__":
    main()
