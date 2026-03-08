#!/usr/bin/env python3
"""MLX Quantized MatMul Benchmark

Benchmarks MLX's quantized matmul operations:
  1. QMV (M=1, decode): K=4096, N={4096, 14336} for Q4 and Q8
  2. QMM (batched, M={32, 128, 256}): K=4096, N=4096 for Q4 and Q8

Uses mx.quantize() + mx.quantized_matmul() and nn.QuantizedLinear.

Usage:
    python mlx_quantized_bench.py [--warmup N] [--iters N]
"""

import argparse
import time
import statistics

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WARMUP = 5
ITERS = 20

# QMV dimension sets: (K, N) — matches RMLX quantized_bench.rs
QMV_DIMS = [
    (4096, 14336),  # Mixtral gate/up
    (4096, 4096),   # attention projections
    (4096, 2048),   # per-expert intermediate
    (5120, 1536),   # DeepSeek-V2 expert FFN
]
QMM_K = 4096
QMM_N = 4096
QMM_M_VALUES = [32, 128, 256]
BITS_VALUES = [4, 8]
GROUP_SIZE = 32  # match RMLX bench (group_size=32)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_op(name, fn, warmup=WARMUP, iters=ITERS):
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
    """Compute effective TFLOPS (treating quantized matmul as equivalent FP FLOPs)."""
    flops = 2.0 * M * K * N
    seconds = elapsed_us / 1e6
    if seconds == 0:
        return 0.0
    return flops / seconds / 1e12


def compute_bandwidth_gbps(M, K, N, bits, group_size, elapsed_us):
    """Compute memory bandwidth in GB/s.

    Weight bytes: (K * N * bits / 8) + scales/biases.
    Input bytes: M * K * 2 (float16).
    Output bytes: M * N * 2 (float16).
    """
    weight_bytes = (K * N * bits) / 8
    num_groups = (K // group_size) * N
    scale_bias_bytes = num_groups * 4  # float16 scale + float16 bias per group
    input_bytes = M * K * 2
    output_bytes = M * N * 2
    total_bytes = weight_bytes + scale_bias_bytes + input_bytes + output_bytes
    seconds = elapsed_us / 1e6
    if seconds == 0:
        return 0.0
    return total_bytes / seconds / 1e9


# ---------------------------------------------------------------------------
# Prepare quantized weights
# ---------------------------------------------------------------------------
def make_quantized_weight(K_dim, N_dim, bits, group_size=GROUP_SIZE):
    """Create a quantized weight matrix using mx.quantize().

    Returns (w_quant, scales, biases) for use with mx.quantized_matmul().
    """
    # Create random float16 weight and quantize it
    w_fp = mx.random.normal((N_dim, K_dim)).astype(mx.float16)
    mx.eval(w_fp)
    w_quant, scales, biases = mx.quantize(w_fp, bits=bits, group_size=group_size)
    mx.eval(w_quant, scales, biases)
    return w_quant, scales, biases


def make_quantized_linear(K_dim, N_dim, bits, group_size=GROUP_SIZE):
    """Create an nn.QuantizedLinear layer."""
    layer = nn.QuantizedLinear(K_dim, N_dim, bias=False, bits=bits, group_size=group_size)
    # Initialize with random weights
    w_fp = mx.random.normal((N_dim, K_dim)).astype(mx.float16)
    mx.eval(w_fp)
    w_quant, scales, biases = mx.quantize(w_fp, bits=bits, group_size=group_size)
    layer.weight = w_quant
    layer.scales = scales
    layer.biases = biases
    mx.eval(layer.weight, layer.scales, layer.biases)
    return layer


# ---------------------------------------------------------------------------
# QMV benchmarks (M=1, decode)
# ---------------------------------------------------------------------------
def bench_qmv(warmup, iters):
    """Benchmark quantized matrix-vector multiply (M=1)."""
    print(f"\n{'='*90}")
    print(f"  QMV (M=1, Decode Mode) — mx.quantized_matmul")
    print(f"{'='*90}")
    header = f"{'Config':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s} {'TFLOPS':>8s} {'GB/s':>8s}"
    print(header)
    print("-" * len(header))

    M = 1
    for bits in BITS_VALUES:
        for K, N in QMV_DIMS:
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)

            # --- mx.quantized_matmul ---
            w_quant, scales, biases = make_quantized_weight(K, N, bits)

            def run_qmatmul(x=x, w=w_quant, s=scales, b=biases, _bits=bits):
                return mx.quantized_matmul(
                    x, w, scales=s, biases=b,
                    transpose=True, bits=_bits, group_size=GROUP_SIZE
                )

            mean, std, p50 = bench_op(
                f"qmv Q{bits} M={M} K={K} N={N}", run_qmatmul, warmup, iters
            )
            tflops = compute_tflops(M, K, N, mean)
            gbps = compute_bandwidth_gbps(M, K, N, bits, GROUP_SIZE, mean)
            label = f"quantized_matmul Q{bits} K={K} N={N}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {gbps:8.1f}")

            # --- nn.QuantizedLinear ---
            layer = make_quantized_linear(K, N, bits)

            def run_qlayer(x=x, l=layer):
                return l(x)

            mean, std, p50 = bench_op(
                f"QuantizedLinear Q{bits} M={M} K={K} N={N}", run_qlayer, warmup, iters
            )
            tflops = compute_tflops(M, K, N, mean)
            gbps = compute_bandwidth_gbps(M, K, N, bits, GROUP_SIZE, mean)
            label = f"QuantizedLinear Q{bits} K={K} N={N}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {gbps:8.1f}")

    # --- FP16 baseline for comparison ---
    print()
    for K, N in QMV_DIMS:
        x = mx.random.normal((1, K)).astype(mx.float16)
        w = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(x, w)

        def run_fp16(x=x, w=w):
            return x @ w

        mean, std, p50 = bench_op(f"fp16 M=1 K={K} N={N}", run_fp16, warmup, iters)
        tflops = compute_tflops(1, K, N, mean)
        fp16_gbps = (1 * K * 2 + K * N * 2 + 1 * N * 2) / (mean / 1e6) / 1e9 if mean > 0 else 0
        label = f"fp16 baseline K={K} N={N}"
        print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {fp16_gbps:8.1f}")


# ---------------------------------------------------------------------------
# QMM benchmarks (batched, M>1)
# ---------------------------------------------------------------------------
def bench_qmm(warmup, iters):
    """Benchmark quantized matrix-matrix multiply (M>1)."""
    print(f"\n{'='*90}")
    print(f"  QMM (Batched) — mx.quantized_matmul, K={QMM_K}, N={QMM_N}")
    print(f"{'='*90}")
    header = f"{'Config':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s} {'TFLOPS':>8s} {'GB/s':>8s}"
    print(header)
    print("-" * len(header))

    K = QMM_K
    N = QMM_N

    for bits in BITS_VALUES:
        for M in QMM_M_VALUES:
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)

            # --- mx.quantized_matmul ---
            w_quant, scales, biases = make_quantized_weight(K, N, bits)

            def run_qmatmul(x=x, w=w_quant, s=scales, b=biases, _bits=bits):
                return mx.quantized_matmul(
                    x, w, scales=s, biases=b,
                    transpose=True, bits=_bits, group_size=GROUP_SIZE
                )

            mean, std, p50 = bench_op(
                f"qmm Q{bits} M={M} K={K} N={N}", run_qmatmul, warmup, iters
            )
            tflops = compute_tflops(M, K, N, mean)
            gbps = compute_bandwidth_gbps(M, K, N, bits, GROUP_SIZE, mean)
            label = f"quantized_matmul Q{bits} M={M}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {gbps:8.1f}")

            # --- nn.QuantizedLinear ---
            layer = make_quantized_linear(K, N, bits)

            def run_qlayer(x=x, l=layer):
                return l(x)

            mean, std, p50 = bench_op(
                f"QuantizedLinear Q{bits} M={M} K={K} N={N}", run_qlayer, warmup, iters
            )
            tflops = compute_tflops(M, K, N, mean)
            gbps = compute_bandwidth_gbps(M, K, N, bits, GROUP_SIZE, mean)
            label = f"QuantizedLinear Q{bits} M={M}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {gbps:8.1f}")

        print()

    # --- FP16 baseline ---
    print("  --- FP16 Baseline ---")
    for M in QMM_M_VALUES:
        x = mx.random.normal((M, K)).astype(mx.float16)
        w = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(x, w)

        def run_fp16(x=x, w=w):
            return x @ w

        mean, std, p50 = bench_op(f"fp16 M={M} K={K} N={N}", run_fp16, warmup, iters)
        tflops = compute_tflops(M, K, N, mean)
        fp16_gbps = (M * K * 2 + K * N * 2 + M * N * 2) / (mean / 1e6) / 1e9 if mean > 0 else 0
        label = f"fp16 baseline M={M}"
        print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f} {fp16_gbps:8.1f}")


# ---------------------------------------------------------------------------
# Quantization overhead
# ---------------------------------------------------------------------------
def bench_quantize_overhead(warmup, iters):
    """Measure the cost of mx.quantize() itself (offline, not in hot path)."""
    print(f"\n{'='*90}")
    print(f"  Quantization Overhead (mx.quantize)")
    print(f"{'='*90}")
    header = f"{'Config':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s}"
    print(header)
    print("-" * len(header))

    for bits in BITS_VALUES:
        for K, N in [(4096, 4096), (4096, 14336)]:
            w = mx.random.normal((N, K)).astype(mx.float16)
            mx.eval(w)

            def run_quantize(w=w, _bits=bits):
                return mx.quantize(w, bits=_bits, group_size=GROUP_SIZE)

            mean, std, p50 = bench_op(
                f"quantize Q{bits} [{N},{K}]", run_quantize, warmup, iters
            )
            label = f"mx.quantize Q{bits} [{N}x{K}]"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f}")


# ---------------------------------------------------------------------------
# Dequantize + matmul comparison
# ---------------------------------------------------------------------------
def bench_dequant_matmul(warmup, iters):
    """Compare: quantized_matmul vs dequantize-then-matmul."""
    print(f"\n{'='*90}")
    print(f"  Quantized vs Dequant+Matmul Comparison")
    print(f"{'='*90}")
    header = f"{'Config':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s} {'TFLOPS':>8s}"
    print(header)
    print("-" * len(header))

    K = 4096
    N = 4096

    for bits in BITS_VALUES:
        for M in [1, 32]:
            x = mx.random.normal((M, K)).astype(mx.float16)
            w_quant, scales, biases = make_quantized_weight(K, N, bits)
            mx.eval(x)

            # quantized_matmul (fused)
            def run_fused(x=x, w=w_quant, s=scales, b=biases, _bits=bits):
                return mx.quantized_matmul(
                    x, w, scales=s, biases=b,
                    transpose=True, bits=_bits, group_size=GROUP_SIZE
                )

            mean, std, p50 = bench_op(f"fused Q{bits} M={M}", run_fused, warmup, iters)
            tflops = compute_tflops(M, K, N, mean)
            label = f"quantized_matmul Q{bits} M={M}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f}")

            # dequantize then matmul (unfused)
            def run_unfused(x=x, w=w_quant, s=scales, b=biases, _bits=bits):
                w_fp = mx.dequantize(w, s, b, bits=_bits, group_size=GROUP_SIZE)
                return x @ w_fp.T

            mean, std, p50 = bench_op(f"dequant+mm Q{bits} M={M}", run_unfused, warmup, iters)
            tflops = compute_tflops(M, K, N, mean)
            label = f"dequant+matmul Q{bits} M={M}"
            print(f"  {label:48s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.4f}")

        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX Quantized MatMul Benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    print("=" * 90)
    print("MLX Quantized MatMul Benchmark")
    print(f"  mlx version      : {mx.__version__}")
    print(f"  device           : {mx.default_device()}")
    print(f"  input dtype      : float16")
    print(f"  group_size       : {GROUP_SIZE}")
    print(f"  QMV dims (K,N)   : {QMV_DIMS}")
    print(f"  QMM K,N          : {QMM_K}, {QMM_N}")
    print(f"  QMM M values     : {QMM_M_VALUES}")
    print(f"  bits             : {BITS_VALUES}")
    print(f"  warmup           : {args.warmup}")
    print(f"  iters            : {args.iters}")
    print("=" * 90)

    bench_qmv(args.warmup, args.iters)
    bench_qmm(args.warmup, args.iters)
    bench_quantize_overhead(args.warmup, args.iters)
    bench_dequant_matmul(args.warmup, args.iters)

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    main()
