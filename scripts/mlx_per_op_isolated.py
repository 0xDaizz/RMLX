#!/usr/bin/env python3
"""MLX Per-Operation Isolated Benchmark — focused S=128 comparison baseline.

Measures individual operations (matmul, RMSNorm, SDPA, SiLU, residual add)
in strict isolation with massive warmup, to identify which ops have the
largest gap vs RMLX.

Key differences from mlx_per_op_bench.py:
  - Pure mx.matmul (not nn.Linear) with explicit weight layout
  - 20 warmup + 30 bench iterations per op
  - JIT warmup phase at startup to pre-compile all Metal PSOs
  - Only S=128 (primary) and S=256 (comparison)
  - Reports mean, p50, min, std per op

Usage:
    python mlx_per_op_isolated.py
"""

import statistics
import time

import mlx.core as mx
import mlx.nn as nn

# -- Config (matches RMLX exactly) -----------------------------------------
HIDDEN_SIZE = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_DIM = 2560
RMS_NORM_EPS = 1e-5

WARMUP_ITERS = 20
BENCH_ITERS = 30

# -- Matmul dimensions (merged weight sizes) --------------------------------
# QKV merged: Q=[28*128=3584] + K=[4*128=512] + V=[4*128=512] => N=4608
QKV_N = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  # 4608
O_PROJ_N = HIDDEN_SIZE  # 3584
GATE_UP_N = 2 * INTERMEDIATE_DIM  # 5120
DOWN_N = HIDDEN_SIZE  # 3584


# -- FLOP helpers -----------------------------------------------------------
def gemm_flops(M, N, K):
    return 2.0 * M * N * K


def tflops(flops, time_us):
    if time_us <= 0:
        return 0.0
    return flops / (time_us / 1e6) / 1e12


# -- Timing -----------------------------------------------------------------
def time_op(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time a single op with eval sync. Returns list of times in us."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)

    return times


def stats(times):
    """Return (mean, p50, min, std) from a list of times."""
    s = sorted(times)
    p50 = s[len(s) // 2]
    return statistics.mean(times), p50, min(times), statistics.stdev(times)


# -- Build weights and inputs -----------------------------------------------
def build_weights():
    """Create all weight matrices as [N, K] (transposed layout)."""
    w = {}
    # QKV merged: [4608, 3584]
    w["qkv"] = mx.random.normal((QKV_N, HIDDEN_SIZE)).astype(mx.float16)
    # O proj: [3584, 3584]
    w["o_proj"] = mx.random.normal((O_PROJ_N, NUM_HEADS * HEAD_DIM)).astype(mx.float16)
    # Gate+Up merged: [5120, 3584]
    w["gate_up"] = mx.random.normal((GATE_UP_N, HIDDEN_SIZE)).astype(mx.float16)
    # Down proj: [3584, 2560]
    w["down"] = mx.random.normal((DOWN_N, INTERMEDIATE_DIM)).astype(mx.float16)
    # RMSNorm weight: [3584]
    w["rms_weight"] = mx.random.normal((HIDDEN_SIZE,)).astype(mx.float16)

    mx.eval(*w.values())
    return w


def build_inputs(seq_len):
    """Create all input tensors for a given seq_len."""
    inp = {}
    S = seq_len

    # Hidden state: [1, S, 3584]
    inp["hidden"] = mx.random.normal((1, S, HIDDEN_SIZE)).astype(mx.float16)
    # Second hidden (for residual): [1, S, 3584]
    inp["hidden2"] = mx.random.normal((1, S, HIDDEN_SIZE)).astype(mx.float16)
    # Intermediate (for SiLU*Mul): [1, S, 2560]
    inp["intermediate"] = mx.random.normal((1, S, INTERMEDIATE_DIM)).astype(mx.float16)
    # SDPA inputs: Q=[1, 28, S, 128], K=[1, 4, S, 128], V=[1, 4, S, 128]
    inp["q"] = mx.random.normal((1, NUM_HEADS, S, HEAD_DIM)).astype(mx.float16)
    inp["k"] = mx.random.normal((1, NUM_KV_HEADS, S, HEAD_DIM)).astype(mx.float16)
    inp["v"] = mx.random.normal((1, NUM_KV_HEADS, S, HEAD_DIM)).astype(mx.float16)
    # Causal mask
    inp["mask"] = nn.MultiHeadAttention.create_additive_causal_mask(S).astype(mx.float16)

    mx.eval(*inp.values())
    return inp


# -- JIT warmup (compile all Metal PSOs before measurement) -----------------
def jit_warmup(weights, inputs):
    """Run every op once to trigger Metal shader compilation."""
    w, inp = weights, inputs
    S = inp["hidden"].shape[1]
    scale = HEAD_DIM ** -0.5

    # Matmuls
    mx.eval(mx.matmul(inp["hidden"], w["qkv"].T))
    mx.eval(mx.matmul(inp["hidden"], w["o_proj"].T))
    mx.eval(mx.matmul(inp["hidden"], w["gate_up"].T))
    mx.eval(mx.matmul(inp["intermediate"], w["down"].T))
    # RMSNorm
    mx.eval(mx.fast.rms_norm(inp["hidden"], w["rms_weight"], RMS_NORM_EPS))
    # SDPA
    mx.eval(mx.fast.scaled_dot_product_attention(
        inp["q"], inp["k"], inp["v"], scale=scale, mask=inp["mask"]
    ))
    # SiLU * Mul
    half = INTERMEDIATE_DIM
    gate = inp["intermediate"][..., :half]
    up = inp["intermediate"][..., half:]
    # Use full intermediate for SiLU test
    mx.eval(nn.silu(inp["intermediate"]) * inp["intermediate"])
    # Residual add
    mx.eval(inp["hidden"] + inp["hidden2"])

    print("JIT warmup complete — all Metal PSOs compiled.")


# -- Benchmark ops ----------------------------------------------------------
def bench_ops(weights, inputs):
    """Benchmark each op individually. Returns list of (name, stats_tuple, flops_or_none)."""
    w, inp = weights, inputs
    S = inp["hidden"].shape[1]
    scale = HEAD_DIM ** -0.5
    results = []

    # 1. Pure matmul: QKV — M=S, K=3584, N=4608
    flops = gemm_flops(S, QKV_N, HIDDEN_SIZE)
    times = time_op(lambda: mx.matmul(inp["hidden"], w["qkv"].T))
    results.append(("Matmul: QKV (M=%d K=3584 N=4608)" % S, stats(times), flops))

    # 2. Pure matmul: O proj — M=S, K=3584, N=3584
    flops = gemm_flops(S, O_PROJ_N, NUM_HEADS * HEAD_DIM)
    times = time_op(lambda: mx.matmul(inp["hidden"], w["o_proj"].T))
    results.append(("Matmul: O proj (M=%d K=3584 N=3584)" % S, stats(times), flops))

    # 3. Pure matmul: Gate+Up — M=S, K=3584, N=5120
    flops = gemm_flops(S, GATE_UP_N, HIDDEN_SIZE)
    times = time_op(lambda: mx.matmul(inp["hidden"], w["gate_up"].T))
    results.append(("Matmul: Gate+Up (M=%d K=3584 N=5120)" % S, stats(times), flops))

    # 4. Pure matmul: Down proj — M=S, K=2560, N=3584
    flops = gemm_flops(S, DOWN_N, INTERMEDIATE_DIM)
    times = time_op(lambda: mx.matmul(inp["intermediate"], w["down"].T))
    results.append(("Matmul: Down (M=%d K=2560 N=3584)" % S, stats(times), flops))

    # 5. RMSNorm — [1, S, 3584]
    times = time_op(lambda: mx.fast.rms_norm(inp["hidden"], w["rms_weight"], RMS_NORM_EPS))
    results.append(("RMSNorm [1,%d,3584]" % S, stats(times), None))

    # 6. SDPA — Q=[1,28,S,128], K=[1,4,S,128], V=[1,4,S,128], causal
    sdpa_flops = 2 * (2.0 * 1 * NUM_HEADS * S * S * HEAD_DIM)
    times = time_op(lambda: mx.fast.scaled_dot_product_attention(
        inp["q"], inp["k"], inp["v"], scale=scale, mask=inp["mask"]
    ))
    results.append(("SDPA (Q=28h K/V=4h S=%d causal)" % S, stats(times), sdpa_flops))

    # 7. SiLU * Mul — [1, S, 2560]
    # Split intermediate into two halves to simulate gate * up
    half = INTERMEDIATE_DIM // 2
    gate_in = mx.random.normal((1, S, INTERMEDIATE_DIM)).astype(mx.float16)
    up_in = mx.random.normal((1, S, INTERMEDIATE_DIM)).astype(mx.float16)
    mx.eval(gate_in, up_in)
    times = time_op(lambda: nn.silu(gate_in) * up_in)
    results.append(("SiLU*Mul [1,%d,2560]" % S, stats(times), None))

    # 8. Residual add — two [1, S, 3584]
    times = time_op(lambda: inp["hidden"] + inp["hidden2"])
    results.append(("Residual add [1,%d,3584]" % S, stats(times), None))

    return results


# -- Output -----------------------------------------------------------------
def print_results(results, seq_len):
    """Print markdown table for one seq_len."""
    print(f"\n## S={seq_len}  (warmup={WARMUP_ITERS}, bench={BENCH_ITERS})")
    print()
    print("| Op | Mean (us) | P50 (us) | Min (us) | Std (us) | TFLOPS |")
    print("|-----|----------:|---------:|---------:|---------:|-------:|")

    p50_sum = 0.0
    for name, (mean, p50, mn, std), flops in results:
        p50_sum += p50
        if flops:
            tf = tflops(flops, p50)
            print(f"| {name} | {mean:.1f} | {p50:.1f} | {mn:.1f} | {std:.1f} | {tf:.2f} |")
        else:
            print(f"| {name} | {mean:.1f} | {p50:.1f} | {mn:.1f} | {std:.1f} | -- |")

    print(f"| **Sum (layer estimate)** | | **{p50_sum:.1f}** | | | |")
    print()


def print_comparison(results_128, results_256):
    """Side-by-side P50 comparison."""
    print("\n## Comparison: S=128 vs S=256 (P50 us)")
    print()
    print("| Op | S=128 | S=256 | Ratio |")
    print("|-----|------:|------:|------:|")

    for (n1, s1, _), (n2, s2, _) in zip(results_128, results_256):
        p50_128 = s1[1]
        p50_256 = s2[1]
        ratio = p50_256 / p50_128 if p50_128 > 0 else 0
        # Use shorter name (strip dimension info for readability)
        short = n1.split("(")[0].strip()
        print(f"| {short} | {p50_128:.1f} | {p50_256:.1f} | {ratio:.2f}x |")

    sum_128 = sum(s[1] for _, s, _ in results_128)
    sum_256 = sum(s[1] for _, s, _ in results_256)
    ratio = sum_256 / sum_128 if sum_128 > 0 else 0
    print(f"| **Total** | **{sum_128:.1f}** | **{sum_256:.1f}** | **{ratio:.2f}x** |")
    print()


# -- Main -------------------------------------------------------------------
def main():
    print("# MLX Per-Op Isolated Benchmark")
    print()
    print(
        f"Config: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_DIM}"
    )
    print(f"dtype: float16, warmup={WARMUP_ITERS}, bench={BENCH_ITERS}")
    print(f"Pure mx.matmul (not nn.Linear), weights as [N, K] with x @ w.T")
    print()

    weights = build_weights()

    # -- JIT warmup phase ---------------------------------------------------
    print("--- JIT warmup phase ---")
    inp_128 = build_inputs(128)
    inp_256 = build_inputs(256)
    jit_warmup(weights, inp_128)
    jit_warmup(weights, inp_256)
    print()

    # -- S=128 (primary) ----------------------------------------------------
    print("Benchmarking S=128...", flush=True)
    results_128 = bench_ops(weights, inp_128)
    print_results(results_128, 128)

    # -- S=256 (comparison) -------------------------------------------------
    print("Benchmarking S=256...", flush=True)
    results_256 = bench_ops(weights, inp_256)
    print_results(results_256, 256)

    # -- Side-by-side comparison --------------------------------------------
    print_comparison(results_128, results_256)


if __name__ == "__main__":
    main()
