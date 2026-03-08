#!/usr/bin/env python3
"""MLX Gather MatMul (Expert MoE Dispatch) Benchmark

Benchmarks MLX's expert matmul patterns for MoE models (Mixtral 8x7B style):
  1. Individual matmul per expert (loop dispatch)
  2. Batched gather via block_masked_mm (if available) or concatenated approach

Mixtral 8x7B dims: 8 experts, K=4096, N=14336 (gate_up), N=4096 (down)

Usage:
    python mlx_gather_mm_bench.py [--warmup N] [--iters N]
"""

import argparse
import time
import statistics

import mlx.core as mx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WARMUP = 5
ITERS = 20

NUM_EXPERTS = 8
K = 4096
N_UP = 14336       # gate/up projection width per expert
N_DOWN = 4096      # down projection width per expert
TOP_K = 2          # top-k experts per token (Mixtral uses 2)

M_VALUES = [1, 4, 8, 16, 32]

# Additional dimension sets for micro-benchmarks (to match RMLX bench)
# (label, K, N)
MICRO_DIMS = [
    ("gate/up K=4096 N=14336", 4096, 14336),   # Mixtral gate/up
    ("attn    K=4096 N=4096",  4096, 4096),     # attention projections
    ("expert  K=4096 N=2048",  4096, 2048),     # per-expert intermediate
    ("dsv2    K=5120 N=1536",  5120, 1536),     # DeepSeek-V2 expert FFN
]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench_op(name, fn, warmup=WARMUP, iters=ITERS):
    """Benchmark fn (zero-arg callable returning mx array(s)).

    Returns (mean_us, tflops) for the caller.
    """
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


def compute_tflops(M, K, N, num_experts_active, elapsed_us):
    """Compute TFLOPS for expert matmuls.

    Each expert matmul: 2*M*K*N FLOPs.
    Total = num_experts_active * 2*M*K*N.
    """
    flops = num_experts_active * 2.0 * M * K * N
    seconds = elapsed_us / 1e6
    if seconds == 0:
        return 0.0
    return flops / seconds / 1e12


# ---------------------------------------------------------------------------
# Method 1: Individual matmul per expert (loop)
# ---------------------------------------------------------------------------
def bench_individual_experts(M, K_dim, N_dim, warmup, iters):
    """Each expert gets its own matmul call. Simulates naive MoE dispatch."""
    # Create per-expert weights and per-expert input slices
    expert_weights = []
    for _ in range(TOP_K):
        w = mx.random.normal((K_dim, N_dim)).astype(mx.float16)
        mx.eval(w)
        expert_weights.append(w)

    # Input: M tokens, each routed to TOP_K experts
    x = mx.random.normal((M, K_dim)).astype(mx.float16)
    mx.eval(x)

    def run():
        outputs = []
        for i in range(TOP_K):
            out = x @ expert_weights[i]  # [M, N]
            outputs.append(out)
        return outputs

    return bench_op(f"individual_mm M={M} K={K_dim} N={N_dim} top_k={TOP_K}", run, warmup, iters)


# ---------------------------------------------------------------------------
# Method 2: Concatenated expert weights (single large matmul)
# ---------------------------------------------------------------------------
def bench_concat_experts(M, K_dim, N_dim, warmup, iters):
    """Concatenate expert weights into one big matrix and do a single matmul.

    This is an upper bound — ignores routing (all tokens to all experts).
    Shape: [M, K] @ [K, N*TOP_K] -> [M, N*TOP_K]
    """
    w_concat = mx.random.normal((K_dim, N_dim * TOP_K)).astype(mx.float16)
    x = mx.random.normal((M, K_dim)).astype(mx.float16)
    mx.eval(w_concat, x)

    def run():
        return x @ w_concat  # [M, N*TOP_K]

    return bench_op(f"concat_mm M={M} K={K_dim} N={N_dim}*{TOP_K}", run, warmup, iters)


# ---------------------------------------------------------------------------
# Method 3: Gather-style (group by expert, variable-size matmul)
# ---------------------------------------------------------------------------
def bench_gather_dispatch(M, K_dim, N_dim, warmup, iters):
    """Simulate gather_mm: route tokens to experts, do per-expert matmul
    with gathered (contiguous) token subsets.

    For M tokens with TOP_K=2, each expert gets ~M*TOP_K/NUM_EXPERTS tokens.
    We simulate a balanced assignment.
    """
    expert_weights = []
    for _ in range(NUM_EXPERTS):
        w = mx.random.normal((K_dim, N_dim)).astype(mx.float16)
        mx.eval(w)
        expert_weights.append(w)

    # Simulate routing: each token assigned to TOP_K experts
    total_tokens = M * TOP_K
    tokens_per_expert = max(1, total_tokens // NUM_EXPERTS)

    # Pre-create gathered input per expert
    expert_inputs = []
    for _ in range(NUM_EXPERTS):
        inp = mx.random.normal((tokens_per_expert, K_dim)).astype(mx.float16)
        mx.eval(inp)
        expert_inputs.append(inp)

    def run():
        outputs = []
        for i in range(NUM_EXPERTS):
            if expert_inputs[i].shape[0] > 0:
                out = expert_inputs[i] @ expert_weights[i]
                outputs.append(out)
        return outputs

    return bench_op(
        f"gather_dispatch M={M} experts={NUM_EXPERTS} top_k={TOP_K}",
        run, warmup, iters
    )


# ---------------------------------------------------------------------------
# Method 4: block_masked_mm (if available in MLX)
# ---------------------------------------------------------------------------
def has_block_masked_mm():
    return hasattr(mx, "block_masked_mm")


def bench_block_masked(M, K_dim, N_dim, warmup, iters):
    """Use mx.block_masked_mm for expert dispatch."""
    if not has_block_masked_mm():
        return None, None, None

    # block_masked_mm: A [B, M, K], B [B, K, N], with block masks
    # Batch dim = NUM_EXPERTS
    tokens_per_expert = max(1, (M * TOP_K) // NUM_EXPERTS)

    # Pad M and N to be multiples of block_size (required by block_masked_mm)
    block_size = 32
    padded_M = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    padded_N = ((N_dim + block_size - 1) // block_size) * block_size
    padded_K = ((K_dim + block_size - 1) // block_size) * block_size

    A = mx.random.normal((NUM_EXPERTS, padded_M, padded_K)).astype(mx.float16)
    B = mx.random.normal((NUM_EXPERTS, padded_K, padded_N)).astype(mx.float16)
    mx.eval(A, B)

    mask_M = padded_M // block_size
    mask_N = padded_N // block_size

    # All-ones mask (all experts active)
    lhs_mask = mx.ones((NUM_EXPERTS, mask_M, 1), dtype=mx.bool_)
    rhs_mask = mx.ones((NUM_EXPERTS, 1, mask_N), dtype=mx.bool_)
    mx.eval(lhs_mask, rhs_mask)

    def run():
        return mx.block_masked_mm(A, B, block_size, lhs_mask, rhs_mask)

    try:
        return bench_op(
            f"block_masked_mm M={M} experts={NUM_EXPERTS}",
            run, warmup, iters
        )
    except Exception as e:
        print(f"  [WARNING] block_masked_mm M={M} failed: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Full MoE layer simulation
# ---------------------------------------------------------------------------
def bench_moe_layer(M, warmup, iters):
    """Full MoE forward: router -> top-k -> gather -> expert matmuls -> scatter.

    Mixtral 8x7B: gate [4096, 8] -> softmax -> top-2 -> expert FFN.
    Expert FFN: gate_proj [4096, 14336], up_proj [4096, 14336], down_proj [14336, 4096].
    """
    x = mx.random.normal((M, K)).astype(mx.float16)

    # Router
    w_gate = mx.random.normal((K, NUM_EXPERTS)).astype(mx.float16)

    # Expert weights (gate+up fused, down)
    expert_gate_up = []
    expert_down = []
    for _ in range(NUM_EXPERTS):
        wgu = mx.random.normal((K, N_UP * 2)).astype(mx.float16)
        wd = mx.random.normal((N_UP, N_DOWN)).astype(mx.float16)
        mx.eval(wgu, wd)
        expert_gate_up.append(wgu)
        expert_down.append(wd)

    mx.eval(x, w_gate)

    def run():
        # 1. Router scores
        logits = x @ w_gate  # [M, 8]
        probs = mx.softmax(logits, axis=-1)

        # 2. Top-k selection
        top_indices = mx.argpartition(-probs, kth=TOP_K, axis=-1)[:, :TOP_K]  # [M, 2]
        top_weights = mx.take_along_axis(probs, top_indices, axis=-1)  # [M, 2]

        # 3. Expert dispatch (simplified: each expert processes all tokens assigned to it)
        # In practice you'd gather/scatter, but for benchmarking the matmul cost:
        output = mx.zeros((M, K), dtype=mx.float16)
        for e in range(NUM_EXPERTS):
            # Mask for tokens routed to this expert (across both top-k slots)
            mask = mx.any(top_indices == e, axis=-1)  # [M]

            # For benchmarking, always run the matmul (avoid dynamic shapes)
            gate_up = x @ expert_gate_up[e]  # [M, N_UP*2]
            gate_part = gate_up[:, :N_UP]
            up_part = gate_up[:, N_UP:]
            ffn = (gate_part * mx.sigmoid(gate_part)) * up_part  # SiLU
            down = ffn @ expert_down[e]  # [M, K]

            # Weight by routing probability (simplified)
            output = output + down

        return output

    mean, std, p50 = bench_op(f"moe_layer M={M} E={NUM_EXPERTS} top_k={TOP_K}", run, warmup, iters)
    # FLOPs: per expert FFN = 2*M*K*N_UP*2 (gate_up) + 2*M*N_UP*N_DOWN (down)
    # All 8 experts run (simplified benchmark)
    flops_per_expert = 2.0 * M * K * N_UP * 2 + 2.0 * M * N_UP * N_DOWN
    total_flops = NUM_EXPERTS * flops_per_expert
    tflops = total_flops / (mean / 1e6) / 1e12 if mean > 0 else 0
    return mean, std, p50, tflops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX Gather MatMul (MoE Expert Dispatch) Benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    print("=" * 90)
    print("MLX Gather MatMul (MoE Expert Dispatch) Benchmark")
    print(f"  mlx version      : {mx.__version__}")
    print(f"  device           : {mx.default_device()}")
    print(f"  dtype            : float16")
    print(f"  num_experts      : {NUM_EXPERTS}")
    print(f"  top_k            : {TOP_K}")
    print(f"  micro dims       : {[(l,k,n) for l,k,n in MICRO_DIMS]}")
    print(f"  MoE layer K      : {K}")
    print(f"  MoE layer N_UP   : {N_UP}")
    print(f"  MoE layer N_DOWN : {N_DOWN}")
    print(f"  M values         : {M_VALUES}")
    print(f"  warmup           : {args.warmup}")
    print(f"  iters            : {args.iters}")
    print(f"  block_masked_mm  : {'available' if has_block_masked_mm() else 'NOT available'}")
    print("=" * 90)

    # ---- Expert matmul micro-benchmarks (all MoE dimensions) ----
    for dim_label, K_dim, N_dim in MICRO_DIMS:
        print(f"\n{'='*90}")
        print(f"  Expert MatMul Micro-Benchmarks ({dim_label})")
        print(f"{'='*90}")
        header = f"{'Method':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s} {'TFLOPS':>8s}"
        print(header)
        print("-" * len(header))

        for M in M_VALUES:
            print()

            # Method 1: individual expert matmuls (top_k only)
            mean, std, p50 = bench_individual_experts(M, K_dim, N_dim, args.warmup, args.iters)
            tflops = compute_tflops(M, K_dim, N_dim, TOP_K, mean)
            print(f"  {'individual_mm M=' + str(M):44s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.2f}")

            # Method 2: concatenated single matmul
            mean, std, p50 = bench_concat_experts(M, K_dim, N_dim, args.warmup, args.iters)
            tflops = compute_tflops(M, K_dim, N_dim, TOP_K, mean)
            print(f"  {'concat_mm M=' + str(M):44s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.2f}")

            # Method 3: gather dispatch (all 8 experts, balanced)
            mean, std, p50 = bench_gather_dispatch(M, K_dim, N_dim, args.warmup, args.iters)
            tokens_per_expert = max(1, (M * TOP_K) // NUM_EXPERTS)
            tflops = compute_tflops(tokens_per_expert, K_dim, N_dim, NUM_EXPERTS, mean)
            print(f"  {'gather_dispatch M=' + str(M):44s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.2f}")

            # Method 4: block_masked_mm (if available)
            if has_block_masked_mm():
                mean, std, p50 = bench_block_masked(M, K_dim, N_dim, args.warmup, args.iters)
                if mean is not None:
                    tokens_per_expert = max(1, (M * TOP_K) // NUM_EXPERTS)
                    tflops = compute_tflops(tokens_per_expert, K_dim, N_dim, NUM_EXPERTS, mean)
                    print(f"  {'block_masked_mm M=' + str(M):44s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.2f}")

    # ---- Full MoE layer benchmark ----
    print(f"\n{'='*90}")
    print(f"  Full MoE Layer (router + {NUM_EXPERTS} experts, gate_up + SiLU + down)")
    print(f"{'='*90}")
    header = f"{'Config':<50s} {'mean(us)':>10s} {'std(us)':>10s} {'p50(us)':>10s} {'TFLOPS':>8s}"
    print(header)
    print("-" * len(header))

    for M in M_VALUES:
        mean, std, p50, tflops = bench_moe_layer(M, args.warmup, args.iters)
        label = f"moe_layer M={M}"
        print(f"  {label:44s} {mean:10.1f} {std:10.1f} {p50:10.1f} {tflops:8.2f}")

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    main()
