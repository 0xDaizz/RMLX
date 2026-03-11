#!/usr/bin/env python3
"""MLX single transformer layer PREFILL latency benchmark.

Measures raw-op latency for a Qwen 3.5 MoE expert single transformer layer in
prefill mode (seq_len > 1) for direct comparison with RMLX prefill benchmarks.

Uses mlx.nn modules (Linear, RMSNorm, RoPE) to match the exact computation
performed by rmlx-nn's TransformerBlock::forward().

Usage:
    python mlx_prefill_bench.py [--warmup N] [--iters N] [--seq-lens 128,256,512]
"""

import argparse
import platform
import time
import statistics

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Qwen 3.5 MoE expert config
# ---------------------------------------------------------------------------
CONFIG = {
    "hidden_size": 3584,
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "intermediate_size": 2560,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-5,
}

DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048]
WARMUP = 5
ITERS = 20


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MoEAttention(nn.Module):
    """Qwen 3.5 MoE GQA attention with RoPE."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config["hidden_size"], self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config["hidden_size"], bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config["rope_theta"])

    def __call__(self, x, mask=None):
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE (included for fair comparison with rmlx which applies RoPE in TransformerBlock)
        q = self.rope(q)
        k = self.rope(k)

        # GQA handled natively by mx.fast.scaled_dot_product_attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MoEFFN(nn.Module):
    """Qwen 3.5 MoE expert SwiGLU FFN."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoETransformerBlock(nn.Module):
    """Single Qwen 3.5 MoE expert transformer layer (pre-norm, SwiGLU, GQA+RoPE)."""

    def __init__(self, config):
        super().__init__()
        self.attention = MoEAttention(config)
        self.ffn = MoEFFN(config)
        self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, x, mask=None):
        h = x + self.attention(self.input_layernorm(x), mask=mask)
        out = h + self.ffn(self.post_attention_layernorm(h))
        return out


# ---------------------------------------------------------------------------
# Weight memory estimation
# ---------------------------------------------------------------------------

def estimate_weight_bytes(config):
    """Estimate total weight memory for a single transformer layer in f16 (2 bytes per param)."""
    H = config["hidden_size"]
    D = config["head_dim"]
    Nh = config["num_heads"]
    Nkv = config["num_kv_heads"]
    I = config["intermediate_size"]

    params = 0
    # Attention projections: Q, K, V, O
    params += H * (Nh * D)       # q_proj
    params += H * (Nkv * D)      # k_proj
    params += H * (Nkv * D)      # v_proj
    params += (Nh * D) * H       # o_proj
    # FFN projections: gate, up, down
    params += H * I              # gate_proj
    params += H * I              # up_proj
    params += I * H              # down_proj
    # RMSNorm weights (2x)
    params += H * 2

    bytes_total = params * 2  # float16
    return params, bytes_total


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench_prefill(block, config, seq_len, warmup, iters):
    """Benchmark a single transformer layer in prefill mode for a given seq_len.

    Returns dict with timing stats.
    """
    x = mx.random.normal(shape=(1, seq_len, config["hidden_size"])).astype(mx.float16)

    # Create causal mask
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(mx.float16)

    mx.eval(x, mask)

    # Warmup
    for _ in range(warmup):
        out = block(x, mask=mask)
        mx.eval(out)

    # Benchmark
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = block(x, mask=mask)
        mx.eval(out)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # us

    latencies.sort()
    n = len(latencies)
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0.0
    p50 = latencies[n // 2]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    lo = latencies[0]
    hi = latencies[-1]
    tokens_per_sec = seq_len / (mean / 1e6)

    return {
        "seq_len": seq_len,
        "mean": mean,
        "std": std,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "min": lo,
        "max": hi,
        "tokens_per_sec": tokens_per_sec,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLX single-layer prefill latency benchmark (Qwen 3.5 MoE expert config)"
    )
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup iterations (default {WARMUP})")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help=f"Benchmark iterations (default {ITERS})")
    parser.add_argument("--seq-lens", type=str, default=None,
                        help="Comma-separated seq_len values (default: 128,256,512,1024,2048)")
    args = parser.parse_args()

    seq_lens = DEFAULT_SEQ_LENS
    if args.seq_lens:
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    # Weight stats
    num_params, weight_bytes = estimate_weight_bytes(CONFIG)
    weight_mb = weight_bytes / (1024 * 1024)

    # Header
    print("=" * 88)
    print("MLX Single-Layer Prefill Benchmark (Qwen 3.5 MoE expert config)")
    print("=" * 88)
    print(f"  mlx version   : {mx.__version__}")
    print(f"  device        : {mx.default_device()}")
    print(f"  platform      : {platform.platform()}")
    print(f"  python        : {platform.python_version()}")
    print(f"  dtype         : float16")
    print(f"  hidden_size   : {CONFIG['hidden_size']}")
    print(f"  num_heads     : {CONFIG['num_heads']} (kv_heads={CONFIG['num_kv_heads']})")
    print(f"  head_dim      : {CONFIG['head_dim']}")
    print(f"  intermediate  : {CONFIG['intermediate_size']}")
    print(f"  rope_theta    : {CONFIG['rope_theta']}")
    print(f"  rms_norm_eps  : {CONFIG['rms_norm_eps']}")
    print(f"  layer params  : {num_params:,} ({weight_mb:.1f} MB in f16)")
    print(f"  warmup        : {args.warmup}")
    print(f"  iters         : {args.iters}")
    print(f"  seq_lens      : {seq_lens}")
    print(f"  mode          : prefill (seq_len > 1, no KV cache)")
    print("=" * 88)

    # Build model
    block = MoETransformerBlock(CONFIG)

    # Cast all weights to float16
    block.update(block.apply(lambda x: x.astype(mx.float16)))
    mx.eval(block.parameters())

    # Run benchmarks
    results = []

    print()
    print(f"{'seq_len':>8s}  {'mean':>10s}  {'std':>8s}  {'p50':>10s}  "
          f"{'p95':>10s}  {'min':>10s}  {'max':>10s}  {'tok/s':>12s}")
    print("-" * 88)

    for seq_len in seq_lens:
        r = bench_prefill(block, CONFIG, seq_len, args.warmup, args.iters)
        results.append(r)
        print(f"{r['seq_len']:8d}  {r['mean']:9.1f}us  {r['std']:7.1f}us  {r['p50']:9.1f}us  "
              f"{r['p95']:9.1f}us  {r['min']:9.1f}us  {r['max']:9.1f}us  {r['tokens_per_sec']:11,.0f}")

    print("-" * 88)

    # Scaling analysis
    print()
    print("--- Scaling Analysis ---")
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            seq_ratio = r["seq_len"] / base["seq_len"]
            time_ratio = r["mean"] / base["mean"]
            efficiency = seq_ratio / time_ratio  # 1.0 = perfect linear scaling
            print(f"  seq_len {base['seq_len']:4d} -> {r['seq_len']:4d}  "
                  f"({seq_ratio:5.1f}x tokens)  latency {time_ratio:5.2f}x  "
                  f"scaling_efficiency={efficiency:.3f}")

    # Bandwidth estimate
    print()
    print("--- Bandwidth Estimate ---")
    print(f"  Layer weight memory: {weight_mb:.1f} MB (f16)")
    for r in results:
        # For prefill, compute is matmul-bound (not memory-bound like decode),
        # but we can still estimate effective bandwidth for reference.
        bw_gbs = weight_bytes / (r["mean"] / 1e6) / 1e9
        flops_approx = 2 * num_params * r["seq_len"]  # rough: 2 * params * seq_len FLOPs
        tflops = flops_approx / (r["mean"] / 1e6) / 1e12
        print(f"  seq_len={r['seq_len']:5d}  eff_bw={bw_gbs:6.1f} GB/s  "
              f"approx_throughput={tflops:.2f} TFLOPS")

    print()
    print("NOTE: This benchmark includes RoPE for fair comparison with rmlx's")
    print("TransformerBlock::forward() which applies RoPE inside attention.")
    print("Prefill is compute-bound (unlike decode which is memory-bound),")
    print("so TFLOPS is the more relevant metric than GB/s for prefill.")

    # ---- Compiled (mx.compile) benchmark for fair comparison with RMLX ExecGraph ----
    print()
    print("=" * 88)
    print("MLX Single-Layer Prefill Benchmark — COMPILED (mx.compile)")
    print("=" * 88)

    compiled_block = mx.compile(block)

    compiled_results = []

    print()
    print(f"{'seq_len':>8s}  {'mean':>10s}  {'std':>8s}  {'p50':>10s}  "
          f"{'p95':>10s}  {'min':>10s}  {'max':>10s}  {'tok/s':>12s}")
    print("-" * 88)

    for seq_len in seq_lens:
        r = bench_prefill(compiled_block, CONFIG, seq_len, args.warmup + 5, args.iters)
        compiled_results.append(r)
        print(f"{r['seq_len']:8d}  {r['mean']:9.1f}us  {r['std']:7.1f}us  {r['p50']:9.1f}us  "
              f"{r['p95']:9.1f}us  {r['min']:9.1f}us  {r['max']:9.1f}us  {r['tokens_per_sec']:11,.0f}")

    print("-" * 88)

    # Compiled vs eager comparison
    print()
    print("--- Compiled vs Eager Comparison ---")
    for eager, compiled in zip(results, compiled_results):
        if eager["seq_len"] == compiled["seq_len"]:
            speedup = eager["mean"] / compiled["mean"]
            print(f"  seq_len={eager['seq_len']:5d}  eager={eager['mean']:9.1f}us  "
                  f"compiled={compiled['mean']:9.1f}us  speedup={speedup:.3f}x")

    # Compiled bandwidth estimate
    print()
    print("--- Compiled Bandwidth Estimate ---")
    print(f"  Layer weight memory: {weight_mb:.1f} MB (f16)")
    for r in compiled_results:
        bw_gbs = weight_bytes / (r["mean"] / 1e6) / 1e9
        flops_approx = 2 * num_params * r["seq_len"]
        tflops = flops_approx / (r["mean"] / 1e6) / 1e12
        print(f"  seq_len={r['seq_len']:5d}  eff_bw={bw_gbs:6.1f} GB/s  "
              f"approx_throughput={tflops:.2f} TFLOPS")

    print()
    print("=" * 88)
    print("Done.")


if __name__ == "__main__":
    main()
