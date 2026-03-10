#!/usr/bin/env python3
"""MLX Q4 Quantized MatMul Pipeline Benchmark

Measures two modes for direct comparison with RMLX:
  Mode A — Per-op: single quantized_matmul + eval + sync per iteration
  Mode B — Pipeline: 96 quantized_matmuls (32 layers x 3 FFN ops) -> single eval -> sync

Model configurations (real MoE architectures, ordered smallest→largest for thermal):
  Qwen3.5-MoE:    K=2048, N=512    (tiny expert, 256E top-8)
  DeepSeek-V3:     K=7168, N=2048   (mainstream MoE, 256E top-8)
  MiniMax-Text-01: K=6144, N=9216   (large expert, 32E top-2)
  Mixtral-8x22B:   K=6144, N=16384  (largest expert, 8E top-2)

Usage:
    python mlx_qmm_pipeline_bench.py [--warmup N] [--iters N] [--bits 4] [--group-size 64]
"""

import argparse
import gc
import sys
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants (must match RMLX exactly)
# ---------------------------------------------------------------------------
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_LAYERS = 32

# (name, K_hidden, N_intermediate) — ordered smallest to largest for thermal
MODEL_DIMS = [
    ("Qwen3.5-MoE",    2048,  512),    # tiny expert, 256E top-8 — smallest, run first
    ("DeepSeek-V3",     7168,  2048),   # mainstream MoE, 256E top-8
    ("MiniMax-Text-01", 6144,  9216),   # large expert, 32E top-2
    ("Mixtral-8x22B",   6144,  16384),  # largest expert, 8E top-2 — run last
]

WARMUP = 5
ITERS = 20
BITS = 4
GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------
def estimate_pipeline_memory_mb(k_hidden, n_inter, n_layers, bits, group_size):
    """Estimate GPU memory for pipeline weights in MB."""
    # gate: [N_inter, K_hidden], up: [N_inter, K_hidden], down: [K_hidden, N_inter]
    # Quantized weight bytes: rows * cols * bits / 8
    gate_bytes = n_inter * k_hidden * bits / 8
    up_bytes = n_inter * k_hidden * bits / 8
    down_bytes = k_hidden * n_inter * bits / 8
    # Scales + biases: (rows * cols / group_size) * 4 bytes (f16 scale + f16 bias)
    gate_meta = (n_inter * k_hidden / group_size) * 4
    up_meta = (n_inter * k_hidden / group_size) * 4
    down_meta = (k_hidden * n_inter / group_size) * 4
    per_layer = gate_bytes + up_bytes + down_bytes + gate_meta + up_meta + down_meta
    total = per_layer * n_layers
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Mode A: Per-op benchmark
# ---------------------------------------------------------------------------
def bench_per_op(m, k, n, bits=BITS, group_size=GROUP_SIZE, warmup=WARMUP, iters=ITERS):
    """Benchmark a single quantized_matmul with eval+sync per iteration.

    Returns (p50_us, tflops).
    """
    x = mx.random.normal((m, k)).astype(mx.float16)
    w = mx.random.normal((n, k)).astype(mx.float16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(x, w_q, scales, biases)

    for _ in range(warmup):
        out = mx.quantized_matmul(
            x, w_q, scales, biases,
            transpose=True, group_size=group_size, bits=bits,
        )
        mx.eval(out)
    mx.synchronize()

    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = mx.quantized_matmul(
            x, w_q, scales, biases,
            transpose=True, group_size=group_size, bits=bits,
        )
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # us

    times.sort()
    p50 = times[len(times) // 2]
    tflops = 2.0 * m * k * n / (p50 * 1e-6) / 1e12 if p50 > 0 else 0.0
    return p50, tflops


# ---------------------------------------------------------------------------
# Mode B: Pipeline benchmark
# ---------------------------------------------------------------------------
def bench_pipeline(m, k_hidden, n_inter, bits=BITS, group_size=GROUP_SIZE,
                   n_layers=N_LAYERS, warmup=WARMUP, iters=ITERS):
    """Benchmark 32-layer x 3-op (gate, up, down) pipeline with single eval+sync.

    Returns (total_p50_us, per_layer_us, per_op_us, amortized_tflops).
    """
    # Create input
    x = mx.random.normal((m, k_hidden)).astype(mx.float16)

    # Create per-layer weights: gate/up [N_inter, K_hidden], down [K_hidden, N_inter]
    layer_weights = []
    for _ in range(n_layers):
        w_gate = mx.random.normal((n_inter, k_hidden)).astype(mx.float16)
        w_up = mx.random.normal((n_inter, k_hidden)).astype(mx.float16)
        w_down = mx.random.normal((k_hidden, n_inter)).astype(mx.float16)

        gate_q, gate_s, gate_b = mx.quantize(w_gate, group_size=group_size, bits=bits)
        up_q, up_s, up_b = mx.quantize(w_up, group_size=group_size, bits=bits)
        down_q, down_s, down_b = mx.quantize(w_down, group_size=group_size, bits=bits)

        layer_weights.append({
            "gate": (gate_q, gate_s, gate_b),
            "up": (up_q, up_s, up_b),
            "down": (down_q, down_s, down_b),
        })

    # Materialize all weights on GPU
    mx.eval(*[t for lw in layer_weights for v in lw.values() for t in v])
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        current = x
        for lw in layer_weights:
            gq, gs, gb = lw["gate"]
            uq, us, ub = lw["up"]
            dq, ds, db = lw["down"]
            gate_out = mx.quantized_matmul(
                current, gq, gs, gb,
                transpose=True, group_size=group_size, bits=bits,
            )
            up_out = mx.quantized_matmul(
                current, uq, us, ub,
                transpose=True, group_size=group_size, bits=bits,
            )
            down_out = mx.quantized_matmul(
                gate_out, dq, ds, db,
                transpose=True, group_size=group_size, bits=bits,
            )
            current = down_out
        mx.eval(current)
    mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        current = x
        for lw in layer_weights:
            gq, gs, gb = lw["gate"]
            uq, us, ub = lw["up"]
            dq, ds, db = lw["down"]
            gate_out = mx.quantized_matmul(
                current, gq, gs, gb,
                transpose=True, group_size=group_size, bits=bits,
            )
            up_out = mx.quantized_matmul(
                current, uq, us, ub,
                transpose=True, group_size=group_size, bits=bits,
            )
            down_out = mx.quantized_matmul(
                gate_out, dq, ds, db,
                transpose=True, group_size=group_size, bits=bits,
            )
            current = down_out
        mx.eval(current)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # us

    times.sort()
    p50 = times[len(times) // 2]
    per_layer = p50 / n_layers
    per_op = p50 / (n_layers * 3)
    # Use gate/up shape for TFLOPS (conservative estimate)
    tflops = 2.0 * m * k_hidden * n_inter / (per_op * 1e-6) / 1e12 if per_op > 0 else 0.0
    return p50, per_layer, per_op, tflops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLX Q4 Pipeline Benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Filter models by name substring (e.g. 'DeepSeek' 'Mixtral')",
    )
    parser.add_argument(
        "--m-values", nargs="*", type=int, default=None,
        help="Override M values (e.g. --m-values 1 32 256)",
    )
    parser.add_argument("--per-op-only", action="store_true", help="Skip pipeline mode")
    parser.add_argument("--pipeline-only", action="store_true", help="Skip per-op mode")
    args = parser.parse_args()

    bits = args.bits
    group_size = args.group_size
    warmup = args.warmup
    iters = args.iters
    m_values = args.m_values if args.m_values else M_VALUES

    # Filter models if requested
    models = MODEL_DIMS
    if args.models:
        models = [
            m for m in MODEL_DIMS
            if any(f.lower() in m[0].lower() for f in args.models)
        ]
        if not models:
            print(f"No models matched filter: {args.models}")
            sys.exit(1)

    print("=" * 65)
    print(f"  MLX Q{bits} Pipeline Benchmark (group_size={group_size})")
    print("=" * 65)
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"Warmup: {warmup}, Iters: {iters}")
    print(f"M values: {m_values}")
    print(f"Models: {[m[0] for m in models]}")
    print()

    for model_name, k_hidden, n_inter in models:
        mem_mb = estimate_pipeline_memory_mb(k_hidden, n_inter, N_LAYERS, bits, group_size)
        n_layers_pipeline = N_LAYERS

        # Reduce layers for large models to avoid OOM
        if mem_mb > 10_000:
            n_layers_pipeline = 8
            mem_mb_reduced = estimate_pipeline_memory_mb(
                k_hidden, n_inter, n_layers_pipeline, bits, group_size,
            )
            print(f"--- {model_name} (K={k_hidden}, N={n_inter}) ---")
            print(f"  [Memory: {mem_mb:.0f} MB for 32L, reduced to {n_layers_pipeline}L = {mem_mb_reduced:.0f} MB]")
        else:
            print(f"--- {model_name} (K={k_hidden}, N={n_inter}) ---")
            print(f"  [Memory: {mem_mb:.0f} MB for {N_LAYERS}L pipeline weights]")

        # --- Per-op: gate/up projection (K_hidden -> N_inter) ---
        if not args.pipeline_only:
            print()
            print(f"  Per-op (gate/up K={k_hidden} -> N={n_inter}):")
            for m in m_values:
                p50, tflops = bench_per_op(m, k_hidden, n_inter, bits, group_size, warmup, iters)
                print(f"  M={m:<6d} p50={p50:>9.1f}us   {tflops:.3f}T")

            # --- Per-op: down projection (N_inter -> K_hidden) ---
            print()
            print(f"  Per-op (down N={n_inter} -> K={k_hidden}):")
            for m in m_values:
                p50, tflops = bench_per_op(m, n_inter, k_hidden, bits, group_size, warmup, iters)
                print(f"  M={m:<6d} p50={p50:>9.1f}us   {tflops:.3f}T")

        # --- Pipeline ---
        if not args.per_op_only:
            print()
            n_ops = n_layers_pipeline * 3
            print(f"  Pipeline ({n_layers_pipeline}L x 3 QMMs = {n_ops} ops):")
            for m in m_values:
                try:
                    p50, per_layer, per_op, tflops = bench_pipeline(
                        m, k_hidden, n_inter, bits, group_size,
                        n_layers_pipeline, warmup, iters,
                    )
                    print(
                        f"  M={m:<6d} total={p50:>9.0f}us  "
                        f"per_layer={per_layer:>8.1f}us  "
                        f"per_op={per_op:>8.1f}us  "
                        f"amortized={tflops:.3f}T"
                    )
                except Exception as e:
                    print(f"  M={m:<6d} FAILED: {e}")
                    break

        # Force cleanup between models
        gc.collect()
        print()

    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
