#!/usr/bin/env python3
"""MLX Per-Operation Prefill Benchmark — isolate GPU time for every op in a transformer layer.

Forces mx.eval() after each operation to measure wall-clock time per kernel.
This adds sync overhead but gives accurate relative timing between ops.

Usage:
    python mlx_per_op_bench.py              # default seq_lens
    python mlx_per_op_bench.py 128          # single seq_len
    python mlx_per_op_bench.py 32 128 512   # specific seq_lens
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

# ── Config (matches RMLX exactly) ──────────────────────────────────────────
HIDDEN_SIZE = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_DIM = 2560
RMS_NORM_EPS = 1e-5
ROPE_THETA = 1000000.0

WARMUP_ITERS = 5
BENCH_ITERS = 10

# ── FLOP calculations ──────────────────────────────────────────────────────
def gemm_flops(M, N, K):
    """2*M*N*K for a single matmul."""
    return 2.0 * M * N * K

def tflops(flops, time_us):
    """Convert flops + time_us to TFLOPS."""
    if time_us <= 0:
        return 0.0
    return flops / (time_us / 1e6) / 1e12


# ── Helpers ─────────────────────────────────────────────────────────────────
def time_op(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time a single operation with eval sync, return median time in us."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        mx.eval(result)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]  # median


def time_op_multi(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time an operation that returns multiple arrays."""
    for _ in range(warmup):
        results = fn()
        mx.eval(*results) if isinstance(results, (list, tuple)) else mx.eval(results)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        results = fn()
        mx.eval(*results) if isinstance(results, (list, tuple)) else mx.eval(results)
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]


# ── Build layer components ─────────────────────────────────────────────────
def build_layer():
    """Create all layer components as individual nn.Modules/params."""
    components = {}

    # Attention weights
    components["q_proj"] = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
    components["k_proj"] = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
    components["v_proj"] = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
    components["o_proj"] = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
    components["rope"] = nn.RoPE(HEAD_DIM, traditional=True, base=ROPE_THETA)

    # Norms
    components["input_layernorm"] = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
    components["post_attn_layernorm"] = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)

    # FFN weights
    components["gate_proj"] = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
    components["up_proj"] = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_DIM, bias=False)
    components["down_proj"] = nn.Linear(INTERMEDIATE_DIM, HIDDEN_SIZE, bias=False)

    # Set all to float16
    for key, mod in components.items():
        if hasattr(mod, "set_dtype"):
            mod.set_dtype(mx.float16)

    # Force initialization
    x_init = mx.random.normal((1, HIDDEN_SIZE)).astype(mx.float16)
    x_inter = mx.random.normal((1, INTERMEDIATE_DIM)).astype(mx.float16)
    for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]:
        mx.eval(components[key](x_init))
    mx.eval(components["down_proj"](x_inter))
    mx.eval(components["input_layernorm"](x_init))
    mx.eval(components["post_attn_layernorm"](x_init))

    return components


@mx.compile
def _swiglu(gate, up):
    return nn.silu(gate) * up


# ── Per-op benchmark ───────────────────────────────────────────────────────
def bench_seq_len(comp, seq_len):
    """Benchmark every individual op for a given seq_len. Returns list of (name, time_us, flops_or_none)."""

    scale = HEAD_DIM ** -0.5

    # Pre-generate inputs (eval them so they're materialized)
    x = mx.random.normal((1, seq_len, HIDDEN_SIZE)).astype(mx.float16)
    mx.eval(x)

    # Create causal mask as array
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(mx.float16)
    mx.eval(mask)

    # Pre-compute intermediates for each op's input
    normed_attn = mx.fast.rms_norm(x, comp["input_layernorm"].weight, RMS_NORM_EPS)
    mx.eval(normed_attn)

    q_raw = comp["q_proj"](normed_attn)
    k_raw = comp["k_proj"](normed_attn)
    v_raw = comp["v_proj"](normed_attn)
    mx.eval(q_raw, k_raw, v_raw)

    q_4d = q_raw.reshape(1, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_4d = k_raw.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_4d = v_raw.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    mx.eval(q_4d, k_4d, v_4d)

    q_roped = comp["rope"](q_4d)
    k_roped = comp["rope"](k_4d)
    mx.eval(q_roped, k_roped)

    sdpa_out = mx.fast.scaled_dot_product_attention(q_roped, k_roped, v_4d, scale=scale, mask=mask)
    mx.eval(sdpa_out)

    o_reshaped = sdpa_out.transpose(0, 2, 1, 3).reshape(1, seq_len, NUM_HEADS * HEAD_DIM)
    mx.eval(o_reshaped)

    o_proj_out = comp["o_proj"](o_reshaped)
    mx.eval(o_proj_out)

    attn_out = x + o_proj_out
    mx.eval(attn_out)

    normed_ffn = mx.fast.rms_norm(attn_out, comp["post_attn_layernorm"].weight, RMS_NORM_EPS)
    mx.eval(normed_ffn)

    gate_out = comp["gate_proj"](normed_ffn)
    up_out = comp["up_proj"](normed_ffn)
    mx.eval(gate_out, up_out)

    swiglu_out = _swiglu(gate_out, up_out)
    mx.eval(swiglu_out)

    down_out = comp["down_proj"](swiglu_out)
    mx.eval(down_out)

    # ── Now measure each op individually ───────────────────────────────
    results = []
    S = seq_len
    H = HIDDEN_SIZE
    HD = HEAD_DIM

    # 1. RMSNorm (pre-attention)
    t = time_op(lambda: mx.fast.rms_norm(x, comp["input_layernorm"].weight, RMS_NORM_EPS))
    results.append(("1. RMSNorm (pre-attn)", t, None))

    # 2. Q projection: [1, S, 3584] -> [1, S, 3584]
    q_flops = gemm_flops(S, NUM_HEADS * HD, H)
    t = time_op(lambda: comp["q_proj"](normed_attn))
    results.append(("2. Q proj", t, q_flops))

    # 3. K projection: [1, S, 3584] -> [1, S, 512]
    k_flops = gemm_flops(S, NUM_KV_HEADS * HD, H)
    t = time_op(lambda: comp["k_proj"](normed_attn))
    results.append(("3. K proj", t, k_flops))

    # 4. V projection: [1, S, 3584] -> [1, S, 512]
    v_flops = gemm_flops(S, NUM_KV_HEADS * HD, H)
    t = time_op(lambda: comp["v_proj"](normed_attn))
    results.append(("4. V proj", t, v_flops))

    # 5. Q reshape+transpose
    t = time_op(lambda: q_raw.reshape(1, S, NUM_HEADS, HD).transpose(0, 2, 1, 3))
    results.append(("5. Q reshape+transpose", t, None))

    # 6. K reshape+transpose
    t = time_op(lambda: k_raw.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3))
    results.append(("6. K reshape+transpose", t, None))

    # 7. V reshape+transpose
    t = time_op(lambda: v_raw.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3))
    results.append(("7. V reshape+transpose", t, None))

    # 8. RoPE Q
    t = time_op(lambda: comp["rope"](q_4d))
    results.append(("8. RoPE Q", t, None))

    # 9. RoPE K
    t = time_op(lambda: comp["rope"](k_4d))
    results.append(("9. RoPE K", t, None))

    # 10. SDPA (GQA native: Q=28 heads, K/V=4 heads)
    # FLOPS: 2 * B * Nh * S * S * Hd (QK^T) + 2 * B * Nh * S * S * Hd (attn@V)
    sdpa_flops = 2 * (2.0 * 1 * NUM_HEADS * S * S * HD)
    t = time_op(lambda: mx.fast.scaled_dot_product_attention(q_roped, k_roped, v_4d, scale=scale, mask=mask))
    results.append(("10. SDPA", t, sdpa_flops))

    # 11. Output reshape+transpose
    t = time_op(lambda: sdpa_out.transpose(0, 2, 1, 3).reshape(1, S, NUM_HEADS * HD))
    results.append(("11. O reshape+transpose", t, None))

    # 12. O projection: [1, S, 3584] -> [1, S, 3584]
    o_flops = gemm_flops(S, H, NUM_HEADS * HD)
    t = time_op(lambda: comp["o_proj"](o_reshaped))
    results.append(("12. O proj", t, o_flops))

    # 13. Residual add (attention)
    t = time_op(lambda: x + o_proj_out)
    results.append(("13. Residual add (attn)", t, None))

    # 14. RMSNorm (pre-FFN)
    t = time_op(lambda: mx.fast.rms_norm(attn_out, comp["post_attn_layernorm"].weight, RMS_NORM_EPS))
    results.append(("14. RMSNorm (pre-FFN)", t, None))

    # 15. Gate projection: [1, S, 3584] -> [1, S, 2560]
    gate_flops = gemm_flops(S, INTERMEDIATE_DIM, H)
    t = time_op(lambda: comp["gate_proj"](normed_ffn))
    results.append(("15. Gate proj", t, gate_flops))

    # 16. Up projection: [1, S, 3584] -> [1, S, 2560]
    up_flops = gemm_flops(S, INTERMEDIATE_DIM, H)
    t = time_op(lambda: comp["up_proj"](normed_ffn))
    results.append(("16. Up proj", t, up_flops))

    # 17. SiLU(gate) * up
    t = time_op(lambda: _swiglu(gate_out, up_out))
    results.append(("17. SiLU*up (fused)", t, None))

    # 18. Down projection: [1, S, 2560] -> [1, S, 3584]
    down_flops = gemm_flops(S, H, INTERMEDIATE_DIM)
    t = time_op(lambda: comp["down_proj"](swiglu_out))
    results.append(("18. Down proj", t, down_flops))

    # 19. Residual add (FFN)
    t = time_op(lambda: attn_out + down_out)
    results.append(("19. Residual add (FFN)", t, None))

    # ── Grouped measurements ───────────────────────────────────────────
    grouped = []

    # QKV merged (ops 2+3+4)
    def qkv_merged():
        q = comp["q_proj"](normed_attn)
        k = comp["k_proj"](normed_attn)
        v = comp["v_proj"](normed_attn)
        return (q, k, v)
    qkv_flops = q_flops + k_flops + v_flops
    t = time_op_multi(qkv_merged)
    grouped.append(("QKV merged", t, qkv_flops))

    # RoPE Q+K (ops 8+9)
    def rope_qk():
        rq = comp["rope"](q_4d)
        rk = comp["rope"](k_4d)
        return (rq, rk)
    t = time_op_multi(rope_qk)
    grouped.append(("RoPE Q+K", t, None))

    # Full attention (ops 1-13)
    def full_attention():
        n = mx.fast.rms_norm(x, comp["input_layernorm"].weight, RMS_NORM_EPS)
        q = comp["q_proj"](n)
        k = comp["k_proj"](n)
        v = comp["v_proj"](n)
        q = q.reshape(1, S, NUM_HEADS, HD).transpose(0, 2, 1, 3)
        k = k.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3)
        v = v.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3)
        q = comp["rope"](q)
        k = comp["rope"](k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(1, S, NUM_HEADS * HD)
        o = comp["o_proj"](o)
        return x + o
    t = time_op(full_attention)
    grouped.append(("Full attention (1-13)", t, None))

    # Full FFN (ops 14-19)
    def full_ffn():
        n = mx.fast.rms_norm(attn_out, comp["post_attn_layernorm"].weight, RMS_NORM_EPS)
        g = comp["gate_proj"](n)
        u = comp["up_proj"](n)
        h = _swiglu(g, u)
        d = comp["down_proj"](h)
        return attn_out + d
    t = time_op(full_ffn)
    grouped.append(("Full FFN (14-19)", t, None))

    # Full layer (ops 1-19)
    def full_layer():
        # Attention block
        n = mx.fast.rms_norm(x, comp["input_layernorm"].weight, RMS_NORM_EPS)
        q = comp["q_proj"](n)
        k = comp["k_proj"](n)
        v = comp["v_proj"](n)
        q = q.reshape(1, S, NUM_HEADS, HD).transpose(0, 2, 1, 3)
        k = k.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3)
        v = v.reshape(1, S, NUM_KV_HEADS, HD).transpose(0, 2, 1, 3)
        q = comp["rope"](q)
        k = comp["rope"](k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(1, S, NUM_HEADS * HD)
        o = comp["o_proj"](o)
        h = x + o
        # FFN block
        n2 = mx.fast.rms_norm(h, comp["post_attn_layernorm"].weight, RMS_NORM_EPS)
        g = comp["gate_proj"](n2)
        u = comp["up_proj"](n2)
        hidden = _swiglu(g, u)
        d = comp["down_proj"](hidden)
        return h + d
    t = time_op(full_layer)
    grouped.append(("Full layer (1-19)", t, None))

    return results, grouped


# ── Output formatting ──────────────────────────────────────────────────────
def print_table(results, grouped, seq_len):
    """Print markdown table for a single seq_len."""
    total_time = sum(t for _, t, _ in results)

    print(f"\n## seq_len = {seq_len}")
    print()
    print(f"| # | Op | Time (us) | % of total | TFLOPS |")
    print(f"|---|-----|----------:|----------:|---------:|")

    for name, t, flops in results:
        pct = t / total_time * 100
        tf_str = f"{tflops(flops, t):.2f}" if flops else "—"
        print(f"| | {name} | {t:.1f} | {pct:.1f}% | {tf_str} |")

    print(f"| | **Sum of individual ops** | **{total_time:.1f}** | **100%** | |")
    print()

    # Grouped
    print(f"| | **Grouped** | **Time (us)** | | **TFLOPS** |")
    print(f"|---|-------------|------------:|---|---------:|")
    for name, t, flops in grouped:
        tf_str = f"{tflops(flops, t):.2f}" if flops else "—"
        print(f"| | {name} | {t:.1f} | | {tf_str} |")

    print()


def print_summary(all_results):
    """Print summary comparison across all seq_lens."""
    print("\n## Summary — key ops across seq_lens")
    print()

    # Key ops to track
    key_ops = [
        "1. RMSNorm (pre-attn)",
        "2. Q proj",
        "3. K proj",
        "4. V proj",
        "8. RoPE Q",
        "10. SDPA",
        "12. O proj",
        "14. RMSNorm (pre-FFN)",
        "15. Gate proj",
        "16. Up proj",
        "17. SiLU*up (fused)",
        "18. Down proj",
    ]

    seq_lens = [s for s, _, _ in all_results]
    header = "| Op |" + " | ".join(f" S={s} (us) " for s in seq_lens) + " |"
    sep = "|-----|" + " | ".join("----------:" for _ in seq_lens) + " |"
    print(header)
    print(sep)

    for op_name in key_ops:
        row = f"| {op_name} |"
        for _, results, _ in all_results:
            val = next((t for n, t, _ in results if n == op_name), None)
            row += f" {val:.1f} |" if val else " — |"
        print(row)

    # Grouped summary
    grouped_keys = ["Full attention (1-13)", "Full FFN (14-19)", "Full layer (1-19)"]
    print("|---|" + " | ".join("---" for _ in seq_lens) + " |")
    for gname in grouped_keys:
        row = f"| **{gname}** |"
        for _, _, grouped in all_results:
            val = next((t for n, t, _ in grouped if n == gname), None)
            row += f" **{val:.1f}** |" if val else " — |"
        print(row)

    print()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        seq_lens = [int(s) for s in sys.argv[1:]]
    else:
        seq_lens = [32, 128, 256, 512, 1024]

    print("# MLX Per-Operation Prefill Benchmark")
    print()
    print(
        f"Config: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}/{NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_DIM}"
    )
    print(f"dtype: float16, warmup={WARMUP_ITERS}, bench={BENCH_ITERS}")
    print(f"NOTE: mx.eval() after EACH op — adds sync overhead but gives relative timing")
    print()

    comp = build_layer()

    all_results = []
    for seq_len in seq_lens:
        print(f"Benchmarking seq_len={seq_len}...", end=" ", flush=True)
        results, grouped = bench_seq_len(comp, seq_len)
        all_results.append((seq_len, results, grouped))
        total = sum(t for _, t, _ in results)
        layer_t = next((t for n, t, _ in grouped if n == "Full layer (1-19)"), 0)
        print(f"done (sum={total:.0f}us, fused={layer_t:.0f}us)")

    # Print detailed tables
    for seq_len, results, grouped in all_results:
        print_table(results, grouped, seq_len)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
