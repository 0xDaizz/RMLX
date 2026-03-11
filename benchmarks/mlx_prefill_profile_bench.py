#!/usr/bin/env python3
"""Per-operation prefill profiling for MLX (Llama-3 8B config).

Times each sub-operation individually within a single transformer layer
in prefill mode, with mx.eval() synchronization between each op.
Designed for 1:1 comparison with RMLX per-op profiling.

Usage:
    python mlx_prefill_profile_bench.py [--warmup N] [--iters N] [--seq-lens 128,256,512]
"""

import argparse
import platform
import time
import statistics

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Llama-3 8B config
# ---------------------------------------------------------------------------
CONFIG = {
    "hidden_size": 4096,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
}

DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048]
WARMUP = 3
ITERS = 15


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_op(warmup, iters, fn):
    """Time a function that produces mx array(s).

    Handles both single arrays and tuples of arrays.
    Returns (mean, p50, min) in microseconds.
    """
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> us

    times.sort()
    mean = statistics.mean(times)
    p50 = times[len(times) // 2]
    mn = times[0]
    return mean, p50, mn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLX per-operation prefill profiling (Llama-3 8B config)"
    )
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup iterations (default {WARMUP})")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help=f"Benchmark iterations (default {ITERS})")
    parser.add_argument("--seq-lens", type=str, default=None,
                        help="Comma-separated seq_len values (default: 128,256,512,1024,2048)")
    args = parser.parse_args()

    warmup = args.warmup
    iters = args.iters
    seq_lens = DEFAULT_SEQ_LENS
    if args.seq_lens:
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    cfg = CONFIG
    H = cfg["hidden_size"]
    Nh = cfg["num_heads"]
    Nkv = cfg["num_kv_heads"]
    D = cfg["head_dim"]
    I = cfg["intermediate_size"]
    scale = D ** -0.5

    # Build components
    rms_norm_1 = nn.RMSNorm(H, eps=cfg["rms_norm_eps"])
    rms_norm_2 = nn.RMSNorm(H, eps=cfg["rms_norm_eps"])
    q_proj = nn.Linear(H, Nh * D, bias=False)
    k_proj = nn.Linear(H, Nkv * D, bias=False)
    v_proj = nn.Linear(H, Nkv * D, bias=False)
    o_proj = nn.Linear(Nh * D, H, bias=False)
    gate_proj = nn.Linear(H, I, bias=False)
    up_proj = nn.Linear(H, I, bias=False)
    down_proj = nn.Linear(I, H, bias=False)
    rope = nn.RoPE(D, traditional=False, base=cfg["rope_theta"])

    # Cast all weights to f16
    for m in [rms_norm_1, rms_norm_2, q_proj, k_proj, v_proj, o_proj,
              gate_proj, up_proj, down_proj]:
        m.update(m.apply(lambda p: p.astype(mx.float16)))
    mx.eval(
        rms_norm_1.parameters(), rms_norm_2.parameters(),
        q_proj.parameters(), k_proj.parameters(), v_proj.parameters(),
        o_proj.parameters(), gate_proj.parameters(), up_proj.parameters(),
        down_proj.parameters(),
    )

    # Header
    print("=" * 80)
    print("Per-Operation Prefill Profile (MLX, Llama-3 8B, f16)")
    print("=" * 80)
    print(f"  mlx version   : {mx.__version__}")
    print(f"  device        : {mx.default_device()}")
    print(f"  platform      : {platform.platform()}")
    print(f"  python        : {platform.python_version()}")
    print(f"  dtype         : float16")
    print(f"  hidden_size   : {H}")
    print(f"  num_heads     : {Nh} (kv_heads={Nkv})")
    print(f"  head_dim      : {D}")
    print(f"  intermediate  : {I}")
    print(f"  warmup        : {warmup}")
    print(f"  iters         : {iters}")
    print(f"  seq_lens      : {seq_lens}")
    print("=" * 80)

    for seq_len in seq_lens:
        B = 1

        # Input tensor
        x = mx.random.normal(shape=(B, seq_len, H)).astype(mx.float16)
        mx.eval(x)

        # Pre-compute intermediates so each op is timed in isolation
        normed = rms_norm_1(x)
        mx.eval(normed)

        q = q_proj(normed).reshape(B, seq_len, Nh, D).transpose(0, 2, 1, 3)
        k = k_proj(normed).reshape(B, seq_len, Nkv, D).transpose(0, 2, 1, 3)
        v = v_proj(normed).reshape(B, seq_len, Nkv, D).transpose(0, 2, 1, 3)
        mx.eval(q, k, v)

        q_roped = rope(q)
        k_roped = rope(k)
        mx.eval(q_roped, k_roped)

        attn_out = mx.fast.scaled_dot_product_attention(
            q_roped, k_roped, v, scale=scale
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, seq_len, -1)
        mx.eval(attn_out)

        o_out = o_proj(attn_out)
        mx.eval(o_out)

        h = x + o_out
        mx.eval(h)

        normed2 = rms_norm_2(h)
        mx.eval(normed2)

        gate_out = gate_proj(normed2)
        up_out = up_proj(normed2)
        mx.eval(gate_out, up_out)

        silu_out = nn.silu(gate_out) * up_out
        mx.eval(silu_out)

        ffn_out = down_proj(silu_out)
        mx.eval(ffn_out)

        # ----- Per-op timing -----
        print(f"\n--- seq_len={seq_len} ---")
        print(f"{'operation':>20s} {'mean(us)':>10s} {'p50(us)':>10s} {'min(us)':>10s}")
        print("-" * 55)

        results = {}

        # 1. RMSNorm 1
        m, p, mn = time_op(warmup, iters, lambda: rms_norm_1(x))
        results["rms_norm_1"] = m
        print(f"{'rms_norm_1':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 2. QKV projection (3 separate GEMMs)
        def qkv_fn():
            q_ = q_proj(normed)
            k_ = k_proj(normed)
            v_ = v_proj(normed)
            return (q_, k_, v_)

        m, p, mn = time_op(warmup, iters, qkv_fn)
        results["qkv_proj"] = m
        print(f"{'qkv_proj':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 3. RoPE
        # Use pre-computed reshaped Q/K for isolated RoPE timing
        q_for_rope = q_proj(normed).reshape(B, seq_len, Nh, D).transpose(0, 2, 1, 3)
        k_for_rope = k_proj(normed).reshape(B, seq_len, Nkv, D).transpose(0, 2, 1, 3)
        mx.eval(q_for_rope, k_for_rope)

        m, p, mn = time_op(warmup, iters,
                           lambda: (rope(q_for_rope), rope(k_for_rope)))
        results["rope"] = m
        print(f"{'rope':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 4. SDPA
        m, p, mn = time_op(warmup, iters,
                           lambda: mx.fast.scaled_dot_product_attention(
                               q_roped, k_roped, v, scale=scale))
        results["sdpa"] = m
        print(f"{'sdpa':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 5. O projection
        attn_flat = mx.fast.scaled_dot_product_attention(
            q_roped, k_roped, v, scale=scale
        )
        attn_flat = attn_flat.transpose(0, 2, 1, 3).reshape(B, seq_len, -1)
        mx.eval(attn_flat)

        m, p, mn = time_op(warmup, iters, lambda: o_proj(attn_flat))
        results["o_proj"] = m
        print(f"{'o_proj':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 6. Residual 1
        m, p, mn = time_op(warmup, iters, lambda: x + o_out)
        results["residual_1"] = m
        print(f"{'residual_1':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 7. RMSNorm 2
        m, p, mn = time_op(warmup, iters, lambda: rms_norm_2(h))
        results["rms_norm_2"] = m
        print(f"{'rms_norm_2':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 8. FFN gate + up (2 separate GEMMs)
        m, p, mn = time_op(warmup, iters,
                           lambda: (gate_proj(normed2), up_proj(normed2)))
        results["ffn_gate_up"] = m
        print(f"{'ffn_gate_up':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 9. SiLU * mul
        gate_pre = gate_proj(normed2)
        up_pre = up_proj(normed2)
        mx.eval(gate_pre, up_pre)

        m, p, mn = time_op(warmup, iters,
                           lambda: nn.silu(gate_pre) * up_pre)
        results["silu_mul"] = m
        print(f"{'silu_mul':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 10. FFN down
        m, p, mn = time_op(warmup, iters, lambda: down_proj(silu_out))
        results["ffn_down"] = m
        print(f"{'ffn_down':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # 11. Residual 2
        m, p, mn = time_op(warmup, iters, lambda: h + ffn_out)
        results["residual_2"] = m
        print(f"{'residual_2':>20s} {m:>9.1f} {p:>9.1f} {mn:>9.1f}")

        # Sum of parts
        total_parts = sum(results.values())
        print("-" * 55)
        print(f"{'sum_of_parts':>20s} {total_parts:>9.1f}")

        # 12. Total eager (single eval at the end, no per-op sync)
        def full_eager():
            n = rms_norm_1(x)
            q_ = q_proj(n).reshape(B, seq_len, Nh, D).transpose(0, 2, 1, 3)
            k_ = k_proj(n).reshape(B, seq_len, Nkv, D).transpose(0, 2, 1, 3)
            v_ = v_proj(n).reshape(B, seq_len, Nkv, D).transpose(0, 2, 1, 3)
            q_ = rope(q_)
            k_ = rope(k_)
            a = mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(B, seq_len, -1)
            o = o_proj(a)
            h_ = x + o
            n2 = rms_norm_2(h_)
            g = gate_proj(n2)
            u = up_proj(n2)
            s = nn.silu(g) * u
            d = down_proj(s)
            return h_ + d

        m_eager, p_eager, mn_eager = time_op(warmup, iters, full_eager)
        print(f"{'total_eager':>20s} {m_eager:>9.1f} {p_eager:>9.1f} {mn_eager:>9.1f}")
        overhead_pct = ((total_parts - m_eager) / m_eager * 100
                        if m_eager > 0 else 0)
        print(f"  sync overhead: {overhead_pct:+.1f}% (sum_of_parts vs total_eager)")

        # 13. Total compiled
        compiled_fn = mx.compile(full_eager)
        m_comp, p_comp, mn_comp = time_op(warmup + 5, iters, compiled_fn)
        print(f"{'total_compiled':>20s} {m_comp:>9.1f} {p_comp:>9.1f} {mn_comp:>9.1f}")
        if m_eager > 0:
            compile_speedup = m_eager / m_comp
            print(f"  compile speedup: {compile_speedup:.2f}x vs eager")

        # Percentage breakdown
        print(f"\n  --- Breakdown (% of sum_of_parts) ---")
        for name, val in results.items():
            pct = val / total_parts * 100 if total_parts > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {name:>20s}: {pct:5.1f}%  {bar}")

    print()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
