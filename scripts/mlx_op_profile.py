"""
MLX per-operation profiling for Qwen 3.5 MoE expert single layer.
Measures each operation individually with mx.eval() synchronization.
"""
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Qwen 3.5 MoE expert config
HIDDEN = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 2560
SEQ_LENS = [128, 256, 512, 1024]
WARMUP = 3
ITERS = 10

def profile_op(name, fn, warmup=WARMUP, iters=ITERS):
    """Profile a single operation."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        mx.eval(result) if isinstance(result, mx.array) else [mx.eval(r) for r in result]

    # Bench
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        mx.eval(result) if isinstance(result, mx.array) else [mx.eval(r) for r in result]
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # us

    mean = np.mean(times)
    std = np.std(times)
    return mean, std

def run_profile(seq_len):
    print(f"\n{'='*80}")
    print(f"seq_len={seq_len}")
    print(f"{'='*80}")

    mx.random.seed(42)

    # Create inputs
    x = mx.random.normal((seq_len, HIDDEN)) * 0.02
    x = x.astype(mx.float16)
    mx.eval(x)

    # Weights (random, f16)
    # QKV
    wq = mx.random.normal((HIDDEN, HIDDEN)) * 0.02
    wk = mx.random.normal((HIDDEN, NUM_KV_HEADS * HEAD_DIM)) * 0.02
    wv = mx.random.normal((HIDDEN, NUM_KV_HEADS * HEAD_DIM)) * 0.02
    wo = mx.random.normal((HIDDEN, HIDDEN)) * 0.02
    # FFN
    w_gate = mx.random.normal((HIDDEN, INTERMEDIATE)) * 0.02
    w_up = mx.random.normal((HIDDEN, INTERMEDIATE)) * 0.02
    w_down = mx.random.normal((INTERMEDIATE, HIDDEN)) * 0.02
    # Norms
    norm_w = mx.ones((HIDDEN,))

    wq, wk, wv, wo = [w.astype(mx.float16) for w in [wq, wk, wv, wo]]
    w_gate, w_up, w_down = [w.astype(mx.float16) for w in [w_gate, w_up, w_down]]
    norm_w = norm_w.astype(mx.float16)
    mx.eval(wq, wk, wv, wo, w_gate, w_up, w_down, norm_w)

    # Build causal mask
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(mx.float16)
    mx.eval(mask)

    # RoPE frequencies
    freqs = mx.exp(mx.arange(0, HEAD_DIM, 2, dtype=mx.float32) * (-np.log(1000000.0) / HEAD_DIM))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    mx.eval(freqs)

    results = []

    # 1. RMSNorm
    def rms_norm():
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        return (x * mx.rsqrt(variance + 1e-5) * norm_w)
    mean, std = profile_op("RMSNorm", rms_norm)
    normed = rms_norm(); mx.eval(normed)
    results.append(("RMSNorm (pre-attn)", mean, std))

    # 2. Q projection
    mean, std = profile_op("Q proj GEMM", lambda: normed @ wq.T)
    q = normed @ wq.T; mx.eval(q)
    results.append(("Q proj GEMM", mean, std))

    # 3. K projection
    mean, std = profile_op("K proj GEMM", lambda: normed @ wk)
    k = normed @ wk; mx.eval(k)
    results.append(("K proj GEMM", mean, std))

    # 4. V projection
    mean, std = profile_op("V proj GEMM", lambda: normed @ wv)
    v = normed @ wv; mx.eval(v)
    results.append(("V proj GEMM", mean, std))

    # 5. Reshape Q/K/V to multi-head format
    def reshape_qkv():
        q_r = q.reshape(1, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k_r = k.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v_r = v.reshape(1, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        return q_r, k_r, v_r
    mean, std = profile_op("Reshape QKV", reshape_qkv)
    q_mh, k_mh, v_mh = reshape_qkv(); mx.eval(q_mh, k_mh, v_mh)
    results.append(("Reshape QKV", mean, std))

    # 6. RoPE
    def apply_rope():
        cos_f = mx.cos(freqs).astype(mx.float16)
        sin_f = mx.sin(freqs).astype(mx.float16)
        q1, q2 = q_mh[..., :HEAD_DIM//2], q_mh[..., HEAD_DIM//2:]
        k1, k2 = k_mh[..., :HEAD_DIM//2], k_mh[..., HEAD_DIM//2:]
        q_rot = mx.concatenate([q1 * cos_f - q2 * sin_f, q1 * sin_f + q2 * cos_f], axis=-1)
        k_rot = mx.concatenate([k1 * cos_f - k2 * sin_f, k1 * sin_f + k2 * cos_f], axis=-1)
        return q_rot, k_rot
    mean, std = profile_op("RoPE Q+K", apply_rope)
    q_rope, k_rope = apply_rope(); mx.eval(q_rope, k_rope)
    results.append(("RoPE Q+K", mean, std))

    # 7. SDPA
    scale = HEAD_DIM ** -0.5
    def sdpa():
        return mx.fast.scaled_dot_product_attention(
            q_rope, k_rope, v_mh, scale=scale, mask=mask
        )
    mean, std = profile_op("SDPA", sdpa)
    attn_out = sdpa(); mx.eval(attn_out)
    results.append(("SDPA", mean, std))

    # 8. Reshape output
    def reshape_out():
        return attn_out.transpose(0, 2, 1, 3).reshape(seq_len, HIDDEN)
    mean, std = profile_op("Reshape out", reshape_out)
    attn_flat = reshape_out(); mx.eval(attn_flat)
    results.append(("Reshape out", mean, std))

    # 9. O projection
    mean, std = profile_op("O proj GEMM", lambda: attn_flat @ wo.T)
    o_out = attn_flat @ wo.T; mx.eval(o_out)
    results.append(("O proj GEMM", mean, std))

    # 10. Residual add
    mean, std = profile_op("Residual add 1", lambda: x + o_out)
    residual1 = x + o_out; mx.eval(residual1)
    results.append(("Residual add 1", mean, std))

    # 11. RMSNorm (pre-FFN)
    def rms_norm2():
        variance = mx.mean(mx.square(residual1.astype(mx.float32)), axis=-1, keepdims=True)
        return (residual1 * mx.rsqrt(variance + 1e-5) * norm_w)
    mean, std = profile_op("RMSNorm (pre-FFN)", rms_norm2)
    normed2 = rms_norm2(); mx.eval(normed2)
    results.append(("RMSNorm (pre-FFN)", mean, std))

    # 12. Gate projection
    mean, std = profile_op("Gate proj GEMM", lambda: normed2 @ w_gate)
    gate = normed2 @ w_gate; mx.eval(gate)
    results.append(("Gate proj GEMM", mean, std))

    # 13. Up projection
    mean, std = profile_op("Up proj GEMM", lambda: normed2 @ w_up)
    up = normed2 @ w_up; mx.eval(up)
    results.append(("Up proj GEMM", mean, std))

    # 14. SiLU * gate
    def silu_mul():
        return nn.silu(gate) * up
    mean, std = profile_op("SiLU * mul", silu_mul)
    ffn_act = silu_mul(); mx.eval(ffn_act)
    results.append(("SiLU * mul", mean, std))

    # 15. Down projection
    mean, std = profile_op("Down proj GEMM", lambda: ffn_act @ w_down)
    down = ffn_act @ w_down; mx.eval(down)
    results.append(("Down proj GEMM", mean, std))

    # 16. Residual add
    mean, std = profile_op("Residual add 2", lambda: residual1 + down)
    results.append(("Residual add 2", mean, std))

    # Print table
    total = sum(r[1] for r in results)
    print(f"\n{'Operation':<25} {'Mean (us)':>10} {'Std':>8} {'%':>6}")
    print("-" * 55)
    for name, mean, std in results:
        pct = mean / total * 100
        print(f"{name:<25} {mean:>10.1f} {std:>8.1f} {pct:>5.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<25} {total:>10.1f}")

    return results

if __name__ == "__main__":
    print("MLX Per-Operation Profiling — Qwen 3.5 MoE Expert Single Layer")
    print(f"Config: hidden={HIDDEN}, heads={NUM_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}, ffn={INTERMEDIATE}")

    for sl in SEQ_LENS:
        run_profile(sl)
