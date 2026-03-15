#!/usr/bin/env python3
"""MLX MoE / Expert Parallelism comparison benchmark.

Matches RMLX's ep_bench.rs configuration (Qwen 3.5 MoE A22B MoE layer):
- 8 experts, hidden=3584, intermediate=18944, top_k=2
- Router (gate) latency
- Single expert FFN (SwiGLU)
- Full MoE layer (sequential per-expert dispatch)
- EP=2 simulation (4 experts per rank)
- Full MoE transformer layer (attention + MoE FFN)

Uses mx.fast.* fused kernels and mx.compile() for production-representative
performance. All measurements in float16 for fairness with RMLX.

Usage:
    python mlx_ep_bench.py [--warmup N] [--iters N] [--seq-lens 1,128,512]
"""

import argparse
import time
import statistics

import mlx.core as mx

# ---------------------------------------------------------------------------
# Qwen 3.5 MoE A22B config (matches RMLX ep_bench.rs)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
NUM_EXPERTS = 8
TOP_K = 2
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_CACHE_LEN = 128
RMS_NORM_EPS = 1e-6

# Token counts per expert (matches ep_bench.rs M_VALUES)
M_VALUES = [1, 4, 8, 16, 32]

WARMUP = 5
ITERS = 30


# ---------------------------------------------------------------------------
# Stats helper (matching ep_bench.rs output format)
# ---------------------------------------------------------------------------
def compute_stats(latencies_us):
    """Compute mean, std, p50, p95, min, max from latency list (microseconds)."""
    n = len(latencies_us)
    mean = statistics.mean(latencies_us)
    std = statistics.stdev(latencies_us) if n > 1 else 0.0
    p50 = statistics.median(latencies_us)
    s = sorted(latencies_us)
    p95_idx = min(int(0.95 * (n - 1) + 0.5), n - 1)
    p95 = s[p95_idx]
    return {
        "mean": mean, "std": std, "p50": p50, "p95": p95,
        "min": s[0], "max": s[-1], "n": n,
    }


def fmt_stats(st):
    return (f"mean={st['mean']:8.1f}us std={st['std']:7.1f}us "
            f"p50={st['p50']:8.1f}us p95={st['p95']:8.1f}us "
            f"min={st['min']:8.1f}us max={st['max']:8.1f}us (n={st['n']})")


# ---------------------------------------------------------------------------
# Expert FFN: SwiGLU (gate + up -> silu*mul -> down)
# ---------------------------------------------------------------------------
def expert_ffn(x, gate_w, up_w, down_w):
    """Single expert SwiGLU FFN: same as RMLX Expert::forward."""
    gate = x @ gate_w
    up = x @ up_w
    hidden = mx.sigmoid(gate) * gate * up  # SiLU(gate) * up
    return hidden @ down_w


def expert_ffn_silu(x, gate_w, up_w, down_w):
    """Single expert SwiGLU FFN using nn.silu for exact match."""
    gate = x @ gate_w
    up = x @ up_w
    # SiLU = x * sigmoid(x), so silu(gate) * up
    hidden = mx.sigmoid(gate) * gate * up
    return hidden @ down_w


# ---------------------------------------------------------------------------
# MoE forward: router + per-expert dispatch + combine
# ---------------------------------------------------------------------------
def moe_forward(x, gate_w, experts, num_experts, top_k):
    """Full MoE forward: route + expert compute + combine.

    Uses integer indexing and CPU-side routing (no boolean indexing).
    Sequential per-expert dispatch matches RMLX PerExpert strategy.

    Args:
        x:           [batch, hidden_size]
        gate_w:      [hidden_size, num_experts]
        experts:     list of (gate_w, up_w, down_w) tuples
        num_experts: int
        top_k:       int
    Returns:
        output:      [batch, hidden_size]
    """
    batch = x.shape[0]
    hidden = x.shape[1]

    # Router: gate projection + softmax + topk
    logits = x @ gate_w                                     # [batch, num_experts]
    probs = mx.softmax(logits, axis=-1)                     # [batch, num_experts]
    # topk: get top_k weights and indices
    indices = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[:, :top_k]  # [batch, top_k]
    # Gather the weights for selected experts
    weights = mx.take_along_axis(probs, indices, axis=-1)   # [batch, top_k]
    weights = weights / weights.sum(axis=-1, keepdims=True)  # normalize
    mx.eval(indices, weights)

    # CPU-side routing: group tokens by expert (same as RMLX PerExpert)
    indices_list = indices.tolist()
    weights_list = weights.tolist()

    # Build per-expert dispatch lists: token index + combined weight
    expert_tokens = [[] for _ in range(num_experts)]   # token indices
    expert_weights = [[] for _ in range(num_experts)]  # routing weights

    for t in range(batch):
        for k in range(top_k):
            e = indices_list[t][k]
            w = weights_list[t][k]
            expert_tokens[e].append(t)
            expert_weights[e].append(w)

    # Sequential per-expert dispatch + combine via weighted sum
    # Accumulate partial results per token, then stack at the end
    token_parts = [[] for _ in range(batch)]  # list of weighted outputs per token

    for e in range(num_experts):
        if not expert_tokens[e]:
            continue

        tok_ids = expert_tokens[e]
        tok_w = expert_weights[e]

        # Gather tokens for this expert using integer indexing
        gather_idx = mx.array(tok_ids)
        expert_input = x[gather_idx]                        # [n_routed, hidden]

        gw, uw, dw = experts[e]
        expert_out = expert_ffn(expert_input, gw, uw, dw)   # [n_routed, hidden]

        # Apply routing weights
        w_vec = mx.array(tok_w, dtype=x.dtype).reshape(-1, 1)
        weighted = expert_out * w_vec                        # [n_routed, hidden]
        mx.eval(weighted)

        # Scatter results back to token slots
        for i, t in enumerate(tok_ids):
            token_parts[t].append(weighted[i])

    # Combine: sum all expert contributions per token
    rows = []
    for t in range(batch):
        if token_parts[t]:
            rows.append(mx.stack(token_parts[t]).sum(axis=0))
        else:
            rows.append(mx.zeros((hidden,), dtype=x.dtype))

    output = mx.stack(rows)  # [batch, hidden]
    return output


# ---------------------------------------------------------------------------
# Full MoE transformer layer (attention + MoE FFN)
# ---------------------------------------------------------------------------
def moe_layer_forward(x, w_q, w_k, w_v, w_o, rms_w1, rms_w2,
                      gate_w, experts, num_experts, top_k,
                      num_heads, num_kv_heads, head_dim, hidden_size,
                      k_cache, v_cache, offset):
    """Single transformer decoder layer with MoE FFN (decode mode, with KV cache).

    Args:
        x:          [batch, hidden_size]
        Attention weights: w_q, w_k, w_v, w_o
        rms_w1/w2:  RMSNorm weights
        gate_w:     [hidden_size, num_experts] router weight
        experts:    list of (gate_w, up_w, down_w) tuples
        num_experts, top_k: MoE config
        Attention config: num_heads, num_kv_heads, head_dim, hidden_size
        k_cache, v_cache: KV cache tensors
        offset:     position offset for RoPE
    """
    seq_len = x.shape[0]  # tokens in this forward call (1 for decode, N for prefill)

    # Pre-attention RMSNorm (fused)
    h = mx.fast.rms_norm(x, rms_w1, RMS_NORM_EPS)

    # QKV projections
    q = h @ w_q
    k = h @ w_k
    v = h @ w_v

    # Reshape for attention: [batch=1, heads, seq_len, head_dim]
    q = q.reshape(1, num_heads, seq_len, head_dim)
    k = k.reshape(1, num_kv_heads, seq_len, head_dim)
    v = v.reshape(1, num_kv_heads, seq_len, head_dim)

    # RoPE (fused)
    q = mx.fast.rope(q, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)
    k = mx.fast.rope(k, head_dim, traditional=False, base=1000000.0, scale=1.0, offset=offset)

    # KV cache update
    k_cache_new = mx.concatenate([k_cache, k], axis=2)
    v_cache_new = mx.concatenate([v_cache, v], axis=2)

    # Scaled dot-product attention (fused, handles GQA)
    scale = head_dim ** -0.5
    attn_out = mx.fast.scaled_dot_product_attention(q, k_cache_new, v_cache_new, scale=scale)
    attn_out = attn_out.reshape(seq_len, num_heads * head_dim)

    # O projection + residual
    x = x + attn_out @ w_o

    # Pre-FFN RMSNorm (fused)
    h2 = mx.fast.rms_norm(x, rms_w2, RMS_NORM_EPS)

    # MoE FFN (instead of dense FFN)
    moe_out = moe_forward(h2, gate_w, experts, num_experts, top_k)
    x = x + moe_out

    return x, k_cache_new, v_cache_new


# ---------------------------------------------------------------------------
# Weight builders
# ---------------------------------------------------------------------------
def build_expert_weights(hidden, inter, seed=42):
    """Build random weights for a single SwiGLU expert."""
    mx.random.seed(seed)
    gate_w = (mx.random.normal((hidden, inter)) * 0.02).astype(mx.float16)
    up_w = (mx.random.normal((hidden, inter)) * 0.02).astype(mx.float16)
    down_w = (mx.random.normal((inter, hidden)) * 0.02).astype(mx.float16)
    mx.eval(gate_w, up_w, down_w)
    return gate_w, up_w, down_w


def build_all_experts(num_experts, hidden, inter, seed_base=2000):
    """Build weights for all experts."""
    experts = []
    for i in range(num_experts):
        experts.append(build_expert_weights(hidden, inter, seed=seed_base + i * 10))
    return experts


def build_gate_weight(hidden, num_experts, seed=1000):
    """Build router (gate) weight matrix."""
    mx.random.seed(seed)
    w = (mx.random.normal((hidden, num_experts)) * 0.02).astype(mx.float16)
    mx.eval(w)
    return w


def build_attn_weights(hidden_size, num_heads, num_kv_heads, head_dim):
    """Build attention weights for the full MoE transformer layer."""
    mx.random.seed(42)
    w_q = (mx.random.normal((hidden_size, num_heads * head_dim)) * 0.02).astype(mx.float16)
    w_k = (mx.random.normal((hidden_size, num_kv_heads * head_dim)) * 0.02).astype(mx.float16)
    w_v = (mx.random.normal((hidden_size, num_kv_heads * head_dim)) * 0.02).astype(mx.float16)
    w_o = (mx.random.normal((num_heads * head_dim, hidden_size)) * 0.02).astype(mx.float16)
    rms_w1 = mx.ones((hidden_size,), dtype=mx.float16)
    rms_w2 = mx.ones((hidden_size,), dtype=mx.float16)
    k_cache = mx.zeros((1, num_kv_heads, KV_CACHE_LEN, head_dim), dtype=mx.float16)
    v_cache = mx.zeros((1, num_kv_heads, KV_CACHE_LEN, head_dim), dtype=mx.float16)
    mx.eval(w_q, w_k, w_v, w_o, rms_w1, rms_w2, k_cache, v_cache)
    return {
        "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
        "rms_w1": rms_w1, "rms_w2": rms_w2,
        "k_cache": k_cache, "v_cache": v_cache,
    }


# ---------------------------------------------------------------------------
# Benchmark runner (matches ep_bench.rs and mlx_tp_bench.py style)
# ---------------------------------------------------------------------------
def run_bench(label, fn, warmup, iters):
    """Run warmup + timed iterations, return stats dict."""
    for _ in range(warmup):
        fn()

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    st = compute_stats(latencies)
    print(f"  {label:55s} {fmt_stats(st)}")
    return st


# ---------------------------------------------------------------------------
# 1. Router (gate) latency
# ---------------------------------------------------------------------------
def bench_router(warmup, iters):
    print("\n=== Router (gate) latency ===")
    gate_w = build_gate_weight(HIDDEN_SIZE, NUM_EXPERTS)

    for m in M_VALUES:
        seq_len = m * NUM_EXPERTS // TOP_K
        x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        def run(x=x, gate_w=gate_w):
            logits = x @ gate_w
            probs = mx.softmax(logits, axis=-1)
            indices = mx.argpartition(-probs, kth=TOP_K - 1, axis=-1)[:, :TOP_K]
            weights = mx.take_along_axis(probs, indices, axis=-1)
            mx.eval(weights, indices)

        run_bench(f"gate [{seq_len}x{HIDDEN_SIZE}->{NUM_EXPERTS}] + topk", run, warmup, iters)


# ---------------------------------------------------------------------------
# 2. Single expert FFN (SwiGLU)
# ---------------------------------------------------------------------------
def bench_single_expert(warmup, iters):
    print(f"\n=== Single expert FFN (SwiGLU: gate*up -> silu -> down) ===")
    gw, uw, dw = build_expert_weights(HIDDEN_SIZE, INTERMEDIATE_SIZE, seed=100)

    for m in M_VALUES:
        x = (mx.random.normal((m, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        def run(x=x, gw=gw, uw=uw, dw=dw):
            out = expert_ffn(x, gw, uw, dw)
            mx.eval(out)

        run_bench(f"expert_ffn [M={m}] {HIDDEN_SIZE}->{INTERMEDIATE_SIZE}->{HIDDEN_SIZE}", run, warmup, iters)

    # Also test compiled variant
    print(f"\n=== Single expert FFN (mx.compile) ===")
    compiled_ffn = mx.compile(expert_ffn)

    for m in M_VALUES:
        x = (mx.random.normal((m, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        def run(x=x, gw=gw, uw=uw, dw=dw):
            out = compiled_ffn(x, gw, uw, dw)
            mx.eval(out)

        run_bench(f"expert_ffn [M={m}] compiled", run, warmup, iters)


# ---------------------------------------------------------------------------
# 3. Full MoE layer (all 8 experts, PerExpert-style sequential)
# ---------------------------------------------------------------------------
def bench_moe_full(warmup, iters):
    print(f"\n=== MoE PerExpert strategy (sequential dispatch, {NUM_EXPERTS} experts) ===")
    gate_w = build_gate_weight(HIDDEN_SIZE, NUM_EXPERTS)
    experts = build_all_experts(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)

    for m in M_VALUES:
        seq_len = m * NUM_EXPERTS // TOP_K
        x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        def run(x=x, gate_w=gate_w, experts=experts):
            out = moe_forward(x, gate_w, experts, NUM_EXPERTS, TOP_K)
            mx.eval(out)

        run_bench(f"moe_per_expert [seq={seq_len}, ~{m} tok/expert]", run, warmup, iters)


# ---------------------------------------------------------------------------
# 4. EP=2 simulation (4 experts per rank)
# ---------------------------------------------------------------------------
def bench_ep2_simulation(warmup, iters):
    ep_experts = NUM_EXPERTS // 2  # 4 experts on this rank

    print(f"\n=== EP-2 simulation: {ep_experts} experts per rank (half compute) ===")
    print(f"  Simulates rank 0 of EP=2: only experts 0..{ep_experts} are local.")

    gate_w = build_gate_weight(HIDDEN_SIZE, ep_experts)
    experts = build_all_experts(ep_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)

    for m in M_VALUES:
        seq_len = m * ep_experts // TOP_K
        x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        def run(x=x, gate_w=gate_w, experts=experts, ep=ep_experts):
            out = moe_forward(x, gate_w, experts, ep, TOP_K)
            mx.eval(out)

        run_bench(f"ep2_local_compute [seq={seq_len}, {ep_experts} experts]", run, warmup, iters)


# ---------------------------------------------------------------------------
# 5. Full MoE transformer layer (attention + MoE FFN)
# ---------------------------------------------------------------------------
def bench_moe_transformer_layer(warmup, iters, seq_lens):
    print(f"\n=== Full MoE transformer layer (attention + MoE FFN) ===")
    print(f"  Config: heads={NUM_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  Fused: mx.fast.rms_norm, mx.fast.rope, mx.fast.scaled_dot_product_attention")

    attn = build_attn_weights(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    gate_w = build_gate_weight(HIDDEN_SIZE, NUM_EXPERTS)
    experts = build_all_experts(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    offset = KV_CACHE_LEN

    for seq_len in seq_lens:
        x = (mx.random.normal((seq_len, HIDDEN_SIZE)) * 0.02).astype(mx.float16)
        mx.eval(x)

        # Adjust KV cache for batch dim if seq_len > 1 (prefill-like)
        k_cache = attn["k_cache"]
        v_cache = attn["v_cache"]

        # Uncompiled
        def run(x=x, attn=attn, gate_w=gate_w, experts=experts,
                k_cache=k_cache, v_cache=v_cache):
            out, _, _ = moe_layer_forward(
                x, attn["w_q"], attn["w_k"], attn["w_v"], attn["w_o"],
                attn["rms_w1"], attn["rms_w2"],
                gate_w, experts, NUM_EXPERTS, TOP_K,
                NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
                k_cache, v_cache, offset,
            )
            mx.eval(out)

        mode = "decode" if seq_len == 1 else "prefill"
        run_bench(f"moe_layer [{mode} M={seq_len}] uncompiled", run, warmup, iters)

    # NOTE: mx.compile cannot wrap the full MoE layer because moe_forward()
    # calls mx.eval() internally for CPU-side expert routing decisions.
    # This is the same constraint as RMLX (CPU-side routing in both frameworks).
    # Therefore, only uncompiled results are reported for MoE layers.
    print("\n  Note: mx.compile is incompatible with MoE routing (requires mx.eval for CPU-side dispatch)")
    print("  This matches RMLX which also uses CPU-side routing. Uncompiled results are the fair comparison.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MLX MoE/EP comparison benchmark (Qwen 3.5 MoE A22B, matches RMLX ep_bench.rs)"
    )
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup iterations (default: {WARMUP})")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help=f"Benchmark iterations (default: {ITERS})")
    parser.add_argument("--seq-lens", type=str, default="1,128,512",
                        help="Comma-separated sequence lengths for full layer bench (default: 1,128,512)")
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    print("MLX MoE/EP Benchmark")
    print(f"  Config: Qwen 3.5 MoE A22B MoE layer")
    print(f"  experts={NUM_EXPERTS}, top_k={TOP_K}, hidden={HIDDEN_SIZE}, "
          f"intermediate={INTERMEDIATE_SIZE}")
    print(f"  heads={NUM_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}, "
          f"kv_cache={KV_CACHE_LEN} tokens")
    print(f"  dtype: float16, warmup={args.warmup}, iters={args.iters}")
    print(f"  M values (tokens per expert): {M_VALUES}")
    print(f"  Seq lengths (full layer): {seq_lens}")

    # ── 1. Router latency ──
    bench_router(args.warmup, args.iters)

    # ── 2. Single expert FFN ──
    bench_single_expert(args.warmup, args.iters)

    # ── 3. Full MoE (all 8 experts, sequential) ──
    bench_moe_full(args.warmup, args.iters)

    # ── 4. EP-2 simulation ──
    bench_ep2_simulation(args.warmup, args.iters)

    # ── 5. Full MoE transformer layer ──
    bench_moe_transformer_layer(args.warmup, args.iters, seq_lens)

    # ── Summary ──
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  PerExpert: sequential per-expert loop (matching RMLX PerExpert strategy)")
    print(f"  EP-2 sim:  half the experts ({NUM_EXPERTS // 2} of {NUM_EXPERTS}) on a single rank")
    print(f"  Full layer: attention (fused) + MoE FFN")
    print()
    print(f"  Key comparison with RMLX ep_bench:")
    print(f"    - Same Qwen 3.5 MoE A22B config (8 experts, top_k=2, 3584->18944)")
    print(f"    - Same SwiGLU FFN structure (gate + up -> SiLU*mul -> down)")
    print(f"    - Same dtype (float16)")
    print(f"    - Same M values: {M_VALUES}")
    print()
    print(f"  RMLX additionally benchmarks:")
    print(f"    - GatherMM strategy (batched GEMM across experts)")
    print(f"    - Raw gather_mm kernel (no routing overhead)")
    print(f"    - Real RDMA token exchange (with distributed feature)")
    print()


if __name__ == "__main__":
    main()
