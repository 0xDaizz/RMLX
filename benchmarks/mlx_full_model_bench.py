#!/usr/bin/env python3
"""
MLX full-model forward benchmark — companion to RMLX model_bench.rs

Loads a HuggingFace safetensors model with MLX and measures full
32-layer forward latency for decode (seq_len=1) and prefill (seq_len=512).

Usage:
  python benchmarks/mlx_full_model_bench.py --model-dir ~/models/Meta-Llama-3-8B-Instruct

Output includes mean latency in microseconds that can be passed to
RMLX model_bench via MLX_DECODE_US / MLX_PREFILL_US env vars.
"""

import argparse
import json
import time
import statistics
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


WARMUP = 3
ITERS = 20


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = nn.RoPE(head_dim, base=rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask=mask
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_dim, rms_norm_eps, rope_theta):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta)
        self.mlp = MLP(hidden_size, intermediate_dim)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask=mask, cache=cache)
        x = x + h
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = x + h
        return x


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [
            TransformerBlock(
                config["hidden_size"],
                config["num_attention_heads"],
                config.get("num_key_value_heads", config["num_attention_heads"]),
                config["hidden_size"] // config["num_attention_heads"],
                config["intermediate_size"],
                config.get("rms_norm_eps", 1e-5),
                config.get("rope_theta", 10000.0),
            )
            for _ in range(config["num_hidden_layers"])
        ]
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-5))

    def __call__(self, x, cache=None):
        h = self.embed_tokens(x)
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, mask=mask, cache=layer_cache)
        return self.norm(h)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def __call__(self, x, cache=None):
        h = self.model(x, cache=cache)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(label, fn, warmup=WARMUP, iters=ITERS):
    """Warm up, then time iters iterations."""
    for _ in range(warmup):
        y = fn()
        mx.eval(y)

    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)

    mean = statistics.mean(times_us)
    std = statistics.stdev(times_us) if len(times_us) > 1 else 0.0
    s = sorted(times_us)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]

    print(f"  {label}:")
    print(f"    mean={mean:10.1f}us  std={std:8.1f}us  p50={p50:10.1f}us  "
          f"p95={p95:10.1f}us  min={s[0]:10.1f}us  max={s[-1]:10.1f}us  (n={iters})")
    return mean


def main():
    parser = argparse.ArgumentParser(description="MLX full-model benchmark")
    parser.add_argument("--model-dir", required=True, help="HF model directory with safetensors")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    intermediate_dim = config["intermediate_size"]
    vocab_size = config["vocab_size"]

    print(f"MLX Full-Model Benchmark")
    print(f"Model: {model_dir} ({config.get('model_type', 'unknown')})")
    print(f"Config: hidden={hidden_size}, heads={num_heads}/{num_kv_heads}, "
          f"head_dim={head_dim}, layers={num_layers}, "
          f"intermediate={intermediate_dim}, vocab={vocab_size}")

    # Load model weights from safetensors
    print(f"\nLoading weights from {model_dir}...")
    t0 = time.time()
    model = LlamaForCausalLM(config)

    # Find and load safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    # Use MLX weight loading
    weights = {}
    for sf in st_files:
        w = mx.load(str(sf))
        weights.update(w)

    # Load weights into model
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print(f"  Loaded {len(weights)} tensors in {time.time() - t0:.1f}s")

    # =====================================================================
    # DECODE (seq_len=1)
    # =====================================================================
    print(f"\n========== DECODE BENCHMARK (seq_len=1, {num_layers} layers) ==========")

    tokens_decode = mx.array([[1]])  # [1, 1]

    # With KV cache
    cache = [nn.KVCache() for _ in range(num_layers)]

    def decode_with_cache():
        for c in cache:
            c.offset = 0
            c.keys = None
            c.values = None
        return model(tokens_decode, cache=cache)

    decode_mean = bench("MLX decode (with KV cache)", decode_with_cache,
                        warmup=args.warmup, iters=args.iters)

    # Compiled
    compiled_model = mx.compile(model)
    cache_compiled = [nn.KVCache() for _ in range(num_layers)]

    def decode_compiled():
        for c in cache_compiled:
            c.offset = 0
            c.keys = None
            c.values = None
        return compiled_model(tokens_decode, cache=cache_compiled)

    decode_compiled_mean = bench("MLX decode compiled", decode_compiled,
                                  warmup=args.warmup, iters=args.iters)

    # =====================================================================
    # PREFILL (seq_len=512)
    # =====================================================================
    print(f"\n========== PREFILL BENCHMARK (seq_len=512, {num_layers} layers) ==========")

    tokens_prefill = mx.array([list(range(512))])  # [1, 512]

    cache_prefill = [nn.KVCache() for _ in range(num_layers)]

    def prefill_with_cache():
        for c in cache_prefill:
            c.offset = 0
            c.keys = None
            c.values = None
        return model(tokens_prefill, cache=cache_prefill)

    prefill_mean = bench("MLX prefill (with KV cache)", prefill_with_cache,
                         warmup=args.warmup, iters=args.iters)

    def prefill_compiled():
        for c in cache_prefill:
            c.offset = 0
            c.keys = None
            c.values = None
        return compiled_model(tokens_prefill, cache=cache_prefill)

    prefill_compiled_mean = bench("MLX prefill compiled", prefill_compiled,
                                   warmup=args.warmup, iters=args.iters)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n========== MLX SUMMARY ==========")
    print(f"Model: {model_dir}")
    print(f"DECODE  (seq_len=1):   mean={decode_mean:10.1f}us  compiled={decode_compiled_mean:10.1f}us")
    print(f"PREFILL (seq_len=512): mean={prefill_mean:10.1f}us  compiled={prefill_compiled_mean:10.1f}us")
    print()
    print("To compare with RMLX, run:")
    print(f"  MLX_DECODE_US={decode_mean:.1f} MLX_PREFILL_US={prefill_mean:.1f} \\")
    print(f"    RMLX_MODEL_DIR={model_dir} cargo bench -p rmlx-nn --bench model_bench")
    print("=================================")


if __name__ == "__main__":
    main()
