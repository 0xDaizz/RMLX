# GPU Pipeline — ExecGraph Architecture

## Overview

The rmlx GPU pipeline eliminates per-operation CPU overhead by batching multiple Metal GPU operations into minimal command buffers. The core abstraction, **ExecGraph**, pre-builds a deterministic execution graph that replays transformer layer computations with near-zero CPU involvement.

**Key results:**

| Metric | Baseline | ExecGraph | Improvement |
|--------|----------|-----------|-------------|
| Command buffers / layer | 65 | 5 | 92.3% reduction |
| Command buffers / layer (Phase KO) | 65 | 1 (9 dispatches) | 98.5% reduction |
| Latency / layer | ~112ms | ~6.4ms | 17.4x speedup |
| Latency / layer (Phase KO) | ~109ms | ~1.7ms | 64x speedup |
| Gap vs MLX (60L) | -- | 2.09x faster | RMLX leads |
| CPU-GPU sync overhead | baseline | minimal | 98.5% reduction |
| Numerical parity | -- | max_diff=6.4e-6 | exact match |

---

## The Problem

In a naive Metal execution model, each GPU operation (matmul, RoPE, softmax, add, etc.) creates its own command buffer:

1. Allocate a new `MTLCommandBuffer`
2. Create a `MTLComputeCommandEncoder`
3. Set pipeline state, buffers, and threadgroup sizes
4. Dispatch threads
5. End encoding
6. Commit the command buffer
7. **CPU-GPU synchronization barrier** (wait for completion)

A single transformer layer in a LLaMA-style model executes approximately 65 individual operations. Each operation incurs CPU-side overhead for command buffer creation, encoder setup, and a CPU-GPU synchronization point. At 65 command buffers per layer, the CPU spends more time managing GPU work than the GPU spends executing it.

```
Baseline: 65 CBs/layer x N layers
  [CB1: matmul_qkv] -> sync -> [CB2: rope_q] -> sync -> [CB3: rope_k] -> sync -> ...
  Total: ~112ms per layer
```

---

## ExecGraph Architecture

The GPU pipeline consists of three layered components:

### CommandBatcher

`CommandBatcher` groups multiple encoder operations into a shared command buffer. Instead of each op creating its own CB, the batcher provides a single CB that multiple ops encode into sequentially.

```rust
let mut batcher = CommandBatcher::new(&device);
batcher.begin();

// Multiple ops share one command buffer
matmul.forward_into_cb(&mut batcher, &q_proj, &x)?;
rope.forward_into_cb(&mut batcher, &q, positions)?;
matmul.forward_into_cb(&mut batcher, &k_proj, &x)?;
rope.forward_into_cb(&mut batcher, &k, positions)?;

batcher.commit();  // Single commit for all ops
```

### ExecGraph

`ExecGraph` pre-builds the full execution sequence for a transformer model. Because transformer inference is deterministic (same ops, same buffer sizes, same dispatch geometry every token), the graph records the sequence once and replays it for subsequent tokens.

```rust
// Build once
let graph = ExecGraph::build(&model, &sample_input)?;

// Replay for each token — near-zero CPU overhead
for token in tokens {
    graph.execute(&device, &buffers)?;
}
```

### ICB (Indirect Command Buffers)

`IcbBuilder`, `IcbReplay`, and `IcbCache` provide Metal Indirect Command Buffer support. ICBs allow the GPU to dispatch compute commands without CPU intervention, enabling true zero-CPU-overhead replay for recorded command sequences.

---

## The `_into_cb()` Pattern

The foundation of the GPU pipeline is the `_into_cb()` pattern. Every one of the 14 op modules in rmlx-core implements an `_into_cb()` variant that encodes work into a caller-provided command buffer rather than creating a new one.

**Standard pattern** (creates its own CB):
```rust
fn forward(&self, input: &Array) -> Result<Array> {
    let cb = device.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    // ... encode work ...
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(output)
}
```

**`_into_cb()` pattern** (encodes into caller's CB):
```rust
fn forward_into_cb(
    &self,
    batcher: &mut CommandBatcher,
    input: &Array,
) -> Result<Array> {
    let encoder = batcher.current_encoder();
    // ... encode work into shared encoder ...
    Ok(output)
}
```

All 14 op modules implement this pattern:
- matmul, quantized matmul, softmax, RMS norm, RoPE
- element-wise binary ops, reduce, copy/transpose, indexing
- SiLU, residual add, layer norm, gather, scatter

At the nn layer, this extends to:
- `Linear::forward_into_cb()`
- `Attention::forward_graph()`
- `TransformerBlock::forward_graph()`
- `TransformerModel::forward_graph()`

---

## Command Buffer Reduction

The 65 baseline command buffers per transformer layer are consolidated into 5:

| CB | Operations | Description |
|----|-----------|-------------|
| CB1 | RMS norm + Q/K/V projections (fused) | Pre-attention normalization and QKV linear projections |
| CB2 | Head split + RoPE + cache append | Multi-head reshaping, rotary position encoding, and KV cache update |
| CB3 | SDPA + head concat + O_proj | Fused scaled dot-product attention, head concatenation, and output projection |
| CB4 | Residual + pre-FFN norm | Attention residual connection and feed-forward normalization |
| CB5 | Gate + up + silu_mul + down + residual | Entire FFN fused: gated activation, projections, and final residual |

```
ExecGraph: 5 CBs/layer
  [CB1: Norm+QKV] -> [CB2: Split+RoPE+Cache] -> [CB3: SDPA+Concat+OProj] -> [CB4: Res+Norm] -> [CB5: FFN+Res]
  Total: ~6.4ms per layer
```

This represents a **92.3% reduction** in command buffer count (65 → 5) and a **98.5% reduction** in CPU-GPU synchronization points.

---

## Benchmark Results

Benchmarks measured on a single transformer layer (LLaMA-style architecture, 4096 hidden, M3 Ultra):

### Baseline (per-op command buffers)

- **Command buffers per layer:** 65
- **Latency per layer:** ~112ms
- **CPU-GPU syncs per layer:** 65

### ExecGraph (batched command buffers)

- **Command buffers per layer:** 5
- **Latency per layer:** ~6.4ms
- **CPU-GPU syncs per layer:** 5

### Summary

| Metric | Value |
|--------|-------|
| Speedup | **17.4x** |
| Latency reduction | **94.3%** (~112ms → ~6.4ms) |
| CB reduction | **92.3%** (65 → 5) |
| CPU-GPU sync reduction | **98.5%** |
| Tests passing | **1,142+** |

---

## Weight Pre-caching

Phase 9B-opt introduced weight pre-caching to eliminate runtime transpose overhead. During model initialization, `prepare_weight_t()` creates contiguous transposed copies of weight matrices so that inference never needs to compute transpositions on the fly.

```rust
// During model load / warmup
linear.prepare_weight_t();

// Returns pre-cached contiguous transposed weight
let wt = linear.weight_transposed_contiguous();
```

At the model level, `prepare_weights_for_graph()` recursively prepares all weight matrices:

```rust
model.prepare_weights_for_graph();
// TransformerModel -> TransformerBlock -> Attention + FeedForward -> Linear
```

This eliminates a significant portion of the per-layer overhead that remained after command buffer batching, contributing to the Phase 9B-opt 17.4x speedup.

---

## ExecGraph vs CUDA Graphs

Metal and CUDA take fundamentally different approaches to GPU work batching:

| Aspect | ExecGraph (Metal) | CUDA Graphs |
|--------|-------------------|-------------|
| Strategy | **Re-encode** into minimal CBs | **Capture-replay** recorded stream |
| Recording | Explicit graph construction | Implicit stream capture |
| Flexibility | Can modify buffer bindings between replays | Requires graph rebuild for changes |
| CPU overhead | Near-zero (5 CB commits per layer) | Near-zero (single graph launch) |
| GPU overhead | Same as manual dispatch | Potential driver-level optimizations |
| Memory | No extra memory for graph | Graph object retains captured state |

ExecGraph's re-encode strategy is well-suited to Metal's command buffer model. Rather than capturing and replaying a fixed sequence (like CUDA Graphs), ExecGraph pre-computes the encoding sequence and re-encodes it into fresh command buffers. This preserves flexibility (buffer bindings can change between invocations) while achieving comparable CPU overhead reduction.

---

## Numerical Parity

ExecGraph produces numerically identical results to the baseline per-op execution path:

- **Maximum absolute difference:** 6.4e-6
- **Verification:** All 1,142+ tests pass with both code paths
- **Guarantee:** The `_into_cb()` pattern encodes the exact same compute pipelines, threadgroup sizes, and buffer bindings as the standard `forward()` path. The only difference is command buffer grouping, which does not affect numerical results.

This level of precision (max_diff=6.4e-6) is well within the expected floating-point tolerance for fp16/bf16 transformer computations and confirms that the pipeline optimization does not introduce any numerical divergence.

---

## Phase KO: 9-Dispatch Decode Path

Phase KO extends the ExecGraph pipeline with a minimal-dispatch decode path that reduces the full transformer layer to 9 Metal dispatches (4 encoders with memory barriers) in a single command buffer.

### Dispatch Breakdown

| # | Operation | Encoder |
|---|-----------|---------|
| 1 | Merged QKV GEMV (fused Q+K+V projection) | Encoder 1 |
| 2 | Batched RoPE (Q and K simultaneously) | Encoder 1 |
| 3 | Batched SDPA decode (slab KV cache) | Encoder 2 |
| 4 | Output projection (fused GEMV + bias) | Encoder 2 |
| 5 | Attention residual add | Encoder 2 |
| 6 | Merged gate_up GEMV (fused gate+up projection) | Encoder 3 |
| 7 | Fused SiLU * gate | Encoder 3 |
| 8 | Down projection (fused GEMV + bias) | Encoder 3 |
| 9 | FFN residual add | Encoder 4 |

Memory barriers between encoders replace the heavyweight encoder boundary transitions, reducing the 9 logical dispatches to 4 physical encoder switches.

### Progressive Optimization Results

```text
Baseline (per-op sync):  109,215us  1x
ExecGraph (5 CB):          2,735us  40x
Single-CB (44 enc):        2,049us  53x
9-Dispatch (9->4 enc):     1,739us  64x
RMLX 60L pipeline:         1,204us/layer
MLX compiled 60L:          2,513us/layer
Result:                    2.09x faster
```

### Key Enablers

- **Weight merging**: QKV and gate_up weights merged at load time, eliminating 4 separate GEMV dispatches
- **Slab KV cache**: Single contiguous allocation per layer enables stride-aware SDPA decode
- **StorageModePrivate**: Static weights stored in GPU-only memory (no CPU page table entries)
- **Array::uninit**: Output buffers allocated without zeroing (kernel will overwrite completely)
- **Unretained CB**: On Metal 3+ (M2+), command buffers are not retained after commit
- **_into_encoder pattern**: Ops encode into a shared compute encoder with memory barriers instead of encoder boundaries
