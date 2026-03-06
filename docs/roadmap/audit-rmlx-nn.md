# rmlx-nn Production Readiness Audit

## Metadata

| Field     | Value                                                                      |
|-----------|----------------------------------------------------------------------------|
| Date      | 2026-03-06                                                                 |
| Scope     | `rmlx-nn` crate — all modules                                             |
| Auditors  | Claude Opus 4.6 (internal code quality + MLX comparison + CUDA frontier)   |
| Branch    | `main` @ `9453cbfe`                                                        |

---

## Executive Summary

The `rmlx-nn` crate provides a broad surface area of neural-network layers targeting Apple Silicon Metal inference, including unique primitives such as MLA attention, sliding-window attention, MoE with expert-parallelism, and pre-allocated KV caches. However, the audit reveals **5 P0 critical issues** that make the crate unusable for production model inference today:

1. **Two stub `forward()` implementations** (MLA, SlidingWindowAttention) that discard all computed projections and return garbage output, breaking DeepSeek-V3 and Mistral-family models.
2. A **RotatingKvCache wrap-around correctness bug** that can surface uninitialized memory regions.
3. **Distributed MoE silently drops non-local expert contributions** when `world_size > 1`.
4. A **QuantizedLinear panic** on any non-Float32 input during Q8 batched fallback.

Beyond correctness, major architectural gaps block production serving: there is **no sampling layer** (top-k, top-p, temperature), **no continuous batching**, **no paged KV cache**, and **no RMSNorm nn layer** despite every modern LLM requiring it. Test coverage is severely lacking, with zero tests for 7 modules and no tests for critical cache edge cases.

The crate does have genuine strengths — MLA/sliding-window as reusable layers, MoePipeline with EP and overlap strategies, Megatron-style tensor parallelism, and O(1) Metal KV-cache append — but these cannot be leveraged until the P0 correctness issues are resolved.

---

## P0 — Critical Issues

### P0-1: MLA `forward()` is a stub

- **Source**: `mla.rs:368-401`
- **Impact**: Computes Q, K, V projections but discards them entirely. Returns `o_proj(x)` directly — a linear projection of the raw input with no attention computation. DeepSeek-V3 and any model using Multi-Latent Attention produces incorrect output.
- **Root cause**: Implementation incomplete. No RoPE application, no SDPA call, no cache update.
- **Fix**: Implement full MLA forward: apply RoPE to Q/K, call SDPA, update KV cache, project output.

### P0-2: SlidingWindowAttention `forward()` is a stub

- **Source**: `sliding_window.rs:145-177`
- **Impact**: Projects Q, K, V and builds the sliding-window mask, then discards all with `let _ = (...)` and returns `o_proj(x)`. Mistral, Mixtral, and any sliding-window model is broken.
- **Root cause**: Implementation incomplete.
- **Fix**: Implement full forward: apply mask to SDPA, update cache with window eviction, project output.

### P0-3: RotatingKvCache wrap-around logic incorrect

- **Source**: `attention.rs:490, 589, 599, 683`
- **Impact**: When the circular buffer wraps around, partially filled buffers can include invalid/uninitialized memory regions. The keep-region semantics are broken by the trimming logic — linearization after wrap may return stale data interleaved with valid data.
- **Root cause**: The trim and linearize operations do not correctly account for the fill level vs. capacity boundary.
- **Fix**: Track valid-region start/end separately from write pointer; linearize must copy only valid regions in correct order; add sentinel/zero-fill for unfilled capacity.

### P0-4: Distributed MoE incomplete — non-local experts dropped

- **Source**: `moe.rs:406, 513, 606, 277`
- **Impact**: When `world_size > 1`, tokens routed to non-local experts are silently skipped. No dispatch/combine via `MoeDispatchExchange` is wired into the forward paths. This means distributed MoE produces incorrect results — expert contributions from remote ranks are simply lost.
- **Root cause**: EP dispatch/combine integration not implemented in forward paths.
- **Fix**: Wire `MoeDispatchExchange` dispatch (before expert compute) and combine (after expert compute) into all MoE forward strategies (PerExpert, Batched, Pipeline).

### P0-5: QuantizedLinear Q8 batched fallback panics for non-Float32

- **Source**: `quantized_linear.rs:191, 206`
- **Impact**: `to_vec_checked::<f32>()` hard-asserts that the input dtype is Float32. Any model running in f16 or bf16 (the common case for quantized models) will panic at runtime.
- **Root cause**: Missing dtype conversion before CPU fallback.
- **Fix**: Cast input to f32 before extraction, or implement f16/bf16 paths. Replace `assert!` with `Result` return.

### P0-6: No sampling layer (CUDA frontier)

- **Source**: Entire crate — no sampling module exists.
- **Impact**: There is zero infrastructure for top-k, top-p (nucleus), temperature scaling, repetition penalty, frequency penalty, or any other sampling strategy. The crate literally cannot generate text — it can only produce logits.
- **Fix**: Implement a `Sampler` module with configurable strategies. This is the minimum viable path to text generation.

### P0-7: No continuous batching (CUDA frontier)

- **Source**: All `forward()` signatures take a single sequence.
- **Impact**: Cannot serve multiple concurrent requests. Throughput is limited to one request at a time. This is a fundamental requirement for any inference server.
- **Fix**: Implement request-level scheduling with batch formation, integrating with paged KV cache for memory-efficient multi-request serving.

### P0-8: No paged KV cache (CUDA frontier)

- **Source**: Current KV caches are monolithic pre-allocated buffers.
- **Impact**: Memory is wasted on maximum-length allocations per sequence. Cannot support large batch sizes or long contexts efficiently. vLLM has demonstrated paged attention on Metal.
- **Fix**: Implement block-based KV cache with a block manager, following the PagedAttention design. This is tightly coupled with continuous batching (P0-7).

### P0-9: FlashAttention Metal missing (CUDA frontier)

- **Source**: SDPA is a placeholder; no tiled/fused attention kernel.
- **Impact**: Attention is the dominant cost for long contexts. Without a fused kernel, performance will be uncompetitive. `metal-flash-attention` exists as a reference implementation.
- **Fix**: Integrate or implement a tiled FlashAttention-2 Metal kernel with online softmax.

---

## P1 — Important Issues

### Code Quality

| ID     | Issue                                                                                          | Location                        |
|--------|------------------------------------------------------------------------------------------------|---------------------------------|
| P1-1   | MoE PerExpert path creates excessive temporary allocations (separate CB per token-expert pair)  | `moe.rs:586-618`                |
| P1-2   | QuantizedLinear re-uploads weight buffers to Metal on every `forward()` via `new_buffer_with_data()` | `quantized_linear.rs:174-177` |
| P1-3   | Linear bias path creates N separate command buffers for N batch rows                           | `linear.rs:157-178`             |
| P1-4   | MlaKvCache `advance()` only increments seq_len, no actual data copy (consistent with MLA stub) | `mla.rs:174-183`               |
| P1-5   | GGUF loader `gguf_name_to_rmlx()` always returns `Some` — unrecognized patterns pass through silently | `gguf_loader.rs:331-348`  |
| P1-6   | `TransformerBlock::new()` creates dummy FFN ignoring `ff_type`, always Dense with wrong dims   | `transformer.rs:254-269`        |
| P1-7   | No `load_weights` on Embedding layer                                                          | `embedding.rs`                  |
| P1-8   | `parallel.rs` `read_f32_strided` uses raw pointer arithmetic without bounds check              | `parallel.rs:67-90`             |
| P1-9   | LayerKvCache append APIs don't validate new_tokens vs source tensor shapes/dtypes              | `attention.rs:98, 166, 270`     |
| P1-10  | `BatchKvCache.offsets` never updated when per-sequence caches appended through `get_mut()`     | `attention.rs:854, 898, 944`    |
| P1-11  | QuantizedKvCache incomplete — no dequant/attention-consumer path wired in Attention; keeps full-precision shadow | `attention.rs:1808, 1813, 1957` |
| P1-12  | MoE pipeline "overlap" path mostly serialized by internal `wait_until_completed()`             | `moe_pipeline.rs:188, 272, 691, 748` |
| P1-13  | Sparse ICB path is a stub (`sparse_plan`/`capacity` ignored)                                   | `moe.rs:807, 849`              |
| P1-14  | `capacity_factor` configured but not used                                                      | `moe.rs:107, 892`              |
| P1-15  | Causal correctness caller-dependent — attention doesn't auto-build causal masks, fused SDPA called with `is_causal=false` | `attention.rs:1144, 1251` |
| P1-16  | Quantization support narrow (Q4/Q8 only), no AWQ/GPTQ ingestion, no GGUF-to-QuantizedLinear integration | `quantized_linear.rs:15`, `gguf_loader.rs:213` |
| P1-17  | Model architecture modules provide configs only — no architecture-specific forward or weight-loading | `models/*.rs`             |
| P1-18  | Sliding-window mask eagerly materialized as full `[seq_len, total_seq]` f32 tensor             | `sliding_window.rs:121, 161`    |
| P1-19  | Parallel row path lacks `world_size == group.size()` validation                                | `parallel.rs:289, 393`          |
| P1-20  | Production code uses panic-oriented validation (`assert!`/indexing) instead of `Result`         | `attention.rs:388, 1839`, `parallel.rs:252` |

### Missing NN Layers (vs MLX)

| Priority | Layer / Feature                                      | Notes                                                    |
|----------|------------------------------------------------------|----------------------------------------------------------|
| Critical | **RMSNorm**                                          | Every modern LLM uses it. `rmlx-core` has ops but no nn layer. `TransformerConfig` has `rms_norm_eps` field but no `RMSNorm` struct. |
| High     | Standalone **RoPE** layer                            | Currently inlined / missing from MLA stub                |
| High     | **20+ activation functions** (ReLU, LeakyReLU, ELU, CELU, PReLU, Mish, Hardswish, etc.) | rmlx-nn has only 5 activations |
| High     | **QuantizedEmbedding**, **QQLinear**                 | Needed for fully quantized model loading                 |
| Medium   | **Dropout / Dropout2d / Dropout3d**                  | Required for training support                            |
| Medium   | **BatchNorm, GroupNorm, InstanceNorm**               | Required for CNN architectures                           |
| Medium   | **Conv3d, ConvTranspose1d/2d/3d**                    | Extended convolution support                             |
| Medium   | **MaxPool, AvgPool** (1d/2d/3d)                      | Pooling layers                                           |
| Medium   | **RNN, GRU, LSTM**                                   | Recurrent architectures                                  |
| Medium   | **SinusoidalPositionalEncoding, ALiBi**              | Positional encoding variants                             |
| Medium   | **TransformerEncoder / TransformerDecoder**           | Encoder-decoder model support                            |
| Medium   | **Sequential**, **Module base trait**                 | Container and parameter traversal                        |
| Low      | **Identity, Bilinear**                               | Linear layer variants                                    |
| Low      | **Upsample**                                         | Upsampling support                                       |
| Low      | **14 loss functions**                                | Training losses                                          |
| Low      | **10 weight initializers**                           | Weight initialization utilities                          |
| Low      | `value_and_grad`, `checkpoint`, `average_gradients`  | Training utilities                                       |

### Missing Serving Features (vs CUDA Frontier)

| Priority | Feature                         | Notes                                                              |
|----------|---------------------------------|--------------------------------------------------------------------|
| P1       | **FP8 KV Cache**                | 2-4x KV memory reduction. Important for long-context serving.     |
| P1       | **Prefix Caching**              | No prefix sharing across requests. Critical for chat workloads.   |
| P1       | **AWQ / GPTQ / k-quants**      | Q4/Q8 only. Limited model compatibility with HuggingFace ecosystem.|
| P1       | **Chunked Prefill**             | Missing. Causes latency unfairness for long prompt inputs.        |
| P2       | **More model configs**          | 4 models vs 70+ in vLLM. Need Gemma, Phi, Mistral, Command-R.    |
| P2       | **Cross-Attention**             | Encoder-decoder and vision-language model support.                |
| P2       | **Speculative Decoding**        | 2-3x generation speedup technique.                                |
| P3       | Constrained decoding (regex/JSON) | Structured output generation.                                   |
| P3       | Multi-LoRA batched inference    | Serve multiple fine-tuned variants simultaneously.                |
| P3       | Fused RoPE + Attention kernel   | Single kernel for position encoding + attention.                  |
| P3       | KV cache compression            | Beyond quantization (e.g., eviction, merging).                    |

---

## P2 — Minor Issues

| ID    | Issue                                                                                      | Location                 |
|-------|--------------------------------------------------------------------------------------------|--------------------------|
| P2-1  | `Swish` type alias not re-exported from `lib.rs`                                           | `activations.rs:99`      |
| P2-2  | `ActivationType::from_str_name` doesn't implement `std::str::FromStr`                     | `activations.rs:167-178` |
| P2-3  | `DynamicExecContext.dtype` stored but never exposed via getter                             | `dynamic.rs:20`          |
| P2-4  | `ExpertGroup` doc says "f32 only" but code handles f16/bf16                                | `expert_group.rs:19-21`  |
| P2-5  | `TransformerConfig` doesn't store `intermediate_dim` directly                              | `transformer.rs`         |
| P2-6  | Model configs are function-based, no builder pattern                                       | `models/*.rs`            |
| P2-7  | `QuantizedKvCache` / `QuantizedArray` deeply buried in `attention.rs` (~2000 lines)        | `attention.rs`           |
| P2-8  | `LayerNorm.forward()` restricts input to exactly 2D                                        | `layer_norm.rs`          |
| P2-9  | `Conv1dConfig` / `Conv2dConfig` don't implement `Clone` or `Debug`                         | `conv.rs`                |
| P2-10 | DynamicExecContext, rotating/batch/quantized KV caches, MLA, sliding-window are dead surfaces not wired into transformer path | Multiple |
| P2-11 | `MoePipeline::forward_overlapped` appears test-only                                        | `moe_pipeline.rs`        |
| P2-12 | Dead-code warnings with `--features distributed` (`read_f32_strided`, `cpu_matmul_f32`)    | `parallel.rs`            |
| P2-13 | Unit tests panic on missing Metal device instead of skipping                               | Multiple test modules    |

---

## Test Coverage Gaps

### Modules with zero tests

| Module               | Risk   | Notes                                           |
|----------------------|--------|-------------------------------------------------|
| `mla.rs`             | P0     | Stub forward — tests would catch this instantly  |
| `sliding_window.rs`  | P0     | Stub forward — same                             |
| `layer_norm.rs`      | Medium | Core layer, untested                            |
| `conv.rs`            | Medium | Conv1d/Conv2d untested                          |
| `dynamic.rs`         | Low    | Execution context                               |
| `gguf_loader.rs`     | High   | Weight loading correctness critical             |
| `activations.rs`     | Low    | Simple functions                                |

### Untested critical paths

- `RotatingKvCache` wrap-around, linearize, and trim operations
- `BatchKvCache` filter and extend operations
- `QuantizedKvCache` quantize/dequantize round-trip fidelity
- `QuantizedLinear.forward()` for any dtype
- MoE distributed dispatch/combine (blocked by P0-4)
- Attention causal mask generation and application
- GGUF weight name mapping edge cases

---

## rmlx-nn Unique Strengths

These are capabilities present in `rmlx-nn` that are **not available** as reusable nn layers in MLX's `mlx.nn`:

| Feature                            | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **MLA as reusable layer**          | Multi-Latent Attention exposed as a generic nn layer (MLX only has it in model-specific files) |
| **SlidingWindowAttention layer**   | Dedicated layer with mask construction (MLX inlines in model code)          |
| **MoePipeline with EP**            | Expert-parallelism-aware MoE with SBO/TBO overlap strategies                |
| **ExpertGroup**                    | Batched expert management with group-level operations                       |
| **Pre-allocated Metal KV cache**   | O(1) append via Metal buffer offsets — no reallocation                      |
| **RotatingKvCache**                | Circular buffer KV cache for bounded memory (once P0-3 is fixed)           |
| **QuantizedKvCache**               | Quantized cache storage concept (once P1-11 is completed)                  |
| **BatchKvCache**                   | Multi-sequence cache with per-sequence offset tracking                     |
| **GGUF loader in nn crate**        | Direct GGUF weight ingestion at the nn layer                               |
| **Megatron-style TP**              | `ColumnParallelLinear` and `RowParallelLinear` as first-class layers        |
| **ICB sparse dispatch concept**    | Indirect command buffer path for sparse expert routing (once P1-13 is done) |

---

## Recommended Action Plan

### Phase 1: P0 Correctness Fixes (Week 1-2)

1. **Implement MLA `forward()`** — RoPE, SDPA, cache update. Add end-to-end test with known-good outputs.
2. **Implement SlidingWindowAttention `forward()`** — masked SDPA, cache eviction. Add tests.
3. **Fix RotatingKvCache wrap-around** — correct linearize/trim for partial fills. Add property-based tests for all cache states.
4. **Wire distributed MoE dispatch/combine** — integrate `MoeDispatchExchange` into all forward strategies.
5. **Fix QuantizedLinear dtype handling** — cast or dispatch by dtype, replace panic with `Result`.

### Phase 2: Serving Infrastructure (Week 3-5)

6. **Implement Sampler module** — top-k, top-p, temperature, repetition/frequency penalties. This unblocks text generation.
7. **Implement RMSNorm nn layer** — trivial wrapper around existing `rmlx-core` ops. Unblocks all modern LLM architectures.
8. **Implement paged KV cache** — block manager, block tables, paged attention kernel.
9. **Implement continuous batching** — request scheduler, batch formation, integration with paged KV.

### Phase 3: Attention and Performance (Week 6-8)

10. **Integrate FlashAttention Metal** — tiled kernel with online softmax for long-context performance.
11. **Implement FP8 KV cache** — quantized cache storage with dequant-on-read.
12. **Fix MoePipeline overlap** — remove internal `wait_until_completed()` barriers, achieve true SBO/TBO overlap.
13. **Eliminate per-forward buffer re-uploads** in QuantizedLinear and excessive CB creation in Linear/MoE.

### Phase 4: Model Coverage (Week 9-11)

14. **Add missing activations** — at least ReLU, LeakyReLU, ELU, GELU variants, Mish, Hardswish.
15. **Add standalone RoPE layer** with frequency scaling support.
16. **Implement AWQ/GPTQ/k-quant ingestion** alongside existing Q4/Q8.
17. **Add model configs** — Gemma, Phi, Mistral, Command-R at minimum.
18. **Wire TransformerBlock FFN type** — respect `ff_type` config, support MoE/Dense/Gated variants.
19. **Implement prefix caching and chunked prefill**.

### Phase 5: Robustness (Week 12-13)

20. **Replace panic-oriented validation with `Result`** across all public APIs.
21. **Add bounds checking** to `parallel.rs` raw pointer paths.
22. **Add input validation** to KV cache append APIs.
23. **Comprehensive test coverage** — target all zero-test modules and untested critical paths listed above.

### Phase 6: Polish (Week 14+)

24. Fix P2 issues (re-exports, trait implementations, doc accuracy, dead code).
25. Add training support layers (Dropout, loss functions, initializers) if training is in scope.
26. Add remaining MLX parity layers based on model demand.

---

## Summary

| Severity | Count | Scope                                                        |
|----------|-------|--------------------------------------------------------------|
| **P0**   | 9     | 5 code correctness + 4 missing serving infrastructure       |
| **P1**   | 20 code quality + 16 missing layers + 4 serving features = **40** | Across all audit dimensions |
| **P2**   | 13    | Code hygiene, dead surfaces, minor API issues                |

**Bottom line**: `rmlx-nn` has an ambitious and well-structured design with several innovations over MLX (reusable MLA/sliding-window layers, MoePipeline, Metal-native KV caches, TP primitives). However, it is **not production-ready**. Two of its most prominent attention layers are stubs that return garbage, the KV cache has a correctness bug, distributed MoE drops expert contributions, and there is no path from logits to generated text. The recommended 6-phase plan prioritizes correctness first, then serving infrastructure, then breadth — reflecting the reality that a correct, servable system with fewer layers is more valuable than a broad but broken one.
