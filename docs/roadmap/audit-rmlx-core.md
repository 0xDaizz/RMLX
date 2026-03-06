# rmlx-core Production Readiness Audit

## Metadata

| Field     | Value                                                                              |
|-----------|------------------------------------------------------------------------------------|
| Date      | 2026-03-06                                                                         |
| Scope     | rmlx-core crate (all modules)                                                     |
| Auditors  | Claude Opus 4.6 (internal audit + MLX comparison + CUDA frontier comparison + Codex) |

## Executive Summary

rmlx-core provides a functional foundation for Metal-accelerated ML inference with unique strengths such as built-in LoRA, precision guards, and runtime metrics. The original audit identified **fifteen P0 correctness issues**. **Phase 0 (PR #38) fixed 12 of 15 P0s** including copy.rs panics, softmax unwrap, quantized debug_assert, conv_tiled overflow, FP8 race, binary/softmax empty-tensor dispatch, RoPE freq validation, GEMV/SDPA/gather_qmm dtype/shape validation, and GGUF alignment=0. **Three P0s remain open**: the broken op CB counter (P0-4), SDPA backward write races (P0-5), and SDPA backward dtype validation (P0-13). The crate is missing critical infrastructure that MLX provides (lazy evaluation, graph compilation, multi-stream scheduling) and lacks essential ops (FFT, sort, scan, random, slicing). Compared to CUDA frontier serving systems, rmlx-core has no paged attention, continuous batching, or prefix caching -- features required for competitive LLM serving throughput.

## P0 -- Critical Issues

1. **`copy.rs:567,578,588` -- `unreachable!()` panics for unsupported dtypes.**
   `vectorized_kernel_name()`, `scalar_kernel_name()`, and `strided_kernel_name()` call `unreachable!()` instead of returning `Result`. Since `copy` is invoked by `make_contiguous()`, which is used by nearly every op, a single unsupported dtype triggers a hard process abort with no recovery path.
   **Status: FIXED (Phase 0, PR #38)** -- replaced with `Result<_, KernelError>`.

2. **`softmax.rs:553` -- `.unwrap()` on `shape.last()` in non-test code.**
   If a zero-rank tensor reaches the softmax kernel, this panics. The function should return an error for empty shapes.
   **Status: FIXED (Phase 0, PR #38)** -- replaced with `ok_or` error.

3. **`dtype.rs:87` -- `debug_assert` for quantized block-alignment check.**
   `numel_to_bytes` verifies that `numel` is a multiple of the quantized block size using `debug_assert` only. In release builds, non-aligned element counts silently truncate the byte count, producing undersized GPU buffers that lead to out-of-bounds reads/writes and potential GPU faults or data corruption.
   **Status: FIXED (Phase 0, PR #38)** -- promoted to runtime check returning `DTypeError`.

4. **Op command-buffer counter (`TOTAL_OP_CBS`) is broken.**
   `increment_op_cbs()` is only called from `commit_with_mode()`, but 20+ ops bypass it by calling `cb.commit(); cb.wait_until_completed()` directly. The metrics API reports incorrect values, making performance diagnosis unreliable.

5. **`sdpa_backward.rs:53,119,154,293` -- SDPA backward kernel has write races on dV and dK.**
   Non-atomic `+=` from parallel query-row threads makes gradients nondeterministic/incorrect.

6. **`conv_tiled.rs:158,161,173` -- conv2d_tiled threadgroup buffer hardcoded for 3x3.**
   `TILE_CI * 9` allocation but indexes with `kH*kW`; kernels >3x3 overflow threadgroup memory.
   **Status: FIXED (Phase 0, PR #38)** -- dynamic `TILE_CI*kH*kW` sizing.

7. **`fp8.rs:290,302,706,711` -- FP8 per-token quantization cross-threadgroup race.**
   Barrier is intra-threadgroup only, but dispatch can split one row across multiple threadgroups, racing on `scales[row]`.
   **Status: FIXED (Phase 0, PR #38)** -- split into 2-pass kernel (`compute_scales` + `apply_quant`).

8. **`binary.rs:452,654,664` -- binary_op_async dispatches on empty tensors.**
   Dispatches one thread even when output is empty (grid_numel=1), can touch `out[0]` for logical empty tensors.
   **Status: FIXED (Phase 0, PR #38)** -- early return for `numel==0`.

9. **`softmax.rs:553,554,556,598` -- softmax zero-dim dispatch.**
   Forces `num_rows=1` when shape product is zero; shapes like `[0, D]` dispatch and read/write past logical tensor bounds.
   **Status: FIXED (Phase 0, PR #38)** -- early return for zero-element shapes.

10. **`rope.rs:458,466,725,769` -- rope_ext_into_cb skips freq validation.**
    Skips frequency-table shape/range validation present in `rope_ext`, so malformed freq tensors cause OOB reads.
    **Status: FIXED (Phase 0, PR #38)** -- added shape/range validation matching `rope_ext`.

11. **`gemv.rs:591,617,691,717,778` -- GEMV doesn't validate vec/bias dtype.**
    Kernel selection based only on matrix dtype; mismatched dtype inputs cause misinterpreted memory/OOB reads.
    **Status: FIXED (Phase 0, PR #38)** -- added vec/mat/bias dtype consistency checks.

12. **`sdpa.rs:1089,1117,1164,1181,1319,1328,1356` -- SDPA forward/into-cb validation holes.**
    Forward doesn't enforce q/k/v/mask dtype consistency, can pass non-contiguous mask; into-cb lacks shape/dtype checks, hardcodes non-causal.
    **Status: FIXED (Phase 0, PR #38)** -- added q/k/v/mask validation.

13. **`sdpa_backward.rs:53,203,219,226` -- SDPA backward dtype unchecked.**
    Only checks q dtype; k/v/grad_output dtypes are unchecked despite float* kernel signature.

14. **`quantized.rs:1357,1372,1286,1287,1319` -- gather_qmm lacks shape/buffer-size validation.**
    Relies on caller-supplied dims; kernel does pointer arithmetic from them, malformed inputs drive OOB.
    **Status: FIXED (Phase 0, PR #38)** -- added shape/buffer-size/group_size checks.

15. **`gguf.rs:364,401` -- GGUF parser panics on alignment=0.**
    `div_ceil(0)` panic on malformed file.
    **Status: FIXED (Phase 0, PR #38)** -- guard + offset range validation.

## P1 -- Important Issues

### Code Quality

| ID   | Issue | Location |
|------|-------|----------|
| P1-1 | No async dispatch for most ops. Only binary, copy, and reduce have async paths; 20+ ops hardcode sync `commit` + `wait`. | Various op modules |
| P1-2 | LoRA `forward_gpu()` transposes the A matrix on CPU and re-uploads every call. The transposed buffer should be cached. | `lora.rs` |
| P1-3 | Metrics `snapshot()` uses `Relaxed` ordering, allowing torn reads of multi-word counters on concurrent threads. | `metrics.rs` |
| P1-4 | VJP GPU backward ops only support `Float32`. No f16/bf16 backward kernels exist, blocking mixed-precision training. | VJP modules |
| P1-5 | VJP tape backward does not accumulate gradients for diamond-shaped (fan-in) graphs; later gradients overwrite earlier ones. | VJP tape |
| P1-6 | `HasDType` only implemented for `f32` and `u32`. No `f16`/`bf16` support. | `dtype.rs` |
| P1-7 | `concat_many` uses pairwise reduction, allocating O(n) temporary GPU buffers instead of a single fused copy. | `concat.rs` |
| P1-8 | Many ops lack inline unit tests: gelu, silu, conv, layer_norm, unary, rope, sdpa, fp8, reduce. | Various |
| P1-9 | `vector_add.metal` is orphaned -- never registered or referenced by any Rust code. | `vector_add.metal` |
| P1-10 | Conv APIs don't validate stride>0 / dilation>0 -- division-based output-size math can divide by zero. | `conv.rs:512,690`, `conv_tiled.rs:316` |
| P1-11 | layer_norm/rms_norm don't validate weight/bias dtype -- shape checked but not dtype matching. | `layer_norm.rs:363,406`, `rms_norm.rs:470,515` |
| P1-12 | gather_mm doesn't validate x.dtype == weights.dtype -- sync contiguizes x/weights but not indices; into-cb does no contiguity handling. | `gather_mm.rs:316,321,347,399` |
| P1-13 | MoE integration incomplete -- topk_route returns [N*K] indices/weights but gather_mm/gather_qmm consumes only [batch] one-expert indices. No end-to-end top-k dispatch/merge path. | `topk_route.rs:389`, `gather_mm.rs:241,288`, `quantized.rs:1348` |
| P1-14 | topk_route uses raw buffers without contiguity normalization. | `topk_route.rs:528,553` |
| P1-15 | batched_qkv_proj is CB batching, not true kernel fusion -- 3 separate GEMMs encoded on one CB. Lacks matmul shape/dtype checks. | `fused.rs:52,55,58,61,99` |
| P1-16 | VJP coverage far below forward-op coverage -- GPU VJP only for add/mul/matmul. CPU VJP only for Add/Mul/MatMul/Softmax/RMSNorm/Reduce + Placeholder. | `vjp.rs:5,144,348`, `vjp_gpu.rs:18` |
| P1-17 | Quantized GGUF type mapping only supports Q4_0/Q4_1/Q8_0 -- many GGUF quant types map to None. Legacy `quantized_matmul` is a stub that always errors. | `quantized.rs:589,712`, `gguf.rs:207,216`, `quantized.rs:1539` |
| P1-18 | No dedicated GPU ops for sort/argsort/pad/pool/embedding/cross_entropy -- cross-entropy only in CPU-side LoRA trainer. | `lora.rs:317` |

### Missing Ops (vs MLX)

| Priority | Op Category | Details |
|----------|-------------|---------|
| High | FFT | fft, ifft, rfft, 2D, nD |
| High | Sort / ArgSort | Full sorting primitives |
| High | Scan | cumsum, cumprod |
| High | ArgReduce | argmin, argmax |
| High | Random | uniform, normal, bernoulli (required for training) |
| High | Slicing | slice, slice_update, dynamic_slice (fundamental) |
| Medium | Arange | Sequence generation |
| Medium | LogSumExp | Required for loss functions |
| Medium | Hadamard | Hadamard transform |
| Medium | Ternary / Where | Full conditional selection |
| Medium | MoE kernel | MLX has dedicated moe.h/moe.cpp |
| Medium | BlockMaskedMM, SegmentedMM, QQMatmul, GatherQMM | Specialized matmul variants |
| Medium | FP quantized (FP4, NAX) | Advanced quantization formats |
| Medium | Linalg | SVD, Cholesky, QR, eigenvalues, inverse |
| Medium | Fence / Event system | GPU synchronization primitives |
| Medium | Resident memory management | Memory residency sets |
| Low-Med | Einsum | Einstein summation |
| Low | Custom Metal kernel JIT | User-defined kernels |
| Low | Export / serialization | Model export |

### Missing Infrastructure (vs MLX)

| Priority | Feature | Gap |
|----------|---------|-----|
| Critical | Lazy evaluation / compute graph | MLX has full DAG-based lazy eval; rmlx executes eagerly |
| Critical | Graph compilation / fusion | MLX JIT graph fusion with simplification passes; rmlx has none |
| High | Transform system completeness | rmlx has VJP only; missing JVP, vmap, full transform composition |
| High | Stream / multi-queue scheduling | MLX has per-stream thread + multi-stream parallelism; rmlx is single-stream |
| Medium | Event-based sync (MTL::SharedEvent) | rmlx uses AtomicBool polling instead |
| Medium | Fence-based sync | Not implemented |
| Medium | Memory residency sets | Not implemented |

### Missing Serving Features (vs CUDA Frontier)

| Priority | Feature | Impact |
|----------|---------|--------|
| P0 | Paged Attention | vLLM PagedAttention for non-contiguous KV cache. rmlx SDPA assumes contiguous KV. Metal-feasible (proven by vllm-metal). |
| P0 | Prefix Caching | SGLang RadixAttention achieves 5.8x TTFT speedup. No prefix-sharing in rmlx. Requires paged attention first. |
| P0 | Continuous Batching | Dynamic add/evict at iteration level. rmlx BatchKvCache is static only. |
| P1 | Fused RMSNorm + Residual Add | Eliminates extra memory pass. Straightforward on Metal. |
| P1 | GEMM Epilogue Fusion | CUTLASS-style bias + activation fusion. rmlx matmul has no epilogue support. |
| P1 | Chunked Prefill | Split long prefill to maintain decode latency. rmlx processes full sequences. |
| P1 | Speculative Decoding | 2-2.5x decode speedup. No draft-verify kernel in rmlx. |
| P2 | Variable-Length Batched SDPA | Ragged batching via cu_seqlens. Avoids padding waste. |
| P2 | Sliding Window in SDPA Kernel | rmlx-nn builds CPU mask instead of kernel-level skip. |
| P2 | Fused W4A16 GEMM (AWQ/GPTQ) | rmlx uses dequant-then-matmul, resulting in 2x memory traffic. |
| P2 | FP8 GEMM | Direct FP8 matmul without dequant pass. |
| P2 | MoE Expert Fusion (Grouped GEMM) | Fuse Gate+Up across experts in a single dispatch. |

## P2 -- Minor Issues

| ID   | Issue |
|------|-------|
| P2-1 | Inconsistent error variants: some modules use `NotFound` while others use `InvalidShape` for unsupported dtype. |
| P2-2 | `Array::new` uses `debug_assert` for shape/strides validation; invalid inputs pass silently in release builds. |
| P2-3 | Quantized CPU reference functions use `debug_assert` for slice length validation -- potential UB in release. |
| P2-4 | `PrecisionGuard` thresholds are hardcoded; no runtime or config-based override. |
| P2-5 | `LoraConfig` does not validate that dropout is in [0, 1). |
| P2-6 | GGUF parser does not validate tensor offset ranges -- out-of-bounds access on corrupt files. **FIXED (Phase 0, PR #38)** -- added offset range validation. |
| P2-7 | `fused.rs` depends on matmul/silu kernel names implicitly with no `register()` function or enforced dependency. |
| P2-8 | Per-dispatch temporary `metal::Buffer` allocation for constants instead of a reusable pool. |
| P2-9 | Dead code: `im2col_f32` kernel defined but never dispatched -- host path uses `conv2d_tiled_f32` directly. `conv_tiled.rs:46,341` **FIXED (Phase 0, PR #38)** -- removed. |
| P2-10 | build.rs robustness -- stem-based output-name uniqueness (possible future collisions), multiple unwrap/expect/assert hard-panic build script. `build.rs:32,89,96,110` |
| P2-11 | DType::numel_to_bytes docs/behavior mismatch for quantized types -- docs say ceil/panic; impl is floor with debug_assert. `dtype.rs:81,87,91` **FIXED (Phase 0, PR #38)** -- corrected docs. |

## Optimization Depth Gaps

| Area | rmlx-core | MLX | CUDA Frontier |
|------|-----------|-----|---------------|
| **Matmul** | ~1200 lines, basic tiled GEMM | ~2600 lines Steel GEMM: split-K, gather, masked, segmented, NAX | CUTLASS 3.x: warp-specialized, epilogue fusion, FP8 |
| **SDPA** | Functional SDPA, contiguous KV only | Steel tiled attention + NAX variant | FlashAttention-2/3, PagedAttention, variable-length |
| **Reduce** | Single reduction strategy | Specialized row/col/all kernels | CUB block-reduce, warp-shuffle, multi-pass |
| **Quantized** | Single quantized matmul module | Int + FP quantized + NAX + FP4 + QQ matmul | AWQ/GPTQ fused W4A16, FP8, Marlin |
| **Indexing** | Single indexing module | 6 specialized kernels | Fused gather/scatter with atomics |
| **Binary** | Standard binary ops | `binary_two` for fused binary pairs | Fused elementwise via kernel fusion JIT |
| **Softmax** | Basic softmax kernel | Online softmax + specialized variants | Online softmax, fused with attention |
| **Normalization** | Separate RMSNorm kernel | Fused norm variants | Fused RMSNorm + residual + bias |

## rmlx-core Unique Strengths

These features are present in rmlx-core but absent from (or only available at a higher layer in) MLX:

1. **Built-in LoRA support** -- Native Rust LoRA with GPU forward path. MLX only has LoRA in the Python `mlx-lm` layer.
2. **Precision guard system** -- `PrecisionGuard` for automatic mixed-precision management at the op level.
3. **Runtime metrics** -- Atomic counters for op dispatches, command buffer usage, and allocation tracking (despite P0-4 accuracy bug).
4. **Structured logging** -- Integrated tracing/logging infrastructure.
5. **Graceful shutdown** -- Coordinated GPU resource cleanup on process exit.
6. **Dedicated activation kernels** -- Separate Metal kernels for GeLU and SiLU (MLX inlines these).
7. **MoE TopK routing** -- `topk_route.rs` with dedicated routing logic.

## Codex Supplementary Findings

The following findings were identified by Codex automated analysis and supplement the original audit above.

### P0 Additions (P0-5 through P0-15)

- **P0-5: SDPA backward write races** -- dV and dK accumulation uses non-atomic `+=` from parallel query-row threads, producing nondeterministic gradients. (`sdpa_backward.rs:53,119,154,293`)
- **P0-6: conv2d_tiled threadgroup overflow** -- Threadgroup buffer sized as `TILE_CI * 9` (hardcoded for 3x3) but indexed with `kH*kW`; any kernel larger than 3x3 overflows threadgroup memory. (`conv_tiled.rs:158,161,173`)
  **Status: FIXED (Phase 0, PR #38)** -- dynamic `TILE_CI*kH*kW` sizing.
- **P0-7: FP8 per-token quantization race** -- Threadgroup barrier is intra-group only; dispatch can split one row across multiple threadgroups, causing a race on `scales[row]`. (`fp8.rs:290,302,706,711`)
  **Status: FIXED (Phase 0, PR #38)** -- split into 2-pass kernel (`compute_scales` + `apply_quant`).
- **P0-8: binary_op_async empty-tensor dispatch** -- Dispatches one thread even when output is logically empty (`grid_numel=1`), potentially writing to `out[0]`. (`binary.rs:452,654,664`)
  **Status: FIXED (Phase 0, PR #38)** -- early return for `numel==0`.
- **P0-9: softmax zero-dim dispatch** -- Forces `num_rows=1` when shape product is zero; `[0, D]` shapes dispatch and access past logical tensor bounds. (`softmax.rs:553,554,556,598`)
  **Status: FIXED (Phase 0, PR #38)** -- early return for zero-element shapes.
- **P0-10: rope_ext_into_cb freq validation skip** -- Omits frequency-table shape/range validation present in `rope_ext`; malformed freq tensors cause OOB reads. (`rope.rs:458,466,725,769`)
  **Status: FIXED (Phase 0, PR #38)** -- added shape/range validation matching `rope_ext`.
- **P0-11: GEMV dtype mismatch** -- Kernel selection uses only matrix dtype; mismatched vec/bias dtypes cause misinterpreted memory or OOB reads. (`gemv.rs:591,617,691,717,778`)
  **Status: FIXED (Phase 0, PR #38)** -- added vec/mat/bias dtype consistency checks.
- **P0-12: SDPA forward/into-cb validation gaps** -- Forward does not enforce q/k/v/mask dtype consistency and accepts non-contiguous masks; into-cb lacks shape/dtype checks and hardcodes non-causal. (`sdpa.rs:1089,1117,1164,1181,1319,1328,1356`)
  **Status: FIXED (Phase 0, PR #38)** -- added q/k/v/mask validation.
- **P0-13: SDPA backward dtype unchecked** -- Only q dtype is validated; k/v/grad_output dtypes are unchecked despite float* kernel signatures. (`sdpa_backward.rs:53,203,219,226`)
- **P0-14: gather_qmm validation holes** -- No shape or buffer-size validation; kernel pointer arithmetic from caller-supplied dims can drive OOB access. (`quantized.rs:1357,1372,1286,1287,1319`)
  **Status: FIXED (Phase 0, PR #38)** -- added shape/buffer-size/group_size checks.
- **P0-15: GGUF parser div-by-zero** -- `div_ceil(0)` panics when alignment is zero in a malformed GGUF file. (`gguf.rs:364,401`)
  **Status: FIXED (Phase 0, PR #38)** -- guard + offset range validation.

### P1 Additions (P1-10 through P1-18)

- **P1-10: Conv stride/dilation not validated** -- Division-based output-size math can divide by zero when stride or dilation is zero. (`conv.rs:512,690`, `conv_tiled.rs:316`)
- **P1-11: layer_norm/rms_norm dtype unchecked** -- Weight/bias shape is validated but dtype matching is not. (`layer_norm.rs:363,406`, `rms_norm.rs:470,515`)
- **P1-12: gather_mm dtype mismatch** -- x.dtype == weights.dtype not validated; sync contiguizes x/weights but not indices; into-cb does no contiguity handling. (`gather_mm.rs:316,321,347,399`)
- **P1-13: MoE integration incomplete** -- topk_route returns `[N*K]` indices/weights but gather_mm/gather_qmm consumes only `[batch]` one-expert indices. No end-to-end top-k dispatch/merge path. (`topk_route.rs:389`, `gather_mm.rs:241,288`, `quantized.rs:1348`)
- **P1-14: topk_route contiguity** -- Uses raw buffers without contiguity normalization. (`topk_route.rs:528,553`)
- **P1-15: batched_qkv_proj is not true fusion** -- 3 separate GEMMs encoded on one command buffer; lacks matmul shape/dtype checks. (`fused.rs:52,55,58,61,99`)
- **P1-16: VJP coverage gap** -- GPU VJP only for add/mul/matmul. CPU VJP only for Add/Mul/MatMul/Softmax/RMSNorm/Reduce + Placeholder. (`vjp.rs:5,144,348`, `vjp_gpu.rs:18`)
- **P1-17: Quantized GGUF type coverage** -- Only Q4_0/Q4_1/Q8_0 supported; many GGUF quant types map to None. Legacy `quantized_matmul` is a stub that always errors. (`quantized.rs:589,712`, `gguf.rs:207,216`, `quantized.rs:1539`)
- **P1-18: Missing dedicated GPU ops** -- No sort/argsort/pad/pool/embedding/cross_entropy GPU kernels; cross-entropy only in CPU-side LoRA trainer. (`lora.rs:317`)

### P2 Additions (P2-9 through P2-11)

- **P2-9: Dead code im2col_f32** -- Kernel defined but never dispatched; host path uses `conv2d_tiled_f32` directly. (`conv_tiled.rs:46,341`)
  **Status: FIXED (Phase 0, PR #38)** -- removed.
- **P2-10: build.rs robustness** -- Stem-based output-name uniqueness risks future collisions; multiple `unwrap`/`expect`/`assert` hard-panic in build script. (`build.rs:32,89,96,110`)
- **P2-11: DType::numel_to_bytes mismatch** -- Docs say ceil/panic for quantized types; implementation is floor with `debug_assert`. (`dtype.rs:81,87,91`)
  **Status: FIXED (Phase 0, PR #38)** -- corrected docs.

## Recommended Action Plan

### Phase 0: Immediate P0 Fixes (COMPLETED -- PR #38)

Fixed 12/15 P0 issues and 3 P2 issues:
- C-P0-1 (copy.rs unreachable), C-P0-2 (softmax unwrap), C-P0-3 (quantized debug_assert)
- C-P0-6 (conv_tiled overflow), C-P0-7 (FP8 race), C-P0-8 (binary empty), C-P0-9 (softmax empty)
- C-P0-10 (RoPE freq), C-P0-11 (GEMV dtype), C-P0-12 (SDPA validation), C-P0-14 (gather_qmm), C-P0-15 (GGUF alignment)
- P2-6 (GGUF offset), P2-9 (dead im2col), P2-11 (dtype docs)

Remaining P0: C-P0-4 (op CB counter), C-P0-5 (SDPA backward races), C-P0-13 (SDPA backward dtype).

### Phase 1: Fix Remaining P0 Correctness (1-2 weeks)

1. Replace `unreachable!()` in `copy.rs` with `Result`-returning functions and propagate errors.
2. Replace `.unwrap()` in `softmax.rs:553` with proper error handling for zero-rank tensors.
3. Promote `debug_assert` in `dtype.rs:87` to a runtime check (`assert!` or `Result`) for quantized block alignment.
4. Audit all 20+ ops that call `cb.commit()` directly and route them through `commit_with_mode()` to fix the metrics counter.
5. Fix SDPA backward dV/dK write races with atomic accumulation or per-thread partitioning (P0-5).
6. Make conv2d_tiled threadgroup buffer size dynamic based on `kH*kW` (P0-6).
7. Fix FP8 per-token quantization to ensure one row is fully within one threadgroup, or use atomics for `scales[row]` (P0-7).
8. Guard binary_op_async and softmax against empty/zero-dim tensors with early returns (P0-8, P0-9).
9. Add freq validation to `rope_ext_into_cb` matching `rope_ext` (P0-10).
10. Add dtype consistency checks to GEMV, SDPA forward/backward, and gather_qmm (P0-11 through P0-14).
11. Guard GGUF parser against alignment=0 (P0-15).

### Phase 2: Serving Foundations (3-4 weeks)

1. Implement paged attention with non-contiguous KV cache support.
2. Build continuous batching infrastructure on top of paged KV.
3. Add prefix caching (RadixAttention-style).
4. Fuse RMSNorm + residual add into a single Metal kernel.

### Phase 3: Missing Critical Ops (3-4 weeks)

1. Implement slicing ops (slice, slice_update, dynamic_slice).
2. Implement sort / argsort.
3. Implement scan ops (cumsum, cumprod).
4. Implement argreduce (argmin, argmax).
5. Implement random number generation (uniform, normal, bernoulli).

### Phase 4: Infrastructure (6-8 weeks)

1. Design and implement lazy evaluation with a compute DAG.
2. Build graph compilation and kernel fusion passes.
3. Add multi-stream scheduling with proper event-based synchronization.
4. Complete the transform system (JVP, vmap).

### Phase 5: Optimization Depth (ongoing)

1. Expand matmul to split-K, masked, and epilogue-fused variants.
2. Implement chunked prefill and speculative decoding.
3. Add fused W4A16 GEMM and FP8 GEMM.
4. Expand reduce to specialized row/col/all strategies.
5. Fix VJP for f16/bf16 and diamond-graph gradient accumulation.

### Phase 6: Polish (ongoing)

1. Resolve all P2 issues (error variant consistency, debug_assert promotions, config validation).
2. Add comprehensive unit tests for untested ops.
3. Remove orphaned `vector_add.metal`.
4. Implement constant buffer pooling to replace per-dispatch allocations.

## Summary

| Severity | Count | Fixed (Phase 0) | Remaining |
|----------|-------|-----------------|-----------|
| P0       | 15    | 12              | 3         |
| P1       | 18    | 0               | 18        |
| P2       | 11    | 3               | 8         |
