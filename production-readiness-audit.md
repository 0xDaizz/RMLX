# RMLX Production Readiness Audit — Integrated Report

**Date:** 2026-03-06
**Project:** RMLX — Rust ML Framework for Apple Silicon
**Auditors:** Claude Opus 4.6, Codex (gpt-5.3-codex)
**References:** MLX (`local MLX checkout`), NCCL, DeepEP, Megascale, Tutel, FasterMoE, vLLM, PyTorch MPS, torchrun, DeepSpeed
**Source audits:** `docs/roadmap/audit-rmlx-{cli,metal,alloc,core,rdma,distributed,nn}.md`

---

## 1. Executive Summary

RMLX is a 7-crate Rust ML framework targeting Apple Silicon Metal GPU with RDMA-over-Thunderbolt-5 distributed computing. The project has **genuine architectural strengths** — zero-copy RDMA pipeline, FP8 wire quantization, async compute↔comm overlap, and credit-based flow control that are best-in-class for Apple Silicon. However, the project is **NOT production-ready** due to:

- **31 P0 critical issues** across all crates (correctness bugs, unsafe UB, stub implementations, missing serving infrastructure)
- **4 critical cross-crate integration failures** (distributed MoE not wired, Metal kernels not registered, MrPool bypassed, TP forward incomplete)
- **rmlx-nn has 2 stub `forward()` methods** that return garbage output (MLA, SlidingWindowAttention)
- **rmlx-core has 15 P0 issues** including GPU write races, threadgroup overflow, and empty-tensor dispatch bugs
- **No sampling, no continuous batching, no paged attention** — the system literally cannot generate text

**Overall Production Readiness: 3/10** (individual crate scores: metal 5/10, alloc 4/10, core 3/10, rdma 6/10, distributed 7/10, nn 2/10, cli 4/10)

---

## 2. Architecture Overview

```
┌─────────────┐
│  rmlx-cli   │  SSH launcher, hostfile, config
└──────┬──────┘
       │ env vars (RMLX_RANK, RMLX_BACKEND, RMLX_COORDINATOR)
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  rmlx-nn    │────▶│  rmlx-core  │────▶│ rmlx-metal  │
│  Layers     │     │  Ops/Kernels│     │ GPU abstraction│
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │            ┌──────┴──────┐     ┌──────┴──────┐
       │            │ rmlx-alloc  │     │  (Metal GPU)│
       │            │ Memory mgmt │     └─────────────┘
       │            └─────────────┘
       ▼
┌──────────────────┐     ┌─────────────┐
│ rmlx-distributed │────▶│  rmlx-rdma  │
│ EP/MoE/Collectives│    │ TB5 RDMA    │
└──────────────────┘     └─────────────┘
```

**Dependency flow:** cli → (env) → nn → core → metal/alloc; nn → distributed → rdma → metal

---

## 3. Cross-Crate Integration Issues

These issues span crate boundaries and cannot be found by auditing crates individually. They represent the most impactful blockers for end-to-end functionality.

### X-P0: Critical Cross-Crate Failures

#### X-P0-1: Distributed MoE Exchange Not Wired Into NN Forward Paths
- **Crates:** `rmlx-nn/moe.rs` → `rmlx-distributed/moe_exchange.rs`
- **Description:** `MoeLayer.exchange` is `Option<MoeDispatchExchange>` but is never actually invoked during forward(). When `world_size > 1`, tokens routed to non-local experts are silently dropped. The entire distributed MoE pipeline (dispatch → remote expert compute → combine) is broken.
- **Impact:** Multi-node MoE inference produces incorrect results — expert contributions from remote ranks are lost.
- **Fix:** Wire `MoeDispatchExchange::dispatch()` before expert compute and `::combine()` after in all MoE forward strategies (PerExpert, Batched, Pipeline).

#### X-P0-2: MoE Metal Kernels Defined But Not Registered
- **Crates:** `rmlx-distributed/moe_exchange.rs` → `rmlx-core/kernels/mod.rs` + `rmlx-core/build.rs`
- **Description:** `moe_exchange.rs` defines Metal shader source inline (moe_gather_scatter, dispatch_local, dispatch_scatter_remote, etc.) but never registers them with `KernelRegistry`. rmlx-core's `build.rs` doesn't know about distributed crate's Metal code. These kernels will fail at runtime lookup.
- **Impact:** MoE dispatch Metal shaders cannot execute.
- **Fix:** Either (a) register kernels via JIT in distributed crate init, or (b) move Metal source to rmlx-core's build.rs compilation.

#### X-P0-3: EP Dispatch Bypasses MrPool — Ad-Hoc MR Registration Per Call
- **Crates:** `rmlx-distributed/moe_exchange.rs` → `rmlx-rdma/mr_pool.rs`
- **Description:** EP dispatch/combine code path does not use pre-registered SharedBuffers from MrPool. Each dispatch potentially does fresh `ibv_reg_mr` (~100us overhead). MrPool exists in rmlx-rdma with 6 tiers but is not integrated.
- **Impact:** ~100us per-dispatch overhead instead of <1us from pre-registered pool.
- **Fix:** Wire `EpRuntimeContext.shared_buffer_pool` into moe_exchange route/combine methods.

#### X-P0-4: Tensor Parallel Forward Not Implemented
- **Crates:** `rmlx-nn/parallel.rs` → `rmlx-distributed/group.rs`
- **Description:** `ColumnParallelLinear` and `RowParallelLinear` structs exist but have no `forward()` implementation wired to Group allreduce/allgather. Tensor parallelism is structurally defined but non-functional.
- **Impact:** TP model sharding doesn't work.
- **Fix:** Implement forward() with Group::allreduce() for RowParallel and Group::allgather() for ColumnParallel.

### X-P1: High Cross-Crate Issues

#### X-P1-1: CLI Backend Enum Mismatch with Distributed Init
- **Crates:** `rmlx-cli/launch.rs` → `rmlx-distributed/init.rs`
- **Description:** `rmlx-cli` accepts `ring` as a backend value (`config.rs:13,278-280`) and passes it raw via the `RMLX_BACKEND` environment variable with no validation (`launch.rs:25-26,105`). However, `rmlx-distributed::init` only recognizes `rdma` and `loopback` (`init.rs:37-44,140-143,541-556`) — unknown values silently fall to `Auto` which then resolves to loopback fallback. No validation on either side.

#### X-P1-2: RMLX_COORDINATOR Uses SSH Target Not Control IP
- **Crates:** `rmlx-cli/launch.rs` → `rmlx-distributed/init.rs`
- **Description:** CLI sets coordinator to `hosts[0]` (SSH hostname), ignoring probed control IPs. In multi-network setups (TB5 data + Ethernet control), RDMA bootstrap fails.

#### X-P1-3: Hostfile RDMA Mapping Orphaned
- **Crates:** `rmlx-cli/config.rs` → `rmlx-cli/launch.rs`
- **Description:** `config` writes RDMA device mesh mapping to hostfile, but `launch` only reads the `ssh` field. Never exports `RMLX_IBV_DEVICES`.

#### X-P1-4: ProgressEngine Not Used in EP Dispatch
- **Crates:** `rmlx-distributed/moe_exchange.rs` → `rmlx-rdma/progress.rs`
- **Description:** Transport has ProgressEngine integration but EP dispatch/combine bypasses it, doing raw post_send/post_recv. No async progress tracking for EP operations.

#### X-P1-5: rmlx-core gather_mm Unused by rmlx-nn MoE
- **Crates:** `rmlx-nn/moe.rs` → `rmlx-core/ops/gather_mm.rs`
- **Description:** Batched gather_mm kernel exists in rmlx-core but MoE layer reimplements gathering via per-expert loops instead. Suboptimal dispatch pattern.

#### X-P1-6: ZeroCopyBuffer Completion Tracking Not Integrated with GpuEvent
- **Crates:** `rmlx-alloc/zero_copy.rs` → `rmlx-metal/event.rs`
- **Description:** CompletionTicket requires manual `mark_gpu_complete()` call. No automatic integration with GpuEvent signaling. Forgetting the call leaks the buffer indefinitely.

#### X-P1-7: ICB Sparse Expert Dispatch Not Wired (EP-7 Planned)
- **Crates:** `rmlx-metal/icb_sparse.rs` → `rmlx-nn/moe.rs` → `rmlx-distributed/moe_exchange.rs`
- **Description:** EP-6 implemented `SparseExpertPlan` and `IcbReplay` infrastructure in rmlx-metal. `forward_sparse_icb()` in `moe.rs` builds the ICB plan and capacity vector but discards them (`let _ = sparse_plan; let _ = capacity;`). The Metal ICB indirect dispatch path (EP-7) is planned but not implemented. This blocks GPU-level empty-expert skipping — currently all experts are dispatched even when some have zero tokens.
- **Impact:** Wasted GPU compute on empty experts in sparse MoE. Performance gap vs CUDA implementations that skip empty experts at kernel level.
- **Fix:** Implement EP-7 as planned in `docs/roadmap/phases.md`:
  1. `ExpertGroup::grouped_forward_icb()` — accepts `SparseExpertPlan` + per-expert capacity, encodes only active experts via Metal ICB indirect dispatch
  2. Wire `IcbReplay` integration — cache compiled ICB commands per sparsity pattern, replay without re-encoding
  3. `forward_sparse_icb()` in `moe.rs` — call `grouped_forward_icb()` instead of `grouped_forward()`
  4. Fix the SwiGLU 2D grid issue (M-P0-2) as prerequisite

### X-P2: Medium Cross-Crate Issues

#### X-P2-1: Large Dead Re-Export Surfaces in lib.rs Files
- **Crates:** `rmlx-distributed/lib.rs`, `rmlx-core/lib.rs`, `rmlx-metal/lib.rs`, `rmlx-alloc/lib.rs`, `rmlx-rdma/lib.rs`
- **Description:** All crates re-export many symbols in `lib.rs` but downstream usage is narrow. E.g., `rmlx-distributed` re-exports ~20 symbols but `rmlx-nn` only uses `MoeDispatchExchange`; `rmlx-core` re-exports ~10 symbols but downstream only uses `MetalAllocator`.
- **Impact:** Increased compile times, confusing public API surface, accidental coupling.
- **Fix:** Audit and trim re-exports to actual downstream usage. Consider `pub(crate)` for internal-only symbols.

#### X-P2-2: Error Propagation Loses Structured Types at Crate Boundaries
- **Crates:** `rmlx-distributed/group.rs`, `rmlx-distributed/init.rs`, `rmlx-distributed/transport.rs`, `rmlx-nn/parallel.rs`
- **Description:** `DistributedError` is string-backed (`group.rs:8-33`). RDMA and kernel errors are converted to strings at every boundary (`init.rs:308,336,338,340,345,369`, `transport.rs:838-851`, `parallel.rs:270,278,425`) instead of using typed `From` propagation.
- **Impact:** Callers cannot pattern-match on source errors across crate boundaries. Debugging requires string parsing.
- **Fix:** Add typed error variants with `From` impls for each source error type.

#### X-P2-3: Device vs DeviceRef Type Mismatch Forces Workarounds
- **Crates:** `rmlx-core/array.rs`, `rmlx-nn/parallel.rs`
- **Description:** `Array::from_bytes` requires `&metal::Device` (`array.rs:309-313`) but `rmlx-nn` holds `&metal::DeviceRef`, forcing adapter function `array_from_raw_bytes` (`parallel.rs:111-117,312-316,439`).
- **Impact:** Unnecessary wrapper code, confusing API boundary.
- **Fix:** Unify on `&metal::DeviceRef` at core API boundaries.

#### X-P2-4: vector_add.metal Kernel Orphaned — Compiled But Never Called
- **Crates:** `rmlx-core/kernels/vector_add.metal`, `rmlx-core/build.rs`
- **Description:** `build.rs` compiles all `.metal` sources into metallib (`build.rs:59-61,86-123`) including `vector_add_float`, but no Rust-side dispatch or registry lookup references this kernel.
- **Impact:** Dead code in metallib, wasted compile time, confusing for developers.
- **Fix:** Either wire the kernel to a Rust-side op or remove the `.metal` file.

---

## 4. Per-Crate P0 Issues (Unified)

### 4.1 rmlx-metal (3 P0)

| ID | Issue | Location |
|----|-------|----------|
| M-P0-1 | CaptureScope FFI memory corruption — wrong ObjC pointer, non-null-terminated string | `capture.rs:136,150-155` |
| M-P0-2 | SparseExpertPlan SwiGLU grid under-dispatch by factor of intermediate_dim | `icb_sparse.rs:192-201` |
| M-P0-3 | ExecGraph submit_batch returns unsignaled token → GPU/CPU deadlock | `exec_graph.rs:128-134` |

### 4.2 rmlx-alloc (1 P0, 3 P1)

| ID | Issue | Location |
|----|-------|----------|
| A-P0-1 | Zero-size allocation cache poisoning — returns too-small buffer for subsequent allocs | `allocator.rs:114, cache.rs:42,73` |
| A-P1-1 | Memory limit bypass under concurrency (non-atomic reservation) | `allocator.rs:121-130` |
| A-P1-2 | Stats underflow/corruption on double-free or foreign buffer free | `allocator.rs:190, stats.rs:43` |
| A-P1-3 | Metal 3 ObjC runtime availability gap → process abort on pre-macOS 15 | `residency.rs:75-85` |

### 4.3 rmlx-core (15 P0)

| ID | Issue | Location |
|----|-------|----------|
| C-P0-1 | `copy.rs` unreachable!() panics for unsupported dtypes | `copy.rs:567,578,588` |
| C-P0-2 | softmax .unwrap() on zero-rank tensor | `softmax.rs:553` |
| C-P0-3 | Quantized block-alignment debug_assert only → undersized GPU buffers in release | `dtype.rs:87` |
| C-P0-4 | TOTAL_OP_CBS counter broken — 20+ ops bypass commit_with_mode() | Various ops |
| C-P0-5 | SDPA backward dV/dK non-atomic write races → nondeterministic gradients | `sdpa_backward.rs:53,119,154,293` |
| C-P0-6 | conv2d_tiled threadgroup buffer hardcoded for 3x3 → overflow for larger kernels | `conv_tiled.rs:158,161,173` |
| C-P0-7 | FP8 per-token quantization cross-threadgroup race on scales[row] | `fp8.rs:290,302,706,711` |
| C-P0-8 | binary_op_async dispatches on empty tensors | `binary.rs:452,654,664` |
| C-P0-9 | softmax zero-dim dispatch reads past tensor bounds | `softmax.rs:553,554,556,598` |
| C-P0-10 | rope_ext_into_cb skips freq validation → OOB reads | `rope.rs:458,466,725,769` |
| C-P0-11 | GEMV doesn't validate vec/bias dtype → misinterpreted memory | `gemv.rs:591,617,691,717,778` |
| C-P0-12 | SDPA forward/into-cb validation holes (dtype, contiguity, causal) | `sdpa.rs:1089,1117,1164,1181` |
| C-P0-13 | SDPA backward dtype unchecked for k/v/grad_output | `sdpa_backward.rs:53,203,219,226` |
| C-P0-14 | gather_qmm no shape/buffer-size validation → OOB kernel access | `quantized.rs:1357,1372,1286` |
| C-P0-15 | GGUF parser div-by-zero on alignment=0 | `gguf.rs:364,401` |

### 4.4 rmlx-rdma (2 P0)

| ID | Issue | Location |
|----|-------|----------|
| R-P0-1 | Collectives only support f32 — cannot use for f16/bf16 ML tensors | `collectives.rs` (entire file) |
| R-P0-2 | No reliability layer for UC transport — silent data corruption on packet loss | `qp.rs` (UC mode throughout) |

### 4.5 rmlx-distributed (3 P0 + 2 Codex P0)

| ID | Issue | Location |
|----|-------|----------|
| D-P0-1 | read_buffer_bytes/f32 unsound unsafe + panic | `moe_exchange.rs:46-74` |
| D-P0-2 | EP dispatch/combine correctness bugs (routing, weight accumulation, async lifecycle) | `moe_exchange.rs`, `ep_runtime.rs` |
| D-P0-3 | blocking_exchange_v3 deadlock on asymmetric send/recv | `v3_protocol.rs` |
| D-P0-4 | Async combine uses global expert index against local-only buffer (Codex) | `moe_exchange.rs:3158-3164` |
| D-P0-5 | Zero-copy RDMA PendingOp doesn't own buffer/MR lifetime → use-after-free (Codex) | `transport.rs:549,765,802` |

### 4.6 rmlx-nn (9 P0)

| ID | Issue | Location |
|----|-------|----------|
| N-P0-1 | MLA forward() is a stub — returns o_proj(x) with no attention | `mla.rs:368-401` |
| N-P0-2 | SlidingWindowAttention forward() is a stub — discards Q/K/V | `sliding_window.rs:145-177` |
| N-P0-3 | RotatingKvCache wrap-around returns uninitialized memory | `attention.rs:490,589,599,683` |
| N-P0-4 | Distributed MoE silently drops non-local expert contributions | `moe.rs:406,513,606,277` |
| N-P0-5 | QuantizedLinear Q8 fallback panics for non-Float32 | `quantized_linear.rs:191,206` |
| N-P0-6 | No sampling layer (top-k, top-p, temperature) — cannot generate text | Entire crate |
| N-P0-7 | No continuous batching — single-request throughput only | All forward() signatures |
| N-P0-8 | No paged KV cache — wasteful max-length allocation | All KV caches |
| N-P0-9 | No FlashAttention Metal — uncompetitive attention performance | SDPA placeholder |

### 4.7 rmlx-cli (1 P0)

| ID | Issue | Location |
|----|-------|----------|
| L-P0-1 | SSH option injection via unvalidated host/user input | `ssh.rs:48-64,122-132` |

---

## 5. Unified P1 Issue Summary

| Crate | Count | Key Issues |
|-------|-------|------------|
| rmlx-metal | 2 | ExecGraph unsignaled token deadlock, ICB naming misleading |
| rmlx-alloc | 3 | Concurrency limit bypass, stats underflow, Metal 3 availability |
| rmlx-core | 18 | No async ops, LoRA CPU transpose, VJP f32-only, missing ops (FFT/sort/scan/random/slice), MoE integration incomplete |
| rmlx-rdma | 4 | No pipelined circular buffers, SharedBuffer scalability, ProgressEngine health, collectives don't use ProgressEngine |
| rmlx-distributed | 7 | Ring allreduce chunk rounding, MoePolicy thread safety, fake warmup benchmarks, no fault tolerance, no tree algorithms, f16/bf16 edge cases, SlabRing no backpressure |
| rmlx-nn | 20 | MoE excessive alloc, QuantizedLinear re-uploads weights, Linear N command buffers, MLA/cache stubs, GGUF silently passes unrecognized patterns, TransformerBlock ignores ff_type, missing RMSNorm/RoPE/activations layers |
| rmlx-cli | 7 | Backend enum mismatch, coordinator IP wrong, RDMA mapping orphaned, no signal forwarding, no ConnectTimeout, no backend-aware launch, no TB5 topology discovery |

**Total P1: 61 issues**

---

## 6. Production Readiness Scorecard

| Crate | Correctness | Performance | Feature Completeness | Test Coverage | Cross-Crate Integration | Overall |
|-------|-------------|-------------|---------------------|---------------|------------------------|---------|
| rmlx-metal | 4/10 | 6/10 | 6/10 | 5/10 | 7/10 | **5/10** |
| rmlx-alloc | 3/10 | 5/10 | 4/10 | 3/10 | 5/10 | **4/10** |
| rmlx-core | 2/10 | 4/10 | 3/10 | 3/10 | 5/10 | **3/10** |
| rmlx-rdma | 5/10 | 5/10 | 6/10 | 3/10 | 6/10 | **6/10** |
| rmlx-distributed | 5/10 | 8/10 | 7/10 | 4/10 | 4/10 | **7/10** |
| rmlx-nn | 1/10 | 3/10 | 2/10 | 1/10 | 2/10 | **2/10** |
| rmlx-cli | 5/10 | 7/10 | 4/10 | 6/10 | 3/10 | **4/10** |
| **Cross-Crate** | — | — | — | — | **3/10** | **3/10** |

**Project-wide: 3/10** — Weighted by impact on end-to-end inference capability.

---

## 7. Comparative Summary vs MLX and CUDA Frontier

| Capability | RMLX | MLX | NCCL/DeepEP/vLLM |
|---|---|---|---|
| Zero-copy GPU→RDMA | ★★★★★ | ★★★★ | ★★★★ (GDR) |
| FP8 wire quantization | ★★★★★ | ★★★ | ★★★★ (DeepEP) |
| Async compute↔comm overlap | ★★★★ | ★★★ | ★★★★★ |
| Metal GPU abstraction | ★★★ | ★★★★ | N/A |
| Memory allocator | ★★ | ★★★★ | ★★★★★ (PyTorch CCA) |
| Collective algorithms | ★★ (ring only) | ★★★ (ring+MPI) | ★★★★★ (tree+ring+SHARP) |
| Op coverage | ★★ (~25 ops) | ★★★★★ (~80+ ops) | ★★★★★ |
| NN layer coverage | ★★ (~15 layers) | ★★★★★ (~50+ layers) | ★★★★★ |
| Attention kernels | ★ (placeholder SDPA) | ★★★★ (Steel attention) | ★★★★★ (FlashAttention) |
| Serving infrastructure | ☆ (none) | ★★ (basic) | ★★★★★ (vLLM) |
| Fault tolerance | ★ | ★★ | ★★★★ (Megascale) |
| Python API | ☆ | ★★★★★ | ★★★★★ |
| Wire protocol | ★★★★ (v3 var-len) | ★★★ | ★★★★ |
| Flow control | ★★★★ (credit-based) | ★★★ | ★★★★★ |
| Test coverage | ★★ | ★★★★ | ★★★★★ |

---

## Phase 0 Status: ✅ COMPLETE (2026-03-06)

All P0 safety issues across rmlx-core, rmlx-metal, rmlx-alloc, rmlx-cli, and rmlx-distributed have been fixed in PR #38.
Key fixes: FP8 race condition (2-pass kernel split), conv2d threadgroup overflow, CaptureScope FFI rewrite, SSH injection prevention, ZeroCopyPendingOp UAF fix, and comprehensive dtype/shape validation across GEMV/SDPA/gather_qmm/RoPE/GGUF.

## 8. Unified Remediation Roadmap

### Phase 0: Triage and Emergency Fixes (Week 1)

**Goal:** Make the system not crash, not corrupt memory, not produce garbage.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 0.1 | Replace all `unreachable!()`/`unwrap()`/`assert!()` in non-test production paths with `Result` | core, distributed, metal, nn | C-P0-1, C-P0-2, D-P0-1, M-P0-3, N-P0-5 |
| 0.2 | Fix SDPA backward write races with atomic accumulation | core | C-P0-5 |
| 0.3 | Fix conv2d_tiled threadgroup overflow for non-3x3 | core | C-P0-6 |
| 0.4 | Fix FP8 cross-threadgroup race on scales | core | C-P0-7 |
| 0.5 | Guard empty/zero-dim tensor dispatch (binary, softmax) | core | C-P0-8, C-P0-9 |
| 0.6 | Add dtype validation to GEMV, SDPA, gather_qmm, RoPE | core | C-P0-10 through C-P0-14 |
| 0.7 | Fix GGUF parser alignment=0 | core | C-P0-15 |
| 0.8 | Promote `debug_assert` to `assert` for quantized block alignment | core | C-P0-3 |
| 0.9 | Fix zero-size allocation cache poisoning | alloc | A-P0-1 |
| 0.10 | Fix CaptureScope FFI (device pointer, CString, autorelease) | metal | M-P0-1 |
| 0.11 | Fix SwiGLU dispatch to 2D grid | metal | M-P0-2 |
| 0.12 | Guard ExecGraph submit_batch against empty CB | metal | M-P0-3 |
| 0.13 | Fix SSH option injection | cli | L-P0-1 |

### Phase 1: Correctness and Integration (Weeks 2-3)

**Goal:** Make end-to-end inference produce correct results.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 1.1 | Implement MLA forward() (RoPE + SDPA + cache update) | nn | N-P0-1 |
| 1.2 | Implement SlidingWindowAttention forward() | nn | N-P0-2 |
| 1.3 | Fix RotatingKvCache wrap-around linearize/trim | nn | N-P0-3 |
| 1.4 | Fix QuantizedLinear dtype handling | nn | N-P0-5 |
| 1.5 | Wire MoeDispatchExchange into MoE forward paths | nn, distributed | X-P0-1, N-P0-4 |
| 1.6 | Register distributed Metal kernels with KernelRegistry | distributed, core | X-P0-2 |
| 1.7 | Wire MrPool into EP dispatch/combine | distributed, rdma | X-P0-3 |
| 1.8 | Implement TP forward (ColumnParallel/RowParallel + Group allreduce) | nn, distributed | X-P0-4 |
| 1.9 | Fix EP dispatch/combine correctness (routing, async combine index) | distributed | D-P0-2, D-P0-4 |
| 1.10 | Fix blocking_exchange_v3 deadlock with non-blocking exchange | distributed | D-P0-3 |
| 1.11 | Fix PendingOp lifetime ownership for zero-copy RDMA | distributed | D-P0-5 |
| 1.12 | Generalize RDMA collectives to f16/bf16 | rdma | R-P0-1 |
| 1.13 | Add application-level CRC32 for UC transport | rdma | R-P0-2 |
| 1.14 | Align CLI backend enum with distributed init | cli, distributed | X-P1-1 |
| 1.15 | Fix RMLX_COORDINATOR to use control IP | cli | X-P1-2 |

### Phase 2: Minimum Viable Serving (Weeks 4-6)

**Goal:** Generate text from a loaded model.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 2.1 | Implement Sampler module (top-k, top-p, temperature) | nn | N-P0-6 |
| 2.2 | Implement RMSNorm nn layer | nn | Missing layer |
| 2.3 | Implement standalone RoPE layer | nn | Missing layer |
| 2.4 | Wire TransformerBlock FFN type (Dense/MoE/Gated) | nn | P1-6 |
| 2.5 | Implement paged KV cache + block manager | nn, core | N-P0-8 |
| 2.6 | Implement continuous batching scheduler | nn | N-P0-7 |
| 2.7 | Implement FlashAttention-2 Metal kernel | core, metal | N-P0-9 |
| 2.8 | Route all ops through commit_with_mode() for metrics | core | C-P0-4 |

### Phase 3: Distributed Correctness and EP-7 ICB (Weeks 7-9)

**Goal:** Multi-node MoE inference works correctly with GPU-level expert skipping.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 3.1 | Fix ring allreduce chunk rounding | distributed | D-P1-1 |
| 3.2 | Fix MoePolicy thread safety | distributed | D-P1-2 |
| 3.3 | Replace fake warmup benchmarks with real Metal/RDMA calibration | distributed | D-P1-3 |
| 3.4 | Add heartbeat/watchdog fault detection | distributed | D-P1-4 |
| 3.5 | Add SlabRing backpressure | distributed | D-P1-7 |
| 3.6 | Wire ProgressEngine into EP dispatch | distributed, rdma | X-P1-4 |
| 3.7 | Implement pipelined circular buffers for ring allreduce | rdma | R-P1-1 |
| 3.8 | Add signal forwarding to CLI launcher | cli | L-P1-4 |
| 3.9 | **Implement EP-7: ICB Full Metal Indirect Dispatch** | metal, nn | X-P1-7 |
|   | 3.9a: `ExpertGroup::grouped_forward_icb()` — encode only active experts via Metal ICB indirect dispatch | nn | |
|   | 3.9b: Wire `IcbReplay` cache — compile ICB commands per sparsity pattern, replay without re-encoding | metal, nn | |
|   | 3.9c: `forward_sparse_icb()` in moe.rs — call `grouped_forward_icb()` instead of `grouped_forward()` | nn | |
|   | 3.9d: ExpertGroup internal refactoring for ICB-encoded GEMM dispatch | nn | |
|   | Prereqs: M-P0-2 (SwiGLU 2D grid fix), X-P0-1 (MoE exchange wiring) | | |

### Phase 4: Performance and Allocator (Weeks 10-12)

**Goal:** Competitive performance for single-node and multi-node.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 4.1 | Fix alloc concurrency limit bypass (atomic CAS) | alloc | A-P1-1 |
| 4.2 | Fix alloc stats underflow + ownership validation | alloc | A-P1-2 |
| 4.3 | Wire SmallBufferPool, LeakDetector, ResidencyManager into MetalAllocator | alloc | Orphan modules |
| 4.4 | Add HazardTrackingModeUntracked to buffer creation | metal, alloc | MLX parity |
| 4.5 | Add chip-class tuning (M1/M3/M4 thresholds) | metal | F1 |
| 4.6 | Add binary archive / disk pipeline cache | metal | F7 |
| 4.7 | Implement tree allreduce for small tensors | distributed | D-P1-5 |
| 4.8 | Fix f16/bf16 conversion edge cases (NaN, subnormal) | distributed, core | D-P1-6 |
| 4.9 | Implement block splitting and coalescing (BFC-style) | alloc | CUDA frontier |
| 4.10 | Fuse RMSNorm + residual add kernel | core | Serving feature |
| 4.11 | Use gather_mm in MoE instead of per-expert loops | nn, core | X-P1-5 |

### Phase 5: Feature Breadth (Weeks 13-16)

**Goal:** Support major model architectures and quantization formats.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 5.1 | Implement missing critical ops (slice, sort, scan, argreduce, random) | core | MLX parity |
| 5.2 | Add 15+ activation functions | nn | MLX parity |
| 5.3 | Implement AWQ/GPTQ/k-quant ingestion | nn | P1-16 |
| 5.4 | Add model configs (Gemma, Phi, Mistral, Command-R) | nn | P2-5 |
| 5.5 | Implement prefix caching (RadixAttention-style) | nn | Serving feature |
| 5.6 | Implement chunked prefill | nn, core | Serving feature |
| 5.7 | Add TB5 topology discovery to CLI | cli | L-P1-7 |
| 5.8 | Implement topology-aware routing in distributed | distributed | D-P2-2 |
| 5.9 | Add backend-aware launch paths to CLI | cli | L-P1-6 |

### Phase 6: Infrastructure and Polish (Weeks 17+)

**Goal:** Production monitoring, lazy eval, Python bindings.

| # | Task | Crates | Issues |
|---|------|--------|--------|
| 6.1 | Design and implement lazy evaluation with compute DAG | core | MLX parity (critical infra) |
| 6.2 | Build graph compilation and kernel fusion passes | core | MLX parity (critical infra) |
| 6.3 | Add multi-stream scheduling | metal, core | MLX parity |
| 6.4 | Add latency histogram metrics (p50/p99) | distributed | D-P2-4 |
| 6.5 | PyO3 bindings for core types | all | D-P2-6 |
| 6.6 | Composable allocator trait (RMM-inspired) | alloc | CUDA frontier |
| 6.7 | Replace all eprintln! with tracing crate | rdma, cli | Various P3 |
| 6.8 | Comprehensive test coverage drive (target >80%) | all | All test gaps |

---

## 9. Issue Count Summary

| Severity | metal | alloc | core | rdma | distributed | nn | cli | cross-crate | **Total** |
|----------|-------|-------|------|------|-------------|-----|-----|-------------|-----------|
| **P0** | 3 | 1 | 15 | 2 | 5 | 9 | 1 | 4 | **40** |
| **P1** | 2 | 3 | 18 | 4 | 7 | 20 | 7 | 7 | **68** |
| **P2** | 5 | 5 | 11 | 6 | 8 | 13 | 7 | 4 | **59** |
| **P3** | 2 | 1 | — | 2 | 5 | — | 6 | — | **16** |
| **Total** | 12 | 10 | 44 | 14 | 25 | 42 | 21 | 15 | **183** |

---

## 10. Key Architectural Decisions Needed

Before implementing the roadmap, these architectural decisions should be made:

1. **Lazy vs Eager execution:** MLX uses lazy eval with DAG compilation. Adding this to RMLX is a ~6-8 week effort that touches every op. Should this be Phase 6 or moved earlier? (Recommendation: Phase 6 — correctness first, lazy eval is an optimization.)

2. **Serving architecture:** Should RMLX build its own serving runtime (like vLLM) or be a backend for an existing framework? This affects whether continuous batching and paged attention live in rmlx-nn or a separate crate.

3. **Python bindings priority:** If researcher adoption matters, PyO3 bindings should move to Phase 3-4. If Rust-native is the target, Phase 6 is fine.

4. **Training support scope:** VJP system exists but is incomplete (f32 only, no JVP/vmap). Should training be in scope? This affects whether SDPA backward races (C-P0-5) are urgent.

5. **UC vs RC RDMA:** Currently UC-only. For TB5 direct cable this is fine. If broader deployment is planned, RC mode is needed.

---

## Appendix A: Individual Audit References

For detailed per-crate findings with full line-number references, see:

- `docs/roadmap/audit-rmlx-cli.md` — 184 lines, 1 P0 / 7 P1 / 7 P2 / 6 P3
- `docs/roadmap/audit-rmlx-metal.md` — 201 lines, 3 P0 / 2 P1 / 5 P2
- `docs/roadmap/audit-rmlx-alloc.md` — 166 lines, 1 P0 / 3 P1 / 5 P2
- `docs/roadmap/audit-rmlx-core.md` — 277 lines, 15 P0 / 18 P1 / 11 P2
- `docs/roadmap/audit-rmlx-rdma.md` — 240 lines, 2 P0 / 4 P1 / 6 P2
- `docs/roadmap/audit-rmlx-distributed.md` — 267 lines, 5 P0 / 7 P1 / 8 P2 / 5 P3 + Codex appendix
- `docs/roadmap/audit-rmlx-nn.md` — 282 lines, 9 P0 / 20+16+4 P1 / 13 P2

## Appendix B: Unique RMLX Strengths

Despite the issues, RMLX has genuine innovations that justify continued development:

1. **Zero-copy GPU→RDMA pipeline** — SlabRing → SharedBuffer → RDMA without CPU memcpy. Superior to CUDA's GDR staging.
2. **FP8 E4M3 wire quantization** — Fused Metal shader dequant+scatter. Ahead of most CUDA MoE implementations.
3. **Credit-based UC flow control** — CreditManager prevents receiver overrun without RC overhead.
4. **Dual-port TB5 striping** — StripeEngine leverages both TB5 ports for 2× bandwidth.
5. **GPU doorbell ring** — GPU-visible descriptor ring with CPU proxy for GPU-initiated RDMA.
6. **SparseGuard overflow monitoring** — EMA-based dense fallback for MoE load imbalance.
7. **Native Rust LoRA** — Built-in LoRA with GPU forward (MLX has this only in Python).
8. **MLA/SlidingWindow as reusable layers** — Once completed, these are architectural innovations over MLX.
9. **Precision guard system** — Automatic mixed-precision management at op level.
10. **MoePipeline with EP and overlap** — SBO/TBO strategies for compute↔comm overlap.

---

*Generated by Claude Opus 4.6 + Codex — 2026-03-06*
*This document integrates 7 individual crate audits + cross-crate integration analysis into a unified production readiness assessment.*
