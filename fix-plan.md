# RMLX Production Readiness Fix Plan (v2)

**Date:** 2026-03-06 (revised)
**Source:** `~/rmlx/production-readiness-audit.md` (183 issues across 7 crates + cross-crate)
**Goal:** Bring RMLX from 3/10 to production-ready (target 8/10)

---

## Design Principles of This Plan

1. **Vertical slice first** — Phase 1 delivers a working single-node Llama inference as the first real milestone, not just "all bugs fixed"
2. **Inference before training** — Training-only issues (SDPA backward races, VJP gaps) are explicitly deferred until training is in scope
3. **Each PR is independently reviewable** — No PR touches more than 2 crates or 5 unrelated files
4. **Validation checkpoints** — Each phase ends with a concrete integration test that proves the milestone
5. **P2/P3 items integrated alongside main phases** — Not dumped in a backlog; each is assigned to the phase where it naturally fits

---

## How to Read This Plan

- **Complexity:** S (< 1 day), M (1-3 days), L (3-5 days), XL (1-2 weeks), 2XL (2-4 weeks)
- `→ depends on PR X.Y` = must complete that PR first
- `∥` = can be parallelized with adjacent PRs
- **[T]** = training-only, deferred unless training scope confirmed
- **Bundled P2/P3** items are listed under each PR where they naturally integrate

---

## Prerequisite: Scope Decision

Before starting, make these architectural decisions (from audit Section 10):

| # | Decision | Recommendation | Impact on Plan |
|---|----------|---------------|----------------|
| 1 | Training in scope? | **No** for v1 | Defers SDPA backward, VJP, dropout, loss functions — saves ~6 weeks |
| 2 | Serving architecture? | **Built-in** (rmlx-nn crate) | Paged KV + scheduler live in rmlx-nn |
| 3 | Python bindings priority? | **Phase 6** | Rust-native first |
| 4 | UC vs RC RDMA? | **UC + CRC32** for v1 | Defers RC mode |

This plan assumes: **Inference-only v1, Rust-native, UC RDMA with CRC32.**

---

## Phase 0: Stop the Bleeding (Weeks 1-2)

> **✅ Phase 0 COMPLETE** (2026-03-06, PR #38)
> - All 19 PRs implemented and merged in a single commit
> - 764+ tests pass, 0 failures
> - Codex review completed — 1 true finding fixed (ZeroCopyPendingOp Drop), 3 false positives identified
> - 33 files changed, +2529/-446 lines

**Goal:** No crashes, no UB, no memory corruption, no security vulnerabilities.
**Validation checkpoint:** `cargo test --workspace` passes; no panics/UB under basic single-node inference attempt.

### ✅ PR 0.1: rmlx-core — Replace unreachable!/unwrap panics with Result
- **Issues:** C-P0-1, C-P0-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/copy.rs:567,578,588` — Replace `unreachable!()` in `vectorized_kernel_name()`, `scalar_kernel_name()`, `strided_kernel_name()` with `Err(CoreError::UnsupportedDtype(dtype))`
  - `crates/rmlx-core/src/softmax.rs:553` — Replace `.unwrap()` with `ok_or(CoreError::EmptyShape)?`
- **Tests:** copy with exotic dtype returns error; softmax on zero-rank returns error
- **Bundled:** None

### ✅ PR 0.2: rmlx-core — Promote debug_assert for quantized block alignment
- **Issues:** C-P0-3
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/dtype.rs:87` — Change `debug_assert!` to runtime check returning `Err(CoreError::QuantBlockMisalign { numel, block_size })`
- **Tests:** Non-aligned numel returns error in release build
- **Bundled P2:** P2-11 (docs/behavior mismatch for numel_to_bytes) — fix docs alongside the assert change

### ✅ PR 0.3: rmlx-core — Guard empty/zero-dim tensor dispatch
- **Issues:** C-P0-8, C-P0-9
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/binary.rs:452,654,664` — Early return for `output.numel() == 0`
  - `crates/rmlx-core/src/softmax.rs:553,554,556,598` — Early return for zero-element shapes
- **Tests:** Binary op and softmax with `[0, 4]`, `[3, 0]`, `[]` return empty output without dispatch

### ✅ PR 0.4: rmlx-core — Fix conv2d_tiled threadgroup overflow
- **Issues:** C-P0-6
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/conv_tiled.rs:158,161,173` — Replace `TILE_CI * 9` with `TILE_CI * kH * kW`, pass kH/kW as kernel constants
- **Tests:** conv2d with 5x5 and 7x7 kernels succeeds
- **Bundled P2:** P2-9 (dead im2col_f32 kernel) — remove the dead code in the same file

### ✅ PR 0.5: rmlx-core — Fix FP8 cross-threadgroup race on scales
- **Issues:** C-P0-7
- **Complexity:** M
- **Files:**
  - `crates/rmlx-core/src/fp8.rs:290,302,706,711` — Ensure each row is dispatched within a single threadgroup by adjusting grid sizing. For rows exceeding max threadgroup size, use `atomic_max` on `scales[row]`.
- **Tests:** FP8 quantization with row_size > max_threadgroup_size produces correct scales

### ✅ PR 0.6: rmlx-core — Add RoPE freq validation
- **Issues:** C-P0-10
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/rope.rs:458,466,725,769` — Add frequency tensor shape/range validation matching `rope_ext` version
- **Tests:** Malformed freq tensor returns error

### ✅ PR 0.7: rmlx-core — Add GEMV dtype validation
- **Issues:** C-P0-11
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/gemv.rs:591,617,691,717,778` — Validate `vec.dtype() == mat.dtype()` and `bias.dtype() == mat.dtype()`
- **Tests:** Mismatched dtypes return error

### ✅ PR 0.8: rmlx-core — Add SDPA forward dtype/shape validation
- **Issues:** C-P0-12
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/sdpa.rs:1089,1117,1164,1181` — Validate q/k/v/mask dtype consistency; reject non-contiguous masks; validate shapes
- **Tests:** Mismatched q/k dtype returns error; non-contiguous mask returns error

### ✅ PR 0.9: rmlx-core — Add gather_qmm shape/buffer-size validation
- **Issues:** C-P0-14
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/quantized.rs:1357,1372,1286,1287,1319` — Add shape and buffer-size validation before kernel dispatch
- **Tests:** Malformed input shapes return error

### ✅ PR 0.10: rmlx-core — Fix GGUF parser alignment=0
- **Issues:** C-P0-15
- **Complexity:** S
- **Files:**
  - `crates/rmlx-core/src/gguf.rs:364,401` — Guard `div_ceil` with `if alignment == 0 { return Err(GgufError::InvalidAlignment) }`
- **Tests:** Malformed GGUF with alignment=0 returns error
- **Bundled P2:** P2-6 (GGUF offset validation) — add offset range checking alongside

### ✅ PR 0.11: rmlx-alloc — Fix zero-size allocation cache poisoning
- **Issues:** A-P0-1
- **Complexity:** S
- **Files:**
  - `crates/rmlx-alloc/src/allocator.rs:114` — Reject zero-size: `if size == 0 { return Err(AllocError::ZeroSize) }`
  - `crates/rmlx-alloc/src/cache.rs:42,73` — Validate `buffer.length() >= requested_size` in `acquire()`
- **Tests:** alloc(0) returns ZeroSize error; cache never returns undersized buffer

### ✅ PR 0.12: rmlx-metal — Fix CaptureScope FFI memory corruption
- **Issues:** M-P0-1
- **Complexity:** M
- **Files:**
  - `crates/rmlx-metal/src/capture.rs:136` — Use `device.as_ptr() as *mut Object` for correct ObjC pointer
  - `crates/rmlx-metal/src/capture.rs:150-155` — Use `CString::new(path)?.as_ptr()` for null-terminated string
  - Wrap all ObjC calls in `objc::rc::autoreleasepool(|| { ... })`
- **Tests:** CaptureScope create/start/stop doesn't crash (Metal device required)
- **Bundled P2:** MED-3 (panic paths in capture.rs) — fix `unwrap()`s while editing the file

### ✅ PR 0.13: rmlx-metal — Fix SwiGLU dispatch to 2D grid
- **Issues:** M-P0-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-metal/src/icb_sparse.rs:192-201` — 2D grid `MTLSize::new(max_cap, inter, 1)` with threadgroup `MTLSize::new(min(16, max_cap), min(16, inter), 1)`
- **Tests:** Verify `replay_with_count` preserves intermediate_dim in height dimension

### ✅ PR 0.14: rmlx-metal — Guard ExecGraph submit_batch against empty CB
- **Issues:** M-P0-3
- **Complexity:** S
- **Files:**
  - `crates/rmlx-metal/src/exec_graph.rs:128-134` — If `!self.batcher.has_pending()`, return previous token unchanged (don't increment counter)
- **Tests:** submit_batch with no pending work returns valid token; wait_for on it completes immediately
- **Bundled P2:** MED-5 (StreamNotFound error variant) — add to metal error type while editing exec_graph

### ✅ PR 0.15: rmlx-metal — Document encoder lifecycle contract
- **Issues:** M-CRIT-3 (was missing from v1 plan!)
- **Complexity:** S
- **Files:**
  - `crates/rmlx-metal/src/batcher.rs:95-99,124-126` — Add doc comments: "Caller MUST call `end_encoding()` on the returned encoder before calling `end_encoder()` or `encoder()` again." Add `debug_assert!(self.encoder_active)` in `end_encoder()`.
- **Bundled P2:** MED-1 (wire StreamState.label) — trivial, do alongside batcher docs

### ✅ PR 0.16: rmlx-cli — Fix SSH option injection
- **Issues:** L-P0-1
- **Complexity:** S
- **Files:**
  - `crates/rmlx-cli/src/ssh.rs:48-64,122-132` — Reject host/user strings beginning with `-`; validate `[a-zA-Z0-9._@:-]` charset; insert `--` before destination
  - `crates/rmlx-cli/src/hostfile.rs:22` — Validate host field on parse
- **Tests:** Host `-oProxyCommand=...` rejected; valid hostnames pass

### ✅ PR 0.17: rmlx-distributed — Fix unsound unsafe + panic in buffer readers
- **Issues:** D-P0-1
- **Complexity:** S
- **Files:**
  - `crates/rmlx-distributed/src/moe_exchange.rs:46-74` — Replace `assert!` with `Result`; add GpuEvent fence check; return `Result<&[u8], DistributedError>`
- **Tests:** Out-of-bounds offset returns error

### ✅ PR 0.18: rmlx-distributed — Fix PendingOp lifetime for zero-copy RDMA
- **Issues:** D-P0-5
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/transport.rs:549,765,802` — `PendingOp` holds `Arc<SharedBuffer>` to prevent buffer drop during DMA. Drop impl waits for completion.
- **Tests:** SharedBuffer drop while PendingOp active doesn't cause UB

### ✅ PR 0.19: rmlx-metal — Add !Send to ScopedPool
- **Issues:** M-MED-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-metal/src/autorelease.rs:28` — Add `_marker: PhantomData<*mut ()>` field (automatically prevents Send)
- **Tests:** Compile-time test that ScopedPool is !Send

**Phase 0 Total: 19 PRs (mostly S, 3 M) — ~2 weeks with 1 engineer, ~1 week with 2 parallel**

**Phase 0 Validation:** `cargo test --workspace` passes. Run basic model load + single forward pass (will fail at missing layers, but no panics/crashes).

---

## Phase 1: Vertical Slice — Single-Node Llama Inference (Weeks 3-6)

**Goal:** Load a Llama-7B GGUF model, run inference, generate text on single node.
**Why this order:** This is the shortest path to a demonstrable working system. Every PR here is on the critical path to text generation.

**Validation checkpoint:** `rmlx-cli` loads Llama-7B Q4_0 GGUF, generates 100 tokens of coherent text.

### PR 1.1: rmlx-nn — Implement RMSNorm layer ∥
- **Issues:** Missing critical layer (every LLM needs this)
- **Complexity:** S
- **Files:**
  - Create `crates/rmlx-nn/src/rms_norm.rs` — `RMSNorm { weight: Array, eps: f32 }` wrapping `rmlx_core::ops::rms_norm`
  - Add to `lib.rs` exports
- **Tests:** RMSNorm output matches reference for known inputs
- **Why Phase 1:** Cannot run any Llama/Mistral model without this layer.

### PR 1.2: rmlx-nn — Implement standalone RoPE layer ∥
- **Issues:** Missing critical layer (MLA and standard attention need it)
- **Complexity:** S
- **Files:**
  - Create `crates/rmlx-nn/src/rope.rs` — `RotaryPositionEmbedding` wrapping `rmlx_core::ops::rope_ext`. Support base frequency, scaling (NTK-aware, YaRN).
- **Tests:** RoPE output for known position/frequency inputs matches reference

### PR 1.3: rmlx-nn — Implement basic Attention forward (non-MLA path)
- **Issues:** N-P0-1 (MLA stub), N-P0-2 (SlidingWindow stub) — but we start with standard GQA first
- **Complexity:** L
- **→ depends on:** PR 0.8 (SDPA validation), PR 1.2 (RoPE)
- **Strategy:** Rather than implementing MLA first (which is DeepSeek-specific), implement standard GQA attention that works for Llama/Mistral/Qwen. MLA comes in Phase 5.
- **Files:**
  - `crates/rmlx-nn/src/attention.rs` — Implement `Attention::forward()` for standard GQA: (1) Q/K/V projection, (2) apply RoPE, (3) update KV cache, (4) call SDPA, (5) output projection
- **Tests:** Attention forward with known Q/K/V vs reference; verify KV cache updates
- **Note:** MLA and SlidingWindow stubs remain — they're model-specific and can wait.

### PR 1.4: rmlx-nn — Fix RotatingKvCache wrap-around ∥
- **Issues:** N-P0-3
- **Complexity:** M
- **Files:**
  - `crates/rmlx-nn/src/attention.rs:490,589,599,683` — Track valid region start/end separate from write pointer. Linearize copies only valid regions.
- **Tests:** Property-based: append N tokens to cache of capacity M for all N in [0..3M], verify linearize matches sequential append

### PR 1.5: rmlx-nn — Fix QuantizedLinear dtype handling ∥
- **Issues:** N-P0-5
- **Complexity:** S
- **Files:**
  - `crates/rmlx-nn/src/quantized_linear.rs:191,206` — Cast input to f32 before extraction; replace `assert!` with `Result`
- **Tests:** f16 input doesn't panic
- **Bundled P1:** P1-2 (cache weight buffers) — stop re-uploading weights every forward() while fixing the dtype

### PR 1.6: rmlx-nn — Wire TransformerBlock FFN type
- **Issues:** N-P1-6
- **Complexity:** M
- **→ depends on:** PR 1.1 (RMSNorm)
- **Files:**
  - `crates/rmlx-nn/src/transformer.rs:254-269` — Read `ff_type` from config: Dense = Linear+activation+Linear; Gated = gate_proj*up_proj+down_proj; MoE = MoeLayer
- **Tests:** TransformerBlock with Dense and Gated FFN produces output of correct shape

### PR 1.7: rmlx-nn — Implement Sampler module
- **Issues:** N-P0-6
- **Complexity:** L
- **Files:**
  - Create `crates/rmlx-nn/src/sampler.rs`:
    - `temperature(logits, temp)` — scale by 1/temp
    - `top_k(logits, k)` — mask all but top-k
    - `top_p(logits, p)` — nucleus sampling
    - `sample(logits) -> token_id` — multinomial from filtered distribution
    - `repetition_penalty(logits, past_tokens, penalty)`
- **Tests:** temperature=0 returns argmax; top_k=1 returns argmax; distribution is valid
- **Note:** Sampler needs `sort`/`argsort` from core. If not available, implement CPU-side sorting as fallback within sampler, with TODO for GPU path.

### PR 1.8: rmlx-nn — GGUF loader integration test
- **Issues:** N-P1-5 (GGUF silently passes unrecognized patterns)
- **Complexity:** M
- **→ depends on:** PR 1.1, 1.2, 1.3, 1.5, 1.6
- **Files:**
  - `crates/rmlx-nn/src/gguf_loader.rs:331-348` — Make unrecognized name patterns return `None` (not silently pass through)
  - Add integration test: load Llama-7B-Q4_0.gguf → construct model → run single forward pass → verify output shape
- **Tests:** Full model load + forward pass produces logits of shape [1, vocab_size]

### PR 1.9: rmlx-nn + rmlx-core — End-to-end text generation test
- **Issues:** Validation milestone
- **Complexity:** M
- **→ depends on:** PR 1.7 (Sampler), PR 1.8 (GGUF loader)
- **Files:**
  - Create `crates/rmlx-nn/tests/e2e_generate.rs` — Load model, generate 100 tokens, verify non-garbage output (perplexity check or simple coherence)
  - Fix any issues discovered during integration
- **Tests:** This IS the test

**Phase 1 Total: 9 PRs — ~4 weeks with 1 engineer (1.1-1.5 parallel, then 1.6-1.9 serial)**

**Phase 1 Validation:** Load Llama-7B-Q4_0, generate "The capital of France is" → get coherent continuation.

---

## Phase 2: Distributed Correctness (Weeks 7-10)

**Goal:** Multi-node inference produces correct results. MoE token exchange works.
**Why before serving?** Distributed correctness is a prerequisite for MoE models (Mixtral, DeepSeek). Serving infrastructure (paged KV, continuous batching) can be developed in parallel but is less critical than getting multi-node correct.

**Validation checkpoint:** 2-rank loopback MoE inference produces same results as single-rank dense equivalent.

### PR 2.1: rmlx-distributed — Fix EP dispatch/combine correctness
- **Issues:** D-P0-2, D-P0-4 (async combine global/local index mismatch)
- **Complexity:** L
- **Files:**
  - `crates/rmlx-distributed/src/moe_exchange.rs:3158-3164` — Fix combine to use local expert index against local buffer
  - Fix routing indices and weight accumulation per `docs/roadmap/audit-ep-dispatch-combine.md`
- **Tests:** dispatch→combine round-trip with known expert assignments matches expected output

### PR 2.2: rmlx-distributed — Fix blocking_exchange_v3 deadlock
- **Issues:** D-P0-3
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/v3_protocol.rs` — Interleaved non-blocking exchange: alternate send/recv chunks instead of all-send-then-all-recv
- **Tests:** Asymmetric token counts (rank 0: 100, rank 1: 0) completes without deadlock
- **Bundled P2:** SlabRing TOCTOU fix (Codex P2-2, `slab_ring.rs:200,220,229,261`) — replace load/check/fetch_add with CAS. This is a correctness bug, not just P2.

### PR 2.3: rmlx-nn + rmlx-distributed — Wire MoeDispatchExchange into MoE forward
- **Issues:** X-P0-1, N-P0-4
- **Complexity:** XL
- **→ depends on:** PR 0.17 (buffer reader fix), PR 2.1 (EP correctness)
- **Strategy:** Split into 3 sub-PRs:
  - **2.3a (M):** Wire dispatch before expert compute in Batched strategy (`moe.rs:406`)
  - **2.3b (M):** Wire combine after expert compute, handle routing weight accumulation
  - **2.3c (M):** Wire into PerExpert and Pipeline strategies (`moe.rs:513,586`)
- **Tests:** 2-rank loopback: tokens routed to non-local experts arrive and return

### PR 2.4: rmlx-distributed + rmlx-core — Register MoE Metal kernels
- **Issues:** X-P0-2
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/moe_exchange.rs` — Add `pub fn init_kernels(device: &Device)` using `LibraryCache::compile_source()` for inline Metal shaders
- **Tests:** MoE kernel lookup succeeds after init_kernels()

### PR 2.5: rmlx-distributed + rmlx-rdma — Wire MrPool into EP dispatch
- **Issues:** X-P0-3
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/moe_exchange.rs` — Replace ad-hoc `ibv_reg_mr` with `ep_runtime.shared_buffer_pool.acquire(tier)`
  - `crates/rmlx-distributed/src/ep_runtime.rs` — Add `shared_buffer_pool: MrPool` field
- **Tests:** EP dispatch latency < 10us (vs ~100us without MrPool)

### PR 2.6: rmlx-nn + rmlx-distributed — Implement TP forward
- **Issues:** X-P0-4
- **Complexity:** L
- **Files:**
  - `crates/rmlx-nn/src/parallel.rs` — `ColumnParallelLinear::forward()`: local matmul → `Group::allgather()`. `RowParallelLinear::forward()`: local matmul → `Group::allreduce()`.
- **Note:** Initially f32-only. f16/bf16 collectives come in Phase 4 (PR 4.8).
- **Tests:** 2-rank loopback TP produces same output as single-rank full Linear (f32)
- **Bundled P1:** P1-19 (parallel world_size validation) — add validation while implementing

### PR 2.7: rmlx-rdma — Add application-level CRC32 for UC transport
- **Issues:** R-P0-2
- **Complexity:** M
- **Files:**
  - `crates/rmlx-rdma/src/connection.rs` — Append 4-byte CRC32 to sends; verify on recv; return `RdmaError::DataCorruption` on mismatch
  - Add `crc32fast` dependency
- **Tests:** Corrupted data detected; clean data passes

### PR 2.8: rmlx-cli + rmlx-distributed — Align backend enum + coordinator IP
- **Issues:** X-P1-1, X-P1-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-cli/src/config.rs:13` — Remove or properly map `ring` backend
  - `crates/rmlx-distributed/src/init.rs:37-44` — Error on unknown backend instead of silent Auto
  - `crates/rmlx-cli/src/launch.rs:164` — Use `entries[0].ips[0]` for coordinator, not SSH target
- **Tests:** Unknown backend errors; coordinator IP matches probed IP
- **Bundled P1:** X-P1-3 (export RMLX_IBV_DEVICES from hostfile) — add while fixing launch.rs

**Phase 2 Total: 8 PRs (3 sub-PRs) — ~4 weeks with 1 engineer**

**Phase 2 Validation:** `rmlx launch --world-size 2 --backend loopback` with Mixtral model produces correct MoE output.

---

## Phase 3: Serving Infrastructure (Weeks 11-16)

**Goal:** Serve multiple concurrent requests with text generation.
**Why now:** Single-node inference works (Phase 1), distributed works (Phase 2). Now make it servable.

**Validation checkpoint:** Serve 4 concurrent requests, measure tokens/sec, no OOM.

### PR 3.1: rmlx-core + rmlx-metal — Implement FlashAttention-2 Metal kernel
- **Issues:** N-P0-9
- **Complexity:** 2XL (this is 2-4 weeks of specialized GPU work, not 1-2 weeks)
- **Files:**
  - Create `crates/rmlx-core/kernels/flash_attention.metal` — Tiled FlashAttention-2: online softmax, register-level Q tiling, shared-memory K/V tiling. Reference: philipturner/metal-flash-attention.
  - `crates/rmlx-core/src/sdpa.rs` — Add dispatch path for seq_len > threshold
  - `crates/rmlx-core/build.rs` — Compile flash_attention.metal
- **Strategy:** Start with head_dim=128 (Llama) f16 path. Add f32 and head_dim=64/96 variants iteratively.
- **Tests:** Output matches naive SDPA within f16 tolerance for seq_len=[128, 512, 2048, 8192]
- **Risk:** This is the single highest-risk PR in the plan. Consider vendoring `metal-flash-attention` first, then optimize.

### PR 3.2: rmlx-nn — Implement paged KV cache + block manager
- **Issues:** N-P0-8
- **Complexity:** XL
- **Note:** Does NOT depend on FlashAttention. Works with existing SDPA initially, upgraded to FlashAttention when PR 3.1 lands.
- **Files:**
  - Create `crates/rmlx-nn/src/paged_kv_cache.rs`:
    - `BlockManager` — free list, block tables, copy-on-write for prefix sharing
    - `PagedKvCache` — non-contiguous block storage, block-table indirection for reads
  - Create `crates/rmlx-core/kernels/paged_attention.metal` — Reads K/V via block table indirection
- **Tests:** Block alloc/dealloc; paged attention matches monolithic for same inputs

### PR 3.3: rmlx-nn — Implement continuous batching scheduler
- **Issues:** N-P0-7
- **Complexity:** XL
- **→ depends on:** PR 3.2 (paged KV)
- **Files:**
  - Create `crates/rmlx-nn/src/scheduler.rs`:
    - `Request` { id, tokens, state: Prefill|Decode|Finished, block_table }
    - `BatchScheduler` — FCFS with memory budget awareness, preemption via swap
    - `IterationRunner` — one decode step per iteration, add/evict requests
- **Tests:** 4 concurrent requests complete; scheduler respects memory limit
- **Bundled:** PR 3.1 integration — once FlashAttention lands, upgrade scheduler to use it

### PR 3.4: rmlx-core — Route all ops through commit_with_mode()
- **Issues:** C-P0-4
- **Complexity:** M
- **Note:** This was incorrectly in Phase 2 of v1. It's a metrics fix, belongs in serving phase where metrics matter.
- **Files:**
  - Audit all ops — replace direct `cb.commit(); cb.wait_until_completed()` with centralized commit
- **Tests:** TOTAL_OP_CBS matches actual dispatch count across 10 different ops

### PR 3.5: rmlx-rdma — Generalize collectives to f16/bf16
- **Issues:** R-P0-1
- **Complexity:** M
- **Files:**
  - `crates/rmlx-rdma/src/collectives.rs` — Generic `trait ReduceElement`; implement for f32, f16, bf16
- **Tests:** ring_allreduce with f16 data; verify precision vs f32 reference

### PR 3.6: rmlx-distributed — Fix ring allreduce chunk rounding + f16 edge cases
- **Issues:** D-P1-1, D-P1-6
- **Complexity:** M
- **→ depends on:** PR 3.5 (f16/bf16 collectives)
- **Files:**
  - `crates/rmlx-distributed/src/group.rs:695,711` — Pad chunk_size to element-size multiple
  - `crates/rmlx-distributed/src/group.rs:1063-1114` — Use `half` crate IEEE conversions; handle NaN/subnormal
- **Tests:** Allreduce with non-divisible tensor size; f16 round-trip preserves NaN payloads

### PR 3.7: rmlx-distributed — Fix MoePolicy thread safety
- **Issues:** D-P1-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-distributed/src/moe_policy.rs` — Wrap all state in `RwLock`; Acquire/Release ordering for zone transitions
- **Tests:** Concurrent zone transitions from N threads; no torn reads

### PR 3.8: rmlx-cli — Add signal forwarding
- **Issues:** L-P1-4
- **Complexity:** S
- **Files:**
  - `crates/rmlx-cli/src/launch.rs:169-276` — `ctrlc` crate handler kills all children; SSH remote kill
- **Tests:** SIGINT kills spawned processes

**Phase 3 Total: 8 PRs — ~6 weeks (PR 3.1 is the bottleneck; 3.2-3.8 can proceed in parallel)**

**Phase 3 Validation:** HTTP server (or simple stdin loop) serves 4 concurrent generation requests. FlashAttention achieves >2x speedup over naive SDPA for seq_len=2048.

---

## Phase 4: Performance and Allocator (Weeks 17-22)

**Goal:** Competitive single-node and multi-node performance. Stable memory management.

**Validation checkpoint:** Llama-7B inference at >40 tok/s (decode) on M4 Max. No memory growth over 1000 requests.

### PR 4.1: rmlx-alloc — Fix concurrency limit bypass (atomic CAS)
- **Issues:** A-P1-1
- **Complexity:** M
- **Files:**
  - `crates/rmlx-alloc/src/allocator.rs:121-130` — Atomic CAS reservation loop before allocation
- **Tests:** N-thread stress test doesn't exceed memory limit

### PR 4.2: rmlx-alloc — Fix stats underflow + ownership validation
- **Issues:** A-P1-2
- **Complexity:** S
- **Files:**
  - `crates/rmlx-alloc/src/stats.rs:43` — `fetch_update` with saturating_sub
  - `crates/rmlx-alloc/src/allocator.rs:190` — Track allocated pointers; reject unknown in free()
- **Tests:** Double-free doesn't corrupt stats; freeing unowned buffer returns error
- **Bundled P2:** P2-A1 (SmallBufferPool double-free guard) — same pattern, fix together

### PR 4.3: rmlx-alloc — Wire orphan modules into MetalAllocator
- **Issues:** SmallBufferPool, LeakDetector, ResidencyManager orphaned
- **Complexity:** L
- **Files:**
  - `crates/rmlx-alloc/src/allocator.rs` — Route <256B through SmallBufferPool; call LeakDetector; use ResidencyManager when metal3 feature on
- **Tests:** Small alloc uses pool; leak detector catches leak in test
- **Bundled P1:** A-P1-3 (Metal 3 runtime availability guard in residency.rs) — fix while wiring

### PR 4.4: rmlx-metal — Add chip-class tuning + unretained CB refs
- **Issues:** Metal F1, F2
- **Complexity:** M
- **Files:**
  - `crates/rmlx-metal/src/device.rs` — `ChipTuning` per M1/M3/M4 generation
  - `crates/rmlx-metal/src/command.rs` — Accept tuning; use `new_command_buffer_with_unretained_references()` (safe due to RAII)
- **Tests:** Correct tuning for each architecture; CB creation succeeds with unretained refs

### PR 4.5: rmlx-metal — Add binary archive / disk pipeline cache
- **Issues:** Metal F7
- **Complexity:** L
- **Files:**
  - `crates/rmlx-metal/src/pipeline.rs` — `DiskPipelineCache` using BinaryArchive API; serialize to `~/.cache/rmlx/pipelines/`
- **Tests:** Cold miss compiles+stores; warm hit loads from disk; startup time reduction

### PR 4.6: rmlx-metal + rmlx-alloc — Add HazardTrackingModeUntracked
- **Issues:** MLX parity
- **Complexity:** S
- **Files:**
  - `crates/rmlx-metal/src/buffer.rs` — Add HazardTrackingModeUntracked to buffer creation
- **Tests:** Buffers created with untracked mode; benchmark speedup

### PR 4.7: rmlx-core — Fuse RMSNorm + residual add kernel
- **Issues:** Serving performance
- **Complexity:** M
- **Files:**
  - Create `crates/rmlx-core/kernels/fused_rms_norm_residual.metal`
  - `crates/rmlx-core/src/rms_norm.rs` — Add `rms_norm_residual_add()` path
- **Tests:** Fused matches separate ops; benchmark speedup

### PR 4.8: rmlx-nn + rmlx-core — Use gather_mm in MoE batched strategy
- **Issues:** X-P1-5
- **Complexity:** M
- **→ depends on:** PR 0.9 (gather_qmm validation)
- **Files:**
  - `crates/rmlx-nn/src/moe.rs` — Replace per-expert loop with batched `gather_mm` dispatch
- **Tests:** gather_mm MoE output matches per-expert loop

### PR 4.9: rmlx-distributed — Add SlabRing backpressure
- **Issues:** D-P1-7
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/slab_ring.rs` — Condvar blocking when ring full; `ring_full_count` metric
- **Tests:** Producer blocks when consumer slow; resumes on consumer progress

### PR 4.10: rmlx-distributed + rmlx-rdma — Wire ProgressEngine into EP dispatch
- **Issues:** X-P1-4
- **Complexity:** M
- **→ depends on:** PR 0.18 (PendingOp lifetime fix)
- **Files:**
  - `crates/rmlx-distributed/src/moe_exchange.rs` — Use ProgressEngine::register_op() + PendingOp::wait()
- **Tests:** EP dispatch completes via ProgressEngine; async progress tracked
- **Bundled P1:** P2-R6 (ProgressEngine health escalation) — add consecutive-error threshold

### PR 4.11: rmlx-metal + rmlx-nn — Implement EP-7 ICB Sparse Expert Dispatch
- **Issues:** X-P1-7
- **Complexity:** XL
- **→ depends on:** PR 0.13 (SwiGLU 2D fix), PR 2.3 (MoE exchange wiring)
- **Why Phase 4 (moved from Phase 3):** ICB is a performance optimization. Phase 3 was about correctness.
- **Sub-PRs:**
  - **4.11a (L):** `ExpertGroup::grouped_forward_icb()` — encode only active experts
  - **4.11b (M):** `IcbReplay` cache — per-sparsity-pattern hash, replay without re-encoding
  - **4.11c (M):** Wire `forward_sparse_icb()` in moe.rs — remove `let _ =` stubs, call grouped_forward_icb()
- **Tests:** Empty experts (0 tokens) not dispatched; ICB matches non-ICB output; benchmark
- **Bundled:** Rename ICB modules to `dispatch_replay`/`sparse_dispatch` (HIGH-2)

### PR 4.12: rmlx-alloc — Block splitting and coalescing (BFC-style)
- **Issues:** CUDA frontier
- **Complexity:** 2XL (this is an allocator rewrite, not a single PR — split if needed)
- **Strategy:** Implement alongside existing allocator as an alternative `BfcAllocator`. Switch via feature flag. Don't replace MetalAllocator immediately.
- **Files:**
  - Create `crates/rmlx-alloc/src/bfc.rs` — Block splitting (oversized → requested + remainder), coalescing (merge adjacent free blocks). Red-black tree for O(log n) best-fit.
- **Tests:** Fragmentation stress test: random alloc/free pattern, verify coalescing prevents growth

**Phase 4 Total: 12 PRs (3 sub-PRs) — ~6 weeks with 2 engineers**

---

## Phase 5: Feature Breadth (Weeks 23-30)

**Goal:** Support major model architectures (DeepSeek, Mistral, Gemma, Phi).

### PR 5.1: rmlx-nn — Implement MLA forward() (DeepSeek)
- **Issues:** N-P0-1
- **Complexity:** L
- **→ depends on:** PR 1.2 (RoPE layer)
- **Files:**
  - `crates/rmlx-nn/src/mla.rs:368-401` — Full MLA: compressed latent projection, RoPE on rotary dims, SDPA with latent KV, output projection
  - Fix MlaKvCache.advance() to copy data
- **Tests:** MLA forward matches DeepSeek reference implementation

### PR 5.2: rmlx-nn — Implement SlidingWindowAttention forward() (Mistral)
- **Issues:** N-P0-2
- **Complexity:** L
- **Files:**
  - `crates/rmlx-nn/src/sliding_window.rs:145-177` — Apply sliding window mask to SDPA; cache eviction on window overflow
- **Tests:** Window=128, seq=256: attention only attends to last 128 positions
- **Bundled P1:** P1-18 (kernel-level mask instead of materialized tensor) — implement mask as SDPA parameter, not full tensor

### PR 5.3: rmlx-core — Implement missing critical ops (split into sub-PRs)
- **Issues:** Core P1 missing ops
- **Sub-PRs:**
  - **5.3a (M):** `slice`, `slice_update`, `dynamic_slice`
  - **5.3b (L):** `sort`, `argsort` — Bitonic sort on Metal
  - **5.3c (M):** `cumsum`, `cumprod` — Parallel prefix scan
  - **5.3d (S):** `argmin`, `argmax` — Reduction with index tracking
  - **5.3e (L):** `uniform`, `normal`, `bernoulli` — Metal Philox PRNG
- **Tests:** Each op vs CPU reference

### PR 5.4: rmlx-nn — Add activation functions
- **Issues:** Missing layers
- **Complexity:** M
- **Files:**
  - `crates/rmlx-nn/src/activations.rs` — Add: ReLU, LeakyReLU, ELU, CELU, PReLU, Mish, Hardswish, HardSigmoid, Softplus, Softsign, LogSigmoid, GELU variants, GLU
- **Tests:** Each activation output matches reference

### PR 5.5: rmlx-nn — Implement AWQ/GPTQ/k-quant ingestion
- **Issues:** N-P1-16, C-P1-17
- **Complexity:** L
- **Files:**
  - `crates/rmlx-nn/src/quantized_linear.rs` — AWQ (group quant + scaling) and GPTQ support
  - `crates/rmlx-nn/src/gguf_loader.rs` — Map Q2_K through Q6_K quant types
- **Tests:** Load and inference with GPTQ-quantized model

### PR 5.6: rmlx-nn — Add model configs + architecture forwards
- **Issues:** N-P2-5, N-P1-17
- **Complexity:** XL
- **Files:**
  - `crates/rmlx-nn/src/models/` — Gemma, Phi-3, Mistral, Command-R, Qwen2 configs + forward() + weight loading
- **Tests:** Each architecture loads and produces correct output shapes

### PR 5.7: rmlx-nn — Implement prefix caching
- **Issues:** Serving feature
- **Complexity:** XL
- **→ depends on:** PR 3.2 (paged KV cache)
- **Files:**
  - Create `crates/rmlx-nn/src/prefix_cache.rs` — Radix tree for prompt prefix sharing; reuse KV blocks
- **Tests:** Shared prefix achieves cache hit; TTFT improvement

### PR 5.8: rmlx-nn — Implement chunked prefill
- **Issues:** Serving feature
- **Complexity:** L
- **→ depends on:** PR 3.3 (continuous batching)
- **Files:**
  - `crates/rmlx-nn/src/scheduler.rs` — Split long prefills into 512-token chunks; interleave with decode
- **Tests:** Decode latency maintained during concurrent long prefill

### PR 5.9: rmlx-cli — Add TB5 topology discovery + backend-aware launch
- **Issues:** L-P1-7, L-P1-6
- **Complexity:** XL
- **Files:**
  - `crates/rmlx-cli/src/config.rs` — Parse `system_profiler SPThunderboltDataType -json`; DFS ring extraction
  - `crates/rmlx-cli/src/launch.rs` — Distinct launch paths: RDMA, loopback, ring
- **Tests:** Topology parsing with mock data; each launch path exports correct env

### PR 5.10: rmlx-distributed — Implement tree allreduce + topology-aware routing
- **Issues:** D-P1-5, D-P2-2
- **Complexity:** L
- **Files:**
  - `crates/rmlx-distributed/src/group.rs` — `tree_allreduce()` for <1MB tensors
  - `crates/rmlx-distributed/src/transport.rs` — Hop-count-aware ring ordering from topology env
- **Tests:** Tree allreduce matches ring; topology-aware ring uses optimal ordering

### PR 5.11: rmlx-rdma — Implement pipelined circular buffers
- **Issues:** R-P1-1
- **Complexity:** L
- **Files:**
  - `crates/rmlx-rdma/src/collectives.rs` — `PipelinedRingBuffer` with N=4 slots; overlap send+reduce
- **Tests:** Pipelined throughput > non-pipelined; correctness verified

**Phase 5 Total: 11 PRs (5 sub-PRs) — ~8 weeks with 2 engineers**

---

## Phase 6: Infrastructure and Polish (Weeks 31+)

**Goal:** Lazy eval, Python bindings, production monitoring, and remaining issues.

### PR 6.1 + 6.2: rmlx-core — Lazy evaluation + graph compilation
- **Issues:** MLX parity (critical infrastructure)
- **Complexity:** 2XL + 2XL = 4-8 weeks combined
- **Strategy:** This is the largest single effort. Split into:
  - **6.1a:** Lazy `Array` with compute DAG + `eval()` materialization
  - **6.1b:** Topological sort scheduler + basic multi-op batching
  - **6.2a:** Pattern matching for fusible op sequences
  - **6.2b:** JIT Metal kernel generation for fused patterns
- **Tests:** Lazy eval matches eager; fused kernel correct; benchmark improvement

### PR 6.3: rmlx-metal + rmlx-core — Multi-stream scheduling
- **Issues:** MLX parity
- **Complexity:** L
- **Files:**
  - Wire StreamManager for concurrent compute + copy streams; stream parameter to op dispatch

### PR 6.4: rmlx-distributed — Add heartbeat/watchdog fault detection
- **Issues:** D-P1-4
- **Complexity:** L
- **Files:**
  - Create `crates/rmlx-distributed/src/health.rs` — Per-peer heartbeat, timeout detection, callback on failure
- **Note:** Moved from Phase 3 to Phase 6. Fault tolerance is important but not on critical path for v1.

### PR 6.5: rmlx-distributed — Replace fake warmup benchmarks
- **Issues:** D-P1-3
- **Complexity:** M
- **Files:**
  - `crates/rmlx-distributed/src/warmup.rs` — Real Metal matmul + RDMA ping-pong calibration

### PR 6.6: All crates — PyO3 bindings
- **Issues:** D-P2-6
- **Complexity:** XL
- **Files:**
  - Create `crates/rmlx-python/` with PyO3 for Array, Device, Linear, Attention, MoE, Sampler, Group

### PR 6.7: All crates — Comprehensive test coverage drive
- **Issues:** All test gaps
- **Complexity:** XL (ongoing)
- **Target:** >80% line coverage

### PR 6.8: All crates — Replace eprintln! with tracing
- **Issues:** Various P3
- **Complexity:** S

---

## Integrated P2/P3 Backlog

Rather than a flat list, P2/P3 items are assigned to the phase where they naturally integrate. Items already bundled above are marked ✓.

### During Phase 0 (bundle with safety fixes)
| Item | Bundle With | Status |
|------|-------------|--------|
| M-MED-2: ScopedPool !Send | PR 0.19 (dedicated) | ✓ |
| M-MED-3: capture.rs panic paths | PR 0.12 | ✓ |
| M-MED-5: StreamNotFound variant | PR 0.14 | ✓ |
| M-MED-1: Wire StreamState.label | PR 0.15 | ✓ |
| C-P2-11: dtype.rs docs mismatch | PR 0.2 | ✓ |
| C-P2-9: dead im2col_f32 | PR 0.4 | ✓ |
| C-P2-6: GGUF offset validation | PR 0.10 | ✓ |

### During Phase 1 (bundle with vertical slice)
| Item | Bundle With |
|------|-------------|
| N-P1-2: QuantizedLinear cache weights | PR 1.5 |
| N-P1-5: GGUF strict pattern matching | PR 1.8 |
| C-P1-10: Conv stride/dilation validation | Standalone S PR |
| C-P1-11: layer_norm/rms_norm dtype validation | Standalone S PR |
| C-P2-1: Inconsistent error variants | Standalone S PR |

### During Phase 2 (bundle with distributed)
| Item | Bundle With |
|------|-------------|
| N-P1-19: parallel world_size validation | PR 2.6 |
| X-P1-3: Export RMLX_IBV_DEVICES | PR 2.8 |
| D-Codex-P2-2: SlabRing TOCTOU | PR 2.2 |

### During Phase 3 (bundle with serving)
| Item | Bundle With |
|------|-------------|
| M-MED-4: CB completion rich errors | Standalone S PR |
| C-P1-3: Metrics snapshot ordering | Standalone S PR |
| X-P2-4: Remove orphaned vector_add.metal | Standalone S PR |

### During Phase 4 (bundle with performance)
| Item | Bundle With |
|------|-------------|
| P2-A3: BufferCache LRU LinkedHashMap | PR 4.3 |
| P2-A4: Mutex poison consistency | PR 4.3 |
| P2-A5: Replace busy-spin waits | PR 4.3 |
| P2-R3: Wire RdmaMetrics | PR 4.10 |
| P2-R4: Wire PortFailover | Standalone M PR |
| P2-R5: SharedBuffer lazy MR reg | Standalone M PR |
| N-P1-1: MoE PerExpert CB overhead | PR 4.8 |
| N-P1-12: MoePipeline true overlap | Standalone L PR |
| M-F3: Per-encoder MTLFence | Standalone M PR |
| M-F5: GPU error buffer | Standalone S PR |
| M-F10: MTLHeap exposure | Standalone M PR |
| HIGH-2: Rename ICB modules | PR 4.11 |

### During Phase 5 (bundle with breadth)
| Item | Bundle With |
|------|-------------|
| N-P1-7: Embedding load_weights | PR 5.6 |
| N-P1-3: Linear bias CB consolidation | Standalone S PR |
| N-P1-11: QuantizedKvCache dequant | Standalone L PR |
| N-P1-14: capacity_factor usage | PR 4.11c |
| N-P1-15: Auto-causal mask | PR 5.2 |
| N-P1-18: SlidingWindow kernel mask | PR 5.2 |

### During Phase 6 (bundle with polish)
| Item | Bundle With |
|------|-------------|
| X-P2-1: Trim dead re-exports | Standalone M PR |
| X-P2-2: Typed error variants with From | Standalone L PR |
| X-P2-3: Unify Device/DeviceRef | Standalone M PR |
| X-P1-6: ZeroCopyBuffer auto-complete | Standalone M PR |
| D-P2-3: TCP fallback transport | Standalone L PR |
| D-P2-5: CreditManager safety audit | Standalone S PR |
| D-P2-7: Coordinator timeout | Standalone S PR |
| D-P2-8: PipelineStage error state | Standalone S PR |
| D-P2-4: Latency histogram metrics | Standalone S PR |
| All P3 items | Bundled with test coverage drive |

### Explicitly Deferred (Training Scope) [T]
| Item | Rationale |
|------|-----------|
| C-P0-5: SDPA backward write races | Training only (backward pass) |
| C-P0-13: SDPA backward dtype unchecked | Training only |
| C-P1-4: VJP f16/bf16 support | Training only |
| C-P1-5: VJP diamond-graph accumulation | Training only |
| C-P1-16: VJP coverage expansion | Training only |
| C-P1-8: Missing unit tests for VJP ops | Training only |
| N-P1-4: MlaKvCache advance (training cache) | Addressed when MLA impl'd (PR 5.1) |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| FlashAttention Metal (PR 3.1) takes >4 weeks | High | Blocks serving perf | Vendor `metal-flash-attention` first; optimize later. Paged KV works without FA. |
| MoE wiring (PR 2.3) reveals additional EP bugs | Medium | Delays Phase 2 | EP audit doc already identifies known issues. Budget 1 week buffer. |
| BFC allocator (PR 4.12) destabilizes memory mgmt | Medium | Perf regression | Feature-flag gated; keep MetalAllocator as fallback. |
| Lazy eval (PR 6.1) requires touching every op | High | Scope explosion | Design lazy Array as wrapper; existing ops work unchanged in eager fallback. |
| GGUF loader fails on real models | Medium | Blocks Phase 1 | Test with Llama-7B-Q4_0 early. Budget fix time in PR 1.8. |

---

## Timeline Summary

### With 1 Engineer (Serial)

| Phase | Weeks | Cumulative | Milestone |
|-------|-------|------------|-----------|
| 0: Stop the Bleeding | 2 | Week 2 | No crashes/UB |
| 1: Vertical Slice | 4 | Week 6 | **Single-node Llama generates text** |
| 2: Distributed | 4 | Week 10 | Multi-node MoE correct |
| 3: Serving | 6 | Week 16 | Concurrent request serving |
| 4: Performance | 6 | Week 22 | Competitive tok/s |
| 5: Breadth | 8 | Week 30 | Multi-model support |
| 6: Infrastructure | 10 | Week 40 | Lazy eval, Python, polish |
| **Total** | **~40 weeks** | | **Production ready (8/10)** |

### With 2 Engineers (Parallel where possible)

| Phase | Weeks | Parallel Strategy |
|-------|-------|-------------------|
| 0 | 1.5 | Split: Engineer A = core/alloc, Engineer B = metal/cli/distributed |
| 1 | 3 | PR 1.1-1.5 parallel, then 1.6-1.9 serial |
| 2 | 3 | Engineer A = MoE wiring (2.1-2.5), Engineer B = TP + RDMA (2.6-2.8) |
| 3 | 5 | Engineer A = FlashAttention (3.1), Engineer B = Paged KV + Scheduler (3.2-3.3) |
| 4 | 4 | Engineer A = alloc + metal (4.1-4.6), Engineer B = distributed + ICB (4.9-4.11) |
| 5 | 5 | Engineer A = ops + models (5.3-5.6), Engineer B = serving + CLI (5.7-5.11) |
| 6 | 6 | Engineer A = lazy eval (6.1-6.2), Engineer B = everything else |
| **Total** | **~28 weeks** | |

### With 3 Engineers

| Phase | Weeks | Notes |
|-------|-------|-------|
| 0 | 1 | Trivially parallel |
| 1 | 2.5 | 3rd engineer starts Phase 2 prep (reading EP audit) |
| 2 | 2 | 3rd engineer starts Phase 3 FA research |
| 3 | 4 | FA dedicated engineer; paged KV dedicated; scheduler + misc 3rd |
| 4-6 | 12 | Full parallel |
| **Total** | **~22 weeks** | |

---

## Key Differences from v1 Plan

| Aspect | v1 | v2 (revised) |
|--------|----|----|
| First working milestone | Week 8 (after Phase 1 + half of Phase 2) | **Week 6** (Phase 1: Llama generates text) |
| RMSNorm/RoPE placement | Phase 2 (too late) | **Phase 1** (prerequisite for any LLM) |
| SDPA backward races | Phase 0 (wrong — training only) | **Deferred [T]** (inference-first) |
| MLA/SlidingWindow | Phase 1 (too early for niche models) | **Phase 5** (after standard GQA works) |
| FlashAttention sizing | XL (1-2 weeks) — unrealistic | **2XL (2-4 weeks)** — realistic for Metal |
| ICB (EP-7) placement | Phase 3 "Correctness" (wrong category) | **Phase 4 "Performance"** (it's an optimization) |
| BFC allocator sizing | Single XL PR | **2XL with feature flag** (recognized as epic) |
| P2/P3 backlog | Flat ~60 items | **Integrated into phases** where they naturally fit |
| PR 0.7 (5 ops) | Single PR | **Split into PR 0.6-0.9** (independent reviews) |
| PR 0.8 (GGUF + op counter) | Mixed unrelated fixes | **Split:** GGUF in 0.10, op counter in 3.4 |
| Scope decision | Implicit | **Explicit prerequisite** (inference-only v1) |
| Risk register | None | **5 identified risks** with mitigations |
| Team scaling | "55 weeks" with no parallelization | **1/2/3 engineer timelines** |
| Phase validation | None | **Concrete checkpoint** per phase |

---

*Generated by Claude Opus 4.6 — 2026-03-06 (v2 revised)*
*Source: ~/rmlx/production-readiness-audit.md (183 issues)*
