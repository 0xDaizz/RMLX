# Audit: EP Dispatch/Combine — RMLX vs MLX

> Date: 2026-03-03
> Scope: `rmlx-distributed` (RMLX) vs `~/mlx` on node1 (MLX)
> Status: **Audit complete — fixes required**

---

## Summary

The high-level 3-zone auto backend design (CPU ≤64, Metal ≥320, RDMA) matches MLX.
However, dispatch loop ordering, per-rank capacity layout, and combine kernel caching
have correctness and performance issues that must be addressed.

---

## 1. Critical Discrepancies (Must Fix)

### 1.1 Dispatch Loop Ordering

| | RMLX | MLX |
|---|------|-----|
| Loop | `batch-outer, k-inner` | `k-outer, n-inner` |
| File | `moe_exchange.rs:495-519` | `cpu/distributed.cpp:180-230` |

MLX uses `k-outer` — top-1 selections (k=0) fill expert slots across all tokens first, then top-2 (k=1) fills remaining slots. This matches the DeepSeek/GShard convention where higher-priority expert assignments get slot priority.

RMLX uses `batch-outer` — all k selections for each token are assigned before moving to the next token. Given the same input, this produces **different routing results** than MLX.

**Fix**: Change loop to `k-outer, n-inner`.

### 1.2 Per-Rank Capacity Partitioning

| | RMLX | MLX |
|---|------|-----|
| Method | Global capacity per expert | Per-source-rank capacity partitioning |
| Layout | `[experts, capacity_per_expert, D]` flat | `[experts_per_device, world_size * capacity, D]` |

MLX gives each source rank an independent `capacity` number of slots per expert. RMLX shares a global capacity, so in multi-node scenarios one rank's tokens can overwrite another rank's slots.

**Fix**: Change dispatch layout to `[experts_per_device, world_size * capacity, D]`.

### 1.3 Combine Metal Kernel Not Cached

| | RMLX | MLX |
|---|------|-----|
| Caching | `combine_metal_cache` field exists but is **dead code** | `get_moe_kernel()` with built-in caching |
| File | `moe_exchange.rs` `MoeCombineExchange::combine_metal()` | `metal/moe.cpp` |

Every call creates a new Metal device, library, pipeline, and command queue. Severe performance bug.

**Fix**: Wire up `combine_metal_cache` following the dispatch `metal_cache: Mutex<Option<CachedMetalPipeline>>` pattern.

---

## 2. Medium Discrepancies (Should Fix)

### 2.1 Middle Zone Byte Threshold

| | RMLX | MLX |
|---|------|-----|
| Value | 4KB (`4_096`) | 2MB (`2_097_152`) |
| File | `moe_policy.rs` default | `moe_policy.cpp` `MLX_MOE_EP_GPU_SWITCH_BYTES` |

4KB is **4 orders of magnitude** lower. RMLX selects Metal unnecessarily for almost all middle-zone workloads.

**Fix**: Change default to 2MB, or use `ThresholdCalibration` result.

### 2.2 Hysteresis Path Gap

`moe_policy.rs:select()` line 145 uses raw `self.gpu_min` without hysteresis for the Metal-only path. Hysteresis is applied to CPU-to-Metal and Metal-to-RDMA transitions but not to the middle zone Metal decision.

**Fix**: Replace `self.gpu_min` with `gpu_thresh` on line 145.

### 2.3 Cooldown Semantics Differ

| | RMLX | MLX |
|---|------|-----|
| Method | 32 steps (count-based only) | 5000ms OR 1000 calls (whichever first) |
| File | `moe_policy.rs` | `moe_policy.cpp` |

MLX uses time-based OR call-count-based cooldown. RMLX uses step-count only.

---

## 3. Feature Gaps

### 3.1 Metal Kernels (7 vs 2)

**MLX** (`kernels/moe.h`):
1. `moe_dispatch_local` — scatter via precomputed slot_map
2. `moe_dispatch_scatter_remote` — scatter received remote tokens
3. `moe_combine_gather_remote` — gather expert outputs for remote requests
4. `moe_combine_weighted_sum` — single-source weighted accumulation
5. `moe_combine_weighted_sum_dual_src` — **zero-copy dual-source** weighted accumulation
6. `moe_packet_gather` — pack into 16B-aligned RDMA packets
7. `moe_packet_scatter` — unpack packets into target buffer

**RMLX** (`moe_exchange.rs`):
1. `moe_gather_scatter` — dispatch via atomic cursors
2. `moe_combine` — weighted sum

Missing: dual-source combine, remote scatter, packet gather/scatter, precomputed slot_map.

### 3.2 Multi-dtype Not Supported

| dtype | MLX | RMLX |
|-------|-----|------|
| f32 | Yes | Yes |
| f16 | Yes | No |
| bf16 | Yes | No |

MLX combine accumulates f16/bf16 inputs in **f32 accumulators** for precision. RMLX only supports f32.

### 3.3 Dual-Source Combine

MLX's `moe_combine_weighted_sum_dual_src` kernel reads directly from local and remote buffers without intermediate copies. RMLX copies remote data into a unified `all_expert_outputs` array before combine — an extra O(N*D) copy.

### 3.4 Packet Format

MLX uses 16B-aligned RDMA packet format with dedicated gather/scatter kernels. RMLX uses a 4B expert_id prefix with no alignment.

### 3.5 Autodiff

MLX's AllToAll primitive supports vjp/jvp/vmap. RMLX does not (inference-only, low priority).

---

## 4. RMLX-Only Advantages (Not in MLX)

| Feature | File | Description |
|---------|------|-------------|
| **SparseGuard** | `sparse_guard.rs` | EMA-based overflow detection + capacity auto-tuning + dense fallback |
| **Hysteresis** | `moe_policy.rs` | Prevents backend oscillation (partial bug on one path) |
| **CreditManager** | `credit_manager.rs` | UC recv-credit window management |
| **LayerPipeline** | `pipeline.rs` | Compute-RDMA overlap across layers |
| **Async dispatch/combine** | `moe_exchange.rs` | PendingOp handle-based async path |
| **ThresholdCalibration** | `moe_policy.rs` | Measures CPU/GPU crossover at startup |
| **PerfCounters** | `perf_counters.rs` | Zero-copy KPI tracking |
| **EpRuntimeContext** | `ep_runtime.rs` | Unified runtime hub (transport, progress, shared buffers, GPU events) |

---

## 5. Fix Priority

| Priority | Item | Difficulty | Impact |
|----------|------|------------|--------|
| P0 | 1.1 Dispatch loop ordering | Medium | Correctness — numerical divergence from other frameworks |
| P0 | 1.2 Per-rank capacity | Medium | Correctness — multi-node data collision |
| P0 | 1.3 Combine kernel caching | Low | Performance — Metal reinitialization on every call |
| P1 | 2.1 Byte threshold 4KB to 2MB | Low | Performance — unnecessary GPU kernel launches |
| P1 | 2.2 Hysteresis path | Low | Correctness — intended behavior broken |
| P2 | 3.1 Expand to 7 Metal kernels | High | Performance + functionality |
| P2 | 3.2 Multi-dtype (f16/bf16) | Medium | Practical usability |
| P2 | 3.3 Dual-source combine | Medium | Performance — eliminate O(N*D) copy |
| P3 | 3.4 Packet format 16B alignment | Medium | RDMA efficiency |

---

## 6. MLX Reference File Locations (node1)

| File | Role |
|------|------|
| `~/mlx/mlx/backend/cpu/distributed.cpp` | CPU dispatch/combine (929 lines) |
| `~/mlx/mlx/backend/metal/distributed.cpp` | Metal dispatch/combine (1067 lines) |
| `~/mlx/mlx/backend/metal/moe.cpp` | MoE Metal kernel management |
| `~/mlx/mlx/backend/metal/kernels/moe.h` | 7 Metal kernel sources |
| `~/mlx/mlx/distributed/moe_policy.cpp` | 3-zone policy |
| `~/mlx/mlx/distributed/moe_metrics.cpp` | MoE metrics |
| `~/mlx/mlx/distributed/moe_warmup.cpp` | Warmup protocol |
| `~/mlx/PLAN-expert-parallelism.md` | EP implementation plan |
| `~/mlx/ep-implementation-proposal.md` | EP implementation proposal |
