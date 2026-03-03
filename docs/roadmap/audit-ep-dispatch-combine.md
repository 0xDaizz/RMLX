# Audit: EP Dispatch/Combine — RMLX vs MLX

> Date: 2026-03-03
> Scope: `rmlx-distributed` (RMLX) vs `~/mlx` on a reference host (MLX)
> Status: **All findings RESOLVED** (Phase 0+1+2 audit remediation, 2026-03-03)

---

## Summary

The high-level 3-zone auto backend design (CPU ≤64, Metal ≥320, RDMA) matches MLX.
All 6 findings from the original audit have been resolved in the Phase 0+1+2 full-crate
audit remediation. Dispatch loop ordering, per-rank capacity layout, combine kernel
caching, byte threshold, hysteresis, and cooldown semantics have all been fixed.

---

## 1. Critical Discrepancies -- RESOLVED

### 1.1 Dispatch Loop Ordering -- RESOLVED (commit `07fad80`, D1)

| | RMLX (before) | RMLX (after) | MLX |
|---|------|------|-----|
| Loop | `batch-outer, k-inner` | `k-outer, n-inner` | `k-outer, n-inner` |
| File | `moe_exchange.rs:495-519` | `moe_exchange.rs` (refactored) | `cpu/distributed.cpp:180-230` |

**Resolution**: Loop ordering changed to `k-outer, n-inner` to match DeepSeek/GShard convention.

### 1.2 Per-Rank Capacity Partitioning -- RESOLVED (commit `07fad80`, D2)

| | RMLX (before) | RMLX (after) | MLX |
|---|------|------|-----|
| Method | Global capacity per expert | Per-source-rank capacity partitioning | Per-source-rank capacity partitioning |
| Layout | `[experts, capacity_per_expert, D]` flat | `[experts_per_device, world_size * capacity, D]` | `[experts_per_device, world_size * capacity, D]` |

**Resolution**: Dispatch layout changed to per-source-rank capacity partitioning.

### 1.3 Combine Metal Kernel Not Cached -- RESOLVED (commit `07fad80`, D3)

| | RMLX (before) | RMLX (after) | MLX |
|---|------|------|-----|
| Caching | `combine_metal_cache` field exists but is **dead code** | `combine_metal_cache` fully wired up | `get_moe_kernel()` with built-in caching |

**Resolution**: `combine_metal_cache` wired up following the dispatch `metal_cache: Mutex<Option<CachedMetalPipeline>>` pattern. No more per-call Metal device/library/pipeline/queue creation.

---

## 2. Medium Discrepancies -- RESOLVED

### 2.1 Middle Zone Byte Threshold -- RESOLVED (commit `014875e`, D5)

| | RMLX (before) | RMLX (after) | MLX |
|---|------|------|-----|
| Value | 4KB (`4_096`) | 2MB (`2_097_152`) | 2MB (`2_097_152`) |

**Resolution**: Default byte threshold changed from 4KB to 2MB to match MLX.

### 2.2 Hysteresis Path Gap -- RESOLVED (commit `014875e`, D6)

**Resolution**: `self.gpu_min` replaced with `gpu_thresh` (hysteresis-aware value) on the Metal-only path. Hysteresis is now consistently applied across all zone transitions.

### 2.3 Cooldown Semantics -- RESOLVED (commit `014875e`, D7)

| | RMLX (before) | RMLX (after) | MLX |
|---|------|------|-----|
| Method | 32 steps (count-based only) | Dual: time-based (5000ms) OR count-based (1000 calls) | 5000ms OR 1000 calls (whichever first) |

**Resolution**: Cooldown now uses dual time-based OR call-count-based semantics to match MLX.

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

## 5. Fix Priority -- All P0/P1 RESOLVED

| Priority | Item | Difficulty | Impact | Status |
|----------|------|------------|--------|--------|
| P0 | 1.1 Dispatch loop ordering | Medium | Correctness | **RESOLVED** (D1) |
| P0 | 1.2 Per-rank capacity | Medium | Correctness | **RESOLVED** (D2) |
| P0 | 1.3 Combine kernel caching | Low | Performance | **RESOLVED** (D3) |
| P1 | 2.1 Byte threshold 4KB to 2MB | Low | Performance | **RESOLVED** (D5) |
| P1 | 2.2 Hysteresis path | Low | Correctness | **RESOLVED** (D6) |
| P1 | 2.3 Cooldown semantics | Low | Correctness | **RESOLVED** (D7) |
| P2 | 3.1 Expand to 7 Metal kernels | High | Performance + functionality | Existing (7 kernels already present) |
| P2 | 3.2 Multi-dtype (f16/bf16) | Medium | Practical usability | Future work |
| P2 | 3.3 Dual-source combine | Medium | Performance | Future work |
| P3 | 3.4 Packet format 16B alignment | Medium | RDMA efficiency | Future work |

---

## 6. MLX Reference File Locations (reference host)

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
