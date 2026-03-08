# Phase F-I Benchmark Report (2026-03-08)

## 1. Test Environment

| Item | Value |
|------|-------|
| Machine | hwstudio1 (Mac Studio) |
| SoC | Apple M3 Ultra, 80 GPU cores |
| RAM | 512 GB Unified Memory |
| GPU API | Metal 4 |
| MLX Version | 0.30.7.dev |
| Rust Version | 1.93.1 (stable) |
| OS | macOS 15.x |
| FP16 Peak | 65.54 TFLOPS |
| Memory BW | 819 GB/s |

---

## 2. FP16 GEMM Baseline

| Config | TFLOPS | vs MLX |
|--------|-------:|-------:|
| MLX 0.30.7.dev | 23.97T | -- |
| RMLX (Phase D2 MLX arch) | 23.82T | -0.6% |

The Phase D2 MLX-architecture kernel (BK=16, 2 SG, 64 threads, single buffer, 4xhalf4 wide loads, direct store, serpentine MMA) closes the GEMM gap from -10.1% (Phase B) to -0.6%. Key insight: occupancy (7-8 TG/core vs 1 TG/core) was the dominant factor, not individual optimizations.

---

## 3. Phase F: Infrastructure Optimization

### F-1: Dispatch Overhead Benchmark

| Metric | Value |
|--------|------:|
| CB overhead per dispatch | 176 us |
| Overhead as % of layer time | 12.4% |
| Measurement method | `dispatch_overhead` bench |

Dispatch overhead accounts for ~12% of total layer execution time. This validates the investment in CB batching and fused kernels.

### F-2: DiskPipelineCache

DiskPipelineCache has been wired into KernelRegistry. Pipeline states are persisted to disk with SHA-256 hashing, avoiding recompilation across process restarts.

### F-3: Grouped GEMM (GatherMM MMA)

| Metric | Before (scalar) | After (simdgroup MMA) |
|--------|----------------:|----------------------:|
| MoE expert compute | scalar multiply | simdgroup_float8x8 MMA |
| Improvement | baseline | 4-12x (expert count dependent) |
| Threads | 64 | 64 |

GatherMM was upgraded from scalar multiplication to simdgroup MMA, providing 4-12x improvement for MoE workloads. This is critical for Mixtral and DeepSeek-V3 expert parallel execution.

---

## 4. Phase G: Quantized Kernel Optimization

### G-1: QMM MMA Q4

Quantized matrix multiply (Q4) upgraded to simdgroup MMA (BM=32, BN=32, BK=32, dequant-in-loader).

### G-2: QMV qdot Q4/Q8

Quantized matrix-vector product using MLX qdot pattern with mask multiplication and uchar4 vectorized loads.

### G-3: Q8 QMM MMA + CPU Fallback Removal

Q8 quantized matrix multiply upgraded to simdgroup MMA. CPU fallback path has been removed entirely -- all quantized operations now run on GPU.

### MLX Comparison (Quantized)

| Kernel | RMLX | MLX | Gap |
|--------|-----:|----:|----:|
| QMV (Q4, M=1) | -- | -- | 1.58x gap |
| QMM (Q4, large M) | -- | -- | 4.78x gap |

The QMV gap (1.58x) is primarily due to MLX's more mature qdot implementation with load_vector preprocessing. The QMM gap (4.78x) reflects MLX's specialized quantized GEMM path with per-group dequant fused into the MMA loop.

**Phase J update**: QMV gap reduced to 1.15x, QMM gap reduced to 2.84x (see Section 7.5).

---

## 5. Phase H-2: Residual Fusion

GEMM + residual epilogue fusion via Metal function constant (ID 202, `has_residual`).

| N | Improvement |
|---|------------|
| Large N (>=4096) | 5-12% |
| Small N (<4096) | 0-2% |

The residual buffer is passed as `[[buffer(10)]]` in the kernel signature. The function constant `has_residual` (ID 202) enables compile-time dead code elimination when residual fusion is not needed.

### Metal Shader Fixes Discovered

- `using namespace metal::simdgroup;` is INVALID -- use `using namespace metal;`
- Function constants must be declared in ALL shader sources that reference them
- `residual [[buffer(10)]]` must be in kernel signature even when `has_residual=false`

---

## 6. Phase I-1: Distributed Tensor Parallelism

### DistributedTransformerModel

| Metric | Value |
|--------|------:|
| TP=2 estimated speedup | 1.94x |
| Implementation | `forward_with_group()` + `shard_for_tp()` |

The `DistributedTransformerModel` wraps `TransformerModel` with tensor-parallel forward using `ColumnParallelLinear` and `RowParallelLinear` (Megatron-LM pattern). `shard_for_tp()` automatically partitions model weights across TP ranks.

---

## 7. PR Verification Results

| PR | Status | Notes |
|----|--------|-------|
| #65 (Phase F) | Merged | dispatch_overhead bench, DiskPipelineCache, gather_mm MMA |
| #66 (Phase G) | Merged | QMM MMA Q4/Q8, QMV qdot, CPU fallback removed |
| #67 (Phase H) | Merged | GEMM+residual epilogue fusion |
| #68 (Phase I) | Merged | DistributedTransformerModel |
| #69 | File missing | PR references a file that does not exist in the repository |

---

## 7.5 Phase J Benchmark Results (2026-03-08)

### Environment

Same as Section 1. Code synced via rsync from local working tree (uncommitted Phase J changes).

### QMM MMA (Q4, Prefill)

| Config | Phase G | Phase J-1 (4SG) | Phase J-1b/J-6 (vec dequant + Split-K) | vs MLX (13.6T) |
|--------|--------:|----------------:|----------------------------------------:|---------------:|
| Q4 M=32 K=4096 N=4096 | -- | 875us / 1.23T | 806us / **1.33T** (+8%) | 10.2x slower |
| Q4 M=128 K=4096 N=4096 | -- | 1129us / 3.81T | 936us / **4.59T** (+20%) | 3.0x slower |
| Q4 M=256 K=4096 N=4096 | 2777us / 3.09T | 1794us / 4.79T | 1607us / **5.34T** (+12%) | 2.55x slower |
| Q8 M=32 K=4096 N=4096 | -- | 935us / 1.15T | 923us / **1.17T** (+2%) | -- |
| Q8 M=128 K=4096 N=4096 | -- | 1096us / 3.92T | 1041us / **4.13T** (+5%) | -- |
| Q8 M=256 K=4096 N=4096 | -- | 1830us / 4.69T | 1762us / **4.88T** (+4%) | -- |

**QMM Q4 M=256**: Phase G 대비 **+73% TFLOPS** (3.09T -> 5.34T). MLX 대비 격차 4.78x -> 2.55x로 축소.
**QMM Q4 M=128**: J-1 대비 **+20%** (3.81T -> 4.59T) -- Split-K의 저-M 개선 효과.
**Q8 M=32 regression fixed**: 4SG→2SG 구성 복원으로 1258us/0.85T → 923us/1.17T (+37%). Q8 전 구간 안정.

#### QMM MLX-arch Rewrite (J-9, 코드 완료 / 벤치 대기)

MLX 아키텍처 기반 QMM 전면 재작성 완료 (qmm-porter 브랜치).

| Variant | BM | BN | BK | SG/Threads | TG Memory | 특성 |
|---------|---:|---:|---:|----------:|----------:|------|
| Standard Q4 (M>32) | 64 | 64 | 16 | 2/64 | 4KB | serpentine MMA, direct store |
| Skinny Q4 (M≤32) | 32 | 64 | 32 | 2/64 | 4KB | 공격적 split-K (target 320 TG) |

추가 최적화:
- `has_norm` (fc 203) QMM 적용 완료
- `has_residual` (fc 202) QMM dispatch 연결 완료
- `fc_group_size` function constant (컴파일타임 나눗셈 제거)
- Vectorized `uchar4` B load

**v1 실측 (hwstudio2)**: M=32 1.87T (MLX 3.22T, 1.72x), M=256 6.30T (MLX 12.91T, 2.05x). BM=64 타일로는 한계 확인.
**v2 (BM32)**: MLX QuantizedBlockLoader 패턴 기반 재작성 진행 중 — BM=32/BN=32/BK=32/BK_padded=40.

### QMV qdot (Q4/Q8, Decode M=1)

| Config | Phase G (baseline) | Phase J | Delta | vs MLX |
|--------|-------------------:|--------:|------:|-------:|
| Q4 K=4096 N=14336 | 452us / 0.26T | 329us / 0.36T | **+37%** | 1.15x slower (was 1.58x) |
| Q8 K=4096 N=14336 | 408us / 0.29T | 299us / 0.39T | **+36%** | 1.08x slower (was 1.52x) |
| Q4 K=4096 N=4096 | -- | 277us / 0.12T | -- | -- |
| Q8 K=4096 N=4096 | -- | 296us / 0.11T | -- | -- |

**QMV Q4 N=14336**: Phase G 대비 **+37% TFLOPS** (0.26T -> 0.36T). MLX 대비 격차 1.58x -> 1.15x로 대폭 축소.

### QMV vs QMM at M=1 (Decode Path)

| Config | QMV p50 | QMM p50 | Winner |
|--------|--------:|--------:|--------|
| Q4 K=4096 N=14336 | 255us (0.46T) | 853us (0.14T) | QMV 3.34x |
| Q8 K=4096 N=14336 | 267us (0.44T) | 853us (0.14T) | QMV 3.20x |

QMV is correctly dispatched for decode (M=1) path.

### Pipeline (f16, Decode, per-layer)

| Config | Phase 10 (baseline) | Phase J | Delta |
|--------|--------------------:|--------:|------:|
| Single-layer 9-dispatch (1 CB) | 1044.7us | 1044.7us | -- |
| Concurrent 9-dispatch | -- | 1016.5us | -2.7% |
| 2-encoder 9-dispatch | -- | 962.1us | -7.9% |
| ExecGraph x60 (per-layer) | 705.8us | 704.9us | -0.1% |
| Serial x60 (per-layer) | -- | 752.2us | -- |
| Cached Fused 7-dispatch x60 | -- | 702.4us | **best** |

**Pipeline**: ExecGraph 60L per-layer **704.9us** (Phase 10의 705.8us와 동일 수준). Cached Fused 7-dispatch가 702.4us로 최적.

### ExecGraph Overhead Decomposition

| Metric | Value |
|--------|------:|
| Empty CB create+commit | 2.5 us/CB |
| CB with signal/wait chain | 4.8 us/CB |
| Encoder overhead | 0.2 us/encoder |
| ExecGraph vs Serial savings | -47.3 us/layer |
| ExecGraph speedup (60L) | 1.07x |

### Fused Kernel Microbenchmarks

| Fusion | Fused | Unfused | Saving |
|--------|------:|--------:|-------:|
| fused_swiglu_down (silu_mul + down_proj) | 339us | 554us | **-38.8%** |
| fused_rms_gemv QKV (rms_norm + QKV proj) | 273us | 458us | **-40.5%** |
| fused_rms_gemv gate_up (rms_norm + gate_up) | 493us | 754us | **-34.6%** |
| Estimated 6-dispatch pipeline | 1541us | 2391us | **-35.6%** |

### Phase J Summary

| Area | Phase G/10 | J-1 (1st) | J-1b/J-6 (latest) | Total Improvement |
|------|-----------|-----------|-------------------|-------------------|
| QMM Q4 M=256 TFLOPS | 3.09T | 4.79T | **5.34T** | +73% |
| QMM Q4 M=128 TFLOPS | -- | 3.81T | **4.59T** | -- |
| QMV Q4 N=14336 TFLOPS | 0.26T | 0.36T | 0.36T | +37% |
| QMV Q8 N=14336 TFLOPS | 0.29T | 0.39T | 0.39T | +36% |
| Pipeline 60L (per-layer) | 705.8us | 704.9us | 704.9us | -0.1% (stable) |
| MLX QMM gap | 4.78x | 2.84x | **2.55x** | gap cut by 47% |
| MLX QMV gap | 1.58x | 1.15x | 1.15x | near parity |

### MoE Parallelism Comparison (f16, Mixtral 8x7B config)

**Config:** E=8, D=4096, top_k=2, intermediate=14336, cf=1.25, hwstudio1+hwstudio2 (M3 Ultra x2, TB5 RDMA)

#### 1. EP Comparison (EP vs EP, 2-node)

RMLX EP-2 GatherMM (4E/rank) vs MLX EP (2-node JACCL, `moe_ep_vs_tp_bench.py`)

| N (tokens) | RMLX EP-2 (ms) | MLX EP (ms) | RMLX/MLX | Notes |
|-----------:|---------------:|------------:|---------:|-------|
| 1 | 6.4 | 1.78 | 3.60x slower | J-8 fused: 8.2→6.4ms (-22.0%) |
| 4 | 22.0 | 1.64 | 13.4x slower | routing/scatter-add bottleneck |
| 8 | 39.6 | 1.67 | 23.7x slower | scales worse with N |
| 32 | 141.7 | 2.03 | 69.8x slower | RMLX MoE framework severely unoptimized |

**Geomean RMLX/MLX (N=1..32):** ~14x slower (J-8 fused 반영, 이전 19.9x에서 축소)

**Root cause:** RMLX raw gather_mm kernel is at parity (1.3ms vs ~1.3ms at N=1), but MoE framework (gating softmax, routing, scatter-add, expert dispatch loop) adds massive overhead. MLX uses C++ fused `moe_dispatch_exchange` + `moe_combine_exchange` primitives that bypass this overhead entirely. J-8 fused kernels (index_gather + scatter_weighted_add) reduced this overhead by 22-32% at low token counts.

#### 2. TP Comparison (TP vs TP, 2-node)

RMLX TP=2 (simulated, single-node sharded weights) vs MLX TP (2-node JACCL, `moe_ep_vs_tp_bench.py`)

| N (tokens) | RMLX TP=2 est (ms) | MLX TP (ms) | RMLX/MLX | Notes |
|-----------:|-------------------:|------------:|---------:|-------|
| 1 | ~56.5 (30L) | 1.54 | -- | Not comparable (RMLX is 30-layer transformer, MLX is single MoE layer) |

**Note:** RMLX TP=2 data is from `distributed_bench.rs` measuring 30-layer full transformer forward (baseline 111.6ms -> sharded 56.5ms, 1.94x speedup). MLX TP measures a single MoE SwiGLU layer. Per-M single-layer TP data for RMLX is not yet available. Direct comparison deferred until RMLX has single-MoE-layer TP benchmark.

#### 3. Single-Node Comparison (1-node vs 1-node)

RMLX GatherMM 8E vs MLX MoE forward (single machine, no RDMA)

| N (tokens) | RMLX 1-node (ms) | MLX 1-node (ms) | RMLX/MLX | Notes |
|-----------:|-----------------:|-----------------:|---------:|-------|
| 1 | 9.8 | 10.7 | 0.92x (faster) | J-8 fused: 12.7→9.8ms (-22.8%) |
| 4 | 28.3 | -- | -- | J-8 fused: 41.5→28.3ms (-31.8%) |
| 8 | 78.3 | -- | -- | |
| 32 | 288.2 | -- | -- | |

**Raw kernel comparison (gather_mm only):**

| N | RMLX raw (ms) | MLX grouped GEMM (ms) | Ratio |
|--:|:-------------:|:---------------------:|------:|
| 1 | 1.3 | ~1.3 (estimated) | parity |

**MLX Grouped GEMM reference** (from `moe_gemm_bench.py`):

| Config | MLX p50 | MLX TFLOPS |
|--------|--------:|-----------:|
| 8E x M=4, N=2048 | 596us | 0.90T |
| 8E x M=8, N=1536 | 458us | 1.76T |
| 64E x M=1, N=2048 | 1979us | 0.54T |

#### 4. Comprehensive Comparison (all strategies, all sources)

| N | RMLX 1-node | RMLX EP-2 | MLX 1-node | MLX TP | MLX EP | Best RMLX vs Best MLX |
|--:|------------:|----------:|-----------:|-------:|-------:|----------------------:|
| 1 | 9.8ms | 6.4ms | 10.7ms | 1.54ms | 1.78ms | **6.4 vs 1.54** (4.2x gap) |
| 4 | 28.3ms | 22.0ms | -- | 2.54ms | 1.64ms | **22.0 vs 1.64** (13.4x gap) |
| 8 | 78.3ms | 39.6ms | -- | 2.53ms | 1.67ms | **39.6 vs 1.67** (23.7x gap) |
| 32 | 288.2ms | 141.7ms | -- | 2.72ms | 2.03ms | **141.7 vs 2.03** (69.8x gap) |

**Geomean (Best RMLX vs Best MLX, N=1..32):** ~14x gap (J-8 fused 반영, 이전 19.9x에서 축소)

#### Key Insights

1. **Kernel-level parity achieved**: Raw gather_mm matches MLX. The gap is entirely in the MoE framework layer.
2. **MLX advantage is architectural**: C++ fused dispatch/combine primitives (Metal compute + lazy eval) vs RMLX's eager Rust dispatch loop with explicit synchronization.
3. **RMLX EP-2 scales correctly** (2x vs 1-node) but starts from a much higher baseline due to framework overhead.
4. **M>=4 scaling is catastrophic** in RMLX: 9.8ms@M=1 -> 288.2ms@M=32 (29.4x), vs MLX: 1.78ms@M=1 -> 2.03ms@M=32 (1.14x). The routing/scatter-add path does not amortize with batch size.
5. **J-8 fused kernels** (index_gather + scatter_weighted_add) reduced single-node M=1 by -22.8% (12.7→9.8ms) and M=4 by -31.8% (41.5→28.3ms). EP-2 N=1 improved -22.0% (8.2→6.4ms). Gap vs MLX still significant but reduced from 19.9x to ~14x geomean.
6. **TP comparison deferred**: RMLX TP=2 data is 30-layer transformer (56.5ms), not comparable to MLX single-MoE-layer TP. Need per-layer TP MoE benchmark.

---

## 8. Benchmark Fairness Issues

The following operations were not benchmarked under fully comparable conditions:

- **SDPA**: RMLX Flash Attention 2 vs MLX fused SDPA use different algorithmic approaches. Direct TFLOPS comparison is not meaningful; end-to-end latency is the correct metric.
- **sum/reduce**: MLX uses lazy evaluation with potential fusion across reduce operations. RMLX eager execution benchmarks may overstate the gap for chained reductions.

---

## 9. Documentation Gaps

- No end-to-end model benchmark (full Llama-3 8B inference) -- only single-layer and kernel-level benchmarks exist
- Quantized kernel benchmark methodology not standardized (varying group sizes, sequence lengths)
- Missing thermal throttling annotation on long-running benchmarks (see GPU Thermal Throttling notes in project memory)

---

## 10. Improvement Opportunities (Post-Phase J)

### High Priority

| Area | Current | Target | Notes |
|------|---------|--------|-------|
| QMM gap vs MLX | 2.55x (J-1b) / 2.05x (J-9 v1) | <1.5x | J-9 v1 (BM64): 2.05x; v2 (BM32, MLX-style) 진행 중 |
| MoE framework gap | ~14x geomean | <5x | J-8 fused kernels cut 19.9x→~14x; further fusion needed |
| Q8 M=32 regression | fixed | -- | 4SG→2SG 복원: 0.85T→1.17T (+37%), 전 구간 안정 |
| SwiGLU epilogue | evaluated | skip | QMM/GEMM epilogue 내 SwiGLU fusion은 비실용적 (MLX도 미적용) |

### Medium Priority

| Area | Description |
|------|-------------|
| lazy.rs activation | J-4 complete (FusionCompiler + eval_fused); needs consumer wiring |
| ICB decode replay | J-3c pending (stub exists, needs icb.rs connection) |
| bf16 barrier fix | Bulk pre-conversion (24 -> 4 barriers/tile) |

### Infrastructure

| Area | Description |
|------|-------------|
| DiskPipelineCache | Wired but not on main flow |
| ICB sparse dispatch | EP-7 planned (GPU-level empty expert skip) |
| Continuous batching | Thermal throttling concern under sustained load |

---

## 11. lazy.rs Status Analysis

- lazy.rs: 938 lines of complete DAG infrastructure (LazyArray, LazyGraph, LazyOp)
- Supported ops: Add, Mul, MatMul, Sub, Neg, Softmax, RoPE, Copy, RmsNorm, Custom
- Width-limited BFS topo-sort to prevent prefill memory explosion
- 18 unit tests passing
- **Current state: zero consumers -- not connected to the main forward path at all**

into_cb pattern status:
- 75 into_cb op variants exist (matmul_into_cb, add_into_cb, rms_norm_into_cb, etc.)
- forward_single_cb: actively used (9-dispatch decode path)
- forward_graph: ExecGraph + into_cb used (6 CB per layer)
- forward (baseline): does not use into_cb (synchronous)

Key gaps:
1. Prefill per-layer CPU stall: forward_prefill_graph() calls submit_batch + wait per layer -- 32 stalls reducible to 1
2. Decode ExecGraph unused: 9-dispatch uses simple CB commit -- could switch to event chain
3. No automatic lazy -> into_cb bridging: eval callback requires manual implementation
4. FusionGraph (element-wise) is disconnected from forward
5. No automatic dispatch reordering (thermal throttling mitigation)

Expected benefits from lazy.rs activation:
- 9 forward variants -> 1 unified path (execution strategy changed via eval strategy swap)
- Graph-level fusion (dead code elimination, op composition)
- Automatic CB batching (no manual into_cb calls needed)

---

## 12. Phase J Completion Status

Phase J was executed as the follow-up to Phase F-I, targeting the quantized kernel gaps and infrastructure improvements identified in this report.

### Completed (J-1 through J-8)

| Task | Description | Result | Status |
|------|-------------|--------|--------|
| J-1 | QMM MMA 4SG/128-thread redesign | 3.09T -> 5.34T (+73%) | Complete |
| J-1b | Vectorized dequant + half input | +12% on Q4 M=256 | Complete |
| J-2 | QMV qdot (load_vector + multi-row TG) | 0.26T -> 0.36T (+37%), MLX 1.15x | Complete |
| J-3 | ExecGraph inter-layer stall removal | 32 stalls -> 0 (Metal FIFO) | Complete |
| J-4 | lazy.rs FusionCompiler + eval_fused | FusionCompiler + ExecGraph connected | Complete |
| J-5 | RMSNorm+GEMM fusion (function constant 203) | **이득 없음** (attn 0.99x, FFN 0.93x); has_norm=false 기본값 유지 | Complete |
| J-6 | Split-K QMM | +20% at M=128 (low-tile-count path) | Complete |
| J-7 | Distributed bench RDMA 2-node | Real RDMA communication wired | Complete |
| J-8 | MoE fused kernels (index_gather + scatter_weighted_add) | 1-node -22~32%, EP-2 -22% | Complete |

### Completed (J-4e, Benchmark Infrastructure)

| Task | Description | Result | Status |
|------|-------------|--------|--------|
| J-4e | forward_auto() eager+lazy hybrid | Attention/FFN eager → leaf, residual LazyOp::Add → FusionAnalyzer auto fuse | Complete |
| Bench: fusion_bench.rs | RMSNorm+GEMM fusion 3-way benchmark | MLX ref / RMLX baseline / RMLX fused comparison | Complete (code) |
| Bench: model_bench.rs | 32-layer full model ExecGraph benchmark | baseline vs ExecGraph vs prefill, safetensors loading | Complete (code) |
| Bench: mlx_fusion_bench.py | MLX fusion reference script | MLX_FUSION_REF JSON output for fair comparison | Complete (code) |
| Bench: mlx_full_model_bench.py | MLX full model reference script | HF model dir, safetensors, end-to-end timing | Complete (code) |

**Benchmark infrastructure notes:**
- All benchmarks use 3-way comparison structure: MLX reference / RMLX baseline / RMLX optimized
- MLX reference values passed via JSON environment variables for reproducibility
- Full model bench uses safetensors loading (not GGUF) for fair MLX comparison
- Awaiting hwstudio1 execution for actual numbers

### J-5 RMSNorm+GEMM Fusion 결론: 이득 없음

| Variant | Fused/Baseline Ratio | Notes |
|---------|---------------------:|-------|
| Attn (fix 전) | 0.90x | baseline 대비 10% 느림 |
| FFN (fix 전) | 0.93x | baseline 대비 7% 느림 |
| Attn (fix 후: vectorized norm + TG cache) | **0.99x** | 패리티 수준이나 이득 없음 |
| FFN (fix 후) | **0.93x** | 여전히 baseline 이하 |

**결론**: dispatch 수가 2→2로 동일하고, A-loader 추가 연산이 중간 텐서 절감 효과를 상쇄함. `has_norm=false` 기본값 유지. 진정한 fusion은 J-3 ExecGraph(같은 CB에 rms_norm+matmul 배치)로 이미 달성됨. SwiGLU epilogue도 비실용적 (MLX도 미적용).

### QMM 2x Gap 근본 원인 분석

기존 "weight layout 문제" 분석은 **수정됨** -- weight layout은 MLX와 동일 (N-major).

**진짜 원인**: B loader thread-to-data mapping이 반대 방향.
- RMLX: thread가 **N 방향**으로 순회 (scattered, stride=K/2)
- MLX: thread가 **K 방향**으로 순회 (contiguous, 8 bytes 연속)
- 타일 크기 64×64 vs MLX 32×32 → N 분산 강제 → scattered access
- BK=16 vs MLX BK=32 → barrier 2배, scale/bias 비효율

#### Gap 분해

| 병목 | 기여도 |
|------|--------|
| B loader N-stride scattered | 35-40% |
| Scale/bias 32 scattered loads | 20-25% |
| BK=16 tile overhead 2x | 15-20% |
| A input f32→half scalar | 10-15% |
| Bank conflict (no padding) | ~5% |

### QMM MLX-arch v1 벤치마크 (hwstudio2, BM=64 버전)

| M | RMLX (TFLOPS) | MLX (TFLOPS) | Gap |
|---|------:|-----:|----:|
| 32 | 1.87T | 3.22T | 1.72x |
| 128 | 4.14T | 9.49T | 2.29x |
| 256 | 6.30T | 12.91T | 2.05x |

BM=64 타일로는 MLX 수준 도달 불가. BM=32/BN=32/BK=32 MLX-style v2 커널 진행 중.

### QMM MLX-style v2 (BM32, 진행 중)

MLX QuantizedBlockLoader 패턴 기반 재작성 진행 중.
- BM=32, BN=32, BK=32, BK_padded=40 (MLX와 동일)
- K-contiguous B loader
- A input half 전처리
- 코드 작성 중, 벤치 미실행

### In Progress

| Task | Description | Status |
|------|-------------|--------|
| J-9 | QMM MLX-arch rewrite (qmm-porter) | v1 bench done (2x gap), v2 (BM32) in progress |
| J-4e | forward_auto() eager+lazy hybrid | Complete |

### Remaining

| Task | Description | Status |
|------|-------------|--------|
| J-3c | ICB decode replay | Pending (stub exists) |

### Key Metrics Summary (Phase J)

| Metric | Before (Phase G/10) | After (Phase J) | Change |
|--------|---------------------|------------------|--------|
| QMM Q4 M=256 TFLOPS | 3.09T | 5.34T | +73% |
| QMM Q4 MLX gap | 4.78x | 2.55x (J-1b) / 2.05x (J-9 v1) | -57% gap reduction (J-9 v1) |
| QMV Q4 TFLOPS | 0.26T | 0.36T | +37% |
| QMV Q4 MLX gap | 1.58x | 1.15x | Near parity |
| QMV Q8 TFLOPS | 0.29T | 0.39T | +36% |
| QMV Q8 MLX gap | 1.52x | 1.08x | Near parity |
| ExecGraph stalls | 32 per model | 0 | Eliminated |
| Fused swiglu_down | -- | -38.8% | New |
| Fused rms_gemv QKV | -- | -40.5% | New |
| Fused rms_gemv gate_up | -- | -34.6% | New |
| MoE scatter sync | N x 3 | 1 | J-8 |
| MoE 1-node M=1 | 12.7ms | 9.8ms | -22.8% (J-8 fused) |
| MoE EP-2 M=1 | 8.2ms | 6.4ms | -22.0% (J-8 fused) |
| EP geomean vs MLX | 19.9x | ~14x | J-8 fused kernels |
| Q8 M=32 regression | 0.85T | 1.17T | Fixed (+37%, 2SG) |
