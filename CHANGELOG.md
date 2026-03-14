# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **objc2-metal migration Phase 1**: Full transition from `metal-rs` to `objc2-metal` ecosystem across all crates (91050e8, 2026-03-13)
- **objc2-metal migration Phase 2**: Minimize `unsafe` blocks and raw `msg_send!` calls, consolidate idiomatic patterns (369e28b, 2026-03-14)

### Added
- **Metal 4 feature-gated API**: Add `metal4` feature flag with downstream integration for future Metal 4 capabilities (ca67196, 2026-03-14)

### Fixed
- Resolve prefill bench GPU errors caused by unretained command buffers (89c27d5, 2026-03-14)

---

## [0.10.0] - 2026-03-13

### Added
- **Speculative decoding framework** (PR #80): Draft model runner, verification engine, token tree management, and acceptance sampling primitives for speculative decoding workflows (bf0401a)

### Changed
- **Rust dispatch optimization P0-P6** (PR #79): ArrayPool for zero-allocation buffer reuse, fused norm fallback path, reduced per-dispatch overhead across the entire ops pipeline (f94bd2f)

### Performance
- **NAX GEMM + SDPA strided 4D** (PR #78): P5 NAX 64x128 GEMM kernel, strided 4D SDPA support, prefill dispatch reduced from 12 to 9 dispatches per layer (34597fe, 2026-03-12)
- **Prefill SDPA parity** (PR #77): Q4 QMM/QMV reaching 94% of MLX throughput, SDPA causal masking optimization, set_bytes + uninit buffer allocation elimination (6c3775e, 2026-03-12)

---

## [0.9.0] - 2026-03-11

### Added
- Prefill optimization waves 1-3 with MLX parity improvements (PR #75, PR #76)
- SDPA parity optimizations: NAX causal masking, BK=32 tile size, norm fusion (f125a81)
- Per-operation prefill profiling benchmarks with RMLX + MLX side-by-side comparison

### Changed
- f16 native activation pipeline: eliminate f32-to-f16 roundtrip casts (PR #74)
- Low-M Q4 QMM optimization: BatchQMV, Tiny MMA, GatherQMV, shape-aware dispatch (PR #72)

### Fixed
- GEMM kernel 4SG + threadgroup padding + Split-K occupancy, closing 35% MLX gap (cc7ba8c)
- Metal shader syntax: `constant constexpr` inside kernel corrected to `constexpr` (1858151)

### Performance
- set_bytes + uninit optimization: eliminate per-dispatch buffer allocation overhead (f787181)

---

## [0.8.0] - 2026-03-10

### Added
- **Phase Prefill Parity** (PR #71): MMA SDPA, NAX QMM, Steel dispatch strategy, KV bypass optimization
- **Phase J optimizations** (PR #70): Thunderbolt 5 auto-discovery, additional kernel tuning
- QMM MMA kernel: simdgroup MMA for Q4 and Q8 batched matmul (G-1, G-2, G-3)
- QMV qdot kernels: fast Q4/Q8 matrix-vector multiply
- gather_mm MMA: simdgroup MMA for MoE expert matmul (F-3)
- GEMM + residual epilogue fusion (H-2)
- DistributedTransformerModel with `forward_with_group()` and TP weight sharding API (I-1)
- DiskPipelineCache wired into KernelRegistry (F-2)
- Quantized benchmark suite for QMM/QMV Q4/Q8 (PR #69)
- Dual-queue Attn/MoE overlap benchmark

### Fixed
- bf16 B_f32 race condition + fused bf16 match arm (PR #63)
- ExecGraph sync: use CB `wait_until_completed` instead of event polling (PR #64)

---

## [0.7.0] - 2026-03-08

### Added
- **Phase E**: Prefill optimization, safetensors loader, bf16 GEMM support, dead code cleanup (PR #62)
- GEMM kernel-level optimization: wide_load (2x half4), optimal SG=2x4 layout applied to production (PR #60)
- MLX-architecture GEMM kernel with function constants, expanded to all f16 large matrices (PR #61)

### Changed
- Phase D GEMM dispatch: MLX-arch kernel, N-aware BM dispatch, Grouped GEMM Split-K
- Phase 3 small-M kernels: BM32, BM16 micro-tile, Split-K heuristic improvements
- MoE GEMM: Split-K f16, f32 MLX-arch, N-aware swizzle

### Performance
- **Phase 10 kernel fusion** (PR #54): fused_rms_gemv + fused_swiglu_down achieving 703.4 us/layer decode latency
- **Overhead squeeze** (PR #53): f16 default + framework optimizations, 714 us/layer (6.34x faster than MLX)
- **Serial decode optimizations** (PR #52): CachedDecode, 2-encoder pipeline, pre-resolved dispatch, 1,367 us/layer
- **Phase B prefill alpha** (PR #58): HiPerf GEMM, merged projections, ExecGraph integration
- **Phase A prefill** (PR #57): Single-CB pipeline, GQA slab SDPA

---

## [0.6.0] - 2026-03-07

### Added
- **Phase KO**: Kernel optimization achieving 77x decode speedup, within 5.1% of MLX (PR #47)
- Phase 7: Metal pipeline optimization, 12 items, beats MLX by 4.4% (PR #50)
- GEMV kernel optimizations: 78x decode speedup, 1.98x faster than MLX (PR #51)
- SDPA decode rewrite using MLX sdpa_vector pattern (12.5% faster)
- Lazy evaluation DAG and tracing-based logging replacing eprintln (PR #45)
- Comprehensive test coverage across 7 crates (PR #46)

### Changed
- Phase 5: Feature breadth with multi-model support (Llama, Mixtral, Qwen, DeepSeek) and new ops (PR #43)
- Phase 6: Infrastructure polish, correctness fixes, deferred findings (PR #44)
- Phase 4: Performance and allocator optimizations (PR #42)

### Fixed
- Phase 0: Eliminate crashes, undefined behavior, memory corruption, and security vulnerabilities (PR #41)

### Performance
- GEMV f16 BM8 widened loads: 2x half4 to 4x half4 (32 bytes/thread)
- Metal uniform<T> for GEMV loop bounds, consolidated KV cache encoders
- Multi-layer pipeline benchmarks: 2.09x faster than MLX at 30/60 layers

---

## [0.5.0] - 2026-03-05

### Added
- **Expert Parallelism (EP) phases 1-6**:
  - EP-1: GPU-native top-k routing (PR #25)
  - EP-2: Grouped expert GEMM + weight stacking (PR #31)
  - EP-3: Variable-length v3 protocol (PR #32)
  - EP-4: Compute-communication overlap with TBO + SBO (PR #33)
  - EP-5: FP8 wire format for reduced bandwidth (PR #34)
  - EP-6: ICB sparse expert launch + RDMA slab ring (PR #35)
- EP phases 2-6 wired into MoE forward path with FP8/SlabRing integration (PR #37)
- Phase 3: Serving infrastructure with FlashAttention, paged KV cache, and continuous batching (PR #41)
- Phase 1: Vertical slice for single-node Llama inference

---

## [0.4.0] - 2026-03-04

### Added
- **ExecGraph integration** + production hardening across 6 phases (PR #23)
- Native Rust CLI (`rmlx-cli` crate) replacing Python scripts (PR #22)
- MLX-style `distributed::init()` API (PR #21)

### Changed
- ExecGraph pipeline fused from 8 command buffers to 5 command buffers (17.4x speedup)

### Fixed
- SDPA tile sizes reduced to fit 32KB threadgroup memory
- SDPA general kernel thread count synced with THREADS_PER_TG (128)
- Metal shader `constant` and `thread` address space qualifiers corrected
- rms_norm single-row threshold and benchmark unused assignments

---

## [0.3.0] - 2026-03-03

### Added
- **Phase 2 feature completeness** + audit remediation (PR #19)
- KV cache, API re-exports, per-expert MoE metrics (PR #2)
- Comprehensive kernel rewrite for MLX parity (PR #4)
- 11 serving-support primitives S1-S5 (PR #11)
- RDMA runbook helpers with sanitized host identifiers

### Changed
- Project license switched to MIT
- Core ops + nn layers optimized (C2-C9, N4-N8) (PR #18)
- Metal fence/streams/buffer-cache, alloc alignment/pool/residency (M5-M8, A4-A8) (PR #17)
- MoE policy improvements and RDMA fixes (D5-D10, R1-R3) (PR #16)

### Fixed
- MoE dispatch/combine correctness D1-D4 (PR #13)
- Alloc cache bounds, Metal command pipeline, quantized GEMM (PR #14)
- NN MoE GPU routing, batched execution, GPU matmul (PR #15)

---

## [0.2.0] - 2026-03-01

### Added
- Zero-copy distributed engine: SharedBuffer, GPU-initiated RDMA, CQ progress, TP/EP
- CI pipeline: required checks + required review for hardware-dependent gates

### Fixed
- Production hardening rounds 1-4: panic elimination, lifetime safety, fused kernels, unwrap elimination, capacity mismatch resolution, deadlock prevention, indexing safety
- P0/P1 critical fixes: RDMA transport implementation, MoE protocol, kernel safety, Linear bias

---

## [0.1.0] - 2026-02-28

### Added
- **Phase 0**: Cargo workspace scaffolding + Metal GPU abstraction layer
- **Phase 1**: Zero-copy memory allocation (`rmlx-alloc`) + RDMA ibverbs integration (`rmlx-rdma`)
- **Phase 2A**: Shader vendoring, DType/Array types, KernelRegistry, 7 GPU kernels (40 tests)
- **Phase 2B**: GEMM, quantized matmul, indexing operations (43 tests)
- **Phase 3**: SharedEvent synchronization + dual queue + layer pipeline (52 tests)
- **Phase 4**: EP 3-Zone dispatch + MoE exchange protocol (62 tests)
- **Phase 5A**: `rmlx-nn` inference core — linear, embedding, attention, MoE, transformer, model definitions
- **Phase 6**: Dual Thunderbolt 5 multi-port striping + multi-node topology
- **Phase 7A**: Production hardening and observability (98 tests)
- **Phase 7B**: VJP autodiff framework + LoRA fine-tuning (10 new tests)
- **Phase 7C**: PyO3 Python bindings (15 new tests, 123 total)
- Real MLX shader integration, stride-aware copy, RDMA probe, MoE routing

### Fixed
- IbvSendWr FFI layout mismatch (P1 Critical hotfix)
- P0/P1 hardening: kernel correctness, RDMA safety, distributed protocol fixes
