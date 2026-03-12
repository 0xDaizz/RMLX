//! Buffer slot index constants for Metal kernel dispatches.
//!
//! Replaces magic numbers in `set_buffer(N, ...)` / `set_bytes(N, ...)` calls
//! to prevent silent corruption from slot collisions.
//!
//! Each sub-module corresponds to a kernel family and documents the exact
//! slot layout expected by the Metal shader.

/// Standard GEMM kernels (MlxArch, NaxArch, Full, Skinny, Small, etc.).
///
/// Layout:
/// ```text
/// 0  A           [M, K]
/// 1  B           [K, N]
/// 2  OUT         [M, N]
/// 3  M           (u32)
/// 4  N           (u32)
/// 5  K           (u32)
/// 6  batch_stride_a (u32)
/// 7  batch_stride_b (u32)
/// 8  batch_stride_c (u32)
/// 9  swizzle_log (u32, variant-dependent)
/// 10 residual    [M, N]  (fused residual-add epilogue)
/// 11 norm_weight [K]     (fused RMSNorm on A-tile load)
/// 12 inv_rms     [M]     (pre-computed inverse RMS per row)
/// 13 gate_result [M, N]  (fused SwiGLU epilogue)
/// ```
pub mod gemm {
    pub const A: u64 = 0;
    pub const B: u64 = 1;
    pub const OUT: u64 = 2;
    pub const M: u64 = 3;
    pub const N: u64 = 4;
    pub const K: u64 = 5;
    pub const BATCH_STRIDE_A: u64 = 6;
    pub const BATCH_STRIDE_B: u64 = 7;
    pub const BATCH_STRIDE_C: u64 = 8;
    pub const SWIZZLE_LOG: u64 = 9;
    /// Residual tensor for fused GEMM + residual-add epilogue.
    pub const RESIDUAL: u64 = 10;
    /// RMSNorm weight vector for fused norm-GEMM.
    pub const NORM_WEIGHT: u64 = 11;
    /// Pre-computed inverse-RMS values for fused norm-GEMM.
    pub const INV_RMS: u64 = 12;
    /// Gate result for fused SwiGLU epilogue.
    pub const GATE_RESULT: u64 = 13;
}

/// Split-K GEMM pass-1 kernel.
///
/// Layout:
/// ```text
/// 0  A         [M, K]
/// 1  B         [K, N]
/// 2  PARTIAL   [n_splits * M * N]  (f32 accumulator)
/// 3  M         (u32)
/// 4  N         (u32)
/// 5  K         (u32)
/// 6  N_SPLITS  (u32)
/// ```
pub mod splitk {
    pub const A: u64 = 0;
    pub const B: u64 = 1;
    pub const PARTIAL: u64 = 2;
    pub const M: u64 = 3;
    pub const N: u64 = 4;
    pub const K: u64 = 5;
    pub const N_SPLITS: u64 = 6;
}

/// Split-K reduce (pass-2) kernel.
///
/// Layout:
/// ```text
/// 0  PARTIAL   [n_splits * M * N]  (f32 input)
/// 1  OUT       [M, N]
/// 2  M         (u32)
/// 3  N         (u32)
/// 4  N_SPLITS  (u32)
/// ```
pub mod splitk_reduce {
    pub const PARTIAL: u64 = 0;
    pub const OUT: u64 = 1;
    pub const M: u64 = 2;
    pub const N: u64 = 3;
    pub const N_SPLITS: u64 = 4;
}

/// Grouped GEMM for MoE (single-pass, no Split-K).
///
/// Layout:
/// ```text
/// 0  A_STACKED   [sum(M_i), K]
/// 1  B_STACKED   [num_experts, K, N]
/// 2  OUT         [sum(M_i), N]
/// 3  OFFSETS     problem offsets buffer
/// 4  TILE_MAP    tile-to-problem map
/// 5  TILE_OFF    per-expert tile offsets
/// 6  K           (u32)
/// 7  N           (u32)
/// ```
pub mod grouped_gemm {
    pub const A_STACKED: u64 = 0;
    pub const B_STACKED: u64 = 1;
    pub const OUT: u64 = 2;
    pub const OFFSETS: u64 = 3;
    pub const TILE_MAP: u64 = 4;
    pub const TILE_OFF: u64 = 5;
    pub const K: u64 = 6;
    pub const N: u64 = 7;
}

/// Grouped Split-K GEMM pass-1 (MoE with K-splitting).
///
/// Same as `grouped_gemm` but adds TOTAL_M and N_SPLITS scalars.
///
/// Layout:
/// ```text
/// 0  A_STACKED
/// 1  B_STACKED
/// 2  PARTIAL
/// 3  OFFSETS
/// 4  TILE_MAP
/// 5  TILE_OFF
/// 6  K           (u32)
/// 7  N           (u32)
/// 8  TOTAL_M     (u32)
/// 9  N_SPLITS    (u32)
/// ```
pub mod grouped_splitk {
    pub const A_STACKED: u64 = 0;
    pub const B_STACKED: u64 = 1;
    pub const PARTIAL: u64 = 2;
    pub const OFFSETS: u64 = 3;
    pub const TILE_MAP: u64 = 4;
    pub const TILE_OFF: u64 = 5;
    pub const K: u64 = 6;
    pub const N: u64 = 7;
    pub const TOTAL_M: u64 = 8;
    pub const N_SPLITS: u64 = 9;
}

/// RMS normalization kernel (standard 3-buffer layout).
///
/// Layout:
/// ```text
/// 0  INPUT      [rows, axis_size]
/// 1  WEIGHT     [axis_size]  (or dummy when has_w=0)
/// 2  OUT        [rows, axis_size]
/// 3  AXIS_SIZE  (u32)
/// 4  EPS        (f32)
/// 5  W_STRIDE   (u32)
/// 6  HAS_W      (u32, 0 or 1)
/// ```
pub mod rms_norm {
    pub const INPUT: u64 = 0;
    pub const WEIGHT: u64 = 1;
    pub const OUT: u64 = 2;
    pub const AXIS_SIZE: u64 = 3;
    pub const EPS: u64 = 4;
    pub const W_STRIDE: u64 = 5;
    pub const HAS_W: u64 = 6;
}

/// Fused RMSNorm + residual-add kernel.
///
/// Layout:
/// ```text
/// 0  INPUT      [rows, axis_size]
/// 1  RESIDUAL   [rows, axis_size]  (mutated in-place)
/// 2  WEIGHT     [axis_size]
/// 3  OUT        [rows, axis_size]
/// 4  AXIS_SIZE  (u32)
/// 5  EPS        (f32)
/// 6  W_STRIDE   (u32)
/// 7  HAS_W      (u32)
/// ```
pub mod rms_norm_residual {
    pub const INPUT: u64 = 0;
    pub const RESIDUAL: u64 = 1;
    pub const WEIGHT: u64 = 2;
    pub const OUT: u64 = 3;
    pub const AXIS_SIZE: u64 = 4;
    pub const EPS: u64 = 5;
    pub const W_STRIDE: u64 = 6;
    pub const HAS_W: u64 = 7;
}

/// Inverse-RMS kernel (used by fused norm-GEMM).
///
/// Layout:
/// ```text
/// 0  INPUT      [rows, axis_size]
/// 1  OUT        [rows]  (f32 inv_rms per row)
/// 2  AXIS_SIZE  (u32)
/// 3  EPS        (f32)
/// ```
pub mod inv_rms {
    pub const INPUT: u64 = 0;
    pub const OUT: u64 = 1;
    pub const AXIS_SIZE: u64 = 2;
    pub const EPS: u64 = 3;
}

/// Fused SiLU-gate (element-wise `silu(gate) * up`).
///
/// Layout:
/// ```text
/// 0  GATE_OUT   [numel]
/// 1  UP_OUT     [numel]
/// 2  OUT        [numel]
/// 3  NUMEL      (u32)
/// ```
pub mod silu_gate {
    pub const GATE_OUT: u64 = 0;
    pub const UP_OUT: u64 = 1;
    pub const OUT: u64 = 2;
    pub const NUMEL: u64 = 3;
}

/// Fused SiLU-gate strided (merged gate+up tensor).
///
/// Layout:
/// ```text
/// 0  MERGED     [seq_len, 2*gate_dim]
/// 1  OUT        [seq_len, gate_dim]
/// 2  GATE_DIM   (u32)
/// 3  TOTAL_DIM  (u32)
/// 4  SEQ_LEN    (u32)
/// ```
pub mod silu_gate_strided {
    pub const MERGED: u64 = 0;
    pub const OUT: u64 = 1;
    pub const GATE_DIM: u64 = 2;
    pub const TOTAL_DIM: u64 = 3;
    pub const SEQ_LEN: u64 = 4;
}

/// Fused SwiGLU down-projection GEMV.
///
/// Layout:
/// ```text
/// 0  MAT        weight matrix
/// 1  GATE_UP    merged gate+up input
/// 2  OUT        output
/// 3  M          (u32)
/// 4  K          (u32)
/// 5  BIAS       residual / h_buf
/// ```
pub mod fused_swiglu_down {
    pub const MAT: u64 = 0;
    pub const GATE_UP: u64 = 1;
    pub const OUT: u64 = 2;
    pub const M: u64 = 3;
    pub const K: u64 = 4;
    pub const BIAS: u64 = 5;
}

/// Fused RMSNorm + GEMV kernel.
///
/// Layout:
/// ```text
/// 0  INPUT      [M, K]
/// 1  NORM_W     [K]
/// 2  MAT        weight matrix
/// 3  OUT        output
/// 4  M          (u32)
/// 5  K          (u32)
/// 6  EPS        (f32)
/// 7  W_STRIDE   (u32)
/// ```
pub mod fused_rms_gemv {
    pub const INPUT: u64 = 0;
    pub const NORM_W: u64 = 1;
    pub const MAT: u64 = 2;
    pub const OUT: u64 = 3;
    pub const M: u64 = 4;
    pub const K: u64 = 5;
    pub const EPS: u64 = 6;
    pub const W_STRIDE: u64 = 7;
}

// =========================================================================
// Quantized kernels (QMV / QMM / Gather)
// =========================================================================

/// QMV (quantized matrix-vector) qdot kernels.
///
/// Shared by all QMV variants: `affine_qmv_fast_*`, `affine_qmv_batched_*`,
/// and their split-K phase-1 counterparts.
///
/// Layout:
/// ```text
/// 0  WEIGHTS       packed Q4/Q8 weight buffer
/// 1  SCALES        per-group scale factors
/// 2  BIASES        per-group bias values
/// 3  INPUT         input vector / matrix (f16 or f32)
/// 4  OUTPUT        output buffer (or partial sums for split-K phase 1)
/// 5  PARAMS        packed params (out_features, in_features, group_size, M/bits)
/// 6  SPLITK_PARAMS (split-K only) (k_partitions, k_per_part)
/// ```
pub mod qmv {
    pub const WEIGHTS: u64 = 0;
    pub const SCALES: u64 = 1;
    pub const BIASES: u64 = 2;
    pub const INPUT: u64 = 3;
    pub const OUTPUT: u64 = 4;
    pub const PARAMS: u64 = 5;
    /// Split-K phase 1 only: (k_partitions, k_per_part).
    pub const SPLITK_PARAMS: u64 = 6;
}

/// QMM MMA kernels (affine_qmm_mma_q4, affine_qmm_mma_f16_q4, etc.).
///
/// Standard (non-Steel) MMA-based quantized matmul with optional fusion.
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed Q4/Q8 weight buffer
/// 2  SCALES        per-group scale factors
/// 3  BIASES        per-group bias values
/// 4  OUT           output [M, N]
/// 5  M             (u32)
/// 6  N             (u32)
/// 7  K             (u32)
/// 8  SWIZZLE_LOG   (u32) or K_PARTITIONS for tiny/skinny
/// 9  NORM_WEIGHT   [K] (fused RMSNorm, unused → dummy)
/// 10 INV_RMS       [M] (fused RMSNorm, unused → dummy)
/// 11 RESIDUAL      [M, N] (fused residual-add, unused → dummy)
/// 12 GATE_RESULT   [M, N] (fused SwiGLU, unused → dummy)
/// ```
pub mod qmm_mma {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const OUT: u64 = 4;
    pub const M: u64 = 5;
    pub const N: u64 = 6;
    pub const K: u64 = 7;
    /// Swizzle log (standard MMA) or k_partitions (tiny/skinny).
    pub const SWIZZLE_LOG: u64 = 8;
    /// RMSNorm weight (fused norm-QMM, else dummy).
    pub const NORM_WEIGHT: u64 = 9;
    /// Pre-computed inverse-RMS (fused norm-QMM, else dummy).
    pub const INV_RMS: u64 = 10;
    /// Residual tensor (fused residual-add, else dummy).
    pub const RESIDUAL: u64 = 11;
    /// Gate result for fused SwiGLU (else dummy).
    pub const GATE_RESULT: u64 = 12;
}

/// QMM Steel Split-K kernels (affine_qmm_steel_splitk_q4, steel4sg).
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed Q4 weight buffer
/// 2  SCALES        per-group scale factors
/// 3  BIASES        per-group bias values
/// 4  PARTIAL       partial sums buffer (or dummy when k_partitions==1)
/// 5  OUT           output [M, N]
/// 6  PARAMS        (M, N, K) as [u32; 3]
/// 7  SPLIT_PARAMS  (k_partition_size, k_partitions) as [u32; 2]
/// ```
pub mod qmm_steel_splitk {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const PARTIAL: u64 = 4;
    pub const OUT: u64 = 5;
    pub const PARAMS: u64 = 6;
    pub const SPLIT_PARAMS: u64 = 7;
}

/// QMM Steel kernels (affine_qmm_steel_q4) — non-split-K variant.
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed Q4 weight buffer
/// 2  SCALES        per-group scale factors
/// 3  BIASES        per-group bias values
/// 4  OUT           output [M, N]
/// 5  M             (u32)
/// 6  N             (u32)
/// 7  K             (u32)
/// 8  SWIZZLE_LOG   (u32)
/// 9  RESIDUAL      [M, N] (fused residual-add, else dummy)
/// 10 GATE_RESULT   [M, N] (fused SwiGLU, else dummy)
/// ```
pub mod qmm_steel {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const OUT: u64 = 4;
    pub const M: u64 = 5;
    pub const N: u64 = 6;
    pub const K: u64 = 7;
    pub const SWIZZLE_LOG: u64 = 8;
    /// Residual tensor (fused residual-add, else dummy).
    pub const RESIDUAL: u64 = 9;
    /// Gate result for fused SwiGLU (else dummy).
    pub const GATE_RESULT: u64 = 10;
}

/// QMM NAX kernels (affine_qmm_nax_q4, affine_qmm_nax_v2_q4).
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed Q4 weight buffer
/// 2  SCALES        per-group scale factors (f16)
/// 3  BIASES        per-group bias values (f16)
/// 4  OUT           output [M, N]
/// 5  M             (u32)
/// 6  N             (u32)
/// 7  K             (u32)
/// ```
pub mod qmm_nax {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const OUT: u64 = 4;
    pub const M: u64 = 5;
    pub const N: u64 = 6;
    pub const K: u64 = 7;
}

/// QMM Skinny f16 kernels (affine_qmm_skinny_f16_q4).
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed Q4 weight buffer
/// 2  SCALES        per-group scale factors
/// 3  BIASES        per-group bias values
/// 4  OUT           output [M, N]
/// 5  M             (u32)
/// 6  N             (u32)
/// 7  K             (u32)
/// 8  K_PARTITIONS  (u32)
/// 9  PARTIAL       partial sums buffer (or dummy when k_partitions==1)
/// ```
pub mod qmm_skinny {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const OUT: u64 = 4;
    pub const M: u64 = 5;
    pub const N: u64 = 6;
    pub const K: u64 = 7;
    pub const K_PARTITIONS: u64 = 8;
    pub const PARTIAL: u64 = 9;
}

/// QMM Legacy scalar kernel (affine_qmm, affine_qmm_mma_q8).
///
/// Layout:
/// ```text
/// 0  X             input activations [M, K]
/// 1  WEIGHTS       packed weight buffer
/// 2  SCALES        per-group scale factors
/// 3  BIASES        per-group bias values
/// 4  OUT           output [M, N]
/// 5  PARAMS        packed params (M, N, K, group_size)
/// ```
pub mod qmm_legacy {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const OUT: u64 = 4;
    pub const PARAMS: u64 = 5;
}

/// QMM Split-K reduce kernels (qmm_splitk_accum, steel_splitk_reduce).
///
/// Layout:
/// ```text
/// 0  PARTIAL       [k_partitions * M * N] f32 partial sums
/// 1  OUT           [M, N] output
/// 2  PARAMS        (N, k_partitions, partition_stride)
/// ```
pub mod qmm_splitk_reduce {
    pub const PARTIAL: u64 = 0;
    pub const OUT: u64 = 1;
    pub const PARAMS: u64 = 2;
}

/// Tiny/Skinny QMM reduce kernels (tiny_qmm_reduce, skinny_qmm_f16_reduce).
///
/// Layout:
/// ```text
/// 0  PARTIAL       [k_partitions * M * N] f32 partial sums
/// 1  OUT           [M, N] output
/// 2  N             (u32)
/// 3  K_PARTITIONS  (u32)
/// 4  MN_TOTAL      (u32) = M * N
/// ```
pub mod qmm_tiny_reduce {
    pub const PARTIAL: u64 = 0;
    pub const OUT: u64 = 1;
    pub const N: u64 = 2;
    pub const K_PARTITIONS: u64 = 3;
    pub const MN_TOTAL: u64 = 4;
}

/// GatherQMM scalar kernel (index-based QMM for MoE).
///
/// Layout:
/// ```text
/// 0  X             input activations [batch * M, K]
/// 1  WEIGHTS       packed weights [n_experts, N, K/2]
/// 2  SCALES        per-group scales
/// 3  BIASES        per-group biases
/// 4  INDICES       expert indices [batch]
/// 5  OUT           output [batch, M, N]
/// 6  BATCH         (u32)
/// 7  M             (u32)
/// 8  N             (u32)
/// 9  K             (u32)
/// 10 GROUP_SIZE    (u32)
/// 11 N_EXPERTS     (u32)
/// ```
pub mod gather_qmm {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const INDICES: u64 = 4;
    pub const OUT: u64 = 5;
    pub const BATCH: u64 = 6;
    pub const M: u64 = 7;
    pub const N: u64 = 8;
    pub const K: u64 = 9;
    pub const GROUP_SIZE: u64 = 10;
    pub const N_EXPERTS: u64 = 11;
}

/// GatherQMV fast kernels (gather_qmv_fast_q4, gather_qmv_fast_f16_q4).
///
/// Layout:
/// ```text
/// 0  X             input vectors [batch, K]
/// 1  WEIGHTS       packed weights [n_experts, N, K/2]
/// 2  SCALES        per-group scales
/// 3  BIASES        per-group biases
/// 4  INDICES       expert indices [batch]
/// 5  OUT           output [batch, N]
/// 6  PARAMS        (N, K, group_size, batch)
/// ```
pub mod gather_qmv {
    pub const X: u64 = 0;
    pub const WEIGHTS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const BIASES: u64 = 3;
    pub const INDICES: u64 = 4;
    pub const OUT: u64 = 5;
    pub const PARAMS: u64 = 6;
}

/// AWQ dequantization kernel (awq_dequant_q4).
///
/// Layout:
/// ```text
/// 0  QWEIGHT       packed uint32 weights [rows, cols/8]
/// 1  QZEROS        packed uint32 zero points [num_groups, cols/8]
/// 2  SCALES        f32 scale factors [num_groups, cols]
/// 3  OUT           f32 output [rows, cols]
/// 4  PARAMS        (rows, cols, group_size, cols_div8)
/// ```
pub mod awq {
    pub const QWEIGHT: u64 = 0;
    pub const QZEROS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const OUT: u64 = 3;
    pub const PARAMS: u64 = 4;
}

/// AWQ dequantization with g_idx reorder.
///
/// Layout:
/// ```text
/// 0  QWEIGHT       packed uint32 weights
/// 1  QZEROS        packed uint32 zero points
/// 2  SCALES        f32 scale factors
/// 3  G_IDX         group index permutation
/// 4  OUT           f32 output
/// 5  PARAMS        (rows, cols, group_size, cols_div8)
/// ```
pub mod awq_gidx {
    pub const QWEIGHT: u64 = 0;
    pub const QZEROS: u64 = 1;
    pub const SCALES: u64 = 2;
    pub const G_IDX: u64 = 3;
    pub const OUT: u64 = 4;
    pub const PARAMS: u64 = 5;
}

// =========================================================================
// SDPA (Scaled Dot-Product Attention) kernels
// =========================================================================

/// SDPA basic kernels (sdpa, sdpa_batched, decode variants).
///
/// Used by: `sdpa`, `sdpa_batched`, `sdpa_into_cb`, `sdpa_batched_into_cb`,
/// `sdpa_prefill_gqa_slab`, `sdpa_decode_batched_slab`, `sdpa_decode_preresolved`.
///
/// Layout:
/// ```text
/// 0  Q             query buffer
/// 1  K             key buffer
/// 2  V             value buffer
/// 3  OUT           output buffer
/// 4  MASK          attention mask (or dummy)
/// 5  PARAMS        packed params struct (content varies by variant)
/// 6  SCALE         softmax scale factor (f32)
/// ```
pub mod sdpa {
    pub const Q: u64 = 0;
    pub const K: u64 = 1;
    pub const V: u64 = 2;
    pub const OUT: u64 = 3;
    pub const MASK: u64 = 4;
    pub const PARAMS: u64 = 5;
    pub const SCALE: u64 = 6;
}

/// SDPA MMA prefill kernels (sdpa_prefill_mma_f16, sdpa_prefill_mma_bk32_f16).
///
/// Layout:
/// ```text
/// 0  Q             query slab
/// 1  K             key slab
/// 2  V             value slab
/// 3  OUT           output buffer
/// 4  MASK          attention mask (or dummy)
/// 5  N             seq_len (u32)
/// 6  S             kv_len (u32)
/// 7  D             head_dim (u32)
/// 8  GQA_FACTOR    num_heads / num_kv_heads (u32)
/// 9  SCALE         softmax scale (f32)
/// 10 STRIDE_S      KV seq stride (u32)
/// 11 NUM_Q_HEADS   total Q heads (u32)
/// 12 V_HEAD_STRIDE V head stride (u32)
/// 13 V_ROW_STRIDE  V row stride (u32)
/// ```
pub mod sdpa_mma {
    pub const Q: u64 = 0;
    pub const K: u64 = 1;
    pub const V: u64 = 2;
    pub const OUT: u64 = 3;
    pub const MASK: u64 = 4;
    pub const N: u64 = 5;
    pub const S: u64 = 6;
    pub const D: u64 = 7;
    pub const GQA_FACTOR: u64 = 8;
    pub const SCALE: u64 = 9;
    pub const STRIDE_S: u64 = 10;
    pub const NUM_Q_HEADS: u64 = 11;
    pub const V_HEAD_STRIDE: u64 = 12;
    pub const V_ROW_STRIDE: u64 = 13;
}

/// SDPA NAX prefill kernels (sdpa_prefill_nax_f16) — no mask buffer.
///
/// Layout:
/// ```text
/// 0  Q             query slab
/// 1  K             key slab
/// 2  V             value slab
/// 3  OUT           output buffer
/// 4  N             seq_len (u32)
/// 5  S             kv_len (u32)
/// 6  D             head_dim (u32)
/// 7  GQA_FACTOR    num_heads / num_kv_heads (u32)
/// 8  SCALE         softmax scale (f32)
/// 9  STRIDE_S      KV seq stride (u32)
/// 10 NUM_Q_HEADS   total Q heads (u32)
/// 11 V_HEAD_STRIDE V head stride (u32)
/// 12 V_ROW_STRIDE  V row stride (u32)
/// ```
pub mod sdpa_nax {
    pub const Q: u64 = 0;
    pub const K: u64 = 1;
    pub const V: u64 = 2;
    pub const OUT: u64 = 3;
    pub const N: u64 = 4;
    pub const S: u64 = 5;
    pub const D: u64 = 6;
    pub const GQA_FACTOR: u64 = 7;
    pub const SCALE: u64 = 8;
    pub const STRIDE_S: u64 = 9;
    pub const NUM_Q_HEADS: u64 = 10;
    pub const V_HEAD_STRIDE: u64 = 11;
    pub const V_ROW_STRIDE: u64 = 12;
}

/// SDPA diagnostic QKT kernel (sdpa_nax_diag_qkt_f16).
///
/// Layout:
/// ```text
/// 0  Q             query [N, D]
/// 1  K             key [S, D]
/// 2  OUT           score matrix [N, S]
/// 3  N             seq_len (u32)
/// 4  S             kv_len (u32)
/// 5  D             head_dim (u32)
/// 6  SCALE         softmax scale (f32)
/// ```
pub mod sdpa_diag_qkt {
    pub const Q: u64 = 0;
    pub const K: u64 = 1;
    pub const OUT: u64 = 2;
    pub const N: u64 = 3;
    pub const S: u64 = 4;
    pub const D: u64 = 5;
    pub const SCALE: u64 = 6;
}

/// SDPA diagnostic single-MMA kernel (sdpa_nax_diag_single_mma).
///
/// Layout:
/// ```text
/// 0  Q             [16, 16] f16
/// 1  K             [32, 16] f16
/// 2  OUT           [16, 32] f32
/// ```
pub mod sdpa_diag_single_mma {
    pub const Q: u64 = 0;
    pub const K: u64 = 1;
    pub const OUT: u64 = 2;
}
