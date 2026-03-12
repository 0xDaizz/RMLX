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
