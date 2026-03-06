//! JIT element-wise kernel fusion engine.
//!
//! Generates Metal source for chains of element-wise ops (unary, binary,
//! ternary), compiles them at runtime, and caches the compiled pipelines.
//!
//! This benefits per-op forward paths and prefill — the 9-dispatch decode
//! path uses hand-fused kernels and is unaffected.
//!
//! # Limits (from MLX)
//!
//! - `MAX_FUSION_DEPTH`: 11 ops max in a single fused kernel
//! - `MAX_FUSION_ARRAYS`: 24 distinct input/output arrays

pub mod codegen;
pub mod graph;

pub use codegen::FusionCodegen;
pub use graph::{FusableOp, FusionGraph};

/// Maximum number of ops in a fused kernel.
pub const MAX_FUSION_DEPTH: usize = 11;

/// Maximum number of distinct input/output arrays.
pub const MAX_FUSION_ARRAYS: usize = 24;
