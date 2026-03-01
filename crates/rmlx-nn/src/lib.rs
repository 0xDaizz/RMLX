//! rmlx-nn — Neural network layers for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod attention;
pub mod embedding;
pub mod linear;
pub mod models;
pub mod moe;
pub mod parallel;
pub mod transformer;

// ── Re-exports of core types ──
pub use attention::{Attention, AttentionConfig};
pub use embedding::{Embedding, EmbeddingConfig};
pub use linear::{Linear, LinearConfig};
pub use moe::{MoeConfig, MoeForwardMetrics, MoeLayer};
pub use transformer::{FeedForwardType, TransformerBlock, TransformerConfig, TransformerModel};
