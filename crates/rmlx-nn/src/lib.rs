//! rmlx-nn — Neural network layers for RMLX

pub mod attention;
pub mod embedding;
pub mod linear;
pub mod models;
pub mod moe;
pub mod transformer;

// ── Re-exports of core types ──
pub use attention::{Attention, AttentionConfig};
pub use embedding::{Embedding, EmbeddingConfig};
pub use linear::{Linear, LinearConfig};
pub use moe::{MoeConfig, MoeLayer};
pub use transformer::{FeedForwardType, TransformerBlock, TransformerConfig, TransformerModel};
