//! rmlx-nn — Neural network layers for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod activations;
pub mod attention;
pub mod conv;
pub mod dynamic;
pub mod embedding;
pub mod gguf_loader;
pub mod layer_norm;
pub mod linear;
pub mod mla;
pub mod models;
pub mod moe;
pub mod parallel;
pub mod quantized_linear;
pub mod sliding_window;
pub mod transformer;

// ── Re-exports of core types ──
pub use activations::{Activation, ActivationType, GELUFast, SiLU, Sigmoid, Tanh, GELU};
pub use attention::{
    Attention, AttentionConfig, BatchKvCache, LayerKvCache, QuantizedArray, QuantizedKvCache,
    RotatingKvCache,
};
pub use conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
pub use dynamic::DynamicExecContext;
pub use embedding::{Embedding, EmbeddingConfig};
pub use gguf_loader::{GgufLoadError, GgufWeightMap};
pub use layer_norm::{LayerNorm, LayerNormConfig};
pub use linear::{Linear, LinearConfig};
pub use mla::{Mla, MlaConfig, MlaKvCache};
pub use moe::{load_balance_loss, MoeConfig, MoeForwardMetrics, MoeLayer};
pub use quantized_linear::{QuantBits, QuantizedLinear, QuantizedLinearConfig};
pub use sliding_window::{SlidingWindowAttention, SlidingWindowAttentionConfig};
pub use transformer::{
    FeedForward, FeedForwardType, TransformerBlock, TransformerConfig, TransformerModel,
};
