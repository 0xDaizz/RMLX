//! rmlx-nn — Neural network layers for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod activations;
pub mod attention;
pub mod conv;
pub mod dynamic;
pub mod embedding;
pub mod expert_group;
pub mod gguf_loader;
pub mod layer_norm;
pub mod linear;
pub mod mla;
pub mod models;
pub mod moe;
pub mod moe_pipeline;
pub mod paged_kv_cache;
pub mod parallel;
pub mod prefill_plan;
pub mod prefill_pool;
pub mod prefix_cache;
pub mod quantized_linear;
pub mod rms_norm;
pub mod rope;
pub mod safetensors_loader;
pub mod sampler;
pub mod scheduler;
pub mod sliding_window;
pub mod transformer;

// ── Re-exports of core types ──
pub use activations::{
    Activation, ActivationType, GELUFast, HardSigmoid, HardSwish, LeakyReLU, Mish, QuickGELU, ReLU,
    SiLU, Sigmoid, Softplus, Softsign, Tanh, ELU, GELU, GLU, SELU,
};
pub use attention::{
    Attention, AttentionConfig, BatchKvCache, LayerKvCache, QuantizedArray, QuantizedKvCache,
    RotatingKvCache,
};
pub use conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
pub use dynamic::DynamicExecContext;
pub use embedding::{Embedding, EmbeddingConfig};
pub use expert_group::ExpertGroup;
pub use gguf_loader::{GgufLoadError, GgufWeightMap};
pub use layer_norm::{LayerNorm, LayerNormConfig};
pub use linear::{Linear, LinearConfig};
pub use mla::{Mla, MlaConfig, MlaKvCache};
pub use moe::{load_balance_loss, Expert, MoeConfig, MoeForwardMetrics, MoeLayer, MoeStrategy};
pub use moe_pipeline::{MoePipeline, MoePipelineConfig};
pub use paged_kv_cache::{BlockId, BlockManager, PagedKvCache, PagedKvError};
pub use parallel::{ColumnParallelLinear, RowParallelLinear, TpError};
pub use prefill_plan::{PlanCache, PlanStep, PrefillPlan};
pub use prefill_pool::{PrefillBufferPool, Slot as PrefillSlot, NUM_SLOTS as PREFILL_NUM_SLOTS};
pub use prefix_cache::{PrefixCache, PrefixMatch};
pub use quantized_linear::{
    AwqLinear, GptqLinear, KQuantConfig, KQuantType, QuantBits, QuantizedLinear,
    QuantizedLinearConfig,
};
pub use rms_norm::{RMSNorm, RMSNormConfig};
pub use rope::{RotaryPositionEmbedding, RotaryPositionEmbeddingConfig};
pub use safetensors_loader::{QuantizationConfig, SafetensorsLoadError, SafetensorsWeightMap};
pub use sampler::{Sampler, SamplerConfig};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerError, SchedulerOutput, SeqMeta};
pub use sliding_window::{SlidingWindowAttention, SlidingWindowAttentionConfig};
pub use transformer::{
    fused_norm_threshold, set_fused_norm_threshold, CachedDecode, FeedForward, FeedForwardType,
    ForwardMode, TransformerBlock, TransformerConfig, TransformerModel,
};
