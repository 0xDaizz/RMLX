//! Transformer block: attention + MLP (or MoE).

pub enum FeedForwardType {
    Dense { intermediate_dim: usize },
    MoE { config: super::moe::MoeConfig },
}

pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub ff_type: FeedForwardType,
}

pub struct TransformerBlock {
    layer_idx: usize,
    config: TransformerConfig,
}

impl TransformerBlock {
    pub fn new(layer_idx: usize, config: TransformerConfig) -> Self {
        Self { layer_idx, config }
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

pub struct TransformerModel {
    config: TransformerConfig,
    num_layers: usize,
}

impl TransformerModel {
    pub fn new(config: TransformerConfig) -> Self {
        let num_layers = config.num_layers;
        Self { config, num_layers }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
