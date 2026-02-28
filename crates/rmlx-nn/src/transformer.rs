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

impl TransformerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".into());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if self.num_kv_heads > self.num_heads {
            return Err("num_kv_heads > num_heads".into());
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err("num_heads must be divisible by num_kv_heads".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        Ok(())
    }
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
