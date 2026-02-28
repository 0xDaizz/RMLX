//! Multi-head attention with KV cache support.

pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

pub struct Attention {
    config: AttentionConfig,
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.config.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    pub fn hidden_size(&self) -> usize {
        self.config.num_heads * self.config.head_dim
    }

    pub fn is_gqa(&self) -> bool {
        self.config.num_kv_heads < self.config.num_heads
    }
}
