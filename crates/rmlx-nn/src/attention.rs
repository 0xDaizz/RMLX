//! Multi-head attention with KV cache support.

pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<(), String> {
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
        Ok(())
    }
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
