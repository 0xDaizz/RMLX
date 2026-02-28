//! Token embedding: lookup table for discrete tokens.

pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
}

pub struct Embedding {
    config: EmbeddingConfig,
}

impl Embedding {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }
}
