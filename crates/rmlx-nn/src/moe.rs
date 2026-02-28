//! Mixture of Experts layer with top-k gating.

pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

pub struct MoeLayer {
    config: MoeConfig,
}

impl MoeLayer {
    pub fn new(config: MoeConfig) -> Self {
        Self { config }
    }

    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.config.num_experts_per_token
    }

    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }
}
