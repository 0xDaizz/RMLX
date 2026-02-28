//! Linear (fully connected) layer: y = x @ W^T + bias

/// Linear layer configuration.
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
}

/// Linear layer with optional bias.
pub struct Linear {
    config: LinearConfig,
    // In a full implementation, these would hold Array references.
    // For now, we track the config and delegate compute to rmlx-core ops.
}

impl Linear {
    pub fn new(config: LinearConfig) -> Self {
        Self { config }
    }

    pub fn in_features(&self) -> usize {
        self.config.in_features
    }

    pub fn out_features(&self) -> usize {
        self.config.out_features
    }

    pub fn has_bias(&self) -> bool {
        self.config.has_bias
    }
}
