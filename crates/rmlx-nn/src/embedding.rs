//! Token embedding: lookup table for discrete tokens.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue};

pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
}

pub struct Embedding {
    config: EmbeddingConfig,
    weight: Option<Array>,
}

impl Embedding {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            weight: None,
        }
    }

    /// Create an embedding layer with a pre-loaded weight table.
    ///
    /// `weight` shape: [vocab_size, embed_dim]
    pub fn from_array(config: EmbeddingConfig, weight: Array) -> Result<Self, KernelError> {
        if weight.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "weight must be 2D [vocab_size, embed_dim], got {}D",
                weight.ndim()
            )));
        }
        if weight.shape()[0] != config.vocab_size {
            return Err(KernelError::InvalidShape(format!(
                "weight shape[0]={} != vocab_size={}",
                weight.shape()[0],
                config.vocab_size
            )));
        }
        if weight.shape()[1] != config.embed_dim {
            return Err(KernelError::InvalidShape(format!(
                "weight shape[1]={} != embed_dim={}",
                weight.shape()[1],
                config.embed_dim
            )));
        }
        Ok(Self {
            config,
            weight: Some(weight),
        })
    }

    /// Forward pass: gather rows from the weight table by token IDs.
    ///
    /// `token_ids`: slice of token indices in [0, vocab_size)
    /// Returns: [seq_len, embed_dim]
    pub fn forward(
        &self,
        token_ids: &[u32],
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let weight = self
            .weight
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("Embedding: weights not loaded".into()))?;

        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let vocab_size = self.config.vocab_size as u32;
        let dev = registry.device().raw();

        // Validate token IDs are in range [0, vocab_size)
        for (i, &tid) in token_ids.iter().enumerate() {
            if tid >= vocab_size {
                return Err(KernelError::InvalidShape(format!(
                    "token_ids[{i}] = {tid} out of range [0, {vocab_size})"
                )));
            }
        }

        // Build flat index array: for each token, generate embed_dim consecutive indices
        // index[i * embed_dim + j] = token_ids[i] * embed_dim + j
        // Use u64 arithmetic to avoid overflow for large vocab_size * embed_dim products.
        let mut flat_indices: Vec<u32> = Vec::with_capacity(seq_len * embed_dim);
        for &tid in token_ids {
            for j in 0..embed_dim {
                let flat_idx = (tid as usize) * embed_dim + j;
if flat_idx > (u32::MAX as u64).try_into().unwrap() {
                    return Err(KernelError::InvalidShape(format!(
                        "embedding flat index overflow: tid={tid}, embed_dim={embed_dim}"
                    )));
                }
                flat_indices.push(flat_idx as u32);
            }
        }

        // Now that u32 implements HasDType, we can use Array::from_slice directly.
        let indices_arr = Array::from_slice(dev, &flat_indices, vec![seq_len * embed_dim]);

        // Flatten weight to 1D for gather
        let weight_flat = weight.reshape(vec![self.config.vocab_size * embed_dim])?;

        // Gather: output[i] = weight_flat[indices[i]]
        let gathered =
            rmlx_core::ops::indexing::gather(registry, &weight_flat, &indices_arr, queue)?;

        // Reshape to [seq_len, embed_dim]
        gathered.reshape(vec![seq_len, embed_dim])
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    pub fn has_weights(&self) -> bool {
        self.weight.is_some()
    }

    pub fn weight(&self) -> Option<&Array> {
        self.weight.as_ref()
    }
}
