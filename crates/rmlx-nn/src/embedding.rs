//! Token embedding: lookup table for discrete tokens.

use metal::MTLResourceOptions;
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};

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
        queue: &metal::CommandQueue,
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
        let mut flat_indices: Vec<u32> = Vec::with_capacity(seq_len * embed_dim);
        for &tid in token_ids {
            for j in 0..embed_dim {
                flat_indices.push(tid * embed_dim as u32 + j as u32);
            }
        }

        // Create a raw Metal buffer for u32 indices (u32 doesn't impl HasDType)
        let byte_size = (flat_indices.len() * std::mem::size_of::<u32>()) as u64;
        let idx_buffer = dev.new_buffer_with_data(
            flat_indices.as_ptr() as *const std::ffi::c_void,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        // Wrap in Array with Float32 dtype — the gather shader reads as uint* so the
        // dtype is irrelevant; we just need the correct buffer, shape, and element count.
        let indices_arr = Array::new(
            idx_buffer,
            vec![seq_len * embed_dim],
            vec![1],
            DType::Float32, // placeholder dtype — gather reads buffer(1) as uint*
            0,
        );

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
