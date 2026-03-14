//! Rotary Position Embedding (RoPE) neural network layer.
//!
//! Wraps `rmlx_core::ops::rope` as an nn module with precomputed
//! cos/sin frequency tables. Supports NTK-aware scaling via the
//! `scale` config parameter which adjusts the effective base frequency.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

/// Configuration for [`RotaryPositionEmbedding`].
pub struct RotaryPositionEmbeddingConfig {
    /// Dimension of each attention head (must be even).
    pub head_dim: usize,
    /// Maximum sequence length for precomputed frequency tables.
    pub max_seq_len: usize,
    /// Base frequency for the inverse-frequency computation (typically 10000.0).
    pub base_freq: f32,
    /// NTK-aware scaling factor applied to positions.
    /// `theta = scale * position * inv_freq`.
    /// Use `1.0` for no scaling.
    pub scale: f32,
    /// `true` for GPT-NeoX style (2k, 2k+1) pair indexing,
    /// `false` for LLaMA/Mistral split-half (k, k+half_dim).
    pub traditional: bool,
}

impl RotaryPositionEmbeddingConfig {
    /// Create a config with common defaults (base_freq=10000.0, scale=1.0, traditional=false).
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base_freq: 10000.0,
            scale: 1.0,
            traditional: false,
        }
    }
}

/// Rotary Position Embedding module.
///
/// Precomputes cos/sin frequency tables of shape `[max_seq_len, head_dim/2]`
/// on construction and uses them in the forward pass via the core RoPE kernel.
///
/// Frequency computation:
/// ```text
/// inv_freq[k] = 1.0 / (base_freq ^ (2k / head_dim))
/// cos_freqs[pos][k] = cos(scale * pos * inv_freq[k])
/// sin_freqs[pos][k] = sin(scale * pos * inv_freq[k])
/// ```
pub struct RotaryPositionEmbedding {
    /// Precomputed cosine frequency table `[max_seq_len, head_dim/2]`.
    cos_freqs: Array,
    /// Precomputed sine frequency table `[max_seq_len, head_dim/2]`.
    sin_freqs: Array,
    /// Layer configuration.
    config: RotaryPositionEmbeddingConfig,
}

impl RotaryPositionEmbedding {
    /// Create a new RoPE layer, precomputing cos/sin frequency tables.
    ///
    /// The `device` is used to allocate GPU buffers for the tables.
    /// The `dtype` parameter is ignored for the frequency tables themselves
    /// (which are always f32 as required by the core kernel), but is stored
    /// for documentation purposes.
    ///
    /// # Errors
    ///
    /// Returns an error if `head_dim` is zero, odd, or if `max_seq_len` is zero.
    pub fn new(
        config: RotaryPositionEmbeddingConfig,
        device: &ProtocolObject<dyn MTLDevice>,
        _dtype: DType,
    ) -> Result<Self, KernelError> {
        if config.head_dim == 0 {
            return Err(KernelError::InvalidShape(
                "RotaryPositionEmbedding: head_dim must be > 0".into(),
            ));
        }
        if config.head_dim % 2 != 0 {
            return Err(KernelError::InvalidShape(format!(
                "RotaryPositionEmbedding: head_dim must be even, got {}",
                config.head_dim
            )));
        }
        if config.max_seq_len == 0 {
            return Err(KernelError::InvalidShape(
                "RotaryPositionEmbedding: max_seq_len must be > 0".into(),
            ));
        }

        // Precompute frequency tables on the CPU, then upload to GPU.
        let (cos_table, sin_table) = ops::rope::precompute_freqs(
            config.max_seq_len,
            config.head_dim,
            config.base_freq,
            config.scale,
        )?;

        let half_dim = config.head_dim / 2;
        let shape = vec![config.max_seq_len, half_dim];

        let cos_freqs = Array::from_slice(device, &cos_table, shape.clone());
        let sin_freqs = Array::from_slice(device, &sin_table, shape);

        Ok(Self {
            cos_freqs,
            sin_freqs,
            config,
        })
    }

    /// Forward pass: apply rotary position embedding to `input`.
    ///
    /// # Arguments
    ///
    /// * `registry` - Kernel registry with RoPE kernels registered.
    /// * `input` - `[seq_len, head_dim]` or `[batch*n_heads, seq_len, head_dim]`.
    /// * `offset` - Position offset for incremental/KV-cache decoding.
    /// * `queue` - Metal command queue.
    ///
    /// # Returns
    ///
    /// Array with the same shape as `input`, with rotary embeddings applied.
    pub fn forward(
        &self,
        registry: &KernelRegistry,
        input: &Array,
        offset: u32,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        ops::rope::rope_ext(
            registry,
            input,
            &self.cos_freqs,
            &self.sin_freqs,
            offset,
            1.0, // scale already baked into precomputed tables
            self.config.traditional,
            true, // forward rotation
            queue,
        )
    }

    /// Reference to the layer configuration.
    pub fn config(&self) -> &RotaryPositionEmbeddingConfig {
        &self.config
    }

    /// Reference to the precomputed cosine frequency table.
    pub fn cos_freqs(&self) -> &Array {
        &self.cos_freqs
    }

    /// Reference to the precomputed sine frequency table.
    pub fn sin_freqs(&self) -> &Array {
        &self.sin_freqs
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    /// Maximum sequence length supported by the precomputed tables.
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }
}
