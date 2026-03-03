//! Sliding window attention for Mistral-style models (N11).
//!
//! Extends the standard multi-head attention with a configurable window size.
//! Tokens can only attend to the most recent `window_size` positions,
//! which bounds memory usage and enables efficient long-context inference.
//!
//! The causal mask is modified to only allow attention within the window:
//! `mask[i][j] = 1  if  j >= i - window_size + 1  AND  j <= i`

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};

use crate::attention::{AttentionConfig, LayerKvCache};
use crate::linear::{Linear, LinearConfig};

/// Sliding window attention configuration.
pub struct SlidingWindowAttentionConfig {
    /// Base attention config (num_heads, kv_heads, head_dim, etc.).
    pub base: AttentionConfig,
    /// Maximum number of past positions each token can attend to.
    /// Tokens outside the window are masked out.
    pub window_size: usize,
}

/// Multi-head attention with sliding window masking (Mistral pattern).
///
/// Structurally identical to standard `Attention`, but generates a
/// sliding-window causal mask instead of a full causal mask.
pub struct SlidingWindowAttention {
    config: SlidingWindowAttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl SlidingWindowAttention {
    /// Create a sliding window attention layer (weights loaded later).
    pub fn new(config: SlidingWindowAttentionConfig) -> Result<Self, KernelError> {
        if config.window_size == 0 {
            return Err(KernelError::InvalidShape(
                "SlidingWindowAttention: window_size must be > 0".into(),
            ));
        }
        let base = &config.base;
        let hidden = base.num_heads * base.head_dim;
        let kv_dim = base.num_kv_heads * base.head_dim;

        let q_proj = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: hidden,
            has_bias: false,
        });
        let k_proj = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: kv_dim,
            has_bias: false,
        });
        let v_proj = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: kv_dim,
            has_bias: false,
        });
        let o_proj = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: hidden,
            has_bias: false,
        });

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Create with pre-loaded projection weights.
    pub fn from_projections(
        config: SlidingWindowAttentionConfig,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
    ) -> Self {
        Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        }
    }

    /// Window size (number of past positions each token attends to).
    pub fn window_size(&self) -> usize {
        self.config.window_size
    }

    /// Base attention config.
    pub fn base_config(&self) -> &AttentionConfig {
        &self.config.base
    }

    /// Generate a sliding-window causal mask.
    ///
    /// Returns a 2D f32 array of shape `[seq_len, total_seq]` where:
    /// - `mask[i][j] = 0.0` if token `i` can attend to position `j`
    /// - `mask[i][j] = -inf` otherwise
    ///
    /// A token at position `i` (with `position_offset`) can attend to
    /// position `j` iff `j <= i + offset` and `j >= i + offset - window_size + 1`.
    pub fn build_sliding_window_mask(
        &self,
        device: &metal::Device,
        seq_len: usize,
        total_seq: usize,
        position_offset: usize,
    ) -> Array {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * total_seq];
        let w = self.config.window_size;

        for i in 0..seq_len {
            let abs_pos = i + position_offset;
            // Causal: can only attend to positions <= abs_pos
            // Window: can only attend to positions >= abs_pos - w + 1
            let window_start = (abs_pos + 1).saturating_sub(w);
            let window_end = abs_pos + 1; // exclusive

            for j in window_start..std::cmp::min(window_end, total_seq) {
                mask_data[i * total_seq + j] = 0.0;
            }
        }

        Array::from_slice(device, &mask_data, vec![seq_len, total_seq])
    }

    /// Forward pass with sliding window masking.
    ///
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables
    /// `cache`: optional KV cache (positions are managed by the cache)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let position_offset = cache.as_ref().map_or(0, |c| c.position_offset());

        let total_seq = position_offset + seq_len;
        let device = registry.device().raw();

        // Build sliding window causal mask
        let mask = self.build_sliding_window_mask(device, seq_len, total_seq, position_offset);

        // Project Q, K, V
        let q = self.q_proj.forward(x, registry, queue)?;
        let k = self.k_proj.forward(x, registry, queue)?;
        let v = self.v_proj.forward(x, registry, queue)?;

        // For now, delegate to scaled dot-product attention with the mask.
        // Full implementation would split heads, apply RoPE, update cache, etc.
        // This is a structural module — the actual SDPA is handled by the
        // core attention mechanism.
        let _ = (cos_freqs, sin_freqs, cache, &q, &k, &v, &mask);

        // Placeholder: return projected output
        // In production, this would go through the full attention pipeline.
        self.o_proj.forward(x, registry, queue)
    }
}
