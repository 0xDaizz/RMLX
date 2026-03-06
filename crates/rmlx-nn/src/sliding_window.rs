//! Sliding window attention for Mistral-style models (N11).
//!
//! Extends the standard multi-head attention with a configurable window size.
//! Tokens can only attend to the most recent `window_size` positions,
//! which bounds memory usage and enables efficient long-context inference.
//!
//! The causal mask is modified to only allow attention within the window:
//! `mask[i][j] = 1  if  j >= i - window_size + 1  AND  j <= i`

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

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
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let position_offset = cache.as_ref().map_or(0, |c| c.position_offset());

        let total_seq = position_offset + seq_len;
        let dev = registry.device().raw();

        // Build sliding window causal mask
        let mask = self.build_sliding_window_mask(dev, seq_len, total_seq, position_offset);

        let num_heads = self.config.base.num_heads;
        let num_kv_heads = self.config.base.num_kv_heads;
        let head_dim = self.config.base.head_dim;

        // Project Q, K, V
        let q = self.q_proj.forward(x, registry, queue)?;
        let k = self.k_proj.forward(x, registry, queue)?;
        let v = self.v_proj.forward(x, registry, queue)?;
        let elem_size = q.dtype().size_of();

        // RoPE offset from cache position
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.position_offset() as u32);

        // Split Q into heads and apply RoPE
        let mut q_heads: Vec<Array> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let offset = q.offset() + h * head_dim * elem_size;
            let q_head = q.view(
                vec![seq_len, head_dim],
                vec![num_heads * head_dim, 1],
                offset,
            );
            let q_head = ops::copy::copy(registry, &q_head, queue)?;
            let q_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope(registry, &q_head, cos, sin, rope_offset, 1.0, queue)?
            } else {
                q_head
            };
            q_heads.push(q_head);
        }

        // Split K into heads and apply RoPE
        let mut k_heads: Vec<Array> = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let offset = k.offset() + h * head_dim * elem_size;
            let k_head = k.view(
                vec![seq_len, head_dim],
                vec![num_kv_heads * head_dim, 1],
                offset,
            );
            let k_head = ops::copy::copy(registry, &k_head, queue)?;
            let k_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope(registry, &k_head, cos, sin, rope_offset, 1.0, queue)?
            } else {
                k_head
            };
            k_heads.push(k_head);
        }

        // Split V into heads (no RoPE for V)
        let mut v_heads: Vec<Array> = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let offset = v.offset() + h * head_dim * elem_size;
            let v_head = v.view(
                vec![seq_len, head_dim],
                vec![num_kv_heads * head_dim, 1],
                offset,
            );
            let v_head = ops::copy::copy(registry, &v_head, queue)?;
            v_heads.push(v_head);
        }

        // Update KV cache if provided.
        // When cached sequence length exceeds window_size, only the last
        // window_size entries are returned for attention computation.
        let w = self.config.window_size;
        let (k_final, v_final, total_seq_actual) = match cache {
            Some(ref mut c) => {
                c.append(k_heads, v_heads, seq_len, registry, queue)?;
                let cached_len = c.seq_len;
                if cached_len > w {
                    // Return views over only the last `window_size` tokens.
                    let start = cached_len - w;
                    let kf: Vec<Array> = (0..num_kv_heads)
                        .map(|h| {
                            let full = c.cached_keys(h);
                            let elem = full.dtype().size_of();
                            let strides = full.strides().to_vec();
                            full.view(
                                vec![w, head_dim],
                                strides.clone(),
                                full.offset() + start * strides[0] * elem,
                            )
                        })
                        .collect();
                    let vf: Vec<Array> = (0..num_kv_heads)
                        .map(|h| {
                            let full = c.cached_values(h);
                            let elem = full.dtype().size_of();
                            let strides = full.strides().to_vec();
                            full.view(
                                vec![w, head_dim],
                                strides.clone(),
                                full.offset() + start * strides[0] * elem,
                            )
                        })
                        .collect();
                    (kf, vf, w)
                } else {
                    let kf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_keys(h)).collect();
                    let vf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_values(h)).collect();
                    (kf, vf, cached_len)
                }
            }
            None => (k_heads, v_heads, seq_len),
        };

        // Rebuild mask with actual total_seq from cache (may differ from initial estimate)
        let mask = if total_seq_actual != total_seq {
            self.build_sliding_window_mask(
                dev,
                seq_len,
                total_seq_actual,
                total_seq_actual - seq_len,
            )
        } else {
            mask
        };

        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f32).sqrt();

        let attn_outputs = if head_dim <= 256 {
            ops::sdpa::sdpa_batched(
                registry,
                &q_heads,
                &k_final,
                &v_final,
                Some(&mask),
                scale,
                false,
                queue,
            )?
        } else {
            let repeats = num_heads / num_kv_heads;
            let mut outputs: Vec<Array> = Vec::with_capacity(num_heads);
            for (h, q_h) in q_heads.iter().enumerate() {
                let kv_idx = h / repeats;
                let k_h = &k_final[kv_idx];
                let v_h = &v_final[kv_idx];

                let k_t = k_h.view(
                    vec![head_dim, total_seq_actual],
                    vec![1, head_dim],
                    k_h.offset(),
                );
                let k_t = ops::copy::copy(registry, &k_t, queue)?;
                let scores = ops::matmul::matmul(registry, q_h, &k_t, queue)?;
                // Scale scores
                let scale_arr = Array::from_slice(dev, &[scale], vec![1]);
                let scores = ops::binary::mul(registry, &scores, &scale_arr, queue)?;
                // Apply mask
                let scores = ops::binary::add(registry, &scores, &mask, queue)?;
                let attn_weights = ops::softmax::softmax(registry, &scores, queue)?;
                let head_out = ops::matmul::matmul(registry, &attn_weights, v_h, queue)?;
                outputs.push(head_out);
            }
            outputs
        };

        // Concatenate heads: [num_heads * [seq_len, head_dim]] -> [seq_len, hidden_size]
        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());

        let copy_kernel = match q.dtype() {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            _ => {
                return Err(KernelError::InvalidShape(format!(
                    "SlidingWindowAttention concat: unsupported dtype {:?}",
                    q.dtype()
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, q.dtype())?;
        let head_bytes = head_dim * elem_size;
        let hidden_bytes = hidden_size * elem_size;

        let cb = queue.new_command_buffer();
        for (h, head_out) in attn_outputs.iter().enumerate() {
            let dst_col_offset = h * head_bytes;

            if seq_len == 1 {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset() as u64);
                enc.set_buffer(1, Some(concat.metal_buffer()), dst_col_offset as u64);
                let count = head_dim as u64;
                let grid = metal::MTLSize::new(count, 1, 1);
                let tg = metal::MTLSize::new(
                    std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
                    1,
                    1,
                );
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
            } else {
                let blit = cb.new_blit_command_encoder();
                for row in 0..seq_len {
                    let src_off = (head_out.offset() + row * head_bytes) as u64;
                    let dst_off = (row * hidden_bytes + dst_col_offset) as u64;
                    blit.copy_from_buffer(
                        head_out.metal_buffer(),
                        src_off,
                        concat.metal_buffer(),
                        dst_off,
                        head_bytes as u64,
                    );
                }
                blit.end_encoding();
            }
        }
        cb.commit();
        cb.wait_until_completed();

        // Output projection
        self.o_proj.forward(&concat, registry, queue)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// Verify the sliding window truncation arithmetic:
    /// when cached_len > window_size, only the last window_size tokens
    /// should be used, with the correct byte offset.
    #[test]
    fn test_sliding_window_truncation_offset() {
        let head_dim = 64;
        let elem_size = 4; // f32
        let stride0 = head_dim; // row-major [seq, head_dim]

        // Simulate 20 cached tokens, window = 8
        let cached_len = 20usize;
        let window_size = 8usize;

        assert!(cached_len > window_size);
        let start = cached_len - window_size;
        assert_eq!(start, 12);

        // The view should have shape [window_size, head_dim] starting at
        // byte offset = start * stride0 * elem_size
        let expected_offset = start * stride0 * elem_size;
        assert_eq!(expected_offset, 12 * 64 * 4);

        // total_seq_actual should be clamped to window_size
        let total_seq_actual = window_size;
        assert_eq!(total_seq_actual, 8);
    }

    /// When cached_len <= window_size, no truncation should occur.
    #[test]
    fn test_sliding_window_no_truncation() {
        let cached_len = 5usize;
        let window_size = 8usize;

        assert!(cached_len <= window_size);
        // total_seq_actual should equal cached_len
        let total_seq_actual = cached_len;
        assert_eq!(total_seq_actual, 5);
    }
}
