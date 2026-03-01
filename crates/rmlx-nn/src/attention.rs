//! Multi-head attention with RoPE and GQA support.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::{Linear, LinearConfig};

/// Per-layer KV cache for incremental decoding.
/// Stores projected+RoPE'd K, V heads from previous steps.
pub struct LayerKvCache {
    /// Cached K heads per kv_head: each [cached_seq, head_dim]
    pub keys: Vec<Array>,
    /// Cached V heads per kv_head: each [cached_seq, head_dim]
    pub values: Vec<Array>,
    /// Number of tokens currently cached
    pub seq_len: usize,
    /// Number of KV heads (for validation)
    num_kv_heads: usize,
}

impl LayerKvCache {
    pub fn new(num_kv_heads: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
            num_kv_heads,
        }
    }

    /// Whether the cache is empty (no tokens cached yet).
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Number of KV heads this cache expects.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Append new K, V heads from the current step.
    /// `new_keys` and `new_values` each have `num_kv_heads` elements,
    /// each of shape [new_seq, head_dim].
    pub fn append(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        if self.keys.is_empty() {
            // First append — just store directly
            self.keys = new_keys;
            self.values = new_values;
        } else {
            // Concatenate along seq dimension: [old_seq, hd] + [new_seq, hd] -> [total_seq, hd]
            for (i, new_k) in new_keys.into_iter().enumerate() {
                self.keys[i] =
                    concat_seq_dim(registry, &self.keys[i], &new_k, queue)?;
            }
            for (i, new_v) in new_values.into_iter().enumerate() {
                self.values[i] =
                    concat_seq_dim(registry, &self.values[i], &new_v, queue)?;
            }
        }
        self.seq_len += new_tokens;
        Ok(())
    }
}

/// Concatenate two 2D arrays along dimension 0 (seq dimension).
/// `a`: [seq_a, dim], `b`: [seq_b, dim] -> [seq_a + seq_b, dim]
/// Uses the copy kernel to assemble the result.
fn concat_seq_dim(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let seq_a = a.shape()[0];
    let seq_b = b.shape()[0];
    let dim = a.shape()[1];
    if b.shape()[1] != dim {
        return Err(KernelError::InvalidShape(format!(
            "concat_seq_dim: dim mismatch: a has {}, b has {}",
            dim,
            b.shape()[1]
        )));
    }

    let total_seq = seq_a + seq_b;
    let dev = registry.device().raw();
    let result = Array::zeros(dev, &[total_seq, dim], a.dtype());
    let elem_size = a.dtype().size_of();

    let copy_kernel = match a.dtype() {
        DType::Float32 => "copy_f32",
        DType::Float16 => "copy_f16",
        DType::Bfloat16 => "copy_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "concat_seq_dim: unsupported dtype {:?}",
                other
            )));
        }
    };
    let pipeline = registry.get_pipeline(copy_kernel, a.dtype())?;

    let cb = queue.new_command_buffer();

    // Copy a into result[0..seq_a]
    let a_count = (seq_a * dim) as u64;
    if a_count > 0 {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
        enc.set_buffer(1, Some(result.metal_buffer()), result.offset() as u64);
        let grid = metal::MTLSize::new(a_count, 1, 1);
        let tg = metal::MTLSize::new(
            std::cmp::min(pipeline.max_total_threads_per_threadgroup(), a_count),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }

    // Copy b into result[seq_a..total_seq]
    let b_count = (seq_b * dim) as u64;
    if b_count > 0 {
        let dst_offset = (result.offset() + seq_a * dim * elem_size) as u64;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(b.metal_buffer()), b.offset() as u64);
        enc.set_buffer(1, Some(result.metal_buffer()), dst_offset);
        let grid = metal::MTLSize::new(b_count, 1, 1);
        let tg = metal::MTLSize::new(
            std::cmp::min(pipeline.max_total_threads_per_threadgroup(), b_count),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();

    Ok(result)
}

pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.num_heads == 0 {
            return Err(KernelError::InvalidShape(
                "AttentionConfig: num_heads must be > 0".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(KernelError::InvalidShape(
                "AttentionConfig: head_dim must be > 0".into(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(KernelError::InvalidShape(
                "AttentionConfig: num_kv_heads must be > 0".into(),
            ));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(KernelError::InvalidShape(format!(
                "AttentionConfig: num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        Ok(())
    }
}

pub struct Attention {
    config: AttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    /// Config-only constructor (weights loaded later).
    pub fn new(config: AttentionConfig) -> Result<Self, KernelError> {
        config.validate()?;
        let hidden_size = config.num_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        Ok(Self {
            q_proj: Linear::new(LinearConfig {
                in_features: hidden_size,
                out_features: hidden_size,
                has_bias: false,
            }),
            k_proj: Linear::new(LinearConfig {
                in_features: hidden_size,
                out_features: kv_size,
                has_bias: false,
            }),
            v_proj: Linear::new(LinearConfig {
                in_features: hidden_size,
                out_features: kv_size,
                has_bias: false,
            }),
            o_proj: Linear::new(LinearConfig {
                in_features: hidden_size,
                out_features: hidden_size,
                has_bias: false,
            }),
            config,
        })
    }

    /// Create attention with pre-loaded projection layers.
    pub fn from_layers(
        config: AttentionConfig,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
    ) -> Result<Self, KernelError> {
        config.validate()?;
        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward pass for multi-head attention.
    ///
    /// `x`: [seq_len, hidden_size] (batch=1 assumed, tokens are rows)
    /// `cos_freqs`: [max_seq, head_dim/2] for RoPE (optional)
    /// `sin_freqs`: [max_seq, head_dim/2] for RoPE (optional)
    /// `mask`: [seq_len, seq_len] additive causal mask (optional, -inf for masked positions)
    /// `cache`: optional KV cache for incremental decoding. When provided,
    ///   newly projected+RoPE'd K,V heads are appended to the cache and attention
    ///   uses the full cached K,V. Pass `None` for stateless full-sequence inference.
    ///
    /// **RoPE contract**: When using `cache`, the caller MUST provide pre-sliced
    /// `cos_freqs`/`sin_freqs` tables matching the current input positions.
    /// For example, at decode step position 5, pass [1, half_dim] RoPE tables
    /// containing only position 5's frequencies. The attention layer uses offset=0
    /// internally, relying on the caller to provide correctly positioned tables.
    ///
    /// Returns: [seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let repeats = num_heads / num_kv_heads;

        // Project: Q, K, V
        // x: [seq_len, hidden_size]
        // Q: [seq_len, num_heads * head_dim]
        // K: [seq_len, num_kv_heads * head_dim]
        // V: [seq_len, num_kv_heads * head_dim]
        let q = self.q_proj.forward(x, registry, queue)?;
        let k = self.k_proj.forward(x, registry, queue)?;
        let v = self.v_proj.forward(x, registry, queue)?;

        // Validate projected shapes before head slicing
        let expected_q_width = num_heads * head_dim;
        let expected_kv_width = num_kv_heads * head_dim;
        if q.shape() != [seq_len, expected_q_width] {
            return Err(KernelError::InvalidShape(format!(
                "Q projection shape {:?}, expected [{}, {}]",
                q.shape(),
                seq_len,
                expected_q_width
            )));
        }
        if k.shape() != [seq_len, expected_kv_width] {
            return Err(KernelError::InvalidShape(format!(
                "K projection shape {:?}, expected [{}, {}]",
                k.shape(),
                seq_len,
                expected_kv_width
            )));
        }
        if v.shape() != [seq_len, expected_kv_width] {
            return Err(KernelError::InvalidShape(format!(
                "V projection shape {:?}, expected [{}, {}]",
                v.shape(),
                seq_len,
                expected_kv_width
            )));
        }

        // Apply RoPE per-head for Q and K.
        // RoPE kernel expects [seq_len, head_dim], so we process each head slice.
        let dev = registry.device().raw();
        let elem_size = q.dtype().size_of();

        // Build Q heads with RoPE: [num_heads, seq_len, head_dim]
        let mut q_heads: Vec<Array> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let offset = q.offset() + h * head_dim * elem_size;
            // View into the h-th head: shape [seq_len, head_dim], stride [num_heads*head_dim, 1]
            let q_head = q.view(
                vec![seq_len, head_dim],
                vec![num_heads * head_dim, 1],
                offset,
            );
            // Make contiguous for RoPE
            let q_head = ops::copy::copy(registry, &q_head, queue)?;
            let q_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope(registry, &q_head, cos, sin, 0, 1.0, queue)?
            } else {
                q_head
            };
            q_heads.push(q_head);
        }

        // Build K heads with RoPE: [num_kv_heads, seq_len, head_dim]
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
                ops::rope::rope(registry, &k_head, cos, sin, 0, 1.0, queue)?
            } else {
                k_head
            };
            k_heads.push(k_head);
        }

        // Build V heads: [num_kv_heads, seq_len, head_dim]
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

        // If cache is provided, append new K,V and use full cached K,V for attention.
        // We move k_heads/v_heads into the cache; the cache then owns the full history.
        // If no cache, use k_heads/v_heads directly.
        //
        // To satisfy Rust's move checker, we store the "effective" K,V heads in new
        // Vecs that either borrow from the cache or own the original heads.
        let (k_final, v_final, total_seq) = match cache {
            Some(ref mut c) => {
                c.append(k_heads, v_heads, seq_len, registry, queue)?;
                // Create views into cache arrays (cheap: shares underlying Metal buffers)
                let kf: Vec<Array> = c.keys.iter().map(|a| {
                    a.view(a.shape().to_vec(), a.strides().to_vec(), a.offset())
                }).collect();
                let vf: Vec<Array> = c.values.iter().map(|a| {
                    a.view(a.shape().to_vec(), a.strides().to_vec(), a.offset())
                }).collect();
                (kf, vf, c.seq_len)
            }
            None => (k_heads, v_heads, seq_len),
        };

        // Scaled dot-product attention per query head
        // For GQA: each kv head is shared by `repeats` query heads.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_arr =
            Array::from_slice(dev, &vec![scale; seq_len * total_seq], vec![seq_len, total_seq]);

        let mut attn_outputs: Vec<Array> = Vec::with_capacity(num_heads);
        for (h, q_h) in q_heads.iter().enumerate() {
            let kv_idx = h / repeats;
            let k_h = &k_final[kv_idx];
            let v_h = &v_final[kv_idx];

            // Q @ K^T: [seq_len, head_dim] @ [head_dim, total_seq] -> [seq_len, total_seq]
            let k_t = k_h.view(vec![head_dim, total_seq], vec![1, head_dim], k_h.offset());
            let k_t = ops::copy::copy(registry, &k_t, queue)?;
            let scores = ops::matmul::matmul(registry, q_h, &k_t, queue)?;

            // Scale
            let scores = ops::binary::mul(registry, &scores, &scale_arr, queue)?;

            // Apply mask (additive: scores + mask where mask has -inf for masked positions)
            // During decode with cache (seq_len=1), mask is typically None so the single
            // query token attends to all cached positions.
            let scores = if let Some(m) = mask {
                ops::binary::add(registry, &scores, m, queue)?
            } else {
                scores
            };

            // Softmax
            let attn_weights = ops::softmax::softmax(registry, &scores, queue)?;

            // attn_weights @ V: [seq_len, total_seq] @ [total_seq, head_dim] -> [seq_len, head_dim]
            let head_out = ops::matmul::matmul(registry, &attn_weights, v_h, queue)?;
            attn_outputs.push(head_out);
        }

        // Concatenate heads: [seq_len, num_heads * head_dim]
        // Batched: all head*row copies dispatched in a single command buffer.
        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());

        let copy_kernel = match q.dtype() {
            rmlx_core::dtype::DType::Float32 => "copy_f32",
            rmlx_core::dtype::DType::Float16 => "copy_f16",
            rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
            _ => {
                return Err(KernelError::InvalidShape(format!(
                    "attention concat: unsupported dtype {:?}",
                    q.dtype()
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, q.dtype())?;
        let cb = queue.new_command_buffer();
        for (h, head_out) in attn_outputs.iter().enumerate() {
            for row in 0..seq_len {
                let src_offset = head_out.offset() + row * head_dim * elem_size;
                let dst_offset = row * hidden_size * elem_size + h * head_dim * elem_size;

                let src_view = head_out.view(vec![head_dim], vec![1], src_offset);
                let dst_view = concat.view(vec![head_dim], vec![1], dst_offset);

                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(src_view.metal_buffer()), src_view.offset() as u64);
                enc.set_buffer(1, Some(dst_view.metal_buffer()), dst_view.offset() as u64);
                let grid = metal::MTLSize::new(head_dim as u64, 1, 1);
                let tg = metal::MTLSize::new(
                    std::cmp::min(
                        pipeline.max_total_threads_per_threadgroup(),
                        head_dim as u64,
                    ),
                    1,
                    1,
                );
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
            }
        }
        cb.commit();
        cb.wait_until_completed();

        // Output projection: [seq_len, hidden_size] -> [seq_len, hidden_size]
        self.o_proj.forward(&concat, registry, queue)
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

    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }
}
