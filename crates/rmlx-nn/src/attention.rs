//! Multi-head attention with RoPE and GQA support.
//!
//! KV cache uses pre-allocated buffers with O(1) append (no full-history copy).

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::{Linear, LinearConfig};

// ---------------------------------------------------------------------------
// KV Cache — pre-allocated, O(1) append
// ---------------------------------------------------------------------------

/// Per-layer KV cache for incremental decoding.
///
/// Uses a pre-allocated contiguous buffer per KV head with step-based indexing.
/// Appending new tokens writes only the new data — no full-history copy.
pub struct LayerKvCache {
    /// Cached K heads per kv_head: each [max_seq, head_dim], pre-allocated.
    pub keys: Vec<Array>,
    /// Cached V heads per kv_head: each [max_seq, head_dim], pre-allocated.
    pub values: Vec<Array>,
    /// Number of tokens currently cached (position offset for next append).
    pub seq_len: usize,
    /// Maximum sequence length this cache was pre-allocated for.
    max_seq_len: usize,
    /// Number of KV heads (for validation).
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
}

impl LayerKvCache {
    /// Create a new **empty** cache (no pre-allocation).
    /// Compatible with old code that did not pre-allocate.
    pub fn new(num_kv_heads: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
            max_seq_len: 0,
            num_kv_heads,
            head_dim: 0,
        }
    }

    /// Create a pre-allocated cache with room for `max_seq_len` tokens.
    ///
    /// Each KV head gets a single [max_seq_len, head_dim] buffer up-front.
    /// Subsequent `append` calls write into the next slot(s) with no reallocation.
    pub fn preallocated(
        device: &metal::Device,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Self {
        let mut keys = Vec::with_capacity(num_kv_heads);
        let mut values = Vec::with_capacity(num_kv_heads);
        for _ in 0..num_kv_heads {
            keys.push(Array::zeros(device, &[max_seq_len, head_dim], dtype));
            values.push(Array::zeros(device, &[max_seq_len, head_dim], dtype));
        }
        Self {
            keys,
            values,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
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

    /// Current cached sequence length (also the RoPE position offset).
    pub fn position_offset(&self) -> usize {
        self.seq_len
    }

    /// Append new K, V heads from the current step.
    ///
    /// For pre-allocated caches, this copies only `new_tokens` rows into the
    /// next available slots — O(new_tokens), not O(total_cached).
    ///
    /// For legacy (non-pre-allocated) caches, falls back to concat (as before).
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

        if self.max_seq_len > 0 {
            // Pre-allocated path: write into slot [seq_len .. seq_len + new_tokens]
            if self.seq_len + new_tokens > self.max_seq_len {
                return Err(KernelError::InvalidShape(format!(
                    "LayerKvCache: overflow: {} cached + {} new > {} max",
                    self.seq_len, new_tokens, self.max_seq_len
                )));
            }
            self.append_preallocated(new_keys, new_values, new_tokens, registry, queue)?;
        } else if self.keys.is_empty() {
            // Legacy path, first append
            self.keys = new_keys;
            self.values = new_values;
            if let Some(k) = self.keys.first() {
                self.head_dim = k.shape()[1];
            }
        } else {
            // Legacy path, concat
            for (i, new_k) in new_keys.into_iter().enumerate() {
                self.keys[i] = concat_seq_dim(registry, &self.keys[i], &new_k, queue)?;
            }
            for (i, new_v) in new_values.into_iter().enumerate() {
                self.values[i] = concat_seq_dim(registry, &self.values[i], &new_v, queue)?;
            }
        }
        self.seq_len += new_tokens;
        Ok(())
    }

    /// O(1)-per-token append into pre-allocated buffers.
    fn append_preallocated(
        &self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        let elem_size = self.keys[0].dtype().size_of();
        let copy_kernel = match self.keys[0].dtype() {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "LayerKvCache: unsupported dtype {:?}",
                    other
                )))
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, self.keys[0].dtype())?;
        let count = (new_tokens * self.head_dim) as u64;
        if count == 0 {
            return Ok(());
        }

        // Single command buffer for all heads
        let cb = queue.new_command_buffer();
        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        for i in 0..self.num_kv_heads {
            // Copy new keys into slot
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset() as u64);
            enc.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                (self.keys[i].offset() + dst_row_offset) as u64,
            );
            let grid = metal::MTLSize::new(count, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            // Copy new values into slot
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(new_values[i].metal_buffer()), new_values[i].offset() as u64);
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                (self.values[i].offset() + dst_row_offset) as u64,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    /// Get a view of cached keys for head `h`, shape [seq_len, head_dim].
    pub fn cached_keys(&self, head: usize) -> Array {
        let a = &self.keys[head];
        // Return a view of only the filled portion [0..seq_len, :]
        a.view(
            vec![self.seq_len, self.head_dim],
            a.strides().to_vec(),
            a.offset(),
        )
    }

    /// Get a view of cached values for head `h`, shape [seq_len, head_dim].
    pub fn cached_values(&self, head: usize) -> Array {
        let a = &self.values[head];
        a.view(
            vec![self.seq_len, self.head_dim],
            a.strides().to_vec(),
            a.offset(),
        )
    }
}

/// Concatenate two 2D arrays along dimension 0 (seq dimension).
/// `a`: [seq_a, dim], `b`: [seq_b, dim] -> [seq_a + seq_b, dim]
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

// ---------------------------------------------------------------------------
// Attention config and module
// ---------------------------------------------------------------------------

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
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables (optional)
    /// `mask`: additive causal mask (optional)
    /// `cache`: optional pre-allocated KV cache for incremental decoding
    ///
    /// Returns: [seq_len, hidden_size]
    #[allow(clippy::too_many_arguments)]
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

        // Project Q, K, V
        let q = self.q_proj.forward(x, registry, queue)?;
        let k = self.k_proj.forward(x, registry, queue)?;
        let v = self.v_proj.forward(x, registry, queue)?;

        let expected_q_width = num_heads * head_dim;
        let expected_kv_width = num_kv_heads * head_dim;
        if q.shape() != [seq_len, expected_q_width] {
            return Err(KernelError::InvalidShape(format!(
                "Q projection shape {:?}, expected [{}, {}]",
                q.shape(), seq_len, expected_q_width
            )));
        }
        if k.shape() != [seq_len, expected_kv_width] {
            return Err(KernelError::InvalidShape(format!(
                "K projection shape {:?}, expected [{}, {}]",
                k.shape(), seq_len, expected_kv_width
            )));
        }

        let dev = registry.device().raw();
        let elem_size = q.dtype().size_of();

        // RoPE offset from cache position
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        // Split into heads and apply RoPE
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

        // Append to KV cache (O(1) with pre-allocated cache)
        let (k_final, v_final, total_seq) = match cache {
            Some(ref mut c) => {
                c.append(k_heads, v_heads, seq_len, registry, queue)?;
                let kf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_keys(h)).collect();
                let vf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_values(h)).collect();
                let ts = c.seq_len;
                (kf, vf, ts)
            }
            None => (k_heads, v_heads, seq_len),
        };

        // Scaled dot-product attention per query head
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Try fused SDPA path (Flash Attention style — no intermediate score matrix)
        let attn_outputs = if head_dim <= 128 {
            // Fused path: single kernel per head, Q@K^T + scale + mask + softmax + @V
            ops::sdpa::sdpa_batched(
                registry, &q_heads, &k_final, &v_final, mask, scale, queue,
            )?
        } else {
            // Unfused fallback for very large head dims
            let mut outputs: Vec<Array> = Vec::with_capacity(num_heads);
            for (h, q_h) in q_heads.iter().enumerate() {
                let kv_idx = h / repeats;
                let k_h = &k_final[kv_idx];
                let v_h = &v_final[kv_idx];

                let k_t = k_h.view(vec![head_dim, total_seq], vec![1, head_dim], k_h.offset());
                let k_t = ops::copy::copy(registry, &k_t, queue)?;
                let scores = ops::matmul::matmul(registry, q_h, &k_t, queue)?;
                let scores = scale_scores(&scores, scale, registry, queue)?;
                let scores = if let Some(m) = mask {
                    ops::binary::add(registry, &scores, m, queue)?
                } else {
                    scores
                };
                let attn_weights = ops::softmax::softmax(registry, &scores, queue)?;
                let head_out = ops::matmul::matmul(registry, &attn_weights, v_h, queue)?;
                outputs.push(head_out);
            }
            outputs
        };

        // Concatenate heads — batch all copies in one command buffer
        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());

        let copy_kernel = match q.dtype() {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
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

                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), src_offset as u64);
                enc.set_buffer(1, Some(concat.metal_buffer()), dst_offset as u64);
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

        // Output projection
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

/// Scale attention scores by a scalar factor.
///
/// Tries broadcasting `scalar * matrix` first. If binary ops don't yet support
/// broadcasting, falls back to a manual element-wise scale via a filled array.
fn scale_scores(
    scores: &Array,
    scale: f32,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let dev = registry.device().raw();
    // Try scalar broadcast: create [1] scalar and rely on broadcasting
    let scale_arr = Array::from_slice(dev, &[scale], vec![1]);
    match ops::binary::mul(registry, scores, &scale_arr, queue) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Fallback: fill a full-sized array with the scale factor
            let numel = scores.numel();
            let data = vec![scale; numel];
            let scale_full = Array::from_slice(dev, &data, scores.shape().to_vec());
            ops::binary::mul(registry, scores, &scale_full, queue)
        }
    }
}
