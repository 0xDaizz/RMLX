//! Multi-head attention with RoPE and GQA support.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::{Linear, LinearConfig};

pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl AttentionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if self.num_kv_heads > self.num_heads {
            return Err("num_kv_heads > num_heads".into());
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err("num_heads must be divisible by num_kv_heads".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
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
    pub fn new(config: AttentionConfig) -> Self {
        let hidden_size = config.num_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        Self {
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
        }
    }

    /// Create attention with pre-loaded projection layers.
    pub fn from_layers(
        config: AttentionConfig,
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

    /// Forward pass for multi-head attention.
    ///
    /// `x`: [seq_len, hidden_size] (batch=1 assumed, tokens are rows)
    /// `cos_freqs`: [max_seq, head_dim/2] for RoPE (optional)
    /// `sin_freqs`: [max_seq, head_dim/2] for RoPE (optional)
    /// `mask`: [seq_len, seq_len] additive causal mask (optional, -inf for masked positions)
    ///
    /// Returns: [seq_len, hidden_size]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
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

        // Scaled dot-product attention per query head
        // For GQA: each kv head is shared by `repeats` query heads.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_arr =
            Array::from_slice(dev, &vec![scale; seq_len * seq_len], vec![seq_len, seq_len]);

        let mut attn_outputs: Vec<Array> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let kv_idx = h / repeats;
            let q_h = &q_heads[h];
            let k_h = &k_heads[kv_idx];
            let v_h = &v_heads[kv_idx];

            // Q @ K^T: [seq_len, head_dim] @ [head_dim, seq_len] -> [seq_len, seq_len]
            let k_t = k_h.view(vec![head_dim, seq_len], vec![1, head_dim], k_h.offset());
            let k_t = ops::copy::copy(registry, &k_t, queue)?;
            let scores = ops::matmul::matmul(registry, q_h, &k_t, queue)?;

            // Scale
            let scores = ops::binary::mul(registry, &scores, &scale_arr, queue)?;

            // Apply mask (additive: scores + mask where mask has -inf for masked positions)
            let scores = if let Some(m) = mask {
                ops::binary::add(registry, &scores, m, queue)?
            } else {
                scores
            };

            // Softmax
            let attn_weights = ops::softmax::softmax(registry, &scores, queue)?;

            // attn_weights @ V: [seq_len, seq_len] @ [seq_len, head_dim] -> [seq_len, head_dim]
            let head_out = ops::matmul::matmul(registry, &attn_weights, v_h, queue)?;
            attn_outputs.push(head_out);
        }

        // Concatenate heads: [seq_len, num_heads * head_dim]
        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());
        for (h, head_out) in attn_outputs.iter().enumerate() {
            // Copy head_out into the correct column slice of concat
            for row in 0..seq_len {
                let src_offset = head_out.offset() + row * head_dim * elem_size;
                let dst_offset = row * hidden_size * elem_size + h * head_dim * elem_size;

                let src_view = head_out.view(vec![head_dim], vec![1], src_offset);
                let dst_view = concat.view(vec![head_dim], vec![1], dst_offset);

                let copy_kernel = match q.dtype() {
                    rmlx_core::dtype::DType::Float32 => "copy_f32",
                    rmlx_core::dtype::DType::Float16 => "copy_f16",
                    rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
                    _ => unreachable!(),
                };
                let pipeline = registry.get_pipeline(copy_kernel, q.dtype())?;
                let cb = queue.new_command_buffer();
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
                cb.commit();
                cb.wait_until_completed();
            }
        }

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
