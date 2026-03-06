//! Multi-head Latent Attention (MLA) for DeepSeek-V3 (N12).
//!
//! MLA compresses KV representations into a low-rank latent space before
//! caching, dramatically reducing KV cache memory. Instead of caching
//! per-head K and V tensors, a single compressed latent vector `c_kv` is
//! cached per token position.
//!
//! Architecture (from DeepSeek-V2/V3 papers):
//!
//! ```text
//! Input x: [seq, hidden]
//!
//! -- KV compression --
//! c_kv = W_dkv(x)                   [seq, kv_lora_rank]      (down-project)
//! k_nope = W_uk(c_kv)               [seq, num_heads * (head_dim - rope_head_dim)]
//! v = W_uv(c_kv)                    [seq, num_heads * v_head_dim]
//! k_rope_input = W_kr(x)            [seq, rope_head_dim]     (decoupled RoPE)
//! k = concat(k_nope, RoPE(k_rope))  per-head
//!
//! -- Q compression --
//! c_q = W_dq(x)                     [seq, q_lora_rank]       (down-project)
//! q_nope = W_uq(c_q)               [seq, num_heads * (head_dim - rope_head_dim)]
//! q_rope_input = slice/proj(c_q)    [seq, rope_head_dim]
//! q = concat(q_nope, RoPE(q_rope))  per-head
//!
//! -- Attention --
//! Standard scaled dot-product attention on (Q, K, V).
//! ```
//!
//! The key benefit: only `c_kv` (size `kv_lora_rank`) is cached per token
//! instead of `num_kv_heads * 2 * head_dim` values. For DeepSeek-V3 this
//! is 512 floats vs. 128*128*2 = 32768 floats per token.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::{Linear, LinearConfig};

/// Configuration for Multi-head Latent Attention (MLA).
pub struct MlaConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Full head dimension (nope + rope portions).
    pub head_dim: usize,
    /// Value head dimension (can differ from head_dim in MLA).
    pub v_head_dim: usize,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// KV compression rank (e.g. 512 for DeepSeek-V3).
    pub kv_lora_rank: usize,
    /// Query compression rank (e.g. 1536 for DeepSeek-V3).
    pub q_lora_rank: usize,
    /// Dimension of the decoupled RoPE portion of each head.
    pub rope_head_dim: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl MlaConfig {
    /// Validate the MLA configuration.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.num_heads == 0 {
            return Err(KernelError::InvalidShape(
                "MlaConfig: num_heads must be > 0".into(),
            ));
        }
        if self.head_dim == 0 || self.v_head_dim == 0 {
            return Err(KernelError::InvalidShape(
                "MlaConfig: head_dim and v_head_dim must be > 0".into(),
            ));
        }
        if self.kv_lora_rank == 0 {
            return Err(KernelError::InvalidShape(
                "MlaConfig: kv_lora_rank must be > 0".into(),
            ));
        }
        if self.q_lora_rank == 0 {
            return Err(KernelError::InvalidShape(
                "MlaConfig: q_lora_rank must be > 0".into(),
            ));
        }
        if self.rope_head_dim > self.head_dim {
            return Err(KernelError::InvalidShape(format!(
                "MlaConfig: rope_head_dim ({}) must be <= head_dim ({})",
                self.rope_head_dim, self.head_dim
            )));
        }
        Ok(())
    }

    /// Non-RoPE (nope) dimension per head: head_dim - rope_head_dim.
    pub fn nope_head_dim(&self) -> usize {
        self.head_dim - self.rope_head_dim
    }
}

/// Per-token latent KV cache for MLA.
///
/// Instead of caching full K and V per head, we cache only the compressed
/// latent `c_kv` and the decoupled RoPE key component. This reduces
/// KV cache memory by a factor of ~(2 * num_heads * head_dim / kv_lora_rank).
pub struct MlaKvCache {
    /// Compressed KV latent: [max_seq, kv_lora_rank], pre-allocated.
    c_kv_cache: Array,
    /// Decoupled RoPE key component (already rotated): [max_seq, rope_head_dim].
    k_rope_cache: Array,
    /// Number of tokens currently cached.
    seq_len: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// KV compression rank.
    kv_lora_rank: usize,
    /// RoPE head dimension.
    rope_head_dim: usize,
}

impl MlaKvCache {
    /// Create a pre-allocated MLA KV cache.
    pub fn new(
        device: &metal::Device,
        max_seq_len: usize,
        kv_lora_rank: usize,
        rope_head_dim: usize,
        dtype: rmlx_core::dtype::DType,
    ) -> Self {
        let c_kv_cache = Array::zeros(device, &[max_seq_len, kv_lora_rank], dtype);
        let k_rope_cache = Array::zeros(device, &[max_seq_len, rope_head_dim], dtype);
        Self {
            c_kv_cache,
            k_rope_cache,
            seq_len: 0,
            max_seq_len,
            kv_lora_rank,
            rope_head_dim,
        }
    }

    /// Current number of cached tokens.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Position offset for RoPE (same as seq_len).
    pub fn position_offset(&self) -> usize {
        self.seq_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get a view of the cached c_kv latents: [seq_len, kv_lora_rank].
    pub fn cached_c_kv(&self) -> Array {
        self.c_kv_cache.view(
            vec![self.seq_len, self.kv_lora_rank],
            self.c_kv_cache.strides().to_vec(),
            self.c_kv_cache.offset(),
        )
    }

    /// Get a view of the cached RoPE keys: [seq_len, rope_head_dim].
    pub fn cached_k_rope(&self) -> Array {
        self.k_rope_cache.view(
            vec![self.seq_len, self.rope_head_dim],
            self.k_rope_cache.strides().to_vec(),
            self.k_rope_cache.offset(),
        )
    }

    /// Advance the sequence length after appending tokens.
    pub fn advance(&mut self, new_tokens: usize) -> Result<(), KernelError> {
        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "MlaKvCache: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }
        self.seq_len += new_tokens;
        Ok(())
    }
}

/// Multi-head Latent Attention (MLA) module.
///
/// Implements the DeepSeek-V2/V3 MLA mechanism where KV representations
/// are compressed into a shared low-rank latent space before caching.
pub struct Mla {
    config: MlaConfig,
    /// KV down-projection: [hidden, kv_lora_rank]
    w_dkv: Linear,
    /// K up-projection from latent: [kv_lora_rank, num_heads * nope_head_dim]
    w_uk: Linear,
    /// V up-projection from latent: [kv_lora_rank, num_heads * v_head_dim]
    w_uv: Linear,
    /// Decoupled K RoPE projection: [hidden, rope_head_dim]
    w_kr: Linear,
    /// Q down-projection: [hidden, q_lora_rank]
    w_dq: Linear,
    /// Q up-projection: [q_lora_rank, num_heads * head_dim]
    w_uq: Linear,
    /// Output projection: [num_heads * v_head_dim, hidden]
    o_proj: Linear,
}

impl Mla {
    /// Create an MLA module (weights loaded later via weight loading).
    pub fn new(config: MlaConfig) -> Result<Self, KernelError> {
        config.validate()?;

        let hidden = config.hidden_size;
        let nope_dim = config.nope_head_dim();
        let kv_rank = config.kv_lora_rank;
        let q_rank = config.q_lora_rank;
        let num_heads = config.num_heads;
        let rope_dim = config.rope_head_dim;
        let v_dim = config.v_head_dim;

        let w_dkv = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: kv_rank,
            has_bias: false,
        });
        let w_uk = Linear::new(LinearConfig {
            in_features: kv_rank,
            out_features: num_heads * nope_dim,
            has_bias: false,
        });
        let w_uv = Linear::new(LinearConfig {
            in_features: kv_rank,
            out_features: num_heads * v_dim,
            has_bias: false,
        });
        let w_kr = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: rope_dim,
            has_bias: false,
        });
        let w_dq = Linear::new(LinearConfig {
            in_features: hidden,
            out_features: q_rank,
            has_bias: false,
        });
        let w_uq = Linear::new(LinearConfig {
            in_features: q_rank,
            out_features: num_heads * config.head_dim,
            has_bias: false,
        });
        let o_proj = Linear::new(LinearConfig {
            in_features: num_heads * v_dim,
            out_features: hidden,
            has_bias: false,
        });

        Ok(Self {
            config,
            w_dkv,
            w_uk,
            w_uv,
            w_kr,
            w_dq,
            w_uq,
            o_proj,
        })
    }

    /// Create an MLA module with pre-loaded projection layers.
    #[allow(clippy::too_many_arguments)]
    pub fn from_projections(
        config: MlaConfig,
        w_dkv: Linear,
        w_uk: Linear,
        w_uv: Linear,
        w_kr: Linear,
        w_dq: Linear,
        w_uq: Linear,
        o_proj: Linear,
    ) -> Result<Self, KernelError> {
        config.validate()?;
        Ok(Self {
            config,
            w_dkv,
            w_uk,
            w_uv,
            w_kr,
            w_dq,
            w_uq,
            o_proj,
        })
    }

    /// Reference to the MLA configuration.
    pub fn config(&self) -> &MlaConfig {
        &self.config
    }

    /// Compute the compressed KV latent from input.
    ///
    /// `x`: [seq_len, hidden_size]
    /// Returns `c_kv`: [seq_len, kv_lora_rank]
    pub fn compress_kv(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        self.w_dkv.forward(x, registry, queue)
    }

    /// Compute the decoupled RoPE key from input.
    ///
    /// `x`: [seq_len, hidden_size]
    /// Returns: [seq_len, rope_head_dim]
    pub fn compute_k_rope(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        self.w_kr.forward(x, registry, queue)
    }

    /// Expand compressed KV latent into full K (nope portion) and V.
    ///
    /// `c_kv`: [seq_len, kv_lora_rank]
    /// Returns (k_nope, v):
    ///   - k_nope: [seq_len, num_heads * nope_head_dim]
    ///   - v: [seq_len, num_heads * v_head_dim]
    pub fn expand_kv(
        &self,
        c_kv: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(Array, Array), KernelError> {
        let k_nope = self.w_uk.forward(c_kv, registry, queue)?;
        let v = self.w_uv.forward(c_kv, registry, queue)?;
        Ok((k_nope, v))
    }

    /// Compute compressed Q and expand to full Q heads.
    ///
    /// `x`: [seq_len, hidden_size]
    /// Returns q: [seq_len, num_heads * head_dim]
    pub fn compute_q(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let c_q = self.w_dq.forward(x, registry, queue)?;
        self.w_uq.forward(&c_q, registry, queue)
    }

    /// Forward pass for MLA.
    ///
    /// `x`: [seq_len, hidden_size]
    /// `cos_freqs`, `sin_freqs`: RoPE frequency tables for the decoupled portion
    /// `cache`: optional MLA KV cache
    ///
    /// Returns: [seq_len, hidden_size]
    ///
    /// This is a structural implementation that performs the projection steps.
    /// The full attention computation (Q@K^T/sqrt(d) + mask, softmax, @V)
    /// is delegated to the core SDPA kernel after head splitting.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mut cache: Option<&mut MlaKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let nope_dim = self.config.nope_head_dim();
        let rope_dim = self.config.rope_head_dim;
        let v_dim = self.config.v_head_dim;
        let elem_size = x.dtype().size_of();

        // Step 1: Compress KV and compute decoupled RoPE key
        let c_kv = self.compress_kv(x, registry, queue)?;
        let k_rope_raw = self.compute_k_rope(x, registry, queue)?;

        // Step 2: Compute Q (compressed then expanded)
        // q shape: [seq_len, num_heads * head_dim]
        let q = self.compute_q(x, registry, queue)?;

        // Step 3: Apply RoPE to the rope portions of Q and K
        // k_rope_raw is [seq_len, rope_dim] — shared across all heads (decoupled RoPE).
        // Q rope portion: for each head h, extract q[seq, h*head_dim + nope_dim .. h*head_dim + head_dim]
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        let k_rope = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            ops::rope::rope(registry, &k_rope_raw, cos, sin, rope_offset, 1.0, queue)?
        } else {
            ops::copy::copy(registry, &k_rope_raw, queue)?
        };

        // Step 5: Update cache if provided
        // We store c_kv and k_rope (post-RoPE) into the cache.
        // On subsequent calls, we retrieve and expand from cache.
        let (c_kv_full, k_rope_full, total_seq) = if let Some(ref mut ca) = cache {
            // Copy new c_kv into cache at [seq_len_old .. seq_len_old + seq_len]
            Self::copy_into_cache_buffer(
                &ca.c_kv_cache,
                &c_kv,
                ca.seq_len,
                seq_len,
                ca.kv_lora_rank,
                registry,
                queue,
            )?;
            Self::copy_into_cache_buffer(
                &ca.k_rope_cache,
                &k_rope,
                ca.seq_len,
                seq_len,
                ca.rope_head_dim,
                registry,
                queue,
            )?;
            ca.advance(seq_len)?;
            let ts = ca.seq_len;
            (ca.cached_c_kv(), ca.cached_k_rope(), ts)
        } else {
            (c_kv, k_rope, seq_len)
        };

        // Re-expand from full cached c_kv to get full-sequence k_nope and v
        let (k_nope_full, v_full) = self.expand_kv(&c_kv_full, registry, queue)?;

        // Step 6: Split into per-head tensors and assemble full K per head.
        //
        // For each head h:
        //   q_h = q[:, h*head_dim .. (h+1)*head_dim]   -> [seq_len, head_dim]
        //   k_nope_h = k_nope_full[:, h*nope_dim .. (h+1)*nope_dim] -> [total_seq, nope_dim]
        //   v_h = v_full[:, h*v_dim .. (h+1)*v_dim]     -> [total_seq, v_dim]
        //
        // Then apply RoPE to the rope portion of q_h, and concat k_nope_h with k_rope_full.
        //
        // For SDPA: Q, K, V must all be [tokens, head_dim].
        // MLA key: k_rope is shared across heads (decoupled), so K per head =
        //   concat(k_nope_h, k_rope_full) along dim=1.
        // This requires nope_dim + rope_dim == head_dim, which is guaranteed by config.

        let dev = registry.device().raw();
        let q_total_width = num_heads * head_dim;
        let k_nope_total_width = num_heads * nope_dim;
        let v_total_width = num_heads * v_dim;

        let mut q_heads: Vec<Array> = Vec::with_capacity(num_heads);
        let mut k_heads: Vec<Array> = Vec::with_capacity(num_heads);
        let mut v_heads: Vec<Array> = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            // --- Q head: [seq_len, head_dim] ---
            let q_offset = q.offset() + h * head_dim * elem_size;
            let q_head_view = q.view(vec![seq_len, head_dim], vec![q_total_width, 1], q_offset);
            let q_head = ops::copy::copy(registry, &q_head_view, queue)?;

            // Apply RoPE to the rope portion of Q.
            // Split q_head into [seq_len, nope_dim] and [seq_len, rope_dim],
            // apply RoPE to rope portion, then concatenate back.
            let q_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                if rope_dim > 0 {
                    // Extract nope portion: q_head[:, 0..nope_dim]
                    let q_nope_view =
                        q_head.view(vec![seq_len, nope_dim], vec![head_dim, 1], q_head.offset());
                    let q_nope = ops::copy::copy(registry, &q_nope_view, queue)?;

                    // Extract rope portion: q_head[:, nope_dim..head_dim]
                    let q_rope_view = q_head.view(
                        vec![seq_len, rope_dim],
                        vec![head_dim, 1],
                        q_head.offset() + nope_dim * elem_size,
                    );
                    let q_rope = ops::copy::copy(registry, &q_rope_view, queue)?;

                    // Apply RoPE to rope portion
                    let q_rope_rotated =
                        ops::rope::rope(registry, &q_rope, cos, sin, rope_offset, 1.0, queue)?;

                    // Concatenate nope + rotated_rope along axis=1
                    ops::concat::concat(registry, &q_nope, &q_rope_rotated, 1, queue)?
                } else {
                    q_head
                }
            } else {
                q_head
            };
            q_heads.push(q_head);

            // --- K head: concat(k_nope_h, k_rope_full) -> [total_seq, head_dim] ---
            let kn_offset = k_nope_full.offset() + h * nope_dim * elem_size;
            let k_nope_h_view = k_nope_full.view(
                vec![total_seq, nope_dim],
                vec![k_nope_total_width, 1],
                kn_offset,
            );
            let k_nope_h = ops::copy::copy(registry, &k_nope_h_view, queue)?;

            // k_rope_full is [total_seq, rope_dim], shared across heads
            let k_h = ops::concat::concat(registry, &k_nope_h, &k_rope_full, 1, queue)?;
            k_heads.push(k_h);

            // --- V head: [total_seq, v_dim] ---
            let v_offset = v_full.offset() + h * v_dim * elem_size;
            let v_h_view = v_full.view(vec![total_seq, v_dim], vec![v_total_width, 1], v_offset);
            let v_h = ops::copy::copy(registry, &v_h_view, queue)?;
            v_heads.push(v_h);
        }

        // Step 7: SDPA
        // SDPA requires Q [N, D], K [S, D], V [S, D] where D matches across Q/K/V.
        // In MLA with v_head_dim != head_dim, the SDPA kernel will reject the shape.
        // TODO: Support v_head_dim != head_dim via a custom attention kernel.
        if v_dim != head_dim {
            return Err(KernelError::InvalidShape(format!(
                "MLA forward: v_head_dim ({}) != head_dim ({}) is not yet supported by SDPA kernel",
                v_dim, head_dim,
            )));
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Use batched SDPA (all heads use same K rope, but we already assembled per-head K).
        // k_heads and v_heads have 1 per query head (no GQA in MLA — all heads share same latent).
        let attn_outputs = ops::sdpa::sdpa_batched(
            registry, &q_heads, &k_heads, &v_heads, None, scale, false, queue,
        )?;

        // Step 8: Concatenate head outputs -> [seq_len, num_heads * v_dim]
        let hidden_out = num_heads * v_dim;
        let concat_out = Array::zeros(dev, &[seq_len, hidden_out], x.dtype());

        let copy_kernel = match x.dtype() {
            rmlx_core::dtype::DType::Float32 => "copy_f32",
            rmlx_core::dtype::DType::Float16 => "copy_f16",
            rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "MLA concat: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;
        let head_bytes = v_dim * elem_size;
        let hidden_bytes = hidden_out * elem_size;

        let cb = queue.new_command_buffer();
        for (h, head_out) in attn_outputs.iter().enumerate() {
            let dst_col_offset = h * head_bytes;

            if seq_len == 1 {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset() as u64);
                enc.set_buffer(1, Some(concat_out.metal_buffer()), dst_col_offset as u64);
                let count = v_dim as u64;
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
                        concat_out.metal_buffer(),
                        dst_off,
                        head_bytes as u64,
                    );
                }
                blit.end_encoding();
            }
        }
        cb.commit();
        cb.wait_until_completed();

        // Step 9: Output projection -> [seq_len, hidden_size]
        self.o_proj.forward(&concat_out, registry, queue)
    }

    /// Copy `src` ([new_tokens, width]) into a pre-allocated cache buffer at row offset.
    fn copy_into_cache_buffer(
        cache_buf: &Array,
        src: &Array,
        row_offset: usize,
        new_tokens: usize,
        width: usize,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        let elem_size = src.dtype().size_of();
        let copy_kernel = match src.dtype() {
            rmlx_core::dtype::DType::Float32 => "copy_f32",
            rmlx_core::dtype::DType::Float16 => "copy_f16",
            rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "MlaKvCache: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, src.dtype())?;
        let count = (new_tokens * width) as u64;
        if count == 0 {
            return Ok(());
        }

        let dst_byte_offset = row_offset * width * elem_size;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
        enc.set_buffer(
            1,
            Some(cache_buf.metal_buffer()),
            (cache_buf.offset() + dst_byte_offset) as u64,
        );
        let grid = metal::MTLSize::new(count, 1, 1);
        let tg = metal::MTLSize::new(
            std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        Ok(())
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    /// KV compression rank.
    pub fn kv_lora_rank(&self) -> usize {
        self.config.kv_lora_rank
    }

    /// Memory savings factor vs. standard MHA KV cache.
    ///
    /// Standard MHA caches: 2 * num_heads * head_dim floats per token
    /// MLA caches: kv_lora_rank + rope_head_dim floats per token
    pub fn cache_compression_ratio(&self) -> f32 {
        let standard = 2 * self.config.num_heads * self.config.head_dim;
        let mla = self.config.kv_lora_rank + self.config.rope_head_dim;
        standard as f32 / mla as f32
    }
}
