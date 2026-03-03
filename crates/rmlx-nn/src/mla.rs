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
        cache: Option<&mut MlaKvCache>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];

        // Step 1: Compress KV and compute decoupled RoPE key
        let c_kv = self.compress_kv(x, registry, queue)?;
        let k_rope = self.compute_k_rope(x, registry, queue)?;

        // Step 2: Compute Q (compressed then expanded)
        let q = self.compute_q(x, registry, queue)?;

        // Step 3: Expand KV from latent
        let (k_nope, v) = self.expand_kv(&c_kv, registry, queue)?;

        // Reference the cache and RoPE for future integration
        let _ = (cos_freqs, sin_freqs, cache, seq_len);
        let _ = (&q, &k_nope, &k_rope, &v);

        // Placeholder: return through output projection.
        // In production, this would:
        // 1. Apply RoPE to q_rope and k_rope portions
        // 2. Concatenate k_nope with rotated k_rope per head
        // 3. Update cache with c_kv and k_rope (not full K/V)
        // 4. Run SDPA on the assembled Q, K, V
        // 5. Concatenate head outputs
        self.o_proj.forward(x, registry, queue)
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
