//! Multi-head attention with RoPE and GQA support.
//!
//! KV cache uses pre-allocated buffers with O(1) append (no full-history copy).

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::{Linear, LinearConfig};

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions,
};
use rmlx_metal::{ComputePass, MTLSize};

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
    /// Contiguous K slab [num_kv_heads * max_seq_len * head_dim] for batched SDPA
    keys_slab: Option<Array>,
    /// Contiguous V slab [num_kv_heads * max_seq_len * head_dim] for batched SDPA
    values_slab: Option<Array>,
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
            keys_slab: None,
            values_slab: None,
        }
    }

    /// Create a pre-allocated cache with room for `max_seq_len` tokens.
    ///
    /// Each KV head gets a single [max_seq_len, head_dim] buffer up-front.
    /// Subsequent `append` calls write into the next slot(s) with no reallocation.
    pub fn preallocated(
        device: &ProtocolObject<dyn MTLDevice>,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Self {
        let elem_size = dtype.size_of();
        let head_size = max_seq_len * head_dim; // elements per head

        // Allocate contiguous slabs
        let k_slab = Array::zeros(device, &[num_kv_heads * max_seq_len * head_dim], dtype);
        let v_slab = Array::zeros(device, &[num_kv_heads * max_seq_len * head_dim], dtype);

        // Create per-head views into slabs (backward compatible)
        let mut keys = Vec::with_capacity(num_kv_heads);
        let mut values = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let offset = h * head_size * elem_size;
            keys.push(Array::new(
                k_slab.metal_buffer().to_owned(),
                vec![max_seq_len, head_dim],
                vec![head_dim, 1],
                dtype,
                offset,
            ));
            values.push(Array::new(
                v_slab.metal_buffer().to_owned(),
                vec![max_seq_len, head_dim],
                vec![head_dim, 1],
                dtype,
                offset,
            ));
        }

        Self {
            keys,
            values,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            keys_slab: Some(k_slab),
            values_slab: Some(v_slab),
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

    /// Remove the last `n` tokens from the cache.
    ///
    /// This is an O(1) metadata-only operation: the pre-allocated buffers
    /// are unchanged; only the logical cursor (`seq_len`) is rewound.
    /// Used by speculative decoding to roll back rejected draft tokens.
    ///
    /// Returns the new sequence length after trimming.
    pub fn trim(&mut self, n: usize) -> usize {
        self.seq_len = self.seq_len.saturating_sub(n);
        self.seq_len
    }

    /// Validate consistency among input tensors when the cache is empty (first append).
    /// Checks that all tensors are 2D, have the same dtype, the same head_dim, and
    /// seq dimension matches new_tokens.
    fn validate_first_append_inputs(
        new_keys: &[Array],
        new_values: &[Array],
        new_tokens: usize,
        context: &str,
    ) -> Result<(), KernelError> {
        // Use the first key tensor as the reference for dtype and head_dim
        let ref_tensor = new_keys.first().or(new_values.first());
        let (ref_dtype, ref_head_dim) = match ref_tensor {
            Some(t) => {
                let shape = t.shape();
                if shape.len() != 2 {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: first tensor expected 2D [seq, head_dim], got {:?}",
                        shape
                    )));
                }
                (t.dtype(), shape[1])
            }
            None => return Ok(()), // no tensors to validate
        };

        for (kind, tensors) in [("keys", new_keys), ("values", new_values)] {
            for (i, tensor) in tensors.iter().enumerate() {
                if tensor.dtype() != ref_dtype {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] dtype mismatch: expected {:?}, got {:?}",
                        ref_dtype,
                        tensor.dtype()
                    )));
                }
                let shape = tensor.shape();
                if shape.len() != 2 {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] expected 2D [seq, head_dim], got {:?}",
                        shape
                    )));
                }
                if shape[1] != ref_head_dim {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] head_dim mismatch: expected {}, got {}",
                        ref_head_dim, shape[1]
                    )));
                }
                if shape[0] != new_tokens {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] seq mismatch: expected {}, got {}",
                        new_tokens, shape[0]
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_cached_inputs(
        &self,
        new_keys: &[Array],
        new_values: &[Array],
        new_tokens: usize,
        context: &str,
    ) -> Result<(), KernelError> {
        let cache_dtype = self.keys[0].dtype();
        for (kind, tensors) in [("keys", new_keys), ("values", new_values)] {
            for (i, tensor) in tensors.iter().enumerate() {
                if tensor.dtype() != cache_dtype {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] dtype mismatch: expected {:?}, got {:?}",
                        cache_dtype,
                        tensor.dtype()
                    )));
                }

                let shape = tensor.shape();
                if shape.len() != 2 {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] expected 2D [seq, head_dim], got {:?}",
                        shape
                    )));
                }
                if shape[1] != self.head_dim {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] head_dim mismatch: expected {}, got {}",
                        self.head_dim, shape[1]
                    )));
                }
                if shape[0] != new_tokens {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] seq mismatch: expected {}, got {}",
                        new_tokens, shape[0]
                    )));
                }
            }
        }
        Ok(())
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
            // Legacy path, first append — validate consistency among inputs
            Self::validate_first_append_inputs(
                &new_keys,
                &new_values,
                new_tokens,
                "LayerKvCache::append",
            )?;
            self.keys = new_keys;
            self.values = new_values;
            if let Some(k) = self.keys.first() {
                self.head_dim = k.shape()[1];
            }
        } else {
            // Legacy path, concat
            self.validate_cached_inputs(
                &new_keys,
                &new_values,
                new_tokens,
                "LayerKvCache::append",
            )?;
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        self.validate_cached_inputs(
            &new_keys,
            &new_values,
            new_tokens,
            "LayerKvCache::append_preallocated",
        )?;
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
        let count = new_tokens * self.head_dim;
        if count == 0 {
            return Ok(());
        }

        // Single command buffer + single encoder for all heads
        let cb = queue.commandBuffer().unwrap();
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        let grid = MTLSize {
            width: (count),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
            height: 1,
            depth: 1,
        };

        for i in 0..self.num_kv_heads {
            // Copy new keys into slot
            enc.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset());
            enc.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_row_offset,
            );
            enc.dispatch_threads(grid, tg);

            // Copy new values into slot
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset(),
            );
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_row_offset,
            );
            enc.dispatch_threads(grid, tg);
        }
        enc.end();

        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// Append new K, V heads into the pre-allocated cache using an existing CB.
    ///
    /// Encodes copy dispatches for each KV head into `cb`, writing new tokens
    /// into pre-allocated slots at `[seq_len..seq_len+new_tokens]`.
    /// Does NOT commit or wait -- the caller manages the CB lifecycle.
    ///
    /// Only works for pre-allocated caches (`max_seq_len > 0`).
    pub fn append_into_cb(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_into_cb: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        if self.max_seq_len == 0 {
            return Err(KernelError::InvalidShape(
                "LayerKvCache::append_into_cb requires pre-allocated cache".to_string(),
            ));
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_into_cb: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }
        self.validate_cached_inputs(
            &new_keys,
            &new_values,
            new_tokens,
            "LayerKvCache::append_into_cb",
        )?;

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
        let count = new_tokens * self.head_dim;
        if count == 0 {
            self.seq_len += new_tokens;
            return Ok(());
        }

        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        let raw_enc = cb.computeCommandEncoder().unwrap();

        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);

        for i in 0..self.num_kv_heads {
            // Copy new keys into slot
            enc.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset());
            enc.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_row_offset,
            );
            let grid = MTLSize {
                width: (count),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);

            // Copy new values into slot
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset(),
            );
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_row_offset,
            );
            enc.dispatch_threads(grid, tg);
        }
        enc.end();

        self.seq_len += new_tokens;
        Ok(())
    }

    /// Append new K, V into the pre-allocated slab cache using a single batched
    /// GPU dispatch (1 dispatch instead of 2 * num_kv_heads).
    ///
    /// Requires:
    ///   - Pre-allocated slab cache (`keys_slab` / `values_slab` present)
    ///   - f16 dtype
    ///   - `src_k` / `src_v` are contiguous with layout `[num_kv_heads * new_tokens, head_dim]`
    ///     (head-major, i.e. output of `deinterleave_heads` / `rope_multihead`)
    ///
    /// Falls back to `append_into_cb` if preconditions are not met.
    pub fn append_batched_into_cb(
        &mut self,
        src_k: &Array,
        src_v: &Array,
        new_tokens: usize,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<(), KernelError> {
        // Check preconditions for the batched path
        let has_slab = self.keys_slab.is_some() && self.values_slab.is_some();
        let is_f16 = !self.keys.is_empty() && self.keys[0].dtype() == DType::Float16;

        if !has_slab || !is_f16 || self.max_seq_len == 0 {
            // Fallback: split into per-head views and use the old path
            let elem_size = src_k.dtype().size_of();
            let head_elems = new_tokens * self.head_dim;
            let k_heads: Vec<Array> = (0..self.num_kv_heads)
                .map(|h| {
                    src_k.view(
                        vec![new_tokens, self.head_dim],
                        vec![self.head_dim, 1],
                        src_k.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            let v_heads: Vec<Array> = (0..self.num_kv_heads)
                .map(|h| {
                    src_v.view(
                        vec![new_tokens, self.head_dim],
                        vec![self.head_dim, 1],
                        src_v.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            return self.append_into_cb(k_heads, v_heads, new_tokens, registry, cb);
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_batched_into_cb: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        let k_slab = self.keys_slab.as_ref().unwrap();
        let v_slab = self.values_slab.as_ref().unwrap();

        ops::copy::kv_cache_copy_batched_f16_into_cb(
            src_k,
            src_v,
            k_slab,
            v_slab,
            self.num_kv_heads,
            new_tokens,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            registry,
            cb,
        )?;

        self.seq_len += new_tokens;
        Ok(())
    }

    /// Batched KV cache append using a pre-existing compute command encoder.
    ///
    /// Like `append_batched_into_cb` but dispatches into an already-open encoder,
    /// avoiding encoder create/destroy overhead.
    pub fn append_batched_encode(
        &mut self,
        src_k: &Array,
        src_v: &Array,
        new_tokens: usize,
        registry: &KernelRegistry,
        encoder: ComputePass<'_>,
    ) -> Result<(), KernelError> {
        let has_slab = self.keys_slab.is_some() && self.values_slab.is_some();
        let is_f16 = !self.keys.is_empty() && self.keys[0].dtype() == DType::Float16;

        if !has_slab || !is_f16 || self.max_seq_len == 0 {
            // Fallback: split into per-head views and use the existing encoder path
            let elem_size = src_k.dtype().size_of();
            let head_elems = new_tokens * self.head_dim;
            let k_heads: Vec<Array> = (0..self.num_kv_heads)
                .map(|h| {
                    src_k.view(
                        vec![new_tokens, self.head_dim],
                        vec![self.head_dim, 1],
                        src_k.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            let v_heads: Vec<Array> = (0..self.num_kv_heads)
                .map(|h| {
                    src_v.view(
                        vec![new_tokens, self.head_dim],
                        vec![self.head_dim, 1],
                        src_v.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            return self.append_into_encoder(k_heads, v_heads, new_tokens, registry, encoder);
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_batched_encode: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        let k_slab = self.keys_slab.as_ref().unwrap();
        let v_slab = self.values_slab.as_ref().unwrap();

        ops::copy::kv_cache_copy_batched_f16_encode(
            src_k,
            src_v,
            k_slab,
            v_slab,
            self.num_kv_heads,
            new_tokens,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            registry,
            encoder,
        )?;

        self.seq_len += new_tokens;
        Ok(())
    }

    /// Like append_into_cb but dispatches into an already-open encoder.
    pub fn append_into_encoder(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        encoder: ComputePass<'_>,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_into_cb: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        if self.max_seq_len == 0 {
            return Err(KernelError::InvalidShape(
                "LayerKvCache::append_into_cb requires pre-allocated cache".to_string(),
            ));
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_into_cb: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }
        self.validate_cached_inputs(
            &new_keys,
            &new_values,
            new_tokens,
            "LayerKvCache::append_into_cb",
        )?;

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
        let count = new_tokens * self.head_dim;
        if count == 0 {
            self.seq_len += new_tokens;
            return Ok(());
        }

        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        encoder.set_pipeline(&pipeline);

        for i in 0..self.num_kv_heads {
            // Copy new keys into slot
            encoder.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset());
            encoder.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_row_offset,
            );
            let grid = MTLSize {
                width: count,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count),
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, tg);

            // Copy new values into slot
            encoder.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset(),
            );
            encoder.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_row_offset,
            );
            encoder.dispatch_threads(grid, tg);
        }

        self.seq_len += new_tokens;
        Ok(())
    }

    /// Append KV using pre-resolved copy PSO (no registry lookup or input validation).
    ///
    /// # Safety
    ///
    /// Caller must ensure inputs have correct shapes, dtypes, and head counts.
    /// This skips `validate_cached_inputs` for performance — use only from
    /// `CachedDecode` where dimensions are validated once at init time.
    pub fn append_preresolved_into_encoder(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        copy_pso: &ProtocolObject<dyn MTLComputePipelineState>,
        encoder: ComputePass<'_>,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_preresolved: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        if self.max_seq_len == 0 {
            return Err(KernelError::InvalidShape(
                "LayerKvCache::append_preresolved requires pre-allocated cache".to_string(),
            ));
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_preresolved: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        let elem_size = self.keys[0].dtype().size_of();
        let count = new_tokens * self.head_dim;
        if count == 0 {
            self.seq_len += new_tokens;
            return Ok(());
        }

        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        encoder.set_pipeline(copy_pso);

        for i in 0..self.num_kv_heads {
            encoder.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset());
            encoder.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_row_offset,
            );
            let grid = MTLSize {
                width: count,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: std::cmp::min(copy_pso.maxTotalThreadsPerThreadgroup(), count),
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, tg);

            encoder.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset(),
            );
            encoder.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_row_offset,
            );
            encoder.dispatch_threads(grid, tg);
        }

        self.seq_len += new_tokens;
        Ok(())
    }

    /// Append KV using raw buffer references (no Vec<Array> allocation).
    ///
    /// `k_buf`/`v_buf` are the source buffers containing contiguous per-head data.
    /// `k_base_offset`/`v_base_offset` are byte offsets to the first K/V head.
    /// Each head's data is `head_dim` elements starting at base + h * head_dim * elem_size.
    /// `copy_max_tg` is the pre-cached max threadgroup size for the copy PSO.
    ///
    /// # Safety
    /// Same as `append_preresolved_into_encoder` — caller validates dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn append_direct_into_encoder(
        &mut self,
        k_buf: &ProtocolObject<dyn MTLBuffer>,
        k_base_offset: u64,
        v_buf: &ProtocolObject<dyn MTLBuffer>,
        v_base_offset: u64,
        new_tokens: usize,
        copy_pso: &ProtocolObject<dyn MTLComputePipelineState>,
        copy_max_tg: u64,
        encoder: ComputePass<'_>,
    ) -> Result<(), KernelError> {
        if self.max_seq_len == 0 {
            return Err(KernelError::InvalidShape(
                "LayerKvCache::append_direct requires pre-allocated cache".to_string(),
            ));
        }

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "LayerKvCache::append_direct: overflow: {} cached + {} new > {} max",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        let elem_size = self.keys[0].dtype().size_of();
        let count = new_tokens * self.head_dim;
        if count == 0 {
            self.seq_len += new_tokens;
            return Ok(());
        }

        let dst_row_offset = self.seq_len * self.head_dim * elem_size;
        let head_stride = self.head_dim * elem_size;

        encoder.set_pipeline(copy_pso);

        let grid = MTLSize {
            width: count,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: std::cmp::min(copy_max_tg as usize, count),
            height: 1,
            depth: 1,
        };

        for i in 0..self.num_kv_heads {
            let src_k_off = k_base_offset as usize + i * head_stride;
            encoder.set_buffer(0, Some(k_buf), src_k_off);
            encoder.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_row_offset,
            );
            encoder.dispatch_threads(grid, tg);

            let src_v_off = v_base_offset as usize + i * head_stride;
            encoder.set_buffer(0, Some(v_buf), src_v_off);
            encoder.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_row_offset,
            );
            encoder.dispatch_threads(grid, tg);
        }

        self.seq_len += new_tokens;
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

    /// Get K slab as flat view for batched SDPA.
    /// Returns None if not using slab layout or cache is empty.
    pub fn keys_slab_view(&self) -> Option<Array> {
        let slab = self.keys_slab.as_ref()?;
        if self.seq_len == 0 {
            return None;
        }
        Some(Array::new(
            slab.metal_buffer().to_owned(),
            vec![self.num_kv_heads * self.max_seq_len * self.head_dim],
            vec![1],
            slab.dtype(),
            slab.offset(),
        ))
    }

    /// Get V slab as flat view for batched SDPA.
    /// Returns None if not using slab layout or cache is empty.
    pub fn values_slab_view(&self) -> Option<Array> {
        let slab = self.values_slab.as_ref()?;
        if self.seq_len == 0 {
            return None;
        }
        Some(Array::new(
            slab.metal_buffer().to_owned(),
            vec![self.num_kv_heads * self.max_seq_len * self.head_dim],
            vec![1],
            slab.dtype(),
            slab.offset(),
        ))
    }

    /// Max sequence length (stride between KV heads in the slab).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ---------------------------------------------------------------------------
// RotatingKvCache — circular buffer KV cache with optional keep region
// ---------------------------------------------------------------------------

/// Circular-buffer KV cache following mlx-lm's rotating cache design.
///
/// Tokens are written into a fixed-size ring buffer. When `keep > 0`, the
/// first `keep` positions are pinned (e.g. for a system prompt) and the
/// circular region spans `[keep .. max_size)`.
///
/// This avoids ever re-allocating or shifting the full history — only the
/// write pointer advances.
pub struct RotatingKvCache {
    /// Pre-allocated K buffers per KV head: [max_size, head_dim]
    keys: Vec<Array>,
    /// Pre-allocated V buffers per KV head: [max_size, head_dim]
    values: Vec<Array>,
    /// Total tokens processed (monotonically increasing)
    offset: usize,
    /// Current circular write position
    write_idx: usize,
    /// Maximum buffer size (circular wrap point)
    max_size: usize,
    /// Number of tokens at the start to preserve (system prompt)
    keep: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl RotatingKvCache {
    /// Create a new rotating KV cache with pre-allocated buffers.
    ///
    /// `max_size`: total ring-buffer capacity (including the `keep` region).
    /// `keep`: number of initial positions that are never overwritten.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        num_kv_heads: usize,
        head_dim: usize,
        max_size: usize,
        keep: usize,
        dtype: DType,
    ) -> Self {
        assert!(keep < max_size, "keep must be < max_size");
        let mut keys = Vec::with_capacity(num_kv_heads);
        let mut values = Vec::with_capacity(num_kv_heads);
        for _ in 0..num_kv_heads {
            keys.push(Array::zeros(device, &[max_size, head_dim], dtype));
            values.push(Array::zeros(device, &[max_size, head_dim], dtype));
        }
        Self {
            keys,
            values,
            offset: 0,
            write_idx: 0,
            max_size,
            keep,
            num_kv_heads,
            head_dim,
        }
    }

    /// Total tokens processed so far (monotonically increasing).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of valid tokens currently in the buffer.
    /// This is `min(offset, max_size)` — once the buffer is full, it stays full.
    pub fn current_len(&self) -> usize {
        std::cmp::min(self.offset, self.max_size)
    }

    /// Whether the cache has received any tokens yet.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    fn validate_copy_inputs(
        &self,
        new_keys: &[Array],
        new_values: &[Array],
        context: &str,
    ) -> Result<(), KernelError> {
        let cache_dtype = self.keys[0].dtype();
        for (kind, tensors) in [("keys", new_keys), ("values", new_values)] {
            for (i, tensor) in tensors.iter().enumerate() {
                if tensor.dtype() != cache_dtype {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] dtype mismatch: expected {:?}, got {:?}",
                        cache_dtype,
                        tensor.dtype()
                    )));
                }

                let shape = tensor.shape();
                if shape.len() != 2 {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] expected 2D [seq, head_dim], got {:?}",
                        shape
                    )));
                }
                if shape[1] != self.head_dim {
                    return Err(KernelError::InvalidShape(format!(
                        "{context}: {kind}[{i}] head_dim mismatch: expected {}, got {}",
                        self.head_dim, shape[1]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Append new K/V tokens into the rotating buffer.
    ///
    /// **Single-token decode** (`new_tokens == 1`): writes at `write_idx`,
    /// then advances `write_idx`. On wrap, skips past the `keep` region.
    ///
    /// **Multi-token prefill** (`new_tokens > 1`): if the write would cross
    /// the wrap boundary, linearizes first via `_temporal_order`, then writes
    /// contiguously and trims.
    pub fn append(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "RotatingKvCache::append: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        if new_tokens == 0 {
            return Ok(());
        }

        if new_tokens == 1 {
            // Fast path: single-token decode — write at write_idx
            self.copy_at_index(&new_keys, &new_values, self.write_idx, 1, registry, queue)?;
            self.write_idx += 1;
            if self.write_idx >= self.max_size {
                self.write_idx = self.keep;
            }
        } else {
            // Multi-token prefill path
            let end = self.write_idx + new_tokens;
            if end <= self.max_size {
                // Fits without wrapping — write contiguously
                self.copy_at_index(
                    &new_keys,
                    &new_values,
                    self.write_idx,
                    new_tokens,
                    registry,
                    queue,
                )?;
                self.write_idx = if end >= self.max_size { self.keep } else { end };
            } else {
                // Would wrap — linearize the buffer first, then concat + trim
                self.linearize_and_append(new_keys, new_values, new_tokens, registry, queue)?;
            }
        }

        self.offset += new_tokens;
        Ok(())
    }

    /// Compute the temporal ordering indices for the ring buffer.
    ///
    /// Returns `(keep, ring_start, ring_end)` where the logical order is:
    /// `[0..keep] + [write_idx..max_size] + [keep..write_idx]`
    ///
    /// `ring_start` is the oldest non-keep position (i.e. `write_idx`),
    /// and `ring_end` is `write_idx` (the newest written positions wrap here).
    fn _temporal_order(&self) -> (usize, usize, usize) {
        // keep region: [0 .. keep] — always in order
        // oldest ring region: [write_idx .. max_size]
        // newest ring region: [keep .. write_idx]
        (self.keep, self.write_idx, self.max_size)
    }

    /// Linearize the buffer (reorder to temporal order), append new data,
    /// then trim back to `max_size`.
    fn linearize_and_append(
        &mut self,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        let (keep, ring_start, ring_end) = self._temporal_order();
        let current_len = self.current_len();
        let dev = registry.device().raw();
        let dtype = self.keys[0].dtype();

        for h in 0..self.num_kv_heads {
            // Build linearized K
            let linear_k = self.linearize_head(
                &self.keys[h],
                keep,
                ring_start,
                ring_end,
                current_len,
                registry,
                queue,
            )?;
            // Concat with new tokens
            let combined_k = concat_seq_dim(registry, &linear_k, &new_keys[h], queue)?;
            // Trim to max_size (preserve keep region, drop oldest ring tokens)
            let total = combined_k.shape()[0];
            let trimmed_k = if total > self.max_size {
                self.trim_to_max(&combined_k, total, registry, queue)?
            } else {
                combined_k
            };
            // Write back into pre-allocated buffer
            let new_buf = Array::zeros(dev, &[self.max_size, self.head_dim], dtype);
            let filled = trimmed_k.shape()[0];
            self.copy_into_buffer(&new_buf, &trimmed_k, 0, filled, registry, queue)?;
            self.keys[h] = new_buf;

            // Build linearized V
            let linear_v = self.linearize_head(
                &self.values[h],
                keep,
                ring_start,
                ring_end,
                current_len,
                registry,
                queue,
            )?;
            let combined_v = concat_seq_dim(registry, &linear_v, &new_values[h], queue)?;
            let total = combined_v.shape()[0];
            let trimmed_v = if total > self.max_size {
                self.trim_to_max(&combined_v, total, registry, queue)?
            } else {
                combined_v
            };
            let new_buf = Array::zeros(dev, &[self.max_size, self.head_dim], dtype);
            let filled = trimmed_v.shape()[0];
            self.copy_into_buffer(&new_buf, &trimmed_v, 0, filled, registry, queue)?;
            self.values[h] = new_buf;
        }

        // After linearize+append+trim, buffer is in order — write_idx is at the filled point
        let total_after = current_len + new_tokens;
        if total_after >= self.max_size {
            // Buffer is full after this operation; write_idx wraps to keep
            self.write_idx = self.keep;
        } else {
            self.write_idx = total_after;
        }

        Ok(())
    }

    /// Reorder a single head buffer into temporal order.
    #[allow(clippy::too_many_arguments)]
    fn linearize_head(
        &self,
        buf: &Array,
        keep: usize,
        ring_start: usize,
        ring_end: usize,
        current_len: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let dev = registry.device().raw();
        let dtype = buf.dtype();
        let elem_size = dtype.size_of();

        // The buffer hasn't wrapped yet if offset < max_size.
        // In that case the valid data is simply [0..current_len] in physical order.
        if self.offset < self.max_size {
            let result = Array::zeros(dev, &[current_len, self.head_dim], dtype);
            if current_len > 0 {
                self.copy_into_buffer(&result, buf, 0, current_len, registry, queue)?;
            }
            return Ok(result);
        }

        // Temporal order: [0..keep] + [ring_start..ring_end] + [keep..ring_start]
        let part1_len = keep;
        let part2_len = ring_end - ring_start;
        let part3_len = ring_start - keep;
        let total = part1_len + part2_len + part3_len;
        let result = Array::zeros(dev, &[total, self.head_dim], dtype);

        let copy_kernel = match dtype {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "RotatingKvCache: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, dtype)?;
        let cb = queue.commandBuffer().unwrap();

        // Part 1: keep region [0..keep]
        if part1_len > 0 {
            let count = part1_len * self.head_dim;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), buf.offset());
            enc.set_buffer(1, Some(result.metal_buffer()), result.offset());
            let grid = MTLSize {
                width: (count),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
            enc.end();
        }

        // Part 2: oldest ring portion [ring_start..ring_end]
        if part2_len > 0 {
            let count = part2_len * self.head_dim;
            let src_off = buf.offset() + ring_start * self.head_dim * elem_size;
            let dst_off = result.offset() + part1_len * self.head_dim * elem_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), src_off);
            enc.set_buffer(1, Some(result.metal_buffer()), dst_off);
            let grid = MTLSize {
                width: (count),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
            enc.end();
        }

        // Part 3: newest ring portion [keep..ring_start]
        if part3_len > 0 {
            let count = part3_len * self.head_dim;
            let src_off = buf.offset() + keep * self.head_dim * elem_size;
            let dst_off = result.offset() + (part1_len + part2_len) * self.head_dim * elem_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), src_off);
            enc.set_buffer(1, Some(result.metal_buffer()), dst_off);
            let grid = MTLSize {
                width: (count),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
            enc.end();
        }

        cb.commit();
        cb.waitUntilCompleted();
        Ok(result)
    }

    /// Trim a linearized array to `max_size`, preserving the keep region.
    ///
    /// Input `arr` has layout: `[keep_tokens | ring_tokens | new_tokens]`
    /// with total length `total > max_size`. We must preserve `[0..keep]`
    /// and keep the most recent `max_size - keep` tokens from the remainder.
    fn trim_to_max(
        &self,
        arr: &Array,
        total: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let dev = registry.device().raw();
        let dtype = arr.dtype();
        let elem_size = dtype.size_of();
        let ring_capacity = self.max_size - self.keep;
        // Number of non-keep tokens to skip (oldest ring entries)
        let skip = total - self.max_size;

        if self.keep == 0 {
            // No keep region — just take the last max_size tokens
            let new_offset = arr.offset() + skip * self.head_dim * elem_size;
            return Ok(arr.view(
                vec![self.max_size, self.head_dim],
                arr.strides().to_vec(),
                new_offset,
            ));
        }

        // Build result: [keep_region] + [most recent ring_capacity tokens]
        let result = Array::zeros(dev, &[self.max_size, self.head_dim], dtype);

        // Copy keep region from arr[0..keep]
        self.copy_into_buffer(
            &result,
            &arr.view(
                vec![self.keep, self.head_dim],
                arr.strides().to_vec(),
                arr.offset(),
            ),
            0,
            self.keep,
            registry,
            queue,
        )?;

        // Copy most recent ring_capacity tokens from arr[keep + skip .. total]
        let src_start = self.keep + skip;
        let src = arr.view(
            vec![ring_capacity, self.head_dim],
            arr.strides().to_vec(),
            arr.offset() + src_start * self.head_dim * elem_size,
        );
        self.copy_into_buffer(&result, &src, self.keep, ring_capacity, registry, queue)?;

        Ok(result)
    }

    /// Copy `count` rows from `src` into `dst` starting at row `dst_row`.
    fn copy_into_buffer(
        &self,
        dst: &Array,
        src: &Array,
        dst_row: usize,
        count: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        if count == 0 {
            return Ok(());
        }
        let elem_size = dst.dtype().size_of();
        let copy_kernel = match dst.dtype() {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "RotatingKvCache: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, dst.dtype())?;
        let num_elems = count * self.head_dim;
        let cb = queue.commandBuffer().unwrap();
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(src.metal_buffer()), src.offset());
        enc.set_buffer(
            1,
            Some(dst.metal_buffer()),
            dst.offset() + dst_row * self.head_dim * elem_size,
        );
        let grid = MTLSize {
            width: (num_elems),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), num_elems)),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// Copy `new_tokens` rows from each head's new K/V into the buffer at `dst_row`.
    fn copy_at_index(
        &self,
        new_keys: &[Array],
        new_values: &[Array],
        dst_row: usize,
        new_tokens: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        self.validate_copy_inputs(new_keys, new_values, "RotatingKvCache::copy_at_index")?;
        let elem_size = self.keys[0].dtype().size_of();
        let copy_kernel = match self.keys[0].dtype() {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "RotatingKvCache: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, self.keys[0].dtype())?;
        let count = new_tokens * self.head_dim;
        if count == 0 {
            return Ok(());
        }

        let cb = queue.commandBuffer().unwrap();
        let dst_byte_offset = dst_row * self.head_dim * elem_size;

        for i in 0..self.num_kv_heads {
            // Keys
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(new_keys[i].metal_buffer()), new_keys[i].offset());
            enc.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                self.keys[i].offset() + dst_byte_offset,
            );
            let grid = MTLSize {
                width: (count),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
            enc.end();

            // Values
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset(),
            );
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                self.values[i].offset() + dst_byte_offset,
            );
            enc.dispatch_threads(grid, tg);
            enc.end();
        }

        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// Get cached keys for a specific KV head in temporal order.
    ///
    /// Returns shape `[current_len, head_dim]`. When the buffer has wrapped,
    /// the data is linearized so that the oldest non-keep token comes first
    /// and the most recently written token comes last.
    pub fn cached_keys(
        &self,
        head: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let (keep, ring_start, ring_end) = self._temporal_order();
        let current_len = self.current_len();
        self.linearize_head(
            &self.keys[head],
            keep,
            ring_start,
            ring_end,
            current_len,
            registry,
            queue,
        )
    }

    /// Get cached values for a specific KV head in temporal order.
    ///
    /// Returns shape `[current_len, head_dim]`. When the buffer has wrapped,
    /// the data is linearized so that the oldest non-keep token comes first
    /// and the most recently written token comes last.
    pub fn cached_values(
        &self,
        head: usize,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let (keep, ring_start, ring_end) = self._temporal_order();
        let current_len = self.current_len();
        self.linearize_head(
            &self.values[head],
            keep,
            ring_start,
            ring_end,
            current_len,
            registry,
            queue,
        )
    }
}

// ---------------------------------------------------------------------------
// BatchKvCache — per-sequence caches for batched decoding
// ---------------------------------------------------------------------------

/// Batched KV cache that holds one `LayerKvCache` per sequence in a batch.
///
/// Useful for serving multiple independent sequences simultaneously with
/// per-sequence offsets and the ability to filter / reorder the batch.
pub struct BatchKvCache {
    /// Per-sequence caches.
    caches: Vec<LayerKvCache>,
    /// Per-sequence token offset (total tokens processed).
    offsets: Vec<usize>,
    /// Number of sequences in the batch.
    batch_size: usize,
}

impl BatchKvCache {
    /// Create a new batch of pre-allocated KV caches.
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Self {
        let mut caches = Vec::with_capacity(batch_size);
        let offsets = vec![0usize; batch_size];
        for _ in 0..batch_size {
            caches.push(LayerKvCache::preallocated(
                device,
                num_kv_heads,
                head_dim,
                max_seq_len,
                dtype,
            ));
        }
        Self {
            caches,
            offsets,
            batch_size,
        }
    }

    /// Number of sequences in the batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get an immutable reference to the cache for sequence `batch_idx`.
    pub fn get(&self, batch_idx: usize) -> &LayerKvCache {
        &self.caches[batch_idx]
    }

    /// Get a mutable reference to the cache for sequence `batch_idx`.
    pub fn get_mut(&mut self, batch_idx: usize) -> &mut LayerKvCache {
        &mut self.caches[batch_idx]
    }

    /// Reset a specific sequence's cache, re-allocating fresh buffers.
    pub fn reset(
        &mut self,
        batch_idx: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) {
        self.caches[batch_idx] =
            LayerKvCache::preallocated(device, num_kv_heads, head_dim, max_seq_len, dtype);
        self.offsets[batch_idx] = 0;
    }

    /// Keep only the sequences at the given `indices`, reordering accordingly.
    ///
    /// After this call, `batch_size` equals `indices.len()`.
    pub fn filter(&mut self, indices: &[usize]) {
        let new_caches: Vec<LayerKvCache> = indices
            .iter()
            .map(|&idx| {
                // Move the cache out, replacing with an empty placeholder
                let mut placeholder = LayerKvCache::new(self.caches[idx].num_kv_heads());
                std::mem::swap(&mut placeholder, &mut self.caches[idx]);
                placeholder
            })
            .collect();
        let new_offsets: Vec<usize> = indices.iter().map(|&idx| self.offsets[idx]).collect();
        self.batch_size = indices.len();
        self.caches = new_caches;
        self.offsets = new_offsets;
    }

    /// Append caches from another `BatchKvCache`, growing the batch.
    pub fn extend(&mut self, mut other: BatchKvCache) {
        self.caches.append(&mut other.caches);
        self.offsets.append(&mut other.offsets);
        self.batch_size += other.batch_size;
    }

    /// Per-sequence offsets (total tokens processed per sequence).
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Maximum offset across all sequences.
    pub fn max_offset(&self) -> usize {
        self.offsets.iter().copied().max().unwrap_or(0)
    }
}

/// Concatenate two 2D arrays along dimension 0 (seq dimension).
/// `a`: [seq_a, dim], `b`: [seq_b, dim] -> [seq_a + seq_b, dim]
fn concat_seq_dim(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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

    let cb = queue.commandBuffer().unwrap();

    // Copy a into result[0..seq_a]
    let a_count = seq_a * dim;
    if a_count > 0 {
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(a.metal_buffer()), a.offset());
        enc.set_buffer(1, Some(result.metal_buffer()), result.offset());
        let grid = MTLSize {
            width: (a_count),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), a_count)),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end();
    }

    // Copy b into result[seq_a..total_seq]
    let b_count = seq_b * dim;
    if b_count > 0 {
        let dst_offset = result.offset() + seq_a * dim * elem_size;
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(b.metal_buffer()), b.offset());
        enc.set_buffer(1, Some(result.metal_buffer()), dst_offset);
        let grid = MTLSize {
            width: (b_count),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), b_count)),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        enc.end();
    }

    cb.commit();
    cb.waitUntilCompleted();

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

/// Intermediate state from attention D1-D3 + KV blit, for 2-encoder decode path.
///
/// Holds everything needed for `Attention::encode_phase2_into()` to encode
/// D4-D5 into a caller-supplied encoder without a full GPU barrier.
pub struct DecodePhase1 {
    /// Roped Q+K buffer (Q portion used for SDPA).
    pub qk_roped_buf: rmlx_metal::MtlBuffer,
    /// Byte offset into `qk_roped_buf` where Q starts.
    pub qk_roped_offset: usize,
    /// Q dimension (num_heads * head_dim).
    pub q_dim: usize,
    /// Original input `x` reshaped as [hidden_size] for residual in D5.
    pub x: Array,
    /// DType of the QKV output (for building views).
    pub dtype: rmlx_core::dtype::DType,
}

pub struct Attention {
    config: AttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Merged QKV weight [q_out + k_out + v_out, in_features] for 9-dispatch path.
    qkv_merged_weight: Option<Array>,
    /// Transposed merged QKV weight [in_features, q_out + k_out + v_out] for prefill GEMM.
    qkv_merged_weight_t: Option<Array>,
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
            qkv_merged_weight: None,
            qkv_merged_weight_t: None,
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
            qkv_merged_weight: None,
            qkv_merged_weight_t: None,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let repeats = num_heads / num_kv_heads;

        // Project Q, K, V (single command buffer for all 3)
        let proj_cb = queue.commandBuffer().unwrap();
        let q = self.q_proj.forward_into_cb(x, registry, &proj_cb)?;
        let k = self.k_proj.forward_into_cb(x, registry, &proj_cb)?;
        let v = self.v_proj.forward_into_cb(x, registry, &proj_cb)?;
        proj_cb.commit();
        proj_cb.waitUntilCompleted();

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

        let dev = registry.device().raw();
        let elem_size = q.dtype().size_of();

        // RoPE offset from cache position
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        // Fused deinterleave + RoPE: 1 dispatch for all Q heads, 1 for all K heads
        // (replaces num_heads*2 + num_kv_heads*2 separate dispatches)
        let q_heads: Vec<Array>;
        let k_heads: Vec<Array>;
        let v_heads: Vec<Array>;

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            // Fused path: deinterleave + RoPE in single dispatch each
            let q_batched = ops::rope::rope_multihead(
                registry,
                &q,
                cos,
                sin,
                num_heads,
                rope_offset,
                q.strides()[0],
                queue,
            )?;
            let k_batched = ops::rope::rope_multihead(
                registry,
                &k,
                cos,
                sin,
                num_kv_heads,
                rope_offset,
                k.strides()[0],
                queue,
            )?;
            // V doesn't need RoPE, just deinterleave
            let v_batched =
                ops::rope::deinterleave_heads(registry, &v, num_kv_heads, v.strides()[0], queue)?;

            // Create per-head views into the batched outputs (zero-copy)
            let head_elems = seq_len * head_dim;
            q_heads = (0..num_heads)
                .map(|h| {
                    q_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        q_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            k_heads = (0..num_kv_heads)
                .map(|h| {
                    k_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        k_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            v_heads = (0..num_kv_heads)
                .map(|h| {
                    v_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        v_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
        } else {
            // No RoPE — just deinterleave
            let q_batched =
                ops::rope::deinterleave_heads(registry, &q, num_heads, q.strides()[0], queue)?;
            let k_batched =
                ops::rope::deinterleave_heads(registry, &k, num_kv_heads, k.strides()[0], queue)?;
            let v_batched =
                ops::rope::deinterleave_heads(registry, &v, num_kv_heads, v.strides()[0], queue)?;

            let head_elems = seq_len * head_dim;
            q_heads = (0..num_heads)
                .map(|h| {
                    q_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        q_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            k_heads = (0..num_kv_heads)
                .map(|h| {
                    k_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        k_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            v_heads = (0..num_kv_heads)
                .map(|h| {
                    v_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        v_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
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

        // Try fused SDPA path (Flash Attention 2 — no intermediate score matrix)
        let attn_outputs = if head_dim <= 256 {
            // Fused FA2 path: single kernel per head, supports D up to 256
            // Q@K^T + scale + mask + softmax + @V with online softmax
            ops::sdpa::sdpa_batched(
                registry, &q_heads, &k_final, &v_final, mask, scale, false, queue,
            )?
        } else {
            // Unfused fallback for head dims > 256
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

        // Concatenate heads: pack SDPA outputs into contiguous batch-major
        // buffer, then interleave into [seq_len, hidden_size] with a single
        // compute dispatch. For seq_len=1, this is equivalent to a flat copy.
        //
        // Step 1: Pack per-head outputs into [num_heads * seq_len, head_dim]
        // Step 2: Single interleave_heads dispatch → [seq_len, hidden_size]
        let hidden_size = num_heads * head_dim;

        if seq_len == 1 {
            // Fast path for decode: direct flat copy into concat buffer
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
            let head_bytes = head_dim * elem_size;

            let cb = queue.commandBuffer().unwrap();
            for (h, head_out) in attn_outputs.iter().enumerate() {
                let dst_col_offset = h * head_bytes;
                let raw_enc = cb.computeCommandEncoder().unwrap();
                let enc = ComputePass::new(&raw_enc);
                enc.set_pipeline(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset());
                enc.set_buffer(1, Some(concat.metal_buffer()), dst_col_offset);
                let count = head_dim;
                let grid = MTLSize {
                    width: (count),
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                    height: 1,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
                enc.end();
            }
            cb.commit();
            cb.waitUntilCompleted();

            return self.o_proj.forward(&concat, registry, queue);
        }

        // Prefill path (seq_len > 1): pack into batch-major, then interleave
        let head_elems = seq_len * head_dim;
        let packed = Array::zeros(dev, &[num_heads * seq_len, head_dim], q.dtype());
        {
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
            let cb = queue.commandBuffer().unwrap();
            for (h, head_out) in attn_outputs.iter().enumerate() {
                let dst_offset = h * head_elems * elem_size;
                let raw_enc = cb.computeCommandEncoder().unwrap();
                let enc = ComputePass::new(&raw_enc);
                enc.set_pipeline(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset());
                enc.set_buffer(1, Some(packed.metal_buffer()), dst_offset);
                let count = head_elems;
                let grid = MTLSize {
                    width: (count),
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                    height: 1,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
                enc.end();
            }
            cb.commit();
            cb.waitUntilCompleted();
        }
        // Single interleave dispatch: [num_heads, seq_len, head_dim] → [seq_len, hidden_size]
        let concat = ops::rope::interleave_heads(registry, &packed, num_heads, seq_len, queue)?;

        // Output projection
        self.o_proj.forward(&concat, registry, queue)
    }

    /// Forward pass from pre-projected Q, K, V tensors (no QKV projection, no O projection).
    ///
    /// Used by tensor-parallel forward: each rank computes its local QKV shard,
    /// then calls this to run RoPE + SDPA + head concatenation on local heads.
    /// The caller handles O projection separately (via RowParallelLinear).
    ///
    /// - `q`: [seq_len, local_num_heads * head_dim] — pre-projected Q for this rank's heads
    /// - `k`: [seq_len, local_num_kv_heads * head_dim]
    /// - `v`: [seq_len, local_num_kv_heads * head_dim]
    /// - `local_num_heads`, `local_num_kv_heads`: head counts for this rank
    ///
    /// Returns: [seq_len, local_num_heads * head_dim] — concatenated attention output
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn forward_from_qkv(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        local_num_heads: usize,
        local_num_kv_heads: usize,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let head_dim = self.config.head_dim;
        let seq_len = q.shape()[0];
        let repeats = local_num_heads / local_num_kv_heads;
        let elem_size = q.dtype().size_of();

        // RoPE offset from cache position
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        let q_heads: Vec<Array>;
        let k_heads: Vec<Array>;
        let v_heads: Vec<Array>;

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            let q_batched = ops::rope::rope_multihead(
                registry,
                q,
                cos,
                sin,
                local_num_heads,
                rope_offset,
                q.strides()[0],
                queue,
            )?;
            let k_batched = ops::rope::rope_multihead(
                registry,
                k,
                cos,
                sin,
                local_num_kv_heads,
                rope_offset,
                k.strides()[0],
                queue,
            )?;
            let v_batched = ops::rope::deinterleave_heads(
                registry,
                v,
                local_num_kv_heads,
                v.strides()[0],
                queue,
            )?;

            let head_elems = seq_len * head_dim;
            q_heads = (0..local_num_heads)
                .map(|h| {
                    q_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        q_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            k_heads = (0..local_num_kv_heads)
                .map(|h| {
                    k_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        k_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            v_heads = (0..local_num_kv_heads)
                .map(|h| {
                    v_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        v_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
        } else {
            let q_batched =
                ops::rope::deinterleave_heads(registry, q, local_num_heads, q.strides()[0], queue)?;
            let k_batched = ops::rope::deinterleave_heads(
                registry,
                k,
                local_num_kv_heads,
                k.strides()[0],
                queue,
            )?;
            let v_batched = ops::rope::deinterleave_heads(
                registry,
                v,
                local_num_kv_heads,
                v.strides()[0],
                queue,
            )?;

            let head_elems = seq_len * head_dim;
            q_heads = (0..local_num_heads)
                .map(|h| {
                    q_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        q_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            k_heads = (0..local_num_kv_heads)
                .map(|h| {
                    k_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        k_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
            v_heads = (0..local_num_kv_heads)
                .map(|h| {
                    v_batched.view(
                        vec![seq_len, head_dim],
                        vec![head_dim, 1],
                        v_batched.offset() + h * head_elems * elem_size,
                    )
                })
                .collect();
        }

        // Append to KV cache
        let (k_final, v_final, total_seq) = match cache {
            Some(ref mut c) => {
                c.append(k_heads, v_heads, seq_len, registry, queue)?;
                let kf: Vec<Array> = (0..local_num_kv_heads).map(|h| c.cached_keys(h)).collect();
                let vf: Vec<Array> = (0..local_num_kv_heads)
                    .map(|h| c.cached_values(h))
                    .collect();
                let ts = c.seq_len;
                (kf, vf, ts)
            }
            None => (k_heads, v_heads, seq_len),
        };

        // SDPA
        let scale = 1.0 / (head_dim as f32).sqrt();
        let dev = registry.device().raw();

        let attn_outputs = if head_dim <= 256 {
            ops::sdpa::sdpa_batched(
                registry, &q_heads, &k_final, &v_final, mask, scale, false, queue,
            )?
        } else {
            let mut outputs: Vec<Array> = Vec::with_capacity(local_num_heads);
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

        // Concatenate heads → [seq_len, local_num_heads * head_dim]
        let local_hidden = local_num_heads * head_dim;

        if seq_len == 1 {
            let concat = Array::zeros(dev, &[seq_len, local_hidden], q.dtype());
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
            let head_bytes = head_dim * elem_size;
            let cb = queue.commandBuffer().unwrap();
            for (h, head_out) in attn_outputs.iter().enumerate() {
                let dst_col_offset = h * head_bytes;
                let raw_enc = cb.computeCommandEncoder().unwrap();
                let enc = ComputePass::new(&raw_enc);
                enc.set_pipeline(&pipeline);
                enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset());
                enc.set_buffer(1, Some(concat.metal_buffer()), dst_col_offset);
                let count = head_dim;
                let grid = MTLSize {
                    width: (count),
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                    height: 1,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
                enc.end();
            }
            cb.commit();
            cb.waitUntilCompleted();
            Ok(concat)
        } else {
            let head_elems = seq_len * head_dim;
            let packed = Array::zeros(dev, &[local_num_heads * seq_len, head_dim], q.dtype());
            {
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
                let cb = queue.commandBuffer().unwrap();
                for (h, head_out) in attn_outputs.iter().enumerate() {
                    let dst_offset = h * head_elems * elem_size;
                    let raw_enc = cb.computeCommandEncoder().unwrap();
                    let enc = ComputePass::new(&raw_enc);
                    enc.set_pipeline(&pipeline);
                    enc.set_buffer(0, Some(head_out.metal_buffer()), head_out.offset());
                    enc.set_buffer(1, Some(packed.metal_buffer()), dst_offset);
                    let count = head_elems;
                    let grid = MTLSize {
                        width: (count),
                        height: 1,
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: (std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), count)),
                        height: 1,
                        depth: 1,
                    };
                    enc.dispatch_threads(grid, tg);
                    enc.end();
                }
                cb.commit();
                cb.waitUntilCompleted();
            }
            let concat =
                ops::rope::interleave_heads(registry, &packed, local_num_heads, seq_len, queue)?;
            Ok(concat)
        }
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

    /// Access the merged QKV weight (if prepared).
    pub fn qkv_merged_weight(&self) -> Option<&Array> {
        self.qkv_merged_weight.as_ref()
    }

    /// Access the O-projection weight.
    pub fn o_proj_weight(&self) -> Option<&Array> {
        self.o_proj.weight()
    }

    /// Shard this attention layer's weights for Tensor Parallelism.
    ///
    /// Column-parallel: Q, K, V projection weights are sharded by output rows
    /// (each rank gets its local head slice).
    /// Row-parallel: O projection weight is sharded by input columns
    /// (each rank holds columns for its local heads).
    ///
    /// Also updates the config to reflect the local head counts.
    ///
    /// # Panics
    /// Panics if weights are not loaded or head counts are not divisible by `world_size`.
    #[cfg(feature = "distributed")]
    pub(crate) fn shard_for_tp(&mut self, rank: u32, world_size: u32) -> Result<(), KernelError> {
        use crate::parallel::{ColumnParallelLinear, RowParallelLinear};

        if world_size <= 1 {
            return Ok(());
        }
        if self.config.num_heads % (world_size as usize) != 0 {
            return Err(KernelError::InvalidShape(format!(
                "num_heads ({}) not divisible by world_size ({})",
                self.config.num_heads, world_size
            )));
        }
        if self.config.num_kv_heads % (world_size as usize) != 0 {
            return Err(KernelError::InvalidShape(format!(
                "num_kv_heads ({}) not divisible by world_size ({})",
                self.config.num_kv_heads, world_size
            )));
        }

        let local_num_heads = self.config.num_heads / (world_size as usize);
        let local_num_kv_heads = self.config.num_kv_heads / (world_size as usize);
        let head_dim = self.config.head_dim;
        let hidden_size = self.config.num_heads * head_dim;
        let local_q_out = local_num_heads * head_dim;
        let local_kv_out = local_num_kv_heads * head_dim;

        // Shard Q weight: [full_q_out, hidden] → [local_q_out, hidden]
        let q_w = self.q_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("shard_for_tp: q_proj weight not loaded".into())
        })?;
        let q_shard = ColumnParallelLinear::shard_weight(q_w, rank, world_size);
        self.q_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: local_q_out,
                has_bias: false,
            },
            q_shard,
            None,
        )?;

        // Shard K weight
        let k_w = self.k_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("shard_for_tp: k_proj weight not loaded".into())
        })?;
        let k_shard = ColumnParallelLinear::shard_weight(k_w, rank, world_size);
        self.k_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: local_kv_out,
                has_bias: false,
            },
            k_shard,
            None,
        )?;

        // Shard V weight
        let v_w = self.v_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("shard_for_tp: v_proj weight not loaded".into())
        })?;
        let v_shard = ColumnParallelLinear::shard_weight(v_w, rank, world_size);
        self.v_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_size,
                out_features: local_kv_out,
                has_bias: false,
            },
            v_shard,
            None,
        )?;

        // Shard O weight: row-parallel — [hidden, hidden] → [hidden, local_hidden]
        let o_w = self.o_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("shard_for_tp: o_proj weight not loaded".into())
        })?;
        let o_shard = RowParallelLinear::shard_weight(o_w, rank, world_size);
        let local_o_in = hidden_size / (world_size as usize);
        self.o_proj = Linear::from_arrays(
            LinearConfig {
                in_features: local_o_in,
                out_features: hidden_size,
                has_bias: false,
            },
            o_shard,
            None,
        )?;

        // Update config to local head counts
        self.config.num_heads = local_num_heads;
        self.config.num_kv_heads = local_num_kv_heads;

        // Invalidate merged QKV weights (they're for the full model)
        self.qkv_merged_weight = None;
        self.qkv_merged_weight_t = None;

        Ok(())
    }

    // -------------------------------------------------------------------
    // ExecGraph path
    // -------------------------------------------------------------------

    /// Pre-cache transposed weights for all projection layers.
    ///
    /// Call once after weight loading. Eliminates per-pass weight copies
    /// in the ExecGraph path.
    pub fn prepare_weights_for_graph(
        &mut self,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        self.q_proj.prepare_weight_t(registry, queue)?;
        self.k_proj.prepare_weight_t(registry, queue)?;
        self.v_proj.prepare_weight_t(registry, queue)?;
        self.o_proj.prepare_weight_t(registry, queue)?;
        Ok(())
    }

    /// Single-CB decode path for seq_len=1.
    ///
    /// Encodes ALL attention ops into one command buffer with minimal dispatches:
    /// - Zero-copy head views instead of per-head copy_into_cb (saves 48 dispatches)
    /// - Batched RoPE: reshape Q/K to 3D and call rope once each (saves 38 dispatches)
    /// - Blit concat: 1 blit encoder instead of 32 compute encoders
    ///
    /// Does NOT commit or wait — the caller manages the CB lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_single_cb(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;

        // RMS norm
        let normed =
            ops::rms_norm::rms_norm_into_cb(registry, x, Some(norm_weight), rms_norm_eps, cb)?;

        // Q/K/V projections (all into same CB)
        let q = self.q_proj.forward_into_cb(&normed, registry, cb)?;
        let k = self.k_proj.forward_into_cb(&normed, registry, cb)?;
        let v = self.v_proj.forward_into_cb(&normed, registry, cb)?;

        let elem_size = q.dtype().size_of();
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        // --- Head split + optional RoPE ---
        // For seq_len=1, each head's data is contiguous in memory:
        // Q is [1, num_heads * head_dim], head h starts at offset + h * head_dim * elem_size
        let mut q_heads: Vec<Array>;
        let mut k_heads: Vec<Array>;

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            // Batched RoPE: reshape to [num_heads, 1, head_dim], apply rope once
            let q_3d = Array::new(
                q.metal_buffer().to_owned(),
                vec![num_heads, 1, head_dim],
                vec![head_dim, head_dim, 1],
                q.dtype(),
                q.offset(),
            );
            let q_roped = ops::rope::rope_ext_into_cb(
                registry,
                &q_3d,
                cos,
                sin,
                rope_offset,
                1.0,
                false,
                true,
                cb,
            )?;
            // Split back into per-head views
            q_heads = Vec::with_capacity(num_heads);
            for h in 0..num_heads {
                q_heads.push(Array::new(
                    q_roped.metal_buffer().to_owned(),
                    vec![1, head_dim],
                    vec![head_dim, 1],
                    q_roped.dtype(),
                    q_roped.offset() + h * head_dim * elem_size,
                ));
            }

            let k_3d = Array::new(
                k.metal_buffer().to_owned(),
                vec![num_kv_heads, 1, head_dim],
                vec![head_dim, head_dim, 1],
                k.dtype(),
                k.offset(),
            );
            let k_roped = ops::rope::rope_ext_into_cb(
                registry,
                &k_3d,
                cos,
                sin,
                rope_offset,
                1.0,
                false,
                true,
                cb,
            )?;
            k_heads = Vec::with_capacity(num_kv_heads);
            for h in 0..num_kv_heads {
                k_heads.push(Array::new(
                    k_roped.metal_buffer().to_owned(),
                    vec![1, head_dim],
                    vec![head_dim, 1],
                    k_roped.dtype(),
                    k_roped.offset() + h * head_dim * elem_size,
                ));
            }
        } else {
            // No RoPE — just create zero-copy views
            q_heads = Vec::with_capacity(num_heads);
            for h in 0..num_heads {
                q_heads.push(Array::new(
                    q.metal_buffer().to_owned(),
                    vec![1, head_dim],
                    vec![head_dim, 1],
                    q.dtype(),
                    q.offset() + h * head_dim * elem_size,
                ));
            }
            k_heads = Vec::with_capacity(num_kv_heads);
            for h in 0..num_kv_heads {
                k_heads.push(Array::new(
                    k.metal_buffer().to_owned(),
                    vec![1, head_dim],
                    vec![head_dim, 1],
                    k.dtype(),
                    k.offset() + h * head_dim * elem_size,
                ));
            }
        }

        // V heads — zero-copy views (no RoPE on V)
        let mut v_heads: Vec<Array> = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            v_heads.push(Array::new(
                v.metal_buffer().to_owned(),
                vec![1, head_dim],
                vec![head_dim, 1],
                v.dtype(),
                v.offset() + h * head_dim * elem_size,
            ));
        }

        // KV cache append
        let (k_final, v_final) = match cache {
            Some(ref mut c) => {
                c.append_into_cb(k_heads, v_heads, 1, registry, cb)?;
                let kf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_keys(h)).collect();
                let vf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_values(h)).collect();
                (kf, vf)
            }
            None => (k_heads, v_heads),
        };

        // SDPA (per-head, batched into same CB)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_outputs = ops::sdpa::sdpa_batched_into_cb(
            registry, &q_heads, &k_final, &v_final, mask, scale, cb,
        )?;

        // Blit concat: 1 blit encoder instead of 32 compute encoders
        let dev = registry.device().raw();
        let concat = Array::zeros(dev, &[1, hidden_size], q.dtype());
        let head_bytes = head_dim * elem_size;
        let blit = cb.blitCommandEncoder().unwrap();
        for (h, head_out) in attn_outputs.iter().enumerate() {
            let src_offset = head_out.offset();
            let dst_offset = h * head_bytes;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    head_out.metal_buffer(),
                    src_offset,
                    concat.metal_buffer(),
                    dst_offset,
                    head_bytes,
                );
            }
        }
        blit.endEncoding();

        // O projection
        self.o_proj.forward_into_cb(&concat, registry, cb)
    }

    /// Single-CB forward pass for prefill (seq_len >= 1).
    ///
    /// Encodes the entire attention block into the provided command buffer:
    ///   RMS norm → Q/K/V projections → deinterleave + RoPE → KV cache append
    ///   → SDPA → pack heads → interleave → O projection
    ///
    /// **Requires** `prepare_weights_for_graph()` to have been called beforehand
    /// so that projection weights are pre-transposed for GEMM.
    ///
    /// KV cache handling: uses `append_into_cb` which encodes copy dispatches
    /// into the same command buffer (no separate commit/wait).
    ///
    /// Does NOT commit or wait — the caller manages the CB lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_into_cb(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>, // unused: all kernels handle causal masking via is_causal FC
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        let seq_len = x.shape()[0];

        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let v_dim = num_kv_heads * head_dim;
        let (q, k, v) = if let Some(ref qkv_wt) = self.qkv_merged_weight_t {
            // Fused RMSNorm + QKV GEMM (two-pass: inv_rms then norm-GEMM).
            // Eliminates the intermediate [M, K] normalized tensor, saving one
            // full read+write of hidden state.  Uses MlxArch 64×64 tile with
            // has_norm=true function constant.
            let x_2d = if x.ndim() == 1 {
                x.reshape(vec![1, x.shape()[0]])?
            } else {
                x.reshape(vec![x.shape()[0], x.shape()[1]])?
            };
            let qkv = ops::matmul::matmul_norm_gemm_into_cb(
                registry,
                &x_2d,
                qkv_wt,
                norm_weight,
                rms_norm_eps,
                cb,
            )?;
            let total_out = q_dim + k_dim + v_dim;
            let elem_size = qkv.dtype().size_of();
            let q_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, q_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset(),
            );
            let k_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, k_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset() + q_dim * elem_size,
            );
            let v_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, v_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset() + (q_dim + k_dim) * elem_size,
            );
            (q_view, k_view, v_view)
        } else {
            // Non-merged path: separate norm then per-projection GEMM.
            let normed =
                ops::rms_norm::rms_norm_into_cb(registry, x, Some(norm_weight), rms_norm_eps, cb)?;
            let normed_2d = if normed.ndim() == 1 {
                normed.reshape(vec![1, normed.shape()[0]])?
            } else {
                normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
            };
            let wq_t = self.q_proj.weight_transposed_contiguous()?;
            let wk_t = self.k_proj.weight_transposed_contiguous()?;
            let wv_t = self.v_proj.weight_transposed_contiguous()?;
            ops::fused::batched_qkv_proj_into(registry, &normed_2d, &wq_t, &wk_t, &wv_t, cb)?
        };

        let rope_offset = cache.seq_len as u32;

        // Deinterleave + RoPE for Q and K, then deinterleave V.
        // All three are deinterleaved before SDPA so V is contiguous — strided V
        // had terrible cache line utilization (2.8%: head_dim=128 vs stride=4608).
        // Output layout: [num_heads * seq_len, head_dim] (batch-major)
        let q_batched;
        let k_batched;
        let v_batched;

        // Use strides()[0] as input_row_stride — handles both merged QKV
        // (stride = total_qkv) and separate GEMM (stride = shape[1]) paths.
        let q_row_stride = q.strides()[0];
        let k_row_stride = k.strides()[0];
        let v_row_stride = v.strides()[0];

        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            // Check if Q, K, V share the same buffer (merged QKV path)
            if std::ptr::eq(q.metal_buffer(), k.metal_buffer())
                && std::ptr::eq(k.metal_buffer(), v.metal_buffer())
            {
                // Fused Q+K RoPE + V deinterleave in a single dispatch
                let qkv_result = ops::rope::rope_qkv_batch_into_cb(
                    registry,
                    &q, // Q view into merged buffer (offset = 0, stride = total_qkv)
                    cos,
                    sin,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rope_offset,
                    q_row_stride, // same as k/v row stride for merged buffer
                    cb,
                )?;
                q_batched = qkv_result.0;
                k_batched = qkv_result.1;
                v_batched = qkv_result.2;
            } else {
                // Separate Q/K/V buffers — use individual dispatches
                q_batched = ops::rope::rope_multihead_into_cb(
                    registry,
                    &q,
                    cos,
                    sin,
                    num_heads,
                    rope_offset,
                    q_row_stride,
                    cb,
                )?;
                k_batched = ops::rope::rope_multihead_into_cb(
                    registry,
                    &k,
                    cos,
                    sin,
                    num_kv_heads,
                    rope_offset,
                    k_row_stride,
                    cb,
                )?;
                v_batched = ops::rope::deinterleave_heads_into_cb(
                    registry,
                    &v,
                    num_kv_heads,
                    v_row_stride,
                    cb,
                )?;
            }
        } else {
            q_batched =
                ops::rope::deinterleave_heads_into_cb(registry, &q, num_heads, q_row_stride, cb)?;
            k_batched = ops::rope::deinterleave_heads_into_cb(
                registry,
                &k,
                num_kv_heads,
                k_row_stride,
                cb,
            )?;
            v_batched = ops::rope::deinterleave_heads_into_cb(
                registry,
                &v,
                num_kv_heads,
                v_row_stride,
                cb,
            )?;
        }

        // ── 4) KV cache: bypass on initial prefill (cache empty) ──
        let needs_initial_append = cache.seq_len == 0;
        // V is always contiguous after deinterleave — no stride overrides needed.
        let (k_for_sdpa, v_for_sdpa, kv_stride, total_seq) = if needs_initial_append {
            // K/V: deinterleave outputs are contiguous — pass directly to SDPA
            let k_view = k_batched.view(
                k_batched.shape().to_vec(),
                k_batched.strides().to_vec(),
                k_batched.offset(),
            );
            let v_view = v_batched.view(
                v_batched.shape().to_vec(),
                v_batched.strides().to_vec(),
                v_batched.offset(),
            );
            (k_view, v_view, None, seq_len)
        } else {
            // Non-initial: append to cache, then use slab for SDPA
            cache.append_batched_into_cb(&k_batched, &v_batched, seq_len, registry, cb)?;
            let total_seq = cache.seq_len;
            let k_slab = cache.keys_slab_view().ok_or_else(|| {
                KernelError::InvalidShape(
                    "forward_prefill_into_cb: cache has no slab layout (use preallocated cache)"
                        .into(),
                )
            })?;
            let v_slab = cache.values_slab_view().ok_or_else(|| {
                KernelError::InvalidShape(
                    "forward_prefill_into_cb: cache has no slab layout (use preallocated cache)"
                        .into(),
                )
            })?;
            let kv_stride = if cache.max_seq_len() != total_seq {
                Some(cache.max_seq_len())
            } else {
                None // contiguous, no stride override needed
            };
            (k_slab, v_slab, kv_stride, total_seq)
        };

        // Single-dispatch GQA SDPA over all heads using contiguous slabs.
        // Q slab: [num_heads * seq_len * head_dim] from deinterleave/RoPE
        // K/V slabs: from cache slab (or direct RoPE output on initial prefill)
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 4-way SDPA dispatch for prefill (f16, D=128, seq_len>1):
        //   1. NAX      — highest throughput (BQ=64, BK=32, register-only)
        //   2. MMA BK=32 — fewer K-loop iterations, better for longer sequences
        //   3. MMA BK=16 — short sequences or fallback
        //   4. GQA slab  — scalar fallback for non-f16 or non-D128
        //
        // All MMA/NAX kernels handle causal masking internally via is_causal FC 300,
        // so no explicit mask is needed for standard causal attention.
        let is_f16_d128 = q_batched.dtype() == DType::Float16 && head_dim == 128 && seq_len > 1;

        // NAX: best throughput. Kernel handles causal via is_causal FC — no explicit mask needed.
        let use_nax = is_f16_d128 && registry.device().tuning().supports_nax;
        // BK=32: fewer K-loop iterations, wins when KV sequence is long enough.
        let use_mma_bk32 = is_f16_d128 && !use_nax && total_seq >= 256;

        // All SDPA kernels (MMA + NAX) now write seq-major output.
        // Scalar fallback also writes seq-major when is_f16_d128 is false,
        // but that path uses head-major — handled below.
        let seq_major_output = true;

        let attn_slab = if use_nax {
            ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                scale,
                true, // is_causal
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                cb,
            )?
        } else if use_mma_bk32 {
            // BK=32: fewer K-loop iterations, better for longer sequences
            ops::sdpa::sdpa_prefill_mma_bk32_f16_into_cb(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None, // kernel handles causal internally
                scale,
                true, // is_causal
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                cb,
            )?
        } else if is_f16_d128 {
            // MMA BK=16: short sequences or fallback
            ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None, // kernel handles causal internally
                scale,
                true, // is_causal
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                cb,
            )?
        } else {
            ops::sdpa::sdpa_prefill_gqa_slab_into_cb(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None, // kernel handles causal internally
                scale,
                true, // is_causal
                cb,
            )?
        };

        // KV cache append for initial prefill (cache was empty).
        // V is already deinterleaved (done before SDPA above).
        if needs_initial_append {
            cache.append_batched_into_cb(&k_batched, &v_batched, seq_len, registry, cb)?;
        }

        // All kernels (MMA + NAX) now write seq-major [seq_len, num_heads * head_dim].
        // No interleave dispatch needed.
        let concat = if seq_major_output {
            // Already seq-major: just reshape to [seq_len, hidden_size]
            attn_slab.view(
                vec![seq_len, hidden_size],
                vec![hidden_size, 1],
                attn_slab.offset(),
            )
        } else {
            // Scalar path: head-major → need interleave (kept for non-f16/non-d128 fallback)
            let packed = attn_slab.view(
                vec![num_heads * seq_len, head_dim],
                vec![head_dim, 1],
                attn_slab.offset(),
            );
            if seq_len == 1 {
                packed.view(vec![1, hidden_size], vec![hidden_size, 1], packed.offset())
            } else {
                ops::rope::interleave_heads_into_cb(registry, &packed, num_heads, seq_len, cb)?
            }
        };

        // O projection
        self.o_proj.forward_into_cb(&concat, registry, cb)
    }

    // -----------------------------------------------------------------------
    // Single-encoder path: all ops share one ComputeCommandEncoder
    // -----------------------------------------------------------------------

    /// Forward pass using a single pre-existing compute command encoder.
    ///
    /// Identical logic to `forward_prefill_into_cb` but dispatches all kernels
    /// via a shared encoder, eliminating per-op encoder create/destroy overhead
    /// (~300-500us/layer).
    ///
    /// **Prerequisites:**
    /// - `prepare_merged_qkv_transposed()` must have been called (merged QKV weight).
    /// - `o_proj` must have pre-cached transposed weight.
    /// - The caller manages the encoder lifecycle (creation and `end_encoding()`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_into_encoder(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        encoder: ComputePass<'_>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        let seq_len = x.shape()[0];

        // RMS norm (pre-attention)
        let normed =
            ops::rms_norm::rms_norm_encode(registry, x, Some(norm_weight), rms_norm_eps, encoder)?;
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let v_dim = num_kv_heads * head_dim;
        let (q, k, v) = {
            let normed_2d = if normed.ndim() == 1 {
                normed.reshape(vec![1, normed.shape()[0]])?
            } else {
                normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
            };

            if let Some(ref qkv_wt) = self.qkv_merged_weight_t {
                let qkv = ops::matmul::matmul_encode(registry, &normed_2d, qkv_wt, encoder)?;
                let total_out = q_dim + k_dim + v_dim;
                let elem_size = qkv.dtype().size_of();
                let q_view = Array::new(
                    qkv.metal_buffer().to_owned(),
                    vec![seq_len, q_dim],
                    vec![total_out, 1],
                    qkv.dtype(),
                    qkv.offset(),
                );
                let k_view = Array::new(
                    qkv.metal_buffer().to_owned(),
                    vec![seq_len, k_dim],
                    vec![total_out, 1],
                    qkv.dtype(),
                    qkv.offset() + q_dim * elem_size,
                );
                let v_view = Array::new(
                    qkv.metal_buffer().to_owned(),
                    vec![seq_len, v_dim],
                    vec![total_out, 1],
                    qkv.dtype(),
                    qkv.offset() + (q_dim + k_dim) * elem_size,
                );
                (q_view, k_view, v_view)
            } else {
                return Err(KernelError::InvalidShape(
                    "forward_prefill_into_encoder requires merged QKV weight \
                     (call prepare_merged_qkv_transposed first)"
                        .into(),
                ));
            }
        };

        let rope_offset = cache.seq_len as u32;

        // Deinterleave + RoPE for Q and K, then deinterleave V.
        // All three are deinterleaved before SDPA so V is contiguous — strided V
        // had terrible cache line utilization (2.8%: head_dim=128 vs stride=4608).
        let q_row_stride = q.strides()[0];
        let k_row_stride = k.strides()[0];
        let v_row_stride = v.strides()[0];

        let (q_batched, k_batched) = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            let q_b = ops::rope::rope_multihead_encode(
                registry,
                &q,
                cos,
                sin,
                num_heads,
                rope_offset,
                q_row_stride,
                encoder,
            )?;
            let k_b = ops::rope::rope_multihead_encode(
                registry,
                &k,
                cos,
                sin,
                num_kv_heads,
                rope_offset,
                k_row_stride,
                encoder,
            )?;
            (q_b, k_b)
        } else {
            let q_b = ops::rope::deinterleave_heads_encode(
                registry,
                &q,
                num_heads,
                q_row_stride,
                encoder,
            )?;
            let k_b = ops::rope::deinterleave_heads_encode(
                registry,
                &k,
                num_kv_heads,
                k_row_stride,
                encoder,
            )?;
            (q_b, k_b)
        };
        // Deinterleave V — always done before SDPA for contiguous memory access
        let v_batched = ops::rope::deinterleave_heads_encode(
            registry,
            &v,
            num_kv_heads,
            v_row_stride,
            encoder,
        )?;

        // KV cache: bypass on initial prefill (cache empty)
        let needs_initial_append = cache.seq_len == 0;
        // V is always contiguous after deinterleave — no stride overrides needed.
        let (k_for_sdpa, v_for_sdpa, kv_stride, total_seq) = if needs_initial_append {
            // K/V: deinterleave outputs are contiguous — pass directly to SDPA
            let k_view = k_batched.view(
                k_batched.shape().to_vec(),
                k_batched.strides().to_vec(),
                k_batched.offset(),
            );
            let v_view = v_batched.view(
                v_batched.shape().to_vec(),
                v_batched.strides().to_vec(),
                v_batched.offset(),
            );
            (k_view, v_view, None, seq_len)
        } else {
            // Non-initial: append to cache, then use slab for SDPA
            cache.append_batched_encode(&k_batched, &v_batched, seq_len, registry, encoder)?;
            let total_seq = cache.seq_len;
            let k_slab = cache.keys_slab_view().ok_or_else(|| {
                KernelError::InvalidShape(
                    "forward_prefill_into_encoder: cache has no slab layout".into(),
                )
            })?;
            let v_slab = cache.values_slab_view().ok_or_else(|| {
                KernelError::InvalidShape(
                    "forward_prefill_into_encoder: cache has no slab layout".into(),
                )
            })?;
            let kv_stride = if cache.max_seq_len() != total_seq {
                Some(cache.max_seq_len())
            } else {
                None
            };
            (k_slab, v_slab, kv_stride, total_seq)
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let is_f16_d128 = q_batched.dtype() == DType::Float16 && head_dim == 128 && seq_len > 1;
        let use_nax = is_f16_d128 && registry.device().tuning().supports_nax;
        let use_mma_bk32 = is_f16_d128 && !use_nax && total_seq >= 256;
        // All SDPA kernels (MMA + NAX) now write seq-major output.
        let seq_major_output = true;

        let attn_slab = if use_nax {
            ops::sdpa::sdpa_prefill_nax_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                scale,
                true,
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                encoder,
            )?
        } else if use_mma_bk32 {
            ops::sdpa::sdpa_prefill_mma_bk32_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                encoder,
            )?
        } else if is_f16_d128 {
            ops::sdpa::sdpa_prefill_mma_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                None, // v_head_stride: V is contiguous after deinterleave
                None, // v_row_stride: V is contiguous after deinterleave
                encoder,
            )?
        } else {
            ops::sdpa::sdpa_prefill_gqa_slab_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                encoder,
            )?
        };

        // KV cache append for initial prefill (cache was empty).
        // V is already deinterleaved (done before SDPA above).
        if needs_initial_append {
            cache.append_batched_encode(&k_batched, &v_batched, seq_len, registry, encoder)?;
        }

        // All kernels (MMA + NAX) now write seq-major — no interleave dispatch needed.
        let concat = if seq_major_output {
            attn_slab.view(
                vec![seq_len, hidden_size],
                vec![hidden_size, 1],
                attn_slab.offset(),
            )
        } else {
            // Scalar path: head-major → need interleave (kept for non-f16/non-d128 fallback)
            let packed = attn_slab.view(
                vec![num_heads * seq_len, head_dim],
                vec![head_dim, 1],
                attn_slab.offset(),
            );
            if seq_len == 1 {
                packed.view(vec![1, hidden_size], vec![hidden_size, 1], packed.offset())
            } else {
                ops::rope::interleave_heads_encode(registry, &packed, num_heads, seq_len, encoder)?
            }
        };

        // O projection
        self.o_proj.forward_into_encoder(&concat, registry, encoder)
    }

    // -----------------------------------------------------------------------
    // 9-dispatch path: merged QKV weight + batched SDPA + fused residual
    // -----------------------------------------------------------------------

    /// Merge Q/K/V projection weights into a single [q_out+k_out+v_out, in_features] matrix.
    ///
    /// Must be called once after weights are loaded and before `forward_decode_9dispatch`.
    pub fn prepare_merged_qkv(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), KernelError> {
        let q_w = self.q_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape(
                "Attention::prepare_merged_qkv: q_proj weight not loaded".into(),
            )
        })?;
        let k_w = self.k_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape(
                "Attention::prepare_merged_qkv: k_proj weight not loaded".into(),
            )
        })?;
        let v_w = self.v_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape(
                "Attention::prepare_merged_qkv: v_proj weight not loaded".into(),
            )
        })?;

        // Validate contiguity
        if !q_w.is_contiguous() || !k_w.is_contiguous() || !v_w.is_contiguous() {
            return Err(KernelError::InvalidShape(
                "prepare_merged_qkv: all weights must be contiguous".into(),
            ));
        }
        // Validate matching dtype
        if q_w.dtype() != k_w.dtype() || q_w.dtype() != v_w.dtype() {
            return Err(KernelError::InvalidShape(format!(
                "prepare_merged_qkv: dtype mismatch: Q={:?}, K={:?}, V={:?}",
                q_w.dtype(),
                k_w.dtype(),
                v_w.dtype()
            )));
        }
        // Validate matching in_features (cols)
        if k_w.shape()[1] != q_w.shape()[1] || v_w.shape()[1] != q_w.shape()[1] {
            return Err(KernelError::InvalidShape(format!(
                "prepare_merged_qkv: in_features mismatch: Q={}, K={}, V={}",
                q_w.shape()[1],
                k_w.shape()[1],
                v_w.shape()[1]
            )));
        }

        let q_rows = q_w.shape()[0]; // num_heads * head_dim
        let k_rows = k_w.shape()[0]; // num_kv_heads * head_dim
        let v_rows = v_w.shape()[0]; // num_kv_heads * head_dim
        let cols = q_w.shape()[1]; // hidden_size
        let total_rows = q_rows + k_rows + v_rows;
        let elem_size = q_w.dtype().size_of();
        let total_bytes = total_rows * cols * elem_size;

        let buf = device
            .newBufferWithLength_options(total_bytes, MTLResourceOptions::StorageModeShared)
            .unwrap();

        unsafe {
            let dst = buf.contents().as_ptr() as *mut u8;
            // Copy Q weight
            let q_src = (q_w.metal_buffer().contents().as_ptr() as *const u8).add(q_w.offset());
            std::ptr::copy_nonoverlapping(q_src, dst, q_rows * cols * elem_size);
            // Copy K weight
            let k_src = (k_w.metal_buffer().contents().as_ptr() as *const u8).add(k_w.offset());
            std::ptr::copy_nonoverlapping(
                k_src,
                dst.add(q_rows * cols * elem_size),
                k_rows * cols * elem_size,
            );
            // Copy V weight
            let v_src = (v_w.metal_buffer().contents().as_ptr() as *const u8).add(v_w.offset());
            std::ptr::copy_nonoverlapping(
                v_src,
                dst.add((q_rows + k_rows) * cols * elem_size),
                v_rows * cols * elem_size,
            );
        }

        self.qkv_merged_weight = Some(Array::new(
            buf,
            vec![total_rows, cols],
            vec![cols, 1],
            q_w.dtype(),
            0,
        ));

        // Also create transposed merged weight [cols, total_rows] for prefill GEMM.
        // Column-concatenate the transposed individual weights.
        let buf_t = device
            .newBufferWithLength_options(total_bytes, MTLResourceOptions::StorageModeShared)
            .unwrap();
        unsafe {
            let dst = buf_t.contents().as_ptr() as *mut u8;
            // Transpose: dst[col][row] = src[row][col]
            // src is [total_rows, cols], dst is [cols, total_rows]
            let src = self
                .qkv_merged_weight
                .as_ref()
                .unwrap()
                .metal_buffer()
                .contents()
                .as_ptr() as *const u8;
            for r in 0..total_rows {
                for c in 0..cols {
                    let src_idx = (r * cols + c) * elem_size;
                    let dst_idx = (c * total_rows + r) * elem_size;
                    std::ptr::copy_nonoverlapping(src.add(src_idx), dst.add(dst_idx), elem_size);
                }
            }
        }
        self.qkv_merged_weight_t = Some(Array::new(
            buf_t,
            vec![cols, total_rows],
            vec![total_rows, 1],
            q_w.dtype(),
            0,
        ));

        Ok(())
    }

    /// Convert all static weights to `StorageModePrivate` (GPU-only).
    ///
    /// Call after loading weights and before the inference loop.
    pub fn prepare_weights_private(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) {
        self.q_proj.convert_weights_private(device, queue);
        self.k_proj.convert_weights_private(device, queue);
        self.v_proj.convert_weights_private(device, queue);
        self.o_proj.convert_weights_private(device, queue);
        if let Some(w) = self.qkv_merged_weight.take() {
            self.qkv_merged_weight = Some(w.to_private(device, queue));
        }
        if let Some(w) = self.qkv_merged_weight_t.take() {
            self.qkv_merged_weight_t = Some(w.to_private(device, queue));
        }
    }

    /// 9-dispatch forward decode path for a single transformer attention block.
    ///
    /// Dispatches:
    ///   1. rms_norm
    ///   2. merged QKV gemv
    ///   3. batched Q+K rope (1 dispatch for all heads)
    ///   4. batched SDPA decode (all heads, 1 dispatch)
    ///   5. gemv_bias(W_o, attn, x) — O_proj + residual fused
    ///
    /// Requires `prepare_merged_qkv()` to have been called first.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_9dispatch(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        // Guard: decode path requires seq_len=1
        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_decode_9dispatch: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // --- Single encoder for dispatches 1-3 (rms_norm, QKV gemv, rope) ---
        let raw_encoder_a = cb.computeCommandEncoder().unwrap();
        let encoder_a = ComputePass::new(&raw_encoder_a);

        // Dispatch 1: rms_norm
        let x_2d = if x.ndim() == 1 {
            Array::new(
                x.metal_buffer().to_owned(),
                vec![1, x.shape()[0]],
                vec![x.shape()[0], 1],
                x.dtype(),
                x.offset(),
            )
        } else {
            Array::new(
                x.metal_buffer().to_owned(),
                x.shape().to_vec(),
                x.strides().to_vec(),
                x.dtype(),
                x.offset(),
            )
        };
        let normed = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &x_2d,
            Some(norm_weight),
            rms_norm_eps,
            encoder_a,
        )?;
        // Memory barrier: ensure normed is visible to dispatch 2
        rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());

        // Dispatch 2: merged QKV gemv
        let qkv_weight = self.qkv_merged_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape(
                "Attention::forward_decode_9dispatch: call prepare_merged_qkv() first".into(),
            )
        })?;
        let normed_vec = Array::new(
            normed.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            normed.dtype(),
            normed.offset(),
        );
        let qkv = ops::gemv::gemv_into_encoder(registry, qkv_weight, &normed_vec, encoder_a)?;
        let elem_size = qkv.dtype().size_of();
        // qkv is [q_dim + k_dim + v_dim] flat

        let q_dim = num_heads * head_dim; // 4096
        let k_dim = num_kv_heads * head_dim; // 1024

        let rope_offset = cache.seq_len as u32;

        // Dispatch 3: batched Q+K rope (1 dispatch for all heads)
        // Q[q_dim] and K[k_dim] are contiguous in qkv buffer
        let total_rope_heads = num_heads + num_kv_heads;

        // Source for roped Q+K and un-roped V
        let (qk_roped_buf, qk_roped_offset, v_buf, v_offset);
        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            // Memory barrier: ensure qkv is visible to dispatch 3
            rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());
            let qk_3d = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![total_rope_heads, 1, head_dim],
                vec![head_dim, head_dim, 1],
                qkv.dtype(),
                qkv.offset(),
            );
            let qk_roped = ops::rope::rope_ext_into_encoder(
                registry,
                &qk_3d,
                cos,
                sin,
                rope_offset,
                1.0,
                false,
                true,
                encoder_a,
            )?;
            qk_roped_buf = qk_roped.metal_buffer().to_owned();
            qk_roped_offset = qk_roped.offset();
            // V is from the original qkv buffer (no rope)
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        } else {
            // No RoPE — use qkv directly
            qk_roped_buf = qkv.metal_buffer().to_owned();
            qk_roped_offset = qkv.offset();
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        }

        // Memory barrier: ensure rope output visible to KV copy
        rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());

        let k_roped_flat_offset = qk_roped_offset + q_dim * elem_size;

        // KV cache append — create per-head K and V views
        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            k_heads.push(Array::new(
                qk_roped_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                k_roped_flat_offset + h * head_dim * elem_size,
            ));
            v_heads.push(Array::new(
                v_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                v_offset + h * head_dim * elem_size,
            ));
        }
        cache.append_into_encoder(k_heads, v_heads, 1, registry, encoder_a)?;

        // End encoder A after KV append (no separate blit encoder needed)
        encoder_a.end();

        // --- Single encoder for dispatches 4-5 (SDPA, O_proj+residual) ---
        let raw_encoder_b = cb.computeCommandEncoder().unwrap();
        let encoder_b = ComputePass::new(&raw_encoder_b);

        // Dispatch 4: batched SDPA decode (all heads, 1 dispatch)
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("9dispatch: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("9dispatch: no values slab after append".into())
        })?;
        let seq_len = cache.seq_len; // actual cached length after append
        let max_seq = cache.max_seq_len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q flat view for batched SDPA
        let q_flat = Array::new(
            qk_roped_buf.clone(),
            vec![q_dim],
            vec![1],
            qkv.dtype(),
            qk_roped_offset,
        );
        let attn_out = ops::sdpa::sdpa_decode_batched_slab_stride_into_encoder(
            registry,
            &q_flat,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            Some(max_seq),
            None, // no additive mask for decode
            scale,
            encoder_b,
        )?;
        // attn_out is [num_heads * head_dim] = [hidden_size] flat
        // Memory barrier: ensure attn_out is visible to dispatch 5
        rmlx_metal::memory_barrier_scope_buffers(encoder_b.raw());

        // Dispatch 5: gemv_bias(W_o, attn, x) — O_proj + residual add fused
        let o_weight = self.o_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("Attention: o_proj weight not loaded".into())
        })?;
        let attn_vec = Array::new(
            attn_out.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            attn_out.dtype(),
            attn_out.offset(),
        );
        let x_vec = Array::new(
            x.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            x.dtype(),
            x.offset(),
        );
        let h =
            ops::gemv::gemv_bias_into_encoder(registry, o_weight, &attn_vec, &x_vec, encoder_b)?;

        encoder_b.end();

        // Return as [1, hidden_size]
        Ok(Array::new(
            h.metal_buffer().to_owned(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            h.dtype(),
            h.offset(),
        ))
    }

    /// Phase-1 of 9-dispatch decode: D1-D3 in encoder A + KV cache blit.
    ///
    /// Returns `DecodePhase1` containing the intermediate state needed for
    /// `encode_phase2_into()` to encode D4-D5 into a caller-provided encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_phase1(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<DecodePhase1, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_decode_phase1: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // --- Encoder A: dispatches 1-3 (rms_norm, QKV gemv, rope) ---
        let raw_encoder_a = cb.computeCommandEncoder().unwrap();
        let encoder_a = ComputePass::new(&raw_encoder_a);

        let x_2d = if x.ndim() == 1 {
            Array::new(
                x.metal_buffer().to_owned(),
                vec![1, x.shape()[0]],
                vec![x.shape()[0], 1],
                x.dtype(),
                x.offset(),
            )
        } else {
            Array::new(
                x.metal_buffer().to_owned(),
                x.shape().to_vec(),
                x.strides().to_vec(),
                x.dtype(),
                x.offset(),
            )
        };
        let normed = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &x_2d,
            Some(norm_weight),
            rms_norm_eps,
            encoder_a,
        )?;
        rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());

        let qkv_weight = self.qkv_merged_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape(
                "forward_decode_phase1: call prepare_merged_qkv() first".into(),
            )
        })?;
        let normed_vec = Array::new(
            normed.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            normed.dtype(),
            normed.offset(),
        );
        let qkv = ops::gemv::gemv_into_encoder(registry, qkv_weight, &normed_vec, encoder_a)?;
        let elem_size = qkv.dtype().size_of();

        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let rope_offset = cache.seq_len as u32;
        let total_rope_heads = num_heads + num_kv_heads;

        let (qk_roped_buf, qk_roped_offset, v_buf, v_offset);
        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());
            let qk_3d = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![total_rope_heads, 1, head_dim],
                vec![head_dim, head_dim, 1],
                qkv.dtype(),
                qkv.offset(),
            );
            let qk_roped = ops::rope::rope_ext_into_encoder(
                registry,
                &qk_3d,
                cos,
                sin,
                rope_offset,
                1.0,
                false,
                true,
                encoder_a,
            )?;
            qk_roped_buf = qk_roped.metal_buffer().to_owned();
            qk_roped_offset = qk_roped.offset();
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        } else {
            qk_roped_buf = qkv.metal_buffer().to_owned();
            qk_roped_offset = qkv.offset();
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        }

        // Memory barrier: ensure rope output visible to KV copy
        rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());

        // KV cache blit
        let k_roped_flat_offset = qk_roped_offset + q_dim * elem_size;
        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            k_heads.push(Array::new(
                qk_roped_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                k_roped_flat_offset + h * head_dim * elem_size,
            ));
            v_heads.push(Array::new(
                v_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                v_offset + h * head_dim * elem_size,
            ));
        }
        cache.append_into_encoder(k_heads, v_heads, 1, registry, encoder_a)?;

        encoder_a.end();

        Ok(DecodePhase1 {
            qk_roped_buf,
            qk_roped_offset,
            q_dim,
            x: Array::new(
                x.metal_buffer().to_owned(),
                vec![hidden_size],
                vec![1],
                x.dtype(),
                x.offset(),
            ),
            dtype: qkv.dtype(),
        })
    }

    /// Phase-2 of 9-dispatch decode: encode D4-D5 into a caller-supplied encoder.
    ///
    /// Returns `h` — the residual-updated attention output [1, hidden_size].
    pub fn encode_phase2_into(
        &self,
        phase1: &DecodePhase1,
        cache: &LayerKvCache,
        registry: &KernelRegistry,
        encoder: ComputePass<'_>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;

        // D4: batched SDPA decode
        let k_slab = cache
            .keys_slab_view()
            .ok_or_else(|| KernelError::InvalidShape("encode_phase2_into: no keys slab".into()))?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("encode_phase2_into: no values slab".into())
        })?;
        let seq_len = cache.seq_len;
        let max_seq = cache.max_seq_len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_flat = Array::new(
            phase1.qk_roped_buf.clone(),
            vec![phase1.q_dim],
            vec![1],
            phase1.dtype,
            phase1.qk_roped_offset,
        );
        let attn_out = ops::sdpa::sdpa_decode_batched_slab_stride_into_encoder(
            registry,
            &q_flat,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            Some(max_seq),
            None,
            scale,
            encoder,
        )?;
        rmlx_metal::memory_barrier_scope_buffers(encoder.raw());

        // D5: gemv_bias(W_o, attn, x) — O_proj + residual
        let o_weight = self.o_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("Attention: o_proj weight not loaded".into())
        })?;
        let attn_vec = Array::new(
            attn_out.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            attn_out.dtype(),
            attn_out.offset(),
        );
        let h =
            ops::gemv::gemv_bias_into_encoder(registry, o_weight, &attn_vec, &phase1.x, encoder)?;

        Ok(Array::new(
            h.metal_buffer().to_owned(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            h.dtype(),
            h.offset(),
        ))
    }

    /// 9-dispatch attention decode using concurrent encoders for better GPU scheduling.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_concurrent(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        _mask: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        // Guard: decode path requires seq_len=1
        let seq_dim = if x.ndim() == 1 { 1 } else { x.shape()[0] };
        if seq_dim != 1 {
            return Err(KernelError::InvalidShape(format!(
                "forward_decode_9dispatch: requires seq_len=1, got {}",
                seq_dim
            )));
        }

        // --- Single encoder for dispatches 1-3 (rms_norm, QKV gemv, rope) ---
        let raw_encoder_a = rmlx_metal::new_concurrent_encoder(cb);
        let encoder_a = ComputePass::new(&raw_encoder_a);

        // Dispatch 1: rms_norm
        let x_2d = if x.ndim() == 1 {
            Array::new(
                x.metal_buffer().to_owned(),
                vec![1, x.shape()[0]],
                vec![x.shape()[0], 1],
                x.dtype(),
                x.offset(),
            )
        } else {
            Array::new(
                x.metal_buffer().to_owned(),
                x.shape().to_vec(),
                x.strides().to_vec(),
                x.dtype(),
                x.offset(),
            )
        };
        let normed = ops::rms_norm::rms_norm_into_encoder(
            registry,
            &x_2d,
            Some(norm_weight),
            rms_norm_eps,
            encoder_a,
        )?;
        // Memory barrier: ensure normed is visible to dispatch 2
        rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());

        // Dispatch 2: merged QKV gemv
        let qkv_weight = self.qkv_merged_weight.as_ref().ok_or_else(|| {
            KernelError::InvalidShape(
                "Attention::forward_decode_9dispatch: call prepare_merged_qkv() first".into(),
            )
        })?;
        let normed_vec = Array::new(
            normed.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            normed.dtype(),
            normed.offset(),
        );
        let qkv = ops::gemv::gemv_into_encoder(registry, qkv_weight, &normed_vec, encoder_a)?;
        let elem_size = qkv.dtype().size_of();
        // qkv is [q_dim + k_dim + v_dim] flat

        let q_dim = num_heads * head_dim; // 4096
        let k_dim = num_kv_heads * head_dim; // 1024

        let rope_offset = cache.seq_len as u32;

        // Dispatch 3: batched Q+K rope (1 dispatch for all heads)
        // Q[q_dim] and K[k_dim] are contiguous in qkv buffer
        let total_rope_heads = num_heads + num_kv_heads;

        // Source for roped Q+K and un-roped V
        let (qk_roped_buf, qk_roped_offset, v_buf, v_offset);
        if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            // Memory barrier: ensure qkv is visible to dispatch 3
            rmlx_metal::memory_barrier_scope_buffers(encoder_a.raw());
            let qk_3d = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![total_rope_heads, 1, head_dim],
                vec![head_dim, head_dim, 1],
                qkv.dtype(),
                qkv.offset(),
            );
            let qk_roped = ops::rope::rope_ext_into_encoder(
                registry,
                &qk_3d,
                cos,
                sin,
                rope_offset,
                1.0,
                false,
                true,
                encoder_a,
            )?;
            qk_roped_buf = qk_roped.metal_buffer().to_owned();
            qk_roped_offset = qk_roped.offset();
            // V is from the original qkv buffer (no rope)
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        } else {
            // No RoPE — use qkv directly
            qk_roped_buf = qkv.metal_buffer().to_owned();
            qk_roped_offset = qkv.offset();
            v_buf = qkv.metal_buffer().to_owned();
            v_offset = qkv.offset() + (q_dim + k_dim) * elem_size;
        }

        // End encoder A before cache append (which creates its own encoders)
        encoder_a.end();

        let k_roped_flat_offset = qk_roped_offset + q_dim * elem_size;

        // KV cache append — create per-head K and V views
        let mut k_heads = Vec::with_capacity(num_kv_heads);
        let mut v_heads = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            k_heads.push(Array::new(
                qk_roped_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                k_roped_flat_offset + h * head_dim * elem_size,
            ));
            v_heads.push(Array::new(
                v_buf.clone(),
                vec![1, head_dim],
                vec![head_dim, 1],
                qkv.dtype(),
                v_offset + h * head_dim * elem_size,
            ));
        }
        cache.append_into_cb(k_heads, v_heads, 1, registry, cb)?;

        // --- Single encoder for dispatches 4-5 (SDPA, O_proj+residual) ---
        let raw_encoder_b = rmlx_metal::new_concurrent_encoder(cb);
        let encoder_b = ComputePass::new(&raw_encoder_b);

        // Dispatch 4: batched SDPA decode (all heads, 1 dispatch)
        let k_slab = cache.keys_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("9dispatch: no keys slab after append".into())
        })?;
        let v_slab = cache.values_slab_view().ok_or_else(|| {
            KernelError::InvalidShape("9dispatch: no values slab after append".into())
        })?;
        let seq_len = cache.seq_len; // actual cached length after append
        let max_seq = cache.max_seq_len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q flat view for batched SDPA
        let q_flat = Array::new(
            qk_roped_buf.clone(),
            vec![q_dim],
            vec![1],
            qkv.dtype(),
            qk_roped_offset,
        );
        let attn_out = ops::sdpa::sdpa_decode_batched_slab_stride_into_encoder(
            registry,
            &q_flat,
            &k_slab,
            &v_slab,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            Some(max_seq),
            None, // no additive mask for decode
            scale,
            encoder_b,
        )?;
        // attn_out is [num_heads * head_dim] = [hidden_size] flat
        // Memory barrier: ensure attn_out is visible to dispatch 5
        rmlx_metal::memory_barrier_scope_buffers(encoder_b.raw());

        // Dispatch 5: gemv_bias(W_o, attn, x) — O_proj + residual add fused
        let o_weight = self.o_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("Attention: o_proj weight not loaded".into())
        })?;
        let attn_vec = Array::new(
            attn_out.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            attn_out.dtype(),
            attn_out.offset(),
        );
        let x_vec = Array::new(
            x.metal_buffer().to_owned(),
            vec![hidden_size],
            vec![1],
            x.dtype(),
            x.offset(),
        );
        let h =
            ops::gemv::gemv_bias_into_encoder(registry, o_weight, &attn_vec, &x_vec, encoder_b)?;

        encoder_b.end();

        // Return as [1, hidden_size]
        Ok(Array::new(
            h.metal_buffer().to_owned(),
            vec![1, hidden_size],
            vec![hidden_size, 1],
            h.dtype(),
            h.offset(),
        ))
    }
}

// ---------------------------------------------------------------------------
// QuantizedKvCache — memory-efficient KV cache using affine quantization
// ---------------------------------------------------------------------------

/// A quantized array stored as (packed_uint32, scales, biases) tuple.
/// Uses the same MLX affine format as QuantizedWeight but for cache data.
pub struct QuantizedArray {
    /// Packed uint32 data — each uint32 holds `32 / bits` quantized values.
    pub packed: Array,
    /// Per-group scale factors (f32).
    pub scales: Array,
    /// Per-group bias terms (f32).
    pub biases: Array,
    /// Number of elements per quantization group.
    pub group_size: u32,
    /// Bit width of each quantized value (4 or 8).
    pub bits: u32,
}

/// KV cache that stores keys and values in quantized format.
///
/// Each layer's KV heads are quantized to reduce memory consumption.
/// On attention computation, quantized matmul is used directly without
/// explicit dequantization.
///
/// Memory savings: 128 head_dim, 32 KV heads, 8192 seq ->
///   f16: 128MB, q8: 64MB, q4: 32MB
pub struct QuantizedKvCache {
    /// Per-layer, per-head quantized key cache: `[num_layers][num_kv_heads]`
    keys: Vec<Vec<QuantizedArray>>,
    /// Per-layer, per-head quantized value cache: `[num_layers][num_kv_heads]`
    values: Vec<Vec<QuantizedArray>>,
    /// Per-layer unquantized key accumulator (f32): `[num_layers][num_kv_heads]`
    /// Stores the full-precision concatenation so we can re-quantize after append.
    keys_full: Vec<Vec<Vec<f32>>>,
    /// Per-layer unquantized value accumulator (f32): `[num_layers][num_kv_heads]`
    values_full: Vec<Vec<Vec<f32>>>,
    /// Offset per layer (number of tokens cached).
    offsets: Vec<usize>,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Quantization group size (default 64).
    group_size: u32,
    /// Quantization bit width (4 or 8).
    bits: u32,
}

impl QuantizedKvCache {
    fn validate_layer(&self, layer: usize, context: &str) -> Result<(), KernelError> {
        if layer >= self.offsets.len() {
            return Err(KernelError::InvalidShape(format!(
                "{context}: layer {} out of bounds for {} layers",
                layer,
                self.offsets.len()
            )));
        }
        Ok(())
    }

    fn validate_lookup(&self, layer: usize, head: usize, context: &str) -> Result<(), KernelError> {
        self.validate_layer(layer, context)?;
        if head >= self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "{context}: head {} out of bounds for {} kv heads",
                head, self.num_kv_heads
            )));
        }
        if self.keys[layer].is_empty() {
            return Err(KernelError::InvalidShape(format!(
                "{context}: layer {} cache is empty",
                layer
            )));
        }
        Ok(())
    }

    /// Create a new empty quantized KV cache.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        group_size: u32,
        bits: u32,
    ) -> Self {
        assert!(
            bits == 4 || bits == 8,
            "QuantizedKvCache: bits must be 4 or 8, got {bits}"
        );
        assert!(
            group_size > 0 && (head_dim % group_size as usize == 0),
            "QuantizedKvCache: head_dim ({head_dim}) must be divisible by group_size ({group_size})"
        );

        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let mut keys_full = Vec::with_capacity(num_layers);
        let mut values_full = Vec::with_capacity(num_layers);
        let offsets = vec![0usize; num_layers];
        for _ in 0..num_layers {
            keys.push(Vec::with_capacity(num_kv_heads));
            values.push(Vec::with_capacity(num_kv_heads));
            let mut kf = Vec::with_capacity(num_kv_heads);
            let mut vf = Vec::with_capacity(num_kv_heads);
            for _ in 0..num_kv_heads {
                kf.push(Vec::new());
                vf.push(Vec::new());
            }
            keys_full.push(kf);
            values_full.push(vf);
        }

        Self {
            keys,
            values,
            keys_full,
            values_full,
            offsets,
            num_kv_heads,
            head_dim,
            group_size,
            bits,
        }
    }

    /// Number of cached tokens for a given layer.
    pub fn offset(&self, layer: usize) -> Result<usize, KernelError> {
        self.validate_layer(layer, "QuantizedKvCache::offset")?;
        Ok(self.offsets[layer])
    }

    /// Whether the cache is empty for a given layer.
    pub fn is_empty(&self, layer: usize) -> Result<bool, KernelError> {
        self.validate_layer(layer, "QuantizedKvCache::is_empty")?;
        Ok(self.offsets[layer] == 0)
    }

    /// Append new K, V for a layer.
    ///
    /// The incoming keys/values are in f16/f32 format.
    /// They get quantized using the affine format and appended to the cache.
    ///
    /// For simplicity: quantize the full (old + new) sequence each time.
    /// This is acceptable because quantized caches are used for long sequences
    /// where the quantization overhead is amortized.
    pub fn append(
        &mut self,
        layer: usize,
        new_keys: Vec<Array>,
        new_values: Vec<Array>,
        new_tokens: usize,
        registry: &KernelRegistry,
        _queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(), KernelError> {
        self.validate_layer(layer, "QuantizedKvCache::append")?;
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedKvCache::append: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
        }

        // Validate input tensor shapes
        for (kind, tensors) in [("keys", &new_keys), ("values", &new_values)] {
            for (i, tensor) in tensors.iter().enumerate() {
                let shape = tensor.shape();
                if shape.len() != 2 {
                    return Err(KernelError::InvalidShape(format!(
                        "QuantizedKvCache::append: {kind}[{i}] expected 2D [seq, head_dim], got {:?}",
                        shape
                    )));
                }
                if shape[1] != self.head_dim {
                    return Err(KernelError::InvalidShape(format!(
                        "QuantizedKvCache::append: {kind}[{i}] head_dim mismatch: expected {}, got {}",
                        self.head_dim, shape[1]
                    )));
                }
                if shape[0] != new_tokens {
                    return Err(KernelError::InvalidShape(format!(
                        "QuantizedKvCache::append: {kind}[{i}] seq mismatch: expected {}, got {}",
                        new_tokens, shape[0]
                    )));
                }
            }
        }

        let dev = registry.device().raw();

        // Extend the full-precision accumulators with the new data, then re-quantize.
        for h in 0..self.num_kv_heads {
            // Read new key data as f32
            let new_k_f32 = read_array_as_f32(&new_keys[h])?;
            self.keys_full[layer][h].extend_from_slice(&new_k_f32);

            // Read new value data as f32
            let new_v_f32 = read_array_as_f32(&new_values[h])?;
            self.values_full[layer][h].extend_from_slice(&new_v_f32);
        }

        self.offsets[layer] += new_tokens;
        let total_seq = self.offsets[layer];

        // Re-quantize all heads for this layer
        let mut new_qk = Vec::with_capacity(self.num_kv_heads);
        let mut new_qv = Vec::with_capacity(self.num_kv_heads);
        for h in 0..self.num_kv_heads {
            let k_arr = Array::from_slice(
                dev,
                &self.keys_full[layer][h],
                vec![total_seq, self.head_dim],
            );
            new_qk.push(quantize_to_affine(&k_arr, self.group_size, self.bits, dev)?);

            let v_arr = Array::from_slice(
                dev,
                &self.values_full[layer][h],
                vec![total_seq, self.head_dim],
            );
            new_qv.push(quantize_to_affine(&v_arr, self.group_size, self.bits, dev)?);
        }

        self.keys[layer] = new_qk;
        self.values[layer] = new_qv;

        Ok(())
    }

    /// Get the quantized keys for a specific layer and head.
    pub fn quantized_keys(
        &self,
        layer: usize,
        head: usize,
    ) -> Result<&QuantizedArray, KernelError> {
        self.validate_lookup(layer, head, "QuantizedKvCache::quantized_keys")?;
        Ok(&self.keys[layer][head])
    }

    /// Get the quantized values for a specific layer and head.
    pub fn quantized_values(
        &self,
        layer: usize,
        head: usize,
    ) -> Result<&QuantizedArray, KernelError> {
        self.validate_lookup(layer, head, "QuantizedKvCache::quantized_values")?;
        Ok(&self.values[layer][head])
    }

    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Quantization bit width.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Quantization group size.
    pub fn group_size(&self) -> u32 {
        self.group_size
    }
}

/// Read an Array's data as f32 values, handling f16/bf16 -> f32 conversion.
///
/// Supports Float32 (direct read), Float16, and Bfloat16 (manual conversion).
/// Returns an error for unsupported dtypes instead of panicking.
fn read_array_as_f32(arr: &Array) -> Result<Vec<f32>, KernelError> {
    match arr.dtype() {
        DType::Float32 => Ok(arr.to_vec_checked::<f32>()),
        DType::Float16 => {
            // Float16 is stored as 2 bytes per element. We read raw bytes
            // and convert each u16 bit-pattern to f32 using IEEE 754 rules.
            let bytes = arr.to_bytes();
            let numel = arr.numel();
            let mut result = Vec::with_capacity(numel);
            for i in 0..numel {
                let lo = bytes[i * 2] as u16;
                let hi = bytes[i * 2 + 1] as u16;
                let bits = lo | (hi << 8);
                result.push(f16_to_f32(bits));
            }
            Ok(result)
        }
        DType::Bfloat16 => {
            // BFloat16: 1 sign, 8 exponent, 7 mantissa — same exponent as f32.
            // Convert by shifting the 16-bit pattern left by 16.
            let bytes = arr.to_bytes();
            let numel = arr.numel();
            let mut result = Vec::with_capacity(numel);
            for i in 0..numel {
                let lo = bytes[i * 2] as u16;
                let hi = bytes[i * 2 + 1] as u16;
                let bits = lo | (hi << 8);
                result.push(bf16_to_f32(bits));
            }
            Ok(result)
        }
        other => Err(KernelError::NotFound(format!(
            "read_array_as_f32: unsupported dtype {:?}; expected Float32, Float16, or Bfloat16",
            other
        ))),
    }
}

/// Convert an IEEE 754 half-precision (f16) bit pattern to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // +/- zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 -> normalized f32
            // value = (-1)^sign * 2^(-14) * (mant / 1024)
            let val = (mant as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            // NaN: preserve some mantissa bits
            f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
        }
    } else {
        // Normalized: rebias exponent from 15-bias to 127-bias
        let f32_exp = exp + 127 - 15;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}

/// Convert a BFloat16 bit pattern to f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Quantize a 2D array [seq_len, dim] into MLX affine format.
///
/// Computes per-group min/max, derives scale and bias, and packs
/// values into uint32.
///
/// This is a CPU-side quantization for simplicity. GPU quantization kernel
/// can be added later.
fn quantize_to_affine(
    input: &Array,
    group_size: u32,
    bits: u32,
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<QuantizedArray, KernelError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "quantize_to_affine: expected 2D input, got {:?}",
            shape
        )));
    }
    let seq_len = shape[0];
    let dim = shape[1];
    let total_elems = seq_len * dim;
    let gs = group_size as usize;

    if total_elems == 0 {
        // Empty input: return empty quantized arrays
        return Ok(QuantizedArray {
            packed: Array::from_slice(device, &[0u32; 0], vec![0]),
            scales: Array::from_slice(device, &[0.0f32; 0], vec![0]),
            biases: Array::from_slice(device, &[0.0f32; 0], vec![0]),
            group_size,
            bits,
        });
    }

    // Pad total elements to be a multiple of group_size
    let padded_elems = if total_elems % gs != 0 {
        ((total_elems / gs) + 1) * gs
    } else {
        total_elems
    };

    // Read input as f32
    let data = read_array_as_f32(input)?;

    // Pad with zeros if necessary
    let mut padded = data;
    if padded.len() < padded_elems {
        padded.resize(padded_elems, 0.0);
    }

    let num_groups = padded_elems / gs;
    let max_q = ((1u32 << bits) - 1) as f32;
    let elems_per_u32 = 32 / bits as usize;

    let mut scales_vec = Vec::with_capacity(num_groups);
    let mut biases_vec = Vec::with_capacity(num_groups);
    let mut packed_vec = Vec::with_capacity(padded_elems / elems_per_u32);

    for g in 0..num_groups {
        let start = g * gs;
        let end = start + gs;
        let group = &padded[start..end];

        // Find min and max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in group {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / max_q };
        let bias = min_val;

        scales_vec.push(scale);
        biases_vec.push(bias);

        // Quantize and pack
        // We accumulate `elems_per_u32` quantized values into each u32 word
        let inv_scale = if range == 0.0 { 0.0 } else { 1.0 / scale };
        for chunk_start in (0..gs).step_by(elems_per_u32) {
            let mut word: u32 = 0;
            for k in 0..elems_per_u32 {
                let idx = chunk_start + k;
                let val = if idx < gs { group[idx] } else { 0.0 };
                let q = ((val - bias) * inv_scale).round().clamp(0.0, max_q) as u32;
                word |= q << (k as u32 * bits);
            }
            packed_vec.push(word);
        }
    }

    let packed = Array::from_slice(device, &packed_vec, vec![packed_vec.len()]);
    let scales = Array::from_slice(device, &scales_vec, vec![num_groups]);
    let biases = Array::from_slice(device, &biases_vec, vec![num_groups]);

    Ok(QuantizedArray {
        packed,
        scales,
        biases,
        group_size,
        bits,
    })
}

/// Scale attention scores by a scalar factor.
///
/// Tries broadcasting `scalar * matrix` first. If binary ops don't yet support
/// broadcasting, falls back to a manual element-wise scale via a filled array.
fn scale_scores(
    scores: &Array,
    scale: f32,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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

// ---------------------------------------------------------------------------
// Diagnostic / benchmark-only methods for Attention
// ---------------------------------------------------------------------------

#[cfg(feature = "bench")]
impl Attention {
    // -----------------------------------------------------------------------
    // Cumulative partial encode: encode dispatches 1..stop_after into one encoder
    // -----------------------------------------------------------------------

    /// Encode attention dispatches 1..`stop_after` into the given compute encoder.
    ///
    /// `stop_after` is 1-indexed and clamped to 1..=5:
    ///   1 → RMSNorm only
    ///   2 → RMSNorm + QKV GEMM
    ///   3 → RMSNorm + QKV + RoPE Q+K + V deinterleave
    ///   4 → RMSNorm + QKV + RoPE + SDPA
    ///   5 → RMSNorm + QKV + RoPE + SDPA + O Projection (full attention)
    ///
    /// Returns the output array of the last dispatch encoded.
    /// For stop_after < 5, the output is an intermediate and should be discarded.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_encode_partial(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        encoder: ComputePass<'_>,
        stop_after: usize,
    ) -> Result<Array, KernelError> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        let seq_len = x.shape()[0];

        // ---- Dispatch 1: RMSNorm (pre-attention) ----
        let normed =
            ops::rms_norm::rms_norm_encode(registry, x, Some(norm_weight), rms_norm_eps, encoder)?;
        if stop_after <= 1 {
            return Ok(normed);
        }

        // ---- Dispatch 2: QKV merged GEMM ----
        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let v_dim = num_kv_heads * head_dim;

        let normed_2d = if normed.ndim() == 1 {
            normed.reshape(vec![1, normed.shape()[0]])?
        } else {
            normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
        };

        let qkv_wt = self.qkv_merged_weight_t.as_ref().ok_or_else(|| {
            KernelError::InvalidShape(
                "forward_prefill_encode_partial requires merged QKV weight".into(),
            )
        })?;
        let qkv = ops::matmul::matmul_encode(registry, &normed_2d, qkv_wt, encoder)?;
        if stop_after <= 2 {
            return Ok(qkv);
        }

        let total_out = q_dim + k_dim + v_dim;
        let elem_size = qkv.dtype().size_of();
        let q = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, q_dim],
            vec![total_out, 1],
            qkv.dtype(),
            qkv.offset(),
        );
        let k = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, k_dim],
            vec![total_out, 1],
            qkv.dtype(),
            qkv.offset() + q_dim * elem_size,
        );
        let v = Array::new(
            qkv.metal_buffer().to_owned(),
            vec![seq_len, v_dim],
            vec![total_out, 1],
            qkv.dtype(),
            qkv.offset() + (q_dim + k_dim) * elem_size,
        );

        // ---- Dispatch 3: RoPE Q+K + V deinterleave ----
        let rope_offset = cache.seq_len as u32;
        let q_row_stride = q.strides()[0];
        let k_row_stride = k.strides()[0];
        let v_row_stride = v.strides()[0];

        let (q_batched, k_batched) = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
            let q_b = ops::rope::rope_multihead_encode(
                registry,
                &q,
                cos,
                sin,
                num_heads,
                rope_offset,
                q_row_stride,
                encoder,
            )?;
            let k_b = ops::rope::rope_multihead_encode(
                registry,
                &k,
                cos,
                sin,
                num_kv_heads,
                rope_offset,
                k_row_stride,
                encoder,
            )?;
            (q_b, k_b)
        } else {
            let q_b = ops::rope::deinterleave_heads_encode(
                registry,
                &q,
                num_heads,
                q_row_stride,
                encoder,
            )?;
            let k_b = ops::rope::deinterleave_heads_encode(
                registry,
                &k,
                num_kv_heads,
                k_row_stride,
                encoder,
            )?;
            (q_b, k_b)
        };
        let v_batched = ops::rope::deinterleave_heads_encode(
            registry,
            &v,
            num_kv_heads,
            v_row_stride,
            encoder,
        )?;

        if stop_after <= 3 {
            return Ok(q_batched);
        }

        // ---- Dispatch 4: SDPA ----
        let needs_initial_append = cache.seq_len == 0;
        let (k_for_sdpa, v_for_sdpa, kv_stride, total_seq) = if needs_initial_append {
            let k_view = k_batched.view(
                k_batched.shape().to_vec(),
                k_batched.strides().to_vec(),
                k_batched.offset(),
            );
            let v_view = v_batched.view(
                v_batched.shape().to_vec(),
                v_batched.strides().to_vec(),
                v_batched.offset(),
            );
            (k_view, v_view, None, seq_len)
        } else {
            cache.append_batched_encode(&k_batched, &v_batched, seq_len, registry, encoder)?;
            let total_seq = cache.seq_len;
            let k_slab = cache.keys_slab_view().ok_or_else(|| {
                KernelError::InvalidShape("encode_partial: cache has no slab layout".into())
            })?;
            let v_slab = cache.values_slab_view().ok_or_else(|| {
                KernelError::InvalidShape("encode_partial: cache has no slab layout".into())
            })?;
            let kv_stride = if cache.max_seq_len() != total_seq {
                Some(cache.max_seq_len())
            } else {
                None
            };
            (k_slab, v_slab, kv_stride, total_seq)
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let is_f16_d128 = q_batched.dtype() == DType::Float16 && head_dim == 128 && seq_len > 1;
        let use_nax = is_f16_d128 && registry.device().tuning().supports_nax;
        let use_mma_bk32 = is_f16_d128 && !use_nax && total_seq >= 256;

        let attn_slab = if use_nax {
            ops::sdpa::sdpa_prefill_nax_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                scale,
                true,
                None,
                None,
                encoder,
            )?
        } else if use_mma_bk32 {
            ops::sdpa::sdpa_prefill_mma_bk32_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                None,
                None,
                encoder,
            )?
        } else if is_f16_d128 {
            ops::sdpa::sdpa_prefill_mma_f16_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                None,
                None,
                encoder,
            )?
        } else {
            ops::sdpa::sdpa_prefill_gqa_slab_encode(
                registry,
                &q_batched,
                &k_for_sdpa,
                &v_for_sdpa,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                total_seq,
                kv_stride,
                None,
                scale,
                true,
                encoder,
            )?
        };

        if needs_initial_append {
            cache.append_batched_encode(&k_batched, &v_batched, seq_len, registry, encoder)?;
        }

        if stop_after <= 4 {
            return Ok(attn_slab);
        }

        // ---- Dispatch 5: O Projection ----
        let concat = attn_slab.view(
            vec![seq_len, hidden_size],
            vec![hidden_size, 1],
            attn_slab.offset(),
        );
        self.o_proj.forward_into_encoder(&concat, registry, encoder)
    }

    // -----------------------------------------------------------------------
    // Per-dispatch breakdown: measure each dispatch in its own CB
    // -----------------------------------------------------------------------

    /// Run attention dispatches 1-5 individually, each in its own command buffer,
    /// returning per-dispatch wall-clock timings and the final O-projection output.
    ///
    /// Dispatches:
    ///   1. RMSNorm (pre-attention)
    ///   2. QKV merged GEMM
    ///   3. RoPE Q+K + V deinterleave (fused)
    ///   4. SDPA
    ///   5. O Projection
    ///
    /// Each dispatch is encoded into a fresh CB, committed, and waited on.
    /// This adds CB overhead but gives exact per-dispatch GPU timing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_breakdown(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        cache: &mut LayerKvCache,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<(Array, Vec<(&'static str, std::time::Duration)>), KernelError> {
        use std::time::Instant;

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;
        let seq_len = x.shape()[0];
        let mut timings: Vec<(&'static str, std::time::Duration)> = Vec::with_capacity(5);

        let q_dim = num_heads * head_dim;
        let k_dim = num_kv_heads * head_dim;
        let v_dim = num_kv_heads * head_dim;

        let qkv_wt = self.qkv_merged_weight_t.as_ref().ok_or_else(|| {
            KernelError::InvalidShape("forward_prefill_breakdown requires merged QKV weight".into())
        })?;

        // ---- Dispatch 1: RMSNorm (pre-attention) ----
        let normed = {
            let cb = queue.commandBuffer().unwrap();
            let start = Instant::now();
            let out = ops::rms_norm::rms_norm_into_cb(
                registry,
                x,
                Some(norm_weight),
                rms_norm_eps,
                &*cb,
            )?;
            cb.commit();
            cb.waitUntilCompleted();
            timings.push(("RMSNorm (pre-attn)", start.elapsed()));
            out
        };

        // ---- Dispatch 2: QKV merged GEMM ----
        let normed_2d = if normed.ndim() == 1 {
            normed.reshape(vec![1, normed.shape()[0]])?
        } else {
            normed.reshape(vec![normed.shape()[0], normed.shape()[1]])?
        };

        let (q, k, v) = {
            let cb = queue.commandBuffer().unwrap();
            let start = Instant::now();
            let qkv = ops::matmul::matmul_into_cb(registry, &normed_2d, qkv_wt, &*cb)?;
            cb.commit();
            cb.waitUntilCompleted();
            timings.push(("QKV Merged GEMM", start.elapsed()));
            let total_out = q_dim + k_dim + v_dim;
            let elem_size = qkv.dtype().size_of();
            let q_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, q_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset(),
            );
            let k_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, k_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset() + q_dim * elem_size,
            );
            let v_view = Array::new(
                qkv.metal_buffer().to_owned(),
                vec![seq_len, v_dim],
                vec![total_out, 1],
                qkv.dtype(),
                qkv.offset() + (q_dim + k_dim) * elem_size,
            );
            (q_view, k_view, v_view)
        };

        // ---- Dispatch 3: RoPE Q+K + V deinterleave (fused) ----
        let rope_offset = cache.seq_len as u32;
        let q_row_stride = q.strides()[0];

        let (q_batched, k_batched, v_batched) = {
            let cb = queue.commandBuffer().unwrap();
            let start = Instant::now();
            let result = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                if std::ptr::eq(q.metal_buffer(), k.metal_buffer())
                    && std::ptr::eq(k.metal_buffer(), v.metal_buffer())
                {
                    ops::rope::rope_qkv_batch_into_cb(
                        registry,
                        &q,
                        cos,
                        sin,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        rope_offset,
                        q_row_stride,
                        &*cb,
                    )?
                } else {
                    let q_b = ops::rope::rope_multihead_into_cb(
                        registry,
                        &q,
                        cos,
                        sin,
                        num_heads,
                        rope_offset,
                        q.strides()[0],
                        &*cb,
                    )?;
                    let k_b = ops::rope::rope_multihead_into_cb(
                        registry,
                        &k,
                        cos,
                        sin,
                        num_kv_heads,
                        rope_offset,
                        k.strides()[0],
                        &*cb,
                    )?;
                    let v_b = ops::rope::deinterleave_heads_into_cb(
                        registry,
                        &v,
                        num_kv_heads,
                        v.strides()[0],
                        &*cb,
                    )?;
                    (q_b, k_b, v_b)
                }
            } else {
                let q_b = ops::rope::deinterleave_heads_into_cb(
                    registry,
                    &q,
                    num_heads,
                    q.strides()[0],
                    &*cb,
                )?;
                let k_b = ops::rope::deinterleave_heads_into_cb(
                    registry,
                    &k,
                    num_kv_heads,
                    k.strides()[0],
                    &*cb,
                )?;
                let v_b = ops::rope::deinterleave_heads_into_cb(
                    registry,
                    &v,
                    num_kv_heads,
                    v.strides()[0],
                    &*cb,
                )?;
                (q_b, k_b, v_b)
            };
            cb.commit();
            cb.waitUntilCompleted();
            timings.push(("RoPE Q+K + V deinterleave", start.elapsed()));
            result
        };

        // ---- Dispatch 4: SDPA ----
        let needs_initial_append = cache.seq_len == 0;
        let (k_for_sdpa, v_for_sdpa, kv_stride, total_seq) = if needs_initial_append {
            let k_view = k_batched.view(
                k_batched.shape().to_vec(),
                k_batched.strides().to_vec(),
                k_batched.offset(),
            );
            let v_view = v_batched.view(
                v_batched.shape().to_vec(),
                v_batched.strides().to_vec(),
                v_batched.offset(),
            );
            (k_view, v_view, None, seq_len)
        } else {
            let cache_cb = queue.commandBuffer().unwrap();
            cache.append_batched_into_cb(&k_batched, &v_batched, seq_len, registry, &*cache_cb)?;
            cache_cb.commit();
            cache_cb.waitUntilCompleted();
            let total_seq = cache.seq_len;
            let k_slab = cache.keys_slab_view().ok_or_else(|| {
                KernelError::InvalidShape("breakdown: cache has no slab layout".into())
            })?;
            let v_slab = cache.values_slab_view().ok_or_else(|| {
                KernelError::InvalidShape("breakdown: cache has no slab layout".into())
            })?;
            let kv_stride = if cache.max_seq_len() != total_seq {
                Some(cache.max_seq_len())
            } else {
                None
            };
            (k_slab, v_slab, kv_stride, total_seq)
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let is_f16_d128 = q_batched.dtype() == DType::Float16 && head_dim == 128 && seq_len > 1;
        let use_nax = is_f16_d128 && registry.device().tuning().supports_nax;
        let use_mma_bk32 = is_f16_d128 && !use_nax && total_seq >= 256;

        let attn_slab = {
            let cb = queue.commandBuffer().unwrap();
            let start = Instant::now();
            let slab = if use_nax {
                ops::sdpa::sdpa_prefill_nax_f16_into_cb(
                    registry,
                    &q_batched,
                    &k_for_sdpa,
                    &v_for_sdpa,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    total_seq,
                    kv_stride,
                    scale,
                    true,
                    None,
                    None,
                    &*cb,
                )?
            } else if use_mma_bk32 {
                ops::sdpa::sdpa_prefill_mma_bk32_f16_into_cb(
                    registry,
                    &q_batched,
                    &k_for_sdpa,
                    &v_for_sdpa,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    total_seq,
                    kv_stride,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    &*cb,
                )?
            } else if is_f16_d128 {
                ops::sdpa::sdpa_prefill_mma_f16_into_cb(
                    registry,
                    &q_batched,
                    &k_for_sdpa,
                    &v_for_sdpa,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    total_seq,
                    kv_stride,
                    None,
                    scale,
                    true,
                    None,
                    None,
                    &*cb,
                )?
            } else {
                ops::sdpa::sdpa_prefill_gqa_slab_into_cb(
                    registry,
                    &q_batched,
                    &k_for_sdpa,
                    &v_for_sdpa,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                    total_seq,
                    kv_stride,
                    None,
                    scale,
                    true,
                    &*cb,
                )?
            };
            cb.commit();
            cb.waitUntilCompleted();
            timings.push(("SDPA", start.elapsed()));
            slab
        };

        let concat = attn_slab.view(
            vec![seq_len, hidden_size],
            vec![hidden_size, 1],
            attn_slab.offset(),
        );

        // ---- Dispatch 5: O Projection ----
        let o_out = {
            let cb = queue.commandBuffer().unwrap();
            let start = Instant::now();
            let out = self.o_proj.forward_into_cb(&concat, registry, &*cb)?;
            cb.commit();
            cb.waitUntilCompleted();
            timings.push(("O Projection", start.elapsed()));
            out
        };

        Ok((o_out, timings))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::sync::OnceLock;

    fn test_device() -> &'static rmlx_metal::MtlDevice {
        static DEVICE: OnceLock<rmlx_metal::MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            objc2::rc::autoreleasepool(|_| {
                MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
            })
        })
    }

    #[allow(clippy::type_complexity)]
    fn setup() -> (
        &'static rmlx_metal::MtlDevice,
        KernelRegistry,
        objc2::rc::Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) {
        let device = test_device();
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("no GpuDevice");
        let queue = device.newCommandQueue().unwrap();
        let registry = KernelRegistry::new(gpu);
        ops::copy::register(&registry).expect("failed to register copy kernels");
        (device, registry, queue)
    }

    /// Helper: create an Array from a flat f32 slice with shape [rows, cols].
    fn make_array(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Array {
        assert_eq!(data.len(), rows * cols);
        Array::from_slice(device, data, vec![rows, cols])
    }

    /// Helper: read f32 data from an Array.
    fn read_f32(arr: &Array) -> Vec<f32> {
        arr.to_vec_checked::<f32>()
    }

    #[test]
    fn test_kv_cache_dtype_mismatch() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::preallocated(device, 1, 4, 8, DType::Float32);
        let bad_k = Array::zeros(device, &[1, 4], DType::Float16);
        let good_v = Array::zeros(device, &[1, 4], DType::Float32);
        let result = cache.append(vec![bad_k], vec![good_v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("dtype"));
    }

    #[test]
    fn test_kv_cache_head_dim_mismatch() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::preallocated(device, 1, 4, 8, DType::Float32);
        let bad_k = Array::zeros(device, &[1, 8], DType::Float32);
        let good_v = Array::zeros(device, &[1, 4], DType::Float32);
        let result = cache.append(vec![bad_k], vec![good_v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("head_dim"));
    }

    #[test]
    fn test_kv_cache_seq_dim_mismatch() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::preallocated(device, 1, 4, 8, DType::Float32);
        let bad_k = Array::zeros(device, &[2, 4], DType::Float32);
        let bad_v = Array::zeros(device, &[2, 4], DType::Float32);
        let result = cache.append(vec![bad_k], vec![bad_v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("seq"));
    }

    #[test]
    fn test_quantized_kv_cache_layer_out_of_bounds() {
        let cache = QuantizedKvCache::new(2, 4, 128, 64, 8);
        assert!(cache.offset(5).is_err());
        assert!(cache.is_empty(5).is_err());
    }

    #[test]
    fn test_quantized_kv_cache_lookup_empty() {
        let cache = QuantizedKvCache::new(2, 4, 128, 64, 8);
        assert!(cache.quantized_keys(0, 0).is_err());
        assert!(cache.quantized_values(0, 0).is_err());
    }

    #[test]
    fn test_quantized_kv_cache_head_out_of_bounds() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = QuantizedKvCache::new(2, 2, 128, 64, 8);
        let k0 = Array::zeros(device, &[1, 128], DType::Float32);
        let k1 = Array::zeros(device, &[1, 128], DType::Float32);
        let v0 = Array::zeros(device, &[1, 128], DType::Float32);
        let v1 = Array::zeros(device, &[1, 128], DType::Float32);
        cache
            .append(0, vec![k0, k1], vec![v0, v1], 1, &registry, &queue)
            .unwrap();
        assert!(cache.quantized_keys(0, 5).is_err());
        assert!(cache.quantized_values(0, 5).is_err());
    }

    // --- Issue #91: Legacy first-append validation tests ---

    #[test]
    fn test_legacy_first_append_dtype_mismatch() {
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::new(2);
        let k0 = Array::zeros(device, &[1, 4], DType::Float32);
        let k1 = Array::zeros(device, &[1, 4], DType::Float16);
        let v0 = Array::zeros(device, &[1, 4], DType::Float32);
        let v1 = Array::zeros(device, &[1, 4], DType::Float32);
        let result = cache.append(vec![k0, k1], vec![v0, v1], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("dtype"));
    }

    #[test]
    fn test_legacy_first_append_head_dim_mismatch() {
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::new(2);
        let k0 = Array::zeros(device, &[1, 4], DType::Float32);
        let k1 = Array::zeros(device, &[1, 8], DType::Float32);
        let v0 = Array::zeros(device, &[1, 4], DType::Float32);
        let v1 = Array::zeros(device, &[1, 4], DType::Float32);
        let result = cache.append(vec![k0, k1], vec![v0, v1], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("head_dim"));
    }

    #[test]
    fn test_legacy_first_append_not_2d() {
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::new(1);
        let k = Array::from_slice(device, &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let v = Array::zeros(device, &[1, 4], DType::Float32);
        let result = cache.append(vec![k], vec![v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("2D"));
    }

    #[test]
    fn test_legacy_first_append_seq_mismatch() {
        let (device, registry, queue) = setup();
        let mut cache = LayerKvCache::new(1);
        let k = Array::zeros(device, &[3, 4], DType::Float32);
        let v = Array::zeros(device, &[3, 4], DType::Float32);
        let result = cache.append(vec![k], vec![v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("seq"));
    }

    // --- Issue #92: QuantizedKvCache shape validation tests ---

    #[test]
    fn test_quantized_kv_cache_head_dim_mismatch() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = QuantizedKvCache::new(2, 1, 128, 64, 8);
        // Wrong head_dim: 64 instead of 128
        let k = Array::zeros(device, &[1, 64], DType::Float32);
        let v = Array::zeros(device, &[1, 128], DType::Float32);
        let result = cache.append(0, vec![k], vec![v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("head_dim"));
    }

    #[test]
    fn test_quantized_kv_cache_not_2d() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = QuantizedKvCache::new(2, 1, 128, 64, 8);
        let k = Array::from_slice(device, &vec![0.0f32; 128], vec![128]);
        let v = Array::zeros(device, &[1, 128], DType::Float32);
        let result = cache.append(0, vec![k], vec![v], 1, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("2D"));
    }

    #[test]
    fn test_quantized_kv_cache_seq_mismatch() {
        let _ = test_device(); // ensure Metal device is available
        let (device, registry, queue) = setup();
        let mut cache = QuantizedKvCache::new(2, 1, 128, 64, 8);
        // Say 3 tokens but array has 2 rows
        let k = Array::zeros(device, &[2, 128], DType::Float32);
        let v = Array::zeros(device, &[2, 128], DType::Float32);
        let result = cache.append(0, vec![k], vec![v], 3, &registry, &queue);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("seq"));
    }

    /// Property test: append N tokens one at a time to a cache of capacity M,
    /// verify that linearized output matches the expected sliding window.
    ///
    /// For keep=0: expected = last min(N, M) tokens
    /// For keep>0: expected = first `keep` tokens + last min(N-keep, M-keep) ring tokens
    fn check_rotating_cache(max_size: usize, keep: usize, total_tokens: usize, head_dim: usize) {
        let (device, registry, queue) = setup();
        let num_kv_heads = 1;

        let mut cache = RotatingKvCache::new(
            device,
            num_kv_heads,
            head_dim,
            max_size,
            keep,
            DType::Float32,
        );

        // Build the expected full sequence — token i has all elements = (i+1) as f32
        let mut all_tokens: Vec<Vec<f32>> = Vec::new();
        for t in 0..total_tokens {
            let val = (t + 1) as f32;
            let token_data: Vec<f32> = vec![val; head_dim];
            all_tokens.push(token_data.clone());

            let k_arr = make_array(device, &token_data, 1, head_dim);
            let v_arr = make_array(device, &token_data, 1, head_dim);
            cache
                .append(vec![k_arr], vec![v_arr], 1, &registry, &queue)
                .expect("append failed");
        }

        // Read linearized keys
        let linearized = cache
            .cached_keys(0, &registry, &queue)
            .expect("cached_keys failed");
        let got = read_f32(&linearized);
        let current_len = cache.current_len();

        // Build expected output
        let expected: Vec<f32> = if total_tokens <= max_size {
            // Haven't filled the buffer yet — all tokens in order
            all_tokens.iter().flat_map(|t| t.iter().copied()).collect()
        } else if keep == 0 {
            // No keep region — last max_size tokens
            all_tokens[total_tokens - max_size..]
                .iter()
                .flat_map(|t| t.iter().copied())
                .collect()
        } else {
            // Keep region + most recent ring tokens
            let ring_capacity = max_size - keep;
            let ring_tokens_available = total_tokens - keep;
            let ring_start_idx = if ring_tokens_available > ring_capacity {
                keep + (ring_tokens_available - ring_capacity)
            } else {
                keep
            };
            let mut exp = Vec::new();
            // Keep region
            for t in &all_tokens[..keep] {
                exp.extend_from_slice(t);
            }
            // Ring region (most recent)
            for t in &all_tokens[ring_start_idx..total_tokens] {
                exp.extend_from_slice(t);
            }
            exp
        };

        assert_eq!(
            got.len(),
            expected.len(),
            "length mismatch: got {} expected {} (max_size={}, keep={}, total={})",
            got.len(),
            expected.len(),
            max_size,
            keep,
            total_tokens,
        );
        assert_eq!(
            current_len,
            expected.len() / head_dim,
            "current_len mismatch"
        );
        assert_eq!(
            got, expected,
            "data mismatch (max_size={}, keep={}, total={})\ngot:      {:?}\nexpected: {:?}",
            max_size, keep, total_tokens, got, expected,
        );
    }

    // --- Property tests: no keep region ---

    #[test]
    fn test_rotating_cache_no_wrap() {
        // N < M: no wrapping
        check_rotating_cache(8, 0, 5, 4);
    }

    #[test]
    fn test_rotating_cache_exact_fill() {
        // N == M: exact fill, no wrap
        check_rotating_cache(8, 0, 8, 4);
    }

    #[test]
    fn test_rotating_cache_one_past() {
        // N == M+1: wraps once
        check_rotating_cache(8, 0, 9, 4);
    }

    #[test]
    fn test_rotating_cache_double_capacity() {
        // N == 2M: wraps fully twice
        check_rotating_cache(8, 0, 16, 4);
    }

    #[test]
    fn test_rotating_cache_triple_capacity() {
        // N == 3M: wraps fully three times
        check_rotating_cache(8, 0, 24, 4);
    }

    #[test]
    fn test_rotating_cache_various_no_keep() {
        for max_size in [4, 8, 16] {
            for total in 0..max_size * 3 {
                if total == 0 {
                    continue; // skip empty
                }
                check_rotating_cache(max_size, 0, total, 2);
            }
        }
    }

    // --- Property tests: with keep region ---

    #[test]
    fn test_rotating_cache_keep_no_wrap() {
        // keep=2, N < M: no wrapping
        check_rotating_cache(8, 2, 5, 4);
    }

    #[test]
    fn test_rotating_cache_keep_exact_fill() {
        // keep=2, N == M: exact fill
        check_rotating_cache(8, 2, 8, 4);
    }

    #[test]
    fn test_rotating_cache_keep_one_past() {
        // keep=2, N == M+1: wraps once in ring region
        check_rotating_cache(8, 2, 9, 4);
    }

    #[test]
    fn test_rotating_cache_keep_double() {
        // keep=2, N == 2M
        check_rotating_cache(8, 2, 16, 4);
    }

    #[test]
    fn test_rotating_cache_keep_triple() {
        // keep=2, N == 3M
        check_rotating_cache(8, 2, 24, 4);
    }

    #[test]
    fn test_rotating_cache_various_with_keep() {
        for max_size in [4, 8] {
            for keep in 1..max_size {
                for total in 1..max_size * 3 {
                    check_rotating_cache(max_size, keep, total, 2);
                }
            }
        }
    }

    // --- Multi-token append tests ---

    #[test]
    fn test_rotating_cache_multi_token_no_wrap() {
        let (device, registry, queue) = setup();
        let max_size = 8;
        let head_dim = 4;
        let mut cache = RotatingKvCache::new(device, 1, head_dim, max_size, 0, DType::Float32);

        // Append 3 tokens at once
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let k = make_array(device, &data, 3, head_dim);
        let v = make_array(device, &data, 3, head_dim);
        cache
            .append(vec![k], vec![v], 3, &registry, &queue)
            .expect("append failed");

        let got = read_f32(
            &cache
                .cached_keys(0, &registry, &queue)
                .expect("cached_keys failed"),
        );
        assert_eq!(got, data);
    }

    #[test]
    fn test_rotating_cache_multi_token_wrapping() {
        let (device, registry, queue) = setup();
        let max_size = 4;
        let head_dim = 2;
        let mut cache = RotatingKvCache::new(device, 1, head_dim, max_size, 0, DType::Float32);

        // Fill to capacity with single tokens
        for t in 0..4 {
            let val = (t + 1) as f32;
            let data = vec![val; head_dim];
            let k = make_array(device, &data, 1, head_dim);
            let v = make_array(device, &data, 1, head_dim);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
        }

        // Now append 3 more tokens (wraps via linearize_and_append)
        let new_data: Vec<f32> = vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0];
        let k = make_array(device, &new_data, 3, head_dim);
        let v = make_array(device, &new_data, 3, head_dim);
        cache
            .append(vec![k], vec![v], 3, &registry, &queue)
            .expect("append failed");

        // Expected: last 4 tokens = [4, 5, 6, 7]
        let got = read_f32(
            &cache
                .cached_keys(0, &registry, &queue)
                .expect("cached_keys failed"),
        );
        let expected: Vec<f32> = vec![4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_rotating_cache_offset_monotonic() {
        let (device, registry, queue) = setup();
        let mut cache = RotatingKvCache::new(device, 1, 2, 4, 0, DType::Float32);

        for t in 0..10 {
            let data = vec![(t + 1) as f32; 2];
            let k = make_array(device, &data, 1, 2);
            let v = make_array(device, &data, 1, 2);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
            assert_eq!(cache.offset(), t + 1);
        }
    }

    #[test]
    fn test_rotating_cache_current_len() {
        let (device, registry, queue) = setup();
        let mut cache = RotatingKvCache::new(device, 1, 2, 4, 0, DType::Float32);

        for t in 0..10 {
            let data = vec![(t + 1) as f32; 2];
            let k = make_array(device, &data, 1, 2);
            let v = make_array(device, &data, 1, 2);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
            assert_eq!(cache.current_len(), std::cmp::min(t + 1, 4));
        }
    }

    #[test]
    fn test_layer_kv_cache_trim() {
        let _ = test_device(); // ensure Metal device is available
        let device = test_device();
        let mut cache = LayerKvCache::preallocated(device, 2, 4, 32, DType::Float32);

        // Simulate appending 10 tokens by setting seq_len directly.
        cache.seq_len = 10;
        assert_eq!(cache.position_offset(), 10);

        // Trim 3 tokens.
        let new_len = cache.trim(3);
        assert_eq!(new_len, 7);
        assert_eq!(cache.seq_len, 7);

        // Trim 5 more tokens.
        let new_len = cache.trim(5);
        assert_eq!(new_len, 2);
        assert_eq!(cache.seq_len, 2);

        // Trim more than remaining — saturates to 0.
        let new_len = cache.trim(100);
        assert_eq!(new_len, 0);
        assert_eq!(cache.seq_len, 0);
        assert!(cache.is_empty());
    }
}
