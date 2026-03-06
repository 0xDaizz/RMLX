//! Multi-head attention with RoPE and GQA support.
//!
//! KV cache uses pre-allocated buffers with O(1) append (no full-history copy).

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_metal::exec_graph::{EventToken, ExecGraph};

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
            enc.set_buffer(
                0,
                Some(new_keys[i].metal_buffer()),
                new_keys[i].offset() as u64,
            );
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
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset() as u64,
            );
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
        cb: &metal::CommandBufferRef,
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
            self.seq_len += new_tokens;
            return Ok(());
        }

        let dst_row_offset = self.seq_len * self.head_dim * elem_size;

        for i in 0..self.num_kv_heads {
            // Copy new keys into slot
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(
                0,
                Some(new_keys[i].metal_buffer()),
                new_keys[i].offset() as u64,
            );
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
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset() as u64,
            );
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                (self.values[i].offset() + dst_row_offset) as u64,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
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
        device: &metal::Device,
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        let cb = queue.new_command_buffer();

        // Part 1: keep region [0..keep]
        if part1_len > 0 {
            let count = (part1_len * self.head_dim) as u64;
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), buf.offset() as u64);
            enc.set_buffer(1, Some(result.metal_buffer()), result.offset() as u64);
            let grid = metal::MTLSize::new(count, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        // Part 2: oldest ring portion [ring_start..ring_end]
        if part2_len > 0 {
            let count = (part2_len * self.head_dim) as u64;
            let src_off = (buf.offset() + ring_start * self.head_dim * elem_size) as u64;
            let dst_off = (result.offset() + part1_len * self.head_dim * elem_size) as u64;
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), src_off);
            enc.set_buffer(1, Some(result.metal_buffer()), dst_off);
            let grid = metal::MTLSize::new(count, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        // Part 3: newest ring portion [keep..ring_start]
        if part3_len > 0 {
            let count = (part3_len * self.head_dim) as u64;
            let src_off = (buf.offset() + keep * self.head_dim * elem_size) as u64;
            let dst_off =
                (result.offset() + (part1_len + part2_len) * self.head_dim * elem_size) as u64;
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(buf.metal_buffer()), src_off);
            enc.set_buffer(1, Some(result.metal_buffer()), dst_off);
            let grid = metal::MTLSize::new(count, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        let num_elems = (count * self.head_dim) as u64;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
        enc.set_buffer(
            1,
            Some(dst.metal_buffer()),
            (dst.offset() + dst_row * self.head_dim * elem_size) as u64,
        );
        let grid = metal::MTLSize::new(num_elems, 1, 1);
        let tg = metal::MTLSize::new(
            std::cmp::min(pipeline.max_total_threads_per_threadgroup(), num_elems),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
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
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
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
        let count = (new_tokens * self.head_dim) as u64;
        if count == 0 {
            return Ok(());
        }

        let cb = queue.new_command_buffer();
        let dst_byte_offset = dst_row * self.head_dim * elem_size;

        for i in 0..self.num_kv_heads {
            // Keys
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(
                0,
                Some(new_keys[i].metal_buffer()),
                new_keys[i].offset() as u64,
            );
            enc.set_buffer(
                1,
                Some(self.keys[i].metal_buffer()),
                (self.keys[i].offset() + dst_byte_offset) as u64,
            );
            let grid = metal::MTLSize::new(count, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(pipeline.max_total_threads_per_threadgroup(), count),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            // Values
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(
                0,
                Some(new_values[i].metal_buffer()),
                new_values[i].offset() as u64,
            );
            enc.set_buffer(
                1,
                Some(self.values[i].metal_buffer()),
                (self.values[i].offset() + dst_byte_offset) as u64,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        cb.commit();
        cb.wait_until_completed();
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
        queue: &metal::CommandQueue,
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
        queue: &metal::CommandQueue,
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
        device: &metal::Device,
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
        device: &metal::Device,
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

        // Concatenate heads — N14 optimization: one strided copy per head
        // instead of O(num_heads * seq_len) individual per-row encodes.
        //
        // Each head output is [seq_len, head_dim] contiguous. We copy it into
        // the interleaved output layout [seq_len, num_heads * head_dim] using
        // a single blit per head with `destinationBytesPerRow` stride, which
        // lets the GPU handle the row-by-row scatter in hardware.
        //
        // When blit striding is not possible (non-contiguous heads), we fall
        // back to one compute encode per head that copies all rows at once
        // using a 2D grid (seq_len x head_dim threads).
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
        let head_bytes = head_dim * elem_size;
        let hidden_bytes = hidden_size * elem_size;

        let cb = queue.new_command_buffer();
        for (h, head_out) in attn_outputs.iter().enumerate() {
            let dst_col_offset = h * head_bytes;

            if seq_len == 1 {
                // Single-token decode: one contiguous copy of head_dim elements.
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
                // Multi-token: use MTLBlitCommandEncoder for strided copy.
                // Copy `seq_len` rows of `head_dim` elements each from the
                // contiguous head output into the interleaved concat buffer.
                //
                // Source: contiguous [seq_len, head_dim], row stride = head_bytes
                // Dest: interleaved [seq_len, hidden_size], row stride = hidden_bytes
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
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        self.q_proj.prepare_weight_t(registry, queue)?;
        self.k_proj.prepare_weight_t(registry, queue)?;
        self.v_proj.prepare_weight_t(registry, queue)?;
        self.o_proj.prepare_weight_t(registry, queue)?;
        Ok(())
    }

    /// ExecGraph-based attention forward pass using 4 command buffers.
    ///
    /// Architecture:
    /// - CB1 (current): Q/K/V projections
    /// - CB2: head split + contiguous copy + RoPE + cache append
    /// - CB3: SDPA
    /// - CB4: head concat (interleave) + O_proj
    ///
    /// Returns `(output_array, EventToken)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph(
        &self,
        x: &Array,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
    ) -> Result<(Array, EventToken), KernelError> {
        let seq_len = x.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;

        // ---- CB1: Q/K/V projections ----
        let cb1 = graph.command_buffer();
        let q = self.q_proj.forward_into_cb(x, registry, cb1)?;
        let k = self.k_proj.forward_into_cb(x, registry, cb1)?;
        let v = self.v_proj.forward_into_cb(x, registry, cb1)?;
        let t1 = graph.submit_batch();

        // ---- CB2: head split, contiguous copy, RoPE, cache append ----
        graph.wait_for(t1);
        let cb2 = graph.command_buffer();

        let dev = registry.device().raw();
        let elem_size = q.dtype().size_of();
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        // Split Q into heads and apply RoPE
        let mut q_heads: Vec<Array> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let offset = q.offset() + h * head_dim * elem_size;
            let q_head = q.view(
                vec![seq_len, head_dim],
                vec![num_heads * head_dim, 1],
                offset,
            );
            let q_head = ops::copy::copy_into_cb(registry, &q_head, cb2)?;
            let q_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope_ext_into_cb(
                    registry,
                    &q_head,
                    cos,
                    sin,
                    rope_offset,
                    1.0,
                    false,
                    true,
                    cb2,
                )?
            } else {
                q_head
            };
            q_heads.push(q_head);
        }

        // Split K into heads, copy contiguous, apply RoPE
        let mut k_heads: Vec<Array> = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let offset = k.offset() + h * head_dim * elem_size;
            let k_head = k.view(
                vec![seq_len, head_dim],
                vec![num_kv_heads * head_dim, 1],
                offset,
            );
            let k_head = ops::copy::copy_into_cb(registry, &k_head, cb2)?;
            let k_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope_ext_into_cb(
                    registry,
                    &k_head,
                    cos,
                    sin,
                    rope_offset,
                    1.0,
                    false,
                    true,
                    cb2,
                )?
            } else {
                k_head
            };
            k_heads.push(k_head);
        }

        // Split V into heads, copy contiguous
        let mut v_heads: Vec<Array> = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let offset = v.offset() + h * head_dim * elem_size;
            let v_head = v.view(
                vec![seq_len, head_dim],
                vec![num_kv_heads * head_dim, 1],
                offset,
            );
            let v_head = ops::copy::copy_into_cb(registry, &v_head, cb2)?;
            v_heads.push(v_head);
        }

        // Cache append (into same CB)
        let (k_final, v_final) = match cache {
            Some(ref mut c) => {
                c.append_into_cb(k_heads, v_heads, seq_len, registry, cb2)?;
                let kf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_keys(h)).collect();
                let vf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_values(h)).collect();
                (kf, vf)
            }
            None => (k_heads, v_heads),
        };

        let t2 = graph.submit_batch();

        // ---- CB3: SDPA ----
        graph.wait_for(t2);
        let cb3 = graph.command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_outputs = ops::sdpa::sdpa_batched_into_cb(
            registry, &q_heads, &k_final, &v_final, mask, scale, cb3,
        )?;
        let t3 = graph.submit_batch();

        // ---- CB4: head concat + O_proj ----
        graph.wait_for(t3);
        let cb4 = graph.command_buffer();

        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());
        let head_bytes = head_dim * elem_size;

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

        // Interleave head outputs into [seq_len, hidden_size]
        for (h, head_out) in attn_outputs.iter().enumerate() {
            let dst_col_offset = h * head_bytes;
            if seq_len == 1 {
                let enc = cb4.new_compute_command_encoder();
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
                let hidden_bytes_stride = hidden_size * elem_size;
                let blit = cb4.new_blit_command_encoder();
                for row in 0..seq_len {
                    let src_off = (head_out.offset() + row * head_bytes) as u64;
                    let dst_off = (row * hidden_bytes_stride + dst_col_offset) as u64;
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

        // O projection
        let output = self.o_proj.forward_into_cb(&concat, registry, cb4)?;
        let t4 = graph.submit_batch();

        Ok((output, t4))
    }

    /// Fused ExecGraph attention: norm + projections in CB1 (3 CBs total).
    ///
    /// Architecture (saves 1 CB vs `forward_graph`):
    /// - CB1: RMS norm + Q/K/V projections (fused)
    /// - CB2: head split + RoPE + cache append
    /// - CB3: SDPA + head concat + O_proj
    ///
    /// Returns `(output_array, EventToken)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_fused(
        &self,
        x: &Array,
        norm_weight: &Array,
        rms_norm_eps: f32,
        cos_freqs: Option<&Array>,
        sin_freqs: Option<&Array>,
        mask: Option<&Array>,
        mut cache: Option<&mut LayerKvCache>,
        registry: &KernelRegistry,
        graph: &mut ExecGraph<'_, '_>,
    ) -> Result<(Array, EventToken), KernelError> {
        let seq_len = x.shape()[0];
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;

        // ---- CB1: norm + Q/K/V projections (fused) ----
        let cb1 = graph.command_buffer();
        let normed =
            ops::rms_norm::rms_norm_into_cb(registry, x, Some(norm_weight), rms_norm_eps, cb1)?;
        let q = self.q_proj.forward_into_cb(&normed, registry, cb1)?;
        let k = self.k_proj.forward_into_cb(&normed, registry, cb1)?;
        let v = self.v_proj.forward_into_cb(&normed, registry, cb1)?;
        let t1 = graph.submit_batch();

        // ---- CB2: head split + RoPE + cache append ----
        graph.wait_for(t1);
        let cb2 = graph.command_buffer();

        let dev = registry.device().raw();
        let elem_size = q.dtype().size_of();
        let rope_offset = cache.as_ref().map_or(0u32, |c| c.seq_len as u32);

        let mut q_heads: Vec<Array> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let offset = q.offset() + h * head_dim * elem_size;
            let q_head = q.view(
                vec![seq_len, head_dim],
                vec![num_heads * head_dim, 1],
                offset,
            );
            let q_head = ops::copy::copy_into_cb(registry, &q_head, cb2)?;
            let q_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope_ext_into_cb(
                    registry,
                    &q_head,
                    cos,
                    sin,
                    rope_offset,
                    1.0,
                    false,
                    true,
                    cb2,
                )?
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
            let k_head = ops::copy::copy_into_cb(registry, &k_head, cb2)?;
            let k_head = if let (Some(cos), Some(sin)) = (cos_freqs, sin_freqs) {
                ops::rope::rope_ext_into_cb(
                    registry,
                    &k_head,
                    cos,
                    sin,
                    rope_offset,
                    1.0,
                    false,
                    true,
                    cb2,
                )?
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
            let v_head = ops::copy::copy_into_cb(registry, &v_head, cb2)?;
            v_heads.push(v_head);
        }

        let (k_final, v_final) = match cache {
            Some(ref mut c) => {
                c.append_into_cb(k_heads, v_heads, seq_len, registry, cb2)?;
                let kf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_keys(h)).collect();
                let vf: Vec<Array> = (0..num_kv_heads).map(|h| c.cached_values(h)).collect();
                (kf, vf)
            }
            None => (k_heads, v_heads),
        };

        let t2 = graph.submit_batch();

        // ---- CB3: SDPA + head concat + O_proj ----
        graph.wait_for(t2);
        let cb3 = graph.command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_outputs = ops::sdpa::sdpa_batched_into_cb(
            registry, &q_heads, &k_final, &v_final, mask, scale, cb3,
        )?;

        // Head concat
        let hidden_size = num_heads * head_dim;
        let concat = Array::zeros(dev, &[seq_len, hidden_size], q.dtype());
        let head_bytes = head_dim * elem_size;

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

        for (h, head_out) in attn_outputs.iter().enumerate() {
            let dst_col_offset = h * head_bytes;
            if seq_len == 1 {
                let enc = cb3.new_compute_command_encoder();
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
                let hidden_bytes_stride = hidden_size * elem_size;
                let blit = cb3.new_blit_command_encoder();
                for row in 0..seq_len {
                    let src_off = (head_out.offset() + row * head_bytes) as u64;
                    let dst_off = (row * hidden_bytes_stride + dst_col_offset) as u64;
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

        let output = self.o_proj.forward_into_cb(&concat, registry, cb3)?;
        let t3 = graph.submit_batch();

        Ok((output, t3))
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
    pub fn offset(&self, layer: usize) -> usize {
        self.offsets[layer]
    }

    /// Whether the cache is empty for a given layer.
    pub fn is_empty(&self, layer: usize) -> bool {
        self.offsets[layer] == 0
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
        _queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        if new_keys.len() != self.num_kv_heads || new_values.len() != self.num_kv_heads {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedKvCache::append: expected {} kv heads, got keys={}, values={}",
                self.num_kv_heads,
                new_keys.len(),
                new_values.len()
            )));
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
    pub fn quantized_keys(&self, layer: usize, head: usize) -> &QuantizedArray {
        &self.keys[layer][head]
    }

    /// Get the quantized values for a specific layer and head.
    pub fn quantized_values(&self, layer: usize, head: usize) -> &QuantizedArray {
        &self.values[layer][head]
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
    device: &metal::Device,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (metal::Device, KernelRegistry, metal::CommandQueue) {
        let device = metal::Device::system_default().expect("no Metal device");
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("no GpuDevice");
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        ops::copy::register(&registry).expect("failed to register copy kernels");
        (device, registry, queue)
    }

    /// Helper: create an Array from a flat f32 slice with shape [rows, cols].
    fn make_array(device: &metal::Device, data: &[f32], rows: usize, cols: usize) -> Array {
        assert_eq!(data.len(), rows * cols);
        Array::from_slice(device, data, vec![rows, cols])
    }

    /// Helper: read f32 data from an Array.
    fn read_f32(arr: &Array) -> Vec<f32> {
        arr.to_vec_checked::<f32>()
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
            &device,
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

            let k_arr = make_array(&device, &token_data, 1, head_dim);
            let v_arr = make_array(&device, &token_data, 1, head_dim);
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
        let mut cache = RotatingKvCache::new(&device, 1, head_dim, max_size, 0, DType::Float32);

        // Append 3 tokens at once
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let k = make_array(&device, &data, 3, head_dim);
        let v = make_array(&device, &data, 3, head_dim);
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
        let mut cache = RotatingKvCache::new(&device, 1, head_dim, max_size, 0, DType::Float32);

        // Fill to capacity with single tokens
        for t in 0..4 {
            let val = (t + 1) as f32;
            let data = vec![val; head_dim];
            let k = make_array(&device, &data, 1, head_dim);
            let v = make_array(&device, &data, 1, head_dim);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
        }

        // Now append 3 more tokens (wraps via linearize_and_append)
        let new_data: Vec<f32> = vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0];
        let k = make_array(&device, &new_data, 3, head_dim);
        let v = make_array(&device, &new_data, 3, head_dim);
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
        let mut cache = RotatingKvCache::new(&device, 1, 2, 4, 0, DType::Float32);

        for t in 0..10 {
            let data = vec![(t + 1) as f32; 2];
            let k = make_array(&device, &data, 1, 2);
            let v = make_array(&device, &data, 1, 2);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
            assert_eq!(cache.offset(), t + 1);
        }
    }

    #[test]
    fn test_rotating_cache_current_len() {
        let (device, registry, queue) = setup();
        let mut cache = RotatingKvCache::new(&device, 1, 2, 4, 0, DType::Float32);

        for t in 0..10 {
            let data = vec![(t + 1) as f32; 2];
            let k = make_array(&device, &data, 1, 2);
            let v = make_array(&device, &data, 1, 2);
            cache
                .append(vec![k], vec![v], 1, &registry, &queue)
                .expect("append failed");
            assert_eq!(cache.current_len(), std::cmp::min(t + 1, 4));
        }
    }
}
