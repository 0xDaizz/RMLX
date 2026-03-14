//! Paged KV cache with block manager for efficient memory management.
//!
//! Implements vLLM-style paged attention memory management:
//! - A fixed pool of blocks backed by a single large Metal buffer
//! - Block-table indirection per sequence
//! - Copy-on-write via reference counting for forked sequences

use std::collections::HashMap;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_metal::{MtlBuffer, MtlDevice};

// ---------------------------------------------------------------------------
// BlockId
// ---------------------------------------------------------------------------

/// Opaque identifier for a block in the block pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

// ---------------------------------------------------------------------------
// BlockManager
// ---------------------------------------------------------------------------

/// Manages a fixed-size pool of memory blocks for paged KV caching.
///
/// Each block stores `[block_size, num_kv_heads, head_dim]` elements for both
/// keys and values. The entire pool is backed by a single large Metal buffer
/// (StorageModeShared), divided into equal-sized block segments.
///
/// Supports copy-on-write via reference counting: when a sequence is forked,
/// its blocks are shared (refcount incremented) rather than copied. A physical
/// copy is made only when a shared block is written to.
pub struct BlockManager {
    /// Total number of blocks in the pool.
    num_blocks: usize,
    /// Tokens per block.
    block_size: usize,
    /// Number of layers (each layer has its own K and V storage within each block).
    _num_layers: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Data type for stored KV entries.
    dtype: DType,
    /// The single large Metal buffer backing the entire block pool.
    /// Layout: block_id -> layer -> {K, V} -> [block_size, num_kv_heads, head_dim]
    pool_buffer: MtlBuffer,
    /// Byte size of one block (covering all layers, K+V).
    block_byte_size: usize,
    /// Byte size of one layer-half (K or V) within a block.
    layer_half_byte_size: usize,
    /// Free block IDs (stack-based free list).
    free_list: Vec<BlockId>,
    /// Block table per sequence: maps sequence_id -> ordered list of block IDs.
    block_tables: HashMap<u64, Vec<BlockId>>,
    /// Reference counts per block. Only tracked for blocks with refcount > 1.
    /// Blocks not in this map implicitly have refcount 1 (if allocated).
    ref_counts: HashMap<BlockId, usize>,
    /// Reference to the Metal device (needed for copy-on-write buffer creation).
    device: MtlDevice,
}

impl BlockManager {
    /// Create a new block manager with a pre-allocated pool.
    ///
    /// # Arguments
    /// * `device` - Metal device for buffer allocation
    /// * `num_blocks` - Total number of blocks in the pool
    /// * `block_size` - Number of tokens per block
    /// * `num_layers` - Number of transformer layers
    /// * `num_kv_heads` - Number of KV attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `dtype` - Data type (e.g., Float16, Float32)
    pub fn new(
        device: &MtlDevice,
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        // Each layer-half (K or V) stores [block_size, num_kv_heads, head_dim].
        let elements_per_layer_half = block_size * num_kv_heads * head_dim;
        let layer_half_byte_size = elements_per_layer_half * dtype.size_of();
        // Each block has num_layers * 2 halves (K + V per layer).
        let block_byte_size = num_layers * 2 * layer_half_byte_size;
        let total_bytes = num_blocks * block_byte_size;

        // Allocate at least 1 byte (Metal returns null for zero-length).
        let alloc_size = total_bytes.max(1);
        let pool_buffer = device
            .newBufferWithLength_options(alloc_size, MTLResourceOptions::StorageModeShared)
            .expect("failed to allocate pool buffer");

        // Zero the buffer for deterministic initial state.
        // SAFETY: StorageModeShared buffer contents() is CPU-accessible.
        unsafe {
            std::ptr::write_bytes(pool_buffer.contents().as_ptr() as *mut u8, 0, alloc_size);
        }

        // Initialize free list (all blocks available, in reverse order so pop gives 0 first).
        let free_list: Vec<BlockId> = (0..num_blocks as u32).rev().map(BlockId).collect();

        Self {
            num_blocks,
            block_size,
            _num_layers: num_layers,
            num_kv_heads,
            head_dim,
            dtype,
            pool_buffer,
            block_byte_size,
            layer_half_byte_size,
            free_list,
            block_tables: HashMap::new(),
            ref_counts: HashMap::new(),
            device: device.clone(),
        }
    }

    /// Allocate a free block from the pool.
    ///
    /// Returns `Err` if no blocks are available.
    pub fn allocate(&mut self) -> Result<BlockId, PagedKvError> {
        self.free_list.pop().ok_or(PagedKvError::OutOfBlocks)
    }

    /// Free a block, returning it to the pool.
    ///
    /// If the block has a refcount > 1, the refcount is decremented instead
    /// of actually freeing the block.
    pub fn free(&mut self, block_id: BlockId) {
        let rc = self.ref_counts.get_mut(&block_id);
        match rc {
            Some(count) if *count > 1 => {
                *count -= 1;
            }
            Some(_) => {
                // refcount drops to 1 (or was 1), remove tracking and free.
                self.ref_counts.remove(&block_id);
                self.free_list.push(block_id);
            }
            None => {
                // Not ref-counted (implicitly refcount=1), just free.
                self.free_list.push(block_id);
            }
        }
    }

    /// Number of currently free (unallocated) blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the block table for a sequence.
    ///
    /// Returns an empty slice if the sequence has no blocks.
    pub fn block_table(&self, sequence_id: u64) -> &[BlockId] {
        self.block_tables
            .get(&sequence_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Append a newly allocated block to a sequence's block table.
    ///
    /// Allocates a block from the free list and adds it to the sequence.
    pub fn append_block(&mut self, sequence_id: u64) -> Result<BlockId, PagedKvError> {
        let block_id = self.allocate()?;
        self.block_tables
            .entry(sequence_id)
            .or_default()
            .push(block_id);
        Ok(block_id)
    }

    /// Fork a sequence: share all blocks from `src_seq` to `dst_seq` via CoW.
    ///
    /// The destination sequence gets a copy of the source's block table, and
    /// each shared block's reference count is incremented. Actual data copying
    /// is deferred until a write to a shared block (copy-on-write).
    pub fn fork(&mut self, src_seq: u64, dst_seq: u64) -> Result<(), PagedKvError> {
        let src_blocks = self
            .block_tables
            .get(&src_seq)
            .ok_or(PagedKvError::SequenceNotFound(src_seq))?
            .clone();

        // Increment refcount for each shared block.
        for &block_id in &src_blocks {
            let rc = self.ref_counts.entry(block_id).or_insert(1);
            *rc += 1;
        }

        self.block_tables.insert(dst_seq, src_blocks);
        Ok(())
    }

    /// Remove and return the last block from a sequence's block table.
    /// Used by trim operations to free trailing empty blocks.
    pub fn pop_last_block(&mut self, seq_id: u64) -> Option<BlockId> {
        self.block_tables
            .get_mut(&seq_id)
            .and_then(|table| table.pop())
    }

    /// Check whether a sequence has an entry in the block tables.
    pub fn has_sequence(&self, seq_id: u64) -> bool {
        self.block_tables.contains_key(&seq_id)
    }

    /// Free all blocks owned by a sequence and remove its block table.
    pub fn free_sequence(&mut self, sequence_id: u64) {
        if let Some(blocks) = self.block_tables.remove(&sequence_id) {
            for block_id in blocks {
                self.free(block_id);
            }
        }
    }

    /// Byte offset of a block's layer-half (K or V) within the pool buffer.
    ///
    /// `is_value`: false for keys, true for values.
    fn block_offset(&self, block_id: BlockId, layer: usize, is_value: bool) -> usize {
        let block_start = block_id.0 as usize * self.block_byte_size;
        let layer_offset = layer * 2 * self.layer_half_byte_size;
        let half_offset = if is_value {
            self.layer_half_byte_size
        } else {
            0
        };
        block_start + layer_offset + half_offset
    }

    /// Check if a block is shared (refcount > 1) and needs copy-on-write.
    fn is_shared(&self, block_id: BlockId) -> bool {
        self.ref_counts.get(&block_id).is_some_and(|&rc| rc > 1)
    }

    /// Perform copy-on-write: allocate a new block, copy data, replace in table.
    ///
    /// Returns the new block ID that replaced the shared block at the given
    /// index in the sequence's block table.
    fn cow_copy(&mut self, sequence_id: u64, block_index: usize) -> Result<BlockId, PagedKvError> {
        let old_block = self.block_tables[&sequence_id][block_index];
        let new_block = self.allocate()?;

        // Copy the entire block's data.
        let old_offset = old_block.0 as usize * self.block_byte_size;
        let new_offset = new_block.0 as usize * self.block_byte_size;

        // SAFETY: Both offsets are within the pool buffer bounds, non-overlapping
        // (different block IDs), and the buffer is CPU-accessible (StorageModeShared).
        unsafe {
            let base = self.pool_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(
                base.add(old_offset),
                base.add(new_offset),
                self.block_byte_size,
            );
        }

        // Decrement old block's refcount.
        self.free(old_block);

        // Replace in block table.
        self.block_tables.get_mut(&sequence_id).unwrap()[block_index] = new_block;

        Ok(new_block)
    }

    /// Raw pointer to the pool buffer contents.
    fn pool_ptr(&self) -> *mut u8 {
        self.pool_buffer.contents().as_ptr() as *mut u8
    }
}

// ---------------------------------------------------------------------------
// PagedKvCache
// ---------------------------------------------------------------------------

/// Non-contiguous KV cache using paged memory blocks.
///
/// Stores key and value tensors across scattered fixed-size blocks managed by
/// a [`BlockManager`]. Each sequence maintains a block table that maps logical
/// token positions to physical blocks.
///
/// For now, `get_keys`/`get_values` performs a CPU-side gather (copies scattered
/// blocks into a contiguous output buffer). A GPU-side paged attention kernel
/// will be added later.
pub struct PagedKvCache {
    /// The block manager owning the physical block pool.
    block_manager: BlockManager,
    /// Number of tokens written per (sequence, layer).
    /// Maps (sequence_id, layer) -> token count.
    layer_lengths: HashMap<(u64, usize), usize>,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// # Arguments
    /// * `block_manager` - Pre-configured block manager with allocated pool
    pub fn new(block_manager: BlockManager) -> Self {
        Self {
            block_manager,
            layer_lengths: HashMap::new(),
        }
    }

    /// Access the underlying block manager.
    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    /// Mutable access to the underlying block manager.
    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
    }

    /// Number of tokens currently cached for a sequence (max across all layers).
    pub fn seq_len(&self, seq_id: u64) -> usize {
        self.layer_lengths
            .iter()
            .filter(|(&(sid, _), _)| sid == seq_id)
            .map(|(_, &len)| len)
            .max()
            .unwrap_or(0)
    }

    /// Number of tokens cached for a specific (sequence, layer).
    pub fn layer_len(&self, seq_id: u64, layer: usize) -> usize {
        self.layer_lengths
            .get(&(seq_id, layer))
            .copied()
            .unwrap_or(0)
    }

    /// Append new key and value entries for a sequence at a given layer.
    ///
    /// `key` and `value` should have shape `[num_new_tokens, num_kv_heads, head_dim]`.
    /// New blocks are allocated as needed. Copy-on-write is triggered for
    /// shared blocks that would be written to.
    pub fn append(
        &mut self,
        layer: usize,
        seq_id: u64,
        key: &Array,
        value: &Array,
    ) -> Result<(), PagedKvError> {
        let bm = &self.block_manager;
        let block_size = bm.block_size;
        let num_kv_heads = bm.num_kv_heads;
        let head_dim = bm.head_dim;
        let dtype = bm.dtype;
        let elem_size = dtype.size_of();

        // Validate input shapes.
        let key_shape = key.shape();
        let value_shape = value.shape();
        if key_shape.len() != 3 || key_shape[1] != num_kv_heads || key_shape[2] != head_dim {
            return Err(PagedKvError::ShapeMismatch {
                expected: format!("[N, {num_kv_heads}, {head_dim}]"),
                got: format!("{key_shape:?}"),
            });
        }
        if value_shape != key_shape {
            return Err(PagedKvError::ShapeMismatch {
                expected: format!("{key_shape:?}"),
                got: format!("{value_shape:?}"),
            });
        }
        if key.dtype() != dtype {
            return Err(PagedKvError::DTypeMismatch {
                expected: dtype,
                got: key.dtype(),
            });
        }

        let num_new_tokens = key_shape[0];
        if num_new_tokens == 0 {
            return Ok(());
        }

        let current_len = self
            .layer_lengths
            .get(&(seq_id, layer))
            .copied()
            .unwrap_or(0);
        let token_row_bytes = num_kv_heads * head_dim * elem_size;

        // Read input data pointers.
        let key_ptr = key.metal_buffer().contents().as_ptr() as *const u8;
        let val_ptr = value.metal_buffer().contents().as_ptr() as *const u8;
        let key_offset = key.offset();
        let val_offset = value.offset();

        for i in 0..num_new_tokens {
            let global_pos = current_len + i;
            let block_index = global_pos / block_size;
            let slot_in_block = global_pos % block_size;

            // Ensure we have enough blocks allocated.
            let num_blocks = self.block_manager.block_table(seq_id).len();
            if block_index >= num_blocks {
                // Need a new block.
                self.block_manager.append_block(seq_id)?;
            }

            // Check for copy-on-write.
            let block_id = self.block_manager.block_table(seq_id)[block_index];
            if self.block_manager.is_shared(block_id) {
                self.block_manager.cow_copy(seq_id, block_index)?;
            }

            let block_id = self.block_manager.block_table(seq_id)[block_index];

            // Compute destination offset in pool buffer.
            let key_dst_offset = self.block_manager.block_offset(block_id, layer, false)
                + slot_in_block * token_row_bytes;
            let val_dst_offset = self.block_manager.block_offset(block_id, layer, true)
                + slot_in_block * token_row_bytes;

            // Source offset in input array.
            let src_offset = i * token_row_bytes;

            // SAFETY: Pool buffer is CPU-accessible (StorageModeShared). Offsets
            // are within bounds by construction (block_size * token_row_bytes fits
            // within layer_half_byte_size, and block is within pool).
            unsafe {
                let pool = self.block_manager.pool_ptr();
                std::ptr::copy_nonoverlapping(
                    key_ptr.add(key_offset + src_offset),
                    pool.add(key_dst_offset),
                    token_row_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    val_ptr.add(val_offset + src_offset),
                    pool.add(val_dst_offset),
                    token_row_bytes,
                );
            }
        }

        *self.layer_lengths.entry((seq_id, layer)).or_insert(0) += num_new_tokens;
        Ok(())
    }

    /// Retrieve all cached keys for a sequence at a given layer as a contiguous array.
    ///
    /// Returns an array of shape `[seq_len, num_kv_heads, head_dim]`.
    pub fn get_keys(&self, layer: usize, seq_id: u64) -> Result<Array, PagedKvError> {
        self.gather_layer(layer, seq_id, false)
    }

    /// Retrieve all cached values for a sequence at a given layer as a contiguous array.
    ///
    /// Returns an array of shape `[seq_len, num_kv_heads, head_dim]`.
    pub fn get_values(&self, layer: usize, seq_id: u64) -> Result<Array, PagedKvError> {
        self.gather_layer(layer, seq_id, true)
    }

    /// Gather scattered blocks into a contiguous array (CPU-side).
    fn gather_layer(
        &self,
        layer: usize,
        seq_id: u64,
        is_value: bool,
    ) -> Result<Array, PagedKvError> {
        let bm = &self.block_manager;
        let seq_len = self.layer_len(seq_id, layer);
        if seq_len == 0 {
            return Err(PagedKvError::SequenceNotFound(seq_id));
        }

        let num_kv_heads = bm.num_kv_heads;
        let head_dim = bm.head_dim;
        let dtype = bm.dtype;
        let block_size = bm.block_size;
        let elem_size = dtype.size_of();
        let token_row_bytes = num_kv_heads * head_dim * elem_size;

        let shape = [seq_len, num_kv_heads, head_dim];
        let total_bytes = seq_len * token_row_bytes;

        // Allocate output buffer.
        let out_buffer = bm
            .device
            .newBufferWithLength_options(total_bytes.max(1), MTLResourceOptions::StorageModeShared)
            .expect("failed to allocate output buffer");

        let block_table = bm.block_table(seq_id);
        let pool = bm.pool_ptr();
        let out_ptr = out_buffer.contents().as_ptr() as *mut u8;

        let mut tokens_remaining = seq_len;
        let mut dst_offset = 0usize;

        for (block_idx, &block_id) in block_table.iter().enumerate() {
            let tokens_in_block = if block_idx == block_table.len() - 1 {
                // Last block may be partially filled.
                let remainder = seq_len % block_size;
                if remainder == 0 {
                    block_size
                } else {
                    remainder
                }
            } else {
                block_size
            };
            let copy_tokens = tokens_in_block.min(tokens_remaining);
            let copy_bytes = copy_tokens * token_row_bytes;

            let src_offset = bm.block_offset(block_id, layer, is_value);

            // SAFETY: Source is within pool buffer bounds, destination is within
            // out_buffer bounds. Both are CPU-accessible StorageModeShared.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pool.add(src_offset),
                    out_ptr.add(dst_offset),
                    copy_bytes,
                );
            }

            dst_offset += copy_bytes;
            tokens_remaining -= copy_tokens;
        }

        let strides = vec![num_kv_heads * head_dim, head_dim, 1];
        Ok(Array::new(out_buffer, shape.to_vec(), strides, dtype, 0))
    }

    /// Remove the last `n` tokens from a sequence's cache across all layers.
    ///
    /// Frees any blocks that become completely empty after trimming.
    /// Partially-occupied blocks are kept but their logical length is reduced.
    /// This is used by speculative decoding to roll back rejected draft tokens.
    ///
    /// Returns the new sequence length after trimming, or `PagedKvError::SequenceNotFound`
    /// if the sequence doesn't exist.
    pub fn trim(&mut self, seq_id: u64, n: usize) -> Result<usize, PagedKvError> {
        // Check sequence exists
        if !self.block_manager.has_sequence(seq_id) {
            return Err(PagedKvError::SequenceNotFound(seq_id));
        }

        let block_size = self.block_manager.block_size();
        let old_seq_len = self.seq_len(seq_id);
        let new_seq_len = old_seq_len.saturating_sub(n);

        // Update all layer lengths for this sequence
        let layer_keys: Vec<(u64, usize)> = self
            .layer_lengths
            .keys()
            .filter(|(sid, _)| *sid == seq_id)
            .cloned()
            .collect();

        for key in &layer_keys {
            if let Some(len) = self.layer_lengths.get_mut(key) {
                *len = (*len).saturating_sub(n);
            }
        }

        // Calculate how many blocks are needed before and after trimming
        let old_num_blocks = old_seq_len.div_ceil(block_size);
        let new_num_blocks = if new_seq_len == 0 {
            0
        } else {
            new_seq_len.div_ceil(block_size)
        };

        // Free trailing blocks that are no longer needed
        let blocks_to_free = old_num_blocks.saturating_sub(new_num_blocks);
        for _ in 0..blocks_to_free {
            if let Some(block_id) = self.block_manager.pop_last_block(seq_id) {
                self.block_manager.free(block_id);
            }
        }

        // If new_seq_len is 0, clean up completely
        if new_seq_len == 0 {
            self.free_sequence(seq_id);
            return Ok(0);
        }

        Ok(new_seq_len)
    }

    /// Free all blocks for a sequence and remove its metadata.
    pub fn free_sequence(&mut self, seq_id: u64) {
        self.block_manager.free_sequence(seq_id);
        self.layer_lengths.retain(|&(sid, _), _| sid != seq_id);
    }

    /// Fork a sequence (copy-on-write).
    pub fn fork(&mut self, src_seq: u64, dst_seq: u64) -> Result<(), PagedKvError> {
        let src_len = self.seq_len(src_seq);
        if src_len == 0 {
            return Err(PagedKvError::SequenceNotFound(src_seq));
        }
        self.block_manager.fork(src_seq, dst_seq)?;
        // Copy all per-layer lengths from source to destination.
        let src_entries: Vec<(usize, usize)> = self
            .layer_lengths
            .iter()
            .filter(|(&(sid, _), _)| sid == src_seq)
            .map(|(&(_, layer), &len)| (layer, len))
            .collect();
        for (layer, len) in src_entries {
            self.layer_lengths.insert((dst_seq, layer), len);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from paged KV cache operations.
#[derive(Debug)]
pub enum PagedKvError {
    /// No free blocks available in the pool.
    OutOfBlocks,
    /// Sequence ID not found.
    SequenceNotFound(u64),
    /// Input array shape does not match expected dimensions.
    ShapeMismatch { expected: String, got: String },
    /// Input array dtype does not match cache dtype.
    DTypeMismatch { expected: DType, got: DType },
}

impl std::fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagedKvError::OutOfBlocks => write!(f, "paged KV cache: no free blocks available"),
            PagedKvError::SequenceNotFound(id) => {
                write!(f, "paged KV cache: sequence {id} not found")
            }
            PagedKvError::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "paged KV cache: shape mismatch, expected {expected}, got {got}"
                )
            }
            PagedKvError::DTypeMismatch { expected, got } => {
                write!(
                    f,
                    "paged KV cache: dtype mismatch, expected {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for PagedKvError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::OnceLock;

    fn test_device() -> &'static MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
        })
    }

    // Helper: create a block manager with small parameters for testing.
    fn test_block_manager(num_blocks: usize) -> BlockManager {
        BlockManager::new(
            test_device(),
            num_blocks,
            4, // block_size: 4 tokens per block
            2, // num_layers
            2, // num_kv_heads
            4, // head_dim
            DType::Float32,
        )
    }

    // Helper: create a test array of shape [num_tokens, 2, 4] with sequential f32 values.
    fn test_array(device: &MtlDevice, num_tokens: usize, start_val: f32) -> Array {
        let num_kv_heads = 2;
        let head_dim = 4;
        let numel = num_tokens * num_kv_heads * head_dim;
        let data: Vec<f32> = (0..numel).map(|i| start_val + i as f32).collect();
        Array::from_slice(device, &data, vec![num_tokens, num_kv_heads, head_dim])
    }

    #[test]
    fn test_allocate_free_blocks() {
        let mut bm = test_block_manager(8);
        assert_eq!(bm.num_free_blocks(), 8);

        let b0 = bm.allocate().unwrap();
        let b1 = bm.allocate().unwrap();
        assert_eq!(bm.num_free_blocks(), 6);
        assert_ne!(b0, b1);

        bm.free(b0);
        assert_eq!(bm.num_free_blocks(), 7);

        bm.free(b1);
        assert_eq!(bm.num_free_blocks(), 8);
    }

    #[test]
    fn test_out_of_blocks() {
        let mut bm = test_block_manager(2);
        let _b0 = bm.allocate().unwrap();
        let _b1 = bm.allocate().unwrap();

        match bm.allocate() {
            Err(PagedKvError::OutOfBlocks) => {} // expected
            other => panic!("expected OutOfBlocks, got {other:?}"),
        }
    }

    #[test]
    fn test_append_and_retrieve() {
        let device = test_device();
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;
        let key = test_array(device, 3, 1.0);
        let value = test_array(device, 3, 100.0);

        cache.append(0, seq_id, &key, &value).unwrap();

        assert_eq!(cache.seq_len(seq_id), 3);

        // Retrieve and verify keys.
        let got_keys = cache.get_keys(0, seq_id).unwrap();
        assert_eq!(got_keys.shape(), &[3, 2, 4]);
        let key_data: Vec<f32> = key.to_vec_checked();
        let got_key_data: Vec<f32> = got_keys.to_vec_checked();
        assert_eq!(key_data, got_key_data);

        // Retrieve and verify values.
        let got_values = cache.get_values(0, seq_id).unwrap();
        assert_eq!(got_values.shape(), &[3, 2, 4]);
        let val_data: Vec<f32> = value.to_vec_checked();
        let got_val_data: Vec<f32> = got_values.to_vec_checked();
        assert_eq!(val_data, got_val_data);
    }

    #[test]
    fn test_append_across_block_boundary() {
        let device = test_device();
        // block_size=4, so 6 tokens should span 2 blocks.
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;
        // Append 6 tokens (spans 2 blocks of size 4).
        let key = test_array(device, 6, 1.0);
        let value = test_array(device, 6, 100.0);
        cache.append(0, seq_id, &key, &value).unwrap();

        assert_eq!(cache.seq_len(seq_id), 6);
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 2);

        let got_keys = cache.get_keys(0, seq_id).unwrap();
        let expected: Vec<f32> = key.to_vec_checked();
        let actual: Vec<f32> = got_keys.to_vec_checked();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_incremental_append() {
        let device = test_device();
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;

        // First append: 3 tokens.
        let key1 = test_array(device, 3, 1.0);
        let val1 = test_array(device, 3, 100.0);
        cache.append(0, seq_id, &key1, &val1).unwrap();

        // Second append: 2 more tokens (should cross block boundary since block_size=4).
        let key2 = test_array(device, 2, 50.0);
        let val2 = test_array(device, 2, 200.0);
        cache.append(0, seq_id, &key2, &val2).unwrap();

        assert_eq!(cache.seq_len(seq_id), 5);
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 2);

        // Verify concatenated keys.
        let got_keys = cache.get_keys(0, seq_id).unwrap();
        let k1: Vec<f32> = key1.to_vec_checked();
        let k2: Vec<f32> = key2.to_vec_checked();
        let expected: Vec<f32> = k1.iter().chain(k2.iter()).copied().collect();
        let actual: Vec<f32> = got_keys.to_vec_checked();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_multiple_sequences() {
        let device = test_device();
        let bm = BlockManager::new(device, 16, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        // Sequence 1: 3 tokens.
        let key1 = test_array(device, 3, 1.0);
        let val1 = test_array(device, 3, 100.0);
        cache.append(0, 1, &key1, &val1).unwrap();

        // Sequence 2: 5 tokens.
        let key2 = test_array(device, 5, 50.0);
        let val2 = test_array(device, 5, 200.0);
        cache.append(0, 2, &key2, &val2).unwrap();

        assert_eq!(cache.seq_len(1), 3);
        assert_eq!(cache.seq_len(2), 5);

        // Verify each sequence independently.
        let got1: Vec<f32> = cache.get_keys(0, 1).unwrap().to_vec_checked();
        let exp1: Vec<f32> = key1.to_vec_checked();
        assert_eq!(got1, exp1);

        let got2: Vec<f32> = cache.get_keys(0, 2).unwrap().to_vec_checked();
        let exp2: Vec<f32> = key2.to_vec_checked();
        assert_eq!(got2, exp2);
    }

    #[test]
    fn test_multiple_layers() {
        let device = test_device();
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;

        // Write different data to layer 0 and layer 1.
        let key_l0 = test_array(device, 2, 1.0);
        let val_l0 = test_array(device, 2, 100.0);
        let key_l1 = test_array(device, 2, 500.0);
        let val_l1 = test_array(device, 2, 600.0);

        cache.append(0, seq_id, &key_l0, &val_l0).unwrap();
        cache.append(1, seq_id, &key_l1, &val_l1).unwrap();

        // Verify layer 0.
        let got_k0: Vec<f32> = cache.get_keys(0, seq_id).unwrap().to_vec_checked();
        let exp_k0: Vec<f32> = key_l0.to_vec_checked();
        assert_eq!(got_k0, exp_k0);

        // Verify layer 1.
        let got_k1: Vec<f32> = cache.get_keys(1, seq_id).unwrap().to_vec_checked();
        let exp_k1: Vec<f32> = key_l1.to_vec_checked();
        assert_eq!(got_k1, exp_k1);
    }

    #[test]
    fn test_fork_cow() {
        let device = test_device();
        let bm = BlockManager::new(device, 16, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let src_seq = 1;
        let dst_seq = 2;

        // Populate source sequence.
        let key = test_array(device, 3, 1.0);
        let val = test_array(device, 3, 100.0);
        cache.append(0, src_seq, &key, &val).unwrap();

        // Fork.
        cache.fork(src_seq, dst_seq).unwrap();

        // Both sequences should return the same data.
        let src_keys: Vec<f32> = cache.get_keys(0, src_seq).unwrap().to_vec_checked();
        let dst_keys: Vec<f32> = cache.get_keys(0, dst_seq).unwrap().to_vec_checked();
        assert_eq!(src_keys, dst_keys);

        // Blocks should be shared (same block IDs).
        let src_table = cache.block_manager().block_table(src_seq).to_vec();
        let dst_table = cache.block_manager().block_table(dst_seq).to_vec();
        assert_eq!(src_table, dst_table);

        // Append to dst_seq should trigger CoW.
        let new_key = test_array(device, 1, 999.0);
        let new_val = test_array(device, 1, 888.0);
        cache.append(0, dst_seq, &new_key, &new_val).unwrap();

        // Source should be unaffected.
        let src_keys_after: Vec<f32> = cache.get_keys(0, src_seq).unwrap().to_vec_checked();
        assert_eq!(src_keys, src_keys_after);

        // Destination should have the new token appended.
        assert_eq!(cache.seq_len(dst_seq), 4);
        let dst_keys_after: Vec<f32> = cache.get_keys(0, dst_seq).unwrap().to_vec_checked();
        let expected_dst: Vec<f32> = src_keys
            .iter()
            .chain(new_key.to_vec_checked::<f32>().iter())
            .copied()
            .collect();
        assert_eq!(dst_keys_after, expected_dst);

        // Block tables should now differ (CoW created a new block for dst).
        let dst_table_after = cache.block_manager().block_table(dst_seq).to_vec();
        assert_ne!(src_table, dst_table_after);
    }

    #[test]
    fn test_free_sequence() {
        let mut bm = test_block_manager(8);
        let initial_free = bm.num_free_blocks();

        bm.append_block(1).unwrap();
        bm.append_block(1).unwrap();
        assert_eq!(bm.num_free_blocks(), initial_free - 2);

        bm.free_sequence(1);
        assert_eq!(bm.num_free_blocks(), initial_free);
        assert!(bm.block_table(1).is_empty());
    }

    #[test]
    fn test_trim_partial() {
        let device = test_device();
        // block_size=4, so 6 tokens spans 2 blocks.
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;
        let key = test_array(device, 6, 1.0);
        let value = test_array(device, 6, 100.0);
        cache.append(0, seq_id, &key, &value).unwrap();
        cache.append(1, seq_id, &key, &value).unwrap();
        assert_eq!(cache.seq_len(seq_id), 6);
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 2);

        // Trim 1 token — still need 2 blocks (5 tokens, block_size=4).
        let new_len = cache.trim(seq_id, 1).unwrap();
        assert_eq!(new_len, 5);
        assert_eq!(cache.layer_len(seq_id, 0), 5);
        assert_eq!(cache.layer_len(seq_id, 1), 5);
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 2);
    }

    #[test]
    fn test_trim_frees_block() {
        let device = test_device();
        // block_size=4, so 6 tokens spans 2 blocks.
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;
        let key = test_array(device, 6, 1.0);
        let value = test_array(device, 6, 100.0);
        cache.append(0, seq_id, &key, &value).unwrap();
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 2);
        let free_before = cache.block_manager().num_free_blocks();

        // Trim 3 tokens: 6 -> 3, which fits in 1 block. Should free 1 block.
        let new_len = cache.trim(seq_id, 3).unwrap();
        assert_eq!(new_len, 3);
        assert_eq!(cache.block_manager().block_table(seq_id).len(), 1);
        assert_eq!(cache.block_manager().num_free_blocks(), free_before + 1);

        // Data in the remaining block should still be valid.
        let got_keys = cache.get_keys(0, seq_id).unwrap();
        assert_eq!(got_keys.shape(), &[3, 2, 4]);
    }

    #[test]
    fn test_trim_to_zero() {
        let device = test_device();
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        let seq_id = 1;
        let key = test_array(device, 3, 1.0);
        let value = test_array(device, 3, 100.0);
        cache.append(0, seq_id, &key, &value).unwrap();
        let free_before_append = cache.block_manager().num_free_blocks();

        // Trim all tokens.
        let new_len = cache.trim(seq_id, 10).unwrap();
        assert_eq!(new_len, 0);
        // All blocks freed — 1 block was used.
        assert_eq!(
            cache.block_manager().num_free_blocks(),
            free_before_append + 1
        );
        // Sequence should be fully cleaned up.
        assert_eq!(cache.seq_len(seq_id), 0);
    }

    #[test]
    fn test_trim_nonexistent_sequence() {
        let device = test_device();
        let bm = BlockManager::new(device, 8, 4, 2, 2, 4, DType::Float32);
        let mut cache = PagedKvCache::new(bm);

        match cache.trim(999, 1) {
            Err(PagedKvError::SequenceNotFound(999)) => {} // expected
            other => panic!("expected SequenceNotFound, got {other:?}"),
        }
    }
}
