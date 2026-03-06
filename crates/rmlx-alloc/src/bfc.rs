//! BFC (Best-Fit with Coalescing) allocator for sub-allocating from a single
//! pre-allocated Metal buffer.
//!
//! This allocator manages a contiguous region of GPU memory by splitting free
//! blocks on allocation and coalescing adjacent free blocks on deallocation.
//! It uses a BTreeMap keyed by `(size, BlockId)` for O(log n) best-fit lookup.
//!
//! # Usage
//!
//! ```ignore
//! let device = Arc::new(GpuDevice::system_default()?);
//! let mut bfc = BfcAllocator::new(&device, 64 * 1024 * 1024)?; // 64 MiB region
//! let alloc = bfc.alloc(4096)?;
//! // use alloc.offset and alloc.size to index into the backing Metal buffer
//! bfc.free(alloc)?;
//! ```

use std::collections::BTreeMap;

use crate::AllocError;

/// Minimum alignment for all allocations (256 bytes, matching Metal buffer
/// offset alignment requirements for `setBuffer:offset:atIndex:`).
const MIN_ALIGNMENT: usize = 256;

/// When a free block is larger than the request by at least this many bytes,
/// the remainder is split off into a separate free block. Below this threshold
/// the extra bytes are left as internal fragmentation to avoid block explosion.
const MIN_SPLIT_REMAINDER: usize = 256;

// ---------------------------------------------------------------------------
// Block arena
// ---------------------------------------------------------------------------

/// Opaque handle to a block in the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

/// A contiguous region within the backing buffer.
#[derive(Debug)]
struct BfcBlock {
    /// Byte offset from the start of the backing buffer.
    offset: usize,
    /// Size of this block in bytes.
    size: usize,
    /// Whether the block is currently allocated (in use).
    in_use: bool,
    /// Previous block in address order (None if this is the first block).
    prev: Option<BlockId>,
    /// Next block in address order (None if this is the last block).
    next: Option<BlockId>,
}

/// Simple arena for block storage. Freed slots are recycled via a free list.
struct BlockArena {
    blocks: Vec<Option<BfcBlock>>,
    free_slots: Vec<usize>,
}

impl BlockArena {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    /// Insert a block and return its id.
    fn insert(&mut self, block: BfcBlock) -> BlockId {
        if let Some(slot) = self.free_slots.pop() {
            self.blocks[slot] = Some(block);
            BlockId(slot)
        } else {
            let id = self.blocks.len();
            self.blocks.push(Some(block));
            BlockId(id)
        }
    }

    /// Get a reference to a block.
    fn get(&self, id: BlockId) -> Option<&BfcBlock> {
        self.blocks.get(id.0).and_then(|b| b.as_ref())
    }

    /// Get a mutable reference to a block.
    fn get_mut(&mut self, id: BlockId) -> Option<&mut BfcBlock> {
        self.blocks.get_mut(id.0).and_then(|b| b.as_mut())
    }

    /// Remove a block, recycling its slot.
    fn remove(&mut self, id: BlockId) -> Option<BfcBlock> {
        if id.0 < self.blocks.len() {
            let block = self.blocks[id.0].take();
            if block.is_some() {
                self.free_slots.push(id.0);
            }
            block
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// BFC allocation handle
// ---------------------------------------------------------------------------

/// A handle returned by [`BfcAllocator::alloc`] representing a sub-region of
/// the backing Metal buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BfcAllocation {
    /// Byte offset within the backing Metal buffer.
    pub offset: usize,
    /// Usable size in bytes.
    pub size: usize,
    /// Internal block id (used by `free`).
    block_id: BlockId,
}

// ---------------------------------------------------------------------------
// BFC allocator
// ---------------------------------------------------------------------------

/// Best-Fit with Coalescing allocator.
///
/// Manages sub-allocations from a single contiguous region. The caller is
/// responsible for creating and holding the backing Metal buffer; this
/// allocator only tracks offsets and sizes.
pub struct BfcAllocator {
    /// Total size of the managed region.
    region_size: usize,
    /// Block arena.
    arena: BlockArena,
    /// Free blocks indexed by `(size, BlockId)` for best-fit lookup.
    free_by_size: BTreeMap<(usize, BlockId), ()>,
    /// Current number of bytes in use.
    allocated_bytes: usize,
    /// Peak allocated bytes (high-water mark).
    peak_allocated: usize,
}

impl BfcAllocator {
    /// Create a new BFC allocator managing a region of `region_size` bytes.
    ///
    /// The caller should create a Metal buffer of at least `region_size` bytes
    /// and use the returned [`BfcAllocation::offset`] values to index into it.
    ///
    /// Returns `Err(AllocError::ZeroSize)` if `region_size` is 0.
    pub fn new(region_size: usize) -> Result<Self, AllocError> {
        if region_size == 0 {
            return Err(AllocError::ZeroSize);
        }

        let mut arena = BlockArena::new();
        let mut free_by_size = BTreeMap::new();

        // Create a single free block spanning the entire region.
        let root = arena.insert(BfcBlock {
            offset: 0,
            size: region_size,
            in_use: false,
            prev: None,
            next: None,
        });
        free_by_size.insert((region_size, root), ());

        Ok(Self {
            region_size,
            arena,
            free_by_size,
            allocated_bytes: 0,
            peak_allocated: 0,
        })
    }

    /// Allocate `size` bytes from the managed region.
    ///
    /// The returned [`BfcAllocation`] contains the offset and actual size.
    /// The actual size may be slightly larger than requested due to alignment.
    pub fn alloc(&mut self, size: usize) -> Result<BfcAllocation, AllocError> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }

        let aligned = align_up(size, MIN_ALIGNMENT);

        // Best-fit: find the smallest free block >= aligned size.
        // BTreeMap range from (aligned, BlockId(0)) gives us the first entry
        // with size >= aligned.
        let key = self
            .free_by_size
            .range((aligned, BlockId(0))..)
            .next()
            .map(|(&k, _)| k);

        let (block_size, block_id) = match key {
            Some(k) => k,
            None => {
                return Err(AllocError::OutOfMemory {
                    requested: aligned,
                    available: self.largest_free_block(),
                });
            }
        };

        // Remove from free set.
        self.free_by_size.remove(&(block_size, block_id));

        // Potentially split the block.
        let remainder = block_size - aligned;
        if remainder >= MIN_SPLIT_REMAINDER {
            // Split: shrink current block to `aligned`, create new free block
            // for the remainder.
            let block_offset = self.arena.get(block_id).unwrap().offset;
            let block_next = self.arena.get(block_id).unwrap().next;

            let new_block = BfcBlock {
                offset: block_offset + aligned,
                size: remainder,
                in_use: false,
                prev: Some(block_id),
                next: block_next,
            };
            let new_id = self.arena.insert(new_block);

            // Update the next block's prev pointer.
            if let Some(next_id) = block_next {
                if let Some(next_block) = self.arena.get_mut(next_id) {
                    next_block.prev = Some(new_id);
                }
            }

            // Update current block.
            if let Some(block) = self.arena.get_mut(block_id) {
                block.size = aligned;
                block.in_use = true;
                block.next = Some(new_id);
            }

            // Add remainder to free set.
            self.free_by_size.insert((remainder, new_id), ());
        } else {
            // No split; use the entire block (including remainder as internal
            // fragmentation).
            if let Some(block) = self.arena.get_mut(block_id) {
                block.in_use = true;
            }
        }

        let block = self.arena.get(block_id).unwrap();
        let allocation = BfcAllocation {
            offset: block.offset,
            size: block.size,
            block_id,
        };

        self.allocated_bytes += allocation.size;
        if self.allocated_bytes > self.peak_allocated {
            self.peak_allocated = self.allocated_bytes;
        }

        Ok(allocation)
    }

    /// Free a previously allocated region, coalescing with adjacent free blocks.
    pub fn free(&mut self, allocation: BfcAllocation) -> Result<(), AllocError> {
        let block_id = allocation.block_id;

        // Validate the block exists and is in use.
        {
            let block = self.arena.get(block_id).ok_or(AllocError::InvalidFree)?;
            if !block.in_use {
                return Err(AllocError::InvalidFree);
            }
            if block.offset != allocation.offset || block.size != allocation.size {
                return Err(AllocError::InvalidFree);
            }
        }

        self.allocated_bytes = self.allocated_bytes.saturating_sub(allocation.size);

        // Mark as free.
        self.arena.get_mut(block_id).unwrap().in_use = false;

        // Coalesce with next neighbor.
        let mut current_id = block_id;
        if let Some(next_id) = self.arena.get(current_id).unwrap().next {
            if let Some(next_block) = self.arena.get(next_id) {
                if !next_block.in_use {
                    let next_size = next_block.size;
                    let next_next = next_block.next;

                    // Remove neighbor from free set.
                    self.free_by_size.remove(&(next_size, next_id));

                    // Absorb into current block.
                    if let Some(block) = self.arena.get_mut(current_id) {
                        block.size += next_size;
                        block.next = next_next;
                    }

                    // Update the block after the absorbed neighbor.
                    if let Some(nn_id) = next_next {
                        if let Some(nn_block) = self.arena.get_mut(nn_id) {
                            nn_block.prev = Some(current_id);
                        }
                    }

                    // Remove the absorbed block from arena.
                    self.arena.remove(next_id);
                }
            }
        }

        // Coalesce with prev neighbor.
        if let Some(prev_id) = self.arena.get(current_id).unwrap().prev {
            if let Some(prev_block) = self.arena.get(prev_id) {
                if !prev_block.in_use {
                    let prev_size = prev_block.size;
                    let current_size = self.arena.get(current_id).unwrap().size;
                    let current_next = self.arena.get(current_id).unwrap().next;

                    // Remove prev from free set.
                    self.free_by_size.remove(&(prev_size, prev_id));

                    // Absorb current into prev.
                    if let Some(prev) = self.arena.get_mut(prev_id) {
                        prev.size += current_size;
                        prev.next = current_next;
                    }

                    // Update the block after current.
                    if let Some(cn_id) = current_next {
                        if let Some(cn_block) = self.arena.get_mut(cn_id) {
                            cn_block.prev = Some(prev_id);
                        }
                    }

                    // Remove current from arena (it was absorbed into prev).
                    self.arena.remove(current_id);
                    current_id = prev_id;
                }
            }
        }

        // Add the (possibly coalesced) block to the free set.
        let final_size = self.arena.get(current_id).unwrap().size;
        self.free_by_size.insert((final_size, current_id), ());

        Ok(())
    }

    /// Total size of the managed region.
    pub fn region_size(&self) -> usize {
        self.region_size
    }

    /// Current number of allocated bytes.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Peak allocated bytes (high-water mark).
    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated
    }

    /// Number of free blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_by_size.len()
    }

    /// Total free bytes across all free blocks.
    pub fn free_bytes(&self) -> usize {
        self.region_size - self.allocated_bytes
    }

    /// Size of the largest free block, or 0 if none.
    pub fn largest_free_block(&self) -> usize {
        self.free_by_size
            .keys()
            .next_back()
            .map(|(size, _)| *size)
            .unwrap_or(0)
    }

    /// Reset the peak allocated counter to the current value.
    pub fn reset_peak(&mut self) {
        self.peak_allocated = self.allocated_bytes;
    }
}

/// Round `size` up to the next multiple of `alignment`.
#[inline]
fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Ordering implementations for BlockId (required for BTreeMap keys)
// ---------------------------------------------------------------------------

impl PartialOrd for BlockId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc_free() {
        let mut bfc = BfcAllocator::new(1024 * 1024).unwrap();
        let a = bfc.alloc(4096).unwrap();
        assert!(a.size >= 4096);
        assert_eq!(a.offset, 0);
        assert!(bfc.allocated_bytes() >= 4096);

        bfc.free(a).unwrap();
        assert_eq!(bfc.allocated_bytes(), 0);
        assert_eq!(bfc.free_block_count(), 1);
    }

    #[test]
    fn test_zero_size_region_rejected() {
        assert!(BfcAllocator::new(0).is_err());
    }

    #[test]
    fn test_zero_size_alloc_rejected() {
        let mut bfc = BfcAllocator::new(4096).unwrap();
        assert!(bfc.alloc(0).is_err());
    }

    #[test]
    fn test_block_splitting() {
        let region = 1024 * 1024; // 1 MiB
        let mut bfc = BfcAllocator::new(region).unwrap();

        // Allocate a small block. Since the region is 1 MiB and we request
        // 4096, the initial block should be split into [4096 | remainder].
        let a = bfc.alloc(4096).unwrap();
        assert_eq!(a.offset, 0);
        assert!(a.size >= 4096);

        // There should be exactly one free block (the remainder).
        assert_eq!(bfc.free_block_count(), 1);

        // The free block should have the rest of the region.
        let expected_free = region - a.size;
        assert_eq!(bfc.largest_free_block(), expected_free);

        // Allocate another small block — it should come from the remainder.
        let b = bfc.alloc(4096).unwrap();
        assert_eq!(b.offset, a.size); // immediately after the first block
        assert!(b.size >= 4096);

        bfc.free(a).unwrap();
        bfc.free(b).unwrap();
    }

    #[test]
    fn test_coalescing_adjacent_blocks() {
        let region = 1024 * 1024;
        let mut bfc = BfcAllocator::new(region).unwrap();

        let a = bfc.alloc(4096).unwrap();
        let b = bfc.alloc(4096).unwrap();
        let c = bfc.alloc(4096).unwrap();

        // Free the middle block first.
        bfc.free(b).unwrap();
        // Free block count: the middle is free, plus the tail remainder.
        assert_eq!(bfc.free_block_count(), 2);

        // Free the first block. It should coalesce with the middle.
        bfc.free(a).unwrap();
        // Now the first two freed blocks should be merged.
        assert_eq!(bfc.free_block_count(), 2);

        // Free the last allocated block. All three plus the tail should merge
        // into one big free block covering the entire region.
        bfc.free(c).unwrap();
        assert_eq!(bfc.free_block_count(), 1);
        assert_eq!(bfc.largest_free_block(), region);
        assert_eq!(bfc.allocated_bytes(), 0);
    }

    #[test]
    fn test_coalescing_reverse_order() {
        let region = 1024 * 1024;
        let mut bfc = BfcAllocator::new(region).unwrap();

        let a = bfc.alloc(4096).unwrap();
        let b = bfc.alloc(4096).unwrap();

        // Free in reverse order.
        bfc.free(b).unwrap();
        bfc.free(a).unwrap();

        // Everything should coalesce back to a single free block.
        assert_eq!(bfc.free_block_count(), 1);
        assert_eq!(bfc.largest_free_block(), region);
    }

    #[test]
    fn test_out_of_memory() {
        let mut bfc = BfcAllocator::new(4096).unwrap();
        // Allocate the entire region.
        let _a = bfc.alloc(4096).unwrap();
        // A second allocation should fail.
        let result = bfc.alloc(256);
        assert!(result.is_err());
        match result.unwrap_err() {
            AllocError::OutOfMemory { .. } => {}
            other => panic!("expected OutOfMemory, got: {other}"),
        }
    }

    #[test]
    fn test_double_free_rejected() {
        let mut bfc = BfcAllocator::new(4096).unwrap();
        let a = bfc.alloc(1024).unwrap();
        let a_copy = a; // Copy the allocation handle.
        bfc.free(a).unwrap();
        // Second free with the same handle should fail.
        assert!(bfc.free(a_copy).is_err());
    }

    #[test]
    fn test_best_fit_selection() {
        let region = 1024 * 1024;
        let mut bfc = BfcAllocator::new(region).unwrap();

        // Create blocks of different sizes separated by in-use blocks so that
        // freeing does not coalesce them into a single large block.
        // All sizes are multiples of MIN_ALIGNMENT (256).
        let a = bfc.alloc(1024).unwrap(); // 1 KiB hole candidate
        let _b = bfc.alloc(256).unwrap(); // separator (stays allocated)
        let c = bfc.alloc(4096).unwrap(); // 4 KiB hole candidate
        let _d = bfc.alloc(256).unwrap(); // separator (stays allocated)
        let e = bfc.alloc(2048).unwrap(); // 2 KiB hole candidate (exact fit)
        let _f = bfc.alloc(256).unwrap(); // separator after e (prevents tail coalescing)

        let e_offset = e.offset;

        // Free a (1 KiB), c (4 KiB), e (2 KiB) — creating three isolated holes.
        bfc.free(a).unwrap();
        bfc.free(c).unwrap();
        bfc.free(e).unwrap();

        // Now allocate 2 KiB. Best-fit should pick the 2 KiB hole (e's slot),
        // not the 1 KiB (too small) or 4 KiB (unnecessarily large).
        let g = bfc.alloc(2048).unwrap();
        assert_eq!(g.size, 2048);
        assert_eq!(g.offset, e_offset);

        bfc.free(g).unwrap();
    }

    #[test]
    fn test_fragmentation_stress() {
        let region = 256 * 1024; // 256 KiB
        let mut bfc = BfcAllocator::new(region).unwrap();
        let mut allocations = Vec::new();

        // Allocate many small blocks.
        for _ in 0..100 {
            match bfc.alloc(512) {
                Ok(a) => allocations.push(a),
                Err(_) => break,
            }
        }

        // Free every other block to create fragmentation.
        let mut freed = Vec::new();
        for (i, a) in allocations.drain(..).enumerate() {
            if i % 2 == 0 {
                bfc.free(a).unwrap();
            } else {
                freed.push(a);
            }
        }

        // Re-allocate — should reuse freed holes.
        let mut re_allocs = Vec::new();
        for _ in 0..freed.len() {
            match bfc.alloc(512) {
                Ok(a) => re_allocs.push(a),
                Err(_) => break,
            }
        }

        // Free everything.
        for a in freed {
            bfc.free(a).unwrap();
        }
        for a in re_allocs {
            bfc.free(a).unwrap();
        }

        // After freeing everything, the region should be fully coalesced.
        assert_eq!(bfc.free_block_count(), 1);
        assert_eq!(bfc.largest_free_block(), region);
        assert_eq!(bfc.allocated_bytes(), 0);
    }

    #[test]
    fn test_alignment() {
        let mut bfc = BfcAllocator::new(1024 * 1024).unwrap();

        // Allocate an odd size; offset should still be aligned.
        let a = bfc.alloc(100).unwrap();
        assert_eq!(a.offset % MIN_ALIGNMENT, 0);
        assert!(a.size >= 100);
        assert_eq!(a.size % MIN_ALIGNMENT, 0);

        let b = bfc.alloc(300).unwrap();
        assert_eq!(b.offset % MIN_ALIGNMENT, 0);
        assert!(b.size >= 300);

        bfc.free(a).unwrap();
        bfc.free(b).unwrap();
    }

    #[test]
    fn test_peak_tracking() {
        let mut bfc = BfcAllocator::new(1024 * 1024).unwrap();

        let a = bfc.alloc(4096).unwrap();
        let b = bfc.alloc(8192).unwrap();
        let peak_after_two = bfc.peak_allocated();
        assert!(peak_after_two >= 4096 + 8192);

        bfc.free(a).unwrap();
        // Peak should not decrease after free.
        assert_eq!(bfc.peak_allocated(), peak_after_two);

        bfc.free(b).unwrap();
        assert_eq!(bfc.peak_allocated(), peak_after_two);

        bfc.reset_peak();
        assert_eq!(bfc.peak_allocated(), 0);
    }

    #[test]
    fn test_exact_fit_no_split() {
        // When the remainder would be smaller than MIN_SPLIT_REMAINDER,
        // the block should NOT be split.
        let region = MIN_ALIGNMENT + MIN_SPLIT_REMAINDER - 1;
        let mut bfc = BfcAllocator::new(region).unwrap();

        let a = bfc.alloc(MIN_ALIGNMENT).unwrap();
        // The block should consume the entire region (no split because
        // remainder < MIN_SPLIT_REMAINDER).
        assert_eq!(a.size, region);
        assert_eq!(bfc.free_block_count(), 0);

        bfc.free(a).unwrap();
        assert_eq!(bfc.free_block_count(), 1);
    }
}
