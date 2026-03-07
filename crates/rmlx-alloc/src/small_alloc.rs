//! Small-buffer recycling allocator (A5).
//!
//! Sub-allocates fixed-size slots from a single large Metal buffer.
//! This reduces per-buffer Metal driver overhead compared to creating
//! individual buffers for each small allocation.
//!
//! `SmallBufferPool` pre-allocates a single Metal buffer and divides it
//! into fixed-size slots. Requests up to `MAX_SMALL_ALLOC` (256 bytes)
//! are served from this pool via offset-based sub-allocation.

use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal::Buffer as MetalBuffer;

/// Maximum allocation size served by the small-buffer pool.
pub const MAX_SMALL_ALLOC: usize = 256;

/// Default number of slots: 4096 = 1 MiB total.
const DEFAULT_NUM_SLOTS: usize = 4096;

/// A small allocation checked out from the recycling pool.
#[derive(Debug)]
pub struct SmallAllocation {
    /// Index of the slot in the pool (for returning to the free list).
    slot: usize,
    /// Byte offset within the backing buffer.
    pub offset: usize,
    /// Requested size (may be smaller than slot size).
    pub size: usize,
    /// Clone of the backing buffer (same MTLBuffer, different Rust handle).
    pub buffer: Option<MetalBuffer>,
}

impl SmallAllocation {
    /// The slot index used internally by the pool.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// Inner state protected by a mutex.
struct PoolInner {
    /// Free slot indices, stored as a stack for O(1) pop/push.
    free_list: Vec<usize>,
}

/// Sub-allocating pool backed by a single large Metal buffer.
///
/// All slots share the same underlying `MTLBuffer`, with each slot
/// at a different offset. This gives 16x more slots than the previous
/// per-buffer approach in the same memory footprint, with significantly
/// less Metal driver overhead.
pub struct SmallBufferPool {
    /// The single backing Metal buffer.
    backing_buffer: MetalBuffer,
    /// Slot size in bytes (= `MAX_SMALL_ALLOC`).
    slot_size: usize,
    /// Total number of slots.
    num_slots: usize,
    /// Protected mutable state.
    inner: Mutex<PoolInner>,
}

impl SmallBufferPool {
    /// Create a new small-buffer pool backed by a single large Metal buffer.
    ///
    /// `pool_size` is the total backing buffer size. If `None`, defaults to
    /// `DEFAULT_NUM_SLOTS * MAX_SMALL_ALLOC` (1 MiB with 4096 slots).
    pub fn new(device: &GpuDevice, pool_size: Option<usize>) -> Self {
        let slot_size = MAX_SMALL_ALLOC;
        let num_slots = pool_size
            .map(|s| std::cmp::max(1, s / slot_size))
            .unwrap_or(DEFAULT_NUM_SLOTS);
        let total_bytes = num_slots * slot_size;

        let backing_buffer = device.new_buffer(
            total_bytes as u64,
            rmlx_metal::device::DEFAULT_BUFFER_OPTIONS,
        );
        let free_list = (0..num_slots).rev().collect();

        Self {
            backing_buffer,
            slot_size,
            num_slots,
            inner: Mutex::new(PoolInner { free_list }),
        }
    }

    /// Try to allocate `size` bytes from the pool.
    ///
    /// Returns `None` if the pool is exhausted or `size > MAX_SMALL_ALLOC`.
    pub fn alloc(&self, size: usize) -> Option<SmallAllocation> {
        if size == 0 || size > MAX_SMALL_ALLOC {
            return None;
        }

        let mut inner = self.inner.lock().ok()?;
        let slot = inner.free_list.pop()?;

        Some(SmallAllocation {
            slot,
            offset: slot * self.slot_size,
            size,
            buffer: Some(self.backing_buffer.clone()),
        })
    }

    /// Return a small allocation to the pool.
    ///
    /// Returns `true` if the slot was successfully freed, `false` if the slot
    /// was already in the free list (double-free detection) or out of range.
    pub fn free(&self, alloc: SmallAllocation) -> bool {
        if alloc.slot >= self.num_slots {
            return false;
        }
        if let Ok(mut inner) = self.inner.lock() {
            // Guard: check if slot is already free (double-free detection).
            // All small allocations share one gpu_address (sub-allocated from
            // a single backing buffer), so the allocator's free() path could
            // pop an arbitrary SmallAllocation from the tracking vec. If that
            // allocation was already returned, we'd corrupt a live slot.
            if inner.free_list.contains(&alloc.slot) {
                return false;
            }
            inner.free_list.push(alloc.slot);
            true
        } else {
            false
        }
    }

    /// The single backing Metal buffer.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.backing_buffer
    }

    /// Slot size in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Total number of slots.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Number of currently free slots.
    pub fn free_count(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.free_list.len())
            .unwrap_or(0)
    }

    /// Number of currently allocated slots.
    pub fn allocated_count(&self) -> usize {
        self.num_slots - self.free_count()
    }

    /// Whether this pool performs true sub-allocation from a larger slab.
    ///
    /// This pool sub-allocates from a single backing buffer.
    pub fn is_suballocating(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_buffer_pool_basic() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, Some(256 * 4)); // 4 slots
        assert_eq!(pool.num_slots(), 4);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.allocated_count(), 0);
        assert!(pool.is_suballocating());

        // Allocate 3 slots
        let a1 = pool.alloc(8).expect("alloc 1");
        assert_eq!(a1.size, 8);
        assert!(a1.buffer.is_some());

        let a2 = pool.alloc(128).expect("alloc 2");
        let a2_offset = a2.offset;

        let a3 = pool.alloc(256).expect("alloc 3");

        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.allocated_count(), 3);

        // Free one and reallocate
        pool.free(a2);
        assert_eq!(pool.free_count(), 2);

        let a4 = pool.alloc(64).expect("alloc 4");
        // Should reuse the same slot (LIFO free list)
        assert_eq!(a4.offset, a2_offset);

        // All allocations share the same backing buffer
        assert_eq!(
            a1.buffer.as_ref().unwrap().gpu_address(),
            a4.buffer.as_ref().unwrap().gpu_address(),
        );

        pool.free(a1);
        pool.free(a3);
        pool.free(a4);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_small_buffer_pool_exhaustion() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, Some(256 * 2)); // 2 slots
        let a1 = pool.alloc(16).expect("alloc 1");
        let a2 = pool.alloc(16).expect("alloc 2");

        // Pool is now exhausted
        assert!(pool.alloc(16).is_none());
        assert_eq!(pool.free_count(), 0);

        pool.free(a1);
        pool.free(a2);
    }

    #[test]
    fn test_small_buffer_pool_rejects_large() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, None);
        assert!(pool.alloc(257).is_none());
        assert!(pool.alloc(1024).is_none());
        assert!(pool.alloc(0).is_none());
    }

    #[test]
    fn test_small_buffer_pool_double_free_detected() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, Some(256 * 4)); // 4 slots
        let a1 = pool.alloc(16).expect("alloc 1");
        let slot = a1.slot();
        assert_eq!(pool.free_count(), 3);

        // First free succeeds.
        assert!(pool.free(a1), "first free should succeed");
        assert_eq!(pool.free_count(), 4);

        // Simulate a double-free by constructing a SmallAllocation with the
        // same slot (normally impossible due to ownership, but tests the guard).
        let fake = SmallAllocation {
            slot,
            offset: slot * pool.slot_size(),
            size: 16,
            buffer: None,
        };
        assert!(!pool.free(fake), "double-free should be rejected");
        // Free count must not increase on double-free.
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_small_buffer_pool_default_size() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let pool = SmallBufferPool::new(&device, None);
        assert_eq!(pool.num_slots(), 4096);
        assert_eq!(pool.slot_size(), 256);
        assert!(pool.is_suballocating());
    }
}
