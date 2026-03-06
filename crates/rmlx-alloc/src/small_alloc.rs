//! Small-buffer recycling allocator (A5).
//!
//! Creating individual Metal buffers for each tiny request adds significant
//! Metal runtime overhead.
//!
//! `SmallBufferPool` pre-allocates a fixed set of 256-byte Metal buffers and
//! recycles them through a free-list. Requests up to `MAX_SMALL_ALLOC`
//! (256 bytes) are served from this pool.

use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal::Buffer as MetalBuffer;

/// Maximum allocation size served by the small-buffer pool.
pub const MAX_SMALL_ALLOC: usize = 256;

/// Default pool size: 256 buffers = 64 KiB total.
const DEFAULT_POOL_SIZE: usize = MAX_SMALL_ALLOC * 256;

/// A small allocation checked out from the recycling pool.
#[derive(Debug)]
pub struct SmallAllocation {
    /// Index of the slot in the pool (for returning to the free list).
    slot: usize,
    /// Logical byte offset within the pool.
    pub offset: usize,
    /// Requested size (may be smaller than slot size).
    pub size: usize,
    /// The pooled Metal buffer handed out for this allocation.
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
    /// Buffers currently held by the pool. `None` means the slot is checked out.
    buffers: Vec<Option<MetalBuffer>>,
    /// Free slot indices, stored as a stack for O(1) pop/push.
    free_list: Vec<usize>,
}

/// Recycling pool that hands out pre-allocated fixed-size Metal buffers.
pub struct SmallBufferPool {
    /// Representative buffer kept for backward compatibility with `buffer()`.
    sample_buffer: MetalBuffer,
    /// Slot size in bytes (= `MAX_SMALL_ALLOC`).
    slot_size: usize,
    /// Total number of slots.
    num_slots: usize,
    /// Protected mutable state.
    inner: Mutex<PoolInner>,
}

impl SmallBufferPool {
    /// Create a new small-buffer pool backed by fixed-size Metal buffers.
    ///
    /// `pool_size` is the total backing buffer size (default: 64 KiB).
    /// Each slot is `MAX_SMALL_ALLOC` bytes (256 B).
    pub fn new(device: &GpuDevice, pool_size: Option<usize>) -> Self {
        let total = pool_size.unwrap_or(DEFAULT_POOL_SIZE);
        let slot_size = MAX_SMALL_ALLOC;
        let num_slots = std::cmp::max(1, total / slot_size);

        let mut buffers = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            buffers.push(Some(device.new_buffer(
                slot_size as u64,
                rmlx_metal::device::DEFAULT_BUFFER_OPTIONS,
            )));
        }
        let sample_buffer = buffers[0]
            .as_ref()
            .expect("pool must have a buffer")
            .clone();
        let free_list = (0..num_slots).rev().collect();

        Self {
            sample_buffer,
            slot_size,
            num_slots,
            inner: Mutex::new(PoolInner { buffers, free_list }),
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
        let buffer = match inner.buffers.get_mut(slot).and_then(Option::take) {
            Some(buffer) => buffer,
            None => {
                inner.free_list.push(slot);
                return None;
            }
        };

        Some(SmallAllocation {
            slot,
            offset: slot * self.slot_size,
            size,
            buffer: Some(buffer),
        })
    }

    /// Return a small allocation to the pool.
    pub fn free(&self, alloc: SmallAllocation) {
        if let Ok(mut inner) = self.inner.lock() {
            if alloc.slot < self.num_slots {
                if let Some(buffer) = alloc.buffer {
                    if inner.buffers[alloc.slot].is_none() {
                        inner.buffers[alloc.slot] = Some(buffer);
                        inner.free_list.push(alloc.slot);
                    }
                }
            }
        }
    }

    /// Representative Metal buffer from the pool.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.sample_buffer
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
    /// This recycling pool returns whole buffers, so it never sub-allocates.
    pub fn is_suballocating(&self) -> bool {
        false
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

        // Allocate 3 slots
        let a1 = pool.alloc(8).expect("alloc 1");
        assert_eq!(a1.offset, 0);
        assert_eq!(a1.size, 8);
        assert!(a1.buffer.is_some());

        let a2 = pool.alloc(128).expect("alloc 2");
        assert_eq!(a2.offset, 256);
        let a2_addr = a2
            .buffer
            .as_ref()
            .expect("alloc 2 should carry a buffer")
            .gpu_address();

        let a3 = pool.alloc(256).expect("alloc 3");
        assert_eq!(a3.offset, 512);

        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.allocated_count(), 3);

        // Free one and reallocate
        pool.free(a2);
        assert_eq!(pool.free_count(), 2);

        let a4 = pool.alloc(64).expect("alloc 4");
        // Should reuse slot 1 (LIFO free list).
        assert_eq!(a4.offset, 256);
        assert_eq!(
            a4.buffer
                .as_ref()
                .expect("alloc 4 should carry a buffer")
                .gpu_address(),
            a2_addr
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
        // Requests > MAX_SMALL_ALLOC should be rejected
        assert!(pool.alloc(257).is_none());
        assert!(pool.alloc(1024).is_none());
        // Zero-size should be rejected
        assert!(pool.alloc(0).is_none());
    }
}
