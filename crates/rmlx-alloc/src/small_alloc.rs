//! Small-buffer slab allocator (A5).
//!
//! MLX uses `MTL::Heap` for tiny allocations (<256 B). Creating individual
//! Metal buffers for each tiny request adds significant Metal runtime overhead.
//!
//! `SmallBufferPool` pre-allocates a single Metal buffer (default 1 MiB) and
//! sub-allocates fixed-size slots from it using a free-list bitmap. Requests
//! up to `MAX_SMALL_ALLOC` (256 bytes) are served from this pool.

use std::sync::Mutex;

use rmlx_metal::device::GpuDevice;
use rmlx_metal::metal::{Buffer as MetalBuffer, MTLResourceOptions};

/// Maximum allocation size served by the small-buffer pool.
pub const MAX_SMALL_ALLOC: usize = 256;

/// Default backing buffer size: 1 MiB.
const DEFAULT_POOL_SIZE: usize = 1024 * 1024;

/// A sub-allocation within the small-buffer pool.
#[derive(Debug)]
pub struct SmallAllocation {
    /// Index of the slot in the pool (for returning to the free list).
    slot: usize,
    /// Byte offset within the parent buffer.
    pub offset: usize,
    /// Requested size (may be smaller than slot size).
    pub size: usize,
}

impl SmallAllocation {
    /// The slot index used internally by the pool.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// Inner state protected by a mutex.
struct PoolInner {
    /// Bitmap: true = slot is free.
    free_bitmap: Vec<bool>,
    /// Number of free slots remaining.
    free_count: usize,
}

/// Slab allocator that sub-allocates from a single pre-allocated Metal buffer.
///
/// Each slot is `MAX_SMALL_ALLOC` bytes. The pool hands out `SmallAllocation`
/// descriptors referencing the parent buffer + offset.
pub struct SmallBufferPool {
    /// The backing Metal buffer (shared storage mode).
    buffer: MetalBuffer,
    /// Slot size in bytes (= `MAX_SMALL_ALLOC`).
    slot_size: usize,
    /// Total number of slots.
    num_slots: usize,
    /// Protected mutable state.
    inner: Mutex<PoolInner>,
}

impl SmallBufferPool {
    /// Create a new small-buffer pool backed by a single Metal buffer.
    ///
    /// `pool_size` is the total backing buffer size (default: 1 MiB).
    /// Each slot is `MAX_SMALL_ALLOC` bytes (256 B).
    pub fn new(device: &GpuDevice, pool_size: Option<usize>) -> Self {
        let total = pool_size.unwrap_or(DEFAULT_POOL_SIZE);
        let slot_size = MAX_SMALL_ALLOC;
        let num_slots = total / slot_size;

        let buffer = device.new_buffer(total as u64, MTLResourceOptions::StorageModeShared);
        let free_bitmap = vec![true; num_slots];

        Self {
            buffer,
            slot_size,
            num_slots,
            inner: Mutex::new(PoolInner {
                free_bitmap,
                free_count: num_slots,
            }),
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
        if inner.free_count == 0 {
            return None;
        }
        // Linear scan for first free slot. For 4096 slots (1 MiB / 256 B)
        // this is fast enough; a more sophisticated approach (e.g., next-free
        // index) can be added if profiling shows this as a bottleneck.
        for (i, free) in inner.free_bitmap.iter_mut().enumerate() {
            if *free {
                *free = false;
                inner.free_count -= 1;
                return Some(SmallAllocation {
                    slot: i,
                    offset: i * self.slot_size,
                    size,
                });
            }
        }
        None
    }

    /// Return a small allocation to the pool.
    pub fn free(&self, alloc: SmallAllocation) {
        if let Ok(mut inner) = self.inner.lock() {
            if alloc.slot < self.num_slots {
                inner.free_bitmap[alloc.slot] = true;
                inner.free_count += 1;
            }
        }
    }

    /// Reference to the backing Metal buffer.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
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
            .map(|inner| inner.free_count)
            .unwrap_or(0)
    }

    /// Number of currently allocated slots.
    pub fn allocated_count(&self) -> usize {
        self.num_slots - self.free_count()
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

        let a2 = pool.alloc(128).expect("alloc 2");
        assert_eq!(a2.offset, 256);

        let a3 = pool.alloc(256).expect("alloc 3");
        assert_eq!(a3.offset, 512);

        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.allocated_count(), 3);

        // Free one and reallocate
        pool.free(a2);
        assert_eq!(pool.free_count(), 2);

        let a4 = pool.alloc(64).expect("alloc 4");
        // Should reuse slot 1 (first free)
        assert_eq!(a4.offset, 256);

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
        let _a1 = pool.alloc(16).expect("alloc 1");
        let _a2 = pool.alloc(16).expect("alloc 2");

        // Pool is now exhausted
        assert!(pool.alloc(16).is_none());
        assert_eq!(pool.free_count(), 0);
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
