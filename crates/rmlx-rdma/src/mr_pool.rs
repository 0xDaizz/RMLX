//! Pre-registered MR pool for RDMA operations.
//!
//! Provides tiered, pre-allocated memory regions to avoid per-transfer
//! registration overhead. Slots are acquired via `MrHandle` RAII guards
//! that automatically release on drop.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::context::ProtectionDomain;
use crate::mr::MemoryRegion;
use crate::RdmaError;

/// Number of pipeline slots per tier (double-buffering).
pub const PIPELINE: usize = 2;

const TIER_SIZES: [usize; 6] = [
    4 * 1024,           // 4KB
    64 * 1024,          // 64KB
    256 * 1024,         // 256KB
    1 * 1024 * 1024,    // 1MB
    4 * 1024 * 1024,    // 4MB
    16 * 1024 * 1024,   // 16MB
];

/// Inner data for a pre-registered memory slot, shared via `Arc`.
///
/// Uses `ManuallyDrop<MemoryRegion>` so MrPool::Drop can control deregistration
/// order: MR deregistration must happen before freeing the buffer.
struct MrSlotInner {
    mr: std::mem::ManuallyDrop<MemoryRegion>,
    buffer: NonNull<u8>,
    size: usize,
    in_use: AtomicBool,
}

// SAFETY: MrSlotInner's buffer pointer is heap-allocated via posix_memalign and
// solely owned by this slot. MemoryRegion is Send (see mr.rs). AtomicBool is
// Send+Sync. NonNull<u8> is Send (the pointed-to allocation is owned by this slot).
unsafe impl Send for MrSlotInner {}
// SAFETY: All mutable state is behind AtomicBool (lock-free). MemoryRegion fields
// are only read (lkey, addr, length) which are immutable after registration.
// Buffer pointer is only read (for copying data in/out).
unsafe impl Sync for MrSlotInner {}

// Note: MrSlotInner does NOT implement Drop. The MR is ManuallyDrop'd
// and the buffer is free'd by MrPool::Drop in the correct order.

/// A tier of identically-sized MR slots.
struct MrTier {
    size: usize,
    slots: Vec<Arc<MrSlotInner>>,
}

/// Pool of pre-registered memory regions organized by size tier.
pub struct MrPool {
    tiers: Vec<MrTier>,
}

/// RAII handle to an acquired MR slot. Releases the slot on drop.
///
/// Not `Copy` or `Clone` to prevent double-release.
pub struct MrHandle {
    slot: Arc<MrSlotInner>,
}

impl MrHandle {
    /// Access the underlying MemoryRegion for RDMA operations.
    pub fn mr(&self) -> &MemoryRegion {
        &*self.slot.mr
    }

    /// lkey for SGE construction.
    pub fn lkey(&self) -> u32 {
        self.slot.mr.lkey()
    }

    /// Raw buffer pointer for data copy.
    pub fn buffer(&self) -> *mut u8 {
        self.slot.buffer.as_ptr()
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.slot.size
    }
}

impl Drop for MrHandle {
    fn drop(&mut self) {
        self.slot.in_use.store(false, Ordering::Release);
    }
}

// SAFETY: MrHandle holds an Arc<MrSlotInner> which is Send+Sync.
// Dropping an MrHandle only stores an atomic bool (Send-safe).
unsafe impl Send for MrHandle {}

impl MrPool {
    /// Create a new MR pool with all tiers pre-allocated and registered.
    pub fn new(pd: &ProtectionDomain) -> Result<Self, RdmaError> {
        let alignment: usize = 16384; // Apple Silicon page size
        let mut tiers = Vec::with_capacity(TIER_SIZES.len());

        for &tier_size in &TIER_SIZES {
            let mut slots = Vec::with_capacity(PIPELINE);
            for _ in 0..PIPELINE {
                // SAFETY: posix_memalign is called with valid alignment (power of 2, >= sizeof(void*))
                // and returns a valid heap pointer on success.
                let buffer = unsafe {
                    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
                    let ret = libc::posix_memalign(&mut ptr, alignment, tier_size);
                    if ret != 0 {
                        return Err(RdmaError::MrReg(format!(
                            "posix_memalign failed for tier_size={tier_size}: errno={ret}"
                        )));
                    }
                    // SAFETY: ptr is valid for tier_size bytes after successful posix_memalign.
                    std::ptr::write_bytes(ptr as *mut u8, 0, tier_size);
                    NonNull::new(ptr as *mut u8)
                        .ok_or_else(|| RdmaError::MrReg("posix_memalign returned null".into()))?
                };

                // SAFETY: buffer is valid for tier_size bytes and will remain valid
                // until MrSlotInner is dropped (we control the free order in MrPool::Drop).
                let mr = unsafe {
                    MemoryRegion::register(pd, buffer.as_ptr() as *mut c_void, tier_size)?
                };

                slots.push(Arc::new(MrSlotInner {
                    mr: std::mem::ManuallyDrop::new(mr),
                    buffer,
                    size: tier_size,
                    in_use: AtomicBool::new(false),
                }));
            }
            tiers.push(MrTier {
                size: tier_size,
                slots,
            });
        }

        Ok(Self { tiers })
    }

    /// Acquire an MR handle for at least `needed_bytes`.
    ///
    /// Finds the smallest tier that fits and returns an available slot.
    /// Returns `None` if all slots of the appropriate tier are in use.
    pub fn acquire(&self, needed_bytes: usize) -> Option<MrHandle> {
        for tier in &self.tiers {
            if tier.size >= needed_bytes {
                for slot in &tier.slots {
                    if slot
                        .in_use
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Some(MrHandle {
                            slot: Arc::clone(slot),
                        });
                    }
                }
            }
        }
        None
    }

    /// Number of tiers in the pool.
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Total number of slots across all tiers.
    pub fn total_slots(&self) -> usize {
        self.tiers.iter().map(|t| t.slots.len()).sum()
    }
}

impl Drop for MrPool {
    fn drop(&mut self) {
        // We must deregister MRs before freeing the buffers.
        // Take ownership of tiers so we control the Arc drop order.
        let tiers = std::mem::take(&mut self.tiers);
        for tier in tiers {
            for slot_arc in tier.slots {
                // Try to get exclusive ownership. If other MrHandles still exist,
                // the Arc won't unwrap — but that's a bug (pool outlives handles).
                match Arc::try_unwrap(slot_arc) {
                    Ok(mut inner) => {
                        let buf_ptr = inner.buffer;
                        // Drop the MR first (deregisters via ibv_dereg_mr)
                        // SAFETY: ManuallyDrop::drop is called exactly once here.
                        // The MR is still valid at this point.
                        unsafe { std::mem::ManuallyDrop::drop(&mut inner.mr) };
                        // Now free the buffer
                        // SAFETY: buf_ptr was allocated by posix_memalign and the MR
                        // has been deregistered.
                        unsafe { libc::free(buf_ptr.as_ptr() as *mut libc::c_void) };
                    }
                    Err(arc) => {
                        // Outstanding handles exist — this is a usage bug.
                        // Drop the Arc anyway (MR won't deregister until last handle drops).
                        eprintln!(
                            "[rmlx-rdma] WARN: MrPool dropped with {} outstanding handles for slot size={}",
                            Arc::strong_count(&arc) - 1,
                            arc.size
                        );
                        // The buffer will leak. This is intentional to avoid UB
                        // (freeing memory that an outstanding MrHandle may still reference).
                    }
                }
            }
        }
    }
}
