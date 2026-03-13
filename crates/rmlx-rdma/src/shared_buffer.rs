//! SharedBuffer — Triple-View Zero-Copy buffer for Apple Silicon UMA.
//!
//! A single page-aligned allocation exposed simultaneously as:
//! 1. CPU raw pointer (for memcpy, serialization)
//! 2. Metal GPU buffer (for compute kernels)
//! 3. RDMA memory region(s) (one per connection PD)
//!
//! All three views share the same physical memory on UMA, enabling
//! zero-copy data flow between GPU compute and RDMA transfer.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use rmlx_metal::{MTLResourceOptions, MtlBuffer};

use rmlx_alloc::zero_copy::CompletionTicket;
use rmlx_alloc::AllocError;

use crate::context::ProtectionDomain;
use crate::mr::MemoryRegion;
use crate::RdmaError;

/// Number of buffers per tier (double-buffering for pipeline overlap).
pub const PIPELINE: usize = 2;

/// Identifies a specific connection (reconnect-safe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId {
    pub node_id: u32,
    pub qp_num: u32,
    pub generation: u32,
}

/// A triple-view buffer: CPU + Metal + RDMA over the same physical memory.
pub struct SharedBuffer {
    raw_ptr: NonNull<u8>,
    size: usize,
    slot_index: u8,
    metal_buffer: MtlBuffer,
    rdma_registrations: HashMap<ConnectionId, MemoryRegion>,
    ticket: Option<CompletionTicket>,
}

// SAFETY: SharedBuffer owns its allocation. The raw pointer is heap-allocated
// via posix_memalign and solely owned. Metal buffer is StorageModeShared
// (CPU+GPU coherent on UMA). MemoryRegion is Send. HashMap is Send when
// key/value are Send.
unsafe impl Send for SharedBuffer {}

impl SharedBuffer {
    /// Allocate a new SharedBuffer with CPU + Metal views.
    ///
    /// The Metal device reference must remain valid for the lifetime of
    /// the Metal buffer created here. In practice, the device is held by
    /// the caller for the entire session.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
        slot_index: u8,
    ) -> Result<Self, AllocError> {
        let alignment: usize = 16384; // Apple Silicon page size
        let aligned_size = align_up(size, alignment);

        // Step 1: Page-aligned allocation
        // SAFETY: posix_memalign is called with valid alignment (power of 2, multiple of
        // sizeof(void*)) and returns a valid heap pointer on success.
        let raw_ptr = unsafe {
            let mut ptr: *mut libc::c_void = std::ptr::null_mut();
            let ret = libc::posix_memalign(&mut ptr, alignment, aligned_size);
            if ret != 0 {
                return Err(AllocError::PosixMemalign(ret));
            }
            std::ptr::write_bytes(ptr as *mut u8, 0, aligned_size);
            match NonNull::new(ptr as *mut u8) {
                Some(nn) => nn,
                None => return Err(AllocError::PosixMemalign(-1)),
            }
        };

        // Step 2: Metal NoCopy buffer wrapping the same memory
        // SAFETY: raw_ptr is valid for aligned_size bytes from posix_memalign above.
        // The Metal buffer does not take ownership of the memory (NoCopy with None deallocator).
        let metal_buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(raw_ptr.as_ptr() as *mut c_void).unwrap(),
                    aligned_size,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .unwrap()
        };

        Ok(Self {
            raw_ptr,
            size: aligned_size,
            slot_index,
            metal_buffer,
            rdma_registrations: HashMap::new(),
            ticket: None,
        })
    }

    /// Register this buffer's memory on a connection's protection domain.
    pub fn register_on(
        &mut self,
        conn_id: ConnectionId,
        pd: &ProtectionDomain,
    ) -> Result<(), RdmaError> {
        // SAFETY: raw_ptr is valid for self.size bytes and will outlive
        // the MemoryRegion (we deregister in deregister() or Drop).
        let mr =
            unsafe { MemoryRegion::register(pd, self.raw_ptr.as_ptr() as *mut c_void, self.size)? };
        self.rdma_registrations.insert(conn_id, mr);
        Ok(())
    }

    /// Deregister this buffer from a specific connection.
    pub fn deregister(&mut self, conn_id: &ConnectionId) {
        // MemoryRegion::drop handles ibv_dereg_mr
        self.rdma_registrations.remove(conn_id);
    }

    /// Get the RDMA memory region for a specific connection.
    pub fn rdma_mr(&self, conn_id: &ConnectionId) -> Option<&MemoryRegion> {
        self.rdma_registrations.get(conn_id)
    }

    /// The Metal GPU buffer view.
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        &self.metal_buffer
    }

    /// Raw pointer to the buffer memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.raw_ptr.as_ptr()
    }

    /// Buffer size in bytes (page-aligned).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Slot index within the tier.
    pub fn slot_index(&self) -> u8 {
        self.slot_index
    }

    /// Attach a completion ticket for GPU/RDMA lifecycle tracking.
    pub fn set_ticket(&mut self, ticket: CompletionTicket) {
        self.ticket = Some(ticket);
    }

    /// Returns true if the buffer is available for reuse.
    ///
    /// A buffer is available if no completion ticket is attached, or if the
    /// attached ticket indicates both GPU and RDMA operations have completed.
    pub fn is_available(&self) -> bool {
        match &self.ticket {
            Some(t) => t.is_safe_to_free(),
            None => true,
        }
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // Deregister all RDMA MRs (MemoryRegion::drop handles ibv_dereg_mr)
        self.rdma_registrations.clear();
        // Metal buffer drop is automatic (NoCopy buffer doesn't free memory)
        // Free the underlying allocation
        unsafe { libc::free(self.raw_ptr.as_ptr() as *mut libc::c_void) };
    }
}

/// A tier of identically-sized SharedBuffers.
pub struct SharedBufferTier {
    size: usize,
    buffers: Vec<SharedBuffer>,
}

/// Pool of SharedBuffers organized by size tier.
pub struct SharedBufferPool {
    tiers: Vec<SharedBufferTier>,
}

impl SharedBufferPool {
    /// Create a new pool with `PIPELINE` buffers per tier.
    ///
    /// The Metal device reference must remain valid for the lifetime of
    /// all Metal buffers created in this pool.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        tier_sizes: &[usize],
    ) -> Result<Self, AllocError> {
        let mut tiers = Vec::with_capacity(tier_sizes.len());
        for (tier_idx, &size) in tier_sizes.iter().enumerate() {
            let mut buffers = Vec::with_capacity(PIPELINE);
            for slot in 0..PIPELINE {
                let slot_index = (tier_idx * PIPELINE + slot) as u8;
                let buf = SharedBuffer::new(device, size, slot_index)?;
                buffers.push(buf);
            }
            tiers.push(SharedBufferTier { size, buffers });
        }
        Ok(Self { tiers })
    }

    /// Register all buffers on the given connections.
    pub fn register_all(
        &mut self,
        connections: &[(ConnectionId, &ProtectionDomain)],
    ) -> Result<(), RdmaError> {
        for tier in &mut self.tiers {
            for buf in &mut tier.buffers {
                for &(conn_id, pd) in connections {
                    buf.register_on(conn_id, pd)?;
                }
            }
        }
        Ok(())
    }

    /// Acquire an available buffer of at least `needed_bytes`.
    ///
    /// Finds the smallest tier that fits and returns a mutable reference
    /// to the first available buffer. Returns `None` if all are in use.
    pub fn acquire(&mut self, needed_bytes: usize) -> Option<&mut SharedBuffer> {
        for tier in &mut self.tiers {
            if tier.size >= needed_bytes {
                for buf in &mut tier.buffers {
                    if buf.is_available() {
                        return Some(buf);
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

    /// Total number of buffers across all tiers.
    pub fn total_buffers(&self) -> usize {
        self.tiers.iter().map(|t| t.buffers.len()).sum()
    }
}

/// Round up `n` to the next multiple of `align`. `align` must be a power of 2.
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}
