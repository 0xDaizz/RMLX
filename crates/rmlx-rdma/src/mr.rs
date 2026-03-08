//! RDMA memory region registration

use std::ffi::c_void;

use crate::context::ProtectionDomain;
use crate::ffi::{access_flags, IbvMr, IbverbsLib};
use crate::RdmaError;

/// Default TB5 maximum MR size (16 MB hardware limit, used when probe is unavailable).
pub const DEFAULT_MAX_MR_SIZE: usize = 16 * 1024 * 1024;

/// Get the system page size. Apple Silicon uses 16KB pages.
fn page_size() -> usize {
    // SAFETY: sysconf is always safe to call with _SC_PAGESIZE.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 {
        ps as usize
    } else {
        16384 // Apple Silicon default
    }
}

/// RDMA memory region — wraps ibv_mr.
/// Deregisters the MR on drop.
pub struct MemoryRegion {
    mr: *mut IbvMr,
    lib: &'static IbverbsLib,
}

impl MemoryRegion {
    /// Register a memory region for RDMA access.
    ///
    /// # Safety
    /// - `ptr` must be valid for `size` bytes
    /// - The memory must remain valid until this MR is dropped
    pub unsafe fn register(
        pd: &ProtectionDomain,
        ptr: *mut c_void,
        size: usize,
    ) -> Result<Self, RdmaError> {
        // SAFETY: caller guarantees ptr validity (see fn-level doc).
        unsafe { Self::register_with_limit(pd, ptr, size, DEFAULT_MAX_MR_SIZE) }
    }

    /// Register a memory region with a specific max_mr_size limit.
    ///
    /// # Safety
    /// - `ptr` must be valid for `size` bytes
    /// - The memory must remain valid until this MR is dropped
    pub unsafe fn register_with_limit(
        pd: &ProtectionDomain,
        ptr: *mut c_void,
        size: usize,
        max_mr_size: usize,
    ) -> Result<Self, RdmaError> {
        if size > max_mr_size {
            return Err(RdmaError::MrReg(format!(
                "size {size} exceeds max_mr_size ({max_mr_size})"
            )));
        }

        // Apple Silicon uses 16KB pages. TB5 RDMA requires page-aligned MR
        // registration. Round the pointer down and size up to page boundaries.
        let page_size = page_size();
        let ptr_addr = ptr as usize;
        let aligned_addr = ptr_addr & !(page_size - 1);
        let aligned_ptr = aligned_addr as *mut c_void;
        let aligned_size = (ptr_addr - aligned_addr + size + page_size - 1) & !(page_size - 1);

        let reg_size = aligned_size.min(max_mr_size);
        let flags =
            access_flags::LOCAL_WRITE | access_flags::REMOTE_WRITE | access_flags::REMOTE_READ;
        // SAFETY: pd.raw() is valid, aligned_ptr..aligned_ptr+reg_size covers
        // the original ptr..ptr+size range (caller guarantees ptr validity).
        let mr = unsafe { (pd.lib().reg_mr)(pd.raw(), aligned_ptr, reg_size, flags) };
        if mr.is_null() {
            return Err(RdmaError::MrReg(format!(
                "ibv_reg_mr failed for ptr={ptr:?}, size={size} \
                 (aligned to {aligned_ptr:?}, size={reg_size})"
            )));
        }
        Ok(Self { mr, lib: pd.lib() })
    }

    /// Local key for this memory region.
    pub fn lkey(&self) -> u32 {
        // SAFETY: mr is a valid ibv_mr pointer.
        unsafe { (*self.mr).lkey }
    }

    /// Remote key for this memory region.
    pub fn rkey(&self) -> u32 {
        // SAFETY: mr is a valid ibv_mr pointer.
        unsafe { (*self.mr).rkey }
    }

    /// Registered address.
    pub fn addr(&self) -> *mut c_void {
        // SAFETY: mr is a valid ibv_mr pointer.
        unsafe { (*self.mr).addr }
    }

    /// Registered length.
    pub fn length(&self) -> usize {
        // SAFETY: mr is a valid ibv_mr pointer.
        unsafe { (*self.mr).length }
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        // SAFETY: mr was obtained from ibv_reg_mr and is valid.
        unsafe {
            (self.lib.dereg_mr)(self.mr);
        }
    }
}

// SAFETY: MR handle is safe to share across threads.
unsafe impl Send for MemoryRegion {}
