//! RDMA memory region registration

use std::ffi::c_void;

use crate::context::ProtectionDomain;
use crate::ffi::{access_flags, IbvMr, IbverbsLib};
use crate::RdmaError;

/// Default TB5 maximum MR size (16 MB hardware limit, used when probe is unavailable).
pub const DEFAULT_MAX_MR_SIZE: usize = 16 * 1024 * 1024;

/// RDMA memory region — wraps ibv_mr.
/// Deregisters the MR on drop and frees the page-aligned buffer (if owned).
/// For nocopy MRs, the caller owns the buffer and Drop only deregisters.
pub struct MemoryRegion {
    mr: *mut IbvMr,
    lib: &'static IbverbsLib,
    /// Page-aligned buffer allocated via posix_memalign.
    /// Data is copied here for MR registration.
    /// Freed on drop.
    aligned_buf: *mut c_void,
    aligned_size: usize,
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

        // JACCL compatible: allocate page-aligned buffer and copy data.
        // macOS librdma requires page-aligned pointers for ibv_reg_mr.
        let page_size = page_size();
        let aligned_size = (size + page_size - 1) & !(page_size - 1);
        let aligned_size = aligned_size.max(page_size); // at least one page

        let mut aligned_buf: *mut c_void = std::ptr::null_mut();
        // SAFETY: posix_memalign is called with valid alignment (power of 2)
        // and returns a valid heap pointer on success.
        let ret = unsafe { libc::posix_memalign(&mut aligned_buf, page_size, aligned_size) };
        if ret != 0 || aligned_buf.is_null() {
            return Err(RdmaError::MrReg(format!(
                "posix_memalign failed: ret={ret}, size={aligned_size}"
            )));
        }

        // SAFETY: aligned_buf is valid for aligned_size bytes from posix_memalign.
        // ptr is valid for size bytes (caller guarantee).
        unsafe {
            // Copy data to aligned buffer
            std::ptr::copy_nonoverlapping(ptr as *const u8, aligned_buf as *mut u8, size);
            // Zero remaining bytes
            if aligned_size > size {
                std::ptr::write_bytes((aligned_buf as *mut u8).add(size), 0, aligned_size - size);
            }
        }

        let flags =
            access_flags::LOCAL_WRITE | access_flags::REMOTE_WRITE | access_flags::REMOTE_READ;
        // SAFETY: pd.raw() is valid, aligned_buf is valid for aligned_size bytes.
        let mr = unsafe { (pd.lib().reg_mr)(pd.raw(), aligned_buf, aligned_size, flags) };
        if mr.is_null() {
            // SAFETY: aligned_buf was allocated by posix_memalign.
            unsafe { libc::free(aligned_buf) };
            return Err(RdmaError::MrReg(format!(
                "ibv_reg_mr failed for aligned_buf={aligned_buf:?}, size={aligned_size}"
            )));
        }
        Ok(Self {
            mr,
            lib: pd.lib(),
            aligned_buf,
            aligned_size,
        })
    }

    /// Register an externally-owned buffer directly (no copy, no allocation).
    ///
    /// Designed for UMA architectures (Apple Silicon) where Metal `SharedBuffer`
    /// pointers can be registered directly with RDMA.  The caller retains
    /// ownership of the buffer and must keep it alive until this MR is dropped.
    ///
    /// # Safety
    /// - `ptr` must be page-aligned and valid for `size` bytes
    /// - The memory must remain valid until this MR is dropped
    pub unsafe fn register_nocopy(
        pd: &ProtectionDomain,
        ptr: *mut c_void,
        size: usize,
    ) -> Result<Self, RdmaError> {
        // SAFETY: caller guarantees ptr validity and alignment (see fn-level doc).
        unsafe { Self::register_nocopy_with_limit(pd, ptr, size, DEFAULT_MAX_MR_SIZE) }
    }

    /// Register an externally-owned buffer directly with a specific max_mr_size limit.
    ///
    /// # Safety
    /// - `ptr` must be page-aligned and valid for `size` bytes
    /// - The memory must remain valid until this MR is dropped
    pub unsafe fn register_nocopy_with_limit(
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

        if ptr.is_null() {
            return Err(RdmaError::MrReg("ptr is null".into()));
        }

        // Verify page alignment of the caller-provided pointer.
        let ps = page_size();
        if (ptr as usize) % ps != 0 {
            return Err(RdmaError::MrReg(format!(
                "ptr {ptr:?} is not page-aligned (page_size={ps})"
            )));
        }

        let flags =
            access_flags::LOCAL_WRITE | access_flags::REMOTE_WRITE | access_flags::REMOTE_READ;
        // SAFETY: pd.raw() is valid, ptr is page-aligned and valid for size bytes
        // (caller guarantee).
        let mr = unsafe { (pd.lib().reg_mr)(pd.raw(), ptr, size, flags) };
        if mr.is_null() {
            return Err(RdmaError::MrReg(format!(
                "ibv_reg_mr failed for nocopy ptr={ptr:?}, size={size}"
            )));
        }
        Ok(Self {
            mr,
            lib: pd.lib(),
            // null signals "not owned" — Drop will skip libc::free.
            aligned_buf: std::ptr::null_mut(),
            aligned_size: size,
        })
    }

    /// Returns `true` if this MR was created via `register_nocopy` (caller owns the buffer).
    pub fn is_nocopy(&self) -> bool {
        self.aligned_buf.is_null()
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

    /// Registered address (page-aligned buffer used for ibv_reg_mr).
    /// For nocopy MRs, returns the address stored in the ibv_mr struct.
    pub fn addr(&self) -> *mut c_void {
        if self.aligned_buf.is_null() {
            // nocopy: read from ibv_mr
            // SAFETY: mr is a valid ibv_mr pointer.
            unsafe { (*self.mr).addr }
        } else {
            self.aligned_buf
        }
    }

    /// Registered length (page-aligned size).
    pub fn length(&self) -> usize {
        self.aligned_size
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        unsafe {
            // Retry MR deregistration up to 3 times — TB5 macOS driver can transiently
            // fail dereg when kernel async cleanup hasn't finished yet.
            let mut ret = (self.lib.dereg_mr)(self.mr);
            if ret != 0 {
                for attempt in 1..=3 {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    ret = (self.lib.dereg_mr)(self.mr);
                    if ret == 0 {
                        eprintln!("[mr] ibv_dereg_mr succeeded on retry {attempt}");
                        break;
                    }
                }
            }
            if ret != 0 {
                // MR deregister failed after retries — leak the buffer to avoid
                // kernel state corruption. IOConnectUnmapMemory failure + free
                // would leave a dangling IOMMU mapping.
                eprintln!(
                    "[mr] WARNING: ibv_dereg_mr returned {ret} after retries, leaking aligned_buf"
                );
                return;
            }
            // nocopy MRs have null aligned_buf — caller owns the memory, skip free.
            if !self.aligned_buf.is_null() {
                // SAFETY: aligned_buf was allocated by posix_memalign in register_with_limit.
                libc::free(self.aligned_buf);
            }
        }
    }
}

// SAFETY: MR handle is safe to share across threads.
unsafe impl Send for MemoryRegion {}

fn page_size() -> usize {
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 {
        ps as usize
    } else {
        16384
    }
}
