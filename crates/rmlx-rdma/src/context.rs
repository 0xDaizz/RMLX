//! RDMA device context and protection domain management

use std::ffi::CStr;

use crate::ffi::{IbvContext, IbvPd, IbverbsLib};
use crate::RdmaError;

/// RDMA device context — wraps ibv_context.
/// Owns the context and closes it on drop.
pub struct RdmaContext {
    ctx: *mut IbvContext,
    device_name: String,
    lib: &'static IbverbsLib,
}

impl RdmaContext {
    /// Open the first available RDMA device.
    pub fn open_default() -> Result<Self, RdmaError> {
        let lib = IbverbsLib::load()?;

        // SAFETY: ibv_get_device_list returns a null-terminated array.
        // We check for null and num_devices > 0.
        unsafe {
            let mut num_devices: std::ffi::c_int = 0;
            let dev_list = (lib.get_device_list)(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                if !dev_list.is_null() {
                    (lib.free_device_list)(dev_list);
                }
                return Err(RdmaError::NoDevices);
            }

            let device = *dev_list;
            let name_ptr = (lib.get_device_name)(device);
            let device_name = if name_ptr.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            };

            let ctx = (lib.open_device)(device);
            (lib.free_device_list)(dev_list);

            if ctx.is_null() {
                return Err(RdmaError::DeviceOpen(device_name));
            }

            Ok(Self {
                ctx,
                device_name,
                lib,
            })
        }
    }

    /// Device name (e.g., "mlx5_0" or TB5 device name).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Raw context pointer (for FFI calls).
    pub(crate) fn raw(&self) -> *mut IbvContext {
        self.ctx
    }

    /// Reference to the loaded ibverbs library.
    pub(crate) fn lib(&self) -> &'static IbverbsLib {
        self.lib
    }

    /// Allocate a protection domain on this context.
    pub fn alloc_pd(&self) -> Result<ProtectionDomain, RdmaError> {
        // SAFETY: ctx is a valid ibv_context pointer.
        let pd = unsafe { (self.lib.alloc_pd)(self.ctx) };
        if pd.is_null() {
            return Err(RdmaError::PdAlloc);
        }
        Ok(ProtectionDomain { pd, lib: self.lib })
    }
}

impl Drop for RdmaContext {
    fn drop(&mut self) {
        // SAFETY: ctx was obtained from ibv_open_device and is valid.
        unsafe {
            (self.lib.close_device)(self.ctx);
        }
    }
}

// SAFETY: RdmaContext holds a raw pointer to a thread-safe ibverbs context.
// ibverbs contexts are safe to share across threads.
unsafe impl Send for RdmaContext {}

/// Protection domain — wraps ibv_pd.
pub struct ProtectionDomain {
    pd: *mut IbvPd,
    lib: &'static IbverbsLib,
}

impl ProtectionDomain {
    /// Raw PD pointer (for FFI calls).
    pub(crate) fn raw(&self) -> *mut IbvPd {
        self.pd
    }

    /// Reference to the loaded ibverbs library.
    pub(crate) fn lib(&self) -> &'static IbverbsLib {
        self.lib
    }
}

impl Drop for ProtectionDomain {
    fn drop(&mut self) {
        // SAFETY: pd was obtained from ibv_alloc_pd and is valid.
        unsafe {
            (self.lib.dealloc_pd)(self.pd);
        }
    }
}

// SAFETY: PD is an opaque handle safe to share across threads.
unsafe impl Send for ProtectionDomain {}
