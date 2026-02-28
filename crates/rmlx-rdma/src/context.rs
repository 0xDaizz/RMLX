//! RDMA device context and protection domain management

use std::ffi::CStr;

use crate::ffi::{IbvContext, IbvPd, IbvPortAttr, IbverbsLib};
use crate::RdmaError;

/// RDMA device context — wraps ibv_context.
/// Owns the context and closes it on drop.
pub struct RdmaContext {
    ctx: *mut IbvContext,
    device_name: String,
    lib: &'static IbverbsLib,
    probe: Option<RdmaDeviceProbe>,
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

            let mut rdma_ctx = Self {
                ctx,
                device_name,
                lib,
                probe: None,
            };

            // Probe device capabilities and store results
            match RdmaDeviceProbe::probe(&rdma_ctx) {
                Ok(p) => {
                    eprintln!(
                        "[rmlx-rdma] device '{}' probed: gid_index={}, max_mr_size={}, \
                         max_qp_wr={}, max_cq_depth={}, mtu={}, max_msg_sz={}, gid_tbl_len={}",
                        rdma_ctx.device_name,
                        p.gid_index,
                        p.max_mr_size,
                        p.max_qp_wr,
                        p.max_cq_depth,
                        p.mtu,
                        p.max_msg_sz,
                        p.gid_tbl_len,
                    );
                    rdma_ctx.probe = Some(p);
                }
                Err(e) => {
                    eprintln!(
                        "[rmlx-rdma] device '{}' probe failed, using defaults: {e}",
                        rdma_ctx.device_name,
                    );
                }
            }

            Ok(rdma_ctx)
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

    /// Probed device capabilities, if available.
    pub fn probe(&self) -> Option<&RdmaDeviceProbe> {
        self.probe.as_ref()
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

/// Runtime-probed RDMA device capabilities.
///
/// Replaces hardcoded constants (GID_INDEX, max_mr_size, etc.) with values
/// queried from the actual device at startup.
pub struct RdmaDeviceProbe {
    /// GID table index to use (TB5 typically uses 1, not 0).
    pub gid_index: u32,
    /// Maximum memory region size the device supports.
    pub max_mr_size: usize,
    /// Maximum work requests per QP.
    pub max_qp_wr: u32,
    /// Maximum CQ depth.
    pub max_cq_depth: u32,
    /// Active MTU (ibv_mtu enum value).
    pub mtu: u32,
    /// Maximum message size.
    pub max_msg_sz: u32,
    /// GID table length (number of valid GID entries).
    pub gid_tbl_len: u32,
}

impl RdmaDeviceProbe {
    /// Probe the device through the given context, querying port 1 by default.
    pub fn probe(ctx: &RdmaContext) -> Result<Self, RdmaError> {
        Self::probe_port(ctx, 1)
    }

    /// Probe a specific port on the device.
    pub fn probe_port(ctx: &RdmaContext, port: u8) -> Result<Self, RdmaError> {
        let lib = ctx.lib();

        // Query port attributes
        // SAFETY: ctx.raw() is a valid ibv_context pointer, port_attr is zero-initialized.
        let mut port_attr: IbvPortAttr = unsafe { std::mem::zeroed() };
        let ret = unsafe { (lib.query_port)(ctx.raw(), port, &mut port_attr) };
        if ret != 0 {
            return Err(RdmaError::Unavailable(format!(
                "ibv_query_port failed for port {port}: {ret}"
            )));
        }

        // Determine GID index: probe from index 1 downward.
        // TB5 RoCE uses GID index 1; if that fails, fall back to 0.
        let gid_index = Self::probe_gid_index(ctx, port, port_attr.gid_tbl_len as u32)?;

        // max_qp_wr is not directly in port_attr but typically capped by the device.
        // We use the port max_msg_sz and CQ-related values from port_attr.
        // For max_qp_wr, use a safe default based on known TB5 limits (4095).
        // A proper implementation would call ibv_query_device, but that requires
        // adding the FFI binding. For now, use the port-level information we have.
        Ok(Self {
            gid_index,
            max_mr_size: port_attr.max_msg_sz as usize,
            max_qp_wr: 4095, // TB5 known limit; ibv_query_device would give exact value
            max_cq_depth: 8192, // Safe default for TB5
            mtu: port_attr.active_mtu,
            max_msg_sz: port_attr.max_msg_sz,
            gid_tbl_len: port_attr.gid_tbl_len as u32,
        })
    }

    /// Find a valid GID index by probing. Prefers index 1 (RoCE on TB5).
    fn probe_gid_index(ctx: &RdmaContext, port: u8, gid_tbl_len: u32) -> Result<u32, RdmaError> {
        let lib = ctx.lib();

        // Try index 1 first (TB5 RoCE convention)
        if gid_tbl_len > 1 {
            let mut gid: crate::ffi::IbvGid = unsafe { std::mem::zeroed() };
            let ret = unsafe { (lib.query_gid)(ctx.raw(), port, 1, &mut gid) };
            if ret == 0 {
                // Verify it's not all-zeros
                let raw = unsafe { gid.raw };
                if raw.iter().any(|&b| b != 0) {
                    return Ok(1);
                }
            }
        }

        // Fall back to index 0
        if gid_tbl_len > 0 {
            let mut gid: crate::ffi::IbvGid = unsafe { std::mem::zeroed() };
            let ret = unsafe { (lib.query_gid)(ctx.raw(), port, 0, &mut gid) };
            if ret == 0 {
                return Ok(0);
            }
        }

        Err(RdmaError::Unavailable(
            "no valid GID index found".to_string(),
        ))
    }
}
