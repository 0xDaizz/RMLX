//! RDMA device context and protection domain management

use std::ffi::{c_int, CStr};

use crate::ffi::{IbvContext, IbvDeviceAttr, IbvPd, IbvPortAttr, IbverbsLib};
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
    ///
    /// If `RMLX_RDMA_DEVICE` is set, opens that device by name instead.
    pub fn open_default() -> Result<Self, RdmaError> {
        if let Ok(name) = std::env::var("RMLX_RDMA_DEVICE") {
            return Self::open_by_name(&name);
        }

        let lib = IbverbsLib::load()?;

        // SAFETY: ibv_get_device_list returns a null-terminated array.
        // We check for null and num_devices > 0.
        unsafe {
            let mut num_devices: c_int = 0;
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
                    tracing::info!(
                        target: "rmlx_rdma",
                        device = %rdma_ctx.device_name,
                        gid_index = p.gid_index,
                        max_mr_size = p.max_mr_size,
                        max_qp_wr = p.max_qp_wr,
                        max_cq_depth = p.max_cq_depth,
                        mtu = p.mtu,
                        max_msg_sz = p.max_msg_sz,
                        gid_tbl_len = p.gid_tbl_len,
                        "device probed",
                    );
                    // Log INFO when probed values differ significantly from defaults
                    if p.gid_index != 1 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            gid_index = p.gid_index,
                            "probed gid_index differs from default (1)",
                        );
                    }
                    if p.max_qp_wr != 4095 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            max_qp_wr = p.max_qp_wr,
                            "probed max_qp_wr differs from default (4095)",
                        );
                    }
                    if p.max_cq_depth != 8192 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            max_cq_depth = p.max_cq_depth,
                            "probed max_cq_depth differs from default (8192)",
                        );
                    }
                    rdma_ctx.probe = Some(p);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "rmlx_rdma",
                        device = %rdma_ctx.device_name,
                        %e,
                        "device probe failed, falling back to defaults",
                    );
                }
            }

            Ok(rdma_ctx)
        }
    }

    /// Open a specific RDMA device by name (e.g., "mlx5_0").
    ///
    /// Uses `ibv_devices` CLI to discover the device index, then opens it
    /// directly — avoiding `ibv_get_device_name()` which can hang on
    /// PORT_DOWN devices on macOS.
    pub fn open_by_name(name: &str) -> Result<Self, RdmaError> {
        // Step 1: Find device index via `ibv_devices` CLI output (safe, no hang)
        let index = find_device_index_by_cli(name)?;

        // Step 2: Open the device directly by index (no iteration needed)
        let lib = IbverbsLib::load()?;

        // SAFETY: ibv_get_device_list returns a null-terminated array.
        // We access by the index discovered from CLI, skipping ibv_get_device_name.
        unsafe {
            let mut num_devices: c_int = 0;
            let dev_list = (lib.get_device_list)(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                if !dev_list.is_null() {
                    (lib.free_device_list)(dev_list);
                }
                return Err(RdmaError::NoDevices);
            }

            if index >= num_devices as usize {
                (lib.free_device_list)(dev_list);
                return Err(RdmaError::DeviceOpen(format!(
                    "device '{name}' index {index} out of range ({num_devices} devices)"
                )));
            }

            let device = *dev_list.offset(index as isize);
            let device_name = name.to_string();
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
                    tracing::info!(
                        target: "rmlx_rdma",
                        device = %rdma_ctx.device_name,
                        gid_index = p.gid_index,
                        max_mr_size = p.max_mr_size,
                        max_qp_wr = p.max_qp_wr,
                        max_cq_depth = p.max_cq_depth,
                        mtu = p.mtu,
                        max_msg_sz = p.max_msg_sz,
                        gid_tbl_len = p.gid_tbl_len,
                        "device probed",
                    );
                    // Log INFO when probed values differ significantly from defaults
                    if p.gid_index != 1 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            gid_index = p.gid_index,
                            "probed gid_index differs from default (1)",
                        );
                    }
                    if p.max_qp_wr != 4095 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            max_qp_wr = p.max_qp_wr,
                            "probed max_qp_wr differs from default (4095)",
                        );
                    }
                    if p.max_cq_depth != 8192 {
                        tracing::info!(
                            target: "rmlx_rdma",
                            max_cq_depth = p.max_cq_depth,
                            "probed max_cq_depth differs from default (8192)",
                        );
                    }
                    rdma_ctx.probe = Some(p);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "rmlx_rdma",
                        device = %rdma_ctx.device_name,
                        %e,
                        "device probe failed, falling back to defaults",
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
    ///
    /// Queries both `ibv_query_device` (for device-level caps like max_mr_size,
    /// max_qp_wr, max_cqe) and `ibv_query_port` (for port-level caps like MTU,
    /// max_msg_sz, GID table). Falls back to hardcoded TB5 defaults on failure.
    pub fn probe_port(ctx: &RdmaContext, port: u8) -> Result<Self, RdmaError> {
        let lib = ctx.lib();

        // --- Device-level attributes (ibv_query_device) ---
        // SAFETY: ctx.raw() is a valid ibv_context pointer, dev_attr is zero-initialized.
        let mut dev_attr: IbvDeviceAttr = unsafe { std::mem::zeroed() };
        let dev_query_ok = unsafe { (lib.query_device)(ctx.raw(), &mut dev_attr) } == 0;

        let (max_mr_size, max_qp_wr, max_cq_depth) = if dev_query_ok {
            let probed_mr = dev_attr.max_mr_size as usize;
            let probed_qp_wr = dev_attr.max_qp_wr as u32;
            let probed_cqe = dev_attr.max_cqe as u32;
            tracing::info!(
                target: "rmlx_rdma",
                max_mr_size = probed_mr,
                max_qp_wr = probed_qp_wr,
                max_cqe = probed_cqe,
                "using probed device values",
            );
            (probed_mr, probed_qp_wr, probed_cqe)
        } else {
            const FALLBACK_MR: usize = 16 * 1024 * 1024;
            const FALLBACK_QP_WR: u32 = 4095;
            const FALLBACK_CQ: u32 = 8192;
            tracing::warn!(
                target: "rmlx_rdma",
                max_mr_size = FALLBACK_MR,
                max_qp_wr = FALLBACK_QP_WR,
                max_cq_depth = FALLBACK_CQ,
                "ibv_query_device failed, using hardcoded fallback",
            );
            (FALLBACK_MR, FALLBACK_QP_WR, FALLBACK_CQ)
        };

        // --- Port-level attributes (ibv_query_port) ---
        // SAFETY: ctx.raw() is a valid ibv_context pointer, port_attr is zero-initialized.
        let mut port_attr: IbvPortAttr = unsafe { std::mem::zeroed() };
        let ret = unsafe { (lib.query_port)(ctx.raw(), port, &mut port_attr) };
        if ret != 0 {
            return Err(RdmaError::Unavailable(format!(
                "ibv_query_port failed for port {port}: {ret}"
            )));
        }

        tracing::info!(
            target: "rmlx_rdma",
            active_mtu = port_attr.active_mtu,
            max_msg_sz = port_attr.max_msg_sz,
            gid_tbl_len = port_attr.gid_tbl_len,
            "using probed port values",
        );

        // Determine GID index: probe from index 1 downward.
        // TB5 RoCE uses GID index 1; if that fails, fall back to 0.
        let gid_index = Self::probe_gid_index(ctx, port, port_attr.gid_tbl_len as u32)?;

        Ok(Self {
            gid_index,
            max_mr_size,
            max_qp_wr,
            max_cq_depth,
            mtu: port_attr.active_mtu,
            max_msg_sz: port_attr.max_msg_sz,
            gid_tbl_len: port_attr.gid_tbl_len as u32,
        })
    }

    /// Find a valid GID index by probing.
    ///
    /// Prefers a non-link-local IPv4-mapped GID (e.g. `10.254.0.x` set by
    /// `auto_setup`) over a link-local one (`169.254.x.x`).  Falls back to
    /// index 1 → 0 if no such GID exists.
    fn probe_gid_index(ctx: &RdmaContext, port: u8, gid_tbl_len: u32) -> Result<u32, RdmaError> {
        use std::os::raw::c_int;
        let lib = ctx.lib();

        // Phase 1: Find a non-link-local IPv4-mapped GID (preferred for RDMA over TB5)
        for idx in 0..gid_tbl_len {
            let mut gid: crate::ffi::IbvGid = unsafe { std::mem::zeroed() };
            let ret = unsafe { (lib.query_gid)(ctx.raw(), port, idx as c_int, &mut gid) };
            if ret != 0 {
                continue;
            }
            let raw = unsafe { gid.raw };
            // Check IPv4-mapped format: ::ffff:x.x.x.x
            let is_ipv4_mapped =
                raw[..10] == [0u8; 10] && raw[10] == 0xff && raw[11] == 0xff;
            if !is_ipv4_mapped {
                continue;
            }
            // Skip link-local (169.254.x.x)
            if raw[12] == 0xa9 && raw[13] == 0xfe {
                continue;
            }
            // Found a non-link-local IPv4 GID
            eprintln!(
                "[rdma] probe_gid_index: selected index {idx} (IP={}.{}.{}.{})",
                raw[12], raw[13], raw[14], raw[15]
            );
            return Ok(idx);
        }

        // Phase 2: Fall back to any non-zero GID (try index 1 first for TB5 convention)
        if gid_tbl_len > 1 {
            let mut gid: crate::ffi::IbvGid = unsafe { std::mem::zeroed() };
            let ret = unsafe { (lib.query_gid)(ctx.raw(), port, 1, &mut gid) };
            if ret == 0 {
                let raw = unsafe { gid.raw };
                if raw.iter().any(|&b| b != 0) {
                    eprintln!("[rdma] probe_gid_index: fallback to index 1");
                    return Ok(1);
                }
            }
        }

        // Phase 3: Fall back to index 0
        if gid_tbl_len > 0 {
            let mut gid: crate::ffi::IbvGid = unsafe { std::mem::zeroed() };
            let ret = unsafe { (lib.query_gid)(ctx.raw(), port, 0, &mut gid) };
            if ret == 0 {
                eprintln!("[rdma] probe_gid_index: fallback to index 0");
                return Ok(0);
            }
        }

        Err(RdmaError::Unavailable(
            "no valid GID index found".to_string(),
        ))
    }
}

/// Find the index of a device in the `ibv_devices` CLI output.
///
/// This avoids calling `ibv_get_device_name()` via the C API, which can hang
/// on PORT_DOWN devices on macOS. The CLI lists devices in the same order as
/// `ibv_get_device_list()`, so the positional index is stable.
fn find_device_index_by_cli(name: &str) -> Result<usize, RdmaError> {
    let output = std::process::Command::new("ibv_devices")
        .output()
        .map_err(|e| RdmaError::DeviceOpen(format!("ibv_devices failed: {e}")))?;

    if !output.status.success() {
        return Err(RdmaError::DeviceOpen(
            "ibv_devices returned non-zero".into(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse output like:
    //     device              node GUID
    //     ------          ----------------
    //     rdma_en2        b003616e7ef2ac05
    //     rdma_en3        b103616e7ef2ac05
    //     rdma_en5        b303616e7ef2ac05

    let mut index = 0usize;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("device")
            || trimmed.starts_with("------")
            || trimmed.is_empty()
        {
            continue;
        }
        let dev_name = trimmed.split_whitespace().next().unwrap_or("");
        if dev_name == name {
            return Ok(index);
        }
        index += 1;
    }

    Err(RdmaError::DeviceOpen(format!(
        "device '{name}' not found in ibv_devices output"
    )))
}
