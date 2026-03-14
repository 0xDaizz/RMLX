//! RDMA Queue Pair (UC mode) management for Thunderbolt 5
//!
//! TB5 only supports Unreliable Connected (UC) QPs.
//! This module handles QP creation, state transitions (RESET → INIT → RTR → RTS),
//! and post send/recv operations.

use std::ffi::c_int;
use std::ptr;

use crate::context::{ProtectionDomain, RdmaContext};
use crate::ffi::*;
use crate::RdmaError;

/// TB5-specific default constants (used as fallback when probe is unavailable).
/// CQ/WR depths are aligned with JACCL defaults for interoperability.
pub const IB_PORT: u8 = 1;
pub const DEFAULT_GID_INDEX: c_int = 1;
pub const DEFAULT_CQ_DEPTH: c_int = 64; // send(32) + recv(32), matches JACCL
pub const DEFAULT_MAX_SEND_WR: u32 = 32; // JACCL compatible
pub const DEFAULT_MAX_RECV_WR: u32 = 32; // JACCL compatible
pub const MAX_SEND_SGE: u32 = 1;
pub const MAX_RECV_SGE: u32 = 1;

/// Completion Queue wrapper.
///
/// Wraps an `ibv_cq` and provides safe poll semantics.
/// Destroys the CQ on drop.
pub struct CompletionQueue {
    cq: *mut IbvCq,
    lib: &'static IbverbsLib,
}

impl CompletionQueue {
    /// Create a new completion queue on the given RDMA context.
    ///
    /// Uses probed `max_cq_depth` from the device if available, otherwise
    /// falls back to `DEFAULT_CQ_DEPTH`.
    pub fn new(ctx: &RdmaContext) -> Result<Self, RdmaError> {
        let lib = ctx.lib();
        let cq_depth = ctx
            .probe()
            .map(|p| p.max_cq_depth as c_int)
            .unwrap_or_else(|| {
                tracing::warn!(
                    target: "rmlx_rdma",
                    default = DEFAULT_CQ_DEPTH,
                    "probe unavailable, using DEFAULT_CQ_DEPTH",
                );
                DEFAULT_CQ_DEPTH
            });
        // SAFETY: ctx.raw() is a valid ibv_context pointer obtained from ibv_open_device.
        // We pass null for the completion channel and user context since we poll manually.
        let cq =
            unsafe { (lib.create_cq)(ctx.raw(), cq_depth, ptr::null_mut(), ptr::null_mut(), 0) };
        if cq.is_null() {
            return Err(RdmaError::CqCreate);
        }
        Ok(Self { cq, lib })
    }

    /// Raw CQ pointer for use in QP creation.
    pub(crate) fn raw(&self) -> *mut IbvCq {
        self.cq
    }

    /// Poll for completions. Returns number of completions (0 = none ready).
    pub fn poll(&self, wc: &mut [IbvWc]) -> Result<usize, RdmaError> {
        // SAFETY: self.cq is a valid ibv_cq pointer, wc is a valid mutable slice.
        // SAFETY: self.cq is valid, calls through context->ops vtable (inline in C).
        let n = unsafe { ibv_poll_cq(self.cq, wc.len() as c_int, wc.as_mut_ptr()) };
        if n < 0 {
            return Err(RdmaError::CqPoll(format!("ibv_poll_cq returned {n}")));
        }
        Ok(n as usize)
    }
}

impl Drop for CompletionQueue {
    fn drop(&mut self) {
        // SAFETY: self.cq was obtained from ibv_create_cq and is valid.
        unsafe {
            (self.lib.destroy_cq)(self.cq);
        }
    }
}

// SAFETY: CQ handle is an opaque pointer safe to share across threads.
// ibv_poll_cq is thread-safe per libibverbs documentation.
unsafe impl Send for CompletionQueue {}
unsafe impl Sync for CompletionQueue {}

/// Queue Pair info needed for TCP exchange between peers.
#[derive(Clone)]
pub struct QpInfo {
    pub lid: u16,
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
}

/// Wire format size: lid(2) + qpn(4) + psn(4) + gid(16) = 26 bytes.
pub const QP_INFO_WIRE_SIZE: usize = 2 + 4 + 4 + 16;

impl QpInfo {
    /// Serialize QpInfo to a 26-byte little-endian wire format.
    pub fn to_wire(&self) -> [u8; QP_INFO_WIRE_SIZE] {
        let mut buf = [0u8; QP_INFO_WIRE_SIZE];
        buf[0..2].copy_from_slice(&self.lid.to_le_bytes());
        buf[2..6].copy_from_slice(&self.qpn.to_le_bytes());
        buf[6..10].copy_from_slice(&self.psn.to_le_bytes());
        buf[10..26].copy_from_slice(&self.gid);
        buf
    }

    /// Deserialize QpInfo from a 26-byte little-endian wire format.
    pub fn from_wire(buf: [u8; QP_INFO_WIRE_SIZE]) -> Self {
        let lid = u16::from_le_bytes([buf[0], buf[1]]);
        let qpn = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]);
        let psn = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&buf[10..26]);
        Self { lid, qpn, psn, gid }
    }
}

/// UC Queue Pair for TB5 RDMA.
///
/// Manages the full lifecycle: creation in RESET state, state transitions
/// (INIT → RTR → RTS), and post send/recv operations. Destroys the QP on drop.
pub struct QueuePair {
    qp: *mut IbvQp,
    lib: &'static IbverbsLib,
    local_info: QpInfo,
    gid_index: c_int,
    mtu: u32,
}

impl QueuePair {
    /// Create a UC Queue Pair in RESET state.
    ///
    /// Uses probed `max_qp_wr` from the device if available, otherwise
    /// falls back to `DEFAULT_MAX_SEND_WR` / `DEFAULT_MAX_RECV_WR`.
    pub fn create_uc(
        pd: &ProtectionDomain,
        cq: &CompletionQueue,
        ctx: &RdmaContext,
    ) -> Result<Self, RdmaError> {
        let lib = pd.lib();

        let max_send_wr = ctx.probe().map(|p| p.max_qp_wr).unwrap_or_else(|| {
            tracing::warn!(
                target: "rmlx_rdma",
                default = DEFAULT_MAX_SEND_WR,
                "probe unavailable, using DEFAULT_MAX_SEND_WR",
            );
            DEFAULT_MAX_SEND_WR
        });
        let max_recv_wr = ctx.probe().map(|p| p.max_qp_wr).unwrap_or_else(|| {
            tracing::warn!(
                target: "rmlx_rdma",
                default = DEFAULT_MAX_RECV_WR,
                "probe unavailable, using DEFAULT_MAX_RECV_WR",
            );
            DEFAULT_MAX_RECV_WR
        });

        // SAFETY: We zero-initialize the struct and fill in all required fields.
        let mut init_attr: IbvQpInitAttr = unsafe { std::mem::zeroed() };
        init_attr.send_cq = cq.raw();
        init_attr.recv_cq = cq.raw();
        init_attr.qp_type = qp_type::UC;
        // JACCL 호환 — 각 WR에서 개별적으로 SIGNALED 설정
        init_attr.sq_sig_all = 0;
        init_attr.cap.max_send_wr = max_send_wr;
        init_attr.cap.max_recv_wr = max_recv_wr;
        init_attr.cap.max_send_sge = MAX_SEND_SGE;
        init_attr.cap.max_recv_sge = MAX_RECV_SGE;

        // SAFETY: pd.raw() is a valid ibv_pd pointer, init_attr is fully initialized.
        let qp = unsafe { (lib.create_qp)(pd.raw(), &mut init_attr) };
        if qp.is_null() {
            return Err(RdmaError::QpCreate("ibv_create_qp failed".into()));
        }

        // SAFETY: qp is a valid ibv_qp pointer with accessible qp_num field.
        let qpn = unsafe { (*qp).qp_num };

        let gid_index = ctx
            .probe()
            .map(|p| p.gid_index as c_int)
            .unwrap_or_else(|| {
                tracing::warn!(
                    target: "rmlx_rdma",
                    default = DEFAULT_GID_INDEX,
                    "probe unavailable, using DEFAULT_GID_INDEX",
                );
                DEFAULT_GID_INDEX
            });
        // Always use MTU_1024 for TB5 compatibility (matches JACCL).
        // macOS reports active_mtu=4096 but TB5 UC mode is unreliable at larger MTUs.
        let mtu = mtu::MTU_1024;

        Ok(Self {
            qp,
            lib,
            local_info: QpInfo {
                lid: 0, // filled by query_local_info
                qpn,
                psn: 0,         // filled by query_local_info
                gid: [0u8; 16], // filled by query_local_info
            },
            gid_index,
            mtu,
        })
    }

    /// Query local port attributes and GID, populating local_info.
    ///
    /// Must be called before exchanging QP info with the remote peer.
    /// PSN is fixed to 7 for JACCL compatibility.
    /// Uses probed GID index if available, otherwise falls back to `DEFAULT_GID_INDEX`.
    pub fn query_local_info(&mut self, ctx: &RdmaContext, _rank: u32) -> Result<(), RdmaError> {
        let lib = self.lib;
        let gid_index = ctx
            .probe()
            .map(|p| p.gid_index as c_int)
            .unwrap_or_else(|| {
                tracing::warn!(
                    target: "rmlx_rdma",
                    default = DEFAULT_GID_INDEX,
                    "probe unavailable for query_local_info, using DEFAULT_GID_INDEX",
                );
                DEFAULT_GID_INDEX
            });

        // Query port for LID
        // SAFETY: ctx.raw() is valid, port_attr is zero-initialized and passed by mutable ref.
        let mut port_attr: IbvPortAttr = unsafe { std::mem::zeroed() };
        let ret = unsafe { (lib.query_port)(ctx.raw(), IB_PORT, &mut port_attr) };
        if ret != 0 {
            return Err(RdmaError::QpModify(format!("ibv_query_port failed: {ret}")));
        }

        // Query GID at probed index (RoCE on TB5 typically uses GID index 1, not 0)
        // SAFETY: ctx.raw() is valid, gid is zero-initialized.
        let mut gid: IbvGid = unsafe { std::mem::zeroed() };
        let ret = unsafe { (lib.query_gid)(ctx.raw(), IB_PORT, gid_index, &mut gid) };
        if ret != 0 {
            return Err(RdmaError::QpModify(format!("ibv_query_gid failed: {ret}")));
        }

        self.local_info.lid = port_attr.lid;
        // JACCL compatible: all ranks use PSN=7
        self.local_info.psn = 7;
        // SAFETY: IbvGid union's raw field is always valid to read.
        self.local_info.gid = unsafe { gid.raw };
        eprintln!("[qp] query_local_info: gid_index={gid_index}, lid={}, psn={}, qpn={}, gid={:02x?}", self.local_info.lid, self.local_info.psn, self.local_info.qpn, &self.local_info.gid);

        Ok(())
    }

    /// Get local QP info for TCP exchange with remote peer.
    pub fn local_info(&self) -> &QpInfo {
        &self.local_info
    }

    /// Connect this QP to a remote peer by transitioning through
    /// RESET → INIT → RTR → RTS.
    pub fn connect(&self, remote: &QpInfo) -> Result<(), RdmaError> {
        self.modify_to_init()?;
        eprintln!("[qp] connect: RESET→INIT OK");
        self.modify_to_rtr(remote)?;
        eprintln!(
            "[qp] connect: INIT→RTR OK (remote qpn={}, psn={}, gid={:02x?})",
            remote.qpn, remote.psn, &remote.gid[..4]
        );
        eprintln!("[qp] connect: calling modify_to_rts...");
        self.modify_to_rts()?;
        eprintln!(
            "[qp] connect: RTR→RTS OK (local psn={})",
            self.local_info.psn
        );
        Ok(())
    }

    /// Transition QP from RESET → INIT.
    fn modify_to_init(&self) -> Result<(), RdmaError> {
        // SAFETY: attr is zero-initialized, all required fields are set.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = qp_state::INIT;
        attr.pkey_index = 0;
        attr.port_num = IB_PORT;
        attr.qp_access_flags = (access_flags::LOCAL_WRITE
            | access_flags::REMOTE_WRITE
            | access_flags::REMOTE_READ) as u32;

        let mask = qp_attr_mask::STATE
            | qp_attr_mask::PKEY_INDEX
            | qp_attr_mask::PORT
            | qp_attr_mask::ACCESS_FLAGS;

        // SAFETY: self.qp is a valid ibv_qp pointer, attr is fully set up.
        let ret = unsafe { (self.lib.modify_qp)(self.qp, &mut attr, mask) };
        if ret != 0 {
            return Err(RdmaError::QpModify(format!("RESET→INIT failed: {ret}")));
        }
        Ok(())
    }

    /// Transition QP from INIT → RTR with remote peer info.
    fn modify_to_rtr(&self, remote: &QpInfo) -> Result<(), RdmaError> {
        // SAFETY: attr is zero-initialized, all required fields for RTR are set.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = qp_state::RTR;
        attr.path_mtu = self.mtu;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;

        // Address Handle
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = IB_PORT;

        // JACCL 호환: only set is_global when remote GID has non-zero interface_id
        // JACCL: if (dst.global_identifier.global.interface_id) { is_global=1; ... }
        let has_interface_id = remote.gid[8..16].iter().any(|&b| b != 0);
        if has_interface_id {
            attr.ah_attr.is_global = 1;
            // SAFETY: Constructing IbvGid union from raw bytes is safe.
            attr.ah_attr.grh.dgid = unsafe {
                let mut gid: IbvGid = std::mem::zeroed();
                gid.raw = remote.gid;
                gid
            };
            attr.ah_attr.grh.sgid_index = self.gid_index as u8;
            // JACCL 호환: hop_limit = 1
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.traffic_class = 0;
        }

        let mask = qp_attr_mask::STATE
            | qp_attr_mask::AV
            | qp_attr_mask::PATH_MTU
            | qp_attr_mask::DEST_QPN
            | qp_attr_mask::RQ_PSN;

        // SAFETY: self.qp is valid, attr is fully initialized for RTR transition.
        eprintln!("[qp] modify_to_rtr: sizeof(IbvQpAttr)={}, calling modify_qp (qp={:?}, mask=0x{:x})...",
            std::mem::size_of::<IbvQpAttr>(), self.qp, mask);
        eprintln!("[qp] modify_to_rtr: dest_qpn={}, rq_psn={}, path_mtu={}, is_global={}, gid_index={}",
            attr.dest_qp_num, attr.rq_psn, attr.path_mtu, attr.ah_attr.is_global, attr.ah_attr.grh.sgid_index);
        let ret = unsafe { (self.lib.modify_qp)(self.qp, &mut attr, mask) };
        eprintln!("[qp] modify_to_rtr: modify_qp returned {ret}");
        if ret != 0 {
            return Err(RdmaError::QpModify(format!("INIT→RTR failed: {ret}")));
        }
        Ok(())
    }

    /// Transition QP from RTR → RTS.
    fn modify_to_rts(&self) -> Result<(), RdmaError> {
        // SAFETY: attr is zero-initialized, RTS only needs state and sq_psn.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = qp_state::RTS;
        attr.sq_psn = self.local_info.psn;

        let mask = qp_attr_mask::STATE | qp_attr_mask::SQ_PSN;

        // SAFETY: self.qp is valid, attr is set for RTS transition.
        eprintln!("[qp] modify_to_rts: calling modify_qp (sq_psn={})...", self.local_info.psn);
        let ret = unsafe { (self.lib.modify_qp)(self.qp, &mut attr, mask) };
        eprintln!("[qp] modify_to_rts: modify_qp returned {ret}");
        if ret != 0 {
            return Err(RdmaError::QpModify(format!("RTR→RTS failed: {ret}")));
        }
        Ok(())
    }

    /// Post a send work request.
    pub fn post_send(&self, wr: &mut IbvSendWr) -> Result<(), RdmaError> {
        let mut bad_wr: *mut IbvSendWr = ptr::null_mut();
        // SAFETY: self.qp is valid, wr points to a valid send work request,
        // bad_wr receives the pointer to the first failed WR on error.
        // SAFETY: self.qp is valid, calls through context->ops vtable (inline in C).
        let ret = unsafe { ibv_post_send(self.qp, wr, &mut bad_wr) };
        if ret != 0 {
            return Err(RdmaError::PostFailed(format!(
                "ibv_post_send failed: {ret}"
            )));
        }
        Ok(())
    }

    /// Post a receive work request.
    pub fn post_recv(&self, wr: &mut IbvRecvWr) -> Result<(), RdmaError> {
        let mut bad_wr: *mut IbvRecvWr = ptr::null_mut();
        // SAFETY: self.qp is valid, wr points to a valid recv work request,
        // bad_wr receives the pointer to the first failed WR on error.
        // SAFETY: self.qp is valid, calls through context->ops vtable (inline in C).
        let ret = unsafe { ibv_post_recv(self.qp, wr, &mut bad_wr) };
        if ret != 0 {
            return Err(RdmaError::PostFailed(format!(
                "ibv_post_recv failed: {ret}"
            )));
        }
        Ok(())
    }

    /// Raw QP pointer (for advanced operations).
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> *mut IbvQp {
        self.qp
    }
}

impl Drop for QueuePair {
    fn drop(&mut self) {
        unsafe {
            // Transition QP to RESET to release hardware resources before destroy.
            // This ensures in-flight operations are aborted and the QP's hardware
            // state is cleaned up, preventing EBUSY on subsequent device opens.
            let mut attr: IbvQpAttr = std::mem::zeroed();
            attr.qp_state = qp_state::RESET;
            let _ = (self.lib.modify_qp)(self.qp, &mut attr, qp_attr_mask::STATE);

            let ret = (self.lib.destroy_qp)(self.qp);
            if ret != 0 {
                eprintln!("[qp] WARNING: ibv_destroy_qp returned {ret}");
            }
        }
    }
}

// SAFETY: QP handle is an opaque pointer safe to share across threads.
// ibverbs QP operations are thread-safe.
unsafe impl Send for QueuePair {}
