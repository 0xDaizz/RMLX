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

/// TB5-specific default constants (used as fallback when probe is unavailable)
pub const IB_PORT: u8 = 1;
pub const DEFAULT_GID_INDEX: c_int = 1;
pub const DEFAULT_CQ_DEPTH: c_int = 8192;
pub const DEFAULT_MAX_SEND_WR: u32 = 8192;
pub const DEFAULT_MAX_RECV_WR: u32 = 8192;
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
                eprintln!(
                    "[rmlx-rdma] WARN: probe unavailable, using DEFAULT_CQ_DEPTH={DEFAULT_CQ_DEPTH}"
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
        let n = unsafe { (self.lib.poll_cq)(self.cq, wc.len() as c_int, wc.as_mut_ptr()) };
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
            eprintln!(
                "[rmlx-rdma] WARN: probe unavailable, using DEFAULT_MAX_SEND_WR={DEFAULT_MAX_SEND_WR}"
            );
            DEFAULT_MAX_SEND_WR
        });
        let max_recv_wr = ctx.probe().map(|p| p.max_qp_wr).unwrap_or_else(|| {
            eprintln!(
                "[rmlx-rdma] WARN: probe unavailable, using DEFAULT_MAX_RECV_WR={DEFAULT_MAX_RECV_WR}"
            );
            DEFAULT_MAX_RECV_WR
        });

        // SAFETY: We zero-initialize the struct and fill in all required fields.
        let mut init_attr: IbvQpInitAttr = unsafe { std::mem::zeroed() };
        init_attr.send_cq = cq.raw();
        init_attr.recv_cq = cq.raw();
        init_attr.qp_type = qp_type::UC;
        init_attr.sq_sig_all = 1;
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
                eprintln!(
                "[rmlx-rdma] WARN: probe unavailable, using DEFAULT_GID_INDEX={DEFAULT_GID_INDEX}"
            );
                DEFAULT_GID_INDEX
            });
        let mtu = ctx.probe().map(|p| p.mtu).unwrap_or_else(|| {
            eprintln!(
                "[rmlx-rdma] WARN: probe unavailable, using MTU_1024 ({})",
                mtu::MTU_1024
            );
            mtu::MTU_1024
        });

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
    /// PSN is computed as `rank * 1000 + 42` per TB5 convention.
    /// Uses probed GID index if available, otherwise falls back to `DEFAULT_GID_INDEX`.
    pub fn query_local_info(&mut self, ctx: &RdmaContext, rank: u32) -> Result<(), RdmaError> {
        let lib = self.lib;
        let gid_index = ctx.probe().map(|p| p.gid_index as c_int).unwrap_or_else(|| {
            eprintln!(
                "[rmlx-rdma] WARN: probe unavailable for query_local_info, using DEFAULT_GID_INDEX={DEFAULT_GID_INDEX}"
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
        self.local_info.psn = rank * 1000 + 42;
        // SAFETY: IbvGid union's raw field is always valid to read.
        self.local_info.gid = unsafe { gid.raw };

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
        self.modify_to_rtr(remote)?;
        self.modify_to_rts()?;
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

        // Address Handle with GRH (required for RoCE over TB5)
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = IB_PORT;

        // SAFETY: Constructing IbvGid union from raw bytes is safe.
        attr.ah_attr.grh.dgid = unsafe {
            let mut gid: IbvGid = std::mem::zeroed();
            gid.raw = remote.gid;
            gid
        };
        attr.ah_attr.grh.sgid_index = self.gid_index as u8;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.traffic_class = 0;

        let mask = qp_attr_mask::STATE
            | qp_attr_mask::AV
            | qp_attr_mask::PATH_MTU
            | qp_attr_mask::DEST_QPN
            | qp_attr_mask::RQ_PSN;

        // SAFETY: self.qp is valid, attr is fully initialized for RTR transition.
        let ret = unsafe { (self.lib.modify_qp)(self.qp, &mut attr, mask) };
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
        let ret = unsafe { (self.lib.modify_qp)(self.qp, &mut attr, mask) };
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
        let ret = unsafe { (self.lib.post_send)(self.qp, wr, &mut bad_wr) };
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
        let ret = unsafe { (self.lib.post_recv)(self.qp, wr, &mut bad_wr) };
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
        // SAFETY: self.qp was obtained from ibv_create_qp and is valid.
        unsafe {
            (self.lib.destroy_qp)(self.qp);
        }
    }
}

// SAFETY: QP handle is an opaque pointer safe to share across threads.
// ibverbs QP operations are thread-safe.
unsafe impl Send for QueuePair {}
