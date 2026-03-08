//! ibverbs FFI bindings via dynamic loading (dlopen)
//!
//! Since macOS TB5 RDMA drivers provide librdma.dylib rather than standard
//! libibverbs headers, we load functions at runtime using libloading.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::RdmaError;

// ─── ibverbs opaque types (pointers only, no struct layout needed) ───

/// Opaque protection domain
pub enum IbvPd {}

// ─── ibv_context_ops vtable (from macOS SDK verbs.h) ───
// These 3 ops (poll_cq, post_send, post_recv) are static inline in C headers,
// so Apple's librdma.dylib does NOT export them as dynamic symbols.
// We must call through the context ops vtable directly.

/// Function pointer types for vtable ops
type FnPollCq = unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int;
type FnPostSend = unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int;
type FnPostRecv = unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int;

/// ibv_context_ops — 32 function pointers matching verbs.h layout.
/// Only poll_cq, post_send, post_recv are typed; the rest are opaque void*.
#[repr(C)]
pub struct IbvContextOps {
    _compat_query_device: *const c_void,
    _compat_query_port: *const c_void,
    _compat_alloc_pd: *const c_void,
    _compat_dealloc_pd: *const c_void,
    _compat_reg_mr: *const c_void,
    _compat_rereg_mr: *const c_void,
    _compat_dereg_mr: *const c_void,
    _alloc_mw: *const c_void,
    _bind_mw: *const c_void,
    _dealloc_mw: *const c_void,
    _compat_create_cq: *const c_void,
    pub poll_cq: FnPollCq, // index 11
    _req_notify_cq: *const c_void,
    _compat_cq_event: *const c_void,
    _compat_resize_cq: *const c_void,
    _compat_destroy_cq: *const c_void,
    _compat_create_srq: *const c_void,
    _compat_modify_srq: *const c_void,
    _compat_query_srq: *const c_void,
    _compat_destroy_srq: *const c_void,
    _post_srq_recv: *const c_void,
    _compat_create_qp: *const c_void,
    _compat_query_qp: *const c_void,
    _compat_modify_qp: *const c_void,
    _compat_destroy_qp: *const c_void,
    pub post_send: FnPostSend, // index 25
    pub post_recv: FnPostRecv, // index 26
    _compat_create_ah: *const c_void,
    _compat_destroy_ah: *const c_void,
    _compat_attach_mcast: *const c_void,
    _compat_detach_mcast: *const c_void,
    _compat_async_event: *const c_void,
}

/// ibv_context — non-opaque, we need access to the ops vtable.
#[repr(C)]
pub struct IbvContext {
    pub device: *mut IbvDevice,
    pub ops: IbvContextOps,
    // Remaining fields (cmd_fd, async_fd, num_comp_vectors, mutex, abi_compat) are opaque
    _opaque: [u8; 128], // generous padding
}

/// ibv_cq — non-opaque, we need context pointer for poll_cq vtable call.
#[repr(C)]
pub struct IbvCq {
    pub context: *mut IbvContext,
    _channel: *mut c_void,
    _cq_context: *mut c_void,
    pub handle: u32,
    pub cqe: c_int,
    // Remaining fields (mutex, cond, event counters) are opaque
    _opaque: [u8; 128], // generous padding for pthread types
}
/// Queue pair — we need to read qp_num from this.
/// Only the first fields up to qp_num are defined; the rest is opaque.
#[repr(C)]
pub struct IbvQp {
    pub context: *mut IbvContext,
    pub qp_context: *mut c_void,
    pub pd: *mut IbvPd,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut c_void,
    pub handle: u32,
    pub qp_num: u32,
    pub state: u32,   // ibv_qp_state
    pub qp_type: u32, // ibv_qp_type
    // Remaining fields (mutex, cond, events_completed) are opaque
    _opaque: [u8; 120], // generous padding for pthread types
}
/// Opaque device
pub enum IbvDevice {}

/// Memory region — we need to read fields from this
#[repr(C)]
pub struct IbvMr {
    pub context: *mut IbvContext,
    pub pd: *mut IbvPd,
    pub addr: *mut c_void,
    pub length: usize,
    pub handle: u32,
    pub lkey: u32,
    pub rkey: u32,
}

/// GID global identifier
#[repr(C)]
#[derive(Copy, Clone)]
pub struct IbvGidGlobal {
    pub subnet_prefix: u64,
    pub interface_id: u64,
}

/// GID union
#[repr(C)]
pub union IbvGid {
    pub raw: [u8; 16],
    pub global: IbvGidGlobal,
}

impl Copy for IbvGid {}
impl Clone for IbvGid {
    fn clone(&self) -> Self {
        *self
    }
}

/// Global route header
#[repr(C)]
pub struct IbvGlobalRoute {
    pub dgid: IbvGid,
    pub flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
    _pad: u8,
}

/// Address handle attributes
#[repr(C)]
pub struct IbvAhAttr {
    pub grh: IbvGlobalRoute,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub static_rate: u8,
    pub is_global: u8,
    pub port_num: u8,
    _pad: u8,
}

/// QP capabilities
#[repr(C)]
pub struct IbvQpCap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

/// QP init attributes
#[repr(C)]
pub struct IbvQpInitAttr {
    pub qp_context: *mut c_void,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut c_void,
    pub cap: IbvQpCap,
    pub qp_type: u32,
    pub sq_sig_all: c_int,
}

/// QP attributes for modify/query
#[repr(C)]
pub struct IbvQpAttr {
    pub qp_state: u32,
    pub cur_qp_state: u32,
    pub path_mtu: u32,
    pub path_mig_state: u32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: IbvQpCap,
    pub ah_attr: IbvAhAttr,
    pub alt_ah_attr: IbvAhAttr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    _pad: u8,
    pub rate_limit: u32,
}

/// Device attributes (from ibv_query_device)
#[repr(C)]
pub struct IbvDeviceAttr {
    pub fw_ver: [c_char; 64],
    pub node_guid: u64,
    pub sys_image_guid: u64,
    pub max_mr_size: u64,
    pub page_size_cap: u64,
    pub vendor_id: u32,
    pub vendor_part_id: u32,
    pub hw_ver: u32,
    pub max_qp: c_int,
    pub max_qp_wr: c_int,
    pub device_cap_flags: u32,
    pub max_sge: c_int,
    pub max_sge_rd: c_int,
    pub max_cq: c_int,
    pub max_cqe: c_int,
    pub max_mr: c_int,
    pub max_pd: c_int,
    pub max_qp_rd_atom: c_int,
    pub max_ee_rd_atom: c_int,
    pub max_res_rd_atom: c_int,
    pub max_qp_init_rd_atom: c_int,
    pub max_ee_init_rd_atom: c_int,
    pub atomic_cap: c_int,
    pub max_ee: c_int,
    pub max_rdd: c_int,
    pub max_mw: c_int,
    pub max_raw_ipv6_qp: c_int,
    pub max_raw_ethy_qp: c_int,
    pub max_mcast_grp: c_int,
    pub max_mcast_qp_attach: c_int,
    pub max_total_mcast_qp_attach: c_int,
    pub max_ah: c_int,
    pub max_fmr: c_int,
    pub max_map_per_fmr: c_int,
    pub max_srq: c_int,
    pub max_srq_wr: c_int,
    pub max_srq_sge: c_int,
    pub max_pkeys: u16,
    pub local_ca_ack_delay: u8,
    pub phys_port_cnt: u8,
}

/// Port attributes
#[repr(C)]
pub struct IbvPortAttr {
    pub state: u32,
    pub max_mtu: u32,
    pub active_mtu: u32,
    pub gid_tbl_len: i32,
    pub port_cap_flags: u32,
    pub max_msg_sz: u32,
    pub bad_pkey_cntr: u32,
    pub qkey_viol_cntr: u32,
    pub pkey_tbl_len: u16,
    pub lid: u16,
    pub sm_lid: u16,
    pub lmc: u8,
    pub max_vl_num: u8,
    pub sm_sl: u8,
    pub subnet_timeout: u8,
    pub init_type_reply: u8,
    pub active_width: u8,
    pub active_speed: u8,
    pub phys_state: u8,
    pub link_layer: u8,
    pub flags: u8,
    pub port_cap_flags2: u16,
}

/// Work completion
#[repr(C)]
#[derive(Copy, Clone)]
pub struct IbvWc {
    pub wr_id: u64,
    pub status: u32, // ibv_wc_status
    pub opcode: u32,
    pub vendor_err: u32,
    pub byte_len: u32,
    pub imm_data: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub sl: u8,
    pub dlid_path_bits: u8,
}

/// Send work request — matches C ibv_send_wr layout (~80 bytes).
///
/// # ABI Contract
/// This struct must match the C `ibv_send_wr` layout from libibverbs exactly:
/// - `wr_id` at offset 0 (u64)
/// - `next` pointer at offset 8
/// - `sg_list` pointer at offset 16
/// - `num_sge` (c_int) at offset 24
/// - `opcode` (u32) at offset 28
/// - `send_flags` (u32) at offset 32
/// - `imm_data` (u32) at offset 36 (4 bytes padding follows on 64-bit)
/// - `wr` union at offset 40 (32 bytes)
/// - `qp_type` union at offset 72 (4 bytes)
/// - Total size >= 80 bytes
///
/// The trailing `_qp_type_pad` covers the C union `qp_type { struct { uint32_t remote_srqn; } xrc; }`.
/// The `_trailing_pad` provides safety margin for any platform-specific trailing fields.
#[repr(C)]
pub struct IbvSendWr {
    pub wr_id: u64,
    pub next: *mut IbvSendWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: c_int,
    pub opcode: u32,
    pub send_flags: u32,
    /// Immediate data (also used as invalidate_rkey in some operations).
    pub imm_data: u32,
    /// Work request type-specific data (RDMA/atomic/UD).
    pub wr: IbvWrUnion,
    /// Trailing qp_type union (covers xrc.remote_srqn, 4 bytes).
    _qp_type_pad: u32,
    /// Safety margin for platform-specific trailing fields or future extensions.
    _trailing_pad: [u8; 4],
}

/// Receive work request
#[repr(C)]
pub struct IbvRecvWr {
    pub wr_id: u64,
    pub next: *mut IbvRecvWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: c_int,
}

/// Scatter/gather entry
#[repr(C)]
pub struct IbvSge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

/// Work request union — sized to match the C ibv_send_wr.wr union.
/// The C union's largest variant (atomic) is 28 bytes; with 8-byte alignment → 32 bytes.
/// We keep the rdma-relevant named fields and add padding for the atomic variant's extra space.
#[repr(C)]
pub struct IbvWrUnion {
    /// For UD: address handle pointer. For RDMA: overlaps with remote_addr.
    pub ah: *mut c_void,
    /// For RDMA: remote address. For atomic: compare_add lives here.
    pub remote_addr: u64,
    /// For RDMA/atomic: remote key.
    pub rkey: u32,
    /// Padding to cover atomic variant's swap field and alignment.
    /// atomic variant needs: remote_addr(8)+compare_add(8)+swap(8)+rkey(4) = 28B → 32B aligned
    _atomic_pad: [u8; 12],
}

/// ibv_access_flags
pub mod access_flags {
    use std::ffi::c_int;
    pub const LOCAL_WRITE: c_int = 1;
    pub const REMOTE_WRITE: c_int = 2;
    pub const REMOTE_READ: c_int = 4;
}

/// ibv_wc_status
pub mod wc_status {
    pub const SUCCESS: u32 = 0;
    pub const LOC_LEN_ERR: u32 = 1;
    pub const LOC_QP_OP_ERR: u32 = 2;
    pub const LOC_EEC_OP_ERR: u32 = 3;
    pub const LOC_PROT_ERR: u32 = 4;
    pub const WR_FLUSH_ERR: u32 = 5;
    pub const MW_BIND_ERR: u32 = 6;
    pub const BAD_RESP_ERR: u32 = 7;
    pub const LOC_ACCESS_ERR: u32 = 8;
    pub const REM_INV_REQ_ERR: u32 = 9;
    pub const REM_ACCESS_ERR: u32 = 10;
    pub const REM_OP_ERR: u32 = 11;
    pub const RETRY_EXC_ERR: u32 = 12;
    pub const RNR_RETRY_EXC_ERR: u32 = 13;
    pub const LOC_RDD_VIOL_ERR: u32 = 14;
    pub const REM_INV_RD_REQ_ERR: u32 = 15;
    pub const REM_ABORT_ERR: u32 = 16;
    pub const INV_EECN_ERR: u32 = 17;
    pub const INV_EEC_STATE_ERR: u32 = 18;
    pub const FATAL_ERR: u32 = 19;
    pub const RESP_TIMEOUT_ERR: u32 = 20;
    pub const GENERAL_ERR: u32 = 21;
}

pub fn wc_status_str(status: u32) -> &'static str {
    match status {
        wc_status::SUCCESS => "SUCCESS",
        wc_status::LOC_LEN_ERR => "LOC_LEN_ERR",
        wc_status::LOC_QP_OP_ERR => "LOC_QP_OP_ERR",
        wc_status::LOC_EEC_OP_ERR => "LOC_EEC_OP_ERR",
        wc_status::LOC_PROT_ERR => "LOC_PROT_ERR",
        wc_status::WR_FLUSH_ERR => "WR_FLUSH_ERR",
        wc_status::MW_BIND_ERR => "MW_BIND_ERR",
        wc_status::BAD_RESP_ERR => "BAD_RESP_ERR",
        wc_status::LOC_ACCESS_ERR => "LOC_ACCESS_ERR",
        wc_status::REM_INV_REQ_ERR => "REM_INV_REQ_ERR",
        wc_status::REM_ACCESS_ERR => "REM_ACCESS_ERR",
        wc_status::REM_OP_ERR => "REM_OP_ERR",
        wc_status::RETRY_EXC_ERR => "RETRY_EXC_ERR",
        wc_status::RNR_RETRY_EXC_ERR => "RNR_RETRY_EXC_ERR",
        wc_status::LOC_RDD_VIOL_ERR => "LOC_RDD_VIOL_ERR",
        wc_status::REM_INV_RD_REQ_ERR => "REM_INV_RD_REQ_ERR",
        wc_status::REM_ABORT_ERR => "REM_ABORT_ERR",
        wc_status::INV_EECN_ERR => "INV_EECN_ERR",
        wc_status::INV_EEC_STATE_ERR => "INV_EEC_STATE_ERR",
        wc_status::FATAL_ERR => "FATAL_ERR",
        wc_status::RESP_TIMEOUT_ERR => "RESP_TIMEOUT_ERR",
        wc_status::GENERAL_ERR => "GENERAL_ERR",
        _ => "UNKNOWN",
    }
}

/// ibv_wr_opcode
pub mod wr_opcode {
    pub const SEND: u32 = 0;
    pub const SEND_WITH_IMM: u32 = 1;
}

/// ibv_send_flags
pub mod send_flags {
    pub const SIGNALED: u32 = 1;
}

/// ibv_qp_state
pub mod qp_state {
    pub const RESET: u32 = 0;
    pub const INIT: u32 = 1;
    pub const RTR: u32 = 2;
    pub const RTS: u32 = 3;
    pub const SQD: u32 = 4;
    pub const SQE: u32 = 5;
    pub const ERR: u32 = 6;
}

/// ibv_qp_type
pub mod qp_type {
    pub const RC: u32 = 2;
    pub const UC: u32 = 3;
    pub const UD: u32 = 4;
}

/// ibv_mtu
pub mod mtu {
    pub const MTU_256: u32 = 1;
    pub const MTU_512: u32 = 2;
    pub const MTU_1024: u32 = 3;
    pub const MTU_2048: u32 = 4;
    pub const MTU_4096: u32 = 5;
}

/// ibv_qp_attr_mask
pub mod qp_attr_mask {
    use std::ffi::c_int;
    pub const STATE: c_int = 1;
    pub const CUR_STATE: c_int = 1 << 1;
    pub const EN_SQD_ASYNC_NOTIFY: c_int = 1 << 2;
    pub const ACCESS_FLAGS: c_int = 1 << 3;
    pub const PKEY_INDEX: c_int = 1 << 4;
    pub const PORT: c_int = 1 << 5;
    pub const QKEY: c_int = 1 << 6;
    pub const AV: c_int = 1 << 7;
    pub const PATH_MTU: c_int = 1 << 8;
    pub const TIMEOUT: c_int = 1 << 9;
    pub const RETRY_CNT: c_int = 1 << 10;
    pub const RNR_RETRY: c_int = 1 << 11;
    pub const RQ_PSN: c_int = 1 << 12;
    pub const MAX_QP_RD_ATOMIC: c_int = 1 << 13;
    pub const ALT_PATH: c_int = 1 << 14;
    pub const MIN_RNR_TIMER: c_int = 1 << 15;
    pub const SQ_PSN: c_int = 1 << 16;
    pub const MAX_DEST_RD_ATOMIC: c_int = 1 << 17;
    pub const PATH_MIG_STATE: c_int = 1 << 18;
    pub const CAP: c_int = 1 << 19;
    pub const DEST_QPN: c_int = 1 << 20;
}

/// Dynamically loaded ibverbs library
pub struct IbverbsLib {
    _lib: Library,
    // Device management
    pub get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut IbvDevice,
    pub free_device_list: unsafe extern "C" fn(*mut *mut IbvDevice),
    pub get_device_name: unsafe extern "C" fn(*mut IbvDevice) -> *const c_char,
    pub open_device: unsafe extern "C" fn(*mut IbvDevice) -> *mut IbvContext,
    pub close_device: unsafe extern "C" fn(*mut IbvContext) -> c_int,
    // PD
    pub alloc_pd: unsafe extern "C" fn(*mut IbvContext) -> *mut IbvPd,
    pub dealloc_pd: unsafe extern "C" fn(*mut IbvPd) -> c_int,
    // MR
    pub reg_mr: unsafe extern "C" fn(*mut IbvPd, *mut c_void, usize, c_int) -> *mut IbvMr,
    pub dereg_mr: unsafe extern "C" fn(*mut IbvMr) -> c_int,
    // CQ
    pub create_cq:
        unsafe extern "C" fn(*mut IbvContext, c_int, *mut c_void, *mut c_void, c_int) -> *mut IbvCq,
    pub destroy_cq: unsafe extern "C" fn(*mut IbvCq) -> c_int,
    // NOTE: poll_cq, post_send, post_recv are NOT loaded from the library.
    // Apple's librdma.dylib does not export them (they are static inline in verbs.h).
    // Use ibv_poll_cq(), ibv_post_send(), ibv_post_recv() free functions instead.
    // QP management
    pub create_qp: unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp,
    pub destroy_qp: unsafe extern "C" fn(*mut IbvQp) -> c_int,
    pub modify_qp: unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int,
    // Device/Port/GID queries
    pub query_device: unsafe extern "C" fn(*mut IbvContext, *mut IbvDeviceAttr) -> c_int,
    pub query_port: unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int,
    pub query_gid: unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int,
}

static LIB: OnceLock<Result<IbverbsLib, String>> = OnceLock::new();

impl IbverbsLib {
    /// Try to load librdma.dylib. Returns cached result on subsequent calls.
    pub fn load() -> Result<&'static IbverbsLib, RdmaError> {
        let result = LIB.get_or_init(|| Self::load_inner().map_err(|e| e.to_string()));
        match result {
            Ok(lib) => Ok(lib),
            Err(e) => Err(RdmaError::LibraryNotFound(e.clone())),
        }
    }

    fn load_inner() -> Result<Self, libloading::Error> {
        // SAFETY: We load the library and immediately resolve all needed symbols.
        // The Library is kept alive in the struct for the 'static lifetime via OnceLock.
        unsafe {
            // Try short name first (works when librdma.dylib is on DYLD_LIBRARY_PATH or
            // in the dyld shared cache). Fall back to absolute path for macOS Big Sur+
            // where /usr/lib dylibs only exist in the shared cache.
            let lib = Library::new("librdma.dylib")
                .or_else(|e1| {
                    eprintln!("[rmlx-rdma] dlopen(librdma.dylib) failed: {e1}, trying /usr/lib/librdma.dylib");
                    Library::new("/usr/lib/librdma.dylib")
                })
                .or_else(|e2| {
                    eprintln!("[rmlx-rdma] dlopen(/usr/lib/librdma.dylib) failed: {e2}, trying /usr/lib/rdma/libibverbs.dylib");
                    Library::new("/usr/lib/rdma/libibverbs.dylib")
                })?;

            // Helper macro to load a symbol with an explicit type
            macro_rules! load_sym {
                ($lib:expr, $name:expr, $ty:ty) => {{
                    let sym: Symbol<$ty> = $lib.get($name.as_bytes())?;
                    *sym
                }};
            }

            type FnGetDeviceList = unsafe extern "C" fn(*mut c_int) -> *mut *mut IbvDevice;
            type FnFreeDeviceList = unsafe extern "C" fn(*mut *mut IbvDevice);
            type FnGetDeviceName = unsafe extern "C" fn(*mut IbvDevice) -> *const c_char;
            type FnOpenDevice = unsafe extern "C" fn(*mut IbvDevice) -> *mut IbvContext;
            type FnCloseDevice = unsafe extern "C" fn(*mut IbvContext) -> c_int;
            type FnAllocPd = unsafe extern "C" fn(*mut IbvContext) -> *mut IbvPd;
            type FnDeallocPd = unsafe extern "C" fn(*mut IbvPd) -> c_int;
            type FnRegMr =
                unsafe extern "C" fn(*mut IbvPd, *mut c_void, usize, c_int) -> *mut IbvMr;
            type FnDeregMr = unsafe extern "C" fn(*mut IbvMr) -> c_int;
            type FnCreateCq = unsafe extern "C" fn(
                *mut IbvContext,
                c_int,
                *mut c_void,
                *mut c_void,
                c_int,
            ) -> *mut IbvCq;
            type FnDestroyCq = unsafe extern "C" fn(*mut IbvCq) -> c_int;
            type FnCreateQp = unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp;
            type FnDestroyQp = unsafe extern "C" fn(*mut IbvQp) -> c_int;
            type FnModifyQp = unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int;
            type FnQueryDevice = unsafe extern "C" fn(*mut IbvContext, *mut IbvDeviceAttr) -> c_int;
            type FnQueryPort = unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int;
            type FnQueryGid =
                unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int;

            Ok(Self {
                get_device_list: load_sym!(lib, "ibv_get_device_list", FnGetDeviceList),
                free_device_list: load_sym!(lib, "ibv_free_device_list", FnFreeDeviceList),
                get_device_name: load_sym!(lib, "ibv_get_device_name", FnGetDeviceName),
                open_device: load_sym!(lib, "ibv_open_device", FnOpenDevice),
                close_device: load_sym!(lib, "ibv_close_device", FnCloseDevice),
                alloc_pd: load_sym!(lib, "ibv_alloc_pd", FnAllocPd),
                dealloc_pd: load_sym!(lib, "ibv_dealloc_pd", FnDeallocPd),
                reg_mr: load_sym!(lib, "ibv_reg_mr", FnRegMr),
                dereg_mr: load_sym!(lib, "ibv_dereg_mr", FnDeregMr),
                create_cq: load_sym!(lib, "ibv_create_cq", FnCreateCq),
                destroy_cq: load_sym!(lib, "ibv_destroy_cq", FnDestroyCq),
                create_qp: load_sym!(lib, "ibv_create_qp", FnCreateQp),
                destroy_qp: load_sym!(lib, "ibv_destroy_qp", FnDestroyQp),
                modify_qp: load_sym!(lib, "ibv_modify_qp", FnModifyQp),
                query_device: load_sym!(lib, "ibv_query_device", FnQueryDevice),
                query_port: load_sym!(lib, "ibv_query_port", FnQueryPort),
                query_gid: load_sym!(lib, "ibv_query_gid", FnQueryGid),
                _lib: lib,
            })
        }
    }
}

// ─── Inline vtable wrappers ───
// These replicate the static inline functions from verbs.h that Apple's
// librdma.dylib does NOT export as dynamic symbols.

/// Poll a CQ for work completions (replicates inline ibv_poll_cq from verbs.h).
///
/// # Safety
/// `cq` must be a valid ibv_cq pointer. `wc` must point to valid memory for
/// at least `num_entries` IbvWc structs.
pub unsafe fn ibv_poll_cq(cq: *mut IbvCq, num_entries: c_int, wc: *mut IbvWc) -> c_int {
    // C inline: return cq->context->ops.poll_cq(cq, num_entries, wc);
    unsafe {
        let ctx = (*cq).context;
        ((*ctx).ops.poll_cq)(cq, num_entries, wc)
    }
}

/// Post a list of send work requests (replicates inline ibv_post_send from verbs.h).
///
/// # Safety
/// `qp` must be a valid ibv_qp pointer. `wr` must point to a valid send work request.
pub unsafe fn ibv_post_send(
    qp: *mut IbvQp,
    wr: *mut IbvSendWr,
    bad_wr: *mut *mut IbvSendWr,
) -> c_int {
    // C inline: return qp->context->ops.post_send(qp, wr, bad_wr);
    unsafe {
        let ctx = (*qp).context;
        ((*ctx).ops.post_send)(qp, wr, bad_wr)
    }
}

/// Post a list of recv work requests (replicates inline ibv_post_recv from verbs.h).
///
/// # Safety
/// `qp` must be a valid ibv_qp pointer. `wr` must point to a valid recv work request.
pub unsafe fn ibv_post_recv(
    qp: *mut IbvQp,
    wr: *mut IbvRecvWr,
    bad_wr: *mut *mut IbvRecvWr,
) -> c_int {
    // C inline: return qp->context->ops.post_recv(qp, wr, bad_wr);
    unsafe {
        let ctx = (*qp).context;
        ((*ctx).ops.post_recv)(qp, wr, bad_wr)
    }
}

// ─── Compile-time ABI assertions ───
// These ensure our repr(C) structs match the libibverbs C ABI.
// If any assertion fails, the struct layout has diverged from the C definition.

const _: () = {
    // IbvSendWr: C ibv_send_wr is at least 80 bytes on 64-bit platforms.
    assert!(std::mem::size_of::<IbvSendWr>() >= 80);
    // IbvRecvWr: wr_id(8) + next(8) + sg_list(8) + num_sge(4) + padding(4) = 32, but
    // the C struct may be 40 bytes depending on alignment. Must be at least 28 usable.
    assert!(std::mem::size_of::<IbvRecvWr>() >= 28);
    // IbvSge: addr(8) + length(4) + lkey(4) = exactly 16 bytes (no padding).
    assert!(std::mem::size_of::<IbvSge>() == 16);
    // IbvWc: must be at least 48 bytes to cover all C fields.
    assert!(std::mem::size_of::<IbvWc>() >= 48);
};

// Verify critical field offsets match C ABI (wr_id must always be at offset 0).
const _: () = {
    assert!(std::mem::offset_of!(IbvSendWr, wr_id) == 0);
    assert!(std::mem::offset_of!(IbvRecvWr, wr_id) == 0);
    assert!(std::mem::offset_of!(IbvWc, wr_id) == 0);
    // IbvSge layout: addr at 0, length at 8, lkey at 12
    assert!(std::mem::offset_of!(IbvSge, addr) == 0);
    assert!(std::mem::offset_of!(IbvSge, length) == 8);
    assert!(std::mem::offset_of!(IbvSge, lkey) == 12);
};
