//! ibverbs FFI bindings via dynamic loading (dlopen)
//!
//! Since macOS TB5 RDMA drivers provide librdma.dylib rather than standard
//! libibverbs headers, we load functions at runtime using libloading.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::RdmaError;

// ─── ibverbs opaque types (pointers only, no struct layout needed) ───

/// Opaque ibverbs context
pub enum IbvContext {}
/// Opaque protection domain
pub enum IbvPd {}
/// Opaque completion queue
pub enum IbvCq {}
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
/// SAFETY: The struct is repr(C) and sized to match the C ibv_send_wr.
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
    pub poll_cq: unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int,
    // QP management
    pub create_qp: unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp,
    pub destroy_qp: unsafe extern "C" fn(*mut IbvQp) -> c_int,
    pub modify_qp: unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int,
    // Port/GID queries
    pub query_port: unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int,
    pub query_gid: unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int,
    // Post operations
    pub post_send: unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int,
    pub post_recv: unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int,
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
            let lib = Library::new("librdma.dylib")?;

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
            type FnPollCq = unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int;
            type FnCreateQp = unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp;
            type FnDestroyQp = unsafe extern "C" fn(*mut IbvQp) -> c_int;
            type FnModifyQp = unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int;
            type FnQueryPort = unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int;
            type FnQueryGid =
                unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int;
            type FnPostSend =
                unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int;
            type FnPostRecv =
                unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int;

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
                poll_cq: load_sym!(lib, "ibv_poll_cq", FnPollCq),
                create_qp: load_sym!(lib, "ibv_create_qp", FnCreateQp),
                destroy_qp: load_sym!(lib, "ibv_destroy_qp", FnDestroyQp),
                modify_qp: load_sym!(lib, "ibv_modify_qp", FnModifyQp),
                query_port: load_sym!(lib, "ibv_query_port", FnQueryPort),
                query_gid: load_sym!(lib, "ibv_query_gid", FnQueryGid),
                post_send: load_sym!(lib, "ibv_post_send", FnPostSend),
                post_recv: load_sym!(lib, "ibv_post_recv", FnPostRecv),
                _lib: lib,
            })
        }
    }
}
