//! rmlx-rdma — RDMA communication layer for RMLX (Thunderbolt 5)

#![deny(unsafe_op_in_unsafe_fn)]

pub mod collectives;
pub mod connection;
pub mod connection_manager;
pub mod context;
pub mod coordinator;
pub mod device_file;
pub mod exchange;
pub mod exchange_tag;
pub mod ffi;
pub mod gpu_doorbell;
pub mod mr;
pub mod mr_pool;
pub mod multi_port;
pub mod progress;
pub mod qp;
pub mod rdma_metrics;
pub mod shared_buffer;

// ── Re-exports of core types ──
pub use collectives::{
    apply_reduce_op, chunk_boundaries, ring_allgather, ring_allreduce, ring_reduce_scatter,
    ReduceOp,
};
pub mod crc;
pub use connection::{
    CompletionTracker, PostedOp, PostedOpKind, RdmaConfig, RdmaConnection, RegisteredRecv,
    RegisteredSend,
};
pub use connection_manager::ConnectionManager;
pub use context::{ProtectionDomain, RdmaContext, RdmaDeviceProbe};
pub use coordinator::{all_gather_bytes, all_gather_qp_info, barrier, CoordinatorConfig};
pub use device_file::DeviceMap;
pub use exchange::ExchangeConfig;
pub use exchange_tag::{ExchangeTag, WrIdFields};
pub use gpu_doorbell::{
    DescriptorHandler, DescriptorProxy, DescriptorRing, HandlerResult, ProxyConfig, RdmaDescriptor,
    RdmaOp,
};
pub use mr::MemoryRegion;
pub use mr_pool::{MrHandle, MrPool};
pub use multi_port::{
    DualPortConfig, PortConfig, PortFailover, PortState, StripeEngine, StripePlan, Topology,
};
pub use progress::{
    Completion, OpError, OwnedPendingOp, PendingOp, ProgressConfig, ProgressEngine, ProgressMode,
};
pub use qp::{CompletionQueue, QpInfo, QueuePair};
pub use rdma_metrics::{RdmaMetrics, RdmaMetricsSnapshot};

use std::fmt;

/// Errors from RDMA operations
#[derive(Debug)]
pub enum RdmaError {
    /// librdma.dylib not found or failed to load
    LibraryNotFound(String),
    /// No RDMA devices found
    NoDevices,
    /// Failed to open device
    DeviceOpen(String),
    /// Failed to allocate protection domain
    PdAlloc,
    /// Failed to register memory region
    MrReg(String),
    /// Failed to create completion queue
    CqCreate,
    /// Failed to create queue pair
    QpCreate(String),
    /// Failed to modify queue pair state
    QpModify(String),
    /// Work request posting failed
    PostFailed(String),
    /// Completion queue poll error
    CqPoll(String),
    /// Connection setup failed
    ConnectionFailed(String),
    /// CQ poll timed out waiting for completion
    Timeout(String),
    /// Feature not available (RDMA hardware missing)
    Unavailable(String),
    /// Invalid argument (e.g. SGE out of bounds)
    InvalidArgument(String),
    /// Data corruption detected (CRC32 mismatch on UC transport)
    DataCorruption(String),
}

impl fmt::Display for RdmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound(e) => write!(f, "RDMA library not found: {e}"),
            Self::NoDevices => write!(f, "no RDMA devices found"),
            Self::DeviceOpen(e) => write!(f, "failed to open RDMA device: {e}"),
            Self::PdAlloc => write!(f, "failed to allocate protection domain"),
            Self::MrReg(e) => write!(f, "memory region registration failed: {e}"),
            Self::CqCreate => write!(f, "failed to create completion queue"),
            Self::QpCreate(e) => write!(f, "queue pair creation failed: {e}"),
            Self::QpModify(e) => write!(f, "queue pair state transition failed: {e}"),
            Self::PostFailed(e) => write!(f, "work request post failed: {e}"),
            Self::CqPoll(e) => write!(f, "completion queue poll error: {e}"),
            Self::ConnectionFailed(e) => write!(f, "connection failed: {e}"),
            Self::Timeout(e) => write!(f, "CQ poll timeout: {e}"),
            Self::Unavailable(e) => write!(f, "RDMA unavailable: {e}"),
            Self::InvalidArgument(e) => write!(f, "invalid argument: {e}"),
            Self::DataCorruption(e) => write!(f, "data corruption detected: {e}"),
        }
    }
}

impl std::error::Error for RdmaError {}

/// Check if RDMA is available on this system.
pub fn is_available() -> bool {
    ffi::IbverbsLib::load().is_ok()
}
