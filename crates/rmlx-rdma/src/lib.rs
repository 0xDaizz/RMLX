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
#[allow(deprecated)]
pub use collectives::{
    apply_reduce_op, apply_reduce_op_typed, chunk_boundaries, pipelined_ring_allreduce,
    ring_allgather, ring_allreduce, ring_allreduce_typed, ring_reduce_scatter,
    ring_reduce_scatter_typed, CollectiveDType, PipelinedRingBuffer, ReduceElement, ReduceOp,
    SlotState,
};
pub mod crc;
pub use connection::{
    CompletionTracker, PostedOp, PostedOpKind, RdmaConfig, RdmaConnection, RegisteredRecv,
    RegisteredSend,
};
#[allow(deprecated)]
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

/// Errors from RDMA operations
#[derive(Debug, thiserror::Error)]
pub enum RdmaError {
    /// librdma.dylib not found or failed to load
    #[error("RDMA library not found: {0}")]
    LibraryNotFound(String),
    /// No RDMA devices found
    #[error("no RDMA devices found")]
    NoDevices,
    /// Failed to open device
    #[error("failed to open RDMA device: {0}")]
    DeviceOpen(String),
    /// Failed to allocate protection domain
    #[error("failed to allocate protection domain")]
    PdAlloc,
    /// Failed to register memory region
    #[error("memory region registration failed: {0}")]
    MrReg(String),
    /// Failed to create completion queue
    #[error("failed to create completion queue")]
    CqCreate,
    /// Failed to create queue pair
    #[error("queue pair creation failed: {0}")]
    QpCreate(String),
    /// Failed to modify queue pair state
    #[error("queue pair state transition failed: {0}")]
    QpModify(String),
    /// Work request posting failed
    #[error("work request post failed: {0}")]
    PostFailed(String),
    /// Completion queue poll error
    #[error("completion queue poll error: {0}")]
    CqPoll(String),
    /// Connection setup failed
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    /// CQ poll timed out waiting for completion
    #[error("CQ poll timeout: {0}")]
    Timeout(String),
    /// Feature not available (RDMA hardware missing)
    #[error("RDMA unavailable: {0}")]
    Unavailable(String),
    /// Invalid argument (e.g. SGE out of bounds)
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Data corruption detected (CRC32 mismatch on UC transport)
    #[error("data corruption detected: {0}")]
    DataCorruption(String),
}

/// Check if RDMA is available on this system.
pub fn is_available() -> bool {
    ffi::IbverbsLib::load().is_ok()
}
