//! rmlx-rdma — RDMA communication layer for RMLX (Thunderbolt 5)

pub mod connection;
pub mod context;
pub mod exchange;
pub mod ffi;
pub mod mr;
pub mod multi_port;
pub mod qp;
pub mod rdma_metrics;

// ── Re-exports of core types ──
pub use connection::{CompletionTracker, RdmaConfig, RdmaConnection};
pub use context::{ProtectionDomain, RdmaContext, RdmaDeviceProbe};
pub use mr::MemoryRegion;
pub use multi_port::{
    DualPortConfig, PortConfig, PortFailover, PortState, StripeEngine, StripePlan, Topology,
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
        }
    }
}

impl std::error::Error for RdmaError {}

/// Check if RDMA is available on this system.
pub fn is_available() -> bool {
    ffi::IbverbsLib::load().is_ok()
}
