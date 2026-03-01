//! rmlx-distributed — Distributed computing for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod credit_manager;
pub mod ep_runtime;
pub mod group;
pub mod metrics;
pub mod moe_exchange;
pub mod moe_policy;
pub mod perf_counters;
pub mod pipeline;
pub mod sparse_guard;
pub mod transport;
pub mod warmup;

// ── Re-exports of core types ──
pub use credit_manager::CreditManager;
pub use ep_runtime::EpRuntimeContext;
pub use group::{DistributedError, Group, RdmaTransport};
pub use metrics::{MoeMetrics as MoeMetricsAtomic, MoeMetricsSnapshot};
pub use moe_exchange::{
    AsyncCombineHandle, AsyncDispatchResult, DispatchLayout, DispatchResult, MoeCombineExchange,
    MoeDispatchConfig, MoeDispatchExchange,
};
pub use moe_policy::{MoeBackend, MoePolicy, ThresholdCalibration};
pub use perf_counters::{global_counters, PerfCounters, PerfSnapshot};
pub use pipeline::{
    LayerPipeline, LayerTransferState, PipelineConfig, PipelineStage, PipelineStats,
};
pub use sparse_guard::{GuardAction, SparseGuard};
pub use transport::{RdmaConnectionTransport, RecvCredit};
pub use warmup::{WarmupConfig, WarmupResult, WarmupState};
