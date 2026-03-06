//! rmlx-distributed — Distributed computing for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod credit_manager;
pub mod ep_runtime;
pub mod fp8_exchange;
pub mod group;
pub mod health;
pub mod init;
pub mod metrics;
pub mod moe_exchange;
pub mod moe_kernels;
pub mod moe_policy;
pub mod perf_counters;
pub mod pipeline;
pub mod progress_tracker;
pub mod slab_ring;
pub mod sparse_guard;
pub mod transport;
pub mod v3_protocol;
pub mod warmup;

// ── Re-exports of core types ──
pub use credit_manager::CreditManager;
pub use ep_runtime::{AcquiredBuffer, EpRuntimeContext};
pub use fp8_exchange::Fp8DispatchPayload;
pub use group::{
    AllreduceAlgorithm, DistributedError, Group, RdmaTransport, ReduceDtype, ReduceOp,
    TopologyRing, TREE_ALLREDUCE_THRESHOLD,
};
pub use health::{HealthMonitor, HeartbeatConfig, HeartbeatSender};
pub use init::{init, BackendHint, DistributedContext, InitConfig};
pub use metrics::{MoeMetrics as MoeMetricsAtomic, MoeMetricsSnapshot};
pub use moe_exchange::{
    AsyncCombineHandle, AsyncDispatchResult, DispatchLayout, DispatchResult, ExchangeBuffers,
    MoeCombineExchange, MoeDispatchConfig, MoeDispatchExchange, MoeDtype, WireProtocol,
};
pub use moe_kernels::init_kernels as init_moe_kernels;
pub use moe_policy::{MoeBackend, MoePolicy, ThresholdCalibration};
pub use perf_counters::{global_counters, PerfCounters, PerfSnapshot};
pub use pipeline::{
    LayerPipeline, LayerTransferState, PipelineConfig, PipelineStage, PipelineStats,
};
pub use progress_tracker::ProgressTracker;
pub use slab_ring::{Slab, SlabRing, SlabRingConfig, SlabRingError};
pub use sparse_guard::{GuardAction, SparseGuard};
pub use transport::{RdmaConnectionTransport, RecvCredit};
pub use v3_protocol::{
    blocking_exchange_v3, pack_combine_request_v3, pack_combine_response_v3, pack_dispatch_v3,
    unpack_combine_request_v3, unpack_combine_response_v3, unpack_dispatch_v3, PacketMeta,
    ProtocolError, V3CombineRequest, V3CombineResponse, V3DispatchPacket, V3ReceivedTokens,
};
pub use warmup::{WarmupConfig, WarmupResult, WarmupState};
