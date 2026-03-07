//! rmlx-metal — Metal GPU abstraction layer for RMLX

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unexpected_cfgs)] // objc crate uses deprecated cfg(feature = "cargo-clippy")

// Required for objc::msg_send! macro in autorelease and capture modules.
#[macro_use]
extern crate objc;

pub mod autorelease;
pub mod batcher;
pub mod buffer;
pub mod capture;
pub mod command;
pub mod device;
pub mod event;
pub mod exec_graph;
pub mod fence;
pub mod icb;
pub mod icb_sparse;
pub mod library;
pub mod library_cache;
pub mod managed_buffer;
pub mod msl_version;
pub mod pipeline;
pub mod pipeline_cache;
pub mod queue;
pub mod self_check;
pub mod stream;

// Re-export metal crate for downstream users
pub use metal;

// Re-export core types for convenience
pub use autorelease::ScopedPool;
pub use batcher::CommandBatcher;
pub use capture::{CaptureDestination, CaptureScope};
pub use command::{
    memory_barrier_scope_buffers, new_concurrent_encoder, probe_concurrent_dispatch,
    BarrierTracker, CommandBufferManager, GpuError, GpuErrorStore,
};
pub use device::{
    ArchClass, Architecture, ChipTuning, GpuDevice, DEFAULT_BUFFER_OPTIONS, TRACKED_BUFFER_OPTIONS,
    UNTRACKED_BUFFER_OPTIONS,
};
pub use event::GpuEvent;
pub use exec_graph::ExecGraph;
pub use fence::{FenceError, GpuFence};
pub use icb_sparse::{
    grouped_forward_icb, CachedSparsityPattern, IcbReplayCache, SparseDispatchResult,
    SparseExpertCache, SparseExpertConfig, SparseExpertKey, SparseExpertPlan,
};
pub use library_cache::LibraryCache;
pub use managed_buffer::{BufferAllocator, ManagedBuffer};
pub use msl_version::{DeviceInfo, MslVersion};
pub use pipeline::{FunctionConstant, PipelineCache};
pub use pipeline_cache::DiskPipelineCache;
pub use queue::GpuQueue;
pub use stream::{StreamManager, StreamSync, STREAM_COMPUTE, STREAM_COPY, STREAM_DEFAULT};

/// Errors from Metal operations
#[derive(Debug)]
pub enum MetalError {
    NoDevice,
    ShaderCompile(String),
    PipelineCreate(String),
    LibraryLoad(String),
    KernelNotFound(String),
    StreamNotFound(usize),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device found"),
            Self::ShaderCompile(e) => write!(f, "shader compile error: {e}"),
            Self::PipelineCreate(e) => write!(f, "pipeline creation error: {e}"),
            Self::LibraryLoad(e) => write!(f, "library load error: {e}"),
            Self::KernelNotFound(name) => write!(f, "kernel function not found: {name}"),
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
        }
    }
}

impl std::error::Error for MetalError {}
