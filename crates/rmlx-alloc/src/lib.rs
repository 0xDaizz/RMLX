//! rmlx-alloc — GPU memory allocator for RMLX

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unexpected_cfgs)] // objc crate uses deprecated cfg(feature = "cargo-clippy")

// Required for objc::msg_send! macro in residency module (metal3 feature).
#[cfg(feature = "metal3")]
#[macro_use]
extern crate objc;

pub mod allocator;
pub mod bfc;
pub mod buffer_pool;
pub mod cache;
pub mod leak_detector;
pub mod residency;
pub mod small_alloc;
pub mod stats;
pub mod zero_copy;

// ── Re-exports of core types ──
pub use allocator::MetalAllocator;
pub use bfc::{BfcAllocation, BfcAllocator};
pub use buffer_pool::BufferPool;
pub use cache::BufferCache;
pub use leak_detector::{LeakDetector, LeakReport};
pub use residency::{ResidencyError, ResidencyManager};
pub use small_alloc::{SmallAllocation, SmallBufferPool};
pub use stats::AllocStats;
pub use zero_copy::{
    CompletionError, CompletionFence, CompletionTicket, GpuCompletionHandler, InFlightToken,
    ZeroCopyBuffer,
};

use std::fmt;

/// Errors from allocation operations
#[derive(Debug)]
pub enum AllocError {
    PosixMemalign(i32),
    MetalBufferCreate,
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    PoolExhausted,
    MutexPoisoned,
    ZeroSize,
    /// A dtype-related error (e.g. quantized block misalignment).
    DType(String),
    /// Attempted to free a buffer not owned by this allocator (double-free or
    /// untracked pointer).
    InvalidFree,
    /// Attempted to free a buffer not owned by this allocator, returning the
    /// buffer to the caller.
    InvalidFreeBuffer(rmlx_metal::metal::Buffer),
    /// The buffer's Metal device does not match this allocator's device.
    DeviceMismatch,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PosixMemalign(rc) => write!(f, "posix_memalign failed with code {rc}"),
            Self::MetalBufferCreate => write!(f, "failed to create Metal buffer"),
            Self::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "out of memory: requested {requested} bytes, {available} available"
                )
            }
            Self::PoolExhausted => write!(f, "buffer pool exhausted"),
            Self::MutexPoisoned => write!(f, "allocator mutex poisoned"),
            Self::ZeroSize => write!(f, "zero-size allocation is not allowed"),
            Self::DType(msg) => write!(f, "dtype error: {msg}"),
            Self::InvalidFree => write!(f, "attempted to free an unowned or already-freed buffer"),
            Self::InvalidFreeBuffer(_) => write!(
                f,
                "attempted to free an unowned or already-freed buffer (buffer returned)"
            ),
            Self::DeviceMismatch => write!(f, "buffer device does not match allocator device"),
        }
    }
}

impl std::error::Error for AllocError {}
