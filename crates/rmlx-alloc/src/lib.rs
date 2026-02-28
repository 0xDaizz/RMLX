//! rmlx-alloc — GPU memory allocator for RMLX

pub mod allocator;
pub mod buffer_pool;
pub mod cache;
pub mod leak_detector;
pub mod stats;
pub mod zero_copy;

// ── Re-exports of core types ──
pub use allocator::MetalAllocator;
pub use buffer_pool::BufferPool;
pub use cache::BufferCache;
pub use leak_detector::{LeakDetector, LeakReport};
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
    OutOfMemory { requested: usize, available: usize },
    PoolExhausted,
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
        }
    }
}

impl std::error::Error for AllocError {}
