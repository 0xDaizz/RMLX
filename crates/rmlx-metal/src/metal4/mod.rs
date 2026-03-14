//! Metal 4 API wrappers (macOS 26+, M1+)
//!
//! All types and functions in this module require the `metal4` feature flag
//! and runtime detection via `GpuDevice::supports_metal4()`.
//!
//! - [`CommandAllocator`] — manages encoding memory for Metal 4 command buffers
//! - [`Mtl4CommandBuffer`] — explicit begin/end command buffer lifecycle
//! - [`CommandQueue4`] — batch commit of multiple command buffers

pub mod command;
pub mod compiler;
pub mod compute;
pub mod counter_heap;

pub use command::{CommandAllocator, CommandQueue4, Mtl4CommandBuffer};
pub use counter_heap::{CounterHeap, GpuTimestamp};
