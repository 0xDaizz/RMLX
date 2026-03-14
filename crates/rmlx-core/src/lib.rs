//! rmlx-core — Compute engine for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod arena;
pub mod array;
pub mod dtype;
pub mod formats;
pub mod fusion;
pub mod kernels;
pub mod lazy;
pub mod logging;
pub mod lora;
pub mod metrics;
pub mod ops;
pub mod precision_guard;
pub mod prelude;
pub mod shutdown;
pub mod vjp;

// ── Re-exports of core types ──
pub use array::{broadcast_shape, Array};
pub use dtype::{DType, DTypeError, HasDType};
pub use kernels::{KernelError, KernelRegistry};
pub use lazy::{EvalContext, LazyArray, LazyEvalError, LazyGraph, LazyOp, NodeId};
pub use logging::{LogEntry, LogLevel};
pub use lora::{LoraConfig, LoraLayer, LoraModel, LoraTrainer, TrainConfig};
pub use metrics::{MetricsSnapshot, RuntimeMetrics};
pub use ops::{CommandBufferHandle, ExecMode, LaunchResult};
pub use precision_guard::{GuardAction, PrecisionGuard, PrecisionResult};
pub use rmlx_alloc::{AllocError, MetalAllocator};
pub use shutdown::{ShutdownHandle, ShutdownSignal};
pub use vjp::{AddGrad, GradFn, MatMulGrad, MulGrad, Operation, Tape, TapedValue, VjpError};

/// Path to the AOT-compiled Metal shader library.
/// Set by build.rs at compile time.
/// Empty string if Metal compiler was not available during build.
pub const METALLIB_PATH: &str = env!("RMLX_METALLIB_PATH");

/// Shared test utilities — provides a single Metal device via `OnceLock`
/// to prevent concurrent `MTLCreateSystemDefaultDevice()` failures.
#[cfg(test)]
pub(crate) mod test_utils {
    use rmlx_metal::device::GpuDevice;
    use rmlx_metal::MtlDevice;
    use std::sync::OnceLock;

    /// Returns a shared raw Metal device, created exactly once across all tests.
    pub fn shared_metal_device() -> MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE
            .get_or_init(|| {
                objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
            })
            .clone()
    }

    /// Returns a fresh `GpuDevice` wrapper backed by the shared Metal device.
    ///
    /// Each call creates a new `GpuDevice` (with its own `StreamManager` etc.)
    /// but they all share the same underlying `MTLDevice`, avoiding the
    /// concurrent-creation crash.
    pub fn test_gpu() -> GpuDevice {
        GpuDevice::from_raw_device(shared_metal_device())
    }
}
