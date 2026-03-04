//! rmlx-core — Compute engine for RMLX

#![deny(unsafe_op_in_unsafe_fn)]

pub mod array;
pub mod dtype;
pub mod formats;
pub mod kernels;
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
pub use dtype::{DType, HasDType};
pub use rmlx_alloc::{AllocError, MetalAllocator};
pub use kernels::{KernelError, KernelRegistry};
pub use logging::{LogEntry, LogLevel};
pub use lora::{LoraConfig, LoraLayer, LoraModel, LoraTrainer, TrainConfig};
pub use metrics::{MetricsSnapshot, RuntimeMetrics};
pub use ops::{CommandBufferHandle, ExecMode, LaunchResult};
pub use precision_guard::{GuardAction, PrecisionGuard, PrecisionResult};
pub use shutdown::{ShutdownHandle, ShutdownSignal};
pub use vjp::{AddGrad, GradFn, MatMulGrad, MulGrad, Operation, Tape, TapedValue, VjpError};

/// Path to the AOT-compiled Metal shader library.
/// Set by build.rs at compile time.
/// Empty string if Metal compiler was not available during build.
pub const METALLIB_PATH: &str = env!("RMLX_METALLIB_PATH");
