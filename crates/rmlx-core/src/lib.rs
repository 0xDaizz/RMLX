//! rmlx-core — Compute engine for RMLX

pub mod array;
pub mod dtype;
pub mod kernels;
pub mod logging;
pub mod lora;
pub mod metrics;
pub mod ops;
pub mod precision_guard;
pub mod shutdown;
pub mod vjp;

// ── Re-exports of core types ──
pub use array::Array;
pub use dtype::{DType, HasDType};
pub use kernels::{KernelError, KernelRegistry};
pub use logging::{LogEntry, LogLevel};
pub use lora::{LoraConfig, LoraLayer, LoraModel, LoraTrainer, TrainConfig};
pub use metrics::{MetricsSnapshot, RuntimeMetrics};
pub use ops::{CommandBufferHandle, ExecMode};
pub use precision_guard::{GuardAction, PrecisionGuard, PrecisionResult};
pub use shutdown::{ShutdownHandle, ShutdownSignal};
pub use vjp::{AddGrad, GradFn, MatMulGrad, MulGrad, Operation, Tape, TapedValue};

/// Path to the AOT-compiled Metal shader library.
/// Set by build.rs at compile time.
/// Empty string if Metal compiler was not available during build.
pub const METALLIB_PATH: &str = env!("RMLX_METALLIB_PATH");
