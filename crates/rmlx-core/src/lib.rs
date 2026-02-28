//! rmlx-core — Compute engine for RMLX

pub mod array;
pub mod dtype;
pub mod kernels;
pub mod logging;
pub mod metrics;
pub mod ops;
pub mod precision_guard;
pub mod shutdown;

/// Path to the AOT-compiled Metal shader library.
/// Set by build.rs at compile time.
/// Empty string if Metal compiler was not available during build.
pub const METALLIB_PATH: &str = env!("RMLX_METALLIB_PATH");
