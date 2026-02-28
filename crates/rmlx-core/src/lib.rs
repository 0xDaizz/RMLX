//! rmlx-core — Compute engine for RMLX

/// Path to the AOT-compiled Metal shader library.
/// Set by build.rs at compile time.
/// Empty string if Metal compiler was not available during build.
pub const METALLIB_PATH: &str = env!("RMLX_METALLIB_PATH");
