//! GPU kernel operations for RMLX arrays.

pub mod binary;
pub mod copy;
pub mod gemv;
pub mod indexing;
pub mod matmul;
pub mod quantized;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod softmax;

use crate::array::Array;
use crate::kernels::{KernelError, KernelRegistry};

/// If the array is non-contiguous, return a contiguous copy. Otherwise `None`.
pub(crate) fn make_contiguous(
    array: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Option<Array>, KernelError> {
    if array.is_contiguous() {
        Ok(None)
    } else {
        Ok(Some(copy::copy(registry, array, queue)?))
    }
}

/// Register all built-in kernels with the given registry.
pub fn register_all(registry: &KernelRegistry) -> Result<(), KernelError> {
    copy::register(registry)?;
    binary::register(registry)?;
    reduce::register(registry)?;
    rms_norm::register(registry)?;
    softmax::register(registry)?;
    rope::register(registry)?;
    gemv::register(registry)?;
    matmul::register(registry)?;
    quantized::register(registry)?;
    indexing::register(registry)?;
    Ok(())
}

/// Execution mode for kernel dispatch.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Synchronous: commit and wait immediately (default, safe).
    #[default]
    Sync,
    /// Async: commit but don't wait. Caller must ensure sync before reading.
    Async,
}

/// Commit a command buffer with the given execution mode.
///
/// In `Sync` mode, blocks until the GPU finishes.
/// In `Async` mode, returns immediately after commit — the caller must
/// call `wait_until_completed()` on the command buffer before reading results.
pub fn commit_with_mode(cb: &metal::CommandBufferRef, mode: ExecMode) {
    cb.commit();
    match mode {
        ExecMode::Sync => cb.wait_until_completed(),
        ExecMode::Async => { /* caller manages sync */ }
    }
}
