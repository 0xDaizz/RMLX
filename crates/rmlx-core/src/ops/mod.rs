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
pub mod silu;
pub mod softmax;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::array::Array;
use crate::kernels::{KernelError, KernelRegistry};

/// Safely convert `usize` to `u32`, returning `KernelError::InvalidShape` on overflow.
pub(crate) fn checked_u32(val: usize, name: &str) -> Result<u32, KernelError> {
    u32::try_from(val)
        .map_err(|_| KernelError::InvalidShape(format!("{name} ({val}) exceeds u32::MAX")))
}

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
    silu::register(registry)?;
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

/// Handle for tracking async command buffer completion.
///
/// Returned by `commit_with_mode` in `Async` mode. The caller can poll
/// `is_complete()`, block with `wait()`, or use `wait_timeout()`.
#[derive(Clone)]
pub struct CommandBufferHandle {
    completed: Arc<AtomicBool>,
}

impl CommandBufferHandle {
    /// Create a handle from an existing completion flag (for testing).
    pub fn new_from_flag(completed: Arc<AtomicBool>) -> Self {
        Self { completed }
    }

    /// Returns true if the GPU command buffer has completed.
    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Block until the command buffer completes.
    pub fn wait(&self) {
        while !self.completed.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }

    /// Block until the command buffer completes or the timeout expires.
    /// Returns true if completed, false if timed out.
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while !self.completed.load(Ordering::Acquire) {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

/// Commit a command buffer with the given execution mode.
///
/// In `Sync` mode, blocks until the GPU finishes and returns `None`.
/// In `Async` mode, returns `Some(CommandBufferHandle)` that the caller
/// can use to poll or wait for completion.
pub fn commit_with_mode(
    cb: &metal::CommandBufferRef,
    mode: ExecMode,
) -> Option<CommandBufferHandle> {
    cb.commit();
    match mode {
        ExecMode::Sync => {
            cb.wait_until_completed();
            None
        }
        ExecMode::Async => {
            let completed = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&completed);
            // Retain the command buffer (increments Obj-C refcount) so we
            // can safely wait on it from a background thread.
            let owned_cb = cb.to_owned();
            std::thread::spawn(move || {
                owned_cb.wait_until_completed();
                flag.store(true, Ordering::Release);
            });
            Some(CommandBufferHandle { completed })
        }
    }
}

/// Result of an async kernel launch.
///
/// Wraps the output `Array` (private) together with a `CommandBufferHandle`.
/// The only way to access the output is via `into_array()`, which blocks
/// until the GPU command buffer completes. This prevents reading
/// incomplete GPU results at compile time.
pub struct LaunchResult {
    output: Array,
    handle: CommandBufferHandle,
}

impl std::fmt::Debug for LaunchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaunchResult")
            .field("complete", &self.handle.is_complete())
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl LaunchResult {
    /// Create a new `LaunchResult` from an output array and a completion handle.
    pub fn new(output: Array, handle: CommandBufferHandle) -> Self {
        Self { output, handle }
    }

    /// Check whether the GPU work has completed without blocking.
    pub fn is_complete(&self) -> bool {
        self.handle.is_complete()
    }

    /// Block until the GPU finishes, then return the output array.
    ///
    /// This is the **only** way to obtain the underlying `Array`.
    pub fn into_array(self) -> Array {
        self.handle.wait();
        self.output
    }

    /// Block until the GPU finishes or the timeout expires.
    ///
    /// Returns `Ok(Array)` on completion, `Err(self)` on timeout so the
    /// caller can retry or drop.
    pub fn into_array_timeout(self, timeout: Duration) -> Result<Array, Self> {
        if self.handle.wait_timeout(timeout) {
            Ok(self.output)
        } else {
            Err(self)
        }
    }

    /// Access the completion handle (e.g. for polling).
    pub fn handle(&self) -> &CommandBufferHandle {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_u32_valid() {
        assert_eq!(checked_u32(0, "x").unwrap(), 0u32);
        assert_eq!(checked_u32(42, "x").unwrap(), 42u32);
        assert_eq!(checked_u32(u32::MAX as usize, "x").unwrap(), u32::MAX);
    }

    #[test]
    fn test_checked_u32_overflow() {
        let val = u32::MAX as usize + 1;
        let result = checked_u32(val, "big_dim");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            KernelError::InvalidShape(msg) => {
                assert!(msg.contains("big_dim"), "error should contain name: {msg}");
                assert!(
                    msg.contains("exceeds u32::MAX"),
                    "error should mention overflow: {msg}"
                );
            }
            _ => panic!("expected InvalidShape, got {err:?}"),
        }
    }
}
