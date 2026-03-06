//! GPU capture / profiling support via MTLCaptureManager (M12).
//!
//! Enables GPU trace capture for debugging and profiling Metal workloads
//! with Xcode's GPU profiler or Metal System Trace.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rmlx_metal::capture::{CaptureScope, CaptureDestination};
//!
//! // Capture to a GPU trace file (viewable in Xcode):
//! let scope = CaptureScope::begin(device, CaptureDestination::GpuTraceFile("trace.gputrace"))?;
//! // ... Metal work ...
//! scope.end();
//!
//! // Capture to Xcode debugger (must be attached):
//! let scope = CaptureScope::begin(device, CaptureDestination::DeveloperTools)?;
//! // ... Metal work ...
//! scope.end();
//! ```
//!
//! Note: Requires the `MetalCaptureEnabled` Info.plist key or the
//! `METAL_CAPTURE_ENABLED=1` environment variable to be set.

use crate::MetalError;

/// Destination for GPU trace capture.
#[derive(Debug, Clone)]
pub enum CaptureDestination {
    /// Capture to Xcode GPU debugger (requires Xcode attached).
    DeveloperTools,
    /// Capture to a `.gputrace` file at the given path.
    GpuTraceFile(String),
}

/// RAII scope for GPU trace capture.
///
/// Begins capture on creation and ends it on drop (or explicit `end()` call).
/// Only one capture can be active at a time system-wide.
pub struct CaptureScope {
    /// Whether capture is currently active (to prevent double-end).
    active: bool,
}

impl CaptureScope {
    /// Begin a GPU trace capture on the given device.
    ///
    /// # Errors
    ///
    /// Returns `MetalError::PipelineCreate` if capture cannot be started
    /// (e.g., capture is already active, or the environment does not support it).
    pub fn begin(
        device: &metal::Device,
        destination: CaptureDestination,
    ) -> Result<Self, MetalError> {
        start_capture(device, &destination).map(|()| Self { active: true })
    }

    /// End the capture explicitly.
    ///
    /// This is also called automatically on drop, but explicit `end()` makes
    /// intent clearer and allows checking for errors.
    pub fn end(&mut self) {
        if self.active {
            stop_capture();
            self.active = false;
        }
    }

    /// Whether the capture is currently active.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        self.end();
    }
}

/// Check whether GPU capture is supported in the current environment.
///
/// Returns `true` if `MTLCaptureManager.sharedCaptureManager.supportsDestination`
/// indicates at least one destination is available.
pub fn is_capture_supported() -> bool {
    let manager = metal::CaptureManager::shared();
    manager.supports_destination(metal::MTLCaptureDestination::DeveloperTools)
}

// ---------------------------------------------------------------------------
// Implementation helpers using the `metal` crate's safe(r) bindings
// ---------------------------------------------------------------------------

/// Start a Metal GPU capture via the `metal` crate's CaptureManager API.
///
/// Uses `metal::CaptureManager`, `metal::CaptureDescriptor`, etc. instead of
/// raw `objc` FFI calls, avoiding pointer-cast and C-string memory corruption
/// bugs that existed in the previous implementation.
fn start_capture(
    device: &metal::Device,
    destination: &CaptureDestination,
) -> Result<(), MetalError> {
    let manager = metal::CaptureManager::shared();

    let descriptor = metal::CaptureDescriptor::new();
    descriptor.set_capture_device(device);

    match destination {
        CaptureDestination::DeveloperTools => {
            descriptor.set_destination(metal::MTLCaptureDestination::DeveloperTools);
        }
        CaptureDestination::GpuTraceFile(path) => {
            descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
            descriptor.set_output_url(path);
        }
    }

    manager.start_capture(&descriptor).map_err(|e| {
        MetalError::PipelineCreate(format!(
            "failed to start GPU capture ({destination:?}): {e}; \
             ensure METAL_CAPTURE_ENABLED=1 is set"
        ))
    })
}

/// Stop the active Metal GPU capture.
fn stop_capture() {
    let manager = metal::CaptureManager::shared();
    manager.stop_capture();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_destination_debug() {
        let d1 = CaptureDestination::DeveloperTools;
        let d2 = CaptureDestination::GpuTraceFile("test.gputrace".into());
        // Just verify Debug is implemented.
        let _ = format!("{d1:?}");
        let _ = format!("{d2:?}");
    }

    #[test]
    fn test_is_capture_supported() {
        // Just verify it does not panic. Result depends on environment.
        let _ = is_capture_supported();
    }

    #[test]
    fn test_capture_scope_inactive_end_is_noop() {
        // Verify that ending an already-inactive scope doesn't panic.
        let mut scope = CaptureScope { active: false };
        scope.end();
        assert!(!scope.is_active());
    }

    #[test]
    fn test_capture_scope_drop_inactive_is_safe() {
        // Verify that dropping an inactive scope doesn't panic.
        let scope = CaptureScope { active: false };
        drop(scope);
    }
}
