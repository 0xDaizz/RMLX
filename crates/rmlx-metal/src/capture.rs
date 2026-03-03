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
        // Use Objective-C runtime to access MTLCaptureManager.
        // The `metal` crate (v0.31) does not expose CaptureManager bindings,
        // so we use `objc` directly.
        //
        // SAFETY: These are well-known Metal framework APIs.
        let started = unsafe { start_capture(device, &destination) };
        if started {
            Ok(Self { active: true })
        } else {
            Err(MetalError::PipelineCreate(format!(
                "failed to start GPU capture ({destination:?}): \
                 ensure METAL_CAPTURE_ENABLED=1 is set"
            )))
        }
    }

    /// End the capture explicitly.
    ///
    /// This is also called automatically on drop, but explicit `end()` makes
    /// intent clearer and allows checking for errors.
    pub fn end(&mut self) {
        if self.active {
            unsafe {
                stop_capture();
            }
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
    unsafe { capture_supported_impl() }
}

// ---------------------------------------------------------------------------
// Obj-C runtime helpers
// ---------------------------------------------------------------------------

/// Start a Metal GPU capture via MTLCaptureManager.
///
/// # Safety
/// Calls Objective-C runtime APIs. Must be called on a thread with an
/// autorelease pool (or within a `ScopedPool`).
unsafe fn start_capture(device: &metal::Device, destination: &CaptureDestination) -> bool {
    use objc::runtime::{Object, BOOL, YES};

    let manager_cls = match objc::runtime::Class::get("MTLCaptureManager") {
        Some(cls) => cls,
        None => return false,
    };

    let manager: *mut Object = objc::msg_send![manager_cls, sharedCaptureManager];
    if manager.is_null() {
        return false;
    }

    // Create MTLCaptureDescriptor
    let desc_cls = match objc::runtime::Class::get("MTLCaptureDescriptor") {
        Some(cls) => cls,
        None => return false,
    };
    let desc: *mut Object = objc::msg_send![desc_cls, alloc];
    let desc: *mut Object = objc::msg_send![desc, init];

    // Set capture device
    let raw_device = device as *const metal::Device as *const Object;
    let _: () = objc::msg_send![desc, setCaptureObject: raw_device];

    // Set destination
    match destination {
        CaptureDestination::DeveloperTools => {
            // MTLCaptureDestinationDeveloperTools = 1
            let _: () = objc::msg_send![desc, setDestination: 1i64];
        }
        CaptureDestination::GpuTraceFile(path) => {
            // MTLCaptureDestinationGPUTraceDocument = 2
            let _: () = objc::msg_send![desc, setDestination: 2i64];

            // Set output URL
            let nsstring_cls = objc::runtime::Class::get("NSString").unwrap();
            let path_bytes = path.as_bytes();
            let ns_path: *mut Object = objc::msg_send![
                nsstring_cls,
                stringWithUTF8String: path_bytes.as_ptr()
            ];
            let nsurl_cls = objc::runtime::Class::get("NSURL").unwrap();
            let url: *mut Object = objc::msg_send![
                nsurl_cls,
                fileURLWithPath: ns_path
            ];
            let _: () = objc::msg_send![desc, setOutputURL: url];
        }
    }

    // Start capture with descriptor
    let mut error: *mut Object = std::ptr::null_mut();
    let result: BOOL = objc::msg_send![
        manager,
        startCaptureWithDescriptor: desc
        error: &mut error
    ];

    result == YES
}

/// Stop the active Metal GPU capture.
///
/// # Safety
/// Calls Objective-C runtime APIs.
unsafe fn stop_capture() {
    use objc::runtime::Object;

    let manager_cls = match objc::runtime::Class::get("MTLCaptureManager") {
        Some(cls) => cls,
        None => return,
    };

    let manager: *mut Object = objc::msg_send![manager_cls, sharedCaptureManager];
    if !manager.is_null() {
        let _: () = objc::msg_send![manager, stopCapture];
    }
}

/// Check if capture is supported.
///
/// # Safety
/// Calls Objective-C runtime APIs.
unsafe fn capture_supported_impl() -> bool {
    use objc::runtime::Object;

    let manager_cls = match objc::runtime::Class::get("MTLCaptureManager") {
        Some(cls) => cls,
        None => return false,
    };

    let manager: *mut Object = objc::msg_send![manager_cls, sharedCaptureManager];
    if manager.is_null() {
        return false;
    }

    // Check if developer tools destination is supported (destination = 1)
    let supported: bool = objc::msg_send![manager, supportsDestination: 1i64];
    supported
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
}
