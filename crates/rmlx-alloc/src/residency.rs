//! Metal 3+ residency set management (A6).
//!
//! Wraps the Metal 3 residency set API (`MTLResidencySet`) to ensure buffers
//! are resident in GPU memory before use. This reduces page faults on Apple
//! Silicon GPUs that support Metal 3 (M3 and later).
//!
//! Feature-gated behind `metal3` since older devices (M1, M2) do not support
//! the residency set API.
//!
//! # Implementation
//!
//! When the `metal3` feature is enabled, this module uses `objc::msg_send!`
//! to call the Metal 3 `MTLResidencySet` API directly, since the `metal`
//! Rust crate does not yet expose these bindings.
//!
//! When `metal3` is not enabled, a no-op stub is provided that only tracks
//! buffer counts for diagnostics.

use std::fmt;

/// Error type for residency set operations.
#[derive(Debug)]
pub enum ResidencyError {
    /// The device does not support Metal 3 residency sets.
    CreationFailed(String),
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreationFailed(msg) => write!(f, "residency set creation failed: {msg}"),
        }
    }
}

impl std::error::Error for ResidencyError {}

// ---------------------------------------------------------------------------
// metal3-enabled implementation: real MTLResidencySet via objc runtime
// ---------------------------------------------------------------------------

#[cfg(feature = "metal3")]
mod inner {
    use objc::runtime::Object;
    use rmlx_metal::metal;
    use rmlx_metal::metal::foreign_types::ForeignType;

    use super::ResidencyError;

    /// Manages a set of Metal buffers that should be made resident on the GPU.
    ///
    /// On Metal 3+ devices, this uses `MTLResidencySet` to batch residency
    /// operations. The underlying Objective-C object is created from the
    /// device and released on drop.
    pub struct ResidencyManager {
        /// Raw pointer to the `MTLResidencySet` Objective-C object.
        residency_set: *mut Object,
        /// Number of buffers tracked (for diagnostics).
        buffer_count: usize,
        /// Whether changes have been made since the last commit.
        dirty: bool,
    }

    impl ResidencyManager {
        /// Create a new residency manager backed by a real `MTLResidencySet`.
        ///
        /// Creates an `MTLResidencySetDescriptor`, then calls
        /// `[device newResidencySetWithDescriptor:error:]` to obtain the
        /// residency set.
        ///
        /// # Errors
        ///
        /// Returns `ResidencyError::CreationFailed` if the device does not
        /// support Metal 3 residency sets (i.e., the returned object is null).
        pub fn new(device: &metal::Device) -> Result<Self, ResidencyError> {
            // SAFETY: We are calling well-known Metal 3 Objective-C APIs.
            // MTLResidencySetDescriptor is a simple descriptor class, and
            // newResidencySetWithDescriptor:error: is the standard factory.
            // We use device.as_ptr() (from ForeignType) to get the real
            // ObjC pointer, rather than casting the Rust wrapper address.
            let residency_set: *mut Object = unsafe {
                let desc: *mut Object = msg_send![class!(MTLResidencySetDescriptor), new];

                let raw_device: *mut Object = device.as_ptr() as *mut Object;
                let null_ptr: *mut Object = std::ptr::null_mut();
                let set: *mut Object = msg_send![
                    raw_device,
                    newResidencySetWithDescriptor: desc
                    error: null_ptr
                ];

                // Release the descriptor; the residency set retains what it needs.
                let _: () = msg_send![desc, release];

                set
            };

            if residency_set.is_null() {
                return Err(ResidencyError::CreationFailed(
                    "device may not support Metal 3".to_string(),
                ));
            }

            Ok(Self {
                residency_set,
                buffer_count: 0,
                dirty: false,
            })
        }

        /// Add a buffer to the residency set.
        ///
        /// Calls `[residencySet addAllocation:]` with the raw buffer pointer.
        pub fn add_buffer(&mut self, buffer: &metal::Buffer) {
            // SAFETY: addAllocation: is a Metal 3 MTLResidencySet method.
            // We use buffer.as_ptr() (from ForeignType) to get the real
            // ObjC pointer.
            unsafe {
                let raw_buf: *mut Object = buffer.as_ptr() as *mut Object;
                let _: () = msg_send![self.residency_set, addAllocation: raw_buf];
            }
            self.buffer_count += 1;
            self.dirty = true;
        }

        /// Remove a buffer from the residency set.
        ///
        /// Calls `[residencySet removeAllocation:]` with the raw buffer pointer.
        pub fn remove_buffer(&mut self, buffer: &metal::Buffer) {
            // SAFETY: removeAllocation: is a Metal 3 MTLResidencySet method.
            unsafe {
                let raw_buf: *mut Object = buffer.as_ptr() as *mut Object;
                let _: () = msg_send![self.residency_set, removeAllocation: raw_buf];
            }
            self.buffer_count = self.buffer_count.saturating_sub(1);
            self.dirty = true;
        }

        /// Commit pending residency set changes to the GPU.
        ///
        /// Calls `[residencySet commit]` to apply additions and removals
        /// to the GPU's residency table.
        pub fn commit(&mut self) {
            // SAFETY: commit is a Metal 3 MTLResidencySet method.
            unsafe {
                let _: () = msg_send![self.residency_set, commit];
            }
            self.dirty = false;
        }

        /// Number of buffers currently in the residency set.
        pub fn buffer_count(&self) -> usize {
            self.buffer_count
        }

        /// Whether there are uncommitted changes.
        pub fn is_dirty(&self) -> bool {
            self.dirty
        }

        /// Batch-register multiple buffers and commit in one operation.
        ///
        /// Convenience method for ensuring a set of weight and KV cache
        /// buffers are resident before starting a decode loop.
        pub fn ensure_resident(&mut self, buffers: &[&metal::Buffer]) {
            for buf in buffers {
                self.add_buffer(buf);
            }
            if self.is_dirty() {
                self.commit();
            }
        }
    }

    impl Drop for ResidencyManager {
        fn drop(&mut self) {
            // SAFETY: We own the residency set and release it exactly once.
            if !self.residency_set.is_null() {
                unsafe {
                    let _: () = msg_send![self.residency_set, release];
                }
            }
        }
    }

    // The residency set is an Objective-C object that is not thread-safe.
    // We explicitly do NOT impl Send or Sync.
}

// ---------------------------------------------------------------------------
// Non-metal3 implementation: no-op stubs with counter tracking
// ---------------------------------------------------------------------------

#[cfg(not(feature = "metal3"))]
mod inner {
    use rmlx_metal::metal;

    use super::ResidencyError;

    /// Manages a set of Metal buffers that should be made resident on the GPU.
    ///
    /// When the `metal3` feature is not enabled, this is a no-op stub that
    /// only tracks buffer counts for diagnostics.
    pub struct ResidencyManager {
        /// Number of buffers tracked (for diagnostics).
        buffer_count: usize,
        /// Whether changes have been made since the last commit.
        dirty: bool,
    }

    impl ResidencyManager {
        /// Create a new residency manager (no-op without `metal3` feature).
        ///
        /// The `device` parameter is accepted for API uniformity with the
        /// `metal3` path but is not used.
        ///
        /// # Errors
        ///
        /// This stub never fails, but returns `Result` for API consistency
        /// with the `metal3` path.
        pub fn new(_device: &metal::Device) -> Result<Self, ResidencyError> {
            Ok(Self {
                buffer_count: 0,
                dirty: false,
            })
        }

        /// Add a buffer to the residency set (no-op, tracks count only).
        pub fn add_buffer(&mut self, _buffer: &metal::Buffer) {
            self.buffer_count += 1;
            self.dirty = true;
        }

        /// Remove a buffer from the residency set (no-op, tracks count only).
        pub fn remove_buffer(&mut self, _buffer: &metal::Buffer) {
            self.buffer_count = self.buffer_count.saturating_sub(1);
            self.dirty = true;
        }

        /// Commit pending residency set changes (no-op).
        pub fn commit(&mut self) {
            self.dirty = false;
        }

        /// Number of buffers currently in the residency set.
        pub fn buffer_count(&self) -> usize {
            self.buffer_count
        }

        /// Whether there are uncommitted changes.
        pub fn is_dirty(&self) -> bool {
            self.dirty
        }

        /// Batch-register multiple buffers and commit in one operation.
        ///
        /// Convenience method for ensuring a set of weight and KV cache
        /// buffers are resident before starting a decode loop.
        pub fn ensure_resident(&mut self, buffers: &[&metal::Buffer]) {
            for buf in buffers {
                self.add_buffer(buf);
            }
            if self.is_dirty() {
                self.commit();
            }
        }
    }
}

pub use inner::ResidencyManager;

#[cfg(test)]
#[cfg(not(feature = "metal3"))]
mod tests {
    use super::*;

    /// Test the non-metal3 stub path: counter tracking and dirty flag.
    ///
    /// This test always runs (even without `metal3`) because it exercises
    /// the bookkeeping logic directly.
    #[test]
    fn test_residency_metal3_stub() {
        use rmlx_metal::device::GpuDevice;
        use rmlx_metal::metal::MTLResourceOptions;

        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut mgr = ResidencyManager::new(device.raw()).expect("stub new should not fail");
        assert_eq!(mgr.buffer_count(), 0);
        assert!(!mgr.is_dirty());

        let buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

        mgr.add_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 1);
        assert!(mgr.is_dirty());

        mgr.add_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 2);

        mgr.commit();
        assert!(!mgr.is_dirty());

        mgr.remove_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 1);
        assert!(mgr.is_dirty());

        mgr.remove_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 0);

        // Saturating sub should not underflow
        mgr.remove_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 0);

        mgr.commit();
        assert!(!mgr.is_dirty());
    }

    #[test]
    fn test_residency_manager_stub_lifecycle() {
        use rmlx_metal::device::GpuDevice;
        use rmlx_metal::metal::MTLResourceOptions;

        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut mgr = ResidencyManager::new(device.raw()).expect("stub new should not fail");
        assert_eq!(mgr.buffer_count(), 0);
        assert!(!mgr.is_dirty());

        let buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

        mgr.add_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 1);
        assert!(mgr.is_dirty());

        mgr.commit();
        assert!(!mgr.is_dirty());

        mgr.remove_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 0);
    }

    #[test]
    fn test_residency_manager_with_device() {
        use rmlx_metal::device::GpuDevice;
        use rmlx_metal::metal::MTLResourceOptions;

        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut mgr = ResidencyManager::new(device.raw()).expect("stub new should not fail");
        let buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

        mgr.add_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 1);
        assert!(mgr.is_dirty());

        mgr.commit();
        assert!(!mgr.is_dirty());

        mgr.remove_buffer(&buf);
        assert_eq!(mgr.buffer_count(), 0);
        assert!(mgr.is_dirty());

        mgr.commit();
        assert!(!mgr.is_dirty());
    }

    #[test]
    fn test_residency_error_display() {
        let err = ResidencyError::CreationFailed("test error".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("test error"));
        assert!(msg.contains("residency set creation failed"));
    }

    #[test]
    fn test_residency_batch_ensure_resident() {
        use rmlx_metal::device::GpuDevice;
        use rmlx_metal::metal::MTLResourceOptions;

        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut mgr = ResidencyManager::new(device.raw()).expect("stub new should not fail");
        let buf1 = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
        let buf2 = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
        let buf3 = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

        mgr.ensure_resident(&[&buf1, &buf2, &buf3]);
        assert_eq!(mgr.buffer_count(), 3);
        assert!(!mgr.is_dirty(), "should be committed after ensure_resident");
    }

    #[test]
    fn test_residency_ensure_resident_empty() {
        use rmlx_metal::device::GpuDevice;

        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut mgr = ResidencyManager::new(device.raw()).expect("stub new should not fail");
        mgr.ensure_resident(&[]);
        assert_eq!(mgr.buffer_count(), 0);
        assert!(!mgr.is_dirty());
    }
}
