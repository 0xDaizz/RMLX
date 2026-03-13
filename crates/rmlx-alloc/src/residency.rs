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
//! When the `metal3` feature is enabled, this module uses the objc2-metal
//! `MTLResidencySet` bindings to manage residency sets via typed APIs.
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
// metal3-enabled implementation: real MTLResidencySet via objc2-metal bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "metal3")]
mod inner {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLAllocation, MTLBuffer, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
    };

    use super::ResidencyError;

    /// Manages a set of Metal buffers that should be made resident on the GPU.
    ///
    /// On Metal 3+ devices, this uses `MTLResidencySet` to batch residency
    /// operations. The `Retained` handle is automatically released on drop.
    pub struct ResidencyManager {
        /// Typed handle to the `MTLResidencySet` Objective-C object.
        residency_set: objc2::rc::Retained<ProtocolObject<dyn MTLResidencySet>>,
        /// Number of buffers tracked (for diagnostics).
        buffer_count: usize,
        /// Whether changes have been made since the last commit.
        dirty: bool,
    }

    /// Upcast a `&ProtocolObject<dyn MTLBuffer>` to `&ProtocolObject<dyn MTLAllocation>`.
    ///
    /// MTLBuffer inherits MTLResource which inherits MTLAllocation, so this
    /// is a safe protocol upcast via pointer identity.
    fn as_allocation(
        buf: &ProtocolObject<dyn MTLBuffer>,
    ) -> &ProtocolObject<dyn MTLAllocation> {
        let ptr: *const ProtocolObject<dyn MTLBuffer> = buf;
        // SAFETY: MTLBuffer : MTLResource : MTLAllocation — the ObjC object
        // behind the protocol pointer implements MTLAllocation. The pointer
        // representation is identical (thin pointer to the same ObjC object).
        unsafe { &*(ptr as *const ProtocolObject<dyn MTLAllocation>) }
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
        /// support Metal 3 residency sets.
        pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, ResidencyError> {
            let desc = MTLResidencySetDescriptor::new();
            let residency_set =
                device.newResidencySetWithDescriptor_error(&desc).map_err(|e| {
                    ResidencyError::CreationFailed(format!(
                        "device may not support Metal 3: {e}"
                    ))
                })?;

            Ok(Self {
                residency_set,
                buffer_count: 0,
                dirty: false,
            })
        }

        /// Add a buffer to the residency set.
        ///
        /// Calls `[residencySet addAllocation:]` on the buffer.
        pub fn add_buffer(&mut self, buffer: &ProtocolObject<dyn MTLBuffer>) {
            self.residency_set.addAllocation(as_allocation(buffer));
            self.buffer_count += 1;
            self.dirty = true;
        }

        /// Remove a buffer from the residency set.
        ///
        /// Calls `[residencySet removeAllocation:]` on the buffer.
        pub fn remove_buffer(&mut self, buffer: &ProtocolObject<dyn MTLBuffer>) {
            self.residency_set.removeAllocation(as_allocation(buffer));
            self.buffer_count = self.buffer_count.saturating_sub(1);
            self.dirty = true;
        }

        /// Commit pending residency set changes to the GPU.
        ///
        /// Calls `[residencySet commit]` to apply additions and removals
        /// to the GPU's residency table.
        pub fn commit(&mut self) {
            self.residency_set.commit();
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
        pub fn ensure_resident(&mut self, buffers: &[&ProtocolObject<dyn MTLBuffer>]) {
            for buf in buffers {
                self.add_buffer(buf);
            }
            if self.is_dirty() {
                self.commit();
            }
        }
    }

    // Retained<ProtocolObject<dyn MTLResidencySet>> is automatically released
    // on drop — no manual Drop impl needed.

    // The residency set is an Objective-C object that is not thread-safe.
    // We explicitly do NOT impl Send or Sync.
}

// ---------------------------------------------------------------------------
// Non-metal3 implementation: no-op stubs with counter tracking
// ---------------------------------------------------------------------------

#[cfg(not(feature = "metal3"))]
mod inner {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLBuffer, MTLDevice};

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
        pub fn new(_device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, ResidencyError> {
            Ok(Self {
                buffer_count: 0,
                dirty: false,
            })
        }

        /// Add a buffer to the residency set (no-op, tracks count only).
        pub fn add_buffer(&mut self, _buffer: &ProtocolObject<dyn MTLBuffer>) {
            self.buffer_count += 1;
            self.dirty = true;
        }

        /// Remove a buffer from the residency set (no-op, tracks count only).
        pub fn remove_buffer(&mut self, _buffer: &ProtocolObject<dyn MTLBuffer>) {
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
        pub fn ensure_resident(&mut self, buffers: &[&ProtocolObject<dyn MTLBuffer>]) {
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
        use rmlx_metal::MTLResourceOptions;

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
        use rmlx_metal::MTLResourceOptions;

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
        use rmlx_metal::MTLResourceOptions;

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
        use rmlx_metal::MTLResourceOptions;

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
