//! Metal 3+ residency set management (A6).
//!
//! Wraps the Metal 3 residency set API (`MTLResidencySet`) to ensure buffers
//! are resident in GPU memory before use. This reduces page faults on Apple
//! Silicon GPUs that support Metal 3 (M3 and later).
//!
//! Feature-gated behind `metal3` since older devices (M1, M2) do not support
//! the residency set API.
//!
//! # Current status
//!
//! The `metal` Rust crate does not yet expose `MTLResidencySet` bindings.
//! This module provides stub implementations with the correct interface so
//! that callers can be written against it today. Once upstream bindings land,
//! the stubs should be replaced with real Metal API calls.
//!
//! TODO(metal3): Replace stubs with real `MTLResidencySet` calls when the
//! `metal` crate exposes them (tracked upstream).

/// Manages a set of Metal buffers that should be made resident on the GPU.
///
/// On Metal 3+ devices, this uses `MTLResidencySet` to batch residency
/// operations. On older devices or when the `metal3` feature is not enabled,
/// this is a no-op.
pub struct ResidencyManager {
    /// Number of buffers tracked (for diagnostics).
    buffer_count: usize,
    /// Whether changes have been made since the last commit.
    dirty: bool,
}

impl ResidencyManager {
    /// Placeholder for Metal 3 residency-set backend wiring.
    ///
    /// The `metal3` feature indicates intent to use the real API, so we keep a
    /// `todo!()` path to prevent silent no-op behavior under that feature.
    #[cfg(feature = "metal3")]
    #[inline(always)]
    fn todo_metal3_backend() -> ! {
        todo!(
            "TODO(A6): Implement real MTLResidencySet backend in rmlx-alloc \
             once metal-rs exposes the required bindings"
        )
    }

    /// Create a new residency manager.
    ///
    /// TODO(metal3): Accept `&metal::Device` and create an `MTLResidencySet`
    /// via `device.newResidencySet()` once bindings are available.
    pub fn new() -> Self {
        #[cfg(feature = "metal3")]
        Self::todo_metal3_backend();

        Self {
            buffer_count: 0,
            dirty: false,
        }
    }

    /// Add a buffer to the residency set.
    ///
    /// TODO(metal3): Call `residency_set.addAllocation(buffer)` on the real
    /// Metal residency set.
    pub fn add_buffer(&mut self, _buffer: &rmlx_metal::metal::Buffer) {
        #[cfg(feature = "metal3")]
        Self::todo_metal3_backend();

        self.buffer_count += 1;
        self.dirty = true;
    }

    /// Remove a buffer from the residency set.
    ///
    /// TODO(metal3): Call `residency_set.removeAllocation(buffer)` on the
    /// real Metal residency set.
    pub fn remove_buffer(&mut self, _buffer: &rmlx_metal::metal::Buffer) {
        #[cfg(feature = "metal3")]
        Self::todo_metal3_backend();

        self.buffer_count = self.buffer_count.saturating_sub(1);
        self.dirty = true;
    }

    /// Commit pending residency set changes to the GPU.
    ///
    /// TODO(metal3): Call `residency_set.commit()` to apply additions and
    /// removals to the GPU's residency table.
    pub fn commit(&mut self) {
        #[cfg(feature = "metal3")]
        Self::todo_metal3_backend();

        // Stub: no-op until MTLResidencySet bindings are available.
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
}

impl Default for ResidencyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residency_manager_stub_lifecycle() {
        let mut mgr = ResidencyManager::new();
        assert_eq!(mgr.buffer_count(), 0);
        assert!(!mgr.is_dirty());

        // We can't easily create a real Metal buffer in a unit test without
        // a device, so we test the bookkeeping logic with the stub path.
        // The add/remove/commit methods are no-ops on the Metal side, but
        // the counter tracking should work.
        mgr.buffer_count += 1; // simulate add
        mgr.dirty = true;
        assert_eq!(mgr.buffer_count(), 1);
        assert!(mgr.is_dirty());

        mgr.commit();
        assert!(!mgr.is_dirty());

        mgr.buffer_count = mgr.buffer_count.saturating_sub(1); // simulate remove
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

        let mut mgr = ResidencyManager::new();
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
}
