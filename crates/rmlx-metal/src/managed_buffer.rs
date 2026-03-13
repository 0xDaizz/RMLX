//! Managed buffer integration for cache-aware allocation.
//!
//! Provides [`ManagedBuffer`], a RAII wrapper around a Metal buffer that
//! returns the buffer to an allocator on drop.  This bridges `rmlx-metal`'s
//! buffer usage with `rmlx-alloc`'s [`BufferAllocator`] cache layer.
//!
//! The [`BufferAllocator`] trait is defined here so that `rmlx-alloc`'s
//! `MetalAllocator` can implement it without creating a circular dependency.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::sync::Arc;

use crate::types::MtlBuffer;

/// Trait for a cache-aware buffer allocator.
///
/// This is implemented by `rmlx-alloc::MetalAllocator` to allow
/// `rmlx-metal` code to allocate and return buffers through the cache
/// without a direct crate dependency.
pub trait BufferAllocator: Send + Sync {
    /// Allocate a Metal buffer of at least `size` bytes.
    ///
    /// The allocator may return a cached buffer larger than `size`.
    /// Returns an error string on failure.
    fn alloc(&self, size: usize, options: MTLResourceOptions) -> Result<MtlBuffer, String>;

    /// Return a buffer to the allocator's cache for reuse.
    ///
    /// The buffer should not be used after this call.
    fn free(&self, buffer: MtlBuffer);
}

/// RAII wrapper that returns its buffer to the allocator on drop.
///
/// Wraps a Metal buffer together with a reference to a [`BufferAllocator`].
/// When the `ManagedBuffer` is dropped, the buffer is automatically returned
/// to the allocator's cache instead of being deallocated.
///
/// For buffers managed by the barrier tracker (M2), the buffer is created
/// with `HazardTrackingModeUntracked` so that Metal does not perform
/// redundant hardware-level hazard tracking.
pub struct ManagedBuffer {
    buffer: Option<MtlBuffer>,
    allocator: Arc<dyn BufferAllocator>,
}

impl ManagedBuffer {
    /// Allocate a new managed buffer from the given allocator.
    ///
    /// Uses `StorageModeShared` by default.  The caller can combine this
    /// with `HazardTrackingModeUntracked` via the `options` parameter when
    /// the buffer is managed by the barrier tracker.
    pub fn alloc(
        allocator: Arc<dyn BufferAllocator>,
        size: usize,
        options: MTLResourceOptions,
    ) -> Result<Self, String> {
        let buffer = allocator.alloc(size, options)?;
        Ok(Self {
            buffer: Some(buffer),
            allocator,
        })
    }

    /// Allocate with untracked hazard mode for barrier-tracker-managed buffers.
    ///
    /// Equivalent to calling `alloc` with
    /// `StorageModeShared | HazardTrackingModeUntracked`.
    pub fn alloc_untracked(
        allocator: Arc<dyn BufferAllocator>,
        size: usize,
    ) -> Result<Self, String> {
        Self::alloc(allocator, size, crate::device::UNTRACKED_BUFFER_OPTIONS)
    }

    /// Access the underlying Metal buffer.
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.buffer
            .as_ref()
            .expect("ManagedBuffer accessed after drop")
    }

    /// Buffer length in bytes.
    pub fn length(&self) -> u64 {
        self.buffer().length() as u64
    }

    /// GPU address of the buffer.
    pub fn gpu_address(&self) -> u64 {
        self.buffer().gpuAddress()
    }

    /// Get the raw buffer contents pointer.
    ///
    /// # Safety
    /// - The buffer must use `StorageModeShared`.
    /// - No GPU writes may be in-flight.
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.buffer().contents().as_ptr()
    }

    /// Detach the buffer from automatic recycling and return it.
    ///
    /// After calling this, the buffer will NOT be returned to the allocator
    /// on drop.  The caller takes ownership.
    pub fn take(mut self) -> MtlBuffer {
        self.buffer
            .take()
            .expect("ManagedBuffer::take called after drop")
    }
}

impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.allocator.free(buffer);
        }
    }
}

// Provide Deref so callers can use buffer methods directly.
impl std::ops::Deref for ManagedBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::types::MtlDevice;

    /// A simple test allocator that tracks alloc/free counts.
    struct TestAllocator {
        device: MtlDevice,
        alloc_count: AtomicUsize,
        free_count: AtomicUsize,
    }

    impl TestAllocator {
        fn new(device: MtlDevice) -> Self {
            Self {
                device,
                alloc_count: AtomicUsize::new(0),
                free_count: AtomicUsize::new(0),
            }
        }
    }

    impl BufferAllocator for TestAllocator {
        fn alloc(&self, size: usize, options: MTLResourceOptions) -> Result<MtlBuffer, String> {
            self.alloc_count.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .device
                .newBufferWithLength_options(size, options)
                .unwrap())
        }

        fn free(&self, _buffer: MtlBuffer) {
            self.free_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_managed_buffer_returns_on_drop() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.unwrap();
        let allocator = Arc::new(TestAllocator::new(device));

        {
            let buf = ManagedBuffer::alloc(
                Arc::clone(&allocator) as Arc<dyn BufferAllocator>,
                1024,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
            assert!(buf.length() >= 1024);
            assert_eq!(allocator.alloc_count.load(Ordering::SeqCst), 1);
            assert_eq!(allocator.free_count.load(Ordering::SeqCst), 0);
        }
        // After drop, free should have been called.
        assert_eq!(allocator.free_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_managed_buffer_take_prevents_free() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.unwrap();
        let allocator = Arc::new(TestAllocator::new(device));

        let buf = ManagedBuffer::alloc(
            Arc::clone(&allocator) as Arc<dyn BufferAllocator>,
            512,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();

        let _raw = buf.take();
        // free should NOT be called since we took ownership.
        assert_eq!(allocator.free_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_managed_buffer_untracked() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.unwrap();
        let allocator = Arc::new(TestAllocator::new(device));

        let buf =
            ManagedBuffer::alloc_untracked(Arc::clone(&allocator) as Arc<dyn BufferAllocator>, 256)
                .unwrap();
        assert!(buf.length() >= 256);
        drop(buf);
        assert_eq!(allocator.free_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_managed_buffer_deref() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.unwrap();
        let allocator = Arc::new(TestAllocator::new(device));

        let buf = ManagedBuffer::alloc(
            Arc::clone(&allocator) as Arc<dyn BufferAllocator>,
            2048,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap();

        // Deref should give us access to MTLBuffer methods.
        let _addr = buf.gpuAddress();
        assert!(buf.length() >= 2048);
    }
}
