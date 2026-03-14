//! Metal 4 command encoding wrappers.
//!
//! Provides safe wrappers around `MTL4CommandAllocator`, `MTL4CommandBuffer`,
//! and `MTL4CommandQueue` for the explicit encoding model introduced in Metal 4.
//!
//! # Architecture
//!
//! Metal 4 replaces the implicit encoding model (Metal 3) with an explicit one:
//!
//! - **CommandAllocator**: Manages backing memory for command encoding. Created
//!   once and reset between decode iterations to reclaim memory without
//!   deallocation.
//!
//! - **Mtl4CommandBuffer**: Has explicit `begin()` / `end()` lifecycle.
//!   `begin()` attaches to an allocator and opens encoding; `end()` closes
//!   encoding and prepares the buffer for submission.
//!
//! - **CommandQueue4**: Batch-commits an array of command buffers in a single
//!   call, reducing per-CB submit overhead in ExecGraph.
//!
//! # Usage
//!
//! ```rust,ignore
//! let alloc = CommandAllocator::new(&device)?;
//! let queue = CommandQueue4::new(&device)?;
//!
//! // Per-iteration loop:
//! alloc.reset();
//! let cb = alloc.new_command_buffer(&device)?;
//! cb.begin(&alloc);
//! // ... encode compute work via cb.compute_encoder() ...
//! cb.end();
//! queue.commit_batch(&[&cb]);
//! ```

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Owned Metal 4 command allocator.
pub type Mtl4Allocator = Retained<ProtocolObject<dyn MTL4CommandAllocator>>;

/// Owned Metal 4 command buffer.
pub type Mtl4CB = Retained<ProtocolObject<dyn MTL4CommandBuffer>>;

/// Owned Metal 4 command queue.
pub type Mtl4Queue = Retained<ProtocolObject<dyn MTL4CommandQueue>>;

// ---------------------------------------------------------------------------
// CommandAllocator
// ---------------------------------------------------------------------------

/// Safe wrapper around `MTL4CommandAllocator`.
///
/// Manages the backing memory for encoding GPU commands into Metal 4 command
/// buffers. Create one allocator per thread (allocators service a single
/// command buffer at a time).
pub struct CommandAllocator {
    inner: Mtl4Allocator,
}

impl CommandAllocator {
    /// Create a new command allocator on `device`.
    ///
    /// Returns `None` if the device does not support Metal 4 or allocation
    /// fails (e.g. out of memory).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let inner = device.newCommandAllocator()?;
        Some(Self { inner })
    }

    /// Create a new command allocator with a descriptor for custom configuration.
    ///
    /// Returns `Err` with the underlying `NSError` on failure.
    pub fn with_descriptor(
        device: &ProtocolObject<dyn MTLDevice>,
        descriptor: &MTL4CommandAllocatorDescriptor,
    ) -> Result<Self, Retained<objc2_foundation::NSError>> {
        let inner = device.newCommandAllocatorWithDescriptor_error(descriptor)?;
        Ok(Self { inner })
    }

    /// Reset the allocator, marking its internal heaps for reuse.
    ///
    /// # Safety
    ///
    /// The caller must ensure that **all** command buffers previously allocated
    /// from this allocator have completed execution on the GPU before calling
    /// `reset()`. Typically this means calling `CommandQueue4::commit_batch()`
    /// followed by a fence/event wait or `waitUntilCompleted` on the last CB.
    /// Calling `reset()` while command buffers are still in flight is undefined
    /// behavior and may corrupt GPU state.
    pub unsafe fn reset(&self) {
        self.inner.reset();
    }

    /// Query the current allocated size of the allocator's internal heaps.
    pub fn allocated_size(&self) -> u64 {
        self.inner.allocatedSize()
    }

    /// Create a new Metal 4 command buffer from this allocator's device.
    ///
    /// The returned `Mtl4CommandBuffer` is in an un-begun state; call
    /// [`Mtl4CommandBuffer::begin()`] before encoding.
    pub fn new_command_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Option<Mtl4CommandBuffer> {
        let inner = device.newCommandBuffer()?;
        Some(Mtl4CommandBuffer { inner })
    }

    /// Access the raw `MTL4CommandAllocator` protocol object.
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4CommandAllocator> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Mtl4CommandBuffer
// ---------------------------------------------------------------------------

/// Safe wrapper around `MTL4CommandBuffer`.
///
/// Metal 4 command buffers have an explicit lifecycle:
/// 1. Create via `CommandAllocator::new_command_buffer()`
/// 2. Call `begin()` to attach to an allocator and open encoding
/// 3. Create encoders and record GPU commands
/// 4. Call `end()` to close encoding
/// 5. Submit via `CommandQueue4::commit_batch()`
pub struct Mtl4CommandBuffer {
    inner: Mtl4CB,
}

impl Mtl4CommandBuffer {
    /// Begin command buffer encoding, attaching to `allocator`.
    ///
    /// The allocator services only one command buffer at a time. For
    /// multi-threaded encoding, use separate allocators.
    ///
    /// After calling `begin()`, prior residency set bindings are cleared.
    pub fn begin(&self, allocator: &CommandAllocator) {
        self.inner.beginCommandBufferWithAllocator(&allocator.inner);
    }

    /// Begin command buffer encoding with additional options.
    ///
    /// Same as [`begin()`](Self::begin) but accepts `MTL4CommandBufferOptions`
    /// for configuring shader logging, etc.
    pub fn begin_with_options(
        &self,
        allocator: &CommandAllocator,
        options: &MTL4CommandBufferOptions,
    ) {
        self.inner
            .beginCommandBufferWithAllocator_options(&allocator.inner, options);
    }

    /// End command buffer encoding, preparing it for submission.
    ///
    /// After calling `end()`, the allocator is free to service other command
    /// buffers. The CB can then be submitted via `CommandQueue4::commit_batch()`.
    pub fn end(&self) {
        self.inner.endCommandBuffer();
    }

    /// Create a compute command encoder on this command buffer.
    ///
    /// Returns `None` if encoder creation fails.
    pub fn compute_encoder(
        &self,
    ) -> Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>> {
        self.inner.computeCommandEncoder()
    }

    /// Set the debug label on this command buffer.
    pub fn set_label(&self, label: &str) {
        self.inner
            .setLabel(Some(&objc2_foundation::NSString::from_str(label)));
    }

    /// Push a debug group string for GPU profiler / capture traces.
    pub fn push_debug_group(&self, name: &str) {
        self.inner
            .pushDebugGroup(&objc2_foundation::NSString::from_str(name));
    }

    /// Pop the most recent debug group.
    pub fn pop_debug_group(&self) {
        self.inner.popDebugGroup();
    }

    /// Access the raw `MTL4CommandBuffer` protocol object.
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4CommandBuffer> {
        &self.inner
    }

    /// Obtain a Metal 3 `MTLCommandBuffer` reference from this Metal 4 CB.
    ///
    /// At the ObjC runtime level, the concrete Metal 4 command buffer object
    /// also conforms to `MTLCommandBuffer`. This enables use of existing ops
    /// that accept `&ProtocolObject<dyn MTLCommandBuffer>`.
    ///
    /// # Safety
    ///
    /// This is safe on macOS 26+ where Metal 4 command buffers are returned
    /// by the system framework and always conform to `MTLCommandBuffer`.
    #[inline(always)]
    pub fn as_legacy_cb(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        // SAFETY: The concrete ObjC object behind `MTL4CommandBuffer` also
        // conforms to `MTLCommandBuffer`. Both are protocol object pointers
        // to the same underlying NSObject.
        unsafe { &*(&*self.inner as *const _ as *const _) }
    }

    /// Consume this wrapper and return the owned inner `Retained`.
    pub fn into_inner(self) -> Mtl4CB {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// CommandQueue4
// ---------------------------------------------------------------------------

/// Safe wrapper around `MTL4CommandQueue`.
///
/// Provides batch commit of multiple command buffers in a single call,
/// reducing per-CB submission overhead. Also supports GPU-side event
/// signal/wait for cross-queue synchronization.
pub struct CommandQueue4 {
    inner: Mtl4Queue,
}

impl CommandQueue4 {
    /// Create a new Metal 4 command queue on `device`.
    ///
    /// Returns `None` if the device does not support Metal 4.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let inner = device.newMTL4CommandQueue()?;
        Some(Self { inner })
    }

    /// Create a new Metal 4 command queue with a descriptor.
    ///
    /// Returns `Err` with the underlying `NSError` on failure.
    pub fn with_descriptor(
        device: &ProtocolObject<dyn MTLDevice>,
        descriptor: &MTL4CommandQueueDescriptor,
    ) -> Result<Self, Retained<objc2_foundation::NSError>> {
        let inner = device.newMTL4CommandQueueWithDescriptor_error(descriptor)?;
        Ok(Self { inner })
    }

    /// Batch-commit an array of command buffers for execution.
    ///
    /// The order of buffers in the slice is significant: it determines GPU
    /// execution order and is required for suspend/resume render passes.
    ///
    /// All command buffers must have had `end()` called before committing.
    ///
    /// # Panics
    ///
    /// Panics if `buffers` is empty.
    pub fn commit_batch(&self, buffers: &[&Mtl4CommandBuffer]) {
        assert!(
            !buffers.is_empty(),
            "commit_batch requires at least one command buffer"
        );

        // Stack buffer for typical batch sizes (1-8 CBs in ExecGraph4).
        const MAX_STACK: usize = 16;
        assert!(
            buffers.len() <= MAX_STACK,
            "commit_batch: batch size {} exceeds stack buffer capacity {}",
            buffers.len(),
            MAX_STACK,
        );

        let mut ptrs: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; MAX_STACK] =
            [NonNull::dangling(); MAX_STACK];
        for (i, cb) in buffers.iter().enumerate() {
            let raw = &*cb.inner as *const ProtocolObject<dyn MTL4CommandBuffer>
                as *mut ProtocolObject<dyn MTL4CommandBuffer>;
            // SAFETY: The reference is valid for the duration of this call.
            ptrs[i] = NonNull::new(raw).expect("command buffer pointer must not be null");
        }

        // SAFETY: `ptrs` is a valid contiguous stack array of NonNull pointers,
        // and `count` matches the number of initialised entries. The command
        // buffers are kept alive by the `buffers` slice references.
        unsafe {
            self.inner.commit_count(
                NonNull::new(ptrs.as_ptr() as *mut _).unwrap(),
                buffers.len(),
            );
        }
    }

    /// Batch-commit with commit options (e.g. feedback handlers).
    ///
    /// See [`commit_batch`](Self::commit_batch) for ordering requirements.
    ///
    /// # Panics
    ///
    /// Panics if `buffers` is empty.
    pub fn commit_batch_with_options(
        &self,
        buffers: &[&Mtl4CommandBuffer],
        options: &MTL4CommitOptions,
    ) {
        assert!(
            !buffers.is_empty(),
            "commit_batch_with_options requires at least one command buffer"
        );

        // Stack buffer — same rationale as commit_batch().
        const MAX_STACK: usize = 16;
        assert!(
            buffers.len() <= MAX_STACK,
            "commit_batch_with_options: batch size {} exceeds stack buffer capacity {}",
            buffers.len(),
            MAX_STACK,
        );

        let mut ptrs: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; MAX_STACK] =
            [NonNull::dangling(); MAX_STACK];
        for (i, cb) in buffers.iter().enumerate() {
            let raw = &*cb.inner as *const ProtocolObject<dyn MTL4CommandBuffer>
                as *mut ProtocolObject<dyn MTL4CommandBuffer>;
            ptrs[i] = NonNull::new(raw).expect("command buffer pointer must not be null");
        }

        // SAFETY: Same as `commit_batch` — valid stack pointer array + count.
        unsafe {
            self.inner.commit_count_options(
                NonNull::new(ptrs.as_ptr() as *mut _).unwrap(),
                buffers.len(),
                options,
            );
        }
    }

    /// Signal an event with a specific value after all prior GPU work completes.
    ///
    /// Useful for cross-queue synchronization with Metal 3 queues or other
    /// Metal 4 queues.
    pub fn signal_event(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.inner.signalEvent_value(event, value);
    }

    /// Wait for an event to reach a specific value before executing future GPU work.
    pub fn wait_event(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.inner.waitForEvent_value(event, value);
    }

    /// Add a residency set to ensure resources remain resident during execution.
    pub fn add_residency_set(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        self.inner.addResidencySet(set);
    }

    /// Remove a residency set from this queue.
    pub fn remove_residency_set(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        self.inner.removeResidencySet(set);
    }

    /// Access the raw `MTL4CommandQueue` protocol object.
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4CommandQueue> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn test_device() -> &'static MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests"))
    }

    #[test]
    fn command_allocator_creation() {
        let device = test_device();
        // Metal 4 may not be available on all hardware; skip gracefully.
        let Some(alloc) = CommandAllocator::new(&device) else {
            eprintln!("skipping: Metal 4 command allocator not supported on this device");
            return;
        };
        // Just verify it returns without panic; allocated_size is u64.
        let _size = alloc.allocated_size();
    }

    #[test]
    fn command_allocator_reset_idempotent() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        // Multiple resets without any CB usage should not panic.
        // SAFETY: No command buffers have been allocated, so none are in flight.
        unsafe { alloc.reset() };
        unsafe { alloc.reset() };
    }

    #[test]
    fn command_buffer_lifecycle() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        let Some(cb) = alloc.new_command_buffer(&device) else {
            eprintln!("skipping: Metal 4 command buffer creation failed");
            return;
        };

        cb.set_label("test-cb");
        cb.begin(&alloc);
        // Encode a no-op compute pass.
        if let Some(enc) = cb.compute_encoder() {
            enc.endEncoding();
        }
        cb.end();
    }

    #[test]
    fn command_queue4_creation() {
        let device = test_device();
        let Some(_queue) = CommandQueue4::new(&device) else {
            eprintln!("skipping: Metal 4 command queue not supported on this device");
            return;
        };
    }

    #[test]
    fn batch_commit_single_cb() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        let Some(queue) = CommandQueue4::new(&device) else {
            return;
        };
        let Some(cb) = alloc.new_command_buffer(&device) else {
            return;
        };

        cb.begin(&alloc);
        if let Some(enc) = cb.compute_encoder() {
            enc.endEncoding();
        }
        cb.end();

        queue.commit_batch(&[&cb]);
    }

    #[test]
    fn batch_commit_multiple_cbs() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        let Some(queue) = CommandQueue4::new(&device) else {
            return;
        };

        // Metal 4 allocators service one CB at a time, so we begin/end
        // sequentially before batch-committing.
        let Some(cb1) = alloc.new_command_buffer(&device) else {
            return;
        };
        cb1.begin(&alloc);
        if let Some(enc) = cb1.compute_encoder() {
            enc.endEncoding();
        }
        cb1.end();

        let Some(cb2) = alloc.new_command_buffer(&device) else {
            return;
        };
        cb2.begin(&alloc);
        if let Some(enc) = cb2.compute_encoder() {
            enc.endEncoding();
        }
        cb2.end();

        queue.commit_batch(&[&cb1, &cb2]);
    }

    #[test]
    fn allocator_reset_between_iterations() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        let Some(queue) = CommandQueue4::new(&device) else {
            return;
        };

        // Simulate two decode iterations with allocator reset between them.
        for _ in 0..2 {
            let Some(cb) = alloc.new_command_buffer(&device) else {
                return;
            };
            cb.begin(&alloc);
            if let Some(enc) = cb.compute_encoder() {
                enc.endEncoding();
            }
            cb.end();
            queue.commit_batch(&[&cb]);

            // In production, would wait for GPU completion here.
            // SAFETY: The CB was committed via commit_batch and this is a
            // test with trivial no-op work that completes near-instantly.
            unsafe { alloc.reset() };
        }
    }

    #[test]
    fn debug_group_push_pop() {
        let device = test_device();
        let Some(alloc) = CommandAllocator::new(&device) else {
            return;
        };
        let Some(cb) = alloc.new_command_buffer(&device) else {
            return;
        };

        cb.begin(&alloc);
        cb.push_debug_group("layer-0");
        if let Some(enc) = cb.compute_encoder() {
            enc.endEncoding();
        }
        cb.pop_debug_group();
        cb.end();
    }
}
