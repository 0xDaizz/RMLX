//! Metal 4 unified compute + blit command encoder.
//!
//! [`ComputePass4`] wraps `MTL4ComputeCommandEncoder`, which unifies compute
//! dispatches with blit operations (buffer copy, fill, etc.) in a single
//! encoder pass. This eliminates the encoder-switch overhead that Metal 3 incurs
//! when alternating between `MTLComputeCommandEncoder` and `MTLBlitCommandEncoder`.
//!
//! All methods mirror the ergonomic style of [`crate::ComputePass`], with
//! additional blit operations exposed as first-class methods.

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;

/// A borrowed Metal 4 compute command encoder with integrated blit support.
///
/// Wraps `MTL4ComputeCommandEncoder`, which extends `MTL4CommandEncoder` and
/// combines compute dispatch, buffer copy/fill, and texture copy operations
/// into a single encoder.
///
/// # Lifetime
///
/// The `'a` lifetime ties this to the command buffer that created the encoder.
/// Call [`end`](Self::end) before committing the command buffer.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ComputePass4<'a>(pub(crate) &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>);

impl<'a> ComputePass4<'a> {
    /// Create a new `ComputePass4` wrapping a borrowed MTL4 compute encoder.
    #[inline(always)]
    pub fn new(encoder: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>) -> Self {
        Self(encoder)
    }

    // -----------------------------------------------------------------------
    // Compute operations (same interface as ComputePass)
    // -----------------------------------------------------------------------

    /// Set the active compute pipeline state.
    #[inline(always)]
    pub fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.0.setComputePipelineState(pso);
    }

    /// Set threadgroup memory length at the given index.
    #[inline(always)]
    pub fn set_threadgroup_memory_length(&self, index: u32, length: u64) {
        unsafe {
            self.0
                .setThreadgroupMemoryLength_atIndex(length as usize, index as usize);
        }
    }

    /// Dispatch a grid of threads with automatic threadgroup sizing.
    #[inline(always)]
    pub fn dispatch_threads(&self, grid: MTLSize, tg: MTLSize) {
        self.0.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Dispatch threadgroups (grid is in threadgroup units, not thread units).
    #[inline(always)]
    pub fn dispatch_threadgroups(&self, grid: MTLSize, tg: MTLSize) {
        self.0.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    // -----------------------------------------------------------------------
    // Blit operations (Metal 4 integrated — no separate blit encoder needed)
    // -----------------------------------------------------------------------

    /// Copy bytes between two Metal buffers on the GPU.
    ///
    /// This is a blit operation encoded directly into the compute pass,
    /// avoiding the encoder-switch overhead of Metal 3.
    ///
    /// # Safety
    ///
    /// Caller must ensure offsets and size are within buffer bounds.
    #[inline(always)]
    pub fn copy_buffer(
        &self,
        src: &ProtocolObject<dyn MTLBuffer>,
        src_offset: usize,
        dst: &ProtocolObject<dyn MTLBuffer>,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.0
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src, src_offset, dst, dst_offset, size,
                );
        }
    }

    /// Fill a buffer region with a constant byte value.
    ///
    /// # Safety
    ///
    /// Caller must ensure the range is within the buffer bounds.
    #[inline(always)]
    pub fn fill_buffer(
        &self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        range: std::ops::Range<usize>,
        value: u8,
    ) {
        unsafe {
            self.0.fillBuffer_range_value(
                buf,
                NSRange::new(range.start, range.end - range.start),
                value,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Barrier operations (Metal 4 intra-pass barriers)
    // -----------------------------------------------------------------------

    /// Encode an intra-pass barrier between encoder stages.
    ///
    /// Ensures that commands corresponding to `after_stages` complete before
    /// commands corresponding to `before_stages` begin, within this encoder.
    #[inline(always)]
    pub fn barrier(
        &self,
        after_stages: MTLStages,
        before_stages: MTLStages,
        visibility: MTL4VisibilityOptions,
    ) {
        self.0
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                after_stages,
                before_stages,
                visibility,
            );
    }

    /// Query the stages that have been encoded so far.
    #[inline(always)]
    pub fn stages(&self) -> MTLStages {
        self.0.stages()
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// End encoding on this compute pass.
    #[inline(always)]
    pub fn end(&self) {
        self.0.endEncoding();
    }

    /// Access the raw MTL4 compute command encoder for advanced usage.
    #[inline(always)]
    pub fn raw(&self) -> &ProtocolObject<dyn MTL4ComputeCommandEncoder> {
        self.0
    }

    /// Obtain a Metal 3 `ComputePass` from this Metal 4 encoder.
    ///
    /// At the ObjC runtime level, the concrete Metal 4 encoder object conforms
    /// to **both** `MTL4ComputeCommandEncoder` and `MTLComputeCommandEncoder`.
    /// This method performs an unsafe protocol cast to expose the Metal 3
    /// interface, enabling use of existing ops that accept `ComputePass`.
    ///
    /// # Safety
    ///
    /// This is safe on macOS 26+ where Metal 4 encoders are returned by the
    /// system framework and always conform to `MTLComputeCommandEncoder`.
    #[inline(always)]
    pub fn as_legacy_pass(&self) -> crate::ComputePass<'a> {
        // SAFETY: The concrete ObjC object behind `MTL4ComputeCommandEncoder`
        // also conforms to `MTLComputeCommandEncoder`. Both are `repr(C)`
        // protocol object pointers to the same underlying NSObject.
        let legacy: &'a ProtocolObject<dyn MTLComputeCommandEncoder> =
            unsafe { &*(self.0 as *const _ as *const _) };
        crate::ComputePass::new(legacy)
    }
}
