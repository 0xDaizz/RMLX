//! Zero-cost newtype wrapper around MTLComputeCommandEncoder.
//!
//! [`ComputePass`] provides a safe(r) interface to Metal compute encoding,
//! hiding the verbose objc2-metal method names behind ergonomic methods.
//! All methods are `#[inline(always)]` for zero overhead.

use std::mem::size_of;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;

/// A borrowed compute command encoder with ergonomic helpers.
///
/// Created from `command_buffer.new_compute_command_encoder()` and provides
/// safe wrappers around the most common encoding operations.
///
/// # Lifetime
///
/// The `'a` lifetime ties this to the command buffer that created the encoder.
/// The encoder must be ended (via [`end`]) before the command buffer is committed.
#[repr(transparent)]
pub struct ComputePass<'a>(pub(crate) &'a ProtocolObject<dyn MTLComputeCommandEncoder>);

impl<'a> ComputePass<'a> {
    /// Bind a Metal buffer at the given index with an offset.
    #[inline(always)]
    pub fn set_buffer(
        &self,
        index: u32,
        buf: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
    ) {
        unsafe {
            self.0
                .setBuffer_offset_atIndex(buf, offset as usize, index as usize);
        }
    }

    /// Set a typed value as bytes at the given buffer index.
    ///
    /// Uses `bytemuck::Pod` to ensure the value is safe to transmit as raw bytes.
    #[inline(always)]
    pub fn set_val<T: bytemuck::Pod>(&self, index: u32, value: &T) {
        unsafe {
            self.0.setBytes_length_atIndex(
                NonNull::from(value).cast(),
                size_of::<T>(),
                index as usize,
            );
        }
    }

    /// Set raw bytes at the given buffer index (for pre-packed data).
    #[inline(always)]
    pub fn set_bytes(&self, index: u32, data: *const std::ffi::c_void, len: usize) {
        unsafe {
            let ptr = NonNull::new(data as *mut _).expect("set_bytes: null pointer");
            self.0.setBytes_length_atIndex(ptr, len, index as usize);
        }
    }

    /// Batch-set multiple buffers starting at `start` index.
    ///
    /// More efficient than individual `set_buffer` calls when binding
    /// multiple consecutive buffers (single Obj-C message send).
    #[inline(always)]
    pub fn set_buffers(
        &self,
        start: u32,
        bufs: &[Option<&ProtocolObject<dyn MTLBuffer>>],
        offsets: &[usize],
    ) {
        debug_assert_eq!(bufs.len(), offsets.len(), "bufs and offsets length mismatch");
        unsafe {
            self.0.setBuffers_offsets_withRange(
                NonNull::new(bufs.as_ptr() as *mut _).unwrap(),
                NonNull::new(offsets.as_ptr() as *mut _).unwrap(),
                NSRange::new(start as usize, bufs.len()),
            );
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
        self.0
            .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

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

    /// End encoding on this compute pass.
    #[inline(always)]
    pub fn end(&self) {
        self.0.endEncoding();
    }

    /// Access the raw encoder for advanced usage.
    #[inline(always)]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        self.0
    }
}
