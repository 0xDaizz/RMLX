//! Metal buffer management

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::device::{DEFAULT_BUFFER_OPTIONS, UNTRACKED_BUFFER_OPTIONS};
use crate::types::MtlBuffer;

/// Create a new buffer initialized with data from a typed slice.
///
/// Uses the safe default [`DEFAULT_BUFFER_OPTIONS`] (`StorageModeShared`)
/// so the buffer is CPU+GPU visible with Metal hazard tracking enabled.
pub fn new_buffer_with_data<T>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> MtlBuffer {
    let size = std::mem::size_of_val(data);
    let ptr = data.as_ptr() as *mut c_void;
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(ptr).unwrap(),
                size,
                DEFAULT_BUFFER_OPTIONS,
            )
            .unwrap()
    }
}

/// Create a zero-copy buffer wrapping externally allocated memory.
///
/// Uses [`UNTRACKED_BUFFER_OPTIONS`] for performance-critical zero-copy paths.
/// The caller must ensure synchronisation via the barrier tracker.
///
/// # Safety
/// - `ptr` must be page-aligned (16384 bytes on Apple Silicon).
/// - `ptr` must remain valid for the entire lifetime of the returned buffer.
/// - `size` must not exceed the allocation behind `ptr`.
/// - The caller is responsible for freeing the memory *after* the buffer is dropped.
pub unsafe fn new_buffer_no_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    ptr: *mut c_void,
    size: u64,
) -> MtlBuffer {
    // SAFETY: Caller guarantees ptr is page-aligned, valid for size bytes,
    // and will outlive the returned buffer. We pass None for the deallocator
    // because the caller manages the memory lifetime.
    unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(ptr).unwrap(),
                size as usize,
                UNTRACKED_BUFFER_OPTIONS,
                None,
            )
            .unwrap()
    }
}

/// Read buffer contents as a typed slice.
///
/// # Safety
/// - The buffer must use `StorageModeShared` (CPU-accessible).
/// - No GPU writes may be in-flight to this buffer (the caller must ensure
///   the command buffer that last wrote to this buffer has completed).
/// - `count` must not exceed the number of `T` values that fit in the buffer.
pub unsafe fn read_buffer<T>(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> &[T] {
    // SAFETY: Caller guarantees the buffer is shared-mode, GPU work is complete,
    // and count * size_of::<T>() <= buffer.length(). contents() returns a valid
    // pointer for StorageModeShared buffers on Apple Silicon UMA.
    let ptr = buffer.contents().as_ptr() as *const T;
    // SAFETY: caller guarantees bounds and GPU completion (see fn-level doc).
    unsafe { std::slice::from_raw_parts(ptr, count) }
}
