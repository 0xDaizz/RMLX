//! Metal buffer management

use metal::{Buffer as MTLBuffer, MTLResourceOptions};
use std::ffi::c_void;

/// Create a new buffer initialized with data from a typed slice.
///
/// Uses `StorageModeShared` so the buffer is accessible from both CPU and GPU.
pub fn new_buffer_with_data<T>(device: &metal::Device, data: &[T]) -> MTLBuffer {
    let size = std::mem::size_of_val(data) as u64;
    let ptr = data.as_ptr() as *const c_void;
    device.new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared)
}

/// Create a zero-copy buffer wrapping externally allocated memory.
///
/// # Safety
/// - `ptr` must be page-aligned (16384 bytes on Apple Silicon).
/// - `ptr` must remain valid for the entire lifetime of the returned buffer.
/// - `size` must not exceed the allocation behind `ptr`.
/// - The caller is responsible for freeing the memory *after* the buffer is dropped.
pub unsafe fn new_buffer_no_copy(device: &metal::Device, ptr: *mut c_void, size: u64) -> MTLBuffer {
    // SAFETY: Caller guarantees ptr is page-aligned, valid for size bytes,
    // and will outlive the returned buffer. We pass None for the deallocator
    // because the caller manages the memory lifetime.
    device.new_buffer_with_bytes_no_copy(ptr, size, MTLResourceOptions::StorageModeShared, None)
}

/// Read buffer contents as a typed slice.
///
/// # Safety
/// - The buffer must use `StorageModeShared` (CPU-accessible).
/// - No GPU writes may be in-flight to this buffer (the caller must ensure
///   the command buffer that last wrote to this buffer has completed).
/// - `count` must not exceed the number of `T` values that fit in the buffer.
pub unsafe fn read_buffer<T>(buffer: &MTLBuffer, count: usize) -> &[T] {
    // SAFETY: Caller guarantees the buffer is shared-mode, GPU work is complete,
    // and count * size_of::<T>() <= buffer.length(). contents() returns a valid
    // pointer for StorageModeShared buffers on Apple Silicon UMA.
    let ptr = buffer.contents() as *const T;
    // SAFETY: caller guarantees bounds and GPU completion (see fn-level doc).
    unsafe { std::slice::from_raw_parts(ptr, count) }
}
