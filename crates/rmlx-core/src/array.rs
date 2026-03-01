//! N-dimensional array backed by a Metal buffer.

use metal::Buffer as MTLBuffer;
use metal::MTLResourceOptions;
use rmlx_metal::metal;

use rmlx_alloc::MetalAllocator;

use crate::dtype::{DType, HasDType};

/// An N-dimensional array stored in a Metal GPU buffer.
///
/// The buffer uses `StorageModeShared` (CPU + GPU accessible) on Apple Silicon UMA.
/// Shape, strides, and dtype are tracked as metadata alongside the buffer.
pub struct Array {
    buffer: MTLBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    /// Byte offset into the buffer where this array's data begins.
    offset: usize,
}

impl Array {
    /// Create an array wrapping an existing Metal buffer.
    pub fn new(
        buffer: MTLBuffer,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
        offset: usize,
    ) -> Self {
        Self {
            buffer,
            shape,
            strides,
            dtype,
            offset,
        }
    }

    /// Create an array from a typed slice, allocating a new Metal buffer.
    pub fn from_slice<T: HasDType>(device: &metal::Device, data: &[T], shape: Vec<usize>) -> Self {
        let dtype = T::DTYPE;
        let numel: usize = shape.iter().product();
        debug_assert_eq!(
            data.len(),
            numel,
            "data length ({}) does not match shape product ({})",
            data.len(),
            numel
        );

        let size = std::mem::size_of_val(data) as u64;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let buffer = device.new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared);

        let strides = compute_contiguous_strides(&shape);
        Self {
            buffer,
            shape,
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Create a zero-filled array.
    pub fn zeros(device: &metal::Device, shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let byte_size = dtype.numel_to_bytes(numel) as u64;
        let buffer = device.new_buffer(byte_size, MTLResourceOptions::StorageModeShared);

        // StorageModeShared buffers are zero-initialized by the OS on Apple Silicon.
        let strides = compute_contiguous_strides(shape);
        Self {
            buffer,
            shape: shape.to_vec(),
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Create a zero-filled array using a `MetalAllocator` buffer pool.
    ///
    /// Reuses cached buffers when possible, avoiding fresh allocations.
    /// Falls back to a new allocation if the cache has no buffer of
    /// suitable size.
    pub fn zeros_pooled(
        allocator: &MetalAllocator,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self, rmlx_alloc::AllocError> {
        let numel: usize = shape.iter().product();
        let byte_size = dtype.numel_to_bytes(numel);
        let buffer = allocator.alloc(byte_size)?;

        // Zero the buffer contents (cached buffers may contain stale data).
        // SAFETY: SharedMode buffer contents() is CPU-accessible and valid
        // for buffer.length() bytes.
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, byte_size);
        }

        let strides = compute_contiguous_strides(shape);
        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            strides,
            dtype,
            offset: 0,
        })
    }

    /// Create an array filled with ones (f32 only).
    pub fn ones(device: &metal::Device, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = vec![1.0; numel];
        Self::from_slice(device, &data, shape.to_vec())
    }

    /// Read the buffer contents as a typed Vec (bounds-checked).
    ///
    /// Asserts that `offset + numel * size_of::<T>()` fits within the buffer,
    /// then copies data out. Prefer this over `to_vec_unchecked` in upper layers.
    ///
    /// Note: callers should ensure all GPU command buffers writing to this
    /// buffer have completed before reading, otherwise values may be stale.
    pub fn to_vec_checked<T: HasDType + Clone>(&self) -> Vec<T> {
        assert_eq!(
            T::DTYPE,
            self.dtype,
            "type mismatch: requested {:?} but array is {:?}",
            T::DTYPE,
            self.dtype
        );
        let numel: usize = self.shape.iter().product();
        let byte_size = numel * std::mem::size_of::<T>();
        assert!(
            self.offset + byte_size <= self.buffer.length() as usize,
            "to_vec_checked: offset({}) + data({}) exceeds buffer({})",
            self.offset,
            byte_size,
            self.buffer.length()
        );
        let base = self.buffer.contents() as *const u8;
        // SAFETY: bounds checked above; contents() returns valid CPU-accessible
        // pointer for StorageModeShared buffers.
        unsafe {
            let ptr = base.add(self.offset) as *const T;
            std::slice::from_raw_parts(ptr, numel).to_vec()
        }
    }

    /// Read the buffer contents as a typed Vec (unchecked).
    ///
    /// # Safety
    /// The caller must ensure:
    /// - No GPU writes are in-flight to this buffer
    /// - `T::DTYPE` matches `self.dtype`
    /// - `offset + numel * size_of::<T>()` fits within the buffer
    pub unsafe fn to_vec_unchecked<T: HasDType + Clone>(&self) -> Vec<T> {
        let numel: usize = self.shape.iter().product();
        let base = self.buffer.contents() as *const u8;
        // SAFETY: caller guarantees bounds and GPU completion (see fn-level doc).
        unsafe {
            let ptr = base.add(self.offset) as *const T;
            let slice = std::slice::from_raw_parts(ptr, numel);
            slice.to_vec()
        }
    }

    /// Reference to the underlying Metal buffer.
    pub fn metal_buffer(&self) -> &MTLBuffer {
        &self.buffer
    }

    /// Byte offset into the buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Array shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Array strides (in elements, not bytes).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total data size in bytes.
    pub fn byte_size(&self) -> usize {
        self.dtype.numel_to_bytes(self.numel())
    }

    /// Returns the raw bytes of this array's buffer (from offset, for numel * dtype bytes).
    ///
    /// Note: callers should ensure all GPU command buffers writing to this
    /// buffer have completed before reading, otherwise values may be stale.
    pub fn to_bytes(&self) -> &[u8] {
        let len = self.byte_size();
        assert!(
            self.offset + len <= self.buffer.length() as usize,
            "to_bytes: offset({}) + len({}) exceeds buffer({})",
            self.offset,
            len,
            self.buffer.length()
        );
        let base = self.buffer.contents() as *const u8;
        // SAFETY: bounds checked above; contents() returns valid CPU-accessible
        // pointer for StorageModeShared buffers.
        unsafe { std::slice::from_raw_parts(base.add(self.offset), len) }
    }

    /// Create an Array from raw bytes, allocating a new Metal buffer.
    ///
    /// `bytes.len()` must equal the exact buffer size for the given shape and dtype.
    pub fn from_bytes(device: &metal::Device, bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let expected = dtype.numel_to_bytes(numel);
        assert_eq!(
            bytes.len(),
            expected,
            "from_bytes: bytes length ({}) does not match expected ({}) for shape {:?} dtype {:?}",
            bytes.len(),
            expected,
            shape,
            dtype
        );
        let ptr = bytes.as_ptr() as *const std::ffi::c_void;
        let buffer =
            device.new_buffer_with_data(ptr, bytes.len() as u64, MTLResourceOptions::StorageModeShared);
        let strides = compute_contiguous_strides(&shape);
        Self {
            buffer,
            shape,
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Slice columns [start..end) for tensor parallelism sharding.
    ///
    /// For a 2D array `[rows, cols]`, returns a view of `[rows, end - start]`
    /// sharing the same underlying buffer with adjusted offset and shape.
    ///
    /// # Panics
    /// Panics if the array is not 2D or if the column range is out of bounds.
    pub fn slice_columns(&self, start: usize, end: usize) -> Self {
        assert_eq!(self.ndim(), 2, "slice_columns requires a 2D array, got {}D", self.ndim());
        let cols = self.shape[1];
        assert!(
            start < end && end <= cols,
            "slice_columns: invalid range [{}..{}) for {} columns",
            start,
            end,
            cols
        );
        let elem_bytes = self.dtype.size_of();
        let new_offset = self.offset + start * elem_bytes;
        Self {
            buffer: self.buffer.clone(),
            shape: vec![self.shape[0], end - start],
            strides: self.strides.clone(),
            dtype: self.dtype,
            offset: new_offset,
        }
    }

    /// Whether the array is stored contiguously in memory.
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_contiguous_strides(&self.shape)
    }

    /// Create a view with a new shape (same buffer, zero-copy).
    /// The array must be contiguous and the total element count must match.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, crate::kernels::KernelError> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "reshape: element count mismatch ({} vs {})",
                self.numel(),
                new_numel
            )));
        }
        if !self.is_contiguous() {
            return Err(crate::kernels::KernelError::InvalidShape(
                "reshape requires a contiguous array".into(),
            ));
        }
        let new_strides = compute_contiguous_strides(&new_shape);
        Ok(Self {
            buffer: self.buffer.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// Create a view with custom strides and offset (same buffer, zero-copy).
    pub fn view(&self, shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            buffer: self.buffer.clone(),
            shape,
            strides,
            dtype: self.dtype,
            offset,
        }
    }
}

/// Compute contiguous (row-major) strides for a given shape.
fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(compute_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_contiguous_strides(&[5]), vec![1]);
        assert_eq!(compute_contiguous_strides(&[]), Vec::<usize>::new());
    }
}
