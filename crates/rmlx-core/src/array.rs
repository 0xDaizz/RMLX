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

    /// Read the buffer contents as a typed Vec.
    ///
    /// # Safety
    /// The caller must ensure no GPU writes are in-flight to this buffer.
    pub unsafe fn to_vec<T: HasDType + Clone>(&self) -> Vec<T> {
        debug_assert_eq!(
            T::DTYPE,
            self.dtype,
            "type mismatch: requested {:?} but array is {:?}",
            T::DTYPE,
            self.dtype
        );
        let numel: usize = self.shape.iter().product();
        let base = self.buffer.contents() as *const u8;
        let ptr = base.add(self.offset) as *const T;
        let slice = std::slice::from_raw_parts(ptr, numel);
        slice.to_vec()
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
