//! N-dimensional array backed by a Metal buffer.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer as _;
use rmlx_metal::{MTLResourceOptions, MtlBuffer};

use rmlx_alloc::MetalAllocator;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

#[cfg(feature = "bench")]
use std::cell::Cell;
#[cfg(feature = "bench")]
use std::time::Instant;

use crate::dtype::{DType, HasDType};
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLDevice as _;

// Thread-local allocation statistics for profiling.
// Tracks count and cumulative time of Array::uninit() / Array::zeros() calls.
#[cfg(feature = "bench")]
thread_local! {
    static ALLOC_COUNT: Cell<u64> = Cell::new(0);
    static ALLOC_NANOS: Cell<u64> = Cell::new(0);
    static ALLOC_BYTES: Cell<u64> = Cell::new(0);
}

// ---------------------------------------------------------------------------
// Thread-local buffer pool for recycling Metal buffers
// ---------------------------------------------------------------------------

/// Thread-local buffer pool for recycling Metal buffers.
/// Eliminates ~3ms/iter of `device.new_buffer()` overhead during forward passes.
struct ArrayPool {
    /// Free buffers keyed by allocation size (bytes).
    free: HashMap<usize, Vec<MtlBuffer>>,
    enabled: bool,
    /// Stats
    hits: u64,
    misses: u64,
}

impl ArrayPool {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
            enabled: false,
            hits: 0,
            misses: 0,
        }
    }

    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        size: usize,
    ) -> MtlBuffer {
        if self.enabled {
            if let Some(bufs) = self.free.get_mut(&size) {
                if let Some(buf) = bufs.pop() {
                    self.hits += 1;
                    return buf;
                }
            }
            self.misses += 1;
        }
        device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .unwrap()
    }

    fn recycle(&mut self, buf: MtlBuffer) {
        if self.enabled {
            let size = buf.length();
            self.free.entry(size).or_default().push(buf);
        }
        // If not enabled, buffer is simply dropped (normal Metal ARC dealloc)
    }
}

thread_local! {
    static ARRAY_POOL: RefCell<ArrayPool> = RefCell::new(ArrayPool::new());
}

/// Enable the thread-local buffer pool. Subsequent `Array::uninit()` calls
/// will reuse buffers, and dropped Arrays will return buffers to the pool.
pub fn enable_array_pool() {
    ARRAY_POOL.with(|p| p.borrow_mut().enabled = true);
}

/// Disable the pool and drain all cached buffers (freeing Metal memory).
pub fn disable_array_pool() {
    ARRAY_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        pool.enabled = false;
        pool.free.clear();
        pool.hits = 0;
        pool.misses = 0;
    });
}

/// Get pool statistics: (hits, misses, cached_buffer_count).
pub fn array_pool_stats() -> (u64, u64, usize) {
    ARRAY_POOL.with(|p| {
        let pool = p.borrow();
        let cached: usize = pool.free.values().map(|v| v.len()).sum();
        (pool.hits, pool.misses, cached)
    })
}

#[cfg(feature = "bench")]
pub fn reset_alloc_stats() {
    ALLOC_COUNT.with(|c| c.set(0));
    ALLOC_NANOS.with(|c| c.set(0));
    ALLOC_BYTES.with(|c| c.set(0));
}

#[cfg(feature = "bench")]
pub fn get_alloc_stats() -> (u64, u64, u64) {
    let count = ALLOC_COUNT.with(|c| c.get());
    let nanos = ALLOC_NANOS.with(|c| c.get());
    let bytes = ALLOC_BYTES.with(|c| c.get());
    (count, nanos, bytes)
}

/// An N-dimensional array stored in a Metal GPU buffer.
///
/// The buffer uses `StorageModeShared` (CPU + GPU accessible) on Apple Silicon UMA.
/// Shape, strides, and dtype are tracked as metadata alongside the buffer.
///
/// `Debug` prints shape, strides, and dtype (not buffer contents).
pub struct Array {
    buffer: ManuallyDrop<MtlBuffer>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    /// Byte offset into the buffer where this array's data begins.
    offset: usize,
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("dtype", &self.dtype)
            .field("offset", &self.offset)
            .finish()
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: We take ownership of the buffer via ManuallyDrop::take.
        // This is the only place the buffer is consumed; after drop() returns,
        // Rust will not drop the ManuallyDrop field (that's the whole point).
        let buf = unsafe { ManuallyDrop::take(&mut self.buffer) };
        // Only recycle buffers that own the full allocation (offset == 0).
        // Views (offset > 0) share a buffer with a parent array and must
        // not be recycled — the parent will handle it.
        if self.offset == 0 {
            ARRAY_POOL.with(|p| p.borrow_mut().recycle(buf));
        }
        // else: buf is dropped normally here (Metal ARC decrements refcount)
    }
}

impl Array {
    /// Create an array wrapping an existing Metal buffer.
    pub fn new(
        buffer: MtlBuffer,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
        offset: usize,
    ) -> Self {
        Self {
            buffer: ManuallyDrop::new(buffer),
            shape,
            strides,
            dtype,
            offset,
        }
    }

    /// Create an array from a typed slice, allocating a new Metal buffer.
    pub fn from_slice<T: HasDType>(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        data: &[T],
        shape: Vec<usize>,
    ) -> Self {
        let dtype = T::DTYPE;
        let numel: usize = shape.iter().product();
        debug_assert_eq!(
            data.len(),
            numel,
            "data length ({}) does not match shape product ({})",
            data.len(),
            numel
        );

        let size = std::mem::size_of_val(data);
        // Metal returns null for zero-length buffers; allocate at least 1 byte.
        let buffer = if size == 0 {
            device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .unwrap()
        } else {
            let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
            unsafe {
                device
                    .newBufferWithBytes_length_options(
                        ptr,
                        size,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap()
            }
        };

        let strides = compute_contiguous_strides(&shape);
        Self {
            buffer: ManuallyDrop::new(buffer),
            shape,
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Allocate an uninitialized array. Use ONLY for outputs that will be
    /// fully overwritten by a GPU kernel before any read.
    pub fn uninit(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        shape: &[usize],
        dtype: DType,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let byte_size = dtype
            .numel_to_bytes(numel)
            .expect("numel must be block-aligned for quantized dtypes");
        // Metal returns null for zero-length buffers; allocate at least 1 byte.
        let alloc_size = byte_size.max(1);

        #[cfg(feature = "bench")]
        let t0 = Instant::now();

        let buffer = ARRAY_POOL.with(|p| p.borrow_mut().acquire(device, alloc_size));

        #[cfg(feature = "bench")]
        {
            let elapsed = t0.elapsed().as_nanos() as u64;
            ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            ALLOC_NANOS.with(|c| c.set(c.get() + elapsed));
            ALLOC_BYTES.with(|c| c.set(c.get() + alloc_size as u64));
        }

        let strides = compute_contiguous_strides(shape);
        Self {
            buffer: ManuallyDrop::new(buffer),
            shape: shape.to_vec(),
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Create a zero-filled array.
    ///
    /// Explicitly zeroes the buffer after allocation to guarantee correctness
    /// regardless of platform or Metal driver behavior.
    pub fn zeros(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        shape: &[usize],
        dtype: DType,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let byte_size = dtype
            .numel_to_bytes(numel)
            .expect("numel must be block-aligned for quantized dtypes");
        // Metal returns null for zero-length buffers; allocate at least 1 byte.
        let alloc_size = byte_size.max(1);

        #[cfg(feature = "bench")]
        let t0 = Instant::now();

        let buffer = ARRAY_POOL.with(|p| p.borrow_mut().acquire(device, alloc_size));

        #[cfg(feature = "bench")]
        {
            let elapsed = t0.elapsed().as_nanos() as u64;
            ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            ALLOC_NANOS.with(|c| c.set(c.get() + elapsed));
            ALLOC_BYTES.with(|c| c.set(c.get() + alloc_size as u64));
        }

        // Explicitly zero the buffer. Recycled buffers may contain stale data,
        // and even fresh buffers are not guaranteed to be zeroed by Metal.
        // SAFETY: SharedMode buffer contents() is CPU-accessible and valid
        // for buffer.length() bytes.
        unsafe {
            std::ptr::write_bytes(buffer.contents().as_ptr() as *mut u8, 0, alloc_size);
        }

        let strides = compute_contiguous_strides(shape);
        Self {
            buffer: ManuallyDrop::new(buffer),
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
        let byte_size = dtype
            .numel_to_bytes(numel)
            .map_err(|e| rmlx_alloc::AllocError::DType(e.to_string()))?;
        let (buffer, sub_offset) = allocator.alloc(byte_size)?;

        // Zero the buffer contents (cached buffers may contain stale data).
        // SAFETY: SharedMode buffer contents() is CPU-accessible and valid
        // for buffer.length() bytes. We zero only our sub-allocation region.
        unsafe {
            let base = (buffer.contents().as_ptr() as *mut u8).add(sub_offset);
            std::ptr::write_bytes(base, 0, byte_size);
        }

        let strides = compute_contiguous_strides(shape);
        Ok(Self {
            buffer: ManuallyDrop::new(buffer),
            shape: shape.to_vec(),
            strides,
            dtype,
            offset: sub_offset,
        })
    }

    /// Create an array filled with ones (f32 only).
    pub fn ones(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = vec![1.0; numel];
        Self::from_slice(device, &data, shape.to_vec())
    }

    /// Read the buffer contents as a typed Vec (bounds-checked).
    ///
    /// Asserts that `offset + numel * size_of::<T>()` fits within the buffer,
    /// then copies data out in logical (row-major) order. For non-contiguous
    /// arrays, this iterates using stride-aware indexing rather than a flat
    /// memcpy. Prefer this over `to_vec_unchecked` in upper layers.
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
        let elem_size = std::mem::size_of::<T>();

        if self.is_contiguous() {
            // Fast path: data is laid out contiguously, single memcpy.
            let byte_size = numel * elem_size;
            assert!(
                self.offset + byte_size <= self.buffer.length(),
                "to_vec_checked: offset({}) + data({}) exceeds buffer({})",
                self.offset,
                byte_size,
                self.buffer.length()
            );
            let base = self.buffer.contents().as_ptr() as *const u8;
            // SAFETY: bounds checked above; contents() returns valid CPU-accessible
            // pointer for StorageModeShared buffers.
            unsafe {
                let ptr = base.add(self.offset) as *const T;
                std::slice::from_raw_parts(ptr, numel).to_vec()
            }
        } else {
            // Slow path: non-contiguous layout, gather elements via strides.
            self.gather_strided::<T>(numel)
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

        if self.is_contiguous() {
            // Fast path: data is laid out contiguously, single memcpy.
            let base = self.buffer.contents().as_ptr() as *const u8;
            // SAFETY: caller guarantees bounds and GPU completion (see fn-level doc).
            unsafe {
                let ptr = base.add(self.offset) as *const T;
                let slice = std::slice::from_raw_parts(ptr, numel);
                slice.to_vec()
            }
        } else {
            // Slow path: non-contiguous layout, gather elements via strides.
            self.gather_strided::<T>(numel)
        }
    }

    /// Gather elements in logical (row-major) order from a strided layout.
    ///
    /// Iterates over all multi-dimensional indices, computes the physical
    /// element offset using `strides`, and reads each element individually.
    fn gather_strided<T: Clone>(&self, numel: usize) -> Vec<T> {
        let base = self.buffer.contents().as_ptr() as *const u8;
        let elem_size = std::mem::size_of::<T>();
        let ndim = self.shape.len();
        let mut result = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];

        for _ in 0..numel {
            // Compute the physical offset for the current multi-dim index.
            let mut physical_offset = self.offset;
            for (d, &idx) in indices.iter().enumerate().take(ndim) {
                physical_offset += idx * self.strides[d] * elem_size;
            }

            // SAFETY: the caller (to_vec_checked or to_vec_unchecked) has
            // verified type correctness and buffer bounds.
            unsafe {
                let ptr = base.add(physical_offset) as *const T;
                result.push((*ptr).clone());
            }

            // Increment the multi-dimensional index (row-major order, last dim fastest).
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        result
    }

    /// Reference to the underlying Metal buffer.
    pub fn metal_buffer(&self) -> &MtlBuffer {
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
        self.dtype
            .numel_to_bytes(self.numel())
            .expect("array numel must be block-aligned")
    }

    /// Copy this array's data into a new `StorageModePrivate` buffer.
    ///
    /// The returned array is GPU-only (cannot be read by CPU).
    /// Use for static weights that are loaded once and only read by GPU kernels.
    /// This eliminates CPU page-table mapping overhead for GPU-only buffers.
    pub fn to_private(
        &self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    ) -> Self {
        use objc2_metal::MTLBlitCommandEncoder as _;
        use objc2_metal::MTLCommandBuffer as _;
        use objc2_metal::MTLCommandEncoder as _;
        let byte_size = self.byte_size();
        let private_buf = device
            .newBufferWithLength_options(byte_size.max(4), MTLResourceOptions::StorageModePrivate)
            .unwrap();

        // Blit copy from shared to private
        let cb = queue.commandBuffer().unwrap();
        let blit = cb.blitCommandEncoder().unwrap();
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                self.metal_buffer(),
                self.offset(),
                &private_buf,
                0,
                byte_size,
            )
        };
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        Self::new(
            private_buf,
            self.shape().to_vec(),
            self.strides().to_vec(),
            self.dtype(),
            0, // new buffer starts at offset 0
        )
    }

    /// Returns the raw bytes of this array's buffer (from offset, for numel * dtype bytes).
    ///
    /// Note: callers should ensure all GPU command buffers writing to this
    /// buffer have completed before reading, otherwise values may be stale.
    pub fn to_bytes(&self) -> &[u8] {
        let len = self.byte_size();
        assert!(
            self.offset + len <= self.buffer.length(),
            "to_bytes: offset({}) + len({}) exceeds buffer({})",
            self.offset,
            len,
            self.buffer.length()
        );
        let base = self.buffer.contents().as_ptr() as *const u8;
        // SAFETY: bounds checked above; contents() returns valid CPU-accessible
        // pointer for StorageModeShared buffers.
        unsafe { std::slice::from_raw_parts(base.add(self.offset), len) }
    }

    /// Create an Array from raw bytes, allocating a new Metal buffer.
    ///
    /// `bytes.len()` must equal the exact buffer size for the given shape and dtype.
    pub fn from_bytes(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let numel: usize = shape.iter().product();
        let expected = dtype
            .numel_to_bytes(numel)
            .expect("numel must be block-aligned for quantized dtypes");
        assert_eq!(
            bytes.len(),
            expected,
            "from_bytes: bytes length ({}) does not match expected ({}) for shape {:?} dtype {:?}",
            bytes.len(),
            expected,
            shape,
            dtype
        );
        let ptr = NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void).unwrap();
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let strides = compute_contiguous_strides(&shape);
        Self {
            buffer: ManuallyDrop::new(buffer),
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
        assert_eq!(
            self.ndim(),
            2,
            "slice_columns requires a 2D array, got {}D",
            self.ndim()
        );
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
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: vec![self.shape[0], end - start],
            strides: self.strides.clone(),
            dtype: self.dtype,
            offset: new_offset,
        }
    }

    /// Slice along an arbitrary axis, returning a view for `[start..end)` on
    /// that axis. The returned array shares the same underlying buffer with
    /// adjusted offset, shape, and strides.
    ///
    /// # Panics
    /// Panics if `axis >= ndim`, or if the range is out of bounds.
    pub fn slice(
        &self,
        axis: usize,
        start: usize,
        end: usize,
    ) -> Result<Self, crate::kernels::KernelError> {
        if axis >= self.ndim() {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "slice: axis {} out of range for {}D array",
                axis,
                self.ndim()
            )));
        }
        let dim_size = self.shape[axis];
        if start >= end || end > dim_size {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "slice: invalid range [{}..{}) for axis {} with size {}",
                start, end, axis, dim_size
            )));
        }

        let elem_bytes = self.dtype.size_of();
        let new_offset = self.offset + start * self.strides[axis] * elem_bytes;

        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        // Strides remain unchanged: the view simply starts at a different
        // offset and has a smaller extent along `axis`.
        Ok(Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: self.strides.clone(),
            dtype: self.dtype,
            offset: new_offset,
        })
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
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// Create a view with custom strides and offset (same buffer, zero-copy).
    pub fn view(&self, shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape,
            strides,
            dtype: self.dtype,
            offset,
        }
    }

    /// Remove all size-1 dimensions from the shape, returning a new view.
    ///
    /// For example, shape `[1, 3, 1, 5]` becomes `[3, 5]`. If all dimensions
    /// are size 1, the result is a scalar-shaped array with shape `[1]`.
    pub fn squeeze(&self) -> Self {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            if dim != 1 {
                new_shape.push(dim);
                new_strides.push(self.strides[i]);
            }
        }
        // If all dims were 1, keep at least one dimension.
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }
        Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        }
    }

    /// Insert a size-1 dimension at the given axis, returning a new view.
    ///
    /// For example, shape `[3, 5]` with `axis = 0` becomes `[1, 3, 5]`,
    /// and with `axis = 2` becomes `[3, 5, 1]`.
    ///
    /// # Errors
    /// Returns an error if `axis > ndim`.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, crate::kernels::KernelError> {
        if axis > self.ndim() {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "unsqueeze: axis {} out of range for {}D array (valid: 0..={})",
                axis,
                self.ndim(),
                self.ndim()
            )));
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        // The stride for a size-1 dimension doesn't matter for indexing, but
        // we set it to the product of all dimensions after `axis` so that the
        // array remains "contiguous-looking" when possible.
        let stride_val = if axis < self.strides.len() {
            // Insert before an existing axis: match that axis's stride * dim
            // so that squeezing round-trips correctly.
            self.strides[axis] * self.shape[axis]
        } else {
            // Appending at the end: stride 1.
            1
        };

        new_shape.insert(axis, 1);
        new_strides.insert(axis, stride_val);

        Ok(Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// Transpose (swap) two dimensions, returning a new zero-copy view.
    ///
    /// This swaps the shape and stride entries for `dim0` and `dim1`.
    /// The resulting array shares the same underlying buffer but may be
    /// non-contiguous.
    ///
    /// # Errors
    /// Returns an error if either dimension is out of range.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, crate::kernels::KernelError> {
        let ndim = self.ndim();
        if dim0 >= ndim {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "transpose: dim0={dim0} out of range for {ndim}D array"
            )));
        }
        if dim1 >= ndim {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "transpose: dim1={dim1} out of range for {ndim}D array"
            )));
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Ok(Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// Remove a specific dimension of size 1, returning a new zero-copy view.
    ///
    /// Unlike [`squeeze`](Array::squeeze) which removes all size-1 dimensions,
    /// this targets a single dimension.
    ///
    /// # Errors
    /// Returns an error if `dim >= ndim` or if `shape[dim] != 1`.
    pub fn squeeze_dim(&self, dim: usize) -> Result<Self, crate::kernels::KernelError> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "squeeze_dim: dim={dim} out of range for {ndim}D array"
            )));
        }
        if self.shape[dim] != 1 {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "squeeze_dim: dimension {dim} has size {}, expected 1",
                self.shape[dim]
            )));
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(dim);
        new_strides.remove(dim);
        // If we removed the last dimension, keep at least a scalar shape.
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }
        Ok(Self {
            buffer: ManuallyDrop::new((*self.buffer).clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: self.offset,
        })
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

/// Compute the broadcast-compatible output shape for two input shapes,
/// following NumPy broadcasting rules.
///
/// Shapes are compared element-wise from the trailing dimensions. Two
/// dimensions are compatible when they are equal, or one of them is 1.
/// The output dimension is the maximum of the two.
///
/// # Errors
/// Returns `KernelError::InvalidShape` if the shapes are not broadcast-compatible.
///
/// # Examples
/// ```ignore
/// assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Ok(vec![3, 4]));
/// assert_eq!(broadcast_shape(&[2, 3], &[3]), Ok(vec![2, 3]));
/// assert!(broadcast_shape(&[2, 3], &[4]).is_err());
/// ```
pub fn broadcast_shape(
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, crate::kernels::KernelError> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Iterate from the trailing dimension backwards.
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(crate::kernels::KernelError::InvalidShape(format!(
                "broadcast_shape: incompatible dimensions {} and {} \
                 (shapes {:?} and {:?})",
                da, db, a, b
            )));
        }
    }

    // We built the result in reverse order (trailing-first), so reverse it.
    result.reverse();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_device() -> &'static objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice> {
        use std::sync::OnceLock;
        static DEVICE: OnceLock<rmlx_metal::MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            crate::test_utils::shared_metal_device().expect("Metal GPU required for tests")
        })
    }

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(compute_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_contiguous_strides(&[5]), vec![1]);
        assert_eq!(compute_contiguous_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_broadcast_shape_basic() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[1], &[5]).unwrap(), vec![5]);
        assert_eq!(broadcast_shape(&[5], &[1]).unwrap(), vec![5]);
        assert_eq!(broadcast_shape(&[5], &[5]).unwrap(), vec![5]);
    }

    #[test]
    fn test_broadcast_shape_different_ndim() {
        assert_eq!(
            broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).unwrap(),
            vec![8, 7, 6, 5]
        );
        assert_eq!(
            broadcast_shape(&[256, 256, 3], &[3]).unwrap(),
            vec![256, 256, 3]
        );
        assert_eq!(broadcast_shape(&[], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[3, 4], &[]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shape_incompatible() {
        assert!(broadcast_shape(&[2, 3], &[4]).is_err());
        assert!(broadcast_shape(&[2, 1], &[8, 4, 3]).is_err());
    }

    #[test]
    fn test_broadcast_shape_scalars() {
        assert_eq!(broadcast_shape(&[], &[]).unwrap(), Vec::<usize>::new());
        assert_eq!(broadcast_shape(&[], &[5]).unwrap(), vec![5]);
    }

    // ── C6: View ops tests ──────────────────────────────────────────────

    #[test]
    fn test_transpose_2d() {
        // Simulate a [3, 4] array with contiguous strides [4, 1]
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![3, 4],
            strides: vec![4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        let t = arr.transpose(0, 1).unwrap();
        assert_eq!(t.shape(), &[4, 3]);
        assert_eq!(t.strides(), &[1, 4]);
    }

    #[test]
    fn test_transpose_3d() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(96, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![2, 3, 4],
            strides: vec![12, 4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        let t = arr.transpose(0, 2).unwrap();
        assert_eq!(t.shape(), &[4, 3, 2]);
        assert_eq!(t.strides(), &[1, 4, 12]);
    }

    #[test]
    fn test_transpose_same_dim_is_noop() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![3, 4],
            strides: vec![4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        let t = arr.transpose(1, 1).unwrap();
        assert_eq!(t.shape(), arr.shape());
        assert_eq!(t.strides(), arr.strides());
    }

    #[test]
    fn test_transpose_out_of_range() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![3, 4],
            strides: vec![4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        assert!(arr.transpose(0, 5).is_err());
        assert!(arr.transpose(5, 0).is_err());
    }

    #[test]
    fn test_squeeze_dim() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![1, 3, 4],
            strides: vec![12, 4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        let s = arr.squeeze_dim(0).unwrap();
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(s.strides(), &[4, 1]);
    }

    #[test]
    fn test_squeeze_dim_middle() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![3, 1, 4],
            strides: vec![4, 4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        let s = arr.squeeze_dim(1).unwrap();
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(s.strides(), &[4, 1]);
    }

    #[test]
    fn test_squeeze_dim_non_one_fails() {
        let arr = Array {
            buffer: ManuallyDrop::new(
                test_device()
                    .newBufferWithLength_options(48, MTLResourceOptions::StorageModeShared)
                    .unwrap(),
            ),
            shape: vec![3, 4],
            strides: vec![4, 1],
            dtype: DType::Float32,
            offset: 0,
        };
        assert!(arr.squeeze_dim(0).is_err());
    }

    #[test]
    fn test_view_ops_zero_copy() {
        // All view operations should share the same buffer
        let dev = test_device();
        let arr = Array::from_slice(dev, &[1.0f32; 12], vec![3, 4]);

        let reshaped = arr.reshape(vec![4, 3]).unwrap();
        assert_eq!(
            arr.metal_buffer().gpuAddress(),
            reshaped.metal_buffer().gpuAddress()
        );

        let unsqueezed = arr.unsqueeze(0).unwrap();
        assert_eq!(
            arr.metal_buffer().gpuAddress(),
            unsqueezed.metal_buffer().gpuAddress()
        );
        assert_eq!(unsqueezed.shape(), &[1, 3, 4]);

        let squeezed = unsqueezed.squeeze_dim(0).unwrap();
        assert_eq!(
            arr.metal_buffer().gpuAddress(),
            squeezed.metal_buffer().gpuAddress()
        );
        assert_eq!(squeezed.shape(), &[3, 4]);

        let transposed = arr.transpose(0, 1).unwrap();
        assert_eq!(
            arr.metal_buffer().gpuAddress(),
            transposed.metal_buffer().gpuAddress()
        );
        assert_eq!(transposed.shape(), &[4, 3]);
    }
}
