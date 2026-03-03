//! Dynamic shape support for variable sequence lengths.
//!
//! Pre-allocates buffers at maximum size and dispatches with actual dimensions.
//! This avoids buffer reallocation for varying input sizes.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::KernelError;

/// Context for executing with dynamic shapes.
///
/// Pre-allocates intermediate buffers at `max_seq_len` and dispatches
/// kernels with actual sizes, using only the prefix of each buffer.
pub struct DynamicExecContext {
    /// Maximum sequence length for pre-allocation.
    max_seq_len: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// DType for intermediate buffers.
    dtype: DType,
    /// Pre-allocated intermediate buffers.
    intermediates: Vec<Array>,
}

impl DynamicExecContext {
    /// Create a new dynamic execution context.
    ///
    /// Pre-allocates `num_intermediates` buffers of shape `[max_seq_len, hidden_dim]`.
    pub fn new(
        device: &metal::Device,
        max_seq_len: usize,
        hidden_dim: usize,
        dtype: DType,
        num_intermediates: usize,
    ) -> Self {
        let mut intermediates = Vec::with_capacity(num_intermediates);
        for _ in 0..num_intermediates {
            intermediates.push(Array::zeros(device, &[max_seq_len, hidden_dim], dtype));
        }
        Self {
            max_seq_len,
            hidden_dim,
            dtype,
            intermediates,
        }
    }

    /// Maximum sequence length this context supports.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get a view of intermediate buffer `idx` with actual sequence length.
    ///
    /// Returns a view into the pre-allocated buffer with shape
    /// `[actual_seq_len, hidden_dim]`, avoiding reallocation.
    pub fn get_intermediate(
        &self,
        idx: usize,
        actual_seq_len: usize,
    ) -> Result<Array, KernelError> {
        if idx >= self.intermediates.len() {
            return Err(KernelError::InvalidShape(format!(
                "DynamicExecContext: intermediate index {} >= count {}",
                idx,
                self.intermediates.len()
            )));
        }
        if actual_seq_len > self.max_seq_len {
            return Err(KernelError::InvalidShape(format!(
                "DynamicExecContext: actual_seq_len {} > max_seq_len {}",
                actual_seq_len, self.max_seq_len
            )));
        }
        let buf = &self.intermediates[idx];
        // Return a view of the first actual_seq_len rows.
        Ok(buf.view(
            vec![actual_seq_len, self.hidden_dim],
            buf.strides().to_vec(),
            buf.offset(),
        ))
    }

    /// Number of pre-allocated intermediate buffers.
    pub fn num_intermediates(&self) -> usize {
        self.intermediates.len()
    }

    /// Total pre-allocated memory in bytes.
    pub fn allocated_bytes(&self) -> usize {
        self.intermediates.len()
            * self
                .dtype
                .numel_to_bytes(self.max_seq_len * self.hidden_dim)
    }
}
