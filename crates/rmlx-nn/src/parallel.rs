//! Tensor Parallel layers following the Megatron-LM pattern.
//!
//! - [`ColumnParallelLinear`]: partitions output features across ranks.
//!   Forward: local matmul on full input → allgather output across ranks.
//! - [`RowParallelLinear`]: partitions input features across ranks.
//!   Forward: local matmul on sharded input → allreduce sum.

use rmlx_core::array::Array;
#[cfg(feature = "distributed")]
use rmlx_core::dtype::DType;
#[cfg(feature = "distributed")]
use rmlx_core::kernels::KernelRegistry;
#[cfg(feature = "distributed")]
use rmlx_core::ops;

#[cfg(feature = "distributed")]
use rmlx_distributed::group::{DistributedError, Group, ReduceDtype};

#[cfg(feature = "distributed")]
use objc2::runtime::ProtocolObject;
#[cfg(feature = "distributed")]
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice, MTLResource, MTLResourceOptions};

use crate::quantized_linear::{QuantBits, QuantizedLinear};

/// Column-parallel linear layer (Megatron-LM style).
///
/// Weight shape per rank: `[out_features / world_size, in_features]`
/// Forward: local matmul → allgather output across ranks.
#[derive(Debug)]
pub struct ColumnParallelLinear {
    /// Local weight shard: [out_features / world_size, in_features]
    weight: Array,
    /// Optional bias shard: [out_features / world_size]
    #[cfg_attr(not(feature = "distributed"), allow(dead_code))]
    bias: Option<Array>,
    /// Total output features (before sharding)
    #[cfg_attr(not(feature = "distributed"), allow(dead_code))]
    out_features: usize,
    /// Input features (not sharded)
    #[cfg_attr(not(feature = "distributed"), allow(dead_code))]
    in_features: usize,
    /// This rank's index in the TP group
    rank: u32,
    /// Total number of ranks in TP group
    world_size: u32,
}

/// Row-parallel linear layer (Megatron-LM style).
///
/// Weight shape per rank: `[out_features, in_features / world_size]`
/// Forward: local matmul on sharded input → allreduce sum.
#[derive(Debug)]
pub struct RowParallelLinear {
    /// Local weight shard: [out_features, in_features / world_size]
    weight: Array,
    /// Bias (only on rank 0, others add zero)
    #[cfg_attr(not(feature = "distributed"), allow(dead_code))]
    bias: Option<Array>,
    /// Output features (not sharded)
    #[cfg_attr(not(feature = "distributed"), allow(dead_code))]
    out_features: usize,
    /// Total input features (before sharding)
    #[allow(dead_code)]
    in_features: usize,
    /// This rank's index in the TP group
    rank: u32,
    /// Total number of ranks in TP group
    world_size: u32,
}

/// Read f32 values from a potentially non-contiguous 2D Array, respecting strides.
///
/// Returns a contiguous Vec<f32> of shape [rows, cols] in row-major order.
#[cfg(test)]
fn read_f32_strided(arr: &Array) -> Vec<f32> {
    assert_eq!(arr.dtype(), DType::Float32);
    assert_eq!(arr.ndim(), 2);
    let rows = arr.shape()[0];
    let cols = arr.shape()[1];
    let stride0 = arr.strides()[0]; // in elements
    let stride1 = arr.strides()[1]; // in elements
    let base = arr.metal_buffer().contents().as_ptr() as *const u8;
    let offset = arr.offset();
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let elem_offset = offset + (r * stride0 + c * stride1) * 4;
            // SAFETY: Array buffer is CPU-accessible (StorageModeShared) and
            // we stay within the buffer bounds guaranteed by Array construction.
            let val = unsafe {
                let ptr = base.add(elem_offset) as *const f32;
                *ptr
            };
            out.push(val);
        }
    }
    out
}

/// CPU f32 matmul: a @ b^T.
///
/// a: [m, k] row-major, b: [n, k] row-major (transposed for output [m, n]).
/// Returns [m, n] contiguous row-major.
#[cfg(test)]
fn cpu_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p]; // b transposed
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Create an Array from raw bytes with the given dtype, using a DeviceRef.
///
/// This avoids the `&Device` vs `&DeviceRef` mismatch when the device is
/// obtained from `buffer.device()`.
#[cfg(feature = "distributed")]
#[allow(dead_code)] // retained for future use; RowParallel now uses in-place allreduce
fn array_from_raw_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: DType,
) -> Array {
    let ptr = bytes.as_ptr() as *const std::ffi::c_void;
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new_unchecked(ptr as *mut std::ffi::c_void),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .unwrap();
    let numel: usize = shape.iter().product();
    let elem_bytes = dtype.size_of();
    assert_eq!(
        bytes.len(),
        numel * elem_bytes,
        "byte length ({}) does not match shape {:?} * {}",
        bytes.len(),
        shape,
        elem_bytes
    );
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    Array::new(buffer, shape, strides, dtype, 0)
}

/// Interleave rank-major gathered bytes directly into a Metal buffer, returning
/// an Array in batch-major layout — eliminates the intermediate `Vec<u8>`
/// allocation that `array_from_raw_bytes` would require.
///
/// `gathered` layout: `[rank0_all_rows | rank1_all_rows | ...]`  (rank-major)
/// Output layout:     `[row0_shard0 ++ row0_shard1 ++ ... | row1_... ]` (batch-major)
#[cfg(feature = "distributed")]
fn interleave_gathered_into_array(
    device: &ProtocolObject<dyn MTLDevice>,
    gathered: &[u8],
    batch: usize,
    shard_out: usize,
    full_out: usize,
    world: usize,
    dtype: DType,
) -> Array {
    let elem_bytes = dtype.size_of();
    let shard_bytes = shard_out * elem_bytes;
    let row_bytes = full_out * elem_bytes;
    let output_size = batch * row_bytes;

    // Allocate Metal buffer directly (no intermediate Vec)
    let buffer = device
        .newBufferWithLength_options(output_size, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let dst_base = buffer.contents().as_ptr() as *mut u8;

    // Copy gathered data in batch-major order directly into Metal buffer
    for r in 0..batch {
        for rank in 0..world {
            let src_offset = rank * batch * shard_bytes + r * shard_bytes;
            let dst_offset = r * row_bytes + rank * shard_bytes;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gathered.as_ptr().add(src_offset),
                    dst_base.add(dst_offset),
                    shard_bytes,
                );
            }
        }
    }

    let strides = vec![full_out, 1];
    Array::new(buffer, vec![batch, full_out], strides, dtype, 0)
}

/// Element-wise addition: a[i] += b[i] for raw byte slices interpreted as f32.
#[cfg(feature = "distributed")]
fn add_bias_f32(data: &mut [u8], bias: &[u8], rows: usize, cols: usize) {
    assert_eq!(bias.len(), cols * 4);
    for r in 0..rows {
        for c in 0..cols {
            let offset = (r * cols + c) * 4;
            let val = f32::from_ne_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let b = f32::from_ne_bytes([
                bias[c * 4],
                bias[c * 4 + 1],
                bias[c * 4 + 2],
                bias[c * 4 + 3],
            ]);
            let sum = val + b;
            data[offset..offset + 4].copy_from_slice(&sum.to_ne_bytes());
        }
    }
}

/// f16 ↔ f32 conversion helpers for bias addition.
#[cfg(feature = "distributed")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        // zero or subnormal
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // subnormal: 2^-14 * (frac/1024)
        let val = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        // inf or NaN
        f32::from_bits((sign << 31) | 0x7F800000 | if frac != 0 { 0x400000 } else { 0 })
    } else {
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

#[cfg(feature = "distributed")]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | (new_exp as u32) << 10 | (frac >> 13)) as u16
}

/// Element-wise addition: a[i] += b[i] for raw byte slices interpreted as f16.
#[cfg(feature = "distributed")]
fn add_bias_f16(data: &mut [u8], bias: &[u8], rows: usize, cols: usize) {
    assert_eq!(bias.len(), cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let offset = (r * cols + c) * 2;
            let val = f16_to_f32(u16::from_ne_bytes([data[offset], data[offset + 1]]));
            let b = f16_to_f32(u16::from_ne_bytes([bias[c * 2], bias[c * 2 + 1]]));
            let sum = f32_to_f16(val + b);
            data[offset..offset + 2].copy_from_slice(&sum.to_ne_bytes());
        }
    }
}

/// Error type for tensor-parallel layer construction.
#[derive(Debug, Clone)]
pub struct TpError(pub String);

impl std::fmt::Display for TpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TpError: {}", self.0)
    }
}

impl std::error::Error for TpError {}

/// Sharding mode for quantized tensor-parallel layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardMode {
    /// Column-parallel: shard output dimension, allgather.
    ColumnParallel,
    /// Row-parallel: shard input dimension, allreduce.
    RowParallel,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer from a pre-sharded weight.
    ///
    /// `weight` shape: `[out_features / world_size, in_features]`
    /// `bias` shape (if present): `[out_features / world_size]`
    ///
    /// Returns an error if:
    /// - `world_size` is 0
    /// - `full_out_features` is not divisible by `world_size`
    /// - `weight.shape()[0]` does not equal `full_out_features / world_size`
    /// - `weight.shape()[1]` does not equal `full_in_features`
    /// - `weight` is not 2D
    pub fn new(
        weight: Array,
        bias: Option<Array>,
        full_out_features: usize,
        full_in_features: usize,
        rank: u32,
        world_size: u32,
    ) -> Result<Self, TpError> {
        if world_size == 0 {
            return Err(TpError("world_size must be > 0".into()));
        }
        if full_out_features % (world_size as usize) != 0 {
            return Err(TpError(format!(
                "full_out_features ({}) must be divisible by world_size ({})",
                full_out_features, world_size
            )));
        }
        if weight.ndim() != 2 {
            return Err(TpError(format!(
                "weight must be 2D, got {}D",
                weight.ndim()
            )));
        }
        let expected_rows = full_out_features / (world_size as usize);
        if weight.shape()[0] != expected_rows {
            return Err(TpError(format!(
                "weight.shape()[0] ({}) must equal out_features/world_size ({})",
                weight.shape()[0],
                expected_rows
            )));
        }
        if weight.shape()[1] != full_in_features {
            return Err(TpError(format!(
                "weight.shape()[1] ({}) must equal in_features ({})",
                weight.shape()[1],
                full_in_features
            )));
        }
        Ok(Self {
            weight,
            bias,
            out_features: full_out_features,
            in_features: full_in_features,
            rank,
            world_size,
        })
    }

    /// Reference to the local weight shard.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// This rank's index.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Total TP world size.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Shard a full weight `[out_features, in_features]` for column parallelism.
    ///
    /// Returns columns `[rank * shard_size .. (rank+1) * shard_size]` of the
    /// weight matrix, producing shape `[out_features / world_size, in_features]`.
    ///
    /// For column-parallel, we partition output features (rows of the weight matrix).
    /// This uses `slice_columns` on the transposed view: we slice rows of the
    /// original weight, which correspond to output features.
    ///
    /// # Panics
    /// Panics if `out_features` is not divisible by `world_size`.
    pub fn shard_weight(full_weight: &Array, rank: u32, world_size: u32) -> Array {
        assert_eq!(full_weight.ndim(), 2, "shard_weight requires 2D weight");
        let out_features = full_weight.shape()[0];
        let in_features = full_weight.shape()[1];
        assert_eq!(
            out_features % (world_size as usize),
            0,
            "out_features ({}) must be divisible by world_size ({})",
            out_features,
            world_size
        );
        let shard_size = out_features / (world_size as usize);
        let start_row = (rank as usize) * shard_size;

        // Row slicing: offset by start_row * in_features elements,
        // shape [shard_size, in_features], same strides.
        let elem_bytes = full_weight.dtype().size_of();
        let new_offset = full_weight.offset() + start_row * in_features * elem_bytes;
        full_weight.view(
            vec![shard_size, in_features],
            full_weight.strides().to_vec(),
            new_offset,
        )
    }

    /// Forward pass with a communication group (requires `distributed` feature).
    ///
    /// Steps:
    /// 1. GPU matmul: output_local = input @ weight_shard^T  → [batch, out/world_size]
    /// 2. Allgather: concatenate output_local from all ranks → [batch, out_features]
    #[cfg(feature = "distributed")]
    pub fn forward_with_group(
        &self,
        input: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, DistributedError> {
        assert_eq!(input.ndim(), 2, "input must be 2D [batch, in_features]");
        let dtype = input.dtype();
        assert!(
            dtype == DType::Float32 || dtype == DType::Float16,
            "only f32 and f16 supported, got {:?}",
            dtype
        );
        assert_eq!(
            self.weight.dtype(),
            dtype,
            "weight dtype must match input dtype"
        );

        let batch = input.shape()[0];
        let k = input.shape()[1]; // in_features
        assert_eq!(k, self.in_features, "input in_features mismatch");

        let shard_out = self.weight.shape()[0]; // out_features / world_size

        // ── N3 fix: GPU matmul instead of cpu_matmul_f32 ──
        // Transpose weight: [shard_out, k] -> [k, shard_out] via stride swap (zero-copy)
        let w_t = self
            .weight
            .view(vec![k, shard_out], vec![1, k], self.weight.offset());

        // GPU matmul: [batch, k] @ [k, shard_out] -> [batch, shard_out]
        let local_out_arr = ops::matmul::matmul(registry, input, &w_t, queue)
            .map_err(|e| DistributedError::Protocol(format!("GPU matmul failed: {e}")))?;

        // Add local bias if present (GPU add with broadcast)
        let local_out_arr = if let Some(ref bias) = self.bias {
            // Broadcast bias [shard_out] to [batch, shard_out] via binary add
            // Reshape bias to [1, shard_out] for broadcasting
            let bias_2d = bias.view(vec![1, shard_out], vec![shard_out, 1], bias.offset());
            ops::binary::add(registry, &local_out_arr, &bias_2d, queue)
                .map_err(|e| DistributedError::Protocol(format!("GPU bias add failed: {e}")))?
        } else {
            local_out_arr
        };

        // Read result bytes for allgather
        let local_bytes = local_out_arr.to_bytes();

        // Allgather across ranks → rank-major: [rank0_all_rows][rank1_all_rows]...
        // We need batch-major: [row0_rank0_shard ++ row0_rank1_shard ++ ...][row1_...]
        let gathered = group.allgather(local_bytes)?;
        assert_eq!(
            self.world_size as usize,
            group.size(),
            "TP world_size ({}) does not match group size ({})",
            self.world_size,
            group.size()
        );

        let world = self.world_size as usize;

        // Interleave rank-major gathered data directly into a Metal buffer,
        // skipping the intermediate Vec<u8> allocation.
        let result = interleave_gathered_into_array(
            &input.metal_buffer().device(),
            &gathered,
            batch,
            shard_out,
            self.out_features,
            world,
            dtype,
        );
        Ok(result)
    }
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer from a pre-sharded weight.
    ///
    /// `weight` shape: `[out_features, in_features / world_size]`
    /// `bias` shape (if present): `[out_features]`
    ///
    /// Returns an error if:
    /// - `world_size` is 0
    /// - `full_in_features` is not divisible by `world_size`
    /// - `weight.shape()[0]` does not equal `full_out_features`
    /// - `weight.shape()[1]` does not equal `full_in_features / world_size`
    /// - `weight` is not 2D
    pub fn new(
        weight: Array,
        bias: Option<Array>,
        full_out_features: usize,
        full_in_features: usize,
        rank: u32,
        world_size: u32,
    ) -> Result<Self, TpError> {
        if world_size == 0 {
            return Err(TpError("world_size must be > 0".into()));
        }
        if full_in_features % (world_size as usize) != 0 {
            return Err(TpError(format!(
                "full_in_features ({}) must be divisible by world_size ({})",
                full_in_features, world_size
            )));
        }
        if weight.ndim() != 2 {
            return Err(TpError(format!(
                "weight must be 2D, got {}D",
                weight.ndim()
            )));
        }
        if weight.shape()[0] != full_out_features {
            return Err(TpError(format!(
                "weight.shape()[0] ({}) must equal out_features ({})",
                weight.shape()[0],
                full_out_features
            )));
        }
        let expected_cols = full_in_features / (world_size as usize);
        if weight.shape()[1] != expected_cols {
            return Err(TpError(format!(
                "weight.shape()[1] ({}) must equal in_features/world_size ({})",
                weight.shape()[1],
                expected_cols
            )));
        }
        Ok(Self {
            weight,
            bias,
            out_features: full_out_features,
            in_features: full_in_features,
            rank,
            world_size,
        })
    }

    /// Reference to the local weight shard.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// This rank's index.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Total TP world size.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Shard a full weight `[out_features, in_features]` for row parallelism.
    ///
    /// Returns columns `[rank * shard_size .. (rank+1) * shard_size]` of the
    /// weight matrix, producing shape `[out_features, in_features / world_size]`.
    ///
    /// For row-parallel, we partition input features (columns of the weight matrix).
    ///
    /// # Panics
    /// Panics if `in_features` is not divisible by `world_size`.
    pub fn shard_weight(full_weight: &Array, rank: u32, world_size: u32) -> Array {
        assert_eq!(full_weight.ndim(), 2, "shard_weight requires 2D weight");
        let in_features = full_weight.shape()[1];
        assert_eq!(
            in_features % (world_size as usize),
            0,
            "in_features ({}) must be divisible by world_size ({})",
            in_features,
            world_size
        );
        let shard_size = in_features / (world_size as usize);
        let start_col = (rank as usize) * shard_size;
        let end_col = start_col + shard_size;

        // Column slicing via slice_columns
        full_weight.slice_columns(start_col, end_col)
    }

    /// Forward pass with a communication group (requires `distributed` feature).
    ///
    /// Steps:
    /// 1. GPU matmul: output_partial = sharded_input @ weight_shard^T → [batch, out]
    /// 2. Allreduce sum: output = sum of output_partial across all ranks
    /// 3. Add bias (after allreduce, all ranks have same result)
    #[cfg(feature = "distributed")]
    pub fn forward_with_group(
        &self,
        input: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, DistributedError> {
        assert_eq!(
            input.ndim(),
            2,
            "input must be 2D [batch, in_features/world_size]"
        );
        let dtype = input.dtype();
        assert!(
            dtype == DType::Float32 || dtype == DType::Float16,
            "only f32 and f16 supported, got {:?}",
            dtype
        );
        assert_eq!(
            self.weight.dtype(),
            dtype,
            "weight dtype must match input dtype"
        );

        let batch = input.shape()[0];
        let shard_in = input.shape()[1]; // in_features / world_size
        assert_eq!(
            shard_in,
            self.weight.shape()[1],
            "input shard width must match weight columns"
        );
        let n = self.out_features;

        // ── N3 fix: GPU matmul instead of cpu_matmul_f32 ──
        // Transpose weight: [out, shard_in] -> [shard_in, out] via stride swap (zero-copy)
        let w_t = self
            .weight
            .view(vec![shard_in, n], vec![1, shard_in], self.weight.offset());

        // GPU matmul: [batch, shard_in] @ [shard_in, out] -> [batch, out]
        let mut local_out_arr = ops::matmul::matmul(registry, input, &w_t, queue)
            .map_err(|e| DistributedError::Protocol(format!("GPU matmul failed: {e}")))?;

        // Use dtype-native allreduce — f16 data is sent directly over RDMA (no f32 expansion).
        let reduce_dtype = match dtype {
            DType::Float16 => ReduceDtype::F16,
            DType::Float32 => ReduceDtype::F32,
            _ => ReduceDtype::F32,
        };

        // In-place allreduce: mutate the Metal buffer directly, avoiding a
        // second buffer allocation + memcpy. Safe because:
        // 1. `matmul` uses ExecMode::Sync (waitUntilCompleted) — GPU is done.
        // 2. `local_out_arr` is a fresh buffer owned solely by this stack frame.
        // 3. Apple UMA: StorageModeShared buffers are CPU-accessible.
        {
            let reduce_buf = local_out_arr.to_bytes_mut();
            group.allreduce_in_place(reduce_buf, reduce_dtype)?;
        }

        // Add bias after allreduce (all ranks have the same result)
        if let Some(ref bias) = self.bias {
            let bias_data = bias.to_bytes();
            let buf = local_out_arr.to_bytes_mut();
            match dtype {
                DType::Float16 => add_bias_f16(buf, bias_data, batch, n),
                DType::Float32 => add_bias_f32(buf, bias_data, batch, n),
                _ => {}
            }
        }

        // Return the original Array — its Metal buffer now holds the reduced
        // (and bias-added) result. No new allocation needed.
        Ok(local_out_arr)
    }
}

// ---------------------------------------------------------------------------
// Quantized Tensor-Parallel Layers
// ---------------------------------------------------------------------------

/// Column-parallel quantized linear: shards output dimension across ranks.
///
/// Each rank holds `w_packed[out_features/world_size, in_features]` with matching
/// scales/biases. Forward: local QMV/QMM -> allgather output across ranks.
#[cfg(feature = "distributed")]
pub struct QuantizedColumnParallelLinear {
    /// Sharded quantized weights for this rank.
    pub ql: QuantizedLinear,
    /// This rank's index in the TP group.
    pub rank: u32,
    /// Total number of ranks in TP group.
    pub world_size: u32,
}

/// Row-parallel quantized linear: shards input dimension across ranks.
///
/// Each rank holds `w_packed[out_features, in_features/world_size]` with matching
/// scales/biases. Forward: local QMV/QMM on sharded input -> allreduce sum.
#[cfg(feature = "distributed")]
pub struct QuantizedRowParallelLinear {
    /// Sharded quantized weights for this rank.
    pub ql: QuantizedLinear,
    /// This rank's index in the TP group.
    pub rank: u32,
    /// Total number of ranks in TP group.
    pub world_size: u32,
}

/// Trait for quantized parallel forward pass.
#[cfg(feature = "distributed")]
pub trait QuantizedParallelForward {
    /// Forward pass: quantized matmul + collective communication.
    fn forward(
        &self,
        x: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, TpError>;
}

#[cfg(feature = "distributed")]
impl QuantizedColumnParallelLinear {
    /// Create from a pre-sharded `QuantizedLinear`.
    pub fn new(ql: QuantizedLinear, rank: u32, world_size: u32) -> Self {
        Self {
            ql,
            rank,
            world_size,
        }
    }

    /// Forward: local quantized matmul -> allgather across ranks.
    pub fn forward(
        &self,
        x: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, TpError> {
        // 1. Local QMV/QMM
        let local_out = self
            .ql
            .forward(x, registry, queue)
            .map_err(|e| TpError(format!("quantized forward failed: {e}")))?;

        // 2. Allgather output across ranks
        let local_bytes = local_out.to_bytes();
        let gathered = group
            .allgather(local_bytes)
            .map_err(|e| TpError(format!("allgather failed: {e}")))?;

        let dtype = local_out.dtype();
        let batch = local_out.shape()[0];
        let shard_out = self.ql.out_features();
        let full_out = shard_out * self.world_size as usize;
        let world = self.world_size as usize;

        // Interleave rank-major gathered data directly into a Metal buffer,
        // skipping the intermediate Vec<u8> allocation.
        let result = interleave_gathered_into_array(
            &x.metal_buffer().device(),
            &gathered,
            batch,
            shard_out,
            full_out,
            world,
            dtype,
        );
        Ok(result)
    }
}

#[cfg(feature = "distributed")]
impl QuantizedRowParallelLinear {
    /// Create from a pre-sharded `QuantizedLinear`.
    pub fn new(ql: QuantizedLinear, rank: u32, world_size: u32) -> Self {
        Self {
            ql,
            rank,
            world_size,
        }
    }

    /// Forward: slice input for this rank -> local quantized matmul -> allreduce sum.
    pub fn forward(
        &self,
        x: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, TpError> {
        // 1. Slice input to get this rank's shard
        let shard_in = self.ql.in_features();
        let start = self.rank as usize * shard_in;
        let end = start + shard_in;

        // x is [batch, full_in_features], slice columns [start..end]
        let x_shard = x.slice_columns(start, end);
        // QMM kernels assume contiguous input — slice_columns produces a strided
        // view (strides [full_in, 1]) which is non-contiguous for batch > 1.
        let x_shard = ops::copy::copy(registry, &x_shard, queue)
            .map_err(|e| TpError(format!("contiguous copy failed: {e}")))?;

        // 2. Local QMV/QMM on sharded input
        let mut local_out = self
            .ql
            .forward(&x_shard, registry, queue)
            .map_err(|e| TpError(format!("quantized forward failed: {e}")))?;

        // 3. Allreduce sum across ranks — dtype-native (no f32 expansion)
        // In-place allreduce: mutate the Metal buffer directly, avoiding a
        // second buffer allocation + memcpy. Safe because:
        // 1. QMM forward uses ExecMode::Sync (waitUntilCompleted) — GPU is done.
        // 2. `local_out` is a fresh buffer owned solely by this stack frame.
        // 3. Apple UMA: StorageModeShared buffers are CPU-accessible.
        let dtype = local_out.dtype();
        let reduce_dtype = match dtype {
            DType::Float16 => ReduceDtype::F16,
            DType::Float32 => ReduceDtype::F32,
            _ => ReduceDtype::F32,
        };
        {
            let reduce_buf = local_out.to_bytes_mut();
            group
                .allreduce_in_place(reduce_buf, reduce_dtype)
                .map_err(|e| TpError(format!("allreduce failed: {e}")))?;
        }

        // Return the original Array — its Metal buffer now holds the reduced result.
        Ok(local_out)
    }
}

#[cfg(feature = "distributed")]
impl QuantizedParallelForward for QuantizedColumnParallelLinear {
    fn forward(
        &self,
        x: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, TpError> {
        self.forward(x, group, registry, queue)
    }
}

#[cfg(feature = "distributed")]
impl QuantizedParallelForward for QuantizedRowParallelLinear {
    fn forward(
        &self,
        x: &Array,
        group: &Group,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, TpError> {
        self.forward(x, group, registry, queue)
    }
}

// ---------------------------------------------------------------------------
// Quantized weight sharding helpers
// ---------------------------------------------------------------------------

/// Shard a `QuantizedLinear` along the output dimension (column-parallel).
///
/// Each rank gets rows `[rank * shard_out .. (rank+1) * shard_out]` of the
/// packed weights, scales, and biases.
pub fn shard_quantized_column(
    ql: &QuantizedLinear,
    rank: u32,
    world_size: u32,
) -> Result<QuantizedLinear, TpError> {
    let out = ql.out_features();
    let ws = world_size as usize;
    if out % ws != 0 {
        return Err(TpError(format!(
            "out_features ({out}) must be divisible by world_size ({world_size})"
        )));
    }
    let shard_out = out / ws;
    let start = rank as usize * shard_out;
    let end = start + shard_out;
    let in_f = ql.in_features();

    // w_packed: [out, in_f/pack_factor] row-major
    let pack_factor = match ql.bits() {
        QuantBits::Q4 => 2usize,
        QuantBits::Q8 => 1usize,
    };
    let row_packed = in_f / pack_factor;
    let w_shard: Vec<u8> = ql.w_packed()[start * row_packed..end * row_packed].to_vec();

    // scales/biases: [out, in_f/group_size] row-major
    let scale_cols = in_f / ql.group_size();
    let s_shard: Vec<f32> = ql.scales()[start * scale_cols..end * scale_cols].to_vec();
    let b_shard: Vec<f32> = ql.biases()[start * scale_cols..end * scale_cols].to_vec();

    QuantizedLinear::new(
        w_shard,
        s_shard,
        b_shard,
        in_f,
        shard_out,
        ql.group_size(),
        ql.bits(),
    )
    .map_err(|e| TpError(format!("shard_quantized_column: {e}")))
}

/// Shard a `QuantizedLinear` along the input dimension (row-parallel).
///
/// Each rank gets columns `[rank * shard_in .. (rank+1) * shard_in]` of the
/// packed weights, and the corresponding scale/bias columns.
///
/// `shard_in` must be divisible by `group_size` to maintain quantization group
/// boundaries.
pub fn shard_quantized_row(
    ql: &QuantizedLinear,
    rank: u32,
    world_size: u32,
) -> Result<QuantizedLinear, TpError> {
    let in_f = ql.in_features();
    let ws = world_size as usize;
    if in_f % ws != 0 {
        return Err(TpError(format!(
            "in_features ({in_f}) must be divisible by world_size ({world_size})"
        )));
    }
    let shard_in = in_f / ws;
    if shard_in % ql.group_size() != 0 {
        return Err(TpError(format!(
            "shard_in ({shard_in}) must be divisible by group_size ({})",
            ql.group_size()
        )));
    }
    let start = rank as usize * shard_in;

    let pack_factor = match ql.bits() {
        QuantBits::Q4 => 2usize,
        QuantBits::Q8 => 1usize,
    };
    let full_row = in_f / pack_factor;
    let shard_row = shard_in / pack_factor;
    let col_start = start / pack_factor;

    let out = ql.out_features();
    let mut w_shard = Vec::with_capacity(out * shard_row);
    for r in 0..out {
        let row_offset = r * full_row;
        w_shard.extend_from_slice(
            &ql.w_packed()[row_offset + col_start..row_offset + col_start + shard_row],
        );
    }

    // scales/biases: extract columns for this shard's groups
    let full_scale_cols = in_f / ql.group_size();
    let shard_scale_cols = shard_in / ql.group_size();
    let scale_col_start = start / ql.group_size();

    let mut s_shard = Vec::with_capacity(out * shard_scale_cols);
    let mut b_shard = Vec::with_capacity(out * shard_scale_cols);
    for r in 0..out {
        let row_offset = r * full_scale_cols;
        s_shard.extend_from_slice(
            &ql.scales()
                [row_offset + scale_col_start..row_offset + scale_col_start + shard_scale_cols],
        );
        b_shard.extend_from_slice(
            &ql.biases()
                [row_offset + scale_col_start..row_offset + scale_col_start + shard_scale_cols],
        );
    }

    QuantizedLinear::new(
        w_shard,
        s_shard,
        b_shard,
        shard_in,
        out,
        ql.group_size(),
        ql.bits(),
    )
    .map_err(|e| TpError(format!("shard_quantized_row: {e}")))
}

/// Shard a fused QKV `QuantizedLinear` by segments along the output dimension.
///
/// For fused QKV weights like `[q_out + k_out + v_out, in_features]`, each segment
/// is independently sharded by `world_size`, then the sharded segments are
/// concatenated.
///
/// # Arguments
/// * `segments` -- sizes of each output segment (e.g.,
///   `[q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim]`).
///   Must sum to `ql.out_features()`.
pub fn shard_quantized_column_segments(
    ql: &QuantizedLinear,
    rank: u32,
    world_size: u32,
    segments: &[usize],
) -> Result<QuantizedLinear, TpError> {
    let total_out: usize = segments.iter().sum();
    if total_out != ql.out_features() {
        return Err(TpError(format!(
            "segments sum ({total_out}) != out_features ({})",
            ql.out_features()
        )));
    }
    let ws = world_size as usize;
    for (i, &seg) in segments.iter().enumerate() {
        if seg % ws != 0 {
            return Err(TpError(format!(
                "segment[{i}] ({seg}) must be divisible by world_size ({world_size})"
            )));
        }
    }

    let in_f = ql.in_features();
    let pack_factor = match ql.bits() {
        QuantBits::Q4 => 2usize,
        QuantBits::Q8 => 1usize,
    };
    let row_packed = in_f / pack_factor;
    let scale_cols = in_f / ql.group_size();

    let mut w_shard = Vec::new();
    let mut s_shard = Vec::new();
    let mut b_shard = Vec::new();
    let mut shard_out_total = 0usize;

    let mut row_cursor = 0usize; // tracks starting row in the full weight
    for &seg_size in segments {
        let seg_shard = seg_size / ws;
        let seg_start = row_cursor + rank as usize * seg_shard;
        let seg_end = seg_start + seg_shard;

        // Extract packed weight rows for this segment shard
        w_shard.extend_from_slice(&ql.w_packed()[seg_start * row_packed..seg_end * row_packed]);

        // Extract scales/biases rows for this segment shard
        s_shard.extend_from_slice(&ql.scales()[seg_start * scale_cols..seg_end * scale_cols]);
        b_shard.extend_from_slice(&ql.biases()[seg_start * scale_cols..seg_end * scale_cols]);

        shard_out_total += seg_shard;
        row_cursor += seg_size;
    }

    QuantizedLinear::new(
        w_shard,
        s_shard,
        b_shard,
        in_f,
        shard_out_total,
        ql.group_size(),
        ql.bits(),
    )
    .map_err(|e| TpError(format!("shard_quantized_column_segments: {e}")))
}

/// Factory: shard a `QuantizedLinear` and wrap in a parallel layer.
///
/// # Arguments
/// * `ql` -- Full (un-sharded) quantized linear layer.
/// * `rank` -- This rank's index.
/// * `world_size` -- Total number of TP ranks.
/// * `mode` -- Column or row parallel.
/// * `segments` -- Optional segment sizes for fused QKV column sharding.
#[cfg(feature = "distributed")]
pub fn shard_quantized_linear(
    ql: QuantizedLinear,
    rank: u32,
    world_size: u32,
    mode: ShardMode,
    segments: Option<&[usize]>,
) -> Result<Box<dyn QuantizedParallelForward>, TpError> {
    match mode {
        ShardMode::ColumnParallel => {
            let sharded = if let Some(segs) = segments {
                shard_quantized_column_segments(&ql, rank, world_size, segs)?
            } else {
                shard_quantized_column(&ql, rank, world_size)?
            };
            Ok(Box::new(QuantizedColumnParallelLinear::new(
                sharded, rank, world_size,
            )))
        }
        ShardMode::RowParallel => {
            if segments.is_some() {
                return Err(TpError(
                    "segments not supported for RowParallel sharding".into(),
                ));
            }
            let sharded = shard_quantized_row(&ql, rank, world_size)?;
            Ok(Box::new(QuantizedRowParallelLinear::new(
                sharded, rank, world_size,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_core::dtype::DType;

    use std::sync::OnceLock;

    fn test_device() -> Option<&'static rmlx_metal::MtlDevice> {
        static DEVICE: OnceLock<Option<rmlx_metal::MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| {
                objc2::rc::autoreleasepool(|_| objc2_metal::MTLCreateSystemDefaultDevice())
            })
            .as_ref()
    }

    #[cfg(feature = "distributed")]
    fn setup_registry_and_queue(
        _device: &ProtocolObject<dyn MTLDevice>,
    ) -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("Metal device");
        let queue = gpu.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        rmlx_core::ops::register_all(&registry).expect("register kernels");
        (registry, queue)
    }

    #[test]
    fn test_column_parallel_shard_weight_shapes() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // Full weight: [8, 4] (out_features=8, in_features=4)
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(device, &data, vec![8, 4]);

        // Shard across 2 ranks
        let shard_0 = ColumnParallelLinear::shard_weight(&full_weight, 0, 2);
        let shard_1 = ColumnParallelLinear::shard_weight(&full_weight, 1, 2);

        // Each shard: [4, 4]
        assert_eq!(shard_0.shape(), &[4, 4]);
        assert_eq!(shard_1.shape(), &[4, 4]);

        // Shard across 4 ranks
        for rank in 0..4 {
            let shard = ColumnParallelLinear::shard_weight(&full_weight, rank, 4);
            assert_eq!(shard.shape(), &[2, 4]);
        }
    }

    #[test]
    fn test_column_parallel_shard_weight_data() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // Full weight: [4, 2] = [[0,1],[2,3],[4,5],[6,7]]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(device, &data, vec![4, 2]);

        let shard_0 = ColumnParallelLinear::shard_weight(&full_weight, 0, 2);
        let shard_1 = ColumnParallelLinear::shard_weight(&full_weight, 1, 2);

        // shard_0 = rows [0,1] = [[0,1],[2,3]]
        let vals_0: Vec<f32> = shard_0.to_vec_checked();
        assert_eq!(vals_0, vec![0.0, 1.0, 2.0, 3.0]);

        // shard_1 = rows [2,3] = [[4,5],[6,7]]
        let vals_1: Vec<f32> = shard_1.to_vec_checked();
        assert_eq!(vals_1, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_row_parallel_shard_weight_shapes() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // Full weight: [4, 8] (out_features=4, in_features=8)
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(device, &data, vec![4, 8]);

        // Shard across 2 ranks
        let shard_0 = RowParallelLinear::shard_weight(&full_weight, 0, 2);
        let shard_1 = RowParallelLinear::shard_weight(&full_weight, 1, 2);

        // Each shard: [4, 4]
        assert_eq!(shard_0.shape(), &[4, 4]);
        assert_eq!(shard_1.shape(), &[4, 4]);

        // Shard across 4 ranks
        for rank in 0..4 {
            let shard = RowParallelLinear::shard_weight(&full_weight, rank, 4);
            assert_eq!(shard.shape(), &[4, 2]);
        }
    }

    #[test]
    fn test_column_parallel_new() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let data: Vec<f32> = vec![0.0; 16];
        let weight = Array::from_slice(device, &data, vec![4, 4]);

        let layer = ColumnParallelLinear::new(weight, None, 8, 4, 0, 2).unwrap();
        assert_eq!(layer.rank(), 0);
        assert_eq!(layer.world_size(), 2);
        assert_eq!(layer.weight().shape(), &[4, 4]);
    }

    #[test]
    fn test_row_parallel_new() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let data: Vec<f32> = vec![0.0; 16];
        let weight = Array::from_slice(device, &data, vec![4, 4]);

        let layer = RowParallelLinear::new(weight, None, 4, 8, 1, 2).unwrap();
        assert_eq!(layer.rank(), 1);
        assert_eq!(layer.world_size(), 2);
        assert_eq!(layer.weight().shape(), &[4, 4]);
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::from_slice(device, &data, vec![2, 3]);

        let bytes = arr.to_bytes();
        assert_eq!(bytes.len(), 24); // 6 * 4 bytes

        let arr2 = Array::from_bytes(device, bytes, vec![2, 3], DType::Float32);
        let vals: Vec<f32> = arr2.to_vec_checked();
        assert_eq!(vals, data);
    }

    #[test]
    fn test_slice_columns() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // [2, 4] = [[0,1,2,3],[4,5,6,7]]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let arr = Array::from_slice(device, &data, vec![2, 4]);

        // Slice columns [1..3) → [2, 2] view
        let sliced = arr.slice_columns(1, 3);
        assert_eq!(sliced.shape(), &[2, 2]);
        // Strides should remain [4, 1] (non-contiguous view)
        assert_eq!(sliced.strides(), &[4, 1]);
    }

    // ─── Forward pass tests (requires distributed feature) ───

    #[cfg(feature = "distributed")]
    #[test]
    fn test_cpu_matmul_f32_basic() {
        // 2x3 @ 2x3^T → 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]
        let result = super::cpu_matmul_f32(&a, &b, 2, 3, 2);
        // Row 0: [1,2,3]@[1,0,0]=1, [1,2,3]@[0,1,0]=2
        // Row 1: [4,5,6]@[1,0,0]=4, [4,5,6]@[0,1,0]=5
        assert_eq!(result, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_read_f32_strided_non_contiguous() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // [2, 4] = [[0,1,2,3],[4,5,6,7]]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let arr = Array::from_slice(device, &data, vec![2, 4]);

        // Slice columns [1..3) → [[1,2],[5,6]] with strides [4,1]
        let sliced = arr.slice_columns(1, 3);
        let vals = super::read_f32_strided(&sliced);
        assert_eq!(vals, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_column_parallel_forward_single_rank() {
        use rmlx_distributed::group::Group;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // weight: [2, 3] = [[1,0,0],[0,1,0]]  (identity-like, out=2, in=3)
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let weight = Array::from_slice(device, &weight_data, vec![2, 3]);

        // Single rank: world_size=1, out_features=2, in_features=3
        let layer = ColumnParallelLinear::new(weight, None, 2, 3, 0, 1).unwrap();

        // input: [2, 3] = [[1,2,3],[4,5,6]]
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Array::from_slice(device, &input_data, vec![2, 3]);

        let group = Group::world(1, 0).unwrap();
        let result = layer
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let vals: Vec<f32> = result.to_vec_checked();
        // Row 0: [1,2,3] @ [1,0,0]^T = 1, [1,2,3] @ [0,1,0]^T = 2
        // Row 1: [4,5,6] @ [1,0,0]^T = 4, [4,5,6] @ [0,1,0]^T = 5
        assert_eq!(vals, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_column_parallel_forward_with_bias() {
        use rmlx_distributed::group::Group;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // weight: [2, 2] = [[1,0],[0,1]] (identity)
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let weight = Array::from_slice(device, &weight_data, vec![2, 2]);

        // bias: [2] = [10, 20]
        let bias_data: Vec<f32> = vec![10.0, 20.0];
        let bias = Array::from_slice(device, &bias_data, vec![2]);

        let layer = ColumnParallelLinear::new(weight, Some(bias), 2, 2, 0, 1).unwrap();

        // input: [1, 2] = [[3, 7]]
        let input_data: Vec<f32> = vec![3.0, 7.0];
        let input = Array::from_slice(device, &input_data, vec![1, 2]);

        let group = Group::world(1, 0).unwrap();
        let result = layer
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        let vals: Vec<f32> = result.to_vec_checked();
        // [3,7] @ [[1,0],[0,1]]^T = [3,7], + bias [10,20] = [13, 27]
        assert_eq!(vals, vec![13.0, 27.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_row_parallel_forward_single_rank() {
        use rmlx_distributed::group::Group;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // weight: [2, 3] = [[1,2,3],[4,5,6]]  (out=2, in=3)
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = Array::from_slice(device, &weight_data, vec![2, 3]);

        // Single rank: world_size=1
        let layer = RowParallelLinear::new(weight, None, 2, 3, 0, 1).unwrap();

        // input: [1, 3] = [[1, 1, 1]]
        let input_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let input = Array::from_slice(device, &input_data, vec![1, 3]);

        let group = Group::world(1, 0).unwrap();
        let result = layer
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        let vals: Vec<f32> = result.to_vec_checked();
        // [1,1,1] @ [[1,2,3],[4,5,6]]^T = [1+2+3, 4+5+6] = [6, 15]
        assert_eq!(vals, vec![6.0, 15.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_row_parallel_forward_with_bias() {
        use rmlx_distributed::group::Group;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // weight: [2, 2] = [[1,0],[0,1]] (identity)
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let weight = Array::from_slice(device, &weight_data, vec![2, 2]);

        // bias: [2] = [100, 200]
        let bias_data: Vec<f32> = vec![100.0, 200.0];
        let bias = Array::from_slice(device, &bias_data, vec![2]);

        let layer = RowParallelLinear::new(weight, Some(bias), 2, 2, 0, 1).unwrap();

        // input: [1, 2] = [[5, 3]]
        let input_data: Vec<f32> = vec![5.0, 3.0];
        let input = Array::from_slice(device, &input_data, vec![1, 2]);

        let group = Group::world(1, 0).unwrap();
        let result = layer
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        let vals: Vec<f32> = result.to_vec_checked();
        // [5,3] @ [[1,0],[0,1]]^T = [5,3], + bias [100,200] = [105, 203]
        assert_eq!(vals, vec![105.0, 203.0]);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_column_parallel_forward_mock_2rank() {
        // Simulate 2-rank TP by computing each rank's local output separately
        // then verifying the concatenation matches full matmul.

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // Full weight: [4, 2] = [[1,0],[0,1],[2,0],[0,2]]
        let full_w: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let full_weight = Array::from_slice(device, &full_w, vec![4, 2]);

        // Input: [1, 2] = [[3, 5]]
        let input_data: Vec<f32> = vec![3.0, 5.0];
        let input = Array::from_slice(device, &input_data, vec![1, 2]);

        // Each rank runs as single-rank group to get local matmul output
        let group = rmlx_distributed::group::Group::world(1, 0).unwrap();

        // Rank 0: shard rows [0..2] = [[1,0],[0,1]]
        let shard_0 = ColumnParallelLinear::shard_weight(&full_weight, 0, 2);
        let layer_0 = ColumnParallelLinear::new(shard_0, None, 2, 2, 0, 1).unwrap();
        let out_0 = layer_0
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();
        let vals_0: Vec<f32> = out_0.to_vec_checked();
        assert_eq!(vals_0, vec![3.0, 5.0]); // [3,5] @ I = [3,5]

        // Rank 1: shard rows [2..4] = [[2,0],[0,2]]
        let shard_1 = ColumnParallelLinear::shard_weight(&full_weight, 1, 2);
        let layer_1 = ColumnParallelLinear::new(shard_1, None, 2, 2, 0, 1).unwrap();
        let out_1 = layer_1
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();
        let vals_1: Vec<f32> = out_1.to_vec_checked();
        assert_eq!(vals_1, vec![6.0, 10.0]); // [3,5] @ 2I = [6,10]

        // After allgather, result = concat([3,5], [6,10]) = [3,5,6,10]
        let mut full_result = vals_0;
        full_result.extend_from_slice(&vals_1);

        // Verify against full matmul
        let expected = super::cpu_matmul_f32(&input_data, &full_w, 1, 2, 4);
        assert_eq!(full_result, expected);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_row_parallel_forward_mock_2rank() {
        // Simulate 2-rank TP for RowParallelLinear.
        // Full output = sum of per-rank partial matmul results.

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // Full weight: [2, 4] = [[1,2,3,4],[5,6,7,8]]
        let full_w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let _full_weight = Array::from_slice(device, &full_w, vec![2, 4]);

        // Full input: [1, 4] = [[1, 2, 3, 4]]
        let full_input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Expected: [1,2,3,4] @ [[1,2,3,4],[5,6,7,8]]^T = [30, 70]
        let expected = super::cpu_matmul_f32(&full_input, &full_w, 1, 4, 2);
        assert_eq!(expected, vec![30.0, 70.0]);

        let group = rmlx_distributed::group::Group::world(1, 0).unwrap();

        // Rank 0: weight cols [0..2] = [[1,2],[5,6]], input = [1,2]
        // Create contiguous shard directly (shard_weight returns a strided view
        // which the GPU transpose-view path doesn't handle correctly).
        let shard_0 = Array::from_slice(device, &[1.0f32, 2.0, 5.0, 6.0], vec![2, 2]);

        let input_0 = Array::from_slice(device, &[1.0f32, 2.0], vec![1, 2]);
        let layer_0 = RowParallelLinear::new(shard_0, None, 2, 4, 0, 2).unwrap();
        let out_0 = layer_0
            .forward_with_group(&input_0, &group, &registry, &queue)
            .unwrap();
        let vals_0: Vec<f32> = out_0.to_vec_checked();
        assert_eq!(vals_0, vec![5.0, 17.0]); // [1,2]@[[1,2],[5,6]]^T

        // Rank 1: weight cols [2..4] = [[3,4],[7,8]], input = [3,4]
        let shard_1 = Array::from_slice(device, &[3.0f32, 4.0, 7.0, 8.0], vec![2, 2]);

        let input_1 = Array::from_slice(device, &[3.0f32, 4.0], vec![1, 2]);
        let layer_1 = RowParallelLinear::new(shard_1, None, 2, 4, 1, 2).unwrap();
        let out_1 = layer_1
            .forward_with_group(&input_1, &group, &registry, &queue)
            .unwrap();
        let vals_1: Vec<f32> = out_1.to_vec_checked();
        assert_eq!(vals_1, vec![25.0, 53.0]); // [3,4]@[[3,4],[7,8]]^T

        // After allreduce sum: [5+25, 17+53] = [30, 70]
        let summed: Vec<f32> = vals_0
            .iter()
            .zip(vals_1.iter())
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(summed, expected);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_column_parallel_forward_multibatch_single_rank() {
        use rmlx_distributed::group::Group;

        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let (registry, queue) = setup_registry_and_queue(device);

        // weight: [2, 3] = [[1,0,0],[0,1,0]]  (out=2, in=3)
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let weight = Array::from_slice(device, &weight_data, vec![2, 3]);

        // Single rank, but batch=3 to exercise the interleaving path
        let layer = ColumnParallelLinear::new(weight, None, 2, 3, 0, 1).unwrap();

        // input: [3, 3] — three batch rows
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];
        let input = Array::from_slice(device, &input_data, vec![3, 3]);

        let group = Group::world(1, 0).unwrap();
        let result = layer
            .forward_with_group(&input, &group, &registry, &queue)
            .unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        let vals: Vec<f32> = result.to_vec_checked();
        // Row 0: [1,2,3] @ [1,0,0]^T=1, [1,2,3] @ [0,1,0]^T=2
        // Row 1: [4,5,6] @ [1,0,0]^T=4, [4,5,6] @ [0,1,0]^T=5
        // Row 2: [7,8,9] @ [1,0,0]^T=7, [7,8,9] @ [0,1,0]^T=8
        assert_eq!(vals, vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);
    }

    // ─── Validation tests ───

    #[test]
    fn test_column_parallel_rejects_zero_world_size() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let weight = Array::from_slice(device, &[0.0f32; 4], vec![2, 2]);
        let result = ColumnParallelLinear::new(weight, None, 2, 2, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("world_size"));
    }

    #[test]
    fn test_column_parallel_rejects_indivisible_out_features() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // out_features=3, world_size=2 -> not divisible
        let weight = Array::from_slice(device, &[0.0f32; 4], vec![2, 2]);
        let result = ColumnParallelLinear::new(weight, None, 3, 2, 0, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("divisible"));
    }

    #[test]
    fn test_column_parallel_rejects_mismatched_weight_rows() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // out_features=4, world_size=2 -> expected rows=2, but weight is [3, 2]
        let weight = Array::from_slice(device, &[0.0f32; 6], vec![3, 2]);
        let result = ColumnParallelLinear::new(weight, None, 4, 2, 0, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("shape"));
    }

    #[test]
    fn test_column_parallel_rejects_mismatched_weight_cols() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // weight [2, 3] but in_features=4
        let weight = Array::from_slice(device, &[0.0f32; 6], vec![2, 3]);
        let result = ColumnParallelLinear::new(weight, None, 2, 4, 0, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("in_features"));
    }

    #[test]
    fn test_row_parallel_rejects_zero_world_size() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let weight = Array::from_slice(device, &[0.0f32; 4], vec![2, 2]);
        let result = RowParallelLinear::new(weight, None, 2, 2, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("world_size"));
    }

    #[test]
    fn test_row_parallel_rejects_indivisible_in_features() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // in_features=3, world_size=2 -> not divisible
        let weight = Array::from_slice(device, &[0.0f32; 4], vec![2, 2]);
        let result = RowParallelLinear::new(weight, None, 2, 3, 0, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("divisible"));
    }

    #[test]
    fn test_row_parallel_rejects_mismatched_weight_rows() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // out_features=2 but weight is [3, 2]
        let weight = Array::from_slice(device, &[0.0f32; 6], vec![3, 2]);
        let result = RowParallelLinear::new(weight, None, 2, 2, 0, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("out_features"));
    }

    #[test]
    fn test_row_parallel_rejects_mismatched_weight_cols() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // in_features=4, world_size=2 -> expected cols=2, but weight is [2, 3]
        let weight = Array::from_slice(device, &[0.0f32; 6], vec![2, 3]);
        let result = RowParallelLinear::new(weight, None, 2, 4, 0, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("in_features/world_size"));
    }

    // ─── Quantized tensor-parallel sharding tests ───

    #[test]
    fn test_shard_quantized_column_basic() {
        // 64x64 Q4 with group_size=32
        let in_f = 64;
        let out_f = 64;
        let group_size = 32;
        let groups_per_row = in_f / group_size; // 2
        let packed_per_row = in_f / 2; // 32

        // Fill w_packed with sequential bytes for verification
        let w_packed: Vec<u8> = (0..out_f * packed_per_row)
            .map(|i| (i % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..out_f * groups_per_row).map(|i| i as f32).collect();
        let biases: Vec<f32> = (0..out_f * groups_per_row)
            .map(|i| i as f32 * 0.1)
            .collect();

        let ql = QuantizedLinear::new(
            w_packed.clone(),
            scales.clone(),
            biases.clone(),
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        // Shard into 2 ranks
        let shard_0 = super::shard_quantized_column(&ql, 0, 2).unwrap();
        let shard_1 = super::shard_quantized_column(&ql, 1, 2).unwrap();

        // Each shard should have out_features/2 = 32
        assert_eq!(shard_0.out_features(), 32);
        assert_eq!(shard_0.in_features(), 64);
        assert_eq!(shard_1.out_features(), 32);
        assert_eq!(shard_1.in_features(), 64);

        // Verify packed weights: shard_0 gets first 32 rows, shard_1 gets last 32 rows
        assert_eq!(shard_0.w_packed().len(), 32 * packed_per_row);
        assert_eq!(shard_0.w_packed(), &w_packed[..32 * packed_per_row]);
        assert_eq!(shard_1.w_packed(), &w_packed[32 * packed_per_row..]);

        // Verify scales
        assert_eq!(shard_0.scales().len(), 32 * groups_per_row);
        assert_eq!(shard_0.scales(), &scales[..32 * groups_per_row]);
        assert_eq!(shard_1.scales(), &scales[32 * groups_per_row..]);

        // Verify biases
        assert_eq!(shard_0.biases(), &biases[..32 * groups_per_row]);
        assert_eq!(shard_1.biases(), &biases[32 * groups_per_row..]);
    }

    #[test]
    fn test_shard_quantized_row_basic() {
        // 64x64 Q4 with group_size=32
        let in_f = 64;
        let out_f = 64;
        let group_size = 32;
        let groups_per_row = in_f / group_size; // 2
        let packed_per_row = in_f / 2; // 32

        let w_packed: Vec<u8> = (0..out_f * packed_per_row)
            .map(|i| (i % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..out_f * groups_per_row).map(|i| i as f32).collect();
        let biases: Vec<f32> = (0..out_f * groups_per_row)
            .map(|i| i as f32 * 0.1)
            .collect();

        let ql = QuantizedLinear::new(
            w_packed.clone(),
            scales.clone(),
            biases.clone(),
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        // Shard into 2 ranks (split input dim)
        let shard_0 = super::shard_quantized_row(&ql, 0, 2).unwrap();
        let shard_1 = super::shard_quantized_row(&ql, 1, 2).unwrap();

        // Each shard: full out_features, half in_features
        assert_eq!(shard_0.out_features(), 64);
        assert_eq!(shard_0.in_features(), 32);
        assert_eq!(shard_1.out_features(), 64);
        assert_eq!(shard_1.in_features(), 32);

        // Verify packed weight dimensions
        let shard_packed_per_row = 32 / 2; // 16 bytes per row
        assert_eq!(shard_0.w_packed().len(), 64 * shard_packed_per_row);

        // Verify first row of shard_0 is first half of first row of original
        assert_eq!(
            &shard_0.w_packed()[..shard_packed_per_row],
            &w_packed[..shard_packed_per_row]
        );
        // First row of shard_1 is second half of first row of original
        assert_eq!(
            &shard_1.w_packed()[..shard_packed_per_row],
            &w_packed[shard_packed_per_row..packed_per_row]
        );

        // Verify scales: shard_0 gets first group of each row, shard_1 gets second
        // shard_in=32, group_size=32 => 1 scale column per row per shard
        assert_eq!(shard_0.scales().len(), 64); // 64 rows * 1 group
        assert_eq!(shard_1.scales().len(), 64);

        // For row 0: scales[0..2] in original => shard_0 gets scales[0], shard_1 gets scales[1]
        assert_eq!(shard_0.scales()[0], scales[0]);
        assert_eq!(shard_1.scales()[0], scales[1]);
    }

    #[test]
    fn test_shard_quantized_row_group_size_boundary() {
        // shard_in not divisible by group_size should error
        let in_f = 128;
        let out_f = 32;
        let group_size = 64;
        let packed_per_row = in_f / 2;
        let groups_per_row = in_f / group_size;

        let ql = QuantizedLinear::new(
            vec![0u8; out_f * packed_per_row],
            vec![1.0f32; out_f * groups_per_row],
            vec![0.0f32; out_f * groups_per_row],
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        // world_size=4 => shard_in=32, but group_size=64 => error
        let result = super::shard_quantized_row(&ql, 0, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("divisible by group_size"));
    }

    #[test]
    fn test_shard_quantized_column_not_divisible() {
        // out_features not divisible by world_size
        let in_f = 64;
        let out_f = 48; // not divisible by 5
        let group_size = 32;
        let packed_per_row = in_f / 2;
        let groups_per_row = in_f / group_size;

        let ql = QuantizedLinear::new(
            vec![0u8; out_f * packed_per_row],
            vec![1.0f32; out_f * groups_per_row],
            vec![0.0f32; out_f * groups_per_row],
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        let result = super::shard_quantized_column(&ql, 0, 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("divisible by world_size"));
    }

    #[test]
    fn test_shard_quantized_column_segments() {
        // Fused QKV: [96, 64] with segments [32, 32, 32]
        let in_f = 64;
        let out_f = 96; // 32 + 32 + 32
        let group_size = 32;
        let packed_per_row = in_f / 2; // 32
        let groups_per_row = in_f / group_size; // 2

        let w_packed: Vec<u8> = (0..out_f * packed_per_row)
            .map(|i| (i % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..out_f * groups_per_row).map(|i| i as f32).collect();
        let biases: Vec<f32> = (0..out_f * groups_per_row)
            .map(|i| i as f32 * 0.1)
            .collect();

        let ql = QuantizedLinear::new(
            w_packed.clone(),
            scales.clone(),
            biases.clone(),
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        let segments = [32, 32, 32];

        let shard_0 = super::shard_quantized_column_segments(&ql, 0, 2, &segments).unwrap();
        let shard_1 = super::shard_quantized_column_segments(&ql, 1, 2, &segments).unwrap();

        // Each shard: 3 segments * 16 = 48 output features
        assert_eq!(shard_0.out_features(), 48);
        assert_eq!(shard_0.in_features(), 64);
        assert_eq!(shard_1.out_features(), 48);

        // Verify shard_0 has rows: [0..16] from seg0 ++ [32..48] from seg1 ++ [64..80] from seg2
        let expected_rows_0: Vec<usize> = (0..16).chain(32..48).chain(64..80).collect();
        for (shard_row, &orig_row) in expected_rows_0.iter().enumerate() {
            let shard_slice =
                &shard_0.w_packed()[shard_row * packed_per_row..(shard_row + 1) * packed_per_row];
            let orig_slice = &w_packed[orig_row * packed_per_row..(orig_row + 1) * packed_per_row];
            assert_eq!(shard_slice, orig_slice, "mismatch at shard_row {shard_row}");
        }

        // Verify shard_1 has rows: [16..32] from seg0 ++ [48..64] from seg1 ++ [80..96] from seg2
        let expected_rows_1: Vec<usize> = (16..32).chain(48..64).chain(80..96).collect();
        for (shard_row, &orig_row) in expected_rows_1.iter().enumerate() {
            let shard_slice =
                &shard_1.w_packed()[shard_row * packed_per_row..(shard_row + 1) * packed_per_row];
            let orig_slice = &w_packed[orig_row * packed_per_row..(orig_row + 1) * packed_per_row];
            assert_eq!(shard_slice, orig_slice, "mismatch at shard_row {shard_row}");
        }
    }

    #[test]
    fn test_shard_quantized_column_segments_sum_mismatch() {
        let in_f = 64;
        let out_f = 96;
        let group_size = 32;
        let packed_per_row = in_f / 2;
        let groups_per_row = in_f / group_size;

        let ql = QuantizedLinear::new(
            vec![0u8; out_f * packed_per_row],
            vec![1.0f32; out_f * groups_per_row],
            vec![0.0f32; out_f * groups_per_row],
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        // segments sum != out_features
        let result = super::shard_quantized_column_segments(&ql, 0, 2, &[32, 32]);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("segments sum"));
    }

    #[test]
    fn test_shard_quantized_column_segments_not_divisible() {
        let in_f = 64;
        let out_f = 96;
        let group_size = 32;
        let packed_per_row = in_f / 2;
        let groups_per_row = in_f / group_size;

        let ql = QuantizedLinear::new(
            vec![0u8; out_f * packed_per_row],
            vec![1.0f32; out_f * groups_per_row],
            vec![0.0f32; out_f * groups_per_row],
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        )
        .unwrap();

        // segment 48 not divisible by world_size 5
        let result = super::shard_quantized_column_segments(&ql, 0, 5, &[48, 24, 24]);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("divisible by world_size"));
    }

    #[test]
    fn test_shard_quantized_q8_column() {
        // Test Q8 sharding too
        let in_f = 64;
        let out_f = 32;
        let group_size = 32;
        let groups_per_row = in_f / group_size;
        // Q8: each byte holds 1 value, packed_per_row = in_f
        let packed_per_row = in_f;

        let w_packed: Vec<u8> = (0..out_f * packed_per_row)
            .map(|i| (i % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..out_f * groups_per_row).map(|i| i as f32).collect();
        let biases: Vec<f32> = (0..out_f * groups_per_row)
            .map(|i| i as f32 * 0.1)
            .collect();

        let ql = QuantizedLinear::new(
            w_packed.clone(),
            scales.clone(),
            biases.clone(),
            in_f,
            out_f,
            group_size,
            QuantBits::Q8,
        )
        .unwrap();

        let shard_0 = super::shard_quantized_column(&ql, 0, 2).unwrap();
        assert_eq!(shard_0.out_features(), 16);
        assert_eq!(shard_0.in_features(), 64);
        assert_eq!(shard_0.bits(), QuantBits::Q8);
        assert_eq!(shard_0.w_packed().len(), 16 * packed_per_row);
        assert_eq!(shard_0.w_packed(), &w_packed[..16 * packed_per_row]);
    }

    #[test]
    fn test_shard_mode_enum() {
        assert_ne!(ShardMode::ColumnParallel, ShardMode::RowParallel);
        let mode = ShardMode::ColumnParallel;
        assert_eq!(mode, ShardMode::ColumnParallel);
    }
}
