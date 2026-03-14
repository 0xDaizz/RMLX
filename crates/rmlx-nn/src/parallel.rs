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
use rmlx_distributed::group::{DistributedError, Group};

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
#[cfg(feature = "distributed")]
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
#[cfg(feature = "distributed")]
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

        let elem_bytes = dtype.size_of();
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
        let local_bytes = local_out_arr.to_bytes().to_vec();

        // Allgather across ranks → rank-major: [rank0_all_rows][rank1_all_rows]...
        // We need batch-major: [row0_rank0_shard ++ row0_rank1_shard ++ ...][row1_...]
        let gathered = group.allgather(&local_bytes)?;
        assert_eq!(
            self.world_size as usize,
            group.size(),
            "TP world_size ({}) does not match group size ({})",
            self.world_size,
            group.size()
        );

        let world = self.world_size as usize;
        let shard_bytes = shard_out * elem_bytes; // bytes per rank per row
        let row_bytes = self.out_features * elem_bytes; // bytes per output row

        let mut interleaved = vec![0u8; batch * row_bytes];
        for r in 0..batch {
            for rank in 0..world {
                let src_offset = rank * batch * shard_bytes + r * shard_bytes;
                let dst_offset = r * row_bytes + rank * shard_bytes;
                interleaved[dst_offset..dst_offset + shard_bytes]
                    .copy_from_slice(&gathered[src_offset..src_offset + shard_bytes]);
            }
        }

        // Reconstruct Array from interleaved bytes
        let result = array_from_raw_bytes(
            input.metal_buffer().device(),
            &interleaved,
            vec![batch, self.out_features],
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
        let local_out_arr = ops::matmul::matmul(registry, input, &w_t, queue)
            .map_err(|e| DistributedError::Protocol(format!("GPU matmul failed: {e}")))?;

        let local_bytes = local_out_arr.to_bytes().to_vec();

        // Allreduce sum across ranks → [batch, out_features]
        let mut reduced = group.allreduce(&local_bytes)?;

        // Add bias after allreduce (all ranks have the same summed result)
        if let Some(ref bias) = self.bias {
            let bias_data = bias.to_bytes();
            match dtype {
                DType::Float32 => add_bias_f32(&mut reduced, bias_data, batch, n),
                DType::Float16 => add_bias_f16(&mut reduced, bias_data, batch, n),
                _ => unreachable!(),
            }
        }

        // Reconstruct Array from reduced bytes
        let result = array_from_raw_bytes(
            input.metal_buffer().device(),
            &reduced,
            vec![batch, n],
            dtype,
        );
        Ok(result)
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
        let full_weight = Array::from_slice(device, &full_w, vec![2, 4]);

        // Full input: [1, 4] = [[1, 2, 3, 4]]
        let full_input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Expected: [1,2,3,4] @ [[1,2,3,4],[5,6,7,8]]^T = [30, 70]
        let expected = super::cpu_matmul_f32(&full_input, &full_w, 1, 4, 2);
        assert_eq!(expected, vec![30.0, 70.0]);

        let group = rmlx_distributed::group::Group::world(1, 0).unwrap();

        // Rank 0: weight cols [0..2] = [[1,2],[5,6]], input = [1,2]
        let shard_0 = RowParallelLinear::shard_weight(&full_weight, 0, 2);
        let shard_0_vals = super::read_f32_strided(&shard_0);
        assert_eq!(shard_0_vals, vec![1.0, 2.0, 5.0, 6.0]);

        let input_0 = Array::from_slice(device, &[1.0f32, 2.0], vec![1, 2]);
        let layer_0 = RowParallelLinear::new(shard_0, None, 2, 4, 0, 1).unwrap();
        let out_0 = layer_0
            .forward_with_group(&input_0, &group, &registry, &queue)
            .unwrap();
        let vals_0: Vec<f32> = out_0.to_vec_checked();
        assert_eq!(vals_0, vec![5.0, 17.0]); // [1,2]@[[1,2],[5,6]]^T

        // Rank 1: weight cols [2..4] = [[3,4],[7,8]], input = [3,4]
        let shard_1 = RowParallelLinear::shard_weight(&full_weight, 1, 2);
        let shard_1_vals = super::read_f32_strided(&shard_1);
        assert_eq!(shard_1_vals, vec![3.0, 4.0, 7.0, 8.0]);

        let input_1 = Array::from_slice(device, &[3.0f32, 4.0], vec![1, 2]);
        let layer_1 = RowParallelLinear::new(shard_1, None, 2, 4, 1, 1).unwrap();
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
}
