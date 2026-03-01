//! Tensor Parallel layers following the Megatron-LM pattern.
//!
//! - [`ColumnParallelLinear`]: partitions output features across ranks.
//!   Forward: local matmul on full input → allgather output across ranks.
//! - [`RowParallelLinear`]: partitions input features across ranks.
//!   Forward: local matmul on sharded input → allreduce sum.

use rmlx_core::array::Array;

#[cfg(feature = "distributed")]
use rmlx_distributed::group::Group;

/// Column-parallel linear layer (Megatron-LM style).
///
/// Weight shape per rank: `[out_features / world_size, in_features]`
/// Forward: local matmul → allgather output across ranks.
pub struct ColumnParallelLinear {
    /// Local weight shard: [out_features / world_size, in_features]
    weight: Array,
    /// Optional bias shard: [out_features / world_size]
    bias: Option<Array>,
    /// Total output features (before sharding)
    out_features: usize,
    /// Input features (not sharded)
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
pub struct RowParallelLinear {
    /// Local weight shard: [out_features, in_features / world_size]
    weight: Array,
    /// Bias (only on rank 0, others add zero)
    bias: Option<Array>,
    /// Output features (not sharded)
    out_features: usize,
    /// Total input features (before sharding)
    in_features: usize,
    /// This rank's index in the TP group
    rank: u32,
    /// Total number of ranks in TP group
    world_size: u32,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer from a pre-sharded weight.
    ///
    /// `weight` shape: `[out_features / world_size, in_features]`
    /// `bias` shape (if present): `[out_features / world_size]`
    pub fn new(
        weight: Array,
        bias: Option<Array>,
        full_out_features: usize,
        full_in_features: usize,
        rank: u32,
        world_size: u32,
    ) -> Self {
        Self {
            weight,
            bias,
            out_features: full_out_features,
            in_features: full_in_features,
            rank,
            world_size,
        }
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
        let end_row = start_row + shard_size;

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
    /// 1. Local matmul: output_local = input @ weight_shard^T
    /// 2. Allgather: concatenate output_local from all ranks along feature dim
    #[cfg(feature = "distributed")]
    pub fn forward_with_group(
        &self,
        input: &Array,
        _group: &Group,
    ) -> Result<Array, rmlx_distributed::group::DistributedError> {
        // TODO: wire up actual matmul kernel dispatch
        // let output_local = matmul(input, &self.weight.transpose());
        // let gathered = group.allgather(output_local.to_bytes())?;
        // Array::from_bytes(device, &gathered, [batch, self.out_features], dtype)
        let _ = input;
        Err(rmlx_distributed::group::DistributedError::Transport(
            "ColumnParallelLinear::forward_with_group not yet implemented".to_string(),
        ))
    }
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer from a pre-sharded weight.
    ///
    /// `weight` shape: `[out_features, in_features / world_size]`
    /// `bias` shape (if present): `[out_features]`
    pub fn new(
        weight: Array,
        bias: Option<Array>,
        full_out_features: usize,
        full_in_features: usize,
        rank: u32,
        world_size: u32,
    ) -> Self {
        Self {
            weight,
            bias,
            out_features: full_out_features,
            in_features: full_in_features,
            rank,
            world_size,
        }
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
    /// 1. Local matmul: output_partial = sharded_input @ weight_shard^T
    /// 2. Allreduce sum: output = sum of output_partial across all ranks
    /// 3. Add bias (rank 0 only)
    #[cfg(feature = "distributed")]
    pub fn forward_with_group(
        &self,
        input: &Array,
        _group: &Group,
    ) -> Result<Array, rmlx_distributed::group::DistributedError> {
        // TODO: wire up actual matmul kernel dispatch
        // let output_partial = matmul(sharded_input, &self.weight.transpose());
        // let reduced = group.allreduce(output_partial.to_bytes())?;
        // let output = Array::from_bytes(device, &reduced, [batch, self.out_features], dtype);
        // if self.rank == 0 { add_bias(output, self.bias) }
        let _ = input;
        Err(rmlx_distributed::group::DistributedError::Transport(
            "RowParallelLinear::forward_with_group not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_core::dtype::DType;

    fn test_device() -> metal::Device {
        metal::Device::system_default().expect("no Metal device")
    }

    #[test]
    fn test_column_parallel_shard_weight_shapes() {
        let device = test_device();
        // Full weight: [8, 4] (out_features=8, in_features=4)
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(&device, &data, vec![8, 4]);

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
        let device = test_device();
        // Full weight: [4, 2] = [[0,1],[2,3],[4,5],[6,7]]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(&device, &data, vec![4, 2]);

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
        let device = test_device();
        // Full weight: [4, 8] (out_features=4, in_features=8)
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let full_weight = Array::from_slice(&device, &data, vec![4, 8]);

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
        let device = test_device();
        let data: Vec<f32> = vec![0.0; 16];
        let weight = Array::from_slice(&device, &data, vec![4, 4]);

        let layer = ColumnParallelLinear::new(weight, None, 8, 4, 0, 2);
        assert_eq!(layer.rank(), 0);
        assert_eq!(layer.world_size(), 2);
        assert_eq!(layer.weight().shape(), &[4, 4]);
    }

    #[test]
    fn test_row_parallel_new() {
        let device = test_device();
        let data: Vec<f32> = vec![0.0; 16];
        let weight = Array::from_slice(&device, &data, vec![4, 4]);

        let layer = RowParallelLinear::new(weight, None, 4, 8, 1, 2);
        assert_eq!(layer.rank(), 1);
        assert_eq!(layer.world_size(), 2);
        assert_eq!(layer.weight().shape(), &[4, 4]);
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let device = test_device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::from_slice(&device, &data, vec![2, 3]);

        let bytes = arr.to_bytes();
        assert_eq!(bytes.len(), 24); // 6 * 4 bytes

        let arr2 = Array::from_bytes(&device, bytes, vec![2, 3], DType::Float32);
        let vals: Vec<f32> = arr2.to_vec_checked();
        assert_eq!(vals, data);
    }

    #[test]
    fn test_slice_columns() {
        let device = test_device();
        // [2, 4] = [[0,1,2,3],[4,5,6,7]]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let arr = Array::from_slice(&device, &data, vec![2, 4]);

        // Slice columns [1..3) → [2, 2] view
        let sliced = arr.slice_columns(1, 3);
        assert_eq!(sliced.shape(), &[2, 2]);
        // Strides should remain [4, 1] (non-contiguous view)
        assert_eq!(sliced.strides(), &[4, 1]);
    }
}
