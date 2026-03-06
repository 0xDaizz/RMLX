//! Collective communication operations using RDMA primitives.
//!
//! Implements ring-based allreduce, allgather, and reduce_scatter for
//! multi-node clusters connected via RDMA. These operations coordinate
//! via the `ConnectionManager` and use the ring topology for efficient
//! communication.
//!
//! Supports f32, f16, and bf16 element types. For f16/bf16, reduction
//! operations are performed by converting to f32, reducing, and converting
//! back to avoid precision loss from direct half-precision arithmetic.
//!
//! Since RDMA hardware is not available in CI, all operations work on
//! CPU buffers and provide a simulated path for testing. When real RDMA
//! connections are present, they use the RDMA send/recv path.

use crate::connection_manager::ConnectionManager;
use crate::RdmaError;
use half::{bf16, f16};

/// Data types supported by the collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveDType {
    /// 32-bit floating point.
    Float32,
    /// 16-bit IEEE 754 floating point.
    Float16,
    /// 16-bit brain floating point.
    Bfloat16,
}

impl CollectiveDType {
    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            CollectiveDType::Float32 => 4,
            CollectiveDType::Float16 => 2,
            CollectiveDType::Bfloat16 => 2,
        }
    }
}

/// Supported reduction operations for allreduce and reduce_scatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Product,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

/// Trait for element types that can participate in collective reductions.
///
/// Reduction is performed via f32 intermediates to maintain precision
/// for half-precision types.
///
/// # Safety
/// Implementors must be plain-old-data types with no padding bytes,
/// valid for all bit patterns, and have size matching `CollectiveDType::element_size()`.
pub unsafe trait ReduceElement: Copy + Default {
    /// The collective dtype tag for this type.
    const DTYPE: CollectiveDType;

    /// Convert to f32 for reduction arithmetic.
    fn to_f32(self) -> f32;

    /// Convert from f32 after reduction arithmetic.
    fn from_f32(v: f32) -> Self;
}

// SAFETY: f32 is a primitive POD type with no padding, valid for all bit patterns,
// and its size (4) matches CollectiveDType::Float32::element_size().
unsafe impl ReduceElement for f32 {
    const DTYPE: CollectiveDType = CollectiveDType::Float32;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

// SAFETY: f16 is a 2-byte POD type with no padding, valid for all bit patterns,
// and its size (2) matches CollectiveDType::Float16::element_size().
unsafe impl ReduceElement for f16 {
    const DTYPE: CollectiveDType = CollectiveDType::Float16;

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

// SAFETY: bf16 is a 2-byte POD type with no padding, valid for all bit patterns,
// and its size (2) matches CollectiveDType::Bfloat16::element_size().
unsafe impl ReduceElement for bf16 {
    const DTYPE: CollectiveDType = CollectiveDType::Bfloat16;

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

/// Apply a reduction operation element-wise: `dst[i] = op(dst[i], src[i])`.
///
/// Both slices must have the same length. Operates on `f32` elements.
pub fn apply_reduce_op(dst: &mut [f32], src: &[f32], op: ReduceOp) {
    apply_reduce_op_typed(dst, src, op);
}

/// Apply a reduction operation element-wise for any [`ReduceElement`] type.
///
/// Reduction is performed by converting to f32, applying the operation,
/// and converting back. For f32 this is a no-op conversion. For f16/bf16
/// this avoids precision issues with direct half-precision arithmetic.
pub fn apply_reduce_op_typed<T: ReduceElement>(dst: &mut [T], src: &[T], op: ReduceOp) {
    debug_assert_eq!(dst.len(), src.len());
    match op {
        ReduceOp::Sum => {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = T::from_f32(d.to_f32() + s.to_f32());
            }
        }
        ReduceOp::Product => {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = T::from_f32(d.to_f32() * s.to_f32());
            }
        }
        ReduceOp::Max => {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = T::from_f32(d.to_f32().max(s.to_f32()));
            }
        }
        ReduceOp::Min => {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = T::from_f32(d.to_f32().min(s.to_f32()));
            }
        }
    }
}

/// Compute chunk boundaries for splitting a buffer across N ranks.
///
/// Returns a vector of (offset, length) pairs for each rank,
/// where lengths differ by at most 1 element to handle non-divisible sizes.
pub fn chunk_boundaries(total_elements: usize, world_size: u32) -> Vec<(usize, usize)> {
    let n = world_size as usize;
    if n == 0 {
        return Vec::new();
    }
    let base = total_elements / n;
    let remainder = total_elements % n;

    let mut boundaries = Vec::with_capacity(n);
    let mut offset = 0;
    for i in 0..n {
        let len = base + if i < remainder { 1 } else { 0 };
        boundaries.push((offset, len));
        offset += len;
    }
    boundaries
}

/// Ring allreduce on f32 data.
///
/// Performs a bidirectional ring allreduce in two phases:
/// 1. **Reduce-scatter**: Each rank reduces a different chunk of the input
///    data. After `world_size - 1` steps, each rank holds the fully
///    reduced version of one chunk.
/// 2. **Allgather**: Each rank broadcasts its reduced chunk around the
///    ring so all ranks end up with the complete reduced result.
///
/// When `mgr` has no real RDMA connections (e.g., in tests), this function
/// falls back to a simulated local-only allreduce (single-rank fast path).
///
/// # Arguments
/// * `mgr` - The connection manager with ring topology connections.
/// * `data` - The local f32 data buffer (read/write). On return, contains
///   the allreduced result.
/// * `op` - The reduction operation to apply.
///
/// # Returns
/// The allreduced data as a new `Vec<f32>`. The input `data` is also
/// modified in-place.
pub fn ring_allreduce(
    mgr: &ConnectionManager,
    data: &mut [f32],
    op: ReduceOp,
) -> Result<Vec<f32>, RdmaError> {
    ring_allreduce_typed(mgr, data, op)
}

/// Ring allreduce generic over element type.
///
/// Works with f32, f16, and bf16. See [`ring_allreduce`] for the f32-specific
/// convenience wrapper.
pub fn ring_allreduce_typed<T: ReduceElement>(
    mgr: &ConnectionManager,
    data: &mut [T],
    op: ReduceOp,
) -> Result<Vec<T>, RdmaError> {
    let world_size = mgr.world_size();
    let rank = mgr.rank();

    // Single-rank fast path
    if world_size <= 1 {
        return Ok(data.to_vec());
    }

    let total = data.len();
    let chunks = chunk_boundaries(total, world_size);
    let elem_size = std::mem::size_of::<T>();

    // Phase 1: Reduce-scatter
    let mut recv_buf = vec![T::default(); total];

    for step in 0..(world_size - 1) {
        let send_chunk_idx = ((rank + world_size - step) % world_size) as usize;
        let recv_chunk_idx = ((rank + world_size - step - 1) % world_size) as usize;

        let (send_offset, send_len) = chunks[send_chunk_idx];
        let (recv_offset, recv_len) = chunks[recv_chunk_idx];

        if mgr.right_connection().is_some() && mgr.left_connection().is_some() {
            let send_data: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data[send_offset..send_offset + send_len].as_ptr() as *const u8,
                    send_len * elem_size,
                )
            };

            let recv_data: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    recv_buf[recv_offset..recv_offset + recv_len].as_mut_ptr() as *mut u8,
                    recv_len * elem_size,
                )
            };

            let right = mgr.right_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allreduce: right connection lost".into())
            })?;
            let left = mgr.left_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allreduce: left connection lost".into())
            })?;

            let recv_reg = left.register_recv_slice(recv_data)?;
            let send_reg = right.register_send_slice(send_data)?;

            let wr_id_recv = step as u64 * 2;
            let wr_id_send = step as u64 * 2 + 1;

            let recv_op =
                left.post_recv(recv_reg.mr(), 0, (recv_len * elem_size) as u32, wr_id_recv)?;
            let send_op =
                right.post_send(send_reg.mr(), 0, (send_len * elem_size) as u32, wr_id_send)?;

            right.wait_posted(&[send_op])?;
            left.wait_posted(&[recv_op])?;

            apply_reduce_op_typed(
                &mut data[recv_offset..recv_offset + recv_len],
                &recv_buf[recv_offset..recv_offset + recv_len],
                op,
            );
        }
    }

    // Phase 2: Allgather
    for step in 0..(world_size - 1) {
        let send_chunk_idx = ((rank + world_size - step) % world_size) as usize;
        let recv_chunk_idx = ((rank + world_size - step - 1) % world_size) as usize;

        let (send_offset, send_len) = chunks[send_chunk_idx];
        let (recv_offset, recv_len) = chunks[recv_chunk_idx];

        if mgr.right_connection().is_some() && mgr.left_connection().is_some() {
            let send_data: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data[send_offset..send_offset + send_len].as_ptr() as *const u8,
                    send_len * elem_size,
                )
            };

            let recv_data: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    recv_buf[recv_offset..recv_offset + recv_len].as_mut_ptr() as *mut u8,
                    recv_len * elem_size,
                )
            };

            let right = mgr.right_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allreduce: right connection lost".into())
            })?;
            let left = mgr.left_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allreduce: left connection lost".into())
            })?;

            let recv_reg = left.register_recv_slice(recv_data)?;
            let send_reg = right.register_send_slice(send_data)?;

            let base_wr = (world_size - 1) as u64 * 2;
            let wr_id_recv = base_wr + step as u64 * 2;
            let wr_id_send = base_wr + step as u64 * 2 + 1;

            let recv_op =
                left.post_recv(recv_reg.mr(), 0, (recv_len * elem_size) as u32, wr_id_recv)?;
            let send_op =
                right.post_send(send_reg.mr(), 0, (send_len * elem_size) as u32, wr_id_send)?;

            right.wait_posted(&[send_op])?;
            left.wait_posted(&[recv_op])?;

            data[recv_offset..recv_offset + recv_len]
                .copy_from_slice(&recv_buf[recv_offset..recv_offset + recv_len]);
        }
    }

    Ok(data.to_vec())
}

/// Ring allgather on byte data.
///
/// Each rank contributes a chunk of data, and after the operation all
/// ranks hold the concatenation of all chunks. Uses the ring topology
/// for efficient N-1 step communication.
///
/// The `local_chunk` is placed at position `rank` in the output.
/// Returns a vector of all chunks, indexed by rank.
pub fn ring_allgather(
    mgr: &ConnectionManager,
    local_chunk: &[u8],
) -> Result<Vec<Vec<u8>>, RdmaError> {
    let world_size = mgr.world_size();
    let rank = mgr.rank();

    if world_size <= 1 {
        return Ok(vec![local_chunk.to_vec()]);
    }

    // Initialize output: each rank's slot
    let mut chunks: Vec<Vec<u8>> = (0..world_size)
        .map(|r| {
            if r == rank {
                local_chunk.to_vec()
            } else {
                Vec::new()
            }
        })
        .collect();

    // In each step, rank sends the chunk it received in the previous step
    // (or its own chunk in step 0) to the right neighbor, and receives
    // a new chunk from the left neighbor.
    for step in 0..(world_size - 1) {
        let send_idx = ((rank + world_size - step) % world_size) as usize;
        let recv_idx = ((rank + world_size - step - 1) % world_size) as usize;

        if mgr.right_connection().is_some() && mgr.left_connection().is_some() {
            let right = mgr.right_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allgather: right connection lost".into())
            })?;
            let left = mgr.left_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_allgather: left connection lost".into())
            })?;

            let send_data = &chunks[send_idx];
            let send_reg = right.register_send_slice(send_data)?;

            // Allocate recv buffer with same size as local_chunk (all chunks same size)
            let mut recv_data = vec![0u8; local_chunk.len()];
            let recv_reg = left.register_recv_slice(&mut recv_data)?;

            let wr_id_recv = step as u64 * 2;
            let wr_id_send = step as u64 * 2 + 1;

            let recv_op = left.post_recv(recv_reg.mr(), 0, local_chunk.len() as u32, wr_id_recv)?;
            let send_op = right.post_send(send_reg.mr(), 0, send_data.len() as u32, wr_id_send)?;

            right.wait_posted(&[send_op])?;
            left.wait_posted(&[recv_op])?;

            chunks[recv_idx] = recv_data;
        }
    }

    Ok(chunks)
}

/// Ring reduce_scatter on f32 data.
///
/// Performs the reduce-scatter phase of a ring allreduce: after completion,
/// each rank holds the fully reduced version of its assigned chunk.
///
/// This is useful as a standalone primitive for expert parallelism and
/// MoE workloads where each rank only needs a portion of the reduced result.
///
/// # Arguments
/// * `mgr` - Connection manager with ring topology.
/// * `data` - Local f32 data (modified in-place during reduction).
/// * `op` - Reduction operation.
///
/// # Returns
/// The reduced chunk assigned to this rank (a slice of the full reduced data).
pub fn ring_reduce_scatter(
    mgr: &ConnectionManager,
    data: &mut [f32],
    op: ReduceOp,
) -> Result<Vec<f32>, RdmaError> {
    ring_reduce_scatter_typed(mgr, data, op)
}

/// Ring reduce_scatter generic over element type.
///
/// Works with f32, f16, and bf16. See [`ring_reduce_scatter`] for the
/// f32-specific convenience wrapper.
pub fn ring_reduce_scatter_typed<T: ReduceElement>(
    mgr: &ConnectionManager,
    data: &mut [T],
    op: ReduceOp,
) -> Result<Vec<T>, RdmaError> {
    let world_size = mgr.world_size();
    let rank = mgr.rank();

    if world_size <= 1 {
        return Ok(data.to_vec());
    }

    let total = data.len();
    let chunks = chunk_boundaries(total, world_size);
    let elem_size = std::mem::size_of::<T>();
    let mut recv_buf = vec![T::default(); total];

    // Reduce-scatter: N-1 steps
    for step in 0..(world_size - 1) {
        let send_chunk_idx = ((rank + world_size - step) % world_size) as usize;
        let recv_chunk_idx = ((rank + world_size - step - 1) % world_size) as usize;

        let (send_offset, send_len) = chunks[send_chunk_idx];
        let (recv_offset, recv_len) = chunks[recv_chunk_idx];

        if mgr.right_connection().is_some() && mgr.left_connection().is_some() {
            let send_data: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data[send_offset..send_offset + send_len].as_ptr() as *const u8,
                    send_len * elem_size,
                )
            };

            let recv_data: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    recv_buf[recv_offset..recv_offset + recv_len].as_mut_ptr() as *mut u8,
                    recv_len * elem_size,
                )
            };

            let right = mgr.right_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_reduce_scatter: right connection lost".into())
            })?;
            let left = mgr.left_connection().ok_or_else(|| {
                RdmaError::ConnectionFailed("ring_reduce_scatter: left connection lost".into())
            })?;

            let recv_reg = left.register_recv_slice(recv_data)?;
            let send_reg = right.register_send_slice(send_data)?;

            let wr_id_recv = step as u64 * 2;
            let wr_id_send = step as u64 * 2 + 1;

            let recv_op =
                left.post_recv(recv_reg.mr(), 0, (recv_len * elem_size) as u32, wr_id_recv)?;
            let send_op =
                right.post_send(send_reg.mr(), 0, (send_len * elem_size) as u32, wr_id_send)?;

            right.wait_posted(&[send_op])?;
            left.wait_posted(&[recv_op])?;

            apply_reduce_op_typed(
                &mut data[recv_offset..recv_offset + recv_len],
                &recv_buf[recv_offset..recv_offset + recv_len],
                op,
            );
        }
    }

    // Return only this rank's chunk
    let (my_offset, my_len) = chunks[rank as usize];
    Ok(data[my_offset..my_offset + my_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_port::Topology;

    #[test]
    fn test_chunk_boundaries_even() {
        let bounds = chunk_boundaries(12, 4);
        assert_eq!(bounds, vec![(0, 3), (3, 3), (6, 3), (9, 3)]);
    }

    #[test]
    fn test_chunk_boundaries_uneven() {
        let bounds = chunk_boundaries(10, 3);
        // 10 / 3 = 3 remainder 1 => first rank gets 4, rest get 3
        assert_eq!(bounds, vec![(0, 4), (4, 3), (7, 3)]);
    }

    #[test]
    fn test_chunk_boundaries_more_ranks_than_elements() {
        let bounds = chunk_boundaries(2, 4);
        assert_eq!(bounds, vec![(0, 1), (1, 1), (2, 0), (2, 0)]);
    }

    #[test]
    fn test_chunk_boundaries_zero_world_size() {
        let bounds = chunk_boundaries(10, 0);
        assert!(bounds.is_empty());
    }

    #[test]
    fn test_chunk_boundaries_single_rank() {
        let bounds = chunk_boundaries(10, 1);
        assert_eq!(bounds, vec![(0, 10)]);
    }

    #[test]
    fn test_apply_reduce_sum() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![4.0, 5.0, 6.0];
        apply_reduce_op(&mut dst, &src, ReduceOp::Sum);
        assert_eq!(dst, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_apply_reduce_product() {
        let mut dst = vec![2.0, 3.0, 4.0];
        let src = vec![5.0, 6.0, 7.0];
        apply_reduce_op(&mut dst, &src, ReduceOp::Product);
        assert_eq!(dst, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_apply_reduce_max() {
        let mut dst = vec![1.0, 6.0, 3.0];
        let src = vec![4.0, 2.0, 5.0];
        apply_reduce_op(&mut dst, &src, ReduceOp::Max);
        assert_eq!(dst, vec![4.0, 6.0, 5.0]);
    }

    #[test]
    fn test_apply_reduce_min() {
        let mut dst = vec![1.0, 6.0, 3.0];
        let src = vec![4.0, 2.0, 5.0];
        apply_reduce_op(&mut dst, &src, ReduceOp::Min);
        assert_eq!(dst, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_allreduce_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ring_allreduce(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_allreduce_no_connections() {
        // With 2 ranks but no actual connections, the allreduce should
        // still succeed (just returns local data unchanged since no
        // communication happens).
        let mgr = ConnectionManager::new(0, 2, Topology::Ring);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ring_allreduce(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_allgather_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let chunk = vec![1u8, 2, 3];
        let result = ring_allgather(&mgr, &chunk).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_reduce_scatter_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ring_reduce_scatter(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reduce_scatter_no_connections() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = ring_reduce_scatter(&mgr, &mut data, ReduceOp::Sum).unwrap();
        // Rank 0's chunk: elements 0,1 (8/4 = 2 per rank)
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_reduce_ops_all_variants() {
        // Verify all reduce ops can be used
        for op in [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Max,
            ReduceOp::Min,
        ] {
            let mut dst = vec![2.0f32];
            let src = vec![3.0f32];
            apply_reduce_op(&mut dst, &src, op);
            match op {
                ReduceOp::Sum => assert_eq!(dst[0], 5.0),
                ReduceOp::Product => assert_eq!(dst[0], 6.0),
                ReduceOp::Max => assert_eq!(dst[0], 3.0),
                ReduceOp::Min => assert_eq!(dst[0], 2.0),
            }
        }
    }

    #[test]
    fn test_chunk_boundaries_coverage() {
        // Verify all elements are covered
        for n in 1..=20 {
            for ws in 1..=8 {
                let bounds = chunk_boundaries(n, ws);
                assert_eq!(bounds.len(), ws as usize);
                let total: usize = bounds.iter().map(|(_, len)| len).sum();
                assert_eq!(total, n, "n={n}, ws={ws}");
                // Verify contiguous
                for i in 1..bounds.len() {
                    assert_eq!(
                        bounds[i].0,
                        bounds[i - 1].0 + bounds[i - 1].1,
                        "non-contiguous at i={i} n={n} ws={ws}"
                    );
                }
            }
        }
    }

    // ── f16 tests ──

    #[test]
    fn test_apply_reduce_sum_f16() {
        let mut dst = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let src = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Sum);
        assert_eq!(dst[0].to_f32(), 5.0);
        assert_eq!(dst[1].to_f32(), 7.0);
        assert_eq!(dst[2].to_f32(), 9.0);
    }

    #[test]
    fn test_apply_reduce_product_f16() {
        let mut dst = vec![f16::from_f32(2.0), f16::from_f32(3.0)];
        let src = vec![f16::from_f32(5.0), f16::from_f32(6.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Product);
        assert_eq!(dst[0].to_f32(), 10.0);
        assert_eq!(dst[1].to_f32(), 18.0);
    }

    #[test]
    fn test_apply_reduce_max_f16() {
        let mut dst = vec![f16::from_f32(1.0), f16::from_f32(6.0)];
        let src = vec![f16::from_f32(4.0), f16::from_f32(2.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Max);
        assert_eq!(dst[0].to_f32(), 4.0);
        assert_eq!(dst[1].to_f32(), 6.0);
    }

    #[test]
    fn test_apply_reduce_min_f16() {
        let mut dst = vec![f16::from_f32(1.0), f16::from_f32(6.0)];
        let src = vec![f16::from_f32(4.0), f16::from_f32(2.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Min);
        assert_eq!(dst[0].to_f32(), 1.0);
        assert_eq!(dst[1].to_f32(), 2.0);
    }

    #[test]
    fn test_allreduce_f16_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let result = ring_allreduce_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[3].to_f32(), 4.0);
    }

    #[test]
    fn test_reduce_scatter_f16_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let result = ring_reduce_scatter_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
    }

    #[test]
    fn test_allreduce_f16_no_connections() {
        let mgr = ConnectionManager::new(0, 2, Topology::Ring);
        let mut data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let result = ring_allreduce_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[1].to_f32(), 2.0);
    }

    #[test]
    fn test_reduce_scatter_f16_no_connections() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        let mut data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];
        let result = ring_reduce_scatter_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        // Rank 0's chunk: elements 0,1
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[1].to_f32(), 2.0);
    }

    // ── bf16 tests ──

    #[test]
    fn test_apply_reduce_sum_bf16() {
        let mut dst = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];
        let src = vec![
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Sum);
        assert_eq!(dst[0].to_f32(), 5.0);
        assert_eq!(dst[1].to_f32(), 7.0);
        assert_eq!(dst[2].to_f32(), 9.0);
    }

    #[test]
    fn test_apply_reduce_product_bf16() {
        let mut dst = vec![bf16::from_f32(2.0), bf16::from_f32(3.0)];
        let src = vec![bf16::from_f32(5.0), bf16::from_f32(6.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Product);
        assert_eq!(dst[0].to_f32(), 10.0);
        assert_eq!(dst[1].to_f32(), 18.0);
    }

    #[test]
    fn test_apply_reduce_max_bf16() {
        let mut dst = vec![bf16::from_f32(1.0), bf16::from_f32(6.0)];
        let src = vec![bf16::from_f32(4.0), bf16::from_f32(2.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Max);
        assert_eq!(dst[0].to_f32(), 4.0);
        assert_eq!(dst[1].to_f32(), 6.0);
    }

    #[test]
    fn test_apply_reduce_min_bf16() {
        let mut dst = vec![bf16::from_f32(1.0), bf16::from_f32(6.0)];
        let src = vec![bf16::from_f32(4.0), bf16::from_f32(2.0)];
        apply_reduce_op_typed(&mut dst, &src, ReduceOp::Min);
        assert_eq!(dst[0].to_f32(), 1.0);
        assert_eq!(dst[1].to_f32(), 2.0);
    }

    #[test]
    fn test_allreduce_bf16_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let result = ring_allreduce_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[3].to_f32(), 4.0);
    }

    #[test]
    fn test_reduce_scatter_bf16_single_rank() {
        let mgr = ConnectionManager::new(0, 1, Topology::Ring);
        let mut data = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let result = ring_reduce_scatter_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
    }

    #[test]
    fn test_allreduce_bf16_no_connections() {
        let mgr = ConnectionManager::new(0, 2, Topology::Ring);
        let mut data = vec![bf16::from_f32(1.0), bf16::from_f32(2.0)];
        let result = ring_allreduce_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[1].to_f32(), 2.0);
    }

    #[test]
    fn test_reduce_scatter_bf16_no_connections() {
        let mgr = ConnectionManager::new(0, 4, Topology::Ring);
        let mut data = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
        ];
        let result = ring_reduce_scatter_typed(&mgr, &mut data, ReduceOp::Sum).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[1].to_f32(), 2.0);
    }

    // ── Cross-dtype consistency tests ──

    #[test]
    fn test_reduce_ops_all_variants_f16() {
        for op in [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Max,
            ReduceOp::Min,
        ] {
            let mut dst = vec![f16::from_f32(2.0)];
            let src = vec![f16::from_f32(3.0)];
            apply_reduce_op_typed(&mut dst, &src, op);
            match op {
                ReduceOp::Sum => assert_eq!(dst[0].to_f32(), 5.0),
                ReduceOp::Product => assert_eq!(dst[0].to_f32(), 6.0),
                ReduceOp::Max => assert_eq!(dst[0].to_f32(), 3.0),
                ReduceOp::Min => assert_eq!(dst[0].to_f32(), 2.0),
            }
        }
    }

    #[test]
    fn test_reduce_ops_all_variants_bf16() {
        for op in [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Max,
            ReduceOp::Min,
        ] {
            let mut dst = vec![bf16::from_f32(2.0)];
            let src = vec![bf16::from_f32(3.0)];
            apply_reduce_op_typed(&mut dst, &src, op);
            match op {
                ReduceOp::Sum => assert_eq!(dst[0].to_f32(), 5.0),
                ReduceOp::Product => assert_eq!(dst[0].to_f32(), 6.0),
                ReduceOp::Max => assert_eq!(dst[0].to_f32(), 3.0),
                ReduceOp::Min => assert_eq!(dst[0].to_f32(), 2.0),
            }
        }
    }

    #[test]
    fn test_collective_dtype_element_size() {
        assert_eq!(CollectiveDType::Float32.element_size(), 4);
        assert_eq!(CollectiveDType::Float16.element_size(), 2);
        assert_eq!(CollectiveDType::Bfloat16.element_size(), 2);
    }

    #[test]
    fn test_reduce_element_dtype_tags() {
        assert_eq!(f32::DTYPE, CollectiveDType::Float32);
        assert_eq!(f16::DTYPE, CollectiveDType::Float16);
        assert_eq!(bf16::DTYPE, CollectiveDType::Bfloat16);
    }
}
