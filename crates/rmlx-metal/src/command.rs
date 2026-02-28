//! Metal command buffer and encoder abstractions

use metal::{Buffer, CommandBufferRef, ComputePipelineState, MTLSize};

/// Convenience function to encode a simple 1D compute dispatch.
///
/// Creates a compute command encoder on `cmd_buf`, sets the pipeline state,
/// binds each buffer at consecutive indices (0, 1, 2, ...) with the given
/// offsets, dispatches `num_threads` threads in a 1D grid, and ends encoding.
pub fn encode_compute_1d(
    cmd_buf: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, u64)],
    num_threads: u64,
) {
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);

    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }

    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup_size = std::cmp::min(max_threads, num_threads);

    let grid_size = MTLSize::new(num_threads, 1, 1);
    let group_size = MTLSize::new(threadgroup_size, 1, 1);

    encoder.dispatch_threads(grid_size, group_size);
    encoder.end_encoding();
}
