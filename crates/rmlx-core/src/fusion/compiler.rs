//! Compile and dispatch fused element-wise kernels.
//!
//! Takes a [`FusionGraph`] from the analyzer, generates Metal source via
//! [`FusionCodegen`], compiles it through [`KernelRegistry`]'s JIT cache,
//! and dispatches the fused kernel — either into an ExecGraph CB or
//! standalone via the command queue.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

use super::codegen::FusionCodegen;
use super::graph::FusionGraph;

/// Kernel function name for a fused element-wise kernel, derived from
/// the graph's cache key and dtype.
fn fused_kernel_name(graph: &FusionGraph, dtype: DType) -> String {
    let dt_tag = match dtype {
        DType::Float32 => "f32",
        DType::Float16 => "f16",
        DType::Bfloat16 => "bf16",
        _ => "unk",
    };
    format!("fused_ew_{dt_tag}_{:016x}", graph.cache_key())
}

/// Compile a fused kernel from a [`FusionGraph`] and register it in the
/// registry's JIT cache.
///
/// Returns the kernel function name. If the kernel is already cached,
/// this is a no-op. The `dtype` controls the buffer element types in the
/// generated Metal source (float / half / bfloat16_t).
pub fn compile_fused(
    graph: &FusionGraph,
    codegen: &FusionCodegen,
    registry: &KernelRegistry,
    dtype: DType,
) -> Result<String, KernelError> {
    let name = fused_kernel_name(graph, dtype);

    // Generate Metal source with the unique kernel name so that the Metal
    // function name matches what get_pipeline() will look up.
    let source = codegen
        .generate_named(graph, &name, dtype)
        .map_err(|e| KernelError::CompilationFailed(format!("fusion codegen: {e}")))?;

    registry.register_jit_source_if_absent(&name, &source)?;

    Ok(name)
}

/// Dispatch a compiled fused kernel.
///
/// `inputs`: the external input arrays, in the same order as the
/// FusionGraph's input indices.
///
/// Returns the output array(s). Currently only single-output is supported.
pub fn dispatch_fused(
    graph: &FusionGraph,
    codegen: &FusionCodegen,
    registry: &KernelRegistry,
    inputs: &[&Array],
    queue: &metal::CommandQueue,
) -> Result<Vec<Array>, KernelError> {
    if graph.n_inputs() != inputs.len() {
        return Err(KernelError::InvalidShape(format!(
            "fused dispatch: expected {} inputs, got {}",
            graph.n_inputs(),
            inputs.len()
        )));
    }

    let dtype = inputs[0].dtype();
    let name = compile_fused(graph, codegen, registry, dtype)?;

    // Get pipeline using the unique per-graph kernel name
    let pipeline = registry.get_pipeline(&name, dtype)?;

    // Compute output size (element count of the first input — all must match
    // for element-wise ops after broadcasting).
    let n_elements = inputs[0].shape().iter().product::<usize>();
    let device = registry.device().raw();

    // Allocate output buffer(s)
    let n_outputs = graph.n_outputs().max(1);
    let outputs: Vec<Array> = (0..n_outputs)
        .map(|_| Array::zeros(device, inputs[0].shape(), inputs[0].dtype()))
        .collect();

    // Create CB and encode
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    // Bind input buffers
    for (i, input) in inputs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(input.metal_buffer()), input.offset() as u64);
    }

    // Bind output buffers
    for (o, output) in outputs.iter().enumerate() {
        let buf_idx = (inputs.len() + o) as u64;
        enc.set_buffer(buf_idx, Some(output.metal_buffer()), output.offset() as u64);
    }

    // Bind N (element count)
    let n = n_elements as u32;
    let n_buf = device.new_buffer_with_data(
        &n as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let params_idx = (inputs.len() + outputs.len()) as u64;
    enc.set_buffer(params_idx, Some(&n_buf), 0);

    // Dispatch
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let tg_size = std::cmp::min(max_tg, n_elements as u64);
    let grid = metal::MTLSize::new(n_elements as u64, 1, 1);
    let tg = metal::MTLSize::new(tg_size, 1, 1);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();

    cb.commit();
    cb.wait_until_completed();

    Ok(outputs)
}

/// Dispatch a compiled fused kernel into a command buffer (for ExecGraph use).
///
/// Does NOT commit or wait — the caller manages the CB lifecycle.
pub fn dispatch_fused_into_cb(
    graph: &FusionGraph,
    codegen: &FusionCodegen,
    registry: &KernelRegistry,
    inputs: &[&Array],
    cb: &metal::CommandBufferRef,
) -> Result<Vec<Array>, KernelError> {
    if graph.n_inputs() != inputs.len() {
        return Err(KernelError::InvalidShape(format!(
            "fused dispatch: expected {} inputs, got {}",
            graph.n_inputs(),
            inputs.len()
        )));
    }

    let dtype = inputs[0].dtype();
    let name = compile_fused(graph, codegen, registry, dtype)?;
    let pipeline = registry.get_pipeline(&name, dtype)?;

    let n_elements = inputs[0].shape().iter().product::<usize>();
    let device = registry.device().raw();

    let n_outputs = graph.n_outputs().max(1);
    let outputs: Vec<Array> = (0..n_outputs)
        .map(|_| Array::zeros(device, inputs[0].shape(), inputs[0].dtype()))
        .collect();

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    for (i, input) in inputs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(input.metal_buffer()), input.offset() as u64);
    }
    for (o, output) in outputs.iter().enumerate() {
        let buf_idx = (inputs.len() + o) as u64;
        enc.set_buffer(buf_idx, Some(output.metal_buffer()), output.offset() as u64);
    }

    let n = n_elements as u32;
    let n_buf = device.new_buffer_with_data(
        &n as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let params_idx = (inputs.len() + outputs.len()) as u64;
    enc.set_buffer(params_idx, Some(&n_buf), 0);

    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let tg_size = std::cmp::min(max_tg, n_elements as u64);
    let grid = metal::MTLSize::new(n_elements as u64, 1, 1);
    let tg = metal::MTLSize::new(tg_size, 1, 1);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::graph::{FusableOp, FusionGraph};

    #[test]
    fn test_fused_kernel_name() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let name = fused_kernel_name(&g, DType::Float32);
        assert!(name.starts_with("fused_ew_f32_"));
        // "fused_ew_f32_" (13) + 16 hex chars
        assert_eq!(name.len(), 13 + 16);
    }

    #[test]
    fn test_fused_kernel_name_deterministic() {
        let mut g1 = FusionGraph::new(2);
        g1.add_op(FusableOp::Add, vec![0, 1]);
        g1.set_outputs(1);

        let mut g2 = FusionGraph::new(2);
        g2.add_op(FusableOp::Add, vec![0, 1]);
        g2.set_outputs(1);

        assert_eq!(
            fused_kernel_name(&g1, DType::Float32),
            fused_kernel_name(&g2, DType::Float32)
        );
    }

    #[test]
    fn test_fused_kernel_name_dtype_differs() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let f32_name = fused_kernel_name(&g, DType::Float32);
        let f16_name = fused_kernel_name(&g, DType::Float16);
        let bf16_name = fused_kernel_name(&g, DType::Bfloat16);

        assert_ne!(f32_name, f16_name);
        assert_ne!(f32_name, bf16_name);
        assert_ne!(f16_name, bf16_name);

        assert!(f32_name.contains("f32"));
        assert!(f16_name.contains("f16"));
        assert!(bf16_name.contains("bf16"));
    }
}
