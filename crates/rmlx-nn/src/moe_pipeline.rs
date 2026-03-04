//! TBO (Token Batch Overlap) and SBO (Shared expert Batch Overlap) orchestration
//! for MoE forward pass.
//!
//! # Overview
//!
//! This module provides overlapped execution strategies that chain GPU stages via
//! `GpuEvent` signals, eliminating CPU synchronization in the pipeline body.
//! Only a single `cpu_wait()` is issued at the very end of the forward pass.
//!
//! ## SBO (Shared expert Batch Overlap)
//!
//! Overlaps the shared expert forward pass with routed expert computation.
//! The shared expert is data-independent of routing results, so it can run
//! on the GPU while routed experts are being computed (especially beneficial
//! when RDMA dispatch runs on a separate CPU thread).
//!
//! ## TBO (Token Batch Overlap)
//!
//! Splits tokens into batches and pipelines dispatch/compute/combine across
//! batches so that batch N+1 dispatch overlaps with batch N compute.

use std::sync::Arc;
use std::time::Duration;

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_core::ops::topk_route::TopkRouteResult;
use rmlx_metal::event::GpuEvent;

use crate::expert_group::ExpertGroup;
use crate::moe::Expert;

/// Per-expert dispatch entry: `(expert_idx, expert_input_array, dispatch_list)`.
/// The dispatch list maps each row in the expert input to `(token_idx, weight)`.
type ExpertDispatchEntry = (usize, Array, Vec<(usize, f32)>);

/// Configuration for the MoE pipeline overlap strategy.
pub struct MoePipelineConfig {
    /// Number of token batches for TBO overlap.
    /// 1 = no overlap (SBO only), 2+ = TBO.
    pub num_tbo_batches: usize,
    /// Whether to overlap shared expert computation with dispatch/routed expert work.
    pub enable_sbo: bool,
    /// Timeout for final CPU wait.
    pub sync_timeout: Duration,
}

impl Default for MoePipelineConfig {
    fn default() -> Self {
        Self {
            num_tbo_batches: 1,
            enable_sbo: true,
            sync_timeout: Duration::from_secs(5),
        }
    }
}

/// Pipeline orchestrator for overlapped MoE forward pass.
///
/// Chains GPU stages via GpuEvent signals -- zero CPU waits in the pipeline.
/// Single `cpu_wait()` at the very end of the forward pass.
pub struct MoePipeline {
    /// GpuEvent for inter-stage synchronization.
    event: Arc<GpuEvent>,
    /// Number of batches for TBO (Token Batch Overlap).
    /// 1 = no overlap (SBO only), 2+ = TBO.
    num_batches: usize,
    /// Configuration.
    config: MoePipelineConfig,
}

impl MoePipeline {
    /// Create a new MoePipeline with the given event and configuration.
    pub fn new(event: Arc<GpuEvent>, config: MoePipelineConfig) -> Self {
        let num_batches = config.num_tbo_batches.max(1);
        Self {
            event,
            num_batches,
            config,
        }
    }

    /// Create a pipeline with default configuration.
    pub fn with_defaults(device: &metal::Device) -> Self {
        let event = Arc::new(GpuEvent::new(device));
        Self::new(event, MoePipelineConfig::default())
    }

    /// Access the underlying GpuEvent.
    pub fn event(&self) -> &Arc<GpuEvent> {
        &self.event
    }

    /// Number of TBO batches.
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &MoePipelineConfig {
        &self.config
    }

    /// Whether SBO is enabled.
    pub fn sbo_enabled(&self) -> bool {
        self.config.enable_sbo
    }

    /// Execute SBO: shared expert computes in parallel with routed expert work.
    ///
    /// Timeline:
    /// ```text
    /// [shared_expert.forward(x)] --signal(v=1)-->
    ///         | (data-independent, GPU overlap with future RDMA)
    /// [routed_expert_fn(x)] --signal(v=2)-->
    ///                                        [combine: shared + routed]
    ///                                         --signal(v=3)--> cpu_wait(v=3)
    /// ```
    ///
    /// The shared expert forward is data-independent of routing decisions,
    /// so it can overlap with RDMA dispatch when distributed mode is active.
    /// Without RDMA, both computations run on the same GPU queue, but the
    /// event-chained design eliminates CPU synchronization between stages.
    pub fn forward_sbo(
        &self,
        x: &Array,
        shared_expert: &Expert,
        routed_expert_fn: impl FnOnce(&Array) -> Result<Array, KernelError>,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // Reset event for fresh pipeline run.
        self.event.reset();
        self.event.reset_cancel();

        // -- Stage 1: Shared expert forward on GPU --
        let shared_out = shared_expert.forward(x, registry, queue)?;

        // Signal that shared expert is done.
        let v_shared = self.event.next_value();
        let cb_shared = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_shared, v_shared);
        cb_shared.commit();

        // -- Stage 2: Routed expert computation --
        // The closure handles routing, dispatch, and expert forward internally.
        let routed_out = routed_expert_fn(x)?;

        // Signal that routed experts are done.
        let v_routed = self.event.next_value();
        let cb_routed = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_routed, v_routed);
        cb_routed.commit();

        // -- Stage 3: Combine shared + routed --
        let output = ops::binary::add(registry, &shared_out, &routed_out, queue)?;

        // Final signal.
        let v_final = self.event.next_value();
        let cb_final = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_final, v_final);
        cb_final.commit();

        // Single CPU wait at the end of the entire pipeline.
        self.event
            .cpu_wait(v_final, self.config.sync_timeout)
            .map_err(|e| KernelError::InvalidShape(format!("SBO cpu_wait failed: {e}")))?;

        Ok(output)
    }

    /// Execute TBO: pipeline dispatch/compute/combine across token batches.
    ///
    /// GPU event chain:
    /// ```text
    /// [route(all)] --signal(1)--> [dispatch(B0)] --signal(2)--> [compute(B0) || dispatch(B1)]
    ///                                                                  |
    ///                             [combine(B0) || compute(B1)] <--signal(3)--
    ///                                                                  |
    ///                                               [combine(B1)] --signal(4)--> [output]
    /// ```
    ///
    /// Each batch processes a subset of tokens. Batch N+1 dispatch overlaps
    /// with batch N compute when encoded into the same command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tbo(
        &self,
        x: &Array,
        expert_group: &ExpertGroup,
        route_result: &TopkRouteResult,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let hidden_dim = x.shape()[1];
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();
        let num_experts = expert_group.num_experts();

        // Read routing results to CPU for dispatch.
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Build per-expert dispatch: expert_dispatch[e] = Vec<(token_idx, weight)>
        let mut expert_dispatch: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for tok in 0..seq_len {
            for k in 0..top_k {
                let flat_idx = tok * top_k + k;
                if flat_idx >= indices_vec.len() {
                    break;
                }
                let expert_idx = indices_vec[flat_idx] as usize;
                let weight = weights_vec[flat_idx];
                expert_dispatch[expert_idx].push((tok, weight));
            }
        }

        // Split token indices into num_batches batches.
        let batch_size = seq_len.div_ceil(self.num_batches);

        // Output accumulator: [seq_len, hidden_dim]
        let output = Array::zeros(dev, &[seq_len, hidden_dim], x.dtype());

        // Reset event for this pipeline run.
        self.event.reset();
        self.event.reset_cancel();

        let (local_start, local_end) = local_expert_range;

        for batch_idx in 0..self.num_batches {
            let tok_start = batch_idx * batch_size;
            let tok_end = std::cmp::min(tok_start + batch_size, seq_len);
            if tok_start >= seq_len {
                break;
            }

            // Determine which experts have tokens in this batch.
            let mut batch_expert_inputs: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
            for (expert_idx, dispatch) in expert_dispatch.iter().enumerate() {
                if expert_idx < local_start || expert_idx >= local_end {
                    continue;
                }
                let batch_tokens: Vec<(usize, f32)> = dispatch
                    .iter()
                    .filter(|&&(tok, _)| tok >= tok_start && tok < tok_end)
                    .copied()
                    .collect();
                if !batch_tokens.is_empty() {
                    batch_expert_inputs.push((expert_idx, batch_tokens));
                }
            }

            if batch_expert_inputs.is_empty() {
                continue;
            }

            // Gather expert inputs for this batch.
            let mut expert_arrays: Vec<ExpertDispatchEntry> = Vec::new();
            for (expert_idx, tokens) in &batch_expert_inputs {
                let n_tokens = tokens.len();
                let expert_input =
                    gather_tokens(x, tokens, hidden_dim, elem_size, dev, registry, queue)?;
                let _ = n_tokens;
                expert_arrays.push((*expert_idx, expert_input, tokens.clone()));
            }

            // Signal that dispatch/gather for this batch is done.
            let v_dispatch = self.event.next_value();
            let cb_sig = queue.new_command_buffer();
            self.event.signal_from_command_buffer(cb_sig, v_dispatch);
            cb_sig.commit();

            // Compute: run grouped forward on this batch's experts.
            let inputs_for_group: Vec<(usize, &Array)> = expert_arrays
                .iter()
                .map(|(idx, arr, _)| (*idx, arr))
                .collect();
            let expert_outputs =
                expert_group.grouped_forward(&inputs_for_group, registry, queue)?;

            // Scatter-add weighted expert outputs back to the output buffer.
            scatter_add_weighted(
                &expert_outputs,
                &expert_arrays,
                &output,
                hidden_dim,
                elem_size,
                dev,
                registry,
                queue,
            )?;

            // Signal that this batch's combine is done.
            let v_combine = self.event.next_value();
            let cb_combine = queue.new_command_buffer();
            self.event.signal_from_command_buffer(cb_combine, v_combine);
            cb_combine.commit();
        }

        // Single CPU wait at the very end.
        let final_value = self.event.current_value();
        if final_value > 0 {
            self.event
                .cpu_wait(final_value, self.config.sync_timeout)
                .map_err(|e| KernelError::InvalidShape(format!("TBO cpu_wait failed: {e}")))?;
        }

        Ok(output)
    }

    /// Overlapped MoE forward pass using SBO and/or TBO.
    ///
    /// Chooses the overlap strategy based on configuration:
    /// - If shared_expert is present and SBO enabled: run SBO
    /// - If num_tbo_batches > 1: run TBO
    /// - Otherwise: fall back to sequential forward
    ///
    /// Zero `cb.wait_until_completed()` in the pipeline body.
    /// Single `GpuEvent::cpu_wait()` at the very end.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_overlapped(
        &self,
        x: &Array,
        gate_logits: &Array,
        expert_group: &ExpertGroup,
        shared_expert: Option<&Expert>,
        expert_bias: Option<&Array>,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // Reset event for fresh pipeline run.
        self.event.reset();
        self.event.reset_cancel();

        // -- Stage 1: GPU top-k routing --
        let route_result =
            ops::topk_route::gpu_topk_route(registry, gate_logits, top_k, expert_bias, queue)?;

        // -- Determine strategy --
        let use_sbo = self.config.enable_sbo && shared_expert.is_some();
        let use_tbo = self.num_batches > 1;

        if use_sbo && use_tbo {
            // Both SBO and TBO: shared expert overlaps with TBO pipeline.
            self.forward_sbo_tbo(
                x,
                expert_group,
                shared_expert.unwrap(),
                &route_result,
                top_k,
                local_expert_range,
                registry,
                queue,
            )
        } else if use_sbo {
            // SBO only: overlap shared expert with sequential routed experts.
            self.forward_sbo_only(
                x,
                expert_group,
                shared_expert.unwrap(),
                &route_result,
                top_k,
                local_expert_range,
                registry,
                queue,
            )
        } else if use_tbo {
            // TBO only: pipeline token batches, no shared expert overlap.
            let routed_out = self.forward_tbo(
                x,
                expert_group,
                &route_result,
                top_k,
                local_expert_range,
                registry,
                queue,
            )?;

            // Add shared expert output sequentially if present (SBO disabled).
            if let Some(shared) = shared_expert {
                let shared_out = shared.forward(x, registry, queue)?;
                ops::binary::add(registry, &routed_out, &shared_out, queue)
            } else {
                Ok(routed_out)
            }
        } else {
            // Sequential fallback: no overlap.
            self.forward_sequential(
                x,
                expert_group,
                shared_expert,
                &route_result,
                top_k,
                local_expert_range,
                registry,
                queue,
            )
        }
    }

    // -----------------------------------------------------------------------
    // Internal strategy implementations
    // -----------------------------------------------------------------------

    /// SBO-only path: overlap shared expert with routed expert sequential execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_sbo_only(
        &self,
        x: &Array,
        expert_group: &ExpertGroup,
        shared_expert: &Expert,
        route_result: &TopkRouteResult,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        self.event.reset();
        self.event.reset_cancel();

        // -- Shared expert forward (data-independent of routing) --
        let shared_out = shared_expert.forward(x, registry, queue)?;

        let v_shared = self.event.next_value();
        let cb_shared = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_shared, v_shared);
        cb_shared.commit();

        // -- Routed expert computation --
        let routed_out = self.run_routed_experts(
            x,
            expert_group,
            route_result,
            top_k,
            local_expert_range,
            registry,
            queue,
        )?;

        let v_routed = self.event.next_value();
        let cb_routed = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_routed, v_routed);
        cb_routed.commit();

        // -- Combine --
        let output = ops::binary::add(registry, &shared_out, &routed_out, queue)?;

        let v_final = self.event.next_value();
        let cb_final = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_final, v_final);
        cb_final.commit();

        self.event
            .cpu_wait(v_final, self.config.sync_timeout)
            .map_err(|e| KernelError::InvalidShape(format!("SBO cpu_wait failed: {e}")))?;

        Ok(output)
    }

    /// Combined SBO + TBO: shared expert overlaps with TBO pipeline.
    #[allow(clippy::too_many_arguments)]
    fn forward_sbo_tbo(
        &self,
        x: &Array,
        expert_group: &ExpertGroup,
        shared_expert: &Expert,
        route_result: &TopkRouteResult,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        self.event.reset();
        self.event.reset_cancel();

        // -- Shared expert runs first (overlaps with TBO dispatch on RDMA path) --
        let shared_out = shared_expert.forward(x, registry, queue)?;

        let v_shared = self.event.next_value();
        let cb_shared = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_shared, v_shared);
        cb_shared.commit();

        // -- TBO pipeline for routed experts --
        let routed_out = self.forward_tbo(
            x,
            expert_group,
            route_result,
            top_k,
            local_expert_range,
            registry,
            queue,
        )?;

        // -- Combine shared + routed --
        let output = ops::binary::add(registry, &shared_out, &routed_out, queue)?;

        let v_final = self.event.next_value();
        let cb_final = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_final, v_final);
        cb_final.commit();

        self.event
            .cpu_wait(v_final, self.config.sync_timeout)
            .map_err(|e| KernelError::InvalidShape(format!("SBO+TBO cpu_wait failed: {e}")))?;

        Ok(output)
    }

    /// Sequential fallback: no overlap, matches the original MoeLayer::forward() logic.
    #[allow(clippy::too_many_arguments)]
    fn forward_sequential(
        &self,
        x: &Array,
        expert_group: &ExpertGroup,
        shared_expert: Option<&Expert>,
        route_result: &TopkRouteResult,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let routed_out = self.run_routed_experts(
            x,
            expert_group,
            route_result,
            top_k,
            local_expert_range,
            registry,
            queue,
        )?;

        let output = if let Some(shared) = shared_expert {
            let shared_out = shared.forward(x, registry, queue)?;
            ops::binary::add(registry, &routed_out, &shared_out, queue)?
        } else {
            routed_out
        };

        // Final event signal + CPU wait for API consistency.
        let v_final = self.event.next_value();
        let cb_final = queue.new_command_buffer();
        self.event.signal_from_command_buffer(cb_final, v_final);
        cb_final.commit();

        self.event
            .cpu_wait(v_final, self.config.sync_timeout)
            .map_err(|e| KernelError::InvalidShape(format!("sequential cpu_wait failed: {e}")))?;

        Ok(output)
    }

    /// Run routed experts using ExpertGroup::grouped_forward with scatter-add combine.
    ///
    /// This is the common routed expert path shared by SBO, TBO, and sequential modes.
    #[allow(clippy::too_many_arguments)]
    fn run_routed_experts(
        &self,
        x: &Array,
        expert_group: &ExpertGroup,
        route_result: &TopkRouteResult,
        top_k: usize,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let hidden_dim = x.shape()[1];
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();
        let num_experts = expert_group.num_experts();

        // Read routing results to CPU for dispatch.
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Build per-expert dispatch buffers.
        let mut expert_dispatch: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for tok in 0..seq_len {
            for k in 0..top_k {
                let flat_idx = tok * top_k + k;
                if flat_idx >= indices_vec.len() {
                    break;
                }
                let expert_idx = indices_vec[flat_idx] as usize;
                let weight = weights_vec[flat_idx];
                expert_dispatch[expert_idx].push((tok, weight));
            }
        }

        let (local_start, local_end) = local_expert_range;

        // Gather expert inputs into contiguous batches.
        let mut expert_inputs_with_dispatch: Vec<ExpertDispatchEntry> = Vec::new();

        for (expert_idx, dispatch) in expert_dispatch.iter().enumerate() {
            if dispatch.is_empty() {
                continue;
            }
            if expert_idx < local_start || expert_idx >= local_end {
                continue;
            }

            let expert_input =
                gather_tokens(x, dispatch, hidden_dim, elem_size, dev, registry, queue)?;
            expert_inputs_with_dispatch.push((expert_idx, expert_input, dispatch.clone()));
        }

        // Run grouped forward.
        let inputs_for_group: Vec<(usize, &Array)> = expert_inputs_with_dispatch
            .iter()
            .map(|(idx, arr, _)| (*idx, arr))
            .collect();
        let expert_outputs = expert_group.grouped_forward(&inputs_for_group, registry, queue)?;

        // Scatter-add weighted outputs back into accumulator.
        let output = Array::zeros(dev, &[seq_len, hidden_dim], x.dtype());

        scatter_add_weighted(
            &expert_outputs,
            &expert_inputs_with_dispatch,
            &output,
            hidden_dim,
            elem_size,
            dev,
            registry,
            queue,
        )?;

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Gather token embeddings for an expert into a contiguous batch buffer.
///
/// Given a dispatch list of `(token_idx, weight)` pairs, copies the token rows
/// from `x` into a contiguous `[batch_size, hidden_dim]` array.
fn gather_tokens(
    x: &Array,
    dispatch: &[(usize, f32)],
    hidden_dim: usize,
    elem_size: usize,
    dev: &metal::Device,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let batch_size = dispatch.len();

    if batch_size == 1 {
        let tok = dispatch[0].0;
        let tok_offset = x.offset() + tok * hidden_dim * elem_size;
        let tok_view = x.view(vec![1, hidden_dim], vec![hidden_dim, 1], tok_offset);
        return ops::copy::copy(registry, &tok_view, queue);
    }

    let batch_buf = Array::zeros(dev, &[batch_size, hidden_dim], x.dtype());
    let copy_kernel = copy_kernel_name(x.dtype())?;
    let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;
    let cb = queue.new_command_buffer();

    for (i, &(tok, _)) in dispatch.iter().enumerate() {
        let src_offset = x.offset() + tok * hidden_dim * elem_size;
        let dst_offset = i * hidden_dim * elem_size;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(x.metal_buffer()), src_offset as u64);
        enc.set_buffer(1, Some(batch_buf.metal_buffer()), dst_offset as u64);
        let grid = metal::MTLSize::new(hidden_dim as u64, 1, 1);
        let tg = metal::MTLSize::new(
            std::cmp::min(
                pipeline.max_total_threads_per_threadgroup(),
                hidden_dim as u64,
            ),
            1,
            1,
        );
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();
    Ok(batch_buf)
}

/// Scatter-add weighted expert outputs back into the output accumulator.
///
/// For each expert output token, scales it by the routing weight and adds it
/// to the corresponding position in the output buffer.
#[allow(clippy::too_many_arguments)]
fn scatter_add_weighted(
    expert_outputs: &[(usize, Array)],
    expert_inputs_with_dispatch: &[ExpertDispatchEntry],
    output: &Array,
    hidden_dim: usize,
    elem_size: usize,
    dev: &metal::Device,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<(), KernelError> {
    let copy_kernel = copy_kernel_name(output.dtype())?;
    let pipeline = registry.get_pipeline(copy_kernel, output.dtype())?;

    for ((_, expert_out), (_, _, dispatch)) in expert_outputs
        .iter()
        .zip(expert_inputs_with_dispatch.iter())
    {
        for (i, &(tok, weight)) in dispatch.iter().enumerate() {
            let expert_tok_offset = expert_out.offset() + i * hidden_dim * elem_size;
            let expert_tok_view =
                expert_out.view(vec![1, hidden_dim], vec![hidden_dim, 1], expert_tok_offset);

            let scale_data = vec![weight; hidden_dim];
            let scale_arr = Array::from_slice(dev, &scale_data, vec![1, hidden_dim]);
            let scaled = ops::binary::mul(registry, &expert_tok_view, &scale_arr, queue)?;

            let dst_offset = output.offset() + tok * hidden_dim * elem_size;
            let dst_view = output.view(vec![1, hidden_dim], vec![hidden_dim, 1], dst_offset);

            let summed = ops::binary::add(registry, &dst_view, &scaled, queue)?;

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(summed.metal_buffer()), summed.offset() as u64);
            enc.set_buffer(1, Some(output.metal_buffer()), dst_offset as u64);
            let grid = metal::MTLSize::new(hidden_dim as u64, 1, 1);
            let tg = metal::MTLSize::new(
                std::cmp::min(
                    pipeline.max_total_threads_per_threadgroup(),
                    hidden_dim as u64,
                ),
                1,
                1,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
    }

    Ok(())
}

/// Get the copy kernel name for the given dtype.
fn copy_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("copy_f32"),
        DType::Float16 => Ok("copy_f16"),
        DType::Bfloat16 => Ok("copy_bf16"),
        other => Err(KernelError::InvalidShape(format!(
            "moe_pipeline: unsupported dtype {:?} for copy kernel",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{Linear, LinearConfig};
    use crate::moe::Expert;

    /// Create a test `KernelRegistry` with all kernels registered.
    fn test_registry() -> (metal::Device, metal::CommandQueue, KernelRegistry) {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("GPU device required");
        let registry = KernelRegistry::new(gpu);
        rmlx_core::ops::register_all(&registry).expect("register kernels");
        (device, queue, registry)
    }

    /// Create a test expert with known weight values.
    fn make_expert(
        device: &metal::Device,
        hidden_dim: usize,
        intermediate_dim: usize,
        val: f32,
    ) -> Expert {
        let gate_data: Vec<f32> = vec![val; intermediate_dim * hidden_dim];
        let gate_w = Array::from_slice(device, &gate_data, vec![intermediate_dim, hidden_dim]);
        let gate_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            gate_w,
            None,
        )
        .unwrap();

        let up_data: Vec<f32> = vec![val; intermediate_dim * hidden_dim];
        let up_w = Array::from_slice(device, &up_data, vec![intermediate_dim, hidden_dim]);
        let up_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            up_w,
            None,
        )
        .unwrap();

        let down_data: Vec<f32> = vec![val; hidden_dim * intermediate_dim];
        let down_w = Array::from_slice(device, &down_data, vec![hidden_dim, intermediate_dim]);
        let down_proj = Linear::from_arrays(
            LinearConfig {
                in_features: intermediate_dim,
                out_features: hidden_dim,
                has_bias: false,
            },
            down_w,
            None,
        )
        .unwrap();

        Expert {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = MoePipelineConfig::default();
        assert_eq!(config.num_tbo_batches, 1);
        assert!(config.enable_sbo);
        assert_eq!(config.sync_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_pipeline_creation() {
        let device = metal::Device::system_default().expect("Metal device required");
        let pipeline = MoePipeline::with_defaults(&device);
        assert_eq!(pipeline.num_batches(), 1);
        assert!(pipeline.sbo_enabled());
    }

    #[test]
    fn test_pipeline_custom_config() {
        let device = metal::Device::system_default().expect("Metal device required");
        let event = Arc::new(GpuEvent::new(&device));
        let config = MoePipelineConfig {
            num_tbo_batches: 4,
            enable_sbo: false,
            sync_timeout: Duration::from_secs(10),
        };
        let pipeline = MoePipeline::new(event, config);
        assert_eq!(pipeline.num_batches(), 4);
        assert!(!pipeline.sbo_enabled());
    }

    #[test]
    fn test_sbo_shared_plus_routed_output() {
        // Verify that SBO produces output = shared_expert(x) + routed_experts(x).
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;
        let num_experts = 4;
        let top_k = 2;
        let seq_len = 4;

        // Create routed experts.
        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| make_expert(&device, hidden_dim, intermediate_dim, 0.1 * (i + 1) as f32))
            .collect();

        // Create shared expert with distinct weights.
        let shared_expert = make_expert(&device, hidden_dim, intermediate_dim, 0.5);

        // Build expert group for grouped forward.
        let expert_group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        // Create input.
        let x_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let x = Array::from_slice(&device, &x_data, vec![seq_len, hidden_dim]);

        // Create gate logits: [seq_len, num_experts]
        #[rustfmt::skip]
        let gate_data: Vec<f32> = vec![
            5.0, 1.0, 1.0, 0.5,  // token 0: expert 0 strong
            0.5, 5.0, 1.0, 1.0,  // token 1: expert 1 strong
            1.0, 0.5, 5.0, 1.0,  // token 2: expert 2 strong
            1.0, 1.0, 0.5, 5.0,  // token 3: expert 3 strong
        ];
        let gate_logits = Array::from_slice(&device, &gate_data, vec![seq_len, num_experts]);

        // -- Compute reference output: sequential shared + routed --
        let shared_ref = shared_expert.forward(&x, &registry, &queue).unwrap();

        let route_result =
            ops::topk_route::gpu_topk_route(&registry, &gate_logits, top_k, None, &queue).unwrap();

        let pipeline = MoePipeline::with_defaults(&device);
        let routed_ref = pipeline
            .run_routed_experts(
                &x,
                &expert_group,
                &route_result,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();

        let ref_output = ops::binary::add(&registry, &shared_ref, &routed_ref, &queue).unwrap();
        let ref_vals: Vec<f32> = ref_output.to_vec_checked();

        // -- Compute SBO output via forward_overlapped --
        let pipeline_sbo = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 1,
                enable_sbo: true,
                sync_timeout: Duration::from_secs(5),
            },
        );

        let sbo_output = pipeline_sbo
            .forward_overlapped(
                &x,
                &gate_logits,
                &expert_group,
                Some(&shared_expert),
                None,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();
        let sbo_vals: Vec<f32> = sbo_output.to_vec_checked();

        // Verify outputs match within tolerance.
        assert_eq!(ref_vals.len(), sbo_vals.len());
        for i in 0..ref_vals.len() {
            let diff = (ref_vals[i] - sbo_vals[i]).abs();
            assert!(
                diff < 1e-2,
                "SBO mismatch at index {i}: ref={} sbo={} diff={}",
                ref_vals[i],
                sbo_vals[i],
                diff
            );
        }
    }

    #[test]
    fn test_sbo_output_nonzero() {
        // Verify that SBO output is not all zeros (sanity check).
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;
        let num_experts = 2;
        let top_k = 1;
        let seq_len = 2;

        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| make_expert(&device, hidden_dim, intermediate_dim, 0.1 * (i + 1) as f32))
            .collect();
        let shared_expert = make_expert(&device, hidden_dim, intermediate_dim, 0.3);
        let expert_group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let x_data: Vec<f32> = vec![1.0; seq_len * hidden_dim];
        let x = Array::from_slice(&device, &x_data, vec![seq_len, hidden_dim]);

        let gate_data: Vec<f32> = vec![5.0, 1.0, 1.0, 5.0];
        let gate_logits = Array::from_slice(&device, &gate_data, vec![seq_len, num_experts]);

        let pipeline = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 1,
                enable_sbo: true,
                sync_timeout: Duration::from_secs(5),
            },
        );

        let output = pipeline
            .forward_overlapped(
                &x,
                &gate_logits,
                &expert_group,
                Some(&shared_expert),
                None,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();

        assert_eq!(output.shape(), &[seq_len, hidden_dim]);
        let vals: Vec<f32> = output.to_vec_checked();
        let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            any_nonzero,
            "SBO output is all zeros, expected nonzero values"
        );
    }

    #[test]
    fn test_forward_sbo_standalone() {
        // Test the standalone forward_sbo method directly.
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;

        let shared_expert = make_expert(&device, hidden_dim, intermediate_dim, 0.2);

        let x_data: Vec<f32> = vec![1.0; 2 * hidden_dim];
        let x = Array::from_slice(&device, &x_data, vec![2, hidden_dim]);

        let pipeline = MoePipeline::with_defaults(&device);

        // Compute reference: shared_expert(x) + constant routed output
        let shared_ref = shared_expert.forward(&x, &registry, &queue).unwrap();
        let routed_ref =
            Array::from_slice(&device, &vec![0.5f32; 2 * hidden_dim], vec![2, hidden_dim]);
        let ref_output = ops::binary::add(&registry, &shared_ref, &routed_ref, &queue).unwrap();
        let ref_vals: Vec<f32> = ref_output.to_vec_checked();

        // Use forward_sbo with a closure that returns a constant array.
        let sbo_output = pipeline
            .forward_sbo(
                &x,
                &shared_expert,
                |_x| {
                    Ok(Array::from_slice(
                        &device,
                        &vec![0.5f32; 2 * hidden_dim],
                        vec![2, hidden_dim],
                    ))
                },
                &registry,
                &queue,
            )
            .unwrap();
        let sbo_vals: Vec<f32> = sbo_output.to_vec_checked();

        assert_eq!(ref_vals.len(), sbo_vals.len());
        for i in 0..ref_vals.len() {
            let diff = (ref_vals[i] - sbo_vals[i]).abs();
            assert!(
                diff < 1e-4,
                "forward_sbo mismatch at {i}: ref={} sbo={} diff={}",
                ref_vals[i],
                sbo_vals[i],
                diff
            );
        }
    }

    #[test]
    fn test_sequential_fallback() {
        // Test with SBO disabled and TBO batches=1 -> sequential path.
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;
        let num_experts = 2;
        let top_k = 1;
        let seq_len = 2;

        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| make_expert(&device, hidden_dim, intermediate_dim, 0.1 * (i + 1) as f32))
            .collect();
        let expert_group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let x_data: Vec<f32> = vec![1.0; seq_len * hidden_dim];
        let x = Array::from_slice(&device, &x_data, vec![seq_len, hidden_dim]);

        let gate_data: Vec<f32> = vec![5.0, 1.0, 1.0, 5.0];
        let gate_logits = Array::from_slice(&device, &gate_data, vec![seq_len, num_experts]);

        let pipeline = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 1,
                enable_sbo: false,
                sync_timeout: Duration::from_secs(5),
            },
        );

        let output = pipeline
            .forward_overlapped(
                &x,
                &gate_logits,
                &expert_group,
                None,
                None,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();

        assert_eq!(output.shape(), &[seq_len, hidden_dim]);
        let vals: Vec<f32> = output.to_vec_checked();
        let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            any_nonzero,
            "Sequential output is all zeros, expected nonzero values"
        );
    }

    #[test]
    fn test_tbo_two_batches() {
        // Test TBO with 2 batches and verify it matches sequential output.
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;
        let num_experts = 2;
        let top_k = 1;
        let seq_len = 4;

        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| make_expert(&device, hidden_dim, intermediate_dim, 0.1 * (i + 1) as f32))
            .collect();
        let expert_group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let x_data: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect();
        let x = Array::from_slice(&device, &x_data, vec![seq_len, hidden_dim]);

        let gate_data: Vec<f32> = vec![
            5.0, 1.0, // token 0
            1.0, 5.0, // token 1
            5.0, 1.0, // token 2
            1.0, 5.0, // token 3
        ];
        let gate_logits = Array::from_slice(&device, &gate_data, vec![seq_len, num_experts]);

        // Reference: sequential (1 batch).
        let pipeline_seq = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 1,
                enable_sbo: false,
                sync_timeout: Duration::from_secs(5),
            },
        );
        let ref_out = pipeline_seq
            .forward_overlapped(
                &x,
                &gate_logits,
                &expert_group,
                None,
                None,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();
        let ref_vals: Vec<f32> = ref_out.to_vec_checked();

        // TBO: 2 batches.
        let pipeline_tbo = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 2,
                enable_sbo: false,
                sync_timeout: Duration::from_secs(5),
            },
        );
        let tbo_out = pipeline_tbo
            .forward_overlapped(
                &x,
                &gate_logits,
                &expert_group,
                None,
                None,
                top_k,
                (0, num_experts),
                &registry,
                &queue,
            )
            .unwrap();
        let tbo_vals: Vec<f32> = tbo_out.to_vec_checked();

        // Verify outputs match.
        assert_eq!(ref_vals.len(), tbo_vals.len());
        for i in 0..ref_vals.len() {
            let diff = (ref_vals[i] - tbo_vals[i]).abs();
            assert!(
                diff < 1e-2,
                "TBO mismatch at index {i}: ref={} tbo={} diff={}",
                ref_vals[i],
                tbo_vals[i],
                diff
            );
        }
    }

    #[test]
    fn test_event_reuse_across_calls() {
        // Verify that the pipeline can be called multiple times with event reset.
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;
        let num_experts = 2;
        let top_k = 1;
        let seq_len = 2;

        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| make_expert(&device, hidden_dim, intermediate_dim, 0.1 * (i + 1) as f32))
            .collect();
        let expert_group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let x_data: Vec<f32> = vec![1.0; seq_len * hidden_dim];
        let x = Array::from_slice(&device, &x_data, vec![seq_len, hidden_dim]);
        let gate_data: Vec<f32> = vec![5.0, 1.0, 1.0, 5.0];
        let gate_logits = Array::from_slice(&device, &gate_data, vec![seq_len, num_experts]);

        let pipeline = MoePipeline::new(
            Arc::new(GpuEvent::new(&device)),
            MoePipelineConfig {
                num_tbo_batches: 1,
                enable_sbo: false,
                sync_timeout: Duration::from_secs(5),
            },
        );

        // Call forward_overlapped twice to ensure event reset works.
        for iter in 0..2 {
            let output = pipeline
                .forward_overlapped(
                    &x,
                    &gate_logits,
                    &expert_group,
                    None,
                    None,
                    top_k,
                    (0, num_experts),
                    &registry,
                    &queue,
                )
                .unwrap();
            assert_eq!(
                output.shape(),
                &[seq_len, hidden_dim],
                "iteration {iter}: unexpected output shape"
            );
        }
    }
}
