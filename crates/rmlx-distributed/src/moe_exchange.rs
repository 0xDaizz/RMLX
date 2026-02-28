//! MoE expert dispatch and combine exchange operations.
//!
//! - MoeDispatchExchange: routes tokens to experts across nodes
//! - MoeCombineExchange: gathers expert outputs and applies weighted sum

use std::sync::Mutex;

use crate::group::{ensure_materialized, DistributedError, Group};
use crate::metrics::MoeMetrics as AtomicMoeMetrics;
use crate::moe_policy::{MoeBackend, MoePolicy};
use crate::sparse_guard::{GuardAction, SparseGuard};

/// Metal shader source for the gather/scatter routing kernel.
///
/// For each token, reads expert_index from `indices`, checks if the expert is
/// in the local range [local_start, local_end), and if so copies the token to
/// the output at the expert's offset. Uses atomic counters per expert to track
/// write positions.
const METAL_ROUTE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RouteParams {
    uint batch_size;
    uint top_k;
    uint token_stride;        // bytes per token
    uint local_start;
    uint local_end;
    uint capacity_per_expert;
    uint local_expert_count;
};

kernel void moe_gather_scatter(
    device const uchar*      token_data      [[buffer(0)]],
    device const uint*        expert_indices  [[buffer(1)]],
    device uchar*             output          [[buffer(2)]],
    device atomic_uint*       cursors         [[buffer(3)]],
    constant RouteParams&     params          [[buffer(4)]],
    uint                      tid             [[thread_position_in_grid]])
{
    uint batch_size = params.batch_size;
    uint top_k = params.top_k;
    uint total = batch_size * top_k;
    if (tid >= total) return;

    uint batch_idx = tid / top_k;
    uint expert = expert_indices[tid];

    if (expert < params.local_start || expert >= params.local_end) return;

    uint local_expert = expert - params.local_start;
    uint cursor = atomic_fetch_add_explicit(&cursors[local_expert], 1, memory_order_relaxed);
    if (cursor >= params.capacity_per_expert) return;

    uint src_offset = batch_idx * params.token_stride;
    uint dst_offset = (local_expert * params.capacity_per_expert + cursor) * params.token_stride;

    for (uint i = 0; i < params.token_stride; i++) {
        output[dst_offset + i] = token_data[src_offset + i];
    }
}
"#;

// MoeMetrics is now the atomic version from crate::metrics (re-exported as AtomicMoeMetrics).
// This unifies the previously duplicated non-atomic and atomic implementations.

/// MoE dispatch configuration.
pub struct MoeDispatchConfig {
    /// Number of experts total.
    pub num_experts: usize,
    /// Number of experts per token (top-k).
    pub top_k: usize,
    /// Expert capacity factor (1.0 = exact, >1.0 = overprovisioned).
    pub capacity_factor: f32,
    /// Group for inter-node communication.
    pub group: Group,
}

/// Cached Metal pipeline for route_metal to avoid per-dispatch JIT compilation.
struct CachedMetalPipeline {
    device: metal::Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

/// MoE dispatch exchange: routes tokens to the correct expert rank.
pub struct MoeDispatchExchange {
    config: MoeDispatchConfig,
    policy: MoePolicy,
    metrics: AtomicMoeMetrics,
    guard: SparseGuard,
    metal_cache: Mutex<Option<CachedMetalPipeline>>,
    /// Runtime capacity factor, modified by SparseGuard actions.
    /// Initially set from `config.capacity_factor`.
    runtime_capacity_factor: f32,
}

impl MoeDispatchExchange {
    pub fn new(config: MoeDispatchConfig, policy: MoePolicy) -> Self {
        let runtime_cf = config.capacity_factor;
        Self {
            config,
            policy,
            metrics: AtomicMoeMetrics::new(),
            guard: SparseGuard::new(),
            metal_cache: Mutex::new(None),
            runtime_capacity_factor: runtime_cf,
        }
    }

    /// Current runtime capacity factor (may differ from baseline after guard actions).
    pub fn runtime_capacity_factor(&self) -> f32 {
        self.runtime_capacity_factor
    }

    /// Dispatch tokens to experts based on routing decisions.
    ///
    /// `expert_indices`: [batch_size, top_k] — which expert each token goes to
    /// `expert_weights`: [batch_size, top_k] — gating weights
    /// `token_data`: flat token payload bytes (batch_size * hidden_dim * sizeof(f32))
    ///
    /// Returns the dispatch metadata (which tokens go where) and routed data.
    /// The selected backend determines the execution path:
    /// - CPU: local in-process routing
    /// - Metal: GPU-accelerated local routing
    /// - RDMA: inter-node transfer for remote experts
    pub fn dispatch(
        &mut self,
        batch_size: usize,
        expert_indices: &[u32],
        _expert_weights: &[f32],
        token_data: &[u8],
    ) -> Result<DispatchResult, DistributedError> {
        // Validate expert partition invariants
        let group_size = self.config.group.size();
        let num_experts = self.config.num_experts;
        if group_size > 0 && num_experts % group_size != 0 {
            return Err(DistributedError::Transport(format!(
                "num_experts ({num_experts}) must be divisible by group size ({group_size})"
            )));
        }
        if group_size > 0 && num_experts / group_size == 0 {
            return Err(DistributedError::Transport(format!(
                "experts_per_rank is 0 (num_experts={num_experts}, group_size={group_size})"
            )));
        }
        // Validate expert indices are in range
        for &idx in expert_indices {
            if (idx as usize) >= num_experts {
                return Err(DistributedError::Transport(format!(
                    "expert index {idx} out of range (num_experts={num_experts})"
                )));
            }
        }

        let n_elements = (batch_size * self.config.top_k) as u32;
        let byte_size = n_elements as usize * 4; // f32

        let backend = self.policy.select(n_elements, byte_size);
        self.policy.step();

        // Calculate capacity per expert using runtime capacity factor
        let tokens_per_expert = (batch_size as f32 * self.config.top_k as f32 / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;

        // Count tokens per expert and detect overflow
        let mut expert_counts = vec![0usize; num_experts];
        let mut overflow = 0u64;

        for &idx in expert_indices.iter() {
            let expert = idx as usize;
            if expert < self.config.num_experts {
                expert_counts[expert] += 1;
                if expert_counts[expert] > tokens_per_expert {
                    overflow += 1;
                }
            }
        }

        // Record metrics using atomic counters
        self.metrics.record_dispatch(batch_size as u64);
        match backend {
            MoeBackend::Cpu => self.metrics.record_cpu_dispatch(),
            MoeBackend::Metal => self.metrics.record_metal_dispatch(),
            MoeBackend::Rdma => self.metrics.record_rdma_dispatch(),
        }
        if overflow > 0 {
            self.metrics.record_overflow();
        }

        // Wire SparseGuard: record step and evaluate
        self.guard
            .record_step(overflow as usize, batch_size * self.config.top_k);
        match self.guard.evaluate() {
            GuardAction::IncreaseCapacity(new_factor) => {
                self.runtime_capacity_factor = new_factor as f32;
            }
            GuardAction::DenseFallback => {
                self.metrics.record_dense_fallback();
                self.policy.force_backend(Some(MoeBackend::Cpu));
            }
            GuardAction::Reset => {
                self.policy.force_backend(None);
                self.runtime_capacity_factor = self.config.capacity_factor;
            }
            GuardAction::None => {}
        }

        // Determine which experts are local vs remote
        let experts_per_rank = self.config.num_experts / self.config.group.size();
        let local_start = self.config.group.local_rank() as usize * experts_per_rank;
        let local_end = local_start + experts_per_rank;

        // Route based on selected backend
        let routed = match backend {
            MoeBackend::Cpu => self.route_cpu(
                token_data,
                expert_indices,
                &expert_counts,
                local_start,
                local_end,
            )?,
            MoeBackend::Metal => self.route_metal(
                token_data,
                expert_indices,
                &expert_counts,
                local_start,
                local_end,
            )?,
            MoeBackend::Rdma => self.route_rdma(
                token_data,
                expert_indices,
                &expert_counts,
                local_start,
                local_end,
            )?,
        };

        // Trigger backend switch with cooldown if the selection changed
        if backend != self.policy.current_backend() {
            self.policy.switch_backend(backend);
        }

        Ok(DispatchResult {
            backend,
            tokens_per_expert,
            expert_counts,
            overflow_count: overflow,
            local_expert_range: (local_start, local_end),
            routed_data: routed,
        })
    }

    /// CPU routing: local in-process token routing.
    ///
    /// Gathers tokens destined for local experts into a contiguous output buffer.
    /// Tokens for remote experts are collected in a separate buffer (for potential
    /// later RDMA send or are simply ignored in pure CPU mode).
    fn route_cpu(
        &self,
        token_data: &[u8],
        expert_indices: &[u32],
        expert_counts: &[usize],
        local_start: usize,
        local_end: usize,
    ) -> Result<Vec<u8>, DistributedError> {
        let num_experts = self.config.num_experts;
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let token_stride = token_data.len() / batch_size;

        // Compute capacity per expert
        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.config.capacity_factor)
            .ceil() as usize;

        // Allocate output: local experts * capacity * token_stride
        let local_expert_count = local_end - local_start;
        let mut output = vec![0u8; local_expert_count * capacity_per_expert * token_stride];

        // Track per-expert write cursors (only for local experts)
        let mut cursors = vec![0usize; local_expert_count];

        // Scatter tokens to correct expert slots
        for batch_idx in 0..batch_size {
            for k in 0..self.config.top_k {
                let flat_idx = batch_idx * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;

                if expert >= local_start && expert < local_end {
                    let local_expert = expert - local_start;
                    let cursor = cursors[local_expert];
                    if cursor < capacity_per_expert {
                        let src_start = batch_idx * token_stride;
                        let src_end = src_start + token_stride;
                        let dst_start =
                            (local_expert * capacity_per_expert + cursor) * token_stride;
                        let dst_end = dst_start + token_stride;

                        if src_end <= token_data.len() && dst_end <= output.len() {
                            output[dst_start..dst_end]
                                .copy_from_slice(&token_data[src_start..src_end]);
                        }
                        cursors[local_expert] += 1;
                    }
                    // else: overflow, token dropped
                }
            }
        }

        // Ignore expert_counts beyond what we used — kept for API consistency
        let _ = expert_counts;

        Ok(output)
    }

    /// Metal routing: GPU-accelerated local gather/scatter.
    ///
    /// Uses a cached pipeline state to avoid per-dispatch JIT compilation.
    /// Falls back to the CPU path if the Metal device is not available or
    /// if shader compilation fails.
    fn route_metal(
        &self,
        token_data: &[u8],
        expert_indices: &[u32],
        expert_counts: &[usize],
        local_start: usize,
        local_end: usize,
    ) -> Result<Vec<u8>, DistributedError> {
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Get or create cached pipeline
        let mut cache_guard = self.metal_cache.lock().unwrap();
        if cache_guard.is_none() {
            let device = match metal::Device::system_default() {
                Some(d) => d,
                None => {
                    drop(cache_guard);
                    return self.route_cpu(
                        token_data,
                        expert_indices,
                        expert_counts,
                        local_start,
                        local_end,
                    );
                }
            };
            let options = metal::CompileOptions::new();
            let library = match device.new_library_with_source(METAL_ROUTE_KERNEL, &options) {
                Ok(lib) => lib,
                Err(_) => {
                    drop(cache_guard);
                    return self.route_cpu(
                        token_data,
                        expert_indices,
                        expert_counts,
                        local_start,
                        local_end,
                    );
                }
            };
            let function = match library.get_function("moe_gather_scatter", None) {
                Ok(f) => f,
                Err(_) => {
                    drop(cache_guard);
                    return self.route_cpu(
                        token_data,
                        expert_indices,
                        expert_counts,
                        local_start,
                        local_end,
                    );
                }
            };
            let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
                Ok(p) => p,
                Err(_) => {
                    drop(cache_guard);
                    return self.route_cpu(
                        token_data,
                        expert_indices,
                        expert_counts,
                        local_start,
                        local_end,
                    );
                }
            };
            let queue = device.new_command_queue();
            *cache_guard = Some(CachedMetalPipeline {
                device,
                pipeline,
                queue,
            });
        }
        let cached = cache_guard.as_ref().unwrap();

        let token_stride = token_data.len() / batch_size;
        let num_experts = self.config.num_experts;
        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.config.capacity_factor)
            .ceil() as usize;
        let local_expert_count = local_end - local_start;
        let output_size = local_expert_count * capacity_per_expert * token_stride;

        // Create buffers using cached device
        let shared = metal::MTLResourceOptions::StorageModeShared;
        let token_buf = cached.device.new_buffer_with_data(
            token_data.as_ptr() as *const std::ffi::c_void,
            token_data.len() as u64,
            shared,
        );
        let indices_buf = cached.device.new_buffer_with_data(
            expert_indices.as_ptr() as *const std::ffi::c_void,
            (expert_indices.len() * 4) as u64,
            shared,
        );
        let output_buf = cached.device.new_buffer(output_size.max(1) as u64, shared);
        let cursor_data = vec![0u32; local_expert_count];
        let cursor_buf = cached.device.new_buffer_with_data(
            cursor_data.as_ptr() as *const std::ffi::c_void,
            (local_expert_count * 4).max(4) as u64,
            shared,
        );

        #[repr(C)]
        struct RouteParams {
            batch_size: u32,
            top_k: u32,
            token_stride: u32,
            local_start: u32,
            local_end: u32,
            capacity_per_expert: u32,
            local_expert_count: u32,
        }
        let params = RouteParams {
            batch_size: batch_size as u32,
            top_k: self.config.top_k as u32,
            token_stride: token_stride as u32,
            local_start: local_start as u32,
            local_end: local_end as u32,
            capacity_per_expert: capacity_per_expert as u32,
            local_expert_count: local_expert_count as u32,
        };
        let params_buf = cached.device.new_buffer_with_data(
            &params as *const RouteParams as *const std::ffi::c_void,
            std::mem::size_of::<RouteParams>() as u64,
            shared,
        );

        // Encode and dispatch using cached queue and pipeline
        let cmd_buf = cached.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&cached.pipeline);
        encoder.set_buffer(0, Some(&token_buf), 0);
        encoder.set_buffer(1, Some(&indices_buf), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        encoder.set_buffer(3, Some(&cursor_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        let total_threads = (batch_size * self.config.top_k) as u64;
        let max_tg = cached.pipeline.max_total_threads_per_threadgroup();
        let tg_size = total_threads.min(max_tg);

        encoder.dispatch_threads(
            metal::MTLSize::new(total_threads, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read output
        if output_size == 0 {
            return Ok(Vec::new());
        }
        let output_ptr = output_buf.contents() as *const u8;
        let output = unsafe { std::slice::from_raw_parts(output_ptr, output_size) };

        let _ = expert_counts;

        Ok(output.to_vec())
    }

    /// RDMA routing: inter-node transfer for remote experts.
    ///
    /// Tokens destined for local experts are gathered in-place (same as CPU path).
    /// Tokens destined for remote experts are collected per-peer-rank and sent
    /// via `Group.send()`. Tokens from remote peers are received via `Group.recv()`
    /// and merged into the local output.
    ///
    /// Calls `ensure_materialized()` before any RDMA transfer to ensure buffers
    /// contain valid data.
    fn route_rdma(
        &self,
        token_data: &[u8],
        expert_indices: &[u32],
        expert_counts: &[usize],
        local_start: usize,
        local_end: usize,
    ) -> Result<Vec<u8>, DistributedError> {
        // Ensure token data is materialized before RDMA transfers
        ensure_materialized(&[(token_data.len(), token_data.len())])?;

        let num_experts = self.config.num_experts;
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let token_stride = token_data.len() / batch_size;
        let experts_per_rank = num_experts / self.config.group.size();

        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.config.capacity_factor)
            .ceil() as usize;

        // --- Local expert gather (same as CPU path) ---
        let local_expert_count = local_end - local_start;
        let mut local_output = vec![0u8; local_expert_count * capacity_per_expert * token_stride];
        let mut local_cursors = vec![0usize; local_expert_count];

        // --- Remote expert buffers: one Vec per peer rank ---
        let world_size = self.config.group.size();
        let _local_rank = self.config.group.local_rank() as usize;
        let mut remote_buffers: Vec<Vec<u8>> = vec![Vec::new(); world_size];

        for batch_idx in 0..batch_size {
            for k in 0..self.config.top_k {
                let flat_idx = batch_idx * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;

                let src_start = batch_idx * token_stride;
                let src_end = src_start + token_stride;
                if src_end > token_data.len() {
                    continue;
                }

                if expert >= local_start && expert < local_end {
                    // Local expert
                    let local_expert = expert - local_start;
                    let cursor = local_cursors[local_expert];
                    if cursor < capacity_per_expert {
                        let dst_start =
                            (local_expert * capacity_per_expert + cursor) * token_stride;
                        let dst_end = dst_start + token_stride;
                        if dst_end <= local_output.len() {
                            local_output[dst_start..dst_end]
                                .copy_from_slice(&token_data[src_start..src_end]);
                        }
                        local_cursors[local_expert] += 1;
                    }
                } else {
                    // Remote expert — buffer for the owning rank
                    // Prepend expert_id (u32 LE) so receiver can place in correct slot
                    let target_rank = expert / experts_per_rank;
                    if target_rank < world_size {
                        let expert_id_bytes = (expert as u32).to_le_bytes();
                        remote_buffers[target_rank].extend_from_slice(&expert_id_bytes);
                        remote_buffers[target_rank]
                            .extend_from_slice(&token_data[src_start..src_end]);
                    }
                }
            }
        }

        // --- Phase 1: Exchange payload sizes with each peer via sendrecv ---
        let mut recv_sizes: Vec<usize> = vec![0; world_size];
        for &peer_rank in self.config.group.peers().iter() {
            let pr = peer_rank as usize;
            let size_bytes = (remote_buffers[pr].len() as u64).to_le_bytes();
            let size_data = self
                .config
                .group
                .sendrecv(&size_bytes, peer_rank, 8, peer_rank)
                .map_err(|e| {
                    DistributedError::Transport(format!("size sendrecv with rank {peer_rank}: {e}"))
                })?;
            recv_sizes[pr] = u64::from_le_bytes(size_data[..8].try_into().unwrap()) as usize;
        }

        // --- Phase 2: Exchange actual payloads via sendrecv ---
        let wire_stride = 4 + token_stride; // 4-byte expert_id prefix + token data
        for &peer_rank in self.config.group.peers().iter() {
            let pr = peer_rank as usize;
            let recv_size = recv_sizes[pr];
            let send_buf = &remote_buffers[pr];
            // If nothing to send, use an empty placeholder but still recv
            // If nothing to recv either, skip entirely
            if send_buf.is_empty() && recv_size == 0 {
                continue;
            }
            // Use sendrecv: send our payload, receive peer's payload
            let recv_len = if recv_size > 0 { recv_size } else { 4 }; // min 4 to avoid zero-len
            let send_data = if send_buf.is_empty() {
                &[0u8; 4][..]
            } else {
                send_buf
            };
            let received = self
                .config
                .group
                .sendrecv(send_data, peer_rank, recv_len, peer_rank)
                .map_err(|e| {
                    DistributedError::Transport(format!(
                        "payload sendrecv with rank {peer_rank}: {e}"
                    ))
                })?;
            if recv_size == 0 {
                continue;
            }

            // Merge received tokens into local_output using expert_id prefix
            let mut offset = 0;
            while offset + wire_stride <= received.len() {
                let expert_id =
                    u32::from_le_bytes(received[offset..offset + 4].try_into().unwrap()) as usize;
                if expert_id >= local_start && expert_id < local_end {
                    let local_expert_idx = expert_id - local_start;
                    let cursor = local_cursors[local_expert_idx];
                    if cursor < capacity_per_expert {
                        let dst_start =
                            (local_expert_idx * capacity_per_expert + cursor) * token_stride;
                        let dst_end = dst_start + token_stride;
                        if dst_end <= local_output.len() {
                            local_output[dst_start..dst_end]
                                .copy_from_slice(&received[offset + 4..offset + 4 + token_stride]);
                            local_cursors[local_expert_idx] += 1;
                        }
                    }
                }
                offset += wire_stride;
            }
        }

        let _ = expert_counts;

        Ok(local_output)
    }

    /// Get current metrics (atomic).
    pub fn metrics(&self) -> &AtomicMoeMetrics {
        &self.metrics
    }

    /// Get sparse guard reference.
    pub fn guard(&self) -> &SparseGuard {
        &self.guard
    }

    /// Get mutable sparse guard reference.
    pub fn guard_mut(&mut self) -> &mut SparseGuard {
        &mut self.guard
    }

    /// Get mutable policy reference (for threshold updates).
    pub fn policy_mut(&mut self) -> &mut MoePolicy {
        &mut self.policy
    }

    /// Get policy reference.
    pub fn policy(&self) -> &MoePolicy {
        &self.policy
    }
}

/// Result of a dispatch operation.
#[derive(Debug)]
pub struct DispatchResult {
    /// Which backend was used.
    pub backend: MoeBackend,
    /// Max tokens per expert (capacity).
    pub tokens_per_expert: usize,
    /// Actual token count per expert.
    pub expert_counts: Vec<usize>,
    /// Number of overflow tokens.
    pub overflow_count: u64,
    /// Local expert index range [start, end).
    pub local_expert_range: (usize, usize),
    /// Routed token data for local experts.
    pub routed_data: Vec<u8>,
}

/// MoE combine exchange: gathers expert outputs and applies weighted sum.
pub struct MoeCombineExchange {
    group: Group,
}

impl MoeCombineExchange {
    pub fn new(group: Group) -> Self {
        Self { group }
    }

    /// Combine expert outputs with gating weights (CPU fallback).
    /// `expert_outputs`: Vec of expert output arrays (one per active expert)
    /// `weights`: [batch_size, top_k] gating weights
    /// `indices`: [batch_size, top_k] expert indices
    ///
    /// Returns combined output for each token.
    pub fn combine_cpu(
        &self,
        expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * hidden_dim];

        for batch_idx in 0..batch_size {
            for k in 0..top_k {
                let flat_idx = batch_idx * top_k + k;
                let expert_idx = indices[flat_idx] as usize;
                let weight = weights[flat_idx];

                if expert_idx < expert_outputs.len() {
                    let expert_out = &expert_outputs[expert_idx];
                    // Weighted accumulate into output
                    let out_base = batch_idx * hidden_dim;
                    let exp_base = batch_idx * hidden_dim;
                    if exp_base + hidden_dim <= expert_out.len() {
                        for h in 0..hidden_dim {
                            output[out_base + h] += weight * expert_out[exp_base + h];
                        }
                    }
                }
            }
        }

        output
    }

    /// Metal-accelerated combine: uses GPU for weighted sum.
    ///
    /// Launches a scatter-add kernel that computes, for each output element:
    ///   output[batch_idx * hidden_dim + h] = sum_k weight[batch_idx * top_k + k]
    ///       * expert_outputs[indices[batch_idx * top_k + k]][batch_idx * hidden_dim + h]
    ///
    /// Each thread handles one (batch_idx, h) pair across all top_k experts,
    /// avoiding atomic conflicts. Falls back to combine_cpu if Metal is unavailable
    /// or shader compilation fails.
    pub fn combine_metal(
        &self,
        expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => {
                return self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                );
            }
        };

        if batch_size == 0 || hidden_dim == 0 {
            return vec![0.0f32; batch_size * hidden_dim];
        }

        let num_experts = expert_outputs.len();
        if num_experts == 0 {
            return vec![0.0f32; batch_size * hidden_dim];
        }

        // Flatten expert outputs into a single contiguous buffer:
        // expert_data[expert_idx * batch_size * hidden_dim + batch_idx * hidden_dim + h]
        let expert_stride = batch_size * hidden_dim;
        let mut expert_data = vec![0.0f32; num_experts * expert_stride];
        for (e, expert_out) in expert_outputs.iter().enumerate() {
            let copy_len = expert_out.len().min(expert_stride);
            expert_data[e * expert_stride..e * expert_stride + copy_len]
                .copy_from_slice(&expert_out[..copy_len]);
        }

        // Compile shader
        let kernel_src = r#"
#include <metal_stdlib>
using namespace metal;

struct CombineParams {
    uint batch_size;
    uint top_k;
    uint hidden_dim;
    uint num_experts;
    uint expert_stride; // batch_size * hidden_dim
};

kernel void moe_combine(
    device const float*    expert_data   [[buffer(0)]],
    device const float*    weights       [[buffer(1)]],
    device const uint*     indices       [[buffer(2)]],
    device float*          output        [[buffer(3)]],
    constant CombineParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = params.batch_size * params.hidden_dim;
    if (tid >= total) return;

    uint batch_idx = tid / params.hidden_dim;
    uint h = tid % params.hidden_dim;

    float sum = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        uint flat_idx = batch_idx * params.top_k + k;
        uint expert_idx = indices[flat_idx];
        float w = weights[flat_idx];
        if (expert_idx < params.num_experts) {
            uint offset = expert_idx * params.expert_stride + batch_idx * params.hidden_dim + h;
            sum += w * expert_data[offset];
        }
    }
    output[tid] = sum;
}
"#;

        let options = metal::CompileOptions::new();
        let library = match device.new_library_with_source(kernel_src, &options) {
            Ok(lib) => lib,
            Err(_) => {
                return self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                );
            }
        };
        let function = match library.get_function("moe_combine", None) {
            Ok(f) => f,
            Err(_) => {
                return self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                );
            }
        };
        let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
            Ok(p) => p,
            Err(_) => {
                return self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                );
            }
        };

        let shared = metal::MTLResourceOptions::StorageModeShared;
        let expert_buf = device.new_buffer_with_data(
            expert_data.as_ptr() as *const std::ffi::c_void,
            (expert_data.len() * 4) as u64,
            shared,
        );
        let weights_buf = device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * 4) as u64,
            shared,
        );
        let indices_buf = device.new_buffer_with_data(
            indices.as_ptr() as *const std::ffi::c_void,
            (indices.len() * 4) as u64,
            shared,
        );
        let output_size = batch_size * hidden_dim;
        let output_buf = device.new_buffer((output_size * 4).max(4) as u64, shared);

        #[repr(C)]
        struct CombineParams {
            batch_size: u32,
            top_k: u32,
            hidden_dim: u32,
            num_experts: u32,
            expert_stride: u32,
        }
        let params = CombineParams {
            batch_size: batch_size as u32,
            top_k: top_k as u32,
            hidden_dim: hidden_dim as u32,
            num_experts: num_experts as u32,
            expert_stride: expert_stride as u32,
        };
        let params_buf = device.new_buffer_with_data(
            &params as *const CombineParams as *const std::ffi::c_void,
            std::mem::size_of::<CombineParams>() as u64,
            shared,
        );

        let queue = device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&expert_buf), 0);
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&indices_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        let total_threads = output_size as u64;
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = total_threads.min(max_tg);

        encoder.dispatch_threads(
            metal::MTLSize::new(total_threads, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let output_ptr = output_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(output_ptr, output_size) };
        result.to_vec()
    }

    /// RDMA combine: gathers expert outputs from remote ranks, then combines.
    ///
    /// Remote expert outputs are received via the group's RDMA transport,
    /// merged with local expert outputs, then combined with gating weights.
    #[allow(clippy::too_many_arguments)]
    pub fn combine_rdma(
        &self,
        local_expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
        num_experts: usize,
    ) -> Result<Vec<f32>, DistributedError> {
        // Validate expert partition invariants
        let group_size = self.group.size();
        if group_size > 0 && num_experts % group_size != 0 {
            return Err(DistributedError::Transport(format!(
                "num_experts ({num_experts}) must be divisible by group size ({group_size})"
            )));
        }
        if group_size > 0 && num_experts / group_size == 0 {
            return Err(DistributedError::Transport(format!(
                "experts_per_rank is 0 (num_experts={num_experts}, group_size={group_size})"
            )));
        }
        // Validate expert indices are in range
        for &idx in indices {
            if (idx as usize) >= num_experts {
                return Err(DistributedError::Transport(format!(
                    "expert index {idx} out of range (num_experts={num_experts})"
                )));
            }
        }

        // Ensure local outputs are materialized
        for (i, output) in local_expert_outputs.iter().enumerate() {
            ensure_materialized(&[(output.len(), output.len() * 4)]).map_err(|_| {
                DistributedError::NotMaterialized(format!("expert output {i} not materialized"))
            })?;
        }

        // Single-rank or no-transport fast path: just use local data
        // Multi-rank: exchange expert outputs via sendrecv, then combine
        let world_size = self.group.size();
        if world_size <= 1 || !self.group.has_transport() {
            return Ok(self.combine_cpu(
                local_expert_outputs,
                weights,
                indices,
                batch_size,
                top_k,
                hidden_dim,
            ));
        }

        // Build full expert outputs array (local + placeholder for remote)
        let mut all_expert_outputs = vec![vec![0.0f32; batch_size * hidden_dim]; num_experts];
        let experts_per_rank = num_experts / world_size;
        let local_start = self.group.local_rank() as usize * experts_per_rank;

        // Fill in local expert outputs
        for (i, local_out) in local_expert_outputs.iter().enumerate() {
            let global_idx = local_start + i;
            if global_idx < num_experts && !local_out.is_empty() {
                let copy_len = local_out.len().min(all_expert_outputs[global_idx].len());
                all_expert_outputs[global_idx][..copy_len].copy_from_slice(&local_out[..copy_len]);
            }
        }

        // Exchange expert outputs with peers via sendrecv
        let byte_len = batch_size * hidden_dim * 4;
        for &peer_rank in self.group.peers().iter() {
            let peer_start = peer_rank as usize * experts_per_rank;
            for i in 0..experts_per_rank {
                let local_expert_idx = local_start + i;
                let peer_expert_idx = peer_start + i;
                if local_expert_idx >= num_experts || peer_expert_idx >= num_experts {
                    continue;
                }
                let send_bytes: Vec<u8> = all_expert_outputs[local_expert_idx]
                    .iter()
                    .flat_map(|f| f.to_ne_bytes())
                    .collect();
                let received = self
                    .group
                    .sendrecv(&send_bytes, peer_rank, byte_len, peer_rank)?;
                for (j, chunk) in received.chunks_exact(4).enumerate() {
                    if j < all_expert_outputs[peer_expert_idx].len() {
                        all_expert_outputs[peer_expert_idx][j] =
                            f32::from_ne_bytes(chunk.try_into().unwrap());
                    }
                }
            }
        }

        Ok(self.combine_cpu(
            &all_expert_outputs,
            weights,
            indices,
            batch_size,
            top_k,
            hidden_dim,
        ))
    }

    /// Group reference.
    pub fn group(&self) -> &Group {
        &self.group
    }
}
