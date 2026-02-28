//! MoE expert dispatch and combine exchange operations.
//!
//! - MoeDispatchExchange: routes tokens to experts across nodes
//! - MoeCombineExchange: gathers expert outputs and applies weighted sum

use crate::group::{ensure_materialized, DistributedError, Group};
use crate::moe_policy::{MoeBackend, MoePolicy};

/// Overflow tracking metrics for MoE dispatch.
#[derive(Debug, Clone, Default)]
pub struct MoeMetrics {
    /// Number of tokens dispatched.
    pub tokens_dispatched: u64,
    /// Number of tokens that overflowed expert capacity.
    pub overflow_count: u64,
    /// Backend selection counts.
    pub cpu_dispatches: u64,
    pub metal_dispatches: u64,
    pub rdma_dispatches: u64,
}

impl MoeMetrics {
    /// Overflow ratio = overflow_count / tokens_dispatched.
    pub fn overflow_ratio(&self) -> f64 {
        if self.tokens_dispatched == 0 {
            0.0
        } else {
            self.overflow_count as f64 / self.tokens_dispatched as f64
        }
    }

    /// Record a dispatch event.
    pub fn record_dispatch(&mut self, n_tokens: u64, n_overflow: u64, backend: MoeBackend) {
        self.tokens_dispatched += n_tokens;
        self.overflow_count += n_overflow;
        match backend {
            MoeBackend::Cpu => self.cpu_dispatches += 1,
            MoeBackend::Metal => self.metal_dispatches += 1,
            MoeBackend::Rdma => self.rdma_dispatches += 1,
        }
    }
}

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

/// MoE dispatch exchange: routes tokens to the correct expert rank.
pub struct MoeDispatchExchange {
    config: MoeDispatchConfig,
    policy: MoePolicy,
    metrics: MoeMetrics,
}

impl MoeDispatchExchange {
    pub fn new(config: MoeDispatchConfig, policy: MoePolicy) -> Self {
        Self {
            config,
            policy,
            metrics: MoeMetrics::default(),
        }
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
        let n_elements = (batch_size * self.config.top_k) as u32;
        let byte_size = n_elements as usize * 4; // f32

        let backend = self.policy.select(n_elements, byte_size);
        self.policy.step();

        // Calculate capacity per expert
        let tokens_per_expert = (batch_size as f32 * self.config.top_k as f32
            / self.config.num_experts as f32
            * self.config.capacity_factor)
            .ceil() as usize;

        // Count tokens per expert and detect overflow
        let mut expert_counts = vec![0usize; self.config.num_experts];
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

        self.metrics
            .record_dispatch(batch_size as u64, overflow, backend);

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

    /// Metal routing: GPU-accelerated local routing.
    ///
    /// Uses the same gather/scatter logic as CPU but is intended to be dispatched
    /// via a Metal compute kernel for parallel execution. Currently delegates to
    /// the CPU path as a functional fallback — the Metal kernel dispatch will be
    /// wired once rmlx-metal's compute pipeline is integrated.
    fn route_metal(
        &self,
        token_data: &[u8],
        expert_indices: &[u32],
        expert_counts: &[usize],
        local_start: usize,
        local_end: usize,
    ) -> Result<Vec<u8>, DistributedError> {
        // Metal path: in production this will encode a gather/scatter compute
        // command buffer. For now, functionally equivalent to the CPU path.
        // The Metal shared-memory buffers will be wired when rmlx-metal provides
        // the encoder/command-buffer API.
        self.route_cpu(
            token_data,
            expert_indices,
            expert_counts,
            local_start,
            local_end,
        )
    }

    /// RDMA routing: inter-node transfer for remote experts.
    ///
    /// Tokens destined for local experts are gathered in-place (same as CPU path).
    /// Tokens destined for remote experts are collected per-peer-rank and would be
    /// sent via `Group.send()` / `Group.recv()` in a real multi-node setup.
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
                    let target_rank = expert / experts_per_rank;
                    if target_rank < world_size {
                        remote_buffers[target_rank]
                            .extend_from_slice(&token_data[src_start..src_end]);
                    }
                }
            }
        }

        // In a real multi-node setup, remote_buffers would be sent via:
        //   for (rank, buf) in remote_buffers.iter().enumerate() {
        //       if rank != local_rank { self.config.group.send(rank, buf); }
        //   }
        // And received tokens from peers would be appended to local_output.
        // For now, remote buffers are computed but not transmitted — the
        // Group transport layer will be wired in the RDMA integration phase.

        let _ = expert_counts;
        let _ = &remote_buffers;

        Ok(local_output)
    }

    /// Get current metrics.
    pub fn metrics(&self) -> &MoeMetrics {
        &self.metrics
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

    /// Group reference.
    pub fn group(&self) -> &Group {
        &self.group
    }
}
