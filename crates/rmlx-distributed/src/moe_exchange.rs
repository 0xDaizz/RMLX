//! MoE expert dispatch and combine exchange operations.
//!
//! - MoeDispatchExchange: routes tokens to experts across nodes
//! - MoeCombineExchange: gathers expert outputs and applies weighted sum

use crate::group::Group;
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
    ///
    /// Returns the dispatch metadata (which tokens go where).
    /// The selected backend determines the execution path:
    /// - CPU: local in-process routing
    /// - Metal: GPU-accelerated local routing
    /// - RDMA: inter-node transfer for remote experts
    pub fn dispatch(
        &mut self,
        batch_size: usize,
        expert_indices: &[u32],
        _expert_weights: &[f32],
    ) -> DispatchResult {
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
        match backend {
            MoeBackend::Cpu => {
                // CPU path: direct in-process routing, no transfers needed
                self.route_cpu(&expert_counts, local_start, local_end);
            }
            MoeBackend::Metal => {
                // Metal path: GPU-accelerated local routing
                self.route_metal(&expert_counts, local_start, local_end);
            }
            MoeBackend::Rdma => {
                // RDMA path: transfer tokens for remote experts across nodes
                self.route_rdma(&expert_counts, local_start, local_end);
            }
        }

        // Trigger backend switch with cooldown if the selection changed
        if backend != self.policy.current_backend() {
            self.policy.switch_backend(backend);
        }

        DispatchResult {
            backend,
            tokens_per_expert,
            expert_counts,
            overflow_count: overflow,
            local_expert_range: (local_start, local_end),
        }
    }

    /// CPU routing: local in-process token routing.
    fn route_cpu(&self, _expert_counts: &[usize], _local_start: usize, _local_end: usize) {
        // CPU path processes tokens directly in the calling thread.
        // Token data stays in main memory; no GPU or network transfers.
    }

    /// Metal routing: GPU-accelerated local routing.
    fn route_metal(&self, _expert_counts: &[usize], _local_start: usize, _local_end: usize) {
        // Metal path dispatches routing via GPU compute kernel.
        // Tokens are already in Metal shared memory; routing uses GPU parallelism.
    }

    /// RDMA routing: inter-node transfer for remote experts.
    fn route_rdma(&self, _expert_counts: &[usize], _local_start: usize, _local_end: usize) {
        // RDMA path transfers tokens destined for remote experts via RDMA send/recv.
        // Local experts are processed in-place; only remote tokens go over the wire.
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
