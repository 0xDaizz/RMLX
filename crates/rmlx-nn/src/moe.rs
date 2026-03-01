//! Mixture of Experts layer with top-k gating.

use std::sync::atomic::{AtomicU64, Ordering};

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::Linear;

/// Lightweight forward-pass metrics for MoE routing.
///
/// Records dispatch counts and tokens routed during `MoeLayer::forward()`.
/// Uses atomic counters so the struct can be shared behind `Arc` if needed.
pub struct MoeForwardMetrics {
    /// Total number of forward() calls.
    pub forward_count: AtomicU64,
    /// Total tokens routed across all forward() calls.
    pub total_tokens_routed: AtomicU64,
    /// Per-expert token count.
    pub expert_tokens: Vec<AtomicU64>,
    /// Number of experts tracked (0 = no per-expert tracking).
    pub num_experts: usize,
}

impl MoeForwardMetrics {
    /// Create metrics without per-expert tracking.
    pub fn new() -> Self {
        Self {
            forward_count: AtomicU64::new(0),
            total_tokens_routed: AtomicU64::new(0),
            expert_tokens: Vec::new(),
            num_experts: 0,
        }
    }

    /// Create metrics with per-expert token counters.
    pub fn with_experts(num_experts: usize) -> Self {
        Self {
            forward_count: AtomicU64::new(0),
            total_tokens_routed: AtomicU64::new(0),
            expert_tokens: (0..num_experts).map(|_| AtomicU64::new(0)).collect(),
            num_experts,
        }
    }

    pub fn record_forward(&self, tokens: u64) {
        self.forward_count.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_routed
            .fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record a token routed to a specific expert.
    pub fn record_expert_token(&self, expert_idx: usize) {
        if expert_idx < self.num_experts {
            self.expert_tokens[expert_idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Snapshot of per-expert token counts.
    pub fn expert_tokens_snapshot(&self) -> Vec<u64> {
        self.expert_tokens
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    pub fn forward_count(&self) -> u64 {
        self.forward_count.load(Ordering::Relaxed)
    }

    pub fn total_tokens_routed(&self) -> u64 {
        self.total_tokens_routed.load(Ordering::Relaxed)
    }
}

impl Default for MoeForwardMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    /// Expert capacity factor for dispatch (1.0 = exact, >1.0 = overprovisioned).
    /// Links to SharedBufferPool tier selection in distributed mode.
    /// Default: 1.0
    pub capacity_factor: f32,
}

impl MoeConfig {
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.num_experts == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: num_experts must be > 0".into(),
            ));
        }
        if self.num_experts_per_token == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: num_experts_per_token must be > 0".into(),
            ));
        }
        if self.num_experts_per_token > self.num_experts {
            return Err(KernelError::InvalidShape(format!(
                "MoeConfig: num_experts_per_token ({}) > num_experts ({})",
                self.num_experts_per_token, self.num_experts
            )));
        }
        if self.hidden_dim == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: hidden_dim must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// A single expert FFN (SwiGLU: gate * up then down projection).
pub struct Expert {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Expert {
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate_out = self.gate_proj.forward(x, registry, queue)?;
        let up_out = self.up_proj.forward(x, registry, queue)?;
        let gate_activated = ops::silu::silu(registry, &gate_out, queue)?;
        let hidden = ops::binary::mul(registry, &gate_activated, &up_out, queue)?;
        self.down_proj.forward(&hidden, registry, queue)
    }
}

pub struct MoeLayer {
    config: MoeConfig,
    gate: Option<Linear>,
    experts: Vec<Expert>,
    metrics: MoeForwardMetrics,
}

impl MoeLayer {
    /// Config-only constructor.
    pub fn new(config: MoeConfig) -> Result<Self, KernelError> {
        config.validate()?;
        let metrics = MoeForwardMetrics::with_experts(config.num_experts);
        Ok(Self {
            config,
            gate: None,
            experts: Vec::new(),
            metrics,
        })
    }

    /// Create MoE layer with pre-loaded gate and experts.
    pub fn from_layers(
        config: MoeConfig,
        gate: Linear,
        experts: Vec<Expert>,
    ) -> Result<Self, KernelError> {
        config.validate()?;
        if experts.len() != config.num_experts {
            return Err(KernelError::InvalidShape(format!(
                "experts count {} != config.num_experts {}",
                experts.len(),
                config.num_experts
            )));
        }
        let metrics = MoeForwardMetrics::with_experts(config.num_experts);
        Ok(Self {
            config,
            gate: Some(gate),
            experts,
            metrics,
        })
    }

    /// Forward pass for MoE.
    ///
    /// `x`: [seq_len, hidden_dim]
    /// Returns: [seq_len, hidden_dim]
    ///
    /// For each token:
    /// 1. Compute gate logits via gate projection
    /// 2. Find top-k experts by gate score
    /// 3. Route token through selected experts
    /// 4. Weighted-sum expert outputs
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let gate = self
            .gate
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("MoE: gate not loaded".into()))?;
        if self.experts.is_empty() {
            return Err(KernelError::InvalidShape("MoE: no experts loaded".into()));
        }

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Record metrics
        self.metrics.record_forward(seq_len as u64);

        // Gate logits: [seq_len, num_experts]
        let gate_logits = gate.forward(x, registry, queue)?;

        // Softmax over experts for each token: [seq_len, num_experts]
        let gate_probs = ops::softmax::softmax(registry, &gate_logits, queue)?;

        // Read gate probs to CPU for top-k selection (small enough for CPU routing)
        let probs_vec: Vec<f32> = gate_probs.to_vec_checked();

        // For each token, find top-k expert indices and weights
        let output = Array::zeros(dev, &[seq_len, hidden_dim], x.dtype());

        // Collect per-token expert outputs, then batch-copy into the result
        let mut tok_outputs: Vec<Option<Array>> = Vec::with_capacity(seq_len);

        for tok in 0..seq_len {
            let row_start = tok * num_experts;
            let row_probs = &probs_vec[row_start..row_start + num_experts];

            // Find top-k indices
            let mut indexed: Vec<(usize, f32)> = row_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Normalize top-k weights
            let top_entries: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();
            let weight_sum: f32 = top_entries.iter().map(|(_, w)| w).sum();
            let inv_sum = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                0.0
            };

            // Record per-expert token routing
            for (expert_idx, _raw_weight) in &top_entries {
                self.metrics.record_expert_token(*expert_idx);
            }

            // Extract token input: [1, hidden_dim]
            let tok_offset = x.offset() + tok * hidden_dim * elem_size;
            let tok_view = x.view(vec![1, hidden_dim], vec![hidden_dim, 1], tok_offset);
            let tok_input = ops::copy::copy(registry, &tok_view, queue)?;

            // Accumulate weighted expert outputs
            let mut first = true;
            let mut tok_output: Option<Array> = None;

            for (expert_idx, raw_weight) in &top_entries {
                let w = raw_weight * inv_sum;
                let expert_out = self.experts[*expert_idx].forward(&tok_input, registry, queue)?;

                // Scale expert output by weight
                let scale_data = vec![w; hidden_dim];
                let scale_arr = Array::from_slice(dev, &scale_data, vec![1, hidden_dim]);
                let scaled = ops::binary::mul(registry, &expert_out, &scale_arr, queue)?;

                tok_output = Some(if first {
                    first = false;
                    scaled
                } else {
                    let prev = tok_output.as_ref().ok_or_else(|| {
                        KernelError::InvalidShape(
                            "MoE: tok_output unexpectedly None in accumulation loop".into(),
                        )
                    })?;
                    ops::binary::add(registry, prev, &scaled, queue)?
                });
            }

            tok_outputs.push(tok_output);
        }

        // Batch-copy all token outputs into the result in a single command buffer
        let copy_kernel = match x.dtype() {
            rmlx_core::dtype::DType::Float32 => "copy_f32",
            rmlx_core::dtype::DType::Float16 => "copy_f16",
            rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "MoE forward: unsupported dtype {:?} for copy kernel",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;
        let cb = queue.new_command_buffer();
        for (tok, tok_out) in tok_outputs.iter().enumerate() {
            if let Some(ref out_tok) = tok_out {
                let dst_offset = tok * hidden_dim * elem_size;
                let dst_view = output.view(vec![1, hidden_dim], vec![hidden_dim, 1], dst_offset);
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(out_tok.metal_buffer()), out_tok.offset() as u64);
                enc.set_buffer(1, Some(dst_view.metal_buffer()), dst_view.offset() as u64);
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
        }
        cb.commit();
        cb.wait_until_completed();

        Ok(output)
    }

    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.config.num_experts_per_token
    }

    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Expert capacity factor.
    pub fn capacity_factor(&self) -> f32 {
        self.config.capacity_factor
    }

    /// Access forward-pass metrics.
    pub fn metrics(&self) -> &MoeForwardMetrics {
        &self.metrics
    }
}
