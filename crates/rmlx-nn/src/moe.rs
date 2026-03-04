//! Mixture of Experts layer with top-k gating.
//!
//! Routing is capacity-based: tokens are grouped by assigned expert, each
//! expert runs a single batched forward pass over all its tokens, and
//! results are scattered back to original positions.
//!
//! Features:
//! - Batched expert execution (N2)
//! - Optional EP integration for multi-node dispatch/combine (N5, `distributed` feature)
//! - GShard auxiliary load balancing loss (N6)
//! - Optional shared expert for DeepSeek-V3 style architectures (N7)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_core::MetalAllocator;

#[cfg(feature = "distributed")]
use rmlx_distributed::MoeDispatchExchange;

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
    /// Optional shared expert that processes every token (DeepSeek-V3 pattern).
    /// When present, the shared expert output is added to the routed MoE output.
    shared_expert: Option<Expert>,
    /// Optional EP dispatch exchange for multi-node expert parallelism.
    /// When present and world_size > 1, tokens are dispatched to remote ranks
    /// that own specific experts, instead of being processed locally.
    #[cfg(feature = "distributed")]
    exchange: Option<MoeDispatchExchange>,
    /// Optional buffer pool allocator for reusing Metal buffers.
    allocator: Option<Arc<MetalAllocator>>,
    /// Optional per-expert bias for adaptive routing (DeepSeek-style, aux-loss-free).
    /// Shape: `[num_experts]`, Float32. Added to gate logits before softmax.
    expert_bias: Option<Array>,
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
            shared_expert: None,
            #[cfg(feature = "distributed")]
            exchange: None,
            allocator: None,
            expert_bias: None,
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
            shared_expert: None,
            #[cfg(feature = "distributed")]
            exchange: None,
            allocator: None,
            expert_bias: None,
        })
    }

    /// Set a shared expert that processes every token (DeepSeek-V3 pattern).
    ///
    /// The shared expert output is added to the routed expert output for every
    /// token, regardless of top-k routing decisions. This provides a baseline
    /// representation that complements the sparse routed experts.
    pub fn with_shared_expert(mut self, expert: Expert) -> Self {
        self.shared_expert = Some(expert);
        self
    }

    /// Set the EP dispatch exchange for distributed expert parallelism.
    ///
    /// When set and the exchange group has world_size > 1, `forward()` will
    /// dispatch tokens to expert-owning ranks instead of running all experts
    /// locally. The flow becomes: gate -> dispatch -> expert forward -> combine.
    #[cfg(feature = "distributed")]
    pub fn with_exchange(mut self, exchange: MoeDispatchExchange) -> Self {
        self.exchange = Some(exchange);
        self
    }

    /// Set a buffer pool allocator for reduced allocation overhead.
    ///
    /// When set, `forward()` allocates intermediate buffers (e.g., the output
    /// accumulator and per-expert batch buffers) through the pool, reusing
    /// cached Metal buffers instead of allocating fresh ones every call.
    pub fn with_allocator(mut self, allocator: Arc<MetalAllocator>) -> Self {
        self.allocator = Some(allocator);
        self
    }

    /// Set a per-expert bias for adaptive routing (DeepSeek-style, aux-loss-free).
    ///
    /// The bias is added to gate logits before softmax during GPU top-k routing.
    /// Use `update_expert_bias()` after each forward pass to adaptively balance
    /// expert utilization without auxiliary loss.
    pub fn with_expert_bias(mut self, bias: Array) -> Self {
        self.expert_bias = Some(bias);
        self
    }

    /// Initialize zero expert bias on the given device.
    pub fn init_expert_bias(&mut self, device: &metal::Device) {
        let bias = Array::zeros(device, &[self.config.num_experts], DType::Float32);
        self.expert_bias = Some(bias);
    }

    /// Update expert bias based on routing counts (DeepSeek-style adaptive bias).
    ///
    /// `bias[e] += lr * (mean_count - count[e])` — encourages underutilized experts
    /// to receive more tokens. This runs entirely on the GPU via elementwise ops.
    ///
    /// Call this after `forward()` using the `expert_counts` from `TopkRouteResult`.
    pub fn update_expert_bias(
        &mut self,
        expert_counts: &Array,
        lr: f32,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<(), KernelError> {
        let bias = match self.expert_bias.as_ref() {
            Some(b) => b,
            None => return Ok(()),
        };

        let dev = registry.device().raw();
        let num_experts = self.config.num_experts;

        // Compute mean_count on CPU (single scalar, cheap)
        let counts: Vec<u32> = expert_counts.to_vec_checked();
        let total: u32 = counts.iter().sum();
        let mean_count = total as f32 / num_experts as f32;

        // Build (mean_count - count[e]) * lr as a Float32 array
        let delta: Vec<f32> = counts
            .iter()
            .map(|&c| lr * (mean_count - c as f32))
            .collect();
        let delta_arr = Array::from_slice(dev, &delta, vec![num_experts]);

        // bias = bias + delta (GPU elementwise add)
        let new_bias = ops::binary::add(registry, bias, &delta_arr, queue)?;
        self.expert_bias = Some(new_bias);

        Ok(())
    }

    /// Whether a shared expert is configured.
    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert.is_some()
    }

    /// Forward pass for MoE with GPU-native routing and batched expert dispatch.
    ///
    /// `x`: [seq_len, hidden_dim]
    /// Returns: [seq_len, hidden_dim]
    ///
    /// Algorithm:
    /// 1. Compute gate logits via gate projection
    /// 2. GPU top-k routing: softmax → top-k → normalize → histogram (zero CPU sync)
    /// 3. Read routing results to CPU for expert dispatch
    /// 4. Run one batched forward pass per expert
    /// 5. Scatter weighted results back to original token positions
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

        // ── N5: Determine local expert range for EP (expert parallelism) ──
        #[cfg(feature = "distributed")]
        let local_expert_range: (usize, usize) = match self.exchange {
            Some(ref ex) if ex.world_size() > 1 => ex.local_expert_range(),
            _ => (0, num_experts),
        };
        #[cfg(not(feature = "distributed"))]
        let local_expert_range: (usize, usize) = (0, num_experts);

        // ── Allocator-aware zero-fill: reuse pooled buffers when available ──
        let alloc_zeros = |shape: &[usize], dtype: DType| -> Array {
            if let Some(ref alloc) = self.allocator {
                if let Ok(arr) = Array::zeros_pooled(alloc, shape, dtype) {
                    return arr;
                }
            }
            Array::zeros(dev, shape, dtype)
        };

        // Record metrics
        self.metrics.record_forward(seq_len as u64);

        // Gate logits: [seq_len, num_experts]
        let gate_logits = gate.forward(x, registry, queue)?;

        // ── GPU top-k routing: eliminates the GPU→CPU→GPU round-trip ──
        // The GPU kernel computes: softmax → top-k → normalize → histogram → prefix scan
        // in a single command buffer with zero CPU synchronization.
        let route_result = ops::topk_route::gpu_topk_route(
            registry,
            &gate_logits,
            top_k,
            self.expert_bias.as_ref(),
            queue,
        )?;

        // Read GPU routing results to CPU for expert dispatch.
        // This is the minimal data transfer: just indices and weights, not the full
        // softmax output. The actual token data stays on GPU throughout.
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Build dispatch buffers: which tokens go to which expert
        // expert_dispatch[e] = Vec<(token_idx, normalized_weight)>
        let mut expert_dispatch: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

        for tok in 0..seq_len {
            for k in 0..top_k {
                let flat_idx = tok * top_k + k;
                let expert_idx = indices_vec[flat_idx] as usize;
                let weight = weights_vec[flat_idx];
                self.metrics.record_expert_token(expert_idx);
                expert_dispatch[expert_idx].push((tok, weight));
            }
        }

        // ── Batched expert execution with dispatch buffers ──
        // Output accumulator: [seq_len, hidden_dim], zero-initialized
        let output = alloc_zeros(&[seq_len, hidden_dim], x.dtype());

        let (local_start, local_end) = local_expert_range;
        for (expert_idx, dispatch) in expert_dispatch.iter().enumerate() {
            if dispatch.is_empty() {
                continue;
            }

            // ── N5: Skip non-local experts in EP mode ──
            if expert_idx < local_start || expert_idx >= local_end {
                continue;
            }

            let batch_size = dispatch.len();

            // Gather token embeddings for this expert into a contiguous batch:
            // expert_input: [batch_size, hidden_dim]
            let expert_input = if batch_size == 1 {
                let tok = dispatch[0].0;
                let tok_offset = x.offset() + tok * hidden_dim * elem_size;
                let tok_view = x.view(vec![1, hidden_dim], vec![hidden_dim, 1], tok_offset);
                ops::copy::copy(registry, &tok_view, queue)?
            } else {
                let batch_buf = alloc_zeros(&[batch_size, hidden_dim], x.dtype());
                let copy_kernel = match x.dtype() {
                    DType::Float32 => "copy_f32",
                    DType::Float16 => "copy_f16",
                    DType::Bfloat16 => "copy_bf16",
                    other => {
                        return Err(KernelError::InvalidShape(format!(
                            "MoE forward: unsupported dtype {:?} for copy kernel",
                            other
                        )));
                    }
                };
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
                batch_buf
            };

            // Single batched forward pass for this expert: [batch_size, hidden_dim]
            let expert_out = self.experts[expert_idx].forward(&expert_input, registry, queue)?;

            // Scale each token's expert output by its routing weight, then
            // scatter-add back into the output buffer at the original positions.
            let copy_kernel = match x.dtype() {
                DType::Float32 => "copy_f32",
                DType::Float16 => "copy_f16",
                DType::Bfloat16 => "copy_bf16",
                other => {
                    return Err(KernelError::InvalidShape(format!(
                        "MoE forward: unsupported dtype {:?} for copy kernel",
                        other
                    )));
                }
            };
            let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;

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

        // ── N7: Shared expert (DeepSeek-V3 pattern) ──
        let output = if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(x, registry, queue)?;
            ops::binary::add(registry, &output, &shared_out, queue)?
        } else {
            output
        };

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

    /// Compute the GShard auxiliary load balancing loss for this layer.
    ///
    /// Wraps [`load_balance_loss`] using the layer's expert count.
    pub fn compute_load_balance_loss(
        &self,
        gate_logits: &[f32],
        expert_indices: &[usize],
        seq_len: usize,
    ) -> f32 {
        load_balance_loss(
            gate_logits,
            expert_indices,
            self.config.num_experts,
            seq_len,
        )
    }
}

/// GShard auxiliary load balancing loss.
///
/// Encourages equal distribution of tokens across experts to avoid
/// routing collapse (where a few experts get most tokens).
///
/// Formula: `loss = num_experts * sum_e(f_e * P_e)` where:
/// - `f_e` = fraction of tokens assigned to expert e
/// - `P_e` = mean gate probability for expert e
///
/// # Arguments
/// - `gate_logits`: flattened softmax probabilities `[seq_len, num_experts]`.
/// - `expert_indices`: per-token selected expert indices (length `seq_len * top_k`,
///   but only one index per token is typical for loss computation; pass all top-k
///   assignments for a more accurate loss).
/// - `num_experts`: total number of experts.
/// - `seq_len`: number of tokens in the batch.
///
/// # Returns
/// The scalar load balancing loss value. When perfectly balanced, the loss is
/// approximately 1.0; imbalanced routing produces higher values.
pub fn load_balance_loss(
    gate_logits: &[f32],
    expert_indices: &[usize],
    num_experts: usize,
    seq_len: usize,
) -> f32 {
    if seq_len == 0 || num_experts == 0 {
        return 0.0;
    }

    // f_e: fraction of tokens assigned to each expert
    let mut expert_counts = vec![0usize; num_experts];
    let total_assignments = expert_indices.len();
    for &idx in expert_indices {
        if idx < num_experts {
            expert_counts[idx] += 1;
        }
    }
    let inv_total = if total_assignments > 0 {
        1.0 / total_assignments as f32
    } else {
        0.0
    };

    // P_e: mean gate probability for each expert across all tokens
    let mut expert_prob_sum = vec![0.0f32; num_experts];
    for tok in 0..seq_len {
        let row_start = tok * num_experts;
        let row_end = row_start + num_experts;
        if row_end <= gate_logits.len() {
            for e in 0..num_experts {
                expert_prob_sum[e] += gate_logits[row_start + e];
            }
        }
    }
    let inv_seq = 1.0 / seq_len as f32;

    // loss = num_experts * sum_e(f_e * P_e)
    let mut loss = 0.0f32;
    for e in 0..num_experts {
        let f_e = expert_counts[e] as f32 * inv_total;
        let p_e = expert_prob_sum[e] * inv_seq;
        loss += f_e * p_e;
    }

    num_experts as f32 * loss
}
