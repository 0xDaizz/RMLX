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
#[cfg(feature = "distributed")]
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_core::MetalAllocator;
use rmlx_metal::icb_sparse::{IcbReplayCache, SparseExpertPlan};
// TODO(Phase 6b): use grouped_forward_icb for direct ICB GEMM replay
// use rmlx_metal::icb_sparse::grouped_forward_icb;

#[cfg(feature = "distributed")]
use rmlx_distributed::{DispatchResult, MoeCombineExchange, MoeDispatchExchange};

use crate::expert_group::ExpertGroup;
use crate::linear::Linear;
use crate::moe_pipeline::MoePipeline;

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

/// Strategy for expert dispatch in the forward pass.
///
/// - `PerExpert`: O(E) per-expert loop with individual command buffers (default, backward-compatible).
/// - `Grouped`: Batched gather → grouped GEMM → batched scatter in minimal command buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoeStrategy {
    /// Per-expert loop: one gather + forward + scatter per active expert.
    /// This is the original behavior with ~2E sync points per forward.
    #[default]
    PerExpert,
    /// Grouped dispatch: all experts in a single command buffer via ExpertGroup.
    /// Reduces sync points from ~2E to 1.
    Grouped,
    /// GatherMM dispatch: uses `ops::gather_mm::gather_mm` to process all experts
    /// in a single GPU call per projection (gate, up, down). No per-expert loop.
    /// Requires stacked weights from ExpertGroup.
    GatherMM,
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
    /// Wrapped in Mutex because dispatch() requires &mut self while forward() takes &self.
    #[cfg(feature = "distributed")]
    exchange: Option<Mutex<MoeDispatchExchange>>,
    /// Optional buffer pool allocator for reusing Metal buffers.
    allocator: Option<Arc<MetalAllocator>>,
    /// Optional per-expert bias for adaptive routing (DeepSeek-style, aux-loss-free).
    /// Shape: `[num_experts]`, Float32. Added to gate logits before softmax.
    expert_bias: Option<Array>,
    /// Dispatch strategy: PerExpert (default) or Grouped.
    strategy: MoeStrategy,
    /// Lazily-initialized ExpertGroup for grouped forward.
    /// Built from `self.experts` on first use via `ensure_expert_group()`.
    expert_group: OnceLock<ExpertGroup>,
    /// Optional pipeline for SBO/TBO overlapped execution (Phase 4).
    pipeline: Option<MoePipeline>,
    /// Lazily-initialized ICB sparse plan for skipping empty experts (Phase 6a).
    sparse_plan: OnceLock<SparseExpertPlan>,
    /// Per-sparsity-pattern ICB replay cache (PR 4.11).
    /// When the same set of active experts recurs, skip re-analysis overhead.
    icb_replay_cache: std::sync::Mutex<IcbReplayCache>,
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
            strategy: MoeStrategy::default(),
            expert_group: OnceLock::new(),
            pipeline: None,
            sparse_plan: OnceLock::new(),
            icb_replay_cache: std::sync::Mutex::new(IcbReplayCache::new(64)),
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
            strategy: MoeStrategy::default(),
            expert_group: OnceLock::new(),
            pipeline: None,
            sparse_plan: OnceLock::new(),
            icb_replay_cache: std::sync::Mutex::new(IcbReplayCache::new(64)),
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
        self.exchange = Some(Mutex::new(exchange));
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

    /// Set the dispatch strategy for expert execution.
    ///
    /// - `PerExpert` (default): per-expert loop with individual command buffers.
    /// - `Grouped`: batched gather → grouped GEMM → batched scatter in minimal CBs.
    pub fn with_strategy(mut self, strategy: MoeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the MoePipeline for SBO/TBO overlapped execution.
    ///
    /// When set with `MoeStrategy::Grouped`, `forward()` will use the pipeline
    /// to overlap shared expert computation (SBO) or batch dispatch/compute (TBO).
    pub fn with_pipeline(mut self, pipeline: MoePipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Pre-set a sparse expert plan for ICB dispatch.
    ///
    /// When set, `forward_grouped()` will auto-select the ICB path when
    /// more than 50% of experts are empty (sparsity > 50%).
    pub fn with_sparse_plan(self, plan: SparseExpertPlan) -> Self {
        let _ = self.sparse_plan.set(plan);
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
    /// Dispatches between `PerExpert` (original O(E) loop) and `Grouped`
    /// (batched gather → grouped GEMM → batched scatter) strategies.
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
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;

        // ── N5: Determine local expert range for EP (expert parallelism) ──
        #[cfg(feature = "distributed")]
        let local_expert_range: (usize, usize) = match self.exchange {
            Some(ref ex) => {
                let guard = ex.lock().unwrap();
                if guard.world_size() > 1 {
                    guard.local_expert_range()
                } else {
                    (0, num_experts)
                }
            }
            _ => (0, num_experts),
        };
        #[cfg(not(feature = "distributed"))]
        let local_expert_range: (usize, usize) = (0, num_experts);

        // Record metrics
        self.metrics.record_forward(seq_len as u64);

        // Gate logits: [seq_len, num_experts]
        let gate_logits = gate.forward(x, registry, queue)?;

        // ── GPU top-k routing: eliminates the GPU→CPU→GPU round-trip ──
        let route_result = ops::topk_route::gpu_topk_route(
            registry,
            &gate_logits,
            top_k,
            self.expert_bias.as_ref(),
            queue,
        )?;

        // ── EP dispatch/combine wiring (X-P0-1, N-P0-4) ──
        // When an exchange is configured with world_size > 1 and the group has
        // real transport, use the dispatch -> compute -> combine pipeline so
        // tokens reach remote experts. In loopback mode (no transport) or
        // single-rank, fall through to the local-only path.
        #[cfg(feature = "distributed")]
        let use_ep_exchange = match self.exchange {
            Some(ref ex) => {
                let guard = ex.lock().unwrap();
                guard.world_size() > 1
            }
            None => false,
        };
        #[cfg(not(feature = "distributed"))]
        let use_ep_exchange = false;

        // ── Strategy dispatch ──
        // Note: forward_pipelined (via forward_grouped) may already include
        // the shared expert output when SBO is enabled. In that case we must
        // NOT add it again here.
        let (routed_output, shared_already_applied) = if use_ep_exchange {
            // EP path: dispatch tokens to remote experts, compute local, combine results.
            #[cfg(feature = "distributed")]
            {
                (
                    self.exchange_and_compute(
                        x,
                        &route_result,
                        local_expert_range,
                        registry,
                        queue,
                    )?,
                    false,
                )
            }
            #[cfg(not(feature = "distributed"))]
            {
                unreachable!("use_ep_exchange is always false without distributed feature")
            }
        } else {
            // Local-only path: all experts are processed on this rank.
            match self.strategy {
                MoeStrategy::PerExpert => (
                    self.forward_per_expert(x, &route_result, local_expert_range, registry, queue)?,
                    false,
                ),
                MoeStrategy::Grouped => {
                    let sbo_will_handle_shared = self
                        .pipeline
                        .as_ref()
                        .is_some_and(|p| p.config().enable_sbo)
                        && self.shared_expert.is_some();
                    (
                        self.forward_grouped(
                            x,
                            &route_result,
                            local_expert_range,
                            registry,
                            queue,
                        )?,
                        sbo_will_handle_shared,
                    )
                }
                MoeStrategy::GatherMM => (
                    self.forward_gather_mm(x, &route_result, local_expert_range, registry, queue)?,
                    false,
                ),
            }
        };

        // ── N7: Shared expert (DeepSeek-V3 pattern) ──
        // Skip if the pipelined SBO path already combined shared + routed.
        let output = if !shared_already_applied {
            if let Some(ref shared) = self.shared_expert {
                let shared_out = shared.forward(x, registry, queue)?;
                ops::binary::add(registry, &routed_output, &shared_out, queue)?
            } else {
                routed_output
            }
        } else {
            routed_output
        };

        Ok(output)
    }

    /// EP exchange-and-compute pipeline: dispatch -> local expert forward -> combine.
    ///
    /// When expert parallelism is active (world_size > 1), tokens routed to
    /// non-local experts must be sent to the owning rank, and results combined
    /// back. This method orchestrates the full pipeline:
    ///
    /// 1. **Dispatch**: serialize token data, call `MoeDispatchExchange::dispatch()`
    ///    to route tokens. Local expert tokens are packed into the dispatch result;
    ///    remote tokens are exchanged via the group transport (RDMA or loopback).
    /// 2. **Compute**: run the local expert forward pass on dispatched tokens
    ///    (which now include tokens received from other ranks).
    /// 3. **Combine**: call `MoeCombineExchange::combine_with_layout()` to send
    ///    expert outputs back to originating ranks and apply routing weights.
    ///
    /// In loopback mode (group has no real transport), the dispatch/combine
    /// exchange is local-only: all tokens stay on the same rank but are
    /// correctly routed through the dispatch layout, ensuring the code path
    /// is exercised even in single-process testing.
    ///
    /// TODO: Wire the full RDMA async path (dispatch_async + combine_async)
    /// for production multi-node inference.
    #[cfg(feature = "distributed")]
    fn exchange_and_compute(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        _local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // The combine exchange operates on Vec<f32>; non-f32 dtypes would
        // require a cast before combine. Guard against this for now.
        if x.dtype() != DType::Float32 {
            return Err(KernelError::InvalidShape(format!(
                "EP exchange_and_compute requires Float32 input, got {:?}. \
                 Non-f32 EP support requires adding a dtype cast before combine.",
                x.dtype()
            )));
        }

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Read routing results to CPU for dispatch
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Serialize token data to bytes for the dispatch exchange.
        // The exchange operates on raw byte buffers regardless of dtype.
        let token_bytes: &[u8] = x.to_bytes();

        // ── Phase 1: Dispatch ──
        // Lock the exchange and call dispatch to route tokens.
        let dispatch_result: DispatchResult = {
            let mut guard = self
                .exchange
                .as_ref()
                .ok_or_else(|| {
                    KernelError::InvalidShape("exchange_and_compute called without exchange".into())
                })?
                .lock()
                .map_err(|e| KernelError::InvalidShape(format!("exchange lock poisoned: {e}")))?;
            guard
                .dispatch(seq_len, &indices_vec, &weights_vec, token_bytes)
                .map_err(|e| KernelError::InvalidShape(format!("EP dispatch failed: {e}")))?
        };

        let layout = &dispatch_result.layout;
        let (local_start, local_end) = layout.local_expert_range;
        let local_expert_count = local_end - local_start;
        let tokens_per_expert = layout.tokens_per_expert;

        // ── Phase 2: Local expert compute on dispatched tokens ──
        // The dispatch result contains routed_data: tokens packed by local expert.
        // We need to unpack per-expert token batches, run each expert's forward,
        // and collect outputs.
        let routed_data = &dispatch_result.routed_data;
        let token_stride = if seq_len > 0 {
            token_bytes.len() / seq_len
        } else {
            hidden_dim * elem_size
        };

        // Build per-expert output buffers from the dispatched tokens.
        // expert_outputs[i] contains the f32 output for local expert i.
        let mut expert_outputs: Vec<Vec<f32>> = Vec::with_capacity(local_expert_count);

        for local_idx in 0..local_expert_count {
            let expert_idx = local_start + local_idx;
            let count = dispatch_result.expert_counts[expert_idx];

            if count == 0 {
                expert_outputs.push(Vec::new());
                continue;
            }

            // Compute the byte range for this expert in routed_data.
            // Tokens are packed contiguously per expert: expert 0's tokens first,
            // then expert 1's, etc. Each expert has up to tokens_per_expert slots.
            let expert_byte_offset = local_idx * tokens_per_expert * token_stride;
            let expert_byte_end = expert_byte_offset + count * token_stride;

            if expert_byte_end > routed_data.len() {
                // Insufficient data — skip this expert (defensive)
                expert_outputs.push(Vec::new());
                continue;
            }

            let expert_token_bytes = &routed_data[expert_byte_offset..expert_byte_end];

            // Reconstruct Array from bytes for expert forward pass
            let expert_input =
                Array::from_bytes(dev, expert_token_bytes, vec![count, hidden_dim], x.dtype());

            // Run expert forward
            let expert_out = self.experts[expert_idx].forward(&expert_input, registry, queue)?;

            // Read expert output back to CPU f32 for combine.
            // The combine exchange operates on Vec<f32> per expert.
            // For non-f32 dtypes, we would need a cast here; for now we
            // assert f32 since the combine path requires it.
            let out_f32: Vec<f32> = expert_out.to_vec_checked::<f32>();
            expert_outputs.push(out_f32);
        }

        // ── Phase 3: Combine ──
        // Use combine_with_layout to scatter expert outputs back to original
        // token positions with routing weights applied.
        let exchange_guard = self
            .exchange
            .as_ref()
            .unwrap()
            .lock()
            .map_err(|e| KernelError::InvalidShape(format!("exchange lock poisoned: {e}")))?;
        let ws = exchange_guard.world_size();
        let local_range = exchange_guard.local_expert_range();
        let local_rank = if ws > 0 {
            local_range.0 as u32 / (num_experts as u32 / ws as u32)
        } else {
            0
        };
        drop(exchange_guard);

        let group =
            rmlx_distributed::Group::new((0..ws as u32).collect(), local_rank, ws as u32)
                .map_err(|e| KernelError::InvalidShape(format!("Group creation failed: {e}")))?;

        let combiner = MoeCombineExchange::new(group);
        let combined_f32 = combiner
            .combine_with_layout(
                &expert_outputs,
                &weights_vec,
                &indices_vec,
                seq_len,
                top_k,
                hidden_dim,
                num_experts,
                &dispatch_result.layout,
            )
            .map_err(|e| KernelError::InvalidShape(format!("EP combine failed: {e}")))?;

        // Convert combined f32 output back to Array
        let output = Array::from_slice(dev, &combined_f32, vec![seq_len, hidden_dim]);

        Ok(output)
    }

    /// Original per-expert loop: one gather + forward + scatter per active expert.
    /// ~2E sync points per forward (backward-compatible path).
    fn forward_per_expert(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // ── Allocator-aware zero-fill ──
        let alloc_zeros = |shape: &[usize], dtype: DType| -> Array {
            if let Some(ref alloc) = self.allocator {
                if let Ok(arr) = Array::zeros_pooled(alloc, shape, dtype) {
                    return arr;
                }
            }
            Array::zeros(dev, shape, dtype)
        };

        // Read routing results to CPU
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Build dispatch table
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

        // Output accumulator: [seq_len, hidden_dim], zero-initialized
        let output = alloc_zeros(&[seq_len, hidden_dim], x.dtype());

        let (local_start, local_end) = local_expert_range;
        for (expert_idx, dispatch) in expert_dispatch.iter().enumerate() {
            if dispatch.is_empty() {
                continue;
            }
            if expert_idx < local_start || expert_idx >= local_end {
                continue;
            }

            let batch_size = dispatch.len();

            // Gather token embeddings for this expert
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

            // Single batched forward pass for this expert
            let expert_out = self.experts[expert_idx].forward(&expert_input, registry, queue)?;

            // Scatter-add weighted results back
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

        Ok(output)
    }

    /// GatherMM forward path: uses `ops::gather_mm` to dispatch all expert
    /// projections in single GPU calls (one per projection stage).
    ///
    /// Instead of iterating over experts, this method:
    /// 1. Flattens all (token, expert) assignments into a single batch
    /// 2. Reshapes tokens as `[N, 1, D]` for gather_mm's `[batch, m_per_batch, k]`
    /// 3. Calls `gather_mm` once per projection (gate, up, down) using the
    ///    stacked weights from `ExpertGroup`
    /// 4. Applies SwiGLU activation between gate/up and down projections
    /// 5. Scatter-adds weighted results back to token positions
    fn forward_gather_mm(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let expert_group = self.ensure_expert_group(registry, queue)?;

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let top_k = self.config.num_experts_per_token;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Read routing results to CPU
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        let (local_start, local_end) = local_expert_range;

        // Build flat list of (token_idx, expert_idx, weight) for local experts only
        let mut assignments: Vec<(usize, u32, f32)> = Vec::new();
        for tok in 0..seq_len {
            for k in 0..top_k {
                let flat_idx = tok * top_k + k;
                let expert_idx = indices_vec[flat_idx];
                let eidx = expert_idx as usize;
                if eidx >= local_start && eidx < local_end {
                    let weight = weights_vec[flat_idx];
                    assignments.push((tok, expert_idx, weight));
                    self.metrics.record_expert_token(eidx);
                }
            }
        }

        // Output accumulator
        let output = {
            let alloc_zeros = |shape: &[usize], dtype: DType| -> Array {
                if let Some(ref alloc) = self.allocator {
                    if let Ok(arr) = Array::zeros_pooled(alloc, shape, dtype) {
                        return arr;
                    }
                }
                Array::zeros(dev, shape, dtype)
            };
            alloc_zeros(&[seq_len, hidden_dim], x.dtype())
        };

        if assignments.is_empty() {
            return Ok(output);
        }

        let n_assign = assignments.len();

        // Build the gathered input: [n_assign, 1, hidden_dim]
        // and the expert index tensor: [n_assign] (UInt32)
        let gathered_input = Array::zeros(dev, &[n_assign, 1, hidden_dim], x.dtype());
        {
            let copy_kernel = match x.dtype() {
                DType::Float32 => "copy_f32",
                DType::Float16 => "copy_f16",
                DType::Bfloat16 => "copy_bf16",
                other => {
                    return Err(KernelError::InvalidShape(format!(
                        "MoE gather_mm: unsupported dtype {:?}",
                        other
                    )));
                }
            };
            let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;
            let cb = queue.new_command_buffer();
            for (i, &(tok, _, _)) in assignments.iter().enumerate() {
                let src_offset = x.offset() + tok * hidden_dim * elem_size;
                let dst_offset = i * hidden_dim * elem_size;
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(x.metal_buffer()), src_offset as u64);
                enc.set_buffer(1, Some(gathered_input.metal_buffer()), dst_offset as u64);
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
        }

        let expert_indices: Vec<u32> = assignments.iter().map(|&(_, eidx, _)| eidx).collect();
        let indices_arr = Array::from_slice(dev, &expert_indices, vec![n_assign]);

        // gather_mm for gate projection:
        // gathered_input: [n_assign, 1, hidden_dim]
        // gate_weights:   [num_experts, hidden_dim, intermediate_dim]
        // -> gate_out:    [n_assign, 1, intermediate_dim]
        let gate_out = ops::gather_mm::gather_mm(
            registry,
            &gathered_input,
            &expert_group.gate_weights,
            &indices_arr,
            queue,
        )?;

        // gather_mm for up projection:
        // -> up_out: [n_assign, 1, intermediate_dim]
        let up_out = ops::gather_mm::gather_mm(
            registry,
            &gathered_input,
            &expert_group.up_weights,
            &indices_arr,
            queue,
        )?;

        // SwiGLU: silu(gate_out) * up_out
        // Reshape to 2D for silu/mul ops, then back to 3D for down projection
        let gate_2d = gate_out.view(
            vec![n_assign, intermediate_dim],
            vec![intermediate_dim, 1],
            gate_out.offset(),
        );
        let up_2d = up_out.view(
            vec![n_assign, intermediate_dim],
            vec![intermediate_dim, 1],
            up_out.offset(),
        );
        let gate_activated = ops::silu::silu(registry, &gate_2d, queue)?;
        let hidden = ops::binary::mul(registry, &gate_activated, &up_2d, queue)?;

        // Reshape hidden to 3D for gather_mm: [n_assign, 1, intermediate_dim]
        let hidden_3d = hidden.view(
            vec![n_assign, 1, intermediate_dim],
            vec![intermediate_dim, intermediate_dim, 1],
            hidden.offset(),
        );

        // gather_mm for down projection:
        // hidden_3d:      [n_assign, 1, intermediate_dim]
        // down_weights:   [num_experts, intermediate_dim, hidden_dim]
        // -> down_out:    [n_assign, 1, hidden_dim]
        let down_out = ops::gather_mm::gather_mm(
            registry,
            &hidden_3d,
            &expert_group.down_weights,
            &indices_arr,
            queue,
        )?;

        // Scatter-add weighted results back to output positions
        {
            let copy_kernel = match x.dtype() {
                DType::Float32 => "copy_f32",
                DType::Float16 => "copy_f16",
                DType::Bfloat16 => "copy_bf16",
                other => {
                    return Err(KernelError::InvalidShape(format!(
                        "MoE gather_mm scatter: unsupported dtype {:?}",
                        other
                    )));
                }
            };
            let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;

            for (i, &(tok, _, weight)) in assignments.iter().enumerate() {
                let expert_tok_offset = down_out.offset() + i * hidden_dim * elem_size;
                let expert_tok_view =
                    down_out.view(vec![1, hidden_dim], vec![hidden_dim, 1], expert_tok_offset);

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

        Ok(output)
    }

    /// Grouped forward path: batched gather → ExpertGroup GEMM → batched scatter.
    /// Reduces sync points from ~2E to 1 by encoding all work into minimal CBs.
    fn forward_grouped(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let expert_group = self.ensure_expert_group(registry, queue)?;

        // Record per-expert metrics from routing result (before potential pipeline delegation)
        {
            let counts: Vec<u32> = route_result.expert_counts.to_vec_checked();
            for (eid, &count) in counts.iter().enumerate() {
                if eid < self.metrics.num_experts && count > 0 {
                    self.metrics.expert_tokens[eid]
                        .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // If pipeline is configured, delegate to pipelined path (SBO/TBO)
        if let Some(ref pipeline) = self.pipeline {
            return self.forward_pipelined(
                x,
                route_result,
                expert_group,
                pipeline,
                local_expert_range,
                registry,
                queue,
            );
        }

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Read routing to CPU
        let indices_vec: Vec<u32> = route_result.expert_indices.to_vec_checked();
        let weights_vec: Vec<f32> = route_result.expert_weights.to_vec_checked();

        // Build dispatch table (metrics already recorded above from expert_counts)
        let mut expert_dispatch: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for tok in 0..seq_len {
            for k in 0..top_k {
                let flat_idx = tok * top_k + k;
                let expert_idx = indices_vec[flat_idx] as usize;
                let weight = weights_vec[flat_idx];
                expert_dispatch[expert_idx].push((tok, weight));
            }
        }

        // ICB sparse dispatch: auto-select when sparsity > 50%
        if let Some(sparse_plan) = self.sparse_plan.get() {
            let counts: Vec<u32> = route_result.expert_counts.to_vec_checked();
            let active = counts.iter().filter(|&&c| c > 0).count();
            if active * 2 < num_experts {
                return self.forward_sparse_icb(
                    x,
                    route_result,
                    sparse_plan,
                    &expert_dispatch,
                    local_expert_range,
                    registry,
                    queue,
                );
            }
        }

        // Batched gather: ALL experts in ONE command buffer
        let expert_inputs = gather_all_experts(
            x,
            &expert_dispatch,
            local_expert_range,
            hidden_dim,
            elem_size,
            dev,
            registry,
            queue,
            self.allocator.as_ref(),
        )?;

        // Grouped forward: ALL expert GEMMs in ONE command buffer
        let input_refs: Vec<(usize, &Array)> =
            expert_inputs.iter().map(|(idx, arr)| (*idx, arr)).collect();
        let expert_outputs = expert_group.grouped_forward(&input_refs, registry, queue)?;

        // Batched scatter-add: ONE command buffer
        let output = {
            let alloc_zeros = |shape: &[usize], dtype: DType| -> Array {
                if let Some(ref alloc) = self.allocator {
                    if let Ok(arr) = Array::zeros_pooled(alloc, shape, dtype) {
                        return arr;
                    }
                }
                Array::zeros(dev, shape, dtype)
            };
            alloc_zeros(&[seq_len, hidden_dim], x.dtype())
        };
        scatter_add_all_experts(
            &expert_outputs,
            &expert_dispatch,
            &output,
            hidden_dim,
            elem_size,
            dev,
            registry,
            queue,
        )?;

        Ok(output)
    }

    /// Lazily build ExpertGroup from self.experts.
    fn ensure_expert_group(
        &self,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<&ExpertGroup, KernelError> {
        if let Some(group) = self.expert_group.get() {
            return Ok(group);
        }
        let group = ExpertGroup::from_experts(&self.experts, registry, queue)?;
        // If another thread raced us, discard our copy and use theirs.
        let _ = self.expert_group.set(group);
        Ok(self.expert_group.get().unwrap())
    }

    /// Pipelined forward path using MoePipeline (SBO/TBO).
    #[allow(clippy::too_many_arguments)]
    fn forward_pipelined(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        expert_group: &ExpertGroup,
        pipeline: &MoePipeline,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let top_k = self.config.num_experts_per_token;

        // SBO: overlap shared expert with routed experts when shared_expert is present
        if pipeline.config().enable_sbo {
            if let Some(ref shared) = self.shared_expert {
                return pipeline.forward_sbo(
                    x,
                    shared,
                    |_x| {
                        pipeline.forward_tbo(
                            x,
                            expert_group,
                            route_result,
                            top_k,
                            local_expert_range,
                            registry,
                            queue,
                        )
                    },
                    registry,
                    queue,
                );
            }
        }

        // TBO only (no shared expert or SBO disabled)
        pipeline.forward_tbo(
            x,
            expert_group,
            route_result,
            top_k,
            local_expert_range,
            registry,
            queue,
        )
    }

    /// ICB sparse dispatch path: skip empty experts via indirect command buffer.
    ///
    /// Uses `grouped_forward_icb()` from rmlx-metal to encode only active
    /// experts into the command buffer. The `IcbReplayCache` tracks sparsity
    /// patterns so that repeated patterns (common in decode phase) can skip
    /// re-analysis of the active expert set.
    ///
    /// Falls back to `ExpertGroup::grouped_forward()` for the actual GEMM
    /// computation since the Metal ICB GEMM encoding path requires further
    /// pipeline state setup (TODO: wire actual ICB GEMM replay in Phase 6b).
    #[allow(clippy::too_many_arguments)]
    fn forward_sparse_icb(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        sparse_plan: &SparseExpertPlan,
        expert_dispatch: &[Vec<(usize, f32)>],
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Read expert counts from routing result
        let counts: Vec<u32> = route_result.expert_counts.to_vec_checked();

        // Check the ICB replay cache for this sparsity pattern.
        // If cached, we can skip re-computing the active mask.
        {
            let mut cache = self.icb_replay_cache.lock().unwrap();
            cache.record(&counts);
        }

        // Gather inputs for active experts only (skips empty experts)
        let expert_inputs = gather_all_experts(
            x,
            expert_dispatch,
            local_expert_range,
            hidden_dim,
            elem_size,
            dev,
            registry,
            queue,
            self.allocator.as_ref(),
        )?;

        // Use grouped forward for the active experts via ExpertGroup.
        // The sparse_plan's replay_sparse is available for direct ICB GEMM
        // dispatch when pipeline states are fully wired (Phase 6b TODO).
        // For now, the key optimization is that we only gather/scatter/compute
        // for active experts -- empty experts are completely skipped.
        let expert_group = self.ensure_expert_group(registry, queue)?;
        let input_refs: Vec<(usize, &Array)> =
            expert_inputs.iter().map(|(idx, arr)| (*idx, arr)).collect();

        // Log sparsity for debugging: how many experts were actually active
        let active_count = counts.iter().filter(|&&c| c > 0).count();
        let _total_experts = counts.len();
        let _ = (sparse_plan, active_count); // sparse_plan used for future ICB GEMM replay

        let expert_outputs = expert_group.grouped_forward(&input_refs, registry, queue)?;

        // Scatter-add results
        let output = {
            let alloc_zeros = |shape: &[usize], dtype: DType| -> Array {
                if let Some(ref alloc) = self.allocator {
                    if let Ok(arr) = Array::zeros_pooled(alloc, shape, dtype) {
                        return arr;
                    }
                }
                Array::zeros(dev, shape, dtype)
            };
            alloc_zeros(&[seq_len, hidden_dim], x.dtype())
        };
        scatter_add_all_experts(
            &expert_outputs,
            expert_dispatch,
            &output,
            hidden_dim,
            elem_size,
            dev,
            registry,
            queue,
        )?;

        Ok(output)
    }

    /// Access the ICB replay cache for inspection or manual management.
    pub fn icb_replay_cache(&self) -> &std::sync::Mutex<IcbReplayCache> {
        &self.icb_replay_cache
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

// ---------------------------------------------------------------------------
// Batched gather/scatter helpers for grouped forward path
// ---------------------------------------------------------------------------

/// Gather ALL active experts' tokens in a single command buffer.
///
/// For each local expert with tokens, copies the assigned token rows from `x`
/// into a contiguous batch buffer. All copy dispatches are encoded into one CB,
/// with a single commit + wait at the end.
///
/// Returns `Vec<(expert_idx, gathered_array)>` for experts with tokens.
#[allow(clippy::too_many_arguments)]
fn gather_all_experts(
    x: &Array,
    expert_dispatch: &[Vec<(usize, f32)>],
    local_expert_range: (usize, usize),
    hidden_dim: usize,
    elem_size: usize,
    dev: &metal::Device,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
    allocator: Option<&Arc<MetalAllocator>>,
) -> Result<Vec<(usize, Array)>, KernelError> {
    let (local_start, local_end) = local_expert_range;

    // Collect active experts
    let active: Vec<(usize, &Vec<(usize, f32)>)> = expert_dispatch
        .iter()
        .enumerate()
        .filter(|(idx, d)| !d.is_empty() && *idx >= local_start && *idx < local_end)
        .collect();

    if active.is_empty() {
        return Ok(Vec::new());
    }

    let copy_kernel = match x.dtype() {
        DType::Float32 => "copy_f32",
        DType::Float16 => "copy_f16",
        DType::Bfloat16 => "copy_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "MoE gather: unsupported dtype {:?}",
                other
            )));
        }
    };
    let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;

    // Pre-allocate batch buffers
    let mut results: Vec<(usize, Array)> = Vec::with_capacity(active.len());
    for &(expert_idx, dispatch) in &active {
        let batch_size = dispatch.len();
        let batch_buf = if let Some(alloc) = allocator {
            Array::zeros_pooled(alloc, &[batch_size, hidden_dim], x.dtype())
                .unwrap_or_else(|_| Array::zeros(dev, &[batch_size, hidden_dim], x.dtype()))
        } else {
            Array::zeros(dev, &[batch_size, hidden_dim], x.dtype())
        };
        results.push((expert_idx, batch_buf));
    }

    // Encode all copies into ONE command buffer
    let cb = queue.new_command_buffer();
    for (ri, &(_, dispatch)) in active.iter().enumerate() {
        let batch_buf = &results[ri].1;
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
    }
    cb.commit();
    cb.wait_until_completed();

    Ok(results)
}

/// Scatter ALL expert outputs back into the output array with one CB per active expert.
///
/// For each expert output, scales by routing weight and adds to the output at
/// the original token position. The scale+add ops use their own internal CBs,
/// then all copy-back operations within one expert are batched into a single CB.
/// This gives O(active_experts) sync points instead of O(total_tokens).
#[allow(clippy::too_many_arguments)]
fn scatter_add_all_experts(
    expert_outputs: &[(usize, Array)],
    expert_dispatch: &[Vec<(usize, f32)>],
    output: &Array,
    hidden_dim: usize,
    elem_size: usize,
    dev: &metal::Device,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<(), KernelError> {
    if expert_outputs.is_empty() {
        return Ok(());
    }

    let copy_kernel = match output.dtype() {
        DType::Float32 => "copy_f32",
        DType::Float16 => "copy_f16",
        DType::Bfloat16 => "copy_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "MoE scatter: unsupported dtype {:?}",
                other
            )));
        }
    };
    let pipeline = registry.get_pipeline(copy_kernel, output.dtype())?;

    // For each expert output, scale by weight and scatter-add to output.
    // Tokens within one expert have distinct output positions, so copies
    // within one expert can be batched safely. Between experts, ordering is
    // maintained by sequential per-expert processing (needed because top-k
    // routing can assign the same token to multiple experts).
    for &(expert_idx, ref expert_out) in expert_outputs {
        let dispatch = &expert_dispatch[expert_idx];

        // Collect scale+add results, deferring copies to a batched CB.
        let mut copy_tasks: Vec<(Array, usize)> = Vec::with_capacity(dispatch.len());
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

            copy_tasks.push((summed, dst_offset));
        }

        // ONE CB for all copy-back operations in this expert
        if !copy_tasks.is_empty() {
            let cb = queue.new_command_buffer();
            for (summed, dst_offset) in &copy_tasks {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(summed.metal_buffer()), summed.offset() as u64);
                enc.set_buffer(1, Some(output.metal_buffer()), *dst_offset as u64);
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
        }
    }

    Ok(())
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
