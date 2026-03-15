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

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLCommandQueue, MTLComputePipelineState as _, MTLDevice,
};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_core::MetalAllocator;
use rmlx_metal::icb_sparse::{IcbReplayCache, SparseExpertPlan};
use rmlx_metal::{ComputePass, MTLSize};
// TODO(Phase 6b): use grouped_forward_icb for direct ICB GEMM replay
// use rmlx_metal::icb_sparse::grouped_forward_icb;

#[cfg(feature = "distributed")]
use rmlx_distributed::{AsyncCombineHandle, AsyncDispatchResult};
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

#[derive(Clone)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    /// Expert capacity factor for dispatch (1.0 = exact, >1.0 = overprovisioned).
    /// Links to SharedBufferPool tier selection in distributed mode.
    /// Default: 1.0
    pub capacity_factor: f32,
    /// Enable FP8 (E4M3) quantization for RDMA token exchange (Phase 6b).
    ///
    /// When true, tokens are quantized to FP8 before dispatch and dequantized
    /// after combine, halving RDMA bandwidth usage (f16→fp8 = 2x reduction).
    /// Requires `distributed` feature and `MoeDispatchConfig::enable_fp8` to
    /// also be set on the underlying exchange.
    ///
    /// Default: false
    pub enable_fp8: bool,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
    ///
    /// For RDMA-backed EP, the [`Group`] inside `MoeDispatchConfig` must be
    /// created with a transport (`Group::with_transport`). The dispatch path
    /// uses `group.sendrecv()` for token exchange, and the combine path clones
    /// the same transport-backed group into `MoeCombineExchange`. If the group
    /// has no transport, the RDMA backend falls back to local Metal routing.
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
    pub fn init_expert_bias(&mut self, device: &ProtocolObject<dyn MTLDevice>) {
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
        self.metrics.record_forward(seq_len.try_into().unwrap());

        // Gate logits: [seq_len, num_experts]
        let gate_logits = gate.forward(x, registry, queue)?;

        // ── GPU top-k routing: eliminates the GPU→CPU→GPU round-trip ──
        // gpu_topk_route accepts f16 natively (f32 accumulation inside)
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
    /// Phase 3a/3b: When `runtime_ctx` is attached to the dispatch exchange,
    /// uses the async dispatch/combine path for compute-RDMA overlap.
    /// Falls back to blocking dispatch/combine when `runtime_ctx` is `None`.
    #[cfg(feature = "distributed")]
    fn exchange_and_compute(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        _local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let input_dtype = x.dtype();
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

        // TODO(Phase 6b — FP8 exchange): When `self.config.enable_fp8` is true,
        // quantize tokens to FP8 before dispatch and dequantize after combine:
        //
        //   1. Pre-dispatch:  fp8_payload = fp8_exchange::quantize_for_dispatch(registry, x, queue)
        //   2. Dispatch:      guard.dispatch_fp8(seq_len, &indices_vec, &weights_vec, x, registry, queue)
        //                     — returns DispatchResult with FP8-encoded routed_data
        //   3. Post-dispatch: fp8_exchange::unpack_from_wire(&routed_data, count, hidden_dim)
        //                     → (fp8_bytes, scale_bytes)
        //   4. Dequantize:    fp8_exchange::dequantize_received(registry, &fp8_arr, &scales, hidden_dim, queue)
        //                     → Float16 expert inputs for compute
        //   5. Combine:       Same as non-FP8 path (expert outputs are f32)
        //
        // Blocked on: need to split the dispatch result's routed_data per-expert and
        // dequantize each expert's tokens separately before feeding to expert.forward().
        // The wire format uses interleaved [fp8_token | scale] per token, requiring
        // unpack_from_wire before dequantize_received.
        if self.config.enable_fp8 {
            return Err(KernelError::InvalidShape(
                "FP8 exchange (Phase 6b) is not yet wired in exchange_and_compute. \
                 Set enable_fp8 = false or implement the quantize→dispatch→dequantize pipeline."
                    .into(),
            ));
        }

        // Check if async path is available (runtime_ctx attached)
        let use_async = {
            let guard = self
                .exchange
                .as_ref()
                .ok_or_else(|| {
                    KernelError::InvalidShape("exchange_and_compute called without exchange".into())
                })?
                .lock()
                .map_err(|e| KernelError::InvalidShape(format!("exchange lock poisoned: {e}")))?;
            guard.has_runtime_ctx()
        };

        if use_async {
            self.exchange_and_compute_async(
                x,
                &indices_vec,
                &weights_vec,
                token_bytes,
                seq_len,
                hidden_dim,
                top_k,
                num_experts,
                elem_size,
                input_dtype,
                dev,
                registry,
                queue,
            )
        } else {
            self.exchange_and_compute_blocking(
                x,
                &indices_vec,
                &weights_vec,
                token_bytes,
                seq_len,
                hidden_dim,
                top_k,
                num_experts,
                elem_size,
                input_dtype,
                dev,
                registry,
                queue,
            )
        }
    }

    /// Blocking dispatch/combine path (original implementation).
    ///
    /// Used when `runtime_ctx` is not attached to the dispatch exchange.
    #[cfg(feature = "distributed")]
    #[allow(clippy::too_many_arguments)]
    fn exchange_and_compute_blocking(
        &self,
        x: &Array,
        indices_vec: &[u32],
        weights_vec: &[f32],
        token_bytes: &[u8],
        seq_len: usize,
        hidden_dim: usize,
        top_k: usize,
        num_experts: usize,
        elem_size: usize,
        input_dtype: DType,
        dev: &ProtocolObject<dyn MTLDevice>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        // ── Phase 1: Dispatch (blocking) ──
        let dispatch_result: DispatchResult = {
            let mut guard =
                self.exchange.as_ref().unwrap().lock().map_err(|e| {
                    KernelError::InvalidShape(format!("exchange lock poisoned: {e}"))
                })?;
            guard
                .dispatch(seq_len, indices_vec, weights_vec, token_bytes)
                .map_err(|e| KernelError::InvalidShape(format!("EP dispatch failed: {e}")))?
        };

        let layout = &dispatch_result.layout;
        let (local_start, local_end) = layout.local_expert_range;
        let local_expert_count = local_end - local_start;
        let tokens_per_expert = layout.tokens_per_expert;

        // ── Phase 2: Local expert compute ──
        let routed_data = &dispatch_result.routed_data;
        let token_stride = if seq_len > 0 {
            token_bytes.len() / seq_len
        } else {
            hidden_dim * elem_size
        };

        let expert_outputs = self.compute_local_experts(
            x,
            routed_data,
            &dispatch_result.expert_counts,
            local_start,
            local_expert_count,
            tokens_per_expert,
            token_stride,
            hidden_dim,
            dev,
            registry,
            queue,
        )?;

        // ── Phase 3: Combine (blocking) ──
        let exchange_guard = self
            .exchange
            .as_ref()
            .unwrap()
            .lock()
            .map_err(|e| KernelError::InvalidShape(format!("exchange lock poisoned: {e}")))?;
        let group = exchange_guard.group().clone();
        drop(exchange_guard);

        let combiner = MoeCombineExchange::new(group);
        let combined_f32 = combiner
            .combine_with_layout(
                &expert_outputs,
                weights_vec,
                indices_vec,
                seq_len,
                top_k,
                hidden_dim,
                num_experts,
                &dispatch_result.layout,
            )
            .map_err(|e| KernelError::InvalidShape(format!("EP combine failed: {e}")))?;

        // Convert combined f32 output back to Array
        let output = Array::from_slice(dev, &combined_f32, vec![seq_len, hidden_dim]);
        let output = if input_dtype != DType::Float32 {
            ops::copy::copy_cast(registry, &output, input_dtype, queue)?
        } else {
            output
        };

        Ok(output)
    }

    /// Async-aware dispatch/combine path (Phase 3a/3b).
    ///
    /// Current implementation:
    /// - Phase 1 (dispatch): Uses blocking `dispatch()` which already delegates to
    ///   zero-copy RDMA internally when `runtime_ctx` is attached.
    /// - Phase 2 (compute): Runs local expert forward passes on dispatched tokens.
    /// - Phase 3 (combine): Uses `combine_with_layout()` with `runtime_ctx` attached
    ///   to the combiner, enabling the zero-copy combine path.
    ///
    /// Phase 3a/3b: `dispatch_async()` + `combine_async_start/finish` are now
    /// fully wired via `AcquiredBuffer::shared_buffer()` accessor.
    #[cfg(feature = "distributed")]
    #[allow(clippy::too_many_arguments)]
    fn exchange_and_compute_async(
        &self,
        x: &Array,
        indices_vec: &[u32],
        weights_vec: &[f32],
        token_bytes: &[u8],
        seq_len: usize,
        hidden_dim: usize,
        top_k: usize,
        num_experts: usize,
        elem_size: usize,
        input_dtype: DType,
        dev: &ProtocolObject<dyn MTLDevice>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        use std::sync::Arc;

        let token_stride = if seq_len > 0 {
            token_bytes.len() / seq_len
        } else {
            hidden_dim * elem_size
        };

        // ── Extract runtime context and group from exchange guard ──
        let (runtime_ctx, group, progress_tracker) = {
            let guard =
                self.exchange.as_ref().unwrap().lock().map_err(|e| {
                    KernelError::InvalidShape(format!("exchange lock poisoned: {e}"))
                })?;
            let ctx = guard.runtime_ctx().cloned().ok_or_else(|| {
                KernelError::InvalidShape("exchange_and_compute_async requires runtime_ctx".into())
            })?;
            let grp = guard.group().clone();
            let pt = guard.progress_tracker().cloned();
            (ctx, grp, pt)
        };

        let world_size = group.size();
        let local_rank = group.local_rank() as usize;

        // ── Phase 1: Acquire buffers and call dispatch_async ──
        // Estimate buffer size: worst case all tokens for local experts
        let experts_per_rank = num_experts / world_size;
        let local_expert_count = experts_per_rank;
        let capacity_per_expert = (seq_len as f32 * top_k as f32 / num_experts as f32 * 2.0) // 2x capacity factor for safety
            .ceil() as usize;
        let rank_cap = world_size * capacity_per_expert;
        let local_buf_size = local_expert_count * rank_cap * token_stride;
        // Peer buffer size: each peer may receive all our remote-expert tokens
        let wire_stride = 4 + token_stride; // expert_id prefix + token data
        let peer_buf_size = experts_per_rank * capacity_per_expert * wire_stride;

        // Acquire local buffer for routing local expert tokens
        let local_buf = runtime_ctx
            .acquire_buffer(local_buf_size)
            .map_err(|e| KernelError::InvalidShape(format!("acquire local_buf failed: {e}")))?;

        // Acquire per-peer send/recv buffers
        let (send_bufs, recv_bufs) = runtime_ctx
            .acquire_send_recv_buffers(peer_buf_size, world_size, local_rank)
            .map_err(|e| {
                KernelError::InvalidShape(format!("acquire send/recv buffers failed: {e}"))
            })?;

        // Build conn_ids indexed by rank (None for local_rank)
        let conn_id_slice = runtime_ctx.conn_ids();
        let conn_ids_opt: Vec<Option<&rmlx_distributed::ConnectionId>> = (0..world_size)
            .map(|r| {
                if r == local_rank {
                    None
                } else if r < conn_id_slice.len() {
                    Some(&conn_id_slice[r])
                } else {
                    None
                }
            })
            .collect();

        // Build Arc<SharedBuffer> slices for dispatch_async
        let peer_send_arcs: Vec<Option<Arc<rmlx_distributed::SharedBuffer>>> = send_bufs
            .iter()
            .map(|opt| opt.as_ref().map(|ab| Arc::clone(ab.shared_buffer())))
            .collect();
        let peer_recv_arcs: Vec<Option<Arc<rmlx_distributed::SharedBuffer>>> = recv_bufs
            .iter()
            .map(|opt| opt.as_ref().map(|ab| Arc::clone(ab.shared_buffer())))
            .collect();

        // Call dispatch_async: scatters local tokens into local_buf + posts RDMA
        let async_dispatch: AsyncDispatchResult = {
            let mut guard =
                self.exchange.as_ref().unwrap().lock().map_err(|e| {
                    KernelError::InvalidShape(format!("exchange lock poisoned: {e}"))
                })?;
            guard
                .dispatch_async(
                    seq_len,
                    indices_vec,
                    weights_vec,
                    token_bytes,
                    local_buf.shared_buffer(),
                    &peer_send_arcs,
                    &peer_recv_arcs,
                    &conn_ids_opt,
                    runtime_ctx.transport(),
                )
                .map_err(|e| KernelError::InvalidShape(format!("dispatch_async failed: {e}")))?
        };

        let layout = &async_dispatch.layout;
        let (local_start, local_end) = layout.local_expert_range;
        let local_expert_count_actual = local_end - local_start;
        let tokens_per_expert = layout.tokens_per_expert;

        // ── Phase 2: Compute local experts while RDMA is in flight ──
        // Read routed data from local_buf (local-rank tokens already scattered)
        let routed_data = unsafe {
            std::slice::from_raw_parts(local_buf.ptr, local_buf_size.min(local_buf.size))
        };

        let expert_outputs = self.compute_local_experts(
            x,
            routed_data,
            &layout.expert_counts,
            local_start,
            local_expert_count_actual,
            tokens_per_expert,
            token_stride,
            hidden_dim,
            dev,
            registry,
            queue,
        )?;

        // ── Phase 2b: Wait for dispatch RDMA ops to complete ──
        if let Some(ref tracker) = progress_tracker {
            async_dispatch
                .wait_tracked(tracker, std::time::Duration::from_secs(5))
                .map_err(|e| {
                    KernelError::InvalidShape(format!("dispatch RDMA wait failed: {e}"))
                })?;
        }

        // ── Phase 3: Async combine ──
        // Acquire fresh send/recv buffers for the combine phase
        let combine_buf_size = experts_per_rank * tokens_per_expert * hidden_dim * 4; // f32
        let (combine_send_bufs, combine_recv_bufs) = runtime_ctx
            .acquire_send_recv_buffers(combine_buf_size, world_size, local_rank)
            .map_err(|e| {
                KernelError::InvalidShape(format!("acquire combine send/recv buffers failed: {e}"))
            })?;

        let combine_send_arcs: Vec<Option<Arc<rmlx_distributed::SharedBuffer>>> = combine_send_bufs
            .iter()
            .map(|opt| opt.as_ref().map(|ab| Arc::clone(ab.shared_buffer())))
            .collect();
        let combine_recv_arcs: Vec<Option<Arc<rmlx_distributed::SharedBuffer>>> = combine_recv_bufs
            .iter()
            .map(|opt| opt.as_ref().map(|ab| Arc::clone(ab.shared_buffer())))
            .collect();

        let combiner =
            MoeCombineExchange::new(group.clone()).with_runtime_context(Arc::clone(&runtime_ctx));
        let combiner = if let Some(ref pt) = progress_tracker {
            combiner.with_progress_tracker(Arc::clone(pt))
        } else {
            combiner
        };

        // Start async combine: posts RDMA sends/recvs for expert outputs
        let combine_handle: AsyncCombineHandle = combiner
            .combine_async_start(
                &expert_outputs,
                layout,
                seq_len,
                hidden_dim,
                &combine_send_arcs,
                &combine_recv_arcs,
                &conn_ids_opt,
                runtime_ctx.transport(),
            )
            .map_err(|e| KernelError::InvalidShape(format!("combine_async_start failed: {e}")))?;

        // Wait for combine RDMA ops
        if let Some(ref tracker) = progress_tracker {
            combine_handle
                .wait_tracked(tracker, std::time::Duration::from_secs(5))
                .map_err(|e| KernelError::InvalidShape(format!("combine RDMA wait failed: {e}")))?;
        }

        // Finalize combine: unpack received data + weighted scatter-add
        let combined_f32 = combiner
            .combine_async_finish(
                &expert_outputs,
                weights_vec,
                indices_vec,
                seq_len,
                top_k,
                hidden_dim,
                num_experts,
                layout,
                &combine_recv_arcs,
            )
            .map_err(|e| KernelError::InvalidShape(format!("combine_async_finish failed: {e}")))?;

        // Convert combined f32 output back to Array
        let output = Array::from_slice(dev, &combined_f32, vec![seq_len, hidden_dim]);
        let output = if input_dtype != DType::Float32 {
            ops::copy::copy_cast(registry, &output, input_dtype, queue)?
        } else {
            output
        };

        Ok(output)
    }

    /// Run local expert forward passes on dispatched token data.
    ///
    /// Shared between blocking and async paths. Takes pre-routed byte data
    /// packed contiguously per expert.
    #[cfg(feature = "distributed")]
    #[allow(clippy::too_many_arguments)]
    fn compute_local_experts(
        &self,
        x: &Array,
        routed_data: &[u8],
        expert_counts: &[usize],
        local_start: usize,
        local_expert_count: usize,
        tokens_per_expert: usize,
        token_stride: usize,
        hidden_dim: usize,
        dev: &ProtocolObject<dyn MTLDevice>,
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<Vec<f32>>, KernelError> {
        let mut expert_outputs: Vec<Vec<f32>> = Vec::with_capacity(local_expert_count);

        for local_idx in 0..local_expert_count {
            let expert_idx = local_start + local_idx;
            let count = expert_counts[expert_idx];

            if count == 0 {
                expert_outputs.push(Vec::new());
                continue;
            }

            let expert_byte_offset = local_idx * tokens_per_expert * token_stride;
            let expert_byte_end = expert_byte_offset + count * token_stride;

            if expert_byte_end > routed_data.len() {
                expert_outputs.push(Vec::new());
                continue;
            }

            let expert_token_bytes = &routed_data[expert_byte_offset..expert_byte_end];
            let expert_input =
                Array::from_bytes(dev, expert_token_bytes, vec![count, hidden_dim], x.dtype());

            let expert_out = self.experts[expert_idx].forward(&expert_input, registry, queue)?;

            let expert_out_f32 = if expert_out.dtype() != DType::Float32 {
                ops::copy::copy_cast(registry, &expert_out, DType::Float32, queue)?
            } else {
                expert_out
            };
            let out_f32: Vec<f32> = expert_out_f32.to_vec_checked::<f32>();
            expert_outputs.push(out_f32);
        }

        Ok(expert_outputs)
    }

    /// Original per-expert loop: one gather + forward + scatter per active expert.
    /// ~2E sync points per forward (backward-compatible path).
    fn forward_per_expert(
        &self,
        x: &Array,
        route_result: &ops::topk_route::TopkRouteResult,
        local_expert_range: (usize, usize),
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
                let cb = queue.commandBuffer().unwrap();
                for (i, &(tok, _)) in dispatch.iter().enumerate() {
                    let src_offset = x.offset() + tok * hidden_dim * elem_size;
                    let dst_offset = i * hidden_dim * elem_size;
                    let raw_enc = cb.computeCommandEncoder().unwrap();
                    let enc = ComputePass::new(&raw_enc);
                    enc.set_pipeline(&pipeline);
                    enc.set_buffer(0, Some(x.metal_buffer()), src_offset);
                    enc.set_buffer(1, Some(batch_buf.metal_buffer()), dst_offset);
                    let grid = MTLSize {
                        width: hidden_dim,
                        height: 1,
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), hidden_dim),
                        height: 1,
                        depth: 1,
                    };
                    enc.dispatch_threads(grid, tg);
                    enc.end();
                }
                cb.commit();
                cb.waitUntilCompleted();
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

                let scale_arr = make_scale_array(dev, weight, hidden_dim, x.dtype());
                let scaled = ops::binary::mul(registry, &expert_tok_view, &scale_arr, queue)?;

                let dst_offset = output.offset() + tok * hidden_dim * elem_size;
                let dst_view = output.view(vec![1, hidden_dim], vec![hidden_dim, 1], dst_offset);

                let summed = ops::binary::add(registry, &dst_view, &scaled, queue)?;

                let cb = queue.commandBuffer().unwrap();
                let raw_enc = cb.computeCommandEncoder().unwrap();
                let enc = ComputePass::new(&raw_enc);
                enc.set_pipeline(&pipeline);
                enc.set_buffer(0, Some(summed.metal_buffer()), summed.offset());
                enc.set_buffer(1, Some(output.metal_buffer()), dst_offset);
                let grid = MTLSize {
                    width: hidden_dim,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), hidden_dim),
                    height: 1,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
                enc.end();
                cb.commit();
                cb.waitUntilCompleted();
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let expert_group = self.ensure_expert_group(registry, queue)?;

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let top_k = self.config.num_experts_per_token;
        let dev = registry.device().raw();

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

        // Build token index array for fused gather
        let tok_indices: Vec<u32> = assignments.iter().map(|&(tok, _, _)| tok as u32).collect();
        let tok_indices_arr = Array::from_slice(dev, &tok_indices, vec![n_assign]);

        // Fused index_gather: single dispatch gathers all assigned token rows
        let x_2d = if x.shape().len() == 2 {
            x.view(x.shape().to_vec(), x.strides().to_vec(), x.offset())
        } else {
            x.view(vec![seq_len, hidden_dim], vec![hidden_dim, 1], x.offset())
        };
        let gathered_2d = ops::moe_kernels::index_gather(registry, &x_2d, &tok_indices_arr, queue)?;
        // Reshape to [n_assign, 1, hidden_dim] for gather_mm
        let gathered_input = gathered_2d.view(
            vec![n_assign, 1, hidden_dim],
            vec![hidden_dim, hidden_dim, 1],
            gathered_2d.offset(),
        );

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

        // Fused scatter-weighted-add: single dispatch replaces N×3 sync points
        {
            // Reshape down_out from [n_assign, 1, hidden_dim] to [n_assign, hidden_dim]
            let down_2d = down_out.view(
                vec![n_assign, hidden_dim],
                vec![hidden_dim, 1],
                down_out.offset(),
            );

            // Build weights array (f32) for the scatter kernel
            let weight_vals: Vec<f32> = assignments.iter().map(|&(_, _, w)| w).collect();
            let weights_arr = Array::from_slice(dev, &weight_vals, vec![n_assign]);

            ops::moe_kernels::scatter_weighted_add(
                registry,
                &down_2d,
                &output,
                &tok_indices_arr,
                &weights_arr,
                queue,
            )?;
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        let expert_group = self.ensure_expert_group(registry, queue)?;

        // Record per-expert metrics from routing result (before potential pipeline delegation)
        {
            let counts: Vec<u32> = route_result.expert_counts.to_vec_checked();
            for (eid, &count) in counts.iter().enumerate() {
                if eid < self.metrics.num_experts && count > 0 {
                    self.metrics.expert_tokens[eid].fetch_add(
                        (count as usize).try_into().unwrap(),
                        std::sync::atomic::Ordering::Relaxed,
                    );
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
    dev: &ProtocolObject<dyn MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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
    let cb = queue.commandBuffer().unwrap();
    for (ri, &(_, dispatch)) in active.iter().enumerate() {
        let batch_buf = &results[ri].1;
        for (i, &(tok, _)) in dispatch.iter().enumerate() {
            let src_offset = x.offset() + tok * hidden_dim * elem_size;
            let dst_offset = i * hidden_dim * elem_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(x.metal_buffer()), src_offset);
            enc.set_buffer(1, Some(batch_buf.metal_buffer()), dst_offset);
            let grid = MTLSize {
                width: hidden_dim,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), hidden_dim),
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
            enc.end();
        }
    }
    cb.commit();
    cb.waitUntilCompleted();

    Ok(results)
}

/// Convert f32 to f16 bit representation (IEEE 754 half precision).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign as u16
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | frac) as u16
    }
}

/// Create a scale array filled with `weight` matching the given dtype.
fn make_scale_array(
    dev: &ProtocolObject<dyn MTLDevice>,
    weight: f32,
    hidden_dim: usize,
    dtype: DType,
) -> Array {
    match dtype {
        DType::Float32 => {
            let scale_data = vec![weight; hidden_dim];
            Array::from_slice(dev, &scale_data, vec![1, hidden_dim])
        }
        DType::Float16 | DType::Bfloat16 => {
            let w_bits = f32_to_f16_bits(weight);
            let f16_bytes: Vec<u8> = (0..hidden_dim).flat_map(|_| w_bits.to_le_bytes()).collect();
            Array::from_bytes(dev, &f16_bytes, vec![1, hidden_dim], dtype)
        }
        _ => {
            // Fallback to f32
            let scale_data = vec![weight; hidden_dim];
            Array::from_slice(dev, &scale_data, vec![1, hidden_dim])
        }
    }
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
    dev: &ProtocolObject<dyn MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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

            let scale_arr = make_scale_array(dev, weight, hidden_dim, output.dtype());
            let scaled = ops::binary::mul(registry, &expert_tok_view, &scale_arr, queue)?;

            let dst_offset = output.offset() + tok * hidden_dim * elem_size;
            let dst_view = output.view(vec![1, hidden_dim], vec![hidden_dim, 1], dst_offset);

            let summed = ops::binary::add(registry, &dst_view, &scaled, queue)?;

            copy_tasks.push((summed, dst_offset));
        }

        // ONE CB for all copy-back operations in this expert
        if !copy_tasks.is_empty() {
            let cb = queue.commandBuffer().unwrap();
            for (summed, dst_offset) in &copy_tasks {
                let raw_enc = cb.computeCommandEncoder().unwrap();
                let enc = ComputePass::new(&raw_enc);
                enc.set_pipeline(&pipeline);
                enc.set_buffer(0, Some(summed.metal_buffer()), summed.offset());
                enc.set_buffer(1, Some(output.metal_buffer()), *dst_offset);
                let grid = MTLSize {
                    width: hidden_dim,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), hidden_dim),
                    height: 1,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
                enc.end();
            }
            cb.commit();
            cb.waitUntilCompleted();
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
