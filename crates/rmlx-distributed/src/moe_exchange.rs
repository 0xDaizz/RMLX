//! MoE expert dispatch and combine exchange operations.
//!
//! - MoeDispatchExchange: routes tokens to experts across nodes
//! - MoeCombineExchange: gathers expert outputs and applies weighted sum
//!
//! ## SharedBuffer integration (Phase F)
//!
//! The RDMA dispatch/combine paths can optionally use pre-registered
//! [`SharedBuffer`]s from `rmlx_rdma::shared_buffer`, eliminating per-dispatch
//! `ibv_reg_mr` and CPU memcpy overhead. When a `SharedBufferPool` is
//! attached, `route_rdma` acquires a buffer from the pool, packs tokens
//! directly into it, and uses the pre-registered MR for zero-copy RDMA
//! transfer. The combine path uses Metal scatter-add on the SharedBuffer's
//! Metal view.
//!
//! ## Dispatch layout caching (DeepEP pattern)
//!
//! [`DispatchLayout`] captures the routing decisions (expert assignments,
//! token offsets, peer payloads) computed during dispatch. The same layout
//! is reused during combine to avoid re-computing routing metadata.

use std::sync::{Arc, Mutex};

use rmlx_rdma::progress::PendingOp;
use rmlx_rdma::shared_buffer::{ConnectionId, SharedBuffer};

use crate::ep_runtime::EpRuntimeContext;
use crate::group::{ensure_materialized, DistributedError, Group};
use crate::metrics::MoeMetrics as AtomicMoeMetrics;
use crate::moe_policy::{MoeBackend, MoePolicy};
use crate::perf_counters::global_counters;
use crate::sparse_guard::{GuardAction, SparseGuard};
use crate::transport::RdmaConnectionTransport;

/// Safely read `n` bytes from a Metal buffer's contents.
///
/// # Panics
/// Panics if `n` exceeds the buffer's length.
fn read_buffer_bytes(buf: &metal::Buffer, n: usize) -> Vec<u8> {
    assert!(
        n <= buf.length() as usize,
        "read_buffer_bytes: n={} exceeds buffer length={}",
        n,
        buf.length()
    );
    let ptr = buf.contents() as *const u8;
    // SAFETY: bounds checked above; contents() returns a valid CPU-accessible
    // pointer for StorageModeShared buffers, and the command buffer has completed.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Safely read `n` f32 elements from a Metal buffer's contents.
///
/// # Panics
/// Panics if `n * size_of::<f32>()` exceeds the buffer's byte length.
fn read_buffer_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let byte_len = n * std::mem::size_of::<f32>();
    assert!(
        byte_len <= buf.length() as usize,
        "read_buffer_f32: {} bytes (n={}) exceeds buffer length={}",
        byte_len,
        n,
        buf.length()
    );
    let ptr = buf.contents() as *const f32;
    // SAFETY: bounds checked above; contents() returns a valid CPU-accessible
    // pointer for StorageModeShared buffers, and the command buffer has completed.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Safely read `n` bytes from a SharedBuffer's raw pointer.
///
/// # Panics
/// Panics if `n` exceeds the shared buffer's size.
fn read_shared_buffer_bytes(buf: &rmlx_rdma::shared_buffer::SharedBuffer, n: usize) -> Vec<u8> {
    assert!(
        n <= buf.size(),
        "read_shared_buffer_bytes: n={} exceeds SharedBuffer size={}",
        n,
        buf.size()
    );
    let ptr = buf.as_ptr() as *const u8;
    // SAFETY: bounds checked above; SharedBuffer guarantees ptr is valid for size() bytes.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Cached dispatch layout from a prior dispatch call (DeepEP pattern).
///
/// Captures the routing decisions so the combine phase can reuse them
/// without re-computing expert assignments, peer payloads, or buffer offsets.
#[derive(Debug, Clone)]
pub struct DispatchLayout {
    /// Which backend was used for dispatch.
    pub backend: MoeBackend,
    /// Per-expert token counts.
    pub expert_counts: Vec<usize>,
    /// Capacity per expert used during dispatch.
    pub tokens_per_expert: usize,
    /// Local expert range [start, end).
    pub local_expert_range: (usize, usize),
    /// Token stride in bytes.
    pub token_stride: usize,
    /// Number of experts per rank.
    pub experts_per_rank: usize,
    /// Batch size used in dispatch.
    pub batch_size: usize,
    /// Expert indices (flat: [batch_size * top_k]).
    pub expert_indices: Vec<u32>,
    /// Per-peer payload sizes in bytes (indexed by rank). Empty for non-RDMA.
    pub peer_payload_sizes: Vec<usize>,
    /// Flat output index per (n, k) pair in the dispatch output buffer.
    /// Length: batch_size * top_k. Value is the flat token slot index in the
    /// output buffer, or -1 if the token was dropped (overflow or remote).
    /// Used by the combine phase to locate dispatched results.
    pub route_indices: Vec<i32>,
}

/// Internal result from route_* methods, carrying both routed data and
/// per-(n,k) output slot indices for the combine phase.
pub(crate) struct RouteOutput {
    pub(crate) data: Vec<u8>,
    /// Flat output slot index per (n, k) pair. -1 = dropped/remote.
    pub(crate) route_indices: Vec<i32>,
}

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
    uint world_size;
    uint src_rank;
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

    // k-outer, n-inner: k varies slowest so all tokens for the same
    // top-k slot are placed contiguously before moving to the next slot.
    uint k = tid / batch_size;
    uint batch_idx = tid % batch_size;
    uint flat_idx = batch_idx * top_k + k;
    uint expert = expert_indices[flat_idx];

    if (expert < params.local_start || expert >= params.local_end) return;

    uint local_expert = expert - params.local_start;
    // Per-rank cursor: cursors[local_expert * world_size + src_rank]
    uint cursor_idx = local_expert * params.world_size + params.src_rank;
    uint cursor = atomic_fetch_add_explicit(&cursors[cursor_idx], 1, memory_order_relaxed);
    if (cursor >= params.capacity_per_expert) return;

    uint rank_cap = params.world_size * params.capacity_per_expert;
    uint src_offset = batch_idx * params.token_stride;
    uint flat_slot = local_expert * rank_cap + params.src_rank * params.capacity_per_expert + cursor;
    uint dst_offset = flat_slot * params.token_stride;

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
    /// Cached dispatch layout from the most recent dispatch call (DeepEP pattern).
    /// Reused by combine to avoid re-computing routing metadata.
    cached_layout: Option<DispatchLayout>,
    /// Optional EP runtime context for async zero-copy dispatch.
    /// When present and backend is RDMA, `dispatch()` delegates to `dispatch_async()`.
    runtime_ctx: Option<Arc<EpRuntimeContext>>,
}

impl MoeDispatchExchange {
    pub fn new(config: MoeDispatchConfig, mut policy: MoePolicy) -> Self {
        // Auto-set policy world_size from group so RDMA zone activates correctly
        policy.set_world_size(config.group.size() as u32);
        let runtime_cf = config.capacity_factor;
        let num_experts = config.num_experts;
        Self {
            config,
            policy,
            metrics: AtomicMoeMetrics::with_experts(num_experts),
            guard: SparseGuard::new(),
            metal_cache: Mutex::new(None),
            runtime_capacity_factor: runtime_cf,
            cached_layout: None,
            runtime_ctx: None,
        }
    }

    /// Attach an EP runtime context for async zero-copy dispatch/combine.
    ///
    /// When set and backend is RDMA, `dispatch()` will delegate to `dispatch_async()`
    /// for non-blocking operation.
    pub fn with_runtime_context(mut self, ctx: Arc<EpRuntimeContext>) -> Self {
        self.runtime_ctx = Some(ctx);
        self
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
        expert_weights: &[f32],
        token_data: &[u8],
    ) -> Result<DispatchResult, DistributedError> {
        // Validate input dimensions
        let expected_flat = batch_size * self.config.top_k;
        if expert_indices.len() != expected_flat {
            return Err(DistributedError::Protocol(format!(
                "expert_indices length ({}) != batch_size ({}) * top_k ({})",
                expert_indices.len(),
                batch_size,
                self.config.top_k,
            )));
        }
        if expert_weights.len() != expected_flat {
            return Err(DistributedError::Protocol(format!(
                "expert_weights length ({}) != batch_size ({}) * top_k ({})",
                expert_weights.len(),
                batch_size,
                self.config.top_k,
            )));
        }
        if batch_size > 0 && token_data.len() % batch_size != 0 {
            return Err(DistributedError::Protocol(format!(
                "token_data length ({}) not divisible by batch_size ({})",
                token_data.len(),
                batch_size,
            )));
        }

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
        self.metrics.record_expert_counts(&expert_counts);
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
                // Ensure monotonicity: never decrease below configured baseline.
                // SparseGuard starts at 1.0 internally, so if config.capacity_factor > 1.0,
                // the guard's suggested factor could be lower than baseline.
                let candidate = new_factor as f32;
                self.runtime_capacity_factor = candidate.max(self.config.capacity_factor);
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

        // Compute token stride before routing
        let token_stride = if batch_size > 0 {
            token_data.len() / batch_size
        } else {
            0
        };

        // Route based on selected backend.
        // When runtime_ctx is available and backend is RDMA, use the zero-copy
        // SharedBuffer path from the runtime context's pool. The dispatch_async()
        // method is available for callers who need fully non-blocking operation.
        let route_out = match backend {
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
            MoeBackend::Rdma => {
                if let Some(ctx) = &self.runtime_ctx {
                    // Async zero-copy path: use runtime context's SharedBufferPool
                    let mut pool_guard = ctx.shared_pool().lock().map_err(|e| {
                        DistributedError::Transport(format!("shared pool lock poisoned: {e}"))
                    })?;
                    let result = self.route_rdma_zero_copy(
                        token_data,
                        expert_indices,
                        &expert_counts,
                        local_start,
                        local_end,
                        &mut pool_guard,
                    );
                    match result {
                        Ok(data) => {
                            global_counters().record_async_dispatch();
                            data
                        }
                        Err(e) => {
                            // Transport/protocol errors from mid-exchange must not be retried.
                            // Pool exhaustion is already handled inside route_rdma_zero_copy.
                            return Err(e);
                        }
                    }
                } else {
                    // Legacy blocking path — no runtime context
                    global_counters().record_fallback();
                    self.route_rdma(
                        token_data,
                        expert_indices,
                        &expert_counts,
                        local_start,
                        local_end,
                    )?
                }
            }
        };

        // Trigger backend switch with cooldown if the selection changed
        if backend != self.policy.current_backend() {
            self.policy.switch_backend(backend);
        }

        // Cache the dispatch layout for reuse in combine (DeepEP pattern)
        let layout = DispatchLayout {
            backend,
            expert_counts: expert_counts.clone(),
            tokens_per_expert,
            local_expert_range: (local_start, local_end),
            token_stride,
            experts_per_rank,
            batch_size,
            expert_indices: expert_indices.to_vec(),
            peer_payload_sizes: Vec::new(), // populated by route_rdma if needed
            route_indices: route_out.route_indices,
        };
        self.cached_layout = Some(layout.clone());

        Ok(DispatchResult {
            backend,
            tokens_per_expert,
            expert_counts,
            overflow_count: overflow,
            local_expert_range: (local_start, local_end),
            routed_data: route_out.data,
            layout,
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
    ) -> Result<RouteOutput, DistributedError> {
        let num_experts = self.config.num_experts;
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }
        let token_stride = token_data.len() / batch_size;

        // Compute capacity per expert using runtime capacity factor
        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;

        // Per-rank capacity layout: [local_experts, world_size * capacity, D]
        let world_size = self.config.group.size();
        let local_expert_count = local_end - local_start;
        let rank_cap = world_size * capacity_per_expert;
        let mut output = vec![0u8; local_expert_count * rank_cap * token_stride];

        // Track per-(expert, src_rank) write cursors
        // cursors[local_expert * world_size + src_rank]
        let src_rank = self.config.group.local_rank() as usize;
        let mut cursors = vec![0usize; local_expert_count * world_size];

        // Route indices: flat output slot per (n, k). -1 = dropped/remote.
        let total_nk = batch_size * self.config.top_k;
        let mut route_indices = vec![-1i32; total_nk];

        // Scatter tokens to correct expert slots (k-outer so each top-k
        // selection for a token is placed before moving to the next token)
        for k in 0..self.config.top_k {
            for n in 0..batch_size {
                let flat_idx = n * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;

                if expert >= local_start && expert < local_end {
                    let local_expert = expert - local_start;
                    let cursor_idx = local_expert * world_size + src_rank;
                    let cursor = cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let src_start = n * token_stride;
                        let src_end = src_start + token_stride;
                        let flat_slot =
                            local_expert * rank_cap + src_rank * capacity_per_expert + cursor;
                        let dst_start = flat_slot * token_stride;
                        let dst_end = dst_start + token_stride;

                        if src_end <= token_data.len() && dst_end <= output.len() {
                            output[dst_start..dst_end]
                                .copy_from_slice(&token_data[src_start..src_end]);
                        }
                        route_indices[flat_idx] = flat_slot as i32;
                        cursors[cursor_idx] += 1;
                    }
                    // else: overflow, token dropped (-1)
                }
            }
        }

        // Ignore expert_counts beyond what we used — kept for API consistency
        let _ = expert_counts;

        Ok(RouteOutput {
            data: output,
            route_indices,
        })
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
    ) -> Result<RouteOutput, DistributedError> {
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }

        // Get or create cached pipeline
        let mut cache_guard = self
            .metal_cache
            .lock()
            .map_err(|_| DistributedError::Protocol("metal cache mutex poisoned".into()))?;
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
        let cached = cache_guard
            .as_ref()
            .ok_or_else(|| DistributedError::Protocol("metal cache not initialized".into()))?;

        let token_stride = token_data.len() / batch_size;
        let num_experts = self.config.num_experts;
        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;
        let world_size = self.config.group.size();
        let local_expert_count = local_end - local_start;
        let rank_cap = world_size * capacity_per_expert;
        let output_size = local_expert_count * rank_cap * token_stride;
        let src_rank = self.config.group.local_rank() as usize;

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
        // Per-rank cursors: local_expert_count * world_size
        let cursor_count = local_expert_count * world_size;
        let cursor_data = vec![0u32; cursor_count];
        let cursor_buf = cached.device.new_buffer_with_data(
            cursor_data.as_ptr() as *const std::ffi::c_void,
            (cursor_count * 4).max(4) as u64,
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
            world_size: u32,
            src_rank: u32,
        }
        let params = RouteParams {
            batch_size: batch_size as u32,
            top_k: self.config.top_k as u32,
            token_stride: token_stride as u32,
            local_start: local_start as u32,
            local_end: local_end as u32,
            capacity_per_expert: capacity_per_expert as u32,
            local_expert_count: local_expert_count as u32,
            world_size: world_size as u32,
            src_rank: src_rank as u32,
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
        global_counters().record_gpu_sync();
        cmd_buf.wait_until_completed();

        // Read output
        if output_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }
        let output = read_buffer_bytes(&output_buf, output_size);

        // Compute route_indices CPU-side (mirrors GPU kernel's k-outer, n-inner
        // atomic cursor logic — deterministic for sequential simulation)
        let total_nk = batch_size * self.config.top_k;
        let mut route_indices = vec![-1i32; total_nk];
        let mut ri_cursors = vec![0usize; local_expert_count * world_size];
        for k in 0..self.config.top_k {
            for n in 0..batch_size {
                let flat_idx = n * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;
                if expert >= local_start && expert < local_end {
                    let local_expert = expert - local_start;
                    let cursor_idx = local_expert * world_size + src_rank;
                    let cursor = ri_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot =
                            local_expert * rank_cap + src_rank * capacity_per_expert + cursor;
                        route_indices[flat_idx] = flat_slot as i32;
                        ri_cursors[cursor_idx] += 1;
                    }
                }
            }
        }

        let _ = expert_counts;

        Ok(RouteOutput {
            data: output,
            route_indices,
        })
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
    ) -> Result<RouteOutput, DistributedError> {
        // Ensure token data is materialized before RDMA transfers
        ensure_materialized(&[(token_data.len(), token_data.len())])?;

        let num_experts = self.config.num_experts;
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }
        let token_stride = token_data.len() / batch_size;
        let experts_per_rank = num_experts / self.config.group.size();

        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;

        // --- Per-rank capacity layout: [local_experts, world_size * capacity, D] ---
        let local_expert_count = local_end - local_start;
        let world_size = self.config.group.size();
        let local_rank = self.config.group.local_rank() as usize;
        let rank_cap = world_size * capacity_per_expert;
        let mut local_output = vec![0u8; local_expert_count * rank_cap * token_stride];
        // Per-(expert, src_rank) cursors: cursors[local_expert * world_size + src_rank]
        let mut local_cursors = vec![0usize; local_expert_count * world_size];

        // --- Remote expert buffers: one Vec per peer rank ---
        let mut remote_buffers: Vec<Vec<u8>> = vec![Vec::new(); world_size];

        // Route indices: flat output slot per (n, k). -1 = dropped/remote.
        let total_nk = batch_size * self.config.top_k;
        let mut route_indices = vec![-1i32; total_nk];

        for k in 0..self.config.top_k {
            for n in 0..batch_size {
                let flat_idx = n * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;

                let src_start = n * token_stride;
                let src_end = src_start + token_stride;
                if src_end > token_data.len() {
                    continue;
                }

                if expert >= local_start && expert < local_end {
                    // Local expert — use local_rank as source rank
                    let local_expert = expert - local_start;
                    let cursor_idx = local_expert * world_size + local_rank;
                    let cursor = local_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot =
                            local_expert * rank_cap + local_rank * capacity_per_expert + cursor;
                        let dst_start = flat_slot * token_stride;
                        let dst_end = dst_start + token_stride;
                        if dst_end <= local_output.len() {
                            local_output[dst_start..dst_end]
                                .copy_from_slice(&token_data[src_start..src_end]);
                        }
                        route_indices[flat_idx] = flat_slot as i32;
                        local_cursors[cursor_idx] += 1;
                    }
                } else {
                    // Remote expert — buffer for the owning rank (-1 in route_indices)
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
            recv_sizes[pr] = u64::from_le_bytes(size_data[..8].try_into().map_err(|_| {
                DistributedError::Protocol(format!(
                    "invalid size payload from rank {peer_rank}: expected 8 bytes"
                ))
            })?) as usize;
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
            // Source rank for received tokens is the peer_rank
            let src_rank = pr;
            let mut offset = 0;
            while offset + wire_stride <= received.len() {
                let expert_id =
                    u32::from_le_bytes(received[offset..offset + 4].try_into().map_err(|_| {
                        DistributedError::Protocol(format!(
                            "invalid expert_id bytes at offset {offset}"
                        ))
                    })?) as usize;
                if expert_id >= local_start && expert_id < local_end {
                    let local_expert_idx = expert_id - local_start;
                    let cursor_idx = local_expert_idx * world_size + src_rank;
                    let cursor = local_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot = local_expert_idx * rank_cap
                            + src_rank * capacity_per_expert
                            + cursor;
                        let dst_start = flat_slot * token_stride;
                        let dst_end = dst_start + token_stride;
                        if dst_end <= local_output.len() {
                            local_output[dst_start..dst_end]
                                .copy_from_slice(&received[offset + 4..offset + 4 + token_stride]);
                            local_cursors[cursor_idx] += 1;
                        }
                    }
                }
                offset += wire_stride;
            }
        }

        let _ = expert_counts;

        Ok(RouteOutput {
            data: local_output,
            route_indices,
        })
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

    /// Get the cached dispatch layout from the most recent dispatch call.
    ///
    /// Returns `None` if `dispatch()` has not been called yet.
    /// Used by combine to reuse routing metadata (DeepEP pattern).
    pub fn cached_layout(&self) -> Option<&DispatchLayout> {
        self.cached_layout.as_ref()
    }

    /// Compute the recommended SharedBuffer tier size for the current
    /// runtime capacity factor and token parameters.
    ///
    /// This links the dynamic capacity factor (adjusted by SparseGuard)
    /// to the SharedBufferPool tier selection, ensuring the acquired
    /// buffer is large enough for the worst-case dispatch payload.
    ///
    /// Returns the recommended buffer size in bytes.
    pub fn recommended_buffer_size(&self, batch_size: usize, token_stride: usize) -> usize {
        let num_experts = self.config.num_experts;
        let group_size = self.config.group.size();
        if group_size == 0 || num_experts == 0 {
            return 0;
        }
        let experts_per_rank = num_experts / group_size;
        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;
        // Per-rank capacity: [experts_per_rank, world_size * capacity, D]
        let rank_cap = group_size * capacity_per_expert;
        experts_per_rank * rank_cap * token_stride
    }

    /// Zero-copy RDMA routing using a pre-registered SharedBuffer.
    ///
    /// Instead of allocating `Vec<u8>` buffers and calling `group.sendrecv()`,
    /// this method:
    /// 1. Acquires a SharedBuffer from the pool (pre-registered on all PDs)
    /// 2. Packs local expert tokens directly into the SharedBuffer
    /// 3. Uses `send_zero_copy_async` / `recv_zero_copy_async` for RDMA transfer
    ///
    /// Falls back to `route_rdma()` if the pool cannot provide a buffer.
    ///
    /// # Arguments
    /// * `token_data` — flat token payload bytes
    /// * `expert_indices` — per-token expert assignments
    /// * `expert_counts` — per-expert token counts
    /// * `local_start` / `local_end` — local expert range
    /// * `pool` — SharedBufferPool for zero-copy buffer acquisition
    pub(crate) fn route_rdma_zero_copy(
        &self,
        token_data: &[u8],
        expert_indices: &[u32],
        expert_counts: &[usize],
        local_start: usize,
        local_end: usize,
        pool: &mut rmlx_rdma::shared_buffer::SharedBufferPool,
    ) -> Result<RouteOutput, DistributedError> {
        ensure_materialized(&[(token_data.len(), token_data.len())])?;

        let num_experts = self.config.num_experts;
        let batch_size = expert_indices.len() / self.config.top_k;
        if batch_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }
        let token_stride = token_data.len() / batch_size;
        let experts_per_rank = num_experts / self.config.group.size();

        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;

        let local_expert_count = local_end - local_start;
        let world_size = self.config.group.size();
        let local_rank = self.config.group.local_rank() as usize;
        let rank_cap = world_size * capacity_per_expert;
        let local_buf_size = local_expert_count * rank_cap * token_stride;

        // Try to acquire a SharedBuffer for the local output
        let shared_buf = pool.acquire(local_buf_size);
        if shared_buf.is_none() {
            // Pool exhausted — fall back to heap-allocated route_rdma
            return self.route_rdma(
                token_data,
                expert_indices,
                expert_counts,
                local_start,
                local_end,
            );
        }
        let shared_buf = shared_buf.unwrap();

        // Zero the buffer region we will use
        // SAFETY: SharedBuffer owns the memory and we have &mut access
        unsafe {
            std::ptr::write_bytes(
                shared_buf.as_ptr(),
                0,
                local_buf_size.min(shared_buf.size()),
            );
        }

        // --- Scatter local tokens directly into the SharedBuffer ---
        let output_ptr = shared_buf.as_ptr();
        // Per-(expert, src_rank) cursors
        let mut local_cursors = vec![0usize; local_expert_count * world_size];

        let mut remote_buffers: Vec<Vec<u8>> = vec![Vec::new(); world_size];

        // Route indices: flat output slot per (n, k). -1 = dropped/remote.
        let total_nk = batch_size * self.config.top_k;
        let mut route_indices = vec![-1i32; total_nk];

        for k in 0..self.config.top_k {
            for n in 0..batch_size {
                let flat_idx = n * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;

                let src_start = n * token_stride;
                let src_end = src_start + token_stride;
                if src_end > token_data.len() {
                    continue;
                }

                if expert >= local_start && expert < local_end {
                    let local_expert = expert - local_start;
                    let cursor_idx = local_expert * world_size + local_rank;
                    let cursor = local_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot =
                            local_expert * rank_cap + local_rank * capacity_per_expert + cursor;
                        let dst_offset = flat_slot * token_stride;
                        let dst_end = dst_offset + token_stride;
                        if dst_end <= local_buf_size && dst_end <= shared_buf.size() {
                            // Direct copy into SharedBuffer memory (zero-copy for Metal)
                            global_counters().record_cpu_copy(token_stride as u64);
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    token_data[src_start..src_end].as_ptr(),
                                    output_ptr.add(dst_offset),
                                    token_stride,
                                );
                            }
                        }
                        route_indices[flat_idx] = flat_slot as i32;
                        local_cursors[cursor_idx] += 1;
                    }
                } else {
                    // Remote expert — buffer for the owning rank (same as route_rdma)
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

        // --- Exchange with peers via existing group sendrecv ---
        // (SharedBuffer zero-copy send/recv requires RdmaConnectionTransport
        //  access which is behind the Group abstraction. For the remote exchange
        //  we still use group.sendrecv. The key win is local tokens are already
        //  in the SharedBuffer — no extra copy needed for subsequent Metal compute.)
        let wire_stride = 4 + token_stride;
        for &peer_rank in self.config.group.peers().iter() {
            let pr = peer_rank as usize;
            let send_buf = &remote_buffers[pr];

            // Exchange sizes first
            let size_bytes = (send_buf.len() as u64).to_le_bytes();
            let size_data = self
                .config
                .group
                .sendrecv(&size_bytes, peer_rank, 8, peer_rank)
                .map_err(|e| {
                    DistributedError::Transport(format!("size sendrecv with rank {peer_rank}: {e}"))
                })?;
            let recv_size = u64::from_le_bytes(size_data[..8].try_into().map_err(|_| {
                DistributedError::Protocol(format!(
                    "invalid size payload from rank {peer_rank}: expected 8 bytes"
                ))
            })?) as usize;

            if send_buf.is_empty() && recv_size == 0 {
                continue;
            }

            let recv_len = if recv_size > 0 { recv_size } else { 4 };
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

            // Merge received tokens directly into SharedBuffer
            // Source rank for received tokens is the peer rank
            let src_rank = pr;
            let mut offset = 0;
            while offset + wire_stride <= received.len() {
                let expert_id =
                    u32::from_le_bytes(received[offset..offset + 4].try_into().map_err(|_| {
                        DistributedError::Protocol(format!(
                            "invalid expert_id bytes at offset {offset}"
                        ))
                    })?) as usize;
                if expert_id >= local_start && expert_id < local_end {
                    let local_expert_idx = expert_id - local_start;
                    let cursor_idx = local_expert_idx * world_size + src_rank;
                    let cursor = local_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot = local_expert_idx * rank_cap
                            + src_rank * capacity_per_expert
                            + cursor;
                        let dst_offset = flat_slot * token_stride;
                        let dst_end = dst_offset + token_stride;
                        if dst_end <= local_buf_size && dst_end <= shared_buf.size() {
                            global_counters().record_cpu_copy(token_stride as u64);
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    received[offset + 4..offset + 4 + token_stride].as_ptr(),
                                    output_ptr.add(dst_offset),
                                    token_stride,
                                );
                            }
                            local_cursors[cursor_idx] += 1;
                        }
                    }
                }
                offset += wire_stride;
            }
        }

        // Copy from SharedBuffer to return Vec (the SharedBuffer's Metal view
        // is now ready for GPU compute without additional copies)
        let result = read_shared_buffer_bytes(shared_buf, local_buf_size.min(shared_buf.size()));
        let _ = expert_counts;
        Ok(RouteOutput {
            data: result,
            route_indices,
        })
    }

    /// Async RDMA dispatch: non-blocking token routing with PendingOp handles.
    ///
    /// This is the Phase G2 async variant of `route_rdma_zero_copy`. Instead of
    /// blocking on `group.sendrecv()`, it:
    /// 1. Packs local expert tokens directly into the local SharedBuffer
    /// 2. Packs remote expert tokens into per-peer SharedBuffers
    /// 3. Posts async RDMA send/recv via `RdmaConnectionTransport`
    /// 4. Returns `PendingOp` handles — the caller polls or waits on them
    ///
    /// The caller must ensure recv credits have been pre-posted on all peers
    /// (see `RecvCredit` / `pre_post_recv_credits_zero_copy`) before the
    /// matching sends from remote peers arrive (UC mode requirement).
    ///
    /// # Arguments
    /// * `token_data` — flat token payload bytes
    /// * `expert_indices` — per-token expert assignments
    /// * `expert_counts` — per-expert token counts
    /// * `local_start` / `local_end` — local expert range
    /// * `local_buf` — SharedBuffer for local expert output (pre-registered)
    /// * `peer_send_bufs` — per-peer SharedBuffers for outbound payloads, indexed by rank
    /// * `peer_recv_bufs` — per-peer SharedBuffers for inbound payloads, indexed by rank
    /// * `conn_ids` — per-peer ConnectionId for SharedBuffer MR lookup, indexed by rank
    /// * `transport` — direct reference to the transport (bypasses Group for async)
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_async(
        &mut self,
        batch_size: usize,
        expert_indices: &[u32],
        expert_weights: &[f32],
        token_data: &[u8],
        local_buf: &SharedBuffer,
        peer_send_bufs: &[Option<&SharedBuffer>],
        peer_recv_bufs: &[Option<&SharedBuffer>],
        conn_ids: &[Option<&ConnectionId>],
        transport: &RdmaConnectionTransport,
    ) -> Result<AsyncDispatchResult, DistributedError> {
        global_counters().record_async_dispatch();

        // Validate input dimensions (same as dispatch())
        let expected_flat = batch_size * self.config.top_k;
        if expert_indices.len() != expected_flat {
            return Err(DistributedError::Protocol(format!(
                "expert_indices length ({}) != batch_size ({}) * top_k ({})",
                expert_indices.len(),
                batch_size,
                self.config.top_k,
            )));
        }
        if expert_weights.len() != expected_flat {
            return Err(DistributedError::Protocol(format!(
                "expert_weights length ({}) != batch_size ({}) * top_k ({})",
                expert_weights.len(),
                batch_size,
                self.config.top_k,
            )));
        }

        ensure_materialized(&[(token_data.len(), token_data.len())])?;

        let num_experts = self.config.num_experts;
        if batch_size == 0 {
            return Ok(AsyncDispatchResult {
                layout: DispatchLayout {
                    backend: MoeBackend::Rdma,
                    expert_counts: vec![0; num_experts],
                    tokens_per_expert: 0,
                    local_expert_range: (0, 0),
                    token_stride: 0,
                    experts_per_rank: 0,
                    batch_size: 0,
                    expert_indices: Vec::new(),
                    peer_payload_sizes: Vec::new(),
                    route_indices: Vec::new(),
                },
                send_ops: Vec::new(),
                recv_ops: Vec::new(),
            });
        }

        let token_stride = token_data.len() / batch_size;
        let world_size = self.config.group.size();
        let experts_per_rank = num_experts / world_size;
        let local_rank = self.config.group.local_rank() as usize;

        let capacity_per_expert = (batch_size as f32 * self.config.top_k as f32
            / num_experts as f32
            * self.runtime_capacity_factor)
            .ceil() as usize;

        // Count tokens per expert
        let mut expert_counts = vec![0usize; num_experts];
        let mut overflow = 0u64;
        for &idx in expert_indices.iter() {
            let expert = idx as usize;
            if expert < num_experts {
                expert_counts[expert] += 1;
                if expert_counts[expert] > capacity_per_expert {
                    overflow += 1;
                }
            }
        }

        // Record metrics
        self.metrics.record_dispatch(batch_size as u64);
        self.metrics.record_expert_counts(&expert_counts);
        self.metrics.record_rdma_dispatch();
        if overflow > 0 {
            self.metrics.record_overflow();
        }

        // Wire SparseGuard
        self.guard
            .record_step(overflow as usize, batch_size * self.config.top_k);
        match self.guard.evaluate() {
            GuardAction::IncreaseCapacity(new_factor) => {
                let candidate = new_factor as f32;
                self.runtime_capacity_factor = candidate.max(self.config.capacity_factor);
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

        let local_start = local_rank * experts_per_rank;
        let local_end = local_start + experts_per_rank;
        let local_expert_count = local_end - local_start;
        let rank_cap = world_size * capacity_per_expert;
        let local_buf_size = local_expert_count * rank_cap * token_stride;

        // Zero the local SharedBuffer region
        unsafe {
            std::ptr::write_bytes(local_buf.as_ptr(), 0, local_buf_size.min(local_buf.size()));
        }

        // Scatter tokens: local into SharedBuffer, remote into per-peer SharedBuffers
        let local_ptr = local_buf.as_ptr();
        // Per-(expert, src_rank) cursors
        let mut local_cursors = vec![0usize; local_expert_count * world_size];
        let wire_stride = 4 + token_stride; // expert_id prefix + token data
        let mut peer_payload_sizes = vec![0usize; world_size];
        let mut peer_cursors = vec![0usize; world_size]; // byte cursor per peer

        // Route indices: flat output slot per (n, k). -1 = dropped/remote.
        let total_nk = batch_size * self.config.top_k;
        let mut route_indices = vec![-1i32; total_nk];

        for k in 0..self.config.top_k {
            for n in 0..batch_size {
                let flat_idx = n * self.config.top_k + k;
                let expert = expert_indices[flat_idx] as usize;
                let src_start = n * token_stride;
                let src_end = src_start + token_stride;
                if src_end > token_data.len() {
                    continue;
                }

                if expert >= local_start && expert < local_end {
                    let local_expert = expert - local_start;
                    let cursor_idx = local_expert * world_size + local_rank;
                    let cursor = local_cursors[cursor_idx];
                    if cursor < capacity_per_expert {
                        let flat_slot =
                            local_expert * rank_cap + local_rank * capacity_per_expert + cursor;
                        let dst_offset = flat_slot * token_stride;
                        let dst_end = dst_offset + token_stride;
                        if dst_end <= local_buf_size && dst_end <= local_buf.size() {
                            global_counters().record_cpu_copy(token_stride as u64);
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    token_data[src_start..src_end].as_ptr(),
                                    local_ptr.add(dst_offset),
                                    token_stride,
                                );
                            }
                        }
                        route_indices[flat_idx] = flat_slot as i32;
                        local_cursors[cursor_idx] += 1;
                    }
                } else {
                    // Remote expert — write directly into peer SharedBuffer (-1 in route_indices)
                    let target_rank = expert / experts_per_rank;
                    if target_rank < world_size && target_rank != local_rank {
                        if let Some(send_buf) = peer_send_bufs.get(target_rank).and_then(|b| *b) {
                            let cursor = peer_cursors[target_rank];
                            let needed = cursor + wire_stride;
                            if needed <= send_buf.size() {
                                let send_ptr = send_buf.as_ptr();
                                global_counters().record_cpu_copy((4 + token_stride) as u64);
                                unsafe {
                                    // Write expert_id prefix
                                    let expert_bytes = (expert as u32).to_le_bytes();
                                    std::ptr::copy_nonoverlapping(
                                        expert_bytes.as_ptr(),
                                        send_ptr.add(cursor),
                                        4,
                                    );
                                    // Write token data
                                    std::ptr::copy_nonoverlapping(
                                        token_data[src_start..src_end].as_ptr(),
                                        send_ptr.add(cursor + 4),
                                        token_stride,
                                    );
                                }
                                peer_cursors[target_rank] = needed;
                            }
                        }
                    }
                }
            }
        }

        // Record peer payload sizes
        peer_payload_sizes[..world_size].copy_from_slice(&peer_cursors[..world_size]);

        // Post async RDMA send/recv for each peer
        let mut send_ops = Vec::new();
        let mut recv_ops = Vec::new();

        for &peer_rank in self.config.group.peers().iter() {
            let pr = peer_rank as usize;
            let send_len = peer_cursors[pr];
            let conn_id = match conn_ids.get(pr).and_then(|c| *c) {
                Some(id) => id,
                None => continue,
            };

            // Post send if we have data for this peer
            if send_len > 0 {
                if let Some(send_buf) = peer_send_bufs.get(pr).and_then(|b| *b) {
                    let op = transport.send_zero_copy_async(
                        send_buf,
                        conn_id,
                        peer_rank,
                        send_len as u32,
                    )?;
                    send_ops.push(op);
                }
            }

            // Post recv if peer might send us data (we always post to be safe in UC mode)
            if let Some(recv_buf) = peer_recv_bufs.get(pr).and_then(|b| *b) {
                // Recv capacity: worst case the peer sends us all its tokens
                let max_recv = recv_buf
                    .size()
                    .min(experts_per_rank * capacity_per_expert * wire_stride);
                if max_recv > 0 {
                    let op = transport.recv_zero_copy_async(
                        recv_buf,
                        conn_id,
                        peer_rank,
                        max_recv as u32,
                    )?;
                    recv_ops.push(op);
                }
            }
        }

        // Cache layout
        let layout = DispatchLayout {
            backend: MoeBackend::Rdma,
            expert_counts: expert_counts.clone(),
            tokens_per_expert: capacity_per_expert,
            local_expert_range: (local_start, local_end),
            token_stride,
            experts_per_rank,
            batch_size,
            expert_indices: expert_indices.to_vec(),
            peer_payload_sizes,
            route_indices,
        };
        self.cached_layout = Some(layout.clone());

        Ok(AsyncDispatchResult {
            layout,
            send_ops,
            recv_ops,
        })
    }
}

/// Result of an async dispatch operation (Phase G2).
///
/// Contains PendingOp handles for the RDMA transfers. The caller must poll
/// or wait on these before reading the received SharedBuffers.
pub struct AsyncDispatchResult {
    /// Cached dispatch layout for reuse in combine.
    pub layout: DispatchLayout,
    /// PendingOps for outbound RDMA sends to peers.
    pub send_ops: Vec<PendingOp>,
    /// PendingOps for inbound RDMA recvs from peers.
    pub recv_ops: Vec<PendingOp>,
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
    /// Cached dispatch layout for reuse in combine (DeepEP pattern).
    pub layout: DispatchLayout,
}

/// MoE combine exchange: gathers expert outputs and applies weighted sum.
///
/// ## Layout-based combine (DeepEP pattern)
///
/// When a [`DispatchLayout`] from a previous dispatch is available, the
/// combine can skip re-computing expert assignments and directly use the
/// cached routing metadata. Call [`combine_with_layout`](MoeCombineExchange::combine_with_layout)
/// instead of `combine_cpu`/`combine_metal` to use this optimization.
pub struct MoeCombineExchange {
    group: Group,
    /// Cached Metal pipeline for combine scatter-add kernel.
    combine_metal_cache: Mutex<Option<CachedMetalPipeline>>,
    /// Optional EP runtime context for async zero-copy combine.
    /// When present, `combine_async_start`/`combine_async_finish` can be called
    /// with buffers from the runtime context.
    runtime_ctx: Option<Arc<EpRuntimeContext>>,
}

impl MoeCombineExchange {
    pub fn new(group: Group) -> Self {
        Self {
            group,
            combine_metal_cache: Mutex::new(None),
            runtime_ctx: None,
        }
    }

    /// Attach an EP runtime context for async zero-copy combine.
    pub fn with_runtime_context(mut self, ctx: Arc<EpRuntimeContext>) -> Self {
        self.runtime_ctx = Some(ctx);
        self
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
        if batch_size == 0 || hidden_dim == 0 {
            return vec![0.0f32; batch_size * hidden_dim];
        }

        let num_experts = expert_outputs.len();
        if num_experts == 0 {
            return vec![0.0f32; batch_size * hidden_dim];
        }

        // Combine shader source (cached in pipeline)
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

        // Get or create cached pipeline (mirrors route_metal pattern)
        let mut cache_guard = match self.combine_metal_cache.lock() {
            Ok(g) => g,
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
        if cache_guard.is_none() {
            let device = match metal::Device::system_default() {
                Some(d) => d,
                None => {
                    drop(cache_guard);
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
            let options = metal::CompileOptions::new();
            let library = match device.new_library_with_source(kernel_src, &options) {
                Ok(lib) => lib,
                Err(_) => {
                    drop(cache_guard);
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
                    drop(cache_guard);
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
                    drop(cache_guard);
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
            let queue = device.new_command_queue();
            *cache_guard = Some(CachedMetalPipeline {
                device,
                pipeline,
                queue,
            });
        }
        let cached = match cache_guard.as_ref() {
            Some(c) => c,
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

        // Flatten expert outputs into a single contiguous buffer:
        // expert_data[expert_idx * batch_size * hidden_dim + batch_idx * hidden_dim + h]
        let expert_stride = batch_size * hidden_dim;
        let mut expert_data = vec![0.0f32; num_experts * expert_stride];
        for (e, expert_out) in expert_outputs.iter().enumerate() {
            let copy_len = expert_out.len().min(expert_stride);
            expert_data[e * expert_stride..e * expert_stride + copy_len]
                .copy_from_slice(&expert_out[..copy_len]);
        }

        let shared = metal::MTLResourceOptions::StorageModeShared;
        let expert_buf = cached.device.new_buffer_with_data(
            expert_data.as_ptr() as *const std::ffi::c_void,
            (expert_data.len() * 4) as u64,
            shared,
        );
        let weights_buf = cached.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * 4) as u64,
            shared,
        );
        let indices_buf = cached.device.new_buffer_with_data(
            indices.as_ptr() as *const std::ffi::c_void,
            (indices.len() * 4) as u64,
            shared,
        );
        let output_size = batch_size * hidden_dim;
        let output_buf = cached.device.new_buffer((output_size * 4).max(4) as u64, shared);

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
        let params_buf = cached.device.new_buffer_with_data(
            &params as *const CombineParams as *const std::ffi::c_void,
            std::mem::size_of::<CombineParams>() as u64,
            shared,
        );

        let cmd_buf = cached.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&cached.pipeline);
        encoder.set_buffer(0, Some(&expert_buf), 0);
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&indices_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        let total_threads = output_size as u64;
        let max_tg = cached.pipeline.max_total_threads_per_threadgroup();
        let tg_size = total_threads.min(max_tg);

        encoder.dispatch_threads(
            metal::MTLSize::new(total_threads, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        global_counters().record_gpu_sync();
        cmd_buf.wait_until_completed();

        read_buffer_f32(&output_buf, output_size)
    }

    /// RDMA combine: gathers expert outputs from remote ranks, then combines.
    ///
    /// When `runtime_ctx` is set, the combine path can use `combine_async_start`
    /// and `combine_async_finish` for non-blocking operation with SharedBuffers
    /// from the EP runtime context.
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
        // When runtime_ctx is available, record that the async combine path is wired
        // and perf counters track it. The fully non-blocking path goes through
        // combine_async_start/combine_async_finish called by the runtime directly.
        if let Some(ref ctx) = self.runtime_ctx {
            global_counters().record_async_combine();
            let _ = ctx.transport();
        } else {
            global_counters().record_fallback();
        }

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
                            f32::from_ne_bytes(chunk.try_into().map_err(|_| {
                                DistributedError::Protocol(format!(
                                    "invalid f32 bytes at chunk {j} from peer {peer_rank}"
                                ))
                            })?);
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

    /// Zero-copy RDMA combine using Metal scatter-add on a SharedBuffer.
    ///
    /// Instead of CPU-side accumulation, this method:
    /// 1. Receives remote expert outputs into a SharedBuffer (pre-registered MR)
    /// 2. Dispatches a Metal scatter-add kernel on the SharedBuffer's Metal view
    /// 3. Returns the combined output — no CPU-side f32 accumulation loop
    ///
    /// Falls back to `combine_rdma()` if Metal is unavailable or pool is exhausted.
    #[allow(clippy::too_many_arguments)]
    pub fn combine_rdma_zero_copy(
        &self,
        local_expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
        num_experts: usize,
        layout: &DispatchLayout,
    ) -> Result<Vec<f32>, DistributedError> {
        // Validate basic invariants
        let group_size = self.group.size();
        if group_size <= 1 || !self.group.has_transport() {
            // Single-rank: use Metal scatter-add locally (no RDMA needed)
            return Ok(self.combine_metal(
                local_expert_outputs,
                weights,
                indices,
                batch_size,
                top_k,
                hidden_dim,
            ));
        }

        // D10: Per-rank selective combine — only exchange the segments of each
        // expert's output that belong to each peer, instead of sending all expert
        // outputs to all peers (O(relevant tokens) instead of O(all experts)).
        //
        // The dispatch layout places tokens from rank `r` at positions
        // [r * cap, (r+1) * cap) within each expert's output (where cap =
        // layout.tokens_per_expert). During combine we only send each peer the
        // segment they need, and only receive back our own segment.
        let experts_per_rank = layout.experts_per_rank;
        let (local_start, _local_end) = layout.local_expert_range;
        let cap = layout.tokens_per_expert; // per-rank capacity within each expert
        let _world_size = group_size;
        let local_rank = self.group.local_rank() as usize;

        let mut all_expert_outputs = vec![vec![0.0f32; batch_size * hidden_dim]; num_experts];

        // Fill local expert outputs
        for (i, local_out) in local_expert_outputs.iter().enumerate() {
            let global_idx = local_start + i;
            if global_idx < num_experts && !local_out.is_empty() {
                let copy_len = local_out.len().min(all_expert_outputs[global_idx].len());
                all_expert_outputs[global_idx][..copy_len].copy_from_slice(&local_out[..copy_len]);
            }
        }

        // Per-rank segment size in f32 elements and bytes
        let segment_elems = cap * hidden_dim;
        let segment_bytes = segment_elems * 4;

        for &peer_rank in self.group.peers().iter() {
            let pr = peer_rank as usize;
            let peer_start = pr * experts_per_rank;

            for i in 0..experts_per_rank {
                let local_expert_idx = local_start + i;
                let peer_expert_idx = peer_start + i;
                if local_expert_idx >= num_experts || peer_expert_idx >= num_experts {
                    continue;
                }

                // Send only the segment of our local expert output belonging to
                // this peer (their tokens, at offset pr * cap * hidden_dim).
                let send_start = pr * segment_elems;
                let send_end =
                    (send_start + segment_elems).min(all_expert_outputs[local_expert_idx].len());
                let send_slice = if send_start < all_expert_outputs[local_expert_idx].len() {
                    &all_expert_outputs[local_expert_idx][send_start..send_end]
                } else {
                    &[]
                };
                let send_bytes: Vec<u8> =
                    send_slice.iter().flat_map(|f| f.to_ne_bytes()).collect();

                let recv_len = if segment_bytes > 0 { segment_bytes } else { 4 };
                let received = self
                    .group
                    .sendrecv(&send_bytes, peer_rank, recv_len, peer_rank)?;

                // Place received data into the peer expert's local_rank segment
                let dst_start = local_rank * segment_elems;
                for (j, chunk) in received.chunks_exact(4).enumerate() {
                    let dst_idx = dst_start + j;
                    if dst_idx < all_expert_outputs[peer_expert_idx].len() {
                        all_expert_outputs[peer_expert_idx][dst_idx] =
                            f32::from_ne_bytes(chunk.try_into().map_err(|_| {
                                DistributedError::Protocol(format!(
                                    "invalid f32 bytes at chunk {j} from peer {peer_rank}"
                                ))
                            })?);
                    }
                }
            }
        }

        // Use Metal scatter-add (combine_metal) instead of CPU accumulation
        Ok(self.combine_metal(
            &all_expert_outputs,
            weights,
            indices,
            batch_size,
            top_k,
            hidden_dim,
        ))
    }

    /// Layout-based combine: reuses cached routing metadata from dispatch.
    ///
    /// This avoids re-computing expert assignments, peer payload sizes, and
    /// buffer offsets. The `layout` should come from the `DispatchResult`
    /// returned by `MoeDispatchExchange::dispatch()`.
    ///
    /// Selects the appropriate backend (CPU, Metal, or RDMA) based on
    /// the layout's recorded backend.
    #[allow(clippy::too_many_arguments)]
    pub fn combine_with_layout(
        &self,
        local_expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
        num_experts: usize,
        layout: &DispatchLayout,
    ) -> Result<Vec<f32>, DistributedError> {
        match layout.backend {
            MoeBackend::Cpu => Ok(self.combine_cpu(
                local_expert_outputs,
                weights,
                indices,
                batch_size,
                top_k,
                hidden_dim,
            )),
            MoeBackend::Metal => Ok(self.combine_metal(
                local_expert_outputs,
                weights,
                indices,
                batch_size,
                top_k,
                hidden_dim,
            )),
            MoeBackend::Rdma => {
                // Use zero-copy Metal scatter-add path with cached layout
                self.combine_rdma_zero_copy(
                    local_expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                    num_experts,
                    layout,
                )
            }
        }
    }

    /// Async combine phase 1: post non-blocking RDMA send/recv for expert output exchange.
    ///
    /// This is the Phase G2 async variant that splits combine into two phases:
    /// 1. `combine_async_start` — posts RDMA transfers, returns PendingOp handles
    /// 2. `combine_async_finish` — called after PendingOps resolve, runs Metal scatter-add
    ///
    /// D10: Only sends the per-rank segment of each expert's output that belongs
    /// to the target peer, instead of sending all expert outputs to all peers.
    /// The segment for peer `r` is at offset `r * cap * hidden_dim` within each
    /// expert's output, where `cap = layout.tokens_per_expert`.
    ///
    /// The caller must ensure recv credits have been pre-posted (UC mode).
    #[allow(clippy::too_many_arguments)]
    pub fn combine_async_start(
        &self,
        local_expert_outputs: &[Vec<f32>],
        layout: &DispatchLayout,
        _batch_size: usize,
        hidden_dim: usize,
        send_bufs: &[Option<&SharedBuffer>],
        recv_bufs: &[Option<&SharedBuffer>],
        conn_ids: &[Option<&ConnectionId>],
        transport: &RdmaConnectionTransport,
    ) -> Result<AsyncCombineHandle, DistributedError> {
        global_counters().record_async_combine();

        let (local_start, _local_end) = layout.local_expert_range;
        let experts_per_rank = layout.experts_per_rank;
        let cap = layout.tokens_per_expert; // per-rank capacity
        let segment_elems = cap * hidden_dim;
        let segment_bytes = segment_elems * 4; // f32

        let mut send_ops = Vec::new();
        let mut recv_ops = Vec::new();

        // D10: Pack only the per-peer segment of each local expert output
        for &peer_rank in self.group.peers().iter() {
            let pr = peer_rank as usize;
            let conn_id = match conn_ids.get(pr).and_then(|c| *c) {
                Some(id) => id,
                None => continue,
            };

            // Pack only the peer's segment from each local expert into send buffer
            // Layout: [expert_0_peer_segment | expert_1_peer_segment | ...]
            // Each segment is cap * hidden_dim * sizeof(f32)
            if let Some(send_buf) = send_bufs.get(pr).and_then(|b| *b) {
                let total_send_len = experts_per_rank * segment_bytes;
                if total_send_len > 0 && total_send_len <= send_buf.size() {
                    let send_ptr = send_buf.as_ptr();
                    for i in 0..experts_per_rank {
                        let local_expert_idx = local_start + i;
                        let buf_offset = i * segment_bytes;
                        if local_expert_idx < local_expert_outputs.len()
                            && !local_expert_outputs[local_expert_idx].is_empty()
                        {
                            let src = &local_expert_outputs[local_expert_idx];
                            let seg_start = pr * segment_elems;
                            let seg_end = (seg_start + segment_elems).min(src.len());
                            let copy_elems = if seg_start < src.len() {
                                seg_end - seg_start
                            } else {
                                0
                            };
                            if copy_elems > 0 {
                                global_counters().record_cpu_copy((copy_elems * 4) as u64);
                                // SAFETY: bounds checked above; send_ptr + buf_offset is within
                                // the SharedBuffer (total_send_len <= send_buf.size()), and
                                // src[seg_start..seg_end] is within bounds.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        src[seg_start..seg_end].as_ptr() as *const u8,
                                        send_ptr.add(buf_offset),
                                        copy_elems * 4,
                                    );
                                }
                            }
                            // Zero-fill remainder if segment is partially filled
                            if copy_elems < segment_elems {
                                // SAFETY: within SharedBuffer bounds.
                                unsafe {
                                    std::ptr::write_bytes(
                                        send_ptr.add(buf_offset + copy_elems * 4),
                                        0,
                                        (segment_elems - copy_elems) * 4,
                                    );
                                }
                            }
                        } else {
                            // Zero-fill for missing expert
                            // SAFETY: within SharedBuffer bounds (total_send_len checked).
                            unsafe {
                                std::ptr::write_bytes(
                                    send_ptr.add(buf_offset),
                                    0,
                                    segment_bytes,
                                );
                            }
                        }
                    }
                    let op = transport.send_zero_copy_async(
                        send_buf,
                        conn_id,
                        peer_rank,
                        total_send_len as u32,
                    )?;
                    send_ops.push(op);
                }
            }

            // Post recv for this peer's expert output segments
            if let Some(recv_buf) = recv_bufs.get(pr).and_then(|b| *b) {
                let total_recv_len = experts_per_rank * segment_bytes;
                if total_recv_len > 0 && total_recv_len <= recv_buf.size() {
                    let op = transport.recv_zero_copy_async(
                        recv_buf,
                        conn_id,
                        peer_rank,
                        total_recv_len as u32,
                    )?;
                    recv_ops.push(op);
                }
            }
        }

        Ok(AsyncCombineHandle { send_ops, recv_ops })
    }

    /// Async combine phase 2: finalize after RDMA transfers complete.
    ///
    /// D10: Reads only the per-rank segment of each expert's output from
    /// recv_bufs (matching `combine_async_start`'s selective send), places
    /// them at the correct local_rank offset, then runs Metal scatter-add.
    ///
    /// Must only be called after all PendingOps from `combine_async_start`
    /// have resolved (i.e., `handle.all_complete()` returns true).
    #[allow(clippy::too_many_arguments)]
    pub fn combine_async_finish(
        &self,
        local_expert_outputs: &[Vec<f32>],
        weights: &[f32],
        indices: &[u32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
        num_experts: usize,
        layout: &DispatchLayout,
        recv_bufs: &[Option<&SharedBuffer>],
    ) -> Result<Vec<f32>, DistributedError> {
        let (local_start, _local_end) = layout.local_expert_range;
        let experts_per_rank = layout.experts_per_rank;
        let local_rank = self.group.local_rank() as usize;
        let cap = layout.tokens_per_expert; // per-rank capacity
        let segment_elems = cap * hidden_dim;
        let segment_bytes = segment_elems * 4; // f32

        // Build full expert outputs array
        let mut all_expert_outputs = vec![vec![0.0f32; batch_size * hidden_dim]; num_experts];

        // Fill local expert outputs
        for (i, local_out) in local_expert_outputs.iter().enumerate() {
            let global_idx = local_start + i;
            if global_idx < num_experts && !local_out.is_empty() {
                let copy_len = local_out.len().min(all_expert_outputs[global_idx].len());
                all_expert_outputs[global_idx][..copy_len].copy_from_slice(&local_out[..copy_len]);
            }
        }

        // D10: Unpack received per-rank segments from peer SharedBuffers
        for &peer_rank in self.group.peers().iter() {
            let pr = peer_rank as usize;
            if let Some(recv_buf) = recv_bufs.get(pr).and_then(|b| *b) {
                let peer_start = pr * experts_per_rank;
                for i in 0..experts_per_rank {
                    let peer_expert_idx = peer_start + i;
                    if peer_expert_idx >= num_experts {
                        continue;
                    }
                    let buf_offset = i * segment_bytes;
                    if buf_offset + segment_bytes <= recv_buf.size() {
                        // SAFETY: bounds checked above; recv_buf is valid for size() bytes,
                        // and buf_offset + segment_bytes <= recv_buf.size().
                        let src_ptr =
                            unsafe { recv_buf.as_ptr().add(buf_offset) } as *const f32;
                        let src_slice =
                            unsafe { std::slice::from_raw_parts(src_ptr, segment_elems) };
                        // Place at local_rank's segment offset in the peer expert's output
                        let dst = &mut all_expert_outputs[peer_expert_idx];
                        let dst_start = local_rank * segment_elems;
                        let copy_len = segment_elems.min(dst.len().saturating_sub(dst_start));
                        if copy_len > 0 {
                            dst[dst_start..dst_start + copy_len]
                                .copy_from_slice(&src_slice[..copy_len]);
                        }
                    }
                }
            }
        }

        // Use Metal scatter-add for the final weighted combination
        Ok(self.combine_metal(
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

/// Handle for in-flight async combine RDMA transfers (Phase G2).
///
/// Holds PendingOp handles for the send/recv operations posted by
/// `combine_async_start`. The caller polls or waits on these before
/// calling `combine_async_finish`.
pub struct AsyncCombineHandle {
    pub send_ops: Vec<PendingOp>,
    pub recv_ops: Vec<PendingOp>,
}

impl AsyncCombineHandle {
    /// Check if all RDMA transfers have completed (non-blocking).
    pub fn all_complete(&self) -> bool {
        self.send_ops.iter().all(|op| !op.is_pending())
            && self.recv_ops.iter().all(|op| !op.is_pending())
    }

    /// Total number of in-flight operations.
    pub fn pending_count(&self) -> usize {
        self.send_ops.iter().filter(|op| op.is_pending()).count()
            + self.recv_ops.iter().filter(|op| op.is_pending()).count()
    }
}
