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

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use rmlx_rdma::shared_buffer::{ConnectionId, SharedBuffer};

use crate::ep_runtime::{AcquiredBuffer, EpRuntimeContext};
use crate::progress_tracker::ProgressTracker;
use crate::transport::ZeroCopyPendingOp;
// FP8 exchange functions are available for callers that set config.enable_fp8.
// They operate on Array types and require KernelRegistry, so the actual
// quantize/dequantize calls happen at the MoeLayer level, not in route_rdma.
use crate::fp8_exchange;
use crate::group::{ensure_materialized, DistributedError, Group};
use crate::metrics::MoeMetrics as AtomicMoeMetrics;
use crate::moe_policy::{MoeBackend, MoePolicy};
use crate::perf_counters::global_counters;
use crate::slab_ring::SlabRing;
use crate::sparse_guard::{GuardAction, SparseGuard};
use crate::transport::RdmaConnectionTransport;
use crate::v3_protocol;

/// Pre-acquired send/recv buffer pairs for EP dispatch/combine exchange.
///
/// Holds one send and one recv `AcquiredBuffer` per peer rank (indexed by rank,
/// with `None` for self-rank). When dropped, all held buffers are automatically
/// released back to the `SharedBufferPool` via their `CompletionTicket`s.
pub struct ExchangeBuffers {
    /// Send buffers indexed by rank. `None` for self-rank.
    pub send_bufs: Vec<Option<AcquiredBuffer>>,
    /// Recv buffers indexed by rank. `None` for self-rank.
    pub recv_bufs: Vec<Option<AcquiredBuffer>>,
    /// World size (number of ranks).
    pub world_size: usize,
    /// Local rank (self-rank, excluded from buffers).
    pub local_rank: usize,
}

impl ExchangeBuffers {
    /// Explicitly release all buffers back to the pool.
    ///
    /// This is equivalent to dropping the struct, but allows the caller to
    /// control the release point precisely. After calling this, all buffer
    /// slots are `None`.
    pub fn release(&mut self) {
        for buf in self.send_bufs.iter_mut() {
            buf.take();
        }
        for buf in self.recv_bufs.iter_mut() {
            buf.take();
        }
    }

    /// Number of active peer buffer pairs (excludes self-rank).
    pub fn peer_count(&self) -> usize {
        self.send_bufs.iter().filter(|b| b.is_some()).count()
    }
}

/// Safely read `n` bytes from a Metal buffer's contents.
///
/// Returns an error if `n` exceeds the buffer's length or the buffer pointer
/// is null.
fn read_buffer_bytes(
    buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) -> Result<Vec<u8>, DistributedError> {
    let buf_len = buf.length();
    if n > buf_len {
        return Err(DistributedError::Protocol(format!(
            "read_buffer_bytes: n={} exceeds buffer length={}",
            n, buf_len
        )));
    }
    let ptr = buf.contents().as_ptr() as *const u8;
    // SAFETY: bounds checked above; contents() returns a valid CPU-accessible
    // pointer for StorageModeShared buffers, and the command buffer has completed.
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
}

/// Safely read `n` f32 elements from a Metal buffer's contents.
///
/// Returns an error if `n * size_of::<f32>()` exceeds the buffer's byte length
/// or the buffer pointer is null.
fn read_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) -> Result<Vec<f32>, DistributedError> {
    let byte_len = n.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
        DistributedError::Protocol(format!(
            "read_buffer_f32: overflow computing byte length for n={}",
            n
        ))
    })?;
    let buf_len = buf.length();
    if byte_len > buf_len {
        return Err(DistributedError::Protocol(format!(
            "read_buffer_f32: {} bytes (n={}) exceeds buffer length={}",
            byte_len, n, buf_len
        )));
    }
    let ptr = buf.contents().as_ptr() as *const f32;
    // SAFETY: bounds checked above; contents() returns a valid CPU-accessible
    // pointer for StorageModeShared buffers, and the command buffer has completed.
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
}

/// Safely read `n` bytes from a SharedBuffer's raw pointer.
///
/// Returns an error if `n` exceeds the shared buffer's size or the pointer
/// is null.
fn read_shared_buffer_bytes(
    buf: &rmlx_rdma::shared_buffer::SharedBuffer,
    n: usize,
) -> Result<Vec<u8>, DistributedError> {
    let buf_size = buf.size();
    if n > buf_size {
        return Err(DistributedError::Protocol(format!(
            "read_shared_buffer_bytes: n={} exceeds SharedBuffer size={}",
            n, buf_size
        )));
    }
    let ptr = buf.as_ptr() as *const u8;
    if ptr.is_null() {
        return Err(DistributedError::Protocol(
            "read_shared_buffer_bytes: SharedBuffer pointer is null".to_string(),
        ));
    }
    // SAFETY: bounds checked above; SharedBuffer guarantees ptr is valid for size() bytes.
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
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

// D14/D15/D16 Metal compute kernel sources were removed — they were never
// referenced from Rust code. If needed in the future, they can be restored
// from git history (commit that added this comment).

/// Dtype selector for Metal multi-dtype kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeDtype {
    /// 32-bit float (4 bytes per element).
    F32 = 0,
    /// IEEE 754 half-precision (2 bytes per element).
    F16 = 1,
    /// Brain floating point (2 bytes per element).
    Bf16 = 2,
}

impl MoeDtype {
    /// Bytes per element for this dtype.
    pub fn element_size(self) -> usize {
        match self {
            MoeDtype::F32 => 4,
            MoeDtype::F16 | MoeDtype::Bf16 => 2,
        }
    }
}

// MoeMetrics is now the atomic version from crate::metrics (re-exported as AtomicMoeMetrics).
// This unifies the previously duplicated non-atomic and atomic implementations.

/// Wire protocol version for RDMA dispatch/combine.
///
/// - `V2`: Fixed-size packets with 4-byte expert_id prefix (default, backward-compatible).
/// - `V3`: Variable-length packets with packed metadata for bandwidth efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WireProtocol {
    /// Fixed-size v2 wire format (4-byte expert_id prefix + token data).
    #[default]
    V2,
    /// Variable-length v3 wire format (packed metadata, eliminates wasted bandwidth).
    V3,
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
    /// Wire protocol version for RDMA dispatch/combine. Default: V2.
    pub wire_protocol: WireProtocol,
    /// Enable FP8 quantization for RDMA token exchange. Default: false.
    pub enable_fp8: bool,
}

impl MoeDispatchConfig {
    /// Create a dispatch config with default protocol settings.
    ///
    /// Uses V2 wire protocol and FP8 disabled. Use struct update syntax
    /// to override: `MoeDispatchConfig { wire_protocol: WireProtocol::V3, ..MoeDispatchConfig::new(...) }`
    pub fn new(num_experts: usize, top_k: usize, capacity_factor: f32, group: Group) -> Self {
        Self {
            num_experts,
            top_k,
            capacity_factor,
            group,
            wire_protocol: WireProtocol::V2,
            enable_fp8: false,
        }
    }
}

/// Cached Metal pipeline for route_metal to avoid per-dispatch JIT compilation.
struct CachedMetalPipeline {
    device: rmlx_metal::MtlDevice,
    pipeline: rmlx_metal::MtlPipeline,
    queue: rmlx_metal::MtlQueue,
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
    /// Optional slab ring for zero-copy GPU→RDMA transfer (Phase 6b).
    /// When present, GPU writes directly into slab's Metal buffer for RDMA send.
    slab_ring: Option<Arc<SlabRing>>,
    /// Optional progress tracker for async dispatch health monitoring.
    /// When present, RDMA dispatch operations are tracked through the
    /// ProgressTracker which monitors consecutive errors and sets health flags.
    progress_tracker: Option<Arc<ProgressTracker>>,
}

impl MoeDispatchExchange {
    pub fn new(config: MoeDispatchConfig, policy: MoePolicy) -> Self {
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
            slab_ring: None,
            progress_tracker: None,
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

    /// Attach a slab ring for zero-copy GPU→RDMA transfer (Phase 6b).
    ///
    /// When set, RDMA dispatch writes GPU data directly into a slab's Metal
    /// buffer, enabling zero-copy RDMA send without intermediate copies.
    pub fn with_slab_ring(mut self, ring: Arc<SlabRing>) -> Self {
        self.slab_ring = Some(ring);
        self
    }

    /// Attach a progress tracker for async dispatch health monitoring.
    ///
    /// When set, async RDMA dispatch operations are polled through the
    /// tracker, which monitors consecutive errors and provides health status.
    pub fn with_progress_tracker(mut self, tracker: Arc<ProgressTracker>) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }

    /// Access the progress tracker, if attached.
    pub fn progress_tracker(&self) -> Option<&Arc<ProgressTracker>> {
        self.progress_tracker.as_ref()
    }

    /// Current runtime capacity factor (may differ from baseline after guard actions).
    pub fn runtime_capacity_factor(&self) -> f32 {
        self.runtime_capacity_factor
    }

    /// Acquire pre-registered send/recv buffers from the runtime's SharedBufferPool.
    ///
    /// Returns an `ExchangeBuffers` struct holding one send and one recv buffer
    /// per peer rank. Buffers are automatically returned to the pool when the
    /// `ExchangeBuffers` is dropped (or explicitly via `release()`).
    ///
    /// Requires a runtime context to be attached via `with_runtime_context()`.
    /// Returns an error if no runtime context is set or buffers cannot be acquired.
    pub fn acquire_exchange_buffers(
        &self,
        payload_size: usize,
    ) -> Result<ExchangeBuffers, DistributedError> {
        let ctx = self.runtime_ctx.as_ref().ok_or_else(|| {
            DistributedError::Transport(
                "acquire_exchange_buffers: no runtime context attached".to_string(),
            )
        })?;

        let world_size = self.config.group.size();
        let local_rank = self.config.group.local_rank() as usize;

        let (send_bufs, recv_bufs) =
            ctx.acquire_send_recv_buffers(payload_size, world_size, local_rank)?;

        Ok(ExchangeBuffers {
            send_bufs,
            recv_bufs,
            world_size,
            local_rank,
        })
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
                    let mut pool_guard = ctx.shared_pool().lock();
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

    /// FP8-aware dispatch: quantizes tokens to FP8 before wire transfer.
    ///
    /// This is a convenience wrapper that:
    /// 1. Calls `fp8_exchange::quantize_for_dispatch()` to quantize tokens
    /// 2. Packs FP8 data + per-token scales into interleaved wire format
    /// 3. Calls `dispatch()` with the FP8 byte payload
    ///
    /// The returned `DispatchResult.routed_data` contains FP8-encoded data.
    /// Callers must use `fp8_exchange::unpack_from_wire()` + `dequantize_received()`
    /// to recover Float16 tokens.
    ///
    /// Requires `config.enable_fp8 == true`.
    pub fn dispatch_fp8(
        &mut self,
        batch_size: usize,
        expert_indices: &[u32],
        expert_weights: &[f32],
        tokens: &rmlx_core::array::Array,
        registry: &rmlx_core::kernels::KernelRegistry,
        queue: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    ) -> Result<DispatchResult, DistributedError> {
        if !self.config.enable_fp8 {
            return Err(DistributedError::Protocol(
                "dispatch_fp8 requires config.enable_fp8 = true".into(),
            ));
        }

        let payload = fp8_exchange::quantize_for_dispatch(registry, tokens, queue)
            .map_err(|e| DistributedError::Protocol(format!("FP8 quantize failed: {e}")))?;

        // Validate batch_size matches quantized token count
        if payload.num_tokens != batch_size {
            return Err(DistributedError::Protocol(format!(
                "dispatch_fp8: batch_size ({}) != quantized token count ({})",
                batch_size, payload.num_tokens
            )));
        }

        let wire_data = payload.pack_for_wire();
        self.dispatch(batch_size, expert_indices, expert_weights, &wire_data)
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
            let device = match objc2_metal::MTLCreateSystemDefaultDevice() {
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
            let options = objc2_metal::MTLCompileOptions::new();
            let library = match device.newLibraryWithSource_options_error(
                &objc2_foundation::NSString::from_str(METAL_ROUTE_KERNEL),
                Some(&options),
            ) {
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
            let function = match library
                .newFunctionWithName(&objc2_foundation::NSString::from_str("moe_gather_scatter"))
            {
                Some(f) => f,
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
            let pipeline = match device.newComputePipelineStateWithFunction_error(&function) {
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
            let queue = device.newCommandQueue().unwrap();
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
        let shared = rmlx_metal::MTLResourceOptions::StorageModeShared;
        // SAFETY: token_data is a valid &[u8] slice; Metal copies bytes into
        // the buffer, so the pointer does not need to outlive this call.
        let token_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(token_data.as_ptr() as *mut std::ffi::c_void).unwrap(),
                token_data.len(),
                shared,
            )
        }
        .unwrap();
        // SAFETY: expert_indices is a valid &[u32] slice; same copy semantics.
        let indices_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(expert_indices.as_ptr() as *mut std::ffi::c_void).unwrap(),
                expert_indices.len() * 4,
                shared,
            )
        }
        .unwrap();
        let output_buf = cached
            .device
            .newBufferWithLength_options(output_size.max(1), shared)
            .unwrap();
        // Per-rank cursors: local_expert_count * world_size
        let cursor_count = local_expert_count * world_size;
        let cursor_buf_byte_len = cursor_count.checked_mul(4).ok_or_else(|| {
            DistributedError::Protocol(format!("cursor buffer size overflow: {cursor_count} * 4"))
        })?;
        let cursor_data = vec![0u32; cursor_count];
        // SAFETY: cursor_data is a valid Vec<u32>; Metal copies bytes.
        let cursor_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(cursor_data.as_ptr() as *mut std::ffi::c_void).unwrap(),
                cursor_buf_byte_len.max(4),
                shared,
            )
        }
        .unwrap();

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
        // SAFETY: params is a valid #[repr(C)] struct on the stack; Metal copies
        // the bytes before returning.
        let params_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(std::ptr::addr_of!(params) as *mut std::ffi::c_void)
                    .unwrap(),
                std::mem::size_of::<RouteParams>(),
                shared,
            )
        }
        .unwrap();

        // Encode and dispatch using cached queue and pipeline
        let cmd_buf = cached.queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&cached.pipeline);
        // SAFETY: all buffers are valid MTLBuffers created above with matching
        // sizes and types expected by the moe_gather_scatter Metal kernel.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&token_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&indices_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&cursor_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buf), 0, 4);
        }

        // D17: 2D grid dispatch (token_stride, batch*top_k, 1) for better GPU
        // utilization. Note: the existing moe_gather_scatter kernel uses 1D tid
        // internally, so we keep grid_y=1 for the legacy kernel but size grid_x
        // to batch*top_k. New D14 kernels use 2D natively.
        let total_threads = batch_size * self.config.top_k;
        let max_tg = cached.pipeline.maxTotalThreadsPerThreadgroup();
        let tg_size = total_threads.min(max_tg);

        encoder.dispatchThreads_threadsPerThreadgroup(
            rmlx_metal::MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            rmlx_metal::MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        global_counters().record_gpu_sync();
        cmd_buf.waitUntilCompleted();

        // Read output
        if output_size == 0 {
            return Ok(RouteOutput {
                data: Vec::new(),
                route_indices: Vec::new(),
            });
        }
        let output = read_buffer_bytes(&output_buf, output_size)?;

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
        let local_buf_size = local_expert_count * rank_cap * token_stride;

        // Use slab ring buffer as pre-allocated memory if available and large enough.
        // CPU-only path: no GPU command buffers involved, so we bypass the
        // acquire/produce/consume/release lifecycle and access slab(0) directly.
        let local_output_ptr = if let Some(ref ring) = self.slab_ring {
            let slab = ring.slab(0);
            if slab.size >= local_buf_size && local_buf_size > 0 {
                let ptr = slab.metal_buffer.contents().as_ptr() as *mut u8;
                // SAFETY: ptr is valid for slab.size bytes (StorageModeShared Metal buffer),
                // and we checked slab.size >= local_buf_size above.
                unsafe {
                    std::ptr::write_bytes(ptr, 0, local_buf_size);
                }
                Some(ptr)
            } else {
                None // Slab too small -- fall back to heap
            }
        } else {
            None
        };

        // Heap fallback
        let mut local_output_vec = if local_output_ptr.is_none() {
            vec![0u8; local_buf_size]
        } else {
            Vec::new() // placeholder, won't be used
        };

        // Get a mutable slice to work with (slab or heap)
        // SAFETY: if local_output_ptr is Some, the pointer is valid for local_buf_size bytes
        // (guaranteed by the slab.size >= local_buf_size check above).
        let local_output: &mut [u8] = if let Some(ptr) = local_output_ptr {
            unsafe { std::slice::from_raw_parts_mut(ptr, local_buf_size) }
        } else {
            &mut local_output_vec
        };

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

        // --- V3 protocol: repack remote buffers if configured ---
        if self.config.wire_protocol == WireProtocol::V3 {
            #[allow(clippy::needless_range_loop)]
            for pr in 0..world_size {
                if remote_buffers[pr].is_empty() {
                    continue;
                }
                // Parse V2 format (4-byte expert_id + token_data) into v3 input tuples
                let v2_stride = 4 + token_stride;
                let mut tokens_for_v3: Vec<(u16, u16, Vec<u8>)> = Vec::new();
                let buf = &remote_buffers[pr];
                let mut off = 0;
                // Receiver's local expert range for this peer rank
                let receiver_local_start = pr * experts_per_rank;
                while off + v2_stride <= buf.len() {
                    let eid = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap_or([0; 4]));
                    // Convert global expert ID to receiver-local expert ID
                    let local_eid = {
                        let raw = (eid as usize).saturating_sub(receiver_local_start);
                        u16::try_from(raw).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 local_eid {} exceeds u16 range",
                                raw
                            ))
                        })?
                    };
                    let position = {
                        let pos_in_peer = tokens_for_v3.len();
                        u16::try_from(pos_in_peer).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 position {} exceeds u16 range",
                                pos_in_peer
                            ))
                        })?
                    };
                    tokens_for_v3.push((
                        local_eid,
                        position,
                        buf[off + 4..off + v2_stride].to_vec(),
                    ));
                    off += v2_stride;
                }
                let refs: Vec<(u16, u16, &[u8])> = tokens_for_v3
                    .iter()
                    .map(|(e, p, d)| (*e, *p, d.as_slice()))
                    .collect();
                let packet = v3_protocol::pack_dispatch_v3(&refs, pr as u32);
                remote_buffers[pr] = packet.wire_data;
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
        let wire_stride = 4 + token_stride; // 4-byte expert_id prefix + token data (V2)
        for &peer_rank in self.config.group.peers().iter() {
            let pr = peer_rank as usize;
            let recv_size = recv_sizes[pr];
            let send_buf = &remote_buffers[pr];
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

            // --- Unpack received data based on wire protocol ---
            let src_rank = pr;
            if self.config.wire_protocol == WireProtocol::V3 {
                // V3 unpack: variable-length packets grouped by expert
                let unpacked =
                    v3_protocol::unpack_dispatch_v3(&received, token_stride, local_expert_count)
                        .map_err(|e| DistributedError::Protocol(format!("v3 unpack: {e}")))?;
                for (local_expert_id, tokens) in unpacked.tokens_by_expert.iter().enumerate() {
                    let expert_id = local_start + local_expert_id;
                    if expert_id >= local_end {
                        continue;
                    }
                    let local_expert_idx = expert_id - local_start;
                    for (_, token_data_v3) in tokens {
                        let cursor_idx = local_expert_idx * world_size + src_rank;
                        let cursor = local_cursors[cursor_idx];
                        if cursor < capacity_per_expert {
                            let flat_slot = local_expert_idx * rank_cap
                                + src_rank * capacity_per_expert
                                + cursor;
                            let dst_start = flat_slot * token_stride;
                            let dst_end = dst_start + token_stride;
                            if dst_end <= local_output.len() && token_data_v3.len() >= token_stride
                            {
                                local_output[dst_start..dst_end]
                                    .copy_from_slice(&token_data_v3[..token_stride]);
                                local_cursors[cursor_idx] += 1;
                            }
                        }
                    }
                }
            } else {
                // V2 unpack: fixed-size packets with expert_id prefix
                let mut offset = 0;
                while offset + wire_stride <= received.len() {
                    let expert_id = u32::from_le_bytes(
                        received[offset..offset + 4].try_into().map_err(|_| {
                            DistributedError::Protocol(format!(
                                "invalid expert_id bytes at offset {offset}"
                            ))
                        })?,
                    ) as usize;
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
                                local_output[dst_start..dst_end].copy_from_slice(
                                    &received[offset + 4..offset + 4 + token_stride],
                                );
                                local_cursors[cursor_idx] += 1;
                            }
                        }
                    }
                    offset += wire_stride;
                }
            }
        }

        let _ = expert_counts;

        // Copy result from whichever buffer was used
        let result_data = if local_output_ptr.is_some() {
            // Copy from slab to Vec for return (preserves current API).
            // No ring state to restore -- slab(0) was used as plain memory.
            local_output[..local_buf_size].to_vec()
        } else {
            local_output_vec
        };

        Ok(RouteOutput {
            data: result_data,
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

    /// Get policy reference (for threshold updates).
    /// MoePolicy uses interior mutability, so `&MoePolicy` suffices for mutation.
    pub fn policy_mut(&self) -> &MoePolicy {
        &self.policy
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

    /// Number of ranks in the communication group.
    pub fn world_size(&self) -> usize {
        self.config.group.size()
    }

    /// Reference to the communication group (includes transport if available).
    pub fn group(&self) -> &Group {
        &self.config.group
    }

    /// Compute the local expert index range `[start, end)` for this rank.
    ///
    /// Experts are evenly partitioned across ranks: rank `r` owns experts
    /// `[r * experts_per_rank, (r+1) * experts_per_rank)`.
    ///
    /// Returns `(0, num_experts)` if `world_size <= 1` (all experts are local).
    pub fn local_expert_range(&self) -> (usize, usize) {
        let ws = self.config.group.size();
        let num_experts = self.config.num_experts;
        if ws <= 1 {
            return (0, num_experts);
        }
        let experts_per_rank = num_experts / ws;
        let local_rank = self.config.group.local_rank() as usize;
        let start = local_rank * experts_per_rank;
        let end = start + experts_per_rank;
        (start, end)
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

        // --- V3 protocol: repack remote buffers if configured ---
        if self.config.wire_protocol == WireProtocol::V3 {
            #[allow(clippy::needless_range_loop)]
            for pr in 0..world_size {
                if remote_buffers[pr].is_empty() {
                    continue;
                }
                // Parse V2 format (4-byte expert_id + token_data) into v3 input tuples
                let v2_stride = 4 + token_stride;
                let mut tokens_for_v3: Vec<(u16, u16, Vec<u8>)> = Vec::new();
                let buf = &remote_buffers[pr];
                let mut off = 0;
                // Receiver's local expert range for this peer rank
                let receiver_local_start = pr * experts_per_rank;
                while off + v2_stride <= buf.len() {
                    let eid = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap_or([0; 4]));
                    // Convert global expert ID to receiver-local expert ID
                    let local_eid = {
                        let raw = (eid as usize).saturating_sub(receiver_local_start);
                        u16::try_from(raw).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 local_eid {} exceeds u16 range",
                                raw
                            ))
                        })?
                    };
                    let position = {
                        let pos_in_peer = tokens_for_v3.len();
                        u16::try_from(pos_in_peer).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 position {} exceeds u16 range",
                                pos_in_peer
                            ))
                        })?
                    };
                    tokens_for_v3.push((
                        local_eid,
                        position,
                        buf[off + 4..off + v2_stride].to_vec(),
                    ));
                    off += v2_stride;
                }
                let refs: Vec<(u16, u16, &[u8])> = tokens_for_v3
                    .iter()
                    .map(|(e, p, d)| (*e, *p, d.as_slice()))
                    .collect();
                let packet = v3_protocol::pack_dispatch_v3(&refs, pr as u32);
                remote_buffers[pr] = packet.wire_data;
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

            // --- Unpack received data based on wire protocol ---
            // Merge received tokens directly into SharedBuffer
            // Source rank for received tokens is the peer rank
            let src_rank = pr;
            if self.config.wire_protocol == WireProtocol::V3 {
                // V3 unpack: variable-length packets grouped by expert
                let unpacked =
                    v3_protocol::unpack_dispatch_v3(&received, token_stride, local_expert_count)
                        .map_err(|e| DistributedError::Protocol(format!("v3 unpack: {e}")))?;
                for (local_expert_id, tokens) in unpacked.tokens_by_expert.iter().enumerate() {
                    let expert_id = local_start + local_expert_id;
                    if expert_id >= local_end {
                        continue;
                    }
                    let local_expert_idx = expert_id - local_start;
                    for (_, token_data_v3) in tokens {
                        let cursor_idx = local_expert_idx * world_size + src_rank;
                        let cursor = local_cursors[cursor_idx];
                        if cursor < capacity_per_expert {
                            let flat_slot = local_expert_idx * rank_cap
                                + src_rank * capacity_per_expert
                                + cursor;
                            let dst_offset = flat_slot * token_stride;
                            let dst_end = dst_offset + token_stride;
                            if dst_end <= local_buf_size
                                && dst_end <= shared_buf.size()
                                && token_data_v3.len() >= token_stride
                            {
                                global_counters().record_cpu_copy(token_stride as u64);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        token_data_v3[..token_stride].as_ptr(),
                                        output_ptr.add(dst_offset),
                                        token_stride,
                                    );
                                }
                                local_cursors[cursor_idx] += 1;
                            }
                        }
                    }
                }
            } else {
                // V2 unpack: fixed-size packets with expert_id prefix
                let mut offset = 0;
                while offset + wire_stride <= received.len() {
                    let expert_id = u32::from_le_bytes(
                        received[offset..offset + 4].try_into().map_err(|_| {
                            DistributedError::Protocol(format!(
                                "invalid expert_id bytes at offset {offset}"
                            ))
                        })?,
                    ) as usize;
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
        }

        // Copy from SharedBuffer to return Vec (the SharedBuffer's Metal view
        // is now ready for GPU compute without additional copies)
        let result = read_shared_buffer_bytes(shared_buf, local_buf_size.min(shared_buf.size()))?;
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
        peer_send_bufs: &[Option<Arc<SharedBuffer>>],
        peer_recv_bufs: &[Option<Arc<SharedBuffer>>],
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
                        if let Some(send_buf) =
                            peer_send_bufs.get(target_rank).and_then(|b| b.as_deref())
                        {
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

        // --- V3 protocol: repack peer SharedBuffers if configured ---
        if self.config.wire_protocol == WireProtocol::V3 {
            #[allow(clippy::needless_range_loop)]
            for pr in 0..world_size {
                let payload_len = peer_cursors[pr];
                if payload_len == 0 {
                    continue;
                }
                let send_buf = match peer_send_bufs.get(pr).and_then(|b| b.as_deref()) {
                    Some(b) => b,
                    None => continue,
                };
                // Read V2 data from SharedBuffer into a temporary Vec
                let v2_data = unsafe {
                    std::slice::from_raw_parts(send_buf.as_ptr() as *const u8, payload_len).to_vec()
                };
                // Parse V2 format (4-byte expert_id + token_data) into v3 input tuples
                let v2_stride = 4 + token_stride;
                let mut tokens_for_v3: Vec<(u16, u16, Vec<u8>)> = Vec::new();
                let mut off = 0;
                let receiver_local_start = pr * experts_per_rank;
                while off + v2_stride <= v2_data.len() {
                    let eid =
                        u32::from_le_bytes(v2_data[off..off + 4].try_into().unwrap_or([0; 4]));
                    let local_eid = {
                        let raw = (eid as usize).saturating_sub(receiver_local_start);
                        u16::try_from(raw).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 local_eid {} exceeds u16 range",
                                raw
                            ))
                        })?
                    };
                    let position = {
                        let pos_in_peer = tokens_for_v3.len();
                        u16::try_from(pos_in_peer).map_err(|_| {
                            DistributedError::Protocol(format!(
                                "V3 position {} exceeds u16 range",
                                pos_in_peer
                            ))
                        })?
                    };
                    tokens_for_v3.push((
                        local_eid,
                        position,
                        v2_data[off + 4..off + v2_stride].to_vec(),
                    ));
                    off += v2_stride;
                }
                let refs: Vec<(u16, u16, &[u8])> = tokens_for_v3
                    .iter()
                    .map(|(e, p, d)| (*e, *p, d.as_slice()))
                    .collect();
                let packet = v3_protocol::pack_dispatch_v3(&refs, pr as u32);
                // Write V3 packet back into SharedBuffer
                let v3_len = packet.wire_data.len();
                if v3_len <= send_buf.size() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            packet.wire_data.as_ptr(),
                            send_buf.as_ptr(),
                            v3_len,
                        );
                    }
                    peer_cursors[pr] = v3_len;
                } else {
                    return Err(DistributedError::Transport(format!(
                        "V3 packet ({} bytes) exceeds SharedBuffer capacity ({} bytes) for peer {}",
                        v3_len,
                        send_buf.size(),
                        pr
                    )));
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
                if let Some(send_buf) = peer_send_bufs.get(pr).and_then(|b| b.clone()) {
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
            if let Some(recv_buf) = peer_recv_bufs.get(pr).and_then(|b| b.clone()) {
                // Recv capacity: worst case the peer sends us all its tokens
                let v2_max = experts_per_rank * capacity_per_expert * wire_stride;
                let max_recv =
                    recv_buf
                        .size()
                        .min(if self.config.wire_protocol == WireProtocol::V3 {
                            4 + v2_max // V3 adds a 4-byte count header
                        } else {
                            v2_max
                        });
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
/// Contains `ZeroCopyPendingOp` handles for the RDMA transfers. Each handle
/// holds an `Arc<SharedBuffer>` reference, preventing the buffer from being
/// freed while the DMA operation is in flight. The caller must poll or wait
/// on these before reading the received SharedBuffers.
pub struct AsyncDispatchResult {
    /// Cached dispatch layout for reuse in combine.
    pub layout: DispatchLayout,
    /// ZeroCopyPendingOps for outbound RDMA sends to peers.
    pub send_ops: Vec<ZeroCopyPendingOp>,
    /// ZeroCopyPendingOps for inbound RDMA recvs from peers.
    pub recv_ops: Vec<ZeroCopyPendingOp>,
}

impl AsyncDispatchResult {
    /// Poll all pending operations through a ProgressTracker.
    ///
    /// Returns the total number of newly completed operations (sends + recvs).
    /// The tracker's health monitoring is updated for each polled result.
    pub fn poll_tracked(&self, tracker: &ProgressTracker) -> usize {
        let sends = tracker.poll_all_zero_copy(&self.send_ops);
        let recvs = tracker.poll_all_zero_copy(&self.recv_ops);
        sends + recvs
    }

    /// Wait for all pending operations through a ProgressTracker.
    ///
    /// Blocks until all send and recv ops complete or an error threshold is
    /// reached. Returns the total number of successfully completed operations.
    pub fn wait_tracked(
        &self,
        tracker: &ProgressTracker,
        per_op_timeout: std::time::Duration,
    ) -> Result<usize, DistributedError> {
        let sends = tracker.wait_all_zero_copy(&self.send_ops, per_op_timeout)?;
        let recvs = tracker.wait_all_zero_copy(&self.recv_ops, per_op_timeout)?;
        Ok(sends + recvs)
    }

    /// Check if all operations have completed (non-blocking).
    pub fn all_complete(&self) -> bool {
        self.send_ops.iter().all(|op| !op.is_pending())
            && self.recv_ops.iter().all(|op| !op.is_pending())
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
    /// Optional progress tracker for async combine health monitoring.
    progress_tracker: Option<Arc<ProgressTracker>>,
}

impl MoeCombineExchange {
    pub fn new(group: Group) -> Self {
        Self {
            group,
            combine_metal_cache: Mutex::new(None),
            runtime_ctx: None,
            progress_tracker: None,
        }
    }

    /// Attach an EP runtime context for async zero-copy combine.
    pub fn with_runtime_context(mut self, ctx: Arc<EpRuntimeContext>) -> Self {
        self.runtime_ctx = Some(ctx);
        self
    }

    /// Attach a progress tracker for async combine health monitoring.
    pub fn with_progress_tracker(mut self, tracker: Arc<ProgressTracker>) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }

    /// Access the progress tracker, if attached.
    pub fn progress_tracker(&self) -> Option<&Arc<ProgressTracker>> {
        self.progress_tracker.as_ref()
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
    ) -> Result<Vec<f32>, DistributedError> {
        if batch_size == 0 || hidden_dim == 0 {
            return Ok(vec![0.0f32; batch_size * hidden_dim]);
        }

        let num_experts = expert_outputs.len();
        if num_experts == 0 {
            return Ok(vec![0.0f32; batch_size * hidden_dim]);
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
                return Ok(self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                ));
            }
        };
        if cache_guard.is_none() {
            let device = match objc2_metal::MTLCreateSystemDefaultDevice() {
                Some(d) => d,
                None => {
                    drop(cache_guard);
                    return Ok(self.combine_cpu(
                        expert_outputs,
                        weights,
                        indices,
                        batch_size,
                        top_k,
                        hidden_dim,
                    ));
                }
            };
            let options = objc2_metal::MTLCompileOptions::new();
            let library = match device.newLibraryWithSource_options_error(
                &objc2_foundation::NSString::from_str(kernel_src),
                Some(&options),
            ) {
                Ok(lib) => lib,
                Err(_) => {
                    drop(cache_guard);
                    return Ok(self.combine_cpu(
                        expert_outputs,
                        weights,
                        indices,
                        batch_size,
                        top_k,
                        hidden_dim,
                    ));
                }
            };
            let function = match library
                .newFunctionWithName(&objc2_foundation::NSString::from_str("moe_combine"))
            {
                Some(f) => f,
                None => {
                    drop(cache_guard);
                    return Ok(self.combine_cpu(
                        expert_outputs,
                        weights,
                        indices,
                        batch_size,
                        top_k,
                        hidden_dim,
                    ));
                }
            };
            let pipeline = match device.newComputePipelineStateWithFunction_error(&function) {
                Ok(p) => p,
                Err(_) => {
                    drop(cache_guard);
                    return Ok(self.combine_cpu(
                        expert_outputs,
                        weights,
                        indices,
                        batch_size,
                        top_k,
                        hidden_dim,
                    ));
                }
            };
            let queue = device.newCommandQueue().unwrap();
            *cache_guard = Some(CachedMetalPipeline {
                device,
                pipeline,
                queue,
            });
        }
        let cached = match cache_guard.as_ref() {
            Some(c) => c,
            None => {
                return Ok(self.combine_cpu(
                    expert_outputs,
                    weights,
                    indices,
                    batch_size,
                    top_k,
                    hidden_dim,
                ));
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

        let shared = rmlx_metal::MTLResourceOptions::StorageModeShared;
        let expert_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(expert_data.as_ptr() as *mut std::ffi::c_void).unwrap(),
                expert_data.len() * 4,
                shared,
            )
        }
        .unwrap();
        let weights_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(weights.as_ptr() as *mut std::ffi::c_void).unwrap(),
                weights.len() * 4,
                shared,
            )
        }
        .unwrap();
        let indices_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(indices.as_ptr() as *mut std::ffi::c_void).unwrap(),
                indices.len() * 4,
                shared,
            )
        }
        .unwrap();
        let output_size = batch_size * hidden_dim;
        let output_buf = cached
            .device
            .newBufferWithLength_options((output_size * 4).max(4), shared)
            .unwrap();

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
        let params_buf = unsafe {
            cached.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(&params as *const CombineParams as *mut std::ffi::c_void)
                    .unwrap(),
                std::mem::size_of::<CombineParams>(),
                shared,
            )
        }
        .unwrap();

        let cmd_buf = cached.queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&cached.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&expert_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&weights_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&indices_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buf), 0, 4);
        }

        // D17: The legacy moe_combine kernel uses 1D tid, so we dispatch
        // 1D here for backward compatibility. The new D14/D15/D16 kernels use
        // uint2 tid and are dispatched with 2D grids (D, batch*top_k, 1).
        let total_threads = output_size;
        let max_tg = cached.pipeline.maxTotalThreadsPerThreadgroup();
        let tg_size = total_threads.min(max_tg);

        encoder.dispatchThreads_threadsPerThreadgroup(
            rmlx_metal::MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            },
            rmlx_metal::MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        global_counters().record_gpu_sync();
        cmd_buf.waitUntilCompleted();

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
            return self.combine_metal(
                local_expert_outputs,
                weights,
                indices,
                batch_size,
                top_k,
                hidden_dim,
            );
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
                let send_bytes: Vec<u8> = send_slice.iter().flat_map(|f| f.to_ne_bytes()).collect();

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
        self.combine_metal(
            &all_expert_outputs,
            weights,
            indices,
            batch_size,
            top_k,
            hidden_dim,
        )
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
        // When route_indices are available from the dispatch layout, use them
        // to look up the correct flat slot in the expert output buffer.
        // With per-rank capacity packing, flat_slot != batch_idx, so we must
        // use route_indices to find the right row.
        let has_route_indices = !layout.route_indices.is_empty();
        let (local_start, _local_end) = layout.local_expert_range;

        match layout.backend {
            MoeBackend::Cpu | MoeBackend::Metal => {
                if has_route_indices {
                    Ok(self.combine_cpu_with_route_indices(
                        local_expert_outputs,
                        weights,
                        batch_size,
                        top_k,
                        hidden_dim,
                        &layout.route_indices,
                        indices,
                        local_start,
                    ))
                } else {
                    // Fallback to legacy batch_idx-based addressing
                    match layout.backend {
                        MoeBackend::Cpu => Ok(self.combine_cpu(
                            local_expert_outputs,
                            weights,
                            indices,
                            batch_size,
                            top_k,
                            hidden_dim,
                        )),
                        _ => self.combine_metal(
                            local_expert_outputs,
                            weights,
                            indices,
                            batch_size,
                            top_k,
                            hidden_dim,
                        ),
                    }
                }
            }
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

    /// Combine using route_indices from DispatchLayout (D4).
    ///
    /// For each (n, k) pair, looks up `slot = route_indices[n * top_k + k]`.
    /// If slot == -1, the token was dropped (overflow or remote) and is skipped.
    /// Otherwise reads from `expert_outputs[expert_idx][slot * hidden_dim..]`
    /// instead of `batch_idx * hidden_dim`.
    #[allow(clippy::too_many_arguments)]
    fn combine_cpu_with_route_indices(
        &self,
        expert_outputs: &[Vec<f32>],
        weights: &[f32],
        batch_size: usize,
        top_k: usize,
        hidden_dim: usize,
        route_indices: &[i32],
        indices: &[u32],
        local_start: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * hidden_dim];

        for batch_idx in 0..batch_size {
            for k in 0..top_k {
                let flat_idx = batch_idx * top_k + k;
                let slot = route_indices[flat_idx];
                // slot == -1 means dropped token; skip it
                if slot < 0 {
                    continue;
                }
                let global_expert_idx = indices[flat_idx] as usize;
                let weight = weights[flat_idx];

                // Convert global expert ID to local index (0-based within
                // this rank's expert range) since expert_outputs only
                // contains this rank's experts.
                if global_expert_idx < local_start {
                    continue;
                }
                let local_idx = global_expert_idx - local_start;

                if local_idx < expert_outputs.len() {
                    let expert_out = &expert_outputs[local_idx];
                    let out_base = batch_idx * hidden_dim;
                    let exp_base = (slot as usize) * hidden_dim;
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
        send_bufs: &[Option<Arc<SharedBuffer>>],
        recv_bufs: &[Option<Arc<SharedBuffer>>],
        conn_ids: &[Option<&ConnectionId>],
        transport: &RdmaConnectionTransport,
    ) -> Result<AsyncCombineHandle, DistributedError> {
        global_counters().record_async_combine();

        let (_local_start, _local_end) = layout.local_expert_range;
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
            if let Some(send_buf) = send_bufs.get(pr).and_then(|b| b.clone()) {
                let total_send_len = experts_per_rank * segment_bytes;
                if total_send_len > 0 && total_send_len <= send_buf.size() {
                    let send_ptr = send_buf.as_ptr();
                    for i in 0..experts_per_rank {
                        // local_expert_outputs is 0-indexed (contains only this
                        // rank's experts), so index with `i` directly, not
                        // `local_start + i` (which is the global expert ID).
                        let buf_offset = i * segment_bytes;
                        if i < local_expert_outputs.len() && !local_expert_outputs[i].is_empty() {
                            let src = &local_expert_outputs[i];
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
                                std::ptr::write_bytes(send_ptr.add(buf_offset), 0, segment_bytes);
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
            if let Some(recv_buf) = recv_bufs.get(pr).and_then(|b| b.clone()) {
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
        recv_bufs: &[Option<Arc<SharedBuffer>>],
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
            if let Some(recv_buf) = recv_bufs.get(pr).and_then(|b| b.as_deref()) {
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
                        let src_ptr = unsafe { recv_buf.as_ptr().add(buf_offset) } as *const f32;
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
        self.combine_metal(
            &all_expert_outputs,
            weights,
            indices,
            batch_size,
            top_k,
            hidden_dim,
        )
    }

    /// Group reference.
    pub fn group(&self) -> &Group {
        &self.group
    }
}

/// Handle for in-flight async combine RDMA transfers (Phase G2).
///
/// Holds `ZeroCopyPendingOp` handles for the send/recv operations posted by
/// `combine_async_start`. Each handle keeps its `SharedBuffer` alive via
/// `Arc` for the duration of the DMA. The caller polls or waits on these
/// before calling `combine_async_finish`.
pub struct AsyncCombineHandle {
    pub send_ops: Vec<ZeroCopyPendingOp>,
    pub recv_ops: Vec<ZeroCopyPendingOp>,
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

    /// Poll all pending operations through a ProgressTracker.
    ///
    /// Returns the total number of newly completed operations.
    pub fn poll_tracked(&self, tracker: &ProgressTracker) -> usize {
        let sends = tracker.poll_all_zero_copy(&self.send_ops);
        let recvs = tracker.poll_all_zero_copy(&self.recv_ops);
        sends + recvs
    }

    /// Wait for all pending operations through a ProgressTracker.
    ///
    /// Returns the total number of successfully completed operations.
    pub fn wait_tracked(
        &self,
        tracker: &ProgressTracker,
        per_op_timeout: std::time::Duration,
    ) -> Result<usize, DistributedError> {
        let sends = tracker.wait_all_zero_copy(&self.send_ops, per_op_timeout)?;
        let recvs = tracker.wait_all_zero_copy(&self.recv_ops, per_op_timeout)?;
        Ok(sends + recvs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn test_device() -> Option<&'static rmlx_metal::MtlDevice> {
        static DEVICE: OnceLock<Option<rmlx_metal::MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| {
                objc2::rc::autoreleasepool(|_| objc2_metal::MTLCreateSystemDefaultDevice())
            })
            .as_ref()
    }

    #[test]
    fn read_buffer_bytes_oob_returns_error() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let buf = device
            .newBufferWithLength_options(64, rmlx_metal::MTLResourceOptions::StorageModeShared)
            .unwrap();
        // Requesting more bytes than the buffer holds should return Err.
        let result = read_buffer_bytes(&buf, 128);
        assert!(result.is_err(), "expected error for OOB read");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("exceeds buffer length"),
            "error message: {msg}"
        );
    }

    #[test]
    fn read_buffer_bytes_exact_len_succeeds() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let buf = device
            .newBufferWithLength_options(64, rmlx_metal::MTLResourceOptions::StorageModeShared)
            .unwrap();
        // Reading exactly the buffer length should succeed.
        let result = read_buffer_bytes(&buf, 64);
        assert!(result.is_ok(), "exact-length read should succeed");
        assert_eq!(result.unwrap().len(), 64);
    }

    #[test]
    fn read_buffer_f32_oob_returns_error() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // 16 bytes = 4 f32 elements
        let buf = device
            .newBufferWithLength_options(16, rmlx_metal::MTLResourceOptions::StorageModeShared)
            .unwrap();
        // Requesting 8 f32s (32 bytes) from a 16-byte buffer should fail.
        let result = read_buffer_f32(&buf, 8);
        assert!(result.is_err(), "expected error for OOB f32 read");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("exceeds buffer length"),
            "error message: {msg}"
        );
    }

    #[test]
    fn read_buffer_f32_exact_len_succeeds() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        // 16 bytes = 4 f32 elements
        let buf = device
            .newBufferWithLength_options(16, rmlx_metal::MTLResourceOptions::StorageModeShared)
            .unwrap();
        let result = read_buffer_f32(&buf, 4);
        assert!(result.is_ok(), "exact-length f32 read should succeed");
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn read_shared_buffer_bytes_oob_returns_error() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let shared_buf = match SharedBuffer::new(device, 64, 0) {
            Ok(b) => b,
            Err(_) => return, // skip if SharedBuffer allocation fails
        };
        // SharedBuffer page-aligns to 16384, so request more than that to trigger OOB.
        let result = read_shared_buffer_bytes(&shared_buf, shared_buf.size() + 1);
        assert!(result.is_err(), "expected error for OOB SharedBuffer read");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("exceeds SharedBuffer size"),
            "error message: {msg}"
        );
    }

    // ---- EP dispatch/combine correctness tests (D-P0-2, D-P0-4) ----

    /// Helper: build a single-rank MoeCombineExchange for unit tests.
    fn make_single_rank_combine() -> MoeCombineExchange {
        let group = Group::new(vec![0], 0, 1).expect("single-rank group");
        MoeCombineExchange::new(group)
    }

    /// Helper: build a DispatchLayout for testing combine_with_layout.
    fn make_test_layout(
        batch_size: usize,
        _top_k: usize,
        num_experts: usize,
        local_start: usize,
        experts_per_rank: usize,
        route_indices: Vec<i32>,
        expert_indices: Vec<u32>,
    ) -> DispatchLayout {
        DispatchLayout {
            backend: MoeBackend::Cpu,
            expert_counts: vec![0; num_experts],
            tokens_per_expert: batch_size, // capacity = batch_size for simplicity
            local_expert_range: (local_start, local_start + experts_per_rank),
            token_stride: 0,
            experts_per_rank,
            batch_size,
            expert_indices: expert_indices.clone(),
            peer_payload_sizes: vec![],
            route_indices,
        }
    }

    #[test]
    fn combine_with_route_indices_global_to_local_mapping() {
        // Scenario: 4 global experts, rank 1 owns experts 2 and 3.
        // local_expert_outputs has 2 entries (index 0 = global expert 2,
        // index 1 = global expert 3).
        let combiner = make_single_rank_combine();
        let hidden_dim = 2;
        let batch_size = 2;
        let top_k = 1;
        let local_start = 2; // rank 1 owns experts [2, 4)
        let experts_per_rank = 2;
        let num_experts = 4;

        // local_expert_outputs[0] = expert 2 output, local_expert_outputs[1] = expert 3 output
        // Each expert buffer has `batch_size * hidden_dim` elements.
        // Using route_indices: token 0 -> slot 0, token 1 -> slot 1
        let expert_2_out = vec![1.0, 2.0, 3.0, 4.0]; // slot 0: [1,2], slot 1: [3,4]
        let expert_3_out = vec![10.0, 20.0, 30.0, 40.0];
        let local_expert_outputs = vec![expert_2_out, expert_3_out];

        // Token 0 -> global expert 2 (slot 0), Token 1 -> global expert 3 (slot 1)
        let indices = vec![2u32, 3u32];
        let weights = vec![0.5f32, 0.8f32];
        let route_indices = vec![0i32, 1i32]; // slot assignments

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        // Token 0: weight 0.5 * expert_2[slot 0] = 0.5 * [1, 2] = [0.5, 1.0]
        // Token 1: weight 0.8 * expert_3[slot 1] = 0.8 * [30, 40] = [24.0, 32.0]
        assert_eq!(result.len(), batch_size * hidden_dim);
        let eps = 1e-6;
        assert!(
            (result[0] - 0.5).abs() < eps,
            "token 0, dim 0: {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < eps,
            "token 0, dim 1: {}",
            result[1]
        );
        assert!(
            (result[2] - 24.0).abs() < eps,
            "token 1, dim 0: {}",
            result[2]
        );
        assert!(
            (result[3] - 32.0).abs() < eps,
            "token 1, dim 1: {}",
            result[3]
        );
    }

    #[test]
    fn combine_with_route_indices_top2_accumulation() {
        // Scenario: top_k=2, each token routed to 2 experts.
        // Tests that weighted outputs are correctly accumulated.
        let combiner = make_single_rank_combine();
        let hidden_dim = 2;
        let batch_size = 1;
        let top_k = 2;
        let local_start = 0;
        let experts_per_rank = 2;
        let num_experts = 2;

        // Expert 0: slot 0 -> [1, 2], Expert 1: slot 0 -> [10, 20]
        let expert_0_out = vec![1.0, 2.0];
        let expert_1_out = vec![10.0, 20.0];
        let local_expert_outputs = vec![expert_0_out, expert_1_out];

        // Token 0 -> expert 0 (slot 0, weight 0.6) and expert 1 (slot 0, weight 0.4)
        let indices = vec![0u32, 1u32];
        let weights = vec![0.6f32, 0.4f32];
        let route_indices = vec![0i32, 0i32];

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        // output = 0.6 * [1, 2] + 0.4 * [10, 20] = [0.6+4.0, 1.2+8.0] = [4.6, 9.2]
        let eps = 1e-5;
        assert!((result[0] - 4.6).abs() < eps, "dim 0: {}", result[0]);
        assert!((result[1] - 9.2).abs() < eps, "dim 1: {}", result[1]);
    }

    #[test]
    fn combine_with_route_indices_dropped_tokens() {
        // Tokens with route_indices == -1 should be skipped (zero output).
        let combiner = make_single_rank_combine();
        let hidden_dim = 2;
        let batch_size = 2;
        let top_k = 1;
        let local_start = 0;
        let experts_per_rank = 1;
        let num_experts = 1;

        let expert_0_out = vec![5.0, 6.0, 7.0, 8.0];
        let local_expert_outputs = vec![expert_0_out];

        let indices = vec![0u32, 0u32];
        let weights = vec![1.0f32, 1.0f32];
        // Token 0 -> slot 0, Token 1 -> dropped (-1)
        let route_indices = vec![0i32, -1i32];

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        let eps = 1e-6;
        // Token 0: 1.0 * [5, 6] = [5.0, 6.0]
        assert!(
            (result[0] - 5.0).abs() < eps,
            "token 0, dim 0: {}",
            result[0]
        );
        assert!(
            (result[1] - 6.0).abs() < eps,
            "token 0, dim 1: {}",
            result[1]
        );
        // Token 1: dropped, should be zero
        assert!(
            (result[2]).abs() < eps,
            "token 1, dim 0 should be 0: {}",
            result[2]
        );
        assert!(
            (result[3]).abs() < eps,
            "token 1, dim 1 should be 0: {}",
            result[3]
        );
    }

    #[test]
    fn combine_with_route_indices_empty_expert_outputs() {
        // When local_expert_outputs is empty, output should be all zeros.
        let combiner = make_single_rank_combine();
        let hidden_dim = 2;
        let batch_size = 1;
        let top_k = 1;
        let local_start = 0;
        let experts_per_rank = 0;
        let num_experts = 2;

        let local_expert_outputs: Vec<Vec<f32>> = vec![];
        let indices = vec![0u32];
        let weights = vec![1.0f32];
        let route_indices = vec![0i32];

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        // No experts to combine, output should be zeros
        assert!(
            result.iter().all(|&x| x == 0.0),
            "expected all zeros: {:?}",
            result
        );
    }

    #[test]
    fn combine_with_route_indices_all_tokens_one_expert() {
        // All tokens assigned to a single expert (stress test for slot indexing).
        let combiner = make_single_rank_combine();
        let hidden_dim = 3;
        let batch_size = 3;
        let top_k = 1;
        let local_start = 1; // rank owns expert 1 only
        let experts_per_rank = 1;
        let num_experts = 2;

        // Expert 1's local output: 3 slots x 3 hidden_dim
        let expert_1_out = vec![
            1.0, 2.0, 3.0, // slot 0
            4.0, 5.0, 6.0, // slot 1
            7.0, 8.0, 9.0, // slot 2
        ];
        let local_expert_outputs = vec![expert_1_out];

        // All tokens -> global expert 1, each at a different slot
        let indices = vec![1u32, 1u32, 1u32];
        let weights = vec![1.0f32, 0.5f32, 2.0f32];
        let route_indices = vec![0i32, 1i32, 2i32];

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        let eps = 1e-6;
        // Token 0: 1.0 * [1,2,3] = [1,2,3]
        assert!((result[0] - 1.0).abs() < eps);
        assert!((result[1] - 2.0).abs() < eps);
        assert!((result[2] - 3.0).abs() < eps);
        // Token 1: 0.5 * [4,5,6] = [2,2.5,3]
        assert!((result[3] - 2.0).abs() < eps);
        assert!((result[4] - 2.5).abs() < eps);
        assert!((result[5] - 3.0).abs() < eps);
        // Token 2: 2.0 * [7,8,9] = [14,16,18]
        assert!((result[6] - 14.0).abs() < eps);
        assert!((result[7] - 16.0).abs() < eps);
        assert!((result[8] - 18.0).abs() < eps);
    }

    #[test]
    fn combine_with_route_indices_out_of_range_expert_skipped() {
        // If a token references an expert outside this rank's range,
        // it should be safely skipped (no panic, zero contribution).
        let combiner = make_single_rank_combine();
        let hidden_dim = 2;
        let batch_size = 2;
        let top_k = 1;
        let local_start = 2; // owns experts [2, 4)
        let experts_per_rank = 2;
        let num_experts = 4;

        let expert_2_out = vec![1.0, 2.0, 3.0, 4.0];
        let expert_3_out = vec![5.0, 6.0, 7.0, 8.0];
        let local_expert_outputs = vec![expert_2_out, expert_3_out];

        // Token 0 -> global expert 0 (not local, should be skipped)
        // Token 1 -> global expert 3 (local, slot 0)
        let indices = vec![0u32, 3u32];
        let weights = vec![1.0f32, 1.0f32];
        let route_indices = vec![0i32, 0i32];

        let layout = make_test_layout(
            batch_size,
            top_k,
            num_experts,
            local_start,
            experts_per_rank,
            route_indices,
            indices.clone(),
        );

        let result = combiner
            .combine_with_layout(
                &local_expert_outputs,
                &weights,
                &indices,
                batch_size,
                top_k,
                hidden_dim,
                num_experts,
                &layout,
            )
            .expect("combine_with_layout should succeed");

        let eps = 1e-6;
        // Token 0: expert 0 is below local_start=2, skipped -> [0, 0]
        assert!((result[0]).abs() < eps, "token 0 dim 0: {}", result[0]);
        assert!((result[1]).abs() < eps, "token 0 dim 1: {}", result[1]);
        // Token 1: expert 3 -> local_idx=1, slot 0 -> expert_3_out[0..2] = [5, 6]
        assert!(
            (result[2] - 5.0).abs() < eps,
            "token 1 dim 0: {}",
            result[2]
        );
        assert!(
            (result[3] - 6.0).abs() < eps,
            "token 1 dim 1: {}",
            result[3]
        );
    }

    #[test]
    fn exchange_buffers_release_clears_all_slots() {
        // ExchangeBuffers::release() should set all slots to None.
        let mut eb = ExchangeBuffers {
            send_bufs: (0..4).map(|_| None).collect(),
            recv_bufs: (0..4).map(|_| None).collect(),
            world_size: 4,
            local_rank: 0,
        };
        assert_eq!(eb.peer_count(), 0);
        eb.release();
        assert!(eb.send_bufs.iter().all(|b| b.is_none()));
        assert!(eb.recv_bufs.iter().all(|b| b.is_none()));
    }

    #[test]
    fn exchange_buffers_peer_count_excludes_none() {
        let eb = ExchangeBuffers {
            send_bufs: (0..3).map(|_| None).collect(),
            recv_bufs: (0..3).map(|_| None).collect(),
            world_size: 3,
            local_rank: 1,
        };
        assert_eq!(eb.peer_count(), 0);
    }
}
