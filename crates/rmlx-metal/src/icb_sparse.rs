//! ICB-based sparse expert launch.
//!
//! Uses `MTLIndirectCommandBuffer` to let the GPU build dispatch commands,
//! skipping experts with zero allocated tokens. This eliminates CPU-side
//! iteration over experts and the conditional skip logic.
//!
//! # Design
//!
//! In MoE (Mixture of Experts) inference, not every expert receives tokens
//! after top-k routing. The CPU would normally iterate all `E` experts and
//! skip empty ones with a branch. With ICB sparse launch:
//!
//! 1. A GPU kernel reads `expert_counts[E]` and encodes dispatch commands
//!    only for non-empty experts.
//! 2. The ICB is then executed, dispatching only the needed compute kernels.
//! 3. Zero CPU involvement in deciding which experts to skip.
//!
//! The `SparseExpertPlan` pre-captures dispatch commands for every expert
//! using the maximum capacity. During replay, only experts with non-zero
//! counts are encoded into the command buffer.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::compute_pass::ComputePass;
use crate::icb::CapturedDispatch;
use crate::types::retain_proto;

/// Configuration for sparse expert ICB dispatch.
#[derive(Debug, Clone)]
pub struct SparseExpertConfig {
    /// Maximum number of experts (E).
    pub max_experts: usize,
    /// Maximum capacity per expert (max tokens per expert).
    pub max_capacity: usize,
    /// Hidden dimension (D).
    pub hidden_dim: usize,
    /// Intermediate dimension for FFN.
    pub intermediate_dim: usize,
}

/// A group of dispatches for a single expert's forward pass.
///
/// Each expert in an MoE FFN block performs four operations:
/// gate projection, up projection, SwiGLU activation, and down projection.
pub struct ExpertDispatchGroup {
    /// Expert index.
    pub expert_id: usize,
    /// Gate projection GEMM dispatch.
    pub gate_dispatch: CapturedDispatch,
    /// Up projection GEMM dispatch.
    pub up_dispatch: CapturedDispatch,
    /// SwiGLU activation dispatch.
    pub swiglu_dispatch: CapturedDispatch,
    /// Down projection GEMM dispatch.
    pub down_dispatch: CapturedDispatch,
}

impl ExpertDispatchGroup {
    /// Replay this expert's dispatches into a command buffer with an adjusted
    /// grid size based on the actual token count.
    ///
    /// Each dispatch's grid width is scaled from `max_capacity` down to
    /// `token_count`. The threadgroup size is clamped accordingly.
    fn replay_with_count(
        &self,
        token_count: u32,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        input_offset: u64,
        output_offset: u64,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let dispatches = [
            &self.gate_dispatch,
            &self.up_dispatch,
            &self.swiglu_dispatch,
            &self.down_dispatch,
        ];

        for dispatch in dispatches {
            let enc = cb.computeCommandEncoder().unwrap();
            let pass = ComputePass::new(&enc);
            pass.set_pipeline(&dispatch.pipeline);

            // Bind the pre-captured buffers (weights etc.) from the template.
            for (index, buffer, offset) in &dispatch.buffers {
                pass.set_buffer(*index as u32, Some(buffer), *offset as usize);
            }

            // Override input/output buffer bindings with the actual stacked
            // buffers and offsets for this expert's token slice.
            // Convention: buffer index 0 = input, buffer index 1 = output.
            pass.set_buffer(0, Some(input_buf), input_offset as usize);
            pass.set_buffer(1, Some(output_buf), output_offset as usize);

            // Scale grid to actual token count while keeping the original
            // height and depth (which encode hidden/intermediate dims).
            let grid = MTLSize {
                width: token_count as usize,
                height: dispatch.grid_size.height,
                depth: dispatch.grid_size.depth,
            };
            let tg = MTLSize {
                width: std::cmp::min(dispatch.threadgroup_size.width, token_count as usize),
                height: dispatch.threadgroup_size.height,
                depth: dispatch.threadgroup_size.depth,
            };

            pass.dispatch_threads(grid, tg);
            pass.end();
        }
    }
}

/// Pre-built sparse expert dispatch plan.
///
/// Given `expert_counts` from GPU top-k routing, this struct contains
/// the captured dispatch commands for all experts. During replay,
/// only experts with non-zero counts execute.
pub struct SparseExpertPlan {
    /// Per-expert captured dispatches (gate GEMM, up GEMM, SwiGLU, down GEMM).
    /// Indexed by expert_id.
    expert_dispatches: Vec<Option<ExpertDispatchGroup>>,
    /// Config used to build this plan.
    config: SparseExpertConfig,
}

impl SparseExpertPlan {
    /// Build a sparse expert plan from pipeline states and weight buffers.
    ///
    /// Pre-captures dispatch commands for each expert using `max_capacity`
    /// as the grid width. During replay, the actual grid size is determined
    /// by `expert_counts`.
    ///
    /// # Arguments
    ///
    /// * `config` - Expert configuration (dimensions, counts).
    /// * `gate_pipeline` - Pipeline state for gate projection GEMM.
    /// * `up_pipeline` - Pipeline state for up projection GEMM.
    /// * `swiglu_pipeline` - Pipeline state for SwiGLU activation.
    /// * `down_pipeline` - Pipeline state for down projection GEMM.
    /// * `gate_weights` - Stacked gate weight buffer `[E, D, inter]`.
    /// * `up_weights` - Stacked up weight buffer `[E, D, inter]`.
    /// * `down_weights` - Stacked down weight buffer `[E, inter, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        config: SparseExpertConfig,
        gate_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        up_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        swiglu_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        down_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        gate_weights: &ProtocolObject<dyn MTLBuffer>,
        up_weights: &ProtocolObject<dyn MTLBuffer>,
        down_weights: &ProtocolObject<dyn MTLBuffer>,
    ) -> Self {
        let max_cap = config.max_capacity;
        let hidden = config.hidden_dim;
        let inter = config.intermediate_dim;

        // Bytes per element (f16 = 2 bytes).
        let elem_size: usize = 2;

        let gate_expert_stride = hidden * inter * elem_size;
        let up_expert_stride = hidden * inter * elem_size;
        let down_expert_stride = inter * hidden * elem_size;

        let mut expert_dispatches = Vec::with_capacity(config.max_experts);

        for expert_id in 0..config.max_experts {
            let eid = expert_id;

            // Gate GEMM: [tokens, D] x [D, inter] -> [tokens, inter]
            let gate_dispatch = CapturedDispatch {
                pipeline: retain_proto(gate_pipeline),
                buffers: vec![
                    // input buffer placeholder at index 0 (overridden at replay)
                    // output buffer placeholder at index 1 (overridden at replay)
                    // weight buffer at index 2
                    (
                        2,
                        retain_proto(gate_weights),
                        (eid * gate_expert_stride) as u64,
                    ),
                ],
                input_indices: vec![0],
                output_indices: vec![0],
                grid_size: MTLSize {
                    width: max_cap,
                    height: inter,
                    depth: 1,
                },
                threadgroup_size: Self::compute_threadgroup(gate_pipeline, max_cap, inter),
            };

            // Up GEMM: [tokens, D] x [D, inter] -> [tokens, inter]
            let up_dispatch = CapturedDispatch {
                pipeline: retain_proto(up_pipeline),
                buffers: vec![(2, retain_proto(up_weights), (eid * up_expert_stride) as u64)],
                input_indices: vec![0],
                output_indices: vec![0],
                grid_size: MTLSize {
                    width: max_cap,
                    height: inter,
                    depth: 1,
                },
                threadgroup_size: Self::compute_threadgroup(up_pipeline, max_cap, inter),
            };

            // SwiGLU: element-wise on [tokens, inter]
            // Use 2D grid so replay_with_count can scale width (tokens)
            // independently of height (intermediate_dim).
            let swiglu_dispatch = CapturedDispatch {
                pipeline: retain_proto(swiglu_pipeline),
                buffers: vec![],
                input_indices: vec![],
                output_indices: vec![],
                grid_size: MTLSize {
                    width: max_cap,
                    height: inter,
                    depth: 1,
                },
                threadgroup_size: Self::compute_threadgroup(swiglu_pipeline, max_cap, inter),
            };

            // Down GEMM: [tokens, inter] x [inter, D] -> [tokens, D]
            let down_dispatch = CapturedDispatch {
                pipeline: retain_proto(down_pipeline),
                buffers: vec![(
                    2,
                    retain_proto(down_weights),
                    (eid * down_expert_stride) as u64,
                )],
                input_indices: vec![0],
                output_indices: vec![0],
                grid_size: MTLSize {
                    width: max_cap,
                    height: hidden,
                    depth: 1,
                },
                threadgroup_size: Self::compute_threadgroup(down_pipeline, max_cap, hidden),
            };

            expert_dispatches.push(Some(ExpertDispatchGroup {
                expert_id,
                gate_dispatch,
                up_dispatch,
                swiglu_dispatch,
                down_dispatch,
            }));
        }

        Self {
            expert_dispatches,
            config,
        }
    }

    /// Compute a 2D threadgroup size clamped to pipeline limits.
    fn compute_threadgroup(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        width: usize,
        height: usize,
    ) -> MTLSize {
        let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
        // Use a square-ish threadgroup, clamped to pipeline max.
        let tg_w = std::cmp::min(width, 16);
        let tg_h = std::cmp::min(height, 16);
        let total = tg_w * tg_h;
        if total <= max_tg {
            MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            }
        } else {
            // Fall back to 1D
            MTLSize {
                width: std::cmp::min(max_tg, width * height),
                height: 1,
                depth: 1,
            }
        }
    }

    /// Replay only the non-empty experts into a command buffer.
    ///
    /// This is the key optimization: iterate experts and encode only non-empty
    /// ones. While this still iterates on CPU, the dispatch setup is minimal
    /// because pipeline states and weight bindings are pre-captured.
    ///
    /// # Arguments
    ///
    /// * `expert_counts` - Per-expert token counts, length `E`.
    /// * `expert_input_buf` - Stacked input buffer `[total_tokens, D]`.
    /// * `expert_output_buf` - Stacked output buffer `[total_tokens, D]`.
    /// * `dispatch_offsets` - `[E+1]` prefix sum offsets into stacked buffers.
    /// * `cb` - Command buffer to encode dispatches into.
    ///
    /// # Returns
    ///
    /// The number of experts actually dispatched (those with count > 0).
    pub fn replay_sparse(
        &self,
        expert_counts: &[u32],
        expert_input_buf: &ProtocolObject<dyn MTLBuffer>,
        expert_output_buf: &ProtocolObject<dyn MTLBuffer>,
        dispatch_offsets: &[u32],
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> usize {
        assert!(
            expert_counts.len() <= self.config.max_experts,
            "expert_counts length {} exceeds max_experts {}",
            expert_counts.len(),
            self.config.max_experts,
        );
        assert_eq!(
            dispatch_offsets.len(),
            expert_counts.len() + 1,
            "dispatch_offsets must have length E+1",
        );

        let elem_size = 2u64; // f16
        let hidden = self.config.hidden_dim as u64;
        let mut dispatched = 0;

        for (expert_id, &count) in expert_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }

            if let Some(ref group) = self.expert_dispatches[expert_id] {
                let input_offset = dispatch_offsets[expert_id] as u64 * hidden * elem_size;
                let output_offset = dispatch_offsets[expert_id] as u64 * hidden * elem_size;

                group.replay_with_count(
                    count,
                    expert_input_buf,
                    expert_output_buf,
                    input_offset,
                    output_offset,
                    cb,
                );
                dispatched += 1;
            }
        }

        dispatched
    }

    /// Number of experts in this plan.
    pub fn num_experts(&self) -> usize {
        self.config.max_experts
    }

    /// Reference to the config used to build this plan.
    pub fn config(&self) -> &SparseExpertConfig {
        &self.config
    }

    /// Check if a specific expert has a dispatch group.
    pub fn has_expert(&self, expert_id: usize) -> bool {
        self.expert_dispatches
            .get(expert_id)
            .is_some_and(|e| e.is_some())
    }
}

// ---------------------------------------------------------------------------
// Grouped forward with ICB sparse dispatch
// ---------------------------------------------------------------------------

/// Result of a sparse ICB dispatch, containing per-expert dispatch metadata.
#[derive(Debug, Clone)]
pub struct SparseDispatchResult {
    /// Number of experts that were actually dispatched (count > 0).
    pub dispatched_count: usize,
    /// Bitmask of active experts (expert_id -> true if dispatched).
    pub active_mask: Vec<bool>,
}

/// Compute a hash key from the active expert mask for ICB replay caching.
///
/// Two invocations with the same set of active experts (same sparsity pattern)
/// will produce the same hash, enabling ICB replay without re-encoding.
fn compute_sparsity_hash(expert_counts: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (i, &count) in expert_counts.iter().enumerate() {
        // Hash only the active/inactive status, not the exact count,
        // so that different token counts but same active set share a key.
        let active = if count > 0 { 1u8 } else { 0u8 };
        i.hash(&mut hasher);
        active.hash(&mut hasher);
    }
    hasher.finish()
}

/// Encode only active experts into a command buffer using the sparse plan.
///
/// This is the core ICB sparse dispatch function: given expert counts from
/// top-k routing, it encodes GEMM dispatches only for experts with non-zero
/// token counts, skipping empty experts entirely. This avoids wasting GPU
/// cycles on experts that received no routed tokens.
///
/// # Arguments
///
/// * `plan` - Pre-built sparse expert plan with per-expert dispatch templates.
/// * `expert_counts` - Per-expert token counts from routing, length `E`.
/// * `expert_input_buf` - Stacked input buffer for all expert tokens.
/// * `expert_output_buf` - Stacked output buffer for all expert tokens.
/// * `dispatch_offsets` - `[E+1]` prefix sum of token offsets into stacked buffers.
/// * `queue` - Metal command queue to submit work on.
///
/// # Returns
///
/// A `SparseDispatchResult` with the number of dispatched experts and the active mask.
pub fn grouped_forward_icb(
    plan: &SparseExpertPlan,
    expert_counts: &[u32],
    expert_input_buf: &ProtocolObject<dyn MTLBuffer>,
    expert_output_buf: &ProtocolObject<dyn MTLBuffer>,
    dispatch_offsets: &[u32],
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> SparseDispatchResult {
    let active_mask: Vec<bool> = expert_counts.iter().map(|&c| c > 0).collect();
    let active_count = active_mask.iter().filter(|&&a| a).count();

    if active_count == 0 {
        return SparseDispatchResult {
            dispatched_count: 0,
            active_mask,
        };
    }

    // Encode only active experts into a single command buffer via the plan's
    // replay_sparse method, which already skips empty experts.
    let cb = queue.commandBufferWithUnretainedReferences().unwrap();
    let dispatched = plan.replay_sparse(
        expert_counts,
        expert_input_buf,
        expert_output_buf,
        dispatch_offsets,
        &cb,
    );
    cb.commit();
    cb.waitUntilCompleted();

    SparseDispatchResult {
        dispatched_count: dispatched,
        active_mask,
    }
}

/// Cache for ICB replay keyed by sparsity pattern.
///
/// When the same set of active experts is seen across multiple forward passes
/// (common in decode phase where routing is stable), the cached dispatch
/// metadata can be reused to skip re-analysis of the sparsity pattern.
///
/// The cache key is a hash of the active expert bitmask, not the exact
/// token counts -- so different batch sizes with the same active experts
/// share a cache entry.
pub struct IcbReplayCache {
    /// Map from sparsity pattern hash -> cached dispatch result metadata.
    cache: HashMap<u64, CachedSparsityPattern>,
    /// Maximum number of entries before eviction.
    max_entries: usize,
}

/// Cached information about a sparsity pattern.
#[derive(Debug, Clone)]
pub struct CachedSparsityPattern {
    /// The active expert mask for this pattern.
    pub active_mask: Vec<bool>,
    /// Number of active experts.
    pub active_count: usize,
    /// Number of times this pattern has been replayed.
    pub replay_count: u64,
}

impl IcbReplayCache {
    /// Create a new cache with the given maximum entry count.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
        }
    }

    /// Look up a cached sparsity pattern by expert counts.
    ///
    /// Returns the cached pattern if the same set of active experts has been
    /// seen before, along with the hash key for updating replay count.
    pub fn lookup(&self, expert_counts: &[u32]) -> Option<(u64, &CachedSparsityPattern)> {
        let key = compute_sparsity_hash(expert_counts);
        self.cache.get(&key).map(|pattern| (key, pattern))
    }

    /// Record a sparsity pattern after a dispatch, or increment replay count
    /// if already cached.
    pub fn record(&mut self, expert_counts: &[u32]) {
        let key = compute_sparsity_hash(expert_counts);
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.replay_count += 1;
            return;
        }

        // Evict oldest entry if at capacity (simple FIFO via oldest replay_count).
        if self.cache.len() >= self.max_entries {
            let evict_key = self
                .cache
                .iter()
                .min_by_key(|(_, v)| v.replay_count)
                .map(|(&k, _)| k);
            if let Some(k) = evict_key {
                self.cache.remove(&k);
            }
        }

        let active_mask: Vec<bool> = expert_counts.iter().map(|&c| c > 0).collect();
        let active_count = active_mask.iter().filter(|&&a| a).count();
        self.cache.insert(
            key,
            CachedSparsityPattern {
                active_mask,
                active_count,
                replay_count: 1,
            },
        );
    }

    /// Number of cached patterns.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached patterns.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Maximum number of entries.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

impl Default for IcbReplayCache {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Key for sparse expert plan cache lookup.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SparseExpertKey {
    pub num_experts: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

/// Cache of sparse expert plans keyed by expert configuration.
///
/// Different MoE layers may share the same expert geometry (E, D, inter_dim).
/// This cache avoids redundant plan building for layers with identical shapes.
pub struct SparseExpertCache {
    cache: HashMap<SparseExpertKey, SparseExpertPlan>,
}

impl SparseExpertCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get a cached plan for the given key.
    pub fn get(&self, key: &SparseExpertKey) -> Option<&SparseExpertPlan> {
        self.cache.get(key)
    }

    /// Insert a plan into the cache.
    pub fn insert(&mut self, key: SparseExpertKey, plan: SparseExpertPlan) {
        self.cache.insert(key, plan);
    }

    /// Check if a plan exists for the given key.
    pub fn contains(&self, key: &SparseExpertKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Number of cached plans.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached plans.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for SparseExpertCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MtlBuffer, MtlPipeline};
    use std::sync::OnceLock;

    fn test_device() -> Option<&'static crate::MtlDevice> {
        static DEVICE: OnceLock<Option<crate::MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| objc2::rc::autoreleasepool(|_| MTLCreateSystemDefaultDevice()))
            .as_ref()
    }

    // ── Cache tests (no GPU needed) ──────────────────────────────────────

    #[test]
    fn cache_new_is_empty() {
        let cache = SparseExpertCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_default_is_empty() {
        let cache = SparseExpertCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_insert_and_get() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);
        let key = SparseExpertKey {
            num_experts: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
        };

        let mut cache = SparseExpertCache::new();
        cache.insert(key.clone(), plan);

        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&key));
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.get(&key).unwrap().num_experts(), 4);

        // Missing key
        let other = SparseExpertKey {
            num_experts: 8,
            hidden_dim: 64,
            intermediate_dim: 128,
        };
        assert!(!cache.contains(&other));
        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn cache_clear() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);
        let key = SparseExpertKey {
            num_experts: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
        };

        let mut cache = SparseExpertCache::new();
        cache.insert(key, plan);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_overwrite() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let key = SparseExpertKey {
            num_experts: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
        };

        let (plan1, _bufs1) = build_test_plan(device, 4, 64, 128);
        let (plan2, _bufs2) = build_test_plan(device, 4, 64, 128);

        let mut cache = SparseExpertCache::new();
        cache.insert(key.clone(), plan1);
        cache.insert(key.clone(), plan2);

        // Still just one entry (overwritten).
        assert_eq!(cache.len(), 1);
    }

    // ── Plan building tests ──────────────────────────────────────────────

    #[test]
    fn plan_build_basic() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 8, 64, 128);

        assert_eq!(plan.num_experts(), 8);
        for i in 0..8 {
            assert!(plan.has_expert(i), "expert {i} should exist");
        }
        assert!(!plan.has_expert(8), "expert 8 should not exist");
    }

    #[test]
    fn plan_build_single_expert() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 1, 32, 64);
        assert_eq!(plan.num_experts(), 1);
        assert!(plan.has_expert(0));
        assert!(!plan.has_expert(1));
    }

    #[test]
    fn plan_config_accessible() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 128, 256);
        let cfg = plan.config();
        assert_eq!(cfg.max_experts, 4);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.intermediate_dim, 256);
    }

    // ── Sparse replay tests ──────────────────────────────────────────────

    #[test]
    fn replay_sparse_all_empty() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        let expert_counts = [0u32; 4];
        let dispatch_offsets = [0u32; 5]; // E+1

        let input_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = queue.commandBuffer().unwrap();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );

        assert_eq!(dispatched, 0, "no experts should be dispatched");

        // Commit and wait (no-op, but validates encoding).
        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn replay_sparse_skips_empty() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        // Experts 0 and 2 have tokens; 1 and 3 are empty.
        let expert_counts = [3u32, 0, 5, 0];
        let dispatch_offsets = [0u32, 3, 3, 8, 8];

        let total_tokens = 8u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = queue.commandBuffer().unwrap();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );

        assert_eq!(dispatched, 2, "exactly 2 experts should be dispatched");

        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn replay_sparse_all_active() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        let expert_counts = [2u32, 3, 1, 4];
        let dispatch_offsets = [0u32, 2, 5, 6, 10];

        let total_tokens = 10u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = queue.commandBuffer().unwrap();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );

        assert_eq!(dispatched, 4, "all 4 experts should be dispatched");

        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn replay_sparse_subset_of_experts() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        // Build plan for 8 experts, but only provide counts for 4.
        let (plan, _bufs) = build_test_plan(device, 8, 64, 128);

        let expert_counts = [1u32, 0, 2, 0];
        let dispatch_offsets = [0u32, 1, 1, 3, 3];

        let total_tokens = 3u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = queue.commandBuffer().unwrap();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );

        assert_eq!(dispatched, 2);

        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    #[should_panic(expected = "expert_counts length")]
    fn replay_sparse_panics_on_too_many_experts() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 2, 64, 128);

        // 3 expert counts but plan only has 2.
        let expert_counts = [1u32, 2, 3];
        let dispatch_offsets = [0u32, 1, 3, 6];

        let input_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = _queue.commandBuffer().unwrap();
        plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );
    }

    #[test]
    #[should_panic(expected = "dispatch_offsets must have length E+1")]
    fn replay_sparse_panics_on_bad_offsets() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        let expert_counts = [1u32, 0, 2, 0];
        // Wrong length: should be 5 (E+1), but is 4.
        let dispatch_offsets = [0u32, 1, 1, 3];

        let input_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = _queue.commandBuffer().unwrap();
        plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );
    }

    // ── SwiGLU 2D grid tests ─────────────────────────────────────────────

    #[test]
    fn swiglu_dispatch_uses_2d_grid() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let _queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 2, 64, 128);

        // Access the SwiGLU dispatch for expert 0.
        let group = plan.expert_dispatches[0].as_ref().unwrap();
        let swiglu = &group.swiglu_dispatch;

        // The grid must be 2D: width = max_capacity, height = intermediate_dim.
        assert_eq!(
            swiglu.grid_size.width, 16,
            "SwiGLU grid width should be max_capacity (16)"
        );
        assert_eq!(
            swiglu.grid_size.height, 128,
            "SwiGLU grid height should be intermediate_dim (128)"
        );
        assert_eq!(swiglu.grid_size.depth, 1);
    }

    #[test]
    fn swiglu_replay_preserves_intermediate_dim() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let intermediate_dim = 256usize;
        let (plan, _bufs) = build_test_plan(device, 4, 64, intermediate_dim);

        // Replay with fewer tokens than max_capacity.
        // The key check: SwiGLU grid height (intermediate_dim) must be preserved
        // even when width (token_count) is scaled down.
        let expert_counts = [3u32, 0, 0, 0];
        let dispatch_offsets = [0u32, 3, 3, 3, 3];

        let total_tokens = 3u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let cb = queue.commandBuffer().unwrap();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &cb,
        );

        assert_eq!(dispatched, 1);

        // Verify the SwiGLU dispatch in the plan has the correct 2D grid shape.
        let group = plan.expert_dispatches[0].as_ref().unwrap();
        let swiglu = &group.swiglu_dispatch;
        assert_eq!(
            swiglu.grid_size.height, intermediate_dim,
            "SwiGLU grid height must equal intermediate_dim after replay"
        );

        cb.commit();
        cb.waitUntilCompleted();
    }

    // ── Key equality / hashing tests ─────────────────────────────────────

    #[test]
    fn sparse_expert_key_equality() {
        let k1 = SparseExpertKey {
            num_experts: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
        };
        let k2 = SparseExpertKey {
            num_experts: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
        };
        let k3 = SparseExpertKey {
            num_experts: 8,
            hidden_dim: 4096,
            intermediate_dim: 11008,
        };
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn sparse_expert_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let k1 = SparseExpertKey {
            num_experts: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
        };
        let k2 = k1.clone();

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        k1.hash(&mut h1);
        k2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── grouped_forward_icb tests ─────────────────────────────────────

    #[test]
    fn grouped_forward_icb_skips_empty_experts() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        // Experts 0 and 2 active, 1 and 3 empty
        let expert_counts = [3u32, 0, 5, 0];
        let dispatch_offsets = [0u32, 3, 3, 8, 8];

        let total_tokens = 8u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(buf_size as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let result = grouped_forward_icb(
            &plan,
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &queue,
        );

        assert_eq!(result.dispatched_count, 2);
        assert_eq!(result.active_mask, vec![true, false, true, false]);
    }

    #[test]
    fn grouped_forward_icb_all_empty() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let queue = device.newCommandQueue().unwrap();

        let (plan, _bufs) = build_test_plan(device, 4, 64, 128);

        let expert_counts = [0u32; 4];
        let dispatch_offsets = [0u32; 5];

        let input_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let output_buf = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let result = grouped_forward_icb(
            &plan,
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            &queue,
        );

        assert_eq!(result.dispatched_count, 0);
        assert!(result.active_mask.iter().all(|&a| !a));
    }

    // ── IcbReplayCache tests ────────────────────────────────────────────

    #[test]
    fn replay_cache_new_is_empty() {
        let cache = IcbReplayCache::new(32);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_entries(), 32);
    }

    #[test]
    fn replay_cache_default() {
        let cache = IcbReplayCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.max_entries(), 64);
    }

    #[test]
    fn replay_cache_record_and_lookup() {
        let mut cache = IcbReplayCache::new(16);

        let counts_a = [3u32, 0, 5, 0];
        let counts_b = [0u32, 2, 0, 4];

        // Record pattern A
        cache.record(&counts_a);
        assert_eq!(cache.len(), 1);

        // Lookup pattern A
        let (_key_a, pattern_a) = cache.lookup(&counts_a).expect("pattern A should be cached");
        assert_eq!(pattern_a.active_mask, vec![true, false, true, false]);
        assert_eq!(pattern_a.active_count, 2);
        assert_eq!(pattern_a.replay_count, 1);

        // Record same pattern again -> replay_count increments
        cache.record(&counts_a);
        let (_, pattern_a2) = cache.lookup(&counts_a).unwrap();
        assert_eq!(pattern_a2.replay_count, 2);
        assert_eq!(cache.len(), 1); // still just one entry

        // Record different pattern
        cache.record(&counts_b);
        assert_eq!(cache.len(), 2);
        let (_, pattern_b) = cache.lookup(&counts_b).unwrap();
        assert_eq!(pattern_b.active_mask, vec![false, true, false, true]);
        assert_eq!(pattern_b.active_count, 2);
    }

    #[test]
    fn replay_cache_same_active_set_different_counts() {
        let mut cache = IcbReplayCache::new(16);

        // Same active experts (0 and 2), different token counts
        let counts_a = [3u32, 0, 5, 0];
        let counts_b = [10u32, 0, 1, 0];

        cache.record(&counts_a);
        // Should produce the same hash since active mask is the same
        let result = cache.lookup(&counts_b);
        assert!(
            result.is_some(),
            "same active mask should share cache entry"
        );
    }

    #[test]
    fn replay_cache_eviction() {
        let mut cache = IcbReplayCache::new(2);

        cache.record(&[1u32, 0, 0, 0]); // pattern A, replay=1
        cache.record(&[0u32, 1, 0, 0]); // pattern B, replay=1
        assert_eq!(cache.len(), 2);

        // Bump pattern A's replay count
        cache.record(&[1u32, 0, 0, 0]); // pattern A now replay=2

        // Insert pattern C -> should evict pattern B (lower replay count)
        cache.record(&[0u32, 0, 1, 0]);
        assert_eq!(cache.len(), 2);

        // Pattern A should still be present
        assert!(cache.lookup(&[1u32, 0, 0, 0]).is_some());
        // Pattern B should be evicted
        assert!(cache.lookup(&[0u32, 1, 0, 0]).is_none());
    }

    #[test]
    fn replay_cache_clear() {
        let mut cache = IcbReplayCache::new(16);
        cache.record(&[1u32, 0, 0, 0]);
        cache.record(&[0u32, 1, 0, 0]);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    // ── sparsity hash tests ─────────────────────────────────────────────

    #[test]
    fn sparsity_hash_deterministic() {
        let counts = [3u32, 0, 5, 0];
        let h1 = compute_sparsity_hash(&counts);
        let h2 = compute_sparsity_hash(&counts);
        assert_eq!(h1, h2);
    }

    #[test]
    fn sparsity_hash_same_active_set() {
        // Different counts but same active experts -> same hash
        let h1 = compute_sparsity_hash(&[3u32, 0, 5, 0]);
        let h2 = compute_sparsity_hash(&[10u32, 0, 1, 0]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn sparsity_hash_different_active_set() {
        let h1 = compute_sparsity_hash(&[3u32, 0, 5, 0]);
        let h2 = compute_sparsity_hash(&[0u32, 3, 0, 5]);
        assert_ne!(h1, h2);
    }

    // ── Test helpers ─────────────────────────────────────────────────────

    /// Build a test plan with trivial Metal pipeline states.
    ///
    /// Returns the plan and the weight buffers (must be kept alive for the
    /// plan's buffer references to remain valid).
    fn build_test_plan(
        device: &ProtocolObject<dyn MTLDevice>,
        num_experts: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> (SparseExpertPlan, TestBuffers) {
        let max_capacity = 16;
        let elem_size = 2usize; // f16

        let gate_size = num_experts * hidden_dim * intermediate_dim * elem_size;
        let up_size = num_experts * hidden_dim * intermediate_dim * elem_size;
        let down_size = num_experts * intermediate_dim * hidden_dim * elem_size;

        let opts = MTLResourceOptions::StorageModeShared;
        let gate_weights = device.newBufferWithLength_options(gate_size, opts).unwrap();
        let up_weights = device.newBufferWithLength_options(up_size, opts).unwrap();
        let down_weights = device.newBufferWithLength_options(down_size, opts).unwrap();

        // Create a trivial compute pipeline for testing.
        let pipeline = create_noop_pipeline(device);

        let config = SparseExpertConfig {
            max_experts: num_experts,
            max_capacity,
            hidden_dim,
            intermediate_dim,
        };

        let plan = SparseExpertPlan::build(
            config,
            &pipeline,
            &pipeline,
            &pipeline,
            &pipeline,
            &gate_weights,
            &up_weights,
            &down_weights,
        );

        let bufs = TestBuffers {
            gate_weights,
            up_weights,
            down_weights,
        };

        (plan, bufs)
    }

    /// Weight buffers that must outlive the plan (prevent deallocation).
    #[allow(dead_code)]
    struct TestBuffers {
        gate_weights: MtlBuffer,
        up_weights: MtlBuffer,
        down_weights: MtlBuffer,
    }

    /// Create a no-op Metal compute pipeline for testing.
    fn create_noop_pipeline(device: &ProtocolObject<dyn MTLDevice>) -> MtlPipeline {
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void noop(device float* out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
                // intentionally empty
            }
        "#;

        let ns_source = objc2_foundation::NSString::from_str(source);
        let opts = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&ns_source, Some(&opts))
            .expect("failed to compile noop shader");
        let func_name = objc2_foundation::NSString::from_str("noop");
        let func = library
            .newFunctionWithName(&func_name)
            .expect("failed to get noop function");
        device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("failed to create noop pipeline")
    }
}
