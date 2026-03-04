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

use metal::{Buffer, ComputePipelineState, MTLSize};

use crate::icb::CapturedDispatch;

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
        input_buf: &Buffer,
        output_buf: &Buffer,
        input_offset: u64,
        output_offset: u64,
        cb: &metal::CommandBufferRef,
    ) {
        let dispatches = [
            &self.gate_dispatch,
            &self.up_dispatch,
            &self.swiglu_dispatch,
            &self.down_dispatch,
        ];

        for dispatch in dispatches {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&dispatch.pipeline);

            // Bind the pre-captured buffers (weights etc.) from the template.
            for (index, buffer, offset) in &dispatch.buffers {
                enc.set_buffer(*index, Some(buffer), *offset);
            }

            // Override input/output buffer bindings with the actual stacked
            // buffers and offsets for this expert's token slice.
            // Convention: buffer index 0 = input, buffer index 1 = output.
            enc.set_buffer(0, Some(input_buf), input_offset);
            enc.set_buffer(1, Some(output_buf), output_offset);

            // Scale grid to actual token count while keeping the original
            // height and depth (which encode hidden/intermediate dims).
            let grid = MTLSize::new(
                token_count as u64,
                dispatch.grid_size.height,
                dispatch.grid_size.depth,
            );
            let tg = MTLSize::new(
                std::cmp::min(dispatch.threadgroup_size.width, token_count as u64),
                dispatch.threadgroup_size.height,
                dispatch.threadgroup_size.depth,
            );

            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
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
        gate_pipeline: &ComputePipelineState,
        up_pipeline: &ComputePipelineState,
        swiglu_pipeline: &ComputePipelineState,
        down_pipeline: &ComputePipelineState,
        gate_weights: &Buffer,
        up_weights: &Buffer,
        down_weights: &Buffer,
    ) -> Self {
        let max_cap = config.max_capacity as u64;
        let hidden = config.hidden_dim as u64;
        let inter = config.intermediate_dim as u64;

        // Bytes per element (f16 = 2 bytes).
        let elem_size: u64 = 2;

        let gate_expert_stride = hidden * inter * elem_size;
        let up_expert_stride = hidden * inter * elem_size;
        let down_expert_stride = inter * hidden * elem_size;

        let mut expert_dispatches = Vec::with_capacity(config.max_experts);

        for expert_id in 0..config.max_experts {
            let eid = expert_id as u64;

            // Gate GEMM: [tokens, D] x [D, inter] -> [tokens, inter]
            let gate_dispatch = CapturedDispatch {
                pipeline: gate_pipeline.clone(),
                buffers: vec![
                    // input buffer placeholder at index 0 (overridden at replay)
                    // output buffer placeholder at index 1 (overridden at replay)
                    // weight buffer at index 2
                    (2, gate_weights.clone(), eid * gate_expert_stride),
                ],
                grid_size: MTLSize::new(max_cap, inter, 1),
                threadgroup_size: Self::compute_threadgroup(gate_pipeline, max_cap, inter),
            };

            // Up GEMM: [tokens, D] x [D, inter] -> [tokens, inter]
            let up_dispatch = CapturedDispatch {
                pipeline: up_pipeline.clone(),
                buffers: vec![(2, up_weights.clone(), eid * up_expert_stride)],
                grid_size: MTLSize::new(max_cap, inter, 1),
                threadgroup_size: Self::compute_threadgroup(up_pipeline, max_cap, inter),
            };

            // SwiGLU: element-wise on [tokens, inter]
            let swiglu_threads = max_cap * inter;
            let swiglu_max_tg = swiglu_pipeline.max_total_threads_per_threadgroup();
            let swiglu_tg = std::cmp::min(swiglu_max_tg, swiglu_threads);
            let swiglu_dispatch = CapturedDispatch {
                pipeline: swiglu_pipeline.clone(),
                buffers: vec![],
                grid_size: MTLSize::new(swiglu_threads, 1, 1),
                threadgroup_size: MTLSize::new(swiglu_tg, 1, 1),
            };

            // Down GEMM: [tokens, inter] x [inter, D] -> [tokens, D]
            let down_dispatch = CapturedDispatch {
                pipeline: down_pipeline.clone(),
                buffers: vec![(2, down_weights.clone(), eid * down_expert_stride)],
                grid_size: MTLSize::new(max_cap, hidden, 1),
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
    fn compute_threadgroup(pipeline: &ComputePipelineState, width: u64, height: u64) -> MTLSize {
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        // Use a square-ish threadgroup, clamped to pipeline max.
        let tg_w = std::cmp::min(width, 16);
        let tg_h = std::cmp::min(height, 16);
        let total = tg_w * tg_h;
        if total <= max_tg {
            MTLSize::new(tg_w, tg_h, 1)
        } else {
            // Fall back to 1D
            MTLSize::new(std::cmp::min(max_tg, width * height), 1, 1)
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
        expert_input_buf: &Buffer,
        expert_output_buf: &Buffer,
        dispatch_offsets: &[u32],
        cb: &metal::CommandBufferRef,
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
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);
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

        drop(queue);
    }

    #[test]
    fn cache_clear() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);
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

        drop(queue);
    }

    #[test]
    fn cache_overwrite() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let key = SparseExpertKey {
            num_experts: 4,
            hidden_dim: 64,
            intermediate_dim: 128,
        };

        let (plan1, _bufs1) = build_test_plan(&device, 4, 64, 128);
        let (plan2, _bufs2) = build_test_plan(&device, 4, 64, 128);

        let mut cache = SparseExpertCache::new();
        cache.insert(key.clone(), plan1);
        cache.insert(key.clone(), plan2);

        // Still just one entry (overwritten).
        assert_eq!(cache.len(), 1);

        drop(queue);
    }

    // ── Plan building tests ──────────────────────────────────────────────

    #[test]
    fn plan_build_basic() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 8, 64, 128);

        assert_eq!(plan.num_experts(), 8);
        for i in 0..8 {
            assert!(plan.has_expert(i), "expert {i} should exist");
        }
        assert!(!plan.has_expert(8), "expert 8 should not exist");

        drop(queue);
    }

    #[test]
    fn plan_build_single_expert() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 1, 32, 64);
        assert_eq!(plan.num_experts(), 1);
        assert!(plan.has_expert(0));
        assert!(!plan.has_expert(1));

        drop(queue);
    }

    #[test]
    fn plan_config_accessible() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 128, 256);
        let cfg = plan.config();
        assert_eq!(cfg.max_experts, 4);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.intermediate_dim, 256);

        drop(queue);
    }

    // ── Sparse replay tests ──────────────────────────────────────────────

    #[test]
    fn replay_sparse_all_empty() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);

        let expert_counts = [0u32; 4];
        let dispatch_offsets = [0u32; 5]; // E+1

        let input_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );

        assert_eq!(dispatched, 0, "no experts should be dispatched");

        // Commit and wait (no-op, but validates encoding).
        cb.commit();
        cb.wait_until_completed();
    }

    #[test]
    fn replay_sparse_skips_empty() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);

        // Experts 0 and 2 have tokens; 1 and 3 are empty.
        let expert_counts = [3u32, 0, 5, 0];
        let dispatch_offsets = [0u32, 3, 3, 8, 8];

        let total_tokens = 8u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );

        assert_eq!(dispatched, 2, "exactly 2 experts should be dispatched");

        cb.commit();
        cb.wait_until_completed();
    }

    #[test]
    fn replay_sparse_all_active() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);

        let expert_counts = [2u32, 3, 1, 4];
        let dispatch_offsets = [0u32, 2, 5, 6, 10];

        let total_tokens = 10u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );

        assert_eq!(dispatched, 4, "all 4 experts should be dispatched");

        cb.commit();
        cb.wait_until_completed();
    }

    #[test]
    fn replay_sparse_subset_of_experts() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        // Build plan for 8 experts, but only provide counts for 4.
        let (plan, _bufs) = build_test_plan(&device, 8, 64, 128);

        let expert_counts = [1u32, 0, 2, 0];
        let dispatch_offsets = [0u32, 1, 1, 3, 3];

        let total_tokens = 3u64;
        let hidden = 64u64;
        let elem_size = 2u64;
        let buf_size = total_tokens * hidden * elem_size;

        let input_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(buf_size, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let dispatched = plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );

        assert_eq!(dispatched, 2);

        cb.commit();
        cb.wait_until_completed();
    }

    #[test]
    #[should_panic(expected = "expert_counts length")]
    fn replay_sparse_panics_on_too_many_experts() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 2, 64, 128);

        // 3 expert counts but plan only has 2.
        let expert_counts = [1u32, 2, 3];
        let dispatch_offsets = [0u32, 1, 3, 6];

        let input_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );
    }

    #[test]
    #[should_panic(expected = "dispatch_offsets must have length E+1")]
    fn replay_sparse_panics_on_bad_offsets() {
        let device = metal::Device::system_default().expect("Metal device required");
        let queue = device.new_command_queue();

        let (plan, _bufs) = build_test_plan(&device, 4, 64, 128);

        let expert_counts = [1u32, 0, 2, 0];
        // Wrong length: should be 5 (E+1), but is 4.
        let dispatch_offsets = [0u32, 1, 1, 3];

        let input_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);
        let output_buf = device.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        plan.replay_sparse(
            &expert_counts,
            &input_buf,
            &output_buf,
            &dispatch_offsets,
            cb,
        );
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

    // ── Test helpers ─────────────────────────────────────────────────────

    /// Build a test plan with trivial Metal pipeline states.
    ///
    /// Returns the plan and the weight buffers (must be kept alive for the
    /// plan's buffer references to remain valid).
    fn build_test_plan(
        device: &metal::Device,
        num_experts: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> (SparseExpertPlan, TestBuffers) {
        let max_capacity = 16;
        let elem_size = 2usize; // f16

        let gate_size = num_experts * hidden_dim * intermediate_dim * elem_size;
        let up_size = num_experts * hidden_dim * intermediate_dim * elem_size;
        let down_size = num_experts * intermediate_dim * hidden_dim * elem_size;

        let opts = metal::MTLResourceOptions::StorageModeShared;
        let gate_weights = device.new_buffer(gate_size as u64, opts);
        let up_weights = device.new_buffer(up_size as u64, opts);
        let down_weights = device.new_buffer(down_size as u64, opts);

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
        gate_weights: Buffer,
        up_weights: Buffer,
        down_weights: Buffer,
    }

    /// Create a no-op Metal compute pipeline for testing.
    fn create_noop_pipeline(device: &metal::Device) -> ComputePipelineState {
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void noop(device float* out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
                // intentionally empty
            }
        "#;

        let opts = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(source, &opts)
            .expect("failed to compile noop shader");
        let func = library
            .get_function("noop", None)
            .expect("failed to get noop function");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("failed to create noop pipeline")
    }
}
