//! GPU-native top-k routing for Mixture-of-Experts (MoE).
//!
//! Eliminates CPU round-trips by performing the full routing pipeline on the GPU:
//! softmax -> top-k selection -> weight normalization -> expert histogram -> prefix scan.
//!
//! Two Metal kernels:
//! - `moe_topk_route_f32`: One-pass fused kernel. Each threadgroup handles one token row:
//!   online softmax of gate logits, top-k selection (K<=8), weight normalization,
//!   and atomic histogram accumulation.
//! - `moe_prefix_scan_u32`: Exclusive prefix sum on expert_counts to produce dispatch
//!   offsets. Single threadgroup, simple scan for E<=256.
//!
//! The output `TopkRouteResult` contains everything needed to dispatch tokens to experts
//! without any GPU->CPU synchronization.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use crate::ops;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLResourceOptions;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

pub const TOPK_ROUTE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>
using namespace metal;

constant constexpr uint SIMD_SIZE = 32;

// ---------------------------------------------------------------------------
// Cross-simdgroup reduction helpers (same pattern as softmax.rs)
// ---------------------------------------------------------------------------

/// Reduce max and normalizer across all simdgroups in the threadgroup.
/// After this call every thread has the same global max_val, and
/// normalizer contains 1.0 / sum(exp(x - max)).
inline void cross_simdgroup_reduce_max_sum(
    thread float& max_val,
    thread float& normalizer,
    threadgroup float* local_max,
    threadgroup float* local_normalizer,
    uint simd_lane_id,
    uint simd_group_id)
{
    // SIMD-level reduction within one simdgroup
    float simd_max_val   = simd_max(max_val);
    // Guard: when all lanes have -INFINITY (empty simdgroup), exp(NaN) would poison the sum.
    if (simd_max_val > -INFINITY) {
        normalizer *= fast::exp(max_val - simd_max_val);
    } else {
        normalizer = 0.0f;
    }
    max_val              = simd_max_val;
    float simd_norm      = simd_sum(normalizer);

    // Cross-simdgroup reduction via shared memory
    if (simd_group_id == 0) {
        local_max[simd_lane_id]        = -INFINITY;
        local_normalizer[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        local_max[simd_group_id]        = max_val;
        local_normalizer[simd_group_id] = simd_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup does the final reduction
    if (simd_group_id == 0) {
        float m = local_max[simd_lane_id];
        float n = local_normalizer[simd_lane_id];
        float final_max = simd_max(m);
        // Guard: same NaN protection for cross-simdgroup level
        if (final_max > -INFINITY) {
            n *= fast::exp(m - final_max);
        } else {
            n = 0.0f;
        }
        float final_norm = simd_sum(n);
        if (simd_lane_id == 0) {
            local_max[0]        = final_max;
            local_normalizer[0] = 1.0f / final_norm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_val    = local_max[0];
    normalizer = local_normalizer[0];  // now holds 1/sum
}

// ---------------------------------------------------------------------------
// Cross-simdgroup sum reduction (for normalizing top-k weights)
// ---------------------------------------------------------------------------

inline float cross_simdgroup_sum(
    float val,
    threadgroup float* scratch,
    uint simd_lane_id,
    uint simd_group_id)
{
    float simd_val = simd_sum(val);

    if (simd_group_id == 0) {
        scratch[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        scratch[simd_group_id] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float n = scratch[simd_lane_id];
        float total = simd_sum(n);
        if (simd_lane_id == 0) {
            scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0];
}

// ---------------------------------------------------------------------------
// Cross-simdgroup argmax: returns (max_value, max_index) across threadgroup
// ---------------------------------------------------------------------------

inline void cross_simdgroup_argmax(
    thread float& val,
    thread uint& idx,
    threadgroup float* shared_val,
    threadgroup uint* shared_idx,
    uint simd_lane_id,
    uint simd_group_id)
{
    // SIMD-level argmax
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(val, offset);
        uint  other_idx = simd_shuffle_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }

    // Cross-simdgroup via shared memory
    if (simd_group_id == 0) {
        shared_val[simd_lane_id] = -INFINITY;
        shared_idx[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        shared_val[simd_group_id] = val;
        shared_idx[simd_group_id] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces across simdgroups
    if (simd_group_id == 0) {
        val = shared_val[simd_lane_id];
        idx = shared_idx[simd_lane_id];
        for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(val, offset);
            uint  other_idx = simd_shuffle_down(idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (simd_lane_id == 0) {
            shared_val[0] = val;
            shared_idx[0] = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    val = shared_val[0];
    idx = shared_idx[0];
}

// ===========================================================================
// moe_topk_route_f32
//
// One-pass fused kernel: softmax -> top-k -> normalize -> histogram.
// One threadgroup per token row. Dispatch [seq_len] threadgroups.
//
// Buffer layout:
//   [0] gate_logits    : device const float*  [N, E]
//   [1] expert_bias    : device const float*  [E]  (may be nullptr)
//   [2] expert_indices : device uint*         [N * K]
//   [3] expert_weights : device float*        [N * K]
//   [4] expert_counts  : device atomic_uint*  [E]
//   [5] num_experts    : constant uint&
//   [6] top_k          : constant uint&
//   [7] has_bias       : constant uint&       (1 if bias present, 0 otherwise)
// ===========================================================================

kernel void moe_topk_route_f32(
    device const float*       gate_logits    [[buffer(0)]],
    device const float*       expert_bias    [[buffer(1)]],
    device       uint*        expert_indices [[buffer(2)]],
    device       float*       expert_weights [[buffer(3)]],
    device       atomic_uint* expert_counts  [[buffer(4)]],
    constant     uint&        num_experts    [[buffer(5)]],
    constant     uint&        top_k          [[buffer(6)]],
    constant     uint&        has_bias       [[buffer(7)]],
    uint row              [[threadgroup_position_in_grid]],
    uint tid              [[thread_position_in_threadgroup]],
    uint tgsize           [[threads_per_threadgroup]],
    uint simd_lane_id     [[thread_index_in_simdgroup]],
    uint simd_group_id    [[simdgroup_index_in_threadgroup]])
{
    // Shared memory for cross-simdgroup reductions
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];
    threadgroup uint  local_idx[SIMD_SIZE];

    // Shared memory for broadcasting top-k results from thread 0
    // Max K=8
    threadgroup float topk_vals[8];
    threadgroup uint  topk_idxs[8];

    const uint E = num_experts;
    const uint K = top_k;
    size_t base = size_t(row) * size_t(E);

    // -----------------------------------------------------------------------
    // Step 1: Online softmax — compute max and normalizer in a single pass.
    // Each thread handles a strided subset of experts.
    // If bias is present, add bias[e] to logits before softmax.
    // -----------------------------------------------------------------------
    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid; i < E; i += tgsize) {
        float val = gate_logits[base + i];
        if (has_bias != 0) {
            val += expert_bias[i];
        }
        float prev_max = max_val;
        max_val        = max(max_val, val);
        normalizer     = normalizer * fast::exp(prev_max - max_val)
                       + fast::exp(val - max_val);
    }

    // Cross-simdgroup reduction: after this, max_val = row max,
    // normalizer = 1.0 / sum(exp(x - max)).
    cross_simdgroup_reduce_max_sum(
        max_val, normalizer,
        local_max, local_normalizer,
        simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;  // 1.0 / sum

    // -----------------------------------------------------------------------
    // Step 2: Top-k selection via repeated argmax.
    // K is small (<=8), so we do K passes. Each pass finds the global argmax
    // of softmax values, excluding previously selected indices.
    // We mark selected experts by setting their softmax value to -1.0.
    //
    // To avoid re-reading gate_logits, we need softmax values. But storing
    // all E values in threadgroup memory may be too large. Instead, each
    // thread recomputes softmax(e) = exp(logit[e] - row_max) * inv_norm
    // on the fly. This is 2 passes over global memory per top-k iteration,
    // but K<=8 and E is small (8-256), so total traffic is bounded.
    // -----------------------------------------------------------------------

    // Mask array: use a simple approach where we broadcast the selected index
    // and each thread skips it in subsequent passes.
    // We store selected indices in topk_idxs shared memory.

    for (uint k = 0; k < K; k++) {
        // Each thread computes its local argmax over its strided slice,
        // excluding previously selected experts.
        float best_val = -1.0f;
        uint  best_idx = 0;

        for (uint i = tid; i < E; i += tgsize) {
            float logit = gate_logits[base + i];
            if (has_bias != 0) {
                logit += expert_bias[i];
            }
            float softmax_val = fast::exp(logit - row_max) * inv_norm;

            // Skip previously selected experts
            bool already_selected = false;
            for (uint prev = 0; prev < k; prev++) {
                if (topk_idxs[prev] == i) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && softmax_val > best_val) {
                best_val = softmax_val;
                best_idx = i;
            }
        }

        // Cross-simdgroup argmax to find global best
        cross_simdgroup_argmax(
            best_val, best_idx,
            local_normalizer, local_idx,
            simd_lane_id, simd_group_id);

        // Thread 0 writes the k-th top expert
        if (tid == 0) {
            topk_vals[k] = best_val;
            topk_idxs[k] = best_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -----------------------------------------------------------------------
    // Step 3: Normalize top-k weights so they sum to 1.0.
    // Only thread 0 does this since K<=8.
    // -----------------------------------------------------------------------
    if (tid == 0) {
        float weight_sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            weight_sum += topk_vals[k];
        }
        float inv_weight_sum = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
        for (uint k = 0; k < K; k++) {
            topk_vals[k] *= inv_weight_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Step 4: Write outputs and atomically increment expert histogram.
    // Thread 0 writes all K entries (K<=8, not worth parallelizing).
    // -----------------------------------------------------------------------
    if (tid == 0) {
        size_t out_base = size_t(row) * size_t(K);
        for (uint k = 0; k < K; k++) {
            expert_indices[out_base + k] = topk_idxs[k];
            expert_weights[out_base + k] = topk_vals[k];
            atomic_fetch_add_explicit(
                &expert_counts[topk_idxs[k]],
                1u,
                memory_order_relaxed);
        }
    }
}

// ===========================================================================
// moe_prefix_scan_u32
//
// Exclusive prefix sum on expert_counts[E] -> dispatch_offsets[E+1].
// Single threadgroup, simple sequential scan. E<=256 so this is trivial.
//
// Buffer layout:
//   [0] expert_counts   : device const uint*  [E]
//   [1] dispatch_offsets : device uint*        [E+1]
//   [2] num_experts     : constant uint&
// ===========================================================================

kernel void moe_prefix_scan_u32(
    device const uint* expert_counts   [[buffer(0)]],
    device       uint* dispatch_offsets [[buffer(1)]],
    constant     uint& num_experts     [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    // Only thread 0 does the work. E<=256 so sequential is fine.
    if (tid != 0) return;

    uint running_sum = 0;
    for (uint e = 0; e < num_experts; e++) {
        dispatch_offsets[e] = running_sum;
        running_sum += expert_counts[e];
    }
    dispatch_offsets[num_experts] = running_sum;
}
"#;

// ---------------------------------------------------------------------------
// Rust API
// ---------------------------------------------------------------------------

/// Result of GPU top-k routing, containing everything needed for expert dispatch.
pub struct TopkRouteResult {
    /// Selected expert indices per token, flat `[N*K]`, UInt32.
    pub expert_indices: Array,
    /// Normalized routing weights per token, flat `[N*K]`, Float32.
    pub expert_weights: Array,
    /// Per-expert token count histogram, `[E]`, UInt32.
    pub expert_counts: Array,
    /// Exclusive prefix sum of expert_counts, `[E+1]`, UInt32.
    /// `dispatch_offsets[e]` is the starting offset for expert `e` in a
    /// compacted token buffer; `dispatch_offsets[E]` is the total count.
    pub dispatch_offsets: Array,
}

/// Register the top-k routing kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("topk_route", TOPK_ROUTE_SHADER_SOURCE)
}

/// Create a u32 Metal constant buffer.
fn make_u32_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: u32,
) -> rmlx_metal::MtlBuffer {
    let opts = MTLResourceOptions::StorageModeShared;
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(&val as *const u32 as *const _ as *mut std::ffi::c_void)
                    .unwrap(),
                4_usize,
                opts,
            )
            .unwrap()
    }
}

/// GPU top-k routing: softmax -> top-k -> normalize -> histogram -> prefix scan. /// /// All computation stays on the GPU with zero CPU synchronization. /// /// # Arguments /// - `gate_logits`: Raw gate logits `[N, E]`, Float16 or Float32 (NOT pre-softmaxed). /// - `top_k`: Number of experts to select per token (typically 2 or 8, max 8). /// - `expert_bias`: Optional `[E]` bias added to logits before softmax (adaptive routing). /// - `queue`: Metal command queue for dispatch. /// /// # Returns /// `TopkRouteResult` with expert_indices, expert_weights, expert_counts, and dispatch_offsets. pub
pub fn gpu_topk_route(
    registry: &KernelRegistry,
    gate_logits: &Array,
    top_k: usize,
    expert_bias: Option<&Array>,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<TopkRouteResult, KernelError> {
    let cb = queue.commandBuffer().unwrap();

    let result = gpu_topk_route_into_cb(registry, gate_logits, top_k, expert_bias, &cb)?;
    super::commit_with_mode(&cb, super::ExecMode::Sync);
    Ok(result)
}

/// Encode GPU top-k routing into an existing command buffer (no commit/wait).
///
/// Same as `gpu_topk_route` but encodes into `cb` for pipelining with other work.
/// The caller is responsible for committing and waiting on the command buffer.
pub fn gpu_topk_route_into_cb(
    registry: &KernelRegistry,
    gate_logits: &Array,
    top_k: usize,
    expert_bias: Option<&Array>,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<TopkRouteResult, KernelError> {
    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------
    if gate_logits.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gpu_topk_route: gate_logits must be 2D [N, E], got {}D",
            gate_logits.ndim()
        )));
    }
    if gate_logits.dtype() != DType::Float32 && gate_logits.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "gpu_topk_route: gate_logits must be Float16 or Float32, got {:?}",
            gate_logits.dtype()
        )));
    }

    // Softmax/topk requires f32 precision to avoid overflow/underflow.
    // Cast f16 → f32 internally so callers can pass f16 directly.
    let gate_logits_f32_owned;
    let gate_logits_f32: &Array = if gate_logits.dtype() != DType::Float32 {
        gate_logits_f32_owned =
            ops::copy::copy_cast_into_cb(registry, gate_logits, DType::Float32, cb)?;
        &gate_logits_f32_owned
    } else {
        gate_logits
    };

    if top_k == 0 || top_k > 8 {
        return Err(KernelError::InvalidShape(format!(
            "gpu_topk_route: top_k must be in [1, 8], got {top_k}"
        )));
    }

    let seq_len = gate_logits.shape()[0];
    let num_experts = gate_logits.shape()[1];

    if top_k > num_experts {
        return Err(KernelError::InvalidShape(format!(
            "gpu_topk_route: top_k ({top_k}) > num_experts ({num_experts})"
        )));
    }
    if num_experts > 256 {
        return Err(KernelError::InvalidShape(format!(
            "gpu_topk_route: num_experts ({num_experts}) exceeds maximum 256"
        )));
    }

    let has_bias: u32 = if expert_bias.is_some() { 1 } else { 0 };

    if let Some(bias) = expert_bias {
        if bias.ndim() != 1 {
            return Err(KernelError::InvalidShape(format!(
                "gpu_topk_route: expert_bias must be 1D [E], got {}D",
                bias.ndim()
            )));
        }
        if bias.shape()[0] != num_experts {
            return Err(KernelError::InvalidShape(format!(
                "gpu_topk_route: expert_bias length {} != num_experts {num_experts}",
                bias.shape()[0]
            )));
        }
        if bias.dtype() != DType::Float32 && bias.dtype() != DType::Float16 {
            return Err(KernelError::InvalidShape(format!(
                "gpu_topk_route: expert_bias must be Float16 or Float32, got {:?}",
                bias.dtype()
            )));
        }
    }

    // Cast expert_bias to f32 if needed (same reasoning as gate_logits)
    let bias_f32_owned;
    let expert_bias_f32: Option<&Array> = match expert_bias {
        Some(bias) if bias.dtype() != DType::Float32 => {
            bias_f32_owned = ops::copy::copy_cast_into_cb(registry, bias, DType::Float32, cb)?;
            Some(&bias_f32_owned)
        }
        other => other,
    };

    let seq_len_u32 = super::checked_u32(seq_len, "seq_len")?;
    let num_experts_u32 = super::checked_u32(num_experts, "num_experts")?;
    let top_k_u32 = super::checked_u32(top_k, "top_k")?;

    let dev = registry.device().raw();

    // -----------------------------------------------------------------------
    // Allocate output arrays
    // -----------------------------------------------------------------------
    let expert_indices = Array::zeros(dev, &[seq_len * top_k], DType::UInt32);
    let expert_weights = Array::zeros(dev, &[seq_len * top_k], DType::Float32);
    let expert_counts = Array::zeros(dev, &[num_experts], DType::UInt32);
    let dispatch_offsets = Array::zeros(dev, &[num_experts + 1], DType::UInt32);

    // -----------------------------------------------------------------------
    // Create a dummy bias buffer if no bias is provided.
    // Metal requires a non-null buffer binding, so we allocate a 4-byte dummy.
    // -----------------------------------------------------------------------
    let dummy_bias_buf;
    let bias_buffer;
    let bias_offset: usize;
    if let Some(bias) = expert_bias_f32 {
        bias_buffer = bias.metal_buffer();
        bias_offset = bias.offset();
    } else {
        dummy_bias_buf = dev
            .newBufferWithLength_options(4_usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        bias_buffer = &dummy_bias_buf;
        bias_offset = 0;
    }

    // -----------------------------------------------------------------------
    // Constant buffers
    // -----------------------------------------------------------------------
    let ne_buf = make_u32_buf(dev, num_experts_u32);
    let k_buf = make_u32_buf(dev, top_k_u32);
    let has_bias_buf = make_u32_buf(dev, has_bias);

    // -----------------------------------------------------------------------
    // Dispatch 1: moe_topk_route_f32
    //   One threadgroup per token row. Threadgroup size = 256 (enough for E<=256).
    // -----------------------------------------------------------------------
    {
        let pipeline = registry.get_pipeline("moe_topk_route_f32", DType::Float32)?;
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(
            0,
            Some(gate_logits_f32.metal_buffer()),
            gate_logits_f32.offset(),
        );
        enc.set_buffer(1, Some(bias_buffer), bias_offset);
        enc.set_buffer(2, Some(expert_indices.metal_buffer()), 0);
        enc.set_buffer(3, Some(expert_weights.metal_buffer()), 0);
        enc.set_buffer(4, Some(expert_counts.metal_buffer()), 0);
        enc.set_buffer(5, Some(&ne_buf), 0);
        enc.set_buffer(6, Some(&k_buf), 0);
        enc.set_buffer(7, Some(&has_bias_buf), 0);
        let tg_size = std::cmp::min(256, pipeline.maxTotalThreadsPerThreadgroup());

        if seq_len > 0 {
            enc.dispatch_threadgroups(
                MTLSize {
                    width: seq_len_u32 as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                },
            );
        }
        enc.end();
    }

    // -----------------------------------------------------------------------
    // Dispatch 2: moe_prefix_scan_u32
    //   Single threadgroup, single thread does the work (E<=256).
    // -----------------------------------------------------------------------
    {
        let pipeline = registry.get_pipeline("moe_prefix_scan_u32", DType::UInt32)?;
        let raw_enc = cb.computeCommandEncoder().unwrap();
        let enc = ComputePass::new(&raw_enc);
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, Some(expert_counts.metal_buffer()), 0);
        enc.set_buffer(1, Some(dispatch_offsets.metal_buffer()), 0);
        enc.set_buffer(2, Some(&ne_buf), 0);
        enc.dispatch_threadgroups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
    }

    Ok(TopkRouteResult {
        expert_indices,
        expert_weights,
        expert_counts,
        dispatch_offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (rmlx_metal::MtlDevice, rmlx_metal::MtlQueue, KernelRegistry) {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal device");
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("GPU device");
        let queue = gpu.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        register(&registry).expect("register topk_route kernels");
        (device, queue, registry)
    }

    #[test]
    fn test_topk_route_basic() {
        let (device, queue, registry) = setup();

        // 4 tokens, 8 experts, top-2
        // Gate logits: each row has one dominant expert
        #[rustfmt::skip]
        let logits_data: Vec<f32> = vec![
            // token 0: expert 2 and 5 are strong
            0.1, 0.1, 5.0, 0.1, 0.1, 3.0, 0.1, 0.1,
            // token 1: expert 0 and 7 are strong
            4.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.0,
            // token 2: expert 3 and 4 are strong
            0.1, 0.1, 0.1, 5.0, 4.0, 0.1, 0.1, 0.1,
            // token 3: expert 1 and 6 are strong
            0.1, 4.0, 0.1, 0.1, 0.1, 0.1, 3.5, 0.1,
        ];

        let gate_logits = Array::from_slice(&device, &logits_data, vec![4, 8]);

        let result =
            gpu_topk_route(&registry, &gate_logits, 2, None, &queue).expect("gpu_topk_route");

        // Check shapes
        assert_eq!(result.expert_indices.shape(), &[8]); // 4 * 2
        assert_eq!(result.expert_weights.shape(), &[8]); // 4 * 2
        assert_eq!(result.expert_counts.shape(), &[8]);
        assert_eq!(result.dispatch_offsets.shape(), &[9]); // 8 + 1

        // Check that expert_indices are reasonable (top-2 of each row)
        let indices = result.expert_indices.to_vec_checked::<u32>();
        // Token 0: should select experts 2 and 5 (in some order)
        let tok0: Vec<u32> = indices[0..2].to_vec();
        assert!(
            tok0.contains(&2),
            "token 0 should select expert 2, got {:?}",
            tok0
        );
        assert!(
            tok0.contains(&5),
            "token 0 should select expert 5, got {:?}",
            tok0
        );

        // Token 1: should select experts 0 and 7
        let tok1: Vec<u32> = indices[2..4].to_vec();
        assert!(
            tok1.contains(&0),
            "token 1 should select expert 0, got {:?}",
            tok1
        );
        assert!(
            tok1.contains(&7),
            "token 1 should select expert 7, got {:?}",
            tok1
        );

        // Check weights sum to ~1.0 per token
        let weights = result.expert_weights.to_vec_checked::<f32>();
        for t in 0..4 {
            let w0 = weights[t * 2];
            let w1 = weights[t * 2 + 1];
            let sum = w0 + w1;
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "token {t} weights should sum to 1.0, got {sum} ({w0} + {w1})"
            );
        }

        // Check expert_counts sums to N*K = 8
        let counts = result.expert_counts.to_vec_checked::<u32>();
        let total: u32 = counts.iter().sum();
        assert_eq!(total, 8, "total expert count should be 8, got {total}");

        // Check dispatch_offsets is a valid exclusive prefix sum
        let offsets = result.dispatch_offsets.to_vec_checked::<u32>();
        assert_eq!(offsets[0], 0, "dispatch_offsets[0] should be 0");
        for e in 0..8 {
            assert_eq!(
                offsets[e + 1],
                offsets[e] + counts[e],
                "dispatch_offsets[{e}+1] should be offsets[{e}] + counts[{e}]"
            );
        }
        assert_eq!(offsets[8], 8, "dispatch_offsets[E] should be N*K=8");
    }

    #[test]
    fn test_topk_route_with_bias() {
        let (device, queue, registry) = setup();

        // 2 tokens, 4 experts, top-1
        // Without bias: token 0 selects expert 0 (logit 3.0)
        // With large bias on expert 2: should flip selection
        let logits_data: Vec<f32> = vec![
            3.0, 1.0, 2.9, 1.0, // token 0
            1.0, 3.0, 1.0, 1.0, // token 1
        ];
        let bias_data: Vec<f32> = vec![0.0, 0.0, 5.0, 0.0]; // large bias on expert 2

        let gate_logits = Array::from_slice(&device, &logits_data, vec![2, 4]);
        let expert_bias = Array::from_slice(&device, &bias_data, vec![4]);

        let result = gpu_topk_route(&registry, &gate_logits, 1, Some(&expert_bias), &queue)
            .expect("gpu_topk_route with bias");

        let indices = result.expert_indices.to_vec_checked::<u32>();
        // Token 0: logit[2] + bias[2] = 2.9 + 5.0 = 7.9, which dominates
        assert_eq!(
            indices[0], 2,
            "with bias, token 0 should select expert 2, got {}",
            indices[0]
        );
    }

    #[test]
    fn test_topk_route_validation() {
        let (device, queue, registry) = setup();

        // 1D input should fail
        let bad = Array::from_slice(&device, &[1.0f32, 2.0, 3.0], vec![3]);
        assert!(gpu_topk_route(&registry, &bad, 2, None, &queue).is_err());

        // top_k=0 should fail
        let logits = Array::from_slice(&device, &[1.0f32; 8], vec![2, 4]);
        assert!(gpu_topk_route(&registry, &logits, 0, None, &queue).is_err());

        // top_k=9 should fail
        assert!(gpu_topk_route(&registry, &logits, 9, None, &queue).is_err());

        // top_k > num_experts should fail
        assert!(gpu_topk_route(&registry, &logits, 5, None, &queue).is_err());
    }

    #[test]
    fn test_topk_route_single_token() {
        let (device, queue, registry) = setup();

        // 1 token, 4 experts, top-2
        let logits_data: Vec<f32> = vec![1.0, 5.0, 0.5, 4.0];
        let gate_logits = Array::from_slice(&device, &logits_data, vec![1, 4]);

        let result = gpu_topk_route(&registry, &gate_logits, 2, None, &queue)
            .expect("single token topk_route");

        let indices = result.expert_indices.to_vec_checked::<u32>();
        assert_eq!(indices.len(), 2);
        // Expert 1 (logit=5.0) and expert 3 (logit=4.0) should be selected
        assert!(
            indices.contains(&1),
            "should select expert 1, got {:?}",
            indices
        );
        assert!(
            indices.contains(&3),
            "should select expert 3, got {:?}",
            indices
        );

        let weights = result.expert_weights.to_vec_checked::<f32>();
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_topk_route_f16_input() {
        let (device, queue, registry) = setup();
        crate::ops::copy::register(&registry).expect("register copy kernels");

        // 2 tokens, 4 experts, top-1
        let logits_f32: Vec<f32> = vec![
            1.0, 5.0, 0.5, 4.0, // token 0: expert 1 dominant
            3.0, 0.1, 0.2, 0.1, // token 1: expert 0 dominant
        ];
        let gate_f32 = Array::from_slice(&device, &logits_f32, vec![2, 4]);

        // Cast to f16 to simulate f16 activation pipeline
        let gate_f16 =
            crate::ops::copy::copy_cast(&registry, &gate_f32, DType::Float16, &queue).unwrap();
        assert_eq!(gate_f16.dtype(), DType::Float16);

        let result = gpu_topk_route(&registry, &gate_f16, 1, None, &queue)
            .expect("gpu_topk_route with f16 input");

        let indices = result.expert_indices.to_vec_checked::<u32>();
        assert_eq!(indices[0], 1, "token 0 should select expert 1");
        assert_eq!(indices[1], 0, "token 1 should select expert 0");

        // Weights should still be f32 (output is always f32)
        assert_eq!(result.expert_weights.dtype(), DType::Float32);
        let weights = result.expert_weights.to_vec_checked::<f32>();
        assert!(
            (weights[0] - 1.0).abs() < 1e-3,
            "top-1 weight should be ~1.0, got {}",
            weights[0]
        );
    }
}
