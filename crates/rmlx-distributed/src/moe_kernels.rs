//! Metal compute kernels for MoE dispatch and combine operations.
//!
//! Provides two specialized kernels:
//! - `moe_scatter_add_f32`: Scatter-add expert outputs back to batch positions
//!   with routing weights (used in combine phase).
//! - `moe_permute_tokens_f32`: Permute tokens from batch order to expert order
//!   (used in dispatch phase).
//!
//! Call [`init_kernels`] once at startup to JIT-compile and register these
//! kernels with the given [`KernelRegistry`].

use rmlx_core::kernels::{KernelError, KernelRegistry};

// ---------------------------------------------------------------------------
// Metal shader source (inline)
// ---------------------------------------------------------------------------

/// Metal shader source for MoE scatter-add and permute kernels.
///
/// `moe_scatter_add_f32`: For each assigned token, atomically accumulates
/// `weight * expert_output[expert_id * cap * hidden + slot * hidden + d]`
/// into `output[position * hidden + d]`. Each thread handles one hidden
/// dimension element for one token assignment.
///
/// `moe_permute_tokens_f32`: Copies token data from batch-ordered `input`
/// into expert-ordered `output` using the provided index mapping. Each
/// thread handles one hidden dimension element for one token.
const MOE_KERNELS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// moe_scatter_add_f32
//
// Grid: total_tokens * hidden_dim threads (1D).
// Thread i handles token (i / hidden_dim), dimension (i % hidden_dim).
// ---------------------------------------------------------------------------
kernel void moe_scatter_add_f32(
    device const float* expert_outputs [[buffer(0)]],  // [num_experts * cap * hidden]
    device const int*   positions      [[buffer(1)]],  // [total_tokens] - batch position
    device const int*   expert_ids     [[buffer(2)]],  // [total_tokens] - which expert
    device const float* weights        [[buffer(3)]],  // [total_tokens] - routing weight
    device float*       output         [[buffer(4)]],  // [batch_size * hidden]
    constant uint&      hidden_dim     [[buffer(5)]],
    constant uint&      cap            [[buffer(6)]],
    constant uint&      total_tokens   [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint token_idx = tid / hidden_dim;
    uint d         = tid % hidden_dim;

    if (token_idx >= total_tokens) return;

    int pos       = positions[token_idx];
    int expert_id = expert_ids[token_idx];
    float w       = weights[token_idx];

    // Compute the slot within the expert's capacity.
    // expert_outputs layout: [expert_id * cap * hidden + slot * hidden + d]
    // The slot is encoded as (token_idx - prefix_sum_of_tokens_before_this_expert),
    // but for simplicity the caller packs tokens contiguously and slot is implicit
    // from the token_idx ordering. We use token_idx directly as a flat index into
    // the expert_outputs buffer which is laid out as [total_tokens * hidden].
    uint src_idx = token_idx * hidden_dim + d;
    uint dst_idx = uint(pos) * hidden_dim + d;

    // Atomic add to handle multiple experts writing to the same batch position.
    // Metal does not have native atomic float add on all devices, so we use a
    // compare-and-swap loop for correctness.
    float val = w * expert_outputs[src_idx];

    // CAS loop for atomic float addition
    device atomic_uint* dst_atomic = (device atomic_uint*)(&output[dst_idx]);
    uint expected = atomic_load_explicit(dst_atomic, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + val;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(
                dst_atomic, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// moe_permute_tokens_f32
//
// Grid: total_tokens * hidden_dim threads (1D).
// Copies input[src_indices[token] * hidden + d] -> output[token * hidden + d].
// ---------------------------------------------------------------------------
kernel void moe_permute_tokens_f32(
    device const float* input        [[buffer(0)]],  // [batch_size * hidden]
    device const int*   src_indices  [[buffer(1)]],  // [total_tokens] - source batch index
    device float*       output       [[buffer(2)]],  // [total_tokens * hidden]
    constant uint&      hidden_dim   [[buffer(3)]],
    constant uint&      total_tokens [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint token_idx = tid / hidden_dim;
    uint d         = tid % hidden_dim;

    if (token_idx >= total_tokens) return;

    int src_pos  = src_indices[token_idx];
    uint src_idx = uint(src_pos) * hidden_dim + d;
    uint dst_idx = token_idx * hidden_dim + d;

    output[dst_idx] = input[src_idx];
}
"#;

/// JIT-compile and register MoE Metal kernels with the kernel registry.
///
/// This registers the following kernel functions:
/// - `moe_scatter_add_f32`
/// - `moe_permute_tokens_f32`
///
/// Safe to call multiple times; subsequent calls are no-ops if the kernels
/// are already registered.
pub fn init_kernels(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source_if_absent("moe_kernels", MOE_KERNELS_SOURCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmlx_core::dtype::DType;
    use rmlx_metal::device::GpuDevice;

    /// Helper: create a GpuDevice, returning None if Metal is unavailable.
    fn try_gpu_device() -> Option<GpuDevice> {
        GpuDevice::system_default().ok()
    }

    #[test]
    fn test_moe_kernels_register_and_lookup() {
        let device = match try_gpu_device() {
            Some(d) => d,
            None => {
                eprintln!("skipping test_moe_kernels_register_and_lookup: no Metal device");
                return;
            }
        };

        let registry = KernelRegistry::new(device);

        // Register MoE kernels.
        init_kernels(&registry).expect("init_kernels should succeed");

        // Verify scatter-add kernel can be looked up and pipeline created.
        let _scatter_pipeline = registry
            .get_pipeline("moe_scatter_add_f32", DType::Float32)
            .expect("moe_scatter_add_f32 pipeline should be found");

        // Verify permute kernel can be looked up and pipeline created.
        let _permute_pipeline = registry
            .get_pipeline("moe_permute_tokens_f32", DType::Float32)
            .expect("moe_permute_tokens_f32 pipeline should be found");

        // Verify idempotency: calling init_kernels again should not error.
        init_kernels(&registry).expect("second init_kernels should be idempotent");
    }

    #[test]
    fn test_moe_kernels_not_found_before_init() {
        let device = match try_gpu_device() {
            Some(d) => d,
            None => {
                eprintln!("skipping test_moe_kernels_not_found_before_init: no Metal device");
                return;
            }
        };

        let registry = KernelRegistry::new(device);

        // Before init_kernels, the MoE kernels should not be found.
        let result = registry.get_pipeline("moe_scatter_add_f32", DType::Float32);
        assert!(
            result.is_err(),
            "moe_scatter_add_f32 should not exist before init_kernels"
        );
    }
}
