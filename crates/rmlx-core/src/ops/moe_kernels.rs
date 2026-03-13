//! Fused MoE helper kernels: index_gather and scatter_weighted_add.
//!
//! These kernels replace per-token loops in `MoeLayer::forward_gather_mm()`
//! that previously required O(n_assign) GPU sync points each.
//!
//! - `index_gather`: Gathers rows from a 2D source by token indices into a
//!   contiguous output buffer — single dispatch replaces N per-token copy encoders.
//! - `scatter_weighted_add`: Multiplies each gathered expert output row by its
//!   routing weight and atomically accumulates into the output at the correct
//!   token position — single dispatch replaces N×3 sync points (mul + add + copy).

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLResourceOptions;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader sources
// ---------------------------------------------------------------------------

/// Metal shader for MoE index_gather and scatter_weighted_add.
///
/// `index_gather_{f32,f16,bf16}`:
///   Gathers rows from src[seq_len, D] into dst[n_assign, D] using
///   token_indices[n_assign]. Each threadgroup handles one assignment row.
///
/// `scatter_weighted_add_{f32,f16,bf16}`:
///   For each assignment i: dst[token_indices[i]] += weights[i] * src[i].
///   Uses atomic CAS loop for f16/bf16 (packed as uint16), native float
///   atomic for f32. Each threadgroup handles one assignment row.
pub const MOE_KERNELS_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ======================================================================
// index_gather: src[seq, D] -> dst[n_assign, D] via token_indices
// ======================================================================

kernel void index_gather_f32(
    device const float*  src            [[buffer(0)]],
    device float*        dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const uint&   D              [[buffer(3)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    for (uint col = tid; col < D; col += tg_size) {
        dst[assign_idx * D + col] = src[tok * D + col];
    }
}

kernel void index_gather_f16(
    device const half*   src            [[buffer(0)]],
    device half*         dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const uint&   D              [[buffer(3)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    for (uint col = tid; col < D; col += tg_size) {
        dst[assign_idx * D + col] = src[tok * D + col];
    }
}

kernel void index_gather_bf16(
    device const bfloat* src            [[buffer(0)]],
    device bfloat*       dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const uint&   D              [[buffer(3)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    for (uint col = tid; col < D; col += tg_size) {
        dst[assign_idx * D + col] = src[tok * D + col];
    }
}

// ======================================================================
// scatter_weighted_add: dst[token_indices[i]] += weights[i] * src[i]
// Uses atomic CAS for correctness when multiple assignments map to same token.
// ======================================================================

// Atomic float add via CAS loop
inline void atomic_add_f32(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

kernel void scatter_weighted_add_f32(
    device const float*  src            [[buffer(0)]],
    device atomic_uint*  dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const float*  weights        [[buffer(3)]],
    device const uint&   D              [[buffer(4)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    float w = weights[assign_idx];
    for (uint col = tid; col < D; col += tg_size) {
        float val = src[assign_idx * D + col] * w;
        atomic_add_f32(&dst[tok * D + col], val);
    }
}

// Atomic half add via CAS on uint16 (reinterpret)
inline void atomic_add_f16(device atomic_uint* addr, uint col, half val) {
    // Pack two f16 values into one uint32 for atomic access.
    // Determine which half of the uint32 this column maps to.
    uint word_idx = col / 2;
    uint half_idx = col % 2;  // 0 = low 16 bits, 1 = high 16 bits

    device atomic_uint* word_addr = &addr[word_idx];
    uint expected = atomic_load_explicit(word_addr, memory_order_relaxed);
    uint desired;
    do {
        // Extract the target half from the word
        half cur;
        if (half_idx == 0) {
            cur = as_type<half>(ushort(expected & 0xFFFF));
        } else {
            cur = as_type<half>(ushort((expected >> 16) & 0xFFFF));
        }
        half updated = cur + val;
        ushort updated_bits = as_type<ushort>(updated);
        if (half_idx == 0) {
            desired = (expected & 0xFFFF0000) | uint(updated_bits);
        } else {
            desired = (expected & 0x0000FFFF) | (uint(updated_bits) << 16);
        }
    } while (!atomic_compare_exchange_weak_explicit(
        word_addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

kernel void scatter_weighted_add_f16(
    device const half*   src            [[buffer(0)]],
    device atomic_uint*  dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const float*  weights        [[buffer(3)]],
    device const uint&   D              [[buffer(4)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    float w = weights[assign_idx];
    // dst is reinterpreted as atomic_uint (32-bit words), each holding 2 halfs
    device atomic_uint* dst_row = &dst[(tok * D) / 2];
    for (uint col = tid; col < D; col += tg_size) {
        half val = half(float(src[assign_idx * D + col]) * w);
        atomic_add_f16(dst_row, col, val);
    }
}

kernel void scatter_weighted_add_bf16(
    device const bfloat* src            [[buffer(0)]],
    device atomic_uint*  dst            [[buffer(1)]],
    device const uint*   token_indices  [[buffer(2)]],
    device const float*  weights        [[buffer(3)]],
    device const uint&   D              [[buffer(4)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint assign_idx = gid;
    uint tok = token_indices[assign_idx];
    float w = weights[assign_idx];
    // Same 2-element packing as f16 (bfloat is also 16-bit)
    device atomic_uint* dst_row = &dst[(tok * D) / 2];
    for (uint col = tid; col < D; col += tg_size) {
        bfloat val = bfloat(float(src[assign_idx * D + col]) * w);
        // Reinterpret bfloat as ushort for atomic packing
        ushort val_bits = as_type<ushort>(val);
        uint half_idx = col % 2;
        uint word_idx = col / 2;
        device atomic_uint* word_addr = &dst_row[word_idx];
        uint expected = atomic_load_explicit(word_addr, memory_order_relaxed);
        uint desired;
        do {
            ushort cur_bits;
            if (half_idx == 0) {
                cur_bits = ushort(expected & 0xFFFF);
            } else {
                cur_bits = ushort((expected >> 16) & 0xFFFF);
            }
            bfloat cur = as_type<bfloat>(cur_bits);
            bfloat updated = bfloat(float(cur) + float(val));
            ushort updated_bits = as_type<ushort>(updated);
            if (half_idx == 0) {
                desired = (expected & 0xFFFF0000) | uint(updated_bits);
            } else {
                desired = (expected & 0x0000FFFF) | (uint(updated_bits) << 16);
            }
        } while (!atomic_compare_exchange_weak_explicit(
            word_addr, &expected, desired,
            memory_order_relaxed, memory_order_relaxed));
    }
}
"#;

/// Register MoE helper kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("moe_kernels", MOE_KERNELS_SHADER_SOURCE)
}

/// Gather rows from `src` by token indices into a contiguous output.
///
/// # Arguments
/// - `src`: Source 2D array `[seq_len, D]`
/// - `token_indices`: `[n_assign]` UInt32 — which row of `src` each assignment reads
///
/// # Returns
/// `[n_assign, D]` with gathered rows.
pub fn index_gather(
    registry: &KernelRegistry,
    src: &Array,
    token_indices: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let n_assign = token_indices.shape()[0];
    let d = src.shape()[src.shape().len() - 1];

    let kernel_name = match src.dtype() {
        DType::Float32 => "index_gather_f32",
        DType::Float16 => "index_gather_f16",
        DType::Bfloat16 => "index_gather_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "index_gather: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &[n_assign, d], src.dtype());

    let d_buf = make_u32_buf(dev, super::checked_u32(d, "D")?);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset());
    enc.set_buffer(1, Some(out.metal_buffer()), 0);
    enc.set_buffer(
        2,
        Some(token_indices.metal_buffer()),
        token_indices.offset(),
    );
    enc.set_buffer(3, Some(&d_buf), 0);
    let tg_size = std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), d);
    // One threadgroup per assignment row
    let grid = MTLSize {
        width: n_assign,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threadgroups(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Scatter expert outputs back to token positions with weighted accumulation.
///
/// Computes: `output[token_indices[i]] += weights[i] * src[i]` for all i.
///
/// # Arguments
/// - `src`: Expert output `[n_assign, D]`
/// - `output`: Pre-zeroed output buffer `[seq_len, D]` — accumulated in-place
/// - `token_indices`: `[n_assign]` UInt32 — destination token index per assignment
/// - `weights`: `[n_assign]` Float32 — routing weight per assignment
pub fn scatter_weighted_add(
    registry: &KernelRegistry,
    src: &Array,
    output: &Array,
    token_indices: &Array,
    weights: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(), KernelError> {
    let n_assign = token_indices.shape()[0];
    let d = src.shape()[src.shape().len() - 1];

    let kernel_name = match src.dtype() {
        DType::Float32 => "scatter_weighted_add_f32",
        DType::Float16 => "scatter_weighted_add_f16",
        DType::Bfloat16 => "scatter_weighted_add_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "scatter_weighted_add: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let dev = registry.device().raw();
    let d_buf = make_u32_buf(dev, super::checked_u32(d, "D")?);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset());
    enc.set_buffer(1, Some(output.metal_buffer()), output.offset());
    enc.set_buffer(
        2,
        Some(token_indices.metal_buffer()),
        token_indices.offset(),
    );
    enc.set_buffer(3, Some(weights.metal_buffer()), weights.offset());
    enc.set_buffer(4, Some(&d_buf), 0);
    let tg_size = std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), d);
    // One threadgroup per assignment row
    let grid = MTLSize {
        width: n_assign,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threadgroups(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let gpu = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu.raw().newCommandQueue().unwrap();
        let registry = KernelRegistry::new(gpu);
        register(&registry).unwrap();
        (registry, queue)
    }

    #[test]
    fn test_index_gather_f32() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        // src: 4 rows × 3 cols
        let src_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let src = Array::from_slice(dev, &src_data, vec![4, 3]);
        let indices = Array::from_slice(dev, &[2u32, 0, 3], vec![3]);

        let result = index_gather(&registry, &src, &indices, &queue).unwrap();
        let out: Vec<f32> = result.to_vec_checked();

        assert_eq!(out, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_scatter_weighted_add_f32() {
        let (registry, queue) = setup();
        let dev = registry.device().raw();

        // src: 3 assignments × 2 cols
        let src_data: Vec<f32> = vec![
            10.0, 20.0, // assignment 0
            30.0, 40.0, // assignment 1
            50.0, 60.0, // assignment 2
        ];
        let src = Array::from_slice(dev, &src_data, vec![3, 2]);

        // output: 2 tokens × 2 cols (zeroed)
        let output = Array::zeros(dev, &[2, 2], DType::Float32);

        // assignments 0 and 2 map to token 0, assignment 1 maps to token 1
        let tok_indices = Array::from_slice(dev, &[0u32, 1, 0], vec![3]);
        let weights = Array::from_slice(dev, &[0.5f32, 1.0, 0.25], vec![3]);

        scatter_weighted_add(&registry, &src, &output, &tok_indices, &weights, &queue).unwrap();
        let out: Vec<f32> = output.to_vec_checked();

        // token 0: 0.5*10 + 0.25*50 = 17.5, 0.5*20 + 0.25*60 = 25.0
        // token 1: 1.0*30 = 30.0, 1.0*40 = 40.0
        assert!((out[0] - 17.5).abs() < 1e-4, "got {}", out[0]);
        assert!((out[1] - 25.0).abs() < 1e-4, "got {}", out[1]);
        assert!((out[2] - 30.0).abs() < 1e-4, "got {}", out[2]);
        assert!((out[3] - 40.0).abs() < 1e-4, "got {}", out[3]);
    }
}
