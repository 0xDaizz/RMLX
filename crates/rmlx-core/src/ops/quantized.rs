//! Quantized matrix multiplication (Q4_0, Q4_1, Q8_0).
//! Full implementation requires vendored MLX quantized.metal (~2508 lines).
//! This provides the Rust dispatch layer with JIT-compilable stub kernels.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

/// Quantized block for Q4_0: 32 elements -> 18 bytes (2 bytes scale + 16 bytes data)
#[repr(C)]
pub struct BlockQ4_0 {
    pub scale: u16, // f16 stored as u16
    pub data: [u8; 16],
}

/// Quantized block for Q8_0: 32 elements -> 34 bytes (2 bytes scale + 32 bytes data)
#[repr(C)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 stored as u16
    pub data: [u8; 32],
}

pub const QUANTIZED_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Dequantize Q8_0 and multiply with f32 vector
// Simplified single-row version for initial implementation
kernel void qmv_q8_0_f32(
    device const uchar* weights [[buffer(0)]],
    device const float* vec [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& out_features [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    if (row >= out_features) return;

    threadgroup float shared_sum[256];

    // Q8_0: each group of 32 elements is stored as 2 bytes (f16 scale) + 32 bytes (int8 data) = 34 bytes
    uint groups_per_row = in_features / 32;
    uint row_bytes = groups_per_row * 34;

    float sum = 0.0;
    for (uint g = tid; g < groups_per_row; g += tgsize) {
        uint group_offset = row * row_bytes + g * 34;
        // Read scale (f16 -> float)
        half scale_h = as_type<half>(*(device const ushort*)(weights + group_offset));
        float scale = float(scale_h);

        // Dequantize and dot product
        for (uint j = 0; j < 32; j++) {
            int8_t q = as_type<int8_t>(weights[group_offset + 2 + j]);
            float w = float(q) * scale;
            sum += w * vec[g * 32 + j];
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) output[row] = shared_sum[0];
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("quantized", QUANTIZED_SHADER_SOURCE)
}

/// Quantized matrix-vector multiply.
/// weights: quantized [out_features, in_features], vec: [in_features] f32.
pub fn quantized_matmul(
    registry: &KernelRegistry,
    weights: &Array,
    vec: &Array,
    out_features: usize,
    in_features: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let kernel_name = match weights.dtype() {
        DType::Q8_0 => "qmv_q8_0_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "quantized_matmul not supported for {:?}",
                weights.dtype()
            )))
        }
    };

    // Validate that in_features is block-aligned for the quantized dtype
    let block_sz = weights
        .dtype()
        .block_size()
        .expect("quantized_matmul requires a quantized dtype");
    if in_features % block_sz != 0 {
        return Err(KernelError::InvalidShape(format!(
            "in_features ({in_features}) must be a multiple of block_size ({block_sz})"
        )));
    }

    // Validate input vector size
    if vec.numel() != in_features {
        return Err(KernelError::InvalidShape(format!(
            "vec.numel() ({}) != in_features ({in_features})",
            vec.numel()
        )));
    }

    // Validate weights buffer size matches expected packed size
    let expected_weight_bytes = weights.dtype().numel_to_bytes(out_features * in_features);
    let actual_weight_bytes = weights.metal_buffer().length() as usize;
    if actual_weight_bytes < expected_weight_bytes {
        return Err(KernelError::InvalidShape(format!(
            "weights buffer too small: {} bytes < expected {} bytes for [{out_features}, {in_features}] {:?}",
            actual_weight_bytes, expected_weight_bytes, weights.dtype()
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, weights.dtype())?;
    let out = Array::zeros(registry.device().raw(), &[out_features], DType::Float32);

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let out_f = super::checked_u32(out_features, "out_features")?;
    let in_f = super::checked_u32(in_features, "in_features")?;
    let of_buf = dev.new_buffer_with_data(&out_f as *const u32 as *const _, 4, opts);
    let if_buf = dev.new_buffer_with_data(&in_f as *const u32 as *const _, 4, opts);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(weights.metal_buffer()), weights.offset() as u64);
    enc.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&of_buf), 0);
    enc.set_buffer(4, Some(&if_buf), 0);

    let tg = std::cmp::min(256, pipeline.max_total_threads_per_threadgroup());
    enc.dispatch_thread_groups(
        metal::MTLSize::new(out_features as u64, 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}
