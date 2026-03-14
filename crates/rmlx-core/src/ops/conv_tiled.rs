//! Optimized convolution with shared memory tiling.
//!
//! Provides a tiled Conv2d kernel that uses threadgroup shared memory for
//! weight reuse, reducing global memory bandwidth for larger input channel
//! counts and kernel sizes. Supports arbitrary kernel sizes (not just 3x3)
//! via dynamically-allocated threadgroup memory.

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
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader for tiled convolution.
///
/// Uses threadgroup shared memory (dynamically sized) to tile weight loads,
/// reducing global memory bandwidth by reusing loaded weight tiles.
/// Supports arbitrary kernel sizes via host-side threadgroup memory allocation.
pub const CONV_TILED_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Tiled Conv2d: uses threadgroup shared memory for weight reuse
// ============================================================================
// Direct tiled convolution with shared memory for the weight tile.
// Each threadgroup loads a tile of weights into shared memory, then all
// threads in the group use it to compute their output elements.
//
// Input:  [B, C_in, H, W]
// Weight: [C_out, C_in/groups, kH, kW]
// Output: [B, C_out, H_out, W_out]
//
// Grid: (W_out, C_out, B * H_out)
//
// Threadgroup memory is allocated dynamically on the host side with size
// TILE_CI * kH * kW * sizeof(float), so any kernel size is supported.

constant constexpr uint TILE_CI = 16;  // Tile over input channels

kernel void conv2d_tiled_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint*      params  [[buffer(4)]],
    threadgroup float*  weight_tile [[threadgroup(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    uint w_out_idx = gid.x;
    uint c_out_idx = gid.y;
    uint bh_idx    = gid.z;

    uint B_       = params[0];
    uint C_in     = params[1];
    uint H        = params[2];
    uint W        = params[3];
    uint C_out    = params[4];
    uint kH       = params[5];
    uint kW       = params[6];
    uint H_out    = params[7];
    uint W_out    = params[8];
    uint stride_h = params[9];
    uint stride_w = params[10];
    uint pad_h    = params[11];
    uint pad_w    = params[12];
    uint dil_h    = params[13];
    uint dil_w    = params[14];
    uint groups   = params[15];
    uint has_bias = params[16];

    uint h_out_idx = bh_idx % H_out;
    uint b_idx     = bh_idx / H_out;

    if (b_idx >= B_ || c_out_idx >= C_out || w_out_idx >= W_out) return;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    uint kernel_size = kH * kW;

    // Iterate over input channels in tiles
    for (uint ci_base = 0; ci_base < c_in_count; ci_base += TILE_CI) {
        uint ci_end = min(ci_base + TILE_CI, c_in_count);
        uint ci_tile_size = ci_end - ci_base;

        // Cooperative load of weight tile (each thread loads some elements)
        uint total_weight_elems = ci_tile_size * kernel_size;
        uint load_tid = lid.x + lid.y * tg_size.x;
        uint load_size = tg_size.x * tg_size.y;
        for (uint idx = load_tid; idx < total_weight_elems; idx += load_size) {
            uint ci_local = idx / kernel_size;
            uint k_idx = idx % kernel_size;
            uint ci = ci_base + ci_local;
            weight_tile[ci_local * kernel_size + k_idx] =
                weight[c_out_idx * c_in_count * kernel_size + ci * kernel_size + k_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial sum using the loaded weight tile
        for (uint ci_local = 0; ci_local < ci_tile_size; ci_local++) {
            uint c_in_idx = c_in_start + ci_base + ci_local;
            for (uint khi = 0; khi < kH; khi++) {
                int h_in = int(h_out_idx * stride_h + khi * dil_h) - int(pad_h);
                if (h_in < 0 || uint(h_in) >= H) continue;
                for (uint kwi = 0; kwi < kW; kwi++) {
                    int w_in = int(w_out_idx * stride_w + kwi * dil_w) - int(pad_w);
                    if (w_in >= 0 && uint(w_in) < W) {
                        float inp = input[b_idx * C_in * H * W + c_in_idx * H * W + uint(h_in) * W + uint(w_in)];
                        float wgt = weight_tile[ci_local * kernel_size + khi * kW + kwi];
                        sum += inp * wgt;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_bias) sum += bias[c_out_idx];
    output[b_idx * C_out * H_out * W_out + c_out_idx * H_out * W_out + h_out_idx * W_out + w_out_idx] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register tiled convolution kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("conv_tiled", CONV_TILED_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a Metal buffer containing a `[u32]` parameter array.
fn make_params_buf(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    params: &[u32],
) -> rmlx_metal::MtlBuffer {
    let byte_len = std::mem::size_of_val(params);
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(
                    params.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void
                )
                .unwrap(),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

/// Create a tiny dummy buffer for bias when absent.
fn make_dummy_buf(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> rmlx_metal::MtlBuffer {
    device
        .newBufferWithLength_options(4_usize, MTLResourceOptions::StorageModeShared)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Tiled 2D convolution using shared memory for weight reuse.
///
/// This is an optimized version of `conv::conv2d` that uses threadgroup
/// shared memory to tile weight loads, reducing global memory bandwidth
/// for larger input channel counts and kernel sizes.
///
/// Falls back to the same interface as `conv::conv2d`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_tiled(
    registry: &KernelRegistry,
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    // Validate shapes (same as conv::conv2d)
    if input.ndim() != 4 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d_tiled: input must be 4D, got {}D",
            input.ndim()
        )));
    }
    if weight.ndim() != 4 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d_tiled: weight must be 4D, got {}D",
            weight.ndim()
        )));
    }
    // If tiled conv doesn't support f16/bf16, fall back to standard conv.
    // The tiled kernel only has an f32 variant; gracefully redirect
    // half-precision inputs to the non-tiled conv2d path which handles them.
    if matches!(input.dtype(), DType::Float16 | DType::Bfloat16) {
        return super::conv::conv2d(
            registry, input, weight, bias, stride, padding, dilation, groups, queue,
        );
    }
    if input.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d_tiled: currently supports f32 only, got {:?}",
            input.dtype()
        )));
    }
    if input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "conv2d_tiled: input dtype {:?} != weight dtype {:?}",
            input.dtype(),
            weight.dtype()
        )));
    }

    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (dil_h, dil_w) = dilation;

    if groups == 0 || c_in % groups != 0 || c_out % groups != 0 {
        return Err(KernelError::InvalidShape(
            "conv2d_tiled: invalid groups configuration".into(),
        ));
    }
    if c_in_per_group != c_in / groups {
        return Err(KernelError::InvalidShape(format!(
            "conv2d_tiled: weight shape[1]={c_in_per_group} != C_in/groups={}",
            c_in / groups
        )));
    }

    let h_out = (h + 2 * pad_h)
        .checked_sub(dil_h * (kh - 1) + 1)
        .map(|v| v / stride_h + 1)
        .ok_or_else(|| {
            KernelError::InvalidShape("conv2d_tiled: output height is non-positive".into())
        })?;
    let w_out = (w + 2 * pad_w)
        .checked_sub(dil_w * (kw - 1) + 1)
        .map(|v| v / stride_w + 1)
        .ok_or_else(|| {
            KernelError::InvalidShape("conv2d_tiled: output width is non-positive".into())
        })?;

    if h_out == 0 || w_out == 0 {
        return Err(KernelError::InvalidShape(
            "conv2d_tiled: output spatial dims are zero".into(),
        ));
    }

    // Ensure contiguous
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let weight_contig = super::make_contiguous(weight, registry, queue)?;
    let weight = weight_contig.as_ref().unwrap_or(weight);

    let pipeline = registry.get_pipeline("conv2d_tiled_f32", DType::Float32)?;
    let out = Array::zeros(
        registry.device().raw(),
        &[batch, c_out, h_out, w_out],
        DType::Float32,
    );

    let has_bias: u32 = if bias.is_some() { 1 } else { 0 };
    let params: Vec<u32> = vec![
        batch as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        h_out as u32,
        w_out as u32,
        stride_h as u32,
        stride_w as u32,
        pad_h as u32,
        pad_w as u32,
        dil_h as u32,
        dil_w as u32,
        groups as u32,
        has_bias,
    ];
    let params_buf = make_params_buf(registry.device().raw(), &params);
    let dummy_buf = make_dummy_buf(registry.device().raw());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset());
    encoder.set_buffer(1, Some(weight.metal_buffer()), weight.offset());
    match bias {
        Some(b) => {
            let bias_contig = super::make_contiguous(b, registry, queue)?;
            let b = bias_contig.as_ref().unwrap_or(b);
            encoder.set_buffer(2, Some(b.metal_buffer()), b.offset());
        }
        None => {
            encoder.set_buffer(2, Some(&dummy_buf), 0);
        }
    }
    encoder.set_buffer(3, Some(out.metal_buffer()), out.offset());
    encoder.set_buffer(4, Some(&params_buf), 0);
    // Allocate dynamic threadgroup memory for weight_tile: TILE_CI * kH * kW floats
    let tile_ci: usize = 16; // must match TILE_CI in the shader
    let tg_mem_bytes = tile_ci * kh * kw * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, tg_mem_bytes as u64);
    // Grid: (W_out, C_out, B * H_out)
    let grid_size = MTLSize {
        width: w_out,
        height: c_out,
        depth: (batch * h_out),
    };
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(w_out, max_threads);
    let tg_y = std::cmp::min(c_out, max_threads / tg_x);
    let tg_z = std::cmp::min(batch * h_out, max_threads / (tg_x * tg_y));
    let threadgroup_size = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: tg_z,
    };

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end();
    super::commit_with_mode(&command_buffer, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(KernelRegistry, rmlx_metal::MtlQueue)> {
        let gpu_dev = crate::test_utils::test_gpu()?;
        let queue = gpu_dev.new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).unwrap();
        crate::ops::copy::register(&registry).unwrap();
        Some((registry, queue))
    }

    #[test]
    fn test_conv2d_tiled_identity_kernel() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        // Input: [1, 1, 3, 3], Weight: [1, 1, 1, 1] (identity kernel)
        let input = Array::from_slice(
            dev,
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let weight = Array::from_slice(dev, &[1.0f32], vec![1, 1, 1, 1]);

        let out = conv2d_tiled(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        let result = out.to_vec_checked::<f32>();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_conv2d_tiled_matches_naive() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        // Also register the non-tiled conv for comparison
        crate::ops::conv::register(&registry).unwrap();

        // Input: [1, 2, 4, 4], Weight: [1, 2, 3, 3]
        let input_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let weight_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.1).collect();

        let input = Array::from_slice(dev, &input_data, vec![1, 2, 4, 4]);
        let weight = Array::from_slice(dev, &weight_data, vec![1, 2, 3, 3]);

        let out_naive = crate::ops::conv::conv2d(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        let out_tiled = conv2d_tiled(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        assert_eq!(out_naive.shape(), out_tiled.shape());

        let naive_vec = out_naive.to_vec_checked::<f32>();
        let tiled_vec = out_tiled.to_vec_checked::<f32>();

        for (i, (n, t)) in naive_vec.iter().zip(tiled_vec.iter()).enumerate() {
            let diff = (n - t).abs();
            assert!(
                diff < 1e-4,
                "mismatch at index {}: naive={}, tiled={}, diff={}",
                i,
                n,
                t,
                diff
            );
        }
    }

    /// Verify that conv2d_tiled works with a 5x5 kernel (would overflow
    /// the old hardcoded `TILE_CI * 9` threadgroup allocation).
    #[test]
    fn test_conv2d_tiled_5x5_kernel() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        // Also register non-tiled conv for reference comparison
        crate::ops::conv::register(&registry).unwrap();

        // Input: [1, 2, 8, 8], Weight: [1, 2, 5, 5]
        let input_data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let weight_data: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.02).collect();

        let input = Array::from_slice(dev, &input_data, vec![1, 2, 8, 8]);
        let weight = Array::from_slice(dev, &weight_data, vec![1, 2, 5, 5]);

        let out_naive = crate::ops::conv::conv2d(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        let out_tiled = conv2d_tiled(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        assert_eq!(out_naive.shape(), out_tiled.shape());
        assert_eq!(out_tiled.shape(), &[1, 1, 4, 4]);

        let naive_vec = out_naive.to_vec_checked::<f32>();
        let tiled_vec = out_tiled.to_vec_checked::<f32>();

        for (i, (n, t)) in naive_vec.iter().zip(tiled_vec.iter()).enumerate() {
            let diff = (n - t).abs();
            assert!(
                diff < 1e-3,
                "5x5 mismatch at index {}: naive={}, tiled={}, diff={}",
                i,
                n,
                t,
                diff
            );
        }
    }

    /// Verify that conv2d_tiled works with a 7x7 kernel (would overflow
    /// the old hardcoded `TILE_CI * 9` threadgroup allocation).
    #[test]
    fn test_conv2d_tiled_7x7_kernel() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        // Also register non-tiled conv for reference comparison
        crate::ops::conv::register(&registry).unwrap();

        // Input: [1, 1, 10, 10], Weight: [1, 1, 7, 7]
        let input_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let weight_data: Vec<f32> = (0..49).map(|i| (i as f32 - 24.0) * 0.02).collect();

        let input = Array::from_slice(dev, &input_data, vec![1, 1, 10, 10]);
        let weight = Array::from_slice(dev, &weight_data, vec![1, 1, 7, 7]);

        let out_naive = crate::ops::conv::conv2d(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        let out_tiled = conv2d_tiled(
            &registry,
            &input,
            &weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &queue,
        )
        .unwrap();

        assert_eq!(out_naive.shape(), out_tiled.shape());
        assert_eq!(out_tiled.shape(), &[1, 1, 4, 4]);

        let naive_vec = out_naive.to_vec_checked::<f32>();
        let tiled_vec = out_tiled.to_vec_checked::<f32>();

        for (i, (n, t)) in naive_vec.iter().zip(tiled_vec.iter()).enumerate() {
            let diff = (n - t).abs();
            assert!(
                diff < 1e-3,
                "7x7 mismatch at index {}: naive={}, tiled={}, diff={}",
                i,
                n,
                t,
                diff
            );
        }
    }
}
