//! Convolution operations (Conv1d and Conv2d) with Metal GPU kernels.
//!
//! Uses implicit GEMM approach (no im2col) for both 1D and 2D convolutions.
//! Supports groups, stride, padding, dilation, and optional bias.
//!
//! ## Layout conventions
//! - Conv1d: Input [B, C_in, W], Weight [C_out, C_in/groups, K], Output [B, C_out, W_out]
//! - Conv2d: Input [B, C_in, H, W], Weight [C_out, C_in/groups, kH, kW], Output [B, C_out, H_out, W_out]

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for convolution kernels.
pub const CONV_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Conv1d kernels
// ============================================================================
// Input:  [B, C_in, W]
// Weight: [C_out, C_in/groups, K]
// Output: [B, C_out, W_out]
// W_out = (W + 2*padding - dilation*(K-1) - 1) / stride + 1
//
// Params buffer layout (11 uint values):
//   [B, C_in, W, C_out, K, W_out, stride, padding, dilation, groups, has_bias]
//
// Grid: (W_out, C_out, B)

kernel void conv1d_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint*      params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint w_out_idx = gid.x;
    uint c_out_idx = gid.y;
    uint b_idx     = gid.z;

    uint B_       = params[0];
    uint C_in     = params[1];
    uint W        = params[2];
    uint C_out    = params[3];
    uint K        = params[4];
    uint W_out    = params[5];
    uint stride   = params[6];
    uint padding  = params[7];
    uint dilation = params[8];
    uint groups   = params[9];
    uint has_bias = params[10];

    if (b_idx >= B_ || c_out_idx >= C_out || w_out_idx >= W_out) return;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint ki = 0; ki < K; ki++) {
            int w_in = int(w_out_idx * stride + ki * dilation) - int(padding);
            if (w_in >= 0 && uint(w_in) < W) {
                float inp = input[b_idx * C_in * W + c_in_idx * W + uint(w_in)];
                float wgt = weight[c_out_idx * c_in_count * K + ci * K + ki];
                sum += inp * wgt;
            }
        }
    }
    if (has_bias) sum += bias[c_out_idx];
    output[b_idx * C_out * W_out + c_out_idx * W_out + w_out_idx] = sum;
}

kernel void conv1d_f16(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half*       output  [[buffer(3)]],
    constant uint*     params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint w_out_idx = gid.x;
    uint c_out_idx = gid.y;
    uint b_idx     = gid.z;

    uint B_       = params[0];
    uint C_in     = params[1];
    uint W        = params[2];
    uint C_out    = params[3];
    uint K        = params[4];
    uint W_out    = params[5];
    uint stride   = params[6];
    uint padding  = params[7];
    uint dilation = params[8];
    uint groups   = params[9];
    uint has_bias = params[10];

    if (b_idx >= B_ || c_out_idx >= C_out || w_out_idx >= W_out) return;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint ki = 0; ki < K; ki++) {
            int w_in = int(w_out_idx * stride + ki * dilation) - int(padding);
            if (w_in >= 0 && uint(w_in) < W) {
                float inp = float(input[b_idx * C_in * W + c_in_idx * W + uint(w_in)]);
                float wgt = float(weight[c_out_idx * c_in_count * K + ci * K + ki]);
                sum += inp * wgt;
            }
        }
    }
    if (has_bias) sum += float(bias[c_out_idx]);
    output[b_idx * C_out * W_out + c_out_idx * W_out + w_out_idx] = half(sum);
}

kernel void conv1d_bf16(
    device const bfloat* input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device bfloat*       output  [[buffer(3)]],
    constant uint*       params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint w_out_idx = gid.x;
    uint c_out_idx = gid.y;
    uint b_idx     = gid.z;

    uint B_       = params[0];
    uint C_in     = params[1];
    uint W        = params[2];
    uint C_out    = params[3];
    uint K        = params[4];
    uint W_out    = params[5];
    uint stride   = params[6];
    uint padding  = params[7];
    uint dilation = params[8];
    uint groups   = params[9];
    uint has_bias = params[10];

    if (b_idx >= B_ || c_out_idx >= C_out || w_out_idx >= W_out) return;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint ki = 0; ki < K; ki++) {
            int w_in = int(w_out_idx * stride + ki * dilation) - int(padding);
            if (w_in >= 0 && uint(w_in) < W) {
                float inp = float(input[b_idx * C_in * W + c_in_idx * W + uint(w_in)]);
                float wgt = float(weight[c_out_idx * c_in_count * K + ci * K + ki]);
                sum += inp * wgt;
            }
        }
    }
    if (has_bias) sum += float(bias[c_out_idx]);
    output[b_idx * C_out * W_out + c_out_idx * W_out + w_out_idx] = bfloat(sum);
}

// ============================================================================
// Conv2d kernels
// ============================================================================
// Input:  [B, C_in, H, W]
// Weight: [C_out, C_in/groups, kH, kW]
// Output: [B, C_out, H_out, W_out]
// H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1
// W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1
//
// Params buffer layout (17 uint values):
//   [B, C_in, H, W, C_out, kH, kW, H_out, W_out,
//    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, has_bias]
//
// Grid: (H_out * W_out, C_out, B)

kernel void conv2d_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint*      params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint spatial_idx = gid.x;
    uint c_out_idx   = gid.y;
    uint b_idx       = gid.z;

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

    if (b_idx >= B_ || c_out_idx >= C_out || spatial_idx >= H_out * W_out) return;

    uint h_out_idx = spatial_idx / W_out;
    uint w_out_idx = spatial_idx % W_out;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint khi = 0; khi < kH; khi++) {
            int h_in = int(h_out_idx * stride_h + khi * dil_h) - int(pad_h);
            if (h_in < 0 || uint(h_in) >= H) continue;
            for (uint kwi = 0; kwi < kW; kwi++) {
                int w_in = int(w_out_idx * stride_w + kwi * dil_w) - int(pad_w);
                if (w_in >= 0 && uint(w_in) < W) {
                    float inp = input[b_idx * C_in * H * W + c_in_idx * H * W + uint(h_in) * W + uint(w_in)];
                    float wgt = weight[c_out_idx * c_in_count * kH * kW + ci * kH * kW + khi * kW + kwi];
                    sum += inp * wgt;
                }
            }
        }
    }
    if (has_bias) sum += bias[c_out_idx];
    output[b_idx * C_out * H_out * W_out + c_out_idx * H_out * W_out + h_out_idx * W_out + w_out_idx] = sum;
}

kernel void conv2d_f16(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half*       output  [[buffer(3)]],
    constant uint*     params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint spatial_idx = gid.x;
    uint c_out_idx   = gid.y;
    uint b_idx       = gid.z;

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

    if (b_idx >= B_ || c_out_idx >= C_out || spatial_idx >= H_out * W_out) return;

    uint h_out_idx = spatial_idx / W_out;
    uint w_out_idx = spatial_idx % W_out;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint khi = 0; khi < kH; khi++) {
            int h_in = int(h_out_idx * stride_h + khi * dil_h) - int(pad_h);
            if (h_in < 0 || uint(h_in) >= H) continue;
            for (uint kwi = 0; kwi < kW; kwi++) {
                int w_in = int(w_out_idx * stride_w + kwi * dil_w) - int(pad_w);
                if (w_in >= 0 && uint(w_in) < W) {
                    float inp = float(input[b_idx * C_in * H * W + c_in_idx * H * W + uint(h_in) * W + uint(w_in)]);
                    float wgt = float(weight[c_out_idx * c_in_count * kH * kW + ci * kH * kW + khi * kW + kwi]);
                    sum += inp * wgt;
                }
            }
        }
    }
    if (has_bias) sum += float(bias[c_out_idx]);
    output[b_idx * C_out * H_out * W_out + c_out_idx * H_out * W_out + h_out_idx * W_out + w_out_idx] = half(sum);
}

kernel void conv2d_bf16(
    device const bfloat* input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device bfloat*       output  [[buffer(3)]],
    constant uint*       params  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint spatial_idx = gid.x;
    uint c_out_idx   = gid.y;
    uint b_idx       = gid.z;

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

    if (b_idx >= B_ || c_out_idx >= C_out || spatial_idx >= H_out * W_out) return;

    uint h_out_idx = spatial_idx / W_out;
    uint w_out_idx = spatial_idx % W_out;

    uint group = c_out_idx / (C_out / groups);
    uint c_in_start = group * (C_in / groups);
    uint c_in_count = C_in / groups;

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in_count; ci++) {
        uint c_in_idx = c_in_start + ci;
        for (uint khi = 0; khi < kH; khi++) {
            int h_in = int(h_out_idx * stride_h + khi * dil_h) - int(pad_h);
            if (h_in < 0 || uint(h_in) >= H) continue;
            for (uint kwi = 0; kwi < kW; kwi++) {
                int w_in = int(w_out_idx * stride_w + kwi * dil_w) - int(pad_w);
                if (w_in >= 0 && uint(w_in) < W) {
                    float inp = float(input[b_idx * C_in * H * W + c_in_idx * H * W + uint(h_in) * W + uint(w_in)]);
                    float wgt = float(weight[c_out_idx * c_in_count * kH * kW + ci * kH * kW + khi * kW + kwi]);
                    sum += inp * wgt;
                }
            }
        }
    }
    if (has_bias) sum += float(bias[c_out_idx]);
    output[b_idx * C_out * H_out * W_out + c_out_idx * H_out * W_out + h_out_idx * W_out + w_out_idx] = bfloat(sum);
}
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the kernel function name for a conv1d operation given the dtype.
fn conv1d_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("conv1d_f32"),
        DType::Float16 => Ok("conv1d_f16"),
        DType::Bfloat16 => Ok("conv1d_bf16"),
        _ => Err(KernelError::InvalidShape(format!(
            "conv1d: unsupported dtype {:?}",
            dtype
        ))),
    }
}

/// Return the kernel function name for a conv2d operation given the dtype.
fn conv2d_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("conv2d_f32"),
        DType::Float16 => Ok("conv2d_f16"),
        DType::Bfloat16 => Ok("conv2d_bf16"),
        _ => Err(KernelError::InvalidShape(format!(
            "conv2d: unsupported dtype {:?}",
            dtype
        ))),
    }
}

/// Create a Metal buffer containing a `[u32]` parameter array.
fn make_params_buf(device: &metal::DeviceRef, params: &[u32]) -> metal::Buffer {
    let byte_len = std::mem::size_of_val(params) as u64;
    device.new_buffer_with_data(
        params.as_ptr() as *const std::ffi::c_void,
        byte_len,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Create a tiny dummy buffer (4 bytes) for the bias slot when bias is absent.
fn make_dummy_buf(device: &metal::DeviceRef) -> metal::Buffer {
    device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register convolution kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("conv", CONV_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 1D convolution.
///
/// # Arguments
/// - `input`: `[B, C_in, W]`
/// - `weight`: `[C_out, C_in/groups, K]`
/// - `bias`: optional `[C_out]`
/// - `stride`, `padding`, `dilation`, `groups`: convolution parameters
///
/// # Returns
/// `[B, C_out, W_out]` where `W_out = (W + 2*padding - dilation*(K-1) - 1) / stride + 1`
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    registry: &KernelRegistry,
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate shapes ---
    if input.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: input must be 3D [B, C_in, W], got {}D",
            input.ndim()
        )));
    }
    if weight.ndim() != 3 {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: weight must be 3D [C_out, C_in/groups, K], got {}D",
            weight.ndim()
        )));
    }
    if input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: input dtype {:?} != weight dtype {:?}",
            input.dtype(),
            weight.dtype()
        )));
    }

    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let w = input.shape()[2];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let k = weight.shape()[2];

    if groups == 0 {
        return Err(KernelError::InvalidShape(
            "conv1d: groups must be >= 1".into(),
        ));
    }
    if c_in % groups != 0 {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: C_in ({c_in}) must be divisible by groups ({groups})"
        )));
    }
    if c_out % groups != 0 {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: C_out ({c_out}) must be divisible by groups ({groups})"
        )));
    }
    if c_in_per_group != c_in / groups {
        return Err(KernelError::InvalidShape(format!(
            "conv1d: weight shape[1]={c_in_per_group} != C_in/groups={}",
            c_in / groups
        )));
    }

    if let Some(b) = bias {
        if b.ndim() != 1 || b.shape()[0] != c_out {
            return Err(KernelError::InvalidShape(format!(
                "conv1d: bias must be [C_out={}], got {:?}",
                c_out,
                b.shape()
            )));
        }
        if b.dtype() != input.dtype() {
            return Err(KernelError::InvalidShape(format!(
                "conv1d: bias dtype {:?} != input dtype {:?}",
                b.dtype(),
                input.dtype()
            )));
        }
    }

    // Compute output width
    let w_out = (w + 2 * padding)
        .checked_sub(dilation * (k - 1) + 1)
        .map(|v| v / stride + 1)
        .ok_or_else(|| {
            KernelError::InvalidShape(format!(
                "conv1d: output width is non-positive (W={w}, K={k}, padding={padding}, dilation={dilation}, stride={stride})"
            ))
        })?;

    if w_out == 0 {
        return Err(KernelError::InvalidShape(
            "conv1d: output width is zero".into(),
        ));
    }

    // Ensure contiguous
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let weight_contig = super::make_contiguous(weight, registry, queue)?;
    let weight = weight_contig.as_ref().unwrap_or(weight);

    let kernel_name = conv1d_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let out_shape = vec![batch, c_out, w_out];
    let out = Array::zeros(registry.device().raw(), &out_shape, input.dtype());

    let has_bias: u32 = if bias.is_some() { 1 } else { 0 };
    let params: Vec<u32> = vec![
        batch as u32,
        c_in as u32,
        w as u32,
        c_out as u32,
        k as u32,
        w_out as u32,
        stride as u32,
        padding as u32,
        dilation as u32,
        groups as u32,
        has_bias,
    ];
    let params_buf = make_params_buf(registry.device().raw(), &params);
    let dummy_buf = make_dummy_buf(registry.device().raw());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(weight.metal_buffer()), weight.offset() as u64);
    match bias {
        Some(b) => {
            let bias_contig = super::make_contiguous(b, registry, queue)?;
            let b = bias_contig.as_ref().unwrap_or(b);
            encoder.set_buffer(2, Some(b.metal_buffer()), b.offset() as u64);
        }
        None => {
            encoder.set_buffer(2, Some(&dummy_buf), 0);
        }
    }
    encoder.set_buffer(3, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(4, Some(&params_buf), 0);

    // Grid: (W_out, C_out, B)
    let grid_size = MTLSize::new(w_out as u64, c_out as u64, batch as u64);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    // Choose threadgroup dimensions: fit as many spatial threads as possible
    let tg_x = std::cmp::min(w_out as u64, max_threads);
    let tg_y = std::cmp::min(c_out as u64, max_threads / tg_x);
    let tg_z = std::cmp::min(batch as u64, max_threads / (tg_x * tg_y));
    let threadgroup_size = MTLSize::new(tg_x, tg_y, tg_z);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// 2D convolution.
///
/// # Arguments
/// - `input`: `[B, C_in, H, W]`
/// - `weight`: `[C_out, C_in/groups, kH, kW]`
/// - `bias`: optional `[C_out]`
/// - `stride`: `(stride_h, stride_w)`
/// - `padding`: `(pad_h, pad_w)`
/// - `dilation`: `(dil_h, dil_w)`
/// - `groups`: number of groups
///
/// # Returns
/// `[B, C_out, H_out, W_out]`
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    registry: &KernelRegistry,
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate shapes ---
    if input.ndim() != 4 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: input must be 4D [B, C_in, H, W], got {}D",
            input.ndim()
        )));
    }
    if weight.ndim() != 4 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: weight must be 4D [C_out, C_in/groups, kH, kW], got {}D",
            weight.ndim()
        )));
    }
    if input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: input dtype {:?} != weight dtype {:?}",
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

    if groups == 0 {
        return Err(KernelError::InvalidShape(
            "conv2d: groups must be >= 1".into(),
        ));
    }
    if c_in % groups != 0 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: C_in ({c_in}) must be divisible by groups ({groups})"
        )));
    }
    if c_out % groups != 0 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: C_out ({c_out}) must be divisible by groups ({groups})"
        )));
    }
    if c_in_per_group != c_in / groups {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: weight shape[1]={c_in_per_group} != C_in/groups={}",
            c_in / groups
        )));
    }

    if let Some(b) = bias {
        if b.ndim() != 1 || b.shape()[0] != c_out {
            return Err(KernelError::InvalidShape(format!(
                "conv2d: bias must be [C_out={}], got {:?}",
                c_out,
                b.shape()
            )));
        }
        if b.dtype() != input.dtype() {
            return Err(KernelError::InvalidShape(format!(
                "conv2d: bias dtype {:?} != input dtype {:?}",
                b.dtype(),
                input.dtype()
            )));
        }
    }

    // Compute output dimensions
    let h_out = (h + 2 * pad_h)
        .checked_sub(dil_h * (kh - 1) + 1)
        .map(|v| v / stride_h + 1)
        .ok_or_else(|| {
            KernelError::InvalidShape(format!(
                "conv2d: output height is non-positive (H={h}, kH={kh}, pad_h={pad_h}, dil_h={dil_h}, stride_h={stride_h})"
            ))
        })?;
    let w_out = (w + 2 * pad_w)
        .checked_sub(dil_w * (kw - 1) + 1)
        .map(|v| v / stride_w + 1)
        .ok_or_else(|| {
            KernelError::InvalidShape(format!(
                "conv2d: output width is non-positive (W={w}, kW={kw}, pad_w={pad_w}, dil_w={dil_w}, stride_w={stride_w})"
            ))
        })?;

    if h_out == 0 || w_out == 0 {
        return Err(KernelError::InvalidShape(format!(
            "conv2d: output spatial dims are zero (H_out={h_out}, W_out={w_out})"
        )));
    }

    // Ensure contiguous
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let weight_contig = super::make_contiguous(weight, registry, queue)?;
    let weight = weight_contig.as_ref().unwrap_or(weight);

    let kernel_name = conv2d_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let out_shape = vec![batch, c_out, h_out, w_out];
    let out = Array::zeros(registry.device().raw(), &out_shape, input.dtype());

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

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(weight.metal_buffer()), weight.offset() as u64);
    match bias {
        Some(b) => {
            let bias_contig = super::make_contiguous(b, registry, queue)?;
            let b = bias_contig.as_ref().unwrap_or(b);
            encoder.set_buffer(2, Some(b.metal_buffer()), b.offset() as u64);
        }
        None => {
            encoder.set_buffer(2, Some(&dummy_buf), 0);
        }
    }
    encoder.set_buffer(3, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(4, Some(&params_buf), 0);

    // Grid: (H_out * W_out, C_out, B)
    let spatial = (h_out * w_out) as u64;
    let grid_size = MTLSize::new(spatial, c_out as u64, batch as u64);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let tg_x = std::cmp::min(spatial, max_threads);
    let tg_y = std::cmp::min(c_out as u64, max_threads / tg_x);
    let tg_z = std::cmp::min(batch as u64, max_threads / (tg_x * tg_y));
    let threadgroup_size = MTLSize::new(tg_x, tg_y, tg_z);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}
