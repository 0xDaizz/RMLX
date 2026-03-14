//! FP8 wire format for MoE token exchange.
//!
//! Halves RDMA bandwidth by quantizing tokens to FP8 E4M3 before wire transfer.
//! f16 -> fp8 = 2x reduction, f32 -> fp8 = 4x reduction at TB5 16 GB/s.
//!
//! Per-token scale: `scale[i] = max(abs(token[i,:])) / 448.0`.
//! 4 bytes scale overhead per token (negligible vs D*2 byte token savings).

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice,
};
use rmlx_metal::{MTLResourceOptions, MTLSize, MtlBuffer};

// ---------------------------------------------------------------------------
// Fused dequant+scatter Metal shader
// ---------------------------------------------------------------------------

/// Metal shader for fused dequant+scatter: depacketize FP8, dequant with
/// per-token scale, and scatter to destination positions in one pass.
const DEQUANT_SCATTER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused dequant (FP8 E4M3 -> f16) + scatter.
//
// Each thread handles one element: (token_idx, col).
// Reads FP8 byte from packet_data, dequantizes using per-token scale,
// writes to output[scatter_indices[token_idx], col].
//
// 2D grid: tid.x = column, tid.y = token index.

kernel void dequant_scatter_fp8e4m3(
    device const uchar* packet_data     [[buffer(0)]],  // [N_tokens * D] FP8
    device const float* scales          [[buffer(1)]],  // [N_tokens] per-token scales
    device const uint*  scatter_indices [[buffer(2)]],  // [N_tokens] dest positions
    device half*        output          [[buffer(3)]],  // [total_tokens, D] output
    constant uint&      num_tokens      [[buffer(4)]],
    constant uint&      hidden_dim      [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint token_idx = tid.y;
    uint col = tid.x;
    if (token_idx >= num_tokens || col >= hidden_dim) return;

    uint src_offset = token_idx * hidden_dim + col;
    float scale = scales[token_idx];

    // Dequantize FP8 E4M3 -> float
    uchar bits = packet_data[src_offset];
    uint sign_bit = (bits >> 7) & 1;
    uint exp  = (bits >> 3) & 0xF;
    uint mant = bits & 0x7;

    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else if (exp == 0) {
        val = (mant / 8.0f) * exp2(-6.0f);
    } else if (exp == 15 && mant == 7) {
        val = NAN;
    } else {
        val = (1.0f + mant / 8.0f) * exp2(float(exp) - 7.0f);
    }
    if (sign_bit) val = -val;

    // Re-apply scale and scatter
    uint dest_row = scatter_indices[token_idx];
    uint dst_offset = dest_row * hidden_dim + col;
    output[dst_offset] = half(val * scale);
}
"#;

// ---------------------------------------------------------------------------
// Payload type
// ---------------------------------------------------------------------------

/// FP8 dispatch packet: quantized tokens + scales for wire transfer.
pub struct Fp8DispatchPayload {
    /// FP8 E4M3 quantized token data: [num_tokens, hidden_dim] as uint8.
    pub fp8_data: Array,
    /// Per-token scale factors: [num_tokens] as Float32.
    pub scales: Array,
    /// Number of tokens.
    pub num_tokens: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl Fp8DispatchPayload {
    /// Total wire bytes: FP8 data + scales.
    pub fn wire_bytes(&self) -> usize {
        // fp8_data: num_tokens * hidden_dim * 1 byte
        // scales: num_tokens * 4 bytes
        self.num_tokens * self.hidden_dim + self.num_tokens * 4
    }

    /// Pack FP8 payload into interleaved wire format:
    /// `[fp8_token0: D bytes][scale0: 4 bytes][fp8_token1: D bytes][scale1: 4 bytes]...`
    ///
    /// This interleaved format is compatible with `route_rdma`'s `token_stride = hidden_dim + 4`.
    pub fn pack_for_wire(&self) -> Vec<u8> {
        let stride = self.hidden_dim + 4; // D bytes FP8 + 4 bytes f32 scale
        let mut wire = Vec::with_capacity(self.num_tokens * stride);
        let fp8_bytes = self.fp8_data.to_bytes();
        let scale_bytes = self.scales.to_bytes();
        for i in 0..self.num_tokens {
            let fp8_start = i * self.hidden_dim;
            wire.extend_from_slice(&fp8_bytes[fp8_start..fp8_start + self.hidden_dim]);
            let scale_start = i * 4;
            wire.extend_from_slice(&scale_bytes[scale_start..scale_start + 4]);
        }
        wire
    }
}

/// Unpack interleaved FP8 wire format back to separate `(fp8_bytes, scale_bytes)`.
///
/// Returns `(fp8_data_flat: Vec<u8>, scales_flat: Vec<u8>)` ready for
/// `Array::from_bytes` reconstruction.
///
/// Returns an error if `wire_data` is too short for the expected
/// `num_tokens * (hidden_dim + 4)` bytes.
pub fn unpack_from_wire(
    wire_data: &[u8],
    num_tokens: usize,
    hidden_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>), KernelError> {
    let stride = hidden_dim + 4;
    let expected_len = num_tokens * stride;
    if wire_data.len() < expected_len {
        return Err(KernelError::InvalidShape(format!(
            "FP8 unpack: wire_data length {} < expected {} (tokens={}, stride={})",
            wire_data.len(),
            expected_len,
            num_tokens,
            stride
        )));
    }
    let mut fp8_data = Vec::with_capacity(num_tokens * hidden_dim);
    let mut scales = Vec::with_capacity(num_tokens * 4);
    for i in 0..num_tokens {
        let off = i * stride;
        fp8_data.extend_from_slice(&wire_data[off..off + hidden_dim]);
        scales.extend_from_slice(&wire_data[off + hidden_dim..off + stride]);
    }
    Ok((fp8_data, scales))
}

/// Returns the per-token wire stride when FP8 is enabled: `hidden_dim` (FP8 bytes) + 4 (f32 scale).
pub fn wire_token_stride(hidden_dim: usize) -> usize {
    hidden_dim + 4
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register the fused dequant+scatter shader.
///
/// Call this once during initialization, after `rmlx_core::ops::fp8::register`.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source_if_absent("fp8_exchange", DEQUANT_SCATTER_SHADER)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Quantize dispatch tokens to FP8 for wire transfer.
///
/// Input: [N, D] Float16 (or Float32, which will cause an error -- promote
/// to f16 externally before calling).
/// Output: `Fp8DispatchPayload` with `fp8_data` and per-token scales.
pub fn quantize_for_dispatch(
    registry: &KernelRegistry,
    tokens: &Array,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Fp8DispatchPayload, KernelError> {
    if tokens.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "quantize_for_dispatch: expected Float16 tokens, got {:?}. \
             Promote to f16 before calling.",
            tokens.dtype()
        )));
    }
    if tokens.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "quantize_for_dispatch: expected 2D input [N, D], got {}D",
            tokens.ndim()
        )));
    }

    let num_tokens = tokens.shape()[0];
    let hidden_dim = tokens.shape()[1];

    let (fp8_data, scales) = ops::fp8::quant_per_token_fp8e4m3(registry, tokens, queue)?;

    Ok(Fp8DispatchPayload {
        fp8_data,
        scales,
        num_tokens,
        hidden_dim,
    })
}

/// Dequantize received FP8 tokens back to Float16.
///
/// Input: `fp8_data` [N, D] as FP8 E4M3, `scales` [N] as Float32.
/// Output: [N, D] Float16.
pub fn dequantize_received(
    registry: &KernelRegistry,
    fp8_data: &Array,
    scales: &Array,
    hidden_dim: usize,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if fp8_data.dtype() != DType::Float8E4M3 {
        return Err(KernelError::InvalidShape(format!(
            "dequantize_received: expected Float8E4M3 fp8_data, got {:?}",
            fp8_data.dtype()
        )));
    }

    // Ensure fp8_data is viewed as 2D [N, D] if needed
    let num_elements = fp8_data.numel();
    if num_elements == 0 || hidden_dim == 0 {
        return Err(KernelError::InvalidShape(
            "dequantize_received: empty input".into(),
        ));
    }

    let num_tokens = num_elements / hidden_dim;
    if num_elements % hidden_dim != 0 {
        return Err(KernelError::InvalidShape(format!(
            "dequantize_received: fp8_data numel ({}) not divisible by hidden_dim ({})",
            num_elements, hidden_dim
        )));
    }

    // If fp8_data is not already shaped [N, D], reshape it
    let fp8_2d = if fp8_data.ndim() == 2
        && fp8_data.shape()[0] == num_tokens
        && fp8_data.shape()[1] == hidden_dim
    {
        None
    } else {
        Some(fp8_data.reshape(vec![num_tokens, hidden_dim])?)
    };
    let fp8_ref = fp8_2d.as_ref().unwrap_or(fp8_data);

    ops::fp8::dequant_per_token_fp8e4m3(registry, fp8_ref, scales, queue)
}

/// Compute wire bytes for FP8 vs raw transfer.
///
/// Returns `(fp8_bytes, raw_bytes)` for bandwidth comparison.
///
/// - `fp8_bytes` = `num_tokens * hidden_dim` (1 byte/element) + `num_tokens * 4` (scales)
/// - `raw_bytes` = `num_tokens * hidden_dim * dtype.size_of()`
pub fn wire_bytes(num_tokens: usize, hidden_dim: usize, dtype: DType) -> (usize, usize) {
    let fp8_bytes = num_tokens * hidden_dim + num_tokens * 4; // data + scales
    let raw_bytes = num_tokens * hidden_dim * dtype.size_of();
    (fp8_bytes, raw_bytes)
}

// ---------------------------------------------------------------------------
// Fused dequant + scatter
// ---------------------------------------------------------------------------

/// Create a constant `uint` buffer on the device.
fn make_u32_buf(device: &ProtocolObject<dyn MTLDevice>, val: u32) -> MtlBuffer {
    // SAFETY: addr_of! produces a valid, aligned pointer to `val` which lives
    // for the duration of this function. The Metal API copies the bytes into the
    // buffer before returning, so the pointer does not need to outlive this call.
    let ptr = std::ptr::NonNull::new(std::ptr::addr_of!(val) as *mut std::ffi::c_void).unwrap();
    unsafe {
        device.newBufferWithBytes_length_options(ptr, 4, MTLResourceOptions::StorageModeShared)
    }
    .unwrap()
}

/// Fused receive kernel: dequant + scatter in one pass.
///
/// Takes packed FP8 data, per-token scales, and scatter indices, then
/// dequantizes from FP8 E4M3 to Float16 and scatters to destination
/// positions in the output buffer -- all in a single GPU kernel dispatch.
///
/// - `packet_data`: [N_tokens * hidden_dim] packed FP8 bytes (contiguous).
/// - `scales`: [N_tokens] per-token scale factors as Float32.
/// - `scatter_indices`: [N_tokens] destination row indices as UInt32.
/// - `hidden_dim`: number of columns per token.
/// - `output`: [total_tokens, hidden_dim] pre-allocated Float16 output buffer.
/// - `queue`: Metal command queue for synchronous execution.
pub fn dequant_scatter_fp8e4m3(
    registry: &KernelRegistry,
    packet_data: &Array,
    scales: &Array,
    scatter_indices: &Array,
    hidden_dim: usize,
    output: &Array,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<(), KernelError> {
    // Validate packet_data
    if packet_data.dtype() != DType::Float8E4M3 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: expected Float8E4M3 packet_data, got {:?}",
            packet_data.dtype()
        )));
    }
    if !packet_data.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "dequant_scatter_fp8e4m3: packet_data must be contiguous".into(),
        ));
    }

    let total_fp8_elements = packet_data.numel();
    if hidden_dim == 0 || total_fp8_elements % hidden_dim != 0 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: packet_data numel ({}) not divisible by hidden_dim ({})",
            total_fp8_elements, hidden_dim
        )));
    }
    let num_tokens = total_fp8_elements / hidden_dim;

    // Validate scales
    if scales.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: expected Float32 scales, got {:?}",
            scales.dtype()
        )));
    }
    if scales.numel() != num_tokens {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: scales numel ({}) != num_tokens ({})",
            scales.numel(),
            num_tokens
        )));
    }

    // Validate scatter_indices
    if scatter_indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: expected UInt32 scatter_indices, got {:?}",
            scatter_indices.dtype()
        )));
    }
    if scatter_indices.numel() != num_tokens {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: scatter_indices numel ({}) != num_tokens ({})",
            scatter_indices.numel(),
            num_tokens
        )));
    }

    // Validate output
    if output.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_scatter_fp8e4m3: expected Float16 output, got {:?}",
            output.dtype()
        )));
    }

    // Ensure the fused shader is registered
    register(registry)?;

    let pipeline = registry.get_pipeline("dequant_scatter_fp8e4m3", DType::Float8E4M3)?;
    let device = registry.device().raw();

    let tokens_buf = make_u32_buf(device, num_tokens as u32);
    let dim_buf = make_u32_buf(device, hidden_dim as u32);

    let command_buffer = queue.commandBuffer().unwrap();
    let encoder = command_buffer.computeCommandEncoder().unwrap();
    encoder.setComputePipelineState(&pipeline);
    // SAFETY: all buffers are valid MTLBuffers with sizes validated above.
    // Buffer indices match the dequant_scatter_fp8e4m3 Metal kernel signature.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(packet_data.metal_buffer()), packet_data.offset(), 0);
        encoder.setBuffer_offset_atIndex(Some(scales.metal_buffer()), scales.offset(), 1);
        encoder.setBuffer_offset_atIndex(
            Some(scatter_indices.metal_buffer()),
            scatter_indices.offset(),
            2,
        );
        encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), output.offset(), 3);
        encoder.setBuffer_offset_atIndex(Some(&tokens_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&dim_buf), 0, 5);
    }

    let grid_size = MTLSize {
        width: hidden_dim,
        height: num_tokens,
        depth: 1,
    };
    let tg_w = std::cmp::min(hidden_dim, pipeline.maxTotalThreadsPerThreadgroup());
    let threadgroup_size = MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_bytes_f16() {
        let (fp8, raw) = wire_bytes(128, 4096, DType::Float16);
        // fp8: 128 * 4096 * 1 + 128 * 4 = 524_288 + 512 = 524_800
        assert_eq!(fp8, 524_800);
        // raw: 128 * 4096 * 2 = 1_048_576
        assert_eq!(raw, 1_048_576);
        // ~2x reduction
        assert!(fp8 < raw);
    }

    #[test]
    fn test_wire_bytes_f32() {
        let (fp8, raw) = wire_bytes(64, 8192, DType::Float32);
        // fp8: 64 * 8192 + 64 * 4 = 524_288 + 256 = 524_544
        assert_eq!(fp8, 524_544);
        // raw: 64 * 8192 * 4 = 2_097_152
        assert_eq!(raw, 2_097_152);
        // ~4x reduction
        assert!(fp8 * 4 < raw * 5); // within 4x with small scale overhead
    }

    #[test]
    fn test_wire_bytes_zero_tokens() {
        let (fp8, raw) = wire_bytes(0, 4096, DType::Float16);
        assert_eq!(fp8, 0);
        assert_eq!(raw, 0);
    }

    #[test]
    fn test_wire_token_stride() {
        assert_eq!(wire_token_stride(4096), 4100);
        assert_eq!(wire_token_stride(8192), 8196);
        assert_eq!(wire_token_stride(0), 4);
    }

    #[test]
    fn test_unpack_from_wire_roundtrip() {
        // Simulate interleaved wire format: 2 tokens, hidden_dim=4
        let hidden_dim = 4;
        let num_tokens = 2;
        let stride = hidden_dim + 4;

        // Build wire data manually:
        // token0: fp8=[0x10, 0x20, 0x30, 0x40], scale=[1.0 as f32 LE bytes]
        // token1: fp8=[0x50, 0x60, 0x70, 0x80], scale=[2.0 as f32 LE bytes]
        let mut wire = Vec::with_capacity(num_tokens * stride);
        wire.extend_from_slice(&[0x10, 0x20, 0x30, 0x40]);
        wire.extend_from_slice(&1.0f32.to_le_bytes());
        wire.extend_from_slice(&[0x50, 0x60, 0x70, 0x80]);
        wire.extend_from_slice(&2.0f32.to_le_bytes());

        let (fp8_data, scales) = unpack_from_wire(&wire, num_tokens, hidden_dim).unwrap();

        assert_eq!(
            fp8_data,
            vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
        );
        assert_eq!(scales.len(), 8); // 2 tokens * 4 bytes each

        // Verify scale values
        let s0 = f32::from_le_bytes(scales[0..4].try_into().unwrap());
        let s1 = f32::from_le_bytes(scales[4..8].try_into().unwrap());
        assert_eq!(s0, 1.0);
        assert_eq!(s1, 2.0);
    }

    #[test]
    fn test_unpack_from_wire_truncated_input() {
        // Wire data too short for 2 tokens with hidden_dim=4 (expects 16 bytes)
        let short_wire = vec![0u8; 10];
        let result = unpack_from_wire(&short_wire, 2, 4);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("wire_data length 10 < expected 16"),
            "unexpected error: {err_msg}"
        );
    }
}
