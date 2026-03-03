//! Quantized matrix-vector multiplication using MLX affine quantization format.
//!
//! MLX affine quantization stores weights as three separate buffers:
//! - `weights`: packed uint32 array (multiple quantized values per uint32)
//! - `scales`: per-group float/half scale factors
//! - `biases`: per-group float/half bias terms
//!
//! Dequantization formula: `value = scale * quantized_int + bias`
//!
//! Supported bit widths: 2, 3, 4, 6, 8
//! Supported group sizes: 32, 64, 128

use metal::Buffer as MTLBuffer;

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

// ---------------------------------------------------------------------------
// Deprecated GGML-format block structs (kept for backward compatibility)
// ---------------------------------------------------------------------------

/// Quantized block for Q4_0: 32 elements -> 18 bytes (2 bytes scale + 16 bytes data).
///
/// DEPRECATED: This uses the legacy GGML interleaved format. Prefer
/// [`QuantizedWeight`] with the MLX affine format for new code.
#[repr(C)]
pub struct BlockQ4_0 {
    pub scale: u16, // f16 stored as u16
    pub data: [u8; 16],
}

/// Quantized block for Q8_0: 32 elements -> 34 bytes (2 bytes scale + 32 bytes data).
///
/// DEPRECATED: This uses the legacy GGML interleaved format. Prefer
/// [`QuantizedWeight`] with the MLX affine format for new code.
#[repr(C)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 stored as u16
    pub data: [u8; 32],
}

// ---------------------------------------------------------------------------
// MLX affine quantization format
// ---------------------------------------------------------------------------

/// Supported quantization bit widths.
const VALID_BITS: &[u32] = &[2, 3, 4, 6, 8];

/// Supported group sizes.
const VALID_GROUP_SIZES: &[u32] = &[32, 64, 128];

/// Quantized weight matrix in MLX affine format.
///
/// Stores three separate Metal buffers rather than interleaved blocks:
/// - `weights_buf`: packed uint32 data. Each uint32 holds `32 / bits` quantized values.
/// - `scales_buf`: one float32 per group (per-group scale factor).
/// - `biases_buf`: one float32 per group (per-group bias / zero-point).
///
/// Dequantization: `w_i = scale_g * q_i + bias_g`
/// where `g = i / group_size`.
pub struct QuantizedWeight {
    /// Packed uint32 weight data.
    pub weights_buf: MTLBuffer,
    /// Per-group scale factors (float32).
    pub scales_buf: MTLBuffer,
    /// Per-group bias terms (float32).
    pub biases_buf: MTLBuffer,
    /// Number of elements per quantization group (32, 64, or 128).
    pub group_size: u32,
    /// Bit width of each quantized value (2, 3, 4, 6, or 8).
    pub bits: u32,
    /// Number of output rows (weight matrix rows).
    pub out_features: usize,
    /// Number of input columns (weight matrix columns).
    pub in_features: usize,
}

impl QuantizedWeight {
    /// Create a new `QuantizedWeight` with validation.
    pub fn new(
        weights_buf: MTLBuffer,
        scales_buf: MTLBuffer,
        biases_buf: MTLBuffer,
        group_size: u32,
        bits: u32,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, KernelError> {
        if !VALID_BITS.contains(&bits) {
            return Err(KernelError::InvalidShape(format!(
                "unsupported bit width {bits}; must be one of {VALID_BITS:?}"
            )));
        }
        if !VALID_GROUP_SIZES.contains(&group_size) {
            return Err(KernelError::InvalidShape(format!(
                "unsupported group_size {group_size}; must be one of {VALID_GROUP_SIZES:?}"
            )));
        }
        if in_features % (group_size as usize) != 0 {
            return Err(KernelError::InvalidShape(format!(
                "in_features ({in_features}) must be a multiple of group_size ({group_size})"
            )));
        }

        let total_elements = out_features * in_features;
        let elems_per_u32 = 32 / bits as usize;
        let expected_weight_u32s = (total_elements + elems_per_u32 - 1) / elems_per_u32;
        let expected_weight_bytes = expected_weight_u32s * 4;
        if (weights_buf.length() as usize) < expected_weight_bytes {
            return Err(KernelError::InvalidShape(format!(
                "weights_buf too small: {} bytes < expected {} bytes for [{out_features}, {in_features}] at {bits} bits",
                weights_buf.length(),
                expected_weight_bytes,
            )));
        }

        let num_groups = total_elements / (group_size as usize);
        let expected_scales_bytes = num_groups * 4; // float32
        if (scales_buf.length() as usize) < expected_scales_bytes {
            return Err(KernelError::InvalidShape(format!(
                "scales_buf too small: {} bytes < expected {} bytes ({num_groups} groups)",
                scales_buf.length(),
                expected_scales_bytes,
            )));
        }
        if (biases_buf.length() as usize) < expected_scales_bytes {
            return Err(KernelError::InvalidShape(format!(
                "biases_buf too small: {} bytes < expected {} bytes ({num_groups} groups)",
                biases_buf.length(),
                expected_scales_bytes,
            )));
        }

        Ok(Self {
            weights_buf,
            scales_buf,
            biases_buf,
            group_size,
            bits,
            out_features,
            in_features,
        })
    }

    /// Number of quantization groups per output row.
    pub fn groups_per_row(&self) -> usize {
        self.in_features / self.group_size as usize
    }

    /// Total number of quantization groups.
    pub fn num_groups(&self) -> usize {
        self.out_features * self.groups_per_row()
    }

    /// Number of packed uint32 values per row.
    pub fn packed_u32s_per_row(&self) -> usize {
        let elems_per_u32 = 32 / self.bits as usize;
        (self.in_features + elems_per_u32 - 1) / elems_per_u32
    }
}

// ---------------------------------------------------------------------------
// Metal shader source -- affine quantized matrix-vector multiply
// ---------------------------------------------------------------------------

pub const QUANTIZED_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// affine_qmv: Quantized matrix-vector multiply using MLX affine format.
//
// For each output row, computes:
//   output[row] = dot( dequantize(weights[row, :]), vec[:] )
//
// where dequantize(q_i) = scales[group] * q_i + biases[group].
//
// Buffers:
//   buffer(0) weights  - packed uint32 data, row-major
//   buffer(1) scales   - float32, one per group, row-major
//   buffer(2) biases   - float32, one per group, row-major
//   buffer(3) vec      - float32 input vector [in_features]
//   buffer(4) output   - float32 output vector [out_features]
//   buffer(5) params   - uint4: (out_features, in_features, group_size, bits)
// -----------------------------------------------------------------------

// Extract a quantized integer value from a packed uint32 word.
// `idx_in_word` is the position of the value within the uint32 (0-based).
inline uint extract_bits(uint word, uint idx_in_word, uint bits) {
    return (word >> (idx_in_word * bits)) & ((1u << bits) - 1u);
}

kernel void affine_qmv(
    device const uint32_t* weights  [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    vec      [[buffer(3)]],
    device float*          output   [[buffer(4)]],
    constant uint4&        params   [[buffer(5)]],
    uint row        [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]])
{
    const uint out_features = params.x;
    const uint in_features  = params.y;
    const uint group_size   = params.z;
    const uint bits         = params.w;

    if (row >= out_features) return;

    const uint elems_per_u32   = 32u / bits;
    const uint groups_per_row  = in_features / group_size;
    const uint u32s_per_group  = group_size / elems_per_u32;
    const uint u32s_per_row    = groups_per_row * u32s_per_group;
    const uint mask            = (1u << bits) - 1u;

    // Each thread accumulates over a strided subset of groups.
    // Thread index within the threadgroup:
    const uint tid = simd_gid * 32u + simd_lid;
    const uint tg_size = simd_count * 32u;

    float accum = 0.0f;

    // Pointers for this row
    device const uint32_t* row_weights = weights + row * u32s_per_row;
    device const float*    row_scales  = scales  + row * groups_per_row;
    device const float*    row_biases  = biases  + row * groups_per_row;

    for (uint g = tid; g < groups_per_row; g += tg_size) {
        float scale = row_scales[g];
        float bias  = row_biases[g];

        // Sum x[i] for this group (needed to fold in the bias term)
        float group_dot = 0.0f;
        float group_xsum = 0.0f;

        uint base_elem = g * group_size;
        device const uint32_t* group_words = row_weights + g * u32s_per_group;

        for (uint w = 0; w < u32s_per_group; w++) {
            uint word = group_words[w];
            uint elem_base = base_elem + w * elems_per_u32;

            for (uint k = 0; k < elems_per_u32; k++) {
                float q = float((word >> (k * bits)) & mask);
                float x = vec[elem_base + k];
                group_dot  += q * x;
                group_xsum += x;
            }
        }

        accum += scale * group_dot + bias * group_xsum;
    }

    // SIMD reduction within each simdgroup
    accum = simd_sum(accum);

    // Cross-simdgroup reduction via threadgroup shared memory
    threadgroup float simd_sums[32]; // max 32 simdgroups per threadgroup
    if (simd_lid == 0) {
        simd_sums[simd_gid] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup sums up all simdgroup partial results
    if (simd_gid == 0) {
        float val = (simd_lid < simd_count) ? simd_sums[simd_lid] : 0.0f;
        val = simd_sum(val);
        if (simd_lid == 0) {
            output[row] = val;
        }
    }
}

// -----------------------------------------------------------------------
// Specialized Q4 kernel with uint16 unpacking for higher throughput.
//
// This kernel is optimized for 4-bit quantization: it reads uint16 words
// and unpacks 4 nibbles per word using bitwise masks, avoiding the
// generic per-element shift loop.
//
// Only valid when bits == 4 and group_size is a multiple of 4.
// -----------------------------------------------------------------------
kernel void affine_qmv_q4(
    device const uint16_t* weights  [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    vec      [[buffer(3)]],
    device float*          output   [[buffer(4)]],
    constant uint4&        params   [[buffer(5)]],
    uint row        [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]])
{
    const uint out_features = params.x;
    const uint in_features  = params.y;
    const uint group_size   = params.z;
    // bits == 4 assumed

    if (row >= out_features) return;

    // 4 nibbles per uint16, so group_size/4 uint16 words per group
    const uint words_per_group = group_size / 4u;
    const uint groups_per_row  = in_features / group_size;
    const uint words_per_row   = groups_per_row * words_per_group;

    const uint tid = simd_gid * 32u + simd_lid;
    const uint tg_size = simd_count * 32u;

    float accum = 0.0f;

    device const uint16_t* row_weights = weights + row * words_per_row;
    device const float*    row_scales  = scales  + row * groups_per_row;
    device const float*    row_biases  = biases  + row * groups_per_row;

    for (uint g = tid; g < groups_per_row; g += tg_size) {
        float scale = row_scales[g];
        float bias  = row_biases[g];

        float group_dot = 0.0f;
        float group_xsum = 0.0f;

        uint base_elem = g * group_size;
        device const uint16_t* ws = row_weights + g * words_per_group;
        device const float*    x  = vec + base_elem;

        for (uint i = 0; i < words_per_group; i++) {
            uint16_t word = ws[i];
            // Unpack 4 nibbles with bitwise masks
            float q0 = float(word & 0x000fu);
            float q1 = float((word >> 4u)  & 0x000fu);
            float q2 = float((word >> 8u)  & 0x000fu);
            float q3 = float((word >> 12u) & 0x000fu);

            float x0 = x[4u * i];
            float x1 = x[4u * i + 1u];
            float x2 = x[4u * i + 2u];
            float x3 = x[4u * i + 3u];

            group_dot  += q0 * x0 + q1 * x1 + q2 * x2 + q3 * x3;
            group_xsum += x0 + x1 + x2 + x3;
        }

        accum += scale * group_dot + bias * group_xsum;
    }

    // SIMD reduction
    accum = simd_sum(accum);

    threadgroup float simd_sums[32];
    if (simd_lid == 0) {
        simd_sums[simd_gid] = accum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        float val = (simd_lid < simd_count) ? simd_sums[simd_lid] : 0.0f;
        val = simd_sum(val);
        if (simd_lid == 0) {
            output[row] = val;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("quantized", QUANTIZED_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Select the kernel name based on the bit width. For 4-bit weights we use
/// the specialized Q4 kernel; everything else goes through the generic kernel.
fn kernel_for_bits(bits: u32) -> &'static str {
    match bits {
        4 => "affine_qmv_q4",
        _ => "affine_qmv",
    }
}

/// Affine quantized matrix-vector multiply using the new MLX-style format.
///
/// Computes `output[i] = dot(dequant(weights[i, :]), vec[:])` for each output row,
/// where `dequant(q) = scale * q + bias` per group.
///
/// # Arguments
/// - `registry`: kernel registry (must have `quantized` source registered).
/// - `qw`: the quantized weight description (buffers + metadata).
/// - `vec`: f32 input vector of length `qw.in_features`.
/// - `queue`: Metal command queue for dispatch.
///
/// # Returns
/// A 1-D f32 `Array` of length `qw.out_features`.
pub fn affine_quantized_matmul(
    registry: &KernelRegistry,
    qw: &QuantizedWeight,
    vec: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate input vector ---
    if vec.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul requires Float32 input vec, got {:?}",
            vec.dtype()
        )));
    }
    if vec.numel() != qw.in_features {
        return Err(KernelError::InvalidShape(format!(
            "vec.numel() ({}) != in_features ({})",
            vec.numel(),
            qw.in_features
        )));
    }

    let kernel_name = kernel_for_bits(qw.bits);

    // Use Float32 as the dtype key (the pipeline is not dtype-templated,
    // the kernel itself handles arbitrary bit widths internally).
    let pipeline = registry.get_pipeline(kernel_name, DType::Float32)?;
    let out = Array::zeros(
        registry.device().raw(),
        &[qw.out_features],
        DType::Float32,
    );

    // Pack (out_features, in_features, group_size, bits) into a uint4.
    let params: [u32; 4] = [
        super::checked_u32(qw.out_features, "out_features")?,
        super::checked_u32(qw.in_features, "in_features")?,
        qw.group_size,
        qw.bits,
    ];

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const _,
        (params.len() * std::mem::size_of::<u32>()) as u64,
        opts,
    );

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&qw.weights_buf), 0);
    enc.set_buffer(1, Some(&qw.scales_buf), 0);
    enc.set_buffer(2, Some(&qw.biases_buf), 0);
    enc.set_buffer(3, Some(vec.metal_buffer()), vec.offset() as u64);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&params_buf), 0);

    let max_tg = pipeline.max_total_threads_per_threadgroup();
    // Use up to 256 threads (8 simdgroups of 32). The kernel uses simd_sum
    // so we need the threadgroup size to be a multiple of 32.
    let tg = std::cmp::min(256, max_tg);
    // Round down to multiple of 32 (simdgroup width on Apple GPU).
    let tg = (tg / 32) * 32;
    let tg = std::cmp::max(tg, 32); // at least one simdgroup

    enc.dispatch_thread_groups(
        metal::MTLSize::new(qw.out_features as u64, 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Legacy API (backward compatibility shim)
// ---------------------------------------------------------------------------

/// Quantized matrix-vector multiply (legacy GGML-format API).
///
/// DEPRECATED: This function exists for backward compatibility with
/// the old interleaved `BlockQ8_0` / `BlockQ4_0` format. New callers
/// should use [`affine_quantized_matmul`] with a [`QuantizedWeight`].
///
/// `weights` is a 1-D `Array` whose raw bytes hold the packed GGML blocks.
/// `vec` is an f32 input vector of length `in_features`.
pub fn quantized_matmul(
    _registry: &KernelRegistry,
    weights: &Array,
    vec: &Array,
    out_features: usize,
    in_features: usize,
    _queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let _kernel_name = match weights.dtype() {
        DType::Q8_0 => "qmv_q8_0_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "quantized_matmul not supported for {:?}",
                weights.dtype()
            )))
        }
    };

    // Validate input vector dtype
    if vec.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "quantized_matmul requires Float32 input vec, got {:?}",
            vec.dtype()
        )));
    }

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

    // Validate weights buffer size accounting for offset
    let expected_weight_bytes = weights.dtype().numel_to_bytes(out_features * in_features);
    let available_bytes = weights.metal_buffer().length() as usize - weights.offset();
    if available_bytes < expected_weight_bytes {
        return Err(KernelError::InvalidShape(format!(
            "weights buffer too small: {} available bytes (buffer {} - offset {}) < expected {} bytes for [{out_features}, {in_features}] {:?}",
            available_bytes, weights.metal_buffer().length(), weights.offset(), expected_weight_bytes, weights.dtype()
        )));
    }

    // Validate vec buffer has enough space at its offset
    let vec_needed = in_features * DType::Float32.size_of();
    let vec_available = vec.metal_buffer().length() as usize - vec.offset();
    if vec_available < vec_needed {
        return Err(KernelError::InvalidShape(format!(
            "vec buffer too small: {} available bytes (buffer {} - offset {}) < expected {} bytes",
            vec_available,
            vec.metal_buffer().length(),
            vec.offset(),
            vec_needed
        )));
    }

    // The legacy GGML kernel is no longer shipped. Return a clear error
    // directing callers to the new affine API.
    Err(KernelError::NotFound(format!(
        "Legacy GGML kernel '{}' is no longer available. \
         Convert weights to MLX affine format and use affine_quantized_matmul() instead.",
        _kernel_name
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_bits() {
        for &b in VALID_BITS {
            assert!([2, 3, 4, 6, 8].contains(&b));
        }
    }

    #[test]
    fn test_valid_group_sizes() {
        for &gs in VALID_GROUP_SIZES {
            assert!([32, 64, 128].contains(&gs));
        }
    }

    #[test]
    fn test_kernel_for_bits_q4() {
        assert_eq!(kernel_for_bits(4), "affine_qmv_q4");
    }

    #[test]
    fn test_kernel_for_bits_generic() {
        assert_eq!(kernel_for_bits(2), "affine_qmv");
        assert_eq!(kernel_for_bits(3), "affine_qmv");
        assert_eq!(kernel_for_bits(6), "affine_qmv");
        assert_eq!(kernel_for_bits(8), "affine_qmv");
    }

    #[test]
    fn test_groups_per_row() {
        // We can't construct a full QuantizedWeight without Metal, but
        // we can verify the arithmetic directly.
        let in_features: usize = 256;
        let group_size: u32 = 64;
        assert_eq!(in_features / group_size as usize, 4);
    }

    #[test]
    fn test_packed_u32s_per_row() {
        // 4-bit: 8 values per u32 -> 128 elements = 16 u32s
        let elems_per_u32 = 32 / 4;
        let in_features = 128usize;
        assert_eq!((in_features + elems_per_u32 - 1) / elems_per_u32, 16);

        // 2-bit: 16 values per u32 -> 128 elements = 8 u32s
        let elems_per_u32 = 32 / 2;
        assert_eq!((in_features + elems_per_u32 - 1) / elems_per_u32, 8);

        // 8-bit: 4 values per u32 -> 128 elements = 32 u32s
        let elems_per_u32 = 32 / 8;
        assert_eq!((in_features + elems_per_u32 - 1) / elems_per_u32, 32);

        // 3-bit: 10 values per u32 -> 128 elements = 13 u32s (ceil)
        let elems_per_u32 = 32 / 3; // 10
        assert_eq!((in_features + elems_per_u32 - 1) / elems_per_u32, 13);
    }

    #[test]
    fn test_deprecated_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }
}
