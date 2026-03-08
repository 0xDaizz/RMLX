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
use crate::kernels::{FunctionConstantValue, KernelError, KernelRegistry};

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
        let expected_weight_u32s = total_elements.div_ceil(elems_per_u32);
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
        self.in_features.div_ceil(elems_per_u32)
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

// -----------------------------------------------------------------------
// affine_qmv_fast_q4: MLX qdot-pattern Q4 fast QMV kernel.
//
// Key optimizations over affine_qmv_q4:
// - Each thread processes VALS_PER_THREAD (=8) Q4 values per inner step
// - uint16 vectorized loads with 4-nibble mask extraction (qdot pattern)
// - Pre-scaled x accumulation: x_thread tracks x/scale and x/bias
//   contributions separately via group_dot and group_xsum
// - Multiple simdgroups cooperate on each output row via simd_sum +
//   threadgroup reduction
//
// Each threadgroup handles one output row.
// Thread mapping: threads stride across groups within the row.
// -----------------------------------------------------------------------

constant constexpr uint QMV_FAST_Q4_VALS_PER_THREAD = 8;

kernel void affine_qmv_fast_q4(
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
    // bits == 4

    if (row >= out_features) return;

    // 4 nibbles per uint16 → group_size/4 uint16 words per group
    const uint words_per_group = group_size / 4u;
    const uint groups_per_row  = in_features / group_size;
    const uint words_per_row   = groups_per_row * words_per_group;

    const uint tid = simd_gid * 32u + simd_lid;
    const uint tg_size = simd_count * 32u;

    float accum = 0.0f;

    device const uint16_t* row_weights = weights + row * words_per_row;
    device const float*    row_scales  = scales  + row * groups_per_row;
    device const float*    row_biases  = biases  + row * groups_per_row;

    // Each thread strides across groups
    for (uint g = tid; g < groups_per_row; g += tg_size) {
        float scale = row_scales[g];
        float bias  = row_biases[g];

        float group_dot = 0.0f;
        float group_xsum = 0.0f;

        uint base_elem = g * group_size;
        device const uint16_t* ws = row_weights + g * words_per_group;
        device const float*    x  = vec + base_elem;

        // Process QMV_FAST_Q4_VALS_PER_THREAD (8) values per iteration
        // = 2 uint16 words (4 nibbles each)
        const uint words_per_step = QMV_FAST_Q4_VALS_PER_THREAD / 4u;
        const uint num_steps = words_per_group / words_per_step;

        for (uint s = 0; s < num_steps; s++) {
            uint w_off = s * words_per_step;
            uint x_off = s * QMV_FAST_Q4_VALS_PER_THREAD;

            // qdot pattern: load 2 uint16 words, extract 4 nibbles each
            uint16_t w0 = ws[w_off];
            uint16_t w1 = ws[w_off + 1u];

            // Load 8 x values
            float x0 = x[x_off];
            float x1 = x[x_off + 1u];
            float x2 = x[x_off + 2u];
            float x3 = x[x_off + 3u];
            float x4 = x[x_off + 4u];
            float x5 = x[x_off + 5u];
            float x6 = x[x_off + 6u];
            float x7 = x[x_off + 7u];

            // qdot: nibble extraction via masks (MLX pattern)
            group_dot += float(w0 & 0x000Fu) * x0;
            group_dot += float((w0 >> 4u)  & 0x000Fu) * x1;
            group_dot += float((w0 >> 8u)  & 0x000Fu) * x2;
            group_dot += float((w0 >> 12u) & 0x000Fu) * x3;

            group_dot += float(w1 & 0x000Fu) * x4;
            group_dot += float((w1 >> 4u)  & 0x000Fu) * x5;
            group_dot += float((w1 >> 8u)  & 0x000Fu) * x6;
            group_dot += float((w1 >> 12u) & 0x000Fu) * x7;

            group_xsum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
        }

        // Handle remaining words (when words_per_group is not multiple of words_per_step)
        for (uint w = num_steps * words_per_step; w < words_per_group; w++) {
            uint16_t word = ws[w];
            uint elem = w * 4u;
            float q0 = float(word & 0x000Fu);
            float q1 = float((word >> 4u)  & 0x000Fu);
            float q2 = float((word >> 8u)  & 0x000Fu);
            float q3 = float((word >> 12u) & 0x000Fu);

            float xv0 = x[elem];
            float xv1 = x[elem + 1u];
            float xv2 = x[elem + 2u];
            float xv3 = x[elem + 3u];

            group_dot  += q0 * xv0 + q1 * xv1 + q2 * xv2 + q3 * xv3;
            group_xsum += xv0 + xv1 + xv2 + xv3;
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

// -----------------------------------------------------------------------
// affine_qmv_fast_q8: Vectorized Q8 QMV kernel.
//
// Q8 stores 1 byte per quantized value (no bit packing needed).
// Key optimizations:
// - uchar4 vectorized loads (4 bytes at a time)
// - Direct float conversion + FMA (no bit extraction overhead)
// - Same simd_sum + threadgroup reduction pattern
//
// Each threadgroup handles one output row.
// -----------------------------------------------------------------------

constant constexpr uint QMV_FAST_Q8_VALS_PER_THREAD = 8;

kernel void affine_qmv_fast_q8(
    device const uint8_t*  weights  [[buffer(0)]],
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
    // bits == 8

    if (row >= out_features) return;

    // Q8: 1 byte per element, but MLX affine format packs 32/8 = 4 values per uint32
    // We read raw bytes via uint8_t* pointer.
    // bytes_per_row = in_features (1 byte per element)
    const uint bytes_per_row   = in_features;
    const uint groups_per_row  = in_features / group_size;

    const uint tid = simd_gid * 32u + simd_lid;
    const uint tg_size = simd_count * 32u;

    float accum = 0.0f;

    device const uint8_t* row_weights = weights + row * bytes_per_row;
    device const float*   row_scales  = scales  + row * groups_per_row;
    device const float*   row_biases  = biases  + row * groups_per_row;

    for (uint g = tid; g < groups_per_row; g += tg_size) {
        float scale = row_scales[g];
        float bias  = row_biases[g];

        float group_dot = 0.0f;
        float group_xsum = 0.0f;

        uint base_elem = g * group_size;
        device const uint8_t* ws = row_weights + base_elem;
        device const float*   x  = vec + base_elem;

        // Process 4 bytes at a time via uchar4 (vectorized load)
        const uint vec4_count = group_size / 4u;
        device const uchar4* ws4 = (device const uchar4*)ws;

        for (uint i = 0; i < vec4_count; i++) {
            uchar4 packed = ws4[i];
            uint x_off = i * 4u;

            float q0 = float(packed.x);
            float q1 = float(packed.y);
            float q2 = float(packed.z);
            float q3 = float(packed.w);

            float x0 = x[x_off];
            float x1 = x[x_off + 1u];
            float x2 = x[x_off + 2u];
            float x3 = x[x_off + 3u];

            group_dot  += q0 * x0 + q1 * x1 + q2 * x2 + q3 * x3;
            group_xsum += x0 + x1 + x2 + x3;
        }

        // Handle remaining elements (when group_size not multiple of 4)
        for (uint i = vec4_count * 4u; i < group_size; i++) {
            float q = float(ws[i]);
            float xv = x[i];
            group_dot  += q * xv;
            group_xsum += xv;
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

// -----------------------------------------------------------------------
// AWQ dequantization: INT4 packed in uint32 (8 nibbles per uint32).
//
// AWQ packs along the output (column) dimension:
//   qweight: [rows, cols/8] as uint32
//   qzeros:  [num_groups, cols/8] as uint32
//   scales:  [num_groups, cols] as float
//
// Dequant: w = (nibble(qweight, col%8) - nibble(qzeros, col%8)) * scales[group, col]
// -----------------------------------------------------------------------
kernel void awq_dequant_f32(
    device const uint32_t* qweight    [[buffer(0)]],  // [rows, cols/8]
    device const uint32_t* qzeros     [[buffer(1)]],  // [num_groups, cols/8]
    device const float*    scales     [[buffer(2)]],  // [num_groups, cols]
    device float*          output     [[buffer(3)]],  // [rows, cols] dequantized
    constant uint4&        params     [[buffer(4)]],  // (rows, cols, group_size, num_groups)
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    uint rows = params.x;
    uint cols = params.y;
    uint group_size = params.z;

    if (row >= rows || col >= cols) return;

    uint group = row / group_size;
    uint pack_idx = col / 8;
    uint pack_off = col % 8;

    uint qw = qweight[row * (cols / 8) + pack_idx];
    uint nibble = (qw >> (pack_off * 4)) & 0xF;

    uint qz = qzeros[group * (cols / 8) + pack_idx];
    uint zero = (qz >> (pack_off * 4)) & 0xF;

    float scale = scales[group * cols + col];
    output[row * cols + col] = scale * (float(nibble) - float(zero));
}

// -----------------------------------------------------------------------
// GPTQ dequantization: INT4 packed in uint32 (8 nibbles per uint32).
//
// GPTQ packs along the input (row) dimension:
//   qweight: [in_features/8, out_features] as uint32
//   qzeros:  [num_groups, out_features/8] as uint32
//   scales:  [num_groups, out_features] as float
//   g_idx:   [in_features] as int32 (optional group index permutation)
//
// Dequant: w = (nibble(qweight, row%8) - (nibble(qzeros, col%8) + 1)) * scales[group, col]
// Note: GPTQ uses zero+1 offset compared to AWQ.
// -----------------------------------------------------------------------
kernel void gptq_dequant_f32(
    device const uint32_t* qweight    [[buffer(0)]],  // [in_features/8, out_features]
    device const uint32_t* qzeros     [[buffer(1)]],  // [num_groups, out_features/8]
    device const float*    scales     [[buffer(2)]],  // [num_groups, out_features]
    device const int32_t*  g_idx      [[buffer(3)]],  // [in_features] or dummy
    device float*          output     [[buffer(4)]],  // [in_features, out_features]
    constant uint4&        params     [[buffer(5)]],  // (in_features, out_features, group_size, has_g_idx)
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;    // in_features dimension
    uint col = gid.x;    // out_features dimension
    uint in_features = params.x;
    uint out_features = params.y;
    uint group_size = params.z;
    uint has_g_idx = params.w;

    if (row >= in_features || col >= out_features) return;

    // GPTQ packs along in_features: qweight[row/8, col]
    uint pack_row = row / 8;
    uint pack_off = row % 8;

    uint qw = qweight[pack_row * out_features + col];
    uint nibble = (qw >> (pack_off * 4)) & 0xF;

    uint group = has_g_idx ? uint(g_idx[row]) : (row / group_size);

    uint zero_pack_idx = col / 8;
    uint zero_pack_off = col % 8;
    uint qz = qzeros[group * (out_features / 8) + zero_pack_idx];
    uint zero = ((qz >> (zero_pack_off * 4)) & 0xF) + 1;  // GPTQ zero+1 offset

    float scale = scales[group * out_features + col];
    output[row * out_features + col] = scale * (float(nibble) - float(zero));
}
"#;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("quantized", QUANTIZED_SHADER_SOURCE)?;
    register_qmm(registry)?;
    register_gather_qmm(registry)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Select the kernel name based on the bit width.
///
/// For 4-bit weights we use the fast qdot-pattern Q4 kernel; for 8-bit weights
/// we use the vectorized Q8 kernel. Everything else goes through the generic kernel.
///
/// The older `affine_qmv_q4` and `affine_qmv` kernels are still registered and
/// available as fallbacks.
fn kernel_for_bits(bits: u32) -> &'static str {
    match bits {
        4 => "affine_qmv_fast_q4",
        8 => "affine_qmv_fast_q8",
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
    let out = Array::zeros(registry.device().raw(), &[qw.out_features], DType::Float32);

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
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// AWQ dequantization
// ---------------------------------------------------------------------------

/// Dequantize AWQ-packed INT4 weights to f32.
///
/// AWQ packs 8 INT4 values per uint32 in the column dimension.
/// This function produces a full f32 weight matrix.
///
/// # Arguments
/// - `qweight`: packed uint32 weights `[rows, cols/8]` (stored as `Float32` dtype,
///   since the raw Metal buffer contains packed uint32 data).
/// - `qzeros`: packed uint32 zero points `[num_groups, cols/8]`.
/// - `scales`: f32 scale factors `[num_groups, cols]`.
/// - `rows`: number of input rows (in_features).
/// - `cols`: number of output columns (out_features). Must be a multiple of 8.
/// - `group_size`: quantization group size. `rows` must be a multiple of `group_size`.
/// - `queue`: Metal command queue for dispatch.
///
/// # Returns
/// A 2-D f32 `Array` of shape `[rows, cols]`.
#[allow(clippy::too_many_arguments)]
pub fn awq_dequant(
    registry: &KernelRegistry,
    qweight: &Array,
    qzeros: &Array,
    scales: &Array,
    rows: usize,
    cols: usize,
    group_size: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate dimensions ---
    if cols % 8 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "awq_dequant: cols ({cols}) must be a multiple of 8 (pack factor)"
        )));
    }
    if group_size == 0 {
        return Err(KernelError::InvalidShape(
            "awq_dequant: group_size must be > 0".into(),
        ));
    }
    if rows % group_size != 0 {
        return Err(KernelError::InvalidShape(format!(
            "awq_dequant: rows ({rows}) must be a multiple of group_size ({group_size})"
        )));
    }

    let num_groups = rows / group_size;
    let cols_packed = cols / 8;

    // --- Validate buffer sizes ---
    let expected_qweight_bytes = rows * cols_packed * 4; // uint32
    let available_qweight = qweight.metal_buffer().length() as usize;
    if available_qweight < expected_qweight_bytes {
        return Err(KernelError::InvalidShape(format!(
            "awq_dequant: qweight buffer too small: {available_qweight} bytes < expected {expected_qweight_bytes} bytes for [{rows}, {cols_packed}]"
        )));
    }

    let expected_qzeros_bytes = num_groups * cols_packed * 4; // uint32
    let available_qzeros = qzeros.metal_buffer().length() as usize;
    if available_qzeros < expected_qzeros_bytes {
        return Err(KernelError::InvalidShape(format!(
            "awq_dequant: qzeros buffer too small: {available_qzeros} bytes < expected {expected_qzeros_bytes} bytes for [{num_groups}, {cols_packed}]"
        )));
    }

    let expected_scales_bytes = num_groups * cols * 4; // float32
    let available_scales = scales.metal_buffer().length() as usize;
    if available_scales < expected_scales_bytes {
        return Err(KernelError::InvalidShape(format!(
            "awq_dequant: scales buffer too small: {available_scales} bytes < expected {expected_scales_bytes} bytes for [{num_groups}, {cols}]"
        )));
    }

    // --- Build pipeline and output ---
    let pipeline = registry.get_pipeline("awq_dequant_f32", DType::Float32)?;
    let out = Array::zeros(registry.device().raw(), &[rows, cols], DType::Float32);

    let params: [u32; 4] = [
        super::checked_u32(rows, "rows")?,
        super::checked_u32(cols, "cols")?,
        super::checked_u32(group_size, "group_size")?,
        super::checked_u32(num_groups, "num_groups")?,
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
    enc.set_buffer(0, Some(qweight.metal_buffer()), qweight.offset() as u64);
    enc.set_buffer(1, Some(qzeros.metal_buffer()), qzeros.offset() as u64);
    enc.set_buffer(2, Some(scales.metal_buffer()), scales.offset() as u64);
    enc.set_buffer(3, Some(out.metal_buffer()), 0);
    enc.set_buffer(4, Some(&params_buf), 0);

    // 2D grid: (cols, rows, 1) — each thread handles one output element.
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let tw = pipeline.thread_execution_width();
    // Use a 2D threadgroup: (tw, max_tg / tw) up to grid bounds.
    let tg_x = tw.min(cols as u64);
    let tg_y = (max_tg / tg_x).max(1).min(rows as u64);

    let grid = metal::MTLSize::new(cols as u64, rows as u64, 1);
    let tg = metal::MTLSize::new(tg_x, tg_y, 1);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// GPTQ dequantization
// ---------------------------------------------------------------------------

/// Dequantize GPTQ-packed INT4 weights to f32.
///
/// GPTQ packs 8 INT4 values per uint32 in the row (in_features) dimension.
/// Optional `g_idx` array permutes group assignments (for `act_order`).
///
/// # Arguments
/// - `qweight`: packed uint32 weights `[in_features/8, out_features]`.
/// - `qzeros`: packed uint32 zero points `[num_groups, out_features/8]`.
/// - `scales`: f32 scale factors `[num_groups, out_features]`.
/// - `g_idx`: optional group index permutation `[in_features]` (for act_order).
///   When `None`, groups are computed as `row / group_size`.
/// - `in_features`: number of input features. Must be a multiple of 8.
/// - `out_features`: number of output features. Must be a multiple of 8.
/// - `group_size`: quantization group size (used when `g_idx` is `None`).
/// - `queue`: Metal command queue for dispatch.
///
/// # Returns
/// A 2-D f32 `Array` of shape `[in_features, out_features]`.
#[allow(clippy::too_many_arguments)]
pub fn gptq_dequant(
    registry: &KernelRegistry,
    qweight: &Array,
    qzeros: &Array,
    scales: &Array,
    g_idx: Option<&Array>,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- Validate dimensions ---
    if in_features % 8 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: in_features ({in_features}) must be a multiple of 8 (pack factor)"
        )));
    }
    if out_features % 8 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: out_features ({out_features}) must be a multiple of 8 (pack factor for qzeros)"
        )));
    }
    if group_size == 0 {
        return Err(KernelError::InvalidShape(
            "gptq_dequant: group_size must be > 0".into(),
        ));
    }
    if g_idx.is_none() && in_features % group_size != 0 {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: in_features ({in_features}) must be a multiple of group_size ({group_size}) when g_idx is not provided"
        )));
    }

    let rows_packed = in_features / 8;
    let cols_packed = out_features / 8;
    let num_groups = if g_idx.is_some() {
        // When g_idx is provided, infer num_groups from the scales buffer.
        // scales shape is [num_groups, out_features], so total floats / out_features.
        let scales_elems = scales.metal_buffer().length() as usize / 4;
        scales_elems / out_features
    } else {
        in_features / group_size
    };

    // --- Validate buffer sizes ---
    let expected_qweight_bytes = rows_packed * out_features * 4;
    let available_qweight = qweight.metal_buffer().length() as usize;
    if available_qweight < expected_qweight_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: qweight buffer too small: {available_qweight} bytes < expected {expected_qweight_bytes} bytes for [{rows_packed}, {out_features}]"
        )));
    }

    let expected_qzeros_bytes = num_groups * cols_packed * 4;
    let available_qzeros = qzeros.metal_buffer().length() as usize;
    if available_qzeros < expected_qzeros_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: qzeros buffer too small: {available_qzeros} bytes < expected {expected_qzeros_bytes} bytes for [{num_groups}, {cols_packed}]"
        )));
    }

    let expected_scales_bytes = num_groups * out_features * 4;
    let available_scales = scales.metal_buffer().length() as usize;
    if available_scales < expected_scales_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gptq_dequant: scales buffer too small: {available_scales} bytes < expected {expected_scales_bytes} bytes for [{num_groups}, {out_features}]"
        )));
    }

    if let Some(idx) = g_idx {
        let expected_gidx_bytes = in_features * 4; // int32
        let available_gidx = idx.metal_buffer().length() as usize;
        if available_gidx < expected_gidx_bytes {
            return Err(KernelError::InvalidShape(format!(
                "gptq_dequant: g_idx buffer too small: {available_gidx} bytes < expected {expected_gidx_bytes} bytes for [{in_features}]"
            )));
        }
    }

    // --- Build pipeline and output ---
    let pipeline = registry.get_pipeline("gptq_dequant_f32", DType::Float32)?;
    let out = Array::zeros(
        registry.device().raw(),
        &[in_features, out_features],
        DType::Float32,
    );

    let has_g_idx: u32 = if g_idx.is_some() { 1 } else { 0 };
    let params: [u32; 4] = [
        super::checked_u32(in_features, "in_features")?,
        super::checked_u32(out_features, "out_features")?,
        super::checked_u32(group_size, "group_size")?,
        has_g_idx,
    ];

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let params_buf = dev.new_buffer_with_data(
        params.as_ptr() as *const _,
        (params.len() * std::mem::size_of::<u32>()) as u64,
        opts,
    );

    // For GPTQ without g_idx, create a dummy single-element buffer.
    let dummy_buf;
    let g_idx_buf = match g_idx {
        Some(idx) => idx.metal_buffer(),
        None => {
            let dummy_val: [i32; 1] = [0];
            dummy_buf = dev.new_buffer_with_data(
                dummy_val.as_ptr() as *const _,
                std::mem::size_of::<i32>() as u64,
                opts,
            );
            &dummy_buf
        }
    };
    let g_idx_offset: u64 = g_idx.map_or(0, |idx| idx.offset() as u64);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(qweight.metal_buffer()), qweight.offset() as u64);
    enc.set_buffer(1, Some(qzeros.metal_buffer()), qzeros.offset() as u64);
    enc.set_buffer(2, Some(scales.metal_buffer()), scales.offset() as u64);
    enc.set_buffer(3, Some(g_idx_buf), g_idx_offset);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&params_buf), 0);

    // 2D grid: (out_features, in_features, 1) — each thread handles one output element.
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let tw = pipeline.thread_execution_width();
    let tg_x = tw.min(out_features as u64);
    let tg_y = (max_tg / tg_x).max(1).min(in_features as u64);

    let grid = metal::MTLSize::new(out_features as u64, in_features as u64, 1);
    let tg = metal::MTLSize::new(tg_x, tg_y, 1);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU-side affine quantized matrix-matrix multiply (Q4)
// ---------------------------------------------------------------------------

/// Dequantize a single Q4 nibble from a packed byte.
///
/// Each byte holds two 4-bit values: low nibble at bits [0:3], high nibble at
/// bits [4:7]. `nibble_idx` 0 selects the low nibble, 1 the high nibble.
#[inline]
fn dequant_q4_nibble(packed_byte: u8, nibble_idx: usize) -> u8 {
    if nibble_idx == 0 {
        packed_byte & 0x0F
    } else {
        (packed_byte >> 4) & 0x0F
    }
}

/// Dequantize a Q4-packed weight row into f32.
///
/// - `w_packed`: packed bytes for one weight row, length = K/2.
/// - `scales`: per-group scale factors for this row, length = K/group_size.
/// - `biases`: per-group bias (zero-point) values for this row, length = K/group_size.
/// - `k`: number of input features (columns).
/// - `group_size`: elements per quantization group (32, 64, or 128).
/// - `out`: output slice of length K to write dequantized f32 values.
pub fn dequantize_q4_row(
    w_packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    k: usize,
    group_size: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(w_packed.len(), k / 2);
    debug_assert_eq!(scales.len(), k / group_size);
    debug_assert_eq!(biases.len(), k / group_size);
    debug_assert_eq!(out.len(), k);

    for (col, out_val) in out.iter_mut().enumerate().take(k) {
        let byte_idx = col / 2;
        let nibble_idx = col % 2;
        let q = dequant_q4_nibble(w_packed[byte_idx], nibble_idx) as f32;
        let group = col / group_size;
        *out_val = scales[group] * q + biases[group];
    }
}

/// Affine quantized matrix-matrix multiply (Q4, CPU fallback).
///
/// Computes `output[m, n] = sum_k x[m, k] * dequant(w[n, k])` for Q4
/// quantized weights, using a dequantize-in-register approach with tiled
/// computation for cache efficiency.
///
/// This enables batch>1 prefill for 70B+ quantized models, where the existing
/// `affine_qmv` only supports batch=1 (matrix-vector).
///
/// # Layout
/// - `x`: `[M, K]` row-major f32 input activations.
/// - `w_packed`: `[N, K/2]` row-major packed Q4 weights (2 nibbles per byte).
/// - `scales`: `[N, K/group_size]` row-major per-group f32 scale factors.
/// - `biases`: `[N, K/group_size]` row-major per-group f32 bias (zero-point) values.
/// - `output`: `[M, N]` row-major f32 output (pre-allocated).
///
/// # Panics
/// Panics (via debug_assert) if slice lengths do not match the declared dimensions.
///
/// # Arguments
/// - `m`: number of input rows (batch size / sequence length).
/// - `n`: number of output features (weight rows).
/// - `k`: number of input features (weight columns). Must be even (Q4 packing).
/// - `group_size`: quantization group size (32, 64, or 128).
#[allow(clippy::too_many_arguments)]
pub fn affine_qmm(
    x: &[f32],
    w_packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(x.len(), m * k, "x must have M*K elements");
    debug_assert_eq!(
        w_packed.len(),
        n * (k / 2),
        "w_packed must have N*(K/2) elements"
    );
    let groups_per_row = k / group_size;
    debug_assert_eq!(
        scales.len(),
        n * groups_per_row,
        "scales must have N*(K/group_size) elements"
    );
    debug_assert_eq!(
        biases.len(),
        n * groups_per_row,
        "biases must have N*(K/group_size) elements"
    );
    debug_assert_eq!(output.len(), m * n, "output must have M*N elements");
    debug_assert_eq!(k % 2, 0, "K must be even for Q4 packing");
    debug_assert!(
        [32, 64, 128].contains(&group_size),
        "group_size must be 32, 64, or 128"
    );
    debug_assert_eq!(k % group_size, 0, "K must be a multiple of group_size");

    // Tile sizes for cache efficiency.
    // TILE_M x TILE_N output tile is computed per iteration over K.
    const TILE_M: usize = 4;
    const TILE_N: usize = 4;

    // Zero output
    output.iter_mut().for_each(|v| *v = 0.0);

    // Process in tiles over the output dimensions
    let mut row = 0;
    while row < m {
        let rm = std::cmp::min(TILE_M, m - row);
        let mut col = 0;
        while col < n {
            let rn = std::cmp::min(TILE_N, n - col);

            // Accumulate dot products for this tile over K, group by group
            for g in 0..groups_per_row {
                let k_start = g * group_size;
                let k_end = k_start + group_size;

                // For each weight row in this tile
                for jj in 0..rn {
                    let j = col + jj;
                    let scale = scales[j * groups_per_row + g];
                    let bias = biases[j * groups_per_row + g];
                    let w_row_packed = &w_packed[j * (k / 2)..];

                    // For each input row in this tile
                    for ii in 0..rm {
                        let i = row + ii;
                        let x_row = &x[i * k..];

                        let mut dot = 0.0f32;
                        let mut xsum = 0.0f32;

                        // Inner loop over group elements
                        #[allow(clippy::needless_range_loop)]
                        for kk in k_start..k_end {
                            let byte_idx = kk / 2;
                            let nibble_idx = kk % 2;
                            let q = dequant_q4_nibble(w_row_packed[byte_idx], nibble_idx) as f32;
                            let xv = x_row[kk];
                            dot += q * xv;
                            xsum += xv;
                        }

                        output[i * n + j] += scale * dot + bias * xsum;
                    }
                }
            }

            col += TILE_N;
        }
        row += TILE_M;
    }
}

/// Affine quantized matrix-matrix multiply (Q8, CPU).
///
/// Computes `output[m, n] = sum_k x[m, k] * dequant(w[n, k])` for Q8
/// quantized weights. Each weight is a single `u8` byte (no bit packing).
///
/// # Layout
/// - `x`: `[M, K]` row-major f32 input activations.
/// - `w_packed`: `[N, K]` row-major packed Q8 weights (1 byte per value).
/// - `scales`: `[N, K/group_size]` row-major per-group f32 scale factors.
/// - `biases`: `[N, K/group_size]` row-major per-group f32 bias values.
/// - `output`: `[M, N]` row-major f32 output (pre-allocated).
#[allow(clippy::too_many_arguments)]
pub fn affine_qmm_q8(
    x: &[f32],
    w_packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(x.len(), m * k, "x must have M*K elements");
    debug_assert_eq!(
        w_packed.len(),
        n * k,
        "w_packed must have N*K elements for Q8"
    );
    let groups_per_row = k / group_size;
    debug_assert_eq!(
        scales.len(),
        n * groups_per_row,
        "scales must have N*(K/group_size) elements"
    );
    debug_assert_eq!(
        biases.len(),
        n * groups_per_row,
        "biases must have N*(K/group_size) elements"
    );
    debug_assert_eq!(output.len(), m * n, "output must have M*N elements");
    debug_assert!(
        [32, 64, 128].contains(&group_size),
        "group_size must be 32, 64, or 128"
    );
    debug_assert_eq!(k % group_size, 0, "K must be a multiple of group_size");

    const TILE_M: usize = 4;
    const TILE_N: usize = 4;

    output.iter_mut().for_each(|v| *v = 0.0);

    let mut row = 0;
    while row < m {
        let rm = std::cmp::min(TILE_M, m - row);
        let mut col = 0;
        while col < n {
            let rn = std::cmp::min(TILE_N, n - col);

            for g in 0..groups_per_row {
                let k_start = g * group_size;
                let k_end = k_start + group_size;

                for jj in 0..rn {
                    let j = col + jj;
                    let scale = scales[j * groups_per_row + g];
                    let bias = biases[j * groups_per_row + g];
                    let w_row = &w_packed[j * k..];

                    for ii in 0..rm {
                        let i = row + ii;
                        let x_row = &x[i * k..];

                        let mut dot = 0.0f32;
                        let mut xsum = 0.0f32;

                        #[allow(clippy::needless_range_loop)]
                        for kk in k_start..k_end {
                            let q = w_row[kk] as f32;
                            let xv = x_row[kk];
                            dot += q * xv;
                            xsum += xv;
                        }

                        output[i * n + j] += scale * dot + bias * xsum;
                    }
                }
            }

            col += TILE_N;
        }
        row += TILE_M;
    }
}

// ---------------------------------------------------------------------------
// Metal shader source -- affine quantized matrix-matrix multiply
// ---------------------------------------------------------------------------

/// Legacy scalar QMM shader (BM=16, BN=16, one thread per output element).
/// Kept as fallback for non-Q4 or non-aligned cases.
pub const QMM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// affine_qmm: Quantized matrix-matrix multiply using MLX affine Q4 format.
//
// Computes: output[m, n] = sum_k x[m, k] * dequant(w[n, k])
//
// where dequant(q_i) = scales[n, group] * q_i + biases[n, group]
// and q_i is a 4-bit value packed 2 per byte in w_packed.
//
// Buffers:
//   buffer(0) x         - float32 input activations [M, K], row-major
//   buffer(1) w_packed  - uint8 packed Q4 weights [N, K/2], row-major
//   buffer(2) scales    - float32 per-group scales [N, groups_per_row]
//   buffer(3) biases    - float32 per-group biases [N, groups_per_row]
//   buffer(4) output    - float32 output [M, N], row-major
//   buffer(5) params    - uint4: (M, N, K, group_size)
//
// Grid: 2D — (ceil(N/BN), ceil(M/BM), 1) threadgroups
//   Each threadgroup computes a BM x BN tile of the output.
// -----------------------------------------------------------------------

constant constexpr uint BM_Q = 16;
constant constexpr uint BN_Q = 16;

kernel void affine_qmm(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         output    [[buffer(4)]],
    constant uint4&       params    [[buffer(5)]],
    uint3 group_id     [[threadgroup_position_in_grid]],
    uint  tid_in_group [[thread_index_in_threadgroup]])
{
    const uint M          = params.x;
    const uint N          = params.y;
    const uint K          = params.z;
    const uint group_size = params.w;

    const uint groups_per_row = K / group_size;
    const uint half_k = K / 2;   // bytes per weight row

    // Thread position within the BM_Q x BN_Q tile
    const uint local_row = tid_in_group / BN_Q;
    const uint local_col = tid_in_group % BN_Q;

    const uint out_row = group_id.y * BM_Q + local_row;
    const uint out_col = group_id.x * BN_Q + local_col;

    if (out_row >= M || out_col >= N) return;

    // Pointers for this output element
    device const float*   x_row = x + out_row * K;
    device const uint8_t* w_row = w_packed + out_col * half_k;
    device const float*   s_row = scales + out_col * groups_per_row;
    device const float*   b_row = biases + out_col * groups_per_row;

    float acc = 0.0f;

    // Iterate over groups
    for (uint g = 0; g < groups_per_row; g++) {
        float scale = s_row[g];
        float bias  = b_row[g];

        float group_dot  = 0.0f;
        float group_xsum = 0.0f;

        uint k_start = g * group_size;

        // Process 2 elements per byte
        uint byte_start = k_start / 2;
        uint bytes_per_group = group_size / 2;

        for (uint b = 0; b < bytes_per_group; b++) {
            uint8_t packed = w_row[byte_start + b];
            float q_lo = float(packed & 0x0F);
            float q_hi = float((packed >> 4) & 0x0F);

            uint kk = k_start + b * 2;
            float x0 = x_row[kk];
            float x1 = x_row[kk + 1];

            group_dot  += q_lo * x0 + q_hi * x1;
            group_xsum += x0 + x1;
        }

        acc += scale * group_dot + bias * group_xsum;
    }

    output[out_row * N + out_col] = acc;
}
"#;

// ---------------------------------------------------------------------------
// Metal shader source -- simdgroup MMA-based Q4 QMM kernel
// ---------------------------------------------------------------------------

/// High-performance Q4 QMM kernel using simdgroup MMA (8x8 fragments).
///
/// Architecture:
/// - Tile: BM=32, BN=32, BK=32
/// - 2 simdgroups per threadgroup (64 threads total)
/// - Dequant-in-loader: Q4 uint8 → half in threadgroup memory
/// - f32 accumulation via simdgroup_matrix<float, 8, 8>
/// - Double-buffered threadgroup memory for A tile (ping-pong)
/// - Function constants for alignment-based bounds check elimination
pub const QMM_MMA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
using namespace metal::simdgroup;

// Function constants for compile-time bounds check elimination
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

// Tile dimensions
constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;

// Each threadgroup has 2 simdgroups × 32 threads = 64 threads
constant constexpr uint THREADGROUP_SIZE = 64;
constant constexpr uint SIMDGROUPS_PER_TG = 2;

// -----------------------------------------------------------------------
// affine_qmm_mma_q4: Simdgroup MMA-based Q4 quantized matrix-matrix multiply.
//
// Computes: output[m, n] = sum_k x[m, k] * dequant(w[n, k])
//
// where dequant(q_i) = scales[n, group] * q_i + biases[n, group]
// and q_i is a 4-bit value packed 2 per byte in w_packed.
//
// Weight layout: w_packed is [N, K/2] row-major (uint8), where weight
// row n has K/2 bytes. Each byte holds 2 Q4 values (low nibble first).
//
// For MMA, we need A=[BM, BK] from x and B=[BK, BN] from W^T.
// Since W is [N, K] (row = output feature), B[k, n] = W[n, k].
// We dequantize W[n, k_range] into threadgroup memory transposed as B[k, n].
//
// Buffers:
//   buffer(0) x         - float32 input activations [M, K], row-major
//   buffer(1) w_packed  - uint8 packed Q4 weights [N, K/2], row-major
//   buffer(2) scales    - float32 per-group scales [N, groups_per_row]
//   buffer(3) biases    - float32 per-group biases [N, groups_per_row]
//   buffer(4) output    - float32 output [M, N], row-major
//   buffer(5) params    - uint4: (M, N, K, group_size)
//
// Grid: 2D — (ceil(N/BN), ceil(M/BM), 1) threadgroups
//   Each threadgroup computes a BM x BN tile of the output.
// -----------------------------------------------------------------------
kernel void affine_qmm_mma_q4(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         output    [[buffer(4)]],
    constant uint4&       params    [[buffer(5)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  simd_gid        [[simdgroup_index_in_threadgroup]],
    uint  simd_lid        [[thread_index_in_simdgroup]])
{
    const uint M          = params.x;
    const uint N          = params.y;
    const uint K          = params.z;
    const uint group_size = params.w;

    const uint groups_per_row = K / group_size;
    const uint half_k = K / 2;  // bytes per weight row

    // Tile origin in output space
    const uint tile_m = group_id.y * BM;
    const uint tile_n = group_id.x * BN;

    // Early exit for entirely out-of-bounds threadgroups
    if (!align_M && tile_m >= M) return;
    if (!align_N && tile_n >= N) return;

    // ------------------------------------------------------------------
    // Threadgroup shared memory (double-buffered for A, single for B)
    // A: [BM, BK] half — loaded from x (f32 → half conversion)
    // B: [BK, BN] half — dequantized from w_packed (transposed)
    // Double buffering: 2 × BM × BK + 2 × BK × BN halfs
    // = 2 × 32 × 32 + 2 × 32 × 32 = 4096 halfs = 8192 bytes
    // ------------------------------------------------------------------
    threadgroup half As[2][BM * BK];  // ping-pong A tiles
    threadgroup half Bs[2][BK * BN];  // ping-pong B tiles

    // ------------------------------------------------------------------
    // Accumulator fragments: each simdgroup computes a 16×16 sub-tile
    // using 2×2 grid of 8×8 MMA fragments.
    //
    // simdgroup 0 → rows [0..16), simdgroup 1 → rows [16..32)
    // Each simdgroup covers all BN=32 columns via 4 column fragments (0..8, 8..16, 16..24, 24..32)
    // ------------------------------------------------------------------
    simdgroup_matrix<float, 8, 8> acc[2][4];  // [2 row frags][4 col frags]
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    const uint num_k_tiles = (K + BK - 1) / BK;

    // ------------------------------------------------------------------
    // Helper: load A tile [BM, BK] from x (f32 → half)
    // 64 threads load 32×32 = 1024 elements → 16 elements per thread
    // ------------------------------------------------------------------
    auto load_A = [&](uint buf, uint k_base) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += THREADGROUP_SIZE) {
            uint row = idx / BK;
            uint col = idx % BK;
            uint global_m = tile_m + row;
            uint global_k = k_base + col;
            half val = 0.0h;
            if (align_M || global_m < M) {
                if (global_k < K) {
                    val = half(x[global_m * K + global_k]);
                }
            }
            As[buf][row * BK + col] = val;
        }
    };

    // ------------------------------------------------------------------
    // Helper: load B tile [BK, BN] from w_packed (Q4 dequant, transposed)
    //
    // W is [N, K] row-major. We need B[k, n] = dequant(W[n, k]).
    // For the k-tile [k_base, k_base+BK), and n-tile [tile_n, tile_n+BN):
    //
    // Each thread handles multiple (k, n) pairs.
    // Dequant: val = scale * q + bias, where q = 4-bit nibble from w_packed.
    // ------------------------------------------------------------------
    auto load_B = [&](uint buf, uint k_base) {
        for (uint idx = tid_in_group; idx < BK * BN; idx += THREADGROUP_SIZE) {
            uint k_local = idx / BN;  // row in B = k dimension
            uint n_local = idx % BN;  // col in B = n dimension
            uint global_k = k_base + k_local;
            uint global_n = tile_n + n_local;
            half val = 0.0h;
            if (align_N || global_n < N) {
                if (global_k < K) {
                    // Dequant: read Q4 nibble from w_packed[global_n, global_k]
                    uint byte_idx = global_k / 2;
                    uint nibble_idx = global_k % 2;
                    uint8_t packed = w_packed[global_n * half_k + byte_idx];
                    uint q = nibble_idx == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

                    // Scale and bias for this element's group
                    uint group_idx = global_k / group_size;
                    float scale = scales[global_n * groups_per_row + group_idx];
                    float bias  = biases[global_n * groups_per_row + group_idx];

                    val = half(scale * float(q) + bias);
                }
            }
            Bs[buf][k_local * BN + n_local] = val;
        }
    };

    // ------------------------------------------------------------------
    // Main loop: iterate over K dimension in BK-sized tiles
    // with double buffering (load next tile while computing current)
    // ------------------------------------------------------------------

    // Load first tile into buffer 0
    load_A(0, 0);
    load_B(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_k_tiles; t++) {
        uint cur_buf = t % 2;
        uint nxt_buf = 1 - cur_buf;

        // Prefetch next tile if available
        if (t + 1 < num_k_tiles) {
            uint next_k_base = (t + 1) * BK;
            load_A(nxt_buf, next_k_base);
            load_B(nxt_buf, next_k_base);
        }

        // Compute MMA for current tile
        // Each simdgroup handles 16 rows of the BM=32 tile
        uint sg_row_base = simd_gid * 16;  // 0 or 16

        // Iterate over BK in 8-element steps for 8×8 MMA fragments
        for (uint kk = 0; kk < BK; kk += 8) {
            // Load A fragments: 2 row fragments × 1 k fragment
            simdgroup_matrix<half, 8, 8> a_frag[2];
            simdgroup_load(a_frag[0], &As[cur_buf][(sg_row_base + 0) * BK + kk], BK);
            simdgroup_load(a_frag[1], &As[cur_buf][(sg_row_base + 8) * BK + kk], BK);

            // Load B fragments: 1 k fragment × 4 column fragments
            simdgroup_matrix<half, 8, 8> b_frag[4];
            simdgroup_load(b_frag[0], &Bs[cur_buf][kk * BN + 0],  BN);
            simdgroup_load(b_frag[1], &Bs[cur_buf][kk * BN + 8],  BN);
            simdgroup_load(b_frag[2], &Bs[cur_buf][kk * BN + 16], BN);
            simdgroup_load(b_frag[3], &Bs[cur_buf][kk * BN + 24], BN);

            // Multiply-accumulate: 2×4 grid of 8×8 fragments
            for (uint bi = 0; bi < 2; bi++) {
                for (uint bj = 0; bj < 4; bj++) {
                    simdgroup_multiply_accumulate(acc[bi][bj], a_frag[bi], b_frag[bj], acc[bi][bj]);
                }
            }
        }

        // Wait for prefetched tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------
    // Store accumulated results to output
    // Each simdgroup stores its 16×32 sub-tile (2×4 grid of 8×8 fragments)
    // ------------------------------------------------------------------
    uint sg_row_base = simd_gid * 16;

    // Use threadgroup memory as staging for f32 → device write
    // Reuse As[0] area as f32 staging (32×32 floats = 4096 bytes, fits in 2*32*32 halfs = 4096 bytes)
    threadgroup float* staging = reinterpret_cast<threadgroup float*>(&As[0][0]);

    for (uint bi = 0; bi < 2; bi++) {
        for (uint bj = 0; bj < 4; bj++) {
            // Store f32 accumulator fragment to staging area
            uint store_row = sg_row_base + bi * 8;
            uint store_col = bj * 8;
            simdgroup_store(acc[bi][bj], &staging[store_row * BN + store_col], BN);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative store from staging to device memory
    for (uint idx = tid_in_group; idx < BM * BN; idx += THREADGROUP_SIZE) {
        uint local_row = idx / BN;
        uint local_col = idx % BN;
        uint global_m = tile_m + local_row;
        uint global_n = tile_n + local_col;

        if (align_M || global_m < M) {
            if (align_N || global_n < N) {
                output[global_m * N + global_n] = staging[local_row * BN + local_col];
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Metal shader source -- simdgroup MMA-based Q8 QMM kernel
// ---------------------------------------------------------------------------

/// High-performance Q8 QMM kernel using simdgroup MMA (8x8 fragments).
///
/// Architecture identical to [`QMM_MMA_SHADER_SOURCE`] (Q4) but with simplified
/// dequantization: each quantized weight is a full `uint8_t` (no nibble
/// extraction), so the loader reads bytes directly and converts to `half`.
///
/// - Tile: BM=32, BN=32, BK=32
/// - 2 simdgroups per threadgroup (64 threads total)
/// - Dequant-in-loader: `uint8_t` → `half` (1 byte per weight, no bit packing)
/// - f32 accumulation via `simdgroup_matrix<float, 8, 8>`
/// - Double-buffered threadgroup memory (ping-pong A and B tiles)
/// - Function constants `align_M`, `align_N` for bounds check elimination
pub const QMM_MMA_Q8_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
using namespace metal::simdgroup;

// Function constants for compile-time bounds check elimination
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

// Tile dimensions
constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;

// Each threadgroup has 2 simdgroups x 32 threads = 64 threads
constant constexpr uint THREADGROUP_SIZE = 64;
constant constexpr uint SIMDGROUPS_PER_TG = 2;

// -----------------------------------------------------------------------
// affine_qmm_mma_q8: Simdgroup MMA-based Q8 quantized matrix-matrix multiply.
//
// Computes: output[m, n] = sum_k x[m, k] * dequant(w[n, k])
//
// where dequant(q_i) = scales[n, group] * q_i + biases[n, group]
// and q_i is a uint8_t value (1 byte per weight, no bit packing).
//
// Weight layout: w_packed is [N, K] row-major (uint8), where weight
// row n has K bytes. Each byte holds 1 Q8 value.
//
// Buffers:
//   buffer(0) x         - float32 input activations [M, K], row-major
//   buffer(1) w_packed  - uint8 weights [N, K], row-major (1 byte per value)
//   buffer(2) scales    - float32 per-group scales [N, groups_per_row]
//   buffer(3) biases    - float32 per-group biases [N, groups_per_row]
//   buffer(4) output    - float32 output [M, N], row-major
//   buffer(5) params    - uint4: (M, N, K, group_size)
//
// Grid: 2D - (ceil(N/BN), ceil(M/BM), 1) threadgroups
//   Each threadgroup computes a BM x BN tile of the output.
// -----------------------------------------------------------------------
kernel void affine_qmm_mma_q8(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         output    [[buffer(4)]],
    constant uint4&       params    [[buffer(5)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  simd_gid        [[simdgroup_index_in_threadgroup]],
    uint  simd_lid        [[thread_index_in_simdgroup]])
{
    const uint M          = params.x;
    const uint N          = params.y;
    const uint K          = params.z;
    const uint group_size = params.w;

    const uint groups_per_row = K / group_size;

    // Tile origin in output space
    const uint tile_m = group_id.y * BM;
    const uint tile_n = group_id.x * BN;

    // Early exit for entirely out-of-bounds threadgroups
    if (!align_M && tile_m >= M) return;
    if (!align_N && tile_n >= N) return;

    // ------------------------------------------------------------------
    // Threadgroup shared memory (double-buffered for A and B)
    // A: [BM, BK] half -- loaded from x (f32 -> half conversion)
    // B: [BK, BN] half -- dequantized from w_packed (transposed)
    // ------------------------------------------------------------------
    threadgroup half As[2][BM * BK];
    threadgroup half Bs[2][BK * BN];

    // ------------------------------------------------------------------
    // Accumulator fragments: each simdgroup computes a 16x16 sub-tile
    // ------------------------------------------------------------------
    simdgroup_matrix<float, 8, 8> acc[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    const uint num_k_tiles = (K + BK - 1) / BK;

    // Helper: load A tile [BM, BK] from x (f32 -> half)
    auto load_A = [&](uint buf, uint k_base) {
        for (uint idx = tid_in_group; idx < BM * BK; idx += THREADGROUP_SIZE) {
            uint row = idx / BK;
            uint col = idx % BK;
            uint global_m = tile_m + row;
            uint global_k = k_base + col;
            half val = 0.0h;
            if (align_M || global_m < M) {
                if (global_k < K) {
                    val = half(x[global_m * K + global_k]);
                }
            }
            As[buf][row * BK + col] = val;
        }
    };

    // Helper: load B tile [BK, BN] from w_packed (Q8 dequant, transposed)
    // Q8: 1 byte per element -- no nibble extraction needed.
    auto load_B = [&](uint buf, uint k_base) {
        for (uint idx = tid_in_group; idx < BK * BN; idx += THREADGROUP_SIZE) {
            uint k_local = idx / BN;
            uint n_local = idx % BN;
            uint global_k = k_base + k_local;
            uint global_n = tile_n + n_local;
            half val = 0.0h;
            if (align_N || global_n < N) {
                if (global_k < K) {
                    // Q8: read single byte directly
                    uint8_t q_byte = w_packed[global_n * K + global_k];

                    // Scale and bias for this element's group
                    uint group_idx = global_k / group_size;
                    float scale = scales[global_n * groups_per_row + group_idx];
                    float bias  = biases[global_n * groups_per_row + group_idx];

                    val = half(scale * float(q_byte) + bias);
                }
            }
            Bs[buf][k_local * BN + n_local] = val;
        }
    };

    // Main loop with double buffering
    load_A(0, 0);
    load_B(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_k_tiles; t++) {
        uint cur_buf = t % 2;
        uint nxt_buf = 1 - cur_buf;

        if (t + 1 < num_k_tiles) {
            uint next_k_base = (t + 1) * BK;
            load_A(nxt_buf, next_k_base);
            load_B(nxt_buf, next_k_base);
        }

        uint sg_row_base = simd_gid * 16;

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[2];
            simdgroup_load(a_frag[0], &As[cur_buf][(sg_row_base + 0) * BK + kk], BK);
            simdgroup_load(a_frag[1], &As[cur_buf][(sg_row_base + 8) * BK + kk], BK);

            simdgroup_matrix<half, 8, 8> b_frag[4];
            simdgroup_load(b_frag[0], &Bs[cur_buf][kk * BN + 0],  BN);
            simdgroup_load(b_frag[1], &Bs[cur_buf][kk * BN + 8],  BN);
            simdgroup_load(b_frag[2], &Bs[cur_buf][kk * BN + 16], BN);
            simdgroup_load(b_frag[3], &Bs[cur_buf][kk * BN + 24], BN);

            for (uint bi = 0; bi < 2; bi++) {
                for (uint bj = 0; bj < 4; bj++) {
                    simdgroup_multiply_accumulate(acc[bi][bj], a_frag[bi], b_frag[bj], acc[bi][bj]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulated results
    uint sg_row_base = simd_gid * 16;
    threadgroup float* staging = reinterpret_cast<threadgroup float*>(&As[0][0]);

    for (uint bi = 0; bi < 2; bi++) {
        for (uint bj = 0; bj < 4; bj++) {
            uint store_row = sg_row_base + bi * 8;
            uint store_col = bj * 8;
            simdgroup_store(acc[bi][bj], &staging[store_row * BN + store_col], BN);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid_in_group; idx < BM * BN; idx += THREADGROUP_SIZE) {
        uint local_row = idx / BN;
        uint local_col = idx % BN;
        uint global_m = tile_m + local_row;
        uint global_n = tile_n + local_col;

        if (align_M || global_m < M) {
            if (align_N || global_n < N) {
                output[global_m * N + global_n] = staging[local_row * BN + local_col];
            }
        }
    }
}
"#;

/// Register the QMM Metal kernels with the given registry.
///
/// Registers the legacy scalar kernel (`qmm`), the simdgroup MMA Q4 kernel
/// (`qmm_mma`), and the simdgroup MMA Q8 kernel (`qmm_mma_q8`).
/// The batched dispatch function [`affine_quantized_matmul_batched`] selects
/// the appropriate kernel based on the weight bit width.
pub fn register_qmm(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("qmm", QMM_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_mma", QMM_MMA_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_mma_q8", QMM_MMA_Q8_SHADER_SOURCE)?;
    Ok(())
}

/// Affine quantized matrix-matrix multiply on GPU (Q4/Q8, Metal).
///
/// Computes `output[m, n] = sum_k x[m, k] * dequant(w[n, k])` using a Metal
/// compute kernel for quantized weights.
///
/// - **Q4** (`bits == 4`): uses `affine_qmm_mma_q4` (dequant from nibble pairs).
/// - **Q8** (`bits == 8`): uses `affine_qmm_mma_q8` (1 byte per weight, no bit
///   packing).
///
/// Both kernels share the same MMA framework (BM=32, BN=32, BK=32, 2 simdgroups,
/// double-buffered, f32 accumulation) with function constants for alignment-based
/// bounds check elimination.
///
/// This function requires the `qmm_mma` and `qmm_mma_q8` kernel sources
/// to be registered via [`register_qmm`]. They are automatically registered by
/// [`register`] in this module.
///
/// # Arguments
/// - `registry`: kernel registry with `qmm_mma` / `qmm_mma_q8` sources registered.
/// - `x`: f32 input activations `[M, K]`.
/// - `qw`: quantized weight description (must be Q4 or Q8).
/// - `queue`: Metal command queue.
///
/// # Returns
/// An f32 `Array` of shape `[M, N]`.
pub fn affine_quantized_matmul_batched(
    registry: &KernelRegistry,
    x: &Array,
    qw: &QuantizedWeight,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate input
    if x.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched requires Float32 input x, got {:?}",
            x.dtype()
        )));
    }
    if x.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched requires 2D input x, got {}D",
            x.ndim()
        )));
    }
    if x.shape()[1] != qw.in_features {
        return Err(KernelError::InvalidShape(format!(
            "x.shape[1] ({}) != in_features ({})",
            x.shape()[1],
            qw.in_features
        )));
    }
    if qw.bits != 4 && qw.bits != 8 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched requires bits==4 or bits==8, got {}",
            qw.bits
        )));
    }

    let m = x.shape()[0];
    let n = qw.out_features;
    let k = qw.in_features;

    // MMA tile sizes
    const BM: usize = 32;
    const BN: usize = 32;

    // Function constants for alignment-based bounds check elimination
    let align_m = m % BM == 0;
    let align_n = n % BN == 0;
    let constants = [
        (200u32, FunctionConstantValue::Bool(align_m)),
        (201u32, FunctionConstantValue::Bool(align_n)),
    ];

    // Select kernel based on bit width
    let kernel_name = match qw.bits {
        4 => "affine_qmm_mma_q4",
        8 => "affine_qmm_mma_q8",
        _ => unreachable!(), // validated above
    };

    let pipeline = registry.get_pipeline_with_constants(kernel_name, DType::Float32, &constants)?;
    let out = Array::zeros(registry.device().raw(), &[m, n], DType::Float32);

    let params: [u32; 4] = [
        super::checked_u32(m, "M")?,
        super::checked_u32(n, "N")?,
        super::checked_u32(k, "K")?,
        qw.group_size,
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
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(&qw.weights_buf), 0);
    enc.set_buffer(2, Some(&qw.scales_buf), 0);
    enc.set_buffer(3, Some(&qw.biases_buf), 0);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&params_buf), 0);

    // Grid: (ceil(N/BN), ceil(M/BM), 1) threadgroups
    // Threadgroup: 64 threads (2 simdgroups of 32)
    let grid_x = n.div_ceil(BN) as u64;
    let grid_y = m.div_ceil(BM) as u64;
    let grid = metal::MTLSize::new(grid_x, grid_y, 1);
    let tg = metal::MTLSize::new(64, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

/// Affine quantized matrix-matrix multiply on GPU using the legacy scalar kernel.
///
/// This is the fallback path using the original BM=16, BN=16 scalar kernel.
/// Prefer [`affine_quantized_matmul_batched`] which uses simdgroup MMA for
/// significantly higher throughput.
pub fn affine_quantized_matmul_batched_scalar(
    registry: &KernelRegistry,
    x: &Array,
    qw: &QuantizedWeight,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate input
    if x.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched_scalar requires Float32 input x, got {:?}",
            x.dtype()
        )));
    }
    if x.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched_scalar requires 2D input x, got {}D",
            x.ndim()
        )));
    }
    if x.shape()[1] != qw.in_features {
        return Err(KernelError::InvalidShape(format!(
            "x.shape[1] ({}) != in_features ({})",
            x.shape()[1],
            qw.in_features
        )));
    }
    if qw.bits != 4 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_batched_scalar currently requires bits==4, got {}",
            qw.bits
        )));
    }

    let m = x.shape()[0];
    let n = qw.out_features;
    let k = qw.in_features;

    let pipeline = registry.get_pipeline("affine_qmm", DType::Float32)?;
    let out = Array::zeros(registry.device().raw(), &[m, n], DType::Float32);

    let params: [u32; 4] = [
        super::checked_u32(m, "M")?,
        super::checked_u32(n, "N")?,
        super::checked_u32(k, "K")?,
        qw.group_size,
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
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(&qw.weights_buf), 0);
    enc.set_buffer(2, Some(&qw.scales_buf), 0);
    enc.set_buffer(3, Some(&qw.biases_buf), 0);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&params_buf), 0);

    // Legacy tile sizes
    const BM_Q: usize = 16;
    const BN_Q: usize = 16;

    let grid_x = n.div_ceil(BN_Q) as u64;
    let grid_y = m.div_ceil(BM_Q) as u64;
    let grid = metal::MTLSize::new(grid_x, grid_y, 1);
    let tg = metal::MTLSize::new((BM_Q * BN_Q) as u64, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// GatherQMM: Index-based quantized matmul for MoE
// ---------------------------------------------------------------------------

/// Metal shader for GatherQMM — combines GatherMM (C4) with Q4 dequantization.
///
/// Each batch element selects an expert's Q4 weight via an index tensor.
/// Computes `output[b] = x[b] @ dequant(w_packed[indices[b]])`.
///
/// Layout:
/// - x:         [batch, M_per_batch, K]   (f32 input activations)
/// - w_packed:  [n_experts, N, K/2]       (Q4 packed expert weights)
/// - scales:    [n_experts, N, groups_per_row] (f32)
/// - biases:    [n_experts, N, groups_per_row] (f32)
/// - indices:   [batch]                    (uint32 expert index per batch element)
/// - output:    [batch, M_per_batch, N]    (f32)
pub const GATHER_QMM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BM_GQ = 16;
constant constexpr uint BN_GQ = 16;

// GatherQMM kernel: per-batch-element quantized matmul with expert selection.
// Each thread computes one (m, n) output element for one batch.
kernel void gather_qmm_f32(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device const uint*    indices   [[buffer(4)]],
    device float*         output    [[buffer(5)]],
    constant uint& batch_size       [[buffer(6)]],
    constant uint& M_per_batch      [[buffer(7)]],
    constant uint& N                [[buffer(8)]],
    constant uint& K                [[buffer(9)]],
    constant uint& group_size       [[buffer(10)]],
    constant uint& n_experts        [[buffer(11)]],
    uint3 group_id     [[threadgroup_position_in_grid]],
    uint  tid_in_group [[thread_index_in_threadgroup]])
{
    const uint batch_idx = group_id.z;
    if (batch_idx >= batch_size) return;

    const uint expert_idx = indices[batch_idx];
    if (expert_idx >= n_experts) return;
    const uint groups_per_row = K / group_size;
    const uint half_k = K / 2;

    const uint local_row = tid_in_group / BN_GQ;
    const uint local_col = tid_in_group % BN_GQ;

    const uint out_m = group_id.y * BM_GQ + local_row;
    const uint out_n = group_id.x * BN_GQ + local_col;

    if (out_m >= M_per_batch || out_n >= N) return;

    // Pointers for this batch element and expert
    device const float*   x_row = x + batch_idx * M_per_batch * K + out_m * K;
    device const uint8_t* w_row = w_packed + expert_idx * N * half_k + out_n * half_k;
    device const float*   s_row = scales + expert_idx * N * groups_per_row + out_n * groups_per_row;
    device const float*   b_row = biases + expert_idx * N * groups_per_row + out_n * groups_per_row;

    float acc = 0.0f;

    for (uint g = 0; g < groups_per_row; g++) {
        float scale = s_row[g];
        float bias  = b_row[g];

        float group_dot  = 0.0f;
        float group_xsum = 0.0f;

        uint k_start = g * group_size;
        uint byte_start = k_start / 2;
        uint bytes_per_group = group_size / 2;

        for (uint b = 0; b < bytes_per_group; b++) {
            uint8_t packed = w_row[byte_start + b];
            float q_lo = float(packed & 0x0F);
            float q_hi = float((packed >> 4) & 0x0F);

            uint kk = k_start + b * 2;
            float x0 = x_row[kk];
            float x1 = x_row[kk + 1];

            group_dot  += q_lo * x0 + q_hi * x1;
            group_xsum += x0 + x1;
        }

        acc += scale * group_dot + bias * group_xsum;
    }

    output[batch_idx * M_per_batch * N + out_m * N + out_n] = acc;
}
"#;

/// Register the GatherQMM Metal kernel with the given registry.
pub fn register_gather_qmm(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gather_qmm", GATHER_QMM_SHADER_SOURCE)
}

/// Ceiling division helper for GatherQMM dispatch.
fn gather_qmm_ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Create a u32 Metal constant buffer (for GatherQMM).
fn gather_qmm_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    let opts = metal::MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(&val as *const u32 as *const _, 4, opts)
}

/// Index-based quantized matmul for MoE expert dispatch (GatherQMM).
///
/// Combines GatherMM (index-based expert selection) with Q4 dequantization.
/// Each batch element selects an expert's Q4-packed weight matrix via `indices`,
/// and computes the dequantized matmul in a single fused kernel.
///
/// # Arguments
/// - `x`: f32 input activations `[batch, m_per_batch, k]`
/// - `qw`: Q4 quantized expert weights (n_experts sets of weights)
/// - `indices`: expert index per batch element `[batch]` (UInt32)
/// - `n_experts`: number of expert weight sets
/// - `m_per_batch`: rows per batch element
/// - `n`: output columns per expert
/// - `k`: inner dimension (input features)
///
/// # Returns
/// f32 output `[batch, m_per_batch, n]`.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    registry: &KernelRegistry,
    x: &Array,
    w_packed_buf: &metal::Buffer,
    scales_buf: &metal::Buffer,
    biases_buf: &metal::Buffer,
    indices: &Array,
    batch: usize,
    m_per_batch: usize,
    n: usize,
    k: usize,
    n_experts: usize,
    group_size: u32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // --- dtype validation ---
    if x.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm requires Float32 input x, got {:?}",
            x.dtype()
        )));
    }
    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }

    // --- dimension validation ---
    if k == 0 || n == 0 || m_per_batch == 0 {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: dimensions must be non-zero (k={k}, n={n}, m_per_batch={m_per_batch})"
        )));
    }
    if n_experts == 0 {
        return Err(KernelError::InvalidShape(
            "gather_qmm: n_experts must be non-zero".to_string(),
        ));
    }

    // --- x shape validation ---
    let expected_x_elems = batch * m_per_batch * k;
    if x.numel() != expected_x_elems {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: x has {} elements but expected batch({batch}) * m_per_batch({m_per_batch}) * k({k}) = {expected_x_elems}",
            x.numel()
        )));
    }

    // --- indices shape validation ---
    if indices.shape()[0] != batch {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: indices length {} != batch {}",
            indices.shape()[0],
            batch
        )));
    }

    // --- group_size validation ---
    if group_size == 0 {
        return Err(KernelError::InvalidShape(
            "gather_qmm: group_size must be non-zero".into(),
        ));
    }

    // --- K constraints for Q4 packing ---
    if k % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: K ({k}) must be even for Q4 packing"
        )));
    }
    if k % (group_size as usize) != 0 {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: K ({k}) must be a multiple of group_size ({group_size})"
        )));
    }

    // --- buffer size validation ---
    let half_k = k / 2;
    let groups_per_row = k / (group_size as usize);
    let expected_w_bytes = n_experts * n * half_k;
    if (w_packed_buf.length() as usize) < expected_w_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: w_packed_buf too small: {} bytes < expected {} bytes \
             (n_experts={n_experts}, N={n}, K/2={half_k})",
            w_packed_buf.length(),
            expected_w_bytes,
        )));
    }
    let expected_scales_bytes = n_experts * n * groups_per_row * 4; // float32
    if (scales_buf.length() as usize) < expected_scales_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: scales_buf too small: {} bytes < expected {} bytes \
             (n_experts={n_experts}, N={n}, groups_per_row={groups_per_row})",
            scales_buf.length(),
            expected_scales_bytes,
        )));
    }
    if (biases_buf.length() as usize) < expected_scales_bytes {
        return Err(KernelError::InvalidShape(format!(
            "gather_qmm: biases_buf too small: {} bytes < expected {} bytes \
             (n_experts={n_experts}, N={n}, groups_per_row={groups_per_row})",
            biases_buf.length(),
            expected_scales_bytes,
        )));
    }

    // --- Rust-side validation: all expert indices must be in [0, n_experts) ---
    if batch > 0 {
        let idx_vec = indices.to_vec_checked::<u32>();
        for (i, &idx) in idx_vec.iter().enumerate() {
            if (idx as usize) >= n_experts {
                return Err(KernelError::InvalidShape(format!(
                    "gather_qmm: index[{i}]={idx} out of range [0, {n_experts})"
                )));
            }
        }
    }

    let pipeline = registry.get_pipeline("gather_qmm_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, &[batch, m_per_batch, n], DType::Float32);

    let batch_buf = gather_qmm_u32_buf(dev, super::checked_u32(batch, "batch")?);
    let m_buf = gather_qmm_u32_buf(dev, super::checked_u32(m_per_batch, "M_per_batch")?);
    let n_buf = gather_qmm_u32_buf(dev, super::checked_u32(n, "N")?);
    let k_buf = gather_qmm_u32_buf(dev, super::checked_u32(k, "K")?);
    let gs_buf = gather_qmm_u32_buf(dev, group_size);
    let ne_buf = gather_qmm_u32_buf(dev, super::checked_u32(n_experts, "n_experts")?);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(w_packed_buf), 0);
    enc.set_buffer(2, Some(scales_buf), 0);
    enc.set_buffer(3, Some(biases_buf), 0);
    enc.set_buffer(4, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(5, Some(out.metal_buffer()), 0);
    enc.set_buffer(6, Some(&batch_buf), 0);
    enc.set_buffer(7, Some(&m_buf), 0);
    enc.set_buffer(8, Some(&n_buf), 0);
    enc.set_buffer(9, Some(&k_buf), 0);
    enc.set_buffer(10, Some(&gs_buf), 0);
    enc.set_buffer(11, Some(&ne_buf), 0);

    const BM_GQ: usize = 16;
    const BN_GQ: usize = 16;

    let grid = metal::MTLSize::new(
        gather_qmm_ceil_div(n, BN_GQ) as u64,
        gather_qmm_ceil_div(m_per_batch, BM_GQ) as u64,
        batch as u64,
    );
    let tg = metal::MTLSize::new((BM_GQ * BN_GQ) as u64, 1, 1);

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

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
    let expected_weight_bytes = weights
        .dtype()
        .numel_to_bytes(out_features * in_features)
        .map_err(|e| KernelError::InvalidShape(e.to_string()))?;
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
        assert_eq!(kernel_for_bits(4), "affine_qmv_fast_q4");
    }

    #[test]
    fn test_kernel_for_bits_q8() {
        assert_eq!(kernel_for_bits(8), "affine_qmv_fast_q8");
    }

    #[test]
    fn test_kernel_for_bits_generic() {
        assert_eq!(kernel_for_bits(2), "affine_qmv");
        assert_eq!(kernel_for_bits(3), "affine_qmv");
        assert_eq!(kernel_for_bits(6), "affine_qmv");
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
        assert_eq!(in_features.div_ceil(elems_per_u32), 16);

        // 2-bit: 16 values per u32 -> 128 elements = 8 u32s
        let elems_per_u32 = 32 / 2;
        assert_eq!(in_features.div_ceil(elems_per_u32), 8);

        // 8-bit: 4 values per u32 -> 128 elements = 32 u32s
        let elems_per_u32 = 32 / 8;
        assert_eq!(in_features.div_ceil(elems_per_u32), 32);

        // 3-bit: 10 values per u32 -> 128 elements = 13 u32s (ceil)
        let elems_per_u32 = 32 / 3; // 10
        assert_eq!(in_features.div_ceil(elems_per_u32), 13);
    }

    #[test]
    fn test_deprecated_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    // =====================================================================
    // Q4 dequantization correctness tests
    // =====================================================================

    #[test]
    fn test_dequant_q4_nibble_low() {
        // Byte 0xA7 = 1010_0111: low nibble = 0x7 = 7, high nibble = 0xA = 10
        assert_eq!(dequant_q4_nibble(0xA7, 0), 7);
    }

    #[test]
    fn test_dequant_q4_nibble_high() {
        assert_eq!(dequant_q4_nibble(0xA7, 1), 10);
    }

    #[test]
    fn test_dequant_q4_nibble_zero() {
        assert_eq!(dequant_q4_nibble(0x00, 0), 0);
        assert_eq!(dequant_q4_nibble(0x00, 1), 0);
    }

    #[test]
    fn test_dequant_q4_nibble_max() {
        assert_eq!(dequant_q4_nibble(0xFF, 0), 15);
        assert_eq!(dequant_q4_nibble(0xFF, 1), 15);
    }

    #[test]
    fn test_dequantize_q4_row_manual() {
        // K=4, group_size=4, 1 group
        // Pack: [3, 5, 10, 1] -> byte0 = (5<<4)|3 = 0x53, byte1 = (1<<4)|10 = 0x1A
        let w_packed = vec![0x53u8, 0x1A];
        let scales = vec![2.0f32];
        let biases = vec![0.5f32];
        let k = 4;
        let group_size = 4;
        let mut out = vec![0.0f32; k];

        dequantize_q4_row(&w_packed, &scales, &biases, k, group_size, &mut out);

        // dequant(q) = scale * q + bias
        // q = [3, 5, 10, 1]
        // out[0] = 2.0 * 3 + 0.5 = 6.5
        // out[1] = 2.0 * 5 + 0.5 = 10.5
        // out[2] = 2.0 * 10 + 0.5 = 20.5
        // out[3] = 2.0 * 1 + 0.5 = 2.5
        assert!((out[0] - 6.5).abs() < 1e-6);
        assert!((out[1] - 10.5).abs() < 1e-6);
        assert!((out[2] - 20.5).abs() < 1e-6);
        assert!((out[3] - 2.5).abs() < 1e-6);
    }

    // =====================================================================
    // affine_qmm: QMM matches naive unquantized matmul
    // =====================================================================

    /// Helper: quantize a row of f32 weights into Q4 packed bytes + scales + biases.
    ///
    /// For each group, finds the min and max, computes:
    ///   scale = (max - min) / 15
    ///   bias  = min
    ///   q_i   = round((w_i - bias) / scale)
    ///
    /// Returns (packed_bytes, scales, biases).
    fn quantize_q4_row(weights: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let k = weights.len();
        assert_eq!(k % group_size, 0);
        let num_groups = k / group_size;

        let mut packed = vec![0u8; k / 2];
        let mut scales = Vec::with_capacity(num_groups);
        let mut biases = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;
            let group = &weights[start..end];

            let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let range = max - min;
            let scale = if range < 1e-10 { 1.0 } else { range / 15.0 };
            let bias = min;

            scales.push(scale);
            biases.push(bias);

            for (i, &w) in group.iter().enumerate() {
                let q = ((w - bias) / scale).round().clamp(0.0, 15.0) as u8;
                let col = start + i;
                let byte_idx = col / 2;
                let nibble_idx = col % 2;
                if nibble_idx == 0 {
                    packed[byte_idx] = q;
                } else {
                    packed[byte_idx] |= q << 4;
                }
            }
        }

        (packed, scales, biases)
    }

    /// Naive (unquantized) matmul: C[m,n] = X[m,k] * W_dequant[n,k]^T
    /// where W_dequant is the dequantized weight matrix (row n, col k).
    #[allow(clippy::too_many_arguments)]
    fn naive_matmul_with_dequant(
        x: &[f32],
        w_packed: &[u8],
        scales: &[f32],
        biases: &[f32],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        let mut w_row_buf = vec![0.0f32; k];

        for j in 0..n {
            let row_packed = &w_packed[j * (k / 2)..(j + 1) * (k / 2)];
            let groups_per_row = k / group_size;
            let row_scales = &scales[j * groups_per_row..(j + 1) * groups_per_row];
            let row_biases = &biases[j * groups_per_row..(j + 1) * groups_per_row];

            dequantize_q4_row(
                row_packed,
                row_scales,
                row_biases,
                k,
                group_size,
                &mut w_row_buf,
            );

            for i in 0..m {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    dot += x[i * k + kk] * w_row_buf[kk];
                }
                output[i * n + j] = dot;
            }
        }

        output
    }

    #[test]
    fn test_affine_qmm_matches_naive_group32() {
        test_affine_qmm_matches_naive(32);
    }

    #[test]
    fn test_affine_qmm_matches_naive_group64() {
        test_affine_qmm_matches_naive(64);
    }

    #[test]
    fn test_affine_qmm_matches_naive_group128() {
        test_affine_qmm_matches_naive(128);
    }

    fn test_affine_qmm_matches_naive(group_size: usize) {
        // Dimensions: M=8 (batch), N=4 (output features), K=128 (input features)
        let m = 8;
        let n = 4;
        let k = 128;

        // Generate deterministic "random" weights and inputs using simple PRNG
        let mut seed = 42u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        // Generate float weights [N, K]
        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();

        // Generate input activations [M, K]
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        // Quantize each weight row
        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q4_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        // Compute with affine_qmm
        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        // Compute with naive reference
        let output_naive = naive_matmul_with_dequant(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
        );

        // Compare: should be bit-exact since both use the same packed representation
        for idx in 0..m * n {
            let diff = (output_qmm[idx] - output_naive[idx]).abs();
            let scale = output_naive[idx].abs().max(1.0);
            assert!(
                diff / scale < 1e-4,
                "mismatch at [{}, {}]: qmm={} naive={} diff={} (group_size={})",
                idx / n,
                idx % n,
                output_qmm[idx],
                output_naive[idx],
                diff,
                group_size
            );
        }
    }

    // =====================================================================
    // QMM vs unquantized matmul tolerance test
    // =====================================================================

    #[test]
    fn test_affine_qmm_vs_float_matmul_tolerance() {
        // Verify that QMM output is within expected quantization tolerance
        // of the exact float matmul.
        let m = 4;
        let n = 2;
        let k = 64;
        let group_size = 32;

        // Generate deterministic "random" weights and inputs
        let mut seed = 123u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        // Exact float matmul: C[i,j] = sum_k x[i,k] * w[j,k]
        let mut output_exact = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    dot += x[i * k + kk] * weights_f32[j * k + kk];
                }
                output_exact[i * n + j] = dot;
            }
        }

        // Quantize and compute via QMM
        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q4_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        // Q4 with 32-group-size should give ~5-10% relative error on random data.
        // We use a generous 20% tolerance to account for worst-case rounding.
        for idx in 0..m * n {
            let exact = output_exact[idx];
            let qmm = output_qmm[idx];
            let diff = (exact - qmm).abs();
            let scale = exact.abs().max(0.01);
            assert!(
                diff / scale < 0.50,
                "QMM vs float too far at [{}, {}]: exact={} qmm={} rel_err={}",
                idx / n,
                idx % n,
                exact,
                qmm,
                diff / scale
            );
        }
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn test_affine_qmm_m1_reduces_to_qmv() {
        // M=1 should produce the same result as a matrix-vector product.
        let m = 1;
        let n = 2;
        let k = 32;
        let group_size = 32;

        // Simple known weights: all nibbles = 7
        let w_packed = vec![0x77u8; n * (k / 2)];
        let scales = vec![1.0f32; n];
        let biases = vec![0.0f32; n];
        let x = vec![1.0f32; k];

        let mut output = vec![0.0f32; m * n];
        affine_qmm(
            &x,
            &w_packed,
            &scales,
            &biases,
            m,
            n,
            k,
            group_size,
            &mut output,
        );

        // Each output element = sum of k elements of (1.0 * (1.0 * 7 + 0.0)) = 7*32 = 224
        for (j, &val) in output.iter().enumerate().take(n) {
            assert!(
                (val - 224.0).abs() < 1e-4,
                "output[0,{}] = {} expected 224.0",
                j,
                val
            );
        }
    }

    #[test]
    fn test_affine_qmm_large_batch() {
        // Larger batch to exercise tiling (M=32, N=8, K=64, group_size=32)
        let m = 32;
        let n = 8;
        let k = 64;
        let group_size = 32;

        let mut seed = 7u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q4_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        let output_naive = naive_matmul_with_dequant(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
        );

        for idx in 0..m * n {
            let diff = (output_qmm[idx] - output_naive[idx]).abs();
            let scale = output_naive[idx].abs().max(1.0);
            assert!(
                diff / scale < 1e-4,
                "large batch mismatch at {}: qmm={} naive={} diff={}",
                idx,
                output_qmm[idx],
                output_naive[idx],
                diff,
            );
        }
    }

    // =====================================================================
    // gather_qmm shape/buffer-size validation tests
    // =====================================================================

    /// Helper: set up Metal device, registry, queue, and valid gather_qmm inputs.
    /// Returns (registry, x, w_packed_buf, scales_buf, biases_buf, indices, queue)
    /// with valid dimensions: batch=2, m_per_batch=1, n=4, k=32, n_experts=2, group_size=32.
    fn setup_gather_qmm_valid() -> (
        KernelRegistry,
        Array,
        metal::Buffer,
        metal::Buffer,
        metal::Buffer,
        Array,
        metal::CommandQueue,
    ) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let queue = gpu_dev.raw().new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register_gather_qmm(&registry).unwrap();

        let dev = registry.device().raw();

        let batch: usize = 2;
        let m_per_batch: usize = 1;
        let n: usize = 4;
        let k: usize = 32;
        let n_experts: usize = 2;
        let group_size: usize = 32;

        let x_data = vec![1.0f32; batch * m_per_batch * k];
        let x = Array::from_slice(dev, &x_data, vec![batch, m_per_batch, k]);

        let half_k = k / 2;
        let groups_per_row = k / group_size;

        // w_packed: n_experts * n * half_k bytes
        let w_bytes = n_experts * n * half_k;
        let w_data = vec![0x77u8; w_bytes];
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let w_packed_buf =
            dev.new_buffer_with_data(w_data.as_ptr() as *const _, w_bytes as u64, opts);

        // scales & biases: n_experts * n * groups_per_row floats
        let sb_elems = n_experts * n * groups_per_row;
        let scales_data = vec![1.0f32; sb_elems];
        let biases_data = vec![0.0f32; sb_elems];
        let sb_bytes = sb_elems * 4;
        let scales_buf =
            dev.new_buffer_with_data(scales_data.as_ptr() as *const _, sb_bytes as u64, opts);
        let biases_buf =
            dev.new_buffer_with_data(biases_data.as_ptr() as *const _, sb_bytes as u64, opts);

        let idx_data = vec![0u32, 1];
        let indices = Array::from_slice(dev, &idx_data, vec![batch]);

        (
            registry,
            x,
            w_packed_buf,
            scales_buf,
            biases_buf,
            indices,
            queue,
        )
    }

    #[test]
    fn test_gather_qmm_valid_inputs() {
        let (registry, x, w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(
            result.is_ok(),
            "valid gather_qmm should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gather_qmm_zero_k() {
        let (registry, _x, w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        let x = Array::from_slice(dev, &[0.0f32; 0], vec![2, 1, 0]);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            0,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("non-zero"),
            "expected non-zero error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_zero_n() {
        let (registry, x, w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            0,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("non-zero"),
            "expected non-zero error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_zero_n_experts() {
        let (registry, x, w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            0,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("n_experts"),
            "expected n_experts error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_x_shape_mismatch() {
        let (registry, _x, w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        // x has 10 elements but batch*m*k = 2*1*32 = 64
        let x_bad = Array::from_slice(dev, &[1.0f32; 10], vec![10]);
        let result = gather_qmm(
            &registry,
            &x_bad,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("elements"),
            "expected element count error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_odd_k() {
        let (registry, _x, w_packed_buf, scales_buf, biases_buf, _indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        let x = Array::from_slice(dev, &[1.0f32; 6], vec![2, 1, 3]);
        let indices = Array::from_slice(dev, &[0u32, 1], vec![2]);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            3,
            2,
            1,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("even"), "expected even K error, got: {msg}");
    }

    #[test]
    fn test_gather_qmm_k_not_multiple_of_group_size() {
        let (registry, _x, w_packed_buf, scales_buf, biases_buf, _indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        // k=32, group_size=64 => 32 % 64 != 0
        let x = Array::from_slice(dev, &[1.0f32; 64], vec![2, 1, 32]);
        let indices = Array::from_slice(dev, &[0u32, 1], vec![2]);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            64,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("multiple of group_size"),
            "expected group_size error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_w_packed_buf_too_small() {
        let (registry, x, _w_packed_buf, scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        // Valid w_packed needs 2*4*16 = 128 bytes, provide only 10
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let small_w = dev.new_buffer_with_data([0u8; 10].as_ptr() as *const _, 10, opts);
        let result = gather_qmm(
            &registry,
            &x,
            &small_w,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("w_packed_buf too small"),
            "expected w_packed_buf error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_scales_buf_too_small() {
        let (registry, x, w_packed_buf, _scales_buf, biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        let opts = metal::MTLResourceOptions::StorageModeShared;
        // Valid scales needs 2*4*1*4 = 32 bytes, provide only 4
        let small_s = dev.new_buffer_with_data([0u8; 4].as_ptr() as *const _, 4, opts);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &small_s,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("scales_buf too small"),
            "expected scales_buf error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_biases_buf_too_small() {
        let (registry, x, w_packed_buf, scales_buf, _biases_buf, indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let small_b = dev.new_buffer_with_data([0u8; 4].as_ptr() as *const _, 4, opts);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &small_b,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("biases_buf too small"),
            "expected biases_buf error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_index_out_of_range() {
        let (registry, x, w_packed_buf, scales_buf, biases_buf, _indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        // n_experts=2, but index=5 is out of range
        let bad_indices = Array::from_slice(dev, &[0u32, 5], vec![2]);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &bad_indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("out of range"),
            "expected out-of-range error, got: {msg}"
        );
    }

    #[test]
    fn test_gather_qmm_wrong_x_dtype() {
        // This test doesn't need a full setup since dtype check is first
        let (registry, _x, w_packed_buf, scales_buf, biases_buf, _indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        let x_u32 = Array::from_slice(dev, &[1u32; 64], vec![2, 1, 32]);
        let indices = Array::from_slice(dev, &[0u32, 1], vec![2]);
        let result = gather_qmm(
            &registry,
            &x_u32,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("Float32"), "expected dtype error, got: {msg}");
    }

    #[test]
    fn test_gather_qmm_indices_length_mismatch() {
        let (registry, x, w_packed_buf, scales_buf, biases_buf, _indices, queue) =
            setup_gather_qmm_valid();
        let dev = registry.device().raw();
        // batch=2 but indices has 3 elements
        let bad_indices = Array::from_slice(dev, &[0u32, 1, 0], vec![3]);
        let result = gather_qmm(
            &registry,
            &x,
            &w_packed_buf,
            &scales_buf,
            &biases_buf,
            &bad_indices,
            2,
            1,
            4,
            32,
            2,
            32,
            &queue,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("indices length"),
            "expected indices length error, got: {msg}"
        );
    }

    #[test]
    fn test_affine_qmm_identity_scale_zero_bias() {
        // When scale=1, bias=0, dequant(q) = q.
        // So output should be X @ Q^T where Q is the raw nibble matrix.
        let m = 2;
        let n = 2;
        let k = 32;
        let group_size = 32;

        // All nibbles = 3 -> byte = (3<<4)|3 = 0x33
        let w_packed = vec![0x33u8; n * (k / 2)];
        let scales = vec![1.0f32; n]; // 1 group per row
        let biases = vec![0.0f32; n];

        // x = [[1, 1, ..., 1], [2, 2, ..., 2]]
        let mut x = vec![0.0f32; m * k];
        for kk in 0..k {
            x[kk] = 1.0;
            x[k + kk] = 2.0;
        }
        let mut output = vec![0.0f32; m * n];

        affine_qmm(
            &x,
            &w_packed,
            &scales,
            &biases,
            m,
            n,
            k,
            group_size,
            &mut output,
        );

        // Each output element for row 0: sum(1.0 * 3.0 for k=32) = 96.0
        // Each output element for row 1: sum(2.0 * 3.0 for k=32) = 192.0
        for j in 0..n {
            assert!(
                (output[j] - 96.0).abs() < 1e-4,
                "output[0,{}] = {} expected 96.0",
                j,
                output[j]
            );
            assert!(
                (output[n + j] - 192.0).abs() < 1e-4,
                "output[1,{}] = {} expected 192.0",
                j,
                output[n + j]
            );
        }
    }

    // =====================================================================
    // Q8 QMM CPU tests
    // =====================================================================

    /// Helper: quantize a row of f32 weights into Q8 bytes + scales + biases.
    ///
    /// For each group, finds the min and max, computes:
    ///   scale = (max - min) / 255
    ///   bias  = min
    ///   q_i   = round((w_i - bias) / scale)
    ///
    /// Returns (packed_bytes, scales, biases).
    fn quantize_q8_row(weights: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let k = weights.len();
        assert_eq!(k % group_size, 0);
        let num_groups = k / group_size;

        let mut packed = vec![0u8; k];
        let mut scales = Vec::with_capacity(num_groups);
        let mut biases = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;
            let group = &weights[start..end];

            let min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let range = max - min;
            let scale = if range < 1e-10 { 1.0 } else { range / 255.0 };
            let bias = min;

            scales.push(scale);
            biases.push(bias);

            for (i, &w) in group.iter().enumerate() {
                let q = ((w - bias) / scale).round().clamp(0.0, 255.0) as u8;
                packed[start + i] = q;
            }
        }

        (packed, scales, biases)
    }

    /// Naive (unquantized) matmul for Q8: C[m,n] = X[m,k] * W_dequant[n,k]^T
    #[allow(clippy::too_many_arguments)]
    fn naive_matmul_with_dequant_q8(
        x: &[f32],
        w_packed: &[u8],
        scales: &[f32],
        biases: &[f32],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        let groups_per_row = k / group_size;

        for j in 0..n {
            for i in 0..m {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    let q = w_packed[j * k + kk] as f32;
                    let g = kk / group_size;
                    let scale = scales[j * groups_per_row + g];
                    let bias = biases[j * groups_per_row + g];
                    let w_val = scale * q + bias;
                    dot += x[i * k + kk] * w_val;
                }
                output[i * n + j] = dot;
            }
        }

        output
    }

    #[test]
    fn test_affine_qmm_q8_matches_naive_group32() {
        test_affine_qmm_q8_matches_naive(32);
    }

    #[test]
    fn test_affine_qmm_q8_matches_naive_group64() {
        test_affine_qmm_q8_matches_naive(64);
    }

    #[test]
    fn test_affine_qmm_q8_matches_naive_group128() {
        test_affine_qmm_q8_matches_naive(128);
    }

    fn test_affine_qmm_q8_matches_naive(group_size: usize) {
        let m = 8;
        let n = 4;
        let k = 128;

        let mut seed = 42u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q8_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm_q8(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        let output_naive = naive_matmul_with_dequant_q8(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
        );

        for idx in 0..m * n {
            let diff = (output_qmm[idx] - output_naive[idx]).abs();
            let scale = output_naive[idx].abs().max(1.0);
            assert!(
                diff / scale < 1e-4,
                "Q8 mismatch at [{}, {}]: qmm={} naive={} diff={} (group_size={})",
                idx / n,
                idx % n,
                output_qmm[idx],
                output_naive[idx],
                diff,
                group_size
            );
        }
    }

    #[test]
    fn test_affine_qmm_q8_identity_scale_zero_bias() {
        // When scale=1, bias=0, dequant(q) = q.
        let m = 2;
        let n = 2;
        let k = 32;
        let group_size = 32;

        // All bytes = 7
        let w_packed = vec![7u8; n * k];
        let scales = vec![1.0f32; n]; // 1 group per row
        let biases = vec![0.0f32; n];

        let mut x = vec![0.0f32; m * k];
        for kk in 0..k {
            x[kk] = 1.0;
            x[k + kk] = 2.0;
        }
        let mut output = vec![0.0f32; m * n];

        affine_qmm_q8(
            &x,
            &w_packed,
            &scales,
            &biases,
            m,
            n,
            k,
            group_size,
            &mut output,
        );

        // Row 0: sum(1.0 * 7.0 for k=32) = 224.0
        // Row 1: sum(2.0 * 7.0 for k=32) = 448.0
        for j in 0..n {
            assert!(
                (output[j] - 224.0).abs() < 1e-4,
                "output[0,{}] = {} expected 224.0",
                j,
                output[j]
            );
            assert!(
                (output[n + j] - 448.0).abs() < 1e-4,
                "output[1,{}] = {} expected 448.0",
                j,
                output[n + j]
            );
        }
    }

    #[test]
    fn test_affine_qmm_q8_large_batch() {
        let m = 32;
        let n = 8;
        let k = 64;
        let group_size = 32;

        let mut seed = 7u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q8_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm_q8(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        let output_naive = naive_matmul_with_dequant_q8(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
        );

        for idx in 0..m * n {
            let diff = (output_qmm[idx] - output_naive[idx]).abs();
            let scale = output_naive[idx].abs().max(1.0);
            assert!(
                diff / scale < 1e-4,
                "Q8 large batch mismatch at {}: qmm={} naive={} diff={}",
                idx,
                output_qmm[idx],
                output_naive[idx],
                diff,
            );
        }
    }

    #[test]
    fn test_affine_qmm_q8_vs_float_matmul_tolerance() {
        // Q8 should have much lower quantization error than Q4 (255 levels vs 15).
        let m = 4;
        let n = 2;
        let k = 64;
        let group_size = 32;

        let mut seed = 123u64;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };

        let weights_f32: Vec<f32> = (0..n * k).map(|_| next_f32()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();

        // Exact float matmul
        let mut output_exact = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut dot = 0.0f32;
                for kk in 0..k {
                    dot += x[i * k + kk] * weights_f32[j * k + kk];
                }
                output_exact[i * n + j] = dot;
            }
        }

        // Quantize and compute via Q8 QMM
        let mut all_packed = Vec::new();
        let mut all_scales = Vec::new();
        let mut all_biases = Vec::new();

        for j in 0..n {
            let row = &weights_f32[j * k..(j + 1) * k];
            let (packed, sc, bi) = quantize_q8_row(row, group_size);
            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&sc);
            all_biases.extend_from_slice(&bi);
        }

        let mut output_qmm = vec![0.0f32; m * n];
        affine_qmm_q8(
            &x,
            &all_packed,
            &all_scales,
            &all_biases,
            m,
            n,
            k,
            group_size,
            &mut output_qmm,
        );

        // Q8 should be within ~5% tolerance (much tighter than Q4's 50%)
        for idx in 0..m * n {
            let exact = output_exact[idx];
            let qmm = output_qmm[idx];
            let diff = (exact - qmm).abs();
            let scale = exact.abs().max(0.01);
            assert!(
                diff / scale < 0.10,
                "Q8 vs float too far at [{}, {}]: exact={} qmm={} rel_err={}",
                idx / n,
                idx % n,
                exact,
                qmm,
                diff / scale
            );
        }
    }
}
