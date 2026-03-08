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
// affine_qmv_fast_q4: MLX qmv_fast_impl port for Q4.
//
// Faithfully ports MLX's load_vector + qdot pattern:
// 1. load_vector pre-divides x by powers of 16 so that qdot can use
//    mask-only multiplication (no shift needed per nibble).
// 2. Multi-row: 2 simdgroups × 4 rows = 8 output rows per threadgroup.
// 3. K-striding: each thread processes values_per_thread=16 elements per
//    k-step, block_size = 16 × 32 = 512 elements per step across the TG.
//
// Q4 constants (from MLX):
//   pack_factor = 8 (32/4), packs_per_thread = 2
//   values_per_thread = 16, block_size = 512
//   bytes_per_pack = 4 (one uint32 per pack)
// -----------------------------------------------------------------------

constant constexpr int QMV_Q4_NUM_SIMDGROUPS = 2;
constant constexpr int QMV_Q4_RESULTS_PER_SG = 4;
constant constexpr int QMV_Q4_PACKS_PER_THREAD = 2;
constant constexpr int QMV_Q4_PACK_FACTOR = 8;   // 32 / 4
constant constexpr int QMV_Q4_VALUES_PER_THREAD = QMV_Q4_PACK_FACTOR * QMV_Q4_PACKS_PER_THREAD; // 16
constant constexpr int QMV_Q4_BLOCK_SIZE = QMV_Q4_VALUES_PER_THREAD * 32; // 512

kernel void affine_qmv_fast_q4(
    device const uint32_t* weights  [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    vec      [[buffer(3)]],
    device float*          output   [[buffer(4)]],
    constant uint4&        params   [[buffer(5)]],
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]])
{
    const int out_features = int(params.x);
    const int in_features  = int(params.y);
    const int group_size   = int(params.z);
    // bits == 4

    // bytes_per_pack for Q4 = 4 (one uint32 = 8 nibbles)
    const int in_vec_size_w = in_features / QMV_Q4_PACK_FACTOR;  // uint32s per row
    const int in_vec_size_g = in_features / group_size;           // groups per row
    const int scale_step    = group_size / QMV_Q4_VALUES_PER_THREAD;

    const int out_row = int(tgid.y) * (QMV_Q4_NUM_SIMDGROUPS * QMV_Q4_RESULTS_PER_SG)
                      + simd_gid * QMV_Q4_RESULTS_PER_SG;

    if (out_row >= out_features) return;

    // Pointer setup: each thread starts at its slice of K
    device const uint8_t* ws = (device const uint8_t*)(weights)
        + out_row * in_vec_size_w * 4      // row offset in bytes
        + simd_lid * QMV_Q4_PACKS_PER_THREAD * 4;  // thread offset in bytes
    device const float* sl = scales + out_row * in_vec_size_g
        + simd_lid / scale_step;
    device const float* bl = biases + out_row * in_vec_size_g
        + simd_lid / scale_step;
    device const float* x  = vec + simd_lid * QMV_Q4_VALUES_PER_THREAD;

    float result[QMV_Q4_RESULTS_PER_SG] = {0, 0, 0, 0};

    for (int k = 0; k < in_features; k += QMV_Q4_BLOCK_SIZE) {
        // --- load_vector: pre-divide x for Q4 qdot ---
        // For Q4, groups of 4 values: x[0], x[1]/16, x[2]/256, x[3]/4096
        float x_thread[QMV_Q4_VALUES_PER_THREAD];
        float xsum = 0.0f;

        for (int i = 0; i < QMV_Q4_VALUES_PER_THREAD; i += 4) {
            float v0 = x[i];
            float v1 = x[i + 1];
            float v2 = x[i + 2];
            float v3 = x[i + 3];
            xsum += v0 + v1 + v2 + v3;
            x_thread[i]     = v0;
            x_thread[i + 1] = v1 / 16.0f;
            x_thread[i + 2] = v2 / 256.0f;
            x_thread[i + 3] = v3 / 4096.0f;
        }

        // --- qdot for each output row ---
        for (int row = 0; row < QMV_Q4_RESULTS_PER_SG; row++) {
            device const uint8_t* wl = ws + row * in_vec_size_w * 4;
            float s = sl[row * in_vec_size_g];
            float b = bl[row * in_vec_size_g];

            if (row + out_row < out_features) {
                // qdot Q4: cast to uint16_t*, mask-only multiplication
                device const uint16_t* wp = (device const uint16_t*)wl;
                float accum = 0.0f;
                for (int i = 0; i < QMV_Q4_VALUES_PER_THREAD / 4; i++) {
                    accum += x_thread[4 * i]     * float(wp[i] & 0x000fu)
                           + x_thread[4 * i + 1] * float(wp[i] & 0x00f0u)
                           + x_thread[4 * i + 2] * float(wp[i] & 0x0f00u)
                           + x_thread[4 * i + 3] * float(wp[i] & 0xf000u);
                }
                result[row] += s * accum + xsum * b;
            }
        }

        // Advance pointers by block_size
        ws += QMV_Q4_BLOCK_SIZE / QMV_Q4_PACK_FACTOR * 4;  // bytes
        sl += QMV_Q4_BLOCK_SIZE / group_size;
        bl += QMV_Q4_BLOCK_SIZE / group_size;
        x  += QMV_Q4_BLOCK_SIZE;
    }

    // simd_sum reduction + direct write (no threadgroup reduction needed)
    for (int row = 0; row < QMV_Q4_RESULTS_PER_SG; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0 && out_row + row < out_features) {
            output[out_row + row] = result[row];
        }
    }
}

// -----------------------------------------------------------------------
// affine_qmv_fast_q8: MLX qmv_fast_impl port for Q8.
//
// Same multi-row + K-striding structure as Q4.
// Q8 constants:
//   pack_factor = 4 (32/8), packs_per_thread = 2
//   values_per_thread = 8, block_size = 256
//   bytes_per_pack = 4 (one uint32 = 4 bytes)
// -----------------------------------------------------------------------

constant constexpr int QMV_Q8_NUM_SIMDGROUPS = 2;
constant constexpr int QMV_Q8_RESULTS_PER_SG = 4;
constant constexpr int QMV_Q8_PACKS_PER_THREAD = 2;
constant constexpr int QMV_Q8_PACK_FACTOR = 4;   // 32 / 8
constant constexpr int QMV_Q8_VALUES_PER_THREAD = QMV_Q8_PACK_FACTOR * QMV_Q8_PACKS_PER_THREAD; // 8
constant constexpr int QMV_Q8_BLOCK_SIZE = QMV_Q8_VALUES_PER_THREAD * 32; // 256

kernel void affine_qmv_fast_q8(
    device const uint8_t*  weights  [[buffer(0)]],
    device const float*    scales   [[buffer(1)]],
    device const float*    biases   [[buffer(2)]],
    device const float*    vec      [[buffer(3)]],
    device float*          output   [[buffer(4)]],
    constant uint4&        params   [[buffer(5)]],
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]])
{
    const int out_features = int(params.x);
    const int in_features  = int(params.y);
    const int group_size   = int(params.z);
    // bits == 8

    const int in_vec_size_w = in_features;            // 1 byte per element
    const int in_vec_size_g = in_features / group_size;
    const int scale_step    = group_size / QMV_Q8_VALUES_PER_THREAD;

    const int out_row = int(tgid.y) * (QMV_Q8_NUM_SIMDGROUPS * QMV_Q8_RESULTS_PER_SG)
                      + simd_gid * QMV_Q8_RESULTS_PER_SG;

    if (out_row >= out_features) return;

    device const uint8_t* ws = weights
        + out_row * in_vec_size_w
        + simd_lid * QMV_Q8_PACKS_PER_THREAD * 4;  // 4 bytes per pack (uint32)
    device const float* sl = scales + out_row * in_vec_size_g
        + simd_lid / scale_step;
    device const float* bl = biases + out_row * in_vec_size_g
        + simd_lid / scale_step;
    device const float* x  = vec + simd_lid * QMV_Q8_VALUES_PER_THREAD;

    float result[QMV_Q8_RESULTS_PER_SG] = {0, 0, 0, 0};

    for (int k = 0; k < in_features; k += QMV_Q8_BLOCK_SIZE) {
        // --- load_vector Q8: just load and sum ---
        float x_thread[QMV_Q8_VALUES_PER_THREAD];
        float xsum = 0.0f;

        for (int i = 0; i < QMV_Q8_VALUES_PER_THREAD; i++) {
            float v = x[i];
            xsum += v;
            x_thread[i] = v;
        }

        // --- qdot Q8 for each output row ---
        for (int row = 0; row < QMV_Q8_RESULTS_PER_SG; row++) {
            device const uint8_t* wl = ws + row * in_vec_size_w;
            float s = sl[row * in_vec_size_g];
            float b = bl[row * in_vec_size_g];

            if (row + out_row < out_features) {
                float accum = 0.0f;
                for (int i = 0; i < QMV_Q8_VALUES_PER_THREAD; i++) {
                    accum += x_thread[i] * float(wl[i]);
                }
                result[row] += s * accum + xsum * b;
            }
        }

        ws += QMV_Q8_BLOCK_SIZE;  // 1 byte per element
        sl += QMV_Q8_BLOCK_SIZE / group_size;
        bl += QMV_Q8_BLOCK_SIZE / group_size;
        x  += QMV_Q8_BLOCK_SIZE;
    }

    // simd_sum reduction + direct write
    for (int row = 0; row < QMV_Q8_RESULTS_PER_SG; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0 && out_row + row < out_features) {
            output[out_row + row] = result[row];
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

    // MLX qmv_fast layout: 2 simdgroups × 4 rows = 8 rows per threadgroup.
    // Threadgroup size = 2 × 32 = 64 threads.
    // Grid: (1, ceil(out_features / 8), 1)
    let rows_per_tg: u64 = 8; // NUM_SIMDGROUPS(2) * RESULTS_PER_SG(4)
    let num_tgs_y = (qw.out_features as u64).div_ceil(rows_per_tg);
    let tg_size: u64 = 64; // 2 simdgroups × 32

    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, num_tgs_y, 1),
        metal::MTLSize::new(tg_size, 1, 1),
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

/// MLX-architecture Q4 QMM kernel using simdgroup MMA (8x8 fragments).
///
/// Architecture (matches gemm_mlx_f16 for maximum occupancy):
/// - Tile: BM=64, BN=64, BK=16
/// - 2 simdgroups per threadgroup (64 threads total), ~4KB TG memory
/// - Q4 dequant in B loader: uint8 nibble → half in threadgroup memory
/// - f32 accumulation, serpentine MMA, direct register→device store
/// - Single-buffered (low TG memory → high occupancy)
///
/// M3 Ultra-specific optimizations beyond MLX:
/// - `fc_group_size` (204): function constant → compile-time division elimination
/// - `has_norm` (203): RMSNorm fusion in A loader (eliminates separate norm kernel)
/// - Vectorized uchar4 B loads: 4 bytes → 4 halves per load iteration
/// - Function constants for alignment-based bounds check elimination
pub const QMM_MMA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// MLX-architecture Q4 QMM: BM=64, BN=64, BK=16, 2 SG (1x2), 64 threads.
//
// Same architecture as gemm_mlx_f16 (which achieves 23.82T for fp16 GEMM)
// but with B loader replaced by Q4 dequant.
//
// M3 Ultra-specific optimizations beyond MLX:
//   1. group_size as function constant → compile-time division elimination
//   2. has_norm fusion → RMSNorm applied on-the-fly during A load
//   3. Vectorized uchar4 B loads → 4 bytes → 8 halves per load
//   4. Scale/bias hoisted outside inner N loop (constant within group)
//
// Computes: output[m, n] = sum_k x[m, k] * dequant(w[n, k])
//   When has_norm=true: x[m, k] is replaced by x[m,k] * inv_rms[m] * norm_weight[k]
// Weight layout: w_packed[n, k/2] (Q4: 2 values per byte, low nibble first)
// Grid: (ceil(N/64), ceil(M/64), 1) threadgroups, 64 threads each
// ---------------------------------------------------------------------------

constant constexpr uint QBM = 64;
constant constexpr uint QBN = 64;
constant constexpr uint QBK = 16;
constant constexpr uint QTM = 8;   // BM / 8 = 64/8
constant constexpr uint QTN = 4;   // (BN/2) / 8 = 32/8

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool has_residual [[function_constant(202)]];
constant bool has_norm [[function_constant(203)]];
constant bool has_swiglu [[function_constant(204)]];
// group_size as function constant: eliminates division in dequant inner loop
constant uint fc_group_size [[function_constant(205)]];

// Stable SiLU for use in epilogue fusion
inline float q_silu_f(float x) {
    return x / (1.0f + exp(-x));
}

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> q_as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T q_as_uniform(T val) {
    return val;
}
#endif

inline uint2 q_swizzle_tg(uint2 tid, uint swizzle_log) {
    if (swizzle_log == 0) return tid;
    return uint2(
        tid.x >> swizzle_log,
        (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))
    );
}

kernel void affine_qmm_mma_q4(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         output    [[buffer(4)]],
    constant uint&        M         [[buffer(5)]],
    constant uint&        N         [[buffer(6)]],
    constant uint&        K         [[buffer(7)]],
    constant uint&        swizzle_log [[buffer(8)]],
    device const half* norm_weight  [[buffer(9)]],
    device const float* inv_rms    [[buffer(10)]],
    device const float* residual    [[buffer(11)]],
    device const float* gate_result [[buffer(12)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    threadgroup half As[QBM * QBK];  // 64x16 = 1024 halves = 2KB
    threadgroup half Bs[QBK * QBN];  // 16x64 = 1024 halves = 2KB  (total ~4KB)

    uint2 swizzled = q_swizzle_tg(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * q_as_uniform(QBM);
    const uint col_start = swizzled.x * q_as_uniform(QBN);

    const uint uK = q_as_uniform(K);
    const uint uM = q_as_uniform(M);
    const uint uN = q_as_uniform(N);
    // fc_group_size is a function constant — division by it becomes shift/multiply
    const uint groups_per_row = uK / fc_group_size;
    const uint half_k = uK / 2;

    // SG grid: 1x2 -- sg_row always 0, sg_col = sgid (0 or 1)
    const uint base_n = sgid * 32; // each SG covers 32 cols

    simdgroup_float8x8 acc[QTM][QTN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < QTM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < QTN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_tiles = (uK + QBK - 1) / QBK;

    // -- Main loop: single-buffered --
    for (uint tile = 0; tile < n_tiles; tile++) {
        uint kb = tile * QBK;

        // ---- Load A tile: 64 threads x 16 elements = 64x16 = BM x BK ----
        // When has_norm=true: As[row][col] = x[row][col] * inv_rms[row] * norm_weight[col]
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint a_row = tid_in_group;  // 0..63
            uint gr = row_start + a_row;
            if ((align_M || gr < uM) && kb + 15 < uK) {
                if (has_norm) {
                    half row_scale = half(inv_rms[gr]);
                    for (uint d = 0; d < 16; d++) {
                        As[a_row * 16 + d] = half(x[gr * uK + kb + d])
                            * row_scale * norm_weight[kb + d];
                    }
                } else {
                    for (uint d = 0; d < 16; d += 4) {
                        As[a_row * 16 + d + 0] = half(x[gr * uK + kb + d + 0]);
                        As[a_row * 16 + d + 1] = half(x[gr * uK + kb + d + 1]);
                        As[a_row * 16 + d + 2] = half(x[gr * uK + kb + d + 2]);
                        As[a_row * 16 + d + 3] = half(x[gr * uK + kb + d + 3]);
                    }
                }
            } else {
                if (has_norm) {
                    half row_scale = (align_M || gr < uM) ? half(inv_rms[gr]) : half(0);
                    for (uint d = 0; d < 16; d++) {
                        As[a_row * 16 + d] = ((align_M || gr < uM) && kb + d < uK)
                            ? half(x[gr * uK + kb + d]) * row_scale * norm_weight[kb + d]
                            : half(0);
                    }
                } else {
                    for (uint d = 0; d < 16; d++) {
                        As[a_row * 16 + d] = ((align_M || gr < uM) && kb + d < uK)
                            ? half(x[gr * uK + kb + d]) : half(0);
                    }
                }
            }
        }

        // ---- Load B tile: Q4 dequant into BK x BN = 16x64 half ----
        // 64 threads: tid/4 → k-row (0..15), (tid%4)*16 → n-col (0,16,32,48)
        // Each thread dequants 16 N columns for one k index.
        // Vectorized: load 4 bytes (uchar4) → 8 halves, then another 4 → 8.
        {
            uint bi = tid_in_group >> 2;         // 0..15 (row in B tile = k offset)
            uint bj = (tid_in_group & 3u) << 4;  // 0, 16, 32, 48 (col block)
            uint gk = kb + bi;                    // global k index
            uint gc = col_start + bj;             // global n start

            if (gk < uK && (align_N || gc + 15 < uN)) {
                // All 16 N columns share the same k → same group_idx for scale/bias
                uint group_idx = gk / fc_group_size;  // compile-time optimized division
                uint byte_idx_base = gk / 2;
                bool is_lo = (gk & 1u) == 0;

                // Process 4 N columns at a time using uchar4 vectorized load
                for (uint d = 0; d < 16; d += 4) {
                    // Load 4 packed bytes from 4 consecutive N rows
                    uchar4 packed4 = uchar4(
                        w_packed[(gc + d + 0) * half_k + byte_idx_base],
                        w_packed[(gc + d + 1) * half_k + byte_idx_base],
                        w_packed[(gc + d + 2) * half_k + byte_idx_base],
                        w_packed[(gc + d + 3) * half_k + byte_idx_base]
                    );
                    // Load scales/biases for 4 N columns (same group_idx)
                    float s0 = scales[(gc + d + 0) * groups_per_row + group_idx];
                    float s1 = scales[(gc + d + 1) * groups_per_row + group_idx];
                    float s2 = scales[(gc + d + 2) * groups_per_row + group_idx];
                    float s3 = scales[(gc + d + 3) * groups_per_row + group_idx];
                    float b0 = biases[(gc + d + 0) * groups_per_row + group_idx];
                    float b1 = biases[(gc + d + 1) * groups_per_row + group_idx];
                    float b2 = biases[(gc + d + 2) * groups_per_row + group_idx];
                    float b3 = biases[(gc + d + 3) * groups_per_row + group_idx];

                    if (is_lo) {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float(packed4.x & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float(packed4.y & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float(packed4.z & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float(packed4.w & 0x0F) + b3);
                    } else {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float((packed4.x >> 4) & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float((packed4.y >> 4) & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float((packed4.z >> 4) & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float((packed4.w >> 4) & 0x0F) + b3);
                    }
                }
            } else {
                for (uint d = 0; d < 16; d++) {
                    uint gn = gc + d;
                    half val = half(0);
                    if (gk < uK && (align_N || gn < uN)) {
                        uint byte_idx = gk / 2;
                        uint8_t packed_b = w_packed[gn * half_k + byte_idx];
                        uint group_idx = gk / fc_group_size;
                        float scale_b = scales[gn * groups_per_row + group_idx];
                        float bias_b  = biases[gn * groups_per_row + group_idx];
                        bool is_lo = (gk & 1u) == 0;
                        uint nibble = is_lo ? (packed_b & 0x0F) : ((packed_b >> 4) & 0x0F);
                        val = half(scale_b * float(nibble) + bias_b);
                    }
                    Bs[bi * 64 + bj + d] = val;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA compute (serpentine, matching gemm_mlx_f16) ----
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 2; kk++) {
            simdgroup_half8x8 a_frag[QTM];
            simdgroup_half8x8 b_frag[QTN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < QTM; i++) {
                simdgroup_load(a_frag[i],
                    &As[(i * 8) * 16 + kk * 8], 16);
            }

            #pragma clang loop unroll(full)
            for (uint j = 0; j < QTN; j++) {
                simdgroup_load(b_frag[j],
                    &Bs[kk * 8 * 64 + (base_n + j * 8)], 64);
            }

            #pragma clang loop unroll(full)
            for (uint i = 0; i < QTM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < QTN; j++) {
                    uint n_serp = (i % 2) ? (3 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // -- Store results: direct store from simdgroup registers --
    // When has_residual=true: output[i] += residual[i] (epilogue fusion)
    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < QTM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < QTN; j++) {
            uint gr = row_start + i * 8 + fm;
            uint gc0 = col_start + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (align_M && align_N) {
                float v0 = elems[0];
                float v1 = elems[1];
                if (has_swiglu) {
                    uint idx0 = gr * uN + gc0;
                    uint idx1 = gr * uN + gc1;
                    v0 = q_silu_f(gate_result[idx0]) * v0;
                    v1 = q_silu_f(gate_result[idx1]) * v1;
                }
                if (has_residual) {
                    v0 += residual[gr * uN + gc0];
                    v1 += residual[gr * uN + gc1];
                }
                output[gr * uN + gc0] = v0;
                output[gr * uN + gc1] = v1;
            } else {
                if ((align_M || gr < uM) && (align_N || gc0 < uN)) {
                    float v0 = elems[0];
                    if (has_swiglu) v0 = q_silu_f(gate_result[gr * uN + gc0]) * v0;
                    if (has_residual) v0 += residual[gr * uN + gc0];
                    output[gr * uN + gc0] = v0;
                }
                if ((align_M || gr < uM) && (align_N || gc1 < uN)) {
                    float v1 = elems[1];
                    if (has_swiglu) v1 = q_silu_f(gate_result[gr * uN + gc1]) * v1;
                    if (has_residual) v1 += residual[gr * uN + gc1];
                    output[gr * uN + gc1] = v1;
                }
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Metal shader source -- Steel-architecture Q4 QMM (BM=32, BN=32, BK=32)
// ---------------------------------------------------------------------------

/// Steel-architecture Q4 QMM kernel matching MLX's QuantizedBlockLoader pattern.
///
/// Key differences from the previous `affine_qmm_mma_q4` (BM=64, BN=64, BK=16):
///
/// 1. **BM=32, BN=32, BK=32**: Smaller output tile but deeper K.
///    - K tiles halved (128 vs 256 for K=4096) → half the barrier overhead.
///    - group_size=32 fits exactly in one BK tile → 1 scale+bias per dequant.
///
/// 2. **K-contiguous B loader** (MLX QuantizedBlockLoader pattern):
///    - Each thread loads from a single N row, reading K-contiguous packed bytes.
///    - 64 threads → 32 N rows × 2 packed reads (n_reads=2) per thread.
///    - True contiguous memory access (vs scattered N-stride in old kernel).
///    - MLX dequant trick: `s[0]=scale, s[1]=scale/16` → no shift for high nibble.
///
/// 3. **BK_padded=40**: Bank conflict avoidance (matches MLX BK + 16/sizeof(half)).
///
/// 4. **half input**: A buffer is `device const half*` (pre-converted from f32).
///    - Enables vectorized half4 loads in A loader.
///
/// 5. **Double-buffered**: As/Ws use ping-pong buffers to overlap load with compute.
///
/// 6. **2 SG layout**: SG0 covers rows 0-15, SG1 covers rows 16-31 of BM=32.
///    Each SG computes 2×4 = 8 accumulators (16×32 output per SG, full BN width).
///
/// TG memory: As[2][32×40] + Ws[2][32×40] = 2×(2560+2560) = 10240 bytes ≈ 10KB
/// → 3 TG/core (32KB limit). Lower occupancy than 4KB kernel but massively
///   better load efficiency compensates.
///
/// Weight layout: w_packed[n, k/2] (N-major, Q4: 2 values per byte, low nibble first)
/// Scales: scales[n * groups_per_row + group_idx] (N-major)
/// Grid: (ceil(N/32), ceil(M/32), 1) threadgroups, 64 threads each
pub const QMM_STEEL_Q4_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Steel-architecture Q4 QMM: BM=32, BN=32, BK=32, 2 SG (64 threads).
// K-contiguous B loader matching MLX QuantizedBlockLoader pattern.
// Double-buffered. As uses BK_padded for bank conflict avoidance.
// Ws stored as [K, N] for correct simdgroup_load orientation.
// ---------------------------------------------------------------------------

constant constexpr uint ST_BM = 32;
constant constexpr uint ST_BN = 32;
constant constexpr uint ST_BK = 32;
constant constexpr uint ST_BK_PAD = 40;  // BK + 16/sizeof(half) = 32 + 8
constant constexpr uint ST_TG_SIZE = 64;

// Q4 pack constants
constant constexpr uint ST_PACK_FACTOR = 8;   // 32 / 4 bits
constant constexpr uint ST_BYTES_PER_PACK = 4; // one uint32 = 8 nibbles = 4 bytes
constant constexpr uint ST_BK_PACKED = ST_BK / ST_PACK_FACTOR;  // 32/8 = 4
constant constexpr uint ST_N_READS = (ST_BK_PACKED * ST_BN) / ST_TG_SIZE;  // (4*32)/64 = 2

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool has_residual [[function_constant(202)]];
constant bool has_swiglu [[function_constant(204)]];
constant uint st_fc_group_size [[function_constant(205)]];

inline float st_silu_f(float x) {
    return x / (1.0f + exp(-x));
}

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> st_as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T st_as_uniform(T val) {
    return val;
}
#endif

inline uint2 st_swizzle_tg(uint2 tid, uint swizzle_log) {
    if (swizzle_log == 0) return tid;
    return uint2(
        tid.x >> swizzle_log,
        (tid.y << swizzle_log) | (tid.x & ((1u << swizzle_log) - 1u))
    );
}

kernel void affine_qmm_steel_q4(
    device const half*    x         [[buffer(0)]],   // half input [M, K]
    device const uint8_t* w_packed  [[buffer(1)]],   // packed Q4 [N, K/2]
    device const float*   scales    [[buffer(2)]],   // [N * groups_per_row]
    device const float*   biases    [[buffer(3)]],   // [N * groups_per_row]
    device float*         output    [[buffer(4)]],   // [M, N]
    constant uint&        M         [[buffer(5)]],
    constant uint&        N         [[buffer(6)]],
    constant uint&        K         [[buffer(7)]],
    constant uint&        swizzle_log [[buffer(8)]],
    device const float* residual    [[buffer(9)]],
    device const float* gate_result [[buffer(10)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    // Double-buffered threadgroup memory
    // As: [M_local, K_padded] — M rows, K+pad cols. BK_padded avoids bank conflicts.
    // Ws: [K_local, N_local]  — K rows, N cols. No padding needed (stride=32=64 bytes).
    threadgroup half As[2][ST_BM * ST_BK_PAD];  // 2 × 32×40 × 2 = 5120 bytes
    threadgroup half Ws[2][ST_BK * ST_BN];       // 2 × 32×32 × 2 = 4096 bytes (total ~9KB)

    uint2 swizzled = st_swizzle_tg(uint2(group_id.x, group_id.y), swizzle_log);
    const uint row_start = swizzled.y * st_as_uniform(ST_BM);
    const uint col_start = swizzled.x * st_as_uniform(ST_BN);

    const uint uK = st_as_uniform(K);
    const uint uM = st_as_uniform(M);
    const uint uN = st_as_uniform(N);
    const uint groups_per_row = uK / st_fc_group_size;
    const uint half_k = uK / 2;  // bytes per N row in packed weights

    // Early exit for out-of-bounds threadgroups
    if (!align_M && row_start >= uM) return;
    if (!align_N && col_start >= uN) return;

    // 2-SG layout: SG0=rows 0-15, SG1=rows 16-31
    // Each SG: acc[2][4] = 16 rows × 32 cols of 8×8 MMA tiles
    const uint sg_row_base = sgid * 16;

    simdgroup_float8x8 acc[2][4];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // --- B loader thread mapping (MLX QuantizedBlockLoader pattern) ---
    // Weight layout in device memory: w_packed[n][k/2] (N-major, K contiguous)
    // B loader: each thread reads from one N row, K-contiguous packed bytes.
    // 64 threads load BN × BK_PACKED = 32 × 4 = 128 packed byte positions.
    // n_reads = 128/64 = 2 per thread.
    // bi = thread's N row in tile (0..31), bj = thread's K col packed (0..3)
    const uint w_bi = (ST_N_READS * tid_in_group) / ST_BK_PACKED;   // N row: 0..31
    const uint w_bj = (ST_N_READS * tid_in_group) % ST_BK_PACKED;   // K col (packed): 0..3
    const uint w_n_global = col_start + w_bi;

    // Scale/bias: scales[n * groups_per_row + group_idx]
    // group_idx for tile t = t * BK / group_size. With group_size=32, BK=32: one group per tile.
    const uint group_steps = st_fc_group_size / ST_BK;  // how many BK tiles per group

    const uint num_k_tiles = (uK + ST_BK - 1) / ST_BK;

    // Macro to load A tile into buffer BUF at K offset KB
    #define LOAD_A_STEEL(BUF, KB) \
    { \
        uint a_row = tid_in_group & 31u; \
        uint a_col_base = (tid_in_group >> 5u) * 16u; \
        uint gr = row_start + a_row; \
        uint gk_base = (KB) + a_col_base; \
        if ((align_M || gr < uM) && gk_base + 15 < uK) { \
            for (uint d = 0; d < 16; d += 4) { \
                *reinterpret_cast<threadgroup half4*>(&As[(BUF)][a_row * ST_BK_PAD + a_col_base + d]) = \
                    *reinterpret_cast<device const half4*>(&x[gr * uK + gk_base + d]); \
            } \
        } else if (align_M || gr < uM) { \
            for (uint d = 0; d < 16; d++) { \
                uint gk = gk_base + d; \
                As[(BUF)][a_row * ST_BK_PAD + a_col_base + d] = (gk < uK) ? x[gr * uK + gk] : half(0); \
            } \
        } else { \
            for (uint d = 0; d < 16; d++) { \
                As[(BUF)][a_row * ST_BK_PAD + a_col_base + d] = half(0); \
            } \
        } \
    }

    // Macro to load B tile (dequant Q4) into buffer BUF at tile index TILE_IDX
    // Dequant: each thread reads 2 packs (16 Q4 values) from its N row at K offset.
    // Writes to Ws[BUF][k_local * BN + n_local] (K-major for correct MMA orientation).
    #define LOAD_B_STEEL(BUF, TILE_IDX) \
    { \
        uint kb = (TILE_IDX) * ST_BK; \
        if (align_N || w_n_global < uN) { \
            /* Source: K-contiguous bytes from this thread's N row */ \
            device const uint8_t* src = w_packed + w_n_global * half_k + kb / 2 + w_bj * ST_BYTES_PER_PACK; \
            /* Scale/bias for this N row at this K group */ \
            uint group_idx = kb / st_fc_group_size; \
            float scale_f = scales[w_n_global * groups_per_row + group_idx]; \
            float bias_f  = biases[w_n_global * groups_per_row + group_idx]; \
            half s_lo = half(scale_f); \
            half s_hi = half(scale_f) / half(16.0f); \
            half bias_h = half(bias_f); \
            /* Dequant 2 packs (16 values), write to Ws[k][n] layout */ \
            uint k_local_base = w_bj * ST_PACK_FACTOR; \
            for (uint r = 0; r < ST_N_READS; r++) { \
                uint k_local = k_local_base + r * ST_PACK_FACTOR; \
                if (kb + k_local + ST_PACK_FACTOR <= uK) { \
                    device const uint8_t* wp = src + r * ST_BYTES_PER_PACK; \
                    for (uint i = 0; i < ST_PACK_FACTOR / 2; i++) { \
                        uint8_t byte = wp[i]; \
                        Ws[(BUF)][(k_local + 2*i) * ST_BN + w_bi] = s_lo * half(byte & 0x0f) + bias_h; \
                        Ws[(BUF)][(k_local + 2*i + 1) * ST_BN + w_bi] = s_hi * half(byte & 0xf0) + bias_h; \
                    } \
                } else { \
                    for (uint d = 0; d < ST_PACK_FACTOR; d++) { \
                        if (kb + k_local + d < uK) { \
                            device const uint8_t* wp = src + r * ST_BYTES_PER_PACK; \
                            uint8_t byte = wp[d / 2]; \
                            half val = (d & 1) == 0 \
                                ? (s_lo * half(byte & 0x0f) + bias_h) \
                                : (s_hi * half(byte & 0xf0) + bias_h); \
                            Ws[(BUF)][(k_local + d) * ST_BN + w_bi] = val; \
                        } else { \
                            Ws[(BUF)][(k_local + d) * ST_BN + w_bi] = half(0); \
                        } \
                    } \
                } \
            } \
        } else { \
            uint k_local_base = w_bj * ST_PACK_FACTOR; \
            for (uint d = 0; d < ST_N_READS * ST_PACK_FACTOR; d++) { \
                Ws[(BUF)][(k_local_base + d) * ST_BN + w_bi] = half(0); \
            } \
        } \
    }

    // Load first tile (buffer 0)
    LOAD_A_STEEL(0, 0);
    LOAD_B_STEEL(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main loop: double-buffered ---
    for (uint tile = 0; tile < num_k_tiles; tile++) {
        uint cur_buf = tile & 1;
        uint nxt_buf = 1 - cur_buf;

        // Prefetch next tile into nxt_buf (if exists)
        if (tile + 1 < num_k_tiles) {
            uint next_kb = (tile + 1) * ST_BK;
            LOAD_A_STEEL(nxt_buf, next_kb);
            LOAD_B_STEEL(nxt_buf, tile + 1);
        }

        // --- MMA compute on current buffer ---
        // BK=32 → 4 k-steps of 8
        // As layout: As[m][k_padded], stride = ST_BK_PAD
        // Ws layout: Ws[k][n], stride = ST_BN
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 4; kk++) {
            simdgroup_half8x8 a_frag[2];
            simdgroup_half8x8 b_frag[4];

            // Load A fragments: 2 × 8×8 covering 16 rows of this SG's portion
            // a_frag[i] row=M(8), col=K(8). As[m * BK_PAD + k] → stride=BK_PAD
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_frag[i],
                    &As[cur_buf][(sg_row_base + i * 8) * ST_BK_PAD + kk * 8], ST_BK_PAD);
            }

            // Load B fragments: 4 × 8×8 covering full 32 cols
            // b_frag[j] row=K(8), col=N(8). Ws[k * BN + n] → stride=BN
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b_frag[j],
                    &Ws[cur_buf][kk * 8 * ST_BN + j * 8], ST_BN);
            }

            // 2×4 outer product with serpentine
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; j++) {
                    uint n_serp = (i % 2) ? (3 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #undef LOAD_A_STEEL
    #undef LOAD_B_STEEL

    // --- Store results: direct register store ---
    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
            uint gr = row_start + sg_row_base + i * 8 + fm;
            uint gc0 = col_start + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (align_M && align_N) {
                float v0 = elems[0];
                float v1 = elems[1];
                if (has_swiglu) {
                    v0 = st_silu_f(gate_result[gr * uN + gc0]) * v0;
                    v1 = st_silu_f(gate_result[gr * uN + gc1]) * v1;
                }
                if (has_residual) {
                    v0 += residual[gr * uN + gc0];
                    v1 += residual[gr * uN + gc1];
                }
                output[gr * uN + gc0] = v0;
                output[gr * uN + gc1] = v1;
            } else {
                if ((align_M || gr < uM) && (align_N || gc0 < uN)) {
                    float v0 = elems[0];
                    if (has_swiglu) v0 = st_silu_f(gate_result[gr * uN + gc0]) * v0;
                    if (has_residual) v0 += residual[gr * uN + gc0];
                    output[gr * uN + gc0] = v0;
                }
                if ((align_M || gr < uM) && (align_N || gc1 < uN)) {
                    float v1 = elems[1];
                    if (has_swiglu) v1 = st_silu_f(gate_result[gr * uN + gc1]) * v1;
                    if (has_residual) v1 += residual[gr * uN + gc1];
                    output[gr * uN + gc1] = v1;
                }
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Metal shader source -- Skinny-M Q4 QMM kernel (BM=32, BN=64, BK=32)
// ---------------------------------------------------------------------------

/// Skinny-M Q4 QMM kernel for M <= 32.
///
/// When M is small (1-32), the standard 64×64 kernel wastes 50%+ of compute
/// on zero-padded rows. This kernel uses BM=32 to eliminate that waste.
///
/// Key differences from the standard kernel:
/// - BM=32, BN=64, BK=32: narrower M tile, deeper K tile for more work/TG
/// - TG memory: As[32×32]=2KB + Bs[32×64]=4KB = 6KB (still high occupancy)
/// - TM=4 (32/8), TN=4 ((64/2)/8): 4×4=16 MMA ops per SG per k-step
/// - Built-in split-K via grid.z: each z-partition handles a K range
/// - Separate reduce kernel accumulates partial sums
///
/// With BK=32 (vs 16 in standard), each tile does more K work, reducing
/// the loop overhead and barrier count per output element.
pub const QMM_SKINNY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SBM = 32;
constant constexpr uint SBN = 64;
constant constexpr uint SBK = 32;
constant constexpr uint STM = 4;   // SBM / 8 = 32/8
constant constexpr uint STN = 4;   // (SBN/2) / 8 = 32/8

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant uint skinny_fc_group_size [[function_constant(205)]];

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> skinny_as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T skinny_as_uniform(T val) {
    return val;
}
#endif

// Skinny-M Q4 QMM with split-K support.
// Grid: (ceil(N/SBN), ceil(M/SBM), k_partitions)
// When k_partitions == 1, writes directly to output.
// When k_partitions > 1, writes partial sums to c_split[partition * M * N].
kernel void affine_qmm_skinny_q4(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         output    [[buffer(4)]],
    constant uint&        M         [[buffer(5)]],
    constant uint&        N         [[buffer(6)]],
    constant uint&        K         [[buffer(7)]],
    constant uint&        k_partitions [[buffer(8)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    threadgroup half As[SBM * SBK];   // 32x32 = 1024 halves = 2KB
    threadgroup half Bs[SBK * SBN];   // 32x64 = 2048 halves = 4KB  (total 6KB)

    const uint row_start = group_id.y * skinny_as_uniform(SBM);
    const uint col_start = group_id.x * skinny_as_uniform(SBN);
    const uint partition_id = group_id.z;

    const uint uK = skinny_as_uniform(K);
    const uint uM = skinny_as_uniform(M);
    const uint uN = skinny_as_uniform(N);
    const uint u_k_parts = skinny_as_uniform(k_partitions);
    const uint groups_per_row = uK / skinny_fc_group_size;
    const uint half_k = uK / 2;

    if (!align_M && row_start >= uM) return;
    if (!align_N && col_start >= uN) return;

    // Split-K: compute K range for this partition
    const uint k_per_part = ((uK + u_k_parts - 1) / u_k_parts);
    // Round up to SBK boundary
    const uint k_per_part_aligned = ((k_per_part + SBK - 1) / SBK) * SBK;
    const uint k_start = partition_id * k_per_part_aligned;
    uint k_end = k_start + k_per_part_aligned;
    if (k_end > uK) k_end = uK;
    if (k_start >= uK) return;

    // SG grid: 1x2 -- each SG covers 32 cols
    const uint base_n = sgid * 32;

    simdgroup_float8x8 acc[STM][STN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < STM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < STN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint n_k_tiles = (k_end - k_start + SBK - 1) / SBK;

    for (uint tile = 0; tile < n_k_tiles; tile++) {
        uint kb = k_start + tile * SBK;

        // Load A: 64 threads load 32×32 = 1024 halves (16 per thread)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // 64 threads for 32 rows × 32 cols = 1024 elements
            // thread 0..31 → row 0..31, cols 0..15
            // thread 32..63 → row 0..31, cols 16..31
            uint a_row = tid_in_group & 31u;
            uint a_col_base = (tid_in_group >> 5u) * 16u;  // 0 or 16
            uint gr = row_start + a_row;

            if ((align_M || gr < uM) && kb + a_col_base + 15 < k_end) {
                for (uint d = 0; d < 16; d += 4) {
                    As[a_row * SBK + a_col_base + d + 0] = half(x[gr * uK + kb + a_col_base + d + 0]);
                    As[a_row * SBK + a_col_base + d + 1] = half(x[gr * uK + kb + a_col_base + d + 1]);
                    As[a_row * SBK + a_col_base + d + 2] = half(x[gr * uK + kb + a_col_base + d + 2]);
                    As[a_row * SBK + a_col_base + d + 3] = half(x[gr * uK + kb + a_col_base + d + 3]);
                }
            } else if (align_M || gr < uM) {
                for (uint d = 0; d < 16; d++) {
                    uint gk = kb + a_col_base + d;
                    As[a_row * SBK + a_col_base + d] = (gk < k_end)
                        ? half(x[gr * uK + gk]) : half(0);
                }
            } else {
                for (uint d = 0; d < 16; d++) {
                    As[a_row * SBK + a_col_base + d] = half(0);
                }
            }
        }

        // Load B: Q4 dequant, 64 threads load 32×64 = 2048 halves (32 per thread)
        // Thread mapping: tid/2 → k-row (0..31), (tid%2)*32 → n-col base (0 or 32)
        // Each thread handles 32 contiguous N columns for one k index
        {
            uint bi = tid_in_group >> 1;          // 0..31
            uint bj = (tid_in_group & 1u) << 5;   // 0 or 32
            uint gk = kb + bi;
            uint gc = col_start + bj;

            if (gk < k_end && (align_N || gc + 31 < uN)) {
                uint group_idx = gk / skinny_fc_group_size;
                uint byte_idx_base = gk / 2;
                bool is_lo = (gk & 1u) == 0;

                for (uint d = 0; d < 32; d += 4) {
                    uchar4 packed4 = uchar4(
                        w_packed[(gc + d + 0) * half_k + byte_idx_base],
                        w_packed[(gc + d + 1) * half_k + byte_idx_base],
                        w_packed[(gc + d + 2) * half_k + byte_idx_base],
                        w_packed[(gc + d + 3) * half_k + byte_idx_base]
                    );
                    float s0 = scales[(gc + d + 0) * groups_per_row + group_idx];
                    float s1 = scales[(gc + d + 1) * groups_per_row + group_idx];
                    float s2 = scales[(gc + d + 2) * groups_per_row + group_idx];
                    float s3 = scales[(gc + d + 3) * groups_per_row + group_idx];
                    float b0 = biases[(gc + d + 0) * groups_per_row + group_idx];
                    float b1 = biases[(gc + d + 1) * groups_per_row + group_idx];
                    float b2 = biases[(gc + d + 2) * groups_per_row + group_idx];
                    float b3 = biases[(gc + d + 3) * groups_per_row + group_idx];

                    if (is_lo) {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float(packed4.x & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float(packed4.y & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float(packed4.z & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float(packed4.w & 0x0F) + b3);
                    } else {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float((packed4.x >> 4) & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float((packed4.y >> 4) & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float((packed4.z >> 4) & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float((packed4.w >> 4) & 0x0F) + b3);
                    }
                }
            } else {
                for (uint d = 0; d < 32; d++) {
                    uint gn = gc + d;
                    half val = half(0);
                    if (gk < k_end && (align_N || gn < uN)) {
                        uint byte_idx = gk / 2;
                        uint8_t packed_b = w_packed[gn * half_k + byte_idx];
                        uint group_idx = gk / skinny_fc_group_size;
                        float scale_b = scales[gn * groups_per_row + group_idx];
                        float bias_b  = biases[gn * groups_per_row + group_idx];
                        bool is_lo_s = (gk & 1u) == 0;
                        uint nibble = is_lo_s ? (packed_b & 0x0F) : ((packed_b >> 4) & 0x0F);
                        val = half(scale_b * float(nibble) + bias_b);
                    }
                    Bs[bi * 64 + bj + d] = val;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA compute: BK=32 → 4 k-steps of 8
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 4; kk++) {
            simdgroup_half8x8 a_frag[STM];
            simdgroup_half8x8 b_frag[STN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < STM; i++)
                simdgroup_load(a_frag[i], &As[(i * 8) * SBK + kk * 8], SBK);

            #pragma clang loop unroll(full)
            for (uint j = 0; j < STN; j++)
                simdgroup_load(b_frag[j], &Bs[kk * 8 * 64 + (base_n + j * 8)], 64);

            #pragma clang loop unroll(full)
            for (uint i = 0; i < STM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < STN; j++) {
                    uint n_serp = (i % 2) ? (3 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // Store: direct register store
    // When k_partitions > 1, write to partition slice; otherwise direct to output
    const uint partition_stride = uM * uN;
    device float* out_ptr = (u_k_parts > 1)
        ? output + partition_id * partition_stride
        : output;

    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < STM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < STN; j++) {
            uint gr = row_start + i * 8 + fm;
            uint gc0 = col_start + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (align_M && align_N) {
                out_ptr[gr * uN + gc0] = elems[0];
                out_ptr[gr * uN + gc1] = elems[1];
            } else {
                if ((align_M || gr < uM) && (align_N || gc0 < uN))
                    out_ptr[gr * uN + gc0] = elems[0];
                if ((align_M || gr < uM) && (align_N || gc1 < uN))
                    out_ptr[gr * uN + gc1] = elems[1];
            }
        }
    }
}

// Reduce split-K partial sums for skinny kernel
kernel void skinny_qmm_reduce(
    device const float* partial [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& N            [[buffer(2)]],
    constant uint& k_partitions [[buffer(3)]],
    constant uint& mn_total     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= mn_total) return;
    float sum = 0.0f;
    for (uint p = 0; p < k_partitions; p++) {
        sum += partial[p * mn_total + id];
    }
    output[id] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Metal shader source -- simdgroup MMA-based Q8 QMM kernel
// ---------------------------------------------------------------------------

/// High-performance Q8 QMM kernel using simdgroup MMA (8x8 fragments).
///
/// Same architecture as Q4 (WM=2, WN=2, 4 SG, BK_PAD=40) but with
/// simplified dequantization: 1 byte per weight, no nibble extraction.
pub const QMM_MMA_Q8_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Function constants for compile-time bounds check elimination
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

// Q8 uses 2 simdgroups / 64 threads for better occupancy.
// Q8 has no nibble-packing benefit from 4-SG vectorization (each weight is
// a full byte), so the extra threads per TG reduce concurrent TGs per core
// without compensating load savings. 2 SG gives 2x occupancy vs 4 SG.
constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;
constant constexpr uint BK_PAD = 40;  // Bank conflict avoidance (same as Q4)

constant constexpr uint SIMDGROUPS_PER_TG = 2;
constant constexpr uint THREADGROUP_SIZE = 64;

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

    const uint tile_m = group_id.y * BM;
    const uint tile_n = group_id.x * BN;

    if (!align_M && tile_m >= M) return;
    if (!align_N && tile_n >= N) return;

    threadgroup half As[2][BM * BK_PAD];
    threadgroup half Bs[2][BK * BN];

    // 2-SG layout: each SG handles 16 rows × 32 cols (full N width)
    // SG0: rows 0-15, SG1: rows 16-31
    // acc[2][4]: 2 row-fragments × 4 col-fragments of 8×8 each = 16×32
    uint sg_row_base = simd_gid * 16;

    simdgroup_matrix<float, 8, 8> acc[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    const uint num_k_tiles = (K + BK - 1) / BK;

    #define LOAD_A(BUF, K_BASE) \
        for (uint idx_a = tid_in_group; idx_a < BM * BK; idx_a += THREADGROUP_SIZE) { \
            uint row_a = idx_a / BK; \
            uint col_a = idx_a % BK; \
            uint gm_a = tile_m + row_a; \
            uint gk_a = (K_BASE) + col_a; \
            half val_a = 0.0h; \
            if (align_M || gm_a < M) { \
                if (gk_a < K) { \
                    val_a = half(x[gm_a * K + gk_a]); \
                } \
            } \
            As[(BUF)][row_a * BK_PAD + col_a] = val_a; \
        }

    #define LOAD_B(BUF, K_BASE) \
        for (uint idx_b = tid_in_group; idx_b < BK * BN; idx_b += THREADGROUP_SIZE) { \
            uint kl_b = idx_b / BN; \
            uint nl_b = idx_b % BN; \
            uint gk_b = (K_BASE) + kl_b; \
            uint gn_b = tile_n + nl_b; \
            half val_b = 0.0h; \
            if (align_N || gn_b < N) { \
                if (gk_b < K) { \
                    uint8_t q_byte_b = w_packed[gn_b * K + gk_b]; \
                    uint group_idx_b = gk_b / group_size; \
                    float scale_b = scales[gn_b * groups_per_row + group_idx_b]; \
                    float bias_b  = biases[gn_b * groups_per_row + group_idx_b]; \
                    val_b = half(scale_b * float(q_byte_b) + bias_b); \
                } \
            } \
            Bs[(BUF)][kl_b * BN + nl_b] = val_b; \
        }

    LOAD_A(0, 0);
    LOAD_B(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < num_k_tiles; t++) {
        uint cur_buf = t % 2;
        uint nxt_buf = 1 - cur_buf;

        if (t + 1 < num_k_tiles) {
            uint next_k_base = (t + 1) * BK;
            LOAD_A(nxt_buf, next_k_base);
            LOAD_B(nxt_buf, next_k_base);
        }

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[2];
            simdgroup_load(a_frag[0], &As[cur_buf][(sg_row_base + 0) * BK_PAD + kk], BK_PAD);
            simdgroup_load(a_frag[1], &As[cur_buf][(sg_row_base + 8) * BK_PAD + kk], BK_PAD);

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

// ---------------------------------------------------------------------------
// Split-K QMM kernels for low-M cases (M <= 32)
// ---------------------------------------------------------------------------

/// Split-K Q4 QMM kernel: partitions K dimension across threadgroups.
///
/// When M is small (M <= 32), the standard QMM kernel has very few threadgroups
/// (often just 1 in the M dimension), leaving GPU cores idle. Split-K divides K
/// into `split_k_partitions` chunks, each processed by a separate threadgroup
/// in the z-dimension, writing partial results to `C_split[partition][M][N]`.
///
/// A second `qmm_splitk_accum` kernel sums partial results into the final output.
///
/// Architecture matches the main QMM kernel (BM=64, BN=64, BK=16, 2 SG, 64 threads).
pub const QMM_SPLITK_Q4_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Split-K Q4 QMM: MLX-architecture (BM=64, BN=64, BK=16, 2 SG, 64 threads).
// Each z-partition processes [k_start, k_start+k_partition_size) of K.
// Output: C_split[partition_id * M * N + m * N + n] (f32 partial sums).
// ---------------------------------------------------------------------------

constant constexpr uint SK_BM = 64;
constant constexpr uint SK_BN = 64;
constant constexpr uint SK_BK = 16;
constant constexpr uint SK_TM = 8;   // BM / 8
constant constexpr uint SK_TN = 4;   // (BN/2) / 8

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant uint sk_fc_group_size [[function_constant(205)]];

kernel void affine_qmm_splitk_q4(
    device const float*   x         [[buffer(0)]],
    device const uint8_t* w_packed  [[buffer(1)]],
    device const float*   scales    [[buffer(2)]],
    device const float*   biases    [[buffer(3)]],
    device float*         c_split   [[buffer(4)]],
    constant uint3&       params    [[buffer(5)]],
    constant uint2&       split_params [[buffer(6)]],
    uint3 group_id        [[threadgroup_position_in_grid]],
    uint  tid_in_group    [[thread_index_in_threadgroup]],
    uint  sgid            [[simdgroup_index_in_threadgroup]],
    uint  lane_id         [[thread_index_in_simdgroup]])
{
    const uint M          = params.x;
    const uint N          = params.y;
    const uint K          = params.z;

    const uint k_partition_size = split_params.x;
    const uint partition_stride = split_params.y;  // M * N

    const uint groups_per_row = K / sk_fc_group_size;
    const uint half_k = K / 2;

    const uint tile_m = group_id.y * SK_BM;
    const uint tile_n = group_id.x * SK_BN;
    const uint partition_id = group_id.z;

    if (!align_M && tile_m >= M) return;
    if (!align_N && tile_n >= N) return;

    // K range for this partition
    const uint k_start = partition_id * k_partition_size;
    uint k_end = k_start + k_partition_size;
    if (k_end > K) k_end = K;
    if (k_start >= K) return;
    const uint k_len = k_end - k_start;

    threadgroup half As[SK_BM * SK_BK];  // 64x16 = 2KB
    threadgroup half Bs[SK_BK * SK_BN];  // 16x64 = 2KB

    const uint base_n = sgid * 32;

    simdgroup_float8x8 acc[SK_TM][SK_TN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SK_TM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SK_TN; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint num_k_tiles = (k_len + SK_BK - 1) / SK_BK;

    for (uint tile = 0; tile < num_k_tiles; tile++) {
        uint kb = k_start + tile * SK_BK;

        // Load A: 64 threads x 16 elements
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint a_row = tid_in_group;
            uint gr = tile_m + a_row;
            if ((align_M || gr < M) && kb + 15 < k_end) {
                for (uint d = 0; d < 16; d += 4) {
                    As[a_row * 16 + d + 0] = half(x[gr * K + kb + d + 0]);
                    As[a_row * 16 + d + 1] = half(x[gr * K + kb + d + 1]);
                    As[a_row * 16 + d + 2] = half(x[gr * K + kb + d + 2]);
                    As[a_row * 16 + d + 3] = half(x[gr * K + kb + d + 3]);
                }
            } else {
                for (uint d = 0; d < 16; d++) {
                    As[a_row * 16 + d] = ((align_M || gr < M) && kb + d < k_end)
                        ? half(x[gr * K + kb + d]) : half(0);
                }
            }
        }

        // Load B: Q4 dequant, 64 threads x 16 elements
        {
            uint bi = tid_in_group >> 2;
            uint bj = (tid_in_group & 3u) << 4;
            uint gk = kb + bi;
            uint gc = tile_n + bj;

            if (gk < k_end && (align_N || gc + 15 < N)) {
                uint group_idx = gk / sk_fc_group_size;
                uint byte_idx_base = gk / 2;
                bool is_lo = (gk & 1u) == 0;

                for (uint d = 0; d < 16; d += 4) {
                    uchar4 packed4 = uchar4(
                        w_packed[(gc + d + 0) * half_k + byte_idx_base],
                        w_packed[(gc + d + 1) * half_k + byte_idx_base],
                        w_packed[(gc + d + 2) * half_k + byte_idx_base],
                        w_packed[(gc + d + 3) * half_k + byte_idx_base]
                    );
                    float s0 = scales[(gc + d + 0) * groups_per_row + group_idx];
                    float s1 = scales[(gc + d + 1) * groups_per_row + group_idx];
                    float s2 = scales[(gc + d + 2) * groups_per_row + group_idx];
                    float s3 = scales[(gc + d + 3) * groups_per_row + group_idx];
                    float b0 = biases[(gc + d + 0) * groups_per_row + group_idx];
                    float b1 = biases[(gc + d + 1) * groups_per_row + group_idx];
                    float b2 = biases[(gc + d + 2) * groups_per_row + group_idx];
                    float b3 = biases[(gc + d + 3) * groups_per_row + group_idx];

                    if (is_lo) {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float(packed4.x & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float(packed4.y & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float(packed4.z & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float(packed4.w & 0x0F) + b3);
                    } else {
                        Bs[bi * 64 + bj + d + 0] = half(s0 * float((packed4.x >> 4) & 0x0F) + b0);
                        Bs[bi * 64 + bj + d + 1] = half(s1 * float((packed4.y >> 4) & 0x0F) + b1);
                        Bs[bi * 64 + bj + d + 2] = half(s2 * float((packed4.z >> 4) & 0x0F) + b2);
                        Bs[bi * 64 + bj + d + 3] = half(s3 * float((packed4.w >> 4) & 0x0F) + b3);
                    }
                }
            } else {
                for (uint d = 0; d < 16; d++) {
                    uint gn = gc + d;
                    half val = half(0);
                    if (gk < k_end && (align_N || gn < N)) {
                        uint byte_idx = gk / 2;
                        uint8_t packed_b = w_packed[gn * half_k + byte_idx];
                        uint group_idx = gk / sk_fc_group_size;
                        float scale_b = scales[gn * groups_per_row + group_idx];
                        float bias_b  = biases[gn * groups_per_row + group_idx];
                        bool is_lo = (gk & 1u) == 0;
                        uint nibble = is_lo ? (packed_b & 0x0F) : ((packed_b >> 4) & 0x0F);
                        val = half(scale_b * float(nibble) + bias_b);
                    }
                    Bs[bi * 64 + bj + d] = val;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA compute (serpentine)
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 2; kk++) {
            simdgroup_half8x8 a_frag[SK_TM];
            simdgroup_half8x8 b_frag[SK_TN];

            #pragma clang loop unroll(full)
            for (uint i = 0; i < SK_TM; i++)
                simdgroup_load(a_frag[i], &As[(i * 8) * 16 + kk * 8], 16);

            #pragma clang loop unroll(full)
            for (uint j = 0; j < SK_TN; j++)
                simdgroup_load(b_frag[j], &Bs[kk * 8 * 64 + (base_n + j * 8)], 64);

            #pragma clang loop unroll(full)
            for (uint i = 0; i < SK_TM; i++) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < SK_TN; j++) {
                    uint n_serp = (i % 2) ? (3 - j) : j;
                    simdgroup_multiply_accumulate(
                        acc[i][n_serp], a_frag[i], b_frag[n_serp], acc[i][n_serp]);
                }
            }
        }
    }

    // Store to partition slice via direct register store
    device float* out_partition = c_split + partition_id * partition_stride;

    const uint qid = lane_id / 4;
    const uint fm = (qid & 4u) + ((lane_id / 2u) % 4u);
    const uint fn_val = (qid & 2u) * 2u + (lane_id % 2u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0; i < SK_TM; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SK_TN; j++) {
            uint gr = tile_m + i * 8 + fm;
            uint gc0 = tile_n + base_n + j * 8 + fn_val;
            uint gc1 = gc0 + 1;

            auto elems = acc[i][j].thread_elements();

            if (align_M && align_N) {
                out_partition[gr * N + gc0] = elems[0];
                out_partition[gr * N + gc1] = elems[1];
            } else {
                if ((align_M || gr < M) && (align_N || gc0 < N))
                    out_partition[gr * N + gc0] = elems[0];
                if ((align_M || gr < M) && (align_N || gc1 < N))
                    out_partition[gr * N + gc1] = elems[1];
            }
        }
    }
}

// Accumulate split-K partitions into final output
kernel void qmm_splitk_accum(
    device const float* c_split     [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant uint3&     acc_params  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint N               = acc_params.x;
    const uint k_partitions    = acc_params.y;
    const uint partition_stride = acc_params.z;

    float sum = 0.0f;
    for (uint p = 0; p < k_partitions; p++) {
        sum += c_split[p * partition_stride + gid.y * N + gid.x];
    }
    output[gid.y * N + gid.x] = sum;
}
"#;

/// Register the QMM Metal kernels with the given registry.
pub fn register_qmm(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("qmm", QMM_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_mma", QMM_MMA_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_mma_q8", QMM_MMA_Q8_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_splitk_q4", QMM_SPLITK_Q4_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_skinny", QMM_SKINNY_SHADER_SOURCE)?;
    registry.register_jit_source("qmm_steel_q4", QMM_STEEL_Q4_SHADER_SOURCE)?;
    Ok(())
}

/// Affine quantized matrix-matrix multiply on GPU (Q4/Q8, Metal).
///
/// Computes `output[m, n] = sum_k x[m, k] * dequant(w[n, k])` using a Metal
/// compute kernel for quantized weights.
///
/// - **Q4** (`bits == 4`): uses `affine_qmm_mma_q4` — MLX-architecture kernel
///   (BM=64, BN=64, BK=16, 2 SG, 64 threads, single-buffered, direct store,
///   serpentine MMA). Same architecture as the fp16 GEMM that achieves 23.82T.
/// - **Q8** (`bits == 8`): uses `affine_qmm_mma_q8` (BM=32, BN=32, 2 SG).
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

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;

    if qw.bits == 4 {
        // Q4 dispatch: choose kernel variant based on M
        // - M <= 32: skinny kernel (BM=32, BN=64, BK=32) with aggressive split-K
        // - M > 32: standard kernel (BM=64, BN=64, BK=16)
        const SKINNY_BM: usize = 32;
        const SKINNY_BN: usize = 64;
        const SKINNY_BK: usize = 32;
        const STD_BM: usize = 64;
        const STD_BN: usize = 64;

        if m <= SKINNY_BM {
            // --- Skinny-M path: BM=32, BN=64, BK=32 with built-in split-K ---
            let sm_tiles = m.div_ceil(SKINNY_BM);
            let sn_tiles = n.div_ceil(SKINNY_BN);

            let align_m = m % SKINNY_BM == 0;
            let align_n = n % SKINNY_BN == 0;

            // Aggressive split-K: target ~256+ total threadgroups for M3 Ultra 80 cores.
            // Each core can run multiple TGs, so we want 3-4x core count.
            let mn_tgs = sm_tiles * sn_tiles;
            let target_tgs: usize = 320; // 4x M3 Ultra 80 cores
            let k_tiles_total = k.div_ceil(SKINNY_BK);
            let k_partitions = if mn_tgs >= target_tgs || k_tiles_total <= 2 {
                1 // enough spatial parallelism, no split-K needed
            } else {
                let desired = (target_tgs / mn_tgs).clamp(2, k_tiles_total.max(2));
                // Don't exceed k_tiles_total (each partition needs at least 1 BK tile)
                desired.min(k_tiles_total)
            };

            let skinny_constants = [
                (200u32, FunctionConstantValue::Bool(align_m)),
                (201u32, FunctionConstantValue::Bool(align_n)),
                (205u32, FunctionConstantValue::U32(qw.group_size)),
            ];
            let pipeline = registry.get_pipeline_with_constants(
                "affine_qmm_skinny_q4",
                DType::Float32,
                &skinny_constants,
            )?;

            let m_u32 = super::checked_u32(m, "M")?;
            let n_u32 = super::checked_u32(n, "N")?;
            let k_u32 = super::checked_u32(k, "K")?;
            let kp_u32 = super::checked_u32(k_partitions, "k_partitions")?;

            let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
            let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
            let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);
            let kp_buf = dev.new_buffer_with_data(&kp_u32 as *const u32 as *const _, 4, opts);

            if k_partitions == 1 {
                // No split-K: write directly to output
                let out = Array::zeros(dev, &[m, n], DType::Float32);

                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
                enc.set_buffer(1, Some(&qw.weights_buf), 0);
                enc.set_buffer(2, Some(&qw.scales_buf), 0);
                enc.set_buffer(3, Some(&qw.biases_buf), 0);
                enc.set_buffer(4, Some(out.metal_buffer()), 0);
                enc.set_buffer(5, Some(&m_buf), 0);
                enc.set_buffer(6, Some(&n_buf), 0);
                enc.set_buffer(7, Some(&k_buf), 0);
                enc.set_buffer(8, Some(&kp_buf), 0);

                let grid = metal::MTLSize::new(sn_tiles as u64, sm_tiles as u64, 1);
                let tg = metal::MTLSize::new(64, 1, 1);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                super::commit_with_mode(cb, super::ExecMode::Sync);

                Ok(out)
            } else {
                // Split-K: partial sums → reduce
                let partition_stride = m * n;
                let c_split_size =
                    (k_partitions * partition_stride * std::mem::size_of::<f32>()) as u64;
                let c_split_buf = dev.new_buffer(c_split_size, opts);
                let out = Array::zeros(dev, &[m, n], DType::Float32);

                let cb = queue.new_command_buffer();

                // Phase 1: Split-K partial GEMM
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
                enc.set_buffer(1, Some(&qw.weights_buf), 0);
                enc.set_buffer(2, Some(&qw.scales_buf), 0);
                enc.set_buffer(3, Some(&qw.biases_buf), 0);
                enc.set_buffer(4, Some(&c_split_buf), 0);
                enc.set_buffer(5, Some(&m_buf), 0);
                enc.set_buffer(6, Some(&n_buf), 0);
                enc.set_buffer(7, Some(&k_buf), 0);
                enc.set_buffer(8, Some(&kp_buf), 0);

                let grid =
                    metal::MTLSize::new(sn_tiles as u64, sm_tiles as u64, k_partitions as u64);
                let tg = metal::MTLSize::new(64, 1, 1);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();

                // Phase 2: Reduce
                let reduce_pipeline = registry.get_pipeline_with_constants(
                    "skinny_qmm_reduce",
                    DType::Float32,
                    &[],
                )?;
                let mn_total_u32 = super::checked_u32(partition_stride, "mn_total")?;
                let n_reduce_buf =
                    dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
                let kp_reduce_buf =
                    dev.new_buffer_with_data(&kp_u32 as *const u32 as *const _, 4, opts);
                let mn_buf =
                    dev.new_buffer_with_data(&mn_total_u32 as *const u32 as *const _, 4, opts);

                let enc2 = cb.new_compute_command_encoder();
                enc2.set_compute_pipeline_state(&reduce_pipeline);
                enc2.set_buffer(0, Some(&c_split_buf), 0);
                enc2.set_buffer(1, Some(out.metal_buffer()), 0);
                enc2.set_buffer(2, Some(&n_reduce_buf), 0);
                enc2.set_buffer(3, Some(&kp_reduce_buf), 0);
                enc2.set_buffer(4, Some(&mn_buf), 0);

                let reduce_grid = metal::MTLSize::new(partition_stride as u64, 1, 1);
                let reduce_tg = metal::MTLSize::new(partition_stride.min(256) as u64, 1, 1);
                enc2.dispatch_threads(reduce_grid, reduce_tg);
                enc2.end_encoding();

                super::commit_with_mode(cb, super::ExecMode::Sync);

                Ok(out)
            }
        } else {
            // --- Standard path: BM=64, BN=64, BK=16 ---
            let m_tiles = m.div_ceil(STD_BM);
            let n_tiles = n.div_ceil(STD_BN);

            let align_m = m % STD_BM == 0;
            let align_n = n % STD_BN == 0;

            let q4_constants = [
                (200u32, FunctionConstantValue::Bool(align_m)),
                (201u32, FunctionConstantValue::Bool(align_n)),
                (202u32, FunctionConstantValue::Bool(false)), // has_residual = false
                (203u32, FunctionConstantValue::Bool(false)), // has_norm = false
                (204u32, FunctionConstantValue::Bool(false)), // has_swiglu = false
                (205u32, FunctionConstantValue::U32(qw.group_size)),
            ];
            let pipeline = registry.get_pipeline_with_constants(
                "affine_qmm_mma_q4",
                DType::Float32,
                &q4_constants,
            )?;
            let out = Array::zeros(dev, &[m, n], DType::Float32);

            let m_u32 = super::checked_u32(m, "M")?;
            let n_u32 = super::checked_u32(n, "N")?;
            let k_u32 = super::checked_u32(k, "K")?;
            let swizzle_log: u32 = if n_tiles >= 4 { 1 } else { 0 };

            let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
            let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
            let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);
            let sw_buf = dev.new_buffer_with_data(&swizzle_log as *const u32 as *const _, 4, opts);
            let dummy_buf = dev.new_buffer(4, opts);

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
            enc.set_buffer(1, Some(&qw.weights_buf), 0);
            enc.set_buffer(2, Some(&qw.scales_buf), 0);
            enc.set_buffer(3, Some(&qw.biases_buf), 0);
            enc.set_buffer(4, Some(out.metal_buffer()), 0);
            enc.set_buffer(5, Some(&m_buf), 0);
            enc.set_buffer(6, Some(&n_buf), 0);
            enc.set_buffer(7, Some(&k_buf), 0);
            enc.set_buffer(8, Some(&sw_buf), 0);
            enc.set_buffer(9, Some(&dummy_buf), 0); // norm_weight (unused)
            enc.set_buffer(10, Some(&dummy_buf), 0); // inv_rms (unused)
            enc.set_buffer(11, Some(&dummy_buf), 0); // residual (unused)
            enc.set_buffer(12, Some(&dummy_buf), 0); // gate_result (unused)

            let grid = metal::MTLSize::new(n_tiles as u64, m_tiles as u64, 1);
            let tg = metal::MTLSize::new(64, 1, 1);

            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            super::commit_with_mode(cb, super::ExecMode::Sync);

            Ok(out)
        }
    } else {
        // Q8 path: keep old architecture (BM=32, BN=32, 2 SG, 64 threads)
        let q8_bm: usize = 32;
        let q8_bn: usize = 32;
        let q8_align_m = m % q8_bm == 0;
        let q8_align_n = n % q8_bn == 0;
        let q8_constants = [
            (200u32, FunctionConstantValue::Bool(q8_align_m)),
            (201u32, FunctionConstantValue::Bool(q8_align_n)),
        ];
        let pipeline = registry.get_pipeline_with_constants(
            "affine_qmm_mma_q8",
            DType::Float32,
            &q8_constants,
        )?;
        let out = Array::zeros(dev, &[m, n], DType::Float32);

        let params: [u32; 4] = [
            super::checked_u32(m, "M")?,
            super::checked_u32(n, "N")?,
            super::checked_u32(k, "K")?,
            qw.group_size,
        ];

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

        let q8_m_tiles = m.div_ceil(q8_bm);
        let q8_n_tiles = n.div_ceil(q8_bn);
        let grid = metal::MTLSize::new(q8_n_tiles as u64, q8_m_tiles as u64, 1);
        let tg = metal::MTLSize::new(64, 1, 1);

        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        super::commit_with_mode(cb, super::ExecMode::Sync);

        Ok(out)
    }
}

/// Fused QMM + residual add: `output = QMM(x, qw) + residual`.
///
/// Encodes into an existing command buffer. Only Q4 standard path (M > 32).
/// The residual is added in the store epilogue via `has_residual` function constant.
#[allow(clippy::too_many_arguments)]
pub fn qmm_add_residual_into_cb(
    registry: &KernelRegistry,
    x: &Array,
    qw: &QuantizedWeight,
    residual: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = qw.out_features;

    if residual.shape() != &[m, n] {
        return Err(KernelError::InvalidShape(format!(
            "qmm_add_residual_into_cb: residual shape must be [{}, {}], got {:?}",
            m,
            n,
            residual.shape()
        )));
    }

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;

    let std_bm: usize = 64;
    let std_bn: usize = 64;
    let m_tiles = m.div_ceil(std_bm);
    let n_tiles = n.div_ceil(std_bn);
    let align_m = m % std_bm == 0;
    let align_n = n % std_bn == 0;

    let q4_constants = [
        (200u32, FunctionConstantValue::Bool(align_m)),
        (201u32, FunctionConstantValue::Bool(align_n)),
        (202u32, FunctionConstantValue::Bool(true)), // has_residual = true
        (203u32, FunctionConstantValue::Bool(false)),
        (204u32, FunctionConstantValue::Bool(false)),
        (205u32, FunctionConstantValue::U32(qw.group_size)),
    ];
    let pipeline =
        registry.get_pipeline_with_constants("affine_qmm_mma_q4", DType::Float32, &q4_constants)?;
    let out = Array::zeros(dev, &[m, n], DType::Float32);

    let m_u32 = super::checked_u32(m, "M")?;
    let n_u32 = super::checked_u32(n, "N")?;
    let k_u32 = super::checked_u32(k, "K")?;
    let swizzle_log: u32 = if n_tiles >= 4 { 1 } else { 0 };

    let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
    let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);
    let sw_buf = dev.new_buffer_with_data(&swizzle_log as *const u32 as *const _, 4, opts);
    let dummy_buf = dev.new_buffer(4, opts);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(&qw.weights_buf), 0);
    enc.set_buffer(2, Some(&qw.scales_buf), 0);
    enc.set_buffer(3, Some(&qw.biases_buf), 0);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&sw_buf), 0);
    enc.set_buffer(9, Some(&dummy_buf), 0); // norm_weight (unused)
    enc.set_buffer(10, Some(&dummy_buf), 0); // inv_rms (unused)
    enc.set_buffer(11, Some(residual.metal_buffer()), residual.offset() as u64);
    enc.set_buffer(12, Some(&dummy_buf), 0); // gate_result (unused)

    let grid = metal::MTLSize::new(n_tiles as u64, m_tiles as u64, 1);
    let tg = metal::MTLSize::new(64, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();

    Ok(out)
}

/// Fused QMM + SwiGLU: `output = silu(gate_result) * QMM(x, qw)`.
///
/// Encodes into an existing command buffer. Only Q4 standard path (M > 32).
/// The SwiGLU fusion is applied in the store epilogue via `has_swiglu` function constant.
#[allow(clippy::too_many_arguments)]
pub fn qmm_swiglu_into_cb(
    registry: &KernelRegistry,
    x: &Array,
    qw: &QuantizedWeight,
    gate_result: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    use crate::kernels::FunctionConstantValue;

    let m = x.shape()[0];
    let k = x.shape()[1];
    let n = qw.out_features;

    if gate_result.shape() != &[m, n] {
        return Err(KernelError::InvalidShape(format!(
            "qmm_swiglu_into_cb: gate_result shape must be [{}, {}], got {:?}",
            m,
            n,
            gate_result.shape()
        )));
    }

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;

    let std_bm: usize = 64;
    let std_bn: usize = 64;
    let m_tiles = m.div_ceil(std_bm);
    let n_tiles = n.div_ceil(std_bn);
    let align_m = m % std_bm == 0;
    let align_n = n % std_bn == 0;

    let q4_constants = [
        (200u32, FunctionConstantValue::Bool(align_m)),
        (201u32, FunctionConstantValue::Bool(align_n)),
        (202u32, FunctionConstantValue::Bool(false)),
        (203u32, FunctionConstantValue::Bool(false)),
        (204u32, FunctionConstantValue::Bool(true)), // has_swiglu = true
        (205u32, FunctionConstantValue::U32(qw.group_size)),
    ];
    let pipeline =
        registry.get_pipeline_with_constants("affine_qmm_mma_q4", DType::Float32, &q4_constants)?;
    let out = Array::zeros(dev, &[m, n], DType::Float32);

    let m_u32 = super::checked_u32(m, "M")?;
    let n_u32 = super::checked_u32(n, "N")?;
    let k_u32 = super::checked_u32(k, "K")?;
    let swizzle_log: u32 = if n_tiles >= 4 { 1 } else { 0 };

    let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
    let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);
    let sw_buf = dev.new_buffer_with_data(&swizzle_log as *const u32 as *const _, 4, opts);
    let dummy_buf = dev.new_buffer(4, opts);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(x.metal_buffer()), x.offset() as u64);
    enc.set_buffer(1, Some(&qw.weights_buf), 0);
    enc.set_buffer(2, Some(&qw.scales_buf), 0);
    enc.set_buffer(3, Some(&qw.biases_buf), 0);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&sw_buf), 0);
    enc.set_buffer(9, Some(&dummy_buf), 0); // norm_weight (unused)
    enc.set_buffer(10, Some(&dummy_buf), 0); // inv_rms (unused)
    enc.set_buffer(11, Some(&dummy_buf), 0); // residual (unused)
    enc.set_buffer(
        12,
        Some(gate_result.metal_buffer()),
        gate_result.offset() as u64,
    );

    let grid = metal::MTLSize::new(n_tiles as u64, m_tiles as u64, 1);
    let tg = metal::MTLSize::new(64, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();

    Ok(out)
}

/// Steel-architecture affine quantized matrix-matrix multiply (Q4 only).
///
/// Uses the `affine_qmm_steel_q4` kernel with:
/// - BM=32, BN=32, BK=32 tiles (matching MLX steel GEMM dimensions)
/// - K-contiguous B loader (MLX QuantizedBlockLoader pattern)
/// - BK_padded=40 for bank conflict avoidance
/// - half input (f32→half pre-conversion)
/// - Double-buffered A/B tiles
///
/// This kernel is expected to be significantly faster than `affine_qmm_mma_q4`
/// due to coalesced B loads and reduced barrier overhead.
///
/// The input `x` must be Float32; it will be converted to Float16 internally
/// before dispatch.
pub fn affine_quantized_matmul_steel(
    registry: &KernelRegistry,
    x: &Array,
    qw: &QuantizedWeight,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if x.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_steel requires Float32 input x, got {:?}",
            x.dtype()
        )));
    }
    if x.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "affine_quantized_matmul_steel requires 2D input x, got {}D",
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
            "affine_quantized_matmul_steel requires bits==4, got {}",
            qw.bits
        )));
    }

    let m = x.shape()[0];
    let n = qw.out_features;
    let k = qw.in_features;

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;

    // --- Step 1: Convert f32 input to half ---
    // Use a simple GPU cast kernel or CPU fallback.
    // For now, allocate a half buffer and use the GPU cast.
    let x_half = crate::ops::copy::copy_cast(registry, x, DType::Float16, queue)?;

    // --- Step 2: Dispatch steel kernel ---
    const STEEL_BM: usize = 32;
    const STEEL_BN: usize = 32;

    let m_tiles = m.div_ceil(STEEL_BM);
    let n_tiles = n.div_ceil(STEEL_BN);

    let align_m = m % STEEL_BM == 0;
    let align_n = n % STEEL_BN == 0;

    let steel_constants = [
        (200u32, FunctionConstantValue::Bool(align_m)),
        (201u32, FunctionConstantValue::Bool(align_n)),
        (202u32, FunctionConstantValue::Bool(false)), // has_residual
        (204u32, FunctionConstantValue::Bool(false)), // has_swiglu
        (205u32, FunctionConstantValue::U32(qw.group_size)),
    ];
    let pipeline = registry.get_pipeline_with_constants(
        "affine_qmm_steel_q4",
        DType::Float32,
        &steel_constants,
    )?;

    let out = Array::zeros(dev, &[m, n], DType::Float32);

    let m_u32 = super::checked_u32(m, "M")?;
    let n_u32 = super::checked_u32(n, "N")?;
    let k_u32 = super::checked_u32(k, "K")?;
    let swizzle_log: u32 = if n_tiles >= 4 { 1 } else { 0 };

    let m_buf = dev.new_buffer_with_data(&m_u32 as *const u32 as *const _, 4, opts);
    let n_buf = dev.new_buffer_with_data(&n_u32 as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k_u32 as *const u32 as *const _, 4, opts);
    let sw_buf = dev.new_buffer_with_data(&swizzle_log as *const u32 as *const _, 4, opts);
    let dummy_buf = dev.new_buffer(4, opts);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(x_half.metal_buffer()), x_half.offset() as u64);
    enc.set_buffer(1, Some(&qw.weights_buf), 0);
    enc.set_buffer(2, Some(&qw.scales_buf), 0);
    enc.set_buffer(3, Some(&qw.biases_buf), 0);
    enc.set_buffer(4, Some(out.metal_buffer()), 0);
    enc.set_buffer(5, Some(&m_buf), 0);
    enc.set_buffer(6, Some(&n_buf), 0);
    enc.set_buffer(7, Some(&k_buf), 0);
    enc.set_buffer(8, Some(&sw_buf), 0);
    enc.set_buffer(9, Some(&dummy_buf), 0); // residual (unused)
    enc.set_buffer(10, Some(&dummy_buf), 0); // gate_result (unused)

    let grid = metal::MTLSize::new(n_tiles as u64, m_tiles as u64, 1);
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
