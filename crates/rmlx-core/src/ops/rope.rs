//! Rotary Position Embedding (RoPE).
//!
//! Supports both **traditional** (GPT-NeoX) and **non-traditional** (LLaMA/Mistral)
//! pair indexing layouts, forward and inverse (adjoint) rotation, batched
//! multi-head inputs, on-the-fly frequency computation, and f16/bf16 I/O.
//!
//! ## Pair indexing
//!
//! - **Traditional** (GPT-NeoX): pairs at `(2k, 2k+1)` within each head.
//! - **Non-traditional** (LLaMA): pairs at `(k, k + half_dim)` (split-half).
//!
//! ## Scale semantics (MLX-compatible)
//!
//! Scale is applied to the *position*, not the input vector:
//! ```text
//! theta = scale * (offset + position) * inv_freq
//! ```

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

pub const ROPE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- helpers for f16/bf16 type punning ----

// RoPE kernel (precomputed cos/sin tables).
//
// Grid layout:
//   x = pair index (0 .. half_dim)
//   y = sequence position within the slice
//   z = batch * n_heads (outer dim, only when 3-D input)
//
// Buffer layout:
//   input / output: [n_batch, seq_len, head_dim]  (n_batch may be 1 for 2-D)
//   cos_freqs / sin_freqs: [max_seq_len, half_dim]
//
// Parameters:
//   seq_len   - number of sequence positions in the input
//   head_dim  - full head dimension (must be even)
//   offset    - position offset for incremental decoding
//   scale     - position scale factor  (theta = scale*(offset+pos)*inv_freq)
//   traditional - 1 for GPT-NeoX (2k,2k+1) pairs, 0 for LLaMA split-half
//   forward     - 1 for forward rotation, 0 for inverse/adjoint

kernel void rope_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  offset      [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;               // 0 .. half_dim-1
    uint seq_pos  = gid.y;               // 0 .. seq_len-1
    uint batch    = gid.z;               // 0 .. n_batch-1

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    // Frequency lookup: row = scale * (offset + seq_pos)
    // For integer positions we index the table; scale is folded into theta
    // when using the on-the-fly variant. For precomputed tables we just
    // index at (offset + seq_pos).
    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    // Inverse rotation: negate sin component (transpose of rotation matrix).
    if (!forward) {
        sin_val = -sin_val;
    }

    // Compute element indices depending on layout.
    uint base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = base + pair_idx;
        idx2 = base + pair_idx + half_dim;
    }

    float x0 = input[idx1];
    float x1 = input[idx2];

    output[idx1] = x0 * cos_val - x1 * sin_val;
    output[idx2] = x0 * sin_val + x1 * cos_val;
}

// f16 variant: read/write half, compute in float
kernel void rope_f16(
    device const half* input       [[buffer(0)]],
    device const float* cos_freqs  [[buffer(1)]],
    device const float* sin_freqs  [[buffer(2)]],
    device half*       output      [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  offset      [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint batch    = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    if (!forward) {
        sin_val = -sin_val;
    }

    uint base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = base + pair_idx;
        idx2 = base + pair_idx + half_dim;
    }

    float x0 = float(input[idx1]);
    float x1 = float(input[idx2]);

    output[idx1] = half(x0 * cos_val - x1 * sin_val);
    output[idx2] = half(x0 * sin_val + x1 * cos_val);
}

// bf16 variant: read/write bfloat, compute in float
kernel void rope_bf16(
    device const bfloat* input      [[buffer(0)]],
    device const float*  cos_freqs  [[buffer(1)]],
    device const float*  sin_freqs  [[buffer(2)]],
    device bfloat*       output     [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  offset      [[buffer(6)]],
    constant float& scale       [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint batch    = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    if (!forward) {
        sin_val = -sin_val;
    }

    uint base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = base + pair_idx;
        idx2 = base + pair_idx + half_dim;
    }

    float x0 = float(input[idx1]);
    float x1 = float(input[idx2]);

    output[idx1] = bfloat(x0 * cos_val - x1 * sin_val);
    output[idx2] = bfloat(x0 * sin_val + x1 * cos_val);
}

// ---- On-the-fly frequency computation variant ----
// Computes theta = scale * (offset + seq_pos) * base^(-2k/dim) without
// requiring precomputed cos/sin tables.

// ---- Multi-head strided RoPE (fused deinterleave + RoPE) ----
//
// Reads from interleaved layout [seq_len, num_heads * head_dim] and writes to
// batch-major [num_heads, seq_len, head_dim], applying RoPE in-flight.
// Grid: (half_dim, seq_len, num_heads)
// This fuses per-head copy + per-head RoPE into a single dispatch.

kernel void rope_multihead_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  num_heads   [[buffer(6)]],
    constant uint&  offset      [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;  // 0 .. half_dim-1
    uint seq_pos  = gid.y;  // 0 .. seq_len-1
    uint head     = gid.z;  // 0 .. num_heads-1

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= num_heads) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];
    if (!forward) sin_val = -sin_val;

    // Input layout: [seq_len, ?], row-major with configurable stride
    uint in_base = seq_pos * input_row_stride + head * head_dim;
    uint in_idx1, in_idx2;
    if (traditional) {
        in_idx1 = in_base + 2 * pair_idx;
        in_idx2 = in_idx1 + 1;
    } else {
        in_idx1 = in_base + pair_idx;
        in_idx2 = in_base + pair_idx + half_dim;
    }

    // Output layout: [num_heads, seq_len, head_dim], contiguous
    uint out_base = (head * seq_len + seq_pos) * head_dim;
    uint out_idx1, out_idx2;
    if (traditional) {
        out_idx1 = out_base + 2 * pair_idx;
        out_idx2 = out_idx1 + 1;
    } else {
        out_idx1 = out_base + pair_idx;
        out_idx2 = out_base + pair_idx + half_dim;
    }

    float x0 = input[in_idx1];
    float x1 = input[in_idx2];
    output[out_idx1] = x0 * cos_val - x1 * sin_val;
    output[out_idx2] = x0 * sin_val + x1 * cos_val;
}

kernel void rope_multihead_f16(
    device const half*  input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device half*        output      [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  num_heads   [[buffer(6)]],
    constant uint&  offset      [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= num_heads) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];
    if (!forward) sin_val = -sin_val;

    uint in_base = seq_pos * input_row_stride + head * head_dim;
    uint in_idx1, in_idx2;
    if (traditional) {
        in_idx1 = in_base + 2 * pair_idx;
        in_idx2 = in_idx1 + 1;
    } else {
        in_idx1 = in_base + pair_idx;
        in_idx2 = in_base + pair_idx + half_dim;
    }

    uint out_base = (head * seq_len + seq_pos) * head_dim;
    uint out_idx1, out_idx2;
    if (traditional) {
        out_idx1 = out_base + 2 * pair_idx;
        out_idx2 = out_idx1 + 1;
    } else {
        out_idx1 = out_base + pair_idx;
        out_idx2 = out_base + pair_idx + half_dim;
    }

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);
    output[out_idx1] = half(x0 * cos_val - x1 * sin_val);
    output[out_idx2] = half(x0 * sin_val + x1 * cos_val);
}

kernel void rope_multihead_bf16(
    device const bfloat* input      [[buffer(0)]],
    device const float*  cos_freqs  [[buffer(1)]],
    device const float*  sin_freqs  [[buffer(2)]],
    device bfloat*       output     [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  head_dim    [[buffer(5)]],
    constant uint&  num_heads   [[buffer(6)]],
    constant uint&  offset      [[buffer(7)]],
    constant uint&  traditional [[buffer(8)]],
    constant uint&  forward     [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= num_heads) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];
    if (!forward) sin_val = -sin_val;

    uint in_base = seq_pos * input_row_stride + head * head_dim;
    uint in_idx1, in_idx2;
    if (traditional) {
        in_idx1 = in_base + 2 * pair_idx;
        in_idx2 = in_idx1 + 1;
    } else {
        in_idx1 = in_base + pair_idx;
        in_idx2 = in_base + pair_idx + half_dim;
    }

    uint out_base = (head * seq_len + seq_pos) * head_dim;
    uint out_idx1, out_idx2;
    if (traditional) {
        out_idx1 = out_base + 2 * pair_idx;
        out_idx2 = out_idx1 + 1;
    } else {
        out_idx1 = out_base + pair_idx;
        out_idx2 = out_base + pair_idx + half_dim;
    }

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);
    output[out_idx1] = bfloat(x0 * cos_val - x1 * sin_val);
    output[out_idx2] = bfloat(x0 * sin_val + x1 * cos_val);
}

// ---- Multi-head deinterleave (no RoPE, just layout transform) ----
// Same layout transform as rope_multihead but without rotation.
// For V heads that don't need RoPE.

kernel void deinterleave_heads_f32(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    constant uint& input_row_stride [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = seq_pos * input_row_stride + head * head_dim + d;
    uint out_idx = (head * seq_len + seq_pos) * head_dim + d;
        output[out_idx] = input[in_idx];
}

kernel void deinterleave_heads_f16(
    device const half* input    [[buffer(0)]],
    device half*       output   [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    constant uint& input_row_stride [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = seq_pos * input_row_stride + head * head_dim + d;
    uint out_idx = (head * seq_len + seq_pos) * head_dim + d;
    output[out_idx] = input[in_idx];
}

kernel void deinterleave_heads_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    constant uint& input_row_stride [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = seq_pos * input_row_stride + head * head_dim + d;
    uint out_idx = (head * seq_len + seq_pos) * head_dim + d;
    output[out_idx] = input[in_idx];
}

// ---- Interleave heads (batch-major -> row-major) ----
// Input: [num_heads, seq_len, head_dim] contiguous
// Output: [seq_len, num_heads * head_dim] contiguous

kernel void interleave_heads_f32(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = (head * seq_len + seq_pos) * head_dim + d;
    uint out_idx = seq_pos * (num_heads * head_dim) + head * head_dim + d;
    output[out_idx] = input[in_idx];
}

kernel void interleave_heads_f16(
    device const half* input    [[buffer(0)]],
    device half*       output   [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = (head * seq_len + seq_pos) * head_dim + d;
    uint out_idx = seq_pos * (num_heads * head_dim) + head * head_dim + d;
    output[out_idx] = input[in_idx];
}

kernel void interleave_heads_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint& seq_len      [[buffer(2)]],
    constant uint& head_dim     [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d        = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;
    if (d >= head_dim || seq_pos >= seq_len || head >= num_heads) return;
    uint in_idx  = (head * seq_len + seq_pos) * head_dim + d;
    uint out_idx = seq_pos * (num_heads * head_dim) + head * head_dim + d;
    output[out_idx] = input[in_idx];
}

// ---- Fused Q+K batch RoPE (single dispatch for both Q and K heads) ----
//
// Reads Q and K heads from a merged QKV buffer [seq_len, qkv_dim] and writes
// to two separate output buffers in batch-major layout.
// Grid: (half_dim, seq_len, num_q_heads + num_kv_heads)
//
// Head indices 0..num_q_heads-1 read from Q region, write to output_q.
// Head indices num_q_heads..num_q_heads+num_kv_heads-1 read from K region,
// write to output_k.

kernel void rope_qk_batch_f16(
    device const half*  input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device half*        output_q    [[buffer(3)]],
    device half*        output_k    [[buffer(4)]],
    constant uint&  seq_len         [[buffer(5)]],
    constant uint&  head_dim        [[buffer(6)]],
    constant uint&  num_q_heads     [[buffer(7)]],
    constant uint&  num_kv_heads    [[buffer(8)]],
    constant uint&  offset          [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;  // 0 .. half_dim-1
    uint seq_pos  = gid.y;  // 0 .. seq_len-1
    uint head     = gid.z;  // 0 .. num_q_heads + num_kv_heads - 1

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    // Frequency lookup
    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    // Determine Q vs K head
    bool is_q = (head < num_q_heads);
    uint local_head = is_q ? head : (head - num_q_heads);
    // Q region starts at offset 0, K region starts at num_q_heads * head_dim
    uint head_offset_in_row = is_q ? 0 : (num_q_heads * head_dim);

    // Input: [seq_len, qkv_dim] row-major with configurable stride
    // Traditional pairing: (2k, 2k+1)
    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);

    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    // Output: [num_heads * seq_len, head_dim] batch-major (traditional pairing)
    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q) {
        output_q[out_idx1] = half(out0);
        output_q[out_idx2] = half(out1);
    } else {
        output_k[out_idx1] = half(out0);
        output_k[out_idx2] = half(out1);
    }
}

kernel void rope_qk_batch_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device float*       output_q    [[buffer(3)]],
    device float*       output_k    [[buffer(4)]],
    constant uint&  seq_len         [[buffer(5)]],
    constant uint&  head_dim        [[buffer(6)]],
    constant uint&  num_q_heads     [[buffer(7)]],
    constant uint&  num_kv_heads    [[buffer(8)]],
    constant uint&  offset          [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    bool is_q = (head < num_q_heads);
    uint local_head = is_q ? head : (head - num_q_heads);
    uint head_offset_in_row = is_q ? 0 : (num_q_heads * head_dim);

    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = input[in_idx1];
    float x1 = input[in_idx2];

    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q) {
        output_q[out_idx1] = out0;
        output_q[out_idx2] = out1;
    } else {
        output_k[out_idx1] = out0;
        output_k[out_idx2] = out1;
    }
}

kernel void rope_qk_batch_bf16(
    device const bfloat* input      [[buffer(0)]],
    device const float*  cos_freqs  [[buffer(1)]],
    device const float*  sin_freqs  [[buffer(2)]],
    device bfloat*       output_q   [[buffer(3)]],
    device bfloat*       output_k   [[buffer(4)]],
    constant uint&  seq_len         [[buffer(5)]],
    constant uint&  head_dim        [[buffer(6)]],
    constant uint&  num_q_heads     [[buffer(7)]],
    constant uint&  num_kv_heads    [[buffer(8)]],
    constant uint&  offset          [[buffer(9)]],
    constant uint&  input_row_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    uint freq_row = offset + seq_pos;
    uint freq_idx = freq_row * half_dim + pair_idx;
    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    bool is_q = (head < num_q_heads);
    uint local_head = is_q ? head : (head - num_q_heads);
    uint head_offset_in_row = is_q ? 0 : (num_q_heads * head_dim);

    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);

    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q) {
        output_q[out_idx1] = bfloat(out0);
        output_q[out_idx2] = bfloat(out1);
    } else {
        output_k[out_idx1] = bfloat(out0);
        output_k[out_idx2] = bfloat(out1);
    }
}

// ---- Fused Q+K+V batch RoPE (single dispatch for Q RoPE, K RoPE, V deinterleave) ----
//
// Extends rope_qk_batch to also deinterleave V heads (copy-only, no rotation).
// Grid: (half_dim, seq_len, num_q_heads + 2*num_kv_heads)
//
// Head indices 0..num_q_heads-1: Q region, deinterleave + RoPE -> output_q
// Head indices num_q_heads..num_q_heads+num_kv_heads-1: K region, deinterleave + RoPE -> output_k
// Head indices num_q_heads+num_kv_heads..num_q_heads+2*num_kv_heads-1: V region, copy-only -> output_v

kernel void rope_qkv_batch_f16(
    device const half*  input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device half*        output_q    [[buffer(3)]],
    device half*        output_k    [[buffer(4)]],
    device half*        output_v    [[buffer(5)]],
    constant uint&  seq_len         [[buffer(6)]],
    constant uint&  head_dim        [[buffer(7)]],
    constant uint&  num_q_heads     [[buffer(8)]],
    constant uint&  num_kv_heads    [[buffer(9)]],
    constant uint&  offset          [[buffer(10)]],
    constant uint&  input_row_stride [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;  // 0 .. half_dim-1
    uint seq_pos  = gid.y;  // 0 .. seq_len-1
    uint head     = gid.z;  // 0 .. num_q_heads + 2*num_kv_heads - 1

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    // Determine Q vs K vs V head
    bool is_q = (head < num_q_heads);
    bool is_k = (!is_q && head < num_q_heads + num_kv_heads);
    // else: is_v

    uint local_head;
    uint head_offset_in_row;
    if (is_q) {
        local_head = head;
        head_offset_in_row = 0;
    } else if (is_k) {
        local_head = head - num_q_heads;
        head_offset_in_row = num_q_heads * head_dim;
    } else {
        // V heads: offset after Q and K regions
        local_head = head - num_q_heads - num_kv_heads;
        head_offset_in_row = (num_q_heads + num_kv_heads) * head_dim;
    }

    // Input: [seq_len, qkv_dim] row-major with configurable stride
    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);

    // Output: [num_heads * seq_len, head_dim] batch-major (traditional pairing)
    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q || is_k) {
        // RoPE rotation
        uint freq_row = offset + seq_pos;
        uint freq_idx = freq_row * half_dim + pair_idx;
        float cos_val = cos_freqs[freq_idx];
        float sin_val = sin_freqs[freq_idx];
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;
        if (is_q) {
            output_q[out_idx1] = half(out0);
            output_q[out_idx2] = half(out1);
        } else {
            output_k[out_idx1] = half(out0);
            output_k[out_idx2] = half(out1);
        }
    } else {
        // V: copy-only (deinterleave, no rotation)
        output_v[out_idx1] = half(x0);
        output_v[out_idx2] = half(x1);
    }
}

kernel void rope_qkv_batch_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_freqs   [[buffer(1)]],
    device const float* sin_freqs   [[buffer(2)]],
    device float*       output_q    [[buffer(3)]],
    device float*       output_k    [[buffer(4)]],
    device float*       output_v    [[buffer(5)]],
    constant uint&  seq_len         [[buffer(6)]],
    constant uint&  head_dim        [[buffer(7)]],
    constant uint&  num_q_heads     [[buffer(8)]],
    constant uint&  num_kv_heads    [[buffer(9)]],
    constant uint&  offset          [[buffer(10)]],
    constant uint&  input_row_stride [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    bool is_q = (head < num_q_heads);
    bool is_k = (!is_q && head < num_q_heads + num_kv_heads);

    uint local_head;
    uint head_offset_in_row;
    if (is_q) {
        local_head = head;
        head_offset_in_row = 0;
    } else if (is_k) {
        local_head = head - num_q_heads;
        head_offset_in_row = num_q_heads * head_dim;
    } else {
        local_head = head - num_q_heads - num_kv_heads;
        head_offset_in_row = (num_q_heads + num_kv_heads) * head_dim;
    }

    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = input[in_idx1];
    float x1 = input[in_idx2];

    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q || is_k) {
        uint freq_row = offset + seq_pos;
        uint freq_idx = freq_row * half_dim + pair_idx;
        float cos_val = cos_freqs[freq_idx];
        float sin_val = sin_freqs[freq_idx];
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;
        if (is_q) {
            output_q[out_idx1] = out0;
            output_q[out_idx2] = out1;
        } else {
            output_k[out_idx1] = out0;
            output_k[out_idx2] = out1;
        }
    } else {
        output_v[out_idx1] = x0;
        output_v[out_idx2] = x1;
    }
}

kernel void rope_qkv_batch_bf16(
    device const bfloat* input      [[buffer(0)]],
    device const float*  cos_freqs  [[buffer(1)]],
    device const float*  sin_freqs  [[buffer(2)]],
    device bfloat*       output_q   [[buffer(3)]],
    device bfloat*       output_k   [[buffer(4)]],
    device bfloat*       output_v   [[buffer(5)]],
    constant uint&  seq_len         [[buffer(6)]],
    constant uint&  head_dim        [[buffer(7)]],
    constant uint&  num_q_heads     [[buffer(8)]],
    constant uint&  num_kv_heads    [[buffer(9)]],
    constant uint&  offset          [[buffer(10)]],
    constant uint&  input_row_stride [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint head     = gid.z;

    uint half_dim = head_dim / 2;
    uint total_heads = num_q_heads + 2 * num_kv_heads;
    if (pair_idx >= half_dim || seq_pos >= seq_len || head >= total_heads) return;

    bool is_q = (head < num_q_heads);
    bool is_k = (!is_q && head < num_q_heads + num_kv_heads);

    uint local_head;
    uint head_offset_in_row;
    if (is_q) {
        local_head = head;
        head_offset_in_row = 0;
    } else if (is_k) {
        local_head = head - num_q_heads;
        head_offset_in_row = num_q_heads * head_dim;
    } else {
        local_head = head - num_q_heads - num_kv_heads;
        head_offset_in_row = (num_q_heads + num_kv_heads) * head_dim;
    }

    uint in_base = seq_pos * input_row_stride + head_offset_in_row + local_head * head_dim;
    uint in_idx1 = in_base + 2 * pair_idx;
    uint in_idx2 = in_idx1 + 1;

    float x0 = float(input[in_idx1]);
    float x1 = float(input[in_idx2]);

    uint out_base = (local_head * seq_len + seq_pos) * head_dim;
    uint out_idx1 = out_base + 2 * pair_idx;
    uint out_idx2 = out_idx1 + 1;

    if (is_q || is_k) {
        uint freq_row = offset + seq_pos;
        uint freq_idx = freq_row * half_dim + pair_idx;
        float cos_val = cos_freqs[freq_idx];
        float sin_val = sin_freqs[freq_idx];
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;
        if (is_q) {
            output_q[out_idx1] = bfloat(out0);
            output_q[out_idx2] = bfloat(out1);
        } else {
            output_k[out_idx1] = bfloat(out0);
            output_k[out_idx2] = bfloat(out1);
        }
    } else {
        output_v[out_idx1] = bfloat(x0);
        output_v[out_idx2] = bfloat(x1);
    }
}

// ---- On-the-fly frequency computation variant ----

kernel void rope_otf_f32(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant uint&  seq_len     [[buffer(2)]],
    constant uint&  head_dim    [[buffer(3)]],
    constant uint&  offset      [[buffer(4)]],
    constant float& scale       [[buffer(5)]],
    constant float& base        [[buffer(6)]],
    constant uint&  traditional [[buffer(7)]],
    constant uint&  forward     [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint batch    = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    // inv_freq = base^(-2*pair_idx / head_dim) = exp(-2*pair_idx/dim * log(base))
    float inv_freq = exp(-float(2 * pair_idx) / float(head_dim) * log(base));
    float L = scale * float(offset + seq_pos);
    float theta = L * inv_freq;
    float cos_val = cos(theta);
    float sin_val = sin(theta);

    if (!forward) {
        sin_val = -sin_val;
    }

    uint elem_base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = elem_base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = elem_base + pair_idx;
        idx2 = elem_base + pair_idx + half_dim;
    }

    float x0 = input[idx1];
    float x1 = input[idx2];

    output[idx1] = x0 * cos_val - x1 * sin_val;
    output[idx2] = x0 * sin_val + x1 * cos_val;
}

kernel void rope_otf_f16(
    device const half* input       [[buffer(0)]],
    device half*       output      [[buffer(1)]],
    constant uint&  seq_len     [[buffer(2)]],
    constant uint&  head_dim    [[buffer(3)]],
    constant uint&  offset      [[buffer(4)]],
    constant float& scale       [[buffer(5)]],
    constant float& base        [[buffer(6)]],
    constant uint&  traditional [[buffer(7)]],
    constant uint&  forward     [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint batch    = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    float inv_freq = exp(-float(2 * pair_idx) / float(head_dim) * log(base));
    float L = scale * float(offset + seq_pos);
    float theta = L * inv_freq;
    float cos_val = cos(theta);
    float sin_val = sin(theta);

    if (!forward) {
        sin_val = -sin_val;
    }

    uint elem_base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = elem_base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = elem_base + pair_idx;
        idx2 = elem_base + pair_idx + half_dim;
    }

    float x0 = float(input[idx1]);
    float x1 = float(input[idx2]);

    output[idx1] = half(x0 * cos_val - x1 * sin_val);
    output[idx2] = half(x0 * sin_val + x1 * cos_val);
}

kernel void rope_otf_bf16(
    device const bfloat* input      [[buffer(0)]],
    device bfloat*       output     [[buffer(1)]],
    constant uint&  seq_len     [[buffer(2)]],
    constant uint&  head_dim    [[buffer(3)]],
    constant uint&  offset      [[buffer(4)]],
    constant float& scale       [[buffer(5)]],
    constant float& base        [[buffer(6)]],
    constant uint&  traditional [[buffer(7)]],
    constant uint&  forward     [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint pair_idx = gid.x;
    uint seq_pos  = gid.y;
    uint batch    = gid.z;

    uint half_dim = head_dim / 2;
    if (pair_idx >= half_dim || seq_pos >= seq_len) return;

    float inv_freq = exp(-float(2 * pair_idx) / float(head_dim) * log(base));
    float L = scale * float(offset + seq_pos);
    float theta = L * inv_freq;
    float cos_val = cos(theta);
    float sin_val = sin(theta);

    if (!forward) {
        sin_val = -sin_val;
    }

    uint elem_base = (batch * seq_len + seq_pos) * head_dim;
    uint idx1, idx2;
    if (traditional) {
        idx1 = elem_base + 2 * pair_idx;
        idx2 = idx1 + 1;
    } else {
        idx1 = elem_base + pair_idx;
        idx2 = elem_base + pair_idx + half_dim;
    }

    float x0 = float(input[idx1]);
    float x1 = float(input[idx2]);

    output[idx1] = bfloat(x0 * cos_val - x1 * sin_val);
    output[idx2] = bfloat(x0 * sin_val + x1 * cos_val);
}
"#;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("rope", ROPE_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helper: kernel name selection
// ---------------------------------------------------------------------------

fn kernel_name_table(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rope_f32"),
        DType::Float16 => Ok("rope_f16"),
        DType::Bfloat16 => Ok("rope_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rope not supported for {:?}",
            dtype
        ))),
    }
}

fn kernel_name_otf(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rope_otf_f32"),
        DType::Float16 => Ok("rope_otf_f16"),
        DType::Bfloat16 => Ok("rope_otf_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rope_otf not supported for {:?}",
            dtype
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helper: parse input shape into (n_batch, seq_len, head_dim)
// ---------------------------------------------------------------------------

/// Returns `(n_batch, seq_len, head_dim)`.
fn parse_input_shape(input: &Array) -> Result<(usize, usize, usize), KernelError> {
    match input.ndim() {
        2 => {
            let seq_len = input.shape()[0];
            let head_dim = input.shape()[1];
            Ok((1, seq_len, head_dim))
        }
        3 => {
            let n_batch = input.shape()[0]; // batch * n_heads
            let seq_len = input.shape()[1];
            let head_dim = input.shape()[2];
            Ok((n_batch, seq_len, head_dim))
        }
        n => Err(KernelError::InvalidShape(format!(
            "rope requires 2-D [seq_len, head_dim] or 3-D [batch*n_heads, seq_len, head_dim] \
             input, got {n}-D"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Public API: precomputed-table variant
// ---------------------------------------------------------------------------

/// Apply Rotary Position Embedding using precomputed cos/sin frequency tables.
///
/// # Arguments
///
/// * `input` - `[seq_len, head_dim]` or `[batch*n_heads, seq_len, head_dim]`.
///   `head_dim` must be even.
/// * `cos_freqs` - `[max_seq_len, head_dim/2]` precomputed cosine values.
/// * `sin_freqs` - `[max_seq_len, head_dim/2]` precomputed sine values.
/// * `offset` - Position offset for incremental/KV-cache decoding.
/// * `scale` - Position scale factor (`theta = scale * (offset + pos) * inv_freq`).
///   When using precomputed tables the scale is already baked in;
///   pass `1.0` if your tables already incorporate scaling.
/// * `traditional` - `true` for GPT-NeoX style (2k, 2k+1) pairing,
///   `false` for LLaMA/Mistral split-half (k, k+half_dim).
/// * `forward` - `true` for normal rotation, `false` for inverse/adjoint.
/// * `queue` - Metal command queue.
///
/// # Backward compatibility
///
/// The original 4-parameter `rope()` function is still available below with
/// the same signature (it delegates here with `traditional=true, forward=true`).
#[allow(clippy::too_many_arguments)]
pub fn rope_ext(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    offset: u32,
    scale: f32,
    traditional: bool,
    forward: bool,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let (n_batch, seq_len, head_dim) = parse_input_shape(input)?;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "head_dim must be even, got {}",
            head_dim
        )));
    }

    // Ensure contiguity.
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let cos_contig = super::make_contiguous(cos_freqs, registry, queue)?;
    let cos_freqs = cos_contig.as_ref().unwrap_or(cos_freqs);
    let sin_contig = super::make_contiguous(sin_freqs, registry, queue)?;
    let sin_freqs = sin_contig.as_ref().unwrap_or(sin_freqs);

    // Validate frequency table dimensions.
    let half_dim = head_dim / 2;
    let freq_rows_needed = (offset as usize) + seq_len;
    if cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "cos_freqs and sin_freqs must be 2-D [max_seq_len, half_dim]".into(),
        ));
    }
    if cos_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs rows ({}) < seq_len ({}) + offset ({})",
            cos_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if sin_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs rows ({}) < seq_len ({}) + offset ({})",
            sin_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if cos_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs cols ({}) != head_dim/2 ({})",
            cos_freqs.shape()[1],
            half_dim,
        )));
    }
    if sin_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs cols ({}) != head_dim/2 ({})",
            sin_freqs.shape()[1],
            half_dim,
        )));
    }

    // Kernel selection.
    let kname = kernel_name_table(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &scale as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: n_batch,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, half_dim),
        height: std::cmp::min(16, seq_len),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();
    super::commit_with_mode(&command_buffer, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API: on-the-fly frequency variant
// ---------------------------------------------------------------------------

/// Apply RoPE computing frequencies on-the-fly from `base` and `scale`.
///
/// No precomputed cos/sin tables needed. Frequencies are:
/// ```text
/// inv_freq_k = base^(-2k / head_dim)
/// theta      = scale * (offset + position) * inv_freq_k
/// ```
///
/// # Arguments
///
/// * `input`       - `[seq_len, head_dim]` or `[batch*n_heads, seq_len, head_dim]`.
/// * `offset`      - Position offset for incremental decoding.
/// * `scale`       - Position scale factor.
/// * `base`        - Frequency base (typically `10000.0`).
/// * `traditional` - `true` for (2k, 2k+1), `false` for split-half.
/// * `forward`     - `true` for forward, `false` for inverse/adjoint.
/// * `queue`       - Metal command queue.
#[allow(clippy::too_many_arguments)]
pub fn rope_otf(
    registry: &KernelRegistry,
    input: &Array,
    offset: u32,
    scale: f32,
    base: f32,
    traditional: bool,
    forward: bool,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let (n_batch, seq_len, head_dim) = parse_input_shape(input)?;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "head_dim must be even, got {}",
            head_dim
        )));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let kname = kernel_name_otf(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &scale as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &base as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let half_dim = head_dim / 2;
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: n_batch,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, half_dim),
        height: std::cmp::min(16, seq_len),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();
    super::commit_with_mode(&command_buffer, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Backward-compatible wrapper
// ---------------------------------------------------------------------------

/// Apply Rotary Position Embedding (backward-compatible API).
///
/// Equivalent to `rope_ext(... , traditional=true, forward=true)`.
///
/// - `input`:     `[seq_len, head_dim]` (head_dim must be even).
/// - `cos_freqs`: `[max_seq_len, head_dim/2]`.
/// - `sin_freqs`: `[max_seq_len, head_dim/2]`.
/// - `offset`:    position offset for incremental decoding.
/// - `scale`:     scaling factor (typically `1.0`).
pub fn rope(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    offset: u32,
    scale: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    rope_ext(
        registry, input, cos_freqs, sin_freqs, offset, scale, true, // traditional
        true, // forward
        queue,
    )
}

// ---------------------------------------------------------------------------
// CPU utility: precompute cos/sin frequency tables
// ---------------------------------------------------------------------------

/// Precompute cosine and sine frequency tables on the CPU.
///
/// Returns `(cos_table, sin_table)` each of shape `[max_seq_len, head_dim/2]`
/// as `Vec<f32>`. Scale is applied to the position:
/// ```text
/// theta[pos, k] = scale * pos * base^(-2k / head_dim)
/// ```
///
/// These can be uploaded to GPU Arrays and passed to `rope_ext`.
pub fn precompute_freqs(
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>), KernelError> {
    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "precompute_freqs: head_dim must be even, got {}",
            head_dim
        )));
    }
    let half_dim = head_dim / 2;

    // Precompute inv_freq: base^(-2k/dim) for k in 0..half_dim
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|k| {
            let exponent = -(2.0 * k as f32) / (head_dim as f32);
            base.powf(exponent)
        })
        .collect();

    let total = max_seq_len * half_dim;
    let mut cos_table = vec![0.0f32; total];
    let mut sin_table = vec![0.0f32; total];

    for pos in 0..max_seq_len {
        let l = scale * (pos as f32);
        for (k, &freq) in inv_freq.iter().enumerate().take(half_dim) {
            let theta = l * freq;
            let idx = pos * half_dim + k;
            cos_table[idx] = theta.cos();
            sin_table[idx] = theta.sin();
        }
    }

    Ok((cos_table, sin_table))
}

// ---------------------------------------------------------------------------
// Into-CB variant (encode into existing command buffer, no commit/wait)
// ---------------------------------------------------------------------------

/// Encode RoPE (extended) into an existing command buffer (no commit/wait).
#[allow(clippy::too_many_arguments)]
pub fn rope_ext_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    offset: u32,
    scale: f32,
    traditional: bool,
    forward: bool,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    let (n_batch, seq_len, head_dim) = parse_input_shape(input)?;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "head_dim must be even, got {}",
            head_dim
        )));
    }

    let half_dim = head_dim / 2;

    // Validate frequency table dimensions (matches rope_ext validation).
    let freq_rows_needed = (offset as usize) + seq_len;
    if cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "cos_freqs and sin_freqs must be 2-D [max_seq_len, half_dim]".into(),
        ));
    }
    if cos_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs rows ({}) < seq_len ({}) + offset ({})",
            cos_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if sin_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs rows ({}) < seq_len ({}) + offset ({})",
            sin_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if cos_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs cols ({}) != head_dim/2 ({})",
            cos_freqs.shape()[1],
            half_dim,
        )));
    }
    if sin_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs cols ({}) != head_dim/2 ({})",
            sin_freqs.shape()[1],
            half_dim,
        )));
    }

    let kname = kernel_name_table(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &scale as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: n_batch,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, half_dim),
        height: std::cmp::min(16, seq_len),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok(out)
}

/// Encode RoPE into an existing compute command encoder (no encoder create/end).
/// Caller is responsible for creating and ending the encoder.
#[allow(clippy::too_many_arguments)]
pub fn rope_ext_into_encoder(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    offset: u32,
    scale: f32,
    traditional: bool,
    forward: bool,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    let (n_batch, seq_len, head_dim) = parse_input_shape(input)?;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "head_dim must be even, got {}",
            head_dim
        )));
    }

    let half_dim = head_dim / 2;

    // Validate frequency table dimensions (matches rope_ext validation).
    let freq_rows_needed = (offset as usize) + seq_len;
    if cos_freqs.ndim() != 2 || sin_freqs.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "cos_freqs and sin_freqs must be 2-D [max_seq_len, half_dim]".into(),
        ));
    }
    if cos_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs rows ({}) < seq_len ({}) + offset ({})",
            cos_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if sin_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs rows ({}) < seq_len ({}) + offset ({})",
            sin_freqs.shape()[0],
            seq_len,
            offset,
        )));
    }
    if cos_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs cols ({}) != head_dim/2 ({})",
            cos_freqs.shape()[1],
            half_dim,
        )));
    }
    if sin_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs cols ({}) != head_dim/2 ({})",
            sin_freqs.shape()[1],
            half_dim,
        )));
    }

    let kname = kernel_name_table(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &scale as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: n_batch,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, half_dim),
        height: std::cmp::min(16, seq_len),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode RoPE using a pre-resolved PSO and pre-allocated output buffer.
/// Skips all validation — caller must ensure correctness.
#[allow(clippy::too_many_arguments)]
pub fn rope_ext_preresolved_into_encoder(
    pso: &rmlx_metal::MtlPipeline,
    input_buf: &rmlx_metal::MtlBuffer,
    input_offset: usize,
    cos_buf: &rmlx_metal::MtlBuffer,
    cos_offset: usize,
    sin_buf: &rmlx_metal::MtlBuffer,
    sin_offset: usize,
    out_buf: &rmlx_metal::MtlBuffer,
    out_offset: usize,
    seq_len: u32,
    head_dim: u32,
    offset: u32,
    scale: f32,
    traditional: u32,
    forward: u32,
    n_batch: u64,
    encoder: ComputePass<'_>,
) {
    let half_dim = head_dim / 2;
    encoder.set_pipeline(pso);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] =
            [Some(input_buf), Some(cos_buf), Some(sin_buf), Some(out_buf)];
        let offsets: [usize; 4] = [input_offset, cos_offset, sin_offset, out_offset];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_len as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &head_dim as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &scale as *const f32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &traditional as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &forward as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: half_dim as usize,
        height: seq_len as usize,
        depth: n_batch as usize,
    };
    let tg = MTLSize {
        width: std::cmp::min(64, half_dim as usize),
        height: std::cmp::min(16, seq_len as usize),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
}

/// Get the table-based RoPE kernel name for a dtype.
pub fn rope_table_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    kernel_name_table(dtype)
}

// ---------------------------------------------------------------------------
// Multi-head strided RoPE: fused deinterleave + RoPE in a single dispatch
// ---------------------------------------------------------------------------

fn kernel_name_multihead(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rope_multihead_f32"),
        DType::Float16 => Ok("rope_multihead_f16"),
        DType::Bfloat16 => Ok("rope_multihead_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rope_multihead not supported for {:?}",
            dtype
        ))),
    }
}

fn kernel_name_deinterleave(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("deinterleave_heads_f32"),
        DType::Float16 => Ok("deinterleave_heads_f16"),
        DType::Bfloat16 => Ok("deinterleave_heads_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "deinterleave_heads not supported for {:?}",
            dtype
        ))),
    }
}

fn kernel_name_interleave(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("interleave_heads_f32"),
        DType::Float16 => Ok("interleave_heads_f16"),
        DType::Bfloat16 => Ok("interleave_heads_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "interleave_heads not supported for {:?}",
            dtype
        ))),
    }
}

/// Fused deinterleave + RoPE for multi-head projections.
///
/// Input: `[seq_len, num_heads * head_dim]` (interleaved, contiguous)
/// Output: `[num_heads, seq_len, head_dim]` (batch-major, contiguous)
///
/// Replaces `num_heads` separate copy + RoPE dispatches with a single dispatch.
#[allow(clippy::too_many_arguments)]
pub fn rope_multihead(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_heads: usize,
    offset: u32,
    input_row_stride: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;
    if head_dim * num_heads != total_dim {
        return Err(KernelError::InvalidShape(format!(
            "rope_multihead: total_dim ({total_dim}) not divisible by num_heads ({num_heads})"
        )));
    }
    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "rope_multihead: head_dim must be even, got {head_dim}"
        )));
    }

    let half_dim = head_dim / 2;
    let kname = kernel_name_multihead(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;
    let trad_u32: u32 = 1; // traditional pairing
    let fwd_u32: u32 = 1; // forward

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(10, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Fused deinterleave + RoPE into an existing command buffer (no commit/wait).
#[allow(clippy::too_many_arguments)]
pub fn rope_multihead_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_heads: usize,
    offset: u32,
    input_row_stride: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;

    let half_dim = head_dim / 2;
    let kname = kernel_name_multihead(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;
    let trad_u32: u32 = 1;
    let fwd_u32: u32 = 1;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(10, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok(out)
}

/// Deinterleave heads without RoPE (for V projections).
///
/// Input: `[seq_len, num_heads * head_dim]` (interleaved)
/// Output: `[num_heads, seq_len, head_dim]` (batch-major, contiguous)
pub fn deinterleave_heads(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    input_row_stride: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;

    let kname = kernel_name_deinterleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(5, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, head_dim).min(max_tg);
    let tg_y = std::cmp::min(4, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Deinterleave heads into an existing command buffer (no commit/wait).
pub fn deinterleave_heads_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    input_row_stride: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;

    let kname = kernel_name_deinterleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(5, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, head_dim).min(max_tg);
    let tg_y = std::cmp::min(4, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok(out)
}

/// Interleave heads back into row-major layout.
///
/// Input: `[num_heads, seq_len, head_dim]` (batch-major, contiguous — flat [num_heads*seq_len, head_dim])
/// Output: `[seq_len, num_heads * head_dim]` (row-major, contiguous)
pub fn interleave_heads(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    seq_len: usize,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    let head_dim = input.shape()[input.ndim() - 1];

    let kname = kernel_name_interleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let dev = registry.device().raw();
    let out = Array::uninit(dev, &[seq_len, num_heads * head_dim], input.dtype());

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    let tg = MTLSize {
        width: std::cmp::min(head_dim, 64),
        height: std::cmp::min(seq_len, 4),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Interleave heads into an existing command buffer (no commit/wait).
pub fn interleave_heads_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    seq_len: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    let head_dim = input.shape()[input.ndim() - 1];

    let kname = kernel_name_interleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[seq_len, num_heads * head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    let tg = MTLSize {
        width: std::cmp::min(head_dim, 64),
        height: std::cmp::min(seq_len, 4),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok(out)
}

// ---------------------------------------------------------------------------
// _encode variants — accept ComputePass instead of &CommandBufferRef
// ---------------------------------------------------------------------------

/// Encode multi-head RoPE into an existing compute command encoder (no encoder create/end).
#[allow(clippy::too_many_arguments)]
pub fn rope_multihead_encode(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_heads: usize,
    offset: u32,
    input_row_stride: usize,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;

    let half_dim = head_dim / 2;
    let kname = kernel_name_multihead(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;
    let trad_u32: u32 = 1;
    let fwd_u32: u32 = 1;

    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 4] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out.metal_buffer()),
        ];
        let offsets: [usize; 4] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(4, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(5, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &trad_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &fwd_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(10, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);

    Ok(out)
}

/// Encode head deinterleaving into an existing compute command encoder (no encoder create/end).
pub fn deinterleave_heads_encode(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    input_row_stride: usize,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    let seq_len = input.shape()[0];
    let total_dim = input.shape()[1];
    let head_dim = total_dim / num_heads;

    let kname = kernel_name_deinterleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[num_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;
    encoder.set_bytes(5, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, head_dim).min(max_tg);
    let tg_y = std::cmp::min(4, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    encoder.dispatch_threads(grid, tg);

    Ok(out)
}

/// Encode head interleaving into an existing compute command encoder (no encoder create/end).
pub fn interleave_heads_encode(
    registry: &KernelRegistry,
    input: &Array,
    num_heads: usize,
    seq_len: usize,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    let head_dim = input.shape()[input.ndim() - 1];

    let kname = kernel_name_interleave(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::uninit(
        registry.device().raw(),
        &[seq_len, num_heads * head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let heads_u32 = super::checked_u32(num_heads, "num_heads")?;

    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 2] =
            [Some(input.metal_buffer()), Some(out.metal_buffer())];
        let offsets: [usize; 2] = [input.offset(), 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(2, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(3, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(4, &heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    let grid = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: num_heads,
    };
    let tg = MTLSize {
        width: std::cmp::min(head_dim, 64),
        height: std::cmp::min(seq_len, 4),
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Fused Q+K batch RoPE: single dispatch for both Q and K heads
// ---------------------------------------------------------------------------

fn kernel_name_qk_batch(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rope_qk_batch_f32"),
        DType::Float16 => Ok("rope_qk_batch_f16"),
        DType::Bfloat16 => Ok("rope_qk_batch_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rope_qk_batch not supported for {:?}",
            dtype
        ))),
    }
}

/// Fused deinterleave + RoPE for Q and K heads in a single dispatch.
///
/// Input: `[seq_len, qkv_dim]` merged QKV buffer (Q region followed by K region).
///   The Q region occupies columns `[0, num_q_heads * head_dim)` and
///   the K region occupies columns `[num_q_heads * head_dim, (num_q_heads + num_kv_heads) * head_dim)`.
///
/// Returns `(q_roped, k_roped)`:
/// - `q_roped`: `[num_q_heads * seq_len, head_dim]` (batch-major, contiguous)
/// - `k_roped`: `[num_kv_heads * seq_len, head_dim]` (batch-major, contiguous)
///
/// `input_row_stride` is the stride (in elements) between consecutive rows of
/// the merged QKV buffer — typically `num_q_heads * head_dim + 2 * num_kv_heads * head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn rope_qk_batch_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    offset: u32,
    input_row_stride: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<(Array, Array), KernelError> {
    let seq_len = input.shape()[0];
    let half_dim = head_dim / 2;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "rope_qk_batch: head_dim must be even, got {head_dim}"
        )));
    }

    let kname = kernel_name_qk_batch(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out_q = Array::uninit(
        registry.device().raw(),
        &[num_q_heads * seq_len, head_dim],
        input.dtype(),
    );
    let out_k = Array::uninit(
        registry.device().raw(),
        &[num_kv_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let q_heads_u32 = super::checked_u32(num_q_heads, "num_q_heads")?;
    let kv_heads_u32 = super::checked_u32(num_kv_heads, "num_kv_heads")?;
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;

    let total_heads = num_q_heads + num_kv_heads;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 5] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out_q.metal_buffer()),
            Some(out_k.metal_buffer()),
        ];
        let offsets: [usize; 5] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0, 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(5, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &q_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &kv_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(10, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: total_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok((out_q, out_k))
}

/// Fused Q+K batch RoPE into an existing compute command encoder (no encoder create/end).
#[allow(clippy::too_many_arguments)]
pub fn rope_qk_batch_encode(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    offset: u32,
    input_row_stride: usize,
    encoder: ComputePass<'_>,
) -> Result<(Array, Array), KernelError> {
    let seq_len = input.shape()[0];
    let half_dim = head_dim / 2;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "rope_qk_batch: head_dim must be even, got {head_dim}"
        )));
    }

    let kname = kernel_name_qk_batch(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out_q = Array::uninit(
        registry.device().raw(),
        &[num_q_heads * seq_len, head_dim],
        input.dtype(),
    );
    let out_k = Array::uninit(
        registry.device().raw(),
        &[num_kv_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let q_heads_u32 = super::checked_u32(num_q_heads, "num_q_heads")?;
    let kv_heads_u32 = super::checked_u32(num_kv_heads, "num_kv_heads")?;
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;

    let total_heads = num_q_heads + num_kv_heads;

    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 5] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out_q.metal_buffer()),
            Some(out_k.metal_buffer()),
        ];
        let offsets: [usize; 5] = [input.offset(), cos_freqs.offset(), sin_freqs.offset(), 0, 0];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(5, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(6, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &q_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &kv_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(10, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: total_heads,
    };
    encoder.dispatch_threads(grid, tg);

    Ok((out_q, out_k))
}

// ---------------------------------------------------------------------------
// Fused Q+K+V batch RoPE: single dispatch for Q RoPE, K RoPE, V deinterleave
// ---------------------------------------------------------------------------

fn kernel_name_qkv_batch(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rope_qkv_batch_f32"),
        DType::Float16 => Ok("rope_qkv_batch_f16"),
        DType::Bfloat16 => Ok("rope_qkv_batch_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rope_qkv_batch not supported for {:?}",
            dtype
        ))),
    }
}

/// Fused deinterleave + RoPE for Q and K heads, plus V deinterleave, in a single dispatch.
///
/// Input: `[seq_len, qkv_dim]` merged QKV buffer.
///   Q region: columns `[0, num_q_heads * head_dim)`
///   K region: columns `[num_q_heads * head_dim, (num_q_heads + num_kv_heads) * head_dim)`
///   V region: columns `[(num_q_heads + num_kv_heads) * head_dim, (num_q_heads + 2*num_kv_heads) * head_dim)`
///
/// Returns `(q_roped, k_roped, v_deinterleaved)`:
/// - `q_roped`: `[num_q_heads * seq_len, head_dim]` (batch-major)
/// - `k_roped`: `[num_kv_heads * seq_len, head_dim]` (batch-major)
/// - `v_deinterleaved`: `[num_kv_heads * seq_len, head_dim]` (batch-major)
#[allow(clippy::too_many_arguments)]
pub fn rope_qkv_batch_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    offset: u32,
    input_row_stride: usize,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<(Array, Array, Array), KernelError> {
    let seq_len = input.shape()[0];
    let half_dim = head_dim / 2;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "rope_qkv_batch: head_dim must be even, got {head_dim}"
        )));
    }

    let kname = kernel_name_qkv_batch(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out_q = Array::uninit(
        registry.device().raw(),
        &[num_q_heads * seq_len, head_dim],
        input.dtype(),
    );
    let out_k = Array::uninit(
        registry.device().raw(),
        &[num_kv_heads * seq_len, head_dim],
        input.dtype(),
    );
    let out_v = Array::uninit(
        registry.device().raw(),
        &[num_kv_heads * seq_len, head_dim],
        input.dtype(),
    );

    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let q_heads_u32 = super::checked_u32(num_q_heads, "num_q_heads")?;
    let kv_heads_u32 = super::checked_u32(num_kv_heads, "num_kv_heads")?;
    let stride_u32 = super::checked_u32(input_row_stride, "input_row_stride")?;

    let total_heads = num_q_heads + 2 * num_kv_heads;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    {
        let bufs: [Option<&ProtocolObject<dyn MTLBuffer>>; 6] = [
            Some(input.metal_buffer()),
            Some(cos_freqs.metal_buffer()),
            Some(sin_freqs.metal_buffer()),
            Some(out_q.metal_buffer()),
            Some(out_k.metal_buffer()),
            Some(out_v.metal_buffer()),
        ];
        let offsets: [usize; 6] = [
            input.offset(),
            cos_freqs.offset(),
            sin_freqs.offset(),
            0,
            0,
            0,
        ];
        encoder.set_buffers(0, &bufs, &offsets);
    }
    encoder.set_bytes(6, &seq_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(7, &dim_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(8, &q_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(9, &kv_heads_u32 as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(10, &offset as *const u32 as *const std::ffi::c_void, 4);
    encoder.set_bytes(11, &stride_u32 as *const u32 as *const std::ffi::c_void, 4);
    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_x = std::cmp::min(64, half_dim).min(max_tg);
    let tg_y = std::cmp::min(16, seq_len).min(max_tg / tg_x);
    let tg = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid = MTLSize {
        width: half_dim,
        height: seq_len,
        depth: total_heads,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end();

    Ok((out_q, out_k, out_v))
}
