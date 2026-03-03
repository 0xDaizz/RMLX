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
use metal::MTLSize;

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
    queue: &metal::CommandQueue,
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

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    // Constant buffers.
    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let seq_buf = dev.new_buffer_with_data(&seq_u32 as *const u32 as *const _, 4, opts);
    let dim_buf = dev.new_buffer_with_data(&dim_u32 as *const u32 as *const _, 4, opts);
    let off_buf = dev.new_buffer_with_data(&offset as *const u32 as *const _, 4, opts);
    let scl_buf = dev.new_buffer_with_data(&scale as *const f32 as *const _, 4, opts);
    let trad_buf = dev.new_buffer_with_data(&trad_u32 as *const u32 as *const _, 4, opts);
    let fwd_buf = dev.new_buffer_with_data(&fwd_u32 as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(cos_freqs.metal_buffer()), cos_freqs.offset() as u64);
    encoder.set_buffer(2, Some(sin_freqs.metal_buffer()), sin_freqs.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(&seq_buf), 0);
    encoder.set_buffer(5, Some(&dim_buf), 0);
    encoder.set_buffer(6, Some(&off_buf), 0);
    encoder.set_buffer(7, Some(&scl_buf), 0);
    encoder.set_buffer(8, Some(&trad_buf), 0);
    encoder.set_buffer(9, Some(&fwd_buf), 0);

    let grid = MTLSize::new(half_dim as u64, seq_len as u64, n_batch as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, half_dim as u64),
        std::cmp::min(16, seq_len as u64),
        1,
    );
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API: precomputed-table variant – encode into caller's command buffer
// ---------------------------------------------------------------------------

/// Encode RoPE (precomputed cos/sin) into an existing command buffer (no commit/wait).
///
/// **Caller must ensure all inputs are contiguous** and shapes are valid.
/// Input: [seq_len, head_dim] (2-D) or [batch*n_heads, seq_len, head_dim] (3-D).
/// cos_freqs, sin_freqs: [max_seq_len, head_dim/2].
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
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let (n_batch, seq_len, head_dim) = parse_input_shape(input)?;

    if head_dim % 2 != 0 {
        return Err(KernelError::InvalidShape(format!(
            "head_dim must be even, got {}",
            head_dim
        )));
    }

    let half_dim = head_dim / 2;

    // Kernel selection.
    let kname = kernel_name_table(input.dtype())?;
    let pipeline = registry.get_pipeline(kname, input.dtype())?;

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    // Constant buffers.
    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let seq_buf = dev.new_buffer_with_data(&seq_u32 as *const u32 as *const _, 4, opts);
    let dim_buf = dev.new_buffer_with_data(&dim_u32 as *const u32 as *const _, 4, opts);
    let off_buf = dev.new_buffer_with_data(&offset as *const u32 as *const _, 4, opts);
    let scl_buf = dev.new_buffer_with_data(&scale as *const f32 as *const _, 4, opts);
    let trad_buf = dev.new_buffer_with_data(&trad_u32 as *const u32 as *const _, 4, opts);
    let fwd_buf = dev.new_buffer_with_data(&fwd_u32 as *const u32 as *const _, 4, opts);

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(cos_freqs.metal_buffer()), cos_freqs.offset() as u64);
    encoder.set_buffer(2, Some(sin_freqs.metal_buffer()), sin_freqs.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(&seq_buf), 0);
    encoder.set_buffer(5, Some(&dim_buf), 0);
    encoder.set_buffer(6, Some(&off_buf), 0);
    encoder.set_buffer(7, Some(&scl_buf), 0);
    encoder.set_buffer(8, Some(&trad_buf), 0);
    encoder.set_buffer(9, Some(&fwd_buf), 0);

    let grid = MTLSize::new(half_dim as u64, seq_len as u64, n_batch as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, half_dim as u64),
        std::cmp::min(16, seq_len as u64),
        1,
    );
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();

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
    queue: &metal::CommandQueue,
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

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let seq_u32 = super::checked_u32(seq_len, "seq_len")?;
    let dim_u32 = super::checked_u32(head_dim, "head_dim")?;
    let trad_u32: u32 = if traditional { 1 } else { 0 };
    let fwd_u32: u32 = if forward { 1 } else { 0 };

    let seq_buf = dev.new_buffer_with_data(&seq_u32 as *const u32 as *const _, 4, opts);
    let dim_buf = dev.new_buffer_with_data(&dim_u32 as *const u32 as *const _, 4, opts);
    let off_buf = dev.new_buffer_with_data(&offset as *const u32 as *const _, 4, opts);
    let scl_buf = dev.new_buffer_with_data(&scale as *const f32 as *const _, 4, opts);
    let base_buf = dev.new_buffer_with_data(&base as *const f32 as *const _, 4, opts);
    let trad_buf = dev.new_buffer_with_data(&trad_u32 as *const u32 as *const _, 4, opts);
    let fwd_buf = dev.new_buffer_with_data(&fwd_u32 as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&seq_buf), 0);
    encoder.set_buffer(3, Some(&dim_buf), 0);
    encoder.set_buffer(4, Some(&off_buf), 0);
    encoder.set_buffer(5, Some(&scl_buf), 0);
    encoder.set_buffer(6, Some(&base_buf), 0);
    encoder.set_buffer(7, Some(&trad_buf), 0);
    encoder.set_buffer(8, Some(&fwd_buf), 0);

    let half_dim = head_dim / 2;
    let grid = MTLSize::new(half_dim as u64, seq_len as u64, n_batch as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, half_dim as u64),
        std::cmp::min(16, seq_len as u64),
        1,
    );
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

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
    queue: &metal::CommandQueue,
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
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 2 == 0, "head_dim must be even");
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

    (cos_table, sin_table)
}
