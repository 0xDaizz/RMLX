//! Fused kernel dispatch patterns for common operation sequences.
//!
//! Reduces dispatch count and intermediate memory traffic by combining
//! commonly co-occurring operations into single command buffer dispatches.
//!
//! # Fusion patterns
//!
//! | Pattern | Current Ops | Fused | Savings |
//! |---------|------------|-------|---------|
//! | Q/K/V Projection | 3 matmuls | Batched GEMM | 3 CBs -> 1 |
//! | SiLU + Mul (SwiGLU) | 2 CBs | 1 CB | 1 less sync |

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Batched Q/K/V projection: single command buffer for 3 matmuls.
///
/// Instead of 3 separate `matmul()` calls (each creating its own CB),
/// this encodes all 3 projections into one command buffer.
///
/// `input`: [seq_len, hidden_size]
/// `wq`, `wk`, `wv`: projection weight matrices (transposed views)
///
/// Returns (Q, K, V) arrays.
pub fn batched_qkv_proj(
    registry: &KernelRegistry,
    input: &Array,
    wq_t: &Array,
    wk_t: &Array,
    wv_t: &Array,
    queue: &metal::CommandQueue,
) -> Result<(Array, Array, Array), KernelError> {
    let input = ensure_contiguous(input, registry, queue)?;

    let m = input.shape()[0] as u32;
    let k = input.shape()[1] as u32;
    let nq = wq_t.shape()[1] as u32;
    let nk = wk_t.shape()[1] as u32;
    let nv = wv_t.shape()[1] as u32;

    let dev = registry.device().raw();
    let q_out = Array::zeros(dev, &[m as usize, nq as usize], input.dtype());
    let k_out = Array::zeros(dev, &[m as usize, nk as usize], input.dtype());
    let v_out = Array::zeros(dev, &[m as usize, nv as usize], input.dtype());

    let kernel_name = gemm_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    // Single command buffer for all 3 projections
    let cb = queue.new_command_buffer();

    // Encode Q projection
    encode_gemm(cb, &pipeline, &input, wq_t, &q_out, m, nq, k, registry)?;

    // Encode K projection
    encode_gemm(cb, &pipeline, &input, wk_t, &k_out, m, nk, k, registry)?;

    // Encode V projection
    encode_gemm(cb, &pipeline, &input, wv_t, &v_out, m, nv, k, registry)?;

    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok((q_out, k_out, v_out))
}

/// Fused SwiGLU: silu(gate(x)) * up(x), single command buffer.
///
/// Combines gate projection, SiLU activation, up projection, and element-wise
/// multiply into a minimal number of encoders on one command buffer.
///
/// `gate_out`: output of gate_proj.forward()
/// `up_out`: output of up_proj.forward()
///
/// Returns: silu(gate_out) * up_out
pub fn fused_silu_mul(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::zeros(dev, gate_out.shape(), gate_out.dtype());
    let numel_buf = make_u32_buf(dev, numel as u32);

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    enc.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    enc.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    enc.set_buffer(3, Some(&numel_buf), 0);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    enc.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();
    super::commit_with_mode(cb, super::ExecMode::Sync);

    Ok(output)
}

/// Encode fused SwiGLU (silu(gate) * up) into an existing command buffer (no commit/wait).
///
/// **Caller must ensure gate_out and up_out are contiguous with matching shapes/dtypes.**
pub fn fused_silu_mul_into_cb(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_cb: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_cb: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul_into_cb: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::uninit(dev, gate_out.shape(), gate_out.dtype());
    let numel_u32 = numel as u32;

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    enc.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    enc.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    enc.set_bytes(3, 4, &numel_u32 as *const u32 as *const std::ffi::c_void);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    enc.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();

    Ok(output)
}

/// Encode fused SiLU * mul into an existing compute command encoder (no encoder create/end).
/// Caller is responsible for creating and ending the encoder.
pub fn fused_silu_mul_into_encoder(
    registry: &KernelRegistry,
    gate_out: &Array,
    up_out: &Array,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    if gate_out.shape() != up_out.shape() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_encoder: shape mismatch: {:?} vs {:?}",
            gate_out.shape(),
            up_out.shape()
        )));
    }
    if gate_out.dtype() != up_out.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "fused_silu_mul_into_encoder: dtype mismatch: {:?} vs {:?}",
            gate_out.dtype(),
            up_out.dtype()
        )));
    }

    let kernel_name = match gate_out.dtype() {
        DType::Float32 => "silu_gate_f32",
        DType::Float16 => "silu_gate_f16",
        DType::Bfloat16 => "silu_gate_bf16",
        other => {
            return Err(KernelError::InvalidShape(format!(
                "fused_silu_mul_into_encoder: unsupported dtype {:?}",
                other
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, gate_out.dtype())?;
    let numel = gate_out.numel();
    let dev = registry.device().raw();
    let output = Array::uninit(dev, gate_out.shape(), gate_out.dtype());
    let numel_u32 = numel as u32;

    let elems_per_thread: u64 = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(gate_out.metal_buffer()), gate_out.offset() as u64);
    encoder.set_buffer(1, Some(up_out.metal_buffer()), up_out.offset() as u64);
    encoder.set_buffer(2, Some(output.metal_buffer()), output.offset() as u64);
    encoder.set_bytes(3, 4, &numel_u32 as *const u32 as *const std::ffi::c_void);

    let tg = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads);
    encoder.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));

    Ok(output)
}

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode fused SiLU * mul using a pre-resolved PSO and pre-allocated output buffer.
#[allow(clippy::too_many_arguments)]
pub fn fused_silu_mul_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    gate_buf: &metal::BufferRef,
    gate_offset: u64,
    up_buf: &metal::BufferRef,
    up_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    numel: u32,
    elems_per_thread: u64,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(gate_buf), gate_offset);
    encoder.set_buffer(1, Some(up_buf), up_offset);
    encoder.set_buffer(2, Some(out_buf), out_offset);
    encoder.set_bytes(3, 4, &numel as *const u32 as *const std::ffi::c_void);
    let tg = std::cmp::min(pso.max_total_threads_per_threadgroup(), grid_threads);
    encoder.dispatch_threads(MTLSize::new(grid_threads, 1, 1), MTLSize::new(tg, 1, 1));
}

/// Get the silu_gate kernel name for a dtype.
pub fn silu_gate_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("silu_gate_f32"),
        DType::Float16 => Ok("silu_gate_f16"),
        DType::Bfloat16 => Ok("silu_gate_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "silu_gate: unsupported dtype {:?}",
            dtype
        ))),
    }
}

/// Get the number of elements per thread for fused silu_mul.
pub fn silu_gate_elems_per_thread(dtype: DType) -> u64 {
    match dtype {
        DType::Float32 => 2,
        _ => 4,
    }
}

/// Batched Q/K/V projection using the CommandBatcher (no commit/wait).
///
/// Same as `batched_qkv_proj` but encodes into a provided command buffer.
pub fn batched_qkv_proj_into(
    registry: &KernelRegistry,
    input: &Array,
    wq_t: &Array,
    wk_t: &Array,
    wv_t: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<(Array, Array, Array), KernelError> {
    // Ensure all inputs are contiguous for the GEMM kernel
    let input = if input.is_contiguous() {
        input.view(
            input.shape().to_vec(),
            input.strides().to_vec(),
            input.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, input, cb)?
    };
    let wq_t = if wq_t.is_contiguous() {
        wq_t.view(
            wq_t.shape().to_vec(),
            wq_t.strides().to_vec(),
            wq_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wq_t, cb)?
    };
    let wk_t = if wk_t.is_contiguous() {
        wk_t.view(
            wk_t.shape().to_vec(),
            wk_t.strides().to_vec(),
            wk_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wk_t, cb)?
    };
    let wv_t = if wv_t.is_contiguous() {
        wv_t.view(
            wv_t.shape().to_vec(),
            wv_t.strides().to_vec(),
            wv_t.offset(),
        )
    } else {
        super::copy::copy_into_cb(registry, wv_t, cb)?
    };

    let m = input.shape()[0] as u32;
    let k = input.shape()[1] as u32;
    let nq = wq_t.shape()[1] as u32;
    let nk = wk_t.shape()[1] as u32;
    let nv = wv_t.shape()[1] as u32;

    let dev = registry.device().raw();
    let q_out = Array::uninit(dev, &[m as usize, nq as usize], input.dtype());
    let k_out = Array::uninit(dev, &[m as usize, nk as usize], input.dtype());
    let v_out = Array::uninit(dev, &[m as usize, nv as usize], input.dtype());

    let kernel_name = gemm_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    encode_gemm(cb, &pipeline, &input, &wq_t, &q_out, m, nq, k, registry)?;
    encode_gemm(cb, &pipeline, &input, &wk_t, &k_out, m, nk, k, registry)?;
    encode_gemm(cb, &pipeline, &input, &wv_t, &v_out, m, nv, k, registry)?;

    Ok((q_out, k_out, v_out))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn gemm_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("gemm_tiled_f32"),
        DType::Float16 => Ok("gemm_tiled_f16"),
        DType::Bfloat16 => Ok("gemm_tiled_bf16"),
        other => Err(KernelError::InvalidShape(format!(
            "fused: unsupported dtype for GEMM: {:?}",
            other
        ))),
    }
}

/// Encode a single GEMM dispatch into a command buffer.
///
/// C[M, N] = A[M, K] @ B[K, N]
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    cb: &metal::CommandBufferRef,
    pipeline: &metal::ComputePipelineState,
    a: &Array,
    b: &Array,
    c: &Array,
    m: u32,
    n: u32,
    k: u32,
    _registry: &KernelRegistry,
) -> Result<(), KernelError> {
    let batch_stride_a = m * k;
    let batch_stride_b = k * n;
    let batch_stride_c = m * n;

    const BM: u64 = 32;
    const BN: u64 = 32;
    let grid_x = (n as u64).div_ceil(BN);
    let grid_y = (m as u64).div_ceil(BM);

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    enc.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    enc.set_buffer(2, Some(c.metal_buffer()), c.offset() as u64);
    enc.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(
        6,
        4,
        &batch_stride_a as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        7,
        4,
        &batch_stride_b as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        8,
        4,
        &batch_stride_c as *const u32 as *const std::ffi::c_void,
    );

    let grid = MTLSize::new(grid_x, grid_y, 1);
    let tg = MTLSize::new(BM * BN, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();

    Ok(())
}

fn make_u32_buf(device: &metal::Device, value: u32) -> metal::Buffer {
    let data = value.to_ne_bytes();
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn ensure_contiguous(
    array: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if array.is_contiguous() {
        Ok(array.view(
            array.shape().to_vec(),
            array.strides().to_vec(),
            array.offset(),
        ))
    } else {
        super::copy::copy(registry, array, queue)
    }
}

// ---------------------------------------------------------------------------
// Fusion B: fused SwiGLU + down-projection
// ---------------------------------------------------------------------------
//
// Combines D8 (silu_gate) + D9 (gemv_bias down_proj) into a single dispatch.
// Buffer layout:
//   0: mat [M, K]       — down_proj weight matrix (row-major)
//   1: gate_up [2*K]    — concatenated [gate_0..gate_{K-1}, up_0..up_{K-1}]
//   2: output [M]
//   3: M (u32)
//   4: K (u32)
//   5: bias [M]         — residual (h_buf)

/// Threadgroup size used for fused SwiGLU+down dispatch.
const FUSED_SWIGLU_DOWN_TG_SIZE: u64 = 256;
/// Number of rows processed per threadgroup (tile-M).
const FUSED_TM: u64 = 4;
/// Number of simdgroups per threadgroup for the BM=8 variant.
const FUSED_BM8: u64 = 8;
/// Rows per threadgroup in BM=8 mode: BM8 * TM = 32.
const FUSED_BM8_ROWS: u64 = FUSED_BM8 * FUSED_TM;
/// Minimum M to use BM=8 variant.
const FUSED_BM8_THRESHOLD: u64 = 256;

pub const FUSED_SWIGLU_DOWN_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint TM = 4;   // rows per threadgroup
constant constexpr uint BM8 = 8;  // simdgroups per threadgroup for bm8 variant

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T as_uniform(T val) {
    return val;
}
#endif

// ---------------------------------------------------------------------------
// SwiGLU helpers — always compute in f32
// ---------------------------------------------------------------------------
inline float stable_sigmoid_f32(float x) {
    float e = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// Compute silu(gate) * up in f32 from scalar values
inline float swiglu_f32(float gate, float up) {
    return gate * stable_sigmoid_f32(gate) * up;
}

// ===========================================================================
// Regular TM=4 variants (M < 256): cross-simdgroup reduction
// ===========================================================================

kernel void fused_swiglu_down_f32(
    device const float* mat     [[buffer(0)]],
    device const float* gate_up [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        // Read gate and up, compute SwiGLU in f32
        float4 g4 = *reinterpret_cast<device const float4*>(gate_up + idx);
        float4 u4 = *reinterpret_cast<device const float4*>(gate_up + K + idx);
        float4 v4 = float4(
            swiglu_f32(g4[0], u4[0]),
            swiglu_f32(g4[1], u4[1]),
            swiglu_f32(g4[2], u4[2]),
            swiglu_f32(g4[3], u4[3])
        );
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
            acc[r] += dot(m4, v4);
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = swiglu_f32(gate_up[i], gate_up[K + i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = val + bias[row_base + r];
            }
        }
    }
}

kernel void fused_swiglu_down_f16(
    device const half*  mat     [[buffer(0)]],
    device const half*  gate_up [[buffer(1)]],
    device       half*  output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const half*  bias    [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        float4 g4 = float4(*reinterpret_cast<device const half4*>(gate_up + idx));
        float4 u4 = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx));
        float4 v4 = float4(
            swiglu_f32(g4[0], u4[0]),
            swiglu_f32(g4[1], u4[1]),
            swiglu_f32(g4[2], u4[2]),
            swiglu_f32(g4[3], u4[3])
        );
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx));
            acc[r] += dot(m4, v4);
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = half(val + float(bias[row_base + r]));
            }
        }
    }
}

kernel void fused_swiglu_down_bf16(
    device const bfloat*  mat     [[buffer(0)]],
    device const bfloat*  gate_up [[buffer(1)]],
    device       bfloat*  output  [[buffer(2)]],
    constant     uint&    M       [[buffer(3)]],
    constant     uint&    K       [[buffer(4)]],
    device const bfloat*  bias    [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // bf16 does not support vector loads — scalar only
    for (uint i = tid; i < K; i += tgsize) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = bfloat(val + float(bias[row_base + r]));
            }
        }
    }
}

// ===========================================================================
// BM=8 variants (M >= 256): no cross-simdgroup barriers, simd_sum only
// ===========================================================================

kernel void fused_swiglu_down_bm8_f32(
    device const float* mat     [[buffer(0)]],
    device const float* gate_up [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered: process 4×float4 (64 bytes) per iteration
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        // Read gate and up vectors, compute SwiGLU
        float4 g4a = *reinterpret_cast<device const float4*>(gate_up + idx);
        float4 u4a = *reinterpret_cast<device const float4*>(gate_up + K + idx);
        float4 g4b = *reinterpret_cast<device const float4*>(gate_up + idx + 4);
        float4 u4b = *reinterpret_cast<device const float4*>(gate_up + K + idx + 4);
        float4 g4c = *reinterpret_cast<device const float4*>(gate_up + idx + 8);
        float4 u4c = *reinterpret_cast<device const float4*>(gate_up + K + idx + 8);
        float4 g4d = *reinterpret_cast<device const float4*>(gate_up + idx + 12);
        float4 u4d = *reinterpret_cast<device const float4*>(gate_up + K + idx + 12);
        float4 v4a = float4(swiglu_f32(g4a[0], u4a[0]), swiglu_f32(g4a[1], u4a[1]),
                            swiglu_f32(g4a[2], u4a[2]), swiglu_f32(g4a[3], u4a[3]));
        float4 v4b = float4(swiglu_f32(g4b[0], u4b[0]), swiglu_f32(g4b[1], u4b[1]),
                            swiglu_f32(g4b[2], u4b[2]), swiglu_f32(g4b[3], u4b[3]));
        float4 v4c = float4(swiglu_f32(g4c[0], u4c[0]), swiglu_f32(g4c[1], u4c[1]),
                            swiglu_f32(g4c[2], u4c[2]), swiglu_f32(g4c[3], u4c[3]));
        float4 v4d = float4(swiglu_f32(g4d[0], u4d[0]), swiglu_f32(g4d[1], u4d[1]),
                            swiglu_f32(g4d[2], u4d[2]), swiglu_f32(g4d[3], u4d[3]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const float* row = mat + (row_base + r) * K + idx;
            float4 m4a = *reinterpret_cast<device const float4*>(row);
            float4 m4b = *reinterpret_cast<device const float4*>(row + 4);
            float4 m4c = *reinterpret_cast<device const float4*>(row + 8);
            float4 m4d = *reinterpret_cast<device const float4*>(row + 12);
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Remainder in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 g4 = *reinterpret_cast<device const float4*>(gate_up + i);
        float4 u4 = *reinterpret_cast<device const float4*>(gate_up + K + i);
        float4 v4 = float4(swiglu_f32(g4[0], u4[0]), swiglu_f32(g4[1], u4[1]),
                           swiglu_f32(g4[2], u4[2]), swiglu_f32(g4[3], u4[3]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + i);
            acc[r] += dot(m4, v4);
        }
    }
    // Scalar remainder
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(gate_up[i], gate_up[K + i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r] + bias[row_base + r];
        }
    }
}

kernel void fused_swiglu_down_bm8_f16(
    device const half*  mat     [[buffer(0)]],
    device const half*  gate_up [[buffer(1)]],
    device       half*  output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const half*  bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered f16: process 4×half4 (32 bytes) per iteration
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 g4a = float4(*reinterpret_cast<device const half4*>(gate_up + idx));
        float4 u4a = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx));
        float4 g4b = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 4));
        float4 u4b = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 4));
        float4 g4c = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 8));
        float4 u4c = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 8));
        float4 g4d = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 12));
        float4 u4d = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 12));
        float4 v4a = float4(swiglu_f32(g4a[0], u4a[0]), swiglu_f32(g4a[1], u4a[1]),
                            swiglu_f32(g4a[2], u4a[2]), swiglu_f32(g4a[3], u4a[3]));
        float4 v4b = float4(swiglu_f32(g4b[0], u4b[0]), swiglu_f32(g4b[1], u4b[1]),
                            swiglu_f32(g4b[2], u4b[2]), swiglu_f32(g4b[3], u4b[3]));
        float4 v4c = float4(swiglu_f32(g4c[0], u4c[0]), swiglu_f32(g4c[1], u4c[1]),
                            swiglu_f32(g4c[2], u4c[2]), swiglu_f32(g4c[3], u4c[3]));
        float4 v4d = float4(swiglu_f32(g4d[0], u4d[0]), swiglu_f32(g4d[1], u4d[1]),
                            swiglu_f32(g4d[2], u4d[2]), swiglu_f32(g4d[3], u4d[3]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const half* row = mat + (row_base + r) * K + idx;
            float4 m4a = float4(*reinterpret_cast<device const half4*>(row));
            float4 m4b = float4(*reinterpret_cast<device const half4*>(row + 4));
            float4 m4c = float4(*reinterpret_cast<device const half4*>(row + 8));
            float4 m4d = float4(*reinterpret_cast<device const half4*>(row + 12));
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Remainder in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 g4 = float4(*reinterpret_cast<device const half4*>(gate_up + i));
        float4 u4 = float4(*reinterpret_cast<device const half4*>(gate_up + K + i));
        float4 v4 = float4(swiglu_f32(g4[0], u4[0]), swiglu_f32(g4[1], u4[1]),
                           swiglu_f32(g4[2], u4[2]), swiglu_f32(g4[3], u4[3]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + i));
            acc[r] += dot(m4, v4);
        }
    }
    // Scalar remainder
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r] + float(bias[row_base + r]));
        }
    }
}

kernel void fused_swiglu_down_bm8_bf16(
    device const bfloat*  mat     [[buffer(0)]],
    device const bfloat*  gate_up [[buffer(1)]],
    device       bfloat*  output  [[buffer(2)]],
    constant     uint&    M       [[buffer(3)]],
    constant     uint&    K       [[buffer(4)]],
    device const bfloat*  bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // bf16 does not support vector loads — scalar only
    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r] + float(bias[row_base + r]));
        }
    }
}

// ===========================================================================
// Interleaved BM8 variants: mat stored as [M/TM, K, TM] instead of [M, K]
// Weight addressing: mat[group * K * TM + k * TM + r]
// where group = row_base / TM, TM=4
// ===========================================================================

kernel void fused_swiglu_down_bm8_f32_interleaved(
    device const float* mat     [[buffer(0)]],
    device const float* gate_up [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const float* bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* tile_base = mat + (row_base / TM) * K * TM;

    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 g4a = *reinterpret_cast<device const float4*>(gate_up + idx);
        float4 u4a = *reinterpret_cast<device const float4*>(gate_up + K + idx);
        float4 g4b = *reinterpret_cast<device const float4*>(gate_up + idx + 4);
        float4 u4b = *reinterpret_cast<device const float4*>(gate_up + K + idx + 4);
        float4 g4c = *reinterpret_cast<device const float4*>(gate_up + idx + 8);
        float4 u4c = *reinterpret_cast<device const float4*>(gate_up + K + idx + 8);
        float4 g4d = *reinterpret_cast<device const float4*>(gate_up + idx + 12);
        float4 u4d = *reinterpret_cast<device const float4*>(gate_up + K + idx + 12);
        float4 v4a = float4(swiglu_f32(g4a[0], u4a[0]), swiglu_f32(g4a[1], u4a[1]),
                            swiglu_f32(g4a[2], u4a[2]), swiglu_f32(g4a[3], u4a[3]));
        float4 v4b = float4(swiglu_f32(g4b[0], u4b[0]), swiglu_f32(g4b[1], u4b[1]),
                            swiglu_f32(g4b[2], u4b[2]), swiglu_f32(g4b[3], u4b[3]));
        float4 v4c = float4(swiglu_f32(g4c[0], u4c[0]), swiglu_f32(g4c[1], u4c[1]),
                            swiglu_f32(g4c[2], u4c[2]), swiglu_f32(g4c[3], u4c[3]));
        float4 v4d = float4(swiglu_f32(g4d[0], u4d[0]), swiglu_f32(g4d[1], u4d[1]),
                            swiglu_f32(g4d[2], u4d[2]), swiglu_f32(g4d[3], u4d[3]));
        // Interleaved weight loads: tile_base[k * TM + r]
        device const float* chunk = tile_base + idx * TM;
        #pragma clang loop unroll(full)
        for (uint sub = 0; sub < 4; sub++) {
            device const float* p = chunk + sub * 4 * TM;
            float4 c0 = *reinterpret_cast<device const float4*>(p);
            float4 c1 = *reinterpret_cast<device const float4*>(p + TM);
            float4 c2 = *reinterpret_cast<device const float4*>(p + 2 * TM);
            float4 c3 = *reinterpret_cast<device const float4*>(p + 3 * TM);
            float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
            acc[0] += c0[0] * v4[0] + c1[0] * v4[1] + c2[0] * v4[2] + c3[0] * v4[3];
            acc[1] += c0[1] * v4[0] + c1[1] * v4[1] + c2[1] * v4[2] + c3[1] * v4[3];
            acc[2] += c0[2] * v4[0] + c1[2] * v4[1] + c2[2] * v4[2] + c3[2] * v4[3];
            acc[3] += c0[3] * v4[0] + c1[3] * v4[1] + c2[3] * v4[2] + c3[3] * v4[3];
        }
    }
    // Remainder in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 g4 = *reinterpret_cast<device const float4*>(gate_up + i);
        float4 u4 = *reinterpret_cast<device const float4*>(gate_up + K + i);
        float4 v4 = float4(swiglu_f32(g4[0], u4[0]), swiglu_f32(g4[1], u4[1]),
                           swiglu_f32(g4[2], u4[2]), swiglu_f32(g4[3], u4[3]));
        device const float* p = tile_base + i * TM;
        float4 c0 = *reinterpret_cast<device const float4*>(p);
        float4 c1 = *reinterpret_cast<device const float4*>(p + TM);
        float4 c2 = *reinterpret_cast<device const float4*>(p + 2 * TM);
        float4 c3 = *reinterpret_cast<device const float4*>(p + 3 * TM);
        acc[0] += c0[0] * v4[0] + c1[0] * v4[1] + c2[0] * v4[2] + c3[0] * v4[3];
        acc[1] += c0[1] * v4[0] + c1[1] * v4[1] + c2[1] * v4[2] + c3[1] * v4[3];
        acc[2] += c0[2] * v4[0] + c1[2] * v4[1] + c2[2] * v4[2] + c3[2] * v4[3];
        acc[3] += c0[3] * v4[0] + c1[3] * v4[1] + c2[3] * v4[2] + c3[3] * v4[3];
    }
    // Scalar remainder
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(gate_up[i], gate_up[K + i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += tile_base[i * TM + r] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r] + bias[row_base + r];
        }
    }
}

kernel void fused_swiglu_down_bm8_f16_interleaved(
    device const half*  mat     [[buffer(0)]],
    device const half*  gate_up [[buffer(1)]],
    device       half*  output  [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  K       [[buffer(4)]],
    device const half*  bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const half* tile_base = mat + (row_base / TM) * K * TM;

    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 g4a = float4(*reinterpret_cast<device const half4*>(gate_up + idx));
        float4 u4a = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx));
        float4 g4b = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 4));
        float4 u4b = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 4));
        float4 g4c = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 8));
        float4 u4c = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 8));
        float4 g4d = float4(*reinterpret_cast<device const half4*>(gate_up + idx + 12));
        float4 u4d = float4(*reinterpret_cast<device const half4*>(gate_up + K + idx + 12));
        float4 v4a = float4(swiglu_f32(g4a[0], u4a[0]), swiglu_f32(g4a[1], u4a[1]),
                            swiglu_f32(g4a[2], u4a[2]), swiglu_f32(g4a[3], u4a[3]));
        float4 v4b = float4(swiglu_f32(g4b[0], u4b[0]), swiglu_f32(g4b[1], u4b[1]),
                            swiglu_f32(g4b[2], u4b[2]), swiglu_f32(g4b[3], u4b[3]));
        float4 v4c = float4(swiglu_f32(g4c[0], u4c[0]), swiglu_f32(g4c[1], u4c[1]),
                            swiglu_f32(g4c[2], u4c[2]), swiglu_f32(g4c[3], u4c[3]));
        float4 v4d = float4(swiglu_f32(g4d[0], u4d[0]), swiglu_f32(g4d[1], u4d[1]),
                            swiglu_f32(g4d[2], u4d[2]), swiglu_f32(g4d[3], u4d[3]));
        // Interleaved weight loads: tile_base[k * TM + r]
        device const half* chunk = tile_base + idx * TM;
        #pragma clang loop unroll(full)
        for (uint sub = 0; sub < 4; sub++) {
            device const half* p = chunk + sub * 4 * TM;
            half4 c0 = *reinterpret_cast<device const half4*>(p);
            half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
            half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
            half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
            float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
            acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
            acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
            acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
            acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
        }
    }
    // Remainder in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 g4 = float4(*reinterpret_cast<device const half4*>(gate_up + i));
        float4 u4 = float4(*reinterpret_cast<device const half4*>(gate_up + K + i));
        float4 v4 = float4(swiglu_f32(g4[0], u4[0]), swiglu_f32(g4[1], u4[1]),
                           swiglu_f32(g4[2], u4[2]), swiglu_f32(g4[3], u4[3]));
        device const half* p = tile_base + i * TM;
        half4 c0 = *reinterpret_cast<device const half4*>(p);
        half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
        half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
        half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
        acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
        acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
        acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
        acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
    }
    // Scalar remainder
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(tile_base[i * TM + r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r] + float(bias[row_base + r]));
        }
    }
}

kernel void fused_swiglu_down_bm8_bf16_interleaved(
    device const bfloat*  mat     [[buffer(0)]],
    device const bfloat*  gate_up [[buffer(1)]],
    device       bfloat*  output  [[buffer(2)]],
    constant     uint&    M       [[buffer(3)]],
    constant     uint&    K       [[buffer(4)]],
    device const bfloat*  bias    [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const bfloat* tile_base = mat + (row_base / TM) * K * TM;

    // bf16 interleaved: scalar only
    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = swiglu_f32(float(gate_up[i]), float(gate_up[K + i]));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(tile_base[i * TM + r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r] + float(bias[row_base + r]));
        }
    }
}
"#;

/// Get the fused SwiGLU+down kernel name for a given dtype and M dimension.
///
/// Returns the BM8 variant for M >= 256, regular variant otherwise.
pub fn fused_swiglu_down_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    let use_bm8 = (m as u64) >= FUSED_BM8_THRESHOLD;
    match (dtype, use_bm8) {
        (DType::Float32, true) => Ok("fused_swiglu_down_bm8_f32"),
        (DType::Float32, false) => Ok("fused_swiglu_down_f32"),
        (DType::Float16, true) => Ok("fused_swiglu_down_bm8_f16"),
        (DType::Float16, false) => Ok("fused_swiglu_down_f16"),
        (DType::Bfloat16, true) => Ok("fused_swiglu_down_bm8_bf16"),
        (DType::Bfloat16, false) => Ok("fused_swiglu_down_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "fused_swiglu_down not supported for {:?}",
            dtype
        ))),
    }
}

/// Get the fused SwiGLU+down interleaved kernel name for a given dtype.
///
/// Only BM8 variants are available (M >= 256 assumed by caller).
pub fn fused_swiglu_down_interleaved_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("fused_swiglu_down_bm8_f32_interleaved"),
        DType::Float16 => Ok("fused_swiglu_down_bm8_f16_interleaved"),
        DType::Bfloat16 => Ok("fused_swiglu_down_bm8_bf16_interleaved"),
        _ => Err(KernelError::NotFound(format!(
            "fused_swiglu_down_interleaved not supported for {:?}",
            dtype
        ))),
    }
}

/// Compute dispatch grid and threadgroup sizes for fused SwiGLU+down with given M.
pub fn fused_swiglu_down_dispatch_sizes(
    m: u32,
    pso: &metal::ComputePipelineState,
) -> (metal::MTLSize, metal::MTLSize) {
    let use_bm8 = (m as u64) >= FUSED_BM8_THRESHOLD;
    let num_threadgroups = if use_bm8 {
        fused_ceil_div(m as u64, FUSED_BM8_ROWS)
    } else {
        fused_ceil_div(m as u64, FUSED_TM)
    };
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, FUSED_BM8)
    } else {
        let tg_size = std::cmp::min(
            FUSED_SWIGLU_DOWN_TG_SIZE,
            pso.max_total_threads_per_threadgroup(),
        );
        MTLSize::new(tg_size, 1, 1)
    };
    (MTLSize::new(num_threadgroups, 1, 1), tg_dim)
}

/// Encode fused SwiGLU+down using a pre-resolved PSO and pre-allocated buffers.
///
/// Buffer layout:
/// - 0: mat [M, K]     (down_proj weights)
/// - 1: gate_up [2*K]  (concatenated gate and up activations)
/// - 2: output [M]
/// - 3: M (u32)
/// - 4: K (u32)
/// - 5: bias [M]       (residual / h_buf)
#[allow(clippy::too_many_arguments)]
pub fn fused_swiglu_down_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    mat_buf: &metal::BufferRef,
    mat_offset: u64,
    gate_up_buf: &metal::BufferRef,
    gate_up_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    m: u32,
    k: u32,
    bias_buf: &metal::BufferRef,
    bias_offset: u64,
    grid: metal::MTLSize,
    tg: metal::MTLSize,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(mat_buf), mat_offset);
    encoder.set_buffer(1, Some(gate_up_buf), gate_up_offset);
    encoder.set_buffer(2, Some(out_buf), out_offset);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.set_buffer(5, Some(bias_buf), bias_offset);
    encoder.dispatch_thread_groups(grid, tg);
}

/// Register all fused kernel shaders with the kernel registry.
pub fn register_fused_kernels(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("fused_swiglu_down", FUSED_SWIGLU_DOWN_SHADER_SOURCE)?;
    registry.register_jit_source("fused_rms_gemv", FUSED_RMS_GEMV_SHADER_SOURCE)
}

/// Compute ceil(a / b) for unsigned integers.
#[allow(clippy::manual_div_ceil)]
fn fused_ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// Fusion A: fused RMS-norm + GEMV
// ---------------------------------------------------------------------------
//
// Combines D1 (rms_norm) + D2 (gemv) into a single dispatch.
// Each threadgroup redundantly computes inv_rms (no cross-TG sync in Metal),
// then performs GEMV with inline normalization.
//
// Buffer layout:
//   0: input [K]         — raw pre-norm input vector
//   1: norm_weight [K]   — RMS norm weight vector
//   2: mat [M, K]        — projection weight matrix (row-major)
//   3: output [M]
//   4: M (u32)
//   5: K (u32)
//   6: eps (f32)         — RMS norm epsilon
//   7: w_stride (u32)    — norm weight stride (usually 1)

/// Threadgroup size used for fused RMS-norm + GEMV dispatch.
const FUSED_RMS_GEMV_TG_SIZE: u64 = 256;

pub const FUSED_RMS_GEMV_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint TM = 4;   // rows per simdgroup (BM8) or per threadgroup (regular)
constant constexpr uint BM8 = 8;  // simdgroups per threadgroup for bm8 variant

#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T as_uniform(T val) {
    return val;
}
#endif

// ===========================================================================
// Regular TM=4 variants (M < 256): cross-simdgroup reduction
// ===========================================================================

kernel void fused_rms_gemv_f32(
    device const float* input       [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* mat         [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];

    // --- Phase 1: compute inv_rms cooperatively ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += tgsize) {
        float v = input[i];
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: GEMV with inline normalization ---
    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (w_stride == 1) {
        uint k4 = as_uniform(K / 4);
        for (uint i = tid; i < k4; i += tgsize) {
            uint idx = i * 4;
            float4 in4 = *reinterpret_cast<device const float4*>(input + idx);
            float4 nw4 = *reinterpret_cast<device const float4*>(norm_weight + idx);
            float4 v4 = in4 * inv_rms * nw4;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
                acc[r] += dot(m4, v4);
            }
        }
        for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
            float v = input[i] * inv_rms * norm_weight[i];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    } else {
        for (uint i = tid; i < K; i += tgsize) {
            float v = input[i] * inv_rms * norm_weight[i * w_stride];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    // Cross-simdgroup reduction
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = val;
            }
        }
    }
}

kernel void fused_rms_gemv_f16(
    device const half*  input       [[buffer(0)]],
    device const half*  norm_weight [[buffer(1)]],
    device const half*  mat         [[buffer(2)]],
    device       half*  output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];

    // --- Phase 1: compute inv_rms cooperatively ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += tgsize) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: GEMV with inline normalization ---
    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (w_stride == 1) {
        uint k4 = as_uniform(K / 4);
        for (uint i = tid; i < k4; i += tgsize) {
            uint idx = i * 4;
            float4 in4 = float4(*reinterpret_cast<device const half4*>(input + idx));
            float4 nw4 = float4(*reinterpret_cast<device const half4*>(norm_weight + idx));
            float4 v4 = in4 * inv_rms * nw4;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx));
                acc[r] += dot(m4, v4);
            }
        }
        for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    } else {
        for (uint i = tid; i < K; i += tgsize) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    // Cross-simdgroup reduction
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = half(val);
            }
        }
    }
}

kernel void fused_rms_gemv_bf16(
    device const bfloat* input       [[buffer(0)]],
    device const bfloat* norm_weight [[buffer(1)]],
    device const bfloat* mat         [[buffer(2)]],
    device       bfloat* output      [[buffer(3)]],
    constant     uint&   M           [[buffer(4)]],
    constant     uint&   K           [[buffer(5)]],
    constant     float&  eps         [[buffer(6)]],
    constant     uint&   w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];

    // --- Phase 1: compute inv_rms cooperatively ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += tgsize) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: GEMV with inline normalization (bf16: scalar only) ---
    uint row_base = tg_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K; i += tgsize) {
        float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    // Cross-simdgroup reduction
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
                output[row_base + r] = bfloat(val);
            }
        }
    }
}

// ===========================================================================
// BM=8 variants (M >= 256): each simdgroup handles TM=4 rows independently
// Phase 1 needs threadgroup_barrier for inv_rms; Phase 2 is barrier-free.
// ===========================================================================

kernel void fused_rms_gemv_bm8_f32(
    device const float* input       [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* mat         [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = input[i];
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (w_stride == 1) {
        uint k16 = as_uniform(K / 16);
        for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
            uint idx = i * 16;
            float4 in4a = *reinterpret_cast<device const float4*>(input + idx);
            float4 nw4a = *reinterpret_cast<device const float4*>(norm_weight + idx);
            float4 v4a = in4a * inv_rms * nw4a;
            float4 in4b = *reinterpret_cast<device const float4*>(input + idx + 4);
            float4 nw4b = *reinterpret_cast<device const float4*>(norm_weight + idx + 4);
            float4 v4b = in4b * inv_rms * nw4b;
            float4 in4c = *reinterpret_cast<device const float4*>(input + idx + 8);
            float4 nw4c = *reinterpret_cast<device const float4*>(norm_weight + idx + 8);
            float4 v4c = in4c * inv_rms * nw4c;
            float4 in4d = *reinterpret_cast<device const float4*>(input + idx + 12);
            float4 nw4d = *reinterpret_cast<device const float4*>(norm_weight + idx + 12);
            float4 v4d = in4d * inv_rms * nw4d;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                device const float* row = mat + (row_base + r) * K + idx;
                float4 m4a = *reinterpret_cast<device const float4*>(row);
                float4 m4b = *reinterpret_cast<device const float4*>(row + 4);
                float4 m4c = *reinterpret_cast<device const float4*>(row + 8);
                float4 m4d = *reinterpret_cast<device const float4*>(row + 12);
                acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
            }
        }
        // Remainder in groups of 4
        for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
            float4 in4 = *reinterpret_cast<device const float4*>(input + i);
            float4 nw4 = *reinterpret_cast<device const float4*>(norm_weight + i);
            float4 v4 = in4 * inv_rms * nw4;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + i);
                acc[r] += dot(m4, v4);
            }
        }
        // Scalar remainder
        for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = input[i] * inv_rms * norm_weight[i];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    } else {
        for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = input[i] * inv_rms * norm_weight[i * w_stride];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r];
        }
    }
}

kernel void fused_rms_gemv_bm8_f16(
    device const half*  input       [[buffer(0)]],
    device const half*  norm_weight [[buffer(1)]],
    device const half*  mat         [[buffer(2)]],
    device       half*  output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (w_stride == 1) {
        uint k16 = as_uniform(K / 16);
        for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
            uint idx = i * 16;
            float4 in4a = float4(*reinterpret_cast<device const half4*>(input + idx));
            float4 nw4a = float4(*reinterpret_cast<device const half4*>(norm_weight + idx));
            float4 v4a = in4a * inv_rms * nw4a;
            float4 in4b = float4(*reinterpret_cast<device const half4*>(input + idx + 4));
            float4 nw4b = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 4));
            float4 v4b = in4b * inv_rms * nw4b;
            float4 in4c = float4(*reinterpret_cast<device const half4*>(input + idx + 8));
            float4 nw4c = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 8));
            float4 v4c = in4c * inv_rms * nw4c;
            float4 in4d = float4(*reinterpret_cast<device const half4*>(input + idx + 12));
            float4 nw4d = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 12));
            float4 v4d = in4d * inv_rms * nw4d;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                device const half* row = mat + (row_base + r) * K + idx;
                float4 m4a = float4(*reinterpret_cast<device const half4*>(row));
                float4 m4b = float4(*reinterpret_cast<device const half4*>(row + 4));
                float4 m4c = float4(*reinterpret_cast<device const half4*>(row + 8));
                float4 m4d = float4(*reinterpret_cast<device const half4*>(row + 12));
                acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
            }
        }
        // Remainder in groups of 4
        for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
            float4 in4 = float4(*reinterpret_cast<device const half4*>(input + i));
            float4 nw4 = float4(*reinterpret_cast<device const half4*>(norm_weight + i));
            float4 v4 = in4 * inv_rms * nw4;
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + i));
                acc[r] += dot(m4, v4);
            }
        }
        // Scalar remainder
        for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    } else {
        for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r]);
        }
    }
}

kernel void fused_rms_gemv_bm8_bf16(
    device const bfloat* input       [[buffer(0)]],
    device const bfloat* norm_weight [[buffer(1)]],
    device const bfloat* mat         [[buffer(2)]],
    device       bfloat* output      [[buffer(3)]],
    constant     uint&   M           [[buffer(4)]],
    constant     uint&   K           [[buffer(5)]],
    constant     float&  eps         [[buffer(6)]],
    constant     uint&   w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization (bf16: scalar only) ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r]);
        }
    }
}

// ===========================================================================
// Interleaved BM8 variants: mat stored as [M/TM, K, TM] instead of [M, K]
// Weight addressing: mat[group * K * TM + k * TM + r]
// where group = row_base / TM, TM=4
// Each cache line (128B) holds 16 k-values x 4 rows = 64 halfs
// ===========================================================================

kernel void fused_rms_gemv_bm8_f32_interleaved(
    device const float* input       [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* mat         [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = input[i];
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization (interleaved) ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* tile_base = mat + (row_base / TM) * K * TM;

    if (w_stride == 1) {
        uint k16 = as_uniform(K / 16);
        for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
            uint idx = i * 16;
            float4 in4a = *reinterpret_cast<device const float4*>(input + idx);
            float4 nw4a = *reinterpret_cast<device const float4*>(norm_weight + idx);
            float4 v4a = in4a * inv_rms * nw4a;
            float4 in4b = *reinterpret_cast<device const float4*>(input + idx + 4);
            float4 nw4b = *reinterpret_cast<device const float4*>(norm_weight + idx + 4);
            float4 v4b = in4b * inv_rms * nw4b;
            float4 in4c = *reinterpret_cast<device const float4*>(input + idx + 8);
            float4 nw4c = *reinterpret_cast<device const float4*>(norm_weight + idx + 8);
            float4 v4c = in4c * inv_rms * nw4c;
            float4 in4d = *reinterpret_cast<device const float4*>(input + idx + 12);
            float4 nw4d = *reinterpret_cast<device const float4*>(norm_weight + idx + 12);
            float4 v4d = in4d * inv_rms * nw4d;
            // Interleaved weight loads: tile_base[k * TM + r]
            device const float* chunk = tile_base + idx * TM;
            #pragma clang loop unroll(full)
            for (uint sub = 0; sub < 4; sub++) {
                device const float* p = chunk + sub * 4 * TM;
                float4 c0 = *reinterpret_cast<device const float4*>(p);
                float4 c1 = *reinterpret_cast<device const float4*>(p + TM);
                float4 c2 = *reinterpret_cast<device const float4*>(p + 2 * TM);
                float4 c3 = *reinterpret_cast<device const float4*>(p + 3 * TM);
                float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
                acc[0] += c0[0] * v4[0] + c1[0] * v4[1] + c2[0] * v4[2] + c3[0] * v4[3];
                acc[1] += c0[1] * v4[0] + c1[1] * v4[1] + c2[1] * v4[2] + c3[1] * v4[3];
                acc[2] += c0[2] * v4[0] + c1[2] * v4[1] + c2[2] * v4[2] + c3[2] * v4[3];
                acc[3] += c0[3] * v4[0] + c1[3] * v4[1] + c2[3] * v4[2] + c3[3] * v4[3];
            }
        }
        // Remainder in groups of 4
        for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
            float4 in4 = *reinterpret_cast<device const float4*>(input + i);
            float4 nw4 = *reinterpret_cast<device const float4*>(norm_weight + i);
            float4 v4 = in4 * inv_rms * nw4;
            device const float* p = tile_base + i * TM;
            float4 c0 = *reinterpret_cast<device const float4*>(p);
            float4 c1 = *reinterpret_cast<device const float4*>(p + TM);
            float4 c2 = *reinterpret_cast<device const float4*>(p + 2 * TM);
            float4 c3 = *reinterpret_cast<device const float4*>(p + 3 * TM);
            acc[0] += c0[0] * v4[0] + c1[0] * v4[1] + c2[0] * v4[2] + c3[0] * v4[3];
            acc[1] += c0[1] * v4[0] + c1[1] * v4[1] + c2[1] * v4[2] + c3[1] * v4[3];
            acc[2] += c0[2] * v4[0] + c1[2] * v4[1] + c2[2] * v4[2] + c3[2] * v4[3];
            acc[3] += c0[3] * v4[0] + c1[3] * v4[1] + c2[3] * v4[2] + c3[3] * v4[3];
        }
        // Scalar remainder
        for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = input[i] * inv_rms * norm_weight[i];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += tile_base[i * TM + r] * v;
            }
        }
    } else {
        for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = input[i] * inv_rms * norm_weight[i * w_stride];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += tile_base[i * TM + r] * v;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r];
        }
    }
}

kernel void fused_rms_gemv_bm8_f16_interleaved(
    device const half*  input       [[buffer(0)]],
    device const half*  norm_weight [[buffer(1)]],
    device const half*  mat         [[buffer(2)]],
    device       half*  output      [[buffer(3)]],
    constant     uint&  M           [[buffer(4)]],
    constant     uint&  K           [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     uint&  w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization (interleaved) ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const half* tile_base = mat + (row_base / TM) * K * TM;

    if (w_stride == 1) {
        uint k16 = as_uniform(K / 16);
        for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
            uint idx = i * 16;
            float4 in4a = float4(*reinterpret_cast<device const half4*>(input + idx));
            float4 nw4a = float4(*reinterpret_cast<device const half4*>(norm_weight + idx));
            float4 v4a = in4a * inv_rms * nw4a;
            float4 in4b = float4(*reinterpret_cast<device const half4*>(input + idx + 4));
            float4 nw4b = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 4));
            float4 v4b = in4b * inv_rms * nw4b;
            float4 in4c = float4(*reinterpret_cast<device const half4*>(input + idx + 8));
            float4 nw4c = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 8));
            float4 v4c = in4c * inv_rms * nw4c;
            float4 in4d = float4(*reinterpret_cast<device const half4*>(input + idx + 12));
            float4 nw4d = float4(*reinterpret_cast<device const half4*>(norm_weight + idx + 12));
            float4 v4d = in4d * inv_rms * nw4d;
            // Interleaved weight loads: tile_base[k * TM + r]
            device const half* chunk = tile_base + idx * TM;
            #pragma clang loop unroll(full)
            for (uint sub = 0; sub < 4; sub++) {
                device const half* p = chunk + sub * 4 * TM;
                half4 c0 = *reinterpret_cast<device const half4*>(p);
                half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
                half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
                half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
                float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
                acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
                acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
                acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
                acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
            }
        }
        // Remainder in groups of 4
        for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
            float4 in4 = float4(*reinterpret_cast<device const half4*>(input + i));
            float4 nw4 = float4(*reinterpret_cast<device const half4*>(norm_weight + i));
            float4 v4 = in4 * inv_rms * nw4;
            device const half* p = tile_base + i * TM;
            half4 c0 = *reinterpret_cast<device const half4*>(p);
            half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
            half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
            half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
            acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
            acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
            acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
            acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
        }
        // Scalar remainder
        for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(tile_base[i * TM + r]) * v;
            }
        }
    } else {
        for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
            float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; r++) {
                acc[r] += float(tile_base[i * TM + r]) * v;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r]);
        }
    }
}

kernel void fused_rms_gemv_bm8_bf16_interleaved(
    device const bfloat* input       [[buffer(0)]],
    device const bfloat* norm_weight [[buffer(1)]],
    device const bfloat* mat         [[buffer(2)]],
    device       bfloat* output      [[buffer(3)]],
    constant     uint&   M           [[buffer(4)]],
    constant     uint&   K           [[buffer(5)]],
    constant     float&  eps         [[buffer(6)]],
    constant     uint&   w_stride    [[buffer(7)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[BM8 * SIMD_SIZE];
    uint tid = simd_group_id * SIMD_SIZE + simd_lane_id;

    // --- Phase 1: compute inv_rms cooperatively (all 256 threads) ---
    float ss_acc = 0.0f;
    for (uint i = tid; i < K; i += BM8 * SIMD_SIZE) {
        float v = float(input[i]);
        ss_acc += v * v;
    }
    ss_acc = simd_sum(ss_acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = ss_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float total = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sums[0] = metal::precise::rsqrt(total / float(K) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = local_sums[0];

    // --- Phase 2: BM8 GEMV with inline normalization (bf16 interleaved: scalar only) ---
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const bfloat* tile_base = mat + (row_base / TM) * K * TM;

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(input[i]) * inv_rms * float(norm_weight[i * w_stride]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(tile_base[i * TM + r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r]);
        }
    }
}
"#;

/// Get the fused RMS-norm + GEMV kernel name for a given dtype and M dimension.
///
/// Returns the BM8 variant for M >= 256, regular variant otherwise.
pub fn fused_rms_gemv_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    let use_bm8 = (m as u64) >= FUSED_BM8_THRESHOLD;
    match (dtype, use_bm8) {
        (DType::Float32, true) => Ok("fused_rms_gemv_bm8_f32"),
        (DType::Float32, false) => Ok("fused_rms_gemv_f32"),
        (DType::Float16, true) => Ok("fused_rms_gemv_bm8_f16"),
        (DType::Float16, false) => Ok("fused_rms_gemv_f16"),
        (DType::Bfloat16, true) => Ok("fused_rms_gemv_bm8_bf16"),
        (DType::Bfloat16, false) => Ok("fused_rms_gemv_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "fused_rms_gemv not supported for {:?}",
            dtype
        ))),
    }
}

/// Get the fused RMS-norm + GEMV interleaved kernel name for a given dtype.
///
/// Only BM8 variants are available (M >= 256 assumed by caller).
pub fn fused_rms_gemv_interleaved_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("fused_rms_gemv_bm8_f32_interleaved"),
        DType::Float16 => Ok("fused_rms_gemv_bm8_f16_interleaved"),
        DType::Bfloat16 => Ok("fused_rms_gemv_bm8_bf16_interleaved"),
        _ => Err(KernelError::NotFound(format!(
            "fused_rms_gemv_interleaved not supported for {:?}",
            dtype
        ))),
    }
}

/// Compute dispatch grid and threadgroup sizes for fused RMS-norm + GEMV.
pub fn fused_rms_gemv_dispatch_sizes(
    m: u32,
    pso: &metal::ComputePipelineState,
) -> (metal::MTLSize, metal::MTLSize) {
    let use_bm8 = (m as u64) >= FUSED_BM8_THRESHOLD;
    let num_threadgroups = if use_bm8 {
        fused_ceil_div(m as u64, FUSED_BM8_ROWS)
    } else {
        fused_ceil_div(m as u64, FUSED_TM)
    };
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, FUSED_BM8)
    } else {
        let tg_size = std::cmp::min(
            FUSED_RMS_GEMV_TG_SIZE,
            pso.max_total_threads_per_threadgroup(),
        );
        MTLSize::new(tg_size, 1, 1)
    };
    (MTLSize::new(num_threadgroups, 1, 1), tg_dim)
}

/// Encode fused RMS-norm + GEMV using a pre-resolved PSO and pre-allocated buffers.
///
/// Buffer layout:
/// - 0: input [K]         (raw pre-norm input vector)
/// - 1: norm_weight [K]   (RMS norm weight vector)
/// - 2: mat [M, K]        (projection weight matrix, row-major)
/// - 3: output [M]
/// - 4: M (u32)
/// - 5: K (u32)
/// - 6: eps (f32)
/// - 7: w_stride (u32)
#[allow(clippy::too_many_arguments)]
pub fn fused_rms_gemv_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    input_buf: &metal::BufferRef,
    input_offset: u64,
    norm_w_buf: &metal::BufferRef,
    norm_w_offset: u64,
    mat_buf: &metal::BufferRef,
    mat_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    m: u32,
    k: u32,
    eps: f32,
    w_stride: u32,
    grid: metal::MTLSize,
    tg: metal::MTLSize,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(input_buf), input_offset);
    encoder.set_buffer(1, Some(norm_w_buf), norm_w_offset);
    encoder.set_buffer(2, Some(mat_buf), mat_offset);
    encoder.set_buffer(3, Some(out_buf), out_offset);
    encoder.set_bytes(4, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
    encoder.set_bytes(7, 4, &w_stride as *const u32 as *const std::ffi::c_void);
    encoder.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_silu_mul_shape_validation() {
        let device = metal::Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(
            rmlx_metal::device::GpuDevice::system_default().expect("GPU device"),
        );
        super::super::silu::register(&registry).ok();

        let a = Array::from_slice(&device, &[1.0f32, 2.0, 3.0], vec![3]);
        let b = Array::from_slice(&device, &[1.0f32, 2.0], vec![2]);

        let result = fused_silu_mul(&registry, &a, &b, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_kernel_name() {
        assert_eq!(gemm_kernel_name(DType::Float32).unwrap(), "gemm_tiled_f32");
        assert_eq!(gemm_kernel_name(DType::Float16).unwrap(), "gemm_tiled_f16");
        assert!(gemm_kernel_name(DType::Q4_0).is_err());
    }
}
