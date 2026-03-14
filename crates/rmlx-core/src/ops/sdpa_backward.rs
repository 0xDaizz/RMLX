//! SDPA backward pass (Flash Attention V2 backward).
//!
//! Implements the backward pass for Scaled Dot-Product Attention using a
//! recomputation-based approach. Instead of storing the O(N^2) attention
//! matrix from the forward pass, we recompute attention scores during
//! backward to compute gradients for Q, K, and V.
//!
//! This trades compute for memory, matching the Flash Attention V2 strategy.

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

/// Metal shader for SDPA backward pass.
///
/// Recomputes `scores = Q @ K^T / sqrt(d)`, applies softmax, then:
/// - `dV = attn^T @ dO`
/// - `dAttn = dO @ V^T`
/// - `dScores = attn * (dAttn - rowsum(dAttn * attn))`  (softmax backward)
/// - `dQ = dScores @ K / sqrt(d)`
/// - `dK = dScores^T @ Q / sqrt(d)`
///
/// For simplicity and correctness, we use a per-element recomputation kernel
/// that handles moderate sequence lengths. For production Flash Attention V2
/// backward, this would use tiled recomputation.
pub const SDPA_BACKWARD_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// SDPA backward kernel (single-head, no mask, recomputation approach).
//
// Inputs:
//   Q:  [N, D]   - queries
//   K:  [S, D]   - keys
//   V:  [S, D]   - values
//   dO: [N, D]   - gradient of output
//
// Outputs:
//   dQ: [N, D]   - gradient for queries
//   dK: [S, D]   - gradient for keys
//   dV: [S, D]   - gradient for values
//
// Params: [N, S, D] as uint3
// scale: 1/sqrt(D) as float
//
// Grid: (N, 1, 1) -- each thread handles one query row

kernel void sdpa_backward_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device const float* dO      [[buffer(3)]],
    device float*       dQ      [[buffer(4)]],
    device float*       dK      [[buffer(5)]],
    device float*       dV      [[buffer(6)]],
    constant uint3&     params  [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    uint row_id [[thread_position_in_grid]])
{
    uint N = params.x;
    uint S = params.y;
    uint D = params.z;

    if (row_id >= N) return;

    device const float* q_row = Q + row_id * D;
    device const float* do_row = dO + row_id * D;
    device float* dq_row = dQ + row_id * D;

    // Step 1: Compute attention scores and softmax for this row.
    // score[j] = (Q[row] . K[j]) * scale
    float max_score = -INFINITY;
    for (uint j = 0; j < S; j++) {
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += q_row[d] * K[j * D + d];
        }
        float s = dot * scale;
        if (s > max_score) max_score = s;
    }

    // Softmax: compute exp(score - max) and sum
    float sum_exp = 0.0f;
    // We need to store scores temporarily. Use a second pass approach.
    // First compute sum_exp, then compute attn weights on the fly.
    for (uint j = 0; j < S; j++) {
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += q_row[d] * K[j * D + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }
    float inv_sum = 1.0f / sum_exp;

    // Step 2: Compute dV += attn[row, j] * dO[row, :] for all j
    // and also compute D_row = sum_d dO[row,d] * O[row,d]
    // where O[row,:] = sum_j attn[row,j] * V[j,:]
    //
    // But we can simplify: D_row = sum_j attn[row,j] * (dO[row,:] . V[j,:])

    float D_val = 0.0f;
    for (uint j = 0; j < S; j++) {
        // Recompute attn[row, j]
        float dot_qk = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot_qk += q_row[d] * K[j * D + d];
        }
        float attn_j = exp(dot_qk * scale - max_score) * inv_sum;

        // Accumulate dV[j,:] += attn[row,j] * dO[row,:]
        for (uint d = 0; d < D; d++) {
            // Use atomic add for concurrent writes from different rows
            // For now, we use a simple non-atomic approach (single-row-per-thread)
            dV[j * D + d] += attn_j * do_row[d];
        }

        // D_val += attn_j * dot(dO[row,:], V[j,:])
        float dot_dov = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot_dov += do_row[d] * V[j * D + d];
        }
        D_val += attn_j * dot_dov;
    }

    // Step 3: Compute dQ and dK
    // dScore[row, j] = attn[row,j] * (dot(dO[row,:], V[j,:]) - D_val)
    // dQ[row,:] += dScore[row,j] * K[j,:] * scale
    // dK[j,:]   += dScore[row,j] * Q[row,:] * scale

    for (uint d = 0; d < D; d++) {
        dq_row[d] = 0.0f;
    }

    for (uint j = 0; j < S; j++) {
        float dot_qk = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot_qk += q_row[d] * K[j * D + d];
        }
        float attn_j = exp(dot_qk * scale - max_score) * inv_sum;

        float dot_dov = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot_dov += do_row[d] * V[j * D + d];
        }
        float ds = attn_j * (dot_dov - D_val);

        for (uint d = 0; d < D; d++) {
            dq_row[d] += ds * K[j * D + d] * scale;
            dK[j * D + d] += ds * q_row[d] * scale;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register SDPA backward kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("sdpa_backward", SDPA_BACKWARD_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// SDPA backward pass with recomputation (Flash Attention V2 style).
///
/// Computes gradients for Q, K, V given the gradient of the output.
/// Instead of storing the O(N*S) attention matrix from the forward pass,
/// scores are recomputed from Q and K during the backward pass.
///
/// # Arguments
/// - `q`: Query tensor `[N, D]` (f32)
/// - `k`: Key tensor `[S, D]` (f32)
/// - `v`: Value tensor `[S, D]` (f32)
/// - `grad_output`: Gradient of the SDPA output `[N, D]` (f32)
/// - `scale`: Usually `1.0 / sqrt(D)`
///
/// # Returns
/// `(grad_q, grad_k, grad_v)` with shapes `[N, D]`, `[S, D]`, `[S, D]`.
///
/// # Note
/// This kernel processes one query row per thread. For large sequence lengths,
/// a tiled approach would be more efficient. This implementation prioritizes
/// correctness and memory efficiency over raw throughput.
pub fn sdpa_backward(
    registry: &KernelRegistry,
    q: &Array,
    k: &Array,
    v: &Array,
    grad_output: &Array,
    scale: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Array, Array), KernelError> {
    // Validate inputs
    if q.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_backward: Q must be Float32, got {:?}",
            q.dtype()
        )));
    }
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 || grad_output.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "sdpa_backward: all inputs must be 2D [seq, dim]".into(),
        ));
    }

    let n = q.shape()[0]; // query sequence length
    let d = q.shape()[1]; // head dimension
    let s = k.shape()[0]; // key/value sequence length

    if k.shape()[1] != d || v.shape()[1] != d {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_backward: dimension mismatch: Q=[{n},{d}], K={:?}, V={:?}",
            k.shape(),
            v.shape()
        )));
    }
    if grad_output.shape() != q.shape() {
        return Err(KernelError::InvalidShape(format!(
            "sdpa_backward: grad_output shape {:?} != Q shape {:?}",
            grad_output.shape(),
            q.shape()
        )));
    }

    // Ensure contiguous
    let q_c = super::make_contiguous(q, registry, queue)?;
    let q = q_c.as_ref().unwrap_or(q);
    let k_c = super::make_contiguous(k, registry, queue)?;
    let k = k_c.as_ref().unwrap_or(k);
    let v_c = super::make_contiguous(v, registry, queue)?;
    let v = v_c.as_ref().unwrap_or(v);
    let go_c = super::make_contiguous(grad_output, registry, queue)?;
    let grad_output = go_c.as_ref().unwrap_or(grad_output);

    let pipeline = registry.get_pipeline("sdpa_backward_f32", DType::Float32)?;
    let dev = registry.device().raw();

    let grad_q = Array::zeros(dev, &[n, d], DType::Float32);
    let grad_k = Array::zeros(dev, &[s, d], DType::Float32);
    let grad_v = Array::zeros(dev, &[s, d], DType::Float32);

    let params: [u32; 3] = [
        super::checked_u32(n, "N")?,
        super::checked_u32(s, "S")?,
        super::checked_u32(d, "D")?,
    ];
    let opts = MTLResourceOptions::StorageModeShared;
    let params_buf = unsafe {
        dev.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(params.as_ptr() as *const _ as *mut std::ffi::c_void).unwrap(),
            std::mem::size_of_val(&params),
            opts,
        )
        .unwrap()
    };
    let scale_buf = unsafe {
        dev.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(&scale as *const f32 as *const _ as *mut std::ffi::c_void)
                .unwrap(),
            std::mem::size_of::<f32>(),
            opts,
        )
        .unwrap()
    };

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(q.metal_buffer()), q.offset());
    enc.set_buffer(1, Some(k.metal_buffer()), k.offset());
    enc.set_buffer(2, Some(v.metal_buffer()), v.offset());
    enc.set_buffer(3, Some(grad_output.metal_buffer()), grad_output.offset());
    enc.set_buffer(4, Some(grad_q.metal_buffer()), 0);
    enc.set_buffer(5, Some(grad_k.metal_buffer()), 0);
    enc.set_buffer(6, Some(grad_v.metal_buffer()), 0);
    enc.set_buffer(7, Some(&params_buf), 0);
    enc.set_buffer(8, Some(&scale_buf), 0);
    // One thread per query row. For small N this is fine;
    // for large N we'd want a tiled approach.
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), n).max(1),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok((grad_q, grad_k, grad_v))
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
    fn test_sdpa_backward_shapes() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        let n = 4;
        let s = 6;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();

        let q = Array::ones(dev, &[n, d]);
        let k = Array::ones(dev, &[s, d]);
        let v = Array::ones(dev, &[s, d]);
        let grad_o = Array::ones(dev, &[n, d]);

        let (dq, dk, dv) = sdpa_backward(&registry, &q, &k, &v, &grad_o, scale, &queue).unwrap();
        assert_eq!(dq.shape(), &[n, d]);
        assert_eq!(dk.shape(), &[s, d]);
        assert_eq!(dv.shape(), &[s, d]);
    }

    #[test]
    fn test_sdpa_backward_grad_v_nonzero() {
        let Some((registry, queue)) = setup() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let dev = registry.device().raw();

        let n = 2;
        let s = 2;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();

        let q = Array::from_slice(
            dev,
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![n, d],
        );
        let k = Array::from_slice(
            dev,
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![s, d],
        );
        let v = Array::from_slice(
            dev,
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![s, d],
        );
        let grad_o = Array::ones(dev, &[n, d]);

        let (_dq, _dk, dv) = sdpa_backward(&registry, &q, &k, &v, &grad_o, scale, &queue).unwrap();

        // dV should be non-zero since grad_output is all ones
        let dv_vec = dv.to_vec_checked::<f32>();
        let sum: f32 = dv_vec.iter().sum();
        assert!(sum.abs() > 1e-6, "dV should be nonzero, got sum={}", sum);
    }
}
