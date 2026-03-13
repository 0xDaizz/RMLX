//! Grouped expert forward pass using weight stacking and batched GEMM.
//!
//! Replaces the per-expert loop in `moe.rs` with a single grouped forward
//! pass that encodes all expert projections into one Metal command buffer.
//!
//! # Design
//!
//! Instead of O(E) individual command buffer commits (one per expert), this
//! module:
//! 1. Stacks per-expert weights into 3D tensors `[E, *, *]`
//! 2. Encodes all expert GEMMs (gate, up, down) + fused SwiGLU into a
//!    **single command buffer**
//! 3. Commits once at the end
//!
//! This dramatically reduces Metal command buffer overhead, which is the
//! dominant cost for small-batch MoE inference on Apple Silicon.
//!
//! # Limitations
//!
//! - Currently f32 only for the weight stacking path. f16/bf16 experts
//!   should fall back to the per-expert loop in `moe.rs`.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use rmlx_core::ops::buffer_slots::{gemm as gslot, silu_gate as sslot};

use crate::moe::Expert;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLResourceOptions};
use rmlx_metal::{ComputePass, MTLSize, MtlBuffer};

/// Manages stacked expert weights and provides grouped forward pass.
///
/// All expert weight matrices are pre-stacked into contiguous 3D tensors
/// so that the forward pass can encode all expert GEMMs into a single
/// Metal command buffer.
pub struct ExpertGroup {
    /// Stacked gate projection weights: `[E, D, intermediate_dim]`
    ///
    /// Stored transposed (relative to Linear's `[out, in]` convention)
    /// so the GEMM can be `input @ gate_weights` directly.
    pub gate_weights: Array,
    /// Stacked up projection weights: `[E, D, intermediate_dim]`
    pub up_weights: Array,
    /// Stacked down projection weights: `[E, intermediate_dim, D]`
    pub down_weights: Array,
    /// Number of experts.
    pub num_experts: usize,
    /// Hidden dimension (model dimension D).
    pub hidden_dim: usize,
    /// Intermediate dimension (FFN inner dim).
    pub intermediate_dim: usize,
}

impl ExpertGroup {
    /// Stack individual expert weights into 3D tensors for batched GEMM.
    ///
    /// Each expert has `gate_proj`, `up_proj`, `down_proj` (`Linear` layers).
    /// `Linear.weight()` has shape `[out_dim, in_dim]` (standard convention).
    ///
    /// This creates:
    /// - `gate_weights`: `[E, D, intermediate_dim]` -- transposed from `[E, inter, D]`
    /// - `up_weights`:   `[E, D, intermediate_dim]`
    /// - `down_weights`: `[E, intermediate_dim, D]` -- transposed from `[E, D, inter]`
    ///
    /// The transposition is done via GPU copy of transposed views so the
    /// resulting tensors are contiguous and ready for GEMM.
    pub fn from_experts(
        experts: &[Expert],
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Self, KernelError> {
        if experts.is_empty() {
            return Err(KernelError::InvalidShape(
                "ExpertGroup: experts slice is empty".into(),
            ));
        }

        let num_experts = experts.len();

        // Infer dimensions from the first expert's gate_proj weight.
        // gate_proj weight shape: [intermediate_dim, hidden_dim]
        let first_gate_w = experts[0].gate_proj.weight().ok_or_else(|| {
            KernelError::InvalidShape("Expert 0: gate_proj has no weights".into())
        })?;
        let intermediate_dim = first_gate_w.shape()[0];
        let hidden_dim = first_gate_w.shape()[1];
        let dtype = first_gate_w.dtype();

        // Validate all experts have consistent shapes
        for (i, expert) in experts.iter().enumerate() {
            let gw = expert.gate_proj.weight().ok_or_else(|| {
                KernelError::InvalidShape(format!("Expert {i}: gate_proj has no weights"))
            })?;
            let uw = expert.up_proj.weight().ok_or_else(|| {
                KernelError::InvalidShape(format!("Expert {i}: up_proj has no weights"))
            })?;
            let dw = expert.down_proj.weight().ok_or_else(|| {
                KernelError::InvalidShape(format!("Expert {i}: down_proj has no weights"))
            })?;

            if gw.shape() != [intermediate_dim, hidden_dim] {
                return Err(KernelError::InvalidShape(format!(
                    "Expert {i}: gate_proj weight shape {:?} != expected [{intermediate_dim}, {hidden_dim}]",
                    gw.shape()
                )));
            }
            if uw.shape() != [intermediate_dim, hidden_dim] {
                return Err(KernelError::InvalidShape(format!(
                    "Expert {i}: up_proj weight shape {:?} != expected [{intermediate_dim}, {hidden_dim}]",
                    uw.shape()
                )));
            }
            if dw.shape() != [hidden_dim, intermediate_dim] {
                return Err(KernelError::InvalidShape(format!(
                    "Expert {i}: down_proj weight shape {:?} != expected [{hidden_dim}, {intermediate_dim}]",
                    dw.shape()
                )));
            }
        }

        let dev = registry.device().raw();
        let elem_size = dtype.size_of();

        // Allocate stacked weight buffers
        // gate_weights: [E, D, intermediate_dim] (transposed from [inter, D])
        // up_weights:   [E, D, intermediate_dim]
        // down_weights: [E, intermediate_dim, D] (transposed from [D, inter])
        let gate_stacked = Array::zeros(dev, &[num_experts, hidden_dim, intermediate_dim], dtype);
        let up_stacked = Array::zeros(dev, &[num_experts, hidden_dim, intermediate_dim], dtype);
        let down_stacked = Array::zeros(dev, &[num_experts, intermediate_dim, hidden_dim], dtype);

        // Stack weights using GPU copies. For each expert:
        // - gate_proj.weight() is [inter, D], we need [D, inter] -> transpose view then copy
        // - up_proj.weight() is [inter, D], we need [D, inter] -> transpose view then copy
        // - down_proj.weight() is [D, inter], we need [inter, D] -> transpose view then copy
        let copy_kernel = match dtype {
            DType::Float32 => "copy_f32",
            DType::Float16 => "copy_f16",
            DType::Bfloat16 => "copy_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "ExpertGroup: unsupported dtype {:?}",
                    other
                )));
            }
        };
        let pipeline = registry.get_pipeline(copy_kernel, dtype)?;

        // Use GPU copy to transpose and stack each expert's weights.
        // We create transposed views (zero-copy stride swap) and then
        // copy them into the contiguous stacked buffer.
        let gate_slice_size = hidden_dim * intermediate_dim;
        let up_slice_size = hidden_dim * intermediate_dim;
        let down_slice_size = intermediate_dim * hidden_dim;

        let cb = queue.commandBuffer().unwrap();

        for (e, expert) in experts.iter().enumerate() {
            let gate_w = expert.gate_proj.weight().unwrap();
            let up_w = expert.up_proj.weight().unwrap();
            let down_w = expert.down_proj.weight().unwrap();

            // gate_w: [inter, D] -> transposed view [D, inter]
            let gate_t = gate_w.view(
                vec![hidden_dim, intermediate_dim],
                vec![1, hidden_dim],
                gate_w.offset(),
            );
            // up_w: [inter, D] -> transposed view [D, inter]
            let up_t = up_w.view(
                vec![hidden_dim, intermediate_dim],
                vec![1, hidden_dim],
                up_w.offset(),
            );
            // down_w: [D, inter] -> transposed view [inter, D]
            let down_t = down_w.view(
                vec![intermediate_dim, hidden_dim],
                vec![1, intermediate_dim],
                down_w.offset(),
            );

            // Copy transposed views into stacked buffers element by element.
            // Since the views are non-contiguous (transposed), we use the
            // strided copy path: copy each element via row-by-row dispatch.
            let gate_dst_offset = e * gate_slice_size * elem_size;
            let up_dst_offset = e * up_slice_size * elem_size;
            let down_dst_offset = e * down_slice_size * elem_size;

            // For transposed copies, we iterate row-by-row of the transposed view
            // and copy each row (which is contiguous in the source) to the destination.
            // gate_t[row] = gate_w column `row`, elements at offsets row, row+D, row+2D, ...
            // These are NOT contiguous, so we copy element by element per row.
            //
            // Actually, since the transposed view's rows are non-contiguous in memory,
            // we use ops::copy to make a contiguous copy, then blit into the stacked buffer.

            // Make contiguous copies of transposed views
            let gate_t_contig = ops::copy::copy(registry, &gate_t, queue)?;
            let up_t_contig = ops::copy::copy(registry, &up_t, queue)?;
            let down_t_contig = ops::copy::copy(registry, &down_t, queue)?;

            // Blit contiguous transposed data into the stacked buffer slices
            let numel_gate = gate_slice_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(gate_t_contig.metal_buffer()), gate_t_contig.offset());
            enc.set_buffer(1, Some(gate_stacked.metal_buffer()), gate_dst_offset);
            let grid = MTLSize { width: numel_gate, height: 1, depth: 1 };
            let tg = MTLSize { width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel_gate), height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end();

            let numel_up = up_slice_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(up_t_contig.metal_buffer()), up_t_contig.offset());
            enc.set_buffer(1, Some(up_stacked.metal_buffer()), up_dst_offset);
            let grid = MTLSize { width: numel_up, height: 1, depth: 1 };
            let tg = MTLSize { width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel_up), height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end();

            let numel_down = down_slice_size;
            let raw_enc = cb.computeCommandEncoder().unwrap();
            let enc = ComputePass::new(&raw_enc);
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, Some(down_t_contig.metal_buffer()), down_t_contig.offset());
            enc.set_buffer(1, Some(down_stacked.metal_buffer()), down_dst_offset);
            let grid = MTLSize { width: numel_down, height: 1, depth: 1 };
            let tg = MTLSize { width: std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), numel_down), height: 1, depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end();
        }

        cb.commit();
        cb.waitUntilCompleted();

        Ok(Self {
            gate_weights: gate_stacked,
            up_weights: up_stacked,
            down_weights: down_stacked,
            num_experts,
            hidden_dim,
            intermediate_dim,
        })
    }

    /// Grouped expert forward pass: all experts in a **single command buffer**.
    ///
    /// Instead of O(E) individual expert forward passes (each with its own CB),
    /// this encodes all work into one CB:
    ///
    /// For each active expert with tokens:
    /// 1. GEMM: `input[batch, D] @ gate_weights[D, inter]` -> `gate_out`
    /// 2. GEMM: `input[batch, D] @ up_weights[D, inter]`   -> `up_out`
    /// 3. Fused SwiGLU: `silu(gate_out) * up_out`           -> `hidden`
    /// 4. GEMM: `hidden[batch, inter] @ down_weights[inter, D]` -> `output`
    ///
    /// All encoded into the same command buffer, committed once.
    ///
    /// # Arguments
    ///
    /// * `expert_inputs` - Vec of `(expert_idx, input_array)` where each
    ///   input_array has shape `[batch_size, D]`. Batch size may vary per expert.
    ///
    /// # Returns
    ///
    /// Vec of `(expert_idx, output_array)` with shape `[batch_size, D]` per expert.
    pub fn grouped_forward(
        &self,
        expert_inputs: &[(usize, &Array)],
        registry: &KernelRegistry,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<(usize, Array)>, KernelError> {
        if expert_inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Validate inputs
        for &(expert_idx, input) in expert_inputs {
            if expert_idx >= self.num_experts {
                return Err(KernelError::InvalidShape(format!(
                    "ExpertGroup: expert_idx {} >= num_experts {}",
                    expert_idx, self.num_experts
                )));
            }
            if input.ndim() != 2 {
                return Err(KernelError::InvalidShape(format!(
                    "ExpertGroup: input must be 2D [batch, D], got {}D",
                    input.ndim()
                )));
            }
            if input.shape()[1] != self.hidden_dim {
                return Err(KernelError::InvalidShape(format!(
                    "ExpertGroup: input dim {} != hidden_dim {}",
                    input.shape()[1],
                    self.hidden_dim
                )));
            }
        }

        let dev = registry.device().raw();
        let dtype = expert_inputs[0].1.dtype();
        let elem_size = dtype.size_of();

        // SwiGLU kernel
        let silu_kernel = match dtype {
            DType::Float32 => "silu_gate_f32",
            DType::Float16 => "silu_gate_f16",
            DType::Bfloat16 => "silu_gate_bf16",
            other => {
                return Err(KernelError::InvalidShape(format!(
                    "ExpertGroup: unsupported dtype {:?} for SiLU",
                    other
                )));
            }
        };
        let silu_pipeline = registry.get_pipeline(silu_kernel, dtype)?;

        // Pre-allocate all output buffers and make inputs contiguous BEFORE
        // encoding into the single command buffer. The ops::copy::copy calls
        // use their own CBs, so we do this outside the main CB.
        struct ExpertWork {
            expert_idx: usize,
            batch_size: usize,
            input: Array, // contiguous input
            gate_out: Array,
            up_out: Array,
            silu_out: Array,
            output: Array,
        }

        let mut work_items: Vec<ExpertWork> = Vec::with_capacity(expert_inputs.len());

        for &(expert_idx, input) in expert_inputs {
            let batch_size = input.shape()[0];

            // Ensure input is contiguous for GEMM
            let input_contig = if input.is_contiguous() {
                input.view(
                    input.shape().to_vec(),
                    input.strides().to_vec(),
                    input.offset(),
                )
            } else {
                ops::copy::copy(registry, input, queue)?
            };

            let gate_out = Array::zeros(dev, &[batch_size, self.intermediate_dim], dtype);
            let up_out = Array::zeros(dev, &[batch_size, self.intermediate_dim], dtype);
            let silu_out = Array::zeros(dev, &[batch_size, self.intermediate_dim], dtype);
            let output = Array::zeros(dev, &[batch_size, self.hidden_dim], dtype);

            work_items.push(ExpertWork {
                expert_idx,
                batch_size,
                input: input_contig,
                gate_out,
                up_out,
                silu_out,
                output,
            });
        }

        // ── Single command buffer for ALL experts ──
        let cb = queue.commandBuffer().unwrap();

        for item in &work_items {
            let m = item.batch_size as u32;
            let d = self.hidden_dim as u32;
            let inter = self.intermediate_dim as u32;
            let e = item.expert_idx;

            // Extract weight slices for this expert via views into the stacked arrays
            let gate_w_offset = e * self.hidden_dim * self.intermediate_dim * elem_size;
            let gate_w = self.gate_weights.view(
                vec![self.hidden_dim, self.intermediate_dim],
                vec![self.intermediate_dim, 1],
                gate_w_offset,
            );

            let up_w_offset = e * self.hidden_dim * self.intermediate_dim * elem_size;
            let up_w = self.up_weights.view(
                vec![self.hidden_dim, self.intermediate_dim],
                vec![self.intermediate_dim, 1],
                up_w_offset,
            );

            let down_w_offset = e * self.intermediate_dim * self.hidden_dim * elem_size;
            let down_w = self.down_weights.view(
                vec![self.intermediate_dim, self.hidden_dim],
                vec![self.hidden_dim, 1],
                down_w_offset,
            );

            // 1. Gate GEMM: [batch, D] @ [D, inter] -> [batch, inter]
            encode_gemm(
&cb,
                &item.input,
                &gate_w,
                &item.gate_out,
                m,
                inter,
                d,
                dtype,
                registry,
            )?;

            // 2. Up GEMM: [batch, D] @ [D, inter] -> [batch, inter]
            encode_gemm(
&cb,
                &item.input,
                &up_w,
                &item.up_out,
                m,
                inter,
                d,
                dtype,
                registry,
            )?;

            // 3. Fused SwiGLU: silu(gate_out) * up_out -> silu_out
            encode_silu_gate(
&cb,
                &silu_pipeline,
                &item.gate_out,
                &item.up_out,
                &item.silu_out,
                registry,
            )?;

            // 4. Down GEMM: [batch, inter] @ [inter, D] -> [batch, D]
            encode_gemm(
&cb,
                &item.silu_out,
                &down_w,
                &item.output,
                m,
                d,
                inter,
                dtype,
                registry,
            )?;
        }

        cb.commit();
        cb.waitUntilCompleted();

        // Collect results
        let results: Vec<(usize, Array)> = work_items
            .into_iter()
            .map(|item| (item.expert_idx, item.output))
            .collect();

        Ok(results)
    }

    /// Number of experts in this group.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Hidden dimension (model dimension D).
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Intermediate dimension (FFN inner dim).
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Encode a single GEMM dispatch into a command buffer (no commit/wait).
///
/// `C[M, N] = A[M, K] @ B[K, N]`
///
/// Selects the best tile configuration and kernel variant based on matrix
/// dimensions via `ops::matmul::select_tile_config`.
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    a: &Array,
    b: &Array,
    c: &Array,
    m: u32,
    n: u32,
    k: u32,
    dtype: DType,
    registry: &KernelRegistry,
) -> Result<(), KernelError> {
    let tile =
        ops::matmul::select_tile_config_with_dtype(m as usize, n as usize, k as usize, dtype);
    let kernel_name = match (tile.variant, dtype) {
        (ops::matmul::TileVariant::Simd, DType::Float32)
        | (ops::matmul::TileVariant::Medium, DType::Float32) => "gemm_simd_f32",
        (ops::matmul::TileVariant::Simd, DType::Float16)
        | (ops::matmul::TileVariant::Medium, DType::Float16) => "gemm_simd_f16",
        (ops::matmul::TileVariant::Simd, DType::Bfloat16)
        | (ops::matmul::TileVariant::Medium, DType::Bfloat16) => "gemm_simd_bf16",
        (ops::matmul::TileVariant::Small, DType::Float32) => "gemm_small_f32",
        (ops::matmul::TileVariant::Small, DType::Float16) => "gemm_small_f16",
        (ops::matmul::TileVariant::Small, DType::Bfloat16) => "gemm_small_bf16",
        (ops::matmul::TileVariant::Skinny, DType::Float32) => "gemm_skinny_f32",
        (ops::matmul::TileVariant::Skinny, DType::Float16) => "gemm_skinny_f16",
        (ops::matmul::TileVariant::Skinny, DType::Bfloat16) => "gemm_skinny_bf16",
        (ops::matmul::TileVariant::Full, DType::Float32) => "gemm_tiled_f32",
        (ops::matmul::TileVariant::Full, DType::Float16) => "gemm_tiled_f16",
        (ops::matmul::TileVariant::Full, DType::Bfloat16) => "gemm_tiled_bf16",
        (ops::matmul::TileVariant::MlxArch, DType::Float16) => "gemm_mlx_f16",
        (ops::matmul::TileVariant::MlxArch, DType::Float32) => "gemm_mlx_f32",
        (ops::matmul::TileVariant::MlxArchSmall, DType::Float16) => "gemm_mlx_small_f16",
        (ops::matmul::TileVariant::MlxArchMicro, DType::Float16) => "gemm_mlx_m16_f16",
        (ops::matmul::TileVariant::NaxArch, DType::Float16) => "gemm_nax_f16",
        (ops::matmul::TileVariant::NaxArch64x128, DType::Float16) => "gemm_nax_64x128_f16",
        (_, other) => {
            return Err(KernelError::InvalidShape(format!(
                "ExpertGroup: unsupported dtype {:?} for GEMM",
                other
            )));
        }
    };

    // MlxArch/MlxArchSmall/NaxArch use function constants for alignment specialization
    let pipeline = if tile.variant == ops::matmul::TileVariant::MlxArch
        || tile.variant == ops::matmul::TileVariant::MlxArchSmall
        || tile.variant == ops::matmul::TileVariant::MlxArchMicro
        || tile.variant == ops::matmul::TileVariant::NaxArch
        || tile.variant == ops::matmul::TileVariant::NaxArch64x128
    {
        let constants = ops::matmul::matmul_align_constants(
            m as usize, n as usize, k as usize, tile.bm, tile.bn, tile.bk,
        );
        registry.get_pipeline_with_constants(kernel_name, dtype, &constants)?
    } else {
        registry.get_pipeline(kernel_name, dtype)?
    };

    let dev = registry.device().raw();
    let m_buf = make_u32_buf(dev, m);
    let n_buf = make_u32_buf(dev, n);
    let k_buf = make_u32_buf(dev, k);
    let batch_stride_a = m * k;
    let batch_stride_b = k * n;
    let batch_stride_c = m * n;
    let bsa_buf = make_u32_buf(dev, batch_stride_a);
    let bsb_buf = make_u32_buf(dev, batch_stride_b);
    let bsc_buf = make_u32_buf(dev, batch_stride_c);

    let bm = tile.bm as usize;
    let bn = tile.bn as usize;

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(gslot::A as u32, Some(a.metal_buffer()), a.offset());
    enc.set_buffer(gslot::B as u32, Some(b.metal_buffer()), b.offset());
    enc.set_buffer(gslot::OUT as u32, Some(c.metal_buffer()), c.offset());
    enc.set_buffer(gslot::M as u32, Some(&m_buf), 0);
    enc.set_buffer(gslot::N as u32, Some(&n_buf), 0);
    enc.set_buffer(gslot::K as u32, Some(&k_buf), 0);
    enc.set_buffer(gslot::BATCH_STRIDE_A as u32, Some(&bsa_buf), 0);
    enc.set_buffer(gslot::BATCH_STRIDE_B as u32, Some(&bsb_buf), 0);
    enc.set_buffer(gslot::BATCH_STRIDE_C as u32, Some(&bsc_buf), 0);

    // Full, Skinny, and MlxArch kernels require swizzle_log (buffer 9)
    let (swizzle_log, swizzle_log_buf) = if matches!(
        tile.variant,
        ops::matmul::TileVariant::Full
            | ops::matmul::TileVariant::Skinny
            | ops::matmul::TileVariant::MlxArch
            | ops::matmul::TileVariant::MlxArchSmall
            | ops::matmul::TileVariant::MlxArchMicro
            | ops::matmul::TileVariant::NaxArch
            | ops::matmul::TileVariant::NaxArch64x128
    ) {
        let s = ops::matmul::compute_swizzle_log(m as usize, n as usize, tile.bm, tile.bn);
        let buf = make_u32_buf(dev, s);
        enc.set_buffer(gslot::SWIZZLE_LOG as u32, Some(&buf), 0);
        (s, Some(buf))
    } else {
        (0, None)
    };

    let tg_threads = match tile.variant {
        ops::matmul::TileVariant::Small => 256_usize,
        ops::matmul::TileVariant::Medium | ops::matmul::TileVariant::Simd => 1024_usize,
        ops::matmul::TileVariant::Skinny | ops::matmul::TileVariant::Full => 256_usize,
        ops::matmul::TileVariant::MlxArch
        | ops::matmul::TileVariant::MlxArchSmall
        | ops::matmul::TileVariant::MlxArchMicro => 64_usize,
        ops::matmul::TileVariant::NaxArch => 512_usize,
        ops::matmul::TileVariant::NaxArch64x128 => 256_usize,
        ops::matmul::TileVariant::NaxArch64x64 => 128_usize,
    };

    let grid_x = ((n as usize).div_ceil(bn)) << swizzle_log;
    let grid_y = ((m as usize).div_ceil(bm)) >> swizzle_log;
    let grid = MTLSize { width: grid_x, height: grid_y, depth: 1 };
    let tg = MTLSize { width: tg_threads, height: 1, depth: 1 };
    enc.dispatch_threadgroups(grid, tg);
    enc.end();
    drop(swizzle_log_buf);

    Ok(())
}

/// Encode fused SwiGLU (silu(gate) * up -> output) into an existing CB.
fn encode_silu_gate(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    gate_out: &Array,
    up_out: &Array,
    output: &Array,
    registry: &KernelRegistry,
) -> Result<(), KernelError> {
    let dev = registry.device().raw();
    let numel = gate_out.numel();
    let numel_buf = make_u32_buf(dev, numel as u32);

    let elems_per_thread: usize = match gate_out.dtype() {
        DType::Float32 => 2,
        _ => 4,
    };
    let grid_threads = numel.div_ceil(elems_per_thread);

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(pipeline);
    enc.set_buffer(sslot::GATE_OUT as u32, Some(gate_out.metal_buffer()), gate_out.offset());
    enc.set_buffer(sslot::UP_OUT as u32, Some(up_out.metal_buffer()), up_out.offset());
    enc.set_buffer(sslot::OUT as u32, Some(output.metal_buffer()), output.offset());
    enc.set_buffer(sslot::NUMEL as u32, Some(&numel_buf), 0);

    let tg = std::cmp::min(pipeline.maxTotalThreadsPerThreadgroup(), grid_threads);
    enc.dispatch_threads(
        MTLSize { width: grid_threads, height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.end();

    Ok(())
}

/// Create a constant `u32` Metal buffer.
fn make_u32_buf(device: &ProtocolObject<dyn MTLDevice>, val: u32) -> MtlBuffer {
    unsafe { device.newBufferWithBytes_length_options(std::ptr::NonNull::new_unchecked(&val as *const u32 as *mut std::ffi::c_void), 4, MTLResourceOptions::StorageModeShared) }.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{Linear, LinearConfig};
    use crate::moe::Expert;

    /// Create a test `KernelRegistry` with all kernels registered.
    fn test_registry() -> (rmlx_metal::MtlDevice, rmlx_metal::MtlQueue, KernelRegistry) {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal device required");
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("GPU device required");
        let queue = gpu.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        rmlx_core::ops::register_all(&registry).expect("register kernels");
        (device, queue, registry)
    }

    /// Create a test expert with known weight values.
    fn make_expert(
        device: &ProtocolObject<dyn MTLDevice>,
        hidden_dim: usize,
        intermediate_dim: usize,
        val: f32,
    ) -> Expert {
        // gate_proj: [intermediate_dim, hidden_dim]
        let gate_data: Vec<f32> = vec![val; intermediate_dim * hidden_dim];
        let gate_w = Array::from_slice(device, &gate_data, vec![intermediate_dim, hidden_dim]);
        let gate_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            gate_w,
            None,
        )
        .unwrap();

        // up_proj: [intermediate_dim, hidden_dim]
        let up_data: Vec<f32> = vec![val; intermediate_dim * hidden_dim];
        let up_w = Array::from_slice(device, &up_data, vec![intermediate_dim, hidden_dim]);
        let up_proj = Linear::from_arrays(
            LinearConfig {
                in_features: hidden_dim,
                out_features: intermediate_dim,
                has_bias: false,
            },
            up_w,
            None,
        )
        .unwrap();

        // down_proj: [hidden_dim, intermediate_dim]
        let down_data: Vec<f32> = vec![val; hidden_dim * intermediate_dim];
        let down_w = Array::from_slice(device, &down_data, vec![hidden_dim, intermediate_dim]);
        let down_proj = Linear::from_arrays(
            LinearConfig {
                in_features: intermediate_dim,
                out_features: hidden_dim,
                has_bias: false,
            },
            down_w,
            None,
        )
        .unwrap();

        Expert {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    #[test]
    fn test_from_experts_basic() {
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;

        let experts = vec![
            make_expert(&device, hidden_dim, intermediate_dim, 0.1),
            make_expert(&device, hidden_dim, intermediate_dim, 0.2),
        ];

        let group = ExpertGroup::from_experts(&experts, &registry, &queue);
        assert!(group.is_ok(), "from_experts failed: {:?}", group.err());

        let group = group.unwrap();
        assert_eq!(group.num_experts(), 2);
        assert_eq!(group.hidden_dim(), hidden_dim);
        assert_eq!(group.intermediate_dim(), intermediate_dim);
        assert_eq!(
            group.gate_weights.shape(),
            &[2, hidden_dim, intermediate_dim]
        );
        assert_eq!(group.up_weights.shape(), &[2, hidden_dim, intermediate_dim]);
        assert_eq!(
            group.down_weights.shape(),
            &[2, intermediate_dim, hidden_dim]
        );
    }

    #[test]
    fn test_from_experts_empty() {
        let (_device, queue, registry) = test_registry();
        let experts: Vec<Expert> = vec![];
        let result = ExpertGroup::from_experts(&experts, &registry, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_grouped_forward_empty_inputs() {
        let (device, queue, registry) = test_registry();
        let experts = vec![
            make_expert(&device, 8, 4, 0.1),
            make_expert(&device, 8, 4, 0.2),
        ];
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let result = group.grouped_forward(&[], &registry, &queue);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_grouped_forward_single_expert() {
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;

        let experts = vec![
            make_expert(&device, hidden_dim, intermediate_dim, 0.1),
            make_expert(&device, hidden_dim, intermediate_dim, 0.2),
        ];
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        // Single token to expert 0
        let input = Array::from_slice(&device, &vec![1.0f32; hidden_dim], vec![1, hidden_dim]);

        let result = group.grouped_forward(&[(0, &input)], &registry, &queue);
        assert!(result.is_ok(), "grouped_forward failed: {:?}", result.err());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, 0); // expert_idx preserved
        assert_eq!(outputs[0].1.shape(), &[1, hidden_dim]);
    }

    #[test]
    fn test_grouped_forward_multiple_experts() {
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;

        let experts = vec![
            make_expert(&device, hidden_dim, intermediate_dim, 0.1),
            make_expert(&device, hidden_dim, intermediate_dim, 0.2),
            make_expert(&device, hidden_dim, intermediate_dim, 0.3),
        ];
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        // Different batch sizes per expert
        let input0 = Array::from_slice(&device, &vec![1.0f32; 2 * hidden_dim], vec![2, hidden_dim]);
        let input2 = Array::from_slice(&device, &vec![0.5f32; 3 * hidden_dim], vec![3, hidden_dim]);

        let expert_inputs: Vec<(usize, &Array)> = vec![(0, &input0), (2, &input2)];
        let result = group.grouped_forward(&expert_inputs, &registry, &queue);
        assert!(result.is_ok(), "grouped_forward failed: {:?}", result.err());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].0, 0);
        assert_eq!(outputs[0].1.shape(), &[2, hidden_dim]);
        assert_eq!(outputs[1].0, 2);
        assert_eq!(outputs[1].1.shape(), &[3, hidden_dim]);
    }

    #[test]
    fn test_grouped_forward_matches_individual() {
        let (device, queue, registry) = test_registry();
        let hidden_dim = 8;
        let intermediate_dim = 4;

        let experts = vec![
            make_expert(&device, hidden_dim, intermediate_dim, 0.1),
            make_expert(&device, hidden_dim, intermediate_dim, 0.2),
        ];

        // Run individual expert forward for reference
        let input = Array::from_slice(&device, &vec![1.0f32; hidden_dim], vec![1, hidden_dim]);
        let ref_out_0 = experts[0].forward(&input, &registry, &queue).unwrap();
        let ref_out_1 = experts[1].forward(&input, &registry, &queue).unwrap();
        let ref_vals_0: Vec<f32> = ref_out_0.to_vec_checked();
        let ref_vals_1: Vec<f32> = ref_out_1.to_vec_checked();

        // Run grouped forward
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();
        let inputs: Vec<(usize, &Array)> = vec![(0, &input), (1, &input)];
        let grouped_outputs = group.grouped_forward(&inputs, &registry, &queue).unwrap();

        let grouped_vals_0: Vec<f32> = grouped_outputs[0].1.to_vec_checked();
        let grouped_vals_1: Vec<f32> = grouped_outputs[1].1.to_vec_checked();

        // Compare results (allow small floating-point tolerance)
        for i in 0..hidden_dim {
            let diff_0 = (ref_vals_0[i] - grouped_vals_0[i]).abs();
            let diff_1 = (ref_vals_1[i] - grouped_vals_1[i]).abs();
            assert!(
                diff_0 < 1e-3,
                "Expert 0, dim {i}: ref={} grouped={} diff={}",
                ref_vals_0[i],
                grouped_vals_0[i],
                diff_0
            );
            assert!(
                diff_1 < 1e-3,
                "Expert 1, dim {i}: ref={} grouped={} diff={}",
                ref_vals_1[i],
                grouped_vals_1[i],
                diff_1
            );
        }
    }

    #[test]
    fn test_grouped_forward_invalid_expert_idx() {
        let (device, queue, registry) = test_registry();
        let experts = vec![make_expert(&device, 8, 4, 0.1)];
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        let input = Array::from_slice(&device, &[1.0f32; 8], vec![1, 8]);
        // Expert index 5 is out of range (only 1 expert)
        let result = group.grouped_forward(&[(5, &input)], &registry, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_grouped_forward_dimension_mismatch() {
        let (device, queue, registry) = test_registry();
        let experts = vec![make_expert(&device, 8, 4, 0.1)];
        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        // Wrong hidden dim (16 instead of 8)
        let input = Array::from_slice(&device, &[1.0f32; 16], vec![1, 16]);
        let result = group.grouped_forward(&[(0, &input)], &registry, &queue);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_stacking_values() {
        let (device, queue, registry) = test_registry();
        let hidden_dim = 4;
        let intermediate_dim = 2;

        // Expert 0: all weights = 0.1, Expert 1: all weights = 0.5
        let experts = vec![
            make_expert(&device, hidden_dim, intermediate_dim, 0.1),
            make_expert(&device, hidden_dim, intermediate_dim, 0.5),
        ];

        let group = ExpertGroup::from_experts(&experts, &registry, &queue).unwrap();

        // Check that expert 0's gate weights are ~0.1 and expert 1's are ~0.5
        // gate_weights: [2, D=4, inter=2]
        let gate_vals: Vec<f32> = group.gate_weights.to_vec_checked();
        let expert_0_slice = &gate_vals[0..hidden_dim * intermediate_dim];
        let expert_1_slice =
            &gate_vals[hidden_dim * intermediate_dim..2 * hidden_dim * intermediate_dim];

        for &v in expert_0_slice {
            assert!(
                (v - 0.1).abs() < 1e-5,
                "Expert 0 gate weight: expected ~0.1, got {}",
                v
            );
        }
        for &v in expert_1_slice {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "Expert 1 gate weight: expected ~0.5, got {}",
                v
            );
        }
    }
}
