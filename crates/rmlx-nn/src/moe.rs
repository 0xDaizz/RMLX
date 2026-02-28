//! Mixture of Experts layer with top-k gating.

use rmlx_core::array::Array;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

use crate::linear::Linear;

pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

impl MoeConfig {
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.num_experts == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: num_experts must be > 0".into(),
            ));
        }
        if self.num_experts_per_token == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: num_experts_per_token must be > 0".into(),
            ));
        }
        if self.num_experts_per_token > self.num_experts {
            return Err(KernelError::InvalidShape(format!(
                "MoeConfig: num_experts_per_token ({}) > num_experts ({})",
                self.num_experts_per_token, self.num_experts
            )));
        }
        if self.hidden_dim == 0 {
            return Err(KernelError::InvalidShape(
                "MoeConfig: hidden_dim must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// A single expert FFN (SwiGLU: gate * up then down projection).
pub struct Expert {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Expert {
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate_out = self.gate_proj.forward(x, registry, queue)?;
        let up_out = self.up_proj.forward(x, registry, queue)?;
        let gate_activated = ops::silu::silu(registry, &gate_out, queue)?;
        let hidden = ops::binary::mul(registry, &gate_activated, &up_out, queue)?;
        self.down_proj.forward(&hidden, registry, queue)
    }
}

pub struct MoeLayer {
    config: MoeConfig,
    gate: Option<Linear>,
    experts: Vec<Expert>,
}

impl MoeLayer {
    /// Config-only constructor.
    pub fn new(config: MoeConfig) -> Result<Self, KernelError> {
        config.validate()?;
        Ok(Self {
            config,
            gate: None,
            experts: Vec::new(),
        })
    }

    /// Create MoE layer with pre-loaded gate and experts.
    pub fn from_layers(
        config: MoeConfig,
        gate: Linear,
        experts: Vec<Expert>,
    ) -> Result<Self, KernelError> {
        config.validate()?;
        if experts.len() != config.num_experts {
            return Err(KernelError::InvalidShape(format!(
                "experts count {} != config.num_experts {}",
                experts.len(),
                config.num_experts
            )));
        }
        Ok(Self {
            config,
            gate: Some(gate),
            experts,
        })
    }

    /// Forward pass for MoE.
    ///
    /// `x`: [seq_len, hidden_dim]
    /// Returns: [seq_len, hidden_dim]
    ///
    /// For each token:
    /// 1. Compute gate logits via gate projection
    /// 2. Find top-k experts by gate score
    /// 3. Route token through selected experts
    /// 4. Weighted-sum expert outputs
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let gate = self
            .gate
            .as_ref()
            .ok_or_else(|| KernelError::InvalidShape("MoE: gate not loaded".into()))?;
        if self.experts.is_empty() {
            return Err(KernelError::InvalidShape("MoE: no experts loaded".into()));
        }

        let seq_len = x.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let top_k = self.config.num_experts_per_token;
        let num_experts = self.config.num_experts;
        let dev = registry.device().raw();
        let elem_size = x.dtype().size_of();

        // Gate logits: [seq_len, num_experts]
        let gate_logits = gate.forward(x, registry, queue)?;

        // Softmax over experts for each token: [seq_len, num_experts]
        let gate_probs = ops::softmax::softmax(registry, &gate_logits, queue)?;

        // Read gate probs to CPU for top-k selection (small enough for CPU routing)
        let probs_vec: Vec<f32> = unsafe { gate_probs.to_vec() };

        // For each token, find top-k expert indices and weights
        let output = Array::zeros(dev, &[seq_len, hidden_dim], x.dtype());

        for tok in 0..seq_len {
            let row_start = tok * num_experts;
            let row_probs = &probs_vec[row_start..row_start + num_experts];

            // Find top-k indices
            let mut indexed: Vec<(usize, f32)> = row_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Normalize top-k weights
            let top_entries: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();
            let weight_sum: f32 = top_entries.iter().map(|(_, w)| w).sum();
            let inv_sum = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                0.0
            };

            // Extract token input: [1, hidden_dim]
            let tok_offset = x.offset() + tok * hidden_dim * elem_size;
            let tok_view = x.view(vec![1, hidden_dim], vec![hidden_dim, 1], tok_offset);
            let tok_input = ops::copy::copy(registry, &tok_view, queue)?;

            // Accumulate weighted expert outputs
            let mut first = true;
            let mut tok_output: Option<Array> = None;

            for (expert_idx, raw_weight) in &top_entries {
                let w = raw_weight * inv_sum;
                let expert_out = self.experts[*expert_idx].forward(&tok_input, registry, queue)?;

                // Scale expert output by weight
                let scale_data = vec![w; hidden_dim];
                let scale_arr = Array::from_slice(dev, &scale_data, vec![1, hidden_dim]);
                let scaled = ops::binary::mul(registry, &expert_out, &scale_arr, queue)?;

                tok_output = Some(if first {
                    first = false;
                    scaled
                } else {
                    ops::binary::add(registry, tok_output.as_ref().unwrap(), &scaled, queue)?
                });
            }

            // Copy accumulated output into the result row
            if let Some(ref out_tok) = tok_output {
                let dst_offset = tok * hidden_dim * elem_size;
                let dst_view = output.view(vec![1, hidden_dim], vec![hidden_dim, 1], dst_offset);
                let copy_kernel = match x.dtype() {
                    rmlx_core::dtype::DType::Float32 => "copy_f32",
                    rmlx_core::dtype::DType::Float16 => "copy_f16",
                    rmlx_core::dtype::DType::Bfloat16 => "copy_bf16",
                    _ => unreachable!(),
                };
                let pipeline = registry.get_pipeline(copy_kernel, x.dtype())?;
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(out_tok.metal_buffer()), out_tok.offset() as u64);
                enc.set_buffer(1, Some(dst_view.metal_buffer()), dst_view.offset() as u64);
                let grid = metal::MTLSize::new(hidden_dim as u64, 1, 1);
                let tg = metal::MTLSize::new(
                    std::cmp::min(
                        pipeline.max_total_threads_per_threadgroup(),
                        hidden_dim as u64,
                    ),
                    1,
                    1,
                );
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
        }

        Ok(output)
    }

    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.config.num_experts_per_token
    }

    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }
}
