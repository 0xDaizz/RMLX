//! LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

use crate::array::Array;
use crate::kernels::{KernelError, KernelRegistry};

/// Configuration for LoRA adapters.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f64) -> Result<Self, KernelError> {
        if rank == 0 {
            return Err(KernelError::InvalidShape("LoRA rank must be > 0".into()));
        }
        Ok(Self {
            rank,
            alpha,
            ..Default::default()
        })
    }

    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// A single LoRA adapter layer.
///
/// For a base linear layer W: (out_features, in_features),
/// LoRA adds a low-rank decomposition: delta_W = scaling * B @ A
/// where A: (rank, in_features), B: (out_features, rank).
///
/// The forward pass computes: y = x @ W^T + scaling * x @ A^T @ B^T
pub struct LoraLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub scaling: f64,
    /// A matrix: (rank, in_features) — row-major
    pub lora_a: Vec<f32>,
    /// B matrix: (out_features, rank) — row-major
    pub lora_b: Vec<f32>,
}

impl LoraLayer {
    /// Create a new LoRA layer. A is initialized with small random-like values
    /// (Kaiming-style 1/sqrt(in_features)), B is zero-initialized so that
    /// the initial LoRA contribution is zero.
    pub fn new(in_features: usize, out_features: usize, config: &LoraConfig) -> Self {
        let rank = config.rank;
        let scaling = config.scaling();

        // A: (rank, in_features) — small init
        let inv_sqrt = 1.0 / (in_features as f64).sqrt();
        let lora_a: Vec<f32> = (0..rank * in_features)
            .map(|i| {
                // Deterministic pseudo-random init for reproducibility
                let t = (i as f64 * 0.618033988749895) % 1.0;
                ((t * 2.0 - 1.0) * inv_sqrt) as f32
            })
            .collect();

        // B: (out_features, rank) — zero init
        let lora_b = vec![0.0f32; out_features * rank];

        Self {
            in_features,
            out_features,
            rank,
            scaling,
            lora_a,
            lora_b,
        }
    }

    /// Create a LoRA layer with explicit A and B matrices (for testing).
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
    ) -> Result<Self, KernelError> {
        if lora_a.len() != rank * in_features {
            return Err(KernelError::InvalidShape(format!(
                "lora_a length {} != rank*in_features {}",
                lora_a.len(),
                rank * in_features
            )));
        }
        if lora_b.len() != out_features * rank {
            return Err(KernelError::InvalidShape(format!(
                "lora_b length {} != out_features*rank {}",
                lora_b.len(),
                out_features * rank
            )));
        }
        Ok(Self {
            in_features,
            out_features,
            rank,
            scaling: alpha / rank as f64,
            lora_a,
            lora_b,
        })
    }

    /// Forward pass: computes the LoRA delta.
    /// `base_output` is the output from the frozen base layer.
    /// `input` is the layer input: (batch_size, in_features).
    /// Returns: base_output + scaling * input @ A^T @ B^T
    pub fn forward(
        &self,
        base_output: &[f32],
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, KernelError> {
        if input.len() != batch_size * self.in_features {
            return Err(KernelError::InvalidShape(format!(
                "input length {} != batch_size*in_features {}",
                input.len(),
                batch_size * self.in_features
            )));
        }
        if base_output.len() != batch_size * self.out_features {
            return Err(KernelError::InvalidShape(format!(
                "base_output length {} != batch_size*out_features {}",
                base_output.len(),
                batch_size * self.out_features
            )));
        }

        // Step 1: h = input @ A^T — (batch, in) @ (in, rank) -> (batch, rank)
        let mut h = vec![0.0f32; batch_size * self.rank];
        for b in 0..batch_size {
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += input[b * self.in_features + i] * self.lora_a[r * self.in_features + i];
                }
                h[b * self.rank + r] = sum;
            }
        }

        // Step 2: delta = h @ B^T — (batch, rank) @ (rank, out) -> (batch, out)
        let mut output = base_output.to_vec();
        let scale = self.scaling as f32;
        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += h[b * self.rank + r] * self.lora_b[o * self.rank + r];
                }
                output[b * self.out_features + o] += scale * sum;
            }
        }

        Ok(output)
    }

    /// GPU-accelerated forward pass using Metal matmul.
    ///
    /// `base_output`: [batch_size, out_features] on GPU
    /// `input`: [batch_size, in_features] on GPU
    /// Returns: base_output + scaling * input @ A^T @ B^T as a GPU Array.
    ///
    /// A is stored as (rank, in_features), so A^T is (in_features, rank).
    /// B is stored as (out_features, rank), so B^T is (rank, out_features).
    pub fn forward_gpu(
        &self,
        base_output: &Array,
        input: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let dev = registry.device().raw();
        let batch_size = input.shape()[0];
        if input.shape()[1] != self.in_features {
            return Err(KernelError::InvalidShape(format!(
                "input shape[1]={} != in_features={}",
                input.shape()[1],
                self.in_features
            )));
        }
        if base_output.shape() != [batch_size, self.out_features] {
            return Err(KernelError::InvalidShape(format!(
                "base_output shape {:?} != [{}, {}]",
                base_output.shape(),
                batch_size,
                self.out_features
            )));
        }

        // Upload A^T as (in_features, rank) from row-major A (rank, in_features)
        let mut a_t = vec![0.0f32; self.in_features * self.rank];
        for r in 0..self.rank {
            for i in 0..self.in_features {
                a_t[i * self.rank + r] = self.lora_a[r * self.in_features + i];
            }
        }
        let a_t_arr = Array::from_slice(dev, &a_t, vec![self.in_features, self.rank]);

        // Upload B^T as (rank, out_features) from row-major B (out_features, rank)
        let mut b_t = vec![0.0f32; self.rank * self.out_features];
        for o in 0..self.out_features {
            for r in 0..self.rank {
                b_t[r * self.out_features + o] = self.lora_b[o * self.rank + r];
            }
        }
        let b_t_arr = Array::from_slice(dev, &b_t, vec![self.rank, self.out_features]);

        // h = input @ A^T — (batch, in) @ (in, rank) -> (batch, rank)
        let h = crate::ops::matmul::matmul(registry, input, &a_t_arr, queue)?;

        // delta = h @ B^T — (batch, rank) @ (rank, out) -> (batch, out)
        let delta = crate::ops::matmul::matmul(registry, &h, &b_t_arr, queue)?;

        // scale delta and add to base_output
        let scale_data = vec![self.scaling as f32; batch_size * self.out_features];
        let scale_arr = Array::from_slice(dev, &scale_data, vec![batch_size, self.out_features]);
        let scaled = crate::ops::binary::mul(registry, &delta, &scale_arr, queue)?;
        let result = crate::ops::binary::add(registry, base_output, &scaled, queue)?;

        Ok(result)
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.rank * self.in_features + self.out_features * self.rank
    }
}

/// A collection of LoRA adapters applied to a model.
pub struct LoraModel {
    pub config: LoraConfig,
    pub layers: Vec<(String, LoraLayer)>,
}

impl LoraModel {
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
        }
    }

    /// Add a LoRA adapter for a named layer.
    pub fn add_adapter(&mut self, name: &str, in_features: usize, out_features: usize) {
        let layer = LoraLayer::new(in_features, out_features, &self.config);
        self.layers.push((name.to_string(), layer));
    }

    /// Get a LoRA layer by name.
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.iter().find(|(n, _)| n == name).map(|(_, l)| l)
    }

    /// Get a mutable LoRA layer by name.
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoraLayer> {
        self.layers
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, l)| l)
    }

    /// Total trainable parameters across all LoRA layers.
    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|(_, l)| l.num_params()).sum()
    }
}

/// Training configuration for LoRA fine-tuning.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            num_epochs: 1,
            batch_size: 1,
        }
    }
}

/// Simple SGD trainer for LoRA parameters.
pub struct LoraTrainer {
    pub config: TrainConfig,
}

impl LoraTrainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    /// Compute cross-entropy loss.
    /// `logits`: (batch_size, vocab_size), `targets`: (batch_size,) indices.
    pub fn compute_loss(
        logits: &[f32],
        targets: &[usize],
        vocab_size: usize,
    ) -> Result<f32, KernelError> {
        let batch_size = targets.len();
        if logits.len() != batch_size * vocab_size {
            return Err(KernelError::InvalidShape(format!(
                "logits length {} != batch_size*vocab_size {}",
                logits.len(),
                batch_size * vocab_size
            )));
        }

        let mut total_loss = 0.0f32;
        for b in 0..batch_size {
            let row = &logits[b * vocab_size..(b + 1) * vocab_size];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_sum_exp = max_val + exp_sum.ln();
            let target_logit = row[targets[b]];
            total_loss += log_sum_exp - target_logit;
        }
        Ok(total_loss / batch_size as f32)
    }

    /// Compute gradient of cross-entropy loss w.r.t. logits.
    /// Returns: d(loss)/d(logits), shape (batch_size, vocab_size).
    pub fn compute_loss_grad(
        logits: &[f32],
        targets: &[usize],
        vocab_size: usize,
    ) -> Result<Vec<f32>, KernelError> {
        let batch_size = targets.len();
        if logits.len() != batch_size * vocab_size {
            return Err(KernelError::InvalidShape(format!(
                "logits length {} != batch_size*vocab_size {}",
                logits.len(),
                batch_size * vocab_size
            )));
        }

        let mut grad = vec![0.0f32; logits.len()];
        let inv_batch = 1.0 / batch_size as f32;

        for b in 0..batch_size {
            let row = &logits[b * vocab_size..(b + 1) * vocab_size];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let exp_sum: f32 = exps.iter().sum();

            for v in 0..vocab_size {
                let softmax_v = exps[v] / exp_sum;
                let target_indicator = if v == targets[b] { 1.0 } else { 0.0 };
                // d(CE)/d(logit_v) = (softmax_v - indicator) / batch_size
                grad[b * vocab_size + v] = (softmax_v - target_indicator) * inv_batch;
            }
        }
        Ok(grad)
    }

    /// Perform one training step on a LoRA layer using SGD.
    /// `input`: (batch_size, in_features)
    /// `base_output`: (batch_size, out_features) — from frozen base
    /// `targets`: (batch_size,) — target class indices
    /// `layer`: the LoRA layer to update
    ///
    /// Returns the loss before the update.
    pub fn train_step(
        &self,
        layer: &mut LoraLayer,
        input: &[f32],
        base_output: &[f32],
        targets: &[usize],
    ) -> Result<f32, KernelError> {
        let batch_size = targets.len();
        let out_features = layer.out_features;

        // Forward
        let logits = layer.forward(base_output, input, batch_size)?;
        let loss = Self::compute_loss(&logits, targets, out_features)?;

        // Backward: gradient of loss w.r.t. logits
        let d_logits = Self::compute_loss_grad(&logits, targets, out_features)?;

        // Backward through LoRA: output = base + scale * input @ A^T @ B^T
        // d_output = d_logits (same shape)
        // h = input @ A^T — (batch, rank)
        let rank = layer.rank;
        let in_features = layer.in_features;
        let scale = layer.scaling as f32;

        let mut h = vec![0.0f32; batch_size * rank];
        for b in 0..batch_size {
            for r in 0..rank {
                let mut sum = 0.0f32;
                for i in 0..in_features {
                    sum += input[b * in_features + i] * layer.lora_a[r * in_features + i];
                }
                h[b * rank + r] = sum;
            }
        }

        // d_B: d_logits^T @ h * scale — (out, batch) @ (batch, rank) -> (out, rank)
        let mut d_b = vec![0.0f32; out_features * rank];
        for o in 0..out_features {
            for r in 0..rank {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    sum += d_logits[b * out_features + o] * h[b * rank + r];
                }
                d_b[o * rank + r] = scale * sum;
            }
        }

        // d_h = d_logits @ B * scale — (batch, out) @ (out, rank) -> (batch, rank)
        let mut d_h = vec![0.0f32; batch_size * rank];
        for b in 0..batch_size {
            for r in 0..rank {
                let mut sum = 0.0f32;
                for o in 0..out_features {
                    sum += d_logits[b * out_features + o] * layer.lora_b[o * rank + r];
                }
                d_h[b * rank + r] = scale * sum;
            }
        }

        // d_A: d_h^T @ input — (rank, batch) @ (batch, in) -> (rank, in)
        let mut d_a = vec![0.0f32; rank * in_features];
        for r in 0..rank {
            for i in 0..in_features {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    sum += d_h[b * rank + r] * input[b * in_features + i];
                }
                d_a[r * in_features + i] = sum;
            }
        }

        // SGD update
        let lr = self.config.learning_rate as f32;
        for (w, g) in layer.lora_a.iter_mut().zip(d_a.iter()) {
            *w -= lr * g;
        }
        for (w, g) in layer.lora_b.iter_mut().zip(d_b.iter()) {
            *w -= lr * g;
        }

        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let c = LoraConfig::default();
        assert_eq!(c.rank, 8);
        assert_eq!(c.alpha, 16.0);
        assert_eq!(c.dropout, 0.0);
        assert_eq!(c.target_modules, vec!["q_proj", "v_proj"]);
    }

    #[test]
    fn test_lora_config_new_valid() {
        let c = LoraConfig::new(4, 8.0).unwrap();
        assert_eq!(c.rank, 4);
        assert_eq!(c.alpha, 8.0);
        assert_eq!(c.scaling(), 2.0); // 8.0 / 4
    }

    #[test]
    fn test_lora_config_new_zero_rank_rejected() {
        let result = LoraConfig::new(0, 16.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_config_scaling() {
        let c = LoraConfig {
            rank: 4,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: Vec::new(),
        };
        assert_eq!(c.scaling(), 4.0); // 16.0 / 4
    }

    #[test]
    fn test_lora_layer_dimensions() {
        let config = LoraConfig::new(4, 8.0).unwrap();
        let layer = LoraLayer::new(128, 256, &config);
        assert_eq!(layer.in_features, 128);
        assert_eq!(layer.out_features, 256);
        assert_eq!(layer.rank, 4);
        assert_eq!(layer.lora_a.len(), 4 * 128); // rank * in_features
        assert_eq!(layer.lora_b.len(), 256 * 4); // out_features * rank
    }

    #[test]
    fn test_lora_layer_b_is_zero_initialized() {
        let config = LoraConfig::new(2, 4.0).unwrap();
        let layer = LoraLayer::new(8, 16, &config);
        assert!(layer.lora_b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lora_layer_num_params() {
        let config = LoraConfig::new(4, 8.0).unwrap();
        let layer = LoraLayer::new(64, 128, &config);
        // rank * in_features + out_features * rank = 4*64 + 128*4 = 256 + 512 = 768
        assert_eq!(layer.num_params(), 768);
    }

    #[test]
    fn test_lora_layer_with_weights_valid() {
        let a = vec![1.0f32; 2 * 4]; // rank=2, in=4
        let b = vec![0.0f32; 8 * 2]; // out=8, rank=2
        let layer = LoraLayer::with_weights(4, 8, 2, 4.0, a, b).unwrap();
        assert_eq!(layer.scaling, 2.0);
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 8);
    }

    #[test]
    fn test_lora_layer_with_weights_bad_a_size() {
        let a = vec![1.0f32; 3]; // wrong size
        let b = vec![0.0f32; 8 * 2];
        let result = LoraLayer::with_weights(4, 8, 2, 4.0, a, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_layer_with_weights_bad_b_size() {
        let a = vec![1.0f32; 2 * 4];
        let b = vec![0.0f32; 5]; // wrong size
        let result = LoraLayer::with_weights(4, 8, 2, 4.0, a, b);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_forward_zero_b_is_identity() {
        // B is zero-initialized, so LoRA delta should be zero.
        // output should equal base_output.
        let config = LoraConfig::new(2, 4.0).unwrap();
        let layer = LoraLayer::new(4, 4, &config);

        let base = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, out=4
        let input = vec![0.5, 0.5, 0.5, 0.5]; // batch=1, in=4

        let result = layer.forward(&base, &input, 1).unwrap();
        assert_eq!(result, base);
    }

    #[test]
    fn test_lora_forward_shape_mismatch_input() {
        let config = LoraConfig::new(2, 4.0).unwrap();
        let layer = LoraLayer::new(4, 4, &config);

        let base = vec![1.0; 4];
        let bad_input = vec![1.0; 3]; // wrong length
        let result = layer.forward(&base, &bad_input, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_forward_shape_mismatch_base() {
        let config = LoraConfig::new(2, 4.0).unwrap();
        let layer = LoraLayer::new(4, 4, &config);

        let bad_base = vec![1.0; 3]; // wrong length
        let input = vec![1.0; 4];
        let result = layer.forward(&bad_base, &input, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_model_add_and_get_adapter() {
        let config = LoraConfig::new(4, 8.0).unwrap();
        let mut model = LoraModel::new(config);
        model.add_adapter("q_proj", 64, 64);
        model.add_adapter("v_proj", 64, 64);

        assert!(model.get_layer("q_proj").is_some());
        assert!(model.get_layer("v_proj").is_some());
        assert!(model.get_layer("k_proj").is_none());
    }

    #[test]
    fn test_lora_model_total_params() {
        let config = LoraConfig::new(4, 8.0).unwrap();
        let mut model = LoraModel::new(config);
        model.add_adapter("q_proj", 64, 128);
        model.add_adapter("v_proj", 64, 128);

        // Each: 4*64 + 128*4 = 256 + 512 = 768. Total: 1536.
        assert_eq!(model.total_params(), 1536);
    }

    #[test]
    fn test_lora_model_get_layer_mut() {
        let config = LoraConfig::new(2, 4.0).unwrap();
        let mut model = LoraModel::new(config);
        model.add_adapter("test", 4, 4);

        let layer = model.get_layer_mut("test").unwrap();
        // Mutate B to non-zero.
        layer.lora_b[0] = 1.0;
        assert_eq!(model.get_layer("test").unwrap().lora_b[0], 1.0);
    }

    #[test]
    fn test_train_config_default() {
        let tc = TrainConfig::default();
        assert_eq!(tc.learning_rate, 1e-4);
        assert_eq!(tc.num_epochs, 1);
        assert_eq!(tc.batch_size, 1);
    }

    #[test]
    fn test_compute_loss_valid() {
        // Simple case: 1 sample, 3 classes, target = class 0.
        let logits = vec![2.0, 1.0, 0.0]; // batch=1, vocab=3
        let targets = vec![0];
        let loss = LoraTrainer::compute_loss(&logits, &targets, 3).unwrap();
        // Loss should be positive (cross-entropy is always >= 0).
        assert!(loss >= 0.0);
        // For these logits, class 0 has highest logit -> loss should be small.
        assert!(loss < 2.0);
    }

    #[test]
    fn test_compute_loss_shape_mismatch() {
        let logits = vec![1.0, 2.0]; // batch*vocab should be 3 not 2
        let targets = vec![0];
        let result = LoraTrainer::compute_loss(&logits, &targets, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_loss_grad_sums_to_zero() {
        // For softmax-based CE, the gradient for each sample sums to ~0.
        let logits = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, vocab=4
        let targets = vec![2];
        let grad = LoraTrainer::compute_loss_grad(&logits, &targets, 4).unwrap();
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-5);
    }
}
