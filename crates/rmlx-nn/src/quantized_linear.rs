//! Quantized linear layer using MLX affine quantization format.
//!
//! Wraps `rmlx-core::ops::quantized` for Q4 (4-bit) and Q8 (8-bit)
//! quantized weight matrices. Automatically selects between:
//! - `affine_quantized_matmul` (QMV) for single-vector inputs (batch=1)
//! - `affine_quantized_matmul_batched` (QMM) for batched inputs (batch>1, Q4 only)
//! - CPU `affine_qmm` fallback for Q4 batched when Metal QMM is unavailable

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops::copy::copy_cast;
use rmlx_core::ops::quantized::{self, QuantizedWeight};

/// Supported quantization bit widths for `QuantizedLinear`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 4-bit quantization (Q4). Supports both QMV and QMM paths.
    Q4 = 4,
    /// 8-bit quantization (Q8). Supports QMV path only.
    Q8 = 8,
}

impl QuantBits {
    /// Return the raw u32 bit width.
    pub fn bits(self) -> u32 {
        self as u32
    }
}

/// Quantized linear layer configuration.
pub struct QuantizedLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
    pub bits: QuantBits,
}

/// Quantized linear layer wrapping MLX affine quantized weights.
///
/// Stores packed quantized weights, per-group scales, and per-group biases
/// as CPU-side `Vec`s. For GPU execution, constructs a [`QuantizedWeight`]
/// on-the-fly and dispatches to the appropriate Metal kernel.
///
/// Two execution paths:
/// - **batch=1**: `affine_quantized_matmul` (single-vector QMV kernel)
/// - **batch>1**: `affine_quantized_matmul_batched` (tiled QMM kernel, Q4 only)
///   or CPU `affine_qmm` fallback for Q8.
pub struct QuantizedLinear {
    /// Packed quantized weight data.
    /// For Q4: each byte holds 2 values (nibbles). Length = out_features * (in_features / 2).
    /// For Q8: each byte holds 1 value. Length = out_features * in_features.
    w_packed: Vec<u8>,
    /// Per-group scale factors. Length = out_features * (in_features / group_size).
    scales: Vec<f32>,
    /// Per-group bias (zero-point) terms. Length = out_features * (in_features / group_size).
    biases: Vec<f32>,
    /// Number of input features (weight columns).
    in_features: usize,
    /// Number of output features (weight rows).
    out_features: usize,
    /// Elements per quantization group (32, 64, or 128).
    group_size: usize,
    /// Quantization bit width.
    bits: QuantBits,
}

impl QuantizedLinear {
    /// Create a new `QuantizedLinear` with validation.
    ///
    /// # Errors
    /// Returns `KernelError::InvalidShape` if dimensions are inconsistent.
    pub fn new(
        w_packed: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        bits: QuantBits,
    ) -> Result<Self, KernelError> {
        // Validate group_size
        if ![32, 64, 128].contains(&group_size) {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: group_size must be 32, 64, or 128, got {group_size}"
            )));
        }

        if in_features == 0 || out_features == 0 {
            return Err(KernelError::InvalidShape(
                "QuantizedLinear: in_features and out_features must be > 0".into(),
            ));
        }

        if in_features % group_size != 0 {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            )));
        }

        // Validate packed weight size
        let expected_packed_len = match bits {
            QuantBits::Q4 => out_features * (in_features / 2),
            QuantBits::Q8 => out_features * in_features,
        };
        if w_packed.len() != expected_packed_len {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: w_packed length ({}) != expected ({expected_packed_len}) \
                 for [{out_features}, {in_features}] at {} bits",
                w_packed.len(),
                bits.bits()
            )));
        }

        // Validate scales and biases
        let groups_per_row = in_features / group_size;
        let expected_scale_len = out_features * groups_per_row;
        if scales.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: scales length ({}) != expected ({expected_scale_len})",
                scales.len()
            )));
        }
        if biases.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: biases length ({}) != expected ({expected_scale_len})",
                biases.len()
            )));
        }

        Ok(Self {
            w_packed,
            scales,
            biases,
            in_features,
            out_features,
            group_size,
            bits,
        })
    }

    /// Forward pass: quantized linear projection.
    ///
    /// `x` shape: `[batch, in_features]` or `[in_features]` (treated as `[1, in_features]`).
    /// Returns: `[batch, out_features]`.
    ///
    /// - batch=1: dispatches to `affine_quantized_matmul` (QMV kernel).
    /// - batch>1 + Q4: dispatches to `affine_quantized_matmul_batched` (QMM kernel).
    /// - batch>1 + Q8: falls back to CPU `affine_qmm` (no Metal Q8 QMM kernel yet).
    pub fn forward(
        &self,
        x: &Array,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        // Normalize input to 2D
        let (batch, x_2d) = if x.ndim() == 1 {
            (1usize, x.reshape(vec![1, x.shape()[0]])?)
        } else if x.ndim() == 2 {
            (x.shape()[0], x.reshape(vec![x.shape()[0], x.shape()[1]])?)
        } else {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: input must be 1D or 2D, got {}D",
                x.ndim()
            )));
        };

        // Cast to f32 if needed — quantized matmul requires f32 input.
        let x_2d = if x_2d.dtype() != DType::Float32 {
            copy_cast(registry, &x_2d, DType::Float32, queue)?
        } else {
            x_2d
        };

        if x_2d.shape()[1] != self.in_features {
            return Err(KernelError::InvalidShape(format!(
                "QuantizedLinear: input features mismatch: {} vs {}",
                x_2d.shape()[1],
                self.in_features
            )));
        }

        // Build QuantizedWeight from our CPU buffers.
        // We need to upload to Metal buffers.
        let dev = registry.device().raw();
        let qw = self.make_quantized_weight(dev)?;

        if batch == 1 {
            // Single vector: use QMV kernel (works for both Q4 and Q8)
            let vec_1d = x_2d.view(vec![self.in_features], vec![1], x_2d.offset());
            let out_1d = quantized::affine_quantized_matmul(registry, &qw, &vec_1d, queue)?;
            // Reshape to [1, out_features]
            out_1d.reshape(vec![1, self.out_features])
        } else {
            // Batched: use QMM kernel for Q4, CPU fallback for Q8
            match self.bits {
                QuantBits::Q4 => {
                    quantized::affine_quantized_matmul_batched(registry, &x_2d, &qw, queue)
                }
                QuantBits::Q8 => {
                    // CPU fallback: read x to CPU, run affine_qmm, upload result
                    self.forward_cpu_fallback(&x_2d, batch, dev)
                }
            }
        }
    }

    /// CPU fallback for Q8 batched matmul (no Metal QMM kernel for Q8 yet).
    fn forward_cpu_fallback(
        &self,
        x_2d: &Array,
        batch: usize,
        dev: &metal::Device,
    ) -> Result<Array, KernelError> {
        let x_vec: Vec<f32> = x_2d.to_vec_checked();
        let mut output = vec![0.0f32; batch * self.out_features];

        // For Q8, we need to adapt the data to the affine_qmm format which
        // expects Q4 nibble packing. Instead, do a simple dequant-and-matmul.
        for m in 0..batch {
            for n in 0..self.out_features {
                let mut acc = 0.0f32;
                let groups_per_row = self.in_features / self.group_size;
                for g in 0..groups_per_row {
                    let k_start = g * self.group_size;
                    let scale = self.scales[n * groups_per_row + g];
                    let bias = self.biases[n * groups_per_row + g];

                    let mut group_dot = 0.0f32;
                    let mut group_xsum = 0.0f32;

                    for kk in k_start..k_start + self.group_size {
                        let q = self.w_packed[n * self.in_features + kk] as f32;
                        let xv = x_vec[m * self.in_features + kk];
                        group_dot += q * xv;
                        group_xsum += xv;
                    }

                    acc += scale * group_dot + bias * group_xsum;
                }
                output[m * self.out_features + n] = acc;
            }
        }

        Ok(Array::from_slice(
            dev,
            &output,
            vec![batch, self.out_features],
        ))
    }

    /// Build a `QuantizedWeight` by uploading CPU buffers to Metal.
    fn make_quantized_weight(&self, dev: &metal::Device) -> Result<QuantizedWeight, KernelError> {
        let opts = metal::MTLResourceOptions::StorageModeShared;

        let weights_buf = dev.new_buffer_with_data(
            self.w_packed.as_ptr() as *const _,
            self.w_packed.len() as u64,
            opts,
        );
        let scales_buf = dev.new_buffer_with_data(
            self.scales.as_ptr() as *const _,
            (self.scales.len() * std::mem::size_of::<f32>()) as u64,
            opts,
        );
        let biases_buf = dev.new_buffer_with_data(
            self.biases.as_ptr() as *const _,
            (self.biases.len() * std::mem::size_of::<f32>()) as u64,
            opts,
        );

        QuantizedWeight::new(
            weights_buf,
            scales_buf,
            biases_buf,
            self.group_size as u32,
            self.bits.bits(),
            self.out_features,
            self.in_features,
        )
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn bits(&self) -> QuantBits {
        self.bits
    }

    /// Reference to the packed weight data.
    pub fn w_packed(&self) -> &[u8] {
        &self.w_packed
    }

    /// Reference to the scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Reference to the bias terms.
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_config_validation() {
        // Valid Q4 config: in=64, out=32, group_size=32
        let in_f = 64;
        let out_f = 32;
        let group_size = 32;
        let w_packed = vec![0u8; out_f * (in_f / 2)];
        let groups_per_row = in_f / group_size;
        let scales = vec![1.0f32; out_f * groups_per_row];
        let biases = vec![0.0f32; out_f * groups_per_row];

        let ql = QuantizedLinear::new(
            w_packed,
            scales,
            biases,
            in_f,
            out_f,
            group_size,
            QuantBits::Q4,
        );
        assert!(ql.is_ok());
        let ql = ql.unwrap();
        assert_eq!(ql.in_features(), 64);
        assert_eq!(ql.out_features(), 32);
        assert_eq!(ql.group_size(), 32);
        assert_eq!(ql.bits(), QuantBits::Q4);
    }

    #[test]
    fn test_quantized_linear_q8_config_validation() {
        let in_f = 128;
        let out_f = 64;
        let group_size = 64;
        let w_packed = vec![0u8; out_f * in_f];
        let groups_per_row = in_f / group_size;
        let scales = vec![1.0f32; out_f * groups_per_row];
        let biases = vec![0.0f32; out_f * groups_per_row];

        let ql = QuantizedLinear::new(
            w_packed,
            scales,
            biases,
            in_f,
            out_f,
            group_size,
            QuantBits::Q8,
        );
        assert!(ql.is_ok());
        assert_eq!(ql.unwrap().bits(), QuantBits::Q8);
    }

    #[test]
    fn test_quantized_linear_invalid_group_size() {
        let result = QuantizedLinear::new(
            vec![0u8; 16],
            vec![1.0f32; 2],
            vec![0.0f32; 2],
            32,
            1,
            16, // invalid group_size
            QuantBits::Q4,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_linear_mismatched_packed_len() {
        let result = QuantizedLinear::new(
            vec![0u8; 10], // wrong length
            vec![1.0f32; 1],
            vec![0.0f32; 1],
            32,
            1,
            32,
            QuantBits::Q4,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_linear_zero_features() {
        let result = QuantizedLinear::new(vec![], vec![], vec![], 0, 0, 32, QuantBits::Q4);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_linear_in_features_not_multiple_of_group() {
        let result = QuantizedLinear::new(
            vec![0u8; 100],
            vec![1.0f32; 10],
            vec![0.0f32; 10],
            33, // not a multiple of 32
            1,
            32,
            QuantBits::Q4,
        );
        assert!(result.is_err());
    }
}
