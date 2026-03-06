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

// ---------------------------------------------------------------------------
// f16-to-f32 conversion helper (avoids `half` crate dependency)
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision (f16) value stored as u16 to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

// ---------------------------------------------------------------------------
// AWQ (Activation-aware Weight Quantization) linear layer
// ---------------------------------------------------------------------------

/// AWQ packing format: INT4 values packed into u32 words, 8 nibbles per u32.
/// Per-group scales (f16/f32) and zeros (f16/f32) for dequantization.
///
/// Dequantization: `w_i = scale_g * (q_i - zero_g)`
/// where `g = i / group_size`.
///
/// AWQ uses **row-major** packing: consecutive elements along the input
/// dimension are packed into the same u32.
pub struct AwqLinear {
    /// Packed INT4 weight data as u32 words. Each u32 holds 8 nibbles.
    /// Length = out_features * (in_features / 8).
    qweight: Vec<u32>,
    /// Per-group scale factors (f32). Length = out_features * num_groups.
    scales: Vec<f32>,
    /// Per-group zero points (f32). Length = out_features * num_groups.
    zeros: Vec<f32>,
    /// Number of input features (weight columns).
    in_features: usize,
    /// Number of output features (weight rows).
    out_features: usize,
    /// Elements per quantization group (64 or 128).
    group_size: usize,
}

impl AwqLinear {
    /// Create a new `AwqLinear` with validation.
    ///
    /// # Arguments
    /// * `qweight` — Packed INT4 weight data (8 nibbles per u32).
    /// * `scales` — Per-group scale factors.
    /// * `zeros` — Per-group zero points.
    /// * `in_features` — Number of input features.
    /// * `out_features` — Number of output features.
    /// * `group_size` — Elements per quantization group (64 or 128).
    ///
    /// # Errors
    /// Returns `KernelError::InvalidShape` if dimensions are inconsistent.
    pub fn new(
        qweight: Vec<u32>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<Self, KernelError> {
        if group_size != 64 && group_size != 128 {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: group_size must be 64 or 128, got {group_size}"
            )));
        }
        if in_features == 0 || out_features == 0 {
            return Err(KernelError::InvalidShape(
                "AwqLinear: in_features and out_features must be > 0".into(),
            ));
        }
        if in_features % group_size != 0 {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            )));
        }
        if in_features % 8 != 0 {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: in_features ({in_features}) must be a multiple of 8 for INT4 packing"
            )));
        }

        let expected_qweight_len = out_features * (in_features / 8);
        if qweight.len() != expected_qweight_len {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: qweight length ({}) != expected ({expected_qweight_len})",
                qweight.len()
            )));
        }

        let num_groups = in_features / group_size;
        let expected_scale_len = out_features * num_groups;
        if scales.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: scales length ({}) != expected ({expected_scale_len})",
                scales.len()
            )));
        }
        if zeros.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: zeros length ({}) != expected ({expected_scale_len})",
                zeros.len()
            )));
        }

        Ok(Self {
            qweight,
            scales,
            zeros,
            in_features,
            out_features,
            group_size,
        })
    }

    /// Dequantize and perform matmul: `output = x @ W^T`.
    ///
    /// Unpacks INT4 nibbles from u32 words, applies per-group scale and zero,
    /// then performs a standard matmul on CPU.
    ///
    /// `x` shape: `[batch, in_features]` or `[in_features]`.
    /// Returns: `[batch, out_features]`.
    pub fn forward(
        &self,
        x: &Array,
        _registry: &KernelRegistry,
        _queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let (batch, x_flat) = self.normalize_input(x)?;
        let x_vec: Vec<f32> = x_flat.to_vec_checked();

        let mut output = vec![0.0f32; batch * self.out_features];
        let num_groups = self.in_features / self.group_size;

        for m in 0..batch {
            for n in 0..self.out_features {
                let mut acc = 0.0f32;
                for g in 0..num_groups {
                    let scale = self.scales[n * num_groups + g];
                    let zero = self.zeros[n * num_groups + g];
                    let k_start = g * self.group_size;

                    for kk in 0..self.group_size {
                        let k = k_start + kk;
                        // Each u32 holds 8 nibbles (INT4 values).
                        // AWQ row-major: nibble index = k within the row.
                        let word_idx = n * (self.in_features / 8) + k / 8;
                        let nibble_idx = k % 8;
                        let q = ((self.qweight[word_idx] >> (nibble_idx * 4)) & 0xF) as f32;
                        let w = scale * (q - zero);
                        acc += w * x_vec[m * self.in_features + k];
                    }
                }
                output[m * self.out_features + n] = acc;
            }
        }

        let dev = _registry.device().raw();
        Ok(Array::from_slice(
            dev,
            &output,
            vec![batch, self.out_features],
        ))
    }

    /// Normalize input to 2D and validate feature dimension.
    fn normalize_input(&self, x: &Array) -> Result<(usize, Array), KernelError> {
        let (batch, x_2d) = if x.ndim() == 1 {
            (1usize, x.reshape(vec![1, x.shape()[0]])?)
        } else if x.ndim() == 2 {
            (x.shape()[0], x.reshape(vec![x.shape()[0], x.shape()[1]])?)
        } else {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: input must be 1D or 2D, got {}D",
                x.ndim()
            )));
        };
        if x_2d.shape()[1] != self.in_features {
            return Err(KernelError::InvalidShape(format!(
                "AwqLinear: input features mismatch: {} vs {}",
                x_2d.shape()[1],
                self.in_features
            )));
        }
        Ok((batch, x_2d))
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
}

// ---------------------------------------------------------------------------
// GPTQ (GPT Quantization) linear layer
// ---------------------------------------------------------------------------

/// GPTQ packing format: INT4 values packed into u32 words, 8 nibbles per u32.
/// Per-group scales (f32) and zeros (f32) for dequantization.
///
/// Dequantization: `w_i = scale_g * (q_i - zero_g)`
///
/// GPTQ uses **column-major** packing: consecutive elements along the output
/// dimension are packed into the same u32, unlike AWQ which packs along the
/// input dimension.
pub struct GptqLinear {
    /// Packed INT4 weight data as u32 words (column-major packing).
    /// Each u32 holds 8 nibbles packed along the output dimension.
    /// Shape conceptually: [in_features, out_features / 8].
    /// Length = in_features * (out_features / 8).
    qweight: Vec<u32>,
    /// Per-group scale factors (f32). Length = num_groups * out_features.
    /// Layout: [num_groups, out_features] — group-major.
    scales: Vec<f32>,
    /// Per-group zero points (f32). Length = num_groups * out_features.
    /// Layout: [num_groups, out_features] — group-major.
    zeros: Vec<f32>,
    /// Number of input features (weight columns).
    in_features: usize,
    /// Number of output features (weight rows).
    out_features: usize,
    /// Elements per quantization group (64 or 128).
    group_size: usize,
}

impl GptqLinear {
    /// Create a new `GptqLinear` with validation.
    ///
    /// # Arguments
    /// * `qweight` — Packed INT4 weight data (column-major, 8 nibbles per u32).
    /// * `scales` — Per-group scale factors, layout [num_groups, out_features].
    /// * `zeros` — Per-group zero points, layout [num_groups, out_features].
    /// * `in_features` — Number of input features.
    /// * `out_features` — Number of output features.
    /// * `group_size` — Elements per quantization group (64 or 128).
    ///
    /// # Errors
    /// Returns `KernelError::InvalidShape` if dimensions are inconsistent.
    pub fn new(
        qweight: Vec<u32>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<Self, KernelError> {
        if group_size != 64 && group_size != 128 {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: group_size must be 64 or 128, got {group_size}"
            )));
        }
        if in_features == 0 || out_features == 0 {
            return Err(KernelError::InvalidShape(
                "GptqLinear: in_features and out_features must be > 0".into(),
            ));
        }
        if in_features % group_size != 0 {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            )));
        }
        if out_features % 8 != 0 {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: out_features ({out_features}) must be a multiple of 8 for column-major INT4 packing"
            )));
        }

        let expected_qweight_len = in_features * (out_features / 8);
        if qweight.len() != expected_qweight_len {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: qweight length ({}) != expected ({expected_qweight_len})",
                qweight.len()
            )));
        }

        let num_groups = in_features / group_size;
        let expected_scale_len = num_groups * out_features;
        if scales.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: scales length ({}) != expected ({expected_scale_len})",
                scales.len()
            )));
        }
        if zeros.len() != expected_scale_len {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: zeros length ({}) != expected ({expected_scale_len})",
                zeros.len()
            )));
        }

        Ok(Self {
            qweight,
            scales,
            zeros,
            in_features,
            out_features,
            group_size,
        })
    }

    /// Dequantize and perform matmul: `output = x @ W^T`.
    ///
    /// Unpacks INT4 nibbles from column-major u32 words, applies per-group
    /// scale and zero, then performs matmul on CPU.
    ///
    /// `x` shape: `[batch, in_features]` or `[in_features]`.
    /// Returns: `[batch, out_features]`.
    pub fn forward(
        &self,
        x: &Array,
        _registry: &KernelRegistry,
        _queue: &metal::CommandQueue,
    ) -> Result<Array, KernelError> {
        let (batch, x_flat) = self.normalize_input(x)?;
        let x_vec: Vec<f32> = x_flat.to_vec_checked();

        let mut output = vec![0.0f32; batch * self.out_features];
        let num_groups = self.in_features / self.group_size;

        for m in 0..batch {
            for n in 0..self.out_features {
                let mut acc = 0.0f32;
                for k in 0..self.in_features {
                    let g = k / self.group_size;
                    // GPTQ scales/zeros: [num_groups, out_features] layout
                    let scale = self.scales[g * self.out_features + n];
                    let zero = self.zeros[g * self.out_features + n];

                    // Column-major packing: qweight[k, n/8], nibble = n%8
                    let word_idx = k * (self.out_features / 8) + n / 8;
                    let nibble_idx = n % 8;
                    let q = ((self.qweight[word_idx] >> (nibble_idx * 4)) & 0xF) as f32;
                    let w = scale * (q - zero);
                    acc += w * x_vec[m * self.in_features + k];
                }
                output[m * self.out_features + n] = acc;
            }
        }

        let _ = num_groups; // suppress unused warning
        let dev = _registry.device().raw();
        Ok(Array::from_slice(
            dev,
            &output,
            vec![batch, self.out_features],
        ))
    }

    /// Normalize input to 2D and validate feature dimension.
    fn normalize_input(&self, x: &Array) -> Result<(usize, Array), KernelError> {
        let (batch, x_2d) = if x.ndim() == 1 {
            (1usize, x.reshape(vec![1, x.shape()[0]])?)
        } else if x.ndim() == 2 {
            (x.shape()[0], x.reshape(vec![x.shape()[0], x.shape()[1]])?)
        } else {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: input must be 1D or 2D, got {}D",
                x.ndim()
            )));
        };
        if x_2d.shape()[1] != self.in_features {
            return Err(KernelError::InvalidShape(format!(
                "GptqLinear: input features mismatch: {} vs {}",
                x_2d.shape()[1],
                self.in_features
            )));
        }
        Ok((batch, x_2d))
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
}

// ---------------------------------------------------------------------------
// K-quant configuration for GGUF loader mapping
// ---------------------------------------------------------------------------

/// K-quant type identifier, corresponding to GGML k-quant types.
///
/// These types use "super blocks" of 256 elements with nested quantization
/// of scales. The actual decompression is complex; this enum is used
/// primarily for type-mapping in the GGUF loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantType {
    /// Q2_K: 2-bit quantization with k-quant super blocks (256 elements).
    Q2K,
    /// Q3_K: 3-bit quantization with k-quant super blocks (256 elements).
    Q3K,
    /// Q4_K: 4-bit quantization with k-quant super blocks (256 elements).
    Q4K,
    /// Q5_K: 5-bit quantization with k-quant super blocks (256 elements).
    Q5K,
    /// Q6_K: 6-bit quantization with k-quant super blocks (256 elements).
    Q6K,
}

impl KQuantType {
    /// Number of elements per super block (always 256 for k-quants).
    pub fn block_size(self) -> usize {
        256
    }

    /// Size of one super block in bytes.
    pub fn type_size(self) -> usize {
        match self {
            KQuantType::Q2K => 84,
            KQuantType::Q3K => 110,
            KQuantType::Q4K => 144,
            KQuantType::Q5K => 176,
            KQuantType::Q6K => 210,
        }
    }

    /// Effective bits per weight element.
    pub fn bits(self) -> u32 {
        match self {
            KQuantType::Q2K => 2,
            KQuantType::Q3K => 3,
            KQuantType::Q4K => 4,
            KQuantType::Q5K => 5,
            KQuantType::Q6K => 6,
        }
    }
}

/// Configuration produced by the GGUF loader when mapping k-quant tensors.
///
/// Contains the raw super-block data and metadata needed to eventually
/// dequantize the weights into a `QuantizedLinear` or a float array.
#[derive(Debug)]
pub struct KQuantConfig {
    /// The k-quant type of this tensor.
    pub quant_type: KQuantType,
    /// Number of output features (rows).
    pub out_features: usize,
    /// Number of input features (columns).
    pub in_features: usize,
    /// Raw super-block data (opaque bytes from the GGUF file).
    pub raw_data: Vec<u8>,
}

impl KQuantConfig {
    /// Dequantize a Q4_K super-block tensor to f32.
    ///
    /// Q4_K format (144 bytes per 256-element super block):
    /// - 2 bytes: f16 d (super-block scale)
    /// - 2 bytes: f16 dmin (super-block min)
    /// - 12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
    /// - 128 bytes: 256 x 4-bit quantized values
    ///
    /// For non-Q4_K types, returns an error (TODO: implement remaining types).
    pub fn dequantize_to_f32(&self) -> Result<Vec<f32>, KernelError> {
        match self.quant_type {
            KQuantType::Q4K => self.dequantize_q4k(),
            other => Err(KernelError::InvalidShape(format!(
                "KQuantConfig: dequantization for {:?} is not yet implemented (TODO)",
                other
            ))),
        }
    }

    /// Q4_K dequantization (simplified reference implementation).
    ///
    /// Each 144-byte super block holds 256 quantized values in 8 sub-blocks
    /// of 32 elements each. Sub-block scales and mins are packed into 12 bytes
    /// using 6-bit values.
    fn dequantize_q4k(&self) -> Result<Vec<f32>, KernelError> {
        let block_size = 256usize;
        let type_size = 144usize;
        let total_elements = self.out_features * self.in_features;
        let num_blocks = total_elements / block_size;

        if self.raw_data.len() != num_blocks * type_size {
            return Err(KernelError::InvalidShape(format!(
                "KQuantConfig Q4_K: raw_data length ({}) != expected ({})",
                self.raw_data.len(),
                num_blocks * type_size
            )));
        }

        let mut output = Vec::with_capacity(total_elements);

        for b in 0..num_blocks {
            let block = &self.raw_data[b * type_size..(b + 1) * type_size];

            // d and dmin are f16 stored as u16
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);

            // 12 bytes of packed 6-bit scales and mins for 8 sub-blocks
            // Simplified: extract lower 4 bits as scale_idx, upper 2 bits contribute
            // TODO: Full 6-bit extraction matching llama.cpp's get_scale_min_k4
            let scales_raw = &block[4..16];
            let mut sub_scales = [0.0f32; 8];
            let mut sub_mins = [0.0f32; 8];
            for i in 0..8 {
                // Approximate extraction: use byte pairs from the 12-byte region
                // Real Q4_K has a complex bit-packing scheme for 6-bit values.
                // This is a simplified version that extracts the low nibble as scale
                // and high nibble as min from the packed bytes.
                let byte_idx = (i * 3) / 2;
                if byte_idx < scales_raw.len() {
                    let raw = scales_raw[byte_idx];
                    sub_scales[i] = (raw & 0x3F) as f32;
                    // Mins from the next set of 6 bytes (offset by 6 in the 12-byte region)
                    let min_byte_idx = 6 + byte_idx;
                    if min_byte_idx < scales_raw.len() {
                        sub_mins[i] = (scales_raw[min_byte_idx] & 0x3F) as f32;
                    }
                }
            }

            // 128 bytes of 4-bit quantized data (256 nibbles)
            let qdata = &block[16..144];
            for sub in 0..8 {
                let sc = d * sub_scales[sub];
                let mn = dmin * sub_mins[sub];
                for j in 0..32 {
                    let elem_idx = sub * 32 + j;
                    let byte_idx = elem_idx / 2;
                    let nibble = if elem_idx % 2 == 0 {
                        qdata[byte_idx] & 0xF
                    } else {
                        qdata[byte_idx] >> 4
                    };
                    output.push(sc * nibble as f32 - mn);
                }
            }
        }

        Ok(output)
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
