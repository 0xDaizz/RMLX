//! RMSNorm neural network module.
//!
//! Wraps `rmlx_core::ops::rms_norm` as an nn module with a learnable
//! weight (scale) parameter.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

/// RMS Normalization configuration.
pub struct RMSNormConfig {
    /// Number of features (last dimension of input).
    pub normalized_shape: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
}

impl RMSNormConfig {
    /// Create an RMSNormConfig with common defaults (eps=1e-5).
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
        }
    }
}

/// RMS Normalization module.
///
/// Applies `y = x * rsqrt(mean(x^2) + eps) * weight` per row.
///
/// Weight shape: `[normalized_shape]`
///
/// Forward input shape: `[*, normalized_shape]` (2D: `[batch, features]`)
/// Forward output shape: same as input
pub struct RMSNorm {
    weight: Array,
    eps: f32,
}

impl RMSNorm {
    /// Create a new RMSNorm with weight initialized to ones.
    ///
    /// The weight is created as f32 on the given device. The `_dtype` parameter
    /// is accepted for API consistency but currently only f32 weights are
    /// supported (the core kernel handles f16/bf16 inputs with f32 accumulation).
    pub fn new(config: &RMSNormConfig, device: &ProtocolObject<dyn MTLDevice>, _dtype: DType) -> Self {
        let weight = Array::ones(device, &[config.normalized_shape]);
        Self {
            weight,
            eps: config.eps,
        }
    }

    /// Create an RMSNorm from a pre-loaded weight array.
    pub fn from_array(weight: Array, eps: f32) -> Result<Self, KernelError> {
        if weight.ndim() != 1 {
            return Err(KernelError::InvalidShape(format!(
                "RMSNorm: weight must be 1D, got {}D",
                weight.ndim()
            )));
        }
        Ok(Self { weight, eps })
    }

    /// Forward pass: apply RMS normalization.
    ///
    /// `input` shape: `[batch, normalized_shape]`
    /// Returns: same shape as input
    pub fn forward(
        &self,
        registry: &KernelRegistry,
        input: &Array,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Array, KernelError> {
        if input.ndim() != 2 {
            return Err(KernelError::InvalidShape(format!(
                "RMSNorm: input must be 2D [batch, features], got {}D",
                input.ndim()
            )));
        }

        if input.shape()[1] != self.weight.shape()[0] {
            return Err(KernelError::InvalidShape(format!(
                "RMSNorm: input features {} != weight size {}",
                input.shape()[1],
                self.weight.shape()[0]
            )));
        }

        ops::rms_norm::rms_norm(registry, input, &self.weight, self.eps, queue)
    }

    /// Epsilon value for numerical stability.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Reference to the weight array.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// Normalized shape (feature dimension size).
    pub fn normalized_shape(&self) -> usize {
        self.weight.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (rmlx_metal::MtlDevice, KernelRegistry, rmlx_metal::MtlQueue) {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("no Metal device");
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("no GpuDevice");
        let queue = gpu.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        ops::rms_norm::register(&registry).expect("failed to register rms_norm kernels");
        (device, registry, queue)
    }

    #[test]
    fn test_rms_norm_construction() {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("no Metal device");
        let config = RMSNormConfig::new(64);

        let norm = RMSNorm::new(&config, &device, DType::Float32);

        assert_eq!(norm.normalized_shape(), 64);
        assert_eq!(norm.eps(), 1e-5);
        assert_eq!(norm.weight().shape(), &[64]);
        assert_eq!(norm.weight().ndim(), 1);
    }

    #[test]
    fn test_rms_norm_forward_shape() {
        let (device, registry, queue) = setup();
        let config = RMSNormConfig::new(8);
        let norm = RMSNorm::new(&config, &device, DType::Float32);

        let input = Array::ones(&device, &[2, 8]);

        let output = norm
            .forward(&registry, &input, &queue)
            .expect("forward failed");

        assert_eq!(output.shape(), &[2, 8]);
        assert_eq!(output.dtype(), DType::Float32);
    }
}
