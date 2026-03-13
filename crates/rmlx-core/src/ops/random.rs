//! Random number generation operations using Philox 4x32 PRNG on Metal.
//!
//! Philox is a counter-based RNG that is well-suited for GPU parallelism:
//! each thread computes its own random numbers from a unique counter value,
//! requiring no inter-thread communication. The 4x32 variant produces 4
//! uint32 outputs per round, which are then converted to floating-point
//! values in the desired distribution.

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

/// Metal shader source for Philox 4x32 RNG kernels.
///
/// Uniform: produces values in [low, high).
/// Normal: uses Box-Muller transform on uniform samples.
pub const RANDOM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Philox 4x32-10 constants.
constant constexpr uint PHILOX_M0 = 0xD2511F53;
constant constexpr uint PHILOX_M1 = 0xCD9E8D57;
constant constexpr uint PHILOX_W0 = 0x9E3779B9;
constant constexpr uint PHILOX_W1 = 0xBB67AE85;

struct PhiloxState {
    uint4 counter;
    uint2 key;
};

// Single Philox round.
inline PhiloxState philox_round(PhiloxState s) {
    uint lo0 = s.counter.x * PHILOX_M0;
    uint hi0 = mulhi(s.counter.x, PHILOX_M0);
    uint lo1 = s.counter.z * PHILOX_M1;
    uint hi1 = mulhi(s.counter.z, PHILOX_M1);

    PhiloxState r;
    r.counter = uint4(hi1 ^ s.counter.y ^ s.key.x,
                      lo1,
                      hi0 ^ s.counter.w ^ s.key.y,
                      lo0);
    r.key = uint2(s.key.x + PHILOX_W0, s.key.y + PHILOX_W1);
    return r;
}

// Philox 4x32-10: 10 rounds of the Philox bijection.
inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    PhiloxState s;
    s.counter = counter;
    s.key = key;
    for (int i = 0; i < 10; i++) {
        s = philox_round(s);
    }
    return s.counter;
}

// Convert uint32 to float in [0, 1).
inline float u32_to_f01(uint x) {
    return float(x >> 8) * (1.0f / 16777216.0f);  // 2^24
}

// Uniform distribution in [low, high).
kernel void random_uniform_f32(
    device       float*  output  [[buffer(0)]],
    constant     uint&   numel   [[buffer(1)]],
    constant     uint&   seed    [[buffer(2)]],
    constant     float&  low     [[buffer(3)]],
    constant     float&  high    [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    // Each thread produces up to 4 values from one Philox call.
    uint base = id * 4;
    if (base >= numel) return;

    uint4 counter = uint4(base, 0, 0, 0);
    uint2 key = uint2(seed, seed ^ 0x12345678);
    uint4 rng = philox4x32_10(counter, key);

    float range = high - low;
    for (uint i = 0; i < 4 && (base + i) < numel; i++) {
        float u = u32_to_f01(rng[i]);
        output[base + i] = low + u * range;
    }
}

// Normal distribution using Box-Muller transform.
kernel void random_normal_f32(
    device       float*  output  [[buffer(0)]],
    constant     uint&   numel   [[buffer(1)]],
    constant     uint&   seed    [[buffer(2)]],
    constant     float&  mean    [[buffer(3)]],
    constant     float&  stddev  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    // Each thread produces up to 4 values: 2 pairs from Box-Muller.
    uint base = id * 4;
    if (base >= numel) return;

    uint4 counter = uint4(base, 0, 0, 0);
    uint2 key = uint2(seed, seed ^ 0x12345678);
    uint4 rng = philox4x32_10(counter, key);

    // Convert to uniform (0, 1) — clamp away from 0 for log().
    float u1 = max(u32_to_f01(rng.x), 1e-7f);
    float u2 = u32_to_f01(rng.y);
    float u3 = max(u32_to_f01(rng.z), 1e-7f);
    float u4 = u32_to_f01(rng.w);

    // Box-Muller pair 1.
    float r1 = sqrt(-2.0f * log(u1));
    float theta1 = 2.0f * M_PI_F * u2;
    float z0 = r1 * cos(theta1);
    float z1 = r1 * sin(theta1);

    // Box-Muller pair 2.
    float r2 = sqrt(-2.0f * log(u3));
    float theta2 = 2.0f * M_PI_F * u4;
    float z2 = r2 * cos(theta2);
    float z3 = r2 * sin(theta2);

    float vals[4] = {z0, z1, z2, z3};
    for (uint i = 0; i < 4 && (base + i) < numel; i++) {
        output[base + i] = mean + vals[i] * stddev;
    }
}
"#;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register random kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("random_ops", RANDOM_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a tensor of uniformly distributed random values in `[low, high)`.
///
/// * `shape` - output tensor shape.
/// * `low`   - lower bound (inclusive).
/// * `high`  - upper bound (exclusive).
/// * `seed`  - RNG seed (u32).
///
/// Output dtype is Float32.
pub fn uniform(
    registry: &KernelRegistry,
    shape: &[usize],
    low: f32,
    high: f32,
    seed: u32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if low >= high {
        return Err(KernelError::InvalidShape(format!(
            "uniform: low ({low}) must be < high ({high})"
        )));
    }

    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), shape, DType::Float32));
    }

    let pipeline = registry.get_pipeline("random_uniform_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, shape, DType::Float32);

    let numel_u32 = super::checked_u32(numel, "numel")?;
    let numel_buf = make_scalar_buf(dev, &numel_u32);
    let seed_buf = make_scalar_buf(dev, &seed);
    let low_buf = make_scalar_buf(dev, &low);
    let high_buf = make_scalar_buf(dev, &high);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(out.metal_buffer()), 0);
    enc.set_buffer(1, Some(&numel_buf), 0);
    enc.set_buffer(2, Some(&seed_buf), 0);
    enc.set_buffer(3, Some(&low_buf), 0);
    enc.set_buffer(4, Some(&high_buf), 0);
    // Each thread handles 4 elements.
    let n_threads = numel.div_ceil(4);
    let tg = std::cmp::min(256usize, pipeline.maxTotalThreadsPerThreadgroup());
    enc.dispatch_threads(
        MTLSize {
            width: n_threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

/// Generate a tensor of normally distributed random values.
///
/// * `shape`  - output tensor shape.
/// * `mean`   - mean of the distribution.
/// * `stddev` - standard deviation.
/// * `seed`   - RNG seed (u32).
///
/// Output dtype is Float32. Uses Philox PRNG + Box-Muller transform.
pub fn normal(
    registry: &KernelRegistry,
    shape: &[usize],
    mean: f32,
    stddev: f32,
    seed: u32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if stddev <= 0.0 {
        return Err(KernelError::InvalidShape(format!(
            "normal: stddev ({stddev}) must be > 0"
        )));
    }

    let numel: usize = shape.iter().product();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), shape, DType::Float32));
    }

    let pipeline = registry.get_pipeline("random_normal_f32", DType::Float32)?;
    let dev = registry.device().raw();
    let out = Array::zeros(dev, shape, DType::Float32);

    let numel_u32 = super::checked_u32(numel, "numel")?;
    let numel_buf = make_scalar_buf(dev, &numel_u32);
    let seed_buf = make_scalar_buf(dev, &seed);
    let mean_buf = make_scalar_buf(dev, &mean);
    let std_buf = make_scalar_buf(dev, &stddev);

    let cb = queue.commandBuffer().unwrap();
    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(0, Some(out.metal_buffer()), 0);
    enc.set_buffer(1, Some(&numel_buf), 0);
    enc.set_buffer(2, Some(&seed_buf), 0);
    enc.set_buffer(3, Some(&mean_buf), 0);
    enc.set_buffer(4, Some(&std_buf), 0);
    let n_threads = numel.div_ceil(4);
    let tg = std::cmp::min(256usize, pipeline.maxTotalThreadsPerThreadgroup());
    enc.dispatch_threads(
        MTLSize {
            width: n_threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    super::commit_with_mode(&cb, super::ExecMode::Sync);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_scalar_buf<T>(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    val: &T,
) -> rmlx_metal::MtlBuffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(
                    val as *const T as *const std::ffi::c_void as *mut std::ffi::c_void,
                )
                .unwrap(),
                std::mem::size_of::<T>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (KernelRegistry, rmlx_metal::MtlQueue) {
        let device = rmlx_metal::device::GpuDevice::system_default().expect("Metal device");
        let queue = device.raw().newCommandQueue().unwrap();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register random kernels");
        (registry, queue)
    }

    #[test]
    fn test_uniform_shape() {
        let (reg, q) = setup();
        let out = uniform(&reg, &[3, 4], 0.0, 1.0, 42, &q).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(out.dtype(), DType::Float32);
    }

    #[test]
    fn test_uniform_range() {
        let (reg, q) = setup();
        let out = uniform(&reg, &[1000], 2.0, 5.0, 42, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        for &v in &vals {
            assert!((2.0..5.0).contains(&v), "value {v} out of [2.0, 5.0)");
        }
    }

    #[test]
    fn test_uniform_deterministic() {
        let (reg, q) = setup();
        let a = uniform(&reg, &[100], 0.0, 1.0, 123, &q).unwrap();
        let b = uniform(&reg, &[100], 0.0, 1.0, 123, &q).unwrap();
        let va: Vec<f32> = a.to_vec_checked();
        let vb: Vec<f32> = b.to_vec_checked();
        assert_eq!(va, vb, "same seed should produce same output");
    }

    #[test]
    fn test_normal_shape() {
        let (reg, q) = setup();
        let out = normal(&reg, &[5, 6], 0.0, 1.0, 42, &q).unwrap();
        assert_eq!(out.shape(), &[5, 6]);
        assert_eq!(out.dtype(), DType::Float32);
    }

    #[test]
    fn test_normal_statistics() {
        let (reg, q) = setup();
        let n = 10000;
        let target_mean = 3.0f32;
        let target_std = 2.0f32;
        let out = normal(&reg, &[n], target_mean, target_std, 42, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();

        let sum: f32 = vals.iter().sum();
        let mean = sum / n as f32;
        let var: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();

        // Loose tolerances for statistical tests.
        assert!(
            (mean - target_mean).abs() < 0.2,
            "mean {mean} too far from {target_mean}"
        );
        assert!(
            (std - target_std).abs() < 0.3,
            "std {std} too far from {target_std}"
        );
    }
}
