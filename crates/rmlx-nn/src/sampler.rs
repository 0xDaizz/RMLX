//! Sampler module for text generation.
//!
//! Converts model logits into token IDs via temperature scaling, top-k/top-p
//! filtering, repetition penalty, softmax, and weighted random sampling.
//!
//! v1: CPU-side fallback for sort/sample operations. GPU sort path is a TODO.

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_core::kernels::{KernelError, KernelRegistry};
use rmlx_core::ops;

// ---------------------------------------------------------------------------
// SamplerConfig
// ---------------------------------------------------------------------------

/// Configuration for the sampling pipeline.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for logit scaling. 0.0 means greedy (argmax).
    pub temperature: f32,
    /// Keep only the top-k highest logits. 0 means disabled.
    pub top_k: usize,
    /// Nucleus sampling threshold. 1.0 means disabled.
    pub top_p: f32,
    /// Repetition penalty factor. 1.0 means no penalty.
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler struct
// ---------------------------------------------------------------------------

/// Stateless sampler that chains: repetition_penalty -> temperature -> top_k -> top_p -> sample.
pub struct Sampler;

impl Sampler {
    /// Convenience method: apply the full sampling pipeline and return a token ID.
    ///
    /// `logits` must be a 1-D f32 array of shape `[vocab_size]`.
    pub fn sample_token(
        logits: &Array,
        past_tokens: &[u32],
        config: &SamplerConfig,
        registry: &KernelRegistry,
        queue: &metal::CommandQueue,
    ) -> Result<u32, KernelError> {
        validate_1d_f32(logits, "sample_token")?;

        // Validate sampler config
        if config.temperature < 0.0 {
            return Err(KernelError::InvalidShape(
                "sample_token: temperature must be >= 0.0 (0.0 = greedy)".to_string(),
            ));
        }
        if config.repetition_penalty <= 0.0 {
            return Err(KernelError::InvalidShape(
                "sample_token: repetition_penalty must be > 0.0".to_string(),
            ));
        }
        if config.top_p <= 0.0 || config.top_p > 1.0 {
            return Err(KernelError::InvalidShape(
                "sample_token: top_p must be in (0.0, 1.0]".to_string(),
            ));
        }

        let device = registry.device().raw();

        // Read logits to CPU once; we'll work on the CPU side for the pipeline.
        let mut data = logits.to_vec_checked::<f32>();
        let vocab_size = data.len();

        if vocab_size == 0 {
            return Err(KernelError::InvalidShape(
                "sample_token: logits array is empty (0 elements)".to_string(),
            ));
        }

        // 1. Repetition penalty
        if config.repetition_penalty != 1.0 && !past_tokens.is_empty() {
            apply_repetition_penalty_cpu(&mut data, past_tokens, config.repetition_penalty);
        }

        // 2. Temperature
        if config.temperature == 0.0 {
            // Greedy: just return argmax
            return Ok(argmax_cpu(&data) as u32);
        }
        if config.temperature != 1.0 {
            let inv_temp = 1.0 / config.temperature;
            for v in data.iter_mut() {
                *v *= inv_temp;
            }
        }

        // 3. Top-k
        if config.top_k > 0 && config.top_k < vocab_size {
            apply_top_k_cpu(&mut data, config.top_k);
        }

        // 4. Top-p (needs softmax, so we go through GPU for that)
        if config.top_p > 0.0 && config.top_p < 1.0 {
            apply_top_p_cpu(&mut data, config.top_p);
        }

        // 5. Sample: compute softmax on GPU, then sample on CPU
        let logits_arr = Array::from_slice(device, &data, vec![vocab_size]);
        sample(&logits_arr, registry, queue)
    }
}

// ---------------------------------------------------------------------------
// Individual sampling functions (public API)
// ---------------------------------------------------------------------------

/// Scale logits by `1 / temp`.
///
/// `logits` must be a 1-D f32 array. Returns a new array with scaled values.
pub fn temperature(
    logits: &Array,
    temp: f32,
    device: &metal::Device,
) -> Result<Array, KernelError> {
    validate_1d_f32(logits, "temperature")?;
    if temp <= 0.0 {
        return Err(KernelError::InvalidShape(
            "temperature must be > 0".to_string(),
        ));
    }
    let inv_temp = 1.0 / temp;
    let data = logits.to_vec_checked::<f32>();
    let scaled: Vec<f32> = data.iter().map(|&v| v * inv_temp).collect();
    Ok(Array::from_slice(device, &scaled, vec![scaled.len()]))
}

/// Mask all but the top-k logits to `-inf`.
///
/// CPU-side fallback: reads logits to host, finds the k-th largest value,
/// masks everything below it.
///
/// TODO: Replace with GPU sort path when a Metal bitonic/radix sort kernel is available.
pub fn top_k(logits: &Array, k: usize, device: &metal::Device) -> Result<Array, KernelError> {
    validate_1d_f32(logits, "top_k")?;
    let mut data = logits.to_vec_checked::<f32>();
    let vocab_size = data.len();

    if k == 0 || k >= vocab_size {
        // No filtering needed — return a copy
        return Ok(Array::from_slice(device, &data, vec![vocab_size]));
    }

    apply_top_k_cpu(&mut data, k);
    Ok(Array::from_slice(device, &data, vec![vocab_size]))
}

/// Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative
/// probability mass exceeds `p`.
///
/// CPU-side fallback: reads logits to host, computes softmax on GPU, sorts by
/// descending probability on CPU, finds the cutoff, masks the rest to `-inf`.
///
/// TODO: Replace with GPU sort path when available.
pub fn top_p(
    logits: &Array,
    p: f32,
    device: &metal::Device,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    validate_1d_f32(logits, "top_p")?;
    if p <= 0.0 || p > 1.0 {
        return Err(KernelError::InvalidShape(
            "top_p must be in (0, 1]".to_string(),
        ));
    }

    let mut data = logits.to_vec_checked::<f32>();
    let vocab_size = data.len();

    if p >= 1.0 {
        return Ok(Array::from_slice(device, &data, vec![vocab_size]));
    }

    // We need softmax probabilities for top-p. Compute on GPU, read back.
    let probs = ops::softmax::softmax(registry, logits, queue)?;
    let prob_data = probs.to_vec_checked::<f32>();

    // Sort indices by descending probability
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| {
        prob_data[b]
            .partial_cmp(&prob_data[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the cutoff: smallest set whose cumulative prob >= p
    let mut cumulative = 0.0f32;
    let mut keep = vec![false; vocab_size];
    for &idx in &indices {
        keep[idx] = true;
        cumulative += prob_data[idx];
        if cumulative >= p {
            break;
        }
    }

    // Mask logits not in the keep set
    for (i, v) in data.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }

    Ok(Array::from_slice(device, &data, vec![vocab_size]))
}

/// Apply softmax and do weighted random sampling to produce a single token ID.
///
/// CPU-side: reads softmax probabilities to host, does weighted random pick.
pub fn sample(
    logits: &Array,
    registry: &KernelRegistry,
    queue: &metal::CommandQueue,
) -> Result<u32, KernelError> {
    validate_1d_f32(logits, "sample")?;

    if logits.shape()[0] == 0 {
        return Err(KernelError::InvalidShape(
            "sample: logits array is empty (0 elements)".to_string(),
        ));
    }

    let probs = ops::softmax::softmax(registry, logits, queue)?;
    let prob_data = probs.to_vec_checked::<f32>();

    // Weighted random sampling using a simple PRNG seeded from system time.
    let r = simple_random_f32();

    let mut cumulative = 0.0f32;
    for (i, &p) in prob_data.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Ok(i as u32);
        }
    }

    // Fallback: return last token (handles floating-point rounding)
    Ok((prob_data.len() - 1) as u32)
}

/// Apply repetition penalty to logits for previously generated tokens.
///
/// For each token in `past_tokens`:
/// - If logit > 0, divide by `penalty`
/// - If logit < 0, multiply by `penalty`
///
/// This follows the approach from the Ctrl paper (Keskar et al., 2019).
pub fn repetition_penalty(
    logits: &Array,
    past_tokens: &[u32],
    penalty: f32,
    device: &metal::Device,
) -> Result<Array, KernelError> {
    validate_1d_f32(logits, "repetition_penalty")?;
    if penalty <= 0.0 {
        return Err(KernelError::InvalidShape(
            "repetition_penalty must be > 0".to_string(),
        ));
    }

    let mut data = logits.to_vec_checked::<f32>();
    let vocab_size = data.len();

    if penalty != 1.0 {
        apply_repetition_penalty_cpu(&mut data, past_tokens, penalty);
    }

    Ok(Array::from_slice(device, &data, vec![vocab_size]))
}

// ---------------------------------------------------------------------------
// CPU-side helpers (in-place)
// ---------------------------------------------------------------------------

/// Apply repetition penalty in-place on a CPU vec.
fn apply_repetition_penalty_cpu(data: &mut [f32], past_tokens: &[u32], penalty: f32) {
    let vocab_size = data.len();
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < vocab_size {
            if data[idx] > 0.0 {
                data[idx] /= penalty;
            } else {
                data[idx] *= penalty;
            }
        }
    }
}

/// Apply top-k masking in-place: set all but the top-k values to -inf.
fn apply_top_k_cpu(data: &mut [f32], k: usize) {
    let vocab_size = data.len();
    if k >= vocab_size {
        return;
    }

    // Find the k-th largest value by partial sort on CPU
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];

    // Mask values below threshold to -inf.
    // To handle ties correctly: allow exactly k values through.
    let mut count = 0usize;
    for v in data.iter_mut() {
        if *v >= threshold && count < k {
            count += 1;
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) filtering in-place using CPU-side softmax.
///
/// This computes a simple CPU softmax (no GPU round-trip) for the filtering step.
fn apply_top_p_cpu(data: &mut [f32], p: f32) {
    let vocab_size = data.len();

    // CPU softmax for probability computation
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = if sum > 0.0 {
        probs.iter().map(|&v| v / sum).collect()
    } else {
        probs
    };

    // Sort indices by descending probability
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the cutoff
    let mut cumulative = 0.0f32;
    let mut keep = vec![false; vocab_size];
    for &idx in &indices {
        keep[idx] = true;
        cumulative += probs[idx];
        if cumulative >= p {
            break;
        }
    }

    // Mask
    for (i, v) in data.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Return the index of the maximum value.
fn argmax_cpu(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i, v)
            } else {
                (best_i, best_v)
            }
        })
        .0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that the array is 1-D and Float32.
fn validate_1d_f32(arr: &Array, fn_name: &str) -> Result<(), KernelError> {
    if arr.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected 1D array, got {}D",
            arr.ndim()
        )));
    }
    if arr.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected Float32, got {:?}",
            arr.dtype()
        )));
    }
    Ok(())
}

/// Simple pseudo-random f32 in [0, 1) using system time.
///
/// This is intentionally minimal to avoid adding an external `rand` dependency.
/// For production use, replace with a proper PRNG (e.g., `rand::thread_rng()`).
fn simple_random_f32() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // Xorshift-style mixing of the nanos value
    let mut x = nanos;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x as f32) / (u32::MAX as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (metal::Device, metal::CommandQueue, KernelRegistry) {
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("GPU device");
        let device = gpu.raw().clone();
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        ops::softmax::register(&registry).expect("register softmax");
        ops::register_all(&registry).unwrap_or(()); // register binary ops etc.
        (device, queue, registry)
    }

    #[test]
    fn test_temperature_zero_returns_argmax() {
        let (device, queue, registry) = setup();

        // logits: token 3 has the highest value
        let logits = Array::from_slice(&device, &[1.0f32, 0.5, 2.0, 10.0, 0.1], vec![5]);

        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };

        let token =
            Sampler::sample_token(&logits, &[], &config, &registry, &queue).expect("sample");
        assert_eq!(token, 3, "temperature=0 should return argmax (index 3)");
    }

    #[test]
    fn test_top_k_1_returns_argmax() {
        let (device, queue, registry) = setup();

        // logits: token 2 has the highest value
        let logits = Array::from_slice(&device, &[1.0f32, 0.5, 10.0, 2.0, 0.1], vec![5]);

        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        // With top_k=1, only the argmax token survives; softmax gives it prob ~1.0
        let token =
            Sampler::sample_token(&logits, &[], &config, &registry, &queue).expect("sample");
        assert_eq!(token, 2, "top_k=1 should return argmax (index 2)");
    }

    #[test]
    fn test_sample_produces_valid_token_ids() {
        let (device, queue, registry) = setup();

        let vocab_size = 100;
        let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.01).collect();
        let logits = Array::from_slice(&device, &logits_data, vec![vocab_size]);

        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        // Sample multiple times and verify all IDs are in range
        for _ in 0..20 {
            let token =
                Sampler::sample_token(&logits, &[], &config, &registry, &queue).expect("sample");
            assert!(
                (token as usize) < vocab_size,
                "token {token} out of range [0, {vocab_size})"
            );
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let (device, _queue, _registry) = setup();

        let logits = Array::from_slice(&device, &[2.0f32, 4.0, 6.0], vec![3]);
        let scaled = temperature(&logits, 2.0, &device).expect("temperature");
        let data = scaled.to_vec_checked::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_masking() {
        let (device, _queue, _registry) = setup();

        let logits = Array::from_slice(&device, &[1.0f32, 5.0, 3.0, 2.0, 4.0], vec![5]);
        let masked = top_k(&logits, 2, &device).expect("top_k");
        let data = masked.to_vec_checked::<f32>();

        // Top-2 values are 5.0 (index 1) and 4.0 (index 4)
        assert_eq!(data[1], 5.0);
        assert_eq!(data[4], 4.0);
        assert_eq!(data[0], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        assert_eq!(data[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let (device, _queue, _registry) = setup();

        let logits = Array::from_slice(&device, &[2.0f32, 4.0, 6.0, 1.0], vec![4]);
        let penalized =
            repetition_penalty(&logits, &[1, 2], 2.0, &device).expect("repetition_penalty");
        let data = penalized.to_vec_checked::<f32>();

        // Token 0 and 3: unchanged
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
        // Token 1 (positive, 4.0): divided by 2.0 -> 2.0
        assert!((data[1] - 2.0).abs() < 1e-6);
        // Token 2 (positive, 6.0): divided by 2.0 -> 3.0
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let (device, _queue, _registry) = setup();

        let logits = Array::from_slice(&device, &[-2.0f32, -4.0, 1.0], vec![3]);
        let penalized =
            repetition_penalty(&logits, &[0, 1], 2.0, &device).expect("repetition_penalty");
        let data = penalized.to_vec_checked::<f32>();

        // Token 0 (negative, -2.0): multiplied by 2.0 -> -4.0
        assert!((data[0] - (-4.0)).abs() < 1e-6);
        // Token 1 (negative, -4.0): multiplied by 2.0 -> -8.0
        assert!((data[1] - (-8.0)).abs() < 1e-6);
        // Token 2: unchanged
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampler_config_defaults() {
        let config = SamplerConfig::default();
        assert!((config.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 0);
        assert!((config.top_p - 1.0).abs() < f32::EPSILON);
        assert!((config.repetition_penalty - 1.0).abs() < f32::EPSILON);
    }
}
