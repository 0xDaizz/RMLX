//! Sampler module for text generation.
//!
//! Converts model logits into token IDs via temperature scaling, top-k/top-p
//! filtering, repetition penalty, softmax, and weighted random sampling.
//!
//! v1: CPU-side fallback for sort/sample operations. GPU sort path is a TODO.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};
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
// Speculative decoding helpers
// ---------------------------------------------------------------------------

/// Result of speculative decoding verification.
#[derive(Debug, Clone)]
pub struct SpecDecodeResult {
    /// Number of draft tokens accepted (0..=num_draft).
    pub num_accepted: usize,
    /// Accepted token IDs (length = num_accepted).
    pub accepted_tokens: Vec<u32>,
    /// Correction token sampled from the target distribution at the
    /// first rejected position (or bonus token if all accepted).
    /// The serving layer appends this after the accepted tokens.
    pub correction_token: u32,
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
        queue: &ProtocolObject<dyn MTLCommandQueue>,
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
// Speculative decoding functions (public API)
// ---------------------------------------------------------------------------

/// Greedy verification for speculative decoding.
///
/// Compares target model argmax with draft tokens at each position.
/// Accepts as long as `argmax(target_logits[i]) == draft_tokens[i]`.
///
/// `target_logits` must be a 2-D f32 array of shape `[num_draft + 1, vocab_size]`.
/// The extra position is for the bonus/correction token.
/// `draft_tokens` has length `num_draft`.
///
/// Returns `SpecDecodeResult` with accepted count and correction token.
pub fn greedy_verify(
    target_logits: &Array,
    draft_tokens: &[u32],
) -> Result<SpecDecodeResult, KernelError> {
    if target_logits.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "greedy_verify: expected 2D target_logits, got {}D",
            target_logits.ndim()
        )));
    }
    if target_logits.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "greedy_verify: expected Float32, got {:?}",
            target_logits.dtype()
        )));
    }

    let num_positions = target_logits.shape()[0];
    let vocab_size = target_logits.shape()[1];
    let num_draft = draft_tokens.len();

    if num_positions < num_draft + 1 {
        return Err(KernelError::InvalidShape(format!(
            "greedy_verify: target_logits has {} rows but need {} (num_draft={} + 1)",
            num_positions,
            num_draft + 1,
            num_draft
        )));
    }

    // Read all logits to CPU once
    let data = target_logits.to_vec_checked::<f32>();

    let mut num_accepted = 0;
    let mut accepted_tokens = Vec::with_capacity(num_draft);

    // Check each draft position
    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let row_start = i * vocab_size;
        let row = &data[row_start..row_start + vocab_size];
        let target_token = argmax_cpu(row) as u32;

        if target_token == draft_tok {
            num_accepted += 1;
            accepted_tokens.push(draft_tok);
        } else {
            // Reject: correction token is target's choice at this position
            return Ok(SpecDecodeResult {
                num_accepted,
                accepted_tokens,
                correction_token: target_token,
            });
        }
    }

    // All accepted: bonus token from the last position (num_draft)
    let bonus_row_start = num_draft * vocab_size;
    let bonus_row = &data[bonus_row_start..bonus_row_start + vocab_size];
    let bonus_token = argmax_cpu(bonus_row) as u32;

    Ok(SpecDecodeResult {
        num_accepted,
        accepted_tokens,
        correction_token: bonus_token,
    })
}

/// Rejection sampling for speculative decoding (Leviathan et al., 2023).
///
/// For each draft position, accepts with probability
/// `min(1, target_prob[token] / draft_prob[token])`.
/// On rejection, samples a correction token from
/// `max(0, target_prob - draft_prob)` (the residual distribution).
/// If all draft tokens are accepted, samples a bonus token from the
/// target distribution at position `num_draft`.
///
/// `target_logits`: `[num_draft + 1, vocab_size]` f32 — raw logits from target model.
/// `draft_probs`: `[num_draft, vocab_size]` f32 — softmax probabilities from draft model.
/// `draft_tokens`: `[num_draft]` — tokens chosen by the draft model.
///
/// Uses the target model's softmax for probability computation.
pub fn rejection_sample(
    target_logits: &Array,
    draft_probs: &Array,
    draft_tokens: &[u32],
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<SpecDecodeResult, KernelError> {
    if target_logits.ndim() != 2 || draft_probs.ndim() != 2 {
        return Err(KernelError::InvalidShape(
            "rejection_sample: target_logits and draft_probs must be 2D".to_string(),
        ));
    }

    let num_draft = draft_tokens.len();
    let vocab_size = target_logits.shape()[1];

    if target_logits.shape()[0] < num_draft + 1 {
        return Err(KernelError::InvalidShape(format!(
            "rejection_sample: target_logits needs {} rows, got {}",
            num_draft + 1,
            target_logits.shape()[0]
        )));
    }
    if draft_probs.shape()[0] < num_draft || draft_probs.shape()[1] != vocab_size {
        return Err(KernelError::InvalidShape(format!(
            "rejection_sample: draft_probs shape mismatch, expected [{}, {}], got {:?}",
            num_draft,
            vocab_size,
            draft_probs.shape()
        )));
    }

    // Compute target probabilities via GPU softmax (row-wise)
    let target_probs = ops::softmax::softmax(registry, target_logits, queue)?;
    let target_data = target_probs.to_vec_checked::<f32>();
    let draft_data = draft_probs.to_vec_checked::<f32>();

    let mut num_accepted = 0;
    let mut accepted_tokens = Vec::with_capacity(num_draft);

    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let tok = draft_tok as usize;
        if tok >= vocab_size {
            return Err(KernelError::InvalidShape(format!(
                "rejection_sample: draft_token {} out of vocab range {}",
                tok, vocab_size
            )));
        }

        let t_row_start = i * vocab_size;
        let d_row_start = i * vocab_size;

        let target_p = target_data[t_row_start + tok];
        let draft_p = draft_data[d_row_start + tok];

        // Accept with probability min(1, target_p / draft_p)
        let accept_prob = if draft_p > 0.0 {
            (target_p / draft_p).min(1.0)
        } else if target_p > 0.0 {
            1.0 // draft assigned 0 prob but target likes it — accept
        } else {
            0.0 // both 0 — reject
        };

        let r = simple_random_f32();
        if r < accept_prob {
            num_accepted += 1;
            accepted_tokens.push(draft_tok);
        } else {
            // Sample correction token from residual distribution:
            // p_residual[j] = max(0, target_p[j] - draft_p[j])
            let mut residual = Vec::with_capacity(vocab_size);
            let mut residual_sum = 0.0f32;
            for j in 0..vocab_size {
                let r_j = (target_data[t_row_start + j] - draft_data[d_row_start + j]).max(0.0);
                residual.push(r_j);
                residual_sum += r_j;
            }

            let correction = if residual_sum > 0.0 {
                // Weighted sample from residual distribution
                let r2 = simple_random_f32() * residual_sum;
                let mut cumulative = 0.0f32;
                let mut chosen = vocab_size - 1;
                for (j, &rj) in residual.iter().enumerate() {
                    cumulative += rj;
                    if r2 < cumulative {
                        chosen = j;
                        break;
                    }
                }
                chosen as u32
            } else {
                // Fallback: argmax of target distribution
                let t_row = &target_data[t_row_start..t_row_start + vocab_size];
                argmax_cpu(t_row) as u32
            };

            return Ok(SpecDecodeResult {
                num_accepted,
                accepted_tokens,
                correction_token: correction,
            });
        }
    }

    // All accepted: sample bonus token from target distribution at position num_draft
    let bonus_row_start = num_draft * vocab_size;
    let bonus_row = &target_data[bonus_row_start..bonus_row_start + vocab_size];
    // Sample from target distribution (not argmax)
    let r3 = simple_random_f32();
    let mut cumulative = 0.0f32;
    let mut bonus_token = (vocab_size - 1) as u32;
    for (j, &p) in bonus_row.iter().enumerate() {
        cumulative += p;
        if r3 < cumulative {
            bonus_token = j as u32;
            break;
        }
    }

    Ok(SpecDecodeResult {
        num_accepted,
        accepted_tokens,
        correction_token: bonus_token,
    })
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
    device: &ProtocolObject<dyn MTLDevice>,
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
pub fn top_k(
    logits: &Array,
    k: usize,
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Array, KernelError> {
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
    device: &ProtocolObject<dyn MTLDevice>,
    registry: &KernelRegistry,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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
    queue: &ProtocolObject<dyn MTLCommandQueue>,
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
    device: &ProtocolObject<dyn MTLDevice>,
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
    use std::sync::OnceLock;

    fn test_device() -> &'static rmlx_metal::MtlDevice {
        static DEVICE: OnceLock<rmlx_metal::MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            objc2::rc::autoreleasepool(|_| {
                objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
            })
        })
    }

    fn setup() -> (
        &'static rmlx_metal::MtlDevice,
        rmlx_metal::MtlQueue,
        KernelRegistry,
    ) {
        let gpu = rmlx_metal::device::GpuDevice::system_default().expect("GPU device");
        let device = test_device();
        let queue = gpu.new_command_queue();
        let registry = KernelRegistry::new(gpu);
        ops::softmax::register(&registry).expect("register softmax");
        ops::register_all(&registry).unwrap_or(()); // register binary ops etc.
        (device, queue, registry)
    }

    #[test]
    fn test_temperature_zero_returns_argmax() {
        let (device, queue, registry) = setup();

        // logits: token 3 has the highest value
        let logits = Array::from_slice(device, &[1.0f32, 0.5, 2.0, 10.0, 0.1], vec![5]);

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
        let logits = Array::from_slice(device, &[1.0f32, 0.5, 10.0, 2.0, 0.1], vec![5]);

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
        let logits = Array::from_slice(device, &logits_data, vec![vocab_size]);

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

        let logits = Array::from_slice(device, &[2.0f32, 4.0, 6.0], vec![3]);
        let scaled = temperature(&logits, 2.0, device).expect("temperature");
        let data = scaled.to_vec_checked::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_masking() {
        let (device, _queue, _registry) = setup();

        let logits = Array::from_slice(device, &[1.0f32, 5.0, 3.0, 2.0, 4.0], vec![5]);
        let masked = top_k(&logits, 2, device).expect("top_k");
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

        let logits = Array::from_slice(device, &[2.0f32, 4.0, 6.0, 1.0], vec![4]);
        let penalized =
            repetition_penalty(&logits, &[1, 2], 2.0, device).expect("repetition_penalty");
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

        let logits = Array::from_slice(device, &[-2.0f32, -4.0, 1.0], vec![3]);
        let penalized =
            repetition_penalty(&logits, &[0, 1], 2.0, device).expect("repetition_penalty");
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

    #[test]
    fn test_greedy_verify_all_accepted() {
        let (device, _queue, _registry) = setup();
        // 3 draft tokens, target logits agree with all + bonus position
        // draft_tokens = [2, 0, 1]
        // target argmax at each position: 2, 0, 1, 3(bonus)
        let logits_data: Vec<f32> = vec![
            // pos 0: token 2 is max
            0.1, 0.2, 10.0, 0.3, 0.1, // pos 1: token 0 is max
            10.0, 0.2, 0.1, 0.3, 0.1, // pos 2: token 1 is max
            0.1, 10.0, 0.2, 0.3, 0.1, // pos 3 (bonus): token 3 is max
            0.1, 0.2, 0.3, 10.0, 0.1,
        ];
        let target_logits = Array::from_slice(device, &logits_data, vec![4, 5]);
        let draft_tokens = vec![2u32, 0, 1];

        let result = greedy_verify(&target_logits, &draft_tokens).expect("greedy_verify");
        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.accepted_tokens, vec![2, 0, 1]);
        assert_eq!(result.correction_token, 3); // bonus token
    }

    #[test]
    fn test_greedy_verify_partial_reject() {
        let (device, _queue, _registry) = setup();
        // draft_tokens = [2, 0, 1], but target disagrees at position 1
        let logits_data: Vec<f32> = vec![
            // pos 0: token 2 is max (agrees)
            0.1, 0.2, 10.0, 0.3, 0.1,
            // pos 1: token 3 is max (disagrees with draft token 0)
            0.1, 0.2, 0.1, 10.0, 0.1, // pos 2: doesn't matter
            0.1, 10.0, 0.2, 0.3, 0.1, // pos 3: doesn't matter
            0.1, 0.2, 0.3, 10.0, 0.1,
        ];
        let target_logits = Array::from_slice(device, &logits_data, vec![4, 5]);
        let draft_tokens = vec![2u32, 0, 1];

        let result = greedy_verify(&target_logits, &draft_tokens).expect("greedy_verify");
        assert_eq!(result.num_accepted, 1); // only first accepted
        assert_eq!(result.accepted_tokens, vec![2]);
        assert_eq!(result.correction_token, 3); // target's choice at rejected pos
    }

    #[test]
    fn test_greedy_verify_none_accepted() {
        let (device, _queue, _registry) = setup();
        // draft_tokens = [0], but target picks token 4
        let logits_data: Vec<f32> = vec![
            // pos 0: token 4 is max (disagrees with draft token 0)
            0.1, 0.2, 0.1, 0.3, 10.0, // pos 1 (bonus): doesn't matter
            10.0, 0.2, 0.3, 0.1, 0.1,
        ];
        let target_logits = Array::from_slice(device, &logits_data, vec![2, 5]);
        let draft_tokens = vec![0u32];

        let result = greedy_verify(&target_logits, &draft_tokens).expect("greedy_verify");
        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.correction_token, 4);
    }
}
