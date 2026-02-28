//! Precision guard — detects NaN/Inf and entropy drift in logits.

/// Result of checking a logits slice for numerical issues.
#[derive(Debug, PartialEq)]
pub enum PrecisionResult {
    Ok,
    HasNaN(usize),
    HasInf(usize),
    EntropyDrift(f64),
}

/// Monitors logits for NaN, Inf, and entropy drift over time.
pub struct PrecisionGuard {
    nan_count: u64,
    inf_count: u64,
    entropy_history: Vec<f64>,
    baseline_entropy: Option<f64>,
    window_size: usize,
    consecutive_drift_windows: usize,
}

impl PrecisionGuard {
    pub fn new(window_size: usize) -> Self {
        Self {
            nan_count: 0,
            inf_count: 0,
            entropy_history: Vec::new(),
            baseline_entropy: None,
            window_size: window_size.max(1),
            consecutive_drift_windows: 0,
        }
    }

    /// Check a logits slice for NaN/Inf values and compute entropy.
    pub fn check_logits(&mut self, logits: &[f32]) -> PrecisionResult {
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;

        for &v in logits {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            }
        }

        if nan_count > 0 {
            self.nan_count += nan_count as u64;
            return PrecisionResult::HasNaN(nan_count);
        }
        if inf_count > 0 {
            self.inf_count += inf_count as u64;
            return PrecisionResult::HasInf(inf_count);
        }

        // Compute entropy from logits via softmax
        if !logits.is_empty() {
            let entropy = compute_entropy(logits);
            self.record_entropy(entropy);

            if let Some(drift) = self.entropy_drift() {
                if drift > 0.30 {
                    return PrecisionResult::EntropyDrift(drift);
                }
            }
        }

        PrecisionResult::Ok
    }

    /// Record an entropy observation.
    pub fn record_entropy(&mut self, entropy: f64) {
        self.entropy_history.push(entropy);

        // Set baseline from the first full window
        if self.baseline_entropy.is_none() && self.entropy_history.len() >= self.window_size {
            let sum: f64 = self.entropy_history[..self.window_size].iter().sum();
            self.baseline_entropy = Some(sum / self.window_size as f64);
        }

        // Evaluate drift at each window boundary
        if self.entropy_history.len() % self.window_size == 0 && self.baseline_entropy.is_some() {
            if let Some(drift) = self.entropy_drift() {
                if drift > 0.30 {
                    self.consecutive_drift_windows += 1;
                } else {
                    self.consecutive_drift_windows = 0;
                }
            }
        }
    }

    /// Compute the relative entropy drift from baseline.
    /// Returns `None` if baseline is not yet established.
    pub fn entropy_drift(&self) -> Option<f64> {
        let baseline = self.baseline_entropy?;
        if baseline == 0.0 {
            return Some(0.0);
        }
        let n = self.entropy_history.len();
        if n < self.window_size {
            return None;
        }
        let recent_start = n.saturating_sub(self.window_size);
        let recent_sum: f64 = self.entropy_history[recent_start..].iter().sum();
        let recent_avg = recent_sum / self.window_size as f64;
        Some(((recent_avg - baseline) / baseline).abs())
    }

    /// Whether the drift level warrants a warning.
    pub fn should_warn(&self) -> bool {
        self.entropy_drift().is_some_and(|d| d > 0.15)
    }

    /// Whether to fall back to higher precision (drift > 0.30 for 2+ windows).
    pub fn should_fallback(&self) -> bool {
        self.consecutive_drift_windows >= 2
    }
}

/// Compute Shannon entropy from logits via softmax.
fn compute_entropy(logits: &[f32]) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&x| ((x - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for e in &exps {
        let p = e / sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}
