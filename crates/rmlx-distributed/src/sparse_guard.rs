//! Sparse expert overflow guard — monitors overflow and triggers capacity adjustments.

/// Action recommended by the SparseGuard after evaluation.
#[derive(Debug, PartialEq)]
pub enum GuardAction {
    /// No action needed.
    None,
    /// Increase expert capacity by this factor.
    IncreaseCapacity(f64),
    /// Fall back to dense computation.
    DenseFallback,
    /// Reset to normal after recovery.
    Reset,
}

/// Monitors overflow ratio and recommends capacity adjustments or dense fallback.
pub struct SparseGuard {
    overflow_ema: f64,
    ema_alpha: f64,
    capacity_factor: f64,
    dense_fallback: bool,
    window_size: usize,
    step_count: usize,
    overflow_count_window: usize,
    total_count_window: usize,
}

impl SparseGuard {
    pub fn new() -> Self {
        Self {
            overflow_ema: 0.0,
            ema_alpha: 0.1,
            capacity_factor: 1.0,
            dense_fallback: false,
            window_size: 100,
            step_count: 0,
            overflow_count_window: 0,
            total_count_window: 0,
        }
    }

    /// Record a step with observed overflow/total counts.
    pub fn record_step(&mut self, overflow_count: usize, total_count: usize) {
        self.overflow_count_window += overflow_count;
        self.total_count_window += total_count;
        self.step_count += 1;
    }

    /// Whether the overflow EMA suggests capacity should be increased.
    pub fn should_increase_capacity(&self) -> bool {
        self.overflow_ema > 0.05
    }

    /// Whether the overflow EMA warrants dense fallback.
    pub fn should_dense_fallback(&self) -> bool {
        self.overflow_ema > 0.20
    }

    /// Current capacity factor.
    pub fn capacity_factor(&self) -> f64 {
        self.capacity_factor
    }

    /// Whether dense fallback is currently active.
    pub fn is_dense_fallback(&self) -> bool {
        self.dense_fallback
    }

    /// Current overflow EMA value (for testing/debugging).
    pub fn overflow_ema(&self) -> f64 {
        self.overflow_ema
    }

    /// Evaluate at the end of a window and return the recommended action.
    pub fn evaluate(&mut self) -> GuardAction {
        if self.step_count < self.window_size {
            return GuardAction::None;
        }

        // Compute window overflow ratio
        let ratio = if self.total_count_window > 0 {
            self.overflow_count_window as f64 / self.total_count_window as f64
        } else {
            0.0
        };

        // Update EMA
        self.overflow_ema = self.ema_alpha * ratio + (1.0 - self.ema_alpha) * self.overflow_ema;

        // Reset window
        self.step_count = 0;
        self.overflow_count_window = 0;
        self.total_count_window = 0;

        // Decide action
        if self.dense_fallback {
            if self.overflow_ema <= 0.05 {
                self.dense_fallback = false;
                self.capacity_factor = 1.0;
                return GuardAction::Reset;
            }
            return GuardAction::None;
        }

        if self.overflow_ema > 0.20 {
            self.dense_fallback = true;
            return GuardAction::DenseFallback;
        }

        if self.overflow_ema > 0.05 {
            let new_factor = (self.capacity_factor * 1.25).min(2.0);
            self.capacity_factor = new_factor;
            return GuardAction::IncreaseCapacity(new_factor);
        }

        GuardAction::None
    }
}

impl Default for SparseGuard {
    fn default() -> Self {
        Self::new()
    }
}
