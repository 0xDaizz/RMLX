//! MoE expert dispatch policy: CPU/GPU/RDMA 3-zone selection.
//!
//! - CPU zone: N <= cpu_max (small tensors, avoid GPU launch overhead)
//! - Metal zone: cpu_max < N <= gpu_threshold (medium tensors, local GPU)
//! - RDMA zone: N > gpu_threshold AND world_size > 1 (large, multi-node)
//!
//! Cooldown: after backend switch, 32 steps before re-switching.
//! Hysteresis: ±hysteresis_band around thresholds to prevent oscillation.

use std::sync::atomic::{AtomicU32, Ordering};

/// Dispatch backend for MoE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    Cpu,
    Metal,
    Rdma,
}

/// 3-zone dispatch policy configuration.
pub struct MoePolicy {
    /// Maximum N for CPU zone.
    cpu_max: u32,
    /// Minimum N for GPU zone (also threshold for RDMA when multi-node).
    gpu_min: u32,
    /// Byte threshold for middle zone (CPU if below, GPU if above).
    byte_threshold: usize,
    /// Cooldown steps after backend switch (prevents oscillation).
    cooldown_steps: u32,
    /// Hysteresis band around thresholds (elements).
    hysteresis_band: u32,
    /// World size (number of ranks). RDMA only when > 1.
    world_size: u32,
    /// Current backend.
    current_backend: MoeBackend,
    /// Steps remaining in cooldown.
    cooldown_remaining: AtomicU32,
    /// Step counter for hysteresis tracking.
    step_count: AtomicU32,
}

impl MoePolicy {
    /// Create a new policy with default thresholds.
    pub fn new() -> Self {
        Self {
            cpu_max: 64,
            gpu_min: 320,
            byte_threshold: 4096, // 4KB
            cooldown_steps: 32,
            hysteresis_band: 16,
            world_size: 1,
            current_backend: MoeBackend::Metal,
            cooldown_remaining: AtomicU32::new(0),
            step_count: AtomicU32::new(0),
        }
    }

    /// Create with custom thresholds (for calibration).
    pub fn with_thresholds(cpu_max: u32, gpu_min: u32, byte_threshold: usize) -> Self {
        Self {
            cpu_max,
            gpu_min,
            byte_threshold,
            ..Self::new()
        }
    }

    /// Set world size (enables RDMA zone when > 1).
    pub fn set_world_size(&mut self, world_size: u32) {
        self.world_size = world_size;
    }

    /// Current world size.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Set hysteresis band.
    pub fn set_hysteresis_band(&mut self, band: u32) {
        self.hysteresis_band = band;
    }

    /// Select dispatch backend for a given element count and byte size.
    ///
    /// 3-zone logic:
    /// - CPU: N <= cpu_max (small messages)
    /// - Metal: cpu_max < N <= gpu_min (medium, local GPU)
    /// - RDMA: N > gpu_min AND world_size > 1 (large, multi-node)
    ///
    /// Hysteresis: when switching away from the current backend, the threshold
    /// is adjusted by ±hysteresis_band to prevent oscillation at zone boundaries.
    pub fn select(&self, n_elements: u32, byte_size: usize) -> MoeBackend {
        // During cooldown, maintain current backend
        let remaining = self.cooldown_remaining.load(Ordering::Relaxed);
        if remaining > 0 {
            self.cooldown_remaining.fetch_sub(1, Ordering::Relaxed);
            return self.current_backend;
        }

        // Apply hysteresis: to leave current zone, must cross threshold + band
        let (cpu_thresh, gpu_thresh) = match self.current_backend {
            MoeBackend::Cpu => {
                // To leave CPU, must exceed cpu_max + band
                (
                    self.cpu_max + self.hysteresis_band,
                    self.gpu_min + self.hysteresis_band,
                )
            }
            MoeBackend::Metal => {
                // To drop to CPU, must go below cpu_max - band
                // To jump to RDMA, must exceed gpu_min + band
                (
                    self.cpu_max.saturating_sub(self.hysteresis_band),
                    self.gpu_min + self.hysteresis_band,
                )
            }
            MoeBackend::Rdma => {
                // To leave RDMA, must drop below gpu_min - band
                (
                    self.cpu_max.saturating_sub(self.hysteresis_band),
                    self.gpu_min.saturating_sub(self.hysteresis_band),
                )
            }
        };

        if n_elements <= cpu_thresh {
            MoeBackend::Cpu
        } else if n_elements > gpu_thresh && self.world_size > 1 {
            // RDMA zone: large messages AND multi-node
            MoeBackend::Rdma
        } else if n_elements >= self.gpu_min {
            // Large but single-node → Metal
            MoeBackend::Metal
        } else {
            // Middle zone: byte threshold decides CPU vs Metal
            if byte_size < self.byte_threshold {
                MoeBackend::Cpu
            } else {
                MoeBackend::Metal
            }
        }
    }

    /// Update backend with cooldown trigger.
    pub fn switch_backend(&mut self, new_backend: MoeBackend) {
        if new_backend != self.current_backend {
            self.current_backend = new_backend;
            self.cooldown_remaining
                .store(self.cooldown_steps, Ordering::Relaxed);
        }
    }

    /// Record a step.
    pub fn step(&self) {
        self.step_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Current step count.
    pub fn step_count(&self) -> u32 {
        self.step_count.load(Ordering::Relaxed)
    }

    /// Current backend.
    pub fn current_backend(&self) -> MoeBackend {
        self.current_backend
    }

    /// Cooldown remaining.
    pub fn cooldown_remaining(&self) -> u32 {
        self.cooldown_remaining.load(Ordering::Relaxed)
    }

    /// CPU max threshold.
    pub fn cpu_max(&self) -> u32 {
        self.cpu_max
    }

    /// GPU min threshold.
    pub fn gpu_min(&self) -> u32 {
        self.gpu_min
    }

    /// Set CPU max threshold (used by calibration).
    pub fn set_cpu_max(&mut self, v: u32) {
        self.cpu_max = v;
    }

    /// Set GPU min threshold (used by calibration).
    pub fn set_gpu_min(&mut self, v: u32) {
        self.gpu_min = v;
    }
}

impl Default for MoePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Startup threshold calibration for CPU vs GPU crossover.
///
/// Measures at startup to find the crossover point N* where GPU_time <= 0.95 * CPU_time.
/// This avoids hard-coding thresholds that may not match the hardware.
pub struct ThresholdCalibration {
    /// Maximum N for CPU zone after calibration.
    pub cpu_max: usize,
    /// Minimum N for GPU zone after calibration.
    pub gpu_min: usize,
    /// Measured crossover N (where GPU becomes faster).
    pub crossover_n: usize,
    /// Whether calibration has been performed.
    pub calibrated: bool,
}

impl ThresholdCalibration {
    pub fn new() -> Self {
        Self {
            cpu_max: 64,
            gpu_min: 320,
            crossover_n: 0,
            calibrated: false,
        }
    }

    /// Run calibration with user-supplied benchmark functions.
    ///
    /// `cpu_bench`: given N, returns CPU execution time in seconds.
    /// `gpu_bench`: given N, returns GPU execution time in seconds.
    ///
    /// Tests N in {32, 64, 128, 256, 384} and finds the crossover point.
    pub fn calibrate(
        &mut self,
        cpu_bench: impl Fn(usize) -> f64,
        gpu_bench: impl Fn(usize) -> f64,
    ) {
        let test_sizes = [32, 64, 128, 256, 384];
        let mut crossover = 0usize;

        for &n in &test_sizes {
            let cpu_time = cpu_bench(n);
            let gpu_time = gpu_bench(n);

            // GPU is faster when gpu_time <= 0.95 * cpu_time
            if gpu_time <= 0.95 * cpu_time {
                crossover = n;
                break;
            }
        }

        if crossover > 0 {
            self.crossover_n = crossover;
            self.cpu_max = 64usize.max(crossover.saturating_sub(32));
            self.gpu_min = 320usize.min(crossover + 32);
        }

        self.calibrated = true;
    }

    /// Apply calibration results to a MoePolicy.
    pub fn apply_to(&self, policy: &mut MoePolicy) {
        if self.calibrated {
            policy.set_cpu_max(self.cpu_max as u32);
            policy.set_gpu_min(self.gpu_min as u32);
        }
    }
}

impl Default for ThresholdCalibration {
    fn default() -> Self {
        Self::new()
    }
}
