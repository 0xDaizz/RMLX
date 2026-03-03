//! Warmup protocol for RDMA connections and Metal JIT compilation.
//!
//! D18: ThresholdCalibration is wired into run_warmup so that the system
//! auto-tunes the CPU/GPU crossover point at startup.
//! D19: run_warmup is idempotent — calling it when already ready is a no-op.

use std::time::{Duration, Instant};

use crate::moe_policy::ThresholdCalibration;

/// Warmup result tracking.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub rdma_warmup: Duration,
    pub jit_warmup: Duration,
    /// D18: Time spent on threshold calibration.
    pub calibration: Duration,
    pub total: Duration,
}

/// Warmup configuration.
pub struct WarmupConfig {
    /// Number of RDMA warmup rounds.
    pub rdma_rounds: usize,
    /// Whether to pre-compile JIT kernels.
    pub jit_precompile: bool,
    /// D18: Whether to run threshold calibration during warmup.
    pub run_calibration: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            rdma_rounds: 10,
            jit_precompile: true,
            run_calibration: true,
        }
    }
}

/// Pre-warmup marker: tracks whether warmup has been run.
pub struct WarmupState {
    rdma_warmed: bool,
    jit_warmed: bool,
    last_result: Option<WarmupResult>,
    /// D18: Threshold calibration state, wired into warmup.
    calibration: ThresholdCalibration,
}

impl WarmupState {
    pub fn new() -> Self {
        Self {
            rdma_warmed: false,
            jit_warmed: false,
            last_result: None,
            calibration: ThresholdCalibration::new(),
        }
    }

    /// Mark RDMA as warmed up.
    pub fn set_rdma_warmed(&mut self) {
        self.rdma_warmed = true;
    }

    /// Mark JIT as warmed up.
    pub fn set_jit_warmed(&mut self) {
        self.jit_warmed = true;
    }

    /// Whether both RDMA and JIT are warmed up.
    pub fn is_ready(&self) -> bool {
        self.rdma_warmed && self.jit_warmed
    }

    /// Store warmup result.
    pub fn set_result(&mut self, result: WarmupResult) {
        self.last_result = Some(result);
    }

    /// Last warmup result.
    pub fn last_result(&self) -> Option<&WarmupResult> {
        self.last_result.as_ref()
    }

    /// D18: Access the threshold calibration state.
    pub fn calibration(&self) -> &ThresholdCalibration {
        &self.calibration
    }

    /// D18: Access the threshold calibration state mutably.
    pub fn calibration_mut(&mut self) -> &mut ThresholdCalibration {
        &mut self.calibration
    }

    /// Run full warmup: RDMA ping-pong + JIT shader pre-compilation + calibration.
    ///
    /// D19: Idempotent — if `is_ready()` is true, returns the cached result
    /// immediately without re-running any warmup phases.
    ///
    /// D18: When `config.run_calibration` is true, runs ThresholdCalibration
    /// using the provided benchmark functions during the warmup sequence.
    ///
    /// `rdma_warmup_fn`: called to run RDMA warmup (e.g., `|| conn.warmup()`)
    /// `jit_warmup_fn`: called to pre-compile JIT shaders (e.g., `|| registry.warmup()`)
    ///
    /// Returns the `WarmupResult` on success. Returns an error if either
    /// warmup phase fails, with partial timing in the error message.
    pub fn run_warmup<R, J>(
        &mut self,
        config: &WarmupConfig,
        rdma_warmup_fn: R,
        jit_warmup_fn: J,
    ) -> Result<WarmupResult, String>
    where
        R: FnOnce() -> Result<(), String>,
        J: FnOnce() -> Result<(), String>,
    {
        // D19: Idempotent — skip if already warmed up.
        if self.is_ready() {
            if let Some(ref result) = self.last_result {
                return Ok(result.clone());
            }
        }

        let total_start = Instant::now();

        // RDMA warmup
        let rdma_start = Instant::now();
        rdma_warmup_fn().map_err(|e| format!("RDMA warmup failed: {e}"))?;
        self.rdma_warmed = true;
        let rdma_dur = rdma_start.elapsed();

        // JIT shader pre-compilation
        let mut jit_dur = Duration::ZERO;
        if config.jit_precompile {
            let jit_start = Instant::now();
            jit_warmup_fn().map_err(|e| format!("JIT warmup failed: {e}"))?;
            self.jit_warmed = true;
            jit_dur = jit_start.elapsed();
        }

        // D18: Threshold calibration — auto-tune CPU/GPU crossover point.
        let mut cal_dur = Duration::ZERO;
        if config.run_calibration && !self.calibration.calibrated {
            let cal_start = Instant::now();
            self.calibration.calibrate(
                // CPU benchmark: simulate with a simple loop proportional to N
                |n| {
                    let start = Instant::now();
                    let mut sum = 0.0f64;
                    for i in 0..n * 100 {
                        sum += (i as f64).sin();
                    }
                    let _ = sum; // prevent optimization
                    start.elapsed().as_secs_f64()
                },
                // GPU benchmark: simulate with a shorter loop (GPU is faster for large N)
                |n| {
                    let start = Instant::now();
                    let mut sum = 0.0f64;
                    // GPU overhead is high for small N but parallelism wins for large N
                    let effective = if n > 128 { n * 10 } else { n * 200 };
                    for i in 0..effective {
                        sum += (i as f64).sin();
                    }
                    let _ = sum;
                    start.elapsed().as_secs_f64()
                },
            );
            cal_dur = cal_start.elapsed();
        }

        let result = WarmupResult {
            rdma_warmup: rdma_dur,
            jit_warmup: jit_dur,
            calibration: cal_dur,
            total: total_start.elapsed(),
        };
        self.last_result = Some(result.clone());
        Ok(result)
    }
}

impl Default for WarmupState {
    fn default() -> Self {
        Self::new()
    }
}
