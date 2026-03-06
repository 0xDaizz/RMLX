//! MoE expert dispatch policy: CPU/GPU/RDMA 3-zone selection.
//!
//! - CPU zone: N <= cpu_max (small tensors, avoid GPU launch overhead)
//! - Metal zone: cpu_max < N <= gpu_threshold (medium tensors, local GPU)
//! - RDMA zone: N > gpu_threshold AND world_size > 1 (large, multi-node)
//!
//! Cooldown: after backend switch, cooldown expires when EITHER 5000ms
//! elapsed OR 1000 calls (matches MLX's OR condition).
//! Hysteresis: ±hysteresis_band around thresholds to prevent oscillation.
//!
//! Thread safety: all mutable state is protected by an internal `RwLock`,
//! making `MoePolicy` `Send + Sync`. Read-heavy paths (`select`) acquire a
//! read lock; mutation paths (`switch_backend`, setters) acquire a write lock.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Cooldown duration threshold (matches MLX).
const COOLDOWN_DURATION: Duration = Duration::from_millis(5000);
/// Cooldown call count threshold (matches MLX).
const COOLDOWN_CALLS: u64 = 1000;

/// Dispatch backend for MoE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    Cpu,
    Metal,
    Rdma,
}

/// Interior mutable state protected by `RwLock`.
struct MoePolicyInner {
    /// Maximum N for CPU zone.
    cpu_max: u32,
    /// Minimum N for GPU zone (also threshold for RDMA when multi-node).
    gpu_min: u32,
    /// Byte threshold for middle zone (CPU if below, GPU if above).
    byte_threshold: usize,
    /// Hysteresis band around thresholds (elements).
    hysteresis_band: u32,
    /// World size (number of ranks). RDMA only when > 1.
    world_size: u32,
    /// Current backend.
    current_backend: MoeBackend,
    /// Timestamp of the last backend switch (for time-based cooldown).
    last_switch_time: Option<Instant>,
    /// Whether cooldown is active (a switch has occurred and not yet expired).
    cooldown_active: bool,
    /// Force a specific backend (overrides all threshold logic). Used for testing.
    forced_backend: Option<MoeBackend>,
    /// Whether Metal GPU is available on this system.
    metal_available: bool,
}

/// 3-zone dispatch policy configuration.
///
/// Thread-safe: all mutable state is behind an internal `RwLock`.
/// `MoePolicy` is `Send + Sync`.
pub struct MoePolicy {
    inner: RwLock<MoePolicyInner>,
    /// Number of `select()` calls since the last backend switch.
    calls_since_switch: AtomicU64,
    /// Step counter for hysteresis tracking.
    step_count: AtomicU32,
}

// Compile-time assertions: MoePolicy must be Send + Sync.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MoePolicy>();
};

impl MoePolicy {
    /// Create a new policy with default thresholds.
    ///
    /// Reads optional environment variable overrides:
    /// - `RMLX_MOE_BACKEND`: force "cpu", "gpu", or "auto" (default)
    /// - `RMLX_MOE_CPU_N_MAX`: override CPU max batch size threshold
    /// - `RMLX_MOE_GPU_N_MIN`: override GPU min batch size threshold
    /// - `RMLX_MOE_GPU_SWITCH_BYTES`: override byte threshold for GPU switching
    pub fn new() -> Self {
        let metal_available = metal::Device::system_default().is_some();

        let mut cpu_max: u32 = 64;
        let mut gpu_min: u32 = 320;
        let mut byte_threshold: usize = 2_097_152; // 2MB — GPU dispatch crossover (matches MLX)
        let mut forced_backend: Option<MoeBackend> = None;

        // D8: environment variable overrides for policy tuning
        if let Ok(val) = std::env::var("RMLX_MOE_BACKEND") {
            match val.to_lowercase().as_str() {
                "cpu" => forced_backend = Some(MoeBackend::Cpu),
                "gpu" | "metal" => forced_backend = Some(MoeBackend::Metal),
                "auto" | "" => {} // default
                other => {
                    eprintln!(
                        "RMLX_MOE_BACKEND: unknown value '{other}', using auto. \
                         Valid: cpu, gpu, metal, auto"
                    );
                }
            }
        }
        if let Ok(val) = std::env::var("RMLX_MOE_CPU_N_MAX") {
            if let Ok(v) = val.parse::<u32>() {
                cpu_max = v;
            }
        }
        if let Ok(val) = std::env::var("RMLX_MOE_GPU_N_MIN") {
            if let Ok(v) = val.parse::<u32>() {
                gpu_min = v;
            }
        }
        if let Ok(val) = std::env::var("RMLX_MOE_GPU_SWITCH_BYTES") {
            if let Ok(v) = val.parse::<usize>() {
                byte_threshold = v;
            }
        }

        Self {
            inner: RwLock::new(MoePolicyInner {
                cpu_max,
                gpu_min,
                byte_threshold,
                hysteresis_band: 16,
                world_size: 1,
                current_backend: MoeBackend::Metal,
                last_switch_time: None,
                cooldown_active: false,
                forced_backend,
                metal_available,
            }),
            calls_since_switch: AtomicU64::new(0),
            step_count: AtomicU32::new(0),
        }
    }

    /// Create with custom thresholds (for calibration).
    pub fn with_thresholds(cpu_max: u32, gpu_min: u32, byte_threshold: usize) -> Self {
        let base = Self::new();
        {
            let mut inner = base.inner.write().unwrap();
            inner.cpu_max = cpu_max;
            inner.gpu_min = gpu_min;
            inner.byte_threshold = byte_threshold;
        }
        base
    }

    /// Set world size (enables RDMA zone when > 1).
    pub fn set_world_size(&self, world_size: u32) {
        self.inner.write().unwrap().world_size = world_size;
    }

    /// Current world size.
    pub fn world_size(&self) -> u32 {
        self.inner.read().unwrap().world_size
    }

    /// Set hysteresis band.
    pub fn set_hysteresis_band(&self, band: u32) {
        self.inner.write().unwrap().hysteresis_band = band;
    }

    /// Force a specific backend, bypassing all threshold logic.
    /// Pass `None` to clear the override and resume normal selection.
    pub fn force_backend(&self, backend: Option<MoeBackend>) {
        self.inner.write().unwrap().forced_backend = backend;
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
        // Acquire read lock for the inner state.
        let inner = self.inner.read().unwrap();

        // If a backend is forced, return it unconditionally
        if let Some(forced) = inner.forced_backend {
            return forced;
        }

        // D9: If Metal is not available, always return CPU
        if !inner.metal_available {
            return MoeBackend::Cpu;
        }

        // D7: During cooldown, maintain current backend.
        // Cooldown expires when EITHER 5000ms elapsed OR 1000 calls.
        if inner.cooldown_active {
            let calls = self.calls_since_switch.fetch_add(1, Ordering::Acquire) + 1;
            let time_expired = inner
                .last_switch_time
                .map(|t| t.elapsed() >= COOLDOWN_DURATION)
                .unwrap_or(true);
            let calls_expired = calls >= COOLDOWN_CALLS;
            if !time_expired && !calls_expired {
                return inner.current_backend;
            }
            // Cooldown has expired — drop read lock, acquire write lock to clear it,
            // then fall through to normal selection.
            let current = inner.current_backend;
            drop(inner);
            {
                let mut w = self.inner.write().unwrap();
                w.cooldown_active = false;
            }
            // Re-acquire read lock for threshold computation below.
            let inner = self.inner.read().unwrap();
            return Self::compute_zone(&inner, n_elements, byte_size, current);
        }

        let current = inner.current_backend;
        Self::compute_zone(&inner, n_elements, byte_size, current)
    }

    /// Pure zone computation (no side effects). Extracted so both the normal
    /// and cooldown-expiry paths can share it.
    fn compute_zone(
        inner: &MoePolicyInner,
        n_elements: u32,
        byte_size: usize,
        current_backend: MoeBackend,
    ) -> MoeBackend {
        // Apply hysteresis: to leave current zone, must cross threshold + band
        let (cpu_thresh, gpu_thresh) = match current_backend {
            MoeBackend::Cpu => {
                // To leave CPU, must exceed cpu_max + band
                (
                    inner.cpu_max + inner.hysteresis_band,
                    inner.gpu_min + inner.hysteresis_band,
                )
            }
            MoeBackend::Metal => {
                // To drop to CPU, must go below cpu_max - band
                // To jump to RDMA, must exceed gpu_min + band
                (
                    inner.cpu_max.saturating_sub(inner.hysteresis_band),
                    inner.gpu_min + inner.hysteresis_band,
                )
            }
            MoeBackend::Rdma => {
                // To leave RDMA, must drop below gpu_min - band
                (
                    inner.cpu_max.saturating_sub(inner.hysteresis_band),
                    inner.gpu_min.saturating_sub(inner.hysteresis_band),
                )
            }
        };

        if n_elements <= cpu_thresh {
            MoeBackend::Cpu
        } else if n_elements > gpu_thresh && inner.world_size > 1 {
            // RDMA zone: large messages AND multi-node
            MoeBackend::Rdma
        } else if n_elements >= gpu_thresh {
            // Large but single-node → Metal
            MoeBackend::Metal
        } else {
            // Middle zone: byte threshold decides CPU vs Metal
            if byte_size < inner.byte_threshold {
                MoeBackend::Cpu
            } else {
                MoeBackend::Metal
            }
        }
    }

    /// Update backend with cooldown trigger.
    ///
    /// D7: resets both time and call counters. Cooldown expires when EITHER
    /// 5000ms elapsed OR 1000 calls have been made to `select()`.
    pub fn switch_backend(&self, new_backend: MoeBackend) {
        let mut inner = self.inner.write().unwrap();
        if new_backend != inner.current_backend {
            inner.current_backend = new_backend;
            inner.last_switch_time = Some(Instant::now());
            self.calls_since_switch.store(0, Ordering::Release);
            inner.cooldown_active = true;
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
        self.inner.read().unwrap().current_backend
    }

    /// Whether cooldown is currently active.
    pub fn cooldown_active(&self) -> bool {
        self.inner.read().unwrap().cooldown_active
    }

    /// Calls since the last backend switch.
    pub fn calls_since_switch(&self) -> u64 {
        self.calls_since_switch.load(Ordering::Relaxed)
    }

    /// CPU max threshold.
    pub fn cpu_max(&self) -> u32 {
        self.inner.read().unwrap().cpu_max
    }

    /// GPU min threshold.
    pub fn gpu_min(&self) -> u32 {
        self.inner.read().unwrap().gpu_min
    }

    /// Byte threshold.
    pub fn byte_threshold(&self) -> usize {
        self.inner.read().unwrap().byte_threshold
    }

    /// Whether Metal is available on this system.
    pub fn metal_available(&self) -> bool {
        self.inner.read().unwrap().metal_available
    }

    /// Set CPU max threshold (used by calibration).
    pub fn set_cpu_max(&self, v: u32) {
        self.inner.write().unwrap().cpu_max = v;
    }

    /// Set GPU min threshold (used by calibration).
    pub fn set_gpu_min(&self, v: u32) {
        self.inner.write().unwrap().gpu_min = v;
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
    pub fn apply_to(&self, policy: &MoePolicy) {
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
