//! Warmup protocol for RDMA connections and Metal JIT compilation.
//!
//! D18: ThresholdCalibration is wired into run_warmup so that the system
//! auto-tunes the CPU/GPU crossover point at startup.
//! D19: run_warmup is idempotent — calling it when already ready is a no-op.

use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use rmlx_metal::metal::{Device as MTLDevice, NSRange};

use crate::moe_policy::ThresholdCalibration;

const GPU_PROXY_DIM: usize = 1024;
const GPU_PROXY_COPY_ITERS: usize = 4;
const MEMORY_BANDWIDTH_BYTES: usize = 16 * 1024 * 1024;

// Estimated fallback when Metal is unavailable and no GPU measurement can be taken.
const ESTIMATED_GPU_MATMUL_GFLOPS: f64 = 256.0;
// Estimated fallback when the measured host-memory pass underflows the timer resolution.
const ESTIMATED_MEMORY_BANDWIDTH_GBPS: f64 = 12.0;
// Estimated fallback because warmup has no peer connection for a real RDMA ping-pong.
const ESTIMATED_RDMA_LATENCY_US: f64 = 50.0;

/// Measured warmup benchmark data.
#[derive(Debug, Clone)]
pub struct WarmupBenchResult {
    /// Approximate GPU matmul throughput derived from a Metal blit-copy proxy.
    pub gpu_matmul_gflops: f64,
    /// RDMA round-trip latency in microseconds.
    pub rdma_latency_us: f64,
    /// Sequential host-memory read bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
}

/// Warmup result tracking.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub rdma_warmup: Duration,
    pub jit_warmup: Duration,
    /// D18: Time spent on threshold calibration.
    pub calibration: Duration,
    /// Measured or estimated warmup benchmark data.
    pub bench: Option<WarmupBenchResult>,
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
    /// Measured or estimated warmup benchmark data from the last run.
    bench_result: Option<WarmupBenchResult>,
    /// D18: Threshold calibration state, wired into warmup.
    calibration: ThresholdCalibration,
}

impl WarmupState {
    pub fn new() -> Self {
        Self {
            rdma_warmed: false,
            jit_warmed: false,
            last_result: None,
            bench_result: None,
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
        self.bench_result = result.bench.clone();
        self.last_result = Some(result);
    }

    /// Last warmup result.
    pub fn last_result(&self) -> Option<&WarmupResult> {
        self.last_result.as_ref()
    }

    /// Last warmup benchmark result.
    pub fn bench_result(&self) -> Option<&WarmupBenchResult> {
        self.bench_result.as_ref()
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
            if let Some(result) = self.last_result.clone() {
                self.bench_result = result.bench.clone();
                return Ok(result);
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
            let calibration_gpu = MTLDevice::system_default();
            let cal_start = Instant::now();
            self.calibration.calibrate(
                |n| run_cpu_calibration_bench(n),
                |n| run_gpu_calibration_bench(calibration_gpu.as_ref(), n),
            );
            cal_dur = cal_start.elapsed();
        }

        let bench = Some(run_warmup_bench(None));
        self.bench_result = bench.clone();

        let result = WarmupResult {
            rdma_warmup: rdma_dur,
            jit_warmup: jit_dur,
            calibration: cal_dur,
            bench,
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

/// Run the warmup benchmark suite and return measured or estimated values.
pub fn run_warmup_bench(device: Option<&rmlx_metal::GpuDevice>) -> WarmupBenchResult {
    let owned_gpu = if device.is_none() {
        MTLDevice::system_default()
    } else {
        None
    };
    let metal_device = device.map(|gpu| gpu.raw()).or_else(|| owned_gpu.as_ref());

    WarmupBenchResult {
        gpu_matmul_gflops: run_gpu_matmul_proxy_bench(metal_device, GPU_PROXY_DIM),
        rdma_latency_us: run_rdma_latency_bench(),
        memory_bandwidth_gbps: run_memory_bandwidth_bench(),
    }
}

fn run_cpu_calibration_bench(n: usize) -> f64 {
    let len = n.saturating_mul(n).max(1);
    let lhs = vec![1.001f32; len];
    let rhs = vec![0.999f32; len];
    let mut acc = vec![0.0f32; len];
    let passes = (n / 32).max(1);

    let start = Instant::now();
    for _ in 0..passes {
        for i in 0..len {
            acc[i] = lhs[i].mul_add(rhs[i], acc[i]);
        }
    }
    black_box(acc[len / 2]);
    start.elapsed().as_secs_f64().max(f64::EPSILON)
}

fn run_gpu_calibration_bench(device: Option<&MTLDevice>, n: usize) -> f64 {
    let dim = n.max(1);
    if let Some(device) = device {
        if let Some(seconds) = timed_gpu_copy_seconds(device, dim, 1) {
            return seconds;
        }
    }

    estimated_gpu_seconds(dim)
}

fn run_gpu_matmul_proxy_bench(device: Option<&MTLDevice>, dim: usize) -> f64 {
    let dim = dim.max(1);
    if let Some(device) = device {
        if let Some(seconds) = timed_gpu_copy_seconds(device, dim, GPU_PROXY_COPY_ITERS) {
            return approx_matmul_gflops(dim, seconds, GPU_PROXY_COPY_ITERS);
        }
    }

    // Estimated fallback when Metal is unavailable or the copy benchmark cannot run.
    ESTIMATED_GPU_MATMUL_GFLOPS
}

fn timed_gpu_copy_seconds(device: &MTLDevice, dim: usize, iterations: usize) -> Option<f64> {
    let element_count = dim.saturating_mul(dim).max(1);
    let byte_len = element_count
        .saturating_mul(size_of::<f32>())
        .max(size_of::<f32>());
    let range = NSRange::new(0, byte_len as u64);
    let src = device.new_buffer(byte_len as u64, rmlx_metal::DEFAULT_BUFFER_OPTIONS);
    let dst = device.new_buffer(byte_len as u64, rmlx_metal::DEFAULT_BUFFER_OPTIONS);
    let queue = device.new_command_queue();

    let init_cb = queue.new_command_buffer();
    let init_blit = init_cb.new_blit_command_encoder();
    init_blit.fill_buffer(&src, range, 0x11);
    init_blit.fill_buffer(&dst, range, 0x00);
    init_blit.end_encoding();
    init_cb.commit();
    init_cb.wait_until_completed();

    let start = Instant::now();
    for _ in 0..iterations.max(1) {
        let cb = queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(&src, 0, &dst, 0, byte_len as u64);
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let seconds = start.elapsed().as_secs_f64();

    (seconds > 0.0).then_some(seconds)
}

fn run_memory_bandwidth_bench() -> f64 {
    let len = MEMORY_BANDWIDTH_BYTES / size_of::<f32>();
    let data: Vec<f32> = (0..len).map(|i| ((i % 251) as f32) * 0.25).collect();
    let slice = black_box(data.as_slice());

    let start = Instant::now();
    let mut sum = 0.0f32;
    for &value in slice {
        sum += value;
    }
    black_box(sum);

    let seconds = start.elapsed().as_secs_f64();
    if seconds > 0.0 {
        (data.len() * size_of::<f32>()) as f64 / seconds / 1e9
    } else {
        // Estimated fallback when the measured pass is smaller than the timer resolution.
        ESTIMATED_MEMORY_BANDWIDTH_GBPS
    }
}

fn run_rdma_latency_bench() -> f64 {
    if !rmlx_rdma::is_available() {
        // Estimated fallback: RDMA hardware or libraries are unavailable on this host.
        return ESTIMATED_RDMA_LATENCY_US;
    }

    // Estimated fallback: RDMA is present, but warmup has no peer connection for a real ping-pong.
    ESTIMATED_RDMA_LATENCY_US
}

fn approx_matmul_gflops(dim: usize, seconds: f64, iterations: usize) -> f64 {
    let flops = 2.0 * (dim as f64).powi(3) * iterations as f64;
    flops / seconds.max(f64::EPSILON) / 1e9
}

fn estimated_gpu_seconds(dim: usize) -> f64 {
    let flops = 2.0 * (dim as f64).powi(3);
    flops / (ESTIMATED_GPU_MATMUL_GFLOPS * 1e9)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn test_warmup_bench_runs() {
        let result = run_warmup_bench(None);
        assert!(result.gpu_matmul_gflops > 0.0);
        assert!(result.memory_bandwidth_gbps > 0.0);
        assert!(result.rdma_latency_us > 0.0);
    }

    #[test]
    fn test_warmup_state_full() {
        let mut state = WarmupState::new();
        let config = WarmupConfig::default();

        let result = state.run_warmup(&config, || Ok(()), || Ok(()));

        assert!(result.is_ok());
        assert!(state.bench_result().is_some());
        assert!(result.unwrap().bench.is_some());
    }

    #[test]
    fn test_warmup_idempotent() {
        let mut state = WarmupState::new();
        let config = WarmupConfig::default();
        let rdma_calls = Cell::new(0usize);
        let jit_calls = Cell::new(0usize);

        let first = state
            .run_warmup(
                &config,
                || {
                    rdma_calls.set(rdma_calls.get() + 1);
                    Ok(())
                },
                || {
                    jit_calls.set(jit_calls.get() + 1);
                    Ok(())
                },
            )
            .unwrap();

        let second = state
            .run_warmup(
                &config,
                || {
                    rdma_calls.set(rdma_calls.get() + 1);
                    Ok(())
                },
                || {
                    jit_calls.set(jit_calls.get() + 1);
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(rdma_calls.get(), 1);
        assert_eq!(jit_calls.get(), 1);
        assert_eq!(first.total, second.total);
        assert!(state.bench_result().is_some());
    }
}
