//! MoE expert dispatch policy: CPU/GPU/RDMA 3-zone selection.
//!
//! - CPU zone: N <= cpu_max (small tensors, avoid GPU launch overhead)
//! - GPU zone: N >= gpu_min (large tensors, GPU-efficient)
//! - Byte threshold zone: cpu_max < N < gpu_min (decided by byte size)
//!
//! Cooldown: after backend switch, 32 steps before re-switching.

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
    /// Minimum N for GPU zone.
    gpu_min: u32,
    /// Byte threshold for middle zone (CPU if below, GPU if above).
    byte_threshold: usize,
    /// Cooldown steps after backend switch (prevents oscillation).
    cooldown_steps: u32,
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

    /// Select dispatch backend for a given element count and byte size.
    pub fn select(&self, n_elements: u32, byte_size: usize) -> MoeBackend {
        // During cooldown, maintain current backend
        let remaining = self.cooldown_remaining.load(Ordering::Relaxed);
        if remaining > 0 {
            self.cooldown_remaining.fetch_sub(1, Ordering::Relaxed);
            return self.current_backend;
        }

        if n_elements <= self.cpu_max {
            MoeBackend::Cpu
        } else if n_elements >= self.gpu_min {
            MoeBackend::Metal
        } else {
            // Middle zone: byte threshold
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
}

impl Default for MoePolicy {
    fn default() -> Self {
        Self::new()
    }
}
