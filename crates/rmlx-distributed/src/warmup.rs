//! Warmup protocol for RDMA connections and Metal JIT compilation.

use std::time::{Duration, Instant};

/// Warmup result tracking.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub rdma_warmup: Duration,
    pub jit_warmup: Duration,
    pub total: Duration,
}

/// Warmup configuration.
pub struct WarmupConfig {
    /// Number of RDMA warmup rounds.
    pub rdma_rounds: usize,
    /// Whether to pre-compile JIT kernels.
    pub jit_precompile: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            rdma_rounds: 10,
            jit_precompile: true,
        }
    }
}

/// Pre-warmup marker: tracks whether warmup has been run.
pub struct WarmupState {
    rdma_warmed: bool,
    jit_warmed: bool,
    last_result: Option<WarmupResult>,
}

impl WarmupState {
    pub fn new() -> Self {
        Self {
            rdma_warmed: false,
            jit_warmed: false,
            last_result: None,
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

    /// Run full warmup: RDMA ping-pong + JIT shader pre-compilation.
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

        let result = WarmupResult {
            rdma_warmup: rdma_dur,
            jit_warmup: jit_dur,
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
