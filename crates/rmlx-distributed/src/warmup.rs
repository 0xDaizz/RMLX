//! Warmup protocol for RDMA connections and Metal JIT compilation.

use std::time::Duration;

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
}

impl Default for WarmupState {
    fn default() -> Self {
        Self::new()
    }
}
