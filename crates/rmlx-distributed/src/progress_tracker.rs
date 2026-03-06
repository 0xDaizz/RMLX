//! ProgressTracker — wires ProgressEngine into EP dispatch with health monitoring.
//!
//! Wraps [`rmlx_rdma::progress::ProgressEngine`] and adds:
//! - Tracking of async RDMA operations from EP dispatch/combine
//! - Consecutive-error threshold: if N consecutive `poll()` calls return errors,
//!   sets a health warning flag and logs a warning
//! - Bulk poll helpers for `ZeroCopyPendingOp` collections

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rmlx_rdma::progress::{Completion, OpError, PendingOp, ProgressEngine};

use crate::group::DistributedError;
use crate::transport::ZeroCopyPendingOp;

/// Default consecutive error threshold before escalating to a health warning.
const DEFAULT_ERROR_THRESHOLD: u32 = 5;

/// Tracks async RDMA operations from EP dispatch/combine and monitors health.
///
/// The tracker wraps a shared `ProgressEngine` and adds a consecutive-error
/// counter. When the counter reaches the configured threshold, a warning is
/// logged and the `health_warning` flag is set.
///
/// # Usage
///
/// ```ignore
/// let engine = Arc::new(ProgressEngine::new());
/// let tracker = ProgressTracker::new(engine);
///
/// // After launching async RDMA ops, register and poll through the tracker
/// let pending = tracker.register_op(wr_id);
/// // ... post RDMA work request ...
/// let result = tracker.poll_op(&pending)?;
/// ```
pub struct ProgressTracker {
    engine: Arc<ProgressEngine>,
    /// Number of consecutive poll errors before escalating.
    error_threshold: u32,
    /// Current consecutive error count.
    consecutive_errors: AtomicU32,
    /// Set to `true` when consecutive errors reach the threshold.
    health_warning: AtomicBool,
}

impl ProgressTracker {
    /// Create a new tracker wrapping the given progress engine.
    pub fn new(engine: Arc<ProgressEngine>) -> Self {
        Self {
            engine,
            error_threshold: DEFAULT_ERROR_THRESHOLD,
            consecutive_errors: AtomicU32::new(0),
            health_warning: AtomicBool::new(false),
        }
    }

    /// Create a tracker with a custom consecutive-error threshold.
    pub fn with_error_threshold(engine: Arc<ProgressEngine>, threshold: u32) -> Self {
        Self {
            engine,
            error_threshold: threshold,
            consecutive_errors: AtomicU32::new(0),
            health_warning: AtomicBool::new(false),
        }
    }

    /// Access the underlying progress engine.
    pub fn engine(&self) -> &Arc<ProgressEngine> {
        &self.engine
    }

    /// Register a wr_id for completion tracking, returning a `PendingOp` handle.
    pub fn register_op(&self, wr_id: u64) -> PendingOp {
        self.engine.register_op(wr_id)
    }

    /// Current consecutive error count.
    pub fn consecutive_error_count(&self) -> u32 {
        self.consecutive_errors.load(Ordering::Acquire)
    }

    /// Whether the health warning flag is set.
    pub fn has_health_warning(&self) -> bool {
        self.health_warning.load(Ordering::Acquire)
    }

    /// Clear the health warning flag and reset the consecutive error counter.
    pub fn clear_health_warning(&self) {
        self.health_warning.store(false, Ordering::Release);
        self.consecutive_errors.store(0, Ordering::Release);
    }

    /// The configured error threshold.
    pub fn error_threshold(&self) -> u32 {
        self.error_threshold
    }

    /// Record a successful poll result — resets the consecutive error counter.
    fn record_success(&self) {
        self.consecutive_errors.store(0, Ordering::Release);
    }

    /// Record a poll error — increments the counter and checks threshold.
    fn record_error(&self) {
        let prev = self.consecutive_errors.fetch_add(1, Ordering::AcqRel);
        let new_count = prev + 1;
        if new_count >= self.error_threshold && !self.health_warning.load(Ordering::Acquire) {
            self.health_warning.store(true, Ordering::Release);
            eprintln!(
                "[rmlx-distributed] ProgressTracker: {} consecutive poll errors — health warning triggered",
                new_count
            );
        }
    }

    /// Non-blocking poll of a `PendingOp`. Updates health tracking.
    ///
    /// Returns `None` if still pending, `Some(Ok(completion))` on success,
    /// or `Some(Err(error))` on failure.
    pub fn poll_op(&self, op: &PendingOp) -> Option<Result<Completion, OpError>> {
        match op.try_poll() {
            None => None,
            Some(Ok(c)) => {
                self.record_success();
                Some(Ok(c))
            }
            Some(Err(e)) => {
                self.record_error();
                Some(Err(e))
            }
        }
    }

    /// Non-blocking poll of a `ZeroCopyPendingOp`. Updates health tracking.
    pub fn poll_zero_copy_op(&self, op: &ZeroCopyPendingOp) -> Option<Result<Completion, OpError>> {
        match op.try_poll() {
            None => None,
            Some(Ok(c)) => {
                self.record_success();
                Some(Ok(c))
            }
            Some(Err(e)) => {
                self.record_error();
                Some(Err(e))
            }
        }
    }

    /// Blocking wait on a `PendingOp` with timeout. Updates health tracking.
    pub fn wait_op(
        &self,
        op: &PendingOp,
        timeout: Duration,
    ) -> Result<Completion, DistributedError> {
        match op.wait(timeout) {
            Ok(c) => {
                self.record_success();
                Ok(c)
            }
            Err(rmlx_rdma::progress::WaitError::OpFailed(e)) => {
                self.record_error();
                Err(DistributedError::Transport(format!(
                    "RDMA op failed: {}",
                    e
                )))
            }
            Err(rmlx_rdma::progress::WaitError::Timeout) => {
                self.record_error();
                Err(DistributedError::Transport("RDMA op timed out".to_string()))
            }
        }
    }

    /// Blocking wait on a `ZeroCopyPendingOp` with timeout. Updates health tracking.
    pub fn wait_zero_copy_op(
        &self,
        op: &ZeroCopyPendingOp,
        timeout: Duration,
    ) -> Result<Completion, DistributedError> {
        match op.wait(timeout) {
            Ok(c) => {
                self.record_success();
                Ok(c)
            }
            Err(rmlx_rdma::progress::WaitError::OpFailed(e)) => {
                self.record_error();
                Err(DistributedError::Transport(format!(
                    "RDMA op failed: {}",
                    e
                )))
            }
            Err(rmlx_rdma::progress::WaitError::Timeout) => {
                self.record_error();
                Err(DistributedError::Transport("RDMA op timed out".to_string()))
            }
        }
    }

    /// Poll all `ZeroCopyPendingOp`s in a collection, returning the count of
    /// newly completed operations. Updates health tracking for each.
    pub fn poll_all_zero_copy(&self, ops: &[ZeroCopyPendingOp]) -> usize {
        let mut completed = 0;
        for op in ops {
            if let Some(result) = op.try_poll() {
                match result {
                    Ok(_) => self.record_success(),
                    Err(_) => self.record_error(),
                }
                completed += 1;
            }
        }
        completed
    }

    /// Wait for all `ZeroCopyPendingOp`s to complete, with a per-op timeout.
    ///
    /// Returns the number of operations that completed successfully.
    /// Failed or timed-out operations increment the error counter.
    pub fn wait_all_zero_copy(
        &self,
        ops: &[ZeroCopyPendingOp],
        per_op_timeout: Duration,
    ) -> Result<usize, DistributedError> {
        let mut success_count = 0;
        for (i, op) in ops.iter().enumerate() {
            if !op.is_pending() {
                // Already resolved — check result
                if let Some(result) = op.try_poll() {
                    match result {
                        Ok(_) => {
                            self.record_success();
                            success_count += 1;
                        }
                        Err(_) => self.record_error(),
                    }
                }
                continue;
            }
            match op.wait(per_op_timeout) {
                Ok(_) => {
                    self.record_success();
                    success_count += 1;
                }
                Err(rmlx_rdma::progress::WaitError::OpFailed(e)) => {
                    self.record_error();
                    if self.has_health_warning() {
                        return Err(DistributedError::Transport(format!(
                            "RDMA op {} failed and health threshold reached: {}",
                            i, e
                        )));
                    }
                }
                Err(rmlx_rdma::progress::WaitError::Timeout) => {
                    self.record_error();
                    if self.has_health_warning() {
                        return Err(DistributedError::Transport(format!(
                            "RDMA op {} timed out and health threshold reached",
                            i
                        )));
                    }
                }
            }
        }
        Ok(success_count)
    }

    /// Number of operations currently pending in the engine.
    pub fn pending_count(&self) -> usize {
        self.engine.pending_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_new_starts_healthy() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(engine);
        assert!(!tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 0);
        assert_eq!(tracker.error_threshold(), DEFAULT_ERROR_THRESHOLD);
    }

    #[test]
    fn tracker_custom_threshold() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::with_error_threshold(engine, 3);
        assert_eq!(tracker.error_threshold(), 3);
    }

    #[test]
    fn poll_pending_op_returns_none_when_pending() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(engine);
        let op = tracker.register_op(100);
        assert!(tracker.poll_op(&op).is_none());
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn poll_completed_op_resets_error_count() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(Arc::clone(&engine));

        // Simulate some errors first
        for _ in 0..3 {
            tracker.record_error();
        }
        assert_eq!(tracker.consecutive_error_count(), 3);

        // Register and synthetically complete an op
        let op = tracker.register_op(200);
        engine.synthetic_complete(200);

        let result = tracker.poll_op(&op);
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn consecutive_errors_trigger_health_warning() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::with_error_threshold(engine, 3);

        assert!(!tracker.has_health_warning());

        tracker.record_error();
        assert!(!tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 1);

        tracker.record_error();
        assert!(!tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 2);

        tracker.record_error();
        assert!(tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 3);
    }

    #[test]
    fn success_resets_consecutive_errors() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::with_error_threshold(engine, 5);

        tracker.record_error();
        tracker.record_error();
        assert_eq!(tracker.consecutive_error_count(), 2);

        tracker.record_success();
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn clear_health_warning_resets_state() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::with_error_threshold(engine, 2);

        tracker.record_error();
        tracker.record_error();
        assert!(tracker.has_health_warning());

        tracker.clear_health_warning();
        assert!(!tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn dispatch_completes_via_progress_engine() {
        // Simulate EP dispatch: register ops, complete them, poll through tracker
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(Arc::clone(&engine));

        // Simulate 3 send ops + 2 recv ops (typical 3-peer dispatch)
        let send_ops: Vec<PendingOp> = (0..3).map(|i| tracker.register_op(1000 + i)).collect();
        let recv_ops: Vec<PendingOp> = (0..2).map(|i| tracker.register_op(2000 + i)).collect();

        assert_eq!(tracker.pending_count(), 5);

        // All should be pending
        for op in &send_ops {
            assert!(tracker.poll_op(op).is_none());
        }
        for op in &recv_ops {
            assert!(tracker.poll_op(op).is_none());
        }

        // Synthetically complete all ops
        for i in 0..3 {
            engine.synthetic_complete(1000 + i);
        }
        for i in 0..2 {
            engine.synthetic_complete(2000 + i);
        }

        // All should now resolve successfully
        for op in &send_ops {
            let result = tracker.poll_op(op);
            assert!(result.is_some());
            assert!(result.unwrap().is_ok());
        }
        for op in &recv_ops {
            let result = tracker.poll_op(op);
            assert!(result.is_some());
            assert!(result.unwrap().is_ok());
        }

        assert!(!tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn async_progress_tracked_correctly() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(Arc::clone(&engine));

        // Register ops at different "times"
        let op1 = tracker.register_op(10);
        let op2 = tracker.register_op(20);
        let op3 = tracker.register_op(30);

        // Complete op2 first (out of order)
        engine.synthetic_complete(20);

        assert!(tracker.poll_op(&op1).is_none()); // still pending
        assert!(tracker.poll_op(&op2).unwrap().is_ok()); // done
        assert!(tracker.poll_op(&op3).is_none()); // still pending

        // Complete op1
        engine.synthetic_complete(10);
        assert!(tracker.poll_op(&op1).unwrap().is_ok());

        // op3 still pending
        assert!(tracker.poll_op(&op3).is_none());

        // Complete op3
        engine.synthetic_complete(30);
        assert!(tracker.poll_op(&op3).unwrap().is_ok());

        assert!(!tracker.has_health_warning());
    }

    #[test]
    fn error_threshold_triggers_on_mixed_results() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::with_error_threshold(Arc::clone(&engine), 3);

        // Simulate: success, error, error, success, error, error, error -> warning
        tracker.record_success();
        assert_eq!(tracker.consecutive_error_count(), 0);

        tracker.record_error();
        tracker.record_error();
        assert_eq!(tracker.consecutive_error_count(), 2);
        assert!(!tracker.has_health_warning());

        tracker.record_success(); // resets
        assert_eq!(tracker.consecutive_error_count(), 0);

        tracker.record_error();
        tracker.record_error();
        tracker.record_error(); // threshold = 3 -> warning
        assert!(tracker.has_health_warning());
        assert_eq!(tracker.consecutive_error_count(), 3);
    }

    #[test]
    fn wait_op_success() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(Arc::clone(&engine));

        let op = tracker.register_op(42);

        // Complete from another thread
        let eng = Arc::clone(&engine);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            eng.synthetic_complete(42);
        });

        let result = tracker.wait_op(&op, Duration::from_secs(5));
        assert!(result.is_ok());
        assert_eq!(tracker.consecutive_error_count(), 0);
    }

    #[test]
    fn wait_op_timeout_increments_errors() {
        let engine = Arc::new(ProgressEngine::new());
        let tracker = ProgressTracker::new(Arc::clone(&engine));

        let op = tracker.register_op(99);
        // Don't complete it — let it timeout
        let result = tracker.wait_op(&op, Duration::from_millis(10));
        assert!(result.is_err());
        assert_eq!(tracker.consecutive_error_count(), 1);
    }
}
