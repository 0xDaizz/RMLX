//! CQ Progress Engine — asynchronous completion processing.
//!
//! Provides two modes of operation:
//! - **Manual**: Caller drives progress via `drive_progress()` in their own loop.
//! - **Background**: A dedicated thread polls the CQ and resolves pending ops.
//!
//! Each RDMA operation gets a `PendingOp` handle that can be polled or awaited
//! without holding any locks during the busy-wait. Per-op completion uses
//! `AtomicU8` for lock-free signaling and `Condvar` for efficient blocking waits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use crate::exchange_tag::{try_decode_wr_id, WrIdFields};
use crate::ffi::{wc_status, wc_status_str, IbvWc};
use crate::qp::CompletionQueue;

// ─── Op states (AtomicU8) ───

const OP_PENDING: u8 = 0;
const OP_DONE: u8 = 1;
const OP_ERR: u8 = 2;

// ─── Types ───

/// Completion information for a successful operation.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The wr_id that completed.
    pub wr_id: u64,
    /// Decoded wr_id fields (if the tag is valid).
    pub fields: Option<WrIdFields>,
    /// Number of bytes transferred (from ibv_wc.byte_len).
    pub byte_len: u32,
    /// Work completion opcode.
    pub opcode: u32,
}

/// Error for a failed RDMA operation.
#[derive(Debug, Clone)]
pub struct OpError {
    pub wr_id: u64,
    pub status: u32,
    pub status_str: &'static str,
    pub vendor_err: u32,
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RDMA op failed: wr_id={} status={}({}) vendor_err={}",
            self.wr_id, self.status_str, self.status, self.vendor_err
        )
    }
}

impl std::error::Error for OpError {}

/// Error from waiting on a `PendingOp`.
#[derive(Debug)]
pub enum WaitError {
    /// The operation failed with an RDMA error.
    OpFailed(OpError),
    /// The wait timed out before the operation completed.
    Timeout,
}

impl std::fmt::Display for WaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpFailed(e) => write!(f, "{e}"),
            Self::Timeout => write!(f, "wait timed out"),
        }
    }
}

impl std::error::Error for WaitError {}

/// Shared state for a single pending operation.
struct OpSlot {
    state: AtomicU8,
    result: OnceLock<Result<Completion, OpError>>,
    notify: (Mutex<()>, Condvar),
}

/// Handle to a pending RDMA operation.
///
/// Returned by `ProgressEngine::register_op`. Can be polled non-blocking
/// via `try_poll()` or waited on blocking via `wait()`.
pub struct PendingOp {
    slot: Arc<OpSlot>,
    wr_id: u64,
}

impl PendingOp {
    /// Non-blocking poll. Returns `Some` if the operation has completed.
    pub fn try_poll(&self) -> Option<Result<Completion, OpError>> {
        let s = self.slot.state.load(Ordering::Acquire);
        if s == OP_PENDING {
            return None;
        }
        // Result must be set before state transitions from PENDING.
        Some(self.slot.result.get().expect("result must be set").clone())
    }

    /// Returns true if the operation is still pending.
    pub fn is_pending(&self) -> bool {
        self.slot.state.load(Ordering::Acquire) == OP_PENDING
    }

    /// Blocking wait with timeout.
    pub fn wait(self, timeout: Duration) -> Result<Completion, WaitError> {
        // Fast path: already done
        if let Some(r) = self.try_poll() {
            return r.map_err(WaitError::OpFailed);
        }

        // Slow path: condvar wait
        let (lock, cvar) = &self.slot.notify;
        let guard = lock.lock().unwrap();
        let _guard = cvar
            .wait_timeout_while(guard, timeout, |_| {
                self.slot.state.load(Ordering::Acquire) == OP_PENDING
            })
            .unwrap();

        match self.try_poll() {
            Some(r) => r.map_err(WaitError::OpFailed),
            None => Err(WaitError::Timeout),
        }
    }

    /// The wr_id this operation is tracking.
    pub fn wr_id(&self) -> u64 {
        self.wr_id
    }
}

/// Progress mode for the engine.
#[derive(Debug, Clone)]
pub enum ProgressMode {
    /// Caller drives progress manually via `drive_progress()`.
    Manual,
    /// A background thread polls the CQ.
    Background(ProgressConfig),
}

/// Configuration for background progress mode.
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    /// Maximum completions to process per poll cycle.
    pub poll_budget: usize,
    /// Yield between empty polls to avoid 100% CPU spin.
    pub yield_on_empty: bool,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            poll_budget: 32,
            yield_on_empty: true,
        }
    }
}

/// CQ progress engine for asynchronous completion processing.
///
/// Thread-safe: the pending map is protected by a Mutex, but the lock is
/// only held briefly to insert/remove entries — never during CQ polling
/// or busy-waiting.
pub struct ProgressEngine {
    /// Map from wr_id to the shared completion slot.
    pending: Arc<Mutex<HashMap<u64, Arc<OpSlot>>>>,
    /// Signal to stop the background thread.
    shutdown: Arc<AtomicBool>,
    /// Background thread join handle.
    bg_handle: Option<std::thread::JoinHandle<()>>,
}

impl ProgressEngine {
    /// Create a new progress engine.
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            bg_handle: None,
        }
    }

    /// Register a wr_id for completion tracking.
    ///
    /// Returns a `PendingOp` handle that will be resolved when the CQ
    /// reports a completion for this wr_id.
    pub fn register_op(&self, wr_id: u64) -> PendingOp {
        let slot = Arc::new(OpSlot {
            state: AtomicU8::new(OP_PENDING),
            result: OnceLock::new(),
            notify: (Mutex::new(()), Condvar::new()),
        });
        self.pending
            .lock()
            .unwrap()
            .insert(wr_id, Arc::clone(&slot));
        PendingOp { slot, wr_id }
    }

    /// Drive progress by polling the CQ up to `budget` completions.
    ///
    /// Returns the number of completions processed. This is the manual-mode
    /// entry point — call it in your event loop.
    ///
    /// No locks are held during the ibv_poll_cq call itself. The pending map
    /// lock is only held briefly to look up and remove completed entries.
    pub fn drive_progress(&self, cq: &CompletionQueue, budget: usize) -> usize {
        let poll_size = budget.min(64);
        let mut wc_buf: Vec<IbvWc> = vec![unsafe { std::mem::zeroed() }; poll_size];
        let mut total = 0;

        // Poll CQ — no lock held here
        let count = match cq.poll(&mut wc_buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[rmlx-rdma] progress: CQ poll error: {e}");
                return 0;
            }
        };

        if count == 0 {
            return 0;
        }

        // Resolve completions — lock held only for HashMap lookups
        let mut pending = self.pending.lock().unwrap();
        for wc in &wc_buf[..count] {
            if let Some(slot) = pending.remove(&wc.wr_id) {
                Self::resolve_slot(&slot, wc);
                total += 1;
            } else {
                // Completion for unregistered wr_id — log and discard.
                // This can happen if the caller used the old CompletionTracker
                // or wait_completions path for some operations.
                eprintln!(
                    "[rmlx-rdma] progress: unregistered wr_id={} status={}",
                    wc.wr_id,
                    wc_status_str(wc.status)
                );
            }
        }

        total
    }

    /// Start a background thread that continuously polls the CQ.
    ///
    /// The CQ must be wrapped in an `Arc` so the background thread can own
    /// a reference. The thread runs until `shutdown()` is called.
    pub fn start_background(&mut self, cq: Arc<CompletionQueue>, config: ProgressConfig) {
        if self.bg_handle.is_some() {
            eprintln!("[rmlx-rdma] progress: background thread already running");
            return;
        }

        let pending = Arc::clone(&self.pending);
        let shutdown = Arc::clone(&self.shutdown);
        self.shutdown.store(false, Ordering::Release);

        let handle = std::thread::Builder::new()
            .name("rmlx-cq-progress".into())
            .spawn(move || {
                let budget = config.poll_budget.min(64);
                let mut wc_buf: Vec<IbvWc> = vec![unsafe { std::mem::zeroed() }; budget];

                while !shutdown.load(Ordering::Acquire) {
                    let count = match cq.poll(&mut wc_buf) {
                        Ok(n) => n,
                        Err(e) => {
                            eprintln!("[rmlx-rdma] progress bg: CQ poll error: {e}");
                            if config.yield_on_empty {
                                std::thread::yield_now();
                            }
                            continue;
                        }
                    };

                    if count == 0 {
                        if config.yield_on_empty {
                            std::thread::yield_now();
                        }
                        continue;
                    }

                    let mut map = pending.lock().unwrap();
                    for wc in &wc_buf[..count] {
                        if let Some(slot) = map.remove(&wc.wr_id) {
                            Self::resolve_slot(&slot, wc);
                        } else {
                            eprintln!(
                                "[rmlx-rdma] progress bg: unregistered wr_id={} status={}",
                                wc.wr_id,
                                wc_status_str(wc.status)
                            );
                        }
                    }
                }
            })
            .expect("failed to spawn CQ progress thread");

        self.bg_handle = Some(handle);
    }

    /// Signal the background thread to stop and wait for it to exit.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.bg_handle.take() {
            let _ = handle.join();
        }
    }

    /// Number of operations currently pending completion.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Resolve a single operation slot from a work completion.
    fn resolve_slot(slot: &OpSlot, wc: &IbvWc) {
        let result = if wc.status == wc_status::SUCCESS {
            Ok(Completion {
                wr_id: wc.wr_id,
                fields: try_decode_wr_id(wc.wr_id),
                byte_len: wc.byte_len,
                opcode: wc.opcode,
            })
        } else {
            Err(OpError {
                wr_id: wc.wr_id,
                status: wc.status,
                status_str: wc_status_str(wc.status),
                vendor_err: wc.vendor_err,
            })
        };

        let new_state = if result.is_ok() { OP_DONE } else { OP_ERR };

        // Set result before state transition — readers check state first,
        // then read result, so result must be visible when state changes.
        let _ = slot.result.set(result);
        slot.state.store(new_state, Ordering::Release);

        // Wake any thread blocked in PendingOp::wait()
        let (lock, cvar) = &slot.notify;
        let _guard = lock.lock().unwrap();
        cvar.notify_all();
    }
}

impl Default for ProgressEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ProgressEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_op_lifecycle() {
        let engine = ProgressEngine::new();
        let op = engine.register_op(42);
        assert!(op.is_pending());
        assert!(op.try_poll().is_none());
        assert_eq!(engine.pending_count(), 1);

        // Simulate completion by directly resolving the slot
        {
            let map = engine.pending.lock().unwrap();
            let slot = map.get(&42).unwrap();
            let _ = slot.result.set(Ok(Completion {
                wr_id: 42,
                fields: None,
                byte_len: 100,
                opcode: 0,
            }));
            slot.state.store(OP_DONE, Ordering::Release);
            let (lock, cvar) = &slot.notify;
            let _g = lock.lock().unwrap();
            cvar.notify_all();
        }

        assert!(!op.is_pending());
        let result = op.try_poll().unwrap();
        assert!(result.is_ok());
        let completion = result.unwrap();
        assert_eq!(completion.wr_id, 42);
        assert_eq!(completion.byte_len, 100);
    }

    #[test]
    fn pending_op_error() {
        let engine = ProgressEngine::new();
        let op = engine.register_op(99);

        {
            let map = engine.pending.lock().unwrap();
            let slot = map.get(&99).unwrap();
            let _ = slot.result.set(Err(OpError {
                wr_id: 99,
                status: wc_status::WR_FLUSH_ERR,
                status_str: wc_status_str(wc_status::WR_FLUSH_ERR),
                vendor_err: 0,
            }));
            slot.state.store(OP_ERR, Ordering::Release);
            let (lock, cvar) = &slot.notify;
            let _g = lock.lock().unwrap();
            cvar.notify_all();
        }

        let result = op.try_poll().unwrap();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status, wc_status::WR_FLUSH_ERR);
        assert_eq!(err.status_str, "WR_FLUSH_ERR");
    }

    #[test]
    fn wait_timeout_returns_err() {
        let engine = ProgressEngine::new();
        let op = engine.register_op(7);

        let result = op.wait(Duration::from_millis(10));
        assert!(matches!(result, Err(WaitError::Timeout)));
    }

    #[test]
    fn wait_resolves_before_timeout() {
        let engine = ProgressEngine::new();
        let op = engine.register_op(55);

        // Resolve from another thread after a short delay
        let pending = Arc::clone(&engine.pending);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            let map = pending.lock().unwrap();
            if let Some(slot) = map.get(&55) {
                let _ = slot.result.set(Ok(Completion {
                    wr_id: 55,
                    fields: None,
                    byte_len: 256,
                    opcode: 0,
                }));
                slot.state.store(OP_DONE, Ordering::Release);
                let (lock, cvar) = &slot.notify;
                let _g = lock.lock().unwrap();
                cvar.notify_all();
            }
        });

        let result = op.wait(Duration::from_secs(5));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().byte_len, 256);
    }

    #[test]
    fn multiple_ops_independent() {
        let engine = ProgressEngine::new();
        let op1 = engine.register_op(1);
        let op2 = engine.register_op(2);
        let op3 = engine.register_op(3);
        assert_eq!(engine.pending_count(), 3);

        // Resolve only op2
        {
            let map = engine.pending.lock().unwrap();
            let slot = map.get(&2).unwrap();
            let _ = slot.result.set(Ok(Completion {
                wr_id: 2,
                fields: None,
                byte_len: 0,
                opcode: 0,
            }));
            slot.state.store(OP_DONE, Ordering::Release);
            let (lock, cvar) = &slot.notify;
            let _g = lock.lock().unwrap();
            cvar.notify_all();
        }

        assert!(op1.is_pending());
        assert!(!op2.is_pending());
        assert!(op3.is_pending());
    }
}
