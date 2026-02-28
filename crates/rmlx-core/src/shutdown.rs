//! Graceful shutdown signalling via atomic flag.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A shutdown signal that can be triggered from any thread.
pub struct ShutdownSignal {
    flag: Arc<AtomicBool>,
}

impl ShutdownSignal {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Trigger the shutdown signal.
    pub fn trigger(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Check whether shutdown has been triggered.
    pub fn is_triggered(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Create a lightweight handle for checking the shutdown state.
    pub fn clone_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            flag: Arc::clone(&self.flag),
        }
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// A read-only handle for checking the shutdown state.
pub struct ShutdownHandle {
    flag: Arc<AtomicBool>,
}

impl ShutdownHandle {
    /// Check whether shutdown has been triggered.
    pub fn is_shutdown(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}
