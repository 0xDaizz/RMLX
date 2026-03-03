//! Autorelease pool management for Objective-C object lifecycle (M9).
//!
//! Metal operations create many temporary Objective-C objects (command buffers,
//! encoders, pipeline states). Without an autorelease pool, these objects
//! accumulate until the thread exits.
//!
//! [`ScopedPool`] provides a RAII wrapper that drains the pool on drop,
//! ensuring deterministic cleanup in tight loops or long-running threads.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rmlx_metal::autorelease::ScopedPool;
//!
//! for batch in batches {
//!     let _pool = ScopedPool::new();
//!     // ... create command buffers, encoders, etc.
//!     // Pool is drained when `_pool` drops at end of iteration.
//! }
//! ```

/// RAII autorelease pool wrapper.
///
/// Creates an `NSAutoreleasePool` on construction and drains it on drop.
/// This is critical for threads that create many temporary Objective-C
/// objects (e.g., inference loops, training steps) to prevent unbounded
/// memory growth from autoreleased objects.
pub struct ScopedPool {
    // Raw pointer to the NSAutoreleasePool Objective-C object.
    pool: *mut objc::runtime::Object,
}

impl ScopedPool {
    /// Create a new autorelease pool scope.
    ///
    /// All Objective-C objects created after this call (on the current thread)
    /// will be autoreleased into this pool and freed when the `ScopedPool`
    /// is dropped.
    pub fn new() -> Self {
        // SAFETY: NSAutoreleasePool is a well-known Foundation class.
        // `alloc` + `init` is the standard Obj-C construction pattern.
        let pool: *mut objc::runtime::Object = unsafe {
            let cls = objc::runtime::Class::get("NSAutoreleasePool")
                .expect("NSAutoreleasePool class not found");
            let pool: *mut objc::runtime::Object = objc::msg_send![cls, alloc];
            let pool: *mut objc::runtime::Object = objc::msg_send![pool, init];
            pool
        };
        Self { pool }
    }
}

impl Default for ScopedPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ScopedPool {
    fn drop(&mut self) {
        // SAFETY: We own the pool and drain it exactly once.
        unsafe {
            let _: () = objc::msg_send![self.pool, drain];
        }
    }
}

// SAFETY: ScopedPool is only valid on the thread that created it.
// Objective-C autorelease pools are per-thread, so Send is NOT safe.
// However, the typical usage pattern (RAII within a single function scope)
// means this is fine as long as we don't implement Send.
// We explicitly do NOT impl Send or Sync.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_pool_create_and_drop() {
        // Verify that creating and dropping a pool does not crash.
        let pool = ScopedPool::new();
        drop(pool);
    }

    #[test]
    fn test_scoped_pool_nested() {
        // Nested pools should work (inner drains first).
        let _outer = ScopedPool::new();
        {
            let _inner = ScopedPool::new();
            // Inner pool drains here.
        }
        // Outer pool drains here.
    }

    #[test]
    fn test_scoped_pool_with_metal_objects() {
        // Create Metal objects inside a pool to verify no crash.
        let _pool = ScopedPool::new();
        let device = metal::Device::system_default().unwrap();
        let _queue = device.new_command_queue();
        let _buf = device.new_buffer(256, metal::MTLResourceOptions::StorageModeShared);
        // Pool drains, releasing any autoreleased intermediates.
    }
}
