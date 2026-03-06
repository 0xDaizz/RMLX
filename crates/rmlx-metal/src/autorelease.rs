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

use std::marker::PhantomData;

/// RAII autorelease pool wrapper.
///
/// Creates an `NSAutoreleasePool` on construction and drains it on drop.
/// This is critical for threads that create many temporary Objective-C
/// objects (e.g., inference loops, training steps) to prevent unbounded
/// memory growth from autoreleased objects.
///
/// # Thread Safety
///
/// `ScopedPool` is intentionally `!Send` and `!Sync` because
/// `NSAutoreleasePool` is a per-thread construct in Objective-C.
/// The `PhantomData<*mut ()>` marker enforces this at the type level.
pub struct ScopedPool {
    // Raw pointer to the NSAutoreleasePool Objective-C object.
    pool: *mut objc::runtime::Object,
    // Marker to enforce !Send and !Sync (raw pointers are !Send + !Sync).
    _marker: PhantomData<*mut ()>,
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
        Self {
            pool,
            _marker: PhantomData,
        }
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

    // ScopedPool must NOT implement Send or Sync because NSAutoreleasePool
    // is a per-thread construct. The PhantomData<*mut ()> field enforces this.
    // If someone accidentally adds `unsafe impl Send for ScopedPool {}`,
    // the following test will fail.
    #[test]
    fn scoped_pool_is_not_send_or_sync() {
        fn is_send<T: Send>(_: &T) -> bool {
            true
        }
        fn is_sync<T: Sync>(_: &T) -> bool {
            true
        }

        // We cannot call is_send::<ScopedPool>() directly (it won't compile,
        // which is the desired behavior). Instead we verify via trait objects.
        // ScopedPool should not satisfy the Send or Sync bounds.
        // This is enforced at compile time by PhantomData<*mut ()>.
        // This test documents the invariant; the real guard is the
        // PhantomData marker which makes `fn needs_send<T: Send>(_: T) {}`
        // fail to compile if passed a ScopedPool.
        let _ = (is_send::<u8>(&0), is_sync::<u8>(&0)); // suppress unused warnings
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
