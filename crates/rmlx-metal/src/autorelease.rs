//! Autorelease pool management for Objective-C object lifecycle (M9).
//!
//! Metal operations create many temporary Objective-C objects (command buffers,
//! encoders, pipeline states). Without an autorelease pool, these objects
//! accumulate until the thread exits.
//!
//! This module re-exports [`objc2::rc::autoreleasepool`], the idiomatic
//! closure-based autorelease pool from the objc2 ecosystem.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rmlx_metal::autoreleasepool;
//!
//! for batch in batches {
//!     autoreleasepool(|_| {
//!         // ... create command buffers, encoders, etc.
//!         // Pool is drained when the closure returns.
//!     });
//! }
//! ```

pub use objc2::rc::autoreleasepool;

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLDevice as _;

    #[test]
    fn test_autoreleasepool_create_and_drain() {
        // Verify that creating and draining a pool does not crash.
        autoreleasepool(|_| {
            // Empty pool.
        });
    }

    #[test]
    fn test_autoreleasepool_nested() {
        // Nested pools should work (inner drains first).
        autoreleasepool(|_| {
            autoreleasepool(|_| {
                // Inner pool drains here.
            });
            // Outer pool drains here.
        });
    }

    #[test]
    fn test_autoreleasepool_with_metal_objects() {
        // Create Metal objects inside a pool to verify no crash.
        use std::sync::OnceLock;
        fn test_device() -> &'static crate::types::MtlDevice {
            static DEVICE: OnceLock<crate::types::MtlDevice> = OnceLock::new();
            DEVICE.get_or_init(|| {
                objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
            })
        }
        autoreleasepool(|_| {
            let device = test_device();
            let _queue = device.newCommandQueue().unwrap();
            let _buf = device
                .newBufferWithLength_options(
                    256,
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .unwrap();
            // Pool drains, releasing any autoreleased intermediates.
        });
    }
}
