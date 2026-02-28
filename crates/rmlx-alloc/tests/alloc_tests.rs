//! Integration tests for rmlx-alloc

use std::sync::Arc;

use rmlx_alloc::allocator::MetalAllocator;
use rmlx_alloc::cache::BufferCache;
use rmlx_alloc::stats::AllocStats;
use rmlx_alloc::zero_copy::ZeroCopyBuffer;
use rmlx_metal::device::GpuDevice;

/// Helper: acquire the system default Metal device or skip.
fn require_gpu() -> Option<Arc<GpuDevice>> {
    match GpuDevice::system_default() {
        Ok(d) => Some(Arc::new(d)),
        Err(_) => {
            eprintln!("skipping test: no Metal device available");
            None
        }
    }
}

#[test]
fn test_zero_copy_buffer_lifecycle() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping test: no Metal device available");
            return;
        }
    };

    // Allocate a 64KB zero-copy buffer
    let buf = ZeroCopyBuffer::new(&device, 65536).expect("allocation");

    // Verify properties
    assert!(buf.size() >= 65536); // page-aligned, may be larger
    assert!(!buf.as_ptr().is_null());

    // In-flight tracking -- only self holds the Arc
    assert_eq!(buf.in_flight_count(), 1);
    let token = buf.acquire_in_flight();
    assert_eq!(buf.in_flight_count(), 2);
    drop(token);
    assert_eq!(buf.in_flight_count(), 1);

    // Fence lifecycle
    let fence = buf.acquire_fence("test_op");
    assert_eq!(buf.in_flight_count(), 2);
    assert_eq!(fence.op_tag(), "test_op");
    fence.release_after_verification();
    assert_eq!(buf.in_flight_count(), 1);

    // Metal buffer should be valid
    let _metal_buf = buf.metal_buffer();

    // Drop should succeed without timeout (no outstanding in-flight refs)
    drop(buf);
}

#[test]
fn test_buffer_cache_acquire_release() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    // Cache with 1 MB capacity
    let mut cache = BufferCache::new(1024 * 1024);

    // Acquire from empty cache -> cache miss -> returns None
    let cached = cache.acquire(4096);
    assert!(cached.is_none(), "empty cache should return None");

    // Allocate a real Metal buffer via the device, then release into cache
    let buf = device.new_buffer(
        4096,
        rmlx_metal::metal::MTLResourceOptions::StorageModeShared,
    );
    let buf_len = buf.length();
    assert!(buf_len >= 4096);
    cache.release(buf);

    // Cache should now contain something
    assert!(cache.cache_size() > 0);

    // Acquire again -- should get the cached buffer back
    let buf2 = cache.acquire(4096);
    assert!(buf2.is_some(), "cache should return the released buffer");
    let buf2 = buf2.unwrap();
    assert!(buf2.length() >= 4096);
}

#[test]
fn test_metal_allocator_with_cache() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    let allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);

    // Allocate and free
    let buf = allocator.alloc(8192).expect("alloc");
    assert!(buf.length() >= 8192);
    allocator.free(buf);

    // Second allocation should use cache (cache hit)
    let buf2 = allocator.alloc(8192).expect("alloc2");
    assert!(buf2.length() >= 8192);

    // Verify stats
    let stats = allocator.stats();
    assert!(stats.total_allocs() >= 1);
    assert!(stats.cache_hits() >= 1);

    allocator.free(buf2);
}

#[test]
fn test_alloc_stats_atomic() {
    let stats = AllocStats::new();

    stats.record_alloc(4096);
    stats.record_alloc(8192);

    assert_eq!(stats.total_allocs(), 2);
    assert_eq!(stats.active(), 4096 + 8192);
    assert_eq!(stats.peak(), 4096 + 8192);

    stats.record_free(4096);
    assert_eq!(stats.total_frees(), 1);
    // active = 8192, peak still = 12288
    assert_eq!(stats.active(), 8192);
    assert_eq!(stats.peak(), 4096 + 8192);
}

#[test]
fn test_alloc_stats_cache_counters() {
    let stats = AllocStats::new();

    stats.record_cache_hit();
    stats.record_cache_hit();
    stats.record_cache_miss();

    assert_eq!(stats.cache_hits(), 2);
    assert_eq!(stats.cache_misses(), 1);
}

#[test]
fn test_allocator_block_limit() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    let mut allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);
    allocator.set_block_limit(16384);

    // First allocation should succeed
    let buf = allocator.alloc(8192).expect("first alloc within limit");
    assert!(buf.length() >= 8192);

    // Second allocation that would exceed the limit should fail
    let result = allocator.alloc(16384);
    assert!(
        result.is_err(),
        "allocation exceeding block limit should fail"
    );

    allocator.free(buf);
}
