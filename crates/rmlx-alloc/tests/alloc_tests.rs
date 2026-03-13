//! Integration tests for rmlx-alloc

use objc2_metal::MTLBuffer as _;
use std::sync::Arc;
use std::time::Duration;

use rmlx_alloc::allocator::MetalAllocator;
use rmlx_alloc::buffer_pool::BufferPool;
use rmlx_alloc::cache::BufferCache;
use rmlx_alloc::leak_detector::LeakDetector;
use rmlx_alloc::stats::AllocStats;
use rmlx_alloc::zero_copy::{CompletionTicket, ZeroCopyBuffer};
use rmlx_metal::device::GpuDevice;
use rmlx_metal::event::GpuEvent;

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
        rmlx_metal::MTLResourceOptions::StorageModeShared,
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
    let (buf, _offset) = allocator.alloc(8192).expect("alloc");
    assert!(buf.length() >= 8192);
    allocator.free(buf).expect("free");

    // Second allocation should use cache (cache hit)
    let (buf2, _offset2) = allocator.alloc(8192).expect("alloc2");
    assert!(buf2.length() >= 8192);

    // Verify stats
    let stats = allocator.stats();
    assert!(stats.total_allocs() >= 1);
    assert!(stats.cache_hits() >= 1);

    allocator.free(buf2).expect("free2");
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

    let allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);
    allocator.set_block_limit(16384);

    // First allocation should succeed
    let (buf, _offset) = allocator.alloc(8192).expect("first alloc within limit");
    assert!(buf.length() >= 8192);

    // Second allocation that would exceed the limit should fail
    let result = allocator.alloc(16384);
    assert!(
        result.is_err(),
        "allocation exceeding block limit should fail"
    );

    allocator.free(buf).expect("free");
}

// --- CompletionTicket + GpuEvent integration tests ---

#[test]
fn test_completion_ticket_basic_lifecycle() {
    let ticket = CompletionTicket::new();

    // Initially not complete
    assert!(!ticket.is_safe_to_free());
    assert!(!ticket.is_gpu_complete());
    assert!(!ticket.is_rdma_complete());

    // Mark GPU complete
    ticket.mark_gpu_complete();
    assert!(ticket.is_gpu_complete());
    assert!(!ticket.is_safe_to_free()); // RDMA still pending

    // Mark RDMA complete
    ticket.mark_rdma_complete();
    assert!(ticket.is_rdma_complete());
    assert!(ticket.is_safe_to_free()); // Both done
}

#[test]
fn test_completion_ticket_wait_all_complete() {
    let ticket = CompletionTicket::new();

    // Complete both before waiting
    ticket.mark_gpu_complete();
    ticket.mark_rdma_complete();

    let result = ticket.wait_all_complete(Duration::from_millis(100));
    assert!(result.is_ok());
}

#[test]
fn test_completion_ticket_wait_timeout() {
    let ticket = CompletionTicket::new();
    // Neither completed -> should timeout
    let result = ticket.wait_all_complete(Duration::from_millis(10));
    assert!(result.is_err());
}

#[test]
fn test_completion_ticket_with_gpu_event() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping test: no Metal device");
            return;
        }
    };

    let event = Arc::new(GpuEvent::new(device.raw()));
    let mut ticket = CompletionTicket::new();
    assert!(!ticket.has_gpu_event());

    let signal_value = ticket.with_gpu_event(Arc::clone(&event));
    assert!(ticket.has_gpu_event());
    assert!(signal_value > 0);

    // Not complete yet (event hasn't been signaled)
    assert!(!ticket.is_safe_to_free());

    // Mark gpu_complete via atomic flag (fallback path)
    ticket.mark_gpu_complete();
    assert!(ticket.is_gpu_complete());

    // Still need RDMA
    assert!(!ticket.is_safe_to_free());

    ticket.mark_rdma_complete();
    assert!(ticket.is_safe_to_free());
}

#[test]
fn test_completion_ticket_clone_shared_state() {
    let ticket = CompletionTicket::new();
    let ticket2 = ticket.clone();

    // Marking via one clone should be visible from the other
    ticket.mark_gpu_complete();
    assert!(ticket2.is_gpu_complete());

    ticket2.mark_rdma_complete();
    assert!(ticket.is_rdma_complete());
    assert!(ticket.is_safe_to_free());
}

// --- BufferPool tests ---

#[test]
fn test_buffer_pool_basic() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping test: no Metal device");
            return;
        }
    };

    let mut pool = BufferPool::new(4096, 4);
    assert_eq!(pool.free_count(), 0);
    assert_eq!(pool.pending_count(), 0);

    // Acquire from empty pool -> allocates new buffer
    let buf = pool.acquire(&device).expect("acquire");
    assert!(buf.size() >= 4096);

    // Release back to pool (no ticket -> safe to free -> goes to free list)
    pool.release(buf);
    assert_eq!(pool.free_count(), 1);

    // Re-acquire should get the pooled buffer
    let buf2 = pool.acquire(&device).expect("acquire2");
    assert!(buf2.size() >= 4096);
    assert_eq!(pool.free_count(), 0);

    pool.release(buf2);
}

#[test]
fn test_buffer_pool_pending_drain() {
    let device = match GpuDevice::system_default() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skipping test: no Metal device");
            return;
        }
    };

    let mut pool = BufferPool::new(4096, 4);

    // Allocate and attach an incomplete ticket
    let mut buf = pool.acquire(&device).expect("acquire");
    let ticket = CompletionTicket::new();
    buf.set_ticket(ticket.clone());

    // Release with incomplete ticket -> goes to pending
    pool.release(buf);
    assert_eq!(pool.free_count(), 0);
    assert_eq!(pool.pending_count(), 1);

    // Acquire triggers drain, but pending buffer is still not complete
    let buf2 = pool.acquire(&device).expect("acquire2");
    assert_eq!(pool.pending_count(), 1); // still pending

    // Now complete the ticket
    ticket.mark_gpu_complete();
    ticket.mark_rdma_complete();

    // Release buf2 and re-acquire -> drain should move completed to free
    pool.release(buf2);
    let _buf3 = pool.acquire(&device).expect("acquire3");
    assert_eq!(pool.pending_count(), 0); // drained
}

// --- PR 4.3: Small-buffer pool wiring tests ---

#[test]
fn test_small_alloc_uses_pool() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    let allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);
    let initial_free = allocator.small_pool().free_count();

    // Allocate a small buffer (<256 B) — should route through SmallBufferPool.
    let (buf, sub_offset) = allocator.alloc(64).expect("small alloc should succeed");
    assert!(buf.length() > 0);
    // Sub-allocation offset must be within the backing buffer.
    assert!(
        sub_offset + 64 <= buf.length(),
        "sub_offset ({sub_offset}) + size (64) must fit within backing buffer length ({})",
        buf.length()
    );

    // The small pool should have one fewer free slot.
    let after_alloc_free = allocator.small_pool().free_count();
    assert_eq!(
        after_alloc_free,
        initial_free - 1,
        "small pool should have consumed one slot"
    );

    // Free the buffer — slot should be returned to the pool.
    allocator.free(buf).expect("free should succeed");
    let after_free = allocator.small_pool().free_count();
    assert_eq!(
        after_free, initial_free,
        "small pool slot should be returned after free"
    );
}

#[test]
fn test_large_alloc_bypasses_small_pool() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    let allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);
    let initial_free = allocator.small_pool().free_count();

    // Allocate a buffer larger than MAX_SMALL_ALLOC (256 B).
    let (buf, large_offset) = allocator.alloc(4096).expect("large alloc should succeed");
    assert!(buf.length() >= 4096);
    assert_eq!(large_offset, 0, "large allocs should have zero offset");

    // Small pool should be untouched.
    assert_eq!(
        allocator.small_pool().free_count(),
        initial_free,
        "small pool should not be used for large allocations"
    );

    allocator.free(buf).expect("free should succeed");
}

// --- PR 4.3: LeakDetector wiring tests ---

#[test]
fn test_leak_detector_tracks_alloc_free() {
    let device = match require_gpu() {
        Some(d) => d,
        None => return,
    };

    let allocator = MetalAllocator::new(Arc::clone(&device), 1024 * 1024);
    assert_eq!(allocator.leak_detector().outstanding_allocs(), 0);

    let (buf1, _offset1) = allocator.alloc(4096).expect("alloc1");
    assert_eq!(allocator.leak_detector().outstanding_allocs(), 1);

    let (buf2, _offset2) = allocator.alloc(8192).expect("alloc2");
    assert_eq!(allocator.leak_detector().outstanding_allocs(), 2);

    allocator.free(buf1).expect("free1");
    assert_eq!(allocator.leak_detector().outstanding_allocs(), 1);

    allocator.free(buf2).expect("free2");
    assert_eq!(allocator.leak_detector().outstanding_allocs(), 0);
    assert_eq!(allocator.leak_detector().outstanding_bytes(), 0);
}

#[test]
fn test_leak_detector_catches_leak() {
    let detector = LeakDetector::new();

    // Simulate allocating 2 MiB without freeing.
    detector.record_alloc(2 * 1024 * 1024);
    assert!(
        detector.has_potential_leak(),
        "detector should flag potential leak for >1 MiB outstanding"
    );

    let report = detector.report();
    assert_eq!(report.outstanding_allocs, 1);
    assert_eq!(report.outstanding_bytes, 2 * 1024 * 1024);
    assert!(report.potential_leak);

    // Free the allocation — leak flag should clear.
    detector.record_free(2 * 1024 * 1024);
    assert!(
        !detector.has_potential_leak(),
        "detector should clear after freeing all allocations"
    );
}
