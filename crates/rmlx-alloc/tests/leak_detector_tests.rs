//! Phase 7A tests for rmlx-alloc: leak_detector.

use rmlx_alloc::leak_detector::LeakDetector;

#[test]
fn test_leak_detector_basic() {
    let d = LeakDetector::new();
    assert_eq!(d.outstanding_allocs(), 0);
    assert_eq!(d.outstanding_bytes(), 0);
    assert!(!d.has_potential_leak());

    d.record_alloc(1024);
    assert_eq!(d.outstanding_allocs(), 1);
    assert_eq!(d.outstanding_bytes(), 1024);

    d.record_free(1024);
    assert_eq!(d.outstanding_allocs(), 0);
    assert_eq!(d.outstanding_bytes(), 0);
}

#[test]
fn test_leak_detector_outstanding() {
    let d = LeakDetector::new();
    d.record_alloc(2 * 1024 * 1024); // 2 MiB
    d.record_alloc(1024);

    assert_eq!(d.outstanding_allocs(), 2);
    assert!(d.has_potential_leak()); // > 1 MiB outstanding

    d.record_free(2 * 1024 * 1024);
    assert_eq!(d.outstanding_allocs(), 1);
    assert!(!d.has_potential_leak()); // only 1024 bytes outstanding
}

#[test]
fn test_leak_detector_report() {
    let d = LeakDetector::new();
    d.record_alloc(4096);
    d.record_alloc(8192);
    d.record_free(4096);

    let report = d.report();
    assert_eq!(report.total_allocs, 2);
    assert_eq!(report.total_frees, 1);
    assert_eq!(report.outstanding_allocs, 1);
    assert_eq!(report.outstanding_bytes, 8192);
    assert!(report.high_water_mark_bytes >= 12288); // peak was 4096+8192
    assert!(!report.potential_leak); // only 8KiB < 1MiB threshold
}
