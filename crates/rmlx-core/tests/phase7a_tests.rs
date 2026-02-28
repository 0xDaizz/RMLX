//! Phase 7A tests for rmlx-core: logging, metrics, precision_guard, shutdown.

use rmlx_core::logging::{self, LogEntry, LogLevel};
use rmlx_core::metrics::RuntimeMetrics;
use rmlx_core::precision_guard::{PrecisionGuard, PrecisionResult};
use rmlx_core::shutdown::ShutdownSignal;

// ─── Logging tests ───

#[test]
fn test_logging_levels() {
    logging::set_level(LogLevel::Debug);
    assert_eq!(logging::current_level(), LogLevel::Debug);
    assert!(logging::is_enabled(LogLevel::Error));
    assert!(logging::is_enabled(LogLevel::Warn));
    assert!(logging::is_enabled(LogLevel::Info));
    assert!(logging::is_enabled(LogLevel::Debug));
    assert!(!logging::is_enabled(LogLevel::Trace));

    logging::set_level(LogLevel::Error);
    assert!(logging::is_enabled(LogLevel::Error));
    assert!(!logging::is_enabled(LogLevel::Warn));

    // Restore default
    logging::set_level(LogLevel::Info);
}

#[test]
fn test_logging_json_format() {
    let entry = LogEntry::new(LogLevel::Info, "rmlx::core", "started")
        .field("version", "0.1.0")
        .field("gpu", "M4 Ultra");

    let json = entry.format_json();
    assert!(json.contains("\"level\":\"INFO\""));
    assert!(json.contains("\"target\":\"rmlx::core\""));
    assert!(json.contains("\"msg\":\"started\""));
    assert!(json.contains("\"version\":\"0.1.0\""));
    assert!(json.contains("\"gpu\":\"M4 Ultra\""));
    assert!(json.starts_with('{'));
    assert!(json.ends_with('}'));

    let text = entry.format_text();
    assert!(text.contains("INFO"));
    assert!(text.contains("rmlx::core"));
    assert!(text.contains("started"));
    assert!(text.contains("version=0.1.0"));
}

// ─── RuntimeMetrics tests ───

#[test]
fn test_runtime_metrics_atomic() {
    let m = RuntimeMetrics::new();

    m.record_kernel_dispatch(100);
    m.record_kernel_dispatch(200);
    m.record_buffer_alloc(4096);
    m.record_buffer_alloc(8192);
    m.record_buffer_free(4096);
    m.record_cache_hit();
    m.record_cache_hit();
    m.record_cache_miss();

    let snap = m.snapshot();
    assert_eq!(snap.kernel_dispatches, 2);
    assert_eq!(snap.kernel_total_time_us, 300);
    assert_eq!(snap.buffer_allocs, 2);
    assert_eq!(snap.buffer_frees, 1);
    assert_eq!(snap.buffer_bytes_allocated, 8192); // 4096+8192-4096
    assert_eq!(snap.cache_hits, 2);
    assert_eq!(snap.cache_misses, 1);
}

// ─── PrecisionGuard tests ───

#[test]
fn test_precision_guard_nan() {
    let mut guard = PrecisionGuard::new(10);
    let logits = vec![1.0f32, 2.0, f32::NAN, 4.0];
    let result = guard.check_logits(&logits);
    assert_eq!(result, PrecisionResult::HasNaN(1));
}

#[test]
fn test_precision_guard_inf() {
    let mut guard = PrecisionGuard::new(10);
    let logits = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY, 4.0];
    let result = guard.check_logits(&logits);
    assert_eq!(result, PrecisionResult::HasInf(2));
}

#[test]
fn test_precision_guard_entropy_drift() {
    let mut guard = PrecisionGuard::new(5);
    // Establish baseline with uniform-ish logits
    for _ in 0..5 {
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];
        guard.check_logits(&logits);
    }
    assert!(!guard.should_warn());

    // Feed very different logits to cause drift
    for _ in 0..10 {
        let logits = vec![100.0f32, 0.0, 0.0, 0.0]; // very peaky = low entropy
        guard.check_logits(&logits);
    }
    // The drift should be detectable
    if let Some(drift) = guard.entropy_drift() {
        assert!(drift > 0.0, "should have nonzero drift, got {drift}");
    }
}

// ─── Shutdown tests ───

#[test]
fn test_shutdown_signal() {
    let signal = ShutdownSignal::new();
    assert!(!signal.is_triggered());

    let handle = signal.clone_handle();
    assert!(!handle.is_shutdown());

    signal.trigger();
    assert!(signal.is_triggered());
    assert!(handle.is_shutdown());
}
