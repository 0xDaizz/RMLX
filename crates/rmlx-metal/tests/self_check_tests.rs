//! Phase 7A tests for rmlx-metal: self_check.

use rmlx_metal::self_check::SelfCheckResult;

#[test]
fn test_self_check_result() {
    // Test the SelfCheckResult struct API without requiring actual Metal device
    let result = SelfCheckResult {
        metal_available: true,
        metal_version: "Apple M4 Ultra".to_string(),
        gpu_family: "Apple M4 Ultra".to_string(),
        max_buffer_length: 16 * 1024 * 1024 * 1024,
        max_threadgroup_memory: 32768,
        shared_memory_size: 512 * 1024 * 1024 * 1024,
        issues: vec![],
        warnings: vec!["test warning".to_string()],
    };
    assert!(result.is_ok()); // no issues = ok
    assert!(result.metal_available);
    assert_eq!(result.warnings.len(), 1);

    let failed = SelfCheckResult {
        metal_available: false,
        metal_version: "unavailable".to_string(),
        gpu_family: "unavailable".to_string(),
        max_buffer_length: 0,
        max_threadgroup_memory: 0,
        shared_memory_size: 0,
        issues: vec!["Metal not available".to_string()],
        warnings: vec![],
    };
    assert!(!failed.is_ok());
    assert!(!failed.metal_available);
}
