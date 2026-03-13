//! Startup self-check for Metal GPU capabilities.

use objc2_metal::*;

/// Result of running the Metal self-check.
#[derive(Debug, Clone)]
pub struct SelfCheckResult {
    pub metal_available: bool,
    pub metal_version: String,
    pub gpu_family: String,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory: u64,
    pub shared_memory_size: u64,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

impl SelfCheckResult {
    /// Whether the self-check passed without issues.
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Run a comprehensive Metal self-check.
pub fn run_self_check() -> SelfCheckResult {
    let metal_available = check_metal_support();
    let (max_buf, max_tg) = check_memory_limits();

    let mut issues = Vec::new();
    let mut warnings = Vec::new();

    if !metal_available {
        issues.push("Metal is not available on this system".to_string());
    }

    if max_buf == 0 {
        warnings.push("Could not determine max buffer length".to_string());
    }

    let (version, family, shared) = query_gpu_info();

    SelfCheckResult {
        metal_available,
        metal_version: version,
        gpu_family: family,
        max_buffer_length: max_buf,
        max_threadgroup_memory: max_tg,
        shared_memory_size: shared,
        issues,
        warnings,
    }
}

/// Check if Metal is available by trying to get the default device.
pub fn check_metal_support() -> bool {
    unsafe { MTLCreateSystemDefaultDevice() }.is_some()
}

/// Query max buffer length and max threadgroup memory from the default device.
pub fn check_memory_limits() -> (u64, u64) {
    match unsafe { MTLCreateSystemDefaultDevice() } {
        Some(device) => {
            let max_buf = device.maxBufferLength() as u64;
            let max_tg = device.maxThreadgroupMemoryLength() as u64;
            (max_buf, max_tg)
        }
        None => (0, 0),
    }
}

/// Query GPU info strings and shared memory size.
fn query_gpu_info() -> (String, String, u64) {
    match unsafe { MTLCreateSystemDefaultDevice() } {
        Some(device) => {
            let name = device.name().to_string();
            let recommended = device.recommendedMaxWorkingSetSize() as u64;
            // Use device name as a proxy for family/version
            (name.clone(), name, recommended)
        }
        None => ("unavailable".to_string(), "unavailable".to_string(), 0),
    }
}
