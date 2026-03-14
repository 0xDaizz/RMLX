//! Metal Shading Language (MSL) version detection (M11).
//!
//! Auto-detects the highest supported MSL version based on the GPU
//! architecture (Apple Silicon generation). Optionally disables fast math
//! for numerical precision.
//!
//! # MSL version mapping
//!
//! | Apple GPU | GPU Family         | Max MSL |
//! |-----------|--------------------|---------|
//! | M1 (G15)  | Apple7             | 2.4     |
//! | M2 (G16)  | Apple8             | 3.1     |
//! | M3 (G17)  | Apple9             | 3.1     |
//! | M4 (G18)  | Apple9 (extended)  | 3.2     |

use crate::device::Architecture;

/// Metal Shading Language version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MslVersion {
    /// MSL 2.4 (M1, A14+)
    V2_4,
    /// MSL 3.1 (M2, M3)
    V3_1,
    /// MSL 3.2 (M4)
    V3_2,
}

impl MslVersion {
    /// Major.minor pair for display/comparison.
    pub fn major_minor(self) -> (u32, u32) {
        match self {
            MslVersion::V2_4 => (2, 4),
            MslVersion::V3_1 => (3, 1),
            MslVersion::V3_2 => (3, 2),
        }
    }
}

impl std::fmt::Display for MslVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor) = self.major_minor();
        write!(f, "MSL {major}.{minor}")
    }
}

/// Extended device information including MSL version and compile options.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Detected MSL version.
    pub msl_version: MslVersion,
    /// Whether fast math is disabled for numerical precision.
    pub fast_math_disabled: bool,
    /// GPU architecture.
    pub architecture: Architecture,
    /// Device name string.
    pub device_name: String,
}

/// Detect the highest supported MSL version based on GPU architecture.
pub fn detect_msl_version(arch: Architecture) -> MslVersion {
    match arch {
        Architecture::Apple { generation } => {
            if generation >= 18 {
                MslVersion::V3_2
            } else if generation >= 16 {
                MslVersion::V3_1
            } else {
                MslVersion::V2_4
            }
        }
        Architecture::Unknown => MslVersion::V2_4,
    }
}

/// Build a [`DeviceInfo`] from a [`GpuDevice`](crate::device::GpuDevice).
///
/// Detects the MSL version and disables fast math by default for
/// numerical precision (important for ML workloads with mixed precision).
pub fn build_device_info(device: &crate::device::GpuDevice) -> DeviceInfo {
    let arch = device.architecture();
    DeviceInfo {
        msl_version: detect_msl_version(arch),
        fast_math_disabled: true, // Default: disable fast math for precision.
        architecture: arch,
        device_name: device.name().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_msl_version_m1() {
        let v = detect_msl_version(Architecture::Apple { generation: 15 });
        assert_eq!(v, MslVersion::V2_4);
    }

    #[test]
    fn test_detect_msl_version_m2() {
        let v = detect_msl_version(Architecture::Apple { generation: 16 });
        assert_eq!(v, MslVersion::V3_1);
    }

    #[test]
    fn test_detect_msl_version_m3() {
        let v = detect_msl_version(Architecture::Apple { generation: 17 });
        assert_eq!(v, MslVersion::V3_1);
    }

    #[test]
    fn test_detect_msl_version_m4() {
        let v = detect_msl_version(Architecture::Apple { generation: 18 });
        assert_eq!(v, MslVersion::V3_2);
    }

    #[test]
    fn test_detect_msl_version_unknown() {
        let v = detect_msl_version(Architecture::Unknown);
        assert_eq!(v, MslVersion::V2_4);
    }

    #[test]
    fn test_msl_version_ordering() {
        assert!(MslVersion::V2_4 < MslVersion::V3_1);
        assert!(MslVersion::V3_1 < MslVersion::V3_2);
    }

    #[test]
    fn test_msl_version_display() {
        assert_eq!(MslVersion::V2_4.to_string(), "MSL 2.4");
        assert_eq!(MslVersion::V3_1.to_string(), "MSL 3.1");
        assert_eq!(MslVersion::V3_2.to_string(), "MSL 3.2");
    }

    #[test]
    fn test_build_device_info() {
        let Ok(dev) = crate::device::GpuDevice::system_default() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let info = build_device_info(&dev);
        assert!(info.fast_math_disabled);
        assert!(!info.device_name.is_empty());
    }
}
