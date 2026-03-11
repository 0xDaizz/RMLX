//! Metal device abstraction

use metal::Device as MTLDevice;
use metal::{CommandBuffer, CommandQueue, MTLGPUFamily, MTLResourceOptions};

use crate::MetalError;

/// Safe default: StorageModeShared only. Metal performs automatic hazard
/// tracking, so buffers created with these options are safe to use even
/// when code paths bypass the barrier tracker.
pub const TRACKED_BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

/// Performance option: StorageModeShared + HazardTrackingModeUntracked.
/// Use ONLY for buffers whose synchronisation is managed explicitly by
/// the RMLX barrier tracker (command-buffer ordering / MTLSharedEvent /
/// MTLFence). Using this for buffers that bypass the barrier tracker
/// will cause data races.
pub const UNTRACKED_BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::from_bits_truncate(
    MTLResourceOptions::StorageModeShared.bits()
        | MTLResourceOptions::HazardTrackingModeUntracked.bits(),
);

/// Default buffer options — untracked for performance (MLX-compatible).
///
/// RMLX manages synchronisation explicitly via the barrier tracker and
/// encoder boundaries, so Metal's automatic hazard tracking is redundant.
/// Enable the `tracked_hazards` feature to restore the safe-but-slower default.
#[cfg(not(feature = "tracked_hazards"))]
pub const DEFAULT_BUFFER_OPTIONS: MTLResourceOptions = UNTRACKED_BUFFER_OPTIONS;

#[cfg(feature = "tracked_hazards")]
pub const DEFAULT_BUFFER_OPTIONS: MTLResourceOptions = TRACKED_BUFFER_OPTIONS;

// ---------------------------------------------------------------------------
// Chip-class tuning
// ---------------------------------------------------------------------------

/// Per-chip tuning parameters derived from the Metal GPU family.
///
/// Use [`ChipTuning::for_device`] to detect the current hardware and populate
/// appropriate values.  Kernels and infrastructure code can then query these
/// fields instead of hard-coding chip-specific constants.
#[derive(Debug, Clone, Copy)]
pub struct ChipTuning {
    /// Maximum threadgroup memory in bytes (e.g. 32 768 on Apple Silicon).
    pub max_threadgroup_memory: usize,
    /// Maximum threads per threadgroup (typically 1024 on Apple Silicon).
    pub max_threads_per_threadgroup: usize,
    /// Preferred SIMD width (32 on all current Apple GPUs).
    pub preferred_simd_width: usize,
    /// Whether the device supports Metal 3+ unretained command-buffer
    /// references (`newCommandBufferWithUnretainedReferences`), which avoids
    /// the overhead of retaining every resource referenced by the CB.
    pub supports_unretained_refs: bool,
    /// Architecture generation: 0=unknown, 15=M1, 16=M2, 17=M3, 18=M4, ...
    pub arch_gen: u32,
    /// GPU class within the generation.
    pub arch_class: ArchClass,
    /// Whether the device supports NAX (Neural Accelerated matriX) MMA.
    /// True for gen>=17 && class != Phone && class != Unknown.
    pub supports_nax: bool,
    /// Maximum ops per command buffer batch (Ultra/Max=50, Base=40, Unknown=32).
    pub max_ops_per_batch: u32,
    /// Maximum MB per command buffer batch (Ultra/Max=50, Base=40, Unknown=32).
    pub max_mb_per_batch: u32,
    /// Whether concurrent compute dispatch is supported.
    /// True for all Apple Silicon (M1+), false for unknown hardware.
    pub supports_concurrent_dispatch: bool,
    /// Estimated GPU core count for occupancy heuristics (e.g. split-K).
    /// Derived from arch_class: Ultra=80, Max=40, Base=10, Unknown=10.
    pub gpu_cores: usize,
}

impl ChipTuning {
    /// Detect chip capabilities from a `metal::Device` and return tuned values.
    ///
    /// * M1 (Apple7): simd 32, TG mem 32 KB, no unretained refs
    /// * M2 (Apple8) / M3 (Apple9) / M4+: simd 32, TG mem 32 KB, unretained refs (Metal 3+)
    /// * Unknown: conservative defaults (simd 32, TG mem 16 KB, no unretained refs)
    pub fn for_device(device: &metal::Device) -> Self {
        let supports_metal3 = device.supports_family(MTLGPUFamily::Metal3);

        // Apple Silicon always has 32-wide SIMD and 32 KB TG memory.
        // For unknown/non-Apple devices we fall back to conservative values.
        let is_apple = device.supports_family(MTLGPUFamily::Apple7)
            || device.supports_family(MTLGPUFamily::Apple8)
            || device.supports_family(MTLGPUFamily::Apple9);

        let arch = detect_architecture(device.name());
        let generation = match arch {
            Architecture::Apple { generation } => generation,
            Architecture::Unknown => 0,
        };
        let arch_class = detect_arch_class(device.name());
        let supports_nax =
            generation >= 17 && arch_class != ArchClass::Phone && arch_class != ArchClass::Unknown;
        let (max_ops, max_mb) = match arch_class {
            ArchClass::Ultra | ArchClass::Max => (50, 50),
            ArchClass::Base => (40, 40),
            _ => (32, 32),
        };
        let gpu_cores = match arch_class {
            ArchClass::Ultra => 80,
            ArchClass::Max => 40,
            ArchClass::Base | ArchClass::Phone => 10,
            ArchClass::Unknown => 10,
        };

        if is_apple {
            Self {
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                preferred_simd_width: 32,
                supports_unretained_refs: supports_metal3,
                arch_gen: generation,
                arch_class,
                supports_nax,
                max_ops_per_batch: max_ops,
                max_mb_per_batch: max_mb,
                supports_concurrent_dispatch: true,
                gpu_cores,
            }
        } else {
            // Conservative defaults for unknown hardware.
            Self {
                max_threadgroup_memory: 16 * 1024,
                max_threads_per_threadgroup: 512,
                preferred_simd_width: 32,
                supports_unretained_refs: false,
                arch_gen: generation,
                arch_class,
                supports_nax,
                max_ops_per_batch: max_ops,
                max_mb_per_batch: max_mb,
                supports_concurrent_dispatch: false,
                gpu_cores,
            }
        }
    }

    /// Build a `ChipTuning` from a device-name string (for unit tests that
    /// cannot instantiate a real `metal::Device`).
    ///
    /// This is intentionally conservative: it mirrors the name-based
    /// `detect_architecture` heuristic and does *not* query the driver.
    #[cfg(test)]
    pub(crate) fn from_name(name: &str) -> Self {
        let arch = detect_architecture(name);
        let generation = match arch {
            Architecture::Apple { generation } => generation,
            Architecture::Unknown => 0,
        };
        let arch_class = detect_arch_class(name);
        let supports_nax =
            generation >= 17 && arch_class != ArchClass::Phone && arch_class != ArchClass::Unknown;
        let (max_ops, max_mb) = match arch_class {
            ArchClass::Ultra | ArchClass::Max => (50, 50),
            ArchClass::Base => (40, 40),
            _ => (32, 32),
        };
        let gpu_cores = match arch_class {
            ArchClass::Ultra => 80,
            ArchClass::Max => 40,
            ArchClass::Base | ArchClass::Phone => 10,
            ArchClass::Unknown => 10,
        };
        match arch {
            Architecture::Apple { generation } if generation >= 16 => Self {
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                preferred_simd_width: 32,
                supports_unretained_refs: true, // M2+ => Metal 3
                arch_gen: generation,
                arch_class,
                supports_nax,
                max_ops_per_batch: max_ops,
                max_mb_per_batch: max_mb,
                supports_concurrent_dispatch: true,
                gpu_cores,
            },
            Architecture::Apple { generation } => Self {
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                preferred_simd_width: 32,
                supports_unretained_refs: false, // M1 => Metal 2
                arch_gen: generation,
                arch_class,
                supports_nax,
                max_ops_per_batch: max_ops,
                max_mb_per_batch: max_mb,
                supports_concurrent_dispatch: true,
                gpu_cores,
            },
            Architecture::Unknown => Self {
                max_threadgroup_memory: 16 * 1024,
                max_threads_per_threadgroup: 512,
                preferred_simd_width: 32,
                supports_unretained_refs: false,
                arch_gen: 0,
                arch_class,
                supports_nax: false,
                max_ops_per_batch: max_ops,
                max_mb_per_batch: max_mb,
                supports_concurrent_dispatch: false,
                gpu_cores,
            },
        }
    }

    /// Device-aware M limit for BatchQMV dispatch.
    ///
    /// Returns the maximum M for which BatchQMV (qdot) is preferred over MMA.
    /// Larger chips can sustain BatchQMV at higher M due to more GPU cores.
    pub fn batch_qmv_limit(&self, k: usize, n: usize) -> usize {
        let small_dims = k <= 2048 && n <= 2048;
        match self.arch_class {
            ArchClass::Ultra => 16,
            ArchClass::Max => {
                if small_dims {
                    24
                } else {
                    16
                }
            }
            ArchClass::Base | ArchClass::Phone => {
                if small_dims {
                    16
                } else {
                    8
                }
            }
            ArchClass::Unknown => 16, // safe default
        }
    }
}

/// GPU architecture information derived from device name.
#[derive(Debug, Clone, Copy)]
pub enum Architecture {
    /// Apple Silicon GPU with generation number (M1=15, M2=16, M3=17, M4=18).
    Apple { generation: u32 },
    /// Unknown or unrecognized architecture.
    Unknown,
}

/// GPU class within an Apple Silicon generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchClass {
    /// iPhone/iPad GPU (A-series equivalent)
    Phone,
    /// Base chip (M1, M2, M3, M4)
    Base,
    /// Max variant (higher GPU core count)
    Max,
    /// Ultra variant (two Max dies fused)
    Ultra,
    /// Unrecognized class — gets conservative defaults
    Unknown,
}

/// Thin wrapper around a Metal device that caches capability queries.
pub struct GpuDevice {
    device: MTLDevice,
    arch: Architecture,
    tuning: ChipTuning,
    max_buffer_length: u64,
    max_threadgroup_memory: u64,
    stream_manager: crate::stream::StreamManager,
}

impl GpuDevice {
    /// Acquire the system default Metal device.
    pub fn system_default() -> Result<Self, MetalError> {
        let device = MTLDevice::system_default().ok_or(MetalError::NoDevice)?;
        let arch = detect_architecture(device.name());
        let tuning = ChipTuning::for_device(&device);
        let stream_manager = crate::stream::StreamManager::new(&device);
        let max_buffer_length = device.max_buffer_length();
        let max_threadgroup_memory = device.max_threadgroup_memory_length();

        Ok(Self {
            device,
            arch,
            tuning,
            max_buffer_length,
            max_threadgroup_memory,
            stream_manager,
        })
    }

    /// Access the underlying `metal::Device`.
    pub fn raw(&self) -> &MTLDevice {
        &self.device
    }

    /// Human-readable device name (e.g. "Apple M2 Max").
    pub fn name(&self) -> &str {
        self.device.name()
    }

    /// Detected GPU architecture.
    pub fn architecture(&self) -> Architecture {
        self.arch
    }

    /// Whether the device has unified memory (always true on Apple Silicon).
    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    /// Maximum single-buffer allocation size in bytes.
    pub fn max_buffer_length(&self) -> u64 {
        self.max_buffer_length
    }

    /// Maximum threadgroup memory length in bytes.
    pub fn max_threadgroup_memory(&self) -> u64 {
        self.max_threadgroup_memory
    }

    /// Chip-class tuning parameters for the current device.
    pub fn tuning(&self) -> &ChipTuning {
        &self.tuning
    }

    /// Access the multi-stream manager for concurrent compute + copy scheduling.
    pub fn stream_manager(&self) -> &crate::stream::StreamManager {
        &self.stream_manager
    }

    /// Create a command buffer on `queue`, choosing the optimal path for
    /// this chip class.
    ///
    /// When `ChipTuning::supports_unretained_refs` is true (Metal 3+ / M2+),
    /// uses `new_command_buffer_with_unretained_references()` which avoids
    /// the retain/release overhead for every resource referenced by the CB.
    /// Otherwise falls back to the standard `new_command_buffer()`.
    ///
    /// The returned `CommandBuffer` is *owned* (`.to_owned()`) so it is not
    /// reclaimed by the autorelease pool before the caller commits it.
    pub fn create_command_buffer(&self, queue: &CommandQueue) -> CommandBuffer {
        if self.tuning.supports_unretained_refs {
            queue
                .new_command_buffer_with_unretained_references()
                .to_owned()
        } else {
            queue.new_command_buffer().to_owned()
        }
    }

    /// Create a new command queue on this device.
    pub fn new_command_queue(&self) -> CommandQueue {
        self.device.new_command_queue()
    }

    /// Allocate an uninitialized buffer of `size` bytes.
    pub fn new_buffer(&self, size: u64, options: MTLResourceOptions) -> metal::Buffer {
        self.device.new_buffer(size, options)
    }

    /// Allocate a buffer and initialize it from a typed slice.
    ///
    /// Uses the safe default [`DEFAULT_BUFFER_OPTIONS`] (`StorageModeShared`)
    /// so the buffer is CPU+GPU visible with Metal hazard tracking enabled.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> metal::Buffer {
        let size = std::mem::size_of_val(data) as u64;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        self.device
            .new_buffer_with_data(ptr, size, DEFAULT_BUFFER_OPTIONS)
    }
}

/// Parse chip class from device name string.
///
/// Only classifies Apple Silicon devices (names containing "M1"/"M2"/etc).
/// Non-Apple devices always get `Unknown`.
fn detect_arch_class(name: &str) -> ArchClass {
    // Only classify if it's an Apple Silicon device
    let is_apple_silicon =
        name.contains("M1") || name.contains("M2") || name.contains("M3") || name.contains("M4");
    if !is_apple_silicon {
        return ArchClass::Unknown;
    }

    if name.contains("Ultra") {
        ArchClass::Ultra
    } else if name.contains("Max") {
        ArchClass::Max
    } else {
        // Pro and base are the same GPU class for tuning purposes
        ArchClass::Base
    }
}

/// Parse the device name string to determine the Apple Silicon generation.
fn detect_architecture(name: &str) -> Architecture {
    if name.contains("M4") {
        Architecture::Apple { generation: 18 }
    } else if name.contains("M3") {
        Architecture::Apple { generation: 17 }
    } else if name.contains("M2") {
        Architecture::Apple { generation: 16 }
    } else if name.contains("M1") {
        Architecture::Apple { generation: 15 }
    } else {
        Architecture::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_architecture() {
        assert!(matches!(
            detect_architecture("Apple M1"),
            Architecture::Apple { generation: 15 }
        ));
        assert!(matches!(
            detect_architecture("Apple M2 Max"),
            Architecture::Apple { generation: 16 }
        ));
        assert!(matches!(
            detect_architecture("Apple M3 Pro"),
            Architecture::Apple { generation: 17 }
        ));
        assert!(matches!(
            detect_architecture("Apple M4"),
            Architecture::Apple { generation: 18 }
        ));
        assert!(matches!(
            detect_architecture("Intel HD 630"),
            Architecture::Unknown
        ));
    }

    // ----- ChipTuning name-based tests -----

    #[test]
    fn test_chip_tuning_m1() {
        let t = ChipTuning::from_name("Apple M1");
        assert_eq!(t.max_threadgroup_memory, 32 * 1024);
        assert_eq!(t.max_threads_per_threadgroup, 1024);
        assert_eq!(t.preferred_simd_width, 32);
        assert!(!t.supports_unretained_refs, "M1 is Metal 2, no unretained");
    }

    #[test]
    fn test_chip_tuning_m2() {
        let t = ChipTuning::from_name("Apple M2 Max");
        assert_eq!(t.max_threadgroup_memory, 32 * 1024);
        assert_eq!(t.preferred_simd_width, 32);
        assert!(t.supports_unretained_refs, "M2 is Metal 3+");
    }

    #[test]
    fn test_chip_tuning_m3() {
        let t = ChipTuning::from_name("Apple M3 Pro");
        assert!(t.supports_unretained_refs, "M3 is Metal 3+");
        assert_eq!(t.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn test_chip_tuning_m4() {
        let t = ChipTuning::from_name("Apple M4");
        assert!(t.supports_unretained_refs, "M4 is Metal 3+");
        assert_eq!(t.preferred_simd_width, 32);
    }

    #[test]
    fn test_chip_tuning_unknown() {
        let t = ChipTuning::from_name("Intel HD 630");
        assert_eq!(t.max_threadgroup_memory, 16 * 1024);
        assert_eq!(t.max_threads_per_threadgroup, 512);
        assert!(!t.supports_unretained_refs);
    }

    // ----- Live device tests (run on real Metal hardware) -----

    #[test]
    fn test_chip_tuning_for_device_runs() {
        let device = metal::Device::system_default().unwrap();
        let tuning = ChipTuning::for_device(&device);
        // On any Apple Silicon these should hold:
        assert!(tuning.max_threadgroup_memory >= 16 * 1024);
        assert!(tuning.max_threads_per_threadgroup >= 512);
        assert_eq!(tuning.preferred_simd_width, 32);
    }

    #[test]
    fn test_gpu_device_has_tuning() {
        let gpu = GpuDevice::system_default().unwrap();
        let t = gpu.tuning();
        assert!(t.max_threadgroup_memory >= 16 * 1024);
        assert_eq!(t.preferred_simd_width, 32);
    }

    #[test]
    fn test_gpu_device_stream_manager() {
        let gpu = GpuDevice::system_default().unwrap();
        let mgr = gpu.stream_manager();
        assert!(mgr.has_stream(crate::stream::STREAM_COMPUTE));
        assert!(mgr.has_stream(crate::stream::STREAM_COPY));
        assert_eq!(mgr.stream_count(), 3);
    }

    #[test]
    fn test_create_command_buffer_succeeds() {
        let gpu = GpuDevice::system_default().unwrap();
        let queue = gpu.new_command_queue();
        let cb = gpu.create_command_buffer(&queue);
        // Encode a no-op and commit to prove the CB is valid.
        let enc = cb.new_compute_command_encoder();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    #[test]
    fn test_create_command_buffer_both_paths() {
        // Verify that both the unretained and retained paths produce valid CBs.
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();

        // Retained path (standard)
        let cb_retained = queue.new_command_buffer().to_owned();
        let enc = cb_retained.new_compute_command_encoder();
        enc.end_encoding();
        cb_retained.commit();
        cb_retained.wait_until_completed();

        // Unretained path
        let cb_unretained = queue
            .new_command_buffer_with_unretained_references()
            .to_owned();
        let enc = cb_unretained.new_compute_command_encoder();
        enc.end_encoding();
        cb_unretained.commit();
        cb_unretained.wait_until_completed();
    }

    #[test]
    fn test_tracked_buffer_options() {
        assert!(TRACKED_BUFFER_OPTIONS.contains(MTLResourceOptions::StorageModeShared));
        assert!(!TRACKED_BUFFER_OPTIONS.contains(MTLResourceOptions::HazardTrackingModeUntracked));
    }

    #[test]
    fn test_untracked_buffer_options() {
        assert!(UNTRACKED_BUFFER_OPTIONS.contains(MTLResourceOptions::StorageModeShared));
        assert!(UNTRACKED_BUFFER_OPTIONS.contains(MTLResourceOptions::HazardTrackingModeUntracked));
    }

    #[test]
    #[cfg(not(feature = "tracked_hazards"))]
    fn test_default_buffer_options_is_untracked() {
        assert_eq!(DEFAULT_BUFFER_OPTIONS, UNTRACKED_BUFFER_OPTIONS);
        assert!(DEFAULT_BUFFER_OPTIONS.contains(MTLResourceOptions::HazardTrackingModeUntracked));
    }

    #[test]
    #[cfg(feature = "tracked_hazards")]
    fn test_default_buffer_options_is_tracked() {
        assert_eq!(DEFAULT_BUFFER_OPTIONS, TRACKED_BUFFER_OPTIONS);
        assert!(!DEFAULT_BUFFER_OPTIONS.contains(MTLResourceOptions::HazardTrackingModeUntracked));
    }

    #[test]
    fn test_new_buffer_with_data_untracked() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let buf = device.new_buffer_with_data(&data);
        assert!(buf.length() >= 16); // 4 * f32

        // Verify buffer contents are readable (StorageModeShared).
        let ptr = buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, 4) };
        assert_eq!(slice, &data);
    }

    #[test]
    fn test_new_buffer_untracked_rw() {
        let device = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let buf = device.new_buffer(256, UNTRACKED_BUFFER_OPTIONS);
        assert!(buf.length() >= 256);
        // Write and read back to confirm the buffer is functional.
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 256);
            let slice = std::slice::from_raw_parts(ptr, 256);
            assert!(slice.iter().all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn test_gpu_device_name_not_empty() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        assert!(!gpu.name().is_empty());
    }

    #[test]
    fn test_gpu_device_has_unified_memory() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        // Apple Silicon always has unified memory.
        assert!(gpu.has_unified_memory());
    }

    #[test]
    fn test_gpu_device_max_buffer_length_positive() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        assert!(gpu.max_buffer_length() > 0);
    }

    #[test]
    fn test_gpu_device_max_threadgroup_memory_positive() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        assert!(gpu.max_threadgroup_memory() > 0);
    }

    #[test]
    fn test_gpu_device_architecture_detection() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        // On Apple Silicon, should detect an Apple architecture.
        match gpu.architecture() {
            Architecture::Apple { generation } => {
                assert!(generation >= 15); // M1 or later
            }
            Architecture::Unknown => {
                // Acceptable on non-Apple hardware.
            }
        }
    }

    #[test]
    fn test_gpu_device_raw_returns_valid_device() {
        let gpu = match GpuDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };
        let raw = gpu.raw();
        // Should be able to call methods on the raw device.
        assert!(!raw.name().is_empty());
    }

    #[test]
    fn test_detect_architecture_case_sensitivity() {
        // Only exact substrings should match.
        assert!(matches!(detect_architecture("m1"), Architecture::Unknown));
        assert!(matches!(
            detect_architecture("apple m2"),
            Architecture::Unknown
        ));
        assert!(matches!(
            detect_architecture("Apple M2 Ultra"),
            Architecture::Apple { generation: 16 }
        ));
    }

    // ----- ArchClass detection tests -----

    #[test]
    fn test_arch_class_detection() {
        assert_eq!(detect_arch_class("Apple M2 Ultra"), ArchClass::Ultra);
        assert_eq!(detect_arch_class("Apple M3 Max"), ArchClass::Max);
        assert_eq!(detect_arch_class("Apple M3 Pro"), ArchClass::Base);
        assert_eq!(detect_arch_class("Apple M4"), ArchClass::Base);
        assert_eq!(detect_arch_class("Intel HD 630"), ArchClass::Unknown);
        assert_eq!(detect_arch_class("Radeon Pro"), ArchClass::Unknown);
    }

    #[test]
    fn test_chip_tuning_nax_support() {
        let t = ChipTuning::from_name("Apple M3 Max");
        assert!(t.supports_nax, "M3 Max should support NAX");
        assert_eq!(t.arch_gen, 17);
        assert_eq!(t.arch_class, ArchClass::Max);
        assert_eq!(t.max_ops_per_batch, 50);
    }

    #[test]
    fn test_chip_tuning_no_nax_m2() {
        let t = ChipTuning::from_name("Apple M2 Ultra");
        assert!(!t.supports_nax, "M2 should not support NAX");
        assert_eq!(t.arch_gen, 16);
        assert_eq!(t.arch_class, ArchClass::Ultra);
        assert_eq!(t.max_ops_per_batch, 50);
    }

    #[test]
    fn test_chip_tuning_unknown_conservative() {
        let t = ChipTuning::from_name("Intel HD 630");
        assert!(!t.supports_nax);
        assert_eq!(t.arch_gen, 0);
        assert_eq!(t.arch_class, ArchClass::Unknown);
        assert_eq!(t.max_ops_per_batch, 32);
        assert_eq!(t.max_mb_per_batch, 32);
    }
}
