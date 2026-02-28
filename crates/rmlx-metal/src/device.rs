//! Metal device abstraction

use metal::Device as MTLDevice;
use metal::{CommandQueue, MTLResourceOptions};

use crate::MetalError;

/// GPU architecture information derived from device name.
#[derive(Debug, Clone, Copy)]
pub enum Architecture {
    /// Apple Silicon GPU with generation number (M1=15, M2=16, M3=17, M4=18).
    Apple { generation: u32 },
    /// Unknown or unrecognized architecture.
    Unknown,
}

/// Thin wrapper around a Metal device that caches capability queries.
pub struct GpuDevice {
    device: MTLDevice,
    arch: Architecture,
    max_buffer_length: u64,
    max_threadgroup_memory: u64,
}

impl GpuDevice {
    /// Acquire the system default Metal device.
    pub fn system_default() -> Result<Self, MetalError> {
        let device = MTLDevice::system_default().ok_or(MetalError::NoDevice)?;
        let arch = detect_architecture(device.name());
        let max_buffer_length = device.max_buffer_length();
        let max_threadgroup_memory = device.max_threadgroup_memory_length();

        Ok(Self {
            device,
            arch,
            max_buffer_length,
            max_threadgroup_memory,
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
    /// Uses `StorageModeShared` so the buffer is accessible to both CPU and GPU.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> metal::Buffer {
        let size = std::mem::size_of_val(data) as u64;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        self.device
            .new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared)
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
}
