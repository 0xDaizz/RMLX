//! Metal shader library loading

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;
use std::path::Path;

use crate::types::*;
use crate::MetalError;

/// Load a pre-compiled `.metallib` file from disk.
pub fn load_metallib(
    device: &ProtocolObject<dyn MTLDevice>,
    path: &Path,
) -> Result<MtlLibrary, MetalError> {
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    device
        .newLibraryWithURL_error(&url)
        .map_err(|e| MetalError::LibraryLoad(e.localizedDescription().to_string()))
}

/// Compile a Metal shader from MSL source string.
///
/// Primarily useful for testing and JIT compilation. Production code should
/// prefer pre-compiled `.metallib` files via [`load_metallib`].
pub fn compile_source(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<MtlLibrary, MetalError> {
    let options = MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))
}
