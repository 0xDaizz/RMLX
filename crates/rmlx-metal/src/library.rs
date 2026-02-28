//! Metal shader library loading

use metal::{CompileOptions, Library};
use std::path::Path;

use crate::MetalError;

/// Load a pre-compiled `.metallib` file from disk.
pub fn load_metallib(device: &metal::Device, path: &Path) -> Result<Library, MetalError> {
    device
        .new_library_with_file(path)
        .map_err(|e| MetalError::LibraryLoad(e.to_string()))
}

/// Compile a Metal shader from MSL source string.
///
/// Primarily useful for testing and JIT compilation. Production code should
/// prefer pre-compiled `.metallib` files via [`load_metallib`].
pub fn compile_source(device: &metal::Device, source: &str) -> Result<Library, MetalError> {
    let options = CompileOptions::new();
    device
        .new_library_with_source(source, &options)
        .map_err(|e| MetalError::ShaderCompile(e.to_string()))
}
