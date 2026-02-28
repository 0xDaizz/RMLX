//! rmlx-metal — Metal GPU abstraction layer for RMLX

pub mod buffer;
pub mod command;
pub mod device;
pub mod library;
pub mod pipeline;
pub mod queue;

// Re-export metal crate for downstream users
pub use metal;

/// Errors from Metal operations
#[derive(Debug)]
pub enum MetalError {
    NoDevice,
    ShaderCompile(String),
    PipelineCreate(String),
    LibraryLoad(String),
    KernelNotFound(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device found"),
            Self::ShaderCompile(e) => write!(f, "shader compile error: {e}"),
            Self::PipelineCreate(e) => write!(f, "pipeline creation error: {e}"),
            Self::LibraryLoad(e) => write!(f, "library load error: {e}"),
            Self::KernelNotFound(name) => write!(f, "kernel function not found: {name}"),
        }
    }
}

impl std::error::Error for MetalError {}
