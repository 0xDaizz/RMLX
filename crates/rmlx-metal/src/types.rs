//! Type aliases for objc2-metal protocol objects.
//!
//! The objc2 ecosystem uses `Retained<ProtocolObject<dyn MTLFoo>>` for Metal
//! protocol types. These aliases keep call-sites concise.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

/// Owned Metal device handle.
pub type MtlDevice = Retained<ProtocolObject<dyn MTLDevice>>;

/// Owned command queue.
pub type MtlQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Owned GPU buffer.
pub type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Owned compute pipeline state.
pub type MtlPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Owned shader library.
pub type MtlLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Owned command buffer.
pub type MtlCB = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// Owned shared event (for cross-queue GPU synchronization).
pub type MtlEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// Owned compute command encoder.
pub type MtlEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Owned function constant values.
pub type MtlFunctionConstants = Retained<MTLFunctionConstantValues>;

/// Owned compile options.
pub type MtlCompileOptions = Retained<MTLCompileOptions>;

/// Owned Metal function.
pub type MtlFunction = Retained<ProtocolObject<dyn MTLFunction>>;

/// Owned capture manager.
pub type MtlCaptureManager = Retained<MTLCaptureManager>;

/// Owned capture descriptor.
pub type MtlCaptureDescriptor = Retained<MTLCaptureDescriptor>;
