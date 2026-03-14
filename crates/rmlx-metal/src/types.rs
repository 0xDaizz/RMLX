//! Type aliases for objc2-metal protocol objects.
//!
//! The objc2 ecosystem uses `Retained<ProtocolObject<dyn MTLFoo>>` for Metal
//! protocol types. These aliases keep call-sites concise.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::Message;
use objc2_metal::*;

/// Retain an unsized protocol-object reference into an owned `Retained`.
///
/// `Retained::retain()` requires `T: Sized` (it lives in `impl<T: Message>`),
/// but `ProtocolObject<dyn MTLFoo>` is unsized.  We work around this by
/// calling `objc_retain` directly and wrapping the result with `from_raw`,
/// which *is* available for `T: ?Sized + Message`.
pub(crate) fn retain_proto<T: ?Sized + Message>(r: &T) -> Retained<T> {
    let ptr = r as *const T as *mut T;
    // SAFETY: `ptr` originates from a valid reference.  `objc_retain` increments
    // the refcount; `from_raw` then wraps the +1 pointer without retaining again.
    unsafe {
        objc2::ffi::objc_retain(ptr as *mut _);
        Retained::from_raw(ptr).unwrap_unchecked()
    }
}

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
