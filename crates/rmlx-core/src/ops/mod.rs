//! GPU kernel operations for RMLX arrays.

pub mod binary;
pub mod copy;
pub mod gemv;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod softmax;

use crate::kernels::{KernelError, KernelRegistry};

/// Register all built-in kernels with the given registry.
pub fn register_all(registry: &KernelRegistry) -> Result<(), KernelError> {
    copy::register(registry)?;
    binary::register(registry)?;
    reduce::register(registry)?;
    rms_norm::register(registry)?;
    softmax::register(registry)?;
    rope::register(registry)?;
    gemv::register(registry)?;
    Ok(())
}
