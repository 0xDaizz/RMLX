//! Metal compute pipeline state

use metal::{ComputePipelineState, Library};
use std::collections::HashMap;

use crate::MetalError;

/// Cache for compiled compute pipeline states, keyed by kernel function name.
///
/// Avoids redundant pipeline compilation when the same kernel is dispatched
/// repeatedly (which is the common case in ML inference).
pub struct PipelineCache {
    device: metal::Device,
    cache: HashMap<String, ComputePipelineState>,
}

impl PipelineCache {
    /// Create a new empty pipeline cache for the given device.
    pub fn new(device: &metal::Device) -> Self {
        Self {
            device: device.clone(),
            cache: HashMap::new(),
        }
    }

    /// Get a cached pipeline or create one from the named kernel function.
    ///
    /// If the pipeline for `name` was already compiled, returns it from cache.
    /// Otherwise, looks up the function in `library`, compiles a pipeline, and
    /// caches it.
    pub fn get_or_create(
        &mut self,
        name: &str,
        library: &Library,
    ) -> Result<&ComputePipelineState, MetalError> {
        if !self.cache.contains_key(name) {
            let function = library
                .get_function(name, None)
                .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreate(e.to_string()))?;

            self.cache.insert(name.to_string(), pipeline);
        }

        Ok(self.cache.get(name).expect("just inserted"))
    }
}
