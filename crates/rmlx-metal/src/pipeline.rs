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
        use std::collections::hash_map::Entry;
        match self.cache.entry(name.to_string()) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let function = library
                    .get_function(name, None)
                    .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

                let pipeline = self
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| MetalError::PipelineCreate(err.to_string()))?;

                Ok(e.insert(pipeline))
            }
        }
    }
}
