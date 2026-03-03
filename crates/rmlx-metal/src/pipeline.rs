//! Metal compute pipeline state
//!
//! Provides a thread-safe pipeline cache that can be shared across threads
//! via `Arc<PipelineCache>`.  Supports both plain kernel functions and
//! functions specialized with Metal function constants.

use metal::{ComputePipelineState, FunctionConstantValues, Library, MTLDataType};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

use crate::MetalError;

// ---------------------------------------------------------------------------
// Function constant value type
// ---------------------------------------------------------------------------

/// A typed value for a Metal function constant.
///
/// Used with [`PipelineCache::get_or_create_specialized`] to create
/// pipeline states from functions that use `[[function_constant(N)]]`
/// attributes in MSL.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionConstant {
    Bool(bool),
    U32(u32),
    F32(f32),
}

impl Eq for FunctionConstant {}

impl std::hash::Hash for FunctionConstant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            FunctionConstant::Bool(v) => v.hash(state),
            FunctionConstant::U32(v) => v.hash(state),
            FunctionConstant::F32(v) => v.to_bits().hash(state),
        }
    }
}

// ---------------------------------------------------------------------------
// Cache key for specialized pipelines
// ---------------------------------------------------------------------------

/// Cache key combining function name and constant values.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SpecializedKey {
    name: String,
    constants: Vec<(u32, FunctionConstant)>,
}

// ---------------------------------------------------------------------------
// PipelineCache
// ---------------------------------------------------------------------------

/// Thread-safe cache for compiled compute pipeline states, keyed by kernel
/// function name (and optionally function constant values).
///
/// Uses a `RwLock<HashMap<...>>` internally so that cache hits (the common
/// case) only take a read lock, while cache misses take a write lock.
///
/// All public methods take `&self`, making it safe to share across threads
/// via `Arc<PipelineCache>`.
pub struct PipelineCache {
    device: metal::Device,
    cache: RwLock<HashMap<String, ComputePipelineState>>,
    specialized_cache: RwLock<HashMap<SpecializedKey, ComputePipelineState>>,
}

// SAFETY: `metal::Device` and `ComputePipelineState` are Objective-C objects
// that are internally reference-counted and thread-safe.  The `RwLock`
// provides the Rust-side synchronization guarantees.
unsafe impl Send for PipelineCache {}
unsafe impl Sync for PipelineCache {}

impl PipelineCache {
    /// Create a new empty pipeline cache for the given device.
    pub fn new(device: &metal::Device) -> Self {
        Self {
            device: device.clone(),
            cache: RwLock::new(HashMap::new()),
            specialized_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get a cached pipeline or create one from the named kernel function.
    ///
    /// If the pipeline for `name` was already compiled, returns a clone from
    /// the cache (a cheap Objective-C retain).  Otherwise, looks up the
    /// function in `library`, compiles a pipeline, caches it, and returns it.
    ///
    /// Takes `&self` so the cache can be shared across threads.
    pub fn get_or_create(
        &self,
        name: &str,
        library: &Library,
    ) -> Result<ComputePipelineState, MetalError> {
        // Fast path: read lock for cache hit.
        {
            let cache = self.cache.read().map_err(|_| {
                MetalError::PipelineCreate("pipeline cache lock poisoned".to_string())
            })?;
            if let Some(pipeline) = cache.get(name) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: write lock for cache miss.
        let mut cache = self.cache.write().map_err(|_| {
            MetalError::PipelineCreate("pipeline cache lock poisoned".to_string())
        })?;

        // Double-check: another thread may have inserted while we waited.
        if let Some(pipeline) = cache.get(name) {
            return Ok(pipeline.clone());
        }

        let function = library
            .get_function(name, None)
            .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| MetalError::PipelineCreate(err.to_string()))?;

        let cloned = pipeline.clone();
        cache.insert(name.to_string(), pipeline);
        Ok(cloned)
    }

    /// Get or create a pipeline specialized with function constant values.
    ///
    /// If `constants` is empty, delegates to [`Self::get_or_create`].
    /// Otherwise, creates `MTLFunctionConstantValues`, sets each entry, and
    /// compiles a specialized function via `newFunctionWithName:constantValues:`.
    ///
    /// The cache key includes both the function name and the constant values,
    /// so different specializations of the same kernel are cached separately.
    pub fn get_or_create_specialized(
        &self,
        name: &str,
        library: &Library,
        constants: &[(u32, FunctionConstant)],
    ) -> Result<ComputePipelineState, MetalError> {
        // Fast path: no constants -> use the plain cache.
        if constants.is_empty() {
            return self.get_or_create(name, library);
        }

        let key = SpecializedKey {
            name: name.to_string(),
            constants: constants.to_vec(),
        };

        // Fast path: read lock for cache hit.
        {
            let cache = self.specialized_cache.read().map_err(|_| {
                MetalError::PipelineCreate("specialized cache lock poisoned".to_string())
            })?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: build the specialized function.
        let mut cache = self.specialized_cache.write().map_err(|_| {
            MetalError::PipelineCreate("specialized cache lock poisoned".to_string())
        })?;

        // Double-check after acquiring write lock.
        if let Some(pipeline) = cache.get(&key) {
            return Ok(pipeline.clone());
        }

        let fcv = FunctionConstantValues::new();
        for (index, constant) in constants {
            match constant {
                FunctionConstant::Bool(v) => {
                    let val: u8 = u8::from(*v);
                    fcv.set_constant_value_at_index(
                        &val as *const u8 as *const c_void,
                        MTLDataType::Bool,
                        *index as u64,
                    );
                }
                FunctionConstant::U32(v) => {
                    fcv.set_constant_value_at_index(
                        v as *const u32 as *const c_void,
                        MTLDataType::UInt,
                        *index as u64,
                    );
                }
                FunctionConstant::F32(v) => {
                    fcv.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        MTLDataType::Float,
                        *index as u64,
                    );
                }
            }
        }

        let function = library
            .get_function(name, Some(fcv))
            .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| MetalError::PipelineCreate(err.to_string()))?;

        let cloned = pipeline.clone();
        cache.insert(key, pipeline);
        Ok(cloned)
    }

    /// Check whether a pipeline for `name` is already cached (plain cache only).
    pub fn contains(&self, name: &str) -> bool {
        self.cache
            .read()
            .map(|c| c.contains_key(name))
            .unwrap_or(false)
    }

    /// Number of cached pipelines (plain + specialized).
    pub fn len(&self) -> usize {
        let plain = self.cache.read().map(|c| c.len()).unwrap_or(0);
        let specialized = self
            .specialized_cache
            .read()
            .map(|c| c.len())
            .unwrap_or(0);
        plain + specialized
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_constant_hash_equality() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(c: &FunctionConstant) -> u64 {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        }

        // Same values should hash equally.
        assert_eq!(
            hash_of(&FunctionConstant::Bool(true)),
            hash_of(&FunctionConstant::Bool(true))
        );
        assert_eq!(
            hash_of(&FunctionConstant::U32(42)),
            hash_of(&FunctionConstant::U32(42))
        );
        assert_eq!(
            hash_of(&FunctionConstant::F32(1.0)),
            hash_of(&FunctionConstant::F32(1.0))
        );

        // Different values should (almost certainly) hash differently.
        assert_ne!(
            hash_of(&FunctionConstant::Bool(true)),
            hash_of(&FunctionConstant::Bool(false))
        );
        assert_ne!(
            hash_of(&FunctionConstant::U32(1)),
            hash_of(&FunctionConstant::F32(1.0))
        );
    }

    #[test]
    fn test_specialized_key_equality() {
        let k1 = SpecializedKey {
            name: "my_kernel".to_string(),
            constants: vec![(0, FunctionConstant::Bool(true))],
        };
        let k2 = SpecializedKey {
            name: "my_kernel".to_string(),
            constants: vec![(0, FunctionConstant::Bool(true))],
        };
        let k3 = SpecializedKey {
            name: "my_kernel".to_string(),
            constants: vec![(0, FunctionConstant::Bool(false))],
        };

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_pipeline_cache_new_empty() {
        let device = metal::Device::system_default().unwrap();
        let cache = PipelineCache::new(&device);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
