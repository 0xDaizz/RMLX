//! Metal compute pipeline state
//!
//! Provides a thread-safe pipeline cache that can be shared across threads
//! via `Arc<PipelineCache>`.  Supports both plain kernel functions and
//! functions specialized with Metal function constants.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::types::*;
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
/// Uses a `RwLock<FxHashMap<...>>` internally so that cache hits (the common
/// case) only take a read lock, while cache misses take a write lock.
///
/// All public methods take `&self`, making it safe to share across threads
/// via `Arc<PipelineCache>`.
pub struct PipelineCache {
    device: MtlDevice,
    cache: RwLock<FxHashMap<String, MtlPipeline>>,
    specialized_cache: RwLock<FxHashMap<SpecializedKey, MtlPipeline>>,
}

impl PipelineCache {
    /// Create a new empty pipeline cache for the given device.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            device: retain_proto(device),
            cache: RwLock::new(FxHashMap::default()),
            specialized_cache: RwLock::new(FxHashMap::default()),
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
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<MtlPipeline, MetalError> {
        // Fast path: read lock for cache hit.
        {
            let cache = self.cache.read();
            if let Some(pipeline) = cache.get(name) {
                return Ok(retain_proto(&**pipeline));
            }
        }

        // Slow path: write lock for cache miss.
        let mut cache = self.cache.write();

        // Double-check: another thread may have inserted while we waited.
        if let Some(pipeline) = cache.get(name) {
            return Ok(retain_proto(&**pipeline));
        }

        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| MetalError::KernelNotFound(name.to_string()))?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| MetalError::PipelineCreate(err.localizedDescription().to_string()))?;

        let cloned = retain_proto(&*pipeline);
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
        library: &ProtocolObject<dyn MTLLibrary>,
        constants: &[(u32, FunctionConstant)],
    ) -> Result<MtlPipeline, MetalError> {
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
            let cache = self.specialized_cache.read();
            if let Some(pipeline) = cache.get(&key) {
                return Ok(retain_proto(&**pipeline));
            }
        }

        // Slow path: build the specialized function.
        let mut cache = self.specialized_cache.write();

        // Double-check after acquiring write lock.
        if let Some(pipeline) = cache.get(&key) {
            return Ok(retain_proto(&**pipeline));
        }

        let fcv = MTLFunctionConstantValues::new();
        apply_function_constants(&fcv, constants);

        let function = library
            .newFunctionWithName_constantValues_error(&NSString::from_str(name), &fcv)
            .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| MetalError::PipelineCreate(err.localizedDescription().to_string()))?;

        let cloned = retain_proto(&*pipeline);
        cache.insert(key, pipeline);
        Ok(cloned)
    }

    /// Check whether a pipeline for `name` is already cached (plain cache only).
    pub fn contains(&self, name: &str) -> bool {
        self.cache.read().contains_key(name)
    }

    /// Number of cached pipelines (plain + specialized).
    pub fn len(&self) -> usize {
        let plain = self.cache.read().len();
        let specialized = self.specialized_cache.read().len();
        plain + specialized
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Shared helper: apply function constants to MTLFunctionConstantValues
// ---------------------------------------------------------------------------

/// Apply a slice of `(index, FunctionConstant)` to a
/// `MTLFunctionConstantValues` object.
///
/// Encapsulates the `unsafe` calls to `setConstantValue_type_atIndex` which
/// require a raw `NonNull<c_void>` pointer.  The pointer is always derived
/// from a local stack value whose lifetime covers the FFI call, so this is
/// safe to call from safe Rust.
pub(crate) fn apply_function_constants(
    fcv: &MTLFunctionConstantValues,
    constants: &[(u32, FunctionConstant)],
) {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    for (index, constant) in constants {
        match constant {
            FunctionConstant::Bool(v) => {
                let val: u8 = u8::from(*v);
                // SAFETY: `&val` is a valid, aligned pointer to a stack-local u8.
                // The Obj-C method copies the value immediately.
                unsafe {
                    fcv.setConstantValue_type_atIndex(
                        NonNull::new(&val as *const u8 as *mut c_void).unwrap(),
                        MTLDataType::Bool,
                        *index as usize,
                    );
                }
            }
            FunctionConstant::U32(v) => {
                // SAFETY: `v` is a valid reference to a u32 from the slice.
                // The Obj-C method copies the value immediately.
                unsafe {
                    fcv.setConstantValue_type_atIndex(
                        NonNull::new(v as *const u32 as *mut c_void).unwrap(),
                        MTLDataType::UInt,
                        *index as usize,
                    );
                }
            }
            FunctionConstant::F32(v) => {
                // SAFETY: `v` is a valid reference to a f32 from the slice.
                // The Obj-C method copies the value immediately.
                unsafe {
                    fcv.setConstantValue_type_atIndex(
                        NonNull::new(v as *const f32 as *mut c_void).unwrap(),
                        MTLDataType::Float,
                        *index as usize,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    use std::sync::OnceLock;

    fn test_device() -> Option<&'static MtlDevice> {
        static DEVICE: OnceLock<Option<MtlDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(|| objc2::rc::autoreleasepool(|_| MTLCreateSystemDefaultDevice()))
            .as_ref()
    }

    #[test]
    fn test_function_constant_hash_equality() {
        use rustc_hash::FxHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(c: &FunctionConstant) -> u64 {
            let mut h = FxHasher::default();
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
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = PipelineCache::new(device);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
