//! Metal library caching by source hash (M10).
//!
//! Compiling MSL source into a `MTLLibrary` is expensive (~5-50 ms per compile).
//! [`LibraryCache`] caches compiled libraries keyed by a 64-bit hash of the
//! source string, using `RwLock` for thread-safe concurrent reads.
//!
//! This is complementary to [`PipelineCache`](crate::pipeline::PipelineCache),
//! which caches individual pipeline states within a library.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::types::*;
use crate::MetalError;

/// Thread-safe cache for compiled Metal shader libraries, keyed by
/// a 64-bit hash of the MSL source string.
///
/// Uses `RwLock<HashMap>` so concurrent cache hits only take a read lock.
pub struct LibraryCache {
    device: MtlDevice,
    cache: RwLock<FxHashMap<u64, MtlLibrary>>,
}

impl LibraryCache {
    /// Create a new empty library cache for the given device.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            device: retain_proto(device),
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    /// Get a cached library or compile from MSL source.
    ///
    /// If a library for this source was already compiled, returns a clone
    /// (cheap Obj-C retain). Otherwise, compiles the source, caches the
    /// result, and returns it.
    pub fn get_or_compile(&self, source: &str) -> Result<MtlLibrary, MetalError> {
        let key = hash_source(source);

        // Fast path: read lock for cache hit.
        {
            let cache = self.cache.read();
            if let Some(lib) = cache.get(&key) {
                return Ok(retain_proto(&**lib));
            }
        }

        // Slow path: compile and insert under write lock.
        let mut cache = self.cache.write();

        // Double-check after acquiring write lock.
        if let Some(lib) = cache.get(&key) {
            return Ok(retain_proto(&**lib));
        }

        let options = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let cloned = retain_proto(&*library);
        cache.insert(key, library);
        Ok(cloned)
    }

    /// Get a cached library by its source hash, without compiling.
    ///
    /// Returns `None` if the source has not been compiled yet.
    pub fn get(&self, source: &str) -> Option<MtlLibrary> {
        let key = hash_source(source);
        self.cache.read().get(&key).map(|lib| retain_proto(&**lib))
    }

    /// Whether a library for the given source is cached.
    pub fn contains(&self, source: &str) -> bool {
        let key = hash_source(source);
        self.cache.read().contains_key(&key)
    }

    /// Number of cached libraries.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached libraries.
    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

/// Hash MSL source string to a 64-bit key.
fn hash_source(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
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
    fn test_library_cache_new_empty() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = LibraryCache::new(device);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_library_cache_compile_and_retrieve() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = LibraryCache::new(device);

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_kernel(device float* out [[buffer(0)]],
                                    uint id [[thread_position_in_grid]]) {
                out[id] = float(id);
            }
        "#;

        assert!(!cache.contains(source));

        let lib = cache.get_or_compile(source).unwrap();
        assert!(cache.contains(source));
        assert_eq!(cache.len(), 1);

        // Second call should hit the cache.
        let lib2 = cache.get_or_compile(source).unwrap();
        // Both should be usable.
        let _ = lib
            .newFunctionWithName(&NSString::from_str("test_kernel"))
            .unwrap();
        let _ = lib2
            .newFunctionWithName(&NSString::from_str("test_kernel"))
            .unwrap();
    }

    #[test]
    fn test_library_cache_different_sources() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = LibraryCache::new(device);

        let source1 = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void k1(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                o[id] = 1.0;
            }
        "#;
        let source2 = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void k2(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                o[id] = 2.0;
            }
        "#;

        cache.get_or_compile(source1).unwrap();
        cache.get_or_compile(source2).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_library_cache_invalid_source() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = LibraryCache::new(device);

        let result = cache.get_or_compile("this is not valid MSL");
        assert!(result.is_err());
        assert_eq!(cache.len(), 0); // Should not cache failed compilations.
    }

    #[test]
    fn test_library_cache_clear() {
        let Some(device) = test_device() else {
            eprintln!("Skipping: no Metal GPU");
            return;
        };
        let cache = LibraryCache::new(device);

        let source = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void kc(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                o[id] = 0.0;
            }
        "#;
        cache.get_or_compile(source).unwrap();
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hash_source_deterministic() {
        let s = "kernel void foo() {}";
        let h1 = hash_source(s);
        let h2 = hash_source(s);
        assert_eq!(h1, h2);

        let h3 = hash_source("kernel void bar() {}");
        assert_ne!(h1, h3);
    }
}
