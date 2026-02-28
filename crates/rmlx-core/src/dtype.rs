//! Data type definitions for RMLX tensors.

use std::fmt;

/// Supported data types for RMLX arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float16,
    Bfloat16,
    /// Unsigned 32-bit integer (for index arrays, token IDs, etc.)
    UInt32,
    /// 4-bit quantization, group size 32, with f16 scale
    Q4_0,
    /// 4-bit quantization, group size 32, with f16 scale and f16 min
    Q4_1,
    /// 8-bit quantization, group size 32, with f16 scale
    Q8_0,
}

impl DType {
    /// Size of one element in bytes (for non-quantized types).
    ///
    /// For quantized types this returns an approximate per-element size.
    /// Use [`numel_to_bytes`](DType::numel_to_bytes) for exact buffer sizing.
    pub fn size_of(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::Bfloat16 => 2,
            DType::UInt32 => 4,
            DType::Q4_0 => 1, // ~0.5625 bytes/element (18 bytes per 32 elements)
            DType::Q4_1 => 1, // ~0.625 bytes/element (20 bytes per 32 elements)
            DType::Q8_0 => 1, // ~1.0625 bytes/element (34 bytes per 32 elements)
        }
    }

    /// Size of one packed block in bytes for quantized types.
    ///
    /// - Q4_0: 2 (f16 scale) + 16 (4-bit × 32 / 8) = 18 bytes
    /// - Q4_1: 2 (f16 scale) + 2 (f16 min) + 16 = 20 bytes
    /// - Q8_0: 2 (f16 scale) + 32 (8-bit × 32 / 8) = 34 bytes
    ///
    /// Returns `None` for non-quantized types.
    pub fn packed_block_size(&self) -> Option<usize> {
        match self {
            DType::Q4_0 => Some(18),
            DType::Q4_1 => Some(20),
            DType::Q8_0 => Some(34),
            _ => None,
        }
    }

    /// Number of elements per quantization block.
    ///
    /// All current quantized formats use a group size of 32.
    /// Returns `None` for non-quantized types.
    pub fn block_size(&self) -> Option<usize> {
        match self {
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => Some(32),
            _ => None,
        }
    }

    /// Returns `true` if this is a quantized dtype.
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q4_0 | DType::Q4_1 | DType::Q8_0)
    }

    /// Compute the exact buffer size in bytes needed for `numel` elements.
    ///
    /// For non-quantized types: `numel * size_of()`.
    /// For quantized types: `ceil(numel / block_size) * packed_block_size`.
    ///
    /// # Panics
    /// Panics if `numel` is not a multiple of `block_size()` for quantized types.
    pub fn numel_to_bytes(&self, numel: usize) -> usize {
        if let (Some(bs), Some(pbs)) = (self.block_size(), self.packed_block_size()) {
            debug_assert!(
                numel % bs == 0,
                "numel ({numel}) must be a multiple of block_size ({bs}) for {self}"
            );
            (numel / bs) * pbs
        } else {
            numel * self.size_of()
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::Bfloat16 => "bfloat16",
            DType::UInt32 => "uint32",
            DType::Q4_0 => "q4_0",
            DType::Q4_1 => "q4_1",
            DType::Q8_0 => "q8_0",
        }
    }
}

/// Trait mapping Rust types to their corresponding `DType`.
pub trait HasDType {
    const DTYPE: DType;
}

impl HasDType for f32 {
    const DTYPE: DType = DType::Float32;
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size_of(), 4);
        assert_eq!(DType::Float16.size_of(), 2);
        assert_eq!(DType::Bfloat16.size_of(), 2);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DType::Float32.name(), "float32");
        assert_eq!(DType::Float16.name(), "float16");
        assert_eq!(DType::Bfloat16.name(), "bfloat16");
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::Float32), "float32");
    }

    #[test]
    fn test_dtype_eq_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DType::Float32);
        set.insert(DType::Float16);
        set.insert(DType::Float32); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_packed_block_size() {
        assert_eq!(DType::Q4_0.packed_block_size(), Some(18));
        assert_eq!(DType::Q4_1.packed_block_size(), Some(20));
        assert_eq!(DType::Q8_0.packed_block_size(), Some(34));
        assert_eq!(DType::Float32.packed_block_size(), None);
        assert_eq!(DType::Float16.packed_block_size(), None);
        assert_eq!(DType::Bfloat16.packed_block_size(), None);
    }

    #[test]
    fn test_block_size() {
        assert_eq!(DType::Q4_0.block_size(), Some(32));
        assert_eq!(DType::Q4_1.block_size(), Some(32));
        assert_eq!(DType::Q8_0.block_size(), Some(32));
        assert_eq!(DType::Float32.block_size(), None);
    }

    #[test]
    fn test_is_quantized() {
        assert!(DType::Q4_0.is_quantized());
        assert!(DType::Q4_1.is_quantized());
        assert!(DType::Q8_0.is_quantized());
        assert!(!DType::Float32.is_quantized());
        assert!(!DType::Float16.is_quantized());
        assert!(!DType::Bfloat16.is_quantized());
    }

    #[test]
    fn test_numel_to_bytes_float() {
        assert_eq!(DType::Float32.numel_to_bytes(100), 400);
        assert_eq!(DType::Float16.numel_to_bytes(100), 200);
        assert_eq!(DType::Bfloat16.numel_to_bytes(100), 200);
    }

    #[test]
    fn test_numel_to_bytes_quantized() {
        // 32 elements = 1 block
        assert_eq!(DType::Q4_0.numel_to_bytes(32), 18);
        assert_eq!(DType::Q4_1.numel_to_bytes(32), 20);
        assert_eq!(DType::Q8_0.numel_to_bytes(32), 34);

        // 64 elements = 2 blocks
        assert_eq!(DType::Q4_0.numel_to_bytes(64), 36);
        assert_eq!(DType::Q4_1.numel_to_bytes(64), 40);
        assert_eq!(DType::Q8_0.numel_to_bytes(64), 68);

        // 1024 elements = 32 blocks
        assert_eq!(DType::Q4_0.numel_to_bytes(1024), 576);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn test_numel_to_bytes_non_aligned_panics() {
        DType::Q4_0.numel_to_bytes(33);
    }

    #[test]
    fn test_uint32_properties() {
        assert_eq!(DType::UInt32.size_of(), 4);
        assert_eq!(DType::UInt32.name(), "uint32");
        assert_eq!(format!("{}", DType::UInt32), "uint32");
        assert!(!DType::UInt32.is_quantized());
        assert_eq!(DType::UInt32.packed_block_size(), None);
        assert_eq!(DType::UInt32.block_size(), None);
        assert_eq!(DType::UInt32.numel_to_bytes(100), 400);
    }

    #[test]
    fn test_uint32_distinct_from_float32() {
        assert_ne!(DType::UInt32, DType::Float32);
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DType::UInt32);
        set.insert(DType::Float32);
        assert_eq!(set.len(), 2);
    }
}
