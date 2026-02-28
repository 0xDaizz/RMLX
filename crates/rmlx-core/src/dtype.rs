//! Data type definitions for RMLX tensors.

use std::fmt;

/// Supported data types for RMLX arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float16,
    Bfloat16,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_of(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::Bfloat16 => 2,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::Bfloat16 => "bfloat16",
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
}
