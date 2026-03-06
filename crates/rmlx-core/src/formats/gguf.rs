//! GGUF binary format parser.
//!
//! Parses the GGUF container format used by llama.cpp and related projects.
//! Supports GGUF v2 and v3.
//!
//! # Binary layout
//!
//! ```text
//! [magic: u32 = 0x46475547] [version: u32] [tensor_count: u64] [kv_count: u64]
//! [metadata_kv x kv_count]
//! [tensor_info x tensor_count]
//! [padding to alignment]
//! [tensor data...]
//! ```

use std::collections::HashMap;
use std::io::{self, Read, Seek};

// ── Error type ──────────────────────────────────────────────────────

/// Error type for GGUF parsing.
#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    InvalidValueType(u32),
    InvalidGgmlType(u32),
    InvalidString,
    InvalidAlignment(u64),
    OffsetOutOfBounds { tensor: String, offset: u64, data_len: u64 },
    Overflow(String),
}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::InvalidMagic(m) => write!(f, "invalid GGUF magic: 0x{m:08X}"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::InvalidValueType(t) => write!(f, "invalid GGUF value type: {t}"),
            GgufError::InvalidGgmlType(t) => write!(f, "invalid GGML type: {t}"),
            GgufError::InvalidString => write!(f, "invalid UTF-8 in GGUF string"),
            GgufError::InvalidAlignment(a) => write!(f, "invalid GGUF alignment: {a}"),
            GgufError::OffsetOutOfBounds { tensor, offset, data_len } => {
                write!(f, "tensor \"{tensor}\" offset {offset} exceeds data length {data_len}")
            }
            GgufError::Overflow(msg) => write!(f, "overflow: {msg}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ── GGUF magic ──────────────────────────────────────────────────────

/// GGUF magic number: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547;

/// Default alignment for tensor data.
const DEFAULT_ALIGNMENT: u64 = 32;

// ── Value types ─────────────────────────────────────────────────────

/// GGUF value types for metadata.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

// ── GGML types ──────────────────────────────────────────────────────

/// GGML tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 26,
    F64 = 30,
}

impl GgmlType {
    /// Create a `GgmlType` from its raw `u32` discriminant.
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            15 => Ok(GgmlType::Q8K),
            26 => Ok(GgmlType::BF16),
            30 => Ok(GgmlType::F64),
            _ => Err(GgufError::InvalidGgmlType(v)),
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K => 256,
        }
    }

    /// Size of one block in bytes.
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::BF16 => 2,
            GgmlType::F64 => 8,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 36,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            GgmlType::Q8K => 292,
        }
    }
}

// ── Tensor info ─────────────────────────────────────────────────────

/// Tensor info from the GGUF header.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Shape in RMLX order (reversed from GGUF's dimension order).
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset within the data section.
    pub offset: u64,
}

// ── Parsed file ─────────────────────────────────────────────────────

/// Parsed GGUF file header.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins (from start of file).
    pub data_offset: u64,
}

// ── DType mapping ───────────────────────────────────────────────────

/// Map a `GgmlType` to an RMLX `DType`.
///
/// Returns `None` for quantization formats not natively supported by RMLX.
/// Callers should dequantize those tensors before use.
pub fn ggml_type_to_dtype(ggml_type: GgmlType) -> Option<crate::dtype::DType> {
    use crate::dtype::DType;
    match ggml_type {
        GgmlType::F32 => Some(DType::Float32),
        GgmlType::F16 => Some(DType::Float16),
        GgmlType::BF16 => Some(DType::Bfloat16),
        GgmlType::Q4_0 => Some(DType::Q4_0),
        GgmlType::Q4_1 => Some(DType::Q4_1),
        GgmlType::Q8_0 => Some(DType::Q8_0),
        _ => None,
    }
}

// ── Low-level readers ───────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, GgufError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, GgufError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16, GgufError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool, GgufError> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    let len = usize::try_from(len)
        .map_err(|_| GgufError::Overflow(format!("string length {len} exceeds usize")))?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidString)
}

fn read_value<R: Read>(r: &mut R, value_type: u32) -> Result<GgufValue, GgufError> {
    match value_type {
        0 => Ok(GgufValue::Uint8(read_u8(r)?)),
        1 => Ok(GgufValue::Int8(read_i8(r)?)),
        2 => Ok(GgufValue::Uint16(read_u16(r)?)),
        3 => Ok(GgufValue::Int16(read_i16(r)?)),
        4 => Ok(GgufValue::Uint32(read_u32(r)?)),
        5 => Ok(GgufValue::Int32(read_i32(r)?)),
        6 => Ok(GgufValue::Float32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_bool(r)?)),
        8 => Ok(GgufValue::String(read_string(r)?)),
        9 => {
            // Array: element_type (u32) + count (u64) + elements
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            let count = usize::try_from(count)
                .map_err(|_| GgufError::Overflow(format!("array count {count} exceeds usize")))?;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::Uint64(read_u64(r)?)),
        11 => Ok(GgufValue::Int64(read_i64(r)?)),
        12 => Ok(GgufValue::Float64(read_f64(r)?)),
        _ => Err(GgufError::InvalidValueType(value_type)),
    }
}

// ── Main parser ─────────────────────────────────────────────────────

/// Parse a GGUF header from a reader (file or byte buffer).
///
/// Reads the magic, version, metadata key-value pairs, and tensor info entries.
/// Computes the `data_offset` (byte position where tensor data begins, aligned
/// to the alignment specified in metadata or 32 bytes by default).
///
/// # Errors
///
/// Returns `GgufError` on I/O errors, invalid magic, unsupported version,
/// or malformed metadata/tensor entries.
pub fn parse_gguf<R: Read + Seek>(reader: &mut R) -> Result<GgufFile, GgufError> {
    // 1. Magic
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    // 2. Version
    let version = read_u32(reader)?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    // 3. Counts
    let tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    // 4. Metadata KV pairs
    let kv_count = usize::try_from(kv_count)
        .map_err(|_| GgufError::Overflow(format!("kv_count {kv_count} exceeds usize")))?;
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)?;
        let value = read_value(reader, value_type)?;
        metadata.insert(key, value);
    }

    // 5. Determine alignment from metadata
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::Uint32(a)) => u64::from(*a),
        Some(GgufValue::Uint64(a)) => *a,
        _ => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 {
        return Err(GgufError::InvalidAlignment(0));
    }

    // 6. Tensor info entries
    let tensor_count = usize::try_from(tensor_count)
        .map_err(|_| GgufError::Overflow(format!("tensor_count {tensor_count} exceeds usize")))?;
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = read_u32(reader)?;
        let n_dims = usize::try_from(n_dims)
            .map_err(|_| GgufError::Overflow(format!("n_dimensions {n_dims} exceeds usize")))?;

        // Read dimensions in GGUF order, then reverse for RMLX row-major order.
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(reader)?);
        }
        dims.reverse();

        let ggml_type_raw = read_u32(reader)?;
        let ggml_type = GgmlType::from_u32(ggml_type_raw)?;
        let offset = read_u64(reader)?;

        tensors.push(GgufTensorInfo {
            name,
            shape: dims,
            ggml_type,
            offset,
        });
    }

    // 7. Compute data_offset: current position aligned up to `alignment`.
    let pos = reader.stream_position()?;
    let data_offset = pos.div_ceil(alignment) * alignment;

    // 8. Validate tensor offsets: each tensor must fit entirely within the data region.
    //    We use the stream length to determine how much data is available.
    let end = reader.seek(io::SeekFrom::End(0))?;
    let data_len = end.saturating_sub(data_offset);
    for t in &tensors {
        // Compute number of elements from shape (product of dims, minimum 1 for scalars).
        let n_elements: u64 = t.shape.iter().copied().map(|d| d as u64).product::<u64>().max(1);
        let block_size = t.ggml_type.block_size() as u64;
        let type_size = t.ggml_type.type_size() as u64;
        // Number of blocks = ceil(n_elements / block_size)
        let n_blocks = n_elements.div_ceil(block_size);
        let tensor_byte_size = n_blocks.checked_mul(type_size).ok_or_else(|| {
            GgufError::OffsetOutOfBounds {
                tensor: t.name.clone(),
                offset: t.offset,
                data_len,
            }
        })?;
        let end_offset = t.offset.checked_add(tensor_byte_size).ok_or_else(|| {
            GgufError::OffsetOutOfBounds {
                tensor: t.name.clone(),
                offset: t.offset,
                data_len,
            }
        })?;
        if end_offset > data_len {
            return Err(GgufError::OffsetOutOfBounds {
                tensor: t.name.clone(),
                offset: t.offset,
                data_len,
            });
        }
    }

    Ok(GgufFile {
        version,
        metadata,
        tensors,
        data_offset,
    })
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── Helper: write little-endian primitives into a Vec ──

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        push_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    // ── GgmlType::from_u32 ──

    #[test]
    fn test_ggml_type_from_u32_valid() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(3).unwrap(), GgmlType::Q4_1);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12).unwrap(), GgmlType::Q4K);
        assert_eq!(GgmlType::from_u32(14).unwrap(), GgmlType::Q6K);
        assert_eq!(GgmlType::from_u32(26).unwrap(), GgmlType::BF16);
        assert_eq!(GgmlType::from_u32(30).unwrap(), GgmlType::F64);
    }

    #[test]
    fn test_ggml_type_from_u32_invalid() {
        assert!(GgmlType::from_u32(4).is_err());
        assert!(GgmlType::from_u32(5).is_err());
        assert!(GgmlType::from_u32(99).is_err());
    }

    // ── block_size / type_size ──

    #[test]
    fn test_block_and_type_size() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.type_size(), 4);

        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::F16.type_size(), 2);

        assert_eq!(GgmlType::BF16.block_size(), 1);
        assert_eq!(GgmlType::BF16.type_size(), 2);

        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);

        assert_eq!(GgmlType::Q4_1.block_size(), 32);
        assert_eq!(GgmlType::Q4_1.type_size(), 20);

        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);

        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.type_size(), 144);

        assert_eq!(GgmlType::Q6K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.type_size(), 210);

        assert_eq!(GgmlType::F64.block_size(), 1);
        assert_eq!(GgmlType::F64.type_size(), 8);
    }

    // ── ggml_type_to_dtype ──

    #[test]
    fn test_ggml_type_to_dtype() {
        use crate::dtype::DType;

        assert_eq!(ggml_type_to_dtype(GgmlType::F32), Some(DType::Float32));
        assert_eq!(ggml_type_to_dtype(GgmlType::F16), Some(DType::Float16));
        assert_eq!(ggml_type_to_dtype(GgmlType::BF16), Some(DType::Bfloat16));
        assert_eq!(ggml_type_to_dtype(GgmlType::Q4_0), Some(DType::Q4_0));
        assert_eq!(ggml_type_to_dtype(GgmlType::Q4_1), Some(DType::Q4_1));
        assert_eq!(ggml_type_to_dtype(GgmlType::Q8_0), Some(DType::Q8_0));

        // Unsupported types should return None.
        assert_eq!(ggml_type_to_dtype(GgmlType::Q5_0), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::Q5_1), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::Q2K), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::Q3K), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::Q4K), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::Q6K), None);
        assert_eq!(ggml_type_to_dtype(GgmlType::F64), None);
    }

    // ── parse_gguf with synthetic binary ──

    /// Build a minimal valid GGUF v3 binary in memory with:
    /// - 1 metadata key: "test.key" = Float32(3.14)
    /// - 1 tensor: "weight" with shape [4, 8] (GGUF order [8, 4]), type F32, offset 0
    fn build_synthetic_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC); // magic
        push_u32(&mut buf, 3); // version
        push_u64(&mut buf, 1); // tensor_count
        push_u64(&mut buf, 1); // kv_count

        // Metadata KV: "test.key" = Float32(3.14)
        push_string(&mut buf, "test.key");
        push_u32(&mut buf, 6); // value_type = FLOAT32
        push_f32(&mut buf, 3.15);

        // Tensor info: "weight", 2 dims, [8, 4] (GGUF order), type F32, offset 0
        push_string(&mut buf, "weight");
        push_u32(&mut buf, 2); // n_dimensions
        push_u64(&mut buf, 8); // dim 0 (columns in GGUF)
        push_u64(&mut buf, 4); // dim 1 (rows in GGUF)
        push_u32(&mut buf, 0); // ggml_type = F32
        push_u64(&mut buf, 0); // offset in data section

        // Pad to 32-byte alignment (simulate tensor data area start)
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Append tensor data: 8 * 4 = 32 F32 elements = 128 bytes
        buf.extend_from_slice(&vec![0u8; 32 * 4]);

        buf
    }

    #[test]
    fn test_parse_synthetic_gguf() {
        let data = build_synthetic_gguf();
        let mut cursor = Cursor::new(&data);
        let file = parse_gguf(&mut cursor).expect("parse should succeed");

        assert_eq!(file.version, 3);
        assert_eq!(file.metadata.len(), 1);

        // Check metadata value
        match file.metadata.get("test.key") {
            Some(GgufValue::Float32(v)) => {
                assert!((v - 3.15).abs() < 1e-5, "expected ~3.15, got {v}");
            }
            other => panic!("expected Float32, got {other:?}"),
        }

        // Check tensor info
        assert_eq!(file.tensors.len(), 1);
        let t = &file.tensors[0];
        assert_eq!(t.name, "weight");
        // GGUF [8, 4] reversed to RMLX [4, 8]
        assert_eq!(t.shape, vec![4, 8]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.offset, 0);

        // data_offset should be aligned to 32
        assert_eq!(file.data_offset % 32, 0);
    }

    #[test]
    fn test_parse_invalid_magic() {
        let mut buf = Vec::new();
        push_u32(&mut buf, 0xDEADBEEF);
        let mut cursor = Cursor::new(&buf);
        match parse_gguf(&mut cursor) {
            Err(GgufError::InvalidMagic(m)) => assert_eq!(m, 0xDEADBEEF),
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_unsupported_version() {
        let mut buf = Vec::new();
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 1); // v1 not supported
        let mut cursor = Cursor::new(&buf);
        match parse_gguf(&mut cursor) {
            Err(GgufError::UnsupportedVersion(1)) => {}
            other => panic!("expected UnsupportedVersion(1), got {other:?}"),
        }
    }

    #[test]
    fn test_parse_string_metadata() {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 0); // no tensors
        push_u64(&mut buf, 1); // 1 kv

        // Metadata: "general.name" = String("test-model")
        push_string(&mut buf, "general.name");
        push_u32(&mut buf, 8); // STRING
        push_string(&mut buf, "test-model");

        let mut cursor = Cursor::new(&buf);
        let file = parse_gguf(&mut cursor).expect("parse should succeed");

        match file.metadata.get("general.name") {
            Some(GgufValue::String(s)) => assert_eq!(s, "test-model"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_array_metadata() {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 0); // no tensors
        push_u64(&mut buf, 1); // 1 kv

        // Metadata: "dims" = Array([Uint32(1), Uint32(2), Uint32(3)])
        push_string(&mut buf, "dims");
        push_u32(&mut buf, 9); // ARRAY
        push_u32(&mut buf, 4); // element type = UINT32
        push_u64(&mut buf, 3); // count
        push_u32(&mut buf, 1);
        push_u32(&mut buf, 2);
        push_u32(&mut buf, 3);

        let mut cursor = Cursor::new(&buf);
        let file = parse_gguf(&mut cursor).expect("parse should succeed");

        match file.metadata.get("dims") {
            Some(GgufValue::Array(arr)) => {
                assert_eq!(arr.len(), 3);
                match (&arr[0], &arr[1], &arr[2]) {
                    (GgufValue::Uint32(1), GgufValue::Uint32(2), GgufValue::Uint32(3)) => {}
                    other => panic!("unexpected array contents: {other:?}"),
                }
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_custom_alignment() {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 0); // no tensors
        push_u64(&mut buf, 1); // 1 kv

        // Metadata: "general.alignment" = Uint32(64)
        push_string(&mut buf, "general.alignment");
        push_u32(&mut buf, 4); // UINT32
        push_u32(&mut buf, 64);

        // Pad to 64 bytes
        while buf.len() % 64 != 0 {
            buf.push(0);
        }

        let mut cursor = Cursor::new(&buf);
        let file = parse_gguf(&mut cursor).expect("parse should succeed");

        assert_eq!(file.data_offset % 64, 0);
    }

    #[test]
    fn test_parse_alignment_zero_returns_error() {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 0); // no tensors
        push_u64(&mut buf, 1); // 1 kv

        // Metadata: "general.alignment" = Uint32(0)
        push_string(&mut buf, "general.alignment");
        push_u32(&mut buf, 4); // UINT32
        push_u32(&mut buf, 0); // alignment = 0 → should error

        let mut cursor = Cursor::new(&buf);
        match parse_gguf(&mut cursor) {
            Err(GgufError::InvalidAlignment(0)) => {} // expected
            other => panic!("expected InvalidAlignment(0), got {other:?}"),
        }
    }

    #[test]
    fn test_parse_tensor_offset_out_of_bounds() {
        let mut buf = Vec::new();

        // Header
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, 1); // 1 tensor
        push_u64(&mut buf, 0); // no kv

        // Tensor info: "bad_tensor", 1 dim, [4], type F32, offset way past end
        push_string(&mut buf, "bad_tensor");
        push_u32(&mut buf, 1); // n_dimensions
        push_u64(&mut buf, 4); // dim 0
        push_u32(&mut buf, 0); // ggml_type = F32
        push_u64(&mut buf, 999_999); // offset far beyond data

        // Add a small amount of "data" (16 bytes) after alignment padding
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&[0u8; 16]); // 16 bytes of tensor data

        let mut cursor = Cursor::new(&buf);
        match parse_gguf(&mut cursor) {
            Err(GgufError::OffsetOutOfBounds { tensor, .. }) => {
                assert_eq!(tensor, "bad_tensor");
            }
            other => panic!("expected OffsetOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = GgufError::InvalidMagic(0xDEAD);
        assert!(format!("{err}").contains("0x0000DEAD"));

        let err = GgufError::UnsupportedVersion(99);
        assert!(format!("{err}").contains("99"));

        let err = GgufError::InvalidValueType(42);
        assert!(format!("{err}").contains("42"));

        let err = GgufError::InvalidGgmlType(100);
        assert!(format!("{err}").contains("100"));

        let err = GgufError::InvalidString;
        assert!(format!("{err}").contains("UTF-8"));

        let err = GgufError::Overflow("too big".into());
        assert!(format!("{err}").contains("too big"));
    }
}
