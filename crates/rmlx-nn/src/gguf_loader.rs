//! GGUF weight loading utilities (N13).
//!
//! Provides high-level functions for loading model weights from GGUF files
//! into RMLX nn modules. Wraps the low-level `rmlx_core::formats::gguf`
//! parser with nn-layer-aware weight mapping.
//!
//! # Workflow
//!
//! 1. Parse the GGUF header to get tensor metadata.
//! 2. Seek/read tensor byte ranges from the GGUF data section.
//! 3. Use [`GgufWeightMap`] to look up tensors by model-layer name.
//! 4. Convert GGUF tensor data into RMLX `Array` objects on the Metal device.
//!
//! # Example
//!
//! ```rust,ignore
//! use rmlx_nn::gguf_loader::{GgufWeightMap, load_gguf_weights};
//!
//! let weights = load_gguf_weights(device, "model.gguf")?;
//! let embed_weight = weights.get("token_embd.weight")?;
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use rmlx_core::array::Array;
use rmlx_core::formats::gguf::{self, GgmlType, GgufError, GgufFile, GgufTensorInfo};

use crate::quantized_linear::KQuantType;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Error type for GGUF weight loading.
#[derive(Debug)]
pub enum GgufLoadError {
    /// GGUF parsing error.
    Parse(GgufError),
    /// I/O error.
    Io(std::io::Error),
    /// Tensor not found in the GGUF file.
    TensorNotFound(String),
    /// Unsupported quantization type (cannot be directly loaded).
    UnsupportedType {
        tensor_name: String,
        ggml_type: GgmlType,
    },
    /// Shape mismatch between expected and found tensor.
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        found: Vec<u64>,
    },
}

impl std::fmt::Display for GgufLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "GGUF parse error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedType {
                tensor_name,
                ggml_type,
            } => {
                write!(
                    f,
                    "unsupported GGML type {ggml_type:?} for tensor {tensor_name}"
                )
            }
            Self::ShapeMismatch {
                tensor_name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "shape mismatch for {tensor_name}: expected {expected:?}, found {found:?}"
                )
            }
        }
    }
}

impl std::error::Error for GgufLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GgufError> for GgufLoadError {
    fn from(e: GgufError) -> Self {
        Self::Parse(e)
    }
}

impl From<std::io::Error> for GgufLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Loaded GGUF weight map: name -> Array on the Metal device.
///
/// Provides convenient accessors for tensor lookup by name with
/// optional shape validation.
pub struct GgufWeightMap {
    /// Tensor arrays keyed by GGUF tensor name.
    tensors: HashMap<String, Array>,
    /// Original tensor info for debugging/inspection.
    tensor_info: HashMap<String, GgufTensorInfo>,
}

impl GgufWeightMap {
    /// Number of tensors in the weight map.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the weight map is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get a tensor by name, returning an error if not found.
    pub fn get(&self, name: &str) -> Result<&Array, GgufLoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| GgufLoadError::TensorNotFound(name.to_string()))
    }

    /// Get a tensor by name with shape validation.
    pub fn get_with_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&Array, GgufLoadError> {
        let array = self.get(name)?;
        if array.shape() != expected_shape {
            // tensor_info should always have an entry if tensors does, but
            // fall back gracefully to avoid panicking on an internal invariant.
            let found = self
                .tensor_info
                .get(name)
                .map(|info| info.shape.clone())
                .unwrap_or_else(|| array.shape().iter().map(|&d| d as u64).collect());
            return Err(GgufLoadError::ShapeMismatch {
                tensor_name: name.to_string(),
                expected: expected_shape.to_vec(),
                found,
            });
        }
        Ok(array)
    }

    /// Take ownership of a tensor, removing it from the map.
    ///
    /// Useful for transferring weights into nn modules without cloning.
    pub fn take(&mut self, name: &str) -> Result<Array, GgufLoadError> {
        self.tensors
            .remove(name)
            .ok_or_else(|| GgufLoadError::TensorNotFound(name.to_string()))
    }

    /// Check if a tensor exists in the map.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Iterate over all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Iterate over all (name, array) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get the original GGUF tensor info for a given name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_info.get(name)
    }
}

/// Parse a GGUF file header without loading tensor data.
///
/// Useful for inspecting model metadata (architecture, vocab, etc.)
/// before deciding which tensors to load.
pub fn parse_gguf_header<P: AsRef<Path>>(path: P) -> Result<GgufFile, GgufLoadError> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let header = gguf::parse_gguf(&mut reader)?;
    Ok(header)
}

/// Load all tensors from a GGUF file onto the given Metal device.
///
/// Reads each tensor's raw bytes from the file's data section and creates
/// Metal-backed `Array` objects. Only natively supported GGML types
/// (F32, F16, BF16, Q4_0, Q4_1, Q8_0) are loaded directly; other types
/// return an error.
///
/// # Arguments
///
/// * `device` - Metal device to create arrays on.
/// * `path` - Path to the GGUF file.
///
/// # Errors
///
/// Returns `GgufLoadError` on parse errors, I/O errors, or unsupported types.
pub fn load_gguf_weights<P: AsRef<Path>>(
    device: &ProtocolObject<dyn MTLDevice>,
    path: P,
) -> Result<GgufWeightMap, GgufLoadError> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut reader = BufReader::new(&file);
    let header = gguf::parse_gguf(&mut reader)?;
    // Drop the BufReader so we can use the file directly for seeking
    drop(reader);

    let mut tensors = HashMap::with_capacity(header.tensors.len());
    let mut tensor_info_map = HashMap::with_capacity(header.tensors.len());

    for info in &header.tensors {
        // Check that the GGML type is supported
        let dtype = gguf::ggml_type_to_dtype(info.ggml_type).ok_or_else(|| {
            GgufLoadError::UnsupportedType {
                tensor_name: info.name.clone(),
                ggml_type: info.ggml_type,
            }
        })?;

        // Compute the byte size of this tensor's data
        let byte_size = compute_tensor_byte_size(info);

        // Seek to the tensor data position
        let data_pos = header.data_offset + info.offset;
        file.seek(SeekFrom::Start(data_pos))?;

        // Read raw bytes
        let mut raw_data = vec![0u8; byte_size];
        file.read_exact(&mut raw_data)?;

        // Convert shape from u64 to usize
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();

        // Create an Array from the raw bytes on the Metal device
        let array = Array::from_bytes(device, &raw_data, shape, dtype);

        tensors.insert(info.name.clone(), array);
        tensor_info_map.insert(info.name.clone(), info.clone());
    }

    Ok(GgufWeightMap {
        tensors,
        tensor_info: tensor_info_map,
    })
}

/// Load selected tensors from a GGUF file by name prefix.
///
/// Only tensors whose name starts with one of the given prefixes are loaded.
/// This is useful for loading only specific layers (e.g., "blk.0." for layer 0).
pub fn load_gguf_weights_filtered<P: AsRef<Path>>(
    device: &ProtocolObject<dyn MTLDevice>,
    path: P,
    prefixes: &[&str],
) -> Result<GgufWeightMap, GgufLoadError> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut reader = BufReader::new(&file);
    let header = gguf::parse_gguf(&mut reader)?;
    drop(reader);

    let matching: Vec<&GgufTensorInfo> = header
        .tensors
        .iter()
        .filter(|t| prefixes.iter().any(|p| t.name.starts_with(p)))
        .collect();

    let mut tensors = HashMap::with_capacity(matching.len());
    let mut tensor_info_map = HashMap::with_capacity(matching.len());

    for info in matching {
        let dtype = gguf::ggml_type_to_dtype(info.ggml_type).ok_or_else(|| {
            GgufLoadError::UnsupportedType {
                tensor_name: info.name.clone(),
                ggml_type: info.ggml_type,
            }
        })?;

        let byte_size = compute_tensor_byte_size(info);
        let data_pos = header.data_offset + info.offset;
        file.seek(SeekFrom::Start(data_pos))?;

        let mut raw_data = vec![0u8; byte_size];
        file.read_exact(&mut raw_data)?;

        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        let array = Array::from_bytes(device, &raw_data, shape, dtype);

        tensors.insert(info.name.clone(), array);
        tensor_info_map.insert(info.name.clone(), info.clone());
    }

    Ok(GgufWeightMap {
        tensors,
        tensor_info: tensor_info_map,
    })
}

/// Compute the total byte size of a GGUF tensor's data.
fn compute_tensor_byte_size(info: &GgufTensorInfo) -> usize {
    let num_elements: u64 = info.shape.iter().product();
    let num_elements = num_elements as usize;
    let block_size = info.ggml_type.block_size();
    let type_size = info.ggml_type.type_size();

    // Number of blocks = ceil(num_elements / block_size)
    let num_blocks = num_elements.div_ceil(block_size);
    num_blocks * type_size
}

/// Known GGUF tensor name prefixes/patterns that we recognize.
///
/// Any tensor name that does not start with one of these patterns (after
/// accounting for the `blk.N.` layer prefix) is considered unrecognized
/// and `gguf_name_to_rmlx` returns `None` for it.
const KNOWN_GGUF_PREFIXES: &[&str] = &["blk.", "token_embd.", "output_norm.", "output."];

/// Suffixes recognized within a `blk.N.` layer tensor.
const KNOWN_BLOCK_SUFFIXES: &[&str] = &[
    ".attn_q.",
    ".attn_k.",
    ".attn_v.",
    ".attn_output.",
    ".ffn_gate.",
    ".ffn_up.",
    ".ffn_down.",
    ".attn_norm.",
    ".ffn_norm.",
];

/// Map GGUF tensor names to the standard naming convention used by RMLX models.
///
/// GGUF files from llama.cpp use a specific naming scheme (e.g., "blk.0.attn_q.weight").
/// This function provides a mapping table for common architectures.
///
/// Returns `None` for unrecognized tensor name patterns instead of silently
/// passing them through with no transformation.
pub fn gguf_name_to_rmlx(gguf_name: &str) -> Option<String> {
    // Check that the name matches a known pattern before transforming.
    let recognized = if gguf_name.starts_with("blk.") {
        // For block tensors, also verify the suffix is known
        KNOWN_BLOCK_SUFFIXES
            .iter()
            .any(|suffix| gguf_name.contains(suffix))
    } else {
        KNOWN_GGUF_PREFIXES
            .iter()
            .any(|prefix| gguf_name.starts_with(prefix))
    };

    if !recognized {
        return None;
    }

    // Common llama.cpp GGUF -> RMLX name mappings
    let name = gguf_name
        .replace("blk.", "layers.")
        .replace(".attn_q.", ".attention.q_proj.")
        .replace(".attn_k.", ".attention.k_proj.")
        .replace(".attn_v.", ".attention.v_proj.")
        .replace(".attn_output.", ".attention.o_proj.")
        .replace(".ffn_gate.", ".feed_forward.gate_proj.")
        .replace(".ffn_up.", ".feed_forward.up_proj.")
        .replace(".ffn_down.", ".feed_forward.down_proj.")
        .replace(".attn_norm.", ".attention_norm.")
        .replace("token_embd.", "embedding.")
        .replace("output_norm.", "norm.")
        .replace("output.", "lm_head.");

    Some(name)
}

// ---------------------------------------------------------------------------
// K-quant type mapping
// ---------------------------------------------------------------------------

/// Map a GGML type to a k-quant type, if applicable.
///
/// Returns `Some(KQuantType)` for Q2_K through Q6_K types, and `None`
/// for all other types (F32, F16, Q4_0, Q8_0, etc.).
///
/// This is used by the GGUF loader to determine whether a tensor uses
/// k-quant super-block encoding and needs special handling during loading.
pub fn ggml_type_to_kquant(ggml_type: GgmlType) -> Option<KQuantType> {
    match ggml_type {
        GgmlType::Q2K => Some(KQuantType::Q2K),
        GgmlType::Q3K => Some(KQuantType::Q3K),
        GgmlType::Q4K => Some(KQuantType::Q4K),
        GgmlType::Q5K => Some(KQuantType::Q5K),
        GgmlType::Q6K => Some(KQuantType::Q6K),
        _ => None,
    }
}

/// Check whether a GGML type is a k-quant type.
pub fn is_kquant_type(ggml_type: GgmlType) -> bool {
    ggml_type_to_kquant(ggml_type).is_some()
}

/// Describe how a k-quant type should be loaded.
///
/// K-quant tensors cannot be directly loaded as RMLX arrays because they
/// use super-block encoding with nested quantization of scales. This
/// function returns a [`KQuantLoadInfo`] describing the type, block
/// size, and bytes-per-block so that callers can read the raw data and
/// create a [`KQuantConfig`](crate::quantized_linear::KQuantConfig).
#[derive(Debug, Clone)]
pub struct KQuantLoadInfo {
    /// The k-quant type.
    pub quant_type: KQuantType,
    /// Elements per super block (always 256).
    pub block_size: usize,
    /// Bytes per super block.
    pub type_size: usize,
    /// Effective bits per weight.
    pub bits: u32,
}

/// Get loading info for a k-quant GGML type.
///
/// Returns `None` if the GGML type is not a k-quant type.
pub fn kquant_load_info(ggml_type: GgmlType) -> Option<KQuantLoadInfo> {
    ggml_type_to_kquant(ggml_type).map(|qt| KQuantLoadInfo {
        quant_type: qt,
        block_size: qt.block_size(),
        type_size: qt.type_size(),
        bits: qt.bits(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // gguf_name_to_rmlx: known patterns
    // -----------------------------------------------------------------------

    #[test]
    fn test_gguf_name_attn_q() {
        let mapped = gguf_name_to_rmlx("blk.0.attn_q.weight");
        assert_eq!(mapped.as_deref(), Some("layers.0.attention.q_proj.weight"));
    }

    #[test]
    fn test_gguf_name_attn_k() {
        let mapped = gguf_name_to_rmlx("blk.3.attn_k.weight");
        assert_eq!(mapped.as_deref(), Some("layers.3.attention.k_proj.weight"));
    }

    #[test]
    fn test_gguf_name_attn_v() {
        let mapped = gguf_name_to_rmlx("blk.1.attn_v.weight");
        assert_eq!(mapped.as_deref(), Some("layers.1.attention.v_proj.weight"));
    }

    #[test]
    fn test_gguf_name_attn_output() {
        let mapped = gguf_name_to_rmlx("blk.0.attn_output.weight");
        assert_eq!(mapped.as_deref(), Some("layers.0.attention.o_proj.weight"));
    }

    #[test]
    fn test_gguf_name_ffn_gate() {
        let mapped = gguf_name_to_rmlx("blk.2.ffn_gate.weight");
        assert_eq!(
            mapped.as_deref(),
            Some("layers.2.feed_forward.gate_proj.weight")
        );
    }

    #[test]
    fn test_gguf_name_ffn_up() {
        let mapped = gguf_name_to_rmlx("blk.5.ffn_up.weight");
        assert_eq!(
            mapped.as_deref(),
            Some("layers.5.feed_forward.up_proj.weight")
        );
    }

    #[test]
    fn test_gguf_name_ffn_down() {
        let mapped = gguf_name_to_rmlx("blk.0.ffn_down.weight");
        assert_eq!(
            mapped.as_deref(),
            Some("layers.0.feed_forward.down_proj.weight")
        );
    }

    #[test]
    fn test_gguf_name_attn_norm() {
        let mapped = gguf_name_to_rmlx("blk.0.attn_norm.weight");
        assert_eq!(mapped.as_deref(), Some("layers.0.attention_norm.weight"));
    }

    #[test]
    fn test_gguf_name_ffn_norm() {
        let mapped = gguf_name_to_rmlx("blk.0.ffn_norm.weight");
        assert_eq!(mapped.as_deref(), Some("layers.0.ffn_norm.weight"));
    }

    #[test]
    fn test_gguf_name_token_embd() {
        let mapped = gguf_name_to_rmlx("token_embd.weight");
        assert_eq!(mapped.as_deref(), Some("embedding.weight"));
    }

    #[test]
    fn test_gguf_name_output_norm() {
        let mapped = gguf_name_to_rmlx("output_norm.weight");
        assert_eq!(mapped.as_deref(), Some("norm.weight"));
    }

    #[test]
    fn test_gguf_name_output() {
        let mapped = gguf_name_to_rmlx("output.weight");
        assert_eq!(mapped.as_deref(), Some("lm_head.weight"));
    }

    // -----------------------------------------------------------------------
    // gguf_name_to_rmlx: unrecognized patterns return None
    // -----------------------------------------------------------------------

    #[test]
    fn test_gguf_name_unknown_returns_none() {
        assert_eq!(gguf_name_to_rmlx("some_random_tensor"), None);
    }

    #[test]
    fn test_gguf_name_empty_returns_none() {
        assert_eq!(gguf_name_to_rmlx(""), None);
    }

    #[test]
    fn test_gguf_name_unknown_block_suffix_returns_none() {
        // blk.0. prefix is recognized, but ".mystery_layer." suffix is not
        assert_eq!(gguf_name_to_rmlx("blk.0.mystery_layer.weight"), None);
    }

    #[test]
    fn test_gguf_name_partial_match_returns_none() {
        // "token_emb" is close to "token_embd" but not a match
        assert_eq!(gguf_name_to_rmlx("token_emb.weight"), None);
    }

    #[test]
    fn test_gguf_name_arbitrary_prefix_returns_none() {
        assert_eq!(gguf_name_to_rmlx("rope_freqs.weight"), None);
        assert_eq!(gguf_name_to_rmlx("model.layers.0.weight"), None);
    }

    // -----------------------------------------------------------------------
    // GgufWeightMap construction and query (synthetic, no Metal device)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gguf_load_error_display() {
        let e = GgufLoadError::TensorNotFound("foo.weight".to_string());
        assert!(e.to_string().contains("foo.weight"));

        let e2 = GgufLoadError::ShapeMismatch {
            tensor_name: "bar".to_string(),
            expected: vec![4, 4],
            found: vec![8, 8],
        };
        assert!(e2.to_string().contains("bar"));
    }

    // -----------------------------------------------------------------------
    // K-quant type mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_kquant_mapping_q2k() {
        let qt = ggml_type_to_kquant(GgmlType::Q2K);
        assert_eq!(qt, Some(KQuantType::Q2K));
        assert!(is_kquant_type(GgmlType::Q2K));
    }

    #[test]
    fn test_kquant_mapping_q3k() {
        let qt = ggml_type_to_kquant(GgmlType::Q3K);
        assert_eq!(qt, Some(KQuantType::Q3K));
    }

    #[test]
    fn test_kquant_mapping_q4k() {
        let qt = ggml_type_to_kquant(GgmlType::Q4K);
        assert_eq!(qt, Some(KQuantType::Q4K));
    }

    #[test]
    fn test_kquant_mapping_q5k() {
        let qt = ggml_type_to_kquant(GgmlType::Q5K);
        assert_eq!(qt, Some(KQuantType::Q5K));
    }

    #[test]
    fn test_kquant_mapping_q6k() {
        let qt = ggml_type_to_kquant(GgmlType::Q6K);
        assert_eq!(qt, Some(KQuantType::Q6K));
    }

    #[test]
    fn test_kquant_mapping_non_kquant_returns_none() {
        assert_eq!(ggml_type_to_kquant(GgmlType::F32), None);
        assert_eq!(ggml_type_to_kquant(GgmlType::F16), None);
        assert_eq!(ggml_type_to_kquant(GgmlType::Q4_0), None);
        assert_eq!(ggml_type_to_kquant(GgmlType::Q8_0), None);
        assert_eq!(ggml_type_to_kquant(GgmlType::BF16), None);
        assert!(!is_kquant_type(GgmlType::F32));
    }

    #[test]
    fn test_kquant_load_info() {
        let info = kquant_load_info(GgmlType::Q4K).unwrap();
        assert_eq!(info.quant_type, KQuantType::Q4K);
        assert_eq!(info.block_size, 256);
        assert_eq!(info.type_size, 144);
        assert_eq!(info.bits, 4);

        let info = kquant_load_info(GgmlType::Q2K).unwrap();
        assert_eq!(info.type_size, 84);
        assert_eq!(info.bits, 2);

        assert!(kquant_load_info(GgmlType::F32).is_none());
    }
}
