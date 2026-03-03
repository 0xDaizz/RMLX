//! GGUF weight loading utilities (N13).
//!
//! Provides high-level functions for loading model weights from GGUF files
//! into RMLX nn modules. Wraps the low-level `rmlx_core::formats::gguf`
//! parser with nn-layer-aware weight mapping.
//!
//! # Workflow
//!
//! 1. Parse the GGUF header to get tensor metadata.
//! 2. Memory-map the file for zero-copy tensor data access.
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
            let info = self.tensor_info.get(name).unwrap();
            return Err(GgufLoadError::ShapeMismatch {
                tensor_name: name.to_string(),
                expected: expected_shape.to_vec(),
                found: info.shape.clone(),
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
    device: &metal::Device,
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
    device: &metal::Device,
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

/// Map GGUF tensor names to the standard naming convention used by RMLX models.
///
/// GGUF files from llama.cpp use a specific naming scheme (e.g., "blk.0.attn_q.weight").
/// This function provides a mapping table for common architectures.
pub fn gguf_name_to_rmlx(gguf_name: &str) -> Option<String> {
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
