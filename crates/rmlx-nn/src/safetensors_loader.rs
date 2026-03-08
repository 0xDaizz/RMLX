//! Safetensors weight loading utilities.
//!
//! Loads model weights from HuggingFace safetensors files into RMLX nn modules.
//! Supports both non-quantized (f16/bf16/f32) tensors and MLX-community quantized
//! models (weight/scales/biases triplet format).
//!
//! # MLX Quantized Format
//!
//! MLX-community quantized models store each linear layer as a triplet:
//! - `weight` (uint32): packed quantized values
//! - `scales` (float16): per-group scale factors
//! - `biases` (float16): per-group bias terms
//!
//! The quantization config is in `config.json`:
//! ```json
//! {"quantization": {"bits": 4, "group_size": 64}}
//! ```
//!
//! Dequantization: `value = scale * quantized_int + bias`
//!
//! # Example
//!
//! ```rust,ignore
//! use rmlx_nn::safetensors_loader::{SafetensorsWeightMap, load_safetensors_weights};
//!
//! let weights = load_safetensors_weights(device, "model.safetensors")?;
//! let embed = weights.get("model.embed_tokens.weight")?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use safetensors::SafeTensors;

use crate::quantized_linear::{QuantBits, QuantizedLinear};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for safetensors weight loading.
#[derive(Debug)]
pub enum SafetensorsLoadError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// Safetensors parsing error.
    Parse(String),
    /// Tensor not found.
    TensorNotFound(String),
    /// Unsupported dtype in the safetensors file.
    UnsupportedDtype { tensor_name: String, dtype: String },
    /// Shape mismatch.
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// Invalid quantization configuration.
    InvalidQuantConfig(String),
    /// Error creating a QuantizedLinear layer.
    QuantizedLinearError(String),
}

impl std::fmt::Display for SafetensorsLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(e) => write!(f, "safetensors parse error: {e}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedDtype { tensor_name, dtype } => {
                write!(f, "unsupported dtype {dtype} for tensor {tensor_name}")
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
            Self::InvalidQuantConfig(msg) => write!(f, "invalid quantization config: {msg}"),
            Self::QuantizedLinearError(msg) => write!(f, "QuantizedLinear error: {msg}"),
        }
    }
}

impl std::error::Error for SafetensorsLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SafetensorsLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Quantization config (from config.json)
// ---------------------------------------------------------------------------

/// Quantization configuration parsed from config.json.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization bit width (4 or 8).
    pub bits: u32,
    /// Elements per quantization group (32, 64, or 128).
    pub group_size: usize,
}

/// Parse quantization config from a config.json file.
///
/// Looks for `{"quantization": {"bits": N, "group_size": M}}`.
/// Returns `None` if the quantization field is absent (non-quantized model).
pub fn parse_quantization_config<P: AsRef<Path>>(
    path: P,
) -> Result<Option<QuantizationConfig>, SafetensorsLoadError> {
    let content = fs::read_to_string(path)?;
    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;

    let quant = match json.get("quantization") {
        Some(v) => v,
        None => return Ok(None),
    };

    let bits = quant.get("bits").and_then(|v| v.as_u64()).ok_or_else(|| {
        SafetensorsLoadError::InvalidQuantConfig("missing or invalid 'bits' field".into())
    })? as u32;

    let group_size = quant
        .get("group_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            SafetensorsLoadError::InvalidQuantConfig("missing or invalid 'group_size' field".into())
        })? as usize;

    if bits != 4 && bits != 8 {
        return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
            "bits must be 4 or 8, got {bits}"
        )));
    }

    if ![32, 64, 128].contains(&group_size) {
        return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
            "group_size must be 32, 64, or 128, got {group_size}"
        )));
    }

    Ok(Some(QuantizationConfig { bits, group_size }))
}

// ---------------------------------------------------------------------------
// Dtype mapping
// ---------------------------------------------------------------------------

/// Map safetensors dtype string to RMLX DType.
fn safetensors_dtype_to_rmlx(dtype: safetensors::Dtype) -> Option<DType> {
    match dtype {
        safetensors::Dtype::F32 => Some(DType::Float32),
        safetensors::Dtype::F16 => Some(DType::Float16),
        safetensors::Dtype::BF16 => Some(DType::Bfloat16),
        safetensors::Dtype::U32 => Some(DType::UInt32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// f16 -> f32 conversion
// ---------------------------------------------------------------------------

/// Convert f16 bytes (little-endian u16) to f32 values.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "f16 data must have even byte count");
    let count = bytes.len() / 2;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out.push(f16_to_f32(bits));
    }
    out
}

/// Convert an IEEE 754 half-precision (f16) value stored as u16 to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

// ---------------------------------------------------------------------------
// HF tensor name mapping
// ---------------------------------------------------------------------------

/// Map HuggingFace tensor names to RMLX naming convention.
///
/// HF format: `model.layers.0.self_attn.q_proj.weight`
/// RMLX format: `layers.0.attention.q_proj.weight`
pub fn hf_name_to_rmlx(hf_name: &str) -> String {
    let name = hf_name
        // Strip "model." prefix
        .strip_prefix("model.")
        .unwrap_or(hf_name);

    name.replace("self_attn.", "attention.")
        .replace("mlp.gate_proj.", "feed_forward.gate_proj.")
        .replace("mlp.up_proj.", "feed_forward.up_proj.")
        .replace("mlp.down_proj.", "feed_forward.down_proj.")
        .replace("input_layernorm.", "attention_norm.")
        .replace("post_attention_layernorm.", "ffn_norm.")
        .replace("embed_tokens.", "embedding.")
        .replace("model.norm.", "norm.")
}

// ---------------------------------------------------------------------------
// SafetensorsWeightMap
// ---------------------------------------------------------------------------

/// Loaded safetensors weight map: name -> Array on the Metal device.
pub struct SafetensorsWeightMap {
    /// Tensor arrays keyed by original safetensors tensor name.
    tensors: HashMap<String, Array>,
    /// Quantized linear layers keyed by layer prefix (e.g., "model.layers.0.self_attn.q_proj").
    quantized_layers: HashMap<String, QuantizedLinear>,
}

impl SafetensorsWeightMap {
    /// Number of regular (non-quantized) tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the weight map has no regular tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Result<&Array, SafetensorsLoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| SafetensorsLoadError::TensorNotFound(name.to_string()))
    }

    /// Get a tensor by name with shape validation.
    pub fn get_with_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&Array, SafetensorsLoadError> {
        let array = self.get(name)?;
        if array.shape() != expected_shape {
            return Err(SafetensorsLoadError::ShapeMismatch {
                tensor_name: name.to_string(),
                expected: expected_shape.to_vec(),
                found: array.shape().to_vec(),
            });
        }
        Ok(array)
    }

    /// Take ownership of a tensor, removing it from the map.
    pub fn take(&mut self, name: &str) -> Result<Array, SafetensorsLoadError> {
        self.tensors
            .remove(name)
            .ok_or_else(|| SafetensorsLoadError::TensorNotFound(name.to_string()))
    }

    /// Check if a tensor exists.
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

    /// Get a quantized linear layer by its prefix name.
    pub fn get_quantized(&self, prefix: &str) -> Option<&QuantizedLinear> {
        self.quantized_layers.get(prefix)
    }

    /// Take ownership of a quantized linear layer.
    pub fn take_quantized(&mut self, prefix: &str) -> Option<QuantizedLinear> {
        self.quantized_layers.remove(prefix)
    }

    /// Number of quantized linear layers.
    pub fn num_quantized(&self) -> usize {
        self.quantized_layers.len()
    }

    /// Iterate over quantized layer prefixes.
    pub fn quantized_names(&self) -> impl Iterator<Item = &str> {
        self.quantized_layers.keys().map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Loading functions
// ---------------------------------------------------------------------------

/// Load all tensors from a single safetensors file.
///
/// For non-quantized models, all tensors are loaded as RMLX Arrays.
/// For quantized models, pass a `QuantizationConfig` to automatically
/// detect weight/scales/biases triplets and create `QuantizedLinear` layers.
pub fn load_safetensors_weights<P: AsRef<Path>>(
    device: &metal::Device,
    path: P,
    quant_config: Option<&QuantizationConfig>,
) -> Result<SafetensorsWeightMap, SafetensorsLoadError> {
    let data = fs::read(path.as_ref())?;
    let st =
        SafeTensors::deserialize(&data).map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;

    if let Some(qc) = quant_config {
        load_quantized(device, &st, qc)
    } else {
        load_plain(device, &st)
    }
}

/// Load all tensors from multiple safetensors files (sharded models).
///
/// Files like `model-00001-of-00003.safetensors`, etc.
pub fn load_safetensors_sharded<P: AsRef<Path>>(
    device: &metal::Device,
    paths: &[P],
    quant_config: Option<&QuantizationConfig>,
) -> Result<SafetensorsWeightMap, SafetensorsLoadError> {
    let mut all_tensors = HashMap::new();
    let mut all_quantized = HashMap::new();

    for path in paths {
        let wmap = load_safetensors_weights(device, path, quant_config)?;
        all_tensors.extend(wmap.tensors);
        all_quantized.extend(wmap.quantized_layers);
    }

    Ok(SafetensorsWeightMap {
        tensors: all_tensors,
        quantized_layers: all_quantized,
    })
}

/// Load non-quantized tensors.
fn load_plain(
    device: &metal::Device,
    st: &SafeTensors,
) -> Result<SafetensorsWeightMap, SafetensorsLoadError> {
    let mut tensors = HashMap::new();

    for (name, view) in st.tensors() {
        let dtype = safetensors_dtype_to_rmlx(view.dtype()).ok_or_else(|| {
            SafetensorsLoadError::UnsupportedDtype {
                tensor_name: name.clone(),
                dtype: format!("{:?}", view.dtype()),
            }
        })?;

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let array = Array::from_bytes(device, data, shape, dtype);
        tensors.insert(name, array);
    }

    Ok(SafetensorsWeightMap {
        tensors,
        quantized_layers: HashMap::new(),
    })
}

/// Load tensors with quantization triplet detection.
///
/// For each tensor name ending in `.weight`, checks if matching `.scales` and
/// `.biases` tensors exist. If so, assembles a `QuantizedLinear`. Otherwise,
/// loads as a regular Array.
fn load_quantized(
    device: &metal::Device,
    st: &SafeTensors,
    qc: &QuantizationConfig,
) -> Result<SafetensorsWeightMap, SafetensorsLoadError> {
    let tensor_names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();

    // Identify quantized triplets: prefix -> (weight_name, scales_name, biases_name)
    let mut triplets: HashMap<String, (String, String, String)> = HashMap::new();
    let mut triplet_members: std::collections::HashSet<String> = std::collections::HashSet::new();

    for name in &tensor_names {
        if name.ends_with(".weight") {
            let prefix = name.strip_suffix(".weight").unwrap();
            let scales_name = format!("{prefix}.scales");
            let biases_name = format!("{prefix}.biases");

            if tensor_names.contains(&scales_name) && tensor_names.contains(&biases_name) {
                triplet_members.insert(name.clone());
                triplet_members.insert(scales_name.clone());
                triplet_members.insert(biases_name.clone());
                triplets.insert(prefix.to_string(), (name.clone(), scales_name, biases_name));
            }
        }
    }

    let mut tensors = HashMap::new();
    let mut quantized_layers = HashMap::new();

    // Load non-quantized tensors
    for name in &tensor_names {
        if triplet_members.contains(name) {
            continue;
        }
        let view = st
            .tensor(name)
            .map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;

        let dtype = safetensors_dtype_to_rmlx(view.dtype()).ok_or_else(|| {
            SafetensorsLoadError::UnsupportedDtype {
                tensor_name: name.clone(),
                dtype: format!("{:?}", view.dtype()),
            }
        })?;

        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let array = Array::from_bytes(device, data, shape, dtype);
        tensors.insert(name.clone(), array);
    }

    // Assemble quantized layers from triplets
    for (prefix, (weight_name, scales_name, biases_name)) in &triplets {
        let w_view = st
            .tensor(weight_name)
            .map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;
        let s_view = st
            .tensor(scales_name)
            .map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;
        let b_view = st
            .tensor(biases_name)
            .map_err(|e| SafetensorsLoadError::Parse(e.to_string()))?;

        let ql = assemble_quantized_linear(&w_view, &s_view, &b_view, qc, prefix)?;
        quantized_layers.insert(prefix.clone(), ql);
    }

    Ok(SafetensorsWeightMap {
        tensors,
        quantized_layers,
    })
}

/// Assemble a QuantizedLinear from a weight/scales/biases triplet.
///
/// MLX quantized format:
/// - weight: [out_features, in_features / pack_factor] as U32
///   pack_factor = 32 / bits (e.g., 8 for Q4)
/// - scales: [out_features, in_features / group_size] as F16
/// - biases: [out_features, in_features / group_size] as F16
fn assemble_quantized_linear(
    w_view: &safetensors::tensor::TensorView,
    s_view: &safetensors::tensor::TensorView,
    b_view: &safetensors::tensor::TensorView,
    qc: &QuantizationConfig,
    prefix: &str,
) -> Result<QuantizedLinear, SafetensorsLoadError> {
    let w_shape = w_view.shape();
    let s_shape = s_view.shape();

    if w_shape.len() != 2 || s_shape.len() != 2 {
        return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
            "{prefix}: weight shape {w_shape:?} or scales shape {s_shape:?} is not 2D"
        )));
    }

    let out_features = w_shape[0];
    let pack_factor = 32 / qc.bits as usize; // 8 for Q4, 4 for Q8
    let in_features = w_shape[1] * pack_factor;
    let group_size = qc.group_size;

    let bits = match qc.bits {
        4 => QuantBits::Q4,
        8 => QuantBits::Q8,
        _ => {
            return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
                "unsupported bits: {}",
                qc.bits
            )));
        }
    };

    // Repack U32 weight data to packed u8 bytes for QuantizedLinear.
    // MLX stores uint32 where each u32 holds `pack_factor` quantized values.
    // QuantizedLinear expects raw packed bytes.
    let w_data = w_view.data();
    let w_packed = repack_mlx_weight(w_data, out_features, in_features, qc.bits)?;

    // Convert f16 scales and biases to f32
    let scales = f16_bytes_to_f32(s_view.data());
    let biases = f16_bytes_to_f32(b_view.data());

    QuantizedLinear::new(
        w_packed,
        scales,
        biases,
        in_features,
        out_features,
        group_size,
        bits,
    )
    .map_err(|e| SafetensorsLoadError::QuantizedLinearError(format!("{prefix}: {e}")))
}

/// Repack MLX uint32 packed weights into the byte layout expected by QuantizedLinear.
///
/// MLX Q4: each u32 holds 8 nibbles (4-bit values).
///   QuantizedLinear expects: out_features * (in_features / 2) bytes,
///   where each byte holds 2 nibbles.
///
/// MLX Q8: each u32 holds 4 bytes (8-bit values).
///   QuantizedLinear expects: out_features * in_features bytes.
fn repack_mlx_weight(
    data: &[u8],
    out_features: usize,
    in_features: usize,
    bits: u32,
) -> Result<Vec<u8>, SafetensorsLoadError> {
    let num_u32 = data.len() / 4;
    if data.len() % 4 != 0 {
        return Err(SafetensorsLoadError::InvalidQuantConfig(
            "weight data length not a multiple of 4".into(),
        ));
    }

    match bits {
        4 => {
            // Each u32 has 8 nibbles. Output: 4 bytes per u32 (8 nibbles -> 4 bytes, 2 per byte)
            let expected_packed = out_features * (in_features / 2);
            let mut packed = Vec::with_capacity(expected_packed);

            for i in 0..num_u32 {
                let word = u32::from_le_bytes([
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ]);
                // Extract 8 nibbles and repack into 4 bytes (2 nibbles per byte)
                for j in 0..4 {
                    let lo = ((word >> (j * 8)) & 0xF) as u8;
                    let hi = ((word >> (j * 8 + 4)) & 0xF) as u8;
                    packed.push(lo | (hi << 4));
                }
            }

            if packed.len() != expected_packed {
                return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
                    "Q4 repacked length {} != expected {expected_packed}",
                    packed.len()
                )));
            }
            Ok(packed)
        }
        8 => {
            // Each u32 has 4 bytes. Just reinterpret as bytes in LE order.
            let expected_packed = out_features * in_features;
            let mut packed = Vec::with_capacity(expected_packed);

            for i in 0..num_u32 {
                packed.push(data[i * 4]);
                packed.push(data[i * 4 + 1]);
                packed.push(data[i * 4 + 2]);
                packed.push(data[i * 4 + 3]);
            }

            if packed.len() != expected_packed {
                return Err(SafetensorsLoadError::InvalidQuantConfig(format!(
                    "Q8 repacked length {} != expected {expected_packed}",
                    packed.len()
                )));
            }
            Ok(packed)
        }
        _ => Err(SafetensorsLoadError::InvalidQuantConfig(format!(
            "unsupported bits: {bits}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // HF name mapping
    // -------------------------------------------------------------------

    #[test]
    fn test_hf_name_to_rmlx_attn() {
        assert_eq!(
            hf_name_to_rmlx("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attention.q_proj.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.3.self_attn.k_proj.weight"),
            "layers.3.attention.k_proj.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.1.self_attn.v_proj.weight"),
            "layers.1.attention.v_proj.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.0.self_attn.o_proj.weight"),
            "layers.0.attention.o_proj.weight"
        );
    }

    #[test]
    fn test_hf_name_to_rmlx_ffn() {
        assert_eq!(
            hf_name_to_rmlx("model.layers.2.mlp.gate_proj.weight"),
            "layers.2.feed_forward.gate_proj.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.5.mlp.up_proj.weight"),
            "layers.5.feed_forward.up_proj.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.0.mlp.down_proj.weight"),
            "layers.0.feed_forward.down_proj.weight"
        );
    }

    #[test]
    fn test_hf_name_to_rmlx_norms() {
        assert_eq!(
            hf_name_to_rmlx("model.layers.0.input_layernorm.weight"),
            "layers.0.attention_norm.weight"
        );
        assert_eq!(
            hf_name_to_rmlx("model.layers.0.post_attention_layernorm.weight"),
            "layers.0.ffn_norm.weight"
        );
    }

    #[test]
    fn test_hf_name_to_rmlx_embedding() {
        assert_eq!(
            hf_name_to_rmlx("model.embed_tokens.weight"),
            "embedding.weight"
        );
    }

    #[test]
    fn test_hf_name_to_rmlx_lm_head() {
        assert_eq!(hf_name_to_rmlx("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn test_hf_name_to_rmlx_no_model_prefix() {
        // Some models might not have the "model." prefix
        assert_eq!(
            hf_name_to_rmlx("layers.0.self_attn.q_proj.weight"),
            "layers.0.attention.q_proj.weight"
        );
    }

    // -------------------------------------------------------------------
    // Quantization config parsing
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_quant_config_valid() {
        let dir = std::env::temp_dir().join("rmlx_st_test_qc");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"quantization": {"bits": 4, "group_size": 64}}"#).unwrap();

        let qc = parse_quantization_config(&path).unwrap().unwrap();
        assert_eq!(qc.bits, 4);
        assert_eq!(qc.group_size, 64);
    }

    #[test]
    fn test_parse_quant_config_q8() {
        let dir = std::env::temp_dir().join("rmlx_st_test_qc8");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"quantization": {"bits": 8, "group_size": 32}}"#).unwrap();

        let qc = parse_quantization_config(&path).unwrap().unwrap();
        assert_eq!(qc.bits, 8);
        assert_eq!(qc.group_size, 32);
    }

    #[test]
    fn test_parse_quant_config_none() {
        let dir = std::env::temp_dir().join("rmlx_st_test_noqc");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"model_type": "llama"}"#).unwrap();

        assert!(parse_quantization_config(&path).unwrap().is_none());
    }

    #[test]
    fn test_parse_quant_config_invalid_bits() {
        let dir = std::env::temp_dir().join("rmlx_st_test_badbits");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"quantization": {"bits": 3, "group_size": 64}}"#).unwrap();

        assert!(parse_quantization_config(&path).is_err());
    }

    #[test]
    fn test_parse_quant_config_invalid_group_size() {
        let dir = std::env::temp_dir().join("rmlx_st_test_badgs");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"quantization": {"bits": 4, "group_size": 16}}"#).unwrap();

        assert!(parse_quantization_config(&path).is_err());
    }

    // -------------------------------------------------------------------
    // f16 -> f32 conversion
    // -------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_one() {
        // 0x3C00 = 1.0 in f16
        assert_eq!(f16_to_f32(0x3C00), 1.0);
    }

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_neg_one() {
        // 0xBC00 = -1.0 in f16
        assert_eq!(f16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_f16_bytes_to_f32_roundtrip() {
        let bytes = [0x00, 0x3C, 0x00, 0xBC]; // 1.0, -1.0
        let vals = f16_bytes_to_f32(&bytes);
        assert_eq!(vals, vec![1.0, -1.0]);
    }

    // -------------------------------------------------------------------
    // Dtype mapping
    // -------------------------------------------------------------------

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(
            safetensors_dtype_to_rmlx(safetensors::Dtype::F32),
            Some(DType::Float32)
        );
        assert_eq!(
            safetensors_dtype_to_rmlx(safetensors::Dtype::F16),
            Some(DType::Float16)
        );
        assert_eq!(
            safetensors_dtype_to_rmlx(safetensors::Dtype::BF16),
            Some(DType::Bfloat16)
        );
        assert_eq!(
            safetensors_dtype_to_rmlx(safetensors::Dtype::U32),
            Some(DType::UInt32)
        );
        assert_eq!(safetensors_dtype_to_rmlx(safetensors::Dtype::I64), None);
    }

    // -------------------------------------------------------------------
    // Weight repacking
    // -------------------------------------------------------------------

    #[test]
    fn test_repack_q4_simple() {
        // One u32 with nibbles: 0,1,2,3,4,5,6,7
        // u32 = 0x76543210
        let word: u32 = 0x76543210;
        let data = word.to_le_bytes().to_vec();

        // out=1, in=8 (pack_factor=8 for Q4)
        let packed = repack_mlx_weight(&data, 1, 8, 4).unwrap();
        // Expected: 4 bytes, each holding 2 nibbles
        assert_eq!(packed.len(), 4); // 1 * (8/2) = 4
                                     // byte0: lo=0, hi=1 -> 0x10
        assert_eq!(packed[0], 0x10);
        // byte1: lo=2, hi=3 -> 0x32
        assert_eq!(packed[1], 0x32);
        // byte2: lo=4, hi=5 -> 0x54
        assert_eq!(packed[2], 0x54);
        // byte3: lo=6, hi=7 -> 0x76
        assert_eq!(packed[3], 0x76);
    }

    #[test]
    fn test_repack_q8_simple() {
        let word: u32 = 0x04030201;
        let data = word.to_le_bytes().to_vec();

        // out=1, in=4 (pack_factor=4 for Q8)
        let packed = repack_mlx_weight(&data, 1, 4, 8).unwrap();
        assert_eq!(packed.len(), 4);
        assert_eq!(packed, vec![1, 2, 3, 4]);
    }

    // -------------------------------------------------------------------
    // Error display
    // -------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = SafetensorsLoadError::TensorNotFound("foo.weight".into());
        assert!(e.to_string().contains("foo.weight"));

        let e = SafetensorsLoadError::UnsupportedDtype {
            tensor_name: "bar".into(),
            dtype: "I64".into(),
        };
        assert!(e.to_string().contains("I64"));
        assert!(e.to_string().contains("bar"));

        let e = SafetensorsLoadError::ShapeMismatch {
            tensor_name: "x".into(),
            expected: vec![4, 4],
            found: vec![8, 8],
        };
        assert!(e.to_string().contains("x"));
    }

    // -------------------------------------------------------------------
    // SafetensorsWeightMap
    // -------------------------------------------------------------------

    #[test]
    fn test_weight_map_basic() {
        let map = SafetensorsWeightMap {
            tensors: HashMap::new(),
            quantized_layers: HashMap::new(),
        };
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.num_quantized(), 0);
        assert!(!map.contains("foo"));
        assert!(map.get("foo").is_err());
    }

    // -------------------------------------------------------------------
    // Integration: load from synthetic safetensors bytes
    // -------------------------------------------------------------------

    #[test]
    fn test_load_plain_safetensors() {
        // Build a minimal safetensors file with a single f32 [2,2] tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &data_bytes)
                .unwrap();
        let tensors_vec = vec![("test_weight", tensor)];
        let serialized = safetensors::tensor::serialize(tensors_vec, &None).unwrap();

        // Write to temp file
        let dir = std::env::temp_dir().join("rmlx_st_load_plain");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        let device = rmlx_metal::device::GpuDevice::system_default()
            .expect("system_default Metal device")
            .raw()
            .clone();

        let wmap = load_safetensors_weights(&device, &path, None).unwrap();
        assert_eq!(wmap.len(), 1);
        assert!(wmap.contains("test_weight"));

        let arr = wmap.get("test_weight").unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.dtype(), DType::Float32);

        let loaded: Vec<f32> = arr.to_vec_checked();
        assert_eq!(loaded, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_load_f16_tensor() {
        // f16 values: 1.0 = 0x3C00
        let data_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C];

        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2, 2], &data_bytes)
                .unwrap();
        let tensors_vec = vec![("embed", tensor)];
        let serialized = safetensors::tensor::serialize(tensors_vec, &None).unwrap();

        let dir = std::env::temp_dir().join("rmlx_st_load_f16");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        let device = rmlx_metal::device::GpuDevice::system_default()
            .expect("system_default Metal device")
            .raw()
            .clone();

        let wmap = load_safetensors_weights(&device, &path, None).unwrap();
        let arr = wmap.get("embed").unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.dtype(), DType::Float16);
    }

    #[test]
    fn test_load_quantized_triplet() {
        // Create a Q4 quantized triplet:
        // - weight: [2, 4] as U32 (out=2, packed_in=4, in=4*8=32)
        // - scales: [2, 1] as F16 (groups = 32/32 = 1 per row)
        // - biases: [2, 1] as F16
        let out_features = 2usize;
        let in_features = 32usize;
        let pack_factor = 8usize; // Q4
        let packed_cols = in_features / pack_factor; // 4

        // Weight data: all zeros
        let w_bytes: Vec<u8> = vec![0u8; out_features * packed_cols * 4];
        let w_tensor = safetensors::tensor::TensorView::new(
            safetensors::Dtype::U32,
            vec![out_features, packed_cols],
            &w_bytes,
        )
        .unwrap();

        // Scales: 1.0 in f16 = 0x3C00
        let s_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x3C]; // 2 values
        let s_tensor = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F16,
            vec![out_features, 1],
            &s_bytes,
        )
        .unwrap();

        // Biases: 0.0 in f16 = 0x0000
        let b_bytes: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00]; // 2 values
        let b_tensor = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F16,
            vec![out_features, 1],
            &b_bytes,
        )
        .unwrap();

        let tensors_vec = vec![
            ("layer.weight", w_tensor),
            ("layer.scales", s_tensor),
            ("layer.biases", b_tensor),
        ];
        let serialized = safetensors::tensor::serialize(tensors_vec, &None).unwrap();

        let dir = std::env::temp_dir().join("rmlx_st_load_quant");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, &serialized).unwrap();

        let device = rmlx_metal::device::GpuDevice::system_default()
            .expect("system_default Metal device")
            .raw()
            .clone();

        let qc = QuantizationConfig {
            bits: 4,
            group_size: 32,
        };
        let wmap = load_safetensors_weights(&device, &path, Some(&qc)).unwrap();

        // Should have 0 regular tensors and 1 quantized layer
        assert_eq!(wmap.len(), 0);
        assert_eq!(wmap.num_quantized(), 1);

        let ql = wmap.get_quantized("layer").unwrap();
        assert_eq!(ql.in_features(), 32);
        assert_eq!(ql.out_features(), 2);
        assert_eq!(ql.group_size(), 32);
        assert_eq!(ql.bits(), QuantBits::Q4);
    }
}
