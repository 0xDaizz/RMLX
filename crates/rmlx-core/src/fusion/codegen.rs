//! Metal source code generation for fused element-wise kernels.

use std::collections::HashMap;
use std::sync::RwLock;

use super::graph::{FusableOp, FusionGraph};
use crate::dtype::DType;

/// JIT codegen engine for fused element-wise kernels.
///
/// Generates Metal source from a `FusionGraph` and caches the generated
/// source by graph hash.
pub struct FusionCodegen {
    /// Cache: graph hash -> generated Metal source.
    cache: RwLock<HashMap<u64, String>>,
}

impl FusionCodegen {
    /// Create a new codegen engine.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Generate Metal source for a fusion graph, using cache if available.
    ///
    /// The `dtype` parameter controls the Metal buffer/variable types:
    /// - `Float32` → `float`
    /// - `Float16` → `half`
    /// - `Bfloat16` → `bfloat16_t`
    ///
    /// Transcendental operations (exp, log, tanh, etc.) are automatically
    /// promoted to f32 for f16/bf16 to preserve numerical accuracy.
    pub fn generate(&self, graph: &FusionGraph, dtype: DType) -> Result<String, String> {
        Self::validate_dtype(dtype)?;

        let key = Self::cache_key_with_dtype(graph, dtype);

        // Check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(source) = cache.get(&key) {
                return Ok(source.clone());
            }
        }

        // Generate with default name
        let source = Self::emit_kernel_named(graph, "fused_elementwise", dtype);

        // Cache
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key, source.clone());
        }

        Ok(source)
    }

    /// Generate Metal source with a specific kernel function name.
    ///
    /// This avoids pipeline cache collisions when multiple fusion graphs
    /// are compiled — each gets a unique function name.
    pub fn generate_named(
        &self,
        graph: &FusionGraph,
        name: &str,
        dtype: DType,
    ) -> Result<String, String> {
        Self::validate_dtype(dtype)?;

        let key = Self::cache_key_with_dtype(graph, dtype);

        if let Ok(cache) = self.cache.read() {
            if let Some(source) = cache.get(&key) {
                return Ok(source.clone());
            }
        }

        let source = Self::emit_kernel_named(graph, name, dtype);

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key, source.clone());
        }

        Ok(source)
    }

    /// Number of cached kernels.
    pub fn cache_size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Clear the codegen cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Validate that the dtype is supported by the fusion codegen.
    fn validate_dtype(dtype: DType) -> Result<(), String> {
        match dtype {
            DType::Float32 | DType::Float16 | DType::Bfloat16 => Ok(()),
            _ => Err(format!(
                "fusion codegen does not support dtype: {:?}",
                dtype
            )),
        }
    }

    /// Compute a cache key that incorporates both the graph topology and dtype.
    fn cache_key_with_dtype(graph: &FusionGraph, dtype: DType) -> u64 {
        // Mix dtype into the graph cache key using a golden-ratio hash.
        graph.cache_key() ^ (dtype as u64).wrapping_mul(0x9e3779b97f4a7c15)
    }

    /// Map DType to Metal type string.
    fn metal_type_str(dtype: DType) -> &'static str {
        match dtype {
            DType::Float32 => "float",
            DType::Float16 => "half",
            DType::Bfloat16 => "bfloat16_t",
            _ => unreachable!("validate_dtype should have caught this"),
        }
    }

    /// Whether an op requires f32 promotion for numerical accuracy in f16/bf16.
    ///
    /// Transcendental functions (exp, log, tanh, etc.) lose significant
    /// precision in half-precision; we promote their inputs to float, compute
    /// in float, then cast back.
    fn needs_f32_promotion(op: &FusableOp) -> bool {
        matches!(
            op,
            FusableOp::Exp
                | FusableOp::Log
                | FusableOp::Sigmoid
                | FusableOp::Tanh
                | FusableOp::SiLU
                | FusableOp::GELU
                | FusableOp::Pow
                | FusableOp::Mod
        )
    }

    /// Emit a Metal kernel source for the given fusion graph with a specific
    /// kernel function name and element dtype.
    fn emit_kernel_named(graph: &FusionGraph, name: &str, dtype: DType) -> String {
        let n_inputs = graph.n_inputs();
        let n_outputs = graph.n_outputs();
        let ops = graph.ops();
        let n_ops = ops.len();
        let mt = Self::metal_type_str(dtype);
        let is_reduced = dtype != DType::Float32;

        let mut src = String::with_capacity(2048);
        src.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

        // Kernel signature
        src.push_str(&format!("kernel void {name}(\n"));

        // Input buffers
        for i in 0..n_inputs {
            src.push_str(&format!("    device const {mt}* in{i} [[buffer({i})]],\n"));
        }

        // Output buffers
        for o in 0..n_outputs {
            let buf_idx = n_inputs + o;
            src.push_str(&format!("    device {mt}* out{o} [[buffer({buf_idx})]],\n"));
        }

        // Size parameter and thread position
        let params_idx = n_inputs + n_outputs;
        src.push_str(&format!("    constant uint& N [[buffer({params_idx})]],\n"));
        src.push_str("    uint tid [[thread_position_in_grid]])\n");
        src.push_str("{\n");
        src.push_str("    if (tid >= N) return;\n\n");

        // Load inputs
        for i in 0..n_inputs {
            src.push_str(&format!("    {mt} v{i} = in{i}[tid];\n"));
        }
        src.push('\n');

        // Compute ops
        for (idx, (op, inputs)) in ops.iter().enumerate() {
            let node_idx = n_inputs + idx;
            let args: Vec<String> = inputs.iter().map(|&i| format!("v{i}")).collect();

            if is_reduced && Self::needs_f32_promotion(op) {
                // Promote inputs to float, compute in float, cast back.
                let promoted_args: Vec<String> =
                    args.iter().map(|a| format!("float({a})")).collect();
                let expr = op.metal_expr(&promoted_args);
                src.push_str(&format!("    {mt} v{node_idx} = {mt}({expr});\n"));
            } else {
                let expr = op.metal_expr(&args);
                src.push_str(&format!("    {mt} v{node_idx} = {expr};\n"));
            }
        }
        src.push('\n');

        // Store outputs (last n_outputs nodes)
        for o in 0..n_outputs {
            let node_idx = n_inputs + n_ops - n_outputs + o;
            src.push_str(&format!("    out{o}[tid] = v{node_idx};\n"));
        }

        src.push_str("}\n");
        src
    }
}

impl Default for FusionCodegen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::graph::{FusableOp, FusionGraph};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLDevice;
    use std::sync::OnceLock;

    fn test_device() -> &'static ProtocolObject<dyn MTLDevice> {
        static DEVICE: OnceLock<Retained<ProtocolObject<dyn MTLDevice>>> = OnceLock::new();
        DEVICE.get_or_init(|| crate::test_utils::shared_metal_device())
    }

    #[test]
    fn test_codegen_simple_add() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");

        assert!(source.contains("kernel void fused_elementwise"));
        assert!(source.contains("device const float* in0"));
        assert!(source.contains("device const float* in1"));
        assert!(source.contains("device float* out0"));
        assert!(source.contains("float v2 = (v0 + v1)"));
        assert!(source.contains("out0[tid] = v2"));
    }

    #[test]
    fn test_codegen_chain() {
        // (a + b) * sigmoid(c)
        let mut g = FusionGraph::new(3);
        let add = g.add_op(FusableOp::Add, vec![0, 1]);
        let sig = g.add_op(FusableOp::Sigmoid, vec![2]);
        g.add_op(FusableOp::Mul, vec![add, sig]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");

        assert!(source.contains("float v3 = (v0 + v1)"));
        assert!(source.contains("float v4 = (1.0f / (1.0f + exp(-v2)))"));
        assert!(source.contains("float v5 = (v3 * v4)"));
        assert!(source.contains("out0[tid] = v5"));
    }

    #[test]
    fn test_codegen_silu() {
        let mut g = FusionGraph::new(1);
        g.add_op(FusableOp::SiLU, vec![0]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");

        assert!(source.contains("float v1 = (v0 / (1.0f + exp(-v0)))"));
    }

    #[test]
    fn test_codegen_caching() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let s1 = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");
        assert_eq!(codegen.cache_size(), 1);

        let s2 = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");
        assert_eq!(codegen.cache_size(), 1);
        assert_eq!(s1, s2);

        codegen.clear_cache();
        assert_eq!(codegen.cache_size(), 0);
    }

    #[test]
    fn test_codegen_rejects_unsupported_dtype() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let result = codegen.generate(&g, DType::UInt32);
        assert!(result.is_err(), "UInt32 should be rejected");
    }

    #[test]
    fn test_codegen_f16_simple_add() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float16)
            .expect("generate failed");

        assert!(source.contains("device const half* in0"));
        assert!(source.contains("device const half* in1"));
        assert!(source.contains("device half* out0"));
        assert!(source.contains("half v2 = (v0 + v1)"));
        // Arithmetic ops should NOT be promoted to f32
        assert!(!source.contains("float(v0)"));
    }

    #[test]
    fn test_codegen_f16_transcendental_promotion() {
        // Sigmoid requires f32 promotion in f16 mode
        let mut g = FusionGraph::new(1);
        g.add_op(FusableOp::Sigmoid, vec![0]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float16)
            .expect("generate failed");

        // Input should be half
        assert!(source.contains("half v0 = in0[tid]"));
        // Sigmoid should be promoted: half(float_expr)
        assert!(source.contains("half v1 = half("));
        assert!(source.contains("float(v0)"));
    }

    #[test]
    fn test_codegen_f16_silu_promotion() {
        let mut g = FusionGraph::new(1);
        g.add_op(FusableOp::SiLU, vec![0]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float16)
            .expect("generate failed");

        // SiLU should be promoted to float, then cast back to half
        assert!(source.contains("half v1 = half("));
        assert!(source.contains("float(v0)"));
    }

    #[test]
    fn test_codegen_bf16_types() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Bfloat16)
            .expect("generate failed");

        assert!(source.contains("device const bfloat16_t* in0"));
        assert!(source.contains("device bfloat16_t* out0"));
        assert!(source.contains("bfloat16_t v2 = (v0 + v1)"));
    }

    #[test]
    fn test_codegen_dtype_cache_separation() {
        // Same graph with different dtypes should produce different cache entries
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let f32_src = codegen.generate(&g, DType::Float32).expect("f32");
        let f16_src = codegen.generate(&g, DType::Float16).expect("f16");

        assert_ne!(f32_src, f16_src);
        assert_eq!(codegen.cache_size(), 2);
    }

    #[test]
    fn test_codegen_compiles() {
        use objc2_metal::MTLDevice as _;
        // Verify the generated source actually compiles with Metal
        let device = test_device();

        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();

        // Test f32
        let source = codegen
            .generate(&g, DType::Float32)
            .expect("generate failed");
        let options = objc2_metal::MTLCompileOptions::new();
        let ns_source = objc2_foundation::NSString::from_str(&source);
        let result = device.newLibraryWithSource_options_error(&ns_source, Some(&options));
        assert!(
            result.is_ok(),
            "f32 source should compile: {:?}",
            result.err()
        );

        // Test f16
        let source_f16 = codegen
            .generate(&g, DType::Float16)
            .expect("generate failed");
        let ns_source_f16 = objc2_foundation::NSString::from_str(&source_f16);
        let result_f16 = device.newLibraryWithSource_options_error(&ns_source_f16, Some(&options));
        assert!(
            result_f16.is_ok(),
            "f16 source should compile: {:?}",
            result_f16.err()
        );
    }

    #[test]
    fn test_codegen_f16_chain_with_promotion() {
        // (a + b) * sigmoid(c) — add is native half, sigmoid is promoted
        let mut g = FusionGraph::new(3);
        let add = g.add_op(FusableOp::Add, vec![0, 1]);
        let sig = g.add_op(FusableOp::Sigmoid, vec![2]);
        g.add_op(FusableOp::Mul, vec![add, sig]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen
            .generate(&g, DType::Float16)
            .expect("generate failed");

        // Add should be native half (no promotion)
        assert!(source.contains("half v3 = (v0 + v1)"));
        // Sigmoid should be promoted
        assert!(source.contains("half v4 = half("));
        // Mul should be native half
        assert!(source.contains("half v5 = (v3 * v4)"));
    }
}
