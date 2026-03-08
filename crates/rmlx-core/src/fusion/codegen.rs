//! Metal source code generation for fused element-wise kernels.

use std::collections::HashMap;
use std::sync::RwLock;

use super::graph::{FusableOp, FusionGraph};

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
    /// Returns an error if the graph contains ops not yet supported by the
    /// codegen (e.g., cast ops that require non-f32 buffer types).
    pub fn generate(&self, graph: &FusionGraph) -> Result<String, String> {
        // Validate: current implementation only supports f32 element-wise ops.
        // Cast ops imply non-f32 inputs or outputs, but the generated kernel
        // hardcodes `device const float*` / `device float*` for all buffers.
        // Allowing cast ops would silently read/write with the wrong element
        // size, causing memory corruption.
        for (op, _) in graph.ops() {
            match op {
                FusableOp::CastF16ToF32 | FusableOp::CastBf16ToF32 | FusableOp::CastF32ToF16 => {
                    return Err(format!(
                        "fusion codegen does not yet support cast ops: {:?}",
                        op
                    ));
                }
                _ => {}
            }
        }

        let key = graph.cache_key();

        // Check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(source) = cache.get(&key) {
                return Ok(source.clone());
            }
        }

        // Generate with default name
        let source = Self::emit_kernel_named(graph, "fused_elementwise");

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
    pub fn generate_named(&self, graph: &FusionGraph, name: &str) -> Result<String, String> {
        for (op, _) in graph.ops() {
            match op {
                FusableOp::CastF16ToF32 | FusableOp::CastBf16ToF32 | FusableOp::CastF32ToF16 => {
                    return Err(format!(
                        "fusion codegen does not yet support cast ops: {:?}",
                        op
                    ));
                }
                _ => {}
            }
        }

        let key = graph.cache_key();

        if let Ok(cache) = self.cache.read() {
            if let Some(source) = cache.get(&key) {
                return Ok(source.clone());
            }
        }

        let source = Self::emit_kernel_named(graph, name);

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

    /// Emit a Metal kernel source for the given fusion graph with a specific
    /// kernel function name.
    fn emit_kernel_named(graph: &FusionGraph, name: &str) -> String {
        let n_inputs = graph.n_inputs();
        let n_outputs = graph.n_outputs();
        let ops = graph.ops();
        let n_ops = ops.len();

        let mut src = String::with_capacity(2048);
        src.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

        // Kernel signature
        src.push_str(&format!("kernel void {name}(\n"));

        // Input buffers
        for i in 0..n_inputs {
            src.push_str(&format!("    device const float* in{i} [[buffer({i})]],\n"));
        }

        // Output buffers
        for o in 0..n_outputs {
            let buf_idx = n_inputs + o;
            src.push_str(&format!(
                "    device float* out{o} [[buffer({buf_idx})]],\n"
            ));
        }

        // Size parameter and thread position
        let params_idx = n_inputs + n_outputs;
        src.push_str(&format!("    constant uint& N [[buffer({params_idx})]],\n"));
        src.push_str("    uint tid [[thread_position_in_grid]])\n");
        src.push_str("{\n");
        src.push_str("    if (tid >= N) return;\n\n");

        // Load inputs
        for i in 0..n_inputs {
            src.push_str(&format!("    float v{i} = in{i}[tid];\n"));
        }
        src.push('\n');

        // Compute ops
        for (idx, (op, inputs)) in ops.iter().enumerate() {
            let node_idx = n_inputs + idx;
            let args: Vec<String> = inputs.iter().map(|&i| format!("v{i}")).collect();
            let expr = op.metal_expr(&args);
            src.push_str(&format!("    float v{node_idx} = {expr};\n"));
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

    #[test]
    fn test_codegen_simple_add() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen.generate(&g).expect("generate failed");

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
        let source = codegen.generate(&g).expect("generate failed");

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
        let source = codegen.generate(&g).expect("generate failed");

        assert!(source.contains("float v1 = (v0 / (1.0f + exp(-v0)))"));
    }

    #[test]
    fn test_codegen_caching() {
        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let s1 = codegen.generate(&g).expect("generate failed");
        assert_eq!(codegen.cache_size(), 1);

        let s2 = codegen.generate(&g).expect("generate failed");
        assert_eq!(codegen.cache_size(), 1);
        assert_eq!(s1, s2);

        codegen.clear_cache();
        assert_eq!(codegen.cache_size(), 0);
    }

    #[test]
    fn test_codegen_rejects_cast_ops() {
        let mut g = FusionGraph::new(1);
        g.add_op(FusableOp::CastF16ToF32, vec![0]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let result = codegen.generate(&g);
        assert!(result.is_err(), "cast ops should be rejected");
        assert!(result.unwrap_err().contains("CastF16ToF32"));

        let mut g2 = FusionGraph::new(1);
        g2.add_op(FusableOp::CastF32ToF16, vec![0]);
        g2.set_outputs(1);
        assert!(codegen.generate(&g2).is_err());

        let mut g3 = FusionGraph::new(1);
        g3.add_op(FusableOp::CastBf16ToF32, vec![0]);
        g3.set_outputs(1);
        assert!(codegen.generate(&g3).is_err());
    }

    #[test]
    fn test_codegen_compiles() {
        // Verify the generated source actually compiles with Metal
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("skipping test: no Metal device");
                return;
            }
        };

        let mut g = FusionGraph::new(2);
        g.add_op(FusableOp::Add, vec![0, 1]);
        g.set_outputs(1);

        let codegen = FusionCodegen::new();
        let source = codegen.generate(&g).expect("generate failed");

        let options = metal::CompileOptions::new();
        let result = device.new_library_with_source(&source, &options);
        assert!(
            result.is_ok(),
            "generated source should compile: {:?}",
            result.err()
        );
    }
}
