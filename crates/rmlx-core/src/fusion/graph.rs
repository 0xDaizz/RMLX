//! Fusable operation graph for element-wise kernel fusion.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Element-wise operations that can be fused into a single kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusableOp {
    // Unary (14)
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Abs,
    Sigmoid,
    Tanh,
    SiLU,
    GELU,
    Relu,
    CastF16ToF32,
    CastF32ToF16,
    CastBf16ToF32,
    // Binary (8)
    Add,
    Mul,
    Sub,
    Div,
    Max,
    Min,
    Pow,
    Mod,
    // Ternary (1)
    Select,
}

impl FusableOp {
    /// Number of inputs this op requires.
    pub fn arity(&self) -> usize {
        match self {
            Self::Neg
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Sigmoid
            | Self::Tanh
            | Self::SiLU
            | Self::GELU
            | Self::Relu
            | Self::CastF16ToF32
            | Self::CastF32ToF16
            | Self::CastBf16ToF32 => 1,
            Self::Add
            | Self::Mul
            | Self::Sub
            | Self::Div
            | Self::Max
            | Self::Min
            | Self::Pow
            | Self::Mod => 2,
            Self::Select => 3,
        }
    }

    /// Metal expression for this op.
    pub fn metal_expr(&self, args: &[String]) -> String {
        match self {
            Self::Neg => format!("(-{})", args[0]),
            Self::Exp => format!("exp({})", args[0]),
            Self::Log => format!("log({})", args[0]),
            Self::Sqrt => format!("sqrt({})", args[0]),
            Self::Rsqrt => format!("rsqrt({})", args[0]),
            Self::Abs => format!("abs({})", args[0]),
            Self::Sigmoid => format!("(1.0f / (1.0f + exp(-{})))", args[0]),
            Self::Tanh => format!("tanh({})", args[0]),
            Self::SiLU => format!("({0} / (1.0f + exp(-{0})))", args[0]),
            Self::GELU => format!(
                "({0} * 0.5f * (1.0f + tanh(0.7978845608f * ({0} + 0.044715f * {0} * {0} * {0}))))",
                args[0]
            ),
            Self::Relu => format!("max({}, 0.0f)", args[0]),
            Self::CastF16ToF32 => format!("float({})", args[0]),
            Self::CastF32ToF16 => format!("half({})", args[0]),
            Self::CastBf16ToF32 => format!("float({})", args[0]),
            Self::Add => format!("({} + {})", args[0], args[1]),
            Self::Mul => format!("({} * {})", args[0], args[1]),
            Self::Sub => format!("({} - {})", args[0], args[1]),
            Self::Div => format!("({} / {})", args[0], args[1]),
            Self::Max => format!("max({}, {})", args[0], args[1]),
            Self::Min => format!("min({}, {})", args[0], args[1]),
            Self::Pow => format!("pow({}, {})", args[0], args[1]),
            Self::Mod => format!("fmod({}, {})", args[0], args[1]),
            Self::Select => format!("({} ? {} : {})", args[0], args[1], args[2]),
        }
    }
}

/// A graph of fusable element-wise operations.
///
/// Each node is an op with references to its input nodes (by index).
/// Leaf nodes (inputs) are represented by their input index.
#[derive(Debug, Clone)]
pub struct FusionGraph {
    /// (op, input_node_indices) for each node.
    ops: Vec<(FusableOp, Vec<usize>)>,
    /// Number of external input arrays.
    n_inputs: usize,
    /// Number of output nodes (always the last `n_outputs` nodes).
    n_outputs: usize,
}

impl FusionGraph {
    /// Create a new fusion graph.
    pub fn new(n_inputs: usize) -> Self {
        Self {
            ops: Vec::new(),
            n_inputs,
            n_outputs: 0,
        }
    }

    /// Add an operation node. Returns the node index.
    ///
    /// `inputs` are indices into the node list. For the first `n_inputs`
    /// nodes, they refer to external input arrays.
    pub fn add_op(&mut self, op: FusableOp, inputs: Vec<usize>) -> usize {
        let idx = self.n_inputs + self.ops.len();
        self.ops.push((op, inputs));
        idx
    }

    /// Mark the last `n` nodes as outputs.
    pub fn set_outputs(&mut self, n: usize) {
        self.n_outputs = n;
    }

    /// Number of operations in the graph.
    pub fn depth(&self) -> usize {
        self.ops.len()
    }

    /// Number of external inputs.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Get the operations.
    pub fn ops(&self) -> &[(FusableOp, Vec<usize>)] {
        &self.ops
    }

    /// Whether this graph exceeds the fusion limits.
    pub fn exceeds_limits(&self) -> bool {
        self.ops.len() > super::MAX_FUSION_DEPTH
            || (self.n_inputs + self.n_outputs) > super::MAX_FUSION_ARRAYS
    }

    /// Compute a stable hash for cache lookup.
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.n_inputs.hash(&mut hasher);
        self.n_outputs.hash(&mut hasher);
        for (op, inputs) in &self.ops {
            op.hash(&mut hasher);
            inputs.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusable_op_arity() {
        assert_eq!(FusableOp::Neg.arity(), 1);
        assert_eq!(FusableOp::Add.arity(), 2);
        assert_eq!(FusableOp::Select.arity(), 3);
    }

    #[test]
    fn test_fusable_op_metal_expr() {
        let args1 = vec!["x".to_string()];
        assert_eq!(FusableOp::Neg.metal_expr(&args1), "(-x)");
        assert_eq!(FusableOp::Exp.metal_expr(&args1), "exp(x)");
        assert_eq!(FusableOp::Relu.metal_expr(&args1), "max(x, 0.0f)");

        let args2 = vec!["a".to_string(), "b".to_string()];
        assert_eq!(FusableOp::Add.metal_expr(&args2), "(a + b)");
        assert_eq!(FusableOp::Mul.metal_expr(&args2), "(a * b)");

        let args3 = vec!["c".to_string(), "x".to_string(), "y".to_string()];
        assert_eq!(FusableOp::Select.metal_expr(&args3), "(c ? x : y)");
    }

    #[test]
    fn test_fusion_graph_basic() {
        // (a + b) * sigmoid(c)
        let mut g = FusionGraph::new(3); // inputs: a, b, c
        let add = g.add_op(FusableOp::Add, vec![0, 1]); // node 3
        let sig = g.add_op(FusableOp::Sigmoid, vec![2]); // node 4
        let _mul = g.add_op(FusableOp::Mul, vec![add, sig]); // node 5
        g.set_outputs(1);

        assert_eq!(g.depth(), 3);
        assert_eq!(g.n_inputs(), 3);
        assert_eq!(g.n_outputs(), 1);
        assert!(!g.exceeds_limits());
    }

    #[test]
    fn test_fusion_graph_cache_key() {
        let mut g1 = FusionGraph::new(2);
        g1.add_op(FusableOp::Add, vec![0, 1]);
        g1.set_outputs(1);

        let mut g2 = FusionGraph::new(2);
        g2.add_op(FusableOp::Add, vec![0, 1]);
        g2.set_outputs(1);

        let mut g3 = FusionGraph::new(2);
        g3.add_op(FusableOp::Mul, vec![0, 1]);
        g3.set_outputs(1);

        assert_eq!(g1.cache_key(), g2.cache_key(), "same graph = same key");
        assert_ne!(
            g1.cache_key(),
            g3.cache_key(),
            "different op = different key"
        );
    }

    #[test]
    fn test_fusion_graph_exceeds_limits() {
        let mut g = FusionGraph::new(2);
        for i in 0..12 {
            g.add_op(FusableOp::Neg, vec![if i == 0 { 0 } else { 2 + i - 1 }]);
        }
        g.set_outputs(1);
        assert!(g.exceeds_limits(), "12 ops exceeds MAX_FUSION_DEPTH=11");
    }
}
