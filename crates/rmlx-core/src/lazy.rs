//! Lazy evaluation infrastructure for RMLX.
//!
//! Provides [`LazyArray`], a wrapper that records operations into a compute DAG
//! instead of executing them eagerly. Calling [`LazyArray::eval`] materializes
//! the entire pending subgraph via topological-sort traversal.
//!
//! This module intentionally omits graph optimization and fusion — those can be
//! layered on top in a future pass.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::array::Array;
use crate::dtype::DType;
use objc2::runtime::ProtocolObject;

// ---------------------------------------------------------------------------
// LazyOp — the operation variants recorded in the graph
// ---------------------------------------------------------------------------

/// An operation recorded in the lazy compute graph.
///
/// Each variant stores the indices of its input nodes within the graph's
/// node vector, plus any auxiliary parameters needed at evaluation time.
#[derive(Debug, Clone)]
pub enum LazyOp {
    /// Element-wise addition of two arrays.
    Add(NodeId, NodeId),
    /// Element-wise multiplication of two arrays.
    Mul(NodeId, NodeId),
    /// Matrix multiplication of two arrays.
    MatMul(NodeId, NodeId),
    /// Element-wise subtraction (lhs - rhs).
    Sub(NodeId, NodeId),
    /// Element-wise negation.
    Neg(NodeId),
    /// Softmax along the last axis.
    Softmax(NodeId),
    /// Rotary positional embedding.
    RoPE(NodeId),
    /// Contiguous copy.
    Copy(NodeId),
    /// RMS normalization.
    RmsNorm(NodeId),
    /// Custom / user-defined operation with a string tag (for extensibility).
    Custom(String, Vec<NodeId>),
}

impl LazyOp {
    /// Return the node IDs of all inputs to this operation.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            LazyOp::Add(a, b) | LazyOp::Mul(a, b) | LazyOp::MatMul(a, b) | LazyOp::Sub(a, b) => {
                vec![*a, *b]
            }
            LazyOp::Neg(a)
            | LazyOp::Softmax(a)
            | LazyOp::RoPE(a)
            | LazyOp::Copy(a)
            | LazyOp::RmsNorm(a) => vec![*a],
            LazyOp::Custom(_, ids) => ids.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeId / LazyNode — graph building blocks
// ---------------------------------------------------------------------------

/// Opaque handle into the graph's node vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

/// A single node in the compute DAG.
#[derive(Debug)]
struct LazyNode {
    /// The operation that produces this node's value.
    /// `None` means the node is a leaf (already materialized).
    op: Option<LazyOp>,
    /// Cached output shape (known at graph-build time for validation).
    shape: Vec<usize>,
    /// Cached output dtype.
    dtype: DType,
    /// The materialized result, if already evaluated.
    value: Option<Array>,
}

// ---------------------------------------------------------------------------
// LazyGraph — the DAG container
// ---------------------------------------------------------------------------

/// A compute DAG that accumulates lazy operations.
///
/// Thread-safe via interior `Mutex`. Nodes are append-only; once added, a
/// node's index never changes.
#[derive(Debug, Clone)]
pub struct LazyGraph {
    inner: Arc<Mutex<Vec<LazyNode>>>,
}

impl Default for LazyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyGraph {
    /// Create an empty graph.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Number of nodes currently in the graph.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add an already-materialized array as a leaf node. Returns its [`NodeId`].
    pub fn add_value(&self, array: Array) -> NodeId {
        let mut nodes = self.inner.lock().unwrap();
        let id = NodeId(nodes.len());
        nodes.push(LazyNode {
            op: None,
            shape: array.shape().to_vec(),
            dtype: array.dtype(),
            value: Some(array),
        });
        id
    }

    /// Add a deferred operation node. Returns its [`NodeId`].
    ///
    /// `shape` and `dtype` describe the expected output and are stored for
    /// downstream validation without requiring eager computation.
    pub fn add_op(&self, op: LazyOp, shape: Vec<usize>, dtype: DType) -> NodeId {
        let mut nodes = self.inner.lock().unwrap();
        let id = NodeId(nodes.len());
        nodes.push(LazyNode {
            op: Some(op),
            shape,
            dtype,
            value: None,
        });
        id
    }

    /// Check if a node has already been materialized.
    pub fn is_materialized(&self, id: NodeId) -> bool {
        let nodes = self.inner.lock().unwrap();
        nodes.get(id.0).is_some_and(|n| n.value.is_some())
    }

    /// Get the shape of a node.
    pub fn shape(&self, id: NodeId) -> Option<Vec<usize>> {
        let nodes = self.inner.lock().unwrap();
        nodes.get(id.0).map(|n| n.shape.clone())
    }

    /// Get the dtype of a node.
    pub fn dtype(&self, id: NodeId) -> Option<DType> {
        let nodes = self.inner.lock().unwrap();
        nodes.get(id.0).map(|n| n.dtype)
    }
}

// ---------------------------------------------------------------------------
// LazyArray — the user-facing handle
// ---------------------------------------------------------------------------

/// A lazily-evaluated array.
///
/// Wraps either:
/// - An already-materialized [`Array`], or
/// - A deferred computation node inside a [`LazyGraph`].
///
/// Call [`eval`](LazyArray::eval) to walk the DAG and materialize all pending
/// computations. The evaluation function is supplied by the caller so that
/// `lazy.rs` does not depend on Metal kernel infrastructure directly.
#[derive(Clone)]
pub struct LazyArray {
    graph: LazyGraph,
    node: NodeId,
}

impl fmt::Debug for LazyArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let materialized = self.graph.is_materialized(self.node);
        f.debug_struct("LazyArray")
            .field("node", &self.node)
            .field("materialized", &materialized)
            .field("shape", &self.graph.shape(self.node))
            .field("dtype", &self.graph.dtype(self.node))
            .finish()
    }
}

impl LazyArray {
    // ── Constructors ─────────────────────────────────────────────────

    /// Wrap an already-materialized array.
    pub fn from_array(graph: &LazyGraph, array: Array) -> Self {
        let node = graph.add_value(array);
        Self {
            graph: graph.clone(),
            node,
        }
    }

    /// Create a lazy array backed by a deferred operation.
    pub fn from_op(graph: &LazyGraph, op: LazyOp, shape: Vec<usize>, dtype: DType) -> Self {
        let node = graph.add_op(op, shape, dtype);
        Self {
            graph: graph.clone(),
            node,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// The node ID within the graph.
    pub fn node_id(&self) -> NodeId {
        self.node
    }

    /// Reference to the underlying graph.
    pub fn graph(&self) -> &LazyGraph {
        &self.graph
    }

    /// Whether the underlying value has already been computed.
    pub fn is_materialized(&self) -> bool {
        self.graph.is_materialized(self.node)
    }

    /// The expected output shape.
    pub fn shape(&self) -> Vec<usize> {
        self.graph.shape(self.node).unwrap_or_default()
    }

    /// The expected output dtype.
    pub fn dtype(&self) -> DType {
        self.graph.dtype(self.node).unwrap_or(DType::Float32)
    }

    // ── Graph builder helpers ────────────────────────────────────────

    /// Record `self + rhs` as a lazy Add node.
    pub fn add(&self, rhs: &LazyArray) -> Result<LazyArray, LazyEvalError> {
        let out_shape = crate::array::broadcast_shape(&self.shape(), &rhs.shape())
            .map_err(|e| LazyEvalError::ShapeMismatch(e.to_string()))?;
        let dtype = self.dtype();
        Ok(LazyArray::from_op(
            &self.graph,
            LazyOp::Add(self.node, rhs.node),
            out_shape,
            dtype,
        ))
    }

    /// Record `self * rhs` as a lazy Mul node.
    pub fn mul(&self, rhs: &LazyArray) -> Result<LazyArray, LazyEvalError> {
        let out_shape = crate::array::broadcast_shape(&self.shape(), &rhs.shape())
            .map_err(|e| LazyEvalError::ShapeMismatch(e.to_string()))?;
        let dtype = self.dtype();
        Ok(LazyArray::from_op(
            &self.graph,
            LazyOp::Mul(self.node, rhs.node),
            out_shape,
            dtype,
        ))
    }

    /// Record `self - rhs` as a lazy Sub node.
    pub fn sub(&self, rhs: &LazyArray) -> Result<LazyArray, LazyEvalError> {
        let out_shape = crate::array::broadcast_shape(&self.shape(), &rhs.shape())
            .map_err(|e| LazyEvalError::ShapeMismatch(e.to_string()))?;
        let dtype = self.dtype();
        Ok(LazyArray::from_op(
            &self.graph,
            LazyOp::Sub(self.node, rhs.node),
            out_shape,
            dtype,
        ))
    }

    /// Record `self @ rhs` (matmul) as a lazy MatMul node.
    ///
    /// Only validates that trailing dimensions are compatible. Full shape
    /// inference follows standard matrix multiplication rules.
    pub fn matmul(&self, rhs: &LazyArray) -> Result<LazyArray, LazyEvalError> {
        let ls = self.shape();
        let rs = rhs.shape();
        if ls.len() < 2 || rs.len() < 2 {
            return Err(LazyEvalError::ShapeMismatch(format!(
                "matmul requires at least 2D arrays, got {:?} and {:?}",
                ls, rs
            )));
        }
        let m = ls[ls.len() - 2];
        let k1 = ls[ls.len() - 1];
        let k2 = rs[rs.len() - 2];
        let n = rs[rs.len() - 1];
        if k1 != k2 {
            return Err(LazyEvalError::ShapeMismatch(format!(
                "matmul inner dimensions mismatch: {} vs {}",
                k1, k2
            )));
        }
        // Output shape: broadcast batch dims + [m, n]
        let batch_a = &ls[..ls.len() - 2];
        let batch_b = &rs[..rs.len() - 2];
        let batch_out = crate::array::broadcast_shape(batch_a, batch_b)
            .map_err(|e| LazyEvalError::ShapeMismatch(e.to_string()))?;
        let mut out_shape = batch_out;
        out_shape.push(m);
        out_shape.push(n);

        let dtype = self.dtype();
        Ok(LazyArray::from_op(
            &self.graph,
            LazyOp::MatMul(self.node, rhs.node),
            out_shape,
            dtype,
        ))
    }

    /// Record unary negation.
    pub fn neg(&self) -> LazyArray {
        LazyArray::from_op(
            &self.graph,
            LazyOp::Neg(self.node),
            self.shape(),
            self.dtype(),
        )
    }

    /// Record softmax.
    pub fn softmax(&self) -> LazyArray {
        LazyArray::from_op(
            &self.graph,
            LazyOp::Softmax(self.node),
            self.shape(),
            self.dtype(),
        )
    }

    /// Record a contiguous copy.
    pub fn copy(&self) -> LazyArray {
        LazyArray::from_op(
            &self.graph,
            LazyOp::Copy(self.node),
            self.shape(),
            self.dtype(),
        )
    }

    // ── Evaluation ───────────────────────────────────────────────────

    /// Evaluate this node and all of its dependencies.
    ///
    /// `eval_fn` is called for each non-materialized node in topological
    /// order. It receives the operation and the already-materialized input
    /// arrays, and must return the output `Array`.
    ///
    /// If the node is already materialized, this is a no-op and returns
    /// immediately.
    pub fn eval<F>(&self, eval_fn: &F) -> Result<(), LazyEvalError>
    where
        F: Fn(&LazyOp, Vec<&Array>) -> Result<Array, LazyEvalError>,
    {
        let order = self.topo_sort()?;
        let mut nodes = self.graph.inner.lock().unwrap();

        for nid in order {
            if nodes[nid.0].value.is_some() {
                continue; // already materialized
            }
            let op = nodes[nid.0]
                .op
                .clone()
                .ok_or(LazyEvalError::MissingOp(nid))?;

            // Gather input arrays (all must be materialized by now due to
            // topological ordering).
            let input_ids = op.inputs();
            // We need to borrow immutably for inputs while mutating the
            // current node. Collect input refs via raw index access since
            // we know they are disjoint from nid.
            let input_ptrs: Vec<*const Array> = input_ids
                .iter()
                .map(|id| {
                    nodes[id.0]
                        .value
                        .as_ref()
                        .ok_or(LazyEvalError::UnmaterializedInput(*id))
                        .map(|a| a as *const Array)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // SAFETY: We only hold shared references to nodes that are
            // distinct from `nid` (topo order guarantees inputs precede
            // the current node, so `id.0 != nid.0`). We immediately
            // consume these references before mutating `nodes[nid.0]`.
            let input_refs: Vec<&Array> = input_ptrs.iter().map(|p| unsafe { &**p }).collect();

            let result = eval_fn(&op, input_refs)?;
            nodes[nid.0].value = Some(result);
        }

        Ok(())
    }

    /// Evaluate with fusion: partition the graph into fusable and standalone
    /// segments, dispatch fused element-wise kernels via JIT codegen, and
    /// run standalone ops through `standalone_fn`.
    ///
    /// When `ctx.exec_graph` is `Some`, fused kernels are encoded into the
    /// ExecGraph's command buffers. Otherwise they dispatch standalone via
    /// the queue.
    ///
    /// `standalone_fn` handles non-fusable ops (MatMul, Softmax, RoPE, etc.)
    /// and receives the `EvalContext` for dispatch.
    pub fn eval_fused<F>(
        &self,
        ctx: &mut EvalContext<'_, '_, '_>,
        standalone_fn: &F,
    ) -> Result<(), LazyEvalError>
    where
        F: Fn(&LazyOp, Vec<&Array>, &mut EvalContext<'_, '_, '_>) -> Result<Array, LazyEvalError>,
    {
        let order = self.topo_sort()?;
        let mut nodes = self.graph.inner.lock().unwrap();

        // Build topo_ops list (only non-materialized nodes)
        let topo_ops: Vec<(NodeId, LazyOp)> = order
            .iter()
            .filter_map(|nid| {
                if nodes[nid.0].value.is_some() {
                    None
                } else {
                    nodes[nid.0].op.clone().map(|op| (*nid, op))
                }
            })
            .collect();

        if topo_ops.is_empty() {
            return Ok(());
        }

        // Build consumer map for partition
        let num_nodes = nodes.len();
        let mut consumers = vec![Vec::new(); num_nodes];
        for (nid, op) in &topo_ops {
            for inp in op.inputs() {
                if inp.0 < num_nodes {
                    consumers[inp.0].push(*nid);
                }
            }
        }

        // Partition into segments
        let segments = crate::fusion::analyzer::partition(&topo_ops, &consumers, num_nodes);

        // Execute each segment
        for segment in &segments {
            match segment {
                crate::fusion::Segment::Fused {
                    graph: fusion_graph,
                    nodes: seg_nodes,
                    input_nodes,
                    ..
                } => {
                    // Try JIT fusion if codegen is available
                    if let Some(codegen) = ctx.codegen {
                        // Gather input arrays
                        let input_arrays: Vec<&Array> = input_nodes
                            .iter()
                            .map(|inp| {
                                nodes[inp.0]
                                    .value
                                    .as_ref()
                                    .expect("fusion input must be materialized")
                            })
                            .collect();

                        let outputs = if let Some(ref mut exec_graph) = ctx.exec_graph {
                            // J-4d: Dispatch into ExecGraph CB
                            let cb = exec_graph.command_buffer();
                            crate::fusion::dispatch_fused_into_cb(
                                fusion_graph,
                                codegen,
                                ctx.registry,
                                &input_arrays,
                                cb,
                            )
                            .map_err(|e| {
                                LazyEvalError::EvalFailed(format!("fused dispatch: {e}"))
                            })?
                        } else {
                            // Standalone dispatch via queue
                            crate::fusion::dispatch_fused(
                                fusion_graph,
                                codegen,
                                ctx.registry,
                                &input_arrays,
                                ctx.queue,
                            )
                            .map_err(|e| {
                                LazyEvalError::EvalFailed(format!("fused dispatch: {e}"))
                            })?
                        };

                        // Store output in the last node of the segment
                        if let Some(output) = outputs.into_iter().next() {
                            let last_node = seg_nodes.last().unwrap();
                            nodes[last_node.0].value = Some(output);
                        }
                    } else {
                        // No codegen: fall back to per-op eval
                        for (nid, op) in &topo_ops {
                            if !seg_nodes.contains(nid) {
                                continue;
                            }
                            if nodes[nid.0].value.is_some() {
                                continue;
                            }
                            let op = op.clone();
                            let input_ids = op.inputs();
                            let input_ptrs: Vec<*const Array> = input_ids
                                .iter()
                                .map(|id| {
                                    nodes[id.0]
                                        .value
                                        .as_ref()
                                        .ok_or(LazyEvalError::UnmaterializedInput(*id))
                                        .map(|a| a as *const Array)
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let input_refs: Vec<&Array> =
                                input_ptrs.iter().map(|p| unsafe { &**p }).collect();
                            let result = standalone_fn(&op, input_refs, ctx)?;
                            nodes[nid.0].value = Some(result);
                        }
                    }
                }
                crate::fusion::Segment::Standalone { node, op } => {
                    if nodes[node.0].value.is_some() {
                        continue;
                    }
                    let input_ids = op.inputs();
                    let input_ptrs: Vec<*const Array> = input_ids
                        .iter()
                        .map(|id| {
                            nodes[id.0]
                                .value
                                .as_ref()
                                .ok_or(LazyEvalError::UnmaterializedInput(*id))
                                .map(|a| a as *const Array)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    // SAFETY: Same reasoning as in eval() — inputs are disjoint from node.
                    let input_refs: Vec<&Array> =
                        input_ptrs.iter().map(|p| unsafe { &**p }).collect();
                    let result = standalone_fn(op, input_refs, ctx)?;
                    nodes[node.0].value = Some(result);
                }
            }
        }

        Ok(())
    }

    /// Extract the materialized array after evaluation.
    ///
    /// Returns `None` if the node has not been evaluated yet.
    pub fn try_get(&self) -> Option<ArrayRef> {
        let nodes = self.graph.inner.lock().unwrap();
        if nodes[self.node.0].value.is_some() {
            Some(ArrayRef {
                graph: self.graph.clone(),
                node: self.node,
            })
        } else {
            None
        }
    }

    // ── Topological sort ─────────────────────────────────────────────

    /// Compute a topological ordering of all nodes reachable from this
    /// node that still need evaluation.
    fn topo_sort(&self) -> Result<Vec<NodeId>, LazyEvalError> {
        let nodes = self.graph.inner.lock().unwrap();
        let n = nodes.len();
        if self.node.0 >= n {
            return Err(LazyEvalError::InvalidNode(self.node));
        }

        // DFS-based topo sort
        #[derive(Clone, Copy, PartialEq)]
        enum State {
            Unvisited,
            InProgress,
            Done,
        }

        let mut state = vec![State::Unvisited; n];
        let mut order = Vec::new();

        // Iterative DFS to avoid stack overflow on deep graphs
        let mut stack: Vec<(NodeId, bool)> = vec![(self.node, false)];

        while let Some((nid, children_pushed)) = stack.pop() {
            if state[nid.0] == State::Done {
                continue;
            }
            if children_pushed {
                state[nid.0] = State::Done;
                order.push(nid);
                continue;
            }
            if state[nid.0] == State::InProgress {
                return Err(LazyEvalError::CycleDetected(nid));
            }
            state[nid.0] = State::InProgress;
            // Push self again with children_pushed=true so we finalize after children
            stack.push((nid, true));

            if let Some(ref op) = nodes[nid.0].op {
                for &input_id in op.inputs().iter().rev() {
                    if input_id.0 >= n {
                        return Err(LazyEvalError::InvalidNode(input_id));
                    }
                    if state[input_id.0] != State::Done {
                        stack.push((input_id, false));
                    }
                }
            }
        }

        Ok(order)
    }
}

// ---------------------------------------------------------------------------
// ArrayRef — safe handle to a materialized value inside the graph
// ---------------------------------------------------------------------------

/// A handle to a materialized array inside a [`LazyGraph`].
///
/// Provides read access to shape/dtype/data without moving the array out
/// of the graph (which would invalidate downstream references).
pub struct ArrayRef {
    graph: LazyGraph,
    node: NodeId,
}

impl ArrayRef {
    /// Access the underlying array by locking the graph.
    ///
    /// The closure receives an immutable reference to the `Array`.
    pub fn with<R, F: FnOnce(&Array) -> R>(&self, f: F) -> R {
        let nodes = self.graph.inner.lock().unwrap();
        let arr = nodes[self.node.0]
            .value
            .as_ref()
            .expect("ArrayRef invariant: node must be materialized");
        f(arr)
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during lazy evaluation.
#[derive(Debug, Clone)]
pub enum LazyEvalError {
    /// A node references an index outside the graph.
    InvalidNode(NodeId),
    /// A leaf node has no operation and no value.
    MissingOp(NodeId),
    /// An input was expected to be materialized but was not.
    UnmaterializedInput(NodeId),
    /// The graph contains a cycle (should be impossible with append-only IDs
    /// but checked defensively).
    CycleDetected(NodeId),
    /// Shape mismatch between operands.
    ShapeMismatch(String),
    /// An error from the evaluation function.
    EvalFailed(String),
}

impl fmt::Display for LazyEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyEvalError::InvalidNode(id) => write!(f, "invalid node {:?}", id),
            LazyEvalError::MissingOp(id) => {
                write!(f, "node {:?} has no op and no value", id)
            }
            LazyEvalError::UnmaterializedInput(id) => {
                write!(f, "input node {:?} not materialized", id)
            }
            LazyEvalError::CycleDetected(id) => {
                write!(f, "cycle detected at node {:?}", id)
            }
            LazyEvalError::ShapeMismatch(msg) => write!(f, "shape mismatch: {}", msg),
            LazyEvalError::EvalFailed(msg) => write!(f, "eval failed: {}", msg),
        }
    }
}

impl std::error::Error for LazyEvalError {}

// ---------------------------------------------------------------------------
// EvalContext — execution context for lazy graph evaluation
// ---------------------------------------------------------------------------

/// Execution context for materializing a lazy compute graph.
///
/// Bundles all the Metal resources needed to dispatch kernels during
/// lazy evaluation. This decouples the lazy graph infrastructure from
/// concrete kernel dispatch, allowing `LazyArray::eval` callers to
/// pass a context instead of individual resource references.
///
/// # Lifetime parameters
///
/// - `'q`: lifetime of the `CommandQueue`
/// - `'e`: lifetime of the `GpuEvent` (used by ExecGraph)
/// - `'r`: lifetime of the `KernelRegistry`
pub struct EvalContext<'q, 'e, 'r> {
    /// Metal device reference.
    pub device: &'q ProtocolObject<dyn objc2_metal::MTLDevice>,
    /// Kernel registry for pipeline state lookups.
    pub registry: &'r crate::kernels::KernelRegistry,
    /// Command queue for dispatch.
    pub queue: &'q rmlx_metal::MtlQueue,
    /// Optional ExecGraph for batched GPU-side execution.
    /// When `Some`, ops should encode into the graph's command buffers
    /// instead of creating standalone CBs. When `None`, ops use
    /// synchronous dispatch via the queue directly.
    pub exec_graph: Option<&'e mut rmlx_metal::exec_graph::ExecGraph<'q, 'e>>,
    /// Optional FusionCodegen for JIT element-wise kernel fusion.
    pub codegen: Option<&'r crate::fusion::FusionCodegen>,
}

impl<'q, 'e, 'r> EvalContext<'q, 'e, 'r> {
    /// Create a minimal context without ExecGraph or fusion.
    pub fn new(
        device: &'q ProtocolObject<dyn objc2_metal::MTLDevice>,
        registry: &'r crate::kernels::KernelRegistry,
        queue: &'q rmlx_metal::MtlQueue,
    ) -> Self {
        Self {
            device,
            registry,
            queue,
            exec_graph: None,
            codegen: None,
        }
    }

    /// Create a context with ExecGraph for batched execution.
    pub fn with_exec_graph(
        device: &'q ProtocolObject<dyn objc2_metal::MTLDevice>,
        registry: &'r crate::kernels::KernelRegistry,
        queue: &'q rmlx_metal::MtlQueue,
        graph: &'e mut rmlx_metal::exec_graph::ExecGraph<'q, 'e>,
    ) -> Self {
        Self {
            device,
            registry,
            queue,
            exec_graph: Some(graph),
            codegen: None,
        }
    }

    /// Attach a FusionCodegen for JIT kernel fusion.
    pub fn with_codegen(mut self, codegen: &'r crate::fusion::FusionCodegen) -> Self {
        self.codegen = Some(codegen);
        self
    }

    /// Whether an ExecGraph is available for batched dispatch.
    pub fn has_exec_graph(&self) -> bool {
        self.exec_graph.is_some()
    }

    /// Whether JIT fusion is available.
    pub fn has_codegen(&self) -> bool {
        self.codegen.is_some()
    }
}

// ---------------------------------------------------------------------------
// BFS width-limited topological sort
// ---------------------------------------------------------------------------

/// BFS topological sort with width limiting.
///
/// Evaluates a DAG in topological order but limits the frontier width
/// to `max_width` nodes. When the frontier exceeds this limit, the
/// algorithm switches to DFS on the deepest branch to reduce the number
/// of live intermediates.
///
/// This prevents memory explosion for wide graphs (e.g., during prefill
/// with many independent attention heads). Does NOT affect the 9-dispatch
/// decode path, which bypasses LazyGraph entirely.
///
/// Returns node IDs in evaluation order.
pub fn topo_sort_bfs_width_limited(
    nodes: &[Option<Vec<NodeId>>], // adjacency: node -> children
    num_nodes: usize,
    max_width: usize,
) -> Vec<NodeId> {
    use std::collections::VecDeque;

    let max_width = max_width.max(1);

    // Compute in-degree
    let mut in_degree = vec![0u32; num_nodes];
    for children in nodes.iter().flatten() {
        for &child in children {
            if child.0 < num_nodes {
                in_degree[child.0] += 1;
            }
        }
    }

    // Initialize frontier with zero-degree nodes
    let mut frontier: VecDeque<NodeId> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            frontier.push_back(NodeId(i));
        }
    }

    let mut order = Vec::with_capacity(num_nodes);

    while !frontier.is_empty() {
        // If frontier exceeds max_width, process deepest node via DFS
        if frontier.len() > max_width {
            // Pop from back (deepest/most recently added)
            let node = frontier.pop_back().unwrap();
            order.push(node);

            if let Some(children) = &nodes[node.0] {
                for &child in children {
                    if child.0 < num_nodes {
                        in_degree[child.0] -= 1;
                        if in_degree[child.0] == 0 {
                            // Push to back for DFS behavior
                            frontier.push_back(child);
                        }
                    }
                }
            }
        } else {
            // Normal BFS: pop from front
            let node = frontier.pop_front().unwrap();
            order.push(node);

            if let Some(children) = &nodes[node.0] {
                for &child in children {
                    if child.0 < num_nodes {
                        in_degree[child.0] -= 1;
                        if in_degree[child.0] == 0 {
                            frontier.push_back(child);
                        }
                    }
                }
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLDevice;
    use std::sync::OnceLock;

    fn test_device() -> &'static ProtocolObject<dyn MTLDevice> {
        static DEVICE: OnceLock<Retained<ProtocolObject<dyn MTLDevice>>> = OnceLock::new();
        DEVICE.get_or_init(|| crate::test_utils::shared_metal_device())
    }

    /// Minimal eval function for testing: uses Metal device to create output
    /// arrays of the correct shape. For Add/Mul/Sub it creates a zeros array
    /// (we only test the graph machinery, not GPU correctness).
    fn test_eval_fn(op: &LazyOp, inputs: Vec<&Array>) -> Result<Array, LazyEvalError> {
        let dev = test_device();

        match op {
            LazyOp::Add(_, _) | LazyOp::Mul(_, _) | LazyOp::Sub(_, _) => {
                let a = inputs[0];
                let b = inputs[1];
                let out_shape = crate::array::broadcast_shape(a.shape(), b.shape())
                    .map_err(|e| LazyEvalError::EvalFailed(e.to_string()))?;
                Ok(Array::zeros(dev, &out_shape, a.dtype()))
            }
            LazyOp::MatMul(_, _) => {
                let a = inputs[0];
                let b = inputs[1];
                let m = a.shape()[a.ndim() - 2];
                let n = b.shape()[b.ndim() - 1];
                Ok(Array::zeros(dev, &[m, n], a.dtype()))
            }
            LazyOp::Neg(_)
            | LazyOp::Softmax(_)
            | LazyOp::RoPE(_)
            | LazyOp::Copy(_)
            | LazyOp::RmsNorm(_) => {
                let a = inputs[0];
                Ok(Array::zeros(dev, a.shape(), a.dtype()))
            }
            LazyOp::Custom(_, _) => Err(LazyEvalError::EvalFailed(
                "custom op not supported in test".into(),
            )),
        }
    }

    #[test]
    fn test_materialized_leaf() {
        let dev = test_device();
        let graph = LazyGraph::new();
        let arr = Array::zeros(dev, &[2, 3], DType::Float32);
        let lazy = LazyArray::from_array(&graph, arr);

        assert!(lazy.is_materialized());
        assert_eq!(lazy.shape(), vec![2, 3]);
        assert_eq!(lazy.dtype(), DType::Float32);

        // Eval is a no-op for already-materialized nodes
        lazy.eval(&test_eval_fn).unwrap();
        assert!(lazy.is_materialized());
    }

    #[test]
    fn test_lazy_add_eval() {
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[4], DType::Float32));
        let b = LazyArray::from_array(&graph, Array::zeros(dev, &[4], DType::Float32));
        let c = a.add(&b).unwrap();

        assert!(!c.is_materialized());
        c.eval(&test_eval_fn).unwrap();
        assert!(c.is_materialized());

        let aref = c.try_get().unwrap();
        aref.with(|arr| {
            assert_eq!(arr.shape(), &[4]);
            assert_eq!(arr.dtype(), DType::Float32);
        });
    }

    #[test]
    fn test_chained_operations() {
        // a + b -> c, c * d -> e, eval(e) should materialize everything
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[3], DType::Float32));
        let b = LazyArray::from_array(&graph, Array::zeros(dev, &[3], DType::Float32));
        let c = a.add(&b).unwrap();

        let d = LazyArray::from_array(&graph, Array::zeros(dev, &[3], DType::Float32));
        let e = c.mul(&d).unwrap();

        assert!(!c.is_materialized());
        assert!(!e.is_materialized());

        e.eval(&test_eval_fn).unwrap();

        // Both c and e should now be materialized
        assert!(c.is_materialized());
        assert!(e.is_materialized());
    }

    #[test]
    fn test_diamond_dag() {
        // a -> b = neg(a), c = neg(a), d = b + c
        // Tests that shared inputs are evaluated only once.
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[5], DType::Float32));
        let b = a.neg();
        let c = a.neg();
        let d = b.add(&c).unwrap();

        d.eval(&test_eval_fn).unwrap();
        assert!(d.is_materialized());
        assert!(b.is_materialized());
        assert!(c.is_materialized());
    }

    #[test]
    fn test_already_materialized_passthrough() {
        let dev = test_device();
        let graph = LazyGraph::new();
        let arr = Array::zeros(dev, &[2, 2], DType::Float32);
        let lazy = LazyArray::from_array(&graph, arr);

        // eval_fn should never be called for a leaf
        let never_fn = |_: &LazyOp, _: Vec<&Array>| -> Result<Array, LazyEvalError> {
            panic!("should not be called for materialized leaf");
        };
        lazy.eval(&never_fn).unwrap();
    }

    #[test]
    fn test_shape_mismatch_error() {
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[3], DType::Float32));
        let b = LazyArray::from_array(&graph, Array::zeros(dev, &[5], DType::Float32));

        let result = a.add(&b);
        assert!(result.is_err());
        match result {
            Err(LazyEvalError::ShapeMismatch(_)) => {} // expected
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_matmul_shape_validation() {
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[2, 3], DType::Float32));
        let b = LazyArray::from_array(&graph, Array::zeros(dev, &[3, 4], DType::Float32));
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), vec![2, 4]);

        // Incompatible inner dims
        let d = LazyArray::from_array(&graph, Array::zeros(dev, &[5, 4], DType::Float32));
        assert!(a.matmul(&d).is_err());

        // 1D arrays should fail
        let e = LazyArray::from_array(&graph, Array::zeros(dev, &[3], DType::Float32));
        assert!(a.matmul(&e).is_err());
    }

    #[test]
    fn test_eval_error_propagation() {
        let dev = test_device();
        let graph = LazyGraph::new();

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[4], DType::Float32));
        let b = a.neg();

        let failing_fn = |_: &LazyOp, _: Vec<&Array>| -> Result<Array, LazyEvalError> {
            Err(LazyEvalError::EvalFailed("intentional failure".into()))
        };

        let result = b.eval(&failing_fn);
        assert!(result.is_err());
        match result {
            Err(LazyEvalError::EvalFailed(msg)) => {
                assert!(msg.contains("intentional"));
            }
            other => panic!("expected EvalFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_graph_node_count() {
        let dev = test_device();
        let graph = LazyGraph::new();

        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        let a = LazyArray::from_array(&graph, Array::zeros(dev, &[4], DType::Float32));
        assert_eq!(graph.len(), 1);

        let b = LazyArray::from_array(&graph, Array::zeros(dev, &[4], DType::Float32));
        assert_eq!(graph.len(), 2);

        let _c = a.add(&b).unwrap();
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_debug_format() {
        let dev = test_device();
        let graph = LazyGraph::new();
        let lazy = LazyArray::from_array(&graph, Array::zeros(dev, &[2], DType::Float32));
        let dbg = format!("{:?}", lazy);
        assert!(dbg.contains("LazyArray"));
        assert!(dbg.contains("materialized"));
    }

    #[test]
    fn test_topo_sort_bfs_linear() {
        // Linear chain: 0 -> 1 -> 2 -> 3
        let nodes: Vec<Option<Vec<NodeId>>> = vec![
            Some(vec![NodeId(1)]),
            Some(vec![NodeId(2)]),
            Some(vec![NodeId(3)]),
            Some(vec![]),
        ];
        let order = topo_sort_bfs_width_limited(&nodes, 4, 20);
        assert_eq!(order.len(), 4);
        // Must respect dependencies
        let pos = |id: usize| order.iter().position(|n| n.0 == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn test_topo_sort_bfs_wide() {
        // Wide fan-out: 0 -> {1,2,3,4,5} -> 6
        let nodes: Vec<Option<Vec<NodeId>>> = vec![
            Some(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4), NodeId(5)]),
            Some(vec![NodeId(6)]),
            Some(vec![NodeId(6)]),
            Some(vec![NodeId(6)]),
            Some(vec![NodeId(6)]),
            Some(vec![NodeId(6)]),
            Some(vec![]),
        ];
        let order = topo_sort_bfs_width_limited(&nodes, 7, 3);
        assert_eq!(order.len(), 7);
        let pos = |id: usize| order.iter().position(|n| n.0 == id).unwrap();
        // 0 must come first, 6 must come last
        assert_eq!(pos(0), 0);
        assert_eq!(pos(6), 6);
    }

    #[test]
    fn test_topo_sort_bfs_disconnected() {
        // Two independent chains: 0->1, 2->3
        let nodes: Vec<Option<Vec<NodeId>>> = vec![
            Some(vec![NodeId(1)]),
            Some(vec![]),
            Some(vec![NodeId(3)]),
            Some(vec![]),
        ];
        let order = topo_sort_bfs_width_limited(&nodes, 4, 20);
        assert_eq!(order.len(), 4);
        let pos = |id: usize| order.iter().position(|n| n.0 == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn test_topo_sort_bfs_width_1() {
        // Extreme: max_width=1, forces pure DFS
        let nodes: Vec<Option<Vec<NodeId>>> = vec![
            Some(vec![NodeId(1), NodeId(2)]),
            Some(vec![NodeId(3)]),
            Some(vec![NodeId(3)]),
            Some(vec![]),
        ];
        let order = topo_sort_bfs_width_limited(&nodes, 4, 1);
        assert_eq!(order.len(), 4);
        let pos = |id: usize| order.iter().position(|n| n.0 == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        // 3 depends on both 1 and 2
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }
}
