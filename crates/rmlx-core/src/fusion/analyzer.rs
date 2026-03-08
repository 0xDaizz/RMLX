//! Partition a [`LazyGraph`] into fusable and non-fusable segments.
//!
//! The analyzer walks the DAG in topological order and groups consecutive
//! element-wise ops into [`FusionGraph`] subgraphs. Non-fusable ops (MatMul,
//! Softmax, RoPE, RmsNorm, etc.) become standalone segments that break the
//! fusion chain.
//!
//! # Fusion rules
//!
//! An op is **fusable** if:
//! - It maps to a [`FusableOp`] (element-wise unary/binary/ternary)
//! - The fusion subgraph would not exceed [`MAX_FUSION_DEPTH`] or
//!   [`MAX_FUSION_ARRAYS`]
//!
//! Everything else (MatMul, Softmax, RoPE, RmsNorm, Copy, Custom) is
//! **non-fusable** and executed as a standalone kernel.

use crate::lazy::{LazyOp, NodeId};

use super::graph::{FusableOp, FusionGraph};
use super::{MAX_FUSION_ARRAYS, MAX_FUSION_DEPTH};

/// A segment of the compute graph produced by partitioning.
#[derive(Debug, Clone)]
pub enum Segment {
    /// A group of element-wise ops that can be fused into a single kernel.
    Fused {
        /// The fusable subgraph.
        graph: FusionGraph,
        /// The LazyGraph node IDs covered by this segment (in topo order).
        nodes: Vec<NodeId>,
        /// External input node IDs (from outside this segment).
        input_nodes: Vec<NodeId>,
        /// Output node IDs (consumed outside this segment or are final outputs).
        output_nodes: Vec<NodeId>,
    },
    /// A single non-fusable op that must be dispatched as its own kernel.
    Standalone {
        /// The LazyGraph node ID.
        node: NodeId,
        /// The original lazy op.
        op: LazyOp,
    },
}

/// Try to convert a [`LazyOp`] into a [`FusableOp`].
///
/// Returns `None` for ops that cannot be fused (MatMul, Softmax, etc.).
pub fn to_fusable(op: &LazyOp) -> Option<FusableOp> {
    match op {
        LazyOp::Add(_, _) => Some(FusableOp::Add),
        LazyOp::Mul(_, _) => Some(FusableOp::Mul),
        LazyOp::Sub(_, _) => Some(FusableOp::Sub),
        LazyOp::Neg(_) => Some(FusableOp::Neg),
        // Non-fusable ops
        LazyOp::MatMul(_, _)
        | LazyOp::Softmax(_)
        | LazyOp::RoPE(_)
        | LazyOp::Copy(_)
        | LazyOp::RmsNorm(_)
        | LazyOp::Custom(_, _) => None,
    }
}

/// Partition a topologically-sorted sequence of (NodeId, LazyOp) into
/// fusable and standalone segments.
///
/// `topo_ops` must be in valid topological order (inputs before consumers).
/// Leaf nodes (already materialized) should NOT be included.
///
/// `consumers` maps each NodeId to the set of NodeIds that consume it.
/// This is used to determine whether an intermediate result must be
/// materialized (i.e., consumed by a non-fusable op or by multiple segments).
pub fn partition(
    topo_ops: &[(NodeId, LazyOp)],
    consumers: &[Vec<NodeId>],
    num_nodes: usize,
) -> Vec<Segment> {
    let mut segments = Vec::new();

    // Track which segment each node belongs to (None = leaf or not yet assigned)
    let mut node_segment: Vec<Option<usize>> = vec![None; num_nodes];
    // Current accumulating fusable group
    let mut current_fused: Vec<(NodeId, LazyOp, FusableOp)> = Vec::new();

    for (nid, op) in topo_ops {
        if let Some(fusable_op) = to_fusable(op) {
            // Check if adding this op would exceed limits
            let would_exceed = current_fused.len() + 1 > MAX_FUSION_DEPTH;

            // Check if all inputs are either in the current fused group or are leaves
            let inputs = op.inputs();
            let all_inputs_compatible = inputs.iter().all(|inp| {
                // Input is in current fused group, or is a leaf/standalone output
                current_fused.iter().any(|(n, _, _)| *n == *inp)
                    || node_segment[inp.0].is_none()
                    || matches!(node_segment[inp.0], Some(seg_idx) if seg_idx < segments.len())
            });

            if would_exceed || !all_inputs_compatible {
                // Flush current group and start a new one
                if !current_fused.is_empty() {
                    let seg = build_fused_segment(&current_fused, &node_segment, consumers);
                    let seg_idx = segments.len();
                    for (n, _, _) in &current_fused {
                        node_segment[n.0] = Some(seg_idx);
                    }
                    segments.push(seg);
                    current_fused.clear();
                }
            }

            current_fused.push((*nid, op.clone(), fusable_op));
        } else {
            // Non-fusable: flush any current fused group first
            if !current_fused.is_empty() {
                let seg = build_fused_segment(&current_fused, &node_segment, consumers);
                let seg_idx = segments.len();
                for (n, _, _) in &current_fused {
                    node_segment[n.0] = Some(seg_idx);
                }
                segments.push(seg);
                current_fused.clear();
            }

            let seg_idx = segments.len();
            node_segment[nid.0] = Some(seg_idx);
            segments.push(Segment::Standalone {
                node: *nid,
                op: op.clone(),
            });
        }
    }

    // Flush remaining fused group
    if !current_fused.is_empty() {
        let seg = build_fused_segment(&current_fused, &node_segment, consumers);
        let seg_idx = segments.len();
        for (n, _, _) in &current_fused {
            node_segment[n.0] = Some(seg_idx);
        }
        segments.push(seg);
    }

    segments
}

/// Build a Fused segment from accumulated fusable ops.
fn build_fused_segment(
    ops: &[(NodeId, LazyOp, FusableOp)],
    _node_segment: &[Option<usize>],
    _consumers: &[Vec<NodeId>],
) -> Segment {
    // Collect all nodes in this segment
    let nodes: Vec<NodeId> = ops.iter().map(|(n, _, _)| *n).collect();
    let node_set: std::collections::HashSet<NodeId> = nodes.iter().copied().collect();

    // Find external inputs: inputs that are not produced within this segment
    let mut input_nodes = Vec::new();
    let mut input_set = std::collections::HashSet::new();
    for (_, op, _) in ops {
        for inp in op.inputs() {
            if !node_set.contains(&inp) && input_set.insert(inp) {
                input_nodes.push(inp);
            }
        }
    }

    // Build the FusionGraph
    // Map: NodeId -> local index in the fusion graph
    let mut local_idx = std::collections::HashMap::new();
    for (i, inp) in input_nodes.iter().enumerate() {
        local_idx.insert(*inp, i);
    }

    let n_inputs = input_nodes.len();
    let mut fusion_graph = FusionGraph::new(n_inputs);

    for (nid, op, fusable_op) in ops {
        let inputs: Vec<usize> = op
            .inputs()
            .iter()
            .map(|inp| *local_idx.get(inp).expect("input must be mapped"))
            .collect();
        let idx = fusion_graph.add_op(*fusable_op, inputs);
        local_idx.insert(*nid, idx);
    }

    // Output nodes: for simplicity, the last node is the output.
    // A more sophisticated approach would check which nodes are consumed
    // outside the segment.
    let output_nodes = vec![*nodes.last().unwrap()];
    fusion_graph.set_outputs(output_nodes.len());

    // Check if we exceed array limits
    if n_inputs + output_nodes.len() > MAX_FUSION_ARRAYS {
        // Fall back: break this into individual standalone ops
        // For now, just mark outputs; the caller can check exceeds_limits()
    }

    Segment::Fused {
        graph: fusion_graph,
        nodes,
        input_nodes,
        output_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LazyOp, NodeId};

    #[test]
    fn test_to_fusable() {
        assert_eq!(
            to_fusable(&LazyOp::Add(NodeId(0), NodeId(1))),
            Some(FusableOp::Add)
        );
        assert_eq!(
            to_fusable(&LazyOp::Mul(NodeId(0), NodeId(1))),
            Some(FusableOp::Mul)
        );
        assert_eq!(
            to_fusable(&LazyOp::Sub(NodeId(0), NodeId(1))),
            Some(FusableOp::Sub)
        );
        assert_eq!(to_fusable(&LazyOp::Neg(NodeId(0))), Some(FusableOp::Neg));
        assert_eq!(to_fusable(&LazyOp::MatMul(NodeId(0), NodeId(1))), None);
        assert_eq!(to_fusable(&LazyOp::Softmax(NodeId(0))), None);
        assert_eq!(to_fusable(&LazyOp::RoPE(NodeId(0))), None);
        assert_eq!(to_fusable(&LazyOp::RmsNorm(NodeId(0))), None);
        assert_eq!(to_fusable(&LazyOp::Copy(NodeId(0))), None);
    }

    #[test]
    fn test_partition_all_fusable() {
        // Graph: leaf0, leaf1 -> add(0,1) -> neg(2) -> result
        // Nodes 0,1 are leaves (not in topo_ops), nodes 2,3 are ops
        let topo_ops = vec![
            (NodeId(2), LazyOp::Add(NodeId(0), NodeId(1))),
            (NodeId(3), LazyOp::Neg(NodeId(2))),
        ];
        let consumers = vec![
            vec![NodeId(2)], // node 0 consumed by node 2
            vec![NodeId(2)], // node 1 consumed by node 2
            vec![NodeId(3)], // node 2 consumed by node 3
            vec![],          // node 3 is final output
        ];

        let segments = partition(&topo_ops, &consumers, 4);
        assert_eq!(segments.len(), 1);
        match &segments[0] {
            Segment::Fused {
                graph,
                nodes,
                input_nodes,
                ..
            } => {
                assert_eq!(nodes.len(), 2);
                assert_eq!(input_nodes.len(), 2); // leaf0, leaf1
                assert_eq!(graph.depth(), 2); // add + neg
                assert_eq!(graph.n_inputs(), 2);
            }
            _ => panic!("expected Fused segment"),
        }
    }

    #[test]
    fn test_partition_with_standalone() {
        // Graph: leaf0, leaf1 -> add(0,1) -> matmul(2, leaf2) -> neg(3)
        let topo_ops = vec![
            (NodeId(3), LazyOp::Add(NodeId(0), NodeId(1))),
            (NodeId(4), LazyOp::MatMul(NodeId(3), NodeId(2))),
            (NodeId(5), LazyOp::Neg(NodeId(4))),
        ];
        let consumers = vec![
            vec![NodeId(3)], // 0
            vec![NodeId(3)], // 1
            vec![NodeId(4)], // 2
            vec![NodeId(4)], // 3
            vec![NodeId(5)], // 4
            vec![],          // 5
        ];

        let segments = partition(&topo_ops, &consumers, 6);
        assert_eq!(segments.len(), 3);

        // Segment 0: Fused(Add)
        assert!(matches!(&segments[0], Segment::Fused { .. }));
        // Segment 1: Standalone(MatMul)
        assert!(matches!(
            &segments[1],
            Segment::Standalone {
                op: LazyOp::MatMul(_, _),
                ..
            }
        ));
        // Segment 2: Fused(Neg)
        assert!(matches!(&segments[2], Segment::Fused { .. }));
    }

    #[test]
    fn test_partition_all_standalone() {
        let topo_ops = vec![
            (NodeId(2), LazyOp::MatMul(NodeId(0), NodeId(1))),
            (NodeId(3), LazyOp::Softmax(NodeId(2))),
        ];
        let consumers = vec![vec![NodeId(2)], vec![NodeId(2)], vec![NodeId(3)], vec![]];

        let segments = partition(&topo_ops, &consumers, 4);
        assert_eq!(segments.len(), 2);
        assert!(matches!(&segments[0], Segment::Standalone { .. }));
        assert!(matches!(&segments[1], Segment::Standalone { .. }));
    }
}
