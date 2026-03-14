//! Static execution plan for transformer prefill.
//!
//! A `PrefillPlan` captures all dispatch parameters for one transformer layer
//! at a given seq_len, pre-resolving PSOs and pre-computing grid/threadgroup sizes.
//! At runtime, `execute()` replays the plan in a tight loop with minimal CPU overhead.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use rmlx_metal::memory_barrier_scope_buffers;
use rmlx_metal::{ComputePass, MTLSize, MtlBuffer, MtlPipeline};
use std::collections::HashMap;

use crate::prefill_pool::{PrefillBufferPool, Slot};

/// A single step in the execution plan.
pub enum PlanStep {
    /// Set the compute pipeline state (pre-resolved).
    SetPipeline(usize),
    /// Bind a buffer from the scratch pool.
    BindPoolBuffer {
        /// Metal argument index
        index: u64,
        /// Which pool slot
        slot: Slot,
        /// Byte offset into the slot
        offset: u64,
    },
    /// Bind a weight buffer (fixed for the lifetime of the model).
    BindWeight {
        /// Metal argument index
        index: u64,
        /// Weight buffer reference index in the plan's weight list
        weight_idx: usize,
    },
    /// Bind the input buffer.
    BindInput {
        /// Metal argument index
        index: u64,
        /// Byte offset
        offset: u64,
    },
    /// Bind scalar constant bytes.
    BindBytes {
        /// Metal argument index
        index: u64,
        /// Scalar value (up to 16 bytes)
        value: [u8; 16],
        /// Number of valid bytes
        len: u64,
    },
    /// Dispatch threadgroups.
    Dispatch {
        grid: [u64; 3],
        threadgroup: [u64; 3],
    },
    /// Dispatch threads (non-uniform).
    DispatchThreads {
        threads: [u64; 3],
        threadgroup: [u64; 3],
    },
    /// Memory barrier (scope: all buffers).
    Barrier,
}

/// Pre-compiled execution plan for one transformer layer at a specific seq_len.
pub struct PrefillPlan {
    steps: Vec<PlanStep>,
    /// Pre-resolved pipeline states, indexed by SetPipeline.
    pipelines: Vec<MtlPipeline>,
    /// Weight buffer references, indexed by BindWeight.
    weights: Vec<MtlBuffer>,
    /// The seq_len this plan was compiled for.
    seq_len: usize,
}

impl PrefillPlan {
    /// Create an empty plan builder.
    pub fn new(seq_len: usize) -> Self {
        Self {
            steps: Vec::with_capacity(64),
            pipelines: Vec::with_capacity(16),
            weights: Vec::with_capacity(32),
            seq_len,
        }
    }

    /// Add a pre-resolved pipeline state, returns its index.
    pub fn add_pipeline(&mut self, pso: MtlPipeline) -> usize {
        let idx = self.pipelines.len();
        self.pipelines.push(pso);
        idx
    }

    /// Add a weight buffer reference, returns its index.
    pub fn add_weight(&mut self, buffer: MtlBuffer) -> usize {
        let idx = self.weights.len();
        self.weights.push(buffer);
        idx
    }

    /// Add a plan step.
    #[inline]
    pub fn push(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    /// The seq_len this plan was compiled for.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of steps in the plan.
    #[inline]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of pre-resolved pipelines.
    #[inline]
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Number of weight buffer references.
    #[inline]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Execute the plan on the given encoder.
    ///
    /// This is the hot loop — all PSO lookups, shape calculations, and buffer
    /// size decisions were done at compile time. This just replays set/dispatch
    /// calls with minimal branching.
    pub fn execute(
        &self,
        encoder: ComputePass<'_>,
        pool: &PrefillBufferPool,
        input: &ProtocolObject<dyn MTLBuffer>,
    ) {
        for step in &self.steps {
            match step {
                PlanStep::SetPipeline(idx) => {
                    encoder.set_pipeline(&self.pipelines[*idx]);
                }
                PlanStep::BindPoolBuffer {
                    index,
                    slot,
                    offset,
                } => {
                    encoder.set_buffer(*index as u32, Some(pool.buffer(*slot)), *offset as usize);
                }
                PlanStep::BindWeight { index, weight_idx } => {
                    encoder.set_buffer(*index as u32, Some(&self.weights[*weight_idx]), 0);
                }
                PlanStep::BindInput { index, offset } => {
                    encoder.set_buffer(*index as u32, Some(input), *offset as usize);
                }
                PlanStep::BindBytes { index, value, len } => {
                    encoder.set_bytes(
                        *index as u32,
                        value.as_ptr() as *const std::ffi::c_void,
                        *len as usize,
                    );
                }
                PlanStep::Dispatch { grid, threadgroup } => {
                    encoder.dispatch_threadgroups(
                        MTLSize {
                            width: grid[0] as usize,
                            height: grid[1] as usize,
                            depth: grid[2] as usize,
                        },
                        MTLSize {
                            width: threadgroup[0] as usize,
                            height: threadgroup[1] as usize,
                            depth: threadgroup[2] as usize,
                        },
                    );
                }
                PlanStep::DispatchThreads {
                    threads,
                    threadgroup,
                } => {
                    encoder.dispatch_threads(
                        MTLSize {
                            width: threads[0] as usize,
                            height: threads[1] as usize,
                            depth: threads[2] as usize,
                        },
                        MTLSize {
                            width: threadgroup[0] as usize,
                            height: threadgroup[1] as usize,
                            depth: threadgroup[2] as usize,
                        },
                    );
                }
                PlanStep::Barrier => {
                    memory_barrier_scope_buffers(encoder.raw());
                }
            }
        }
    }
}

/// Convenience helpers for building `BindBytes` steps from typed values.
impl PlanStep {
    /// Create a `BindBytes` step from a `u32` value.
    pub fn bind_u32(index: u64, value: u32) -> Self {
        let mut buf = [0u8; 16];
        buf[..4].copy_from_slice(&value.to_ne_bytes());
        PlanStep::BindBytes {
            index,
            value: buf,
            len: 4,
        }
    }

    /// Create a `BindBytes` step from a `f32` value.
    pub fn bind_f32(index: u64, value: f32) -> Self {
        let mut buf = [0u8; 16];
        buf[..4].copy_from_slice(&value.to_ne_bytes());
        PlanStep::BindBytes {
            index,
            value: buf,
            len: 4,
        }
    }

    /// Create a `BindBytes` step from two packed `u32` values (8 bytes).
    pub fn bind_u32x2(index: u64, a: u32, b: u32) -> Self {
        let mut buf = [0u8; 16];
        buf[..4].copy_from_slice(&a.to_ne_bytes());
        buf[4..8].copy_from_slice(&b.to_ne_bytes());
        PlanStep::BindBytes {
            index,
            value: buf,
            len: 8,
        }
    }

    /// Create a `BindBytes` step from a raw byte slice (up to 16 bytes).
    ///
    /// Panics if `bytes.len() > 16`.
    pub fn bind_raw(index: u64, bytes: &[u8]) -> Self {
        assert!(bytes.len() <= 16, "BindBytes supports up to 16 bytes");
        let mut buf = [0u8; 16];
        buf[..bytes.len()].copy_from_slice(bytes);
        PlanStep::BindBytes {
            index,
            value: buf,
            len: bytes.len() as u64,
        }
    }
}

/// Per-layer plan cache: seq_len -> compiled plan.
pub struct PlanCache {
    plans: HashMap<usize, PrefillPlan>,
}

impl PlanCache {
    pub fn new() -> Self {
        Self {
            plans: HashMap::new(),
        }
    }

    /// Get a cached plan for the given seq_len, or None.
    pub fn get(&self, seq_len: usize) -> Option<&PrefillPlan> {
        self.plans.get(&seq_len)
    }

    /// Insert a compiled plan.
    pub fn insert(&mut self, plan: PrefillPlan) {
        self.plans.insert(plan.seq_len, plan);
    }

    /// Number of cached plans.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// Remove all cached plans (e.g., after model weight reload).
    pub fn clear(&mut self) {
        self.plans.clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}
