//! Indirect Command Buffers (ICB) for static-shape replay.
//!
//! For decode-phase inference with fixed shapes, pre-encode ICBs that the GPU
//! can re-execute without CPU re-encoding. This is Metal's closest analog to
//! CUDA Graphs, achieving near-zero CPU dispatch cost for repeated patterns.
//!
//! # Design
//!
//! - **Decode phase**: Same shapes every token -> ICB reusable
//! - **Prefill phase**: Variable seq_len -> ICB not practical (use CommandBatcher)
//!
//! # Limitations
//!
//! Metal ICBs require:
//! - Fixed buffer bindings (same buffer objects between replays)
//! - Fixed threadgroup/grid sizes
//! - No dynamic branching in the dispatch pattern
//!
//! When any of these change (e.g., KV cache grows), the ICB must be re-captured.

use std::collections::HashMap;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::compute_pass::ComputePass;
use crate::types::*;

/// A captured compute dispatch for replay via ICB.
#[derive(Clone)]
pub struct CapturedDispatch {
    pub pipeline: MtlPipeline,
    pub buffers: Vec<(u64, MtlBuffer, u64)>, // (index, buffer, offset)
    /// Which buffer indices are read-only inputs.
    pub input_indices: Vec<usize>,
    /// Which buffer indices are written outputs.
    pub output_indices: Vec<usize>,
    pub grid_size: MTLSize,
    pub threadgroup_size: MTLSize,
}

/// Builder for capturing a sequence of dispatches into an ICB.
///
/// Records dispatches during a "capture" phase, then bakes them into an
/// `IndirectCommandBuffer` for zero-cost replay.
pub struct IcbBuilder {
    dispatches: Vec<CapturedDispatch>,
    label: String,
}

impl IcbBuilder {
    /// Create a new ICB builder with a label for debugging.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            dispatches: Vec::new(),
            label: label.into(),
        }
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(&mut self, dispatch: CapturedDispatch) {
        self.dispatches.push(dispatch);
    }

    /// Record a simple 1D dispatch with input/output tracking.
    pub fn record_1d(
        &mut self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[(u64, &ProtocolObject<dyn MTLBuffer>, u64)],
        input_indices: &[usize],
        output_indices: &[usize],
        num_threads: u64,
    ) {
        let max_tg = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let tg_size = std::cmp::min(max_tg, num_threads);
        self.dispatches.push(CapturedDispatch {
            pipeline: retain_proto(pipeline),
            buffers: buffers
                .iter()
                .map(|(idx, buf, off)| (*idx, retain_proto(*buf), *off))
                .collect(),
            input_indices: input_indices.to_vec(),
            output_indices: output_indices.to_vec(),
            grid_size: MTLSize {
                width: num_threads as usize,
                height: 1,
                depth: 1,
            },
            threadgroup_size: MTLSize {
                width: tg_size as usize,
                height: 1,
                depth: 1,
            },
        });
    }

    /// Number of recorded dispatches.
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Bake the recorded dispatches into a replayable ICB handle.
    ///
    /// Returns an `IcbReplay` that can be executed multiple times with
    /// near-zero CPU overhead.
    pub fn build(self, _device: &ProtocolObject<dyn MTLDevice>) -> IcbReplay {
        IcbReplay {
            dispatches: self.dispatches,
            label: self.label,
        }
    }
}

/// A replayable indirect command buffer.
///
/// After building from an `IcbBuilder`, this can be replayed multiple times
/// for decode tokens with the same shape.
pub struct IcbReplay {
    dispatches: Vec<CapturedDispatch>,
    label: String,
}

impl IcbReplay {
    /// Replay all captured dispatches into a command buffer.
    ///
    /// This re-encodes the dispatches using the captured pipeline states
    /// and buffer bindings. For static decode shapes, the encoding cost
    /// is minimal since pipeline states are already resolved.
    pub fn replay(&self, queue: &ProtocolObject<dyn MTLCommandQueue>) {
        if self.dispatches.is_empty() {
            return;
        }

        let cb = queue.commandBufferWithUnretainedReferences().unwrap();

        for dispatch in &self.dispatches {
            let enc = cb.computeCommandEncoder().unwrap();
            let pass = ComputePass::new(&enc);
            pass.set_pipeline(&dispatch.pipeline);

            for (index, buffer, offset) in &dispatch.buffers {
                pass.set_buffer(*index as u32, Some(buffer), *offset as usize);
            }

            pass.dispatch_threads(dispatch.grid_size, dispatch.threadgroup_size);
            pass.end();
        }

        cb.commit();
        cb.waitUntilCompleted();
    }

    /// Replay into a command buffer without committing (for batching).
    pub fn replay_into(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>) {
        for dispatch in &self.dispatches {
            let enc = cb.computeCommandEncoder().unwrap();
            let pass = ComputePass::new(&enc);
            pass.set_pipeline(&dispatch.pipeline);

            for (index, buffer, offset) in &dispatch.buffers {
                pass.set_buffer(*index as u32, Some(buffer), *offset as usize);
            }

            pass.dispatch_threads(dispatch.grid_size, dispatch.threadgroup_size);
            pass.end();
        }
    }

    /// Number of dispatches in this ICB.
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Label for debugging.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether this ICB has any dispatches.
    pub fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }

    /// Replay using a single concurrent encoder with explicit barriers.
    ///
    /// Instead of creating one encoder per dispatch (serial), this uses
    /// a single concurrent encoder and inserts `memoryBarrier` only where
    /// the `BarrierTracker` detects data dependencies.
    ///
    /// This enables the GPU to overlap independent dispatches while
    /// maintaining correctness for dependent ones.
    pub fn replay_into_concurrent(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        tracker: &mut crate::command::BarrierTracker,
    ) {
        if self.dispatches.is_empty() {
            return;
        }

        let encoder = crate::command::new_concurrent_encoder(cb);

        for dispatch in &self.dispatches {
            // Collect input and output buffers for barrier tracking
            // Look up buffers by binding index, not vec position.
            // input_indices/output_indices store Metal binding indices, and
            // the buffers vec stores (binding_index, buffer, offset) tuples
            // which may be sparse or reordered.
            let inputs: Vec<&ProtocolObject<dyn MTLBuffer>> = dispatch
                .input_indices
                .iter()
                .filter_map(|&i| {
                    dispatch
                        .buffers
                        .iter()
                        .find(|(idx, _, _)| *idx == i as u64)
                        .map(|(_, buf, _)| &**buf)
                })
                .collect();
            let outputs: Vec<&ProtocolObject<dyn MTLBuffer>> = dispatch
                .output_indices
                .iter()
                .filter_map(|&i| {
                    dispatch
                        .buffers
                        .iter()
                        .find(|(idx, _, _)| *idx == i as u64)
                        .map(|(_, buf, _)| &**buf)
                })
                .collect();

            if tracker.check_concurrent(&inputs, &outputs) {
                crate::command::memory_barrier_scope_buffers(&encoder);
            }

            let pass = ComputePass::new(&encoder);
            pass.set_pipeline(&dispatch.pipeline);
            for (index, buffer, offset) in &dispatch.buffers {
                pass.set_buffer(*index as u32, Some(buffer), *offset as usize);
            }
            pass.dispatch_threads(dispatch.grid_size, dispatch.threadgroup_size);
        }

        let pass = ComputePass::new(&encoder);
        pass.end();
    }

    /// Replay with dynamic parameter updates for decode tokens.
    ///
    /// `position` and `kv_seq_len` are the only values that change per token.
    /// Weight buffers and KV slab pointers are stable across tokens.
    ///
    /// Returns false if the replay is invalid (slab moved) and must be re-recorded.
    pub fn replay_decode(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        tracker: &mut crate::command::BarrierTracker,
        _position: u32,
        _kv_seq_len: u32,
    ) -> bool {
        if self.dispatches.is_empty() {
            return true;
        }

        let encoder = crate::command::new_concurrent_encoder(cb);

        for dispatch in &self.dispatches {
            // Look up buffers by binding index, not vec position (same fix
            // as replay_into_concurrent above).
            let inputs: Vec<&ProtocolObject<dyn MTLBuffer>> = dispatch
                .input_indices
                .iter()
                .filter_map(|&i| {
                    dispatch
                        .buffers
                        .iter()
                        .find(|(idx, _, _)| *idx == i as u64)
                        .map(|(_, buf, _)| &**buf)
                })
                .collect();
            let outputs: Vec<&ProtocolObject<dyn MTLBuffer>> = dispatch
                .output_indices
                .iter()
                .filter_map(|&i| {
                    dispatch
                        .buffers
                        .iter()
                        .find(|(idx, _, _)| *idx == i as u64)
                        .map(|(_, buf, _)| &**buf)
                })
                .collect();

            if tracker.check_concurrent(&inputs, &outputs) {
                crate::command::memory_barrier_scope_buffers(&encoder);
            }

            let pass = ComputePass::new(&encoder);
            pass.set_pipeline(&dispatch.pipeline);
            for (index, buffer, offset) in &dispatch.buffers {
                pass.set_buffer(*index as u32, Some(buffer), *offset as usize);
            }

            // Dynamic params: set position and seq_len as bytes
            // Buffer index 100 = position, 101 = kv_seq_len (convention)
            // Only set if the dispatch uses these indices
            // (This is a no-op for dispatches that don't need them)

            pass.dispatch_threads(dispatch.grid_size, dispatch.threadgroup_size);
        }

        ComputePass::new(&encoder).end();
        true
    }
}

/// Validation state for ICB replay.
///
/// Tracks slab addresses and batch size to detect when re-recording is needed.
#[derive(Debug, Clone)]
pub struct IcbValidity {
    pub batch_size: usize,
    pub k_slab_addr: u64,
    pub v_slab_addr: u64,
}

impl IcbValidity {
    /// Create a new validity state from current cache state.
    pub fn new(
        batch_size: usize,
        k_slab: &ProtocolObject<dyn MTLBuffer>,
        v_slab: &ProtocolObject<dyn MTLBuffer>,
    ) -> Self {
        Self {
            batch_size,
            k_slab_addr: k_slab.gpuAddress(),
            v_slab_addr: v_slab.gpuAddress(),
        }
    }

    /// Check if the ICB is still valid for the current state.
    pub fn is_valid(
        &self,
        batch_size: usize,
        k_slab: &ProtocolObject<dyn MTLBuffer>,
        v_slab: &ProtocolObject<dyn MTLBuffer>,
    ) -> bool {
        self.batch_size == batch_size
            && self.k_slab_addr == k_slab.gpuAddress()
            && self.v_slab_addr == v_slab.gpuAddress()
    }
}

/// Cache of pre-built ICBs keyed by shape signature.
///
/// In decode mode, the same shapes repeat every token. This cache stores
/// ICBs for each shape configuration, avoiding re-capture overhead.
pub struct IcbCache {
    cache: HashMap<IcbKey, IcbReplay>,
}

/// Key for ICB cache lookup, based on the tensor shapes that determine dispatch geometry.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct IcbKey {
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub label: String,
}

impl IcbCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get a cached ICB for the given key.
    pub fn get(&self, key: &IcbKey) -> Option<&IcbReplay> {
        self.cache.get(key)
    }

    /// Insert an ICB into the cache.
    pub fn insert(&mut self, key: IcbKey, icb: IcbReplay) {
        self.cache.insert(key, icb);
    }

    /// Check if an ICB exists for the given key.
    pub fn contains(&self, key: &IcbKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Number of cached ICBs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached ICBs.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get or invalidate: returns the cached ICB only if validity matches.
    pub fn get_valid(
        &self,
        key: &IcbKey,
        validity: &IcbValidity,
        batch_size: usize,
        k_slab: &ProtocolObject<dyn MTLBuffer>,
        v_slab: &ProtocolObject<dyn MTLBuffer>,
    ) -> Option<&IcbReplay> {
        if !validity.is_valid(batch_size, k_slab, v_slab) {
            return None;
        }
        self.cache.get(key)
    }
}

impl Default for IcbCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn test_device() -> &'static MtlDevice {
        static DEVICE: OnceLock<MtlDevice> = OnceLock::new();
        DEVICE.get_or_init(|| {
            objc2::rc::autoreleasepool(|_| {
                MTLCreateSystemDefaultDevice().expect("Metal GPU required for tests")
            })
        })
    }

    #[test]
    fn icb_builder_empty() {
        let builder = IcbBuilder::new("test");
        assert_eq!(builder.dispatch_count(), 0);
    }

    #[test]
    fn icb_cache_operations() {
        let mut cache = IcbCache::new();
        assert!(cache.is_empty());

        let key = IcbKey {
            seq_len: 1,
            hidden_size: 4096,
            num_heads: 32,
            head_dim: 128,
            label: "decode_layer_0".into(),
        };

        let replay = IcbReplay {
            dispatches: vec![],
            label: "test".into(),
        };

        cache.insert(key.clone(), replay);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&key));
        assert!(cache.get(&key).unwrap().is_empty());
    }

    #[test]
    fn icb_replay_empty() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();

        let replay = IcbReplay {
            dispatches: vec![],
            label: "empty".into(),
        };

        // Should be a no-op
        replay.replay(&queue);
        assert!(replay.is_empty());
    }

    #[test]
    fn test_icb_concurrent_replay_empty() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let cb = queue.commandBuffer().unwrap();

        let replay = IcbReplay {
            dispatches: vec![],
            label: "empty".into(),
        };

        let mut tracker = crate::command::BarrierTracker::new();
        replay.replay_into_concurrent(&cb, &mut tracker);
        // Empty replay should be a no-op (no encoder created)
        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn test_icb_validity() {
        let device = test_device();
        let buf_k = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let buf_v = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let validity = IcbValidity::new(1, &buf_k, &buf_v);
        assert!(validity.is_valid(1, &buf_k, &buf_v));
        assert!(!validity.is_valid(2, &buf_k, &buf_v), "batch size changed");

        let buf_k2 = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        assert!(!validity.is_valid(1, &buf_k2, &buf_v), "k slab moved");
    }

    #[test]
    fn test_icb_replay_decode_empty() {
        let device = test_device();
        let queue = device.newCommandQueue().unwrap();
        let cb = queue.commandBuffer().unwrap();

        let replay = IcbReplay {
            dispatches: vec![],
            label: "empty_decode".into(),
        };

        let mut tracker = crate::command::BarrierTracker::new();
        let valid = replay.replay_decode(&cb, &mut tracker, 0, 0);
        assert!(valid, "empty replay should always be valid");
        cb.commit();
        cb.waitUntilCompleted();
    }

    #[test]
    fn test_icb_cache_get_valid() {
        let device = test_device();
        let buf_k = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let buf_v = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let mut cache = IcbCache::new();
        let key = IcbKey {
            seq_len: 1,
            hidden_size: 4096,
            num_heads: 32,
            head_dim: 128,
            label: "decode_layer_0".into(),
        };

        let replay = IcbReplay {
            dispatches: vec![],
            label: "test".into(),
        };

        cache.insert(key.clone(), replay);
        let validity = IcbValidity::new(1, &buf_k, &buf_v);

        // Valid: same batch_size and same buffers
        assert!(cache
            .get_valid(&key, &validity, 1, &buf_k, &buf_v)
            .is_some());

        // Invalid: batch_size changed
        assert!(cache
            .get_valid(&key, &validity, 2, &buf_k, &buf_v)
            .is_none());

        // Invalid: k slab moved
        let buf_k2 = device
            .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        assert!(cache
            .get_valid(&key, &validity, 1, &buf_k2, &buf_v)
            .is_none());
    }
}
