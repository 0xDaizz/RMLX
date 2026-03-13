//! Pre-allocated scratch buffer pool for transformer prefill.
//!
//! Allocates 4 Metal buffers (slots A–D) sized for the worst-case intermediate
//! tensors in a single transformer layer. Every layer reuses the same buffers,
//! eliminating per-dispatch allocation overhead during prefill.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use rmlx_core::array::Array;
use rmlx_core::dtype::DType;
use rmlx_metal::{MtlBuffer};

/// Slot indices for scratch buffers in the prefill pipeline.
///
/// Based on liveness analysis of a transformer layer's intermediate tensors:
///
/// ```text
/// D1  RMSNorm:     input → [A] normed
/// D2  QKV GEMM:    [A] → [B] qkv
/// D3  RoPE Q:      [B]view → [C] q_roped
/// D4  RoPE K:      [B]view → [C]view k_roped
/// D5  Deinterl V:  [B]view → [D] v_batched    -- B dead after D5
/// D6  SDPA:        [C],[D] → [A] attn_out      -- C,D dead after D6
/// D7  O Proj:      [A] → [B] o_out
/// D8  Res+Norm:    input,[B] → [A] h, [C] normed2
/// D9  Gate+Up:     [C] → [B] gate_up
/// D10 SiLU*Mul:    [B]view → [D] hidden        -- B dead
/// D11 Down Proj:   [D] → [A] ffn_out
/// D12 Residual:    [A] + h_saved → output
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Slot {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

pub const NUM_SLOTS: usize = 4;

/// A pool of 4 pre-allocated Metal buffers that transformer layers reuse
/// during prefill to avoid per-dispatch allocation.
pub struct PrefillBufferPool {
    slots: [MtlBuffer; NUM_SLOTS],
    slot_sizes: [usize; NUM_SLOTS],
    dtype: DType,
}

impl PrefillBufferPool {
    /// Allocate pool for given model dimensions and max sequence length.
    ///
    /// Slot sizes (bytes) for seq_len `s`, hidden `h`, qkv_dim `q`, intermediate `i`:
    /// - A: `s * h * elem`  (normed, attn_out, ffn_out)
    /// - B: `max(s * q, s * i * 2) * elem`  (qkv merged, gate+up merged)
    /// - C: `max(s * num_heads * head_dim, s * h) * elem`  (q_roped, normed2)
    /// - D: `max(s * num_kv_heads * head_dim, s * i) * elem`  (v_batched, hidden)
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        device: &ProtocolObject<dyn MTLDevice>,
        max_seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        dtype: DType,
    ) -> Self {
        let elem = dtype.size_of();
        let s = max_seq_len;

        // qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        let qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim;

        let size_a = s * hidden_size * elem;
        let size_b = std::cmp::max(s * qkv_dim, s * intermediate_size * 2) * elem;
        let size_c = std::cmp::max(s * num_heads * head_dim, s * hidden_size) * elem;
        let size_d = std::cmp::max(s * num_kv_heads * head_dim, s * intermediate_size) * elem;

        let slot_sizes = [size_a, size_b, size_c, size_d];

        let options = MTLResourceOptions::StorageModeShared;
        let alloc = |sz: usize| {
            device
                .newBufferWithLength_options(sz.max(1), options)
                .expect("failed to allocate prefill pool buffer")
        };
        let slots = [alloc(size_a), alloc(size_b), alloc(size_c), alloc(size_d)];

        Self {
            slots,
            slot_sizes,
            dtype,
        }
    }

    /// Get a buffer reference for the given slot.
    #[inline]
    pub fn buffer(&self, slot: Slot) -> &ProtocolObject<dyn MTLBuffer> {
        &self.slots[slot as usize]
    }

    /// Get the allocated size in bytes for a slot.
    #[inline]
    pub fn slot_size(&self, slot: Slot) -> usize {
        self.slot_sizes[slot as usize]
    }

    /// The dtype this pool was allocated for.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Create an [`Array`] view into a slot with given shape.
    ///
    /// The array shares the pool's underlying Metal buffer (zero-copy).
    /// Callers must ensure the requested shape fits within the slot's
    /// allocated size.
    pub fn array_view(&self, slot: Slot, shape: &[usize]) -> Array {
        self.array_view_offset(slot, shape, self.dtype, 0)
    }

    /// Create an [`Array`] view into a slot with given shape, dtype, and byte offset.
    pub fn array_view_offset(
        &self,
        slot: Slot,
        shape: &[usize],
        dtype: DType,
        byte_offset: usize,
    ) -> Array {
        let buffer = self.slots[slot as usize].clone();

        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        Array::new(buffer, shape.to_vec(), strides, dtype, byte_offset)
    }

    /// Total allocated GPU memory in bytes across all slots.
    pub fn total_bytes(&self) -> usize {
        self.slot_sizes.iter().sum()
    }
}

impl std::fmt::Debug for PrefillBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillBufferPool")
            .field("slot_sizes", &self.slot_sizes)
            .field("dtype", &self.dtype)
            .field("total_mb", &(self.total_bytes() as f64 / (1024.0 * 1024.0)))
            .finish()
    }
}
