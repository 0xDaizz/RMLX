//! Indexing operations: gather and scatter with atomic operations.
//!
//! All bounds checking and negative-index wrapping is performed on the GPU,
//! avoiding costly GPU->CPU synchronisation for host-side validation.
//!
//! Scatter operations use atomic compare-exchange loops for floating-point
//! types (Metal has no native `atomic_float`), ensuring correctness under
//! concurrent writes from parallel threads.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

// ---------------------------------------------------------------------------
// ScatterOp enum
// ---------------------------------------------------------------------------

/// Reduction operation applied when multiple source elements scatter to the
/// same output index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterOp {
    /// Last-write-wins (non-deterministic under races, but no data tearing
    /// thanks to 32-bit atomic store).
    Overwrite,
    /// Atomic addition (CAS loop for float types).
    Add,
    /// Atomic max (CAS loop for float types).
    Max,
    /// Atomic min (CAS loop for float types).
    Min,
}

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

pub const INDEXING_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ======================================================================
// Atomic float helpers (CAS loop — Metal lacks native float atomics)
// ======================================================================

inline void atomic_add_f32(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

inline void atomic_max_f32(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float cur = as_type<float>(expected);
        if (cur >= val) return;          // already >= val, nothing to do
        desired = as_type<uint>(val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

inline void atomic_min_f32(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float cur = as_type<float>(expected);
        if (cur <= val) return;          // already <= val, nothing to do
        desired = as_type<uint>(val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

// Atomic overwrite via atomic_exchange (no tearing, but non-deterministic
// when multiple threads target the same index).
inline void atomic_store_f32(device atomic_uint* addr, float val) {
    atomic_exchange_explicit(addr, as_type<uint>(val), memory_order_relaxed);
}

// ======================================================================
// Gather — f32 with uint indices
// ======================================================================

kernel void gather_f32(
    device const float*  src       [[buffer(0)]],
    device const uint*   indices   [[buffer(1)]],
    device float*        output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = 0.0;
    }
}

// ======================================================================
// Gather — f32 with signed (int) indices  (negative wraps around)
// ======================================================================

kernel void gather_signed_f32(
    device const float*  src       [[buffer(0)]],
    device const int*    indices   [[buffer(1)]],
    device float*        output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    int idx = indices[id];
    if (idx < 0) idx += int(src_size);
    if (idx >= 0 && uint(idx) < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = 0.0;
    }
}

// ======================================================================
// Gather — f16 with uint indices
// ======================================================================

kernel void gather_f16(
    device const half*   src       [[buffer(0)]],
    device const uint*   indices   [[buffer(1)]],
    device half*         output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = half(0.0);
    }
}

// ======================================================================
// Gather — f16 with signed (int) indices
// ======================================================================

kernel void gather_signed_f16(
    device const half*   src       [[buffer(0)]],
    device const int*    indices   [[buffer(1)]],
    device half*         output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    int idx = indices[id];
    if (idx < 0) idx += int(src_size);
    if (idx >= 0 && uint(idx) < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = half(0.0);
    }
}

// ======================================================================
// Gather — bf16 with uint indices
// ======================================================================

kernel void gather_bf16(
    device const bfloat* src       [[buffer(0)]],
    device const uint*   indices   [[buffer(1)]],
    device bfloat*       output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = bfloat(0.0);
    }
}

// ======================================================================
// Gather — bf16 with signed (int) indices
// ======================================================================

kernel void gather_signed_bf16(
    device const bfloat* src       [[buffer(0)]],
    device const int*    indices   [[buffer(1)]],
    device bfloat*       output    [[buffer(2)]],
    constant uint&       src_size  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    int idx = indices[id];
    if (idx < 0) idx += int(src_size);
    if (idx >= 0 && uint(idx) < src_size) {
        output[id] = src[idx];
    } else {
        output[id] = bfloat(0.0);
    }
}

// ======================================================================
// Multi-dim gather: output[i] = src[indices[i] * inner_size + (i % inner_size)]
// Supports gathering along an arbitrary axis of a contiguous tensor.
// ======================================================================

kernel void gather_axis_f32(
    device const float*  src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device float*        output      [[buffer(2)]],
    constant uint&       src_dim     [[buffer(3)]],  // size of src along gather axis
    constant uint&       inner_size  [[buffer(4)]],  // product of dims after gather axis
    uint id [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint outer_idx = id / inner_size;  // which index element
    uint idx = indices[outer_idx];
    if (idx < src_dim) {
        // outer block offset for the source is computed by the caller
        // embedding the outer loop into the flat id; here we just index
        // within the gather-axis * inner_size block.
        output[id] = src[idx * inner_size + inner];
    } else {
        output[id] = 0.0;
    }
}

// ======================================================================
// Scatter — f32 (overwrite, atomic store to avoid tearing)
// ======================================================================

kernel void scatter_overwrite_f32(
    device const float*  src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_store_f32(&output[idx], src[id]);
    }
}

// ======================================================================
// Scatter add — f32
// ======================================================================

kernel void scatter_add_f32(
    device const float*  src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_add_f32(&output[idx], src[id]);
    }
}

// ======================================================================
// Scatter max — f32
// ======================================================================

kernel void scatter_max_f32(
    device const float*  src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_max_f32(&output[idx], src[id]);
    }
}

// ======================================================================
// Scatter min — f32
// ======================================================================

kernel void scatter_min_f32(
    device const float*  src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_min_f32(&output[idx], src[id]);
    }
}

// ======================================================================
// Scatter — f16 variants (promote to f32 for CAS, demote back)
//
// f16/bf16 are 16-bit, so we pack two halves into one uint for atomics.
// However, that complicates indexing (alignment). A simpler and still
// correct approach: promote to f32, perform atomic CAS on a uint output
// buffer, then the Rust side copies back to f16. For the overwrite case,
// we simply do a non-atomic write (16-bit writes are atomic on Apple
// Silicon when naturally aligned).
// ======================================================================

kernel void scatter_overwrite_f16(
    device const half*   src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device half*         output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        output[idx] = src[id];
    }
}

kernel void scatter_overwrite_bf16(
    device const bfloat* src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device bfloat*       output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        output[idx] = src[id];
    }
}

// f16 scatter_add: promote to f32, CAS on f32 output buffer.
// The Rust side allocates a f32 output, then converts back if needed.
kernel void scatter_add_f16(
    device const half*   src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_add_f32(&output[idx], float(src[id]));
    }
}

kernel void scatter_add_bf16(
    device const bfloat* src         [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device atomic_uint*  output      [[buffer(2)]],
    constant uint&       output_size [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = indices[id];
    if (idx < output_size) {
        atomic_add_f32(&output[idx], float(src[id]));
    }
}
"#;

// ---------------------------------------------------------------------------
// Rust helper: create a small constant buffer
// ---------------------------------------------------------------------------

fn make_u32_buf(device: &metal::Device, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("indexing", INDEXING_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Gather (backward-compatible API)
// ---------------------------------------------------------------------------

/// Gather elements from `src` at the positions given by `indices`.
///
/// `output[i] = src[indices[i]]`
///
/// * `src`     - 1-D source array (Float32, Float16, or Bfloat16).
/// * `indices` - 1-D index array (UInt32). Out-of-bounds indices produce 0.
///
/// All bounds checking happens on the GPU -- no host-side readback of indices.
pub fn gather(
    registry: &KernelRegistry,
    src: &Array,
    indices: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let kernel_name = match src.dtype() {
        DType::Float32  => "gather_f32",
        DType::Float16  => "gather_f16",
        DType::Bfloat16 => "gather_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gather not supported for {:?}",
                src.dtype()
            )))
        }
    };

    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }

    let src_size = super::checked_u32(src.numel(), "src.numel")?;
    let numel = indices.numel();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), &[0], src.dtype()));
    }

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let out = Array::zeros(registry.device().raw(), indices.shape(), src.dtype());
    let src_size_buf = make_u32_buf(registry.device().raw(), src_size);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    enc.set_buffer(1, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&src_size_buf), 0);

    let grid = metal::MTLSize::new(numel as u64, 1, 1);
    let tg = metal::MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Gather with signed (int32) indices — negative values wrap around
// ---------------------------------------------------------------------------

/// Gather with signed 32-bit indices. Negative values are interpreted as
/// `index + src_size` (Python-style wrapping). Out-of-bounds indices
/// (after wrapping) produce 0.
///
/// `indices` must be a UInt32 array whose bits are reinterpreted as int32
/// in the Metal kernel. (The DType system does not yet have Int32, so
/// callers transmute i32 slices to u32 slices when building the array.)
pub fn gather_signed(
    registry: &KernelRegistry,
    src: &Array,
    indices: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let kernel_name = match src.dtype() {
        DType::Float32  => "gather_signed_f32",
        DType::Float16  => "gather_signed_f16",
        DType::Bfloat16 => "gather_signed_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gather_signed not supported for {:?}",
                src.dtype()
            )))
        }
    };

    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_signed: indices must be UInt32 (bit-cast from i32), got {:?}",
            indices.dtype()
        )));
    }

    let src_size = super::checked_u32(src.numel(), "src.numel")?;
    let numel = indices.numel();
    if numel == 0 {
        return Ok(Array::zeros(registry.device().raw(), &[0], src.dtype()));
    }

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let out = Array::zeros(registry.device().raw(), indices.shape(), src.dtype());
    let src_size_buf = make_u32_buf(registry.device().raw(), src_size);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    enc.set_buffer(1, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&src_size_buf), 0);

    let grid = metal::MTLSize::new(numel as u64, 1, 1);
    let tg = metal::MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Multi-dimensional gather (along an axis)
// ---------------------------------------------------------------------------

/// Gather along a specific axis of a contiguous source tensor.
///
/// For a source of shape `[..., src_dim, ...]` and `indices` of shape
/// `[num_indices]`, the output has the source shape with `src_dim`
/// replaced by `num_indices`.
///
/// Internally flattens into `outer * num_indices * inner` threads where
/// `inner` is the product of dimensions after the gather axis.
///
/// Currently only supports Float32 with UInt32 indices.
pub fn gather_axis(
    registry: &KernelRegistry,
    src: &Array,
    axis: usize,
    indices: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if src.dtype() != DType::Float32 {
        return Err(KernelError::NotFound(format!(
            "gather_axis only supports Float32, got {:?}",
            src.dtype()
        )));
    }
    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "gather_axis: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }
    if axis >= src.ndim() {
        return Err(KernelError::InvalidShape(format!(
            "gather_axis: axis {} out of range for {}D tensor",
            axis,
            src.ndim()
        )));
    }

    let src_shape = src.shape();
    let src_dim = super::checked_u32(src_shape[axis], "src_dim")?;
    let inner_size: usize = src_shape[axis + 1..].iter().product();
    let outer_size: usize = src_shape[..axis].iter().product::<usize>().max(1);
    let num_indices = indices.numel();
    let inner_u32 = super::checked_u32(inner_size, "inner_size")?;

    // Build output shape: replace axis dimension with num_indices.
    let mut out_shape = src_shape.to_vec();
    out_shape[axis] = num_indices;
    let total = outer_size * num_indices * inner_size;
    if total == 0 {
        return Ok(Array::zeros(registry.device().raw(), &out_shape, src.dtype()));
    }

    let pipeline = registry.get_pipeline("gather_axis_f32", src.dtype())?;
    let out = Array::zeros(registry.device().raw(), &out_shape, src.dtype());
    let src_dim_buf = make_u32_buf(registry.device().raw(), src_dim);
    let inner_buf = make_u32_buf(registry.device().raw(), inner_u32);

    // For each outer block, dispatch num_indices * inner_size threads.
    // We encode one dispatch per outer slice for simplicity.
    let elem_bytes = src.dtype().size_of();
    let src_axis_stride = src_shape[axis..].iter().product::<usize>(); // src_dim * inner
    let out_axis_stride = num_indices * inner_size;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(3, Some(&src_dim_buf), 0);
    enc.set_buffer(4, Some(&inner_buf), 0);

    let threads_per_outer = (num_indices * inner_size) as u64;
    let tg = metal::MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), threads_per_outer),
        1,
        1,
    );

    for o in 0..outer_size {
        let src_offset = (src.offset() + o * src_axis_stride * elem_bytes) as u64;
        let out_offset = (o * out_axis_stride * elem_bytes) as u64;
        enc.set_buffer(0, Some(src.metal_buffer()), src_offset);
        enc.set_buffer(1, Some(indices.metal_buffer()), indices.offset() as u64);
        enc.set_buffer(2, Some(out.metal_buffer()), out_offset);

        let grid = metal::MTLSize::new(threads_per_outer, 1, 1);
        enc.dispatch_threads(grid, tg);
    }

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Scatter (generic)
// ---------------------------------------------------------------------------

/// Scatter `src` values into an output buffer of `output_size` elements at
/// positions given by `indices`, combining collisions with `op`.
///
/// `output[indices[i]]  <op>=  src[i]`
///
/// * For `Overwrite`: non-deterministic last-write-wins (but no tearing).
/// * For `Add` / `Max` / `Min`: atomic CAS loop on f32; correct under races.
///
/// The output is zero-initialised before the scatter.
///
/// For `Add`/`Max`/`Min` with f16/bf16 source, the kernel accumulates into
/// an f32 output buffer (promotion). The returned array is Float32.
///
/// For `Overwrite` with f16/bf16, the output keeps the original dtype.
pub fn scatter(
    registry: &KernelRegistry,
    src: &Array,
    indices: &Array,
    output_size: usize,
    op: ScatterOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if indices.dtype() != DType::UInt32 {
        return Err(KernelError::InvalidShape(format!(
            "scatter: indices must be UInt32, got {:?}",
            indices.dtype()
        )));
    }
    if src.numel() != indices.numel() {
        return Err(KernelError::InvalidShape(format!(
            "scatter: src.numel ({}) != indices.numel ({})",
            src.numel(),
            indices.numel()
        )));
    }

    let output_size_u32 = super::checked_u32(output_size, "output_size")?;
    let numel = src.numel();
    if numel == 0 {
        return Ok(Array::zeros(
            registry.device().raw(),
            &[output_size],
            src.dtype(),
        ));
    }

    // Choose kernel name and output dtype.
    let (kernel_name, out_dtype) = match (op, src.dtype()) {
        // Float32
        (ScatterOp::Overwrite, DType::Float32) => ("scatter_overwrite_f32", DType::Float32),
        (ScatterOp::Add,       DType::Float32) => ("scatter_add_f32",       DType::Float32),
        (ScatterOp::Max,       DType::Float32) => ("scatter_max_f32",       DType::Float32),
        (ScatterOp::Min,       DType::Float32) => ("scatter_min_f32",       DType::Float32),

        // Float16 overwrite (native dtype)
        (ScatterOp::Overwrite, DType::Float16) => ("scatter_overwrite_f16", DType::Float16),
        // Float16 add (promote to f32)
        (ScatterOp::Add,       DType::Float16) => ("scatter_add_f16",       DType::Float32),

        // Bfloat16 overwrite (native dtype)
        (ScatterOp::Overwrite, DType::Bfloat16) => ("scatter_overwrite_bf16", DType::Bfloat16),
        // Bfloat16 add (promote to f32)
        (ScatterOp::Add,       DType::Bfloat16) => ("scatter_add_bf16",       DType::Float32),

        _ => {
            return Err(KernelError::NotFound(format!(
                "scatter {:?} not supported for {:?}",
                op,
                src.dtype()
            )))
        }
    };

    // For scatter_min, pre-fill with +inf instead of zero.
    // For scatter_max with negative values, zero init may be wrong, but
    // mirrors PyTorch semantics (output starts at zero).
    let out = if op == ScatterOp::Min {
        let out = Array::zeros(registry.device().raw(), &[output_size], out_dtype);
        // Fill with +inf by writing f32::INFINITY.
        if out_dtype == DType::Float32 {
            let ptr = out.metal_buffer().contents() as *mut f32;
            unsafe {
                for i in 0..output_size {
                    *ptr.add(i) = f32::INFINITY;
                }
            }
        }
        out
    } else {
        Array::zeros(registry.device().raw(), &[output_size], out_dtype)
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let out_size_buf = make_u32_buf(registry.device().raw(), output_size_u32);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    enc.set_buffer(1, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);
    enc.set_buffer(3, Some(&out_size_buf), 0);

    let grid = metal::MTLSize::new(numel as u64, 1, 1);
    let tg = metal::MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Convenience: scatter_add
// ---------------------------------------------------------------------------

/// Atomically add `src` values into an output buffer at positions given by
/// `indices`.
///
/// `output[indices[i]] += src[i]`
///
/// This is the most common scatter variant, used in gradient accumulation,
/// embedding backprop, and histogram-like operations.
pub fn scatter_add(
    registry: &KernelRegistry,
    src: &Array,
    indices: &Array,
    output_size: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    scatter(registry, src, indices, output_size, ScatterOp::Add, queue)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: set up device, queue, and kernel registry for tests.
    fn setup() -> (KernelRegistry, metal::CommandQueue) {
        let device = rmlx_metal::device::GpuDevice::new().expect("Metal device");
        let queue = device.raw().new_command_queue();
        let registry = KernelRegistry::new(device);
        register(&registry).expect("register indexing kernels");
        (registry, queue)
    }

    #[test]
    fn test_gather_basic() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5]);
        let idx = Array::from_slice(dev, &[4u32, 0, 2], vec![3]);
        let out = gather(&reg, &src, &idx, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![50.0, 10.0, 30.0]);
    }

    #[test]
    fn test_gather_oob_produces_zero() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[1.0f32, 2.0, 3.0], vec![3]);
        let idx = Array::from_slice(dev, &[0u32, 99, 2], vec![3]);
        let out = gather(&reg, &src, &idx, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_gather_signed_negative_wrap() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[10.0f32, 20.0, 30.0, 40.0], vec![4]);
        // -1i32 as u32 bits, -2i32 as u32 bits
        let indices: Vec<u32> = vec![
            (-1i32) as u32,
            (-2i32) as u32,
            0u32,
        ];
        let idx = Array::from_slice(dev, &indices, vec![3]);
        let out = gather_signed(&reg, &src, &idx, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![40.0, 30.0, 10.0]);
    }

    #[test]
    fn test_scatter_overwrite() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[100.0f32, 200.0, 300.0], vec![3]);
        let idx = Array::from_slice(dev, &[1u32, 3, 0], vec![3]);
        let out = scatter(&reg, &src, &idx, 5, ScatterOp::Overwrite, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![300.0, 100.0, 0.0, 200.0, 0.0]);
    }

    #[test]
    fn test_scatter_add_accumulates() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let idx = Array::from_slice(dev, &[0u32, 1, 0, 1], vec![4]);
        let out = scatter_add(&reg, &src, &idx, 3, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        // index 0: 1.0 + 3.0 = 4.0
        // index 1: 2.0 + 4.0 = 6.0
        // index 2: 0.0
        assert_eq!(vals, vec![4.0, 6.0, 0.0]);
    }

    #[test]
    fn test_scatter_oob_skipped() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[10.0f32, 20.0, 30.0], vec![3]);
        let idx = Array::from_slice(dev, &[0u32, 999, 2], vec![3]);
        let out = scatter(&reg, &src, &idx, 4, ScatterOp::Add, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        // index 999 is out of bounds, skipped
        assert_eq!(vals, vec![10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn test_scatter_add_convenience() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[5.0f32, 5.0], vec![2]);
        let idx = Array::from_slice(dev, &[0u32, 0], vec![2]);
        let out = scatter_add(&reg, &src, &idx, 2, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![10.0, 0.0]);
    }

    #[test]
    fn test_scatter_max() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[3.0f32, 1.0, 5.0, 2.0], vec![4]);
        let idx = Array::from_slice(dev, &[0u32, 0, 1, 1], vec![4]);
        let out = scatter(&reg, &src, &idx, 2, ScatterOp::Max, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        // index 0: max(0.0, 3.0, 1.0) = 3.0
        // index 1: max(0.0, 5.0, 2.0) = 5.0
        assert_eq!(vals, vec![3.0, 5.0]);
    }

    #[test]
    fn test_scatter_min() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[3.0f32, 1.0, 5.0, 2.0], vec![4]);
        let idx = Array::from_slice(dev, &[0u32, 0, 1, 1], vec![4]);
        let out = scatter(&reg, &src, &idx, 2, ScatterOp::Min, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        // index 0: min(+inf, 3.0, 1.0) = 1.0
        // index 1: min(+inf, 5.0, 2.0) = 2.0
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_gather_empty_indices() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[1.0f32, 2.0], vec![2]);
        let idx = Array::from_slice(dev, &[] as &[u32], vec![0]);
        let out = gather(&reg, &src, &idx, &q).unwrap();
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn test_scatter_empty_src() {
        let (reg, q) = setup();
        let dev = reg.device().raw();
        let src = Array::from_slice(dev, &[] as &[f32], vec![0]);
        let idx = Array::from_slice(dev, &[] as &[u32], vec![0]);
        let out = scatter_add(&reg, &src, &idx, 3, &q).unwrap();
        let vals: Vec<f32> = out.to_vec_checked();
        assert_eq!(vals, vec![0.0, 0.0, 0.0]);
    }
}
