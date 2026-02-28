// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#include <metal_stdlib>
using namespace metal;

#include "defines.h"
#include "utils.h"
#include "bf16.h"

// Stub: Masked GEMV kernel (for attention masking)
// Full implementation requires vendoring from ml-explore/mlx
[[kernel]] void gemv_masked(
    const device float* mat [[buffer(0)]],
    const device float* vec [[buffer(1)]],
    const device bool* mask [[buffer(2)]],
    device float* result [[buffer(3)]],
    constant int& in_vec_size [[buffer(4)]],
    constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Placeholder — actual MLX implementation needed
}
