// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#include <metal_stdlib>
using namespace metal;

#include "defines.h"
#include "utils.h"
#include "bf16.h"

// Stub: GEMV (General Matrix-Vector multiply) kernel
// Full implementation requires vendoring from ml-explore/mlx
[[kernel]] void gemv(
    const device float* mat [[buffer(0)]],
    const device float* vec [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant int& in_vec_size [[buffer(3)]],
    constant int& out_vec_size [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Placeholder — actual MLX implementation needed
}
