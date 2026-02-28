// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#include <metal_stdlib>
using namespace metal;

#include "defines.h"
#include "utils.h"

// Stub: RMSNorm kernel
// Full implementation requires vendoring from ml-explore/mlx
[[kernel]] void rms_norm(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  // Placeholder — actual MLX implementation needed
}
