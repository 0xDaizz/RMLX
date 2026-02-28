// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#ifndef MLX_REDUCTION_OPS_H
#define MLX_REDUCTION_OPS_H

// Stub: reduction operation functors (Sum, Prod, Min, Max, And, Or)
// Full implementation requires vendoring from ml-explore/mlx

struct Sum {
  template <typename T>
  T operator()(T a, T b) { return a + b; }
  template <typename T>
  static constexpr T init = T(0);
};

struct Max {
  template <typename T>
  T operator()(T a, T b) { return a > b ? a : b; }
};

#endif // MLX_REDUCTION_OPS_H
