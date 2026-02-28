// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#ifndef MLX_BINARY_OPS_H
#define MLX_BINARY_OPS_H

// Stub: binary operation functors (Add, Sub, Mul, Div, etc.)
// Full implementation requires vendoring from ml-explore/mlx

struct Add {
  template <typename T>
  T operator()(T a, T b) { return a + b; }
};

struct Sub {
  template <typename T>
  T operator()(T a, T b) { return a - b; }
};

struct Mul {
  template <typename T>
  T operator()(T a, T b) { return a * b; }
};

struct Div {
  template <typename T>
  T operator()(T a, T b) { return a / b; }
};

#endif // MLX_BINARY_OPS_H
