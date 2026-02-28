// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#ifndef MLX_UTILS_H
#define MLX_UTILS_H

#include "defines.h"

// Elem to loc mapping
template <typename IdxT, int NDIM>
METAL_FUNC IdxT elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const size_t* strides) {
  IdxT loc = 0;
  for (int i = NDIM - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Overload for runtime ndim
template <typename IdxT>
METAL_FUNC IdxT elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const size_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

#endif // MLX_UTILS_H
