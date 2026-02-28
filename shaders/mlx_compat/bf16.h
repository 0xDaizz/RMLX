// Copyright (c) 2023 ml-explore. MIT License.
// Vendored from ml-explore/mlx
// STUB: Replace with vendored content from ml-explore/mlx

#ifndef MLX_BF16_H
#define MLX_BF16_H

struct _MLX_BFloat16;

template <typename T>
static constexpr bool can_convert_to_bfloat16 =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr bool can_convert_from_bfloat16 =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

struct _MLX_BFloat16 {
  ushort bits_;

  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  constexpr _MLX_BFloat16(float v, int) thread : bits_(as_type<ushort2>(v).y) {}

  explicit _MLX_BFloat16(float v) thread : bits_(as_type<ushort2>(v).y) {}

  operator float() const thread { return as_type<float>(ushort2(0, bits_)); }
  operator float() const threadgroup {
    return as_type<float>(ushort2(0, bits_));
  }
  operator float() const device { return as_type<float>(ushort2(0, bits_)); }
  operator float() const constant { return as_type<float>(ushort2(0, bits_)); }
};

typedef _MLX_BFloat16 bfloat16_t;

#endif // MLX_BF16_H
