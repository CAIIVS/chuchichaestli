// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <torch/extension.h>

#define C3LI_CHECK_CONTIGUOUS(x)                                              \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define C3LI_CHECK_FLOATING(x)                                                \
  TORCH_CHECK((x).is_floating_point(), #x " must be a floating point tensor")

// Dispatch over the floating point types the transform supports.
#define C3LI_DISPATCH_FLOATING(TYPE, NAME, ...)                               \
  AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

namespace c3li {

constexpr int kThreadsPerBlock = 256;

// Number of blocks needed to cover `total` items.
inline int64_t blocks_for(int64_t total, int64_t threads = kThreadsPerBlock) {
  return (total + threads - 1) / threads;
}

// Split `(batch, groups, spatial...)` into the strides one axis needs.
//
// `outer` counts everything before the axis, `length` is the axis itself and
// `inner` counts everything after it, so a sample sits at
// `((o * length) + i) * inner + q`.
struct AxisLayout {
  int64_t outer;
  int64_t length;
  int64_t inner;
};

inline AxisLayout axis_layout(const torch::Tensor& x, int64_t axis) {
  const int64_t spatial = x.dim() - 2;
  TORCH_CHECK(axis >= 0 && axis < spatial, "axis ", axis,
              " is out of range for ", spatial, " spatial dimension(s)");
  int64_t outer = x.size(0) * x.size(1);
  for (int64_t d = 0; d < axis; ++d) {
    outer *= x.size(2 + d);
  }
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < spatial; ++d) {
    inner *= x.size(2 + d);
  }
  return {outer, x.size(2 + axis), inner};
}

}  // namespace c3li
