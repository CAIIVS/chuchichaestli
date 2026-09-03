// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define C3LI_HD __host__ __device__
#else
#define C3LI_HD
#endif

namespace c3li {

// Signal extension modes, mirroring `MODE_TO_CODE` in
// `chuchichaestli/dwt/modes.py`. The values are part of the interface between
// the Python and the compiled implementations.
enum PadMode : int64_t {
  kZero = 0,
  kConstant = 1,
  kSymmetric = 2,
  kReflect = 3,
  kPeriodic = 4,
  kPeriodization = 5,
  kAntisymmetric = 6,
  kAntireflect = 7,
};

// An out-of-range sample expressed through the samples that exist, as
// `sign * x[index] + lo * x[0] + hi * x[n - 1]`.
struct PadRef {
  int64_t index;
  double sign;
  double lo;
  double hi;
};

// Fold an index back into `[0, n)` by the given extension mode.
//
// The folding rules are applied from the outside in, so the affine transform
// each one contributes is composed onto what is already accumulated rather
// than the other way round: a rule mapping `v -> -v + 2 x[edge]` turns
// `a v + b_lo x[0] + b_hi x[n-1]` into `-a v' + (b_lo + 2 a [edge is lo]) x[0]
// + (b_hi + 2 a [edge is hi]) x[n-1]`.
C3LI_HD inline PadRef pad_resolve(int64_t i, int64_t n, int64_t mode) {
  if (i >= 0 && i < n) {
    return {i, 1.0, 0.0, 0.0};
  }
  if (mode == kZero) {
    return {0, 0.0, 0.0, 0.0};
  }
  double scale = 1.0;
  double lo = 0.0;
  double hi = 0.0;
  // whole-sample folding needs two samples to step between
  const bool degenerate =
      n == 1 && (mode == kReflect || mode == kAntireflect);
  while (!degenerate && (i < 0 || i >= n)) {
    const bool left = i < 0;
    switch (mode) {
      case kConstant:
        i = left ? 0 : n - 1;
        break;
      case kPeriodic:
      case kPeriodization:
        i = ((i % n) + n) % n;
        break;
      case kSymmetric:
        i = left ? -i - 1 : 2 * n - 1 - i;
        break;
      case kReflect:
        i = left ? -i : 2 * (n - 1) - i;
        break;
      case kAntisymmetric:
        i = left ? -i - 1 : 2 * n - 1 - i;
        scale = -scale;
        lo = -lo;
        hi = -hi;
        break;
      case kAntireflect:
        lo += left ? 2.0 * scale : 0.0;
        hi += left ? 0.0 : 2.0 * scale;
        i = left ? -i : 2 * (n - 1) - i;
        scale = -scale;
        break;
      default:
        return {0, 0.0, 0.0, 0.0};
    }
  }
  return {degenerate ? 0 : i, scale, lo, hi};
}

}  // namespace c3li
