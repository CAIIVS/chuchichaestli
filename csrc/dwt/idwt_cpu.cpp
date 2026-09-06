// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// Merge low- and high-pass band pairs back along one spatial axis.
// `(batch, 2 * groups, ...)` in, `(batch, groups, ...)` out.
//
// A transposed convolution puts the contribution of coefficient `k` and tap
// `f` at `2 k + f`, so gathering the output at `t` means collecting every
// `(k, f)` with `2 k + f == t + trim`. That pairs `t` with one phase of the
// filter: `f` shares the parity of `t + trim`, and the coefficients it reads
// run backwards from `(t + trim - parity) / 2`. Splitting the taps by parity
// turns the gather into two ordinary correlations with nothing to test in the
// inner loop. The critically sampled mode wraps the sum around the output
// instead of discarding the overhang.
torch::Tensor idwt_axis_cpu(const torch::Tensor& coeffs,
                            const torch::Tensor& rec_lo,
                            const torch::Tensor& rec_hi, int64_t axis,
                            int64_t mode, int64_t trim, int64_t out_length) {
  C3LI_CHECK_CONTIGUOUS(coeffs);
  C3LI_CHECK_FLOATING(coeffs);
  TORCH_CHECK(coeffs.size(1) % 2 == 0,
              "the channel count must pair a low- and a high-pass band");

  const AxisLayout layout = axis_layout(coeffs, axis);
  const int64_t filter_len = rec_lo.numel();
  const int64_t bands = coeffs.size(1);
  const int64_t groups = bands / 2;
  const int64_t coeff_length = layout.length;
  const bool circular = mode == kPeriodization;
  const int64_t inner = layout.inner;

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  sizes[2 + axis] = out_length;
  torch::Tensor out = torch::empty(sizes, coeffs.options());

  const int64_t per_group = layout.outer / (coeffs.size(0) * bands);
  const int64_t lanes = coeffs.size(0) * groups * per_group;

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_axis_cpu", [&] {
    using acc = acc_t<scalar_t>;
    using Vec = at::vec::Vectorized<scalar_t>;
    const int64_t width = Vec::size();
    const auto* src = coeffs.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = rec_lo.to(coeffs.scalar_type()).contiguous();
    const auto hi_filter = rec_hi.to(coeffs.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();

    // taps of each parity, and the coefficient the first of them reads
    const int64_t taps[2] = {(filter_len + 1) / 2, filter_len / 2};

    parallel_for(lanes, [&](int64_t begin, int64_t end) {
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);
        const int64_t low_pre = (batch * bands + 2 * group) * per_group + pre;
        const scalar_t* low_lane = src + low_pre * coeff_length * inner;
        const scalar_t* high_lane = low_lane + per_group * coeff_length * inner;
        const int64_t out_pre = (batch * groups + group) * per_group + pre;
        scalar_t* out_lane = dst + out_pre * out_length * inner;

        // one output at a time, but its taps chosen once
        const auto gather = [&](int64_t t, int64_t q) {
          const int64_t s = t + trim;
          const int64_t p = s & 1;
          const int64_t first = (s - p) / 2;
          acc value = acc(0);
          for (int64_t j = 0; j < taps[p]; ++j) {
            int64_t k = first - j;
            if (circular) {
              k = ((k % coeff_length) + coeff_length) % coeff_length;
            } else if (k < 0 || k >= coeff_length) {
              continue;
            }
            value += static_cast<acc>(lo[2 * j + p]) *
                         static_cast<acc>(low_lane[k * inner + q]) +
                     static_cast<acc>(hi[2 * j + p]) *
                         static_cast<acc>(high_lane[k * inner + q]);
          }
          return value;
        };

        // where every tap of either parity reads a coefficient that exists
        const int64_t widest = std::max(taps[0], taps[1]);
        int64_t interior_begin = 2 * (widest - 1) - trim + 1;
        int64_t interior_end = 2 * coeff_length - trim - 1;
        interior_begin = std::max<int64_t>(interior_begin, 0);
        interior_end = std::min(interior_end, out_length);
        if (interior_end < interior_begin) {
          interior_end = interior_begin;
        }

        for (int64_t t = 0; t < interior_begin; ++t) {
          for (int64_t q = 0; q < inner; ++q) {
            out_lane[t * inner + q] = static_cast<scalar_t>(gather(t, q));
          }
        }
        for (int64_t t = interior_end; t < out_length; ++t) {
          for (int64_t q = 0; q < inner; ++q) {
            out_lane[t * inner + q] = static_cast<scalar_t>(gather(t, q));
          }
        }

        if (inner == 1) {
          int64_t t = interior_begin;
          if constexpr (vectorizable<scalar_t>) {
            // consecutive outputs of one parity read consecutive coefficients,
            // so each parity loads whole vectors and the two interleave
            for (; t + 2 * width <= interior_end; t += 2 * width) {
              Vec part[2];
              for (int64_t r = 0; r < 2; ++r) {
                const int64_t s = t + r + trim;
                const int64_t p = s & 1;
                const int64_t first = (s - p) / 2;
                Vec sum_lo(scalar_t(0));
                Vec sum_hi(scalar_t(0));
                for (int64_t j = 0; j < taps[p]; ++j) {
                  const int64_t k = first - j;
                  sum_lo = at::vec::fmadd(Vec(lo[2 * j + p]),
                                          Vec::loadu(low_lane + k), sum_lo);
                  sum_hi = at::vec::fmadd(Vec(hi[2 * j + p]),
                                          Vec::loadu(high_lane + k), sum_hi);
                }
                part[r] = sum_lo + sum_hi;
              }
              const auto woven = at::vec::interleave2(part[0], part[1]);
              woven.first.store(out_lane + t);
              woven.second.store(out_lane + t + width);
            }
          }
          for (; t < interior_end; ++t) {
            const int64_t s = t + trim;
            const int64_t p = s & 1;
            const int64_t first = (s - p) / 2;
            acc value = acc(0);
            for (int64_t j = 0; j < taps[p]; ++j) {
              const int64_t k = first - j;
              value += static_cast<acc>(lo[2 * j + p]) *
                           static_cast<acc>(low_lane[k]) +
                       static_cast<acc>(hi[2 * j + p]) *
                           static_cast<acc>(high_lane[k]);
            }
            out_lane[t] = static_cast<scalar_t>(value);
          }
          continue;
        }

        for (int64_t t = interior_begin; t < interior_end; ++t) {
          const int64_t s = t + trim;
          const int64_t p = s & 1;
          const int64_t first = (s - p) / 2;
          scalar_t* row = out_lane + t * inner;
          int64_t q = 0;
          if constexpr (vectorizable<scalar_t>) {
            // the axis is strided, so a whole vector of the trailing one is
            // read for every tap
            for (; q + width <= inner; q += width) {
              Vec sum_lo(scalar_t(0));
              Vec sum_hi(scalar_t(0));
              for (int64_t j = 0; j < taps[p]; ++j) {
                const int64_t k = first - j;
                sum_lo = at::vec::fmadd(
                    Vec(lo[2 * j + p]), Vec::loadu(low_lane + k * inner + q),
                    sum_lo);
                sum_hi = at::vec::fmadd(
                    Vec(hi[2 * j + p]), Vec::loadu(high_lane + k * inner + q),
                    sum_hi);
              }
              (sum_lo + sum_hi).store(row + q);
            }
          }
          for (; q < inner; ++q) {
            acc value = acc(0);
            for (int64_t j = 0; j < taps[p]; ++j) {
              const int64_t k = first - j;
              value += static_cast<acc>(lo[2 * j + p]) *
                           static_cast<acc>(low_lane[k * inner + q]) +
                       static_cast<acc>(hi[2 * j + p]) *
                           static_cast<acc>(high_lane[k * inner + q]);
            }
            row[q] = static_cast<scalar_t>(value);
          }
        }
      }
    });
  });
  return out;
}

}  // namespace c3li
