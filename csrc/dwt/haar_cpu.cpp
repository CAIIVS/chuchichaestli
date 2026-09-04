// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include <cmath>

#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// Transform every spatial axis at once with the Haar wavelet.
//
// Two taps consume no boundary extension on an even axis, so the separable
// transform collapses to one butterfly over the `2**d` corners of each
// `2 x ... x 2` tile, read once and written once. The nest is spelled out per
// rank so the innermost loop walks the contiguous axis.
//
// Band `b` of group `g` lands in channel `g * 2**d + b`, first axis most
// significant, as the per-axis transform produces.
torch::Tensor haar_nd_cpu(const torch::Tensor& x, double scale) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);

  const int64_t dimensions = x.dim() - 2;
  TORCH_CHECK(dimensions >= 1 && dimensions <= 3,
              "the fused transform covers one to three axes");
  const int64_t corners = int64_t{1} << dimensions;

  auto sizes = x.sizes().vec();
  for (int64_t d = 0; d < dimensions; ++d) {
    TORCH_CHECK(x.size(2 + d) % 2 == 0,
                "the fused transform needs every axis to be even");
    sizes[2 + d] = x.size(2 + d) / 2;
  }
  sizes[1] *= corners;
  torch::Tensor out = torch::empty(sizes, x.options());

  const int64_t lanes = x.size(0) * x.size(1);
  const int64_t groups = x.size(1);
  const int64_t last = sizes[2 + dimensions - 1];
  const int64_t mid = dimensions >= 2 ? sizes[2 + dimensions - 2] : 1;
  const int64_t first = dimensions == 3 ? sizes[2] : 1;
  const int64_t in_last = x.size(2 + dimensions - 1);
  const int64_t in_mid = dimensions >= 2 ? x.size(2 + dimensions - 2) : 1;
  const int64_t in_lane = in_last * in_mid * (dimensions == 3 ? x.size(2) : 1);
  const int64_t out_lane = last * mid * first;

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "haar_nd_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const scalar_t gain = static_cast<scalar_t>(std::pow(scale, dimensions));
    // the pairs sit side by side, so one vector load covers two outputs and
    // `deinterleave2` separates them; the tail stays scalar
    using Vec = at::vec::Vectorized<scalar_t>;
    const int64_t width = Vec::size();
    const Vec scaling(gain);

    parallel_for(lanes, [&](int64_t begin, int64_t end) {
      for (int64_t lane = begin; lane < end; ++lane) {
        const int64_t group = lane % groups;
        const int64_t batch = lane / groups;
        const scalar_t* in = src + lane * in_lane;
        scalar_t* base = dst + (batch * sizes[1] + group * corners) * out_lane;

        if (dimensions == 1) {
          scalar_t* lo = base;
          scalar_t* hi = base + out_lane;
          int64_t k = 0;
          for (; k + width <= last; k += width) {
            const auto pair = at::vec::deinterleave2(
                Vec::loadu(in + 2 * k), Vec::loadu(in + 2 * k + width));
            ((pair.first + pair.second) * scaling).store(lo + k);
            ((pair.first - pair.second) * scaling).store(hi + k);
          }
          for (; k < last; ++k) {
            const scalar_t a = in[2 * k];
            const scalar_t b = in[2 * k + 1];
            lo[k] = gain * (a + b);
            hi[k] = gain * (a - b);
          }
        } else if (dimensions == 2) {
          for (int64_t r = 0; r < mid; ++r) {
            const scalar_t* top = in + (2 * r) * in_last;
            const scalar_t* bottom = top + in_last;
            scalar_t* aa = base + r * last;
            scalar_t* ad = aa + out_lane;
            scalar_t* da = ad + out_lane;
            scalar_t* dd = da + out_lane;
            int64_t k = 0;
            for (; k + width <= last; k += width) {
              const auto upper = at::vec::deinterleave2(
                  Vec::loadu(top + 2 * k), Vec::loadu(top + 2 * k + width));
              const auto lower = at::vec::deinterleave2(
                  Vec::loadu(bottom + 2 * k), Vec::loadu(bottom + 2 * k + width));
              const Vec sum_top = upper.first + upper.second;
              const Vec diff_top = upper.first - upper.second;
              const Vec sum_bottom = lower.first + lower.second;
              const Vec diff_bottom = lower.first - lower.second;
              ((sum_top + sum_bottom) * scaling).store(aa + k);
              ((diff_top + diff_bottom) * scaling).store(ad + k);
              ((sum_top - sum_bottom) * scaling).store(da + k);
              ((diff_top - diff_bottom) * scaling).store(dd + k);
            }
            for (; k < last; ++k) {
              const scalar_t a = top[2 * k];
              const scalar_t b = top[2 * k + 1];
              const scalar_t c = bottom[2 * k];
              const scalar_t d = bottom[2 * k + 1];
              const scalar_t sum_top = a + b;
              const scalar_t diff_top = a - b;
              const scalar_t sum_bottom = c + d;
              const scalar_t diff_bottom = c - d;
              aa[k] = gain * (sum_top + sum_bottom);
              ad[k] = gain * (diff_top + diff_bottom);
              da[k] = gain * (sum_top - sum_bottom);
              dd[k] = gain * (diff_top - diff_bottom);
            }
          }
        } else {
          const int64_t plane = in_mid * in_last;
          for (int64_t p = 0; p < first; ++p) {
            for (int64_t r = 0; r < mid; ++r) {
              const scalar_t* rows[4] = {
                  in + (2 * p) * plane + (2 * r) * in_last,
                  in + (2 * p) * plane + (2 * r + 1) * in_last,
                  in + (2 * p + 1) * plane + (2 * r) * in_last,
                  in + (2 * p + 1) * plane + (2 * r + 1) * in_last,
              };
              scalar_t* bands[8];
              for (int64_t b = 0; b < 8; ++b) {
                bands[b] = base + b * out_lane + (p * mid + r) * last;
              }
              int64_t k = 0;
              for (; k + width <= last; k += width) {
                Vec lanes[8];
                for (int64_t q = 0; q < 4; ++q) {
                  const auto pair = at::vec::deinterleave2(
                      Vec::loadu(rows[q] + 2 * k),
                      Vec::loadu(rows[q] + 2 * k + width));
                  lanes[2 * q] = pair.first + pair.second;
                  lanes[2 * q + 1] = pair.first - pair.second;
                }
                for (int64_t q = 0; q < 2; ++q) {
                  const Vec u = lanes[q];
                  const Vec v = lanes[q + 2];
                  lanes[q] = u + v;
                  lanes[q + 2] = u - v;
                  const Vec w = lanes[q + 4];
                  const Vec z = lanes[q + 6];
                  lanes[q + 4] = w + z;
                  lanes[q + 6] = w - z;
                }
                for (int64_t q = 0; q < 4; ++q) {
                  const Vec u = lanes[q];
                  const Vec v = lanes[q + 4];
                  ((u + v) * scaling).store(bands[q] + k);
                  ((u - v) * scaling).store(bands[q + 4] + k);
                }
              }
              for (; k < last; ++k) {
                scalar_t values[8];
                for (int64_t q = 0; q < 4; ++q) {
                  const scalar_t a = rows[q][2 * k];
                  const scalar_t b = rows[q][2 * k + 1];
                  values[2 * q] = a + b;
                  values[2 * q + 1] = a - b;
                }
                // the two remaining axes, as butterflies over the partial sums
                for (int64_t q = 0; q < 2; ++q) {
                  const scalar_t u = values[q];
                  const scalar_t v = values[q + 2];
                  values[q] = u + v;
                  values[q + 2] = u - v;
                  const scalar_t w = values[q + 4];
                  const scalar_t z = values[q + 6];
                  values[q + 4] = w + z;
                  values[q + 6] = w - z;
                }
                for (int64_t q = 0; q < 4; ++q) {
                  const scalar_t u = values[q];
                  const scalar_t v = values[q + 4];
                  bands[q][k] = gain * (u + v);
                  bands[q + 4][k] = gain * (u - v);
                }
              }
            }
          }
        }
      }
    });
  });
  return out;
}

}  // namespace c3li
