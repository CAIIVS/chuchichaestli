// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include <cstring>
#include <vector>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

namespace {

// Samples per tile of the contiguous axis; small enough for the accumulators to
// stay in registers, large enough to fill a vector unit.
constexpr int64_t kTile = 64;

}  // namespace

// Split every band of `x` along one spatial axis, without materializing the
// boundary extension. `(batch, groups, ...)` in, `(batch, 2 * groups, ...)`
// out, low-pass of group `g` at channel `2 g` and high-pass at `2 g + 1`.
//
// `out[k] = sum_f filter[f] * x_ext[2 k + offset - f]`, with
// `offset = filter_len - 1 - pad_lo`, matching the reference implementation.
// The nest keeps index arithmetic out of the innermost loop, which walks the
// contiguous axis and vectorizes.
torch::Tensor dwt_axis_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                           const torch::Tensor& dec_hi, int64_t axis,
                           int64_t mode, int64_t pad_lo, int64_t out_length) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  TORCH_CHECK(dec_lo.numel() == dec_hi.numel(),
              "the two decomposition filters must have the same length");

  const AxisLayout layout = axis_layout(x, axis);
  const int64_t filter_len = dec_lo.numel();
  const int64_t offset = filter_len - 1 - pad_lo;
  const int64_t length = layout.length;
  const int64_t inner = layout.inner;

  auto sizes = x.sizes().vec();
  sizes[1] *= 2;
  sizes[2 + axis] = out_length;
  torch::Tensor out = torch::empty(sizes, x.options());

  const int64_t groups = x.size(1);
  const int64_t per_group = layout.outer / (x.size(0) * groups);

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "dwt_axis_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = dec_lo.to(x.scalar_type()).contiguous();
    const auto hi_filter = dec_hi.to(x.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();
    using Vec = at::vec::Vectorized<scalar_t>;
    const int64_t width = Vec::size();

    // Writing the extension out per lane leaves every output interior, so one
    // loop serves the axis; the slack keeps every load a whole vector.
    const int64_t extent = pad_lo + 2 * out_length + offset;

    // one task per lane: every index but the transformed axis
    parallel_for(layout.outer, [&](int64_t begin, int64_t end) {
      std::vector<scalar_t> ext(
          inner == 1 ? std::max<int64_t>(extent, 0) + 2 * width : 0);
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);

        const scalar_t* lane = src + o * length * inner;
        scalar_t* out_low =
            dst + ((batch * (2 * groups) + 2 * group) * per_group + pre) *
                      out_length * inner;
        scalar_t* out_high = out_low + per_group * out_length * inner;

        // outputs whose taps all land inside the signal form one contiguous
        // run; splitting it out leaves the hot loop branch-free
        const int64_t interior_begin =
            std::max<int64_t>(0, (filter_len - offset) / 2);
        const int64_t interior_end =
            std::min(out_length, (length - offset + 1) / 2);

        if (inner == 1) {
          // in range every mode resolves to the sample itself, so only the
          // two ends need a rule
          const int64_t body = std::min(extent, pad_lo + length);
          for (int64_t j = 0; j < std::min(pad_lo, extent); ++j) {
            const PadRef ref = pad_resolve(j - pad_lo, length, mode);
            ext[j] = static_cast<scalar_t>(ref.sign) * lane[ref.index] +
                     static_cast<scalar_t>(ref.lo) * lane[0] +
                     static_cast<scalar_t>(ref.hi) * lane[length - 1];
          }
          if (body > pad_lo) {
            std::memcpy(ext.data() + pad_lo, lane,
                        (body - pad_lo) * sizeof(scalar_t));
          }
          for (int64_t j = std::max<int64_t>(body, 0); j < extent; ++j) {
            const PadRef ref = pad_resolve(j - pad_lo, length, mode);
            ext[j] = static_cast<scalar_t>(ref.sign) * lane[ref.index] +
                     static_cast<scalar_t>(ref.lo) * lane[0] +
                     static_cast<scalar_t>(ref.hi) * lane[length - 1];
          }
          for (int64_t k = 0; k < out_length; k += width) {
            const scalar_t* tap = ext.data() + 2 * k + offset + pad_lo;
            const int64_t rest = std::min<int64_t>(width, out_length - k);
            Vec acc_low(scalar_t(0));
            Vec acc_high(scalar_t(0));
            for (int64_t f = 0; f < filter_len; ++f) {
              const auto pair = at::vec::deinterleave2(
                  Vec::loadu(tap - f), Vec::loadu(tap - f + width));
              acc_low = acc_low + Vec(lo[f]) * pair.first;
              acc_high = acc_high + Vec(hi[f]) * pair.first;
            }
            acc_low.store(out_low + k, rest);
            acc_high.store(out_high + k, rest);
          }
          continue;
        }

        for (int64_t k = 0; k < out_length; ++k) {
          const int64_t last = 2 * k + offset;
          const int64_t first = last - (filter_len - 1);
          const bool interior = k >= interior_begin && k < interior_end;
          scalar_t* low_row = out_low + k * inner;
          scalar_t* high_row = out_high + k * inner;

          if (inner == 1 && interior) {
            // consecutive outputs read every other sample, so two loads and
            // a de-interleave give a whole vector of them
            const int64_t rest = std::min<int64_t>(width, interior_end - k);
            const int64_t take = 2 * rest;
            Vec acc_low(scalar_t(0));
            Vec acc_high(scalar_t(0));
            for (int64_t f = 0; f < filter_len; ++f) {
              const scalar_t* p = lane + 2 * k + offset - f;
              const Vec first = Vec::loadu(p, std::min<int64_t>(width, take));
              const Vec second = take > width
                                     ? Vec::loadu(p + width, take - width)
                                     : Vec(scalar_t(0));
              const auto pair = at::vec::deinterleave2(first, second);
              acc_low = acc_low + Vec(lo[f]) * pair.first;
              acc_high = acc_high + Vec(hi[f]) * pair.first;
            }
            acc_low.store(low_row, rest);
            acc_high.store(high_row, rest);
            k += rest - 1;
          } else if (inner == 1) {
            // the tail, and every output whose taps reach past an end
            const scalar_t* base = lane + last;
            scalar_t acc_low = 0;
            scalar_t acc_high = 0;
            if (interior) {
              for (int64_t f = 0; f < filter_len; ++f) {
                const scalar_t value = base[-f];
                acc_low += lo[f] * value;
                acc_high += hi[f] * value;
              }
            } else {
              for (int64_t f = 0; f < filter_len; ++f) {
                const PadRef ref = pad_resolve(last - f, length, mode);
                const scalar_t value =
                    static_cast<scalar_t>(ref.sign) * lane[ref.index] +
                    static_cast<scalar_t>(ref.lo) * lane[0] +
                    static_cast<scalar_t>(ref.hi) * lane[length - 1];
                acc_low += lo[f] * value;
                acc_high += hi[f] * value;
              }
            }
            *low_row = acc_low;
            *high_row = acc_high;
          } else if (interior) {
            // every tap reads a sample that exists, so no rule is consulted
            const scalar_t* base = lane + last * inner;
            int64_t q = 0;
            for (; q + width <= inner; q += width) {
              Vec acc_low(scalar_t(0));
              Vec acc_high(scalar_t(0));
              for (int64_t f = 0; f < filter_len; ++f) {
                const Vec value = Vec::loadu(base - f * inner + q);
                acc_low = acc_low + Vec(lo[f]) * value;
                acc_high = acc_high + Vec(hi[f]) * value;
              }
              acc_low.store(low_row + q);
              acc_high.store(high_row + q);
            }
            for (; q < inner; ++q) {
              scalar_t acc_low = 0;
              scalar_t acc_high = 0;
              for (int64_t f = 0; f < filter_len; ++f) {
                const scalar_t value = base[-f * inner + q];
                acc_low += lo[f] * value;
                acc_high += hi[f] * value;
              }
              low_row[q] = acc_low;
              high_row[q] = acc_high;
            }
          } else {
            // the rules depend on the sample index alone, so each tap
            // resolves once and is reused across the contiguous axis
            const scalar_t* edge_lo = lane;
            const scalar_t* edge_hi = lane + (length - 1) * inner;
            for (int64_t q0 = 0; q0 < inner; q0 += kTile) {
              const int64_t span = std::min(kTile, inner - q0);
              scalar_t acc_low[kTile] = {};
              scalar_t acc_high[kTile] = {};
              for (int64_t f = 0; f < filter_len; ++f) {
                const PadRef ref = pad_resolve(last - f, length, mode);
                const scalar_t sign = static_cast<scalar_t>(ref.sign);
                const scalar_t weight_lo = static_cast<scalar_t>(ref.lo);
                const scalar_t weight_hi = static_cast<scalar_t>(ref.hi);
                const scalar_t wl = lo[f];
                const scalar_t wh = hi[f];
                // a zero-extended tap contributes nothing
                if (sign == 0 && weight_lo == 0 && weight_hi == 0) {
                  continue;
                }
                const scalar_t* row = lane + ref.index * inner + q0;
                for (int64_t j = 0; j < span; ++j) {
                  const scalar_t value = sign * row[j] +
                                         weight_lo * edge_lo[q0 + j] +
                                         weight_hi * edge_hi[q0 + j];
                  acc_low[j] += wl * value;
                  acc_high[j] += wh * value;
                }
              }
              for (int64_t j = 0; j < span; ++j) {
                low_row[q0 + j] = acc_low[j];
                high_row[q0 + j] = acc_high[j];
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
