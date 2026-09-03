// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

namespace {

// Samples per tile of the contiguous axis; small enough for the accumulators to
// stay in registers, large enough to fill a vector unit.
constexpr int64_t kTile = 64;

}  // namespace

// Split every band of `x` along one spatial axis into a low- and a high-pass
// half, without materializing the boundary extension.
//
// `x` is `(batch, groups, spatial...)` and the result is
// `(batch, 2 * groups, spatial...)` with the low-pass half of group `g` at
// channel `2 g` and its high-pass half at `2 g + 1`.
//
// The output reads `out[k] = sum_f filter[f] * x_ext[2 k + offset - f]`, which
// is the flipped, strided convolution of the padded signal that the reference
// implementation performs, with `offset = filter_len - 1 - pad_lo`.
//
// The loops are nested rather than flattened so that the index arithmetic and
// the boundary rules are hoisted out of the innermost one, which then walks the
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

    // one task per lane, which is everything the axis is not
    parallel_for(layout.outer, [&](int64_t begin, int64_t end) {
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);

        const scalar_t* lane = src + o * length * inner;
        scalar_t* out_low =
            dst + ((batch * (2 * groups) + 2 * group) * per_group + pre) *
                      out_length * inner;
        scalar_t* out_high = out_low + per_group * out_length * inner;

        // the taps of output `k` span `[2 k + offset - filter_len + 1, 2 k +
        // offset]`, so the outputs whose span lies inside the signal form one
        // contiguous run; splitting it out leaves the hot loop branch-free
        const int64_t interior_begin =
            std::max<int64_t>(0, (filter_len - offset) / 2);
        const int64_t interior_end =
            std::min(out_length, (length - offset + 1) / 2);

        for (int64_t k = 0; k < out_length; ++k) {
          const int64_t last = 2 * k + offset;
          const int64_t first = last - (filter_len - 1);
          const bool interior = k >= interior_begin && k < interior_end;
          scalar_t* low_row = out_low + k * inner;
          scalar_t* high_row = out_high + k * inner;

          if (inner == 1 && interior && k + width <= interior_end) {
            // the axis is the contiguous one, so consecutive outputs read every
            // other sample; a pair of vector loads and one de-interleave give a
            // whole vector of them
            Vec acc_low(scalar_t(0));
            Vec acc_high(scalar_t(0));
            for (int64_t f = 0; f < filter_len; ++f) {
              const scalar_t* p = lane + 2 * k + offset - f;
              const auto pair =
                  at::vec::deinterleave2(Vec::loadu(p), Vec::loadu(p + width));
              acc_low = acc_low + Vec(lo[f]) * pair.first;
              acc_high = acc_high + Vec(hi[f]) * pair.first;
            }
            acc_low.store(low_row);
            acc_high.store(high_row);
            k += width - 1;
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
            // the rules depend on the sample index alone, so they are resolved
            // once per tap and reused across the contiguous axis
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

namespace c3li {

// Transform every axis, repeatedly, each level working on the approximation of
// the one before it.
//
// The lengths and paddings a level needs follow from the one before it, so
// running the recursion here saves a round trip and a copy per level.
std::vector<torch::Tensor> wavedec_axes_cpu(const torch::Tensor& x,
                                            const torch::Tensor& dec_lo,
                                            const torch::Tensor& dec_hi,
                                            int64_t mode, int64_t levels) {
  TORCH_CHECK(levels >= 1, "a decomposition needs at least one level");
  const int64_t dimensions = x.dim() - 2;
  const int64_t filter_len = dec_lo.numel();
  const int64_t corners = int64_t{1} << dimensions;
  const int64_t groups = x.size(1);

  std::vector<torch::Tensor> stacked;
  stacked.reserve(levels);
  torch::Tensor current = x;
  for (int64_t level = 0; level < levels; ++level) {
    torch::Tensor bands = current;
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      const int64_t length = bands.size(2 + axis);
      int64_t pad_lo;
      int64_t out_length;
      if (mode == kPeriodization) {
        TORCH_CHECK(length % 2 == 0,
                    "the fused recursion needs even axes for this mode");
        pad_lo = filter_len / 2 - 1;
        out_length = length / 2;
      } else {
        pad_lo = filter_len - 2;
        out_length = (length + filter_len - 1) / 2;
      }
      bands = dwt_axis_cpu(bands, dec_lo, dec_hi, axis, mode, pad_lo, out_length);
    }
    stacked.push_back(bands);
    if (level + 1 < levels) {
      // the approximation of every group leads its block of subbands
      current = bands.slice(1, 0, groups * corners, corners).contiguous();
    }
  }
  return stacked;
}

}  // namespace c3li
