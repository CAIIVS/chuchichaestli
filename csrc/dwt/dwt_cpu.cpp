// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
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

          if (inner == 1) {
            // the axis is the contiguous one, so there is nothing to tile over:
            // accumulate straight into a scalar, walking the taps
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
            for (int64_t q0 = 0; q0 < inner; q0 += kTile) {
              const int64_t width = std::min(kTile, inner - q0);
              scalar_t acc_low[kTile] = {};
              scalar_t acc_high[kTile] = {};
              for (int64_t f = 0; f < filter_len; ++f) {
                const scalar_t wl = lo[f];
                const scalar_t wh = hi[f];
                const scalar_t* row = base - f * inner + q0;
                for (int64_t j = 0; j < width; ++j) {
                  acc_low[j] += wl * row[j];
                  acc_high[j] += wh * row[j];
                }
              }
              for (int64_t j = 0; j < width; ++j) {
                low_row[q0 + j] = acc_low[j];
                high_row[q0 + j] = acc_high[j];
              }
            }
          } else {
            // the rules depend on the sample index alone, so they are resolved
            // once per tap and reused across the contiguous axis
            const scalar_t* edge_lo = lane;
            const scalar_t* edge_hi = lane + (length - 1) * inner;
            for (int64_t q0 = 0; q0 < inner; q0 += kTile) {
              const int64_t width = std::min(kTile, inner - q0);
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
                for (int64_t j = 0; j < width; ++j) {
                  const scalar_t value = sign * row[j] +
                                         weight_lo * edge_lo[q0 + j] +
                                         weight_hi * edge_hi[q0 + j];
                  acc_low[j] += wl * value;
                  acc_high[j] += wh * value;
                }
              }
              for (int64_t j = 0; j < width; ++j) {
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
