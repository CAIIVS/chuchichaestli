// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

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

    // one task per (outer, output sample, inner) triple
    const int64_t total = layout.outer * out_length * layout.inner;
    parallel_for(total, [&](int64_t begin, int64_t end) {
      for (int64_t flat = begin; flat < end; ++flat) {
        const int64_t q = flat % layout.inner;
        const int64_t k = (flat / layout.inner) % out_length;
        const int64_t o = flat / (layout.inner * out_length);

        // `o` runs over batch, group and the axes before this one
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);
        const int64_t pre = o % per_group;

        const scalar_t* lane =
            src + ((batch * groups + group) * per_group + pre) * layout.length *
                      layout.inner;
        const scalar_t edge_lo = lane[q];
        const scalar_t edge_hi = lane[(layout.length - 1) * layout.inner + q];

        double low = 0.0;
        double high = 0.0;
        for (int64_t f = 0; f < filter_len; ++f) {
          const PadRef ref = pad_resolve(2 * k + offset - f, layout.length, mode);
          const double value = ref.sign * static_cast<double>(
                                              lane[ref.index * layout.inner + q]) +
                               ref.lo * static_cast<double>(edge_lo) +
                               ref.hi * static_cast<double>(edge_hi);
          low += static_cast<double>(lo[f]) * value;
          high += static_cast<double>(hi[f]) * value;
        }

        const int64_t out_pre =
            ((batch * (2 * groups) + 2 * group) * per_group + pre);
        scalar_t* out_lane = dst + out_pre * out_length * layout.inner;
        out_lane[k * layout.inner + q] = static_cast<scalar_t>(low);
        out_lane[per_group * out_length * layout.inner + k * layout.inner + q] =
            static_cast<scalar_t>(high);
      }
    });
  });
  return out;
}

}  // namespace c3li
