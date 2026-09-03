// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// Merge low- and high-pass band pairs of `coeffs` back along one spatial axis.
//
// `coeffs` is `(batch, 2 * groups, spatial...)` and the result is
// `(batch, groups, spatial...)`. The transposed convolution places the
// contribution of coefficient `k` and tap `f` at `2 k + f`, so gathering the
// output at `t` means collecting every `(k, f)` with `2 k + f == t + trim`.
// The critically sampled mode wraps that sum around the output instead, which
// is what folds the overhang back in rather than discarding it.
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
  const int64_t wrap = 2 * coeff_length;

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  sizes[2 + axis] = out_length;
  torch::Tensor out = torch::empty(sizes, coeffs.options());

  const int64_t per_group = layout.outer / (coeffs.size(0) * bands);

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_axis_cpu", [&] {
    const auto* src = coeffs.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = rec_lo.to(coeffs.scalar_type()).contiguous();
    const auto hi_filter = rec_hi.to(coeffs.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();

    const int64_t total = coeffs.size(0) * groups * per_group * out_length *
                          layout.inner;
    parallel_for(total, [&](int64_t begin, int64_t end) {
      for (int64_t flat = begin; flat < end; ++flat) {
        const int64_t q = flat % layout.inner;
        const int64_t t = (flat / layout.inner) % out_length;
        const int64_t rest = flat / (layout.inner * out_length);
        const int64_t pre = rest % per_group;
        const int64_t group = (rest / per_group) % groups;
        const int64_t batch = rest / (per_group * groups);

        const int64_t low_pre =
            (batch * bands + 2 * group) * per_group + pre;
        const scalar_t* low_lane =
            src + low_pre * coeff_length * layout.inner;
        const scalar_t* high_lane =
            low_lane + per_group * coeff_length * layout.inner;

        double value = 0.0;
        for (int64_t f = 0; f < filter_len; ++f) {
          int64_t shifted = t + trim - f;
          if (circular) {
            shifted = ((shifted % wrap) + wrap) % wrap;
          }
          if (shifted < 0 || (shifted & 1) != 0) {
            continue;
          }
          const int64_t k = shifted / 2;
          if (k >= coeff_length) {
            continue;
          }
          value += static_cast<double>(lo[f]) *
                       static_cast<double>(low_lane[k * layout.inner + q]) +
                   static_cast<double>(hi[f]) *
                       static_cast<double>(high_lane[k * layout.inner + q]);
        }

        const int64_t out_pre = (batch * groups + group) * per_group + pre;
        dst[(out_pre * out_length + t) * layout.inner + q] =
            static_cast<scalar_t>(value);
      }
    });
  });
  return out;
}

}  // namespace c3li
