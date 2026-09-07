// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <torch/extension.h>

#include <vector>

#include <ATen/cpu/vec/vec.h>

#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// Split every band of `x` along one spatial axis by lifting: the even and odd
// samples are written out once and then updated in place, step by step.
// Critically sampled, so the axis is even and every index wraps.
//
// Nothing dispatches to this. It is kept for the in-place and int-to-int
// transforms the convolution form cannot express.
torch::Tensor dwt_lift_axis_cpu(const torch::Tensor& x, int64_t axis,
                            const std::vector<int64_t>& on_detail,
                            const std::vector<std::vector<double>>& coeffs,
                            const std::vector<int64_t>& lows,
                            double approx_gain, int64_t approx_delay,
                            double detail_gain, int64_t detail_delay,
                            c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  TORCH_CHECK(on_detail.size() == coeffs.size() && coeffs.size() == lows.size(),
              "every lifting step needs a side, a filter and an exponent");

  const AxisLayout layout = axis_layout(x, axis);
  const int64_t length = layout.length;
  const int64_t inner = layout.inner;
  TORCH_CHECK(length % 2 == 0, "lifting needs an even axis");
  const int64_t half = length / 2;

  auto sizes = x.sizes().vec();
  sizes[1] *= 2;
  sizes[2 + axis] = half;
  torch::Tensor out = resolve_out(out_opt, sizes, x);

  const int64_t groups = x.size(1);
  const int64_t per_group = layout.outer / (x.size(0) * groups);

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "dwt_lift_axis_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();

    parallel_for(layout.outer, [&](int64_t begin, int64_t end) {
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);

        const scalar_t* lane = src + o * length * inner;
        scalar_t* approx =
            dst + ((batch * (2 * groups) + 2 * group) * per_group + pre) *
                      half * inner;
        scalar_t* detail = approx + per_group * half * inner;

        // the lazy wavelet: a rotation, so two runs rather than a modulo
        for (int64_t j = 0; j < half; ++j) {
          const int64_t a = ((j + approx_delay) % half + half) % half;
          const int64_t d = ((j + detail_delay) % half + half) % half;
          for (int64_t q = 0; q < inner; ++q) {
            approx[j * inner + q] =
                static_cast<scalar_t>(approx_gain) * lane[2 * a * inner + q];
            detail[j * inner + q] =
                static_cast<scalar_t>(detail_gain) * lane[(2 * d + 1) * inner + q];
          }
        }

        for (size_t step = 0; step < coeffs.size(); ++step) {
          scalar_t* target = on_detail[step] ? detail : approx;
          const scalar_t* source = on_detail[step] ? approx : detail;
          const int64_t low = lows[step];
          const auto& filt = coeffs[step];
          const int64_t taps = static_cast<int64_t>(filt.size());

          // where every tap lands inside the axis no index has to wrap
          const int64_t begin_in = std::max<int64_t>(0, -low);
          const int64_t end_in = std::min(half, half - low - taps + 1);

          const auto wrapped = [&](int64_t j) {
            for (int64_t q = 0; q < inner; ++q) {
              scalar_t acc = 0;
              for (int64_t i = 0; i < taps; ++i) {
                int64_t idx = j + low + i;
                idx = ((idx % half) + half) % half;
                acc += static_cast<scalar_t>(filt[i]) * source[idx * inner + q];
              }
              target[j * inner + q] += acc;
            }
          };
          for (int64_t j = 0; j < std::min(begin_in, half); ++j) {
            wrapped(j);
          }
          for (int64_t j = std::max(end_in, begin_in); j < half; ++j) {
            wrapped(j);
          }

          if (inner == 1) {
            using Vec = at::vec::Vectorized<scalar_t>;
            const int64_t width = Vec::size();
            int64_t j = begin_in;
            for (; j + width <= end_in; j += width) {
              Vec acc(scalar_t(0));
              for (int64_t i = 0; i < taps; ++i) {
                acc = acc + Vec(static_cast<scalar_t>(filt[i])) *
                                Vec::loadu(source + j + low + i);
              }
              (Vec::loadu(target + j) + acc).store(target + j);
            }
            for (; j < end_in; ++j) {
              scalar_t acc = 0;
              for (int64_t i = 0; i < taps; ++i) {
                acc += static_cast<scalar_t>(filt[i]) * source[j + low + i];
              }
              target[j] += acc;
            }
            continue;
          }
          for (int64_t j = begin_in; j < end_in; ++j) {
            for (int64_t i = 0; i < taps; ++i) {
              const scalar_t c = static_cast<scalar_t>(filt[i]);
              const scalar_t* row = source + (j + low + i) * inner;
              scalar_t* out_row = target + j * inner;
              for (int64_t q = 0; q < inner; ++q) {
                out_row[q] += c * row[q];
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
