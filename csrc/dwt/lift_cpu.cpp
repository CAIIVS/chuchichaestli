// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <torch/extension.h>

#include <vector>

#include <ATen/cpu/vec/vec.h>

#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// Apply the lifting steps to a split lane, or take them back off it.
//
// Each step adds a filtered copy of one channel onto the other, so undoing it
// is the same filter subtracted, and undoing a run of them is the run walked
// backwards. Critically sampled, so every index wraps.
template <typename scalar_t>
void lift_steps(scalar_t* approx, scalar_t* detail, int64_t half,
                int64_t inner, const std::vector<int64_t>& on_detail,
                const std::vector<std::vector<double>>& coeffs,
                const std::vector<int64_t>& lows, bool undo) {
  const int64_t count = static_cast<int64_t>(coeffs.size());
  for (int64_t n = 0; n < count; ++n) {
    const size_t step = static_cast<size_t>(undo ? count - 1 - n : n);
    scalar_t* target = on_detail[step] ? detail : approx;
    const scalar_t* source = on_detail[step] ? approx : detail;
    const int64_t low = lows[step];
    const auto& raw = coeffs[step];
    const int64_t taps = static_cast<int64_t>(raw.size());
    std::vector<scalar_t> filt(static_cast<size_t>(taps));
    for (int64_t i = 0; i < taps; ++i) {
      filt[static_cast<size_t>(i)] =
          static_cast<scalar_t>(undo ? -raw[static_cast<size_t>(i)]
                                     : raw[static_cast<size_t>(i)]);
    }

    // where every tap lands inside the axis no index has to wrap
    const int64_t begin_in = std::max<int64_t>(0, -low);
    const int64_t end_in = std::min(half, half - low - taps + 1);

    const auto wrapped = [&](int64_t j) {
      for (int64_t q = 0; q < inner; ++q) {
        scalar_t acc = 0;
        for (int64_t i = 0; i < taps; ++i) {
          int64_t idx = j + low + i;
          idx = ((idx % half) + half) % half;
          acc += filt[static_cast<size_t>(i)] * source[idx * inner + q];
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
          acc = acc + Vec(filt[static_cast<size_t>(i)]) *
                          Vec::loadu(source + j + low + i);
        }
        (Vec::loadu(target + j) + acc).store(target + j);
      }
      for (; j < end_in; ++j) {
        scalar_t acc = 0;
        for (int64_t i = 0; i < taps; ++i) {
          acc += filt[static_cast<size_t>(i)] * source[j + low + i];
        }
        target[j] += acc;
      }
      continue;
    }
    for (int64_t j = begin_in; j < end_in; ++j) {
      for (int64_t i = 0; i < taps; ++i) {
        const scalar_t c = filt[static_cast<size_t>(i)];
        const scalar_t* row = source + (j + low + i) * inner;
        scalar_t* out_row = target + j * inner;
        for (int64_t q = 0; q < inner; ++q) {
          out_row[q] += c * row[q];
        }
      }
    }
  }
}


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

        lift_steps(approx, detail, half, inner, on_detail, coeffs, lows,
                   /*undo=*/false);
      }
    });
  });
  return out;
}

// Merge a band pair along one spatial axis by lifting, undoing what
// `dwt_lift_axis_cpu` laid down. `(batch, 2 * groups, ...)` in,
// `(batch, groups, ...)` out, the low-pass of group `g` read from channel
// `2 g` and the high-pass from `2 g + 1`.
torch::Tensor idwt_lift_axis_cpu(
    const torch::Tensor& coeffs, int64_t axis,
    const std::vector<int64_t>& on_detail,
    const std::vector<std::vector<double>>& steps,
    const std::vector<int64_t>& lows, double approx_gain, int64_t approx_delay,
    double detail_gain, int64_t detail_delay,
    c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(coeffs);
  C3LI_CHECK_FLOATING(coeffs);
  TORCH_CHECK(on_detail.size() == steps.size() && steps.size() == lows.size(),
              "every lifting step needs a side, a filter and an exponent");
  TORCH_CHECK(coeffs.size(1) % 2 == 0,
              "the channel count must pair a low- and a high-pass band");
  TORCH_CHECK(approx_gain != 0.0 && detail_gain != 0.0,
              "a lifting scale of zero cannot be undone");

  const AxisLayout layout = axis_layout(coeffs, axis);
  const int64_t half = layout.length;
  const int64_t inner = layout.inner;
  const int64_t bands = coeffs.size(1);
  const int64_t groups = bands / 2;

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  sizes[2 + axis] = 2 * half;
  torch::Tensor out = resolve_out(out_opt, sizes, coeffs);

  const int64_t per_group = layout.outer / (coeffs.size(0) * bands);
  const int64_t lanes = coeffs.size(0) * groups * per_group;

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_lift_axis_cpu", [&] {
    const auto* src = coeffs.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();

    parallel_for(lanes, [&](int64_t begin, int64_t end) {
      // the steps run in place, so the bands are taken out of the input first
      std::vector<scalar_t> work(static_cast<size_t>(2 * half * inner));
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);

        const scalar_t* low_in =
            src + ((batch * bands + 2 * group) * per_group + pre) * half * inner;
        const scalar_t* high_in = low_in + per_group * half * inner;
        scalar_t* approx = work.data();
        scalar_t* detail = approx + half * inner;
        std::copy(low_in, low_in + half * inner, approx);
        std::copy(high_in, high_in + half * inner, detail);

        lift_steps(approx, detail, half, inner, on_detail, steps, lows,
                   /*undo=*/true);

        scalar_t* lane =
            dst + ((batch * groups + group) * per_group + pre) * 2 * half * inner;
        for (int64_t j = 0; j < half; ++j) {
          const int64_t a = ((j + approx_delay) % half + half) % half;
          const int64_t d = ((j + detail_delay) % half + half) % half;
          for (int64_t q = 0; q < inner; ++q) {
            lane[2 * a * inner + q] =
                approx[j * inner + q] / static_cast<scalar_t>(approx_gain);
            lane[(2 * d + 1) * inner + q] =
                detail[j * inner + q] / static_cast<scalar_t>(detail_gain);
          }
        }
      }
    });
  });
  return out;
}

}  // namespace c3li
