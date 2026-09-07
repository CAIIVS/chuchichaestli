// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../common/boundary.h"
#include "../common/dispatch.h"

namespace c3li {

namespace {

// One thread per reconstructed sample, gathering the coefficients that reach
// it: the writes stay exclusive, so the folded overhang needs no atomic.
template <typename scalar_t>
__global__ void idwt_axis_kernel(
    const scalar_t* __restrict__ src, scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ lo, const scalar_t* __restrict__ hi,
    int64_t coeff_length, int64_t inner, int64_t out_length,
    int64_t filter_len, int64_t trim, bool circular, int64_t bands,
    int64_t groups, int64_t per_group, int64_t total) {
  const int64_t wrap = 2 * coeff_length;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t flat = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < total; flat += stride) {
    const int64_t q = flat % inner;
    const int64_t t = (flat / inner) % out_length;
    const int64_t rest = flat / (inner * out_length);
    const int64_t pre = rest % per_group;
    const int64_t group = (rest / per_group) % groups;
    const int64_t batch = rest / (per_group * groups);

    const scalar_t* low_lane =
        src + ((batch * bands + 2 * group) * per_group + pre) * coeff_length *
                  inner;
    const scalar_t* high_lane =
        low_lane + per_group * coeff_length * inner;

    using acc = acc_t<scalar_t>;
    acc value = acc(0);
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
      value += static_cast<acc>(lo[f]) *
                   static_cast<acc>(low_lane[k * inner + q]) +
               static_cast<acc>(hi[f]) *
                   static_cast<acc>(high_lane[k * inner + q]);
    }

    const int64_t out_pre = (batch * groups + group) * per_group + pre;
    dst[(out_pre * out_length + t) * inner + q] = static_cast<scalar_t>(value);
  }
}

}  // namespace

torch::Tensor idwt_axis_cuda(const torch::Tensor& coeffs,
                             const torch::Tensor& rec_lo,
                             const torch::Tensor& rec_hi, int64_t axis,
                             int64_t mode, int64_t trim, int64_t out_length,
                             c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(coeffs);
  C3LI_CHECK_FLOATING(coeffs);
  const at::cuda::CUDAGuard guard(coeffs.device());

  const AxisLayout layout = axis_layout(coeffs, axis);
  const int64_t filter_len = rec_lo.numel();
  const int64_t bands = coeffs.size(1);
  const int64_t groups = bands / 2;

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  sizes[2 + axis] = out_length;
  torch::Tensor out = resolve_out(out_opt, sizes, coeffs);

  const int64_t per_group = layout.outer / (coeffs.size(0) * bands);
  const int64_t total =
      coeffs.size(0) * groups * per_group * out_length * layout.inner;
  if (total == 0) {
    return out;
  }

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_axis_cuda", [&] {
    const auto lo_filter = rec_lo.to(coeffs.options()).contiguous();
    const auto hi_filter = rec_hi.to(coeffs.options()).contiguous();
    idwt_axis_kernel<scalar_t>
        <<<blocks_for(total), kThreadsPerBlock, 0,
           at::cuda::getCurrentCUDAStream()>>>(
            coeffs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            lo_filter.data_ptr<scalar_t>(), hi_filter.data_ptr<scalar_t>(),
            layout.length, layout.inner, out_length, filter_len, trim,
            mode == kPeriodization, bands, groups, per_group, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

}  // namespace c3li
