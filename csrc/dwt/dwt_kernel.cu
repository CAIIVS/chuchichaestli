// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../common/boundary.h"
#include "../common/dispatch.h"

namespace c3li {

namespace {

// One thread per output sample, indexed so neighbouring threads differ in the
// contiguous axis and coalesce; a stride of two where that axis is transformed.
template <typename scalar_t, bool with_detail>
__global__ void dwt_axis_kernel(
    const scalar_t* __restrict__ src, scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ lo, const scalar_t* __restrict__ hi,
    int64_t length, int64_t inner, int64_t out_length, int64_t filter_len,
    int64_t offset, int64_t mode, int64_t groups, int64_t per_group,
    int64_t total) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t flat = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < total; flat += stride) {
    const int64_t q = flat % inner;
    const int64_t k = (flat / inner) % out_length;
    const int64_t o = flat / (inner * out_length);

    const int64_t pre = o % per_group;
    const int64_t group = (o / per_group) % groups;
    const int64_t batch = o / (per_group * groups);

    const scalar_t* lane = src + o * length * inner;
    const int64_t last = 2 * k + offset;
    const int64_t first = last - (filter_len - 1);

    using acc = acc_t<scalar_t>;
    acc low = acc(0);
    acc high = acc(0);
    if (first >= 0 && last < length) {
      const scalar_t* base = lane + last * inner + q;
      for (int64_t f = 0; f < filter_len; ++f) {
        const acc value = static_cast<acc>(base[-f * inner]);
        low += static_cast<acc>(lo[f]) * value;
        if constexpr (with_detail) {
          high += static_cast<acc>(hi[f]) * value;
        }
      }
    } else {
      const acc edge_lo = static_cast<acc>(lane[q]);
      const acc edge_hi = static_cast<acc>(lane[(length - 1) * inner + q]);
      for (int64_t f = 0; f < filter_len; ++f) {
        const PadRef ref = pad_resolve(last - f, length, mode);
        const acc value = static_cast<acc>(ref.sign) *
                              static_cast<acc>(lane[ref.index * inner + q]) +
                          static_cast<acc>(ref.lo) * edge_lo +
                          static_cast<acc>(ref.hi) * edge_hi;
        low += static_cast<acc>(lo[f]) * value;
        if constexpr (with_detail) {
          high += static_cast<acc>(hi[f]) * value;
        }
      }
    }

    if constexpr (!with_detail) {
      dst[o * out_length * inner + k * inner + q] =
          static_cast<scalar_t>(low);
      continue;
    }
    scalar_t* out_low =
        dst + ((batch * (2 * groups) + 2 * group) * per_group + pre) *
                  out_length * inner;
    out_low[k * inner + q] = static_cast<scalar_t>(low);
    out_low[per_group * out_length * inner + k * inner + q] =
        static_cast<scalar_t>(high);
  }
}

}  // namespace

namespace {

// Both entry points below differ only in whether the detail half is wanted:
// the low-pass-only form leaves the channel count alone and instantiates the
// kernel without the high-pass arm.
template <bool with_detail>
torch::Tensor dwt_axis_launch(const torch::Tensor& x,
                              const torch::Tensor& dec_lo,
                              const torch::Tensor& dec_hi, int64_t axis,
                              int64_t mode, int64_t pad_lo, int64_t out_length,
                              c10::optional<torch::Tensor> out_opt) {
  // the dispatch macro names itself at compile time, so this cannot be a
  // parameter
  constexpr const char* name =
      with_detail ? "dwt_axis_cuda" : "dwt_lowpass_axis_cuda";
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  const at::cuda::CUDAGuard guard(x.device());

  const AxisLayout layout = axis_layout(x, axis);
  const int64_t filter_len = dec_lo.numel();
  const int64_t offset = filter_len - 1 - pad_lo;

  auto sizes = x.sizes().vec();
  if (with_detail) {
    sizes[1] *= 2;
  }
  sizes[2 + axis] = out_length;
  torch::Tensor out = resolve_out(out_opt, sizes, x);

  const int64_t groups = x.size(1);
  const int64_t per_group = layout.outer / (x.size(0) * groups);
  const int64_t total = layout.outer * out_length * layout.inner;
  if (total == 0) {
    return out;
  }

  C3LI_DISPATCH_FLOATING(x.scalar_type(), name, [&] {
    const auto lo_filter = dec_lo.to(x.options()).contiguous();
    const auto hi_filter = dec_hi.to(x.options()).contiguous();
    dwt_axis_kernel<scalar_t, with_detail>
        <<<blocks_for(total), kThreadsPerBlock, 0,
           at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            lo_filter.data_ptr<scalar_t>(), hi_filter.data_ptr<scalar_t>(),
            layout.length, layout.inner, out_length, filter_len, offset, mode,
            groups, per_group, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

}  // namespace

torch::Tensor dwt_axis_cuda(const torch::Tensor& x, const torch::Tensor& dec_lo,
                            const torch::Tensor& dec_hi, int64_t axis,
                            int64_t mode, int64_t pad_lo, int64_t out_length,
                            c10::optional<torch::Tensor> out_opt) {
  return dwt_axis_launch<true>(x, dec_lo, dec_hi, axis, mode, pad_lo,
                               out_length, out_opt);
}

// Keep only the low-pass half, for an approximation pyramid that would drop
// the detail bands and carry twice the channels into the next axis.
torch::Tensor dwt_lowpass_axis_cuda(const torch::Tensor& x,
                                    const torch::Tensor& dec_lo, int64_t axis,
                                    int64_t mode, int64_t pad_lo,
                                    int64_t out_length,
                                    c10::optional<torch::Tensor> out_opt) {
  return dwt_axis_launch<false>(x, dec_lo, dec_lo, axis, mode, pad_lo,
                                out_length, out_opt);
}

}  // namespace c3li
