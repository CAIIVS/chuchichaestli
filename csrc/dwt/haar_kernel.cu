// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../common/dispatch.h"

namespace c3li {

namespace {

// One thread per tile, reading the `2**d` corners once and writing every
// subband once. Neighbouring threads take neighbouring tiles, so reads coalesce.
template <typename scalar_t>
__global__ void haar_nd_kernel(const scalar_t* __restrict__ src,
                               scalar_t* __restrict__ dst, scalar_t gain,
                               int64_t dimensions, int64_t corners,
                               int64_t in_lane, int64_t out_lane,
                               int64_t groups, int64_t channels, int64_t tiles,
                               int64_t in_stride_0, int64_t in_stride_1,
                               int64_t in_stride_2, int64_t out_stride_0,
                               int64_t out_stride_1, int64_t total) {
  const int64_t in_stride[3] = {in_stride_0, in_stride_1, in_stride_2};
  const int64_t out_stride[3] = {out_stride_0, out_stride_1, 1};
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t flat = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < total; flat += stride) {
    const int64_t tile = flat % tiles;
    const int64_t lane = flat / tiles;

    int64_t rest = tile;
    int64_t offset = 0;
    for (int64_t d = 0; d < dimensions; ++d) {
      const int64_t k = rest / out_stride[d];
      rest -= k * out_stride[d];
      offset += 2 * k * in_stride[d];
    }

    const scalar_t* base = src + lane * in_lane + offset;
    scalar_t values[8];
    for (int64_t c = 0; c < corners; ++c) {
      int64_t corner = 0;
      for (int64_t d = 0; d < dimensions; ++d) {
        if ((c >> (dimensions - 1 - d)) & 1) {
          corner += in_stride[d];
        }
      }
      values[c] = base[corner];
    }

    for (int64_t d = 0; d < dimensions; ++d) {
      const int64_t bit = int64_t{1} << (dimensions - 1 - d);
      for (int64_t c = 0; c < corners; ++c) {
        if ((c & bit) == 0) {
          const scalar_t low = values[c];
          const scalar_t high = values[c | bit];
          values[c] = low + high;
          values[c | bit] = low - high;
        }
      }
    }

    const int64_t group = lane % groups;
    const int64_t batch = lane / groups;
    for (int64_t b = 0; b < corners; ++b) {
      dst[(batch * channels + group * corners + b) * out_lane + tile] =
          gain * values[b];
    }
  }
}

}  // namespace

torch::Tensor haar_nd_cuda(const torch::Tensor& x, double scale) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  const at::cuda::CUDAGuard guard(x.device());

  const int64_t dimensions = x.dim() - 2;
  TORCH_CHECK(dimensions >= 1 && dimensions <= 3,
              "the fused transform covers one to three axes");
  const int64_t corners = int64_t{1} << dimensions;

  auto sizes = x.sizes().vec();
  int64_t tiles = 1;
  for (int64_t d = 0; d < dimensions; ++d) {
    TORCH_CHECK(x.size(2 + d) % 2 == 0,
                "the fused transform needs every axis to be even");
    sizes[2 + d] = x.size(2 + d) / 2;
    tiles *= sizes[2 + d];
  }
  sizes[1] *= corners;
  torch::Tensor out = torch::empty(sizes, x.options());

  int64_t in_stride[3] = {1, 1, 1};
  int64_t out_stride[3] = {1, 1, 1};
  for (int64_t d = dimensions - 2; d >= 0; --d) {
    in_stride[d] = in_stride[d + 1] * x.size(3 + d);
    out_stride[d] = out_stride[d + 1] * sizes[3 + d];
  }
  const int64_t groups = x.size(1);
  const int64_t lanes = x.size(0) * groups;
  const int64_t total = lanes * tiles;
  if (total == 0) {
    return out;
  }

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "haar_nd_cuda", [&] {
    haar_nd_kernel<scalar_t>
        <<<blocks_for(total), kThreadsPerBlock, 0,
           at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            static_cast<scalar_t>(std::pow(scale, dimensions)), dimensions,
            corners, in_stride[0] * x.size(2), out_stride[0] * sizes[2],
            groups, sizes[1], tiles, in_stride[0], in_stride[1], in_stride[2],
            out_stride[0], out_stride[1], total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

}  // namespace c3li
