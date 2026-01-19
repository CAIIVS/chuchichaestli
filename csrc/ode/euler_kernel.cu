// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define THREADS_PER_BLOCK 256

// Forward Euler kernel
template <typename scalar_t>
__global__ void euler_forward_kernel(const scalar_t *__restrict__ y0,
                                     const scalar_t *__restrict__ dy,
                                     scalar_t *__restrict__ y_out,
                                     const scalar_t dt, const int batch_size,
                                     const int dim) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_size = batch_size * dim;

  if (idx < total_size) {
    y_out[idx] = y0[idx] + dt * dy[idx];
  }
}

// Backward kernel for Euler method
template <typename scalar_t>
__global__ void euler_backward_kernel(const scalar_t *__restrict__ grad_output,
                                      const scalar_t *__restrict__ dy,
                                      scalar_t *__restrict__ grad_y0,
                                      scalar_t *__restrict__ grad_dy,
                                      const scalar_t dt, const int batch_size,
                                      const int dim) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_size = batch_size * dim;

  if (idx < total_size) {
    grad_y0[idx] = grad_output[idx];
    grad_dy[idx] = dt * grad_output[idx];
  }
}

// Forward declaration
torch::Tensor euler_cuda_forward(torch::Tensor y0, torch::Tensor dy,
                                 double dt) {

  const int batch_size = y0.size(0);
  const int dim = y0.size(1);

  auto y_out = torch::empty_like(y0);

  const int threads = THREADS_PER_BLOCK;
  const int blocks = (batch_size * dim + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      y0.scalar_type(), "euler_forward_cuda", ([&] {
        euler_forward_kernel<scalar_t><<<blocks, threads>>>(
            y0.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(),
            y_out.data_ptr<scalar_t>(), static_cast<scalar_t>(dt), batch_size,
            dim);
      }));

  return y_out;
}

std::tuple<torch::Tensor, torch::Tensor>
euler_cuda_backward(torch::Tensor grad_output, torch::Tensor dy, double dt) {

  const int batch_size = grad_output.size(0);
  const int dim = grad_output.size(1);

  auto grad_y0 = torch::empty_like(grad_output);
  auto grad_dy = torch::empty_like(dy);

  const int threads = THREADS_PER_BLOCK;
  const int blocks = (batch_size * dim + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "euler_backward_cuda", ([&] {
        euler_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(),
            grad_y0.data_ptr<scalar_t>(), grad_dy.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt), batch_size, dim);
      }));

  return std::make_tuple(grad_y0, grad_dy);
}
