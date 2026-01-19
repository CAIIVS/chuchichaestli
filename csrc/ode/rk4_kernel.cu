// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// RK4 forward kernel
template <typename scalar_t>
__global__ void rk4_forward_kernel(
    const scalar_t* __restrict__ y0,
    const scalar_t* __restrict__ k1,
    const scalar_t* __restrict__ k2,
    const scalar_t* __restrict__ k3,
    const scalar_t* __restrict__ k4,
    scalar_t* __restrict__ y_out,
    const scalar_t dt,
    const int batch_size,
    const int dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * dim;

    if (idx < total_size) {
        // RK4 formula: y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        y_out[idx] = y0[idx] + (dt / 6.0) * (
            k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]
        );
    }
}

// Compute intermediate RK4 stage
template <typename scalar_t>
__global__ void rk4_stage_kernel(
    const scalar_t* __restrict__ y0,
    const scalar_t* __restrict__ k,
    scalar_t* __restrict__ y_stage,
    const scalar_t factor,
    const scalar_t dt,
    const int batch_size,
    const int dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * dim;
    
    if (idx < total_size) {
        y_stage[idx] = y0[idx] + factor * dt * k[idx];
    }
}

// RK4 backward kernel
template <typename scalar_t>
__global__ void rk4_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_y0,
    scalar_t* __restrict__ grad_k1,
    scalar_t* __restrict__ grad_k2,
    scalar_t* __restrict__ grad_k3,
    scalar_t* __restrict__ grad_k4,
    const scalar_t dt,
    const int batch_size,
    const int dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * dim;
    
    if (idx < total_size) {
        const scalar_t grad = grad_output[idx];
        grad_y0[idx] = grad;
        grad_k1[idx] = (dt / 6.0) * grad;
        grad_k2[idx] = (dt / 3.0) * grad;
        grad_k3[idx] = (dt / 3.0) * grad;
        grad_k4[idx] = (dt / 6.0) * grad;
    }
}

torch::Tensor rk4_cuda_forward(
    torch::Tensor y0,
    torch::Tensor k1,
    torch::Tensor k2,
    torch::Tensor k3,
    torch::Tensor k4,
    double dt) {
    
    const int batch_size = y0.size(0);
    const int dim = y0.size(1);
    
    auto y_out = torch::empty_like(y0);
    
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(y0.scalar_type(), "rk4_forward_cuda", ([&] {
        rk4_forward_kernel<scalar_t><<<blocks, threads>>>(
            y0.data_ptr<scalar_t>(),
            k1.data_ptr<scalar_t>(),
            k2.data_ptr<scalar_t>(),
            k3.data_ptr<scalar_t>(),
            k4.data_ptr<scalar_t>(),
            y_out.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt),
            batch_size,
            dim);
    }));
    
    return y_out;
}

torch::Tensor rk4_cuda_stage(
    torch::Tensor y0,
    torch::Tensor k,
    double factor,
    double dt) {
    
    const int batch_size = y0.size(0);
    const int dim = y0.size(1);
    
    auto y_stage = torch::empty_like(y0);
    
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(y0.scalar_type(), "rk4_stage_cuda", ([&] {
        rk4_stage_kernel<scalar_t><<<blocks, threads>>>(
            y0.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            y_stage.data_ptr<scalar_t>(),
            static_cast<scalar_t>(factor),
            static_cast<scalar_t>(dt),
            batch_size,
            dim);
    }));
    
    return y_stage;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
rk4_cuda_backward(
    torch::Tensor grad_output,
    double dt) {
    
    const int batch_size = grad_output.size(0);
    const int dim = grad_output.size(1);
    
    auto grad_y0 = torch::empty_like(grad_output);
    auto grad_k1 = torch::empty_like(grad_output);
    auto grad_k2 = torch::empty_like(grad_output);
    auto grad_k3 = torch::empty_like(grad_output);
    auto grad_k4 = torch::empty_like(grad_output);
    
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "rk4_backward_cuda", ([&] {
        rk4_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_y0.data_ptr<scalar_t>(),
            grad_k1.data_ptr<scalar_t>(),
            grad_k2.data_ptr<scalar_t>(),
            grad_k3.data_ptr<scalar_t>(),
            grad_k4.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt),
            batch_size,
            dim);
    }));
    
    return std::make_tuple(grad_y0, grad_k1, grad_k2, grad_k3, grad_k4);
}
