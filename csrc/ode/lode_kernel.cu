// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

// Linear ODE solver: dy/dt = A*y, solution: y(t) = exp(A*t) * y0
// For diagonal A, this simplifies to: y_i(t) = exp(A_ii * t) * y0_i
template <typename scalar_t>
__global__ void linear_ode_diagonal_forward_kernel(
    const scalar_t* __restrict__ y0,
    const scalar_t* __restrict__ A_diag,
    scalar_t* __restrict__ y_out,
    const scalar_t t,
    const int batch_size,
    const int dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * dim;

    if (idx < total_size) {
        const int dim_idx = idx % dim;
        // y_i(t) = exp(A_ii * t) * y0_i
        y_out[idx] = exp(A_diag[dim_idx] * t) * y0[idx];
    }
}

// Backward pass for diagonal linear ODE
template <typename scalar_t>
__global__ void linear_ode_diagonal_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ y0,
    const scalar_t* __restrict__ A_diag,
    scalar_t* __restrict__ grad_y0,
    scalar_t* __restrict__ grad_A,
    const scalar_t t,
    const int batch_size,
    const int dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * dim;

    if (idx < total_size) {
        const int batch_idx = idx / dim;
        const int dim_idx = idx % dim;
        const scalar_t exp_At = exp(A_diag[dim_idx] * t);
        
        // Gradient w.r.t. y0
        grad_y0[idx] = exp_At * grad_output[idx];
        
        // Gradient w.r.t. A (accumulate across batch)
        atomicAdd(&grad_A[dim_idx], t * exp_At * y0[idx] * grad_output[idx]);
    }
}

// General matrix exponential using scaling and squaring
// exp(A*t) ≈ (I + A*t/2^s)^(2^s) for large enough s
template <typename scalar_t>
__device__ void matrix_exp_scaled(
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ result,
    const scalar_t t,
    const int dim,
    const int s = 5) {

    // Initialize result as identity matrix
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            result[i * dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    const scalar_t scale = t / pow(2.0, s);
    
    // Create scaled matrix: B = I + A * scale
    scalar_t B[64];  // Assuming max dim = 8 (8x8 = 64)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            B[i * dim + j] = (i == j) ? 1.0 : 0.0;
            B[i * dim + j] += A[i * dim + j] * scale;
        }
    }
    
    // Square s times: result = B^(2^s)
    for (int iter = 0; iter < s; iter++) {
        scalar_t temp[64];
        
        // Matrix multiplication: temp = result * result
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                temp[i * dim + j] = 0;
                for (int k = 0; k < dim; k++) {
                    temp[i * dim + j] += result[i * dim + k] * result[k * dim + j];
                }
            }
        }
        
        // Copy temp to result
        for (int i = 0; i < dim * dim; i++) {
            result[i] = temp[i];
        }
    }
}

// Forward kernel for general matrix A
template <typename scalar_t>
__global__ void linear_ode_matrix_forward_kernel(
    const scalar_t* __restrict__ y0,
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ y_out,
    const scalar_t t,
    const int batch_size,
    const int dim) {
    
    const int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Compute matrix exponential exp(A*t)
        __shared__ scalar_t exp_At[64];  // Shared memory for matrix exp
        
        if (threadIdx.x == 0) {
            matrix_exp_scaled(A, exp_At, t, dim);
        }
        __syncthreads();
        
        // Matrix-vector multiplication: y_out = exp_At * y0
        const int i = threadIdx.x;
        if (i < dim) {
            scalar_t sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += exp_At[i * dim + j] * y0[batch_idx * dim + j];
            }
            y_out[batch_idx * dim + i] = sum;
        }
    }
}

// CUDA forward for diagonal A
torch::Tensor linear_ode_diagonal_cuda_forward(
    torch::Tensor y0,
    torch::Tensor A_diag,
    double t) {
    
    const int batch_size = y0.size(0);
    const int dim = y0.size(1);
    
    auto y_out = torch::empty_like(y0);
    
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(y0.scalar_type(), "linear_ode_diagonal_forward_cuda", ([&] {
        linear_ode_diagonal_forward_kernel<scalar_t><<<blocks, threads>>>(
            y0.data_ptr<scalar_t>(),
            A_diag.data_ptr<scalar_t>(),
            y_out.data_ptr<scalar_t>(),
            static_cast<scalar_t>(t),
            batch_size,
            dim);
    }));
    
    return y_out;
}

std::tuple<torch::Tensor, torch::Tensor> linear_ode_diagonal_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor y0,
    torch::Tensor A_diag,
    double t) {
    
    const int batch_size = grad_output.size(0);
    const int dim = grad_output.size(1);
    
    auto grad_y0 = torch::empty_like(grad_output);
    auto grad_A = torch::zeros_like(A_diag);
    
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (batch_size * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "linear_ode_diagonal_backward_cuda", ([&] {
        linear_ode_diagonal_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            y0.data_ptr<scalar_t>(),
            A_diag.data_ptr<scalar_t>(),
            grad_y0.data_ptr<scalar_t>(),
            grad_A.data_ptr<scalar_t>(),
            static_cast<scalar_t>(t),
            batch_size,
            dim);
    }));
    
    return std::make_tuple(grad_y0, grad_A);
}

torch::Tensor linear_ode_matrix_cuda_forward(
    torch::Tensor y0,
    torch::Tensor A,
    double t) {
    
    const int batch_size = y0.size(0);
    const int dim = y0.size(1);
    
    TORCH_CHECK(dim <= 8, "Matrix exponential limited to dim <= 8 for performance");
    
    auto y_out = torch::empty_like(y0);
    
    const int threads = dim;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(y0.scalar_type(), "linear_ode_matrix_forward_cuda", ([&] {
        linear_ode_matrix_forward_kernel<scalar_t><<<blocks, threads>>>(
            y0.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            y_out.data_ptr<scalar_t>(),
            static_cast<scalar_t>(t),
            batch_size,
            dim);
    }));
    
    return y_out;
}
