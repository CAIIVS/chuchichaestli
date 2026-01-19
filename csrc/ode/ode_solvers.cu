// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later

#include <torch/extension.h>

// Forward declarations from kernel files
torch::Tensor euler_cuda_forward(torch::Tensor y0, torch::Tensor dy, double dt);
std::tuple<torch::Tensor, torch::Tensor> euler_cuda_backward(
    torch::Tensor grad_output, torch::Tensor dy, double dt);

torch::Tensor rk4_cuda_forward(
    torch::Tensor y0, torch::Tensor k1, torch::Tensor k2, 
    torch::Tensor k3, torch::Tensor k4, double dt);
torch::Tensor rk4_cuda_stage(torch::Tensor y0, torch::Tensor k, double factor, double dt);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
rk4_cuda_backward(torch::Tensor grad_output, double dt);

torch::Tensor linear_ode_diagonal_cuda_forward(
    torch::Tensor y0, torch::Tensor A_diag, double t);
std::tuple<torch::Tensor, torch::Tensor> linear_ode_diagonal_cuda_backward(
    torch::Tensor grad_output, torch::Tensor y0, torch::Tensor A_diag, double t);
torch::Tensor linear_ode_matrix_cuda_forward(
    torch::Tensor y0, torch::Tensor A, double t);

// Check tensor requirements
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Euler method wrappers
torch::Tensor euler_forward(torch::Tensor y0, torch::Tensor dy, double dt) {
    CHECK_INPUT(y0);
    CHECK_INPUT(dy);
    return euler_cuda_forward(y0, dy, dt);
}

std::tuple<torch::Tensor, torch::Tensor> euler_backward(
    torch::Tensor grad_output, torch::Tensor dy, double dt) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(dy);
    return euler_cuda_backward(grad_output, dy, dt);
}

// RK4 method wrappers
torch::Tensor rk4_forward(
    torch::Tensor y0, torch::Tensor k1, torch::Tensor k2,
    torch::Tensor k3, torch::Tensor k4, double dt) {
    CHECK_INPUT(y0);
    CHECK_INPUT(k1);
    CHECK_INPUT(k2);
    CHECK_INPUT(k3);
    CHECK_INPUT(k4);
    return rk4_cuda_forward(y0, k1, k2, k3, k4, dt);
}

torch::Tensor rk4_stage(torch::Tensor y0, torch::Tensor k, double factor, double dt) {
    CHECK_INPUT(y0);
    CHECK_INPUT(k);
    return rk4_cuda_stage(y0, k, factor, dt);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rk4_backward(torch::Tensor grad_output, double dt) {
    CHECK_INPUT(grad_output);
    return rk4_cuda_backward(grad_output, dt);
}

// Linear ODE solver wrappers
torch::Tensor linear_ode_diagonal_forward(
    torch::Tensor y0, torch::Tensor A_diag, double t) {
    CHECK_INPUT(y0);
    CHECK_INPUT(A_diag);
    return linear_ode_diagonal_cuda_forward(y0, A_diag, t);
}

std::tuple<torch::Tensor, torch::Tensor> linear_ode_diagonal_backward(
    torch::Tensor grad_output, torch::Tensor y0, torch::Tensor A_diag, double t) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(y0);
    CHECK_INPUT(A_diag);
    return linear_ode_diagonal_cuda_backward(grad_output, y0, A_diag, t);
}

torch::Tensor linear_ode_matrix_forward(
    torch::Tensor y0, torch::Tensor A, double t) {
    CHECK_INPUT(y0);
    CHECK_INPUT(A);
    return linear_ode_matrix_cuda_forward(y0, A, t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated ODE solvers for PyTorch";
    
    // Euler method
    m.def("euler_forward", &euler_forward, "Euler forward pass (CUDA)");
    m.def("euler_backward", &euler_backward, "Euler backward pass (CUDA)");
    
    // RK4 method
    m.def("rk4_forward", &rk4_forward, "RK4 forward pass (CUDA)");
    m.def("rk4_stage", &rk4_stage, "RK4 stage computation (CUDA)");
    m.def("rk4_backward", &rk4_backward, "RK4 backward pass (CUDA)");
    
    // Linear ODE solver
    m.def("linear_ode_diagonal_forward", &linear_ode_diagonal_forward, 
          "Linear ODE diagonal forward pass (CUDA)");
    m.def("linear_ode_diagonal_backward", &linear_ode_diagonal_backward,
          "Linear ODE diagonal backward pass (CUDA)");
    m.def("linear_ode_matrix_forward", &linear_ode_matrix_forward,
          "Linear ODE matrix forward pass (CUDA)");
}
