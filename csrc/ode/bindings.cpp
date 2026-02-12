// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later

#include <torch/extension.h>
#include <vector>

#ifdef USE_CUDA
// Forward declarations for CUDA functions
torch::Tensor euler_cuda_forward(torch::Tensor y0, torch::Tensor dy, double dt);
std::tuple<torch::Tensor, torch::Tensor>
euler_cuda_backward(torch::Tensor grad_output, torch::Tensor dy, double dt);

torch::Tensor rk4_cuda_forward(torch::Tensor y0, torch::Tensor k1,
                               torch::Tensor k2, torch::Tensor k3,
                               torch::Tensor k4, double dt);
torch::Tensor rk4_cuda_stage(torch::Tensor y0, torch::Tensor k, double factor,
                             double dt);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rk4_cuda_backward(torch::Tensor grad_output, double dt);

torch::Tensor linear_ode_diagonal_cuda_forward(torch::Tensor y0,
                                               torch::Tensor A_diag, double t);
std::tuple<torch::Tensor, torch::Tensor>
linear_ode_diagonal_cuda_backward(torch::Tensor grad_output, torch::Tensor y0,
                                  torch::Tensor A_diag, double t);
torch::Tensor linear_ode_matrix_cuda_forward(torch::Tensor y0, torch::Tensor A,
                                             double t);
#endif

// CPU fallback implementations (simple reference implementations)
torch::Tensor euler_forward_cpu(torch::Tensor y0, torch::Tensor dy, double dt) {
  return y0 + dt * dy;
}

std::tuple<torch::Tensor, torch::Tensor>
euler_backward_cpu(torch::Tensor grad_output, torch::Tensor dy, double dt) {
  auto grad_y0 = grad_output;
  auto grad_dy = dt * grad_output;
  return std::make_tuple(grad_y0, grad_dy);
}

torch::Tensor rk4_forward_cpu(torch::Tensor y0, torch::Tensor k1,
                              torch::Tensor k2, torch::Tensor k3,
                              torch::Tensor k4, double dt) {
  return y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

torch::Tensor rk4_stage_cpu(torch::Tensor y0, torch::Tensor k, double factor,
                            double dt) {
  return y0 + factor * dt * k;
}

// Dispatcher functions
torch::Tensor euler_forward_dispatch(torch::Tensor y0, torch::Tensor dy,
                                     double dt) {
#ifdef USE_CUDA
  if (y0.is_cuda()) {
    return euler_cuda_forward(y0, dy, dt);
  }
#endif
  return euler_forward_cpu(y0, dy, dt);
}

std::tuple<torch::Tensor, torch::Tensor>
euler_backward_dispatch(torch::Tensor grad_output, torch::Tensor dy,
                        double dt) {
#ifdef USE_CUDA
  if (grad_output.is_cuda()) {
    return euler_cuda_backward(grad_output, dy, dt);
  }
#endif
  return euler_backward_cpu(grad_output, dy, dt);
}

torch::Tensor rk4_forward_dispatch(torch::Tensor y0, torch::Tensor k1,
                                   torch::Tensor k2, torch::Tensor k3,
                                   torch::Tensor k4, double dt) {
#ifdef USE_CUDA
  if (y0.is_cuda()) {
    return rk4_cuda_forward(y0, k1, k2, k3, k4, dt);
  }
#endif
  return rk4_forward_cpu(y0, k1, k2, k3, k4, dt);
}

torch::Tensor rk4_stage_dispatch(torch::Tensor y0, torch::Tensor k,
                                 double factor, double dt) {
#ifdef USE_CUDA
  if (y0.is_cuda()) {
    return rk4_cuda_stage(y0, k, factor, dt);
  }
#endif
  return rk4_stage_cpu(y0, k, factor, dt);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rk4_backward_dispatch(torch::Tensor grad_output, double dt) {
#ifdef USE_CUDA
  if (grad_output.is_cuda()) {
    return rk4_cuda_backward(grad_output, dt);
  }
#endif
  // CPU fallback
  auto grad_y0 = grad_output;
  auto grad_k1 = (dt / 6.0) * grad_output;
  auto grad_k2 = (dt / 3.0) * grad_output;
  auto grad_k3 = (dt / 3.0) * grad_output;
  auto grad_k4 = (dt / 6.0) * grad_output;
  return std::make_tuple(grad_y0, grad_k1, grad_k2, grad_k3, grad_k4);
}

torch::Tensor linear_ode_diagonal_forward_dispatch(torch::Tensor y0,
                                                   torch::Tensor A_diag,
                                                   double t) {
#ifdef USE_CUDA
  if (y0.is_cuda()) {
    return linear_ode_diagonal_cuda_forward(y0, A_diag, t);
  }
#endif
  // CPU fallback
  return torch::exp(A_diag * t).unsqueeze(0) * y0;
}

std::tuple<torch::Tensor, torch::Tensor>
linear_ode_diagonal_backward_dispatch(torch::Tensor grad_output,
                                      torch::Tensor y0, torch::Tensor A_diag,
                                      double t) {
#ifdef USE_CUDA
  if (grad_output.is_cuda()) {
    return linear_ode_diagonal_cuda_backward(grad_output, y0, A_diag, t);
  }
#endif
  // CPU fallback
  auto exp_At = torch::exp(A_diag * t).unsqueeze(0);
  auto grad_y0 = exp_At * grad_output;
  auto grad_A = torch::sum(t * exp_At * y0 * grad_output, 0);
  return std::make_tuple(grad_y0, grad_A);
}

torch::Tensor linear_ode_matrix_forward_dispatch(torch::Tensor y0,
                                                 torch::Tensor A, double t) {
#ifdef USE_CUDA
  if (y0.is_cuda()) {
    return linear_ode_matrix_cuda_forward(y0, A, t);
  }
#endif
  // CPU fallback using PyTorch's matrix exponential
  auto exp_At = torch::linalg_matrix_exp(A * t);
  return torch::matmul(y0, exp_At.t());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ODE solvers with CUDA acceleration and CPU fallback";

  // Euler method
  m.def("euler_forward", &euler_forward_dispatch, "Euler forward pass");
  m.def("euler_backward", &euler_backward_dispatch, "Euler backward pass");

  // RK4 method
  m.def("rk4_forward", &rk4_forward_dispatch, "RK4 forward pass");
  m.def("rk4_stage", &rk4_stage_dispatch, "RK4 stage computation");
  m.def("rk4_backward", &rk4_backward_dispatch, "RK4 backward pass");

  // Linear ODE solver
  m.def("linear_ode_diagonal_forward", &linear_ode_diagonal_forward_dispatch,
        "Linear ODE diagonal forward pass");
  m.def("linear_ode_diagonal_backward", &linear_ode_diagonal_backward_dispatch,
        "Linear ODE diagonal backward pass");
  m.def("linear_ode_matrix_forward", &linear_ode_matrix_forward_dispatch,
        "Linear ODE matrix forward pass");
}
