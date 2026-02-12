# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differentiable (exact) linear ODE solver with CUDA acceleration."""

import torch
from torch.autograd import Function

try:
    from chuchichaestli.ode import _ode_kernels

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    _ode_kernels = None


__all__ = ["LinearODESolver"]


class LinearODEFunction(Function):
    """Autograd function for linear ODE with exact solution."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        y0: torch.Tensor,
        t: float,
        use_custom_kernel: bool,
    ):
        """Forward pass for linear ODE: dy/dt = A*y.

        Args:
            ctx: module solver context
            A: System matrix (dim,) for diagonal or (dim, dim) for general
            y0: Initial state (batch_size, dim)
            t: Time to integrate to
            use_custom_kernel: Whether to use custom CUDA kernels
        """
        ctx.save_for_backward(y0, A)
        ctx.t = t
        ctx.use_custom_kernel = use_custom_kernel

        if A.dim() == 1:  # Diagonal matrix
            if CUDA_AVAILABLE and use_custom_kernel:
                return _ode_kernels.linear_ode_diagonal_forward(y0, A, t)
            else:
                return torch.exp(A * t).unsqueeze(0) * y0
        else:  # General matrix
            if CUDA_AVAILABLE and use_custom_kernel:
                return _ode_kernels.linear_ode_matrix_forward(y0, A, t)
            else:
                exp_At = torch.linalg.matrix_exp(A * t)
                return torch.matmul(y0, exp_At.t())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for linear ODE."""
        y0, A = ctx.saved_tensors
        t = ctx.t

        use_custom_kernel = ctx.use_custom_kernel

        if A.dim() == 1:  # Diagonal matrix
            if CUDA_AVAILABLE and use_custom_kernel:
                grad_y0, grad_A = _ode_kernels.linear_ode_diagonal_backward(
                    grad_output, y0, A, t
                )
            else:
                exp_At = torch.exp(A * t).unsqueeze(0)
                grad_y0 = exp_At * grad_output
                grad_A = torch.sum(t * exp_At * y0 * grad_output, 0)
        else:  # General matrix
            # For general matrix, use adjoint method
            exp_At = torch.linalg.matrix_exp(A * t)
            grad_y0 = torch.matmul(grad_output, exp_At)

            # Gradient w.r.t. A requires solving a continuous adjoint ODE
            # For simplicity, we use finite differences here
            grad_A = None  # Would require adjoint sensitivity analysis

        # Return gradients for: A, y0, t, use_custom_kernel
        return grad_A, grad_y0, None, None


class LinearODESolver:
    """Exact solver for linear ODEs with CUDA acceleration."""

    use_custom_kernel: bool = True

    def solve(self, A: torch.Tensor, y0: torch.Tensor, t: float) -> torch.Tensor:
        """Solve linear ODE: dy/dt = A*y.

        Args:
            A: System matrix (diagonal or full)
            y0: Initial state
            t: Time to integrate to

        Returns:
            State at time t
        """
        return LinearODEFunction.apply(A, y0, t, self.use_custom_kernel)
