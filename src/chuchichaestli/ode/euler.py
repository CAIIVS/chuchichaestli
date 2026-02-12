# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differentiable Euler ODE solver with CUDA acceleration."""

import torch
from torch.autograd import Function
from collections.abc import Callable

try:
    from chuchichaestli.ode import _ode_kernels

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    _ode_kernels = None


__all__ = ["EulerSolver"]


class EulerFunction(Function):
    """Autograd function for Euler method."""

    @staticmethod
    def forward(ctx, func: Callable, y0: torch.Tensor, t0: float, t1: float, dt: float, use_custom_kernel: bool):
        """Forward pass using Euler method.

        Args:
            ctx: module solver context
            func: Function computing dy/dt = f(t, y)
            y0: Initial state (batch_size, dim)
            t0: Initial time
            t1: Final time
            dt: Time step
            use_custom_kernel: Whether to use custom CUDA kernels
        """
        ctx.func = func
        ctx.dt = dt
        ctx.use_custom_kernel = use_custom_kernel

        y = y0
        t = t0
        steps = int((t1 - t0) / dt)

        # Store intermediate values for backward pass
        y_trajectory = [y0]

        for _ in range(steps):
            with torch.no_grad():
                dy = func(t, y)

            if CUDA_AVAILABLE and use_custom_kernel:
                y = _ode_kernels.euler_forward(y, dy, dt)
            else:
                y = y + dt * dy

            y_trajectory.append(y)
            t += dt

        ctx.save_for_backward(*y_trajectory)
        ctx.t0 = t0
        ctx.steps = steps

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass through Euler method."""
        y_trajectory = ctx.saved_tensors
        func = ctx.func
        dt = ctx.dt
        steps = ctx.steps
        t0 = ctx.t0
        use_custom_kernel = ctx.use_custom_kernel

        grad_y = grad_output
        grad_y0 = None

        # Backward pass through time
        for i in range(steps - 1, -1, -1):
            y = y_trajectory[i]
            t = t0 + i * dt

            # Compute Jacobian-vector product
            with torch.enable_grad():
                y_temp = y.detach().requires_grad_(True)
                dy = func(t, y_temp)

                # Compute gradient w.r.t. y
                if CUDA_AVAILABLE and use_custom_kernel:
                    grad_y_prev, grad_dy = _ode_kernels.euler_backward(grad_y, dy, dt)
                else:
                    grad_y_prev = grad_y
                    grad_dy = dt * grad_y

                # Backprop through function
                if dy.requires_grad:
                    dy.backward(grad_dy)
                    grad_y = grad_y_prev + y_temp.grad
                else:
                    grad_y = grad_y_prev

        grad_y0 = grad_y

        # Return gradients for: func, y0, t0, t1, dt, use_custom_kernel
        return None, grad_y0, None, None, None, None


class EulerSolver:
    """Euler ODE solver with CUDA acceleration."""

    use_custom_kernel: bool = True

    def __init__(self, dt: float = 0.01):
        """Initialize Euler solver.

        Args:
            dt: Time step size
        """
        self.dt = dt

    def solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t0: float,
        t1: float,
    ) -> torch.Tensor:
        """Solve ODE using Euler method.

        Args:
            func: Function computing dy/dt = f(t, y)
            y0: Initial state
            t0: Initial time
            t1: Final time

        Returns:
            Final state at time t1
        """
        return EulerFunction.apply(func, y0, t0, t1, self.dt, self.use_custom_kernel)
