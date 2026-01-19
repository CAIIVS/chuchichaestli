# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differentiable Runge-Kutta (4th order) solver with CUDA acceleration."""

import torch
from torch.autograd import Function
from collections.abc import Callable

try:
    from chuchichaestli.ode import _cuda_ode_solvers

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import warnings

    warnings.warn(
        "CUDA ODE solvers not available. Install with: pip install chuchichaestli[ode]"
    )


__all__ = ["RK4Solver"]


class RK4Function(Function):
    """Autograd function for RK4 method."""

    @staticmethod
    def forward(ctx, y0: torch.Tensor, func: Callable, t0: float, t1: float, dt: float):
        """Forward pass using RK4 method.

        Args:
            ctx: module solver context
            y0: Initial state (batch_size, dim)
            func: Function computing dy/dt = f(t, y)
            t0: Initial time
            t1: Final time
            dt: Time step
        """
        ctx.func = func
        ctx.dt = dt

        y = y0
        t = t0
        steps = int((t1 - t0) / dt)

        y_trajectory = [y0]

        for _ in range(steps):
            with torch.no_grad():
                # Compute RK4 stages
                k1 = func(t, y)

                if CUDA_AVAILABLE and y.is_cuda and ctx.use_custom_kernel:
                    y2 = _cuda_solvers.rk4_stage(y, k1, 0.5, dt)
                else:
                    y2 = y + 0.5 * dt * k1
                k2 = func(t + 0.5 * dt, y2)

                if CUDA_AVAILABLE and y.is_cuda and ctx.use_custom_kernel:
                    y3 = _cuda_solvers.rk4_stage(y, k2, 0.5, dt)
                else:
                    y3 = y + 0.5 * dt * k2
                k3 = func(t + 0.5 * dt, y3)

                if CUDA_AVAILABLE and y.is_cuda and ctx.use_custom_kernel:
                    y4 = _cuda_solvers.rk4_stage(y, k3, 1.0, dt)
                else:
                    y4 = y + dt * k3
                k4 = func(t + dt, y4)

                # Combine stages
                if CUDA_AVAILABLE and y.is_cuda and ctx.use_custom_kernel:
                    y = _cuda_solvers.rk4_forward(y, k1, k2, k3, k4, dt)
                else:
                    y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            y_trajectory.append(y)
            t += dt

        ctx.save_for_backward(*y_trajectory)
        ctx.t0 = t0
        ctx.steps = steps

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass through RK4 method."""
        y_trajectory = ctx.saved_tensors
        func = ctx.func
        dt = ctx.dt
        steps = ctx.steps
        t0 = ctx.t0

        grad_y = grad_output

        # Backward pass through time
        for i in range(steps - 1, -1, -1):
            y = y_trajectory[i]
            t = t0 + i * dt

            with torch.enable_grad():
                y_temp = y.detach().requires_grad_(True)

                # Recompute RK4 stages
                k1 = func(t, y_temp)
                y2 = y_temp + 0.5 * dt * k1
                k2 = func(t + 0.5 * dt, y2)
                y3 = y_temp + 0.5 * dt * k2
                k3 = func(t + 0.5 * dt, y3)
                y4 = y_temp + dt * k3
                k4 = func(t + dt, y4)

                # Compute gradients
                if CUDA_AVAILABLE and grad_y.is_cuda and ctx.use_custom_kernel:
                    grad_y0, grad_k1, grad_k2, grad_k3, grad_k4 = (
                        _cuda_solvers.rk4_backward(grad_y, dt)
                    )
                else:
                    grad_y0 = grad_y
                    grad_k1 = (dt / 6.0) * grad_y
                    grad_k2 = (dt / 3.0) * grad_y
                    grad_k3 = (dt / 3.0) * grad_y
                    grad_k4 = (dt / 6.0) * grad_y

                # Backprop through stages
                y_next = y_temp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                y_next.backward(grad_y)
                grad_y = y_temp.grad

        return grad_y, None, None, None, None


class RK4Solver:
    """RK4 ODE solver with CUDA acceleration."""

    use_custom_kernel: bool = True

    def __init__(self, dt: float = 0.01):
        """Initialize RK4 solver.

        Args:
            dt: Time step size
        """
        self.dt = dt

    def solve(
        self, y0: torch.Tensor, func: Callable, t0: float, t1: float
    ) -> torch.Tensor:
        """Solve ODE using RK4 method.

        Args:
            y0: Initial state
            func: Function computing dy/dt = f(t, y)
            t0: Initial time
            t1: Final time

        Returns:
            Final state at time t1
        """
        return RK4Function.apply(y0, func, t0, t1, self.dt)
