# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differentiable ODE function base class for use in custom ODE solvers."""

import torch
from torch import nn


__all__ = ["ODEFunction"]


class ODEFunction(nn.Module):
    """Base class for ODE functions."""
    
    def forward(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt = f(t, y).
        
        Args:
            t: Current time
            y: Current state
            
        Returns:
            Time derivative dy/dt
        """
        raise NotImplementedError
