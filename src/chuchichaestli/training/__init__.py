# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training components: optimizers, schedules, updates and objectives.

Imports nothing from `chuchichaestli.runtime`; the dependency runs the other
way.
"""

from chuchichaestli.training.update import (
    CLIP_FUNCTIONS,
    ClipTypes,
    ReductionTypes,
    Ema,
    Swa,
    UpdatePolicy,
)
from chuchichaestli.training.optim import (
    OPTIMIZER_MAP,
    SCHEDULER_MAP,
    OptimSpec,
    OptimizerTypes,
    SchedulerSpec,
    SchedulerTypes,
    disjoint_params,
)


__all__ = [
    "OptimSpec",
    "SchedulerSpec",
    "OptimizerTypes",
    "SchedulerTypes",
    "OPTIMIZER_MAP",
    "SCHEDULER_MAP",
    "disjoint_params",
    "UpdatePolicy",
    "Ema",
    "Swa",
    "ClipTypes",
    "ReductionTypes",
    "CLIP_FUNCTIONS",
]
