# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training components: optimizers, schedules, updates and objectives.

Imports nothing from `chuchichaestli.runtime`; the dependency runs the other
way.
"""

from chuchichaestli.training.adversarial import (
    ADV_DISC_LOSSES,
    ADV_GEN_LOSSES,
    AdversarialTypes,
)
from chuchichaestli.training.objective import (
    RECONSTRUCTION_LOSSES,
    PerceptualBackboneTypes,
    ReconstructionLossTypes,
    AdaptiveWeight,
    Loss,
    Objective,
    Term,
)
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
    "AdversarialTypes",
    "ADV_GEN_LOSSES",
    "ADV_DISC_LOSSES",
    "ReconstructionLossTypes",
    "PerceptualBackboneTypes",
    "RECONSTRUCTION_LOSSES",
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
    "Loss",
    "Term",
    "Objective",
    "AdaptiveWeight",
]
