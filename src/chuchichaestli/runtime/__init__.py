# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Execution framework of chuchichaestli: programs, stages and the runtime.

A `Program` is a list of stages and a `Phase` is a stage holding stages, so
schedules nest. `Runtime` runs one, reporting every `Event` to the hooks.
"""

from chuchichaestli.runtime.context import Context, C3liContextError
from chuchichaestli.runtime.events import (
    C3liRuntimeError,
    Event,
    EventType,
    Progress,
    Signal,
    filter_priority,
)
from chuchichaestli.runtime.hooks import Cancel, Console, EarlyStop, Jsonl, Timer
from chuchichaestli.runtime.runtime import (
    BACKENDS_PRESETS_MAP,
    BackendsSettings,
    BackendsPresets,
    C3liProgramError,
    Runtime,
)
from chuchichaestli.runtime.stages import (
    Barrier,
    Call,
    Every,
    Export,
    Load,
    Phase,
    Program,
    Repeat,
    StageBlock,
    When,
)
from chuchichaestli.runtime.topology import Local
from chuchichaestli.runtime.traits import (
    CriticalHook,
    Hook,
    Stage,
    Stateful,
    StoreWriterHook,
    Topology,
    is_critical,
    needs_store,
)


__all__ = [
    "Runtime",
    "C3liProgramError",
    "BackendsPresets",
    "BackendsSettings",
    "BACKENDS_PRESETS_MAP",
    "Program",
    "Phase",
    "Repeat",
    "When",
    "Every",
    "StageBlock",
    "Call",
    "Load",
    "Export",
    "Barrier",
    "Context",
    "C3liContextError",
    "Event",
    "EventType",
    "Progress",
    "Signal",
    "C3liRuntimeError",
    "filter_priority",
    "Console",
    "Jsonl",
    "Timer",
    "EarlyStop",
    "Cancel",
    "Stage",
    "Stateful",
    "Hook",
    "CriticalHook",
    "StoreWriterHook",
    "is_critical",
    "needs_store",
    "Topology",
    "Local",
]
