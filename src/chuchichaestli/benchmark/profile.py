# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two ways to ask why a benchmark case runs as fast as it does.

- `profile_case` attributes the time to operators, through `torch.profiler`.
```python
profile_case(
    functools.partial(backend.apply, payload, case),
    case.label(),
    args.device,
    args.profile_repeats,
    args.profile_rows,
)
```
- `perf_case` goes a level down to the hardware counters, through `perf stat`,
  which needs a subprocess to measure:
```python
perf_case(
    case.label(),
    lambda n: [sys.executable, __file__, ..., str(n)],
    args.perf_iterations,
    moved_bytes=traffic(case)
)
```
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def profile_case(
    call: Callable[[], Any],
    title: str,
    device: str,
    repeats: int = 10,
    rows: int = 10,
    warmup: int = 3,
    trace: str | None = None,
) -> None:
    """Print where one case spends its time, per operator.

    Compiled kernels carrying `RECORD_FUNCTION` appear by name alongside the
    ATen operators rather than as an opaque Python frame.

    Args:
        call: Runs the case once.
        title: Names the case in the printed header.
        device: Device the call runs on.
        repeats: Calls to profile.
        rows: Operators to print.
        warmup: Calls to make before profiling, so lazy set-up is not counted.
        trace: Directory to write a Chrome trace of the run into, one file per
            case, for a timeline view rather than a table.
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    for _ in range(warmup):
        call()

    with profile(activities=activities, profile_memory=True) as session:
        for _ in range(repeats):
            call()

    sort = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(f"\n=== {title} ===")
    print(session.key_averages().table(sort_by=sort, row_limit=rows))

    if trace:
        directory = Path(trace)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{re.sub(r'[^A-Za-z0-9]+', '-', title).strip('-')}.json"
        session.export_chrome_trace(str(path))
        print(f"wrote {path}; open it at https://ui.perfetto.dev")


EVENTS = "cycles:u,instructions:u,cache-references:u,cache-misses:u"

# `perf stat -x` writes one row per event, with no thousands separator; the
# human-readable output groups its digits with whatever character the locale
# asks for, which is not something worth parsing
SEPARATOR = ","


def perf_counters(command: Sequence[str], events: str = EVENTS, timeout: float = 1800) -> tuple[dict[str, int], float]:
    """Run a command under `perf stat` and return its counters and wall time.

    The time is the counter's own enabled time, since `perf stat -x` prints no
    summary line to read a wall clock from.

    Args:
        command: Command to measure.
        events: Comma-separated `perf` event list.
        timeout: Seconds to allow the command.
    """
    finished = subprocess.run(
        ["perf", "stat", "-x", SEPARATOR, "-e", events, *command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    counters: dict[str, int] = {}
    seconds = 0.0
    # value,unit,event,enabled ns,percent running[,metric,metric unit]; a counter
    # the machine cannot provide reads `<not supported>` where its value would
    # be, and is left out rather than counted as a zero
    for line in finished.stderr.splitlines():
        fields = line.split(SEPARATOR)
        if len(fields) < 5:
            continue
        value, _unit, event, enabled, _running = fields[:5]
        if not value.isdigit():
            continue
        counters[event] = int(value)
        if enabled.isdigit():
            seconds = max(seconds, int(enabled) / 1e9)
    return counters, seconds


def perf_case(
    title: str,
    command: Callable[[int], Sequence[str]],
    iterations: int,
    moved_bytes: float = float("nan"),
    events: str = EVENTS,
) -> None:
    """Report the hardware counters of one case, through `perf stat`.

    Start-up costs far more than the loop, so the counters of a run that does
    nothing are subtracted from those of a run that does the work.

    Args:
        title: Names the case in the printed header.
        command: Returns the command running the case a given number of times.
        iterations: Number of times the worker runs the case.
        moved_bytes: Bytes one run of the case has to move, for the achieved
            bandwidth; omitted from the report if not known.
        events: Comma-separated `perf` event list.
    """
    print(f"\n=== {title} ===")
    try:
        loaded, loaded_seconds = perf_counters(command(iterations), events)
        idle, idle_seconds = perf_counters(command(0), events)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  perf could not run: {exc}")
        return
    if not loaded:
        print("  no counters; is /proc/sys/kernel/perf_event_paranoid too strict?")
        return

    counters = {name: value - idle.get(name, 0) for name, value in loaded.items()}
    seconds = loaded_seconds - idle_seconds
    # a counter the work did not move further than the idle run did says nothing
    # about the work; more iterations is the way out, not a number
    lost = [name for name, value in counters.items() if value <= 0]
    for name, value in counters.items():
        if value > 0:
            print(f"  {name:24s} {value:>15,}")
        else:
            print(f"  {name:24s} {'under the idle run':>18s}")
    if lost:
        print(f"  ({', '.join(lost)} need more than {iterations} iterations to show)")

    cycles = counters.get("cycles:u", 0)
    instructions = counters.get("instructions:u", 0)
    references = counters.get("cache-references:u", 0)
    misses = counters.get("cache-misses:u", 0)
    if cycles > 0 and instructions > 0:
        print(f"  {'instructions per cycle':24s} {instructions / cycles:>15.2f}")
    if references > 0 and misses > 0:
        print(f"  {'cache miss rate':24s} {misses / references:>14.1%}")
    if seconds > 0 and iterations > 0:
        print(f"  {'per call':24s} {seconds / iterations * 1e6:>13.1f} us")
        if moved_bytes == moved_bytes:  # not nan
            moved = moved_bytes * iterations
            print(f"  {'achieved bandwidth':24s} {moved / seconds / 1e9:>13.1f} GB/s")
