# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the shared benchmark measuring machinery."""

import inspect
import json
import os
import sys
from dataclasses import dataclass

import pytest
import torch
from chuchichaestli.benchmark.utils import (
    ALLOCATOR_ENV,
    Backend,
    Case,
    Result,
    TensorCase,
    as_measurements,
    load_rows,
    numpy_input,
    peak_memory,
    pin_allocator,
    plot,
    repeated,
    report,
    report_rows,
    sweep,
    torch_input,
    total_energy,
    write,
)

from .benchmark_cases import BAD, GOOD, WORKER, Hand, halve, namespace, reference, square


# the source of truth is the signature, so the test cannot drift from it
DEFAULT_PALETTE = inspect.signature(plot).parameters["palette"].default


class TestCaseProtocol:
    """A benchmark's own case satisfies the protocol by naming itself."""

    def test_a_case_with_every_member_matches(self):
        """`Case` is structural, so no benchmark has to import a base class."""
        assert isinstance(square(4), Case)
        assert isinstance(Hand(4), Case)

    def test_a_case_missing_a_member_does_not(self):
        """A case that cannot name itself could not be reported."""
        assert not isinstance(object(), Case)

    def test_a_case_owing_nothing_to_the_base_still_runs(self):
        """Subclassing `TensorCase` is a convenience, never a requirement."""
        args = namespace(backends=["good"])
        results = sweep([Hand(4)], {"good": GOOD}, args)
        assert [r.status for r in results] == ["ok"]
        assert results[0].row()["case"] == "4x4"


class TestRender:
    """Putting a case field into something json and csv can hold."""

    def test_a_dtype_loses_its_prefix(self):
        """`torch.float32` is not json, and `torch.` says nothing in a column."""
        assert TensorCase.render(torch.float32) == "float32"

    def test_a_tuple_becomes_a_shape(self):
        """Shapes read as `256x256` everywhere else, so they do here too."""
        assert TensorCase.render((2, 3, 64)) == "2x3x64"
        assert TensorCase.render([64]) == "64"

    def test_everything_else_is_left_alone(self):
        """Scalars already survive a round trip through json."""
        assert TensorCase.render(3) == 3
        assert TensorCase.render("db4") == "db4"
        assert TensorCase.render(None) is None

    def test_a_subclass_can_teach_it_a_field_type_of_its_own(self):
        """It is reached through the instance, so overriding it takes effect."""

        @dataclass(frozen=True, kw_only=True)
        class Fancy(TensorCase):
            flag: bool = True

            @staticmethod
            def render(value):
                """Spell booleans out, and defer on everything else."""
                return "yes" if value is True else TensorCase.render(value)

        assert Fancy(shape=(4,)).fields() == {"shape": "4", "dtype": "float32", "flag": "yes"}


class TestTensorCase:
    """The half of a case every tensor benchmark would write the same way."""

    def test_the_input_carries_the_leading_axes(self):
        """A sweep varies the spatial extent; batch and channel are the run's."""
        assert square(8).sample().shape == (2, 1, 8, 8)

    def test_the_input_is_the_same_every_time(self):
        """Every backend must be given the very same numbers to work on."""
        assert torch.equal(square(8).sample(), square(8).sample())

    def test_the_dtype_is_honoured(self):
        """The pywt-parity run is in double precision."""
        assert square(4, dtype=torch.float64).sample().dtype is torch.float64

    def test_sizes_count_the_leading_axes(self):
        """A transform moves every batch, so a bandwidth figure has to as well."""
        case = square(8)
        assert case.elements == 2 * 1 * 8 * 8
        assert case.nbytes == case.elements * 4
        assert square(8, dtype=torch.float64).nbytes == case.nbytes * 2

    def test_fields_render_every_field_of_the_subclass_too(self):
        """A benchmark adds an axis and gets a column for it, without saying so."""
        assert square(4).fields() == {"shape": "4x4", "dtype": "float32", "kind": "square"}

    def test_the_default_label_names_the_shape(self):
        """A one-axis sweep needs nothing more; anything else overrides it."""
        assert TensorCase(shape=(2, 3)).label() == "2x3"

    def test_it_is_frozen_and_keyword_only(self):
        """Cases are dict keys and are built field by field, so both matter."""
        with pytest.raises(TypeError):
            TensorCase((2, 3))
        case = TensorCase(shape=(2, 3))
        with pytest.raises(Exception):
            case.shape = (4,)


class TestBackend:
    """What a backend has to supply, and what it gets for free."""

    def test_arrays_defaults_to_detaching_a_flat_sequence(self):
        """The common case needs no converter of its own."""
        arrays = GOOD.arrays([torch.ones(2, requires_grad=True)])
        assert len(arrays) == 1 and arrays[0].shape == (2,)

    def test_arrays_uses_the_converter_when_given(self):
        """A backend returning some other structure supplies its own order."""
        backend = Backend("x", halve, torch_input, lambda r: [r[0][0].numpy()])
        assert backend.arrays([[torch.ones(2)]])[0].shape == (2,)

    def test_a_backend_supports_every_case_by_default(self):
        """A backend rules nothing out unless it says so."""
        assert GOOD.supports(square(4)) == ""


class TestResult:
    """One measurement, or the reason there is none."""

    def test_an_unmeasured_result_reports_nan(self):
        """A skipped backend has no time, and says so in a way json can hold."""
        result = Result("b", "cpu", square(4), "not installed")
        assert result.median_ms != result.median_ms
        assert result.iqr_ms != result.iqr_ms

    def test_row_carries_the_case_label_and_its_fields(self):
        """`case` is what the repeated sweep merges on, so it must be there."""
        row = Result("b", "cpu", square(4), "ok").row()
        assert row["case"] == "4x4"
        assert row["shape"] == "4x4" and row["dtype"] == "float32" and row["kind"] == "square"
        assert row["backend"] == "b" and row["status"] == "ok"


class TestProbe:
    """Why a backend cannot run a case, before anything is timed."""

    def test_ok(self):
        """A backend that runs is cleared for timing."""
        assert GOOD.probe("cpu", square(4)) == "ok"

    def test_wrong_device(self):
        """A cpu-only backend is not asked for a gpu number."""
        cpu_only = Backend("x", halve, torch_input, devices=("cpu",))
        assert cpu_only.probe("cuda", square(4)) == "no cuda"

    def test_unsupported_case(self):
        """The support test names the reason in the benchmark's own vocabulary."""
        picky = Backend("x", halve, torch_input, supports=lambda case: "too big")
        assert picky.probe("cpu", square(4)) == "too big"

    def test_missing_library_is_not_installed(self):
        """An optional comparison library that is absent is skipped, not fatal."""

        def missing(x, case):
            raise ImportError("no module named nope")

        assert Backend("x", missing, torch_input).probe("cpu", square(4)) == "not installed"

    def test_any_other_failure_disqualifies_and_is_quoted(self):
        """A backend that cannot express the case reports its own first line."""

        def broken(x, case):
            raise ValueError("axis out of range\nsecond line")

        assert Backend("x", broken, torch_input).probe("cpu", square(4)) == "axis out of range"

    def test_a_silent_failure_falls_back_to_the_type(self):
        """An exception with no message still has to say something."""

        def broken(x, case):
            raise RuntimeError

        assert Backend("x", broken, torch_input).probe("cpu", square(4)) == "RuntimeError"


class TestCheck:
    """The gate a backend passes before it is timed at all."""

    def test_no_reference_passes(self):
        """With nothing to check against, nothing is claimed."""
        assert BAD.check("cpu", square(4), square(4).sample(), None, torch.float32)

    def test_agreement_passes(self):
        """A backend that matches the reference is timed."""
        case = square(4)
        x = case.sample()
        assert GOOD.check("cpu", case, x, reference(case, x), torch.float32)

    def test_disagreement_fails(self):
        """A fast wrong answer is not a result."""
        case = square(4)
        x = case.sample()
        assert not BAD.check("cpu", case, x, reference(case, x), torch.float32)

    def test_a_different_band_count_fails(self):
        """Missing an output is a mismatch, not a shape error."""
        case = square(4)
        x = case.sample()
        short = Backend("x", lambda x, c: [], torch_input)
        assert not short.check("cpu", case, x, reference(case, x), torch.float32)

    def test_a_different_shape_fails(self):
        """The right values in the wrong layout are still wrong."""
        case = square(4)
        x = case.sample()
        flat = Backend("x", lambda x, c: [(x * 0.5).reshape(-1)], torch_input)
        assert not flat.check("cpu", case, x, reference(case, x), torch.float32)

    def test_float64_is_held_to_a_tighter_tolerance(self):
        """Double precision must not hide behind the float32 gate.

        Only where the values sit near zero, which is exactly where detail
        coefficients live and where the relative tolerance buys nothing.
        """
        case = square(4)
        x = torch.zeros(4, 4, dtype=torch.float64)
        expected = [x.numpy()]
        nudged = Backend("x", lambda x, c: [x + 1e-7], torch_input)
        assert nudged.check("cpu", case, x, expected, torch.float32)
        assert not nudged.check("cpu", case, x, expected, torch.float64)


class TestTotalEnergy:
    """Summing a result whatever structure it came back in."""

    def test_tensor(self):
        """A bare tensor is its own sum."""
        assert total_energy(torch.ones(3)).item() == 3.0

    def test_sequence(self):
        """A list of bands adds up."""
        assert total_energy([torch.ones(3), torch.ones(2)]).item() == 5.0

    def test_mapping_and_nesting(self):
        """A decomposition of keyed levels is walked to the leaves."""
        coeffs = [torch.ones(1), {"ad": torch.ones(2), "da": [torch.ones(3)]}]
        assert total_energy(coeffs).item() == 6.0

    def test_it_keeps_the_graph(self):
        """The backward direction times a real backward, so grad must survive."""
        x = torch.ones(3, requires_grad=True)
        total_energy([x * 2]).backward()
        assert x.grad is not None


class TestTimedCall:
    """What the timer actually runs."""

    def test_forward_returns_the_result(self):
        """The forward direction times the work and nothing else."""
        call = GOOD.timed_call(torch.ones(2), square(2), "forward")
        assert torch.equal(call()[0], torch.full((2,), 0.5))

    def test_backward_accumulates_grad(self):
        """The backward direction times the gradient, not just the forward."""
        x = torch.ones(2, requires_grad=True)
        GOOD.timed_call(x, square(2), "backward")()
        assert torch.equal(x.grad, torch.full((2,), 0.5))


class TestPeakMemory:
    """Peak allocation, where it means anything."""

    def test_cpu_is_not_reported(self):
        """The caching allocator hides torch tensors from `tracemalloc`."""
        value = peak_memory(lambda: torch.ones(1024), "cpu")
        assert value != value

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_cuda_is_reported(self):
        """On the GPU the number is real, and an allocation moves it."""
        assert peak_memory(lambda: torch.ones(2**18, device="cuda"), "cuda") > 0


class TestSweep:
    """The driver: probe, check, then time."""

    def test_a_correct_backend_is_timed(self):
        """A backend that agrees with the reference gets a measurement."""
        args = namespace()
        results = sweep([square(8)], {"good": GOOD}, args, reference)
        assert [r.status for r in results] == ["ok"]
        assert results[0].median_ms > 0

    def test_a_wrong_backend_is_excluded(self):
        """The gate runs before the timer, so a mismatch is never timed."""
        args = namespace(backends=["bad"])
        results = sweep([square(8)], {"bad": BAD}, args, reference)
        assert [r.status for r in results] == ["MISMATCH"]
        assert results[0].measurement is None

    def test_an_unavailable_backend_is_reported_not_run(self):
        """The reason a backend is missing survives into the report."""
        cpu_only = Backend("gpuless", halve, torch_input, devices=("cpu",))
        args = namespace(device="cuda", backends=["gpuless"])
        results = sweep([square(8)], {"gpuless": cpu_only}, args)
        assert [r.status for r in results] == ["no cuda"]

    def test_a_non_differentiable_backend_is_n_a_backwards(self):
        """PyWavelets has no autograd; that is not a failure."""
        numpy_backend = Backend(
            "np", lambda x, c: [x * 0.5], numpy_input, differentiable=False
        )
        args = namespace(backends=["np"], direction="backward")
        results = sweep([square(8)], {"np": numpy_backend}, args)
        assert [r.status for r in results] == ["n/a"]

    def test_backward_times_a_gradient(self):
        """The payload is asked for grad before the timer sees it."""
        args = namespace(direction="backward")
        results = sweep([square(8)], {"good": GOOD}, args, reference)
        assert [r.status for r in results] == ["ok"]

    def test_every_case_and_backend_is_visited(self):
        """One row per backend per case, in the order they were asked for."""
        args = namespace(backends=["good", "bad"])
        results = sweep([square(4), square(8)], {"good": GOOD, "bad": BAD}, args)
        assert [(r.backend, r.case.label()) for r in results] == [
            ("good", "4x4"), ("bad", "4x4"), ("good", "8x8"), ("bad", "8x8")
        ]


class TestReport:
    """What lands on stdout."""

    def test_measurements_and_skips_are_both_shown(self, capsys):
        """A run is not readable if it hides why a backend is missing."""
        args = namespace(backends=["good", "bad"])
        results = sweep([square(8)], {"good": GOOD, "bad": BAD}, args, reference)
        report(results, args)
        out = capsys.readouterr().out
        assert "8x8" in out
        assert "not measured" in out and "MISMATCH" in out

    def test_nothing_measured_still_prints_the_reasons(self, capsys):
        """A sweep where every backend was skipped must not print an empty page."""
        report([Result("b", "cpu", square(4), "not installed")], namespace())
        assert "not installed" in capsys.readouterr().out

    def test_peak_memory_is_reported_on_the_gpu(self, capsys):
        """Allocations are the headline number for a memory-motion argument."""
        results = sweep([square(8)], {"good": GOOD}, namespace())
        results[0].peak_mb = 12.5
        report(results, namespace(device="cuda"))
        out = capsys.readouterr().out
        assert "peak memory / MiB" in out and "12.50" in out


class TestWrite:
    """The result files."""

    def test_json_round_trips(self, tmp_path, capsys):
        """The json is the input of the repeated sweep, so it must reload."""
        path = tmp_path / "out.json"
        args = namespace(json=str(path))
        write(sweep([square(8)], {"good": GOOD}, args), args)
        rows = json.loads(path.read_text())
        assert rows[0]["case"] == "8x8" and rows[0]["backend"] == "good"

    def test_csv_has_a_header_and_a_row_per_result(self, tmp_path, capsys):
        """The csv is for reading elsewhere, so the columns have to be named."""
        path = tmp_path / "out.csv"
        args = namespace(csv=str(path))
        write(sweep([square(8)], {"good": GOOD}, args), args)
        lines = path.read_text().splitlines()
        assert lines[0] == "backend,device,case,shape,dtype,kind,status,median_ms,iqr_ms,peak_mb,label,threads"
        assert len(lines) == 2

    def test_it_draws_the_plot_it_was_asked_for(self, tmp_path, capsys):
        """`--plot` goes through the same rows as `--json`, grouped as asked."""
        pytest.importorskip("matplotlib")
        path = tmp_path / "out.png"
        args = namespace(plot=str(path))
        write(sweep([square(8)], {"good": GOOD}, args), args, lambda row: row["kind"])
        assert path.stat().st_size > 0

    def test_nothing_is_written_when_nothing_was_asked_for(self, tmp_path):
        """The default run leaves no files behind."""
        write(sweep([square(8)], {"good": GOOD}, namespace()), namespace())
        assert list(tmp_path.iterdir()) == []


class TestLoadRows:
    """Reading saved results back, so a slow sweep is measured once."""

    def rows(self, backend="good", case="8x8", median=1.0, status="ok", **extra):
        """Return one result row."""
        return [{"backend": backend, "case": case, "status": status, "median_ms": median, **extra}]

    def written(self, tmp_path, name, rows):
        """Write rows to a json file and return its path."""
        path = tmp_path / name
        path.write_text(json.dumps(rows))
        return str(path)

    def test_one_file_round_trips(self, tmp_path):
        """What a run writes is what a redraw has to be able to read."""
        path = self.written(tmp_path, "a.json", self.rows())
        assert load_rows([path]) == self.rows()

    def test_files_are_merged(self, tmp_path):
        """A sweep split one backend per process is drawn as one figure."""
        a = self.written(tmp_path, "a.json", self.rows("good"))
        b = self.written(tmp_path, "b.json", self.rows("bad"))
        assert sorted(r["backend"] for r in load_rows([a, b])) == ["bad", "good"]

    def test_a_repeated_measurement_takes_the_later_file(self, tmp_path, capsys):
        """Re-measuring one case should not draw it twice."""
        a = self.written(tmp_path, "a.json", self.rows(median=1.0))
        b = self.written(tmp_path, "b.json", self.rows(median=2.0))
        merged = load_rows([a, b])
        assert len(merged) == 1 and merged[0]["median_ms"] == 2.0
        assert "measured more than once" in capsys.readouterr().out

    def test_the_same_sweep_at_two_thread_counts_keeps_both(self, tmp_path, capsys):
        """`Compare` gives each thread count its own table, so both must survive."""
        a = self.written(tmp_path, "a.json", self.rows(median=1.0, threads=1))
        b = self.written(tmp_path, "b.json", self.rows(median=2.0, threads=16))
        merged = load_rows([a, b])
        assert sorted(r["threads"] for r in merged) == [1, 16]
        assert "measured more than once" not in capsys.readouterr().out

    def test_two_devices_are_never_the_same_measurement(self, tmp_path, capsys):
        """A cpu and a gpu row are different results even with nothing else to tell them apart."""
        a = self.written(tmp_path, "cpu.json", self.rows(median=1.0, device="cpu"))
        b = self.written(tmp_path, "cuda.json", self.rows(median=2.0, device="cuda"))
        merged = load_rows([a, b])
        assert sorted(r["device"] for r in merged) == ["cpu", "cuda"]
        assert "measured more than once" not in capsys.readouterr().out

    def test_a_missing_file_is_named(self, tmp_path):
        """The path is the only useful thing to say."""
        with pytest.raises(SystemExit, match="no such result file"):
            load_rows([str(tmp_path / "nope.json")])

    def test_something_that_is_not_json(self, tmp_path):
        """A truncated or half-written file should not read as an empty sweep."""
        path = tmp_path / "bad.json"
        path.write_text("{oh no")
        with pytest.raises(SystemExit, match="not valid json"):
            load_rows([str(path)])

    def test_json_that_is_not_result_rows(self, tmp_path):
        """Any other json would otherwise fail much later, in the plot."""
        for content, message in (('{"a": 1}', "list of result rows"), ("[1, 2]", "other than result rows")):
            path = tmp_path / "x.json"
            path.write_text(content)
            with pytest.raises(SystemExit, match=message):
                load_rows([str(path)])


class TestAsMeasurements:
    """Rebuilding timer measurements from rows, so `Compare` can lay them out."""

    def test_the_median_survives_the_round_trip(self):
        """It is the number the report shows, so it is the one that must match."""
        rows = [{"backend": "good", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 1.5}]
        assert as_measurements(rows)[0].median == pytest.approx(1.5e-3)

    def test_the_label_and_threads_are_carried_over(self):
        """They are what `Compare` groups its tables by."""
        rows = [{"backend": "good", "case": "8x8", "device": "cpu", "status": "ok",
                 "median_ms": 1.0, "label": "wavelet transform (forward, cpu)", "threads": 16}]
        spec = as_measurements(rows)[0].task_spec
        assert spec.label == "wavelet transform (forward, cpu)" and spec.num_threads == 16
        assert spec.sub_label == "8x8" and spec.description == "good"

    def test_a_row_written_before_they_were_recorded_still_reports(self):
        """Older result files should not have to be re-measured to be redrawn."""
        rows = [{"backend": "good", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 1.0}]
        spec = as_measurements(rows)[0].task_spec
        assert spec.label == "saved results (cpu)" and spec.num_threads == 1

    def test_unmeasured_rows_are_left_out(self):
        """A backend that never ran has no time to compare."""
        rows = [{"backend": "ptwt", "case": "8x8", "device": "cpu", "status": "not installed", "median_ms": None}]
        assert as_measurements(rows) == []


class TestReportRows:
    """Printing what was loaded, the way a live run prints it."""

    def test_it_prints_a_comparison_table(self, capsys):
        """A redraw should look like the run it came from, not like a new format."""
        rows = [{"backend": "good", "case": "8x8", "device": "cpu", "status": "ok", "median_ms": 1.5}]
        report_rows(rows)
        out = capsys.readouterr().out
        assert "8x8" in out and "good" in out and "threads" in out

    def test_it_keeps_the_reasons_a_backend_is_missing(self, capsys):
        """The saved file records them, so a redraw should not drop them."""
        report_rows([{"backend": "ptwt", "case": "8x8", "status": "not installed", "median_ms": None}])
        out = capsys.readouterr().out
        assert "not measured" in out and "not installed" in out


class TestPlot:
    """The figure."""

    def test_it_draws_a_file(self, tmp_path, capsys):
        """A plot is only useful if it reaches the disk."""
        pytest.importorskip("matplotlib")
        path = tmp_path / "out.png"
        rows = [
            {"backend": "a", "case": "8x8", "status": "ok", "median_ms": 1.0},
            {"backend": "b", "case": "8x8", "status": "ok", "median_ms": 2.0},
        ]
        plot(rows, str(path))
        assert path.stat().st_size > 0

    def test_it_groups_by_the_given_key(self, tmp_path):
        """A benchmark chooses what shares an axis tick."""
        pytest.importorskip("matplotlib")
        path = tmp_path / "out.png"
        rows = [{"backend": "a", "case": "8x8", "side": 8, "status": "ok", "median_ms": 1.0}]
        plot(rows, str(path), lambda row: f"side {row['side']}")
        assert path.stat().st_size > 0

    def test_the_bars_use_the_documentation_palette(self, tmp_path):
        """A figure dropped into the docs should look like it belongs there."""
        pytest.importorskip("matplotlib")
        import matplotlib.colors
        import matplotlib.pyplot as plt

        rows = [
            {"backend": name, "case": "8x8", "status": "ok", "median_ms": 1.0}
            for name in ("a", "b", "c")
        ]
        plot(rows, str(tmp_path / "out.png"))
        drawn = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in plt.gcf().axes[0].patches]
        plt.close("all")
        assert drawn == list(DEFAULT_PALETTE[:3])

    def test_more_backends_than_colours_wrap_round(self, tmp_path):
        """A sweep is not limited to the five stops the gradient happens to have."""
        pytest.importorskip("matplotlib")
        import matplotlib.colors
        import matplotlib.pyplot as plt

        rows = [
            {"backend": f"b{i}", "case": "8x8", "status": "ok", "median_ms": 1.0}
            for i in range(7)
        ]
        plot(rows, str(tmp_path / "out.png"))
        drawn = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in plt.gcf().axes[0].patches]
        plt.close("all")
        assert drawn == [*DEFAULT_PALETTE, DEFAULT_PALETTE[0], DEFAULT_PALETTE[1]]

    def test_nothing_measured_draws_nothing(self, tmp_path, capsys):
        """A sweep with no timings says so rather than writing an empty figure."""
        path = tmp_path / "out.png"
        plot([{"backend": "a", "case": "8x8", "status": "MISMATCH"}], str(path))
        assert "nothing to plot" in capsys.readouterr().out
        assert not path.exists()


class TestPinAllocator:
    """Putting every process in the same allocator steady state."""

    def test_it_returns_when_the_thresholds_are_already_set(self, monkeypatch):
        """The re-exec happens once; a second call must not loop."""
        for key, value in ALLOCATOR_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr("os.execve", lambda *a: pytest.fail("re-exec'd twice"))
        pin_allocator()

    def test_it_re_execs_with_the_thresholds_added(self, monkeypatch):
        """The thresholds are read at startup, so only an exec can set them."""
        monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)
        monkeypatch.setattr(sys, "argv", ["bench.py", "--device", "cpu"])
        seen = {}
        monkeypatch.setattr("os.execve", lambda path, argv, env: seen.update(argv=argv, env=env))
        pin_allocator()
        assert seen["argv"][1:] == ["bench.py", "--device", "cpu"]
        assert all(seen["env"][k] == v for k, v in ALLOCATOR_ENV.items())


class TestRepeated:
    """Measuring in fresh processes, because a timer only sees its own."""

    def test_it_merges_the_runs_and_reports_the_spread(self, tmp_path, monkeypatch, capsys):
        """The median of the medians, and how far apart the processes were."""
        script = tmp_path / "worker.py"
        script.write_text(WORKER)
        monkeypatch.setattr(sys, "argv", ["worker.py", "--scale", "2.0", "--repeats", "3"])
        out = tmp_path / "merged.json"
        repeated(str(script), namespace(repeats=3, json=str(out)))

        printed = capsys.readouterr().out
        assert "3 runs, each in its own process" in printed
        assert "8x8" in printed and "threads" in printed  # the Compare table
        assert "spread across runs: median 1.00x, 0 of 1 above 1.15x" in printed

        rows = json.loads(out.read_text())
        timed = [row for row in rows if row["status"] == "ok"]
        assert timed[0]["median_ms"] == 2.0
        assert timed[0]["spread"] == 1.0
        assert timed[0]["runs"] == [2.0, 2.0, 2.0]

    def test_a_backend_that_could_not_run_is_still_listed(self, tmp_path, monkeypatch, capsys):
        """A normal run says why a backend is missing; repeating should not hide it."""
        script = tmp_path / "worker.py"
        script.write_text(WORKER)
        monkeypatch.setattr(sys, "argv", ["worker.py", "--scale", "1.0"])
        out = tmp_path / "merged.json"
        repeated(str(script), namespace(repeats=2, json=str(out)))

        assert "not measured" in capsys.readouterr().out
        statuses = {row["status"] for row in json.loads(out.read_text())}
        assert statuses == {"ok", "not installed"}

    def test_an_unstable_case_is_named(self, tmp_path, monkeypatch, capsys):
        """The spread is the whole point of repeating, so a wide one has to show."""
        script = tmp_path / "worker.py"
        script.write_text(WORKER.replace(
            'float(sys.argv[sys.argv.index("--scale") + 1])',
            'float(sys.argv[sys.argv.index("--scale") + 1]) * (1 + 0.5 * int(os.environ.get("N", "0")))',
        ).replace("import json, sys", "import json, os, sys"))
        calls = {"n": 0}
        real = repeated.__globals__["subprocess"].run

        def counted(command, **kwargs):
            calls["n"] += 1
            return real(command, env={**os.environ, "N": str(calls["n"] - 1)}, **kwargs)

        monkeypatch.setattr(repeated.__globals__["subprocess"], "run", counted)
        monkeypatch.setattr(sys, "argv", ["worker.py", "--scale", "1.0"])
        repeated(str(script), namespace(repeats=2))
        printed = capsys.readouterr().out
        assert "1 of 1 above 1.15x" in printed and "1.50x" in printed

    def test_a_failed_run_stops_the_sweep(self, tmp_path, monkeypatch):
        """A missing run would silently narrow the spread the others report."""
        script = tmp_path / "worker.py"
        script.write_text("import sys; sys.exit(1)")
        monkeypatch.setattr(sys, "argv", ["worker.py"])
        with pytest.raises(SystemExit, match="run 1 failed"):
            repeated(str(script), namespace(repeats=2))

    def test_the_child_does_not_inherit_the_output_flags(self, tmp_path, monkeypatch, capsys):
        """Only the parent writes the merged file; the children write temporaries."""
        script = tmp_path / "worker.py"
        script.write_text(WORKER + '\nassert "--csv" not in sys.argv, "csv leaked"\n')
        monkeypatch.setattr(
            sys, "argv", ["worker.py", "--scale", "1.0", "--csv", "x.csv", "--plot", "x.png"]
        )
        repeated(str(script), namespace(repeats=1))
        assert "1 runs" in capsys.readouterr().out
