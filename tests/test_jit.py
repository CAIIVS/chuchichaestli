# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for compiling the native extensions where they will run.

A wheel ships the extension sources rather than a compiled object, so what
these assert is the packaging contract -- that the sources travel, and that
nothing else does -- and the decisions taken before a compiler is ever run.
The compile itself belongs to `test_dwt_kernels.py`, which exercises whatever
it produced.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils import cpp_extension

from chuchichaestli import _jit


ROOT = Path(__file__).resolve().parents[1]


def build_hook():
    """Load the build hook the way a build loads it.

    It sits outside the package on purpose, so a plain import would not find
    it.
    """
    spec = importlib.util.spec_from_file_location("hook", ROOT / "hatch_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSources:
    """Tests for finding the sources to compile."""

    def test_the_sources_are_found(self):
        """Test that the sources to compile can be located at all."""
        root = _jit.source_root()
        assert root is not None
        assert (root / "common" / "boundary.h").exists()
        assert (root / "dwt" / "bindings.cpp").exists()

    def test_the_byproducts_of_a_hipify_pass_are_left_out(self, tmp_path):
        """Test that an already translated source is not translated twice."""
        for name in ("a.cpp", "a_hip.cpp", "b.cu", "b.hip"):
            (tmp_path / name).touch()
        assert _jit.sources(tmp_path, build_gpu=False) == [str(tmp_path / "a.cpp")]
        assert _jit.sources(tmp_path, build_gpu=True) == [
            str(tmp_path / "a.cpp"),
            str(tmp_path / "b.cu"),
        ]


    def test_the_sources_are_compiled_from_a_copy(self, tmp_path):
        """Test that a build never writes into the package it was installed as."""
        # hipify writes its translation beside the source it read, and an
        # installed package is not always writable, let alone willing
        root = _jit.source_root()
        where = tmp_path / "build"
        staged = _jit.staged(root / "dwt", where)
        assert staged == where / "src" / "dwt"
        assert not staged.is_relative_to(root)
        assert (staged / "bindings.cpp").read_bytes() == (
            root / "dwt" / "bindings.cpp"
        ).read_bytes()
        assert (where / "src" / "common" / "boundary.h").exists()


class TestWheel:
    """Tests for what a wheel carries, which is what makes a build possible."""

    def test_the_wheel_carries_what_a_build_would_compile(self):
        """Test that every source a build asks for is one the wheel carries."""
        carried = {Path(p).resolve() for p in build_hook().shipped()}
        root = _jit.source_root()
        compiled = {
            Path(p).resolve() for p in _jit.sources(root / "dwt", build_gpu=True)
        }
        assert compiled and compiled <= carried

    def test_the_wheel_carries_the_shared_headers(self):
        """Test that the headers the sources include travel with them."""
        carried = build_hook().shipped()
        assert str(_jit.source_root() / "common" / "boundary.h") in carried

    def test_no_build_byproduct_travels_with_the_wheel(self):
        """Test that nothing machine-specific is shipped to another machine."""
        # `force-include` bypasses the exclusions the rest of the wheel goes
        # through, so this is the only thing keeping them out
        for source, destination in build_hook().shipped().items():
            assert Path(source).suffix not in {".so", ".o", ".hip"}
            assert not Path(source).stem.endswith("_hip")
            assert destination.startswith("chuchichaestli/csrc/")

    def test_the_sources_land_where_the_loader_looks_for_them(self):
        """Test that the layout in the wheel is the one the loader resolves."""
        # the loader resolves them against the installed package, so the
        # destinations have to spell out that same layout
        inside = _jit.PACKAGE.name + "/csrc"
        assert all(d.startswith(inside) for d in build_hook().shipped().values())


class TestDecision:
    """Tests for deciding whether to compile at all."""

    @pytest.mark.parametrize(
        "value,expected",
        [("1", True), ("on", True), ("0", False), ("no", False), ("", None)],
    )
    def test_a_switch_reads_its_three_states(self, monkeypatch, value, expected):
        """Test that on, off and unset are told apart."""
        monkeypatch.setenv("C3LI_SWITCH", value)
        assert _jit.setting("C3LI_SWITCH") is expected

    def test_the_switch_turns_the_build_off(self, monkeypatch):
        """Test that a compile can be refused outright."""
        monkeypatch.setenv("C3LI_JIT_KERNELS", "0")
        assert "C3LI_JIT_KERNELS" in _jit.declined("dwt", forced=False)
        assert _jit.load("dwt") is None

    def test_an_explicit_request_overrides_the_switch(self, monkeypatch):
        """Test that asking for a build in person beats the standing setting."""
        monkeypatch.setenv("C3LI_JIT_KERNELS", "0")
        assert _jit.declined("dwt", forced=True) is None

    def test_an_extension_that_is_not_installed_is_declined(self):
        """Test that a name with no sources behind it is turned down."""
        assert "not installed" in _jit.declined("nowhere", forced=True)

    def test_a_missing_toolchain_declines(self, monkeypatch):
        """Test that a machine which cannot compile is not asked to."""
        monkeypatch.delenv("C3LI_JIT_KERNELS", raising=False)
        monkeypatch.setattr(_jit, "missing", lambda: (["ninja"], ["a C++ compiler"]))
        reason = _jit.declined("dwt", forced=False)
        assert "ninja" in reason and "C++ compiler" in reason

    def test_a_missing_setuptools_declines(self, monkeypatch):
        """Test that the module torch builds through is reached, not assumed."""
        # `torch.utils.cpp_extension` imports setuptools as it loads, so
        # probing through it would raise before it could report the shortfall
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] == "setuptools":
                    raise ImportError("setuptools is not installed")
                return None

        monkeypatch.delenv("C3LI_JIT_KERNELS", raising=False)
        monkeypatch.setattr(sys, "meta_path", [Block(), *sys.meta_path])
        stale = [
            n
            for n in sys.modules
            if n.partition(".")[0] == "setuptools"
            or n.startswith("torch.utils.cpp_extension")
        ]
        for name in stale:
            monkeypatch.delitem(sys.modules, name)
        assert _jit.missing() == (["setuptools"], [])
        assert "setuptools" in _jit.declined("dwt", forced=False)
        assert _jit.load("dwt") is None

    def test_the_switch_asks_for_a_build_without_probing(self, monkeypatch):
        """Test that the probes can be waived for a toolchain they miss."""
        monkeypatch.setenv("C3LI_JIT_KERNELS", "1")
        monkeypatch.setattr(_jit, "missing", lambda: (["ninja"], []))
        assert _jit.declined("dwt", forced=False) is None


class TestHint:
    """Tests for saying so when an install away from having the kernels."""

    @pytest.fixture(autouse=True)
    def _unhinted(self, monkeypatch):
        """Each test starts having said nothing yet."""
        monkeypatch.setattr(_jit, "_HINTED", False)
        monkeypatch.delenv("C3LI_JIT_KERNELS", raising=False)

    def test_a_missing_ninja_is_worth_saying_once(self, monkeypatch, capsys):
        """Test that a shortfall pip can fix is reported, and only once."""
        monkeypatch.setattr(_jit, "missing", lambda: (["ninja"], []))
        assert _jit.load("dwt") is None
        assert _jit.load("dwt") is None
        said = capsys.readouterr().err
        assert said.count("chuchichaestli[jit]") == 1

    def test_a_missing_compiler_passes_in_silence(self, monkeypatch, capsys):
        """Test that nothing is said about what an install cannot supply."""
        monkeypatch.setattr(_jit, "missing", lambda: ([], ["a C++ compiler"]))
        assert _jit.load("dwt") is None
        assert capsys.readouterr().err == ""

    def test_an_opt_out_is_not_argued_with(self, monkeypatch, capsys):
        """Test that switching the build off is taken at face value."""
        monkeypatch.setenv("C3LI_JIT_KERNELS", "0")
        monkeypatch.setattr(_jit, "missing", lambda: (["ninja"], []))
        assert _jit.load("dwt") is None
        assert capsys.readouterr().err == ""


class TestCache:
    """Tests for where a compiled object is kept."""

    def test_the_path_names_the_torch_it_was_built_against(self):
        """Test that an object is not reused across a torch upgrade."""
        # an extension built against one release does not load into the next,
        # and torch's own cache path does not distinguish them
        where = str(_jit.build_directory("_dwt_kernels"))
        assert torch.__version__.replace("+", "_") in where
        assert where.endswith("_dwt_kernels")

    def test_the_path_follows_the_torch_environment_variable(
        self, monkeypatch, tmp_path
    ):
        """Test that torch's own cache location is honoured."""
        monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
        assert _jit.build_directory("_dwt_kernels").is_relative_to(tmp_path)


class TestFlags:
    """Tests for the compiler flags a build is given."""

    def test_the_hook_and_the_runtime_probe_the_same_way(self):
        """Test that both routes into a build compile with the same flags."""
        # a source install and a just-in-time build have to agree, or an
        # object compiled by one is wrong for the other
        # the hook executes the module afresh, so the functions are distinct
        # objects; what matters is that they were read from the same file
        shared = build_hook().toolchain()
        for probe in ("vector_flags", "pybind_include"):
            assert getattr(shared, probe).__code__.co_filename == _jit.__file__

    def test_a_vendored_pybind11_is_not_shadowed(self, monkeypatch, tmp_path):
        """Test that torch's own pybind11 headers win over a standalone copy."""
        # a `-I` is searched ahead of torch's `-isystem`, so naming another
        # pybind11 would cross an ABI boundary the extension has to hold
        (tmp_path / "torch" / "include" / "pybind11").mkdir(parents=True)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "torch" / "__init__.py"))
        assert _jit.pybind_include() == []

    def test_a_torch_without_pybind11_gets_the_path_spelled_out(
        self, monkeypatch, tmp_path
    ):
        """Test that a torch carrying no headers is given somewhere to look."""
        pybind11 = pytest.importorskip("pybind11")
        (tmp_path / "torch" / "include").mkdir(parents=True)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "torch" / "__init__.py"))
        assert _jit.pybind_include() == [pybind11.get_include()]

    def test_the_vector_flags_follow_what_torch_selected(self):
        """Test that the capability macro matches the path torch itself takes."""
        capability = torch.backends.cpu.get_cpu_capability()
        flags = _jit.vector_flags()
        if capability in ("AVX512", "AVX2"):
            assert f"-DCPU_CAPABILITY_{capability}" in flags
        else:
            assert flags == []

    def test_msvc_gets_its_own_spelling_of_the_vector_flags(self, monkeypatch):
        """Test that Windows is not handed flags its compiler rejects."""
        monkeypatch.setattr(cpp_extension, "IS_WINDOWS", True)
        monkeypatch.setattr(
            torch.backends.cpu, "get_cpu_capability", lambda: "AVX2"
        )
        assert _jit.vector_flags() == ["/arch:AVX2", "-DCPU_CAPABILITY_AVX2"]

    def test_openmp_follows_the_backend_torch_was_built_on(self, monkeypatch):
        """Test that the flag is passed only where `at::parallel_for` needs it."""
        # on a native thread pool it is dead weight, and Apple clang has no
        # libomp to link against
        info = torch.__config__.parallel_info
        monkeypatch.setattr(
            torch.__config__, "parallel_info", lambda: "ATen parallel backend: native"
        )
        assert _jit.openmp_flags() == ([], [])
        monkeypatch.setattr(torch.__config__, "parallel_info", info)
        assert _jit.openmp_flags() == (["-fopenmp"], ["-fopenmp"])

    def test_a_cuda_build_needs_architectures_it_can_name(self, monkeypatch):
        """Test that a CUDA build with no card in sight is not attempted."""
        # torch reads them off the visible cards, so a CUDA build with none in
        # sight cannot produce them and must not be asked to
        monkeypatch.delenv("C3LI_FORCE_GPU", raising=False)
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setattr(torch.version, "cuda", "12.4")
        monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/usr/local/cuda")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert _jit.gpu_target() == (False, False)
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
        assert _jit.gpu_target() == (True, False)
