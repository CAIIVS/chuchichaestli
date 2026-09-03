# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the optional native extensions under `csrc`.

The extensions are an accelerator, never a requirement: every call site falls
back to a pure-torch path, so this hook is written to skip rather than fail.

They are built in place, for a source or editable install, and the wheel target
excludes `*.so`: a wheel carries no platform tag, so shipping a compiled object
in one would hand a machine-specific binary to every other machine. Whether
torch is importable during a packaging build depends on the front end, so the
exclusion rather than the import is what keeps a wheel portable.

    uv sync --no-build-isolation      # build them alongside the install
    C3LI_SKIP_EXTENSIONS=1 uv build   # or leave them out entirely

Environment:
    C3LI_SKIP_EXTENSIONS: skip every extension when set to `1`.
    C3LI_SKIP_<NAME>: skip one extension, e.g. `C3LI_SKIP_DWT`.
    C3LI_FORCE_GPU: compile the GPU sources even without a compiler probe.
"""

import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


ROOT = Path(__file__).parent
CSRC = ROOT / "csrc"
SHARED = CSRC / "common"

# `csrc/<name>` builds `chuchichaestli.<package>._<name>_kernels`
PACKAGE_OF = {"dwt": "chuchichaestli.dwt", "ode": "chuchichaestli.ode"}

ROCM_ARCHS = (
    "gfx900", "gfx906", "gfx908", "gfx90a",
    "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201",
)


def have(program: str) -> bool:
    """Whether a compiler answers on `PATH`.

    Args:
        program: Executable to probe.
    """
    try:
        subprocess.run(
            [program, "--version"], capture_output=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def cpu_features() -> set[str]:
    """Instruction set extensions the building machine advertises.

    The extensions are compiled in place for a source install and the wheel
    target excludes the result, so the object may be tuned to this machine.
    """
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def vector_flags() -> list[str]:
    """Compiler flags enabling the widest vector unit this machine has.

    `at::vec` selects its implementation from the capability macro, and falls
    back to a scalar one that is slower than plain scalar code, so the flags
    are worth probing for.
    """
    features = cpu_features()
    if {"avx512f", "avx512dq", "avx512vl"} <= features:
        return ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mfma",
                "-DCPU_CAPABILITY_AVX512"]
    if "avx2" in features:
        return ["-mavx2", "-mfma", "-DCPU_CAPABILITY_AVX2"]
    return []


def skipped(name: str) -> bool:
    """Whether one extension was switched off.

    Args:
        name: Directory name under `csrc`.
    """
    return os.environ.get(f"C3LI_SKIP_{name.upper()}", "0") == "1"


def discover(build_gpu: bool, use_rocm: bool) -> list:
    """Collect one extension per `csrc` directory that carries bindings.

    Args:
        build_gpu: Whether to compile the GPU sources too.
        use_rocm: Whether the GPU sources go through hipify first.
    """
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    extensions = []
    for directory in sorted(p for p in CSRC.iterdir() if p.is_dir()):
        name = directory.name
        if not (directory / "bindings.cpp").exists() or skipped(name):
            continue
        package = PACKAGE_OF.get(name, f"chuchichaestli.{name}")

        sources = sorted(str(p) for p in directory.glob("*.cpp"))
        flags = ["-O3", *vector_flags()]
        if build_gpu:
            if use_rocm:
                from torch.utils.hipify.hipify_python import hipify

                hipify(
                    project_directory=str(ROOT),
                    output_directory=str(ROOT),
                    includes=[f"{directory}/*"],
                    extensions=(".cu", ".cuh", ".cpp"),
                    show_detailed=False,
                    is_pytorch_extension=True,
                )
                sources = sorted(
                    str(p)
                    for p in directory.iterdir()
                    if p.suffix in (".hip",) or p.name.endswith("_hip.cpp")
                ) or sources
            else:
                sources += sorted(str(p) for p in directory.glob("*.cu"))
            flags.append("-DC3LI_WITH_GPU")

        factory = CUDAExtension if build_gpu else CppExtension
        extra = {"cxx": flags + ["-fopenmp"]}
        if build_gpu:
            gpu_flags = ["-O3", "-DC3LI_WITH_GPU"]
            if use_rocm:
                gpu_flags += [f"--offload-arch={arch}" for arch in ROCM_ARCHS]
            extra["nvcc"] = gpu_flags
        extensions.append(
            factory(
                name=f"{package}._{name}_kernels",
                sources=sources,
                include_dirs=[str(SHARED)],
                extra_compile_args=extra,
                extra_link_args=["-fopenmp"],
            )
        )
    return extensions


class CustomBuildHook(BuildHookInterface):
    """Compile the `csrc` extensions in place, or leave them out."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build the extensions, reporting rather than raising on any failure.

        Args:
            version: Build version, unused.
            build_data: Build data, unused.
        """
        if os.environ.get("C3LI_SKIP_EXTENSIONS", "0") == "1":
            print("chuchichaestli: extensions skipped by C3LI_SKIP_EXTENSIONS")
            return
        try:
            self._build()
        except Exception as exc:  # noqa: BLE001 - a build failure must not be fatal
            print(f"chuchichaestli: extensions skipped ({exc}); pure-Python install")

    def _build(self) -> None:
        """Probe the toolchain and run the extension build."""
        try:
            import torch
        except ImportError:
            print("chuchichaestli: torch is not importable here; extensions skipped")
            return

        use_rocm = torch.version.hip is not None
        forced = os.environ.get("C3LI_FORCE_GPU", "0") == "1"
        compiler = have("hipcc") if use_rocm else have("nvcc")
        build_gpu = forced or (
            compiler and (use_rocm or torch.version.cuda is not None)
        )

        extensions = discover(build_gpu, use_rocm)
        if not extensions:
            print("chuchichaestli: no extensions to build")
            return

        from setuptools import setup
        from torch.utils.cpp_extension import BuildExtension

        kind = "GPU" if build_gpu else "CPU"
        names = ", ".join(e.name.rsplit(".", 1)[-1] for e in extensions)
        print(f"chuchichaestli: building {kind} extensions: {names}")
        setup(
            name="chuchichaestli_extensions",
            ext_modules=extensions,
            cmdclass={"build_ext": BuildExtension},
            script_args=["build_ext", "--inplace"],
        )
