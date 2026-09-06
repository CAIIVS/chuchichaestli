# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the optional native extensions under `csrc`.

The extensions are accelerators, never requirements, so this hook is written
to skip rather than fail. It builds them in place, which serves a source or
editable install; the wheel carries the sources instead, put there by
`shipped()`, and `chuchichaestli/_jit.py` compiles those on first use. The
probes are read from that module, so both routes compile alike.

    python hatch_build.py             # build them in place, into the source tree
    C3LI_SKIP_EXTENSIONS=1 uv build   # or leave them out entirely

Environment:
    C3LI_SKIP_EXTENSIONS: skip every extension when set to `1`.
    C3LI_SKIP_<NAME>: skip one extension, e.g. `C3LI_SKIP_DWT`.
    C3LI_FORCE_GPU: compile the GPU sources even without a compiler probe.
"""

import os
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # running this file directly, without the build backend
    BuildHookInterface = object


ROOT = Path(__file__).parent
CSRC = ROOT / "csrc"
SHARED = CSRC / "common"

# `csrc/<name>` builds `chuchichaestli.<package>._<name>_kernels`
PACKAGE_OF = {"dwt": "chuchichaestli.dwt", "ode": "chuchichaestli.ode"}

_TOOLCHAIN = None


def toolchain():
    """The compile probes, shared with the just-in-time build.

    Loaded by path rather than imported: the package it sits in pulls in torch,
    which a packaging environment may not carry, while the module itself needs
    nothing but the standard library. Loaded on demand, so that a tree without
    it still packages as a pure-Python wheel by way of the report in `build`.
    """
    global _TOOLCHAIN
    if _TOOLCHAIN is None:
        import importlib.util

        path = ROOT / "src" / "chuchichaestli" / "_jit.py"
        spec = importlib.util.spec_from_file_location("chuchichaestli_jit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TOOLCHAIN = module
    return _TOOLCHAIN


def discover(build_gpu: bool, use_rocm: bool) -> list:
    """Collect one extension per `csrc` directory that carries bindings.

    Args:
        build_gpu: Whether to compile the GPU sources too.
        use_rocm: Whether the GPU sources go through hipify first.
    """
    from torch.utils.cpp_extension import IS_WINDOWS, CppExtension, CUDAExtension

    jit = toolchain()
    omp_compile, omp_link = jit.openmp_flags()
    # `-g1` keeps line tables a profiler needs but saves size compared to `-g`
    optimise = ["/O2"] if IS_WINDOWS else ["-O3", "-g1"]

    extensions = []
    for directory in sorted(p for p in CSRC.iterdir() if p.is_dir()):
        name = directory.name
        if not (directory / "bindings.cpp").exists():
            continue
        if os.environ.get(f"C3LI_SKIP_{name.upper()}", "0") == "1":
            continue

        with_gpu = build_gpu and any(directory.glob("*.cu"))
        sources = jit.sources(directory, with_gpu)
        if with_gpu and use_rocm:
            from torch.utils.hipify.hipify_python import hipify

            hipify(
                project_directory=str(ROOT),
                output_directory=str(ROOT),
                includes=[f"{directory}/*"],
                extensions=(".cu", ".cuh", ".cpp"),
                show_detailed=False,
                is_pytorch_extension=True,
            )

            # hipify rewrites a source only where there's something to translate
            def hipified(source: str) -> str:
                path = Path(source)
                stem = path.stem + (".hip" if path.suffix == ".cu" else "_hip.cpp")
                translated = path.with_name(stem)
                return str(translated if translated.exists() else path)

            sources = [hipified(p) for p in sources]

        flags = [*optimise, *omp_compile, *jit.vector_flags()]
        gpu_flags = [*optimise]
        if with_gpu:
            flags.append("-DC3LI_WITH_GPU")
            gpu_flags.append("-DC3LI_WITH_GPU")
        factory = CUDAExtension if with_gpu else CppExtension
        extensions.append(
            factory(
                name=f"{PACKAGE_OF.get(name, f'chuchichaestli.{name}')}._{name}_kernels",
                sources=sources,
                include_dirs=[str(SHARED), *jit.pybind_include()],
                extra_compile_args={"cxx": flags, "nvcc": gpu_flags},
                extra_link_args=omp_link,
            )
        )
    return extensions


def build() -> None:
    """Compile the extensions in place, reporting rather than raising."""
    if os.environ.get("C3LI_SKIP_EXTENSIONS", "0") == "1":
        print("chuchichaestli: extensions skipped by C3LI_SKIP_EXTENSIONS")
        return
    try:
        _build()
    except Exception as exc:  # noqa: BLE001 - a build failure must not be fatal
        print(f"chuchichaestli: extensions skipped ({exc}); pure-Python install")


def _build() -> None:
    """Probe the toolchain and run the extension build."""
    try:
        from setuptools import setup
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        print("chuchichaestli: torch is not importable here; extensions skipped")
        return

    build_gpu, use_rocm = toolchain().gpu_target()
    extensions = discover(build_gpu, use_rocm)
    if not extensions:
        print("chuchichaestli: no extensions to build")
        return

    kind = "GPU" if build_gpu else "CPU"
    names = ", ".join(e.name.rsplit(".", 1)[-1] for e in extensions)
    print(f"chuchichaestli: building {kind} extensions: {names}")
    setup(
        name="chuchichaestli_extensions",
        ext_modules=extensions,
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext", "--inplace"],
    )


def shipped() -> dict[str, str]:
    """The extension sources a wheel carries, keyed by their path inside it.

    `force-include` bypasses the file selection the rest of the wheel goes
    through, so the exclusions are made here instead: an object built in place
    is machine-specific, and hipify's output is already translated.
    """
    carried = {}
    for directory in sorted(p for p in CSRC.iterdir() if p.is_dir()):
        if directory != SHARED and not (directory / "bindings.cpp").exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix in {".cpp", ".cu", ".cuh", ".h"} and not path.stem.endswith(
                "_hip"
            ):
                carried[str(path)] = f"chuchichaestli/csrc/{directory.name}/{path.name}"
    return carried


class CustomBuildHook(BuildHookInterface):
    """Compile the `csrc` extensions in place, and ship their sources."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build the extensions, and put their sources into the wheel.

        Args:
            version: Build version, unused.
            build_data: Build data the sources are added to.
        """
        if self.target_name != "wheel":
            return
        if CSRC.is_dir():
            build_data.setdefault("force_include", {}).update(shipped())
        build()


if __name__ == "__main__":
    build()
