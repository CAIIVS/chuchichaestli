# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the native extensions under `csrc` into custom kernels.

A wheel ships the sources for custom kernels and the first call compiles them
against the installed torch. Custom kernels are optional: short of a toolchain,
or with `C3LI_JIT_KERNELS=0`, the build is declined and falls back on the
pure-torch path.

Environment:
  - `C3LI_JIT_KERNELS`: `0` never to compile, `1` to compile without probing.
  - `C3LI_FORCE_GPU`: compile the GPU sources even without a compiler probe.
  - `PYTORCH_ROCM_ARCH`: architectures for which to emit ROCm code.
  - `TORCH_EXTENSIONS_DIR`: where the compiled objects are cached.

Usage:
```sh
chuchichaestli-build-kernels
```
"""

import os
import re
import shutil
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent

_HINTED = False

_SWITCH = dict.fromkeys(("1", "true", "yes", "on"), True)
_SWITCH.update(dict.fromkeys(("0", "false", "no", "off"), False))


__all__ = [
    "build_directory",
    "declined",
    "missing",
    "openmp_flags",
    "load",
    "pybind_include",
    "source_root",
    "sources",
    "vector_flags",
]


def setting(name: str) -> bool | None:
    """Read a tri-state switch from the environment.

    Args:
        name: Variable to read.

    Returns:
        What it was set to, or `None` when it was left alone.
    """
    return _SWITCH.get(os.environ.get(name, "").strip().lower())


def vector_flags() -> list[str]:
    """Compiler flags for the widest vector unit torch selected.

    Absent the capability macro, `at::vec` picks a scalar path slower than
    plain scalar code. MSVC spells the same thing its own way and rejects the
    GNU flags outright.
    """
    import torch
    from torch.utils.cpp_extension import IS_WINDOWS

    capability = torch.backends.cpu.get_cpu_capability()
    if capability == "AVX512":
        if IS_WINDOWS:
            return ["/arch:AVX512", "-DCPU_CAPABILITY_AVX512"]
        return ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mfma",
                "-DCPU_CAPABILITY_AVX512"]
    if capability == "AVX2":
        if IS_WINDOWS:
            return ["/arch:AVX2", "-DCPU_CAPABILITY_AVX2"]
        return ["-mavx2", "-mfma", "-DCPU_CAPABILITY_AVX2"]
    return []


def openmp_flags() -> tuple[list[str], list[str]]:
    """Compile and link flags for OpenMP, where torch is built on it.

    `at::parallel_for` inlines an OpenMP pragma only for that backend; on a
    native thread pool the flag is dead weight, and on Apple clang a link
    error.

    Returns:
        The compile flags and the link flags.
    """
    import torch
    from torch.utils.cpp_extension import IS_WINDOWS

    if "OpenMP" not in torch.__config__.parallel_info():
        return [], []
    if IS_WINDOWS:
        return ["/openmp"], []
    return ["-fopenmp"], ["-fopenmp"]


def pybind_include() -> list[str]:
    """Locate pybind11, for a torch that does not vendor its own.

    A `-I` is searched ahead of torch's `-isystem`, so naming a standalone copy
    where torch has one would shadow it across a pybind11 ABI boundary.
    """
    import torch

    if (Path(torch.__file__).parent / "include" / "pybind11").is_dir():
        return []
    try:
        import pybind11
    except ImportError:
        return []
    return [pybind11.get_include()]


def source_root() -> Path | None:
    """Where the sources live: inside the package, or at the repository root."""
    for candidate in (PACKAGE / "csrc", PACKAGE.parents[1] / "csrc"):
        if (candidate / "common").is_dir():
            return candidate
    return None


def sources(directory: Path, build_gpu: bool) -> list[str]:
    """The sources of one extension, less what hipify already translated.

    Args:
        directory: Extension directory under `csrc`.
        build_gpu: Whether the GPU sources are wanted too.
    """
    files = [p for p in sorted(directory.glob("*.cpp")) if not p.stem.endswith("_hip")]
    if build_gpu:
        files += sorted(directory.glob("*.cu"))
    return [str(p) for p in files]


def staged(directory: Path, where: Path) -> Path:
    """Copy the sources into the cache, out of the installed package.

    hipify writes beside the source it reads, and `site-packages` is not always
    writable. Its earlier output is left behind, being already translated.

    Args:
        directory: Extension directory under `csrc`.
        where: Directory the compiled object is cached in.

    Returns:
        The copy of `directory` to compile from.
    """
    root = where / "src"
    translated = shutil.ignore_patterns("*_hip.*", "*.hip")
    for original in (directory.parent / "common", directory):
        shutil.copytree(original, root / original.name, ignore=translated,
                        dirs_exist_ok=True)
    return root / directory.name


def missing() -> tuple[list[str], list[str]]:
    """What a build is short of, split by whether pip can supply it.

    `torch.utils.cpp_extension` imports setuptools as it loads, so failing to
    reach it is the setuptools check, and has to come before anything else
    touches that module. What is checked here is only what decides whether to
    try: torch verifies ninja and the compiler's ABI again as it builds.

    Returns:
        What `pip` could install, and what it could not.
    """
    try:
        from torch.utils.cpp_extension import (
            check_compiler_ok_for_platform,
            get_cxx_compiler,
            is_ninja_available,
        )
    except ImportError:
        return ["setuptools"], []
    installable = [] if is_ninja_available() else ["ninja"]
    named = get_cxx_compiler()
    usable = check_compiler_ok_for_platform(named)
    return installable, [] if usable else [f"a C++ compiler ({named})"]


def gpu_target() -> tuple[bool, bool]:
    """Whether to compile the GPU sources, and whether hipify runs first.

    A CUDA build reads its architectures off the visible cards, so one with no
    device to look at needs `TORCH_CUDA_ARCH_LIST` instead.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    use_rocm = torch.version.hip is not None
    if setting("C3LI_FORCE_GPU"):
        return True, use_rocm
    if use_rocm:
        return ROCM_HOME is not None, True
    if torch.version.cuda is None or CUDA_HOME is None:
        return False, False
    named = bool(os.environ.get("TORCH_CUDA_ARCH_LIST"))
    return named or torch.cuda.device_count() > 0, False


def build_directory(name: str) -> Path:
    """Where the compiled object is cached, keyed by the torch it was built for.

    torch's own cache path omits the release, and an object does not load into
    the next one.

    Args:
        name: Module name of the extension.
    """
    import torch

    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        from torch.utils.cpp_extension import get_default_build_root

        root = get_default_build_root()
    if torch.version.hip is not None:
        accelerator = f"rocm{torch.version.hip}"
    elif torch.version.cuda is not None:
        accelerator = f"cu{torch.version.cuda}"
    else:
        accelerator = "cpu"
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    tag = f"{tag}-{accelerator}-{torch.__version__}"
    return Path(root) / "chuchichaestli" / re.sub(r"[^\w.-]", "_", tag) / name


def declined(name: str, forced: bool) -> str | None:
    """Why the extension will not be compiled, if it will not be.

    Args:
        name: Directory name under `csrc`.
        forced: Whether the caller asked for the build explicitly.
    """
    if not forced and setting("C3LI_JIT_KERNELS") is False:
        return "C3LI_JIT_KERNELS is off"
    root = source_root()
    if root is None or not (root / name / "bindings.cpp").exists():
        return f"the {name} sources are not installed"
    if not (forced or setting("C3LI_JIT_KERNELS") or not any(missing())):
        return "no " + ", ".join(sum(missing(), []))
    return None


def load(name: str, *, forced: bool = False, verbose: bool = False):
    """Compile one `csrc` extension and import it, or give up quietly.

    Concurrent callers are serialised by the lock torch keeps in the build
    directory.

    Args:
        name: Directory name under `csrc`, e.g. `dwt`.
        forced: Compile even where the probes would decline, and say why if
            that fails.
        verbose: Report the compiler's own output.

    Returns:
        The imported module, or `None` if it could not be built.
    """
    global _HINTED

    reason = declined(name, forced)
    if reason is not None:
        if forced or verbose:
            print(f"chuchichaestli: {name} kernels skipped ({reason})", file=sys.stderr)
        elif (
            not _HINTED
            and missing()[0]
            and source_root() is not None
            and setting("C3LI_JIT_KERNELS") is None
        ):
            _HINTED = True
            print(
                f"chuchichaestli: the {name} kernels were not built ({reason});"
                " install `chuchichaestli[jit]` for them, or set"
                " C3LI_JIT_KERNELS=0 to stop looking",
                file=sys.stderr,
            )
        return None
    try:
        return _load(name, source_root() / name, verbose)
    except Exception as exc:  # noqa: BLE001 - an accelerator may never be fatal
        print(f"chuchichaestli: {name} kernels unavailable ({exc})", file=sys.stderr)
        return None


def _load(name: str, directory: Path, verbose: bool):
    """Run the build, retrying without the GPU sources if those fail.

    Args:
        name: Directory name under `csrc`.
        directory: That directory.
        verbose: Report the compiler's own output.
    """
    from torch.utils.cpp_extension import LIB_EXT

    module = f"_{name}_kernels"
    where = build_directory(module)
    where.mkdir(parents=True, exist_ok=True)
    if not any(where.glob(f"*{LIB_EXT}")):
        print(
            f"chuchichaestli: compiling the {name} kernels into {where}; this"
            " happens once (C3LI_JIT_KERNELS=0 skips it)",
            file=sys.stderr,
        )
    build_gpu, _ = gpu_target()
    if not build_gpu:
        return _compile(module, directory, where, False, verbose)
    try:
        return _compile(module, directory, where, True, verbose)
    except Exception as exc:  # noqa: BLE001 - the CPU kernels are still worth having
        print(
            f"chuchichaestli: the {name} GPU sources did not compile ({exc});"
            " building the CPU kernels alone",
            file=sys.stderr,
        )
    return _compile(module, directory, where, False, verbose)


def _compile(
    module: str, directory: Path, where: Path, build_gpu: bool, verbose: bool
):
    """Hand one extension to torch to compile.

    Args:
        module: Module name to build.
        directory: Extension directory under `csrc`.
        where: Directory the compiled object is cached in.
        build_gpu: Whether to compile the GPU sources too.
        verbose: Report the compiler's own output.
    """
    from torch.utils.cpp_extension import IS_WINDOWS
    from torch.utils.cpp_extension import load as compile_extension

    directory = staged(directory, where)
    # torch hipifies everything on `extra_include_paths`
    vendored = [f"-I{path}" for path in pybind_include()]
    optimise = ["/O2"] if IS_WINDOWS else ["-O3"]
    omp_compile, omp_link = openmp_flags()
    cflags = [*optimise, *omp_compile, *vector_flags(), *vendored]
    gpu_flags = [*optimise, *vendored]
    if build_gpu:
        cflags.append("-DC3LI_WITH_GPU")
        gpu_flags.append("-DC3LI_WITH_GPU")
    return compile_extension(
        name=module,
        sources=sources(directory, build_gpu),
        extra_cflags=cflags,
        extra_cuda_cflags=gpu_flags,
        extra_ldflags=omp_link,
        extra_include_paths=[str(directory.parent / "common")],
        build_directory=str(where),
        with_cuda=build_gpu,
        verbose=verbose,
    )


def main() -> int:
    """Compile every installed extension now, rather than on first use."""
    root = source_root()
    if root is None:
        print("chuchichaestli: no extension sources are installed", file=sys.stderr)
        return 1
    failed = 0
    for directory in sorted(p for p in root.iterdir() if (p / "bindings.cpp").exists()):
        if load(directory.name, forced=True, verbose=True) is None:
            failed += 1
        else:
            print(f"chuchichaestli: the {directory.name} kernels are ready")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
