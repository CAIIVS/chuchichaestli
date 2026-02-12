import os
import subprocess
import shutil
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def check_nvcc() -> bool:
    """Check whether system has access to a NVIDIA CUDA compiler."""
    has_nvcc = False
    try:
        subprocess.check_output(["nvcc", "--version"])
        has_nvcc = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return has_nvcc


def check_hipcc() -> bool:
    """Check whether system has access to a ROCm HIP compiler."""
    has_hipcc = False
    try:
        subprocess.check_output(["hipcc", "--version"])
        has_hipcc = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return has_hipcc


def hipify_module(source_dir: str | Path, file_patterns: list[str] = None):
    """Convert CUDA files to HIP files using PyTorch's hipify_python."""
    try:
        from torch.utils.hipify import hipify_python
    except ImportError:
        print("Warning: hipify_python not available, skipping HIP file generation")
        return []

    # Convert to absolute path
    abs_source_dir = os.path.abspath(source_dir)
    print(f"Hipifying CUDA files in {abs_source_dir}...")

    # Run hipify on the directory
    results = hipify_python.hipify(
        project_directory=abs_source_dir,
        output_directory=abs_source_dir,
        includes=file_patterns if file_patterns else ["*"],
        show_detailed=True,
        is_pytorch_extension=True,
        extensions=(".cu", ".cpp"),  # Only process .cu and .cpp files
    )

    # Collect generated .hip files
    hip_files = []
    for filename, result in results.items():
        if hasattr(result, "hip_file_path") and result.hip_file_path:
            hip_files.append(result.hip_file_path)
            print(f"Generated: {result.hip_file_path}")
        elif filename.endswith(".cu"):
            # Assume .hip file was created with same name
            hip_file = filename.replace(".cu", ".hip")
            if os.path.exists(hip_file):
                hip_files.append(hip_file)
                print(f"Generated: {hip_file}")

    return hip_files


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # check for compilers
        has_nvcc = check_nvcc()
        has_hipcc = check_hipcc()

        # check for `torch` and determine backend
        has_torch = False
        use_rocm = False
        try:
            import torch

            if torch.cuda.is_available() or torch.version.cuda or torch.version.hip:
                has_torch = True
                # Check if this is a ROCm build of PyTorch
                if torch.version.hip:
                    use_rocm = True
                    print("ROCm/HIP PyTorch detected")
                elif torch.version.cuda:
                    print("CUDA PyTorch detected")
            else:
                print("`torch` is CPU-only. Skipping GPU extensions.")
        except ImportError:
            print("`torch` not found. Skipping GPU extensions.")

        # add override
        force_cuda_extensions = os.environ.get("FORCE_CUDA_EXTENSIONS", "0") == "1"
        # skip_voxelize = os.environ.get("C3LI_SKIP_VOXELIZE", "0") == "1"
        skip_ode = os.environ.get("C3LI_SKIP_ODE", "0") == "1"

        # check if torch is available (required for both CPU and GPU builds)
        if not has_torch and not force_cuda_extensions:
            print("`torch` not found. Skipping extensions.")
            return

        # determine build mode: GPU (CUDA/HIP) or CPU-only
        has_gpu_compiler = has_nvcc or (use_rocm and has_hipcc)
        build_gpu = force_cuda_extensions or (has_torch and has_gpu_compiler)

        # proceed with build
        if build_gpu:
            if use_rocm:
                print("ROCm environment detected: Building extensions with HIP...")
                print(
                    "Note: Ensure CUDA kernels are HIP-compatible or use hipify to convert them."
                )
            else:
                print(
                    "CUDA environment detected: Building extensions with GPU support..."
                )
            from torch.utils.cpp_extension import CUDAExtension as ExtensionClass
        else:
            print("No GPU compiler found. Building CPU-only extensions...")
            from torch.utils.cpp_extension import CPPExtension as ExtensionClass

        from torch.utils.cpp_extension import BuildExtension
        from setuptools import setup

        include_dirs = [os.path.abspath("csrc/common")]
        ext_modules = []

        # Compiler flags based on backend
        if build_gpu:
            if use_rocm:
                # ROCm/HIP compiler flags
                compile_args = {
                    "cxx": ["-O3", "-fPIC", "-DUSE_CUDA"],
                    "nvcc": [
                        "-O3",
                        "-DUSE_ROCM",
                        "-DUSE_CUDA",
                        "--offload-arch=gfx900",
                        "--offload-arch=gfx906",
                        "--offload-arch=gfx908",
                        "--offload-arch=gfx90a",
                        "--offload-arch=gfx1030",
                        "--offload-arch=gfx1100",
                        "--offload-arch=gfx1151",
                        "--offload-arch=gfx1200",
                        "--offload-arch=gfx1201",
                    ],
                }
            else:
                # CUDA compiler flags
                compile_args = {
                    "cxx": ["-O3", "-DUSE_CUDA"],
                    "nvcc": ["-O3", "-DUSE_CUDA"],
                }
        else:
            # CPU-only compiler flags
            compile_args = {"cxx": ["-O3"]}

        # # voxelize extension
        # if not skip_voxelize and 0:
        #     voxelize_sources = ["csrc/voxelize/bindings.cpp"]
        #     if build_gpu:
        #         voxelize_sources.append("csrc/voxelize/kernels.cu")
        #     ext_modules.append(
        #         ExtensionClass(
        #             name="chuchichaestli.data.voxelize",
        #             sources=voxelize_sources,
        #             include_dirs=include_dirs,
        #             extra_compile_args=compile_args
        #         )
        #     )

        # ode extension
        if not skip_ode:
            if use_rocm:
                # Generate HIP files from CUDA sources in csrc/ode directory
                print("Generating HIP files from CUDA sources...")
                ode_sources = hipify_module("csrc/ode")
                print(ode_sources)

                # Use generated HIP files
                # Note: .cpp files become _hip.cpp, .cu files become .hip
                ode_sources = [
                    "csrc/ode/bindings_hip.cpp",
                    "csrc/ode/lode_kernel.hip",
                    "csrc/ode/euler_kernel.hip",
                    "csrc/ode/rk4_kernel.hip",
                    # "csrc/ode/imex_euler_kernel.hip",
                    # "csrc/ode/dopri45_kernel.hip",
                ]
            else:
                ode_sources = ["csrc/ode/bindings.cpp"]
                # Use CUDA versions
                if build_gpu:
                    ode_sources.extend(
                        [
                            "csrc/ode/lode_kernel.cu",
                            "csrc/ode/euler_kernel.cu",
                            "csrc/ode/rk4_kernel.cu",
                            # "csrc/ode/imex_euler_kernel.cu",
                            # "csrc/ode/dopri45_kernel.cu",
                        ]
                    )
            ext_modules.append(
                ExtensionClass(
                    name="chuchichaestli.ode._ode_kernels",
                    sources=ode_sources,
                    include_dirs=include_dirs,
                    extra_compile_args=compile_args,
                )
            )

        # run the build
        if ext_modules:
            setup_args = {
                "name": "chuchichaestli_extensions",
                "ext_modules": ext_modules,
                "cmdclass": {"build_ext": BuildExtension},
                "script_args": ["build_ext", "--inplace"],
            }
            try:
                setup(**setup_args)
                if build_gpu:
                    print("GPU extensions built successfully.")
                else:
                    print("CPU-only extensions built successfully.")
            except Exception as e:
                print(f"Extension build failed: {e}")
                print("Falling back to pure Python install.")
