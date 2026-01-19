import os
import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # check for CUDA compiler: `nvcc`
        has_nvcc = False
        try:
            subprocess.check_output(["nvcc", "--version"])
            has_nvcc = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("`nvcc` not found. Skipping CUDA extensions.")

        # check for `torch`
        has_torch = False
        try:
            import torch
            if torch.cuda.is_available() or torch.version.cuda:
                has_torch = True
            else:
                print("`torch` is CPU-only. Skipping CUDA extensions.")
        except ImportError:
            print("`torch` not found. Skipping CUDA extensions.")

        # add override
        force_cuda_extensions = os.environ.get("FORCE_CUDA_EXTENSIONS", "0") == "1"
        skip_ode = os.environ.get("C3LI_SKIP_ODE", "0") == "1"
        do_build = (has_nvcc and has_torch) or force_cuda_extensions
        if not do_build:
            return

        # proceed with build
        print("CUDA environment detected: Building extensions...")    
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        from setuptools import setup

        include_dirs = [os.path.abspath("csrc/common")]
        ext_modules = []

        # ode extension
        if not skip_ode:
            ext_modules.append(
                CUDAExtension(
                    name="chuchichaestli.ode._cuda_ode_solvers",
                    sources=[
                        "csrc/ode/binding.cpp",
                        "csrc/ode/lode_kernel.cu",
                        "csrc/ode/euler_kernel.cu",
                        "csrc/ode/rk4_kernel.cu",
                    ],
                    include_dirs=include_dirs,
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
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
                print("CUDA extensions busilt successfully.")
            except:
                print("CUDA extension build failed.")
                print("Falling back to pure Python install.")
