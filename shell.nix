# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Traditional dev shells.
#
# Usage:
#   nix-shell                          # CPU (default)
#   nix-shell --argstr variant rocm    # ROCm
#   nix-shell --argstr variant cuda    # CUDA
{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
  pkgsCuda ? import <nixpkgs> { config.cudaSupport = true; config.allowUnfree = true; },
  variant ? "cpu",
}:
let

  # Merged GPU runtime directories
  rocmMerged = with pkgs; symlinkJoin {
    name = "rocm-merged";
    paths = with rocmPackages; [
      clr rocthrust rocprim hipsparse
      hipblas hipblaslt hipblas-common hipsolver
    ];
  };

  cudaMerged = with pkgsCuda; symlinkJoin {
    name = "cuda-merged";
    paths = with cudaPackages; [
      cuda_cudart cuda_nvcc libcublas libcusparse libcusolver
    ];
  };

  
  # The default torch also drives Apple's GPU on macOS, which has no ROCm or CUDA
  isDarwin = pkgs.stdenv.hostPlatform.isDarwin;
  defaultLabel = if isDarwin then "CPU and MPS" else "CPU-only";
  gpuHint = pkgs.lib.optionalString (!isDarwin) "  (--argstr variant rocm/cuda for GPU support)";

  # Common system build inputs (shared across all variants)
  commonBuildInputs = with pkgs; [
    stdenv.cc.cc.lib
    glib
    zlib
    libGL
    fontconfig
    libx11
    libxkbcommon
    freetype
    dbus
    libsForQt5.wrapQtAppsHook
  ];

  
  # The package itself, editable from the checkout the shell was entered in
  editable = p: p.mkPythonEditablePackage {
    pname = "chuchichaestli";
    inherit (pkgs.callPackage ./nix/chuchichaestli.nix { src = ./.; }) version;
    root = "$REPO_ROOT/src";
  };

  # Python environments (runtime + all optional deps)
  makeCpuPyEnv = dist:
    dist.withPackages (p: with p; [
      numpy
      h5py
      safetensors
      torch
      torchvision
      scikit-learn
      pytest
      pytest-cov
      ruff
    ] ++ (import ./nix/extras.nix { python3Packages = p; qt = true; }).all
      ++ [ (editable p) ]);

  makeRocmPyEnv = dist:
    dist.withPackages (p: with p; [
      numpy
      h5py
      safetensors
      torchWithRocm
      (torchvision.override { torch = torchWithRocm; })
      scikit-learn
      pytest
      pytest-cov
      ruff
    ] ++ (import ./nix/extras.nix { python3Packages = p; qt = true; }).all
      ++ [ (editable p) ]);

  makeCudaPyEnv = dist:
    dist.withPackages (p: with p; [
      numpy
      h5py
      safetensors
      torchWithCuda
      (torchvision.override { torch = torchWithCuda; })
      scikit-learn
      pytest
      pytest-cov
      ruff
    ] ++ (import ./nix/extras.nix { python3Packages = p; qt = true; }).all
      ++ [ (editable p) ]);

  
  # Shell definitions
  cpuShell = pkgs.mkShell {
    packages = [ pkgs.uv (makeCpuPyEnv pkgs.python313) ];
    buildInputs = commonBuildInputs;
    shellHook = ''
      export UV_PYTHON_PREFERENCE="only-system"
      export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
      export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc.lib zlib libGL
      ])}:$LD_LIBRARY_PATH"
      echo "${defaultLabel} environment loaded${gpuHint}"
    '';
  };

  rocmShell = pkgs.mkShell {
    packages = [ pkgs.uv rocmMerged (makeRocmPyEnv pkgs.python313) ];
    buildInputs = commonBuildInputs;
    shellHook = ''
      export UV_PYTHON_PREFERENCE="only-system"
      export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
      export ROCM_PATH="${rocmMerged}"
      export HIP_PATH="${rocmMerged}"
      export CPLUS_INCLUDE_PATH="${rocmMerged}/include:''${CPLUS_INCLUDE_PATH:-}"
      export C_INCLUDE_PATH="${rocmMerged}/include:''${C_INCLUDE_PATH:-}"
      export LD_LIBRARY_PATH="${rocmMerged}/lib:${pkgs.lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc.lib zlib libGL
      ])}:$LD_LIBRARY_PATH"
      echo "ROCm environment loaded"
    '';
  };

  cudaShell = pkgsCuda.mkShell {
    packages = [ pkgsCuda.uv cudaMerged (makeCudaPyEnv pkgsCuda.python313) ];
    buildInputs = with pkgsCuda; [
      stdenv.cc.cc.lib
      glib
      zlib
      libGL
      fontconfig
      libx11
      libxkbcommon
      freetype
      dbus
      libsForQt5.wrapQtAppsHook
    ];
    shellHook = ''
      export UV_PYTHON_PREFERENCE="only-system"
      export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
      export CUDA_PATH="${cudaMerged}"
      export CUDA_HOME="${cudaMerged}"
      export CPLUS_INCLUDE_PATH="${cudaMerged}/include:''${CPLUS_INCLUDE_PATH:-}"
      export C_INCLUDE_PATH="${cudaMerged}/include:''${C_INCLUDE_PATH:-}"
      export LD_LIBRARY_PATH="${cudaMerged}/lib:${pkgsCuda.lib.makeLibraryPath (with pkgsCuda; [
        stdenv.cc.cc.lib zlib libGL
      ])}:$LD_LIBRARY_PATH"
      echo "CUDA environment loaded"
    '';
  };

  shells = {
    cpu  = cpuShell;
    rocm = rocmShell;
    cuda = cudaShell;
  };
in
  shells.${variant}
