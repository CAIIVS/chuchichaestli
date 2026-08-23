# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pkgsCuda = import nixpkgs {
          inherit system;
          config.cudaSupport = true;
          config.allowUnfree = true;
        };

        
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

        
        # Common system build inputs
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

        
        # Python environments (runtime + all optional deps)
        makeCpuPyEnv = dist:
          dist.withPackages (p: with p; [
            numpy
            h5py
            safetensors
            (matplotlib.override { enableQt = true; })
            torch
            torchvision
            scikit-learn
            pytest
            pytest-cov
            ruff
          ]);

        makeRocmPyEnv = dist:
          dist.withPackages (p: with p; [
            numpy
            h5py
            safetensors
            (matplotlib.override { enableQt = true; })
            torchWithRocm
            (torchvision.override { torch = torchWithRocm; })
            scikit-learn
            pytest
            pytest-cov
            ruff
          ]);

        makeCudaPyEnv = dist:
          dist.withPackages (p: with p; [
            numpy
            h5py
            safetensors
            (matplotlib.override { enableQt = true; })
            torchWithCuda
            (torchvision.override { torch = torchWithCuda; })
            scikit-learn
            pytest
            pytest-cov
            ruff
          ]);
      in {
        
        # Packages built from self (git-tracked source tree)
        packages = {
          default = pkgs.callPackage ./nix/chuchichaestli.nix { src = self; };
          cpu     = pkgs.callPackage ./nix/chuchichaestli.nix { src = self; };
          rocm    = pkgs.callPackage ./nix/chuchichaestli.nix {
            src = self;
            torchPackage = pkgs.python313Packages.torchWithRocm;
          };
          cuda    = pkgsCuda.callPackage ./nix/chuchichaestli.nix {
            src = self;
            torchPackage = pkgsCuda.python313Packages.torchWithCuda;
          };
        };

        
        # Dev shells
        devShells = {
          default = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.python313 (makeCpuPyEnv pkgs.python313) ];
            buildInputs = commonBuildInputs;
            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
                stdenv.cc.cc.lib zlib libGL
              ])}:$LD_LIBRARY_PATH"
              echo "CPU-only environment loaded  (nix develop .#rocm or .#cuda for GPU support)"
            '';
          };

          cpu = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.python313 (makeCpuPyEnv pkgs.python313) ];
            buildInputs = commonBuildInputs;
            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
                stdenv.cc.cc.lib zlib libGL
              ])}:$LD_LIBRARY_PATH"
              echo "CPU-only environment loaded"
            '';
          };

          rocm = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.python313 rocmMerged (makeRocmPyEnv pkgs.python313) ];
            buildInputs = commonBuildInputs;
            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system"
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

          cuda = pkgsCuda.mkShell {
            packages = [ pkgsCuda.uv pkgsCuda.python313 cudaMerged (makeCudaPyEnv pkgsCuda.python313) ];
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
        };

      }
    ) // {
      # Overlays are system-independent
      overlays.default = import ./nix/overlay.nix;
    };
}
