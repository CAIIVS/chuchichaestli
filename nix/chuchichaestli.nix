# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Package derivation for chuchichaestli.
#
# Meant to be called via pkgs.callPackage, which auto-injects
# python313Packages from whichever pkgs instantiation is used.
# Override torchPackage to select the GPU backend:
#
#   pkgs.callPackage ./nix/chuchichaestli.nix {}                            # CPU
#   pkgs.callPackage ./nix/chuchichaestli.nix {                             # ROCm
#     torchPackage = pkgs.python313Packages.torchWithRocm;
#   }
#   pkgsCuda.callPackage ./nix/chuchichaestli.nix {                         # CUDA
#     torchPackage = pkgsCuda.python313Packages.torchWithCuda;
#   }
{
  lib,
  python313Packages,
  torchPackage ? python313Packages.torch,
  torchvisionPackage ? python313Packages.torchvision.override { torch = torchPackage; },
  src ? ../.
}:
python313Packages.buildPythonPackage {
  pname = "chuchichaestli";
  version = "0.2.18";
  pyproject = true;
  inherit src;

  build-system = with python313Packages; [ hatchling ];

  dependencies = with python313Packages; [
    numpy
    h5py
    safetensors
    torchPackage
    torchvisionPackage
  ];

  doCheck = false;

  meta = with lib; {
    description = "Where you find all the state-of-the-art cooking utensils (salt, pepper, gradient descent... the usual).";
    homepage = "https://github.com/CAIIVS/chuchichaestli";
    license = licenses.gpl3Plus;
  };
}
