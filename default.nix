# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Traditional Nix package install.
#
# Usage:
#   nix-build                          # CPU (default torch)
#   nix-build --arg torchPackage \
#     '(import <nixpkgs> {}).python313Packages.torchWithRocm'   # ROCm
#   nix-build --arg torchPackage \
#     '(let p = import <nixpkgs> { config.cudaSupport = true; }; in p.python313Packages.torchWithCuda)'
{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:
pkgs.callPackage ./nix/chuchichaestli.nix { src = ./.; }
