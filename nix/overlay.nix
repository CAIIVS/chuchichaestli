# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Nixpkgs overlay exposing chuchichaestli package variants.
#
# Provides:
#   pkgs.chuchichaestli      — CPU-only build (default torch)
#   pkgs.chuchichaestliRocm  — AMD ROCm build
#
# For the CUDA variant, apply this overlay to a pkgsCuda instantiation
# (import nixpkgs { config.cudaSupport = true; }) and call
# pkgsCuda.callPackage ./chuchichaestli.nix { torchPackage = ...; }.
final: prev: {
  chuchichaestli = final.callPackage ./chuchichaestli.nix {};

  chuchichaestliRocm = final.callPackage ./chuchichaestli.nix {
    torchPackage = final.python313Packages.torchWithRocm;
  };
}
