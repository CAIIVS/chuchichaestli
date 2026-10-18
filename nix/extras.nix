# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Python packages of the extras in pyproject.toml, by extra.
#
#   (import ./nix/extras.nix { python3Packages = p; qt = true; }).all
{
  python3Packages,
  qt ? false,
}:
let
  p = python3Packages;
  matplotlib = if qt then p.matplotlib.override { enableQt = true; } else p.matplotlib;
  replaced = [ "mlflow-skinny" "mlflow-tracing" "matplotlib" ];
  mlflowRest = builtins.filter
    (dep:
      p.hasPythonModule dep
      && !(dep ? interpreter)
      && !(builtins.elem (dep.pname or "") replaced))
    p.mlflow.propagatedBuildInputs;
in
rec {
  viz = [ matplotlib ];
  git = [ p.gitpython ];
  tensorboard = [ p.tensorboard ];
  wandb = [ p.wandb ];
  mlflow = [ p.mlflow-skinny matplotlib ] ++ mlflowRest;
  all = viz ++ git ++ tensorboard ++ wandb ++ mlflow;
}
