{
  pkgs ? import <nixpkgs> {},
  src ? ./.,
}:
let 
  pythonPackage = pkgs.python313Packages.buildPythonApplication {
    pname = "chuchichaestli";
    version = "0.2.14";
    format = "pyproject";
    build-system = with pkgs.python313Packages; [hatchling];
    propagatedBuildInputs = with pkgs.python313Packages; [
      numpy
      h5py
      torch
      torchvision
    ];
    src = src;
    doCheck = false;
    meta = {
      description = "Where you find all the state-of-the-art cooking utensils (salt, pepper, gradient descent... the usual).";
      license = pkgs.lib.licenses.gpl3Plus;
    };
  };
in

pythonPackage
