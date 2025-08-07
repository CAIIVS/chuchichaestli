{
  pkgs ? import <nixpkgs> {},
  src ? ./.,
}:
let 
  pythonPackage = pkgs.python312Packages.buildPythonApplication {
    pname = "chuchichaestli";
    version = "0.2.10";
    format = "pyproject";
    build-system = with pkgs.python312Packages; [hatchling];
    propagatedBuildInputs = with pkgs.python312Packages; [
      numpy
      h5py
      torch
      torchvision
      psutil
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
