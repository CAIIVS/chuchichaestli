{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        makeEnvsForPython = dist: dist.withPackages (p: with p; [
          numpy
          h5py
          torch
          torchvision
          psutil
        ]);
        pyDists = with pkgs; [
          python312
        ];
        pyEnvs = map makeEnvsForPython pyDists;
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              uv
            ] ++ pyDists ++ pyEnvs;

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
            '';      
          };
      }
    );
}
