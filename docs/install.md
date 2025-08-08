# Install

Tagged releases are available as PyPI packages. To install the latest
package, run:

```bash
pip install chuchichaestli
```

For the bleeding-edge package directly from a git branch, use
```bash
pip install git+https://github.com/CAIIVS/chuchichaestli.git@<latest-branch>
```

(and replace `<latest-branch>` with the actual branch name) or clone
the repository and run the following command in the root directory of
the repository:

```bash
pip install -e .
```


## uv

`chuchichaestli` is developed using [`uv`](https://docs.astral.sh/uv/)
and thus provides a `uv.lock` file which should make installing the
package easier, faster, and universal. In the project, run

```bash
uv sync [--all-groups]
```

To add `chuchichaestli` to your own project simply useful
```bash
uv add chuchichaestli
```
and it will appear as a dependency in your `pyproject.toml`.


## nix

If reproducibility is of utmost importance, you might want to look
into `nix`. `chuchichaestli` is packaged as nix package (see
`default.nix`). To install it, you can add the following to your 
nix-config and rebuild it

```nix
{pkgs, ...}: let
	remote = builtins.fetchurl {
		url = "https://raw.githubusercontent.com/CAIIVS/chuchichaestli/refs/heads/main/default.nix";
		sha256 = "sha256:0a838l8h2qv4c95zi68r1nr8ndmn8929f53js04g3h15ii3zbskb";
	};
	chuchichaestli = pkgs.callPackage remote {
		src = pkgs.fetchFromGitHub {
		owner = "CAIIVS";
		repo = "chuchichaestli";
		rev = "main";
		sha256 = "sha256:0l5q6j7kav2lsy1pl1izqa8f31q32r7fz47qhim45gjawp838vrw";
	};
in {
	environment.systemPackages = with pkgs; [
		chuchichaestli
	];
}
```
