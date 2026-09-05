# chuchichaestli

`chuchichaestli` is a collection of model architectures and other
useful bits of code in use at the Intelligent Vision Systems group at
the University of Applied Sciences Zurich (ZHAW).


## Installation

Tagged releases are available as PyPI packages. To install the latest
package, run:

```bash
pip install chuchichaestli
```

For the bleeding-edge package directly from a git branch, clone the
repository and run the following command in the root directory of the
repository:

```bash
pip install -e .
```

Alternatively, you can install the package directly from GitHub:

```bash
pip install git+https://github.com/CAIIVS/chuchichaestli.git
```

Note: we recommend [`uv`](https://docs.astral.sh/uv/) for running
examples in
[`examples/`](https://github.com/CAIIVS/chuchichaestli/tree/main/examples)
or benchmarks in
[`benches/`](https://github.com/CAIIVS/chuchichaestli/tree/main/benches).


### Native kernels

`chuchichaestli` provides optional, optimized, custom CPU and GPU
kernels.  Without them everything still works, just a little slower.

They are compiled on first use, against the installed torch, which
takes a C++ compiler (`gcc` on Linux, `clang` on macOS, MSVC on
Windows) plus:

```bash
pip install chuchichaestli[jit]
```

That yields the CPU kernels. The custom GPU kernels additionally need
`nvcc` or `hipcc`; without one the build quietly settles for CPU.

Run 

```bash
chuchichaestli-build-kernels
```

to compile up front instead, or set `C3LI_JIT_KERNELS=0` to skip them
entirely.


#### Benchmarks

The native kernels are benchmarked against the pure-torch path and against
external packages, on both CPU and GPU, with every backend
checked against a reference before it is timed; see
[Benchmarks](https://github.com/CAIIVS/chuchichaestli/blob/main/docs/benchmarks.md#dwt)
for how to run them and benchmark results.


## Development

Releases and packages are automatically created with various GitHub
action workflows. The general development workflow is as follows

1) checkout to a non-default branch to apply patches, features, etc.
  * each commit to a non-default branch triggers an install test (on latest Python versions) and a package build
2) open a PR
  * at opening and each subsequent commit to that PR, the build is uploaded to TestPyPI and a dev version is incremented (e.g. v1.2.0 -> v1.2.0-dev0)
3) once the PR is reviewed, it can be merged
  * at merge, the micro version is incremented on the main branch and tagged (e.g. v1.2.0-dev7 -> v1.2.1)
  * subsequently, a package is built, published to PyPI, and released on GitHub with the latest version tag
4) every once in a while it is necessary to bump minor or major versions
  * minor and major version can be triggered, by manually dispatching the `on_dispatch.yml` workflow 
  * on the web interface (using the option `minor` or `major`), or
  * on CLI with `gh workflow run on_dispatch.yml -f type=minor` (or `-f type=major`)


### Reusable workflows

In `.github/workflows/` there are several reusable workflows which
provide the basic utility for the triggered jobs:

* `test-install.yml`
  - test install on the supported Python endpoints, 3.10 and 3.13, by default;
    pass `full-matrix: true` to sweep every supported version (the release workflow does)
  - ruff linting checks (stop build if error occurs)
  - run unit tests with pytest and collect coverage (fails below the `coverage-threshold` input)
  - upload test results (for prosperity)
* `build-package.yml`
  - build package with `uv build` on Python 3.13 by default; pass `versions`
    (a JSON array) to build on several, if wheels are ABI-specific
  - upload one dist artifact per Python version, named `<artifact-name>-<version>`
* `github-release.yml`
  - download all dist artifacts matching `<artifact-name>-*` and merge them
  - sign package dist with Sigstore
  - create and upload GitHub release
* `publish-to-pypi.yml`
  - download all dist artifacts matching `<artifact-name>-*` and merge them
  - publish to PyPI, or to TestPyPI when called with `test: true`
* `deploy-docs.yml`
  - build the docs with MkDocs and publish them to GitHub Pages
  - uses the Pages artifact flow (`upload-pages-artifact` + `deploy-pages`), not a `gh-pages` push
  - can be run manually via `gh workflow run deploy-docs.yml`
* `phdenzel/pyverto@v*`
  - use pyverto to increment a version
  - commit and push changes


### Triggered workflows

* `on_branches.yml`
  - triggers on commit to any branch (except `main`) not in a PR (pull request)
  - runs `test-install-python-version` and `build-package`
* `on_pr.yml`
  - triggers on commit to a PR
  - runs `test-install-python-version`, `build-package`, `publish-to-testpypi`, and `version-bump-dev`
* `on_merge.yml`
  - triggers on PR merge
  - runs `version-bump-on-merge` (increments micro version on main)
* `on_push.yml`
  - triggers on push to the main branch upon automatic version change
  - runs `test-install-python-version`, `build-package`, `publish-to-pypi`, `github-release`, `deploy-docs`
* `on_dispatch.yml`
  - triggers on dispatch (e.g. by running `gh workflow run on-dispatch.yml -f type=minor`)
  - runs `version-bump` (increments chosen type of version on main)
