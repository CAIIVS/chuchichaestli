# chuchichaestli

`chuchichaestli` is a collection of model architectures and other useful bits of code in use at the Intelligent Vision Systems group at the University of Applied Sciences Zurich (ZHAW).


## Installation

Tagged releases are available as PyPI packages. To install the latest package, run:

```bash
pip install chuchichaestli
```

For the bleeding-edge package directly from the git main, clone the repository and run the following command in the root directory of the repository:

```bash
pip install -e .
```

Alternatively, you can install the package directly from GitHub:

```bash
pip install git+https://github.com/CAIIVS/chuchichaestli.git
```

### Native kernels

`chuchichaestli` provides optional, optimized, custom CPU and GPU kernels.
Without them everything still works, just a little slower.

They are compiled on first use, against the installed torch, which takes a C++
compiler plus:

```bash
pip install chuchichaestli[jit]
```

That yields the CPU kernels. The custom GPU kernels additionally need `nvcc` or
`hipcc`; without one the build quietly settles for CPU.

Run `chuchichaestli-build-kernels` to compile up front instead, or set
`C3LI_JIT_KERNELS=0` to skip them entirely.

#### Benchmarks

`benches/dwt_impl.py` times the kernels against the pure-torch path,
PyWavelets, ptwt and pytorch_wavelets, checking each against a reference before
timing it. Pin the run, take one backend per process, and repeat: some cases
settle into one of two speeds for a whole process at a time, so a single run
looks steady and still disagrees with the next one by more than the difference
being measured. `--repeats` reports the median across processes and marks the
rows whose spread makes them untrustworthy.

```bash
taskset -c 0-15 uv run --with pywavelets --with ptwt \
  python benches/dwt_impl.py --device cuda --backends c3li-kernel \
  --min-run-time 3.0 --repeats 5 --json bench.json
```

On an AMD gfx1151 (ROCm 7.2, float32, 3 levels, forward, batch `2x1`) the
kernels are the fastest of the four in all 54 cases, in microseconds:

| case | c3li-torch | c3li-kernel | ptwt | pytorch_wavelets |
| --- | --- | --- | --- | --- |
| 1d db8 symmetric 4096 | 847 | **70** | 499 | -- |
| 2d db4 symmetric 256x256 | 1474 | **108** | 518 | 1208 |
| 3d haar zero 64x64x64 | 930 | **163** | 639 | -- |
| 3d db8 zero 64x64x64 | 2222 | **318** | 46857 | -- |

On CPU with one thread they are fastest in all 54 too, by a median of 5.8x over
the pure-torch path and 3.3x over PyWavelets. On 16 cores they lead in 49 of
54: the five they do not are all `zero` mode, where torch pads with a constant
rather than gathering indices, and where its convolution parallelizes better
than these kernels do.

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
