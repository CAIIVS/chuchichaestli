# Chuchichaestli

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
  - test install on various Python versions (by default 3.10-3.12)
  - ruff linting checks (stop build if error occurs)
  - run unit tests with pytest
  - upload test results (for prosperity)
* `build-package.yml`
  - build package for Python version 3.x
  - upload package dist artifacts (by name)
* `github-release.yml`
  - download package dist artifact (by name)
  - sign package dist with Sigstore
  - create and upload GitHub release
* `phdenzel/hatch-bump@v*`
  - use hatch to increment a version
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
  - runs `test-install-python-version`, `build-package`, `publish-to-pypi`, `github-release`
* `on_dispatch.yml`
  - triggers on dispatch (e.g. by running `gh workflow run on-dispatch.yml -f type=minor`)
  - runs `version-bump` (increments chosen type of version on main)
