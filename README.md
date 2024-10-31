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

### Creating a new release

On the current version of the main branch, do the follwing.

1. **Create an empty commit:**
   ```bash
   git commit --allow-empty -m "<Your commit message here>"
   ```

2. **Add a tag to the empty commit:**
   ```bash
   git tag -a <tag_name> -m "Tagging empty commit"
   ```
   Replace `<tag_name>` with the new version The `-a` option creates an annotated tag, and the `-m` option allows you to add a message to the tag.

3. **Push the empty commit and tag:**
   ```bash
   git push --tags
   ```

This triggers a GitHub action that creates a PR for you to accept. Accepting the PR updates the version and triggers the release pipeline.
