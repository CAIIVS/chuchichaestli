# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A git repository's state: the commit checked out and the changes since."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from chuchichaestli.utils.registry import require_module

__all__ = ["GitState", "git_state"]


@dataclass(frozen=True)
class GitState:
    """A repository's checked-out commit and its uncommitted changes.

    Args:
        commit: Hash of the commit checked out.
        dirty: Whether tracked files differ from it, staged or not; untracked
            files do not count, as for `git describe --dirty`. `None` when
            unknown, the state having been read without GitPython.
        diff: Those differences as a patch, `git diff HEAD`, when dirty.
    """

    commit: str
    dirty: bool | None = None
    diff: str | None = None


def git_state(path: Path | None = None) -> GitState | None:
    """Return the state of the repository holding a directory, or `None` if unreadable.

    Args:
        path: Directory inside the repository; the working directory if absent.
            Without GitPython, only its commit is read, from its files.
    """
    path = path or Path.cwd()
    try:
        git = require_module("git", "git", "recording the repository's state")
    except ImportError:
        commit = _read_commit(path)
        return None if commit is None else GitState(commit=commit)
    try:
        repo = git.Repo(path, search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return None
    with repo:
        try:
            commit = repo.head.commit.hexsha
            dirty = repo.is_dirty(untracked_files=False)
            diff = repo.git.diff("HEAD") if dirty else None
        except (ValueError, git.GitCommandError, OSError):
            return None
    return GitState(commit=commit, dirty=dirty, diff=diff)


def _git_dir(path: Path) -> Path | None:
    """Return the git directory of the repository holding a directory, if any.

    Args:
        path: Directory to look from, then its parents.
    """
    for directory in (path, *path.parents):
        dot = directory / ".git"
        if dot.is_dir():
            return dot
        if dot.is_file():
            # a worktree or submodule names its git directory in a `.git` file
            content = dot.read_text().strip()
            if content.startswith("gitdir: "):
                return (directory / content.removeprefix("gitdir: ")).resolve()
    return None


def _read_commit(path: Path) -> str | None:
    """Return the commit checked out, read from the repository's files.

    Args:
        path: Directory inside the repository.
    """
    try:
        repo = _git_dir(path)
        if repo is None:
            return None
        head = (repo / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head.removeprefix("ref: ")
        shared = repo / "commondir"
        common = (
            (repo / shared.read_text().strip()).resolve() if shared.is_file() else repo
        )
        for base in (repo, common):
            loose = base / ref
            if loose.is_file():
                return loose.read_text().strip() or None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                commit, _, name = line.partition(" ")
                if name == ref and not line.startswith(("#", "^")):
                    return commit
    except OSError:
        return None
    return None
