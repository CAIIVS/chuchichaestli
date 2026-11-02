# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for reading a repository's commit, dirtiness and diff."""

import sys
from pathlib import Path

import pytest

from chuchichaestli.utils.git import GitState, git_state

SHA, OLD = "a" * 40, "b" * 40


def repository(root: Path, files: dict[str, str] | None = None):
    """Make a repository with GitPython, with one commit holding the given files.

    Args:
        root: Directory the repository is made in.
        files: File names and contents to commit; one small file by default.
    """
    git = pytest.importorskip("git")
    author = git.Actor("Test", "test@example.com")
    files = files or {"a.txt": "one\n"}
    repo = git.Repo.init(root)
    for name, text in files.items():
        (root / name).write_text(text)
    repo.index.add(list(files))
    repo.index.commit("init", author=author, committer=author)
    return repo


@pytest.fixture
def without_gitpython(monkeypatch):
    """Make GitPython unimportable, as when it is not installed."""
    monkeypatch.setitem(sys.modules, "git", None)


def files_of(root: Path, head: str = "ref: refs/heads/main") -> Path:
    """Lay out the files git keeps for a repository, without git.

    Args:
        root: Directory the repository is made in.
        head: What `HEAD` holds.
    """
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n")
    return git_dir


def test_a_clean_repository_reports_its_commit_alone(tmp_path):
    """Nothing changed, so there is no diff to store."""
    repo = repository(tmp_path)
    assert git_state(tmp_path) == GitState(repo.head.commit.hexsha, False, None)


def test_an_unstaged_change_makes_it_dirty_and_shows_in_the_diff(tmp_path):
    """The code that ran differs from the commit; the patch says how."""
    repository(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    state = git_state(tmp_path)
    assert state.dirty
    assert "+two" in state.diff


def test_a_staged_change_counts_as_well(tmp_path):
    """Staged is not committed: the diff is against the commit, not the index."""
    repo = repository(tmp_path)
    (tmp_path / "a.txt").write_text("one\nthree\n")
    repo.index.add(["a.txt"])
    state = git_state(tmp_path)
    assert state.dirty
    assert "+three" in state.diff


def test_untracked_files_do_not_make_it_dirty(tmp_path):
    """Outputs and data often sit untracked, and git describe agrees."""
    repository(tmp_path)
    (tmp_path / "results.csv").write_text("x\n")
    assert git_state(tmp_path).dirty is False


def test_the_repository_is_found_from_a_subdirectory(tmp_path):
    """A run usually starts somewhere inside the repository, not at its root."""
    repo = repository(tmp_path)
    inner = tmp_path / "experiments" / "a"
    inner.mkdir(parents=True)
    assert git_state(inner).commit == repo.head.commit.hexsha


def test_outside_a_repository_there_is_no_state(tmp_path):
    """Nothing to read is not an error."""
    assert git_state(tmp_path) is None


def test_a_branch_without_a_commit_has_no_state(tmp_path):
    """A fresh repository names a branch that holds nothing yet."""
    pytest.importorskip("git").Repo.init(tmp_path)
    assert git_state(tmp_path) is None


def test_without_gitpython_the_commit_is_read_from_the_files(
    tmp_path, without_gitpython
):
    """The commit still identifies the code; whether it is dirty is unknown."""
    git_dir = files_of(tmp_path)
    (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n")
    assert git_state(tmp_path) == GitState(SHA, dirty=None, diff=None)


def test_without_gitpython_a_packed_ref_is_read_and_a_loose_one_wins(
    tmp_path, without_gitpython
):
    """`git gc` packs refs; a branch moved since then has a newer loose file."""
    git_dir = files_of(tmp_path)
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled\n{OLD} refs/heads/main\n^{'c' * 40}\n"
    )
    assert git_state(tmp_path).commit == OLD
    (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n")
    assert git_state(tmp_path).commit == SHA


def test_without_gitpython_a_detached_head_holds_the_commit(
    tmp_path, without_gitpython
):
    """Checking out a commit rather than a branch."""
    files_of(tmp_path, head=SHA)
    assert git_state(tmp_path).commit == SHA


def test_without_gitpython_a_worktree_reads_the_shared_refs(
    tmp_path, without_gitpython
):
    """A worktree's `.git` is a file; its refs live in the main repository."""
    git_dir = files_of(tmp_path / "main")
    (git_dir / "refs" / "heads" / "feature").write_text(SHA + "\n")
    linked = git_dir / "worktrees" / "feature"
    linked.mkdir(parents=True)
    (linked / "HEAD").write_text("ref: refs/heads/feature\n")
    (linked / "commondir").write_text("../..\n")
    tree = tmp_path / "feature"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {linked}\n")
    assert git_state(tree).commit == SHA


def test_without_gitpython_the_repository_is_found_from_a_subdirectory(
    tmp_path, without_gitpython
):
    """The fallback looks upward as git does."""
    git_dir = files_of(tmp_path)
    (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n")
    inner = tmp_path / "experiments" / "a"
    inner.mkdir(parents=True)
    assert git_state(inner).commit == SHA


def test_without_gitpython_a_branch_without_a_commit_has_no_state(
    tmp_path, without_gitpython
):
    """The branch file does not exist yet, so there is nothing to report."""
    files_of(tmp_path)
    assert git_state(tmp_path) is None


def test_without_gitpython_outside_a_repository_there_is_no_state(
    tmp_path, without_gitpython
):
    """Nothing to read is not an error, with or without GitPython."""
    assert git_state(tmp_path) is None
