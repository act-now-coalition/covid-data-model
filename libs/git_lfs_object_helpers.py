"""
Helpers for accessing Git LFS objects from commit history.

Provides a surface for easy loading of previous versions of Git LFS data.
"""

from typing import Optional
import datetime
import subprocess
import re
import pathlib
import structlog
import git
from libs.datasets import dataset_utils

_logger = structlog.getLogger(__name__)


def find_commit(
    repo: git.Repo,
    path: pathlib.Path,
    before: str = None,
    previous_commit: bool = False,
    commit_sha: str = None,
) -> Optional[git.Commit]:
    """Find a commit for a given path matching query options.

    If no query options are specified, returns latest commit.

    Args:
        repo: Git Repo.
        path: file path.
        before: Optional ISO format date string.  If set, will return the first commit
            before this date.
        previous_commit: Returns the previous commit for the file.
        commit_sha: Commit SHA.

    Returns: Commit if matching commit found.
    """
    if commit_sha:
        return repo.commit(commit_sha)
    if previous_commit:
        commit_iterator = repo.iter_commits(paths=path)
        _ = next(commit_iterator)
        return next(commit_iterator)

    if before:
        before = datetime.datetime.fromisoformat(before)

    for commit in repo.iter_commits(paths=path):
        if not before:
            return commit

        if commit.committed_datetime >= before:
            continue

        return commit


def read_data_for_commit(repo: git.Repo, path: pathlib.Path, commit: git.Commit) -> bytes:
    """Read data for a commit, fetching LFS data if necessary.

    Args:
        repo: Git Repo
        path: File path.
        commit: Commit object to read lfs data for.

    Returns: Bytes for file at commit.
    """
    # blob expects relative path, converts to relative from the repo root if path is absolute.
    if path.absolute() == path:
        root = pathlib.Path(repo.common_dir).parent
        path = path.relative_to(root)

    blob = commit.tree / str(path)
    pointer_text = blob.data_stream.read()
    return subprocess.check_output(["git", "lfs", "smudge"], input=pointer_text)


# TODO(chris): Streamline options for choosing the correct commit. Instead of passing specific
# before, previous_commit, or commit options, pass a more generic filter that uses `git rev-list`
# to return the selected commit.
def get_data_for_path(
    path: pathlib.Path,
    repo: git.Repo = None,
    before: str = None,
    previous_commit=False,
    commit: str = None,
) -> bytes:
    """Loads LFS data for a given path.

    Args:
        repo: Git Repo.
        path: file path.
        before: Optional ISO format date string.  If set, will return the first commit
            before this date.
        previous_commit: Returns the previous commit for the file.
        commit_sha: Commit SHA.

    """
    repo = repo or git.Repo(dataset_utils.REPO_ROOT)
    commit = find_commit(
        repo, path, before=before, previous_commit=previous_commit, commit_sha=commit
    )
    return read_data_for_commit(repo, path, commit)
