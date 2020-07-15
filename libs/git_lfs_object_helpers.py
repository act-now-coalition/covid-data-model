"""
Helpers for accessing Git LFS objects from commit history.

Provides a surface for easy loading of previous versions of Git LFS data.
"""

from typing import Optional
import re
import pathlib
import structlog


import git

_logger = structlog.getLogger(__name__)


def _get_lfs_object_path(repo: git.Repo, object_oid: str) -> pathlib.Path:
    lfs_objects = pathlib.Path(repo.git_dir) / "lfs" / "objects"
    path = lfs_objects / object_oid[:2] / object_oid[2:4] / object_oid
    return path


def _get_object_sha_from_lfs_pointer(lfs_data: str) -> str:
    match = re.search("oid sha256:(.*)", lfs_data)
    return match.groups(0)[0]


def find_commit(
    repo: git.Repo, path: pathlib.Path, on_or_before: str = None, previous_commit: bool = False
) -> Optional[git.Commit]:
    """Find a commit for a given path matching query options.

    If no query options are specified, returns latest commit.

    Args:
        repo: Git Repo.
        path: file path.
        on_or_before: Optional ISO format date string.  If set, will return the first commit on or
            before this date.
        previous_commit: Returns the previous commit for the file.

    Returns: Commit if matching commit found.
    """
    print(list(repo.iter_commits(paths=path)))
    if previous_commit:
        commit_iterator = repo.iter_commits(paths=path)
        _ = next(commit_iterator)
        return next(commit_iterator)

    if on_or_before:
        on_or_before = datetime.datetime.fromisoformat(on_or_before)

    for commit in repo.iter_commits(paths=path):
        if not on_or_before:
            return commit

        if commit.committed_datetime > on_or_before:
            continue

        return commit


def read_lfs_data_for_commit(repo: git.Repo, path: str, commit: git.Commit) -> bytes:
    """Read LFS Data for a commit, fetching data if necessary.

    Args:
        repo: Git Repo
        path: File path.
        commit: Commit object to read lfs data for.

    Returns: Bytes for file at commit.
    """
    blob = commit.tree / path
    # TODO(chris): Find a better way to identify a Git LFS pointer file.
    is_pointer = blob.size < 200
    if not is_pointer:
        return blob.data_stream.read()

    _logger.debug("Loading commit data from path at commit", path=path, commit=commit)
    pointer_text = blob.data_stream.read().decode()
    object_sha = _get_object_sha_from_lfs_pointer(pointer_text)
    object_path = _get_lfs_object_path(repo, object_sha)

    if not object_path.exists():
        lfs_fetch_command = f"git lfs fetch origin {commit.hexsha}"
        _logger.info("Fetching missing reference", fetch_command=lfs_fetch_command)
        subprocess.check_output(lfs_fetch_command, shell=True)

    return object_path.read_bytes()


def get_data_for_path(
    repo: git.Repo, path: str, on_or_before: str = None, previous_commit=False
) -> bytes:
    commit = find_commit(repo, path, on_or_before=on_or_before, previous_commit=previous_commit)
    return read_lfs_data_for_commit(repo, path, commit)
