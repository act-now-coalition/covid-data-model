import click
from contextlib import contextmanager
from datetime import datetime
import git
import json
import logging
import os
import pytz
from typing import Optional

from .dataset_utils import LOCAL_PUBLIC_DATA_PATH


_logger = logging.getLogger(__name__)


class DataVersion(object):
    '''
    Encapsulates some state about the source data with the
    intention of using this information to make results
    reproducible or comparable against new methods acting on
    consistent data snapshots.
    '''
    def __init__(self, git_hash: str, is_dirty: bool):
        self.git_hash = git_hash
        self.is_dirty = is_dirty
        self.now = datetime.utcnow().replace(tzinfo=pytz.utc)

    def write_file(self, data_type: str, output_dir: str):
        filename = os.path.join(output_dir, f'{data_type}.version.json')
        with open(filename, 'w') as f:
            json.dump({
                'when': str(self.now),
                'gitHash': self.git_hash,
                'dirty': self.is_dirty
            }, f)


@contextmanager
def _repo_at_hash(repo: git.Repo, git_hash: str):
    # HEAD is a symbolic reference, grab what it points to
    previous_head = repo.head.ref
    # Jump to a detached head referencing the commit passed in
    repo.head.set_reference(git_hash)
    repo.head.reset(index=True, working_tree=True)
    # this translates to calling `git lfs fetch` directly on the repo
    # See: https://gitpython.readthedocs.io/en/stable/tutorial.html#using-git-directly
    repo.git.lfs('fetch')
    yield
    # Reset to whatever was previously checked out
    previous_head.checkout()

@contextmanager
def public_data_hash(git_hash: Optional[str]):
    '''
    If given a git hash, attempts to set the local covid-data-public
    repository to the given hash. Yields the git hash so that it can
    be recorded along with the data generated.
    '''
    if git_hash is not None:
        repo = git.Repo(LOCAL_PUBLIC_DATA_PATH)
        if repo.is_dirty():
            raise RuntimeError('Cannot set covid-data-public repo hash, working tree is dirty')
        _logger.info(f'Using git hash {git_hash}')
        with _repo_at_hash(repo, git_hash):
            yield git_hash
    else:
        yield git_hash

@contextmanager
def data_version(git_hash: Optional[str]):
    # Handle legacy datasets
    os.environ['COVID_DATA_PUBLIC'] = str(LOCAL_PUBLIC_DATA_PATH)
    repo = git.Repo(str(LOCAL_PUBLIC_DATA_PATH))
    is_dirty = repo.is_dirty()
    if git_hash:
        if is_dirty:
            raise RuntimeError('Cannot set covid-data-public repo hash, working tree is dirty')
        with _repo_at_hash(repo, git_hash):
            logging.info(f'Using covid-data-public at version {git_hash}')
            yield DataVersion(git_hash, is_dirty)
    else:
        git_hash = repo.head.ref.commit.hexsha
        logging.info(f'Using covid-data-public at version {"*" if is_dirty else ""}{git_hash}')
        yield DataVersion(git_hash, is_dirty)


def with_git_version_click_option(func):
    """Adds an additional git-hash option and loads the repo at the specified hash."""

    @click.option(
        '--git-hash',
        type=str,
        help='''
        | Git hash of the commit in covid-data-public to use.
        | If provided, covid-data-public must have no pending changes.
        | If omitted, the repository will be used as-is'''
    )
    def run_in_context(git_hash, **kwargs):
        with data_version(git_hash) as version:
            return func(version=version, **kwargs)

    return run_in_context
