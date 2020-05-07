import git
import logging
import os.path
import os

import tempfile

local_public_data_dir = tempfile.TemporaryDirectory()
local_public_data = local_public_data_dir.name

_logger = logging.getLogger(__name__)

PUBLIC_DATA_REPO = 'https://github.com/covid-projections/covid-data-public'


def get_public_data_base_url():
    # COVID_DATA_PUBLIC could be set, to for instance
    # "https://raw.githubusercontent.com/covid-projections/covid-data-public/master"
    # which would not locally copy the data.

    if not os.getenv('COVID_DATA_PUBLIC', False):
        create_local_copy_public_data()
    return os.getenv('COVID_DATA_PUBLIC')


# TODO: support passing a git hash
def _clone_and_hydrate_repo(repo_url: str, target_dir: str):
    repo = git.Repo.clone_from(repo_url, target_dir)
    # this translates to calling `git lfs fetch` directly on the repo
    # See: https://gitpython.readthedocs.io/en/stable/tutorial.html#using-git-directly
    repo.git.lfs('fetch')


def create_local_copy_public_data():
    """
    Creates a local copy of the public data repository. This is done to avoid
    downloading the file again for each intervention type.
    """
    public_data_local_url = f'file://localhost{local_public_data}/'
    _logger.info(f"Creating a Local Copy of {PUBLIC_DATA_REPO} at {public_data_local_url}")
    _clone_and_hydrate_repo(PUBLIC_DATA_REPO, local_public_data)

    os.environ['COVID_DATA_PUBLIC'] = public_data_local_url
