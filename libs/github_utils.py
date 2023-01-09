from typing import Tuple
import shutil
import click
import tempfile
import io
import shutil

import zipfile
import pathlib
import logging

import pydantic
import requests
import git


_logger = logging.getLogger(__name__)

REPO = "act-now-coalition/covid-data-model"

REPO_GIT_URL = f"https://api.github.com/repos/{REPO}"
# GitHub assigned ID for "Build & Publish API artifacts" workflow.
WORKFLOW_ID = "988804"


class GitSummary(pydantic.BaseModel):
    """Summary of a github repo state."""

    sha: str
    branch: str
    is_dirty: bool

    @classmethod
    def from_repo_path(cls, path: pathlib.Path) -> "GitSummary":
        repo = git.Repo(path)

        try:
            branch = str(repo.head.ref)
        except TypeError:
            branch = "detached"

        return cls(sha=repo.head.commit.hexsha, branch=branch, is_dirty=repo.is_dirty())


def _get_artifact_zip_url(run_number: int = None, latest: bool = True) -> Tuple[str, int]:

    workflow_runs_url = f"{REPO_GIT_URL}/actions/workflows/{WORKFLOW_ID}/runs?status=completed"

    runs = requests.get(workflow_runs_url).json()["workflow_runs"]
    if latest:
        # Workflows are returned latest first.
        run = runs[0]
    else:
        # TODO: Only requests the first page of workflows, if run id is not from the
        # 30 most recent runs, it will not be able to be found.
        for run in runs:
            if run["run_number"] == run_number:
                break
        else:
            raise Exception("Could not find workflow with run number {run_number}.")

    run_id = run["id"]

    artifacts_runs_url = f"{REPO_GIT_URL}/actions/runs/{run_id}/artifacts"

    artifacts = requests.get(artifacts_runs_url).json()["artifacts"]
    if not len(artifacts) == 1:
        raise AssertionError(
            "Multiple artifacts detected, missing logic to choose correct artifact."
        )

    artifact = artifacts[0]
    return artifact["archive_download_url"], run["run_number"]


def download_model_artifact(
    github_token: str, output_dir: pathlib.Path, run_number: int = None, prompt_overwrite=True
):
    """Downloads and extracts model output artifact.

    Args:
        github_token: GitHub token with read access.
        output_dir: Output directory to extract artifact to.
        run_number: If specified, will download the specific run number.
        prompt_overwrite: If true, shows a prompt before overwriting exsting output artifact.
            If False, does not overwrite.
    """
    latest = False if run_number else True
    artifact_url, run_number = _get_artifact_zip_url(run_number=run_number, latest=latest)

    _logger.info(f"Downloading artifact from run number {run_number}")
    response = requests.get(
        artifact_url, stream=True, headers={"Authorization": f"token {github_token}"}
    )
    artifact_zip = zipfile.ZipFile(io.BytesIO(response.content))
    _logger.info("Extracting downloaded artifact")

    snapshot_zip = zipfile.ZipFile(artifact_zip.open("api-results.zip"))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        snapshot_zip.extractall(tmp_path)
        data_path = tmp_path / "data"

        for path in data_path.iterdir():
            output_path = output_dir / path.name
            if output_path.exists() and prompt_overwrite:
                if click.confirm(f"Overwrite existing path {output_path}?", default=True):
                    shutil.rmtree(output_path)

            _logger.info(f"Extracted {path.name} to  {str(output_dir)}")
            shutil.move(path, output_dir / path.name)
