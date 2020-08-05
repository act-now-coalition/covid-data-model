from typing import Type, Tuple
import os
import io
import pathlib
import datetime

import git
import structlog
import pydantic

from libs import git_lfs_object_helpers
from libs.datasets.dataset_base import DatasetBase
from libs.datasets.dataset_utils import DatasetType
from libs.datasets import dataset_utils
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def form_filename(dataset_type: DatasetType) -> str:
    return f"{dataset_type.value}.json"


class DatasetPointer(pydantic.BaseModel):
    """Describes a persisted combined dataset."""

    dataset_type: DatasetType

    path: pathlib.Path

    # Sha of covid-data-public for dataset
    data_git_info: GitSummary

    # Sha of covid-data-model used to create dataset.
    model_git_info: GitSummary

    # When local file was saved.
    updated_at: datetime.datetime

    @property
    def filename(self) -> str:
        return self.path.filename

    def save_dataset(self, dataset: DatasetBase) -> pathlib.Path:
        dataset.to_csv(self.path)
        _logger.info("Successfully saved dataset", path=str(self.path))
        return self.path

    def load_dataset(
        self, before: str = None, previous_commit: bool = False, commit: str = None
    ) -> DatasetBase:
        """Load dataset from file specified by pointer.

        If options are specified, will load from git history of the file.

        Args:
            before: If set, returns dataset from first commit for file before date.
            previous_commit: If true, returns the dataset from previous commit.
            commit: SHA of specific commit.

        Returns: Instantiated dataset.
        """
        path = self.path
        # If the path is not absolute, assume that the file was created from the repository
        # root. Helpful when loading files from scripts not placed at repo root.
        if not path.is_absolute():
            path = dataset_utils.REPO_ROOT / path

        if before or previous_commit or commit:
            lfs_data = git_lfs_object_helpers.get_data_for_path(
                path, before=before, previous_commit=previous_commit, commit=commit
            )
            lfs_buf = io.BytesIO(lfs_data)
            return self.dataset_type.dataset_class.load_csv(lfs_buf)

        return self.dataset_type.dataset_class.load_csv(path)

    def save(self, directory: pathlib.Path) -> pathlib.Path:
        filename = form_filename(self.dataset_type)
        path = directory / filename
        path.write_text(self.json(indent=2))
        return path
