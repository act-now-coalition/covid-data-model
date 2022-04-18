import pathlib
import datetime

import structlog
import pydantic

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

    # Sha of covid-data-model used to create dataset.
    model_git_info: GitSummary

    # When local file was saved.
    updated_at: datetime.datetime

    @property
    def path_absolute(self) -> str:
        # If the path is not absolute, assume that the file was created from the repository
        # root. Helpful when loading files from scripts not placed at repo root.
        if not self.path.is_absolute():
            return str(dataset_utils.REPO_ROOT / self.path)
        return str(self.path)

    def path_wide_dates(self) -> pathlib.Path:
        return pathlib.Path(self.path_absolute.replace(".csv", "-wide-dates.csv.gz"))

    def path_annotation(self) -> pathlib.Path:
        return pathlib.Path(self.path_absolute.replace(".csv", "-annotations.csv"))

    def path_static(self) -> pathlib.Path:
        return pathlib.Path(self.path_absolute.replace(".csv", "-static.csv"))

    def save(self, directory: pathlib.Path) -> pathlib.Path:
        filename = form_filename(self.dataset_type)
        path = directory / filename
        path.write_text(self.json(indent=2))
        return path
