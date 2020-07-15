from typing import Type, Tuple
import os
import tempfile
import pathlib
import datetime

import structlog
import numpy as np
import pandas as pd
import pydantic
from covidactnow.datapublic import common_df

from libs.datasets import dataset_base
from libs.datasets import combined_datasets
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets import dataset_utils
from libs.datasets.dataset_base import DatasetBase
from libs.datasets.dataset_utils import DatasetType
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

    def load_dataset(self) -> DatasetBase:
        return self.dataset_type.dataset_class.load_csv(self.path)

    def save(self, directory: pathlib.Path) -> pathlib.Path:
        filename = form_filename(self.dataset_type)
        path = directory / filename
        path.write_text(self.json(indent=2))
        return path
