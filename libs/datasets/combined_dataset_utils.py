from typing import Tuple
import pathlib
import datetime

import structlog

from libs.datasets import dataset_base
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets import dataset_utils
from libs.datasets.dataset_pointer import DatasetPointer
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def persist_dataset(
    dataset: dataset_base.SaveableDatasetInterface,
    data_directory: pathlib.Path,
    data_public_path: pathlib.Path = dataset_utils.LOCAL_PUBLIC_DATA_PATH,
) -> DatasetPointer:
    """Saves dataset and associated pointer in same data directory.

    Args:
        dataset: Dataset to persist.
        data_directory: Data directory
        data_public_path: Path to covid data public folder.

    Returns: DatasetPointer describing persisted dataset.
    """
    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    dataset_type = dataset.dataset_type

    dataset_path = data_directory / f"{dataset_type.value}.csv"
    dataset_pointer = DatasetPointer(
        dataset_type=dataset_type,
        path=dataset_path,
        data_git_info=data_git_info,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    dataset_pointer.save_dataset(dataset)
    dataset_pointer.save(data_directory)
    return dataset_pointer


def update_data_public_head(
    data_directory: pathlib.Path,
    latest_dataset: latest_values_dataset.LatestValuesDataset,
    timeseries_dataset: timeseries.MultiRegionDataset,
) -> Tuple[DatasetPointer, DatasetPointer]:
    """Persists US latest and timeseries dataset and saves dataset pointers for Latest tag.

    Args:
        data_directory: Directory to save dataset and pointer.
        latest_dataset: The LatestValuesDataset to persist for debugging. It is not read downstream.
        timeseries_dataset: The dataset to persist.

    Returns: Tuple of DatasetPointers to latest and timeseries datasets.
    """
    latest_pointer = persist_dataset(latest_dataset, data_directory)

    timeseries_pointer = persist_dataset(timeseries_dataset, data_directory)
    return latest_pointer, timeseries_pointer
