import pathlib
import datetime
import structlog

from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_pointer import DatasetPointer
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def persist_dataset(
    dataset: timeseries.MultiRegionDataset, data_directory: pathlib.Path,
) -> DatasetPointer:
    """Saves dataset and associated pointer in same data directory.

    Args:
        dataset: Dataset to persist.
        data_directory: Data directory

    Returns: DatasetPointer describing persisted dataset.
    """
    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)

    dataset_type = dataset.dataset_type

    dataset_path = data_directory / f"{dataset_type.value}.csv"
    dataset_pointer = DatasetPointer(
        dataset_type=dataset_type,
        path=dataset_path,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    dataset.write_to_dataset_pointer(dataset_pointer)
    dataset_pointer.save(data_directory)
    return dataset_pointer
