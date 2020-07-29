import logging
import os

from libs.datasets import dataset_utils

_logger = logging.getLogger(__name__)


def set_covid_data_public():
    if dataset_utils.LOCAL_PUBLIC_DATA_PATH.exists():
        return

    # When initializing a notebook using binder, covid-data-public is cloned to
    # the repo root.  This checks if that directory exists and
    path = dataset_utils.REPO_ROOT / "covid-data-public"
    if path.exists():
        os.environ["COVID_DATA_PUBLIC"] = str(path)
        dataset_utils.set_global_public_data_path()
        _logger.info(f"Found covid-data-public in repo root, now pointing to {path}")
        assert dataset_utils.LOCAL_PUBLIC_DATA_PATH.exists()
        return

    raise Exception("Could not find covid-data-public")
