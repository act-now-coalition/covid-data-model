from typing import Dict
import logging
import os
import pandas as pd

from libs.datasets import dataset_utils
from libs.datasets.sources import covid_county_data
from libs.datasets.sources import covid_tracking_source
from libs.datasets.sources import nytimes_dataset
from libs.datasets.sources import texas_hospitalizations

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


def load_data_sources_by_name() -> Dict[str, pd.DataFrame]:

    sources = [
        covid_county_data.CovidCountyDataDataSource,
        texas_hospitalizations.TexasHospitalizations,
        covid_tracking_source.CovidTrackingDataSource,
        nytimes_dataset.NYTimesDataset,
    ]
    source_map = {}
    for source_cls in sources:
        dataset = source_cls.local().multi_region_dataset().static_and_timeseries_latest_with_fips()

        dataset["source"] = source_cls.__name__
        source_map[source_cls.__name__] = dataset

    return source_map
