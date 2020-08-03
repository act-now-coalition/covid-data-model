from typing import Dict
import logging
import os
import pandas as pd

from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.sources import cmdc
from libs.datasets.sources import cds_dataset
from libs.datasets.sources import covid_tracking_source
from libs.datasets.sources import nytimes_dataset
from libs.datasets.sources import jhu_dataset
from libs.datasets.sources import nha_hospitalization
from libs.datasets.sources import texas_hospitalizations
from libs.datasets.combined_datasets import US_STATES_FILTER

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


def load_data_sources_by_name() -> Dict[str, TimeseriesDataset]:

    sources = [
        cmdc.CmdcDataSource,
        cds_dataset.CDSDataset,
        jhu_dataset.JHUDataset,
        nha_hospitalization.NevadaHospitalAssociationData,
        texas_hospitalizations.TexasHospitalizations,
        covid_tracking_source.CovidTrackingDataSource,
        nytimes_dataset.NYTimesDataset,
    ]
    source_map = {}
    for source_cls in sources:
        dataset = TimeseriesDataset.build_from_data_source(source_cls.local())
        dataset = US_STATES_FILTER.apply(dataset)
        dataset.data["source"] = source_cls.__name__
        source_map[source_cls.__name__] = dataset

    return source_map


def load_combined_timeseries(
    sources: Dict[str, TimeseriesDataset], timeseries: TimeseriesDataset
) -> TimeseriesDataset:
    timeseries_data = timeseries.data.copy()
    timeseries_data["source"] = "Combined Data"

    combined_timeseries = TimeseriesDataset(
        pd.concat([timeseries_data] + [source.data for source in sources.values()])
    )
    return combined_timeseries
