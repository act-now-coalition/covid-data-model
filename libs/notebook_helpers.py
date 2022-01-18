from typing import Dict
import logging
import pandas as pd

from libs.datasets.sources import nytimes_dataset
from libs.datasets.sources import can_scraper_local_dashboard_providers

_logger = logging.getLogger(__name__)


def load_data_sources_by_name() -> Dict[str, pd.DataFrame]:

    sources = [
        nytimes_dataset.NYTimesDataset,
        can_scraper_local_dashboard_providers.CANScraperStateProviders,
    ]
    source_map = {}
    for source_cls in sources:
        dataset = source_cls.make_dataset().static_and_timeseries_latest_with_fips()

        dataset["source"] = source_cls.__name__
        source_map[source_cls.__name__] = dataset

    return source_map
