import pathlib
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import structlog
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers

from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset
from functools import lru_cache
import pandas as pd

_log = structlog.get_logger()


class DataSource(object):
    """Represents a single dataset source; loads data and produces a MultiRegionDataset."""

    # Name of source
    # TODO(tom): Make an enum of these.
    SOURCE_NAME = None

    # Fields expected to be in the DataFrame loaded by common_df.read_csv
    EXPECTED_FIELDS: Optional[List[CommonFields]] = None

    # Path of the CSV to be loaded by the default `make_dataset` implementation.
    COMMON_DF_CSV_PATH: Optional[Union[pathlib.Path, str]] = None

    # Fields that are ignored when warning about missing and extra fields. By default some fields
    # that contain redundant information about the location are ignored because cleaning them up
    # isn't worth the effort.
    IGNORED_FIELDS = (CommonFields.COUNTY, CommonFields.COUNTRY, CommonFields.STATE)

    @classmethod
    def _load_data(cls) -> pd.DataFrame:
        assert cls.COMMON_DF_CSV_PATH, f"No path in {cls}"
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.COMMON_DF_CSV_PATH
        return common_df.read_csv(input_path, set_index=False)

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """Default implementation of make_dataset that loads timeseries data from a CSV."""
        data = cls._load_data()
        expected_fields = pd.Index({*cls.EXPECTED_FIELDS, *TIMESERIES_INDEX_FIELDS})
        # Keep only the expected fields.
        found_expected_fields = data.columns.intersection(expected_fields)
        data = data[found_expected_fields]
        extra_fields = data.columns.difference(expected_fields).difference(cls.IGNORED_FIELDS)
        missing_fields = expected_fields.difference(data.columns).difference(cls.IGNORED_FIELDS)
        if not extra_fields.empty:
            _log.info(
                "DataSource produced extra unexpected fields, which were dropped.",
                cls=cls.SOURCE_NAME,
                extra_fields=extra_fields,
            )
        if not missing_fields.empty:
            _log.info(
                "DataSource failed to produce all expected fields",
                cls=cls.SOURCE_NAME,
                missing_fields=missing_fields,
            )

        return MultiRegionDataset.from_fips_timeseries_df(data).add_provenance_all(cls.SOURCE_NAME)


# TODO(tom): Clean up the mess that is subclasses of DataSource and
#  instances of DataSourceAndRegionMasks
class CanScraperBase(DataSource):
    # The method called to transform the DataFrame returned by CanScraperLoader into what is
    # consumed by DataSource.make_dataset. Must be set in subclasses.
    TRANSFORM_METHOD: Callable[[pd.DataFrame], pd.DataFrame]

    @staticmethod
    @lru_cache(None)
    def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
        return ccd_helpers.CanScraperLoader.load()

    @classmethod
    def _load_data(cls) -> pd.DataFrame:
        assert cls.TRANSFORM_METHOD
        ccd_dataset = CanScraperBase._get_covid_county_dataset()
        return cls.TRANSFORM_METHOD(ccd_dataset)
