import abc
import pathlib
from typing import List
from typing import Optional
from typing import Union

import structlog
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers

from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.taglib import UrlStr
from libs.datasets.timeseries import MultiRegionDataset
from functools import lru_cache
import pandas as pd

_log = structlog.get_logger()


class DataSource(object):
    """Represents a single dataset source; loads data and produces a MultiRegionDataset."""

    # Attributes set in subclasses and copied to a taglib.Source
    # TODO(tom): Make SOURCE_TYPE an enum when cleaning the mess that is subclasses of DataSource.
    # DataSource class name
    SOURCE_TYPE: str
    SOURCE_NAME: Optional[str] = None
    SOURCE_URL: Optional[UrlStr] = None

    # Fields expected to be in the DataFrame loaded by common_df.read_csv
    EXPECTED_FIELDS: List[CommonFields]

    # Path of the CSV to be loaded by the default `make_dataset` implementation.
    COMMON_DF_CSV_PATH: Optional[Union[pathlib.Path, str]] = None

    # Fields that are ignored when warning about missing and extra fields. By default some fields
    # that contain redundant information about the location are ignored because cleaning them up
    # isn't worth the effort.
    IGNORED_FIELDS = (CommonFields.COUNTY, CommonFields.COUNTRY, CommonFields.STATE)

    @classmethod
    def source_tag(cls) -> taglib.Source:
        # TODO(tom): Make a @property https://docs.python.org/3.9/library/functions.html#classmethod
        return taglib.Source(type=cls.SOURCE_TYPE, url=cls.SOURCE_URL, name=cls.SOURCE_NAME)

    @classmethod
    def _check_data(cls, data: pd.DataFrame):
        expected_fields = pd.Index({*cls.EXPECTED_FIELDS, *TIMESERIES_INDEX_FIELDS})
        # Keep only the expected fields.
        found_expected_fields = data.columns.intersection(expected_fields)
        data = data[found_expected_fields]
        extra_fields = data.columns.difference(expected_fields).difference(cls.IGNORED_FIELDS)
        missing_fields = expected_fields.difference(data.columns).difference(cls.IGNORED_FIELDS)
        if not extra_fields.empty:
            _log.info(
                "DataSource produced extra unexpected fields, which were dropped.",
                cls=cls.SOURCE_TYPE,
                extra_fields=extra_fields,
            )
        if not missing_fields.empty:
            _log.info(
                "DataSource failed to produce all expected fields",
                cls=cls.SOURCE_TYPE,
                missing_fields=missing_fields,
            )
        return data

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """Default implementation of make_dataset that loads timeseries data from a CSV."""
        assert cls.COMMON_DF_CSV_PATH, f"No path in {cls}"
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.COMMON_DF_CSV_PATH
        data = common_df.read_csv(input_path, set_index=False)
        data = cls._check_data(data)
        return MultiRegionDataset.from_fips_timeseries_df(data).add_tag_all_bucket(cls.source_tag())


# TODO(tom): Once using Python 3.9 replace all this metaclass stuff with @classmethod and
#  @property on an EXPECTED_FIELDS method in CanScraperBase. Also try to remove the disable=E1101
#  in many places in the code.
class _CanScraperBaseMeta(type(abc.ABC)):
    @property
    def EXPECTED_FIELDS(cls):
        return [v.common_field for v in cls.VARIABLES if v.common_field is not None]


# TODO(tom): Clean up the mess that is subclasses of DataSource and
#  instances of DataSourceAndRegionMasks
class CanScraperBase(DataSource, abc.ABC, metaclass=_CanScraperBaseMeta):
    # Must be set in subclasses.
    VARIABLES: List[ccd_helpers.ScraperVariable]

    @staticmethod
    @lru_cache(None)
    def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
        return ccd_helpers.CanScraperLoader.load()

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """Default implementation of make_dataset that loads data from the parquet file."""
        ccd_dataset = cls._get_covid_county_dataset()
        rows, source_df = ccd_dataset.query_multiple_variables(
            # pylint: disable=E1101
            cls.VARIABLES,
            log_provider_coverage_warnings=True,
            source_type=cls.SOURCE_TYPE,
        )
        # TODO(tom): Once downstream can handle it return all buckets, not just 'all'.
        rows = rows.xs("all", level=PdFields.DEMOGRAPHIC_BUCKET, drop_level=False)
        data = rows.unstack(PdFields.VARIABLE)
        data = cls._check_data(data)
        ds = MultiRegionDataset(timeseries_bucketed=data)
        if not source_df.empty:
            # For each FIPS-VARIABLE pair keep the source_url row with the last DATE.
            source_tag_df = (
                source_df.sort_values(CommonFields.DATE)
                .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE], sort=False)
                .last()
                .reset_index()
                .drop(columns=[CommonFields.DATE])
                .copy()
            )
            timeseries._tag_df_add_all_bucket_in_place(source_tag_df)
            ds = ds.append_tag_df(source_tag_df)
        return ds
