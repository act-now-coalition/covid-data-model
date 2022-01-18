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


def format_sample_of_df(df: pd.DataFrame) -> str:
    """Formats a sample of a DataFrame as a string, suitable for dumping to a log."""
    return df.to_string(
        line_width=120, max_rows=10, max_cols=5, max_colwidth=40, show_dimensions=True
    )


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
    IGNORED_FIELDS = (
        CommonFields.COUNTY,
        CommonFields.COUNTRY,
        CommonFields.STATE,
        CommonFields.AGGREGATE_LEVEL,
        CommonFields.DATE,
        CommonFields.FIPS,
    )

    @classmethod
    def source_tag(cls) -> taglib.Source:
        # TODO(tom): Make a @property https://docs.python.org/3.9/library/functions.html#classmethod
        return taglib.Source(type=cls.SOURCE_TYPE, url=cls.SOURCE_URL, name=cls.SOURCE_NAME)

    @classmethod
    def _check_and_removed_unexpected_data(cls, data: pd.DataFrame):
        # data may be indexed by location_id,date (in CanSraperBase, ready for MultiRegionDataset)
        # or not (in DataSource, which calls from_fips_timeseries_df). TODO(tom): reduce the
        #  number of code paths.
        dates_sequence: Optional[Union[pd.Series, pd.Index]] = None
        if CommonFields.DATE in data.columns:
            dates_sequence = pd.to_datetime(data.loc[:, CommonFields.DATE])
        elif CommonFields.DATE in data.index.names:
            dates_sequence = data.index.get_level_values(CommonFields.DATE)
        if dates_sequence is not None:
            old_dates_mask = dates_sequence < "2020-01-01"
            if old_dates_mask.any():
                _log.warning(
                    "Dropping old data",
                    cls=cls.SOURCE_TYPE,
                    dropped_df=format_sample_of_df(data.loc[old_dates_mask]),
                )
                data = data.loc[~old_dates_mask]
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
    def _load_data(cls) -> pd.DataFrame:
        """Loads the CSV, override to inject data in a test."""
        assert cls.COMMON_DF_CSV_PATH, f"No path in {cls}"
        data_root = dataset_utils.DATA_DIRECTORY
        input_path = data_root / cls.COMMON_DF_CSV_PATH
        return common_df.read_csv(input_path, set_index=False)

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """Default implementation of make_dataset that loads timeseries data from a CSV."""
        data = cls._load_data()
        data = cls._check_and_removed_unexpected_data(data)
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
        # fetch and persist Parquet file from GCS, then load it
        ccd_helpers.CanScraperLoader.persist_parquet()
        return ccd_helpers.CanScraperLoader.load_from_local()

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
        data = rows.unstack(PdFields.VARIABLE)
        data = cls._check_and_removed_unexpected_data(data).sort_index(ascending=True)
        ds = MultiRegionDataset(timeseries_bucketed=data)
        if not source_df.empty:
            # For each FIPS-VARIABLE pair keep the source_url row with the last DATE.
            source_tag_df = (
                source_df.sort_values(CommonFields.DATE)
                .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE], sort=False)
                .last()
                .reset_index()
                .drop(columns=[CommonFields.DATE])
                # copy before calling tag_df_add_all_bucket_in_place, just to be safe.
                .copy()
            )
            timeseries.tag_df_add_all_bucket_in_place(source_tag_df)
            ds = ds.append_tag_df(source_tag_df)
        return ds
