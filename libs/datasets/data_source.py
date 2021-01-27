import pathlib
from typing import List
from typing import Optional
from typing import Union

import structlog
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields

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
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """Default implementation of make_dataset that loads timeseries data from a CSV."""
        assert cls.COMMON_DF_CSV_PATH, f"No path in {cls}"
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.COMMON_DF_CSV_PATH
        data = common_df.read_csv(input_path, set_index=False)
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
