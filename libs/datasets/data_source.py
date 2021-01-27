import pathlib
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset
from functools import lru_cache


class DataSource(object):
    """Represents a single dataset source, loads data and cleans data."""

    # Name of dataset source
    SOURCE_NAME = None

    # Fields expected in the DataFrame loaded by common_df.read_csv
    EXPECTED_FIELDS: Optional[List[CommonFields]] = None

    # Path of the CSV to be loaded by the default `make_dataset` implementation.
    COMMON_DF_CSV_PATH: Optional[Union[pathlib.Path, str]] = None

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        assert cls.COMMON_DF_CSV_PATH, f"No path in {cls}"
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.COMMON_DF_CSV_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls.make_timeseries_dataset(data)

    @classmethod
    def make_static_dataset(cls, data: pd.DataFrame) -> MultiRegionDataset:
        return MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)

    @classmethod
    def make_timeseries_dataset(cls, data: pd.DataFrame) -> MultiRegionDataset:
        if cls.EXPECTED_FIELDS:
            data = data[data.columns.intersection(cls.EXPECTED_FIELDS + TIMESERIES_INDEX_FIELDS)]
            # TODO(tom): Warn about unexpected fields?
        return MultiRegionDataset.from_fips_timeseries_df(data).add_provenance_all(cls.SOURCE_NAME)
