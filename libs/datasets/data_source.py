from typing import Type, Optional

import pandas as pd

from libs.datasets.dataset_utils import STATIC_INDEX_FIELDS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset
from functools import lru_cache


class DataSource(object):
    """Represents a single dataset source, loads data and cleans data."""

    COMMON_FIELD_MAP = None

    # Name of dataset source
    SOURCE_NAME = None

    # Indicates if NYC data is aggregated into one NYC county or not.
    HAS_AGGREGATED_NYC_BOROUGH = False

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @classmethod
    def make_dataset(cls) -> MultiRegionDataset:
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        return cls.local().multi_region_dataset()

    def local(cls) -> "DataSource":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")

    @classmethod
    def make_static_dataset(cls, data: pd.DataFrame) -> MultiRegionDataset:
        return MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)

    @classmethod
    def make_timeseries_dataset(cls, data: pd.DataFrame) -> MultiRegionDataset:
        return MultiRegionDataset.from_fips_timeseries_df(data).add_provenance_all(cls.SOURCE_NAME)

    def multi_region_dataset(self) -> MultiRegionDataset:
        if set(self.INDEX_FIELD_MAP.keys()) == set(TIMESERIES_INDEX_FIELDS):
            return self.make_timeseries_dataset(self.data)

        if set(self.INDEX_FIELD_MAP.keys()) == set(STATIC_INDEX_FIELDS):
            return self.make_static_dataset(self.data)

        raise ValueError("Unexpected index fields")

    @classmethod
    def _rename_to_common_fields(cls: Type["DataSource"], df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of the DataFrame with only common columns in the class field maps."""
        all_fields_map = {**cls.COMMON_FIELD_MAP, **cls.INDEX_FIELD_MAP}
        to_common_fields = {value: key for key, value in all_fields_map.items()}
        final_columns = to_common_fields.values()
        return df.rename(columns=to_common_fields)[final_columns]
