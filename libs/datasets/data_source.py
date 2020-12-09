from typing import Type, Optional

import pandas as pd

from libs.datasets.dataset_utils import STATIC_INDEX_FIELDS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset
from functools import lru_cache


class DataSource(object):
    """Represents a single dataset source, loads data and cleans data."""

    # Subclass must implement custom class of fields in dataset.
    class Fields(object):
        pass

    INDEX_FIELD_MAP = None

    COMMON_FIELD_MAP = None

    # Name of dataset source
    SOURCE_NAME = None

    # Indicates if NYC data is aggregated into one NYC county or not.
    HAS_AGGREGATED_NYC_BOROUGH = False

    def __init__(self, data: pd.DataFrame, provenance: Optional[pd.Series] = None):
        self.data = data
        self.provenance = provenance

    @classmethod
    def local(cls) -> "DataSource":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")

    @lru_cache(None)
    def multi_region_dataset(self) -> MultiRegionDataset:
        if set(self.INDEX_FIELD_MAP.keys()) == set(TIMESERIES_INDEX_FIELDS):
            dataset = MultiRegionDataset.from_fips_timeseries_df(self.data).add_provenance_all(
                self.SOURCE_NAME
            )
            # TODO(tom): DataSource.provenance is only set by
            # CovidCountyDataDataSource.synthesize_test_metrics. Factor it out into something
            # that reads and creates a MultiRegionDataset.
            # if self.provenance is not None:
            #     dataset.add_fips_provenance(self.provenance)
            return dataset

        if set(self.INDEX_FIELD_MAP.keys()) == set(STATIC_INDEX_FIELDS):
            return MultiRegionDataset.new_without_timeseries().add_fips_static_df(self.data)

        raise ValueError("Unexpected index fields")

    @classmethod
    def _rename_to_common_fields(cls: Type["DataSource"], df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of the DataFrame with only common columns in the class field maps."""
        all_fields_map = {**cls.COMMON_FIELD_MAP, **cls.INDEX_FIELD_MAP}
        to_common_fields = {value: key for key, value in all_fields_map.items()}
        final_columns = to_common_fields.values()
        return df.rename(columns=to_common_fields)[final_columns]
