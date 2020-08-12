from typing import Type

import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.dataset_utils import AggregationLevel
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

    # Flag to indicate whether or not to fill missing state level data with county data
    # when converting to either a TimeseriesDataset or LatestValuesDataset.
    # Some data sources provide state level data, while others don't, however, due to how
    # some data sources report data, aggregating on the county level data may lead to incorrect
    # assumptions about missing vs data that is just zero.
    FILL_MISSING_STATE_LEVEL_DATA = True

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""
        is_state = self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        return self.data[is_state]

    @property
    def county_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only county data."""
        is_county = self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        return self.data[is_county]

    @classmethod
    def local(cls) -> "DataSource":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")

    @lru_cache(None)
    def beds(self) -> LatestValuesDataset:
        """Builds generic beds dataset"""
        return self.latest_values()

    @lru_cache(None)
    def population(self) -> LatestValuesDataset:
        """Builds generic beds dataset"""
        return self.latest_values()

    @lru_cache(None)
    def timeseries(self) -> TimeseriesDataset:
        """Build TimeseriesDataset from this data source."""
        if set(self.INDEX_FIELD_MAP.keys()) != set(TimeseriesDataset.INDEX_FIELDS):
            raise ValueError("Index fields must match")

        return TimeseriesDataset.from_source(
            self, fill_missing_state=self.FILL_MISSING_STATE_LEVEL_DATA
        )

    @lru_cache(None)
    def latest_values(self) -> LatestValuesDataset:
        if set(self.INDEX_FIELD_MAP.keys()) == set(TimeseriesDataset.INDEX_FIELDS):
            return LatestValuesDataset(self.timeseries().latest_values())

        if set(self.INDEX_FIELD_MAP.keys()) != set(LatestValuesDataset.INDEX_FIELDS):
            raise ValueError("Index fields must match")

        return LatestValuesDataset.from_source(
            self, fill_missing_state=self.FILL_MISSING_STATE_LEVEL_DATA
        )

    @classmethod
    def _rename_to_common_fields(cls: Type["DataSource"], df: pd.DataFrame) -> pd.DataFrame:
        all_fields_map = {**cls.COMMON_FIELD_MAP, **cls.INDEX_FIELD_MAP}
        to_common_fields = {value: key for key, value in all_fields_map.items()}
        final_columns = to_common_fields.values()
        return df.rename(columns=to_common_fields)[final_columns]
