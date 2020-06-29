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

    def all_fields_map(self):
        return {**self.COMMON_FIELD_MAP, **self.INDEX_FIELD_MAP}

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
        return LatestValuesDataset.build_from_data_source(self)

    @lru_cache(None)
    def population(self) -> LatestValuesDataset:
        """Builds generic beds dataset"""
        return LatestValuesDataset.build_from_data_source(self)

    @lru_cache(None)
    def timeseries(self) -> TimeseriesDataset:
        """Builds generic beds dataset"""
        return TimeseriesDataset.build_from_data_source(self)
