import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.dataset_utils import AggregationLevel
from functools import lru_cache
from libs.datasets.common_fields import CommonFields


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

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def all_fields_map(self):
        return {**self.COMMON_FIELD_MAP, **self.INDEX_FIELD_MAP}

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""
        is_state = (
            self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        )
        return self.data[is_state]

    @property
    def county_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only county data."""
        is_county = (
            self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        )
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
