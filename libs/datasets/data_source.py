import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.population import PopulationDataset
from libs.datasets.location_metadata import MetadataDataset
from libs.datasets.dataset_utils import AggregationLevel
from functools import lru_cache


class DataSource(object):
    """Represents a single dataset source, loads data and cleans data."""

    # Subclass must implement custom class of fields in dataset.
    class Fields(object):
        pass

    # Subclasses must define mapping from Timeseries fields.
    # eg: {TimseriesDataset.Fields.DATE: Fields.Date}
    # Optional if dataset does not support timeseries data.
    TIMESERIES_FIELD_MAP = None

    # Map of field names from BedsDataset.Fields to dataset source fields.
    # Optional if dataset does not support converting to beds data.
    BEDS_FIELD_MAP = None

    # Map of field names from PopulationDataset.Fields to dataset source fields.
    # Optional if dataset does not support population data.
    POPULATION_FIELD_MAP = None

    METADATA_FIELD_MAP = None

    # Name of dataset source
    SOURCE_NAME = None

    # Indicates if NYC data is aggregated into one NYC county or not.
    HAS_AGGREGATED_NYC_BOROUGH = False

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""
        is_state = (
            self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        )
        return self.data[is_state]

    @property
    def county_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only county data."""
        is_county = (
            self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        )
        return self.data[is_county]

    @classmethod
    def local(cls) -> "cls":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")

    def beds(self) -> "BedsDataset":
        """Builds generic beds dataset"""
        return BedsDataset.from_source(self)

    def population(self) -> "PopulationDataset":
        """Builds generic beds dataset"""
        return PopulationDataset.from_source(self)

    def metadata(self) -> MetadataDataset:
        if not self.METADATA_FIELD_MAP and self.TIMESERIES_FIELD_MAP:
            data = TimeseriesDataset.from_source(self).latest_values()
            return MetadataDataset(data)
        return MetadataDataset.from_source(self)

    def timeseries(self, fill_na: bool = True) -> "TimeseriesDataset":
        """Builds generic timeseries dataset.

        Args:
            fill_na: If True, fills all NA values with 0.
        """

        return TimeseriesDataset.from_source(self, fill_na=fill_na)
