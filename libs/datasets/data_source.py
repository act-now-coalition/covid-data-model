import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.population import PopulationDataset


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

    # Name of dataset source
    SOURCE_NAME = None

    def __init__(self, data: pd.DataFrame):
        self.data = data

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

    def timeseries(self) -> "TimeseriesDataset":
        """Builds generic beds dataset"""
        return TimeseriesDataset.from_source(self)
