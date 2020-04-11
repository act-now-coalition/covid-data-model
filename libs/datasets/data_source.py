import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.population import PopulationDataset
from libs.datasets import common_fields


class DataSource(object):
    """Represents a single dataset source, loads data and cleans data."""

    # map from custom data source names to common fields
    COMMON_FIELD_MAP = None

    def __init__(self, data: pd.DataFrame, preprocess=True):
        self.original_data = data

        if preprocess:
            data = self._remap_common_field_names(data)
            data = self.apply_source_specific_cleaning(data)
            data = self.standardize_common_field_values(data)
            data = self.filter_and_aggregate_duplicates(data)

        self.data = data

    @classmethod
    def _remap_common_field_names(cls, data):
        return data.rename(columns=cls.COMMON_FIELD_MAP)

    @classmethod
    def apply_source_specific_cleaning(cls, data):
        """Applies source specific data cleaning and adds aggregate level."""
        raise NotImplemented("Subclass must implement")

    @classmethod
    def filter_and_aggregate_duplicates(cls, data):
        """Provides source specific logic to filter and aggregate duplicate values."""
        raise NotImplemented("Subclass must implement")

    @classmethod
    def standardize_common_field_values(cls, data):
        """Standardizes all common field values. """
        return common_fields.standardize_common_fields_data(data)

    @property
    def state_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.STATE).data

    @property
    def county_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.COUNTY).data

    @property
    def country_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.COUNTRY).data

    def get_subset(
        self,
        aggregation_level,
        country=None,
        state=None,
        county=None,
        fips=None,
    ) -> "TimeseriesDataset":
        data = self.data

        if aggregation_level:
            data = data[data.aggregate_level == aggregation_level.value]
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if county:
            data = data[data.county == county]
        if fips:
            data = data[data.fips == fips]

        return self.__class__(data, preprocess=False)

    @classmethod
    def local(cls) -> "cls":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")
