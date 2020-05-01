import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.sources.jhu_dataset import JHUDataset
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.dataset_utils import AggregationLevel


class AggregateActualLatest(object):

    class Fields(object):
        DATE = "date"
        COUNTRY = "country"
        STATE = "state"
        FIPS = "fips"
        AGGREGATE_LEVEL = "aggregate_level"

        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"
        ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate"
        ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate"

        POPULATION = "population"
        COUNTY_NAME = "county_name"

    DATA_SOURCE_MAP = {
        Fields.STAFFED_BEDS: CovidCareMapBeds,
        Fields.LICENSED_BEDS: CovidCareMapBeds,
        Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE: CovidCareMapBeds,
        Fields.DEATHS: JHUDataset,
        Fields.CASES: JHUDataset,
    }

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

    def get_data_for_state(self, state) -> dict:
        """Gets all data for a given state.

        Args:
            state: State abbreviation.

        Returns: Dictionary with all data for a given state.
        """
        data = self.state_data
        row = data[data[self.Fields.STATE] == state]
        if not len(row):
            return {}

        return row.iloc[0].to_dict()

    def get_data_for_fips(self, fips) -> dict:
        """Gets all data for a given fips code.

        Args:
            fips: fips code.

        Returns: Dictionary with all data for a given fips code.
        """
        row = self.data[self.data[self.Fields.FIPS] == fips]
        if not len(row):
            return {}

        return row.iloc[0].to_dict()


class AggregateActualTimeseries(object):

    Fields = TimeseriesDataset.Fields

    DATA_SOURCE_MAP = {
        Fields.CASES: JHUDataset,
        Fields.DEATHS: JHUDataset,
        Fields.CURRENT_HOSPITALIZED: CovidTrackingDataSource,
        Fields.CUMULATIVE_ICU: CovidTrackingDataSource
    }

    def __init__(self, data):
        self.data = data

    @classmethod
    def initialize(cls):
        # Some initial data frame
        data = pd.DataFrame()
        classes = set(cls.DATA_SOURCE_MAP.values())

        data_sources = [
            (source_cls.local().timeseries(), source_cls) for source_cls in classes
        ]
        for data_source, source_cls in data_sources:
            data = cls._add_columns_to_data(data, data_source, source_cls)

        return cls(data)

    @classmethod
    def _add_columns_to_data(cls, data: pd.DataFrame, data_source: TimeseriesDataset):
        """Adds nevada data, replacing any state or county level values that match index.

        Args:
            data: Covid tracking data
            nevada_data: Nevada specific override data.

        Returns: Updated dataframe with
        """
        matching_index_group = [
            cls.Fields.DATE,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        new_data = data_source.data.set_index(matching_index_group)
        data = data.set_index(matching_index_group)

        # Sort indices so that we have chunks of equal length in the
        # correct order so that we can splice in values from nevada data.
        data = data.sort_index()
        new_data = new_data.sort_index()
        data_in_new_data = data.index.isin(new_data.index)
        new_data_in_data = new_data.index.isin(data.index)

        if not sum(data_in_new_data) == sum(new_data_in_data):
            raise ValueError(
                "Number of rows should be the for data to replace"
            )

        # Fill in values with data that matches index in nevada data.
        columns = [
            key
            for key, value in cls.DATA_SOURCE_MAP.items()
            if value == data_source.__class__
        ]
        data.loc[data_in_new_data, columns] = new_data.loc[new_data_in_data, :]

        # Combine updated data with rows not present in covid tracking data.
        return pd.concat([
            data,
            new_data[~new_data_in_data]
        ]).reset_index()
