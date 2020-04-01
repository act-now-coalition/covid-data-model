from typing import List, Iterator
import enum
import pandas as pd
import datetime
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel


class TimeseriesDataset(object):
    """Represents timeseries Case data.

    To make a data source compatible with the timeseries, it must have the required
    fields in the Fields class below + metrics. The other fields are generated
    in the `from_source` method.
    """

    class Fields(object):
        # Required Fields
        DATE = "date"
        COUNTRY = "country"
        STATE = "state"
        FIPS = "fips"
        AGGREGATE_LEVEL = "aggregate_level"

        # Target metrics
        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"

        # Generated in from_source
        COUNTY = "county"
        SOURCE = "source"
        IS_SYNTHETIC = "is_synthetic"
        GENERATED = "generated"

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def states(self) -> List:
        return self.data[self.Fields.STATE].dropna().unique().tolist()

    def county_keys(self) -> List:
        """Returns a list of all (country, state, county) combinations."""
        # Check to make sure all values are county values
        county_values = (
            self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        )
        county_data = self.data[county_values]

        data = county_data.set_index(
            [
                self.Fields.COUNTRY,
                self.Fields.STATE,
                self.Fields.COUNTY,
                self.Fields.FIPS,
            ]
        )
        values = set(data.index.to_list())
        return sorted(values)

    def latest_values(self, aggregation_level):
        """Gets the most recent values for index in array."""
        if aggregation_level == AggregationLevel.COUNTY:
            group = [self.Fields.COUNTRY, self.Fields.STATE, self.Fields.FIPS]
        if aggregation_level == AggregationLevel.STATE:
            group = [self.Fields.COUNTRY, self.Fields.STATE]
        if aggregation_level == AggregationLevel.COUNTRY:
            group = [self.Fields.COUNTRY]

        data = self.data[self.data[self.Fields.AGGREGATE_LEVEL] == aggregation_level.value].reset_index()
        return data.iloc[data.groupby(group).date.idxmax(),:]

    def get_subset(
        self,
        aggregation_level,
        on=None,
        after=None,
        before=None,
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

        if on:
            data = data[data.date == on]
        if after:
            data = data[data.date > after]
        if before:
            data = data[data.date < before]

        return self.__class__(data)

    def get_data(
        self, country=None, state=None, county=None, fips=None
    ) -> pd.DataFrame:
        data = self.data
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if county:
            data = data[data.county == county]
        if fips:
            data = data[data.fips == fips]
        return data

    @classmethod
    def from_source(cls, source: "DataSource", fill_missing_state=True):
        """Loads data from a specific datasource."""
        if not source.TIMESERIES_FIELD_MAP:
            raise ValueError("Source must have field timeseries field map.")

        data = source.data
        to_common_fields = {
            value: key for key, value in source.TIMESERIES_FIELD_MAP.items()
        }
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[cls.Fields.SOURCE] = source.SOURCE_NAME
        data[cls.Fields.GENERATED] = False
        if fill_missing_state:
            state_groupby_fields = [
                cls.Fields.DATE,
                cls.Fields.SOURCE,
                cls.Fields.COUNTRY,
                cls.Fields.STATE,
            ]
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data,
                state_groupby_fields,
                AggregationLevel.COUNTY,
                AggregationLevel.STATE,
            ).reset_index()
            non_matching[cls.Fields.GENERATED] = True
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        # Choosing to sort by date
        data = data.sort_values(cls.Fields.DATE)
        return cls(data)

    def summarize(self):
        dataset_utils.summarize(
            self.data,
            AggregationLevel.COUNTY,
            [self.Fields.DATE, self.Fields.COUNTRY, self.Fields.STATE, self.Fields.FIPS]
        )

        print()
        dataset_utils.summarize(
            self.data,
            AggregationLevel.STATE,
            [self.Fields.DATE, self.Fields.COUNTRY, self.Fields.STATE]
        )
