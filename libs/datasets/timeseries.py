import warnings
from typing import List
import pandas as pd
from libs import us_state_abbrev
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import custom_aggregations
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel


class TimeseriesDataset(dataset_base.DatasetBase):
    """Represents timeseries dataset.

    To make a data source compatible with the timeseries, it must have the required
    fields in the Fields class below + metrics. The other fields are generated
    in the `from_source` method.
    """

    class Fields(CommonFields):
        COUNTRY = CommonIndexFields.COUNTRY
        STATE = CommonIndexFields.STATE
        AGGREGATE_LEVEL = CommonIndexFields.AGGREGATE_LEVEL
        FIPS = CommonIndexFields.FIPS
        DATE = CommonIndexFields.DATE

    INDEX_FIELDS = [
        CommonIndexFields.DATE,
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def states(self) -> List:
        return self.data[self.Fields.STATE].dropna().unique().tolist()

    @property
    def state_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.STATE).data

    @property
    def county_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.COUNTY).data

    def county_keys(self) -> List:
        """Returns a list of all (country, state, county) combinations."""
        # Check to make sure all values are county values
        warnings.warn(
            "Tell Tom you are using this, I'm going to delete it soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        county_values = self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        county_data = self.data[county_values]

        data = county_data.set_index(
            [self.Fields.COUNTRY, self.Fields.STATE, self.Fields.COUNTY, self.Fields.FIPS,]
        )
        values = set(data.index.to_list())
        return sorted(values)

    def latest_values(self, aggregation_level=None) -> pd.DataFrame:
        """Gets the most recent values.

        Args:
            aggregation_level: If specified, only gets latest values for that aggregation,
                otherwise returns values for entire aggretation.

        Return: DataFrame
        """
        if not aggregation_level:
            county = self.latest_values(aggregation_level=AggregationLevel.COUNTY)
            state = self.latest_values(aggregation_level=AggregationLevel.STATE)
            return pd.concat([county, state])

        if aggregation_level == AggregationLevel.COUNTY:
            group = [self.Fields.COUNTRY, self.Fields.STATE, self.Fields.FIPS]
        if aggregation_level == AggregationLevel.STATE:
            group = [self.Fields.COUNTRY, self.Fields.STATE]
        if aggregation_level == AggregationLevel.COUNTRY:
            group = [self.Fields.COUNTRY]

        data = self.data[
            self.data[self.Fields.AGGREGATE_LEVEL] == aggregation_level.value
        ].reset_index()
        # If the groupby raises a ValueError check the dtype of date. If it was loaded
        # by read_csv did you set parse_dates=["date"]?
        return data.iloc[data.groupby(group).date.idxmax(), :]

    def get_subset(
        self,
        aggregation_level,
        on=None,
        after=None,
        before=None,
        country=None,
        state=None,
        fips=None,
        states=None,
    ) -> "TimeseriesDataset":
        data = self.data

        if aggregation_level:
            data = data[data.aggregate_level == aggregation_level.value]
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if fips:
            data = data[data.fips == fips]
        if states:
            data = data[data[self.Fields.STATE].isin(states)]

        if on:
            data = data[data.date == on]
        if after:
            data = data[data.date > after]
        if before:
            data = data[data.date < before]

        return self.__class__(data)

    def get_records_for_fips(self, fips) -> List[dict]:
        """Get data for FIPS code.

        Args:
            fips: FIPS code.

        Returns: List of dictionary records with NA values replaced to be None
        """

        pd_data = self.get_subset(None, fips=fips).data
        pd_data = pd_data.where(pd.notnull(pd_data), None)
        return pd_data.to_dict(orient="records")

    def get_records_for_state(self, state) -> List[dict]:
        """Get data for state.

        Args:
            state: 2 letter state abbrev.

        Returns: List of dictionary records with NA values replaced to be None.
        """
        pd_data = self.get_subset(AggregationLevel.STATE, state=state).data
        return pd_data.where(pd.notnull(pd_data), None).to_dict(orient="records")

    def get_data(self, country=None, state=None, fips=None, states=None) -> pd.DataFrame:
        data = self.data
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if fips:
            data = data[data.fips == fips]
        if states:
            data = data[data[self.Fields.STATE].isin(states)]

        return data

    @classmethod
    def from_source(
        cls, source: "DataSource", fill_missing_state: bool = True
    ) -> "TimeseriesDataset":
        """Loads data from a specific datasource.

        Args:
            source: DataSource to standardize for timeseries dataset
            fill_missing_state: If True, backfills missing state data by
                calculating county level aggregates.

        Returns: Timeseries object.
        """
        data = source.data
        to_common_fields = {value: key for key, value in source.all_fields_map().items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        group = [
            cls.Fields.DATE,
            cls.Fields.COUNTRY,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.STATE,
        ]
        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=source.HAS_AGGREGATED_NYC_BOROUGH
        )

        if fill_missing_state:
            state_groupby_fields = [
                cls.Fields.DATE,
                cls.Fields.COUNTRY,
                cls.Fields.STATE,
            ]
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data, state_groupby_fields, AggregationLevel.COUNTY, AggregationLevel.STATE,
            ).reset_index()
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)
        is_state = data[cls.Fields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        state_fips = data.loc[is_state, cls.Fields.STATE].map(us_state_abbrev.ABBREV_US_FIPS)
        data.loc[is_state, cls.Fields.FIPS] = state_fips

        # Choosing to sort by date
        data = data.sort_values(cls.Fields.DATE)
        return cls(data)

    @classmethod
    def build_from_data_source(cls, source):
        """Build TimeseriesDataset from a data source."""
        if set(source.INDEX_FIELD_MAP.keys()) != set(cls.INDEX_FIELDS):
            raise ValueError("Index fields must match")

        return cls.from_source(source)

    def to_latest_values_dataset(self):
        from libs.datasets.latest_values_dataset import LatestValuesDataset

        return LatestValuesDataset(self.latest_values())

    def summarize(self):
        dataset_utils.summarize(
            self.data,
            AggregationLevel.COUNTY,
            [self.Fields.DATE, self.Fields.COUNTRY, self.Fields.STATE, self.Fields.FIPS,],
        )

        print()
        dataset_utils.summarize(
            self.data,
            AggregationLevel.STATE,
            [self.Fields.DATE, self.Fields.COUNTRY, self.Fields.STATE],
        )
