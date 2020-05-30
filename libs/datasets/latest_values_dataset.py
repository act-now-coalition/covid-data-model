from typing import Type, List

from libs import us_state_abbrev
import pandas as pd
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import custom_aggregations
from libs.datasets import dataset_base
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields


class LatestValuesDataset(dataset_base.DatasetBase):

    INDEX_FIELDS = [
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]
    STATE_GROUP_KEY = [
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
    ]

    class Fields(CommonFields):
        COUNTRY = CommonIndexFields.COUNTRY
        STATE = CommonIndexFields.STATE
        AGGREGATE_LEVEL = CommonIndexFields.AGGREGATE_LEVEL
        FIPS = CommonIndexFields.FIPS

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_source(cls, source: "DataSource", fill_missing_state=True):
        """Loads data from a specific datasource.

        Remaps columns from source dataset, fills in missing data
        by computing aggregates, and adds standardized county names from fips.

        Args:
            source: Data source.
            fill_missing_state: If True, fills in missing state level data by
                aggregating county level for a given state.
        """
        if not source.COMMON_FIELD_MAP and not source.INDEX_FIELD_MAP:
            raise ValueError("Source must have metadata field map.")

        data = source.data
        fields = source.all_fields_map().items()
        to_common_fields = {value: key for key, value in fields}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]

        data = cls._aggregate_new_york_data(data)
        if fill_missing_state:
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data, cls.STATE_GROUP_KEY, AggregationLevel.COUNTY, AggregationLevel.STATE,
            ).reset_index()

            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        # Add state fips
        is_state = data[cls.Fields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        state_fips = data.loc[is_state, cls.Fields.STATE].map(us_state_abbrev.ABBREV_US_FIPS)
        data.loc[is_state, cls.Fields.FIPS] = state_fips

        return cls(data)

    @classmethod
    def build_from_data_source(cls, source):
        from libs.datasets.timeseries import TimeseriesDataset

        if set(source.INDEX_FIELD_MAP.keys()) == set(TimeseriesDataset.INDEX_FIELDS):
            timeseries = TimeseriesDataset.build_from_data_source(source)
            return timeseries.to_latest_values_dataset()

        if set(source.INDEX_FIELD_MAP.keys()) != set(cls.INDEX_FIELDS):
            raise ValueError("Index fields must match")

        return cls.from_source(source)

    def get_subset(
        self, aggregation_level, country=None, state=None, fips=None, states=None,
    ) -> "LatestValuesDataset":
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

        return self.__class__(data)

    @classmethod
    def _aggregate_new_york_data(cls, data):
        # When grouping nyc data, we don't want to count the generated field
        # as a value to sum.
        nyc_data = data[data[cls.Fields.FIPS].isin(custom_aggregations.ALL_NYC_FIPS)]
        if not len(nyc_data):
            return data
        group = cls.STATE_GROUP_KEY
        weighted_all_bed_occupancy = None

        if cls.Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE in data.columns:
            licensed_beds = nyc_data[cls.Fields.LICENSED_BEDS]
            occupancy_rates = nyc_data[cls.Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE]
            weighted_all_bed_occupancy = (
                licensed_beds * occupancy_rates
            ).sum() / licensed_beds.sum()
        weighted_icu_occupancy = None
        if cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE in data.columns:
            icu_beds = nyc_data[cls.Fields.ICU_BEDS]
            occupancy_rates = nyc_data[cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE]
            weighted_icu_occupancy = (icu_beds * occupancy_rates).sum() / icu_beds.sum()

        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=False
        )

        nyc_fips = custom_aggregations.NEW_YORK_COUNTY_FIPS
        if weighted_all_bed_occupancy:
            data.loc[
                data[cls.Fields.FIPS] == nyc_fips, cls.Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
            ] = weighted_all_bed_occupancy

        if weighted_icu_occupancy:
            data.loc[
                data[cls.Fields.FIPS] == nyc_fips, cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE
            ] = weighted_icu_occupancy

        return data

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""

        is_state = self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        return self.data[is_state]

    @property
    def county_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only county data."""
        is_county = self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        return self.data[is_county]

    def get_record_for_state(self, state) -> dict:
        """Gets all data for a given state.

        Args:
            state: State abbreviation.

        Returns: Dictionary with all data for a given state.
        """
        # we map NaNs to none here so that they can be generated via the API easier
        data = self.state_data.where(pd.notnull(self.state_data), None)
        row = data[data[self.Fields.STATE] == state]
        if not len(row):
            return {}

        return row.iloc[0].to_dict()

    def get_record_for_fips(self, fips) -> dict:
        """Gets all data for a given fips code.

        Args:
            fips: fips code.

        Returns: Dictionary with all data for a given fips code.
        """
        # we map NaNs to none here so that they can be generated via the API easier
        row = self.data[self.data[self.Fields.FIPS] == fips].where(pd.notnull(self.data), None)
        if not len(row):
            return {}

        return row.iloc[0].to_dict()
