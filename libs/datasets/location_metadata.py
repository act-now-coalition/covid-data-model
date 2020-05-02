from typing import Type, List

from libs import us_state_abbrev
import pandas as pd
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import custom_aggregations


class MetadataDataset(object):

    class Fields(object):
        COUNTRY = "country"
        STATE = "state"
        FIPS = "fips"
        AGGREGATE_LEVEL = "aggregate_level"

        POPULATION = "population"

        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"
        ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate"
        ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate"

        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"
        CURRENT_HOSPITALIZED = "current_hospitalized"
        CUMULATIVE_HOSPITALIZED = "cumulative_hospitalized"
        CUMULATIVE_ICU = "cumulative_icu"

        # Current values
        CURRENT_ICU = "current_icu"
        CURRENT_VENTILATED = "current_ventilated"

        POSITIVE_TESTS = "positive_tests"
        NEGATIVE_TESTS = "negative_tests"

        # Combined beds are the max of staffed and licensed beds.
        # The need for this may be dataset specific, but there are cases in the
        # DH Beds dataset where a source only has either staffed or licensed.
        # To provide a better best guess, we take the max of the two in `from_source`.
        MAX_BED_COUNT = "max_bed_count"

    STATE_GROUP_KEY = [
        Fields.AGGREGATE_LEVEL,
        Fields.COUNTRY,
        Fields.STATE,
    ]

    COUNTY_GROUP_KEY = [
        Fields.AGGREGATE_LEVEL,
        Fields.COUNTRY,
        Fields.STATE,
        Fields.FIPS,
    ]

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
        if not source.METADATA_FIELD_MAP:
            raise ValueError("Source must have metadata field map.")

        data = source.data

        to_common_fields = {value: key for key, value in source.METADATA_FIELD_MAP.items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]

        data = cls._aggregate_new_york_data(data)
        if fill_missing_state:
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data,
                cls.STATE_GROUP_KEY,
                AggregationLevel.COUNTY,
                AggregationLevel.STATE,
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
                (licensed_beds * occupancy_rates).sum() / licensed_beds.sum()
            )
        weighted_icu_occupancy = None
        if cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE in data.columns:
            icu_beds = nyc_data[cls.Fields.ICU_BEDS]
            occupancy_rates = nyc_data[cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE]
            weighted_icu_occupancy = (
                (icu_beds * occupancy_rates).sum() / icu_beds.sum()
            )

        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=False
        )

        nyc_fips = custom_aggregations.NEW_YORK_COUNTY_FIPS
        if weighted_all_bed_occupancy:
            data.loc[data[cls.Fields.FIPS] == nyc_fips, cls.Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE] = (
                weighted_all_bed_occupancy
            )

        if weighted_icu_occupancy:
            data.loc[data[cls.Fields.FIPS] == nyc_fips, cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE] = (
                weighted_icu_occupancy
            )

        return data

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""
        print("HI")
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
