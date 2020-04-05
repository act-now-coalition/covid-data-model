from typing import Optional
import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import custom_aggregations


class BedsDataset(object):
    class Fields(object):
        COUNTRY = "country"
        STATE = "state"
        FIPS = "fips"
        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"
        AGGREGATE_LEVEL = "aggregate_level"

        COUNTY = "county"
        SOURCE = "source"
        GENERATED = "generated"
        # Combined beds are the max of staffed and licensed beds.
        # The need for this may be dataset specific, but there are cases in the
        # DH Beds dataset where a source only has either staffed or licensed.
        # To provide a better best guess, we take the max of the two in `from_source`.
        MAX_BED_COUNT = "max_bed_count"

    # Key for grouping data at the state level.
    # This includes other keys that should be preserved - for example, this will preserve
    # the source field during a groupby.
    STATE_GROUP_KEY = [
        Fields.SOURCE,
        Fields.AGGREGATE_LEVEL,
        Fields.COUNTRY,
        Fields.STATE,
    ]

    COUNTY_GROUP_KEY = [
        Fields.SOURCE,
        Fields.AGGREGATE_LEVEL,
        Fields.COUNTRY,
        Fields.STATE,
        Fields.FIPS,
    ]

    def __init__(self, data):
        self.data = data
        self.validate()

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
    def from_source(cls, source: "DataSource", fill_missing_state=True):
        """Loads data from a specific datasource.

        Remaps columns from source dataset, fills in missing data
        by computing aggregates, and adds standardized county names from fips.

        Args:
            source: Data source.
            fill_missing_state: If True, fills in missing state level data by
                aggregating county level for a given state.
        """
        if not source.BEDS_FIELD_MAP:
            raise ValueError("Source must have beds field map.")

        data = source.data

        to_common_fields = {value: key for key, value in source.BEDS_FIELD_MAP.items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[cls.Fields.SOURCE] = source.SOURCE_NAME
        data[cls.Fields.GENERATED] = False

        # Generating max bed count.
        columns_to_consider = [cls.Fields.STAFFED_BEDS, cls.Fields.LICENSED_BEDS]
        data[cls.Fields.MAX_BED_COUNT] = data[columns_to_consider].max(axis=1)

        # When grouping nyc data, we don't want to count the generated field
        # as a value to sum.
        group = cls.STATE_GROUP_KEY + [cls.Fields.GENERATED]
        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=False
        )
        if fill_missing_state:
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data,
                cls.STATE_GROUP_KEY,
                AggregationLevel.COUNTY,
                AggregationLevel.STATE,
            ).reset_index()

            non_matching[cls.Fields.GENERATED] = True
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        return cls(data)

    def validate(self):
        dataset_utils.check_index_values_are_unique(
            self.state_data, index=self.STATE_GROUP_KEY
        )
        dataset_utils.check_index_values_are_unique(
            self.county_data, index=self.COUNTY_GROUP_KEY
        )

    def get_state_level(self, state) -> Optional[int]:
        """Get beds for a specific state.

        Args:
            state: State to query.

        Returns: Beds for a state.
        """
        state_filter = self.state_data.state == state
        beds = self.state_data[state_filter]

        if len(beds):
            return beds.iloc[0][self.Fields.MAX_BED_COUNT]
        return None

    def get_county_level(self, state, county=None, fips=None) -> Optional[int]:
        """Get beds for a specific county (from fips code or county).

        Args:
            state: State to query
            county: Optional name of county.
            fips: Optional fips code.

        Returns: Beds for a county.
        """
        if not (county or fips) or (county and fips):
            raise ValueError("Must only pass fips or county")

        data = self.county_data
        state_filter = data.state == state
        if fips:
            location_filter = data.fips == fips
        elif county:
            location_filter = data.county == county

        beds = data[state_filter & location_filter]
        if len(beds):
            return beds.iloc[0][self.Fields.MAX_BED_COUNT]

        return None
