from typing import Type, List, Optional, Iterable, Union, TextIO
import pathlib

import structlog
from more_itertools import first

from covidactnow.datapublic import common_df
from libs import us_state_abbrev
import pandas as pd
import numpy as np
from libs.datasets.dataset_utils import AggregationLevel, make_binary_array
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
    COMMON_INDEX_FIELDS = [CommonFields.FIPS]

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

        data = cls._aggregate_new_york_data(data)
        if fill_missing_state:
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data, cls.STATE_GROUP_KEY, AggregationLevel.COUNTY, AggregationLevel.STATE
            ).reset_index()

            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        # Add state fips
        is_state = data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        state_fips = data.loc[is_state, CommonFields.STATE].map(us_state_abbrev.ABBREV_US_FIPS)
        data.loc[is_state, CommonFields.FIPS] = state_fips

        data = cls._calculate_puerto_rico_bed_occupancy_rate(data)

        return cls(data)

    @classmethod
    def _calculate_puerto_rico_bed_occupancy_rate(cls, data):
        is_pr_county = data[CommonFields.FIPS].str.match("72[0-9][0-9][0-9]")
        pr_data = data.loc[is_pr_county.fillna(False)]
        weighted_icu_occupancy = None
        weighted_all_bed_occupancy = None

        if CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE in pr_data.columns:
            licensed_beds = pr_data[CommonFields.LICENSED_BEDS]
            occupancy_rates = pr_data[CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE]
            weighted_all_bed_occupancy = (
                licensed_beds * occupancy_rates
            ).sum() / licensed_beds.sum()

        if CommonFields.ICU_TYPICAL_OCCUPANCY_RATE in pr_data.columns:
            icu_beds = pr_data[CommonFields.ICU_BEDS]
            occupancy_rates = pr_data[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]
            weighted_icu_occupancy = (icu_beds * occupancy_rates).sum() / icu_beds.sum()

        data.loc[
            data[CommonFields.FIPS] == "72", CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE
        ] = weighted_all_bed_occupancy
        data.loc[
            data[CommonFields.FIPS] == "72", CommonFields.ICU_TYPICAL_OCCUPANCY_RATE
        ] = weighted_icu_occupancy

        return data

    @classmethod
    def _aggregate_new_york_data(cls, data):
        # When grouping nyc data, we don't want to the sum to include invalid FIPS.
        nyc_data = data[data[CommonFields.FIPS].isin(custom_aggregations.ALL_NYC_FIPS)]
        if not len(nyc_data):
            return data
        group = cls.STATE_GROUP_KEY
        weighted_all_bed_occupancy = None

        if CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE in data.columns:
            licensed_beds = nyc_data[CommonFields.LICENSED_BEDS]
            occupancy_rates = nyc_data[CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE]
            weighted_all_bed_occupancy = (
                licensed_beds * occupancy_rates
            ).sum() / licensed_beds.sum()
        weighted_icu_occupancy = None
        if CommonFields.ICU_TYPICAL_OCCUPANCY_RATE in data.columns:
            icu_beds = nyc_data[CommonFields.ICU_BEDS]
            occupancy_rates = nyc_data[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]
            weighted_icu_occupancy = (icu_beds * occupancy_rates).sum() / icu_beds.sum()

        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=False
        )

        nyc_fips = custom_aggregations.NEW_YORK_COUNTY_FIPS
        if weighted_all_bed_occupancy:
            data.loc[
                data[CommonFields.FIPS] == nyc_fips, CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE
            ] = weighted_all_bed_occupancy

        if weighted_icu_occupancy:
            data.loc[
                data[CommonFields.FIPS] == nyc_fips, CommonFields.ICU_TYPICAL_OCCUPANCY_RATE
            ] = weighted_icu_occupancy

        return data

    @property
    def county(self):
        return self.get_subset(aggregation_level=AggregationLevel.COUNTY)

    @property
    def state_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only state data."""

        is_state = self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        return self.data[is_state]

    @property
    def county_data(self) -> pd.DataFrame:
        """Returns a new BedsDataset containing only county data."""
        is_county = self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        return self.data[is_county]

    @property
    def states(self):
        return self.data.state.unique()

    @property
    def all_fips(self) -> List[str]:
        return list(self.data.fips.unique())

    def get_subset(
        self,
        aggregation_level=None,
        country=None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        on: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> "LatestValuesDataset":
        rows_binary_array = make_binary_array(
            self.data,
            aggregation_level=aggregation_level,
            country=country,
            fips=fips,
            state=state,
            states=states,
            on=on,
            after=after,
            before=before,
        )
        return self.__class__(self.data.loc[rows_binary_array, :])

    def get_record_for_fips(self, fips) -> dict:
        """Gets all data for a given fips code.

        Args:
            fips: 2 digits for a state or 5 digits for a county

        Returns: Dictionary with all data for a given fips code.
        """
        return first(self.get_subset(fips=fips).yield_records(), default={})

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        """Load CSV Latest Values Dataset."""
        # Cannot use common_df.read_csv as it doesn't support data without a date index field.
        df = pd.read_csv(path_or_buf, dtype={CommonFields.FIPS: str})
        return cls(df)
