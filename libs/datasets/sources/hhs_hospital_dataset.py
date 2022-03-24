import dataclasses
from functools import lru_cache
from libs import pipeline

from datapublic.common_fields import CommonFields
import pandas as pd
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets import data_source
from libs.datasets.timeseries import MultiRegionDataset


# Early data is noisy due to lack of reporting, etc. so we just trim it off.
DEFAULT_START_DATE = "2020-09-01"
CUSTOM_START_DATES = {
    "02": "2020-10-06",  # Alaska
    "04": "2020-09-02",  # Arizona
    "15": "2020-10-10",  # Hawaii
    "16": "2020-10-17",  # Idaho
    "19": "2020-09-05",  # Iowa
    "21": "2020-10-15",  # Kentucky
    "28": "2020-11-11",  # Mississippi
    "34": "2020-09-01",  # New Jersey
    "38": "2020-11-03",  # North Dakota
    "46": "2020-11-02",  # South Dakota
    "47": "2020-09-20",  # Tennessee
    "53": "2020-10-25",  # Washington
}

# Mapping of ScraperVariable.variable_name to common fields.
FIELD_MAPPING = {
    "adult_icu_beds_capacity": CommonFields.ICU_BEDS,
    "adult_icu_beds_in_use": CommonFields.CURRENT_ICU_TOTAL,
    "adult_icu_beds_in_use_covid": CommonFields.CURRENT_ICU,
    "hospital_beds_capacity": CommonFields.STAFFED_BEDS,
    "hospital_beds_in_use": CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
    "hospital_beds_in_use_covid": CommonFields.CURRENT_HOSPITALIZED,
}


def make_hhs_variable(can_scraper_field, common_field, measurement):
    """Helper to create a ScraperVariable, since we have a bunch of variables to deal with and
    two different measurements to deal with (for county and state data)."""
    return ccd_helpers.ScraperVariable(
        variable_name=can_scraper_field,
        measurement=measurement,
        provider="hhs",
        unit="beds",
        common_field=common_field,
    )


# TODO(michael): HHSHospitalStateDataset and HHSHospitalCountyDataset must be separate classes
# because they both have ScraperVariables that target the same CommonFields. This happens becaues
# the state level and county level data use the same variable names but different measurements
# ("current" and "rolling_average_7_day"). This causes CanScraperBase to overwrite itself as it's
# extracting the variables. Using separate Dataset classes solves this.


class HHSHospitalStateDataset(data_source.CanScraperBase):
    """Data source for HHS state-level hospital data."""

    SOURCE_TYPE = "HHSHospitalState"

    VARIABLES = [
        make_hhs_variable(can_field, common_field, "current")
        for can_field, common_field in FIELD_MAPPING.items()
    ] + [
        # hospital admissions are daily at the state-level and weekly at the
        # county-level so we specify them separate from FIELD_MAPPING.
        ccd_helpers.ScraperVariable(
            variable_name="hospital_admissions_covid",
            measurement="new",
            unit="people",
            provider="hhs",
            common_field=CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID,
        ),
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


class HHSHospitalCountyDataset(data_source.CanScraperBase):
    """Data source for HHS county-level hospital data."""

    SOURCE_TYPE = "HHSHospitalCounty"

    VARIABLES = [
        make_hhs_variable(can_field, common_field, "rolling_average_7_day")
        for can_field, common_field in FIELD_MAPPING.items()
    ] + [
        # hospital admissions are daily at the state-level and weekly at the
        # county-level so we specify them separate from FIELD_MAPPING.
        ccd_helpers.ScraperVariable(
            variable_name="hospital_admissions_covid",
            measurement="new_7_day",
            unit="people",
            provider="hhs",
            common_field=CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID,
        )
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


def modify_dataset(ds: MultiRegionDataset) -> MultiRegionDataset:
    """Post-processes the can scraper data as necessary."""
    ts_copy = ds.timeseries.copy()

    # Filter out some of the early noisy data.
    filtered_ds = dataclasses.replace(
        ds, timeseries=filter_early_data(ts_copy), timeseries_bucketed=None
    )

    # Ensure ICU_BEDS is included in the static fields as well.
    return filtered_ds.latest_in_static(field=CommonFields.ICU_BEDS)


def filter_early_data(df):
    """Filters out early data for each location based on DEFAULT_START_DATE and
    CUSTOM_START_DATES."""
    keep_rows = df.index.get_level_values(CommonFields.DATE) >= pd.to_datetime(DEFAULT_START_DATE)
    df = df.loc[keep_rows]

    for (fips, start_date) in CUSTOM_START_DATES.items():
        location_id = pipeline.fips_to_location_id(fips)
        keep_rows = (df.index.get_level_values(CommonFields.LOCATION_ID) != location_id) | (
            df.index.get_level_values(CommonFields.DATE) >= pd.to_datetime(start_date)
        )
        df = df.loc[keep_rows]

    return df
