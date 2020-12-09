from dataclasses import dataclass
from typing import Any
from typing import Dict, Type, List, NewType
import functools
import pathlib
from typing import Optional

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName

from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.cms_testing_dataset import CMSTestingDataset
from libs.datasets.sources.cdc_testing_dataset import CDCTestingDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.hhs_testing_dataset import HHSTestingDataset
from libs.datasets.sources.can_location_page_urls import CANLocationPageURLS
from libs.pipeline import Region

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


# structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
_log = structlog.get_logger()


class RegionLatestNotFound(IndexError):
    """Requested region's latest values not found in combined data"""

    pass


FeatureDataSourceMap = NewType(
    "FeatureDataSourceMap", Dict[FieldName, List[Type[data_source.DataSource]]]
)


# Below are two instances of feature definitions. These define
# how to assemble values for a specific field.  Right now, we only
# support overlaying values. i.e. a row of
# {CommonFields.POSITIVE_TESTS: [HHSTestingDataset, CovidTrackingDataSource]}
# will first get all values for positive tests in HHSTestingDataset and then overlay any data
# From CovidTracking.
# This is just a start to this sort of definition - in the future, we may want more advanced
# capabilities around what data to apply and how to apply it.
# This structure still hides a lot of what is done under the hood and it's not
# immediately obvious as to the transformations that are or are not applied.
# One way of dealing with this is going from showcasing datasets dependencies
# to showingcasing a dependency graph of transformations.
ALL_TIMESERIES_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CovidTrackingDataSource],
    CommonFields.CUMULATIVE_ICU: [CovidTrackingDataSource],
    CommonFields.CURRENT_HOSPITALIZED: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_ICU: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_ICU_TOTAL: [CovidCountyDataDataSource],
    CommonFields.CURRENT_VENTILATED: [CovidCountyDataDataSource, CovidTrackingDataSource,],
    CommonFields.DEATHS: [NYTimesDataset],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [CovidCountyDataDataSource],
    CommonFields.ICU_BEDS: [CovidCountyDataDataSource],
    CommonFields.NEGATIVE_TESTS: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        HHSTestingDataset,
    ],
    CommonFields.POSITIVE_TESTS: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        HHSTestingDataset,
    ],
    CommonFields.TOTAL_TESTS: [CovidTrackingDataSource],
    # STAFFED_BEDS isn't used right now. Disable to ease refactoring.
    # CommonFields.STAFFED_BEDS: [CovidCountyDataDataSource],
    CommonFields.POSITIVE_TESTS_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_VIRAL: [CovidTrackingDataSource],
    CommonFields.POSITIVE_CASES_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_PEOPLE_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL: [CovidTrackingDataSource],
    CommonFields.TEST_POSITIVITY_14D: [CMSTestingDataset],
    CommonFields.TEST_POSITIVITY_7D: [CDCTestingDataset],
}

ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.AGGREGATE_LEVEL: [FIPSPopulation, CovidCountyDataDataSource],
    CommonFields.COUNTRY: [FIPSPopulation, CovidCountyDataDataSource],
    CommonFields.COUNTY: [FIPSPopulation, CovidCountyDataDataSource],
    CommonFields.FIPS: [FIPSPopulation, CovidCountyDataDataSource],
    CommonFields.STATE: [FIPSPopulation, CovidCountyDataDataSource],
    CommonFields.POPULATION: [FIPSPopulation],
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.ICU_BEDS: [CovidCountyDataDataSource, CovidCareMapBeds],
    CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.LICENSED_BEDS: [CovidCareMapBeds],
    CommonFields.MAX_BED_COUNT: [CovidCareMapBeds],
    # STAFFED_BEDS isn't used right now. Disable to ease refactoring.
    # CommonFields.STAFFED_BEDS: [CovidCountyDataDataSource, CovidCareMapBeds],
    CommonFields.CAN_LOCATION_PAGE_URL: [CANLocationPageURLS],
}


@functools.lru_cache(None)
def load_us_timeseries_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
    before=None,
    previous_commit=False,
    commit: str = None,
) -> MultiRegionDataset:
    filename = dataset_pointer.form_filename(DatasetType.MULTI_REGION)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(before=before, previous_commit=previous_commit, commit=commit)


def get_county_name(region: Region) -> Optional[str]:
    return load_us_timeseries_dataset().get_county_name(region=region)


def provenance_wide_metrics_to_series(wide: pd.DataFrame, log) -> pd.Series:
    """Transforms a DataFrame of provenances with a variable columns to Series with one row per variable.

    Args:
        wide: DataFrame with a row for each fips-date and a column containing the data source for each variable.
            FIPS must be a named index. DATE, if present, must be a named index.

    Returns: A Series of string data source values with fips and variable in the index. In the case
        of multiple sources for a timeseries a warning is logged and the values are joined by ';'.
    """
    assert CommonFields.FIPS in wide.index.names
    assert CommonFields.FIPS not in wide.columns
    assert CommonFields.DATE not in wide.columns
    columns_without_timeseries_point_keys = set(wide.columns) - set(COMMON_FIELDS_TIMESERIES_KEYS)
    long_unindexed = (
        wide.reset_index()
        .melt(id_vars=[CommonFields.FIPS], value_vars=columns_without_timeseries_point_keys)
        .drop_duplicates()
        .dropna(subset=["value"])
    )
    fips_var_grouped = long_unindexed.groupby([CommonFields.FIPS, "variable"], sort=False)["value"]
    dups = fips_var_grouped.transform("size") > 1
    if dups.any():
        log.warning("Multiple rows for a timeseries", bad_data=long_unindexed[dups])
    # https://stackoverflow.com/a/17841321/341400
    joined = fips_var_grouped.agg(lambda col: ";".join(col))
    return joined


@dataclass(frozen=True)
class RegionalData:
    """Identifies a geographical area and wraps access to `combined_datasets` of it."""

    # TODO(tom): Now that OneRegionTimeseriesDataset contains `region` consider replacing
    # uses of this class with it.

    region: Region

    timeseries: OneRegionTimeseriesDataset

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def from_region(region: Region) -> "RegionalData":
        us_timeseries = load_us_timeseries_dataset()
        region_timeseries = us_timeseries.get_one_region(region)
        return RegionalData(region=region, timeseries=region_timeseries)

    @property
    def latest(self) -> Dict[str, Any]:
        return self.timeseries.latest

    @property
    def population(self) -> int:
        """Gets the population for this region."""
        return self.latest[CommonFields.POPULATION]

    @property  # TODO(tom): Change to cached_property when we're using Python 3.8
    def display_name(self) -> str:
        county = self.latest[CommonFields.COUNTY]
        state = self.latest[CommonFields.STATE]
        if county:
            return f"{county}, {state}"
        return state
