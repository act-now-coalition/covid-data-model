from dataclasses import dataclass
from typing import Any
from typing import Collection
from typing import Dict, Type, List, NewType
import functools
import pathlib
from typing import Optional
from typing import Union

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from typing_extensions import final

from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.sources.hhs_hospital_dataset import HHSHospitalDataset
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.sources.usa_facts import UsaFactsDataSource
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.cms_testing_dataset import CMSTestingDataset
from libs.datasets.sources.can_scraper_state_providers import CANScraperStateProviders
from libs.datasets.sources.cdc_testing_dataset import CDCTestingDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.hhs_testing_dataset import HHSTestingDataset
from libs.datasets.sources.can_location_page_urls import CANLocationPageURLS
from libs.datasets.sources.cdc_vaccine_dataset import CDCVaccinesDataset
from libs.pipeline import Region

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


# structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
from libs.pipeline import RegionMask

_log = structlog.get_logger()


class RegionLatestNotFound(IndexError):
    """Requested region's latest values not found in combined data"""

    pass


RegionMaskOrRegions = NewType("RegionMaskOrRegions", Union[RegionMask, Region, Collection[Region]])


@final
@dataclass(frozen=True)
class DataSourceAndRegionMasks:
    """Represents a DataSource class and include/exclude region masks.

    Instances of this class can be used in the same places where the DataSource type (not
    instances of the DataSource) are used.

    Using this class depends on an existing source of all location_ids. Currently it depends on
    the existing combined data and is used to produce a new combined data. A recursive data
    dependency isn't great but works and we don't otherwise have a stable source of locations to
    use for filtering. Fixing this could be part of
    https://trello.com/c/2Pa4DMyu/836-trim-down-the-timeseriesindexfields
    """

    data_source_cls: Type[data_source.DataSource]
    include: Optional[RegionMaskOrRegions] = None
    exclude: Optional[RegionMaskOrRegions] = None

    @property
    def EXPECTED_FIELDS(self):
        """Returns the same property of the wrapped DataSource class."""
        return self.data_source_cls.EXPECTED_FIELDS

    @property
    def SOURCE_NAME(self):
        """Returns the same property of the wrapped DataSource class."""
        return self.data_source_cls.SOURCE_NAME

    def make_dataset(self) -> MultiRegionDataset:
        """Returns the dataset of the wrapped DataSource class, with a subset of the regions."""
        dataset = self.data_source_cls.make_dataset()

        def _get_location_ids(region_mask_or_regions: RegionMaskOrRegions,) -> Collection[str]:
            if isinstance(region_mask_or_regions, RegionMask):
                return region_mask_to_location_ids(region_mask_or_regions)
            elif isinstance(region_mask_or_regions, Region):
                return [region_mask_or_regions.location_id]
            else:
                return [r.location_id for r in region_mask_or_regions]

        if self.include:
            dataset = dataset.get_locations_subset(_get_location_ids(self.include))
        if self.exclude:
            dataset = dataset.remove_locations(_get_location_ids(self.exclude))
        return dataset


def datasource_regions(
    data_source_cls: Type[data_source.DataSource],
    include: Optional[RegionMaskOrRegions] = None,
    *,
    exclude: Optional[RegionMaskOrRegions] = None,
) -> DataSourceAndRegionMasks:
    assert include or exclude
    return DataSourceAndRegionMasks(data_source_cls, include=include, exclude=exclude)


FeatureDataSourceMap = NewType(
    "FeatureDataSourceMap",
    Dict[FieldName, List[Union[DataSourceAndRegionMasks, Type[data_source.DataSource]]]],
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
    CommonFields.CASES: [CANScraperStateProviders, UsaFactsDataSource, NYTimesDataset,],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CovidTrackingDataSource],
    CommonFields.CUMULATIVE_ICU: [CovidTrackingDataSource],
    CommonFields.CURRENT_HOSPITALIZED: [
        CANScraperStateProviders,
        CovidTrackingDataSource,
        TexasHospitalizations,
        HHSHospitalDataset,
    ],
    CommonFields.CURRENT_ICU: [
        CANScraperStateProviders,
        CovidTrackingDataSource,
        TexasHospitalizations,
        HHSHospitalDataset,
        datasource_regions(HHSHospitalDataset, RegionMask(states=["TX"])),
    ],
    CommonFields.CURRENT_ICU_TOTAL: [HHSHospitalDataset],
    CommonFields.CURRENT_VENTILATED: [CovidTrackingDataSource],
    CommonFields.DEATHS: [CANScraperStateProviders, UsaFactsDataSource, NYTimesDataset],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [HHSHospitalDataset],
    CommonFields.ICU_BEDS: [CANScraperStateProviders, HHSHospitalDataset],
    CommonFields.NEGATIVE_TESTS: [CovidTrackingDataSource, HHSTestingDataset],
    CommonFields.POSITIVE_TESTS: [CovidTrackingDataSource, HHSTestingDataset],
    CommonFields.TOTAL_TESTS: [CovidTrackingDataSource],
    # STAFFED_BEDS isn't used right now. Disable to ease refactoring.
    # CommonFields.STAFFED_BEDS: [CANScraperStateProviders, CovidCountyDataDataSource],
    CommonFields.POSITIVE_TESTS_VIRAL: [CANScraperStateProviders, CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_VIRAL: [CANScraperStateProviders, CovidTrackingDataSource],
    CommonFields.POSITIVE_CASES_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_PEOPLE_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL: [CovidTrackingDataSource],
    CommonFields.TEST_POSITIVITY_14D: [CMSTestingDataset],
    CommonFields.TEST_POSITIVITY_7D: [CDCTestingDataset],
    CommonFields.VACCINES_DISTRIBUTED: [CANScraperStateProviders, CDCVaccinesDataset],
    CommonFields.VACCINATIONS_INITIATED: [CANScraperStateProviders, CDCVaccinesDataset],
    CommonFields.VACCINATIONS_COMPLETED: [CANScraperStateProviders, CDCVaccinesDataset],
}

ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.AGGREGATE_LEVEL: [FIPSPopulation],
    CommonFields.COUNTRY: [FIPSPopulation],
    CommonFields.COUNTY: [FIPSPopulation],
    CommonFields.FIPS: [FIPSPopulation],
    CommonFields.STATE: [FIPSPopulation],
    CommonFields.POPULATION: [FIPSPopulation],
    # TODO(michael): We don't really trust the CCM bed numbers and would ideally remove them entirely.
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.ICU_BEDS: [CovidCareMapBeds, HHSHospitalDataset],
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
) -> MultiRegionDataset:
    filename = dataset_pointer.form_filename(DatasetType.MULTI_REGION)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return MultiRegionDataset.read_from_pointer(pointer)


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


def region_mask_to_location_ids(region_mask: RegionMask) -> Collection[str]:
    """Uses the existing combined dataset to transform `region_mask` into a list of location_id."""
    us_static = load_us_timeseries_dataset().static

    rows_key = dataset_utils.make_rows_key(
        us_static, aggregation_level=region_mask.level, states=region_mask.states,
    )

    return us_static.loc[rows_key, :].index
