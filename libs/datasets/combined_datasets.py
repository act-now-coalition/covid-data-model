from dataclasses import dataclass
from typing import Any
from typing import Dict, Type, List, NewType
import functools
import pathlib
from typing import Mapping
from typing import Optional
from typing import TypeVar
from typing import Union

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from typing_extensions import final

from libs.datasets import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets import manual_filter
from libs.datasets.custom_aggregations import ALL_NYC_REGIONS
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.sources.hhs_hospital_dataset import HHSHospitalStateDataset
from libs.datasets.sources.hhs_hospital_dataset import HHSHospitalCountyDataset
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.can_scraper_local_dashboard_providers import CANScraperCountyProviders
from libs.datasets.sources.can_scraper_local_dashboard_providers import CANScraperStateProviders
from libs.datasets.sources.can_scraper_usafacts import CANScraperUSAFactsProvider
from libs.datasets.sources.cdc_testing_dataset import CDCHistoricalTestingDataset
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.hhs_testing_dataset import HHSTestingDataset
from libs.datasets.sources.can_location_page_urls import CANLocationPageURLS
from libs.datasets.sources.cdc_vaccine_dataset import CDCVaccinesDataset
from libs.datasets.sources.cdc_new_vaccine_counties_dataset import CDCNewVaccinesCountiesDataset
from libs.pipeline import Region
from libs.pipeline import RegionMask
from libs.pipeline import RegionMaskOrRegion
from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


# structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
_log = structlog.get_logger()


@final
@dataclass(frozen=True)
class DataSourceAndRegionMasks:
    """Represents a DataSource class and include/exclude region masks.

    Use function `datasource_regions` to create instance of this class.

    Instances of this class can be used in the same places where a DataSource subclass/type (not
    instances of the DataSource) are used. In other words DataSourceAndRegionMasks instances
    implement the same interface as the DataSource type.

    Using this class depends on an existing source of all location_ids. Currently it depends on
    the existing combined data and is used to produce a new combined data. A recursive data
    dependency isn't great but works and we don't otherwise have a stable source of locations to
    use for filtering. Fixing this could be part of
    https://trello.com/c/2Pa4DMyu/836-trim-down-the-timeseriesindexfields
    """

    data_source_cls: Type[data_source.DataSource]
    include: List[RegionMaskOrRegion]
    exclude: List[RegionMaskOrRegion]
    manual_filter_config: Optional[Mapping]

    @property
    def EXPECTED_FIELDS(self):
        """Implements the same interface as the wrapped DataSource class."""
        return self.data_source_cls.EXPECTED_FIELDS

    @property
    def SOURCE_TYPE(self):
        """Implements the same interface as the wrapped DataSource class."""
        return self.data_source_cls.SOURCE_TYPE

    def make_dataset(self) -> MultiRegionDataset:
        """Returns the dataset of the wrapped DataSource class, with a subset of the regions.

        This method implements the same interface as the wrapped DataSource class.
        """
        dataset = self.data_source_cls.make_dataset()

        dataset, _ = dataset.partition_by_region(include=self.include, exclude=self.exclude)
        if self.manual_filter_config:
            dataset = manual_filter.run(dataset, self.manual_filter_config)
        return dataset


T = TypeVar("T")


def to_list(list_or_scalar: Union[None, T, List[T]]) -> List[T]:
    """Returns a list which may be empty, contain the single non-list parameter or the parameter."""
    if isinstance(list_or_scalar, List):
        return list_or_scalar
    elif list_or_scalar is None:
        return []
    else:
        return [list_or_scalar]


def datasource_regions(
    data_source_cls: Type[data_source.DataSource],
    include: Union[None, RegionMaskOrRegion, List[RegionMaskOrRegion]] = None,
    *,
    exclude: Union[None, RegionMaskOrRegion, List[RegionMaskOrRegion]] = None,
    manual_filter: Optional[Mapping] = None,
) -> DataSourceAndRegionMasks:
    """Creates an instance of the `DataSourceAndRegionMasks` class."""
    assert include or exclude, (
        "At least one of include or exclude must be set. If neither are "
        "needed use the DataSource class directly."
    )
    return DataSourceAndRegionMasks(
        data_source_cls,
        include=to_list(include),
        exclude=to_list(exclude),
        manual_filter_config=manual_filter,
    )


FeatureDataSourceMap = NewType(
    "FeatureDataSourceMap",
    Dict[FieldName, List[Union[DataSourceAndRegionMasks, Type[data_source.DataSource]]]],
)

KANSAS_CITY_COUNTIES = [
    # Cass County
    Region.from_fips("29037"),
    # Clay County
    Region.from_fips("29047"),
    # Jackson County
    Region.from_fips("29095"),
    # Platte County
    Region.from_fips("29165"),
]

JOPLIN_COUNTIES = [
    # Jasper
    Region.from_fips("29097"),
    # Newton County
    Region.from_fips("29145"),
]

# 11/15/2021: NE periodically stops reporting county-level case data.
# When this occurs, we access Health Department level data via the state's API,
# and we dissaggregate this data to the county level. In these cases, for NE
# counties we want to use the CANScraperStateProviders data.
# When NE does report county data, we prefer to pull from the NYT,
# so this region mask might not always be in use.
NE_COUNTIES = RegionMask(AggregationLevel.COUNTY, states=["NE"])

# NY Times has cases and deaths for all boroughs aggregated into 36061 / New York County.
# Remove all the NYC data so that USAFacts (which reports each borough separately) is used.
# Remove counties in MO that overlap with Kansas City and Joplin because we don't handle the
# reporting done by city, as documented at
# https://github.com/nytimes/covid-19-data/blob/master/README.md#geographic-exceptions
NYTimesDatasetWithoutExceptions = datasource_regions(
    NYTimesDataset, exclude=[*ALL_NYC_REGIONS, *KANSAS_CITY_COUNTIES, *JOPLIN_COUNTIES],
)

CANScraperUSAFactsProviderWithoutNe = datasource_regions(
    CANScraperUSAFactsProvider, exclude=[NE_COUNTIES]
)

CDCVaccinesCountiesDataset = datasource_regions(
    CDCVaccinesDataset, RegionMask(AggregationLevel.COUNTY)
)

CDCVaccinesStatesAndNationDataset = datasource_regions(
    CDCVaccinesDataset, [RegionMask(AggregationLevel.STATE), RegionMask(AggregationLevel.COUNTRY)]
)

CDC_COUNTY_EXCLUSIONS = [
    # Glacier County, MT - reports 99.9% of 12+ population vaccinated [7/2021]
    Region.from_fips("30035"),
    # Santa Cruz County, AZ - reports 99.9% of 12+ population vaccinated [7/2021]
    Region.from_fips("04023"),
    # Arecibo Municipio, PR - reports 99.9% of 12+ population vaccinated [7/2021]
    Region.from_fips("72013"),
    # Bristol Bay Borough, AK - reports 99.9% of 12+ population vaccinated [7/2021]
    Region.from_fips("02060"),
    # Culebra Municipio, PR - reports 99.9% of 12+ population vaccinated [7/2021]
    Region.from_fips("72049"),
]

# Excluded for a variety of reasons (lower overall coverage, data irregularities, etc.)
CDC_STATE_EXCLUSIONS = RegionMask(
    states=[
        # CA - Data irregularities including 99.9% of 12+ vaccinated in San Diego
        "CA",
        # GA - Very low coverage.
        "GA",
        "NM",
        # PA - Data irregularities including very high rates in several counties
        # (e.g. Montgomery, Chester)
        "PA",
        # SD - Missing a lot of counties and 1st dose data.
        "SD",
        # VA - Very low coverage.
        "VA",
        "VT",
        "WV",
        # CDC reporting 0.1% vaccinated
        "HI",
        # TODO(sean) 10/23/21: Block TX CDC data until we have a chance to properly QA it
        "TX",
    ]
)
CDCNewVaccinesCompletedCountiesWithoutExceptions = datasource_regions(
    CDCNewVaccinesCountiesDataset, exclude=[CDC_STATE_EXCLUSIONS, *CDC_COUNTY_EXCLUSIONS]
)

# CDC is missing 1st dose data for many NE counties and where it's not missing it, it's
# often just barely above 2nd dose data, making it suspicious. So we're just blocking
# it outright.
CDCNewVaccinesInitiatedCountiesWithoutExceptions = datasource_regions(
    CDCNewVaccinesCountiesDataset,
    exclude=[CDC_STATE_EXCLUSIONS, *CDC_COUNTY_EXCLUSIONS, NE_COUNTIES],
)

# Excludes FL counties for vaccine fields. See
# https://trello.com/c/0nVivEMt/1435-fix-florida-data-scraper
CANScraperStateProvidersWithoutFLCounties = datasource_regions(
    CANScraperStateProviders, exclude=RegionMask(level=AggregationLevel.COUNTY, states=["FL"])
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
    CommonFields.CASES: [
        CANScraperStateProviders,
        CANScraperUSAFactsProvider,
        NYTimesDatasetWithoutExceptions,
    ],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.CURRENT_HOSPITALIZED: [
        CANScraperStateProviders,
        HHSHospitalCountyDataset,
        HHSHospitalStateDataset,
    ],
    CommonFields.CURRENT_ICU: [
        CANScraperStateProviders,
        HHSHospitalCountyDataset,
        HHSHospitalStateDataset,
    ],
    CommonFields.CURRENT_ICU_TOTAL: [HHSHospitalCountyDataset, HHSHospitalStateDataset],
    CommonFields.DEATHS: [
        CANScraperStateProviders,
        CANScraperUSAFactsProvider,
        NYTimesDatasetWithoutExceptions,
    ],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [HHSHospitalCountyDataset, HHSHospitalStateDataset],
    CommonFields.ICU_BEDS: [
        CANScraperStateProviders,
        HHSHospitalCountyDataset,
        HHSHospitalStateDataset,
    ],
    CommonFields.STAFFED_BEDS: [
        CANScraperStateProviders,
        HHSHospitalCountyDataset,
        HHSHospitalStateDataset,
    ],
    CommonFields.NEGATIVE_TESTS: [HHSTestingDataset],
    CommonFields.POSITIVE_TESTS: [HHSTestingDataset],
    CommonFields.POSITIVE_TESTS_VIRAL: [CANScraperStateProviders],
    CommonFields.TOTAL_TESTS_VIRAL: [CANScraperStateProviders],
    CommonFields.TEST_POSITIVITY_7D: [CDCHistoricalTestingDataset],
    CommonFields.VACCINES_DISTRIBUTED: [
        CDCVaccinesCountiesDataset,
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
        CDCVaccinesStatesAndNationDataset,
    ],
    CommonFields.VACCINES_ADMINISTERED: [
        CDCVaccinesCountiesDataset,
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
        CDCVaccinesStatesAndNationDataset,
    ],
    CommonFields.VACCINATIONS_INITIATED: [
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
        CDCVaccinesStatesAndNationDataset,
        CDCNewVaccinesInitiatedCountiesWithoutExceptions,
    ],
    CommonFields.VACCINATIONS_COMPLETED: [
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
        CDCVaccinesStatesAndNationDataset,
        CDCNewVaccinesCompletedCountiesWithoutExceptions,
    ],
    CommonFields.VACCINATIONS_INITIATED_PCT: [
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
    ],
    CommonFields.VACCINATIONS_COMPLETED_PCT: [
        CANScraperStateProvidersWithoutFLCounties,
        CANScraperCountyProviders,
    ],
}

ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.POPULATION: [FIPSPopulation],
    CommonFields.CAN_LOCATION_PAGE_URL: [CANLocationPageURLS],
}


@functools.lru_cache(None)
def load_us_timeseries_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
) -> MultiRegionDataset:
    """Returns all combined data. `load_test_dataset` is more suitable for tests."""
    filename = dataset_pointer.form_filename(DatasetType.MULTI_REGION)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return MultiRegionDataset.read_from_pointer(pointer)


def get_county_name(region: Region) -> Optional[str]:
    return dataset_utils.get_geo_data().at[region.location_id, CommonFields.COUNTY]


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
