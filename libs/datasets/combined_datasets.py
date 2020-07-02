from typing import Dict, Type, List, NewType
import logging
import pandas as pd
import structlog
from structlog.threadlocal import tmp_bind

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import data_source
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.sources.cmdc import CmdcDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.jhu_dataset import JHUDataset
from libs.datasets.sources.nha_hospitalization import NevadaHospitalAssociationData
from libs.datasets.sources.cds_dataset import CDSDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets import dataset_filter
from libs.datasets import dataset_cache
from libs import us_state_abbrev


_logger = logging.getLogger(__name__)
FeatureDataSourceMap = NewType(
    "FeatureDataSourceMap", Dict[str, List[Type[data_source.DataSource]]]
)

# Below are two instances of feature definitions. These define
# how to assemble values for a specific field.  Right now, we only
# support overlaying values. i.e. a row of
# {CommonFields.POSITIVE_TESTS: [CDSDataset, CovidTrackingDataSource]}
# will first get all values for positive tests in CDSDataset and then overlay any data
# From CovidTracking.
# This is just a start to this sort of definition - in the future, we may want more advanced
# capabilities around what data to apply and how to apply it.
# This structure still hides a lot of what is done under the hood and it's not
# immediately obvious as to the transformations that are or are not applied.
# One way of dealing with this is going from showcasing datasets dependencies
# to showingcasing a dependency graph of transformations.
ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.DEATHS: [NYTimesDataset],
    CommonFields.RECOVERED: [JHUDataset],
    CommonFields.CUMULATIVE_ICU: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CURRENT_ICU: [CmdcDataSource, CovidTrackingDataSource,],
    CommonFields.CURRENT_ICU_TOTAL: [CmdcDataSource],
    CommonFields.CURRENT_HOSPITALIZED_TOTAL: [NevadaHospitalAssociationData],
    CommonFields.CURRENT_HOSPITALIZED: [
        CmdcDataSource,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_VENTILATED: [
        CmdcDataSource,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
    ],
    CommonFields.COUNTY: [FIPSPopulation],
    CommonFields.POPULATION: [FIPSPopulation],
    CommonFields.STAFFED_BEDS: [CmdcDataSource, CovidCareMapBeds],
    CommonFields.LICENSED_BEDS: [CovidCareMapBeds],
    CommonFields.ICU_BEDS: [CmdcDataSource, CovidCareMapBeds],
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.MAX_BED_COUNT: [CovidCareMapBeds],
    CommonFields.POSITIVE_TESTS: [CmdcDataSource, CDSDataset, CovidTrackingDataSource],
    CommonFields.NEGATIVE_TESTS: [CmdcDataSource, CDSDataset, CovidTrackingDataSource],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [CmdcDataSource],
}

ALL_TIMESERIES_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.DEATHS: [NYTimesDataset],
    CommonFields.RECOVERED: [JHUDataset],
    CommonFields.CUMULATIVE_ICU: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CURRENT_ICU: [CmdcDataSource, CovidTrackingDataSource,],
    CommonFields.CURRENT_ICU_TOTAL: [CmdcDataSource],
    CommonFields.CURRENT_HOSPITALIZED: [
        CmdcDataSource,
        CovidTrackingDataSource,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_HOSPITALIZED_TOTAL: [],
    CommonFields.CURRENT_VENTILATED: [
        CmdcDataSource,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
    ],
    CommonFields.STAFFED_BEDS: [CmdcDataSource],
    CommonFields.LICENSED_BEDS: [],
    CommonFields.MAX_BED_COUNT: [],
    CommonFields.ICU_BEDS: [CmdcDataSource],
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [],
    CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: [],
    CommonFields.POSITIVE_TESTS: [CmdcDataSource, CDSDataset, CovidTrackingDataSource],
    CommonFields.NEGATIVE_TESTS: [CmdcDataSource, CDSDataset, CovidTrackingDataSource],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [CmdcDataSource],
}

US_STATES_FILTER = dataset_filter.DatasetFilter(
    country="USA", states=list(us_state_abbrev.ABBREV_US_STATE.keys())
)


@dataset_cache.cache_dataset_on_disk(TimeseriesDataset)
def build_timeseries_with_all_fields() -> TimeseriesDataset:
    return build_combined_dataset_from_sources(
        TimeseriesDataset, ALL_TIMESERIES_FEATURE_DEFINITION,
    )


@dataset_cache.cache_dataset_on_disk(TimeseriesDataset)
def build_us_timeseries_with_all_fields() -> TimeseriesDataset:
    return build_combined_dataset_from_sources(
        TimeseriesDataset, ALL_TIMESERIES_FEATURE_DEFINITION, filters=[US_STATES_FILTER]
    )


@dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
def build_us_latest_with_all_fields() -> LatestValuesDataset:
    return build_combined_dataset_from_sources(
        LatestValuesDataset, ALL_FIELDS_FEATURE_DEFINITION, filters=[US_STATES_FILTER]
    )


def get_us_latest_for_state(state) -> dict:
    """Gets latest values for a given state."""
    us_latest = build_us_latest_with_all_fields()
    return us_latest.get_record_for_state(state)


def get_us_latest_for_fips(fips) -> dict:
    """Gets latest values for a given fips code."""
    us_latest = build_us_latest_with_all_fields()
    return us_latest.get_record_for_fips(fips)


def _remove_padded_nans(df, columns):
    if df[columns].isna().all().all():
        return df.loc[[False] * len(df), :].reset_index(drop=True)

    first_valid_index = min(df[column].first_valid_index() for column in columns)
    last_valid_index = max(df[column].last_valid_index() for column in columns)
    df = df.iloc[first_valid_index : last_valid_index + 1]
    return df.reset_index(drop=True)


def get_timeseries_for_fips(
    fips: str, columns: List = None, min_range_with_some_value: bool = False
) -> TimeseriesDataset:
    """Gets timeseries for a specific FIPS code.

    Args:
        fips: FIPS code.  Can be county (5 character) or state (2 character) code.
        columns: List of columns, apart from `TimeseriesDataset.INDEX_FIELDS`, to include.
        min_range_with_some_value: If True, removes NaNs that pad values at beginning and end of
            timeseries. Only applicable when columns are specified.

    Returns: Timeseries for fips
    """

    state_ts = build_us_timeseries_with_all_fields().get_subset(None, fips=fips)
    if columns:
        subset = state_ts.data.loc[:, TimeseriesDataset.INDEX_FIELDS + columns].reset_index(
            drop=True
        )

        if min_range_with_some_value:
            subset = _remove_padded_nans(subset, columns)

        state_ts = TimeseriesDataset(subset)

    return state_ts


def get_timeseries_for_state(
    state: str, columns: List = None, min_range_with_some_value: bool = False
) -> TimeseriesDataset:
    """Gets timeseries for a specific state abbreviation.

    Args:
        state: 2-letter state code
        columns: List of columns, apart from `TimeseriesDataset.INDEX_FIELDS`, to include.
        min_range_with_some_value: If True, removes NaNs that pad values at beginning and end of
            timeseries. Only applicable when columns are specified.

    Returns: Timeseries for state
    """

    state_ts = build_us_timeseries_with_all_fields().get_subset(AggregationLevel.STATE, state=state)
    if columns:
        subset = state_ts.data.loc[:, TimeseriesDataset.INDEX_FIELDS + columns].reset_index(
            drop=True
        )

        if min_range_with_some_value:
            subset = _remove_padded_nans(subset, columns)

        state_ts = TimeseriesDataset(subset)

    return state_ts


def load_data_sources(
    feature_definition_config,
) -> Dict[Type[data_source.DataSource], data_source.DataSource]:

    data_source_classes = []
    for classes in feature_definition_config.values():
        data_source_classes.extend(classes)
    loaded_data_sources = {}

    for data_source_class in data_source_classes:
        for data_source_cls in data_source_classes:
            # only load data source once
            if data_source_cls not in loaded_data_sources:
                loaded_data_sources[data_source_cls] = data_source_cls.local()

    return loaded_data_sources


def build_combined_dataset_from_sources(
    target_dataset_cls: Type[dataset_base.DatasetBase],
    feature_definition_config: FeatureDataSourceMap,
    filters: List[dataset_filter.DatasetFilter] = None,
):
    """Builds a combined dataset from a feature definition.

    Args:
        target_dataset_cls: Target dataset class.
        feature_definition_config: Dictionary mapping an output field to the
            data sources that will be used to pull values from.
        filters: A list of dataset filters applied to the datasets before
            assembling features.
    """
    loaded_data_sources = load_data_sources(feature_definition_config)

    # Convert data sources to instances of `target_data_cls`.
    intermediate_datasets = {
        data_source_cls: target_dataset_cls.build_from_data_source(source)
        for data_source_cls, source in loaded_data_sources.items()
    }

    # Apply filters to datasets.
    for key in intermediate_datasets:
        dataset = intermediate_datasets[key]
        for data_filter in filters or []:
            dataset = data_filter.apply(dataset)
        intermediate_datasets[key] = dataset

    # Build feature columns from feature_definition_config.
    data = pd.DataFrame({})
    # structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
    log = structlog.get_logger()
    for field, data_source_classes in feature_definition_config.items():
        for data_source_cls in data_source_classes:
            dataset = intermediate_datasets[data_source_cls]
            with tmp_bind(log, dataset_name=data_source_cls.SOURCE_NAME, field=field) as log:
                try:
                    data = dataset_utils.fill_fields_with_data_source(
                        log, data, dataset.data, target_dataset_cls.INDEX_FIELDS, [field],
                    )
                except Exception:
                    log.exception("trying to fill fields")
                    raise

    return target_dataset_cls(data)
