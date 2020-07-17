from enum import Enum
from itertools import chain
from typing import Dict, Type, List, NewType, Mapping, Optional, MutableMapping
import functools
import pathlib
import os
import logging
import pandas as pd
import structlog
from structlog.threadlocal import tmp_bind

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import DatasetType
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

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


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
    CommonFields.CURRENT_ICU: [CmdcDataSource, CovidTrackingDataSource],
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
    CommonFields.RECOVERED: [JHUDataset],
    CommonFields.CUMULATIVE_ICU: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CURRENT_ICU: [CmdcDataSource, CovidTrackingDataSource],
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
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.DEATHS: [NYTimesDataset],
}

US_STATES_FILTER = dataset_filter.DatasetFilter(
    country="USA", states=list(us_state_abbrev.ABBREV_US_STATE.keys())
)


@dataset_cache.cache_dataset_on_disk(TimeseriesDataset)
def build_us_timeseries_with_all_fields(skip_cache=False) -> TimeseriesDataset:
    return _build_combined_dataset_from_sources(
        TimeseriesDataset, ALL_TIMESERIES_FEATURE_DEFINITION, filter=US_STATES_FILTER,
    )


@dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
def build_us_latest_with_all_fields(skip_cache=False) -> LatestValuesDataset:
    return _build_combined_dataset_from_sources(
        LatestValuesDataset, ALL_FIELDS_FEATURE_DEFINITION, filter=US_STATES_FILTER,
    )


@functools.lru_cache(None)
def load_us_timeseries_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
) -> timeseries.TimeseriesDataset:
    """Loads US TimeseriesDataset for """
    filename = dataset_pointer.form_filename(DatasetType.TIMESERIES)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset()


@functools.lru_cache(None)
def load_us_latest_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
) -> latest_values_dataset.LatestValuesDataset:
    filename = dataset_pointer.form_filename(DatasetType.LATEST)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset()


def get_us_latest_for_fips(fips) -> dict:
    """Gets latest values for a given state or county fips code."""
    us_latest = load_us_latest_dataset()
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

    state_ts = load_us_timeseries_dataset().get_subset(None, fips=fips)
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

    state_ts = load_us_timeseries_dataset().get_subset(AggregationLevel.STATE, state=state)
    if columns:
        subset = state_ts.data.loc[:, TimeseriesDataset.INDEX_FIELDS + columns].reset_index(
            drop=True
        )

        if min_range_with_some_value:
            subset = _remove_padded_nans(subset, columns)

        state_ts = TimeseriesDataset(subset)

    return state_ts


def _build_combined_dataset_from_sources(
    target_dataset_cls: Type[dataset_base.DatasetBase],
    feature_definition_config: FeatureDataSourceMap,
    filter: dataset_filter.DatasetFilter,
):
    """Builds a combined dataset from a feature definition.

    Args:
        target_dataset_cls: Target dataset class.
        feature_definition_config: Dictionary mapping an output field to the
            data sources that will be used to pull values from.
        filters: A list of dataset filters applied to the datasets before
            assembling features.
    """
    loaded_data_sources = {
        data_source_cls: data_source_cls.local()
        for data_source_cls in set(chain.from_iterable(feature_definition_config.values()))
    }

    # Convert data sources to instances of `target_data_cls`.
    intermediate_datasets = {
        data_source_cls.SOURCE_NAME: filter.apply(target_dataset_cls.build_from_data_source(source))
        for data_source_cls, source in loaded_data_sources.items()
    }

    datasets: MutableMapping[str, pd.DataFrame] = {}
    for name, dataset_obj in intermediate_datasets.items():
        data_with_index = dataset_obj.data.set_index(target_dataset_cls.COMMON_INDEX_FIELDS)
        if data_with_index.index.duplicated(keep=False).any():
            raise ValueError(f"Duplicate in {name}")
        datasets[name] = data_with_index
        # If any duplicates slip in the following code may help you debug them:
        # https://stackoverflow.com/a/34297689
        # datasets[name] = data_with_index.loc[~data_with_index.duplicated(keep="first"), :]
        # datasets[key] = dataset_obj.data.groupby(target_dataset_cls.COMMON_INDEX_FIELDS).first() fails
        # due to <NA>s: cannot convert to 'float64'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype.

    feature_definition = {
        field_name: [cls.SOURCE_NAME for cls in classes]
        for field_name, classes in feature_definition_config.items()
        if classes
    }

    return target_dataset_cls(_build_dataframe(feature_definition, datasets).reset_index())


class Override(Enum):
    """How data sources override each other when combined."""

    # For each <fips, date>, if a row in a higher priority datasource exists it is used, even
    # for columns with NaN. This is the unexpected behavior from before July 16th that blends
    # timeseries from different sources and XXX
    BY_ROW = 1
    # For each <fips, variable>, use the entire timeseries of the highest priority datasource
    # with at least one real (not NaN) value.
    BY_TIMESERIES = 2
    # For each <fips, variable, date>, use the highest priority datasource that has a real
    # (not NaN) value.
    NAN = 3


def _build_dataframe(
    feature_definitions: Mapping[str, List[str]],
    datasource_dataframes: Mapping[str, pd.DataFrame],
    override=Override.BY_ROW,
) -> pd.DataFrame:
    # structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
    log = structlog.get_logger()

    preserve_columns = [CommonFields.AGGREGATE_LEVEL, CommonFields.STATE, CommonFields.COUNTY]
    all_identifiers = pd.concat(
        df.reset_index().loc[
            :, [CommonFields.FIPS] + list(df.columns.intersection(preserve_columns))
        ]
        for df in datasource_dataframes.values()
    ).drop_duplicates()
    fips_indexed = all_identifiers.set_index(CommonFields.FIPS, verify_integrity=True)

    # Inspired by pd.Series.combine_first()
    dataframes = list(datasource_dataframes.values())
    new_index = dataframes[0].index
    for df in dataframes[1:]:
        new_index = new_index.union(df.index)
    if override in (Override.BY_TIMESERIES, Override.NAN):
        datasource_dataframes = {
            name: df.reindex(new_index, copy=False) for name, df in datasource_dataframes.items()
        }
    log.info("reindexed dataframes")

    # Build feature columns from feature_definitions.
    data = pd.DataFrame(index=new_index)
    for field, data_source_names in feature_definitions.items():
        log.info("working field", field=field)
        field_series = None
        for datasource_name in reversed(data_source_names):
            with tmp_bind(log, dataset_name=datasource_name, field=field) as log:
                this_series = datasource_dataframes[datasource_name][field]
                if field_series is None:
                    field_series = this_series
                elif override == Override.BY_TIMESERIES:
                    keep_higher_priority = field_series.groupby(
                        level=[CommonFields.FIPS]
                    ).transform(lambda x: x.notna().any())
                    field_series = field_series.where(keep_higher_priority, this_series)
                elif override == Override.NAN:
                    field_series = field_series.where(pd.notna(field_series), this_series)
                else:
                    assert override == Override.BY_ROW
                    this_not_in_result = ~this_series.index.isin(field_series.index)
                    field_series = field_series.append(this_series.loc[this_not_in_result])
                dups = field_series.index.duplicated(keep=False)
                if dups.any():
                    log.error("Found duplicates in index")
                    raise ValueError()
        data.loc[:, field] = field_series

    if not fips_indexed.empty:
        # See https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#joining-with-two-multiindexes
        data = data.join(fips_indexed, on=["fips"], how="left")

    return data
