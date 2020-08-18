from enum import Enum
from itertools import chain
from typing import Dict, Type, List, NewType, Mapping, MutableMapping, Tuple
import functools
import pathlib
import pandas as pd
import structlog
from structlog.threadlocal import tmp_bind

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets.data_source import DataSource
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.nha_hospitalization import NevadaHospitalAssociationData
from libs.datasets.sources.cds_dataset import CDSDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets import dataset_filter
from libs import us_state_abbrev

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


# structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
_log = structlog.get_logger()


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
ALL_TIMESERIES_FEATURE_DEFINITION: FeatureDataSourceMap = {
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [],
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CUMULATIVE_ICU: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CURRENT_HOSPITALIZED: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_HOSPITALIZED_TOTAL: [],
    CommonFields.CURRENT_ICU: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_ICU_TOTAL: [CovidCountyDataDataSource],
    CommonFields.CURRENT_VENTILATED: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
    ],
    CommonFields.DEATHS: [NYTimesDataset],
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: [CovidCountyDataDataSource],
    CommonFields.ICU_BEDS: [CovidCountyDataDataSource],
    CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: [],
    CommonFields.LICENSED_BEDS: [],
    CommonFields.MAX_BED_COUNT: [],
    CommonFields.NEGATIVE_TESTS: [CDSDataset, CovidCountyDataDataSource, CovidTrackingDataSource],
    CommonFields.POSITIVE_TESTS: [CDSDataset, CovidCountyDataDataSource, CovidTrackingDataSource],
    CommonFields.RECOVERED: [],
    CommonFields.STAFFED_BEDS: [CovidCountyDataDataSource],
}

ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    **ALL_TIMESERIES_FEATURE_DEFINITION,
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.CURRENT_HOSPITALIZED: [
        CovidCountyDataDataSource,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
        TexasHospitalizations,
    ],
    CommonFields.CURRENT_HOSPITALIZED_TOTAL: [NevadaHospitalAssociationData],
    CommonFields.ICU_BEDS: [CovidCountyDataDataSource, CovidCareMapBeds],
    CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
    CommonFields.LICENSED_BEDS: [CovidCareMapBeds],
    CommonFields.MAX_BED_COUNT: [CovidCareMapBeds],
    CommonFields.POPULATION: [FIPSPopulation],
    CommonFields.STAFFED_BEDS: [CovidCountyDataDataSource, CovidCareMapBeds],
}


US_STATES_FILTER = dataset_filter.DatasetFilter(
    country="USA", states=list(us_state_abbrev.ABBREV_US_STATE.keys())
)


@functools.lru_cache(None)
def load_us_timeseries_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
    before=None,
    previous_commit=False,
    commit: str = None,
) -> timeseries.TimeseriesDataset:
    filename = dataset_pointer.form_filename(DatasetType.TIMESERIES)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(before=before, previous_commit=previous_commit, commit=commit)


@functools.lru_cache(None)
def load_us_latest_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
    before: str = None,
    previous_commit: bool = False,
    commit: str = None,
) -> latest_values_dataset.LatestValuesDataset:

    filename = dataset_pointer.form_filename(DatasetType.LATEST)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(before=before, previous_commit=previous_commit, commit=commit)


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


def build_from_sources(
    target_dataset_cls: Type[dataset_base.DatasetBase],
    loaded_data_sources: Mapping[str, DataSource],
    feature_definition_config: FeatureDataSourceMap,
    filter: dataset_filter.DatasetFilter,
):
    """Builds a combined dataset from a feature definition.

    Args:
        target_dataset_cls: Type of the returned combined dataset.
        loaded_data_sources: Dictionary mapping source name to a DataSource object
        feature_definition_config: Dictionary mapping an output field to the
            data source classes that will be used to pull values from.
        filter: A dataset filters applied to the datasets before
            assembling features.
    """

    feature_definition = {
        field_name: [cls.SOURCE_NAME for cls in classes]
        for field_name, classes in feature_definition_config.items()
        if classes
    }

    datasets: MutableMapping[str, pd.DataFrame] = {}
    for source_name in chain.from_iterable(feature_definition.values()):
        source = loaded_data_sources[source_name]
        if target_dataset_cls == TimeseriesDataset:
            datasets[source_name] = filter.apply(source.timeseries()).indexed_data()
        else:
            assert target_dataset_cls == LatestValuesDataset
            datasets[source_name] = filter.apply(source.latest_values()).indexed_data()

    data, provenance = _build_data_and_provenance(feature_definition, datasets)
    return target_dataset_cls(
        data.reset_index(), provenance=provenance_wide_metrics_to_series(provenance, _log)
    )


class Override(Enum):
    """How data sources override each other when combined."""

    # For each <fips, date>, if a row in a higher priority datasource exists it is used, even
    # for columns with NaN. This is the unexpected behavior from before July 16th that blends
    # timeseries from different sources into a single output timeseries.
    BY_ROW = 1
    # For each <fips, variable>, use the entire timeseries of the highest priority datasource
    # with at least one real (not NaN) value.
    BY_TIMESERIES = 2
    # For each <fips, variable, date>, use the highest priority datasource that has a real
    # (not NaN) value.
    BY_TIMESERIES_POINT = 3  # Deprecated


def _build_data_and_provenance(
    feature_definitions: Mapping[str, List[str]],
    datasource_dataframes: Mapping[str, pd.DataFrame],
    override=Override.BY_TIMESERIES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fips_indexed = dataset_utils.fips_index_geo_data(pd.concat(datasource_dataframes.values()))

    # Inspired by pd.Series.combine_first(). Create a new index which is a union of all the input dataframe
    # index.
    dataframes = list(datasource_dataframes.values())
    new_index = dataframes[0].index
    index_names = dataframes[0].index.names
    for df in dataframes[1:]:
        assert index_names == df.index.names
        new_index = new_index.union(df.index)
    # The following is a failed attempt at a performance optimization. The merge operation spends most of its
    # time boxing datetime values, something that in theory doesn't need to happen for an DatetimeIndex but
    # I've been unable to make that happen when the datetimes are part of a MultiIndex.
    # if index_names == ["fips", "date"]:
    #     arrays = [new_index.get_level_values(0), pd.to_datetime(new_index.get_level_values(1))]
    #     new_index = MultiIndex.from_arrays(arrays=arrays, names=index_names)
    # elif index_names == ["fips"]:
    #     pass
    # else:
    #     raise ValueError("bad new_index.names")

    data, provenance = _merge_data(
        datasource_dataframes, feature_definitions, _log, new_index, override
    )

    if not fips_indexed.empty:
        # See https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#joining-with-two-multiindexes
        data = data.join(fips_indexed, on=["fips"], how="left")

    return data, provenance


def _merge_data(datasource_dataframes, feature_definitions, log, new_index, override):
    if override is Override.BY_ROW:
        return _merge_data_by_row(datasource_dataframes, feature_definitions, log, new_index)
    if override is not Override.BY_TIMESERIES:
        raise ValueError("Invalid override")

    # Not going BY_ROW so reindex the inputs now to avoid reindexing for each field below.
    datasource_dataframes = {
        name: df.reindex(new_index, copy=False) for name, df in datasource_dataframes.items()
    }
    # Build feature columns from feature_definitions.
    data = pd.DataFrame(index=new_index)
    provenance = pd.DataFrame(index=new_index)
    for field_name, data_source_names in feature_definitions.items():
        log.info("Working field", field=field_name)
        field_out = None
        field_provenance = pd.Series(index=new_index, dtype="object")
        # Go through the data sources, starting with the highest priority.
        for datasource_name in reversed(data_source_names):
            with tmp_bind(log, dataset_name=datasource_name, field=field_name) as log:
                datasource_field_in = datasource_dataframes[datasource_name][field_name]
                if field_out is None:
                    # Copy all values from the highest priority input to the output
                    field_provenance.loc[pd.notna(datasource_field_in)] = datasource_name
                    field_out = datasource_field_in
                else:
                    field_out_has_ts = field_out.groupby(
                        level=[CommonFields.FIPS], sort=False
                    ).transform(lambda x: x.notna().any())
                    copy_field_in = (~field_out_has_ts) & pd.notna(datasource_field_in)
                    # Copy from datasource_field_in only on rows where all rows of field_out with that FIPS are NaN.
                    field_provenance.loc[copy_field_in] = datasource_name
                    field_out = field_out.where(~copy_field_in, datasource_field_in)
                dups = field_out.index.duplicated(keep=False)
                if dups.any():
                    log.error("Found duplicates in index")
                    raise ValueError()  # This is bad, somehow the input /still/ has duplicates
        data.loc[:, field_name] = field_out
        provenance.loc[:, field_name] = field_provenance
    return data, provenance


def _merge_data_by_row(datasource_dataframes, feature_definitions, log, new_index):
    # Override.BY_ROW needs to preserve the rows datasource_dataframes

    # Build feature columns from feature_definitions.
    data = pd.DataFrame(index=new_index)
    provenance = pd.DataFrame(index=new_index)
    for field_name, data_source_names in feature_definitions.items():
        log.info("Working field", field=field_name)
        field_out = None
        # A list of Series, that when concatanted, will match 1-1 with the series in field_out
        field_provenance = []
        # Go through the data sources, starting with the highest priority.
        for datasource_name in reversed(data_source_names):
            with tmp_bind(log, dataset_name=datasource_name, field=field_name) as log:
                datasource_field_in = datasource_dataframes[datasource_name][field_name]
                if field_out is None:
                    # Copy all values from the highest priority input to the output
                    field_out = datasource_field_in
                    field_provenance.append(
                        pd.Series(index=datasource_field_in.index, dtype="object").fillna(
                            datasource_name
                        )
                    )
                else:
                    # Copy from datasource_field_in rows that are not yet in field_out
                    this_not_in_result = ~datasource_field_in.index.isin(field_out.index)
                    values_to_append = datasource_field_in.loc[this_not_in_result]
                    field_out = field_out.append(values_to_append)
                    field_provenance.append(
                        pd.Series(index=values_to_append.index, dtype="object").fillna(
                            datasource_name
                        )
                    )
                dups = field_out.index.duplicated(keep=False)
                if dups.any():
                    log.error("Found duplicates in index")
                    raise ValueError()  # This is bad, somehow the input /still/ has duplicates
        data.loc[:, field_name] = field_out
        provenance.loc[:, field_name] = pd.concat(field_provenance)
    return data, provenance


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
