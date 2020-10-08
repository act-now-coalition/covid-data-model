from dataclasses import dataclass
from itertools import chain
from typing import Any
from typing import Dict, Type, List, NewType, Mapping, MutableMapping, Tuple
import functools
import pathlib
from typing import Optional

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import data_source
from libs.datasets import dataset_pointer
from libs.datasets.data_source import DataSource
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets import latest_values_dataset
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations
from libs.datasets.sources.test_and_trace import TestAndTraceData
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.cds_dataset import CDSDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets import dataset_filter
from libs import us_state_abbrev
from libs.pipeline import Region

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


# structlog makes it very easy to bind extra attributes to `log` as it is passed down the stack.
_log = structlog.get_logger()


class RegionLatestNotFound(IndexError):
    """Requested region's latest values not found in combined data"""

    pass


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
    CommonFields.CASES: [NYTimesDataset],
    CommonFields.CONTACT_TRACERS_COUNT: [TestAndTraceData],
    CommonFields.CUMULATIVE_HOSPITALIZED: [CDSDataset, CovidTrackingDataSource],
    CommonFields.CUMULATIVE_ICU: [CDSDataset, CovidTrackingDataSource],
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
    CommonFields.NEGATIVE_TESTS: [CDSDataset, CovidCountyDataDataSource, CovidTrackingDataSource],
    CommonFields.POSITIVE_TESTS: [CDSDataset, CovidCountyDataDataSource, CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS: [CovidTrackingDataSource],
    CommonFields.STAFFED_BEDS: [CovidCountyDataDataSource],
    CommonFields.POSITIVE_TESTS_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_VIRAL: [CovidTrackingDataSource],
    CommonFields.POSITIVE_CASES_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TESTS_PEOPLE_VIRAL: [CovidTrackingDataSource],
    CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL: [CovidTrackingDataSource],
}

ALL_FIELDS_FEATURE_DEFINITION: FeatureDataSourceMap = {
    **ALL_TIMESERIES_FEATURE_DEFINITION,
    CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [CovidCareMapBeds],
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
) -> MultiRegionTimeseriesDataset:
    filename = dataset_pointer.form_filename(DatasetType.MULTI_REGION)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(before=before, previous_commit=previous_commit, commit=commit)


@functools.lru_cache(None)
def load_us_latest_dataset(
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
) -> latest_values_dataset.LatestValuesDataset:
    us_timeseries = load_us_timeseries_dataset(pointer_directory=pointer_directory)
    # Returned object contains a DataFrame with a LOCATION_ID column
    return LatestValuesDataset(us_timeseries.latest_data_with_fips.reset_index())


def get_county_name(region: Region) -> Optional[str]:
    return load_us_timeseries_dataset().get_one_region(region).latest[CommonFields.COUNTY]


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
    # TODO(tom): When LatestValuesDataset is retired return only a MultiRegionTimeseriesDataset
    return target_dataset_cls(
        data.reset_index(), provenance=provenance_wide_metrics_to_series(provenance, _log)
    )


def _build_data_and_provenance(
    feature_definitions: Mapping[str, List[str]], datasource_dataframes: Mapping[str, pd.DataFrame]
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
    new_index = new_index.unique().sort_values()
    assert new_index.is_unique
    assert new_index.is_monotonic_increasing
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

    data, provenance = _merge_data(datasource_dataframes, feature_definitions, _log, new_index)

    if not fips_indexed.empty:
        # See https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#joining-with-two-multiindexes
        data = data.join(fips_indexed, on=["fips"], how="left")

    return data, provenance


def _merge_data(datasource_dataframes, feature_definitions, log, new_index):
    # For each <fips, variable>, use the entire timeseries of the highest priority datasource
    # with at least one real (not NaN) value.

    # Reindex the inputs now to avoid reindexing for each field below.
    datasource_dataframes = {
        name: df.reindex(new_index, copy=False) for name, df in datasource_dataframes.items()
    }
    # Build feature columns from feature_definitions.
    data = pd.DataFrame(index=new_index)
    provenance = pd.DataFrame(index=new_index)
    for field_name, data_source_names in feature_definitions.items():
        datasource_series = [
            (name, datasource_dataframes[name][field_name]) for name in data_source_names
        ]
        field_out, field_provenance = _merge_field(
            datasource_series, new_index, log.bind(field=field_name),
        )
        data.loc[:, field_name] = field_out
        provenance.loc[:, field_name] = field_provenance
    return data, provenance


def _merge_field(
    datasource_series: List[Tuple[str, pd.Series]], new_index, log: structlog.BoundLoggerBase
):
    log.info("Working field")
    field_out = None
    field_provenance = pd.Series(index=new_index, dtype="object")
    # Go through the data sources, starting with the highest priority.
    for datasource_name, datasource_field_in in reversed(datasource_series):
        log_datasource = log.bind(dataset_name=datasource_name)
        if field_out is None:
            # Copy all values from the highest priority input to the output
            field_provenance.loc[pd.notna(datasource_field_in)] = datasource_name
            field_out = datasource_field_in
        else:
            field_out_has_ts = field_out.groupby(level=[CommonFields.FIPS], sort=False).transform(
                lambda x: x.notna().any()
            )
            copy_field_in = (~field_out_has_ts) & pd.notna(datasource_field_in)
            # Copy from datasource_field_in only on rows where all rows of field_out with that FIPS are NaN.
            field_provenance.loc[copy_field_in] = datasource_name
            field_out = field_out.where(~copy_field_in, datasource_field_in)
        dups = field_out.index.duplicated(keep=False)
        if dups.any():
            log_datasource.error("Found duplicates in index")
            raise ValueError()  # This is bad, somehow the input /still/ has duplicates
    return field_out, field_provenance


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

    region: Region

    timeseries: OneRegionTimeseriesDataset

    @staticmethod
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


def get_subset_regions(exclude_county_999: bool, **kwargs) -> List[Region]:
    us_latest = load_us_latest_dataset()
    us_subset = us_latest.get_subset(exclude_county_999=exclude_county_999, **kwargs)
    return [Region.from_fips(fips) for fips in us_subset.data[CommonFields.FIPS].unique()]
