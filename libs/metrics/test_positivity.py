from abc import ABC, abstractmethod
import dataclasses
import pathlib
from itertools import chain
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Set

import structlog

import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields, FieldName
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import timeseries

from libs.datasets.timeseries import MultiRegionDataset


_log = structlog.get_logger()


class Method(ABC):
    """A method of calculating test positivity"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def columns(self) -> Set[FieldName]:
        pass

    @abstractmethod
    def calculate(self, dataset: MultiRegionDataset, diff_days: int) -> MultiRegionDataset:
        """Calculate a DataFrame with LOCATION_ID index and DATE columns.

        Args:
            df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Must contain at least one real value for each of self.columns.
            delta_df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Values are 7-day deltas. Must contain at least one real value for each of self.columns.
        """
        pass


def _append_variable_index_level(df: pd.DataFrame, field_name: FieldName) -> None:
    """Add level `VARIABLE` to df.index with constant value `field_name`, in-place"""
    assert df.index.names == [CommonFields.LOCATION_ID]
    # From https://stackoverflow.com/a/40636392
    df.index = pd.MultiIndex.from_product(
        [df.index, [field_name]], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )


def _make_output_dataset(
    dataset_in: MultiRegionDataset,
    default_provenance: str,
    wide_date_df: pd.DataFrame,
    output_metric: FieldName,
) -> MultiRegionDataset:
    """Returns a dataset with `wide_date_df` in a column named `output_metric` and provenance
    information copied from `dataset_in`.

    Args:
        dataset_in: original MultiRegionDataset. TODO(tom): copy provenance information from this.
        default_provenance: value to put in provenance when no other value is available
        wide_date_df: the timeseries output, copied to the returned dataset
        output_metric: wide_date_df is copied to this metric/column in the returned dataset
    """
    assert wide_date_df.index.names == [CommonFields.LOCATION_ID]
    # Drop all-NA timeseries now, as done in from_timeseries_wide_dates_df. This makes sure
    # `locations` is used to build provenance information for only timeseries in the returned
    # MultiRegionDataset.
    wide_date_df = wide_date_df.dropna("rows", "all")
    locations = wide_date_df.index.get_level_values(CommonFields.LOCATION_ID)
    _append_variable_index_level(wide_date_df, output_metric)

    # TODO(tom): Add support for copying per-location provenance information from dataset_in.
    #  Something like:
    # assert dataset.provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
    # provenance = dataset.provenance.loc[locations, self._column]
    assert dataset_in.provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]

    # Make sure every timeseries has something in `provenance`.
    provenance_index = pd.MultiIndex.from_product(
        [locations, [output_metric]], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )
    provenance = pd.Series([], name=PdFields.PROVENANCE, dtype="object").reindex(
        provenance_index, fill_value=default_provenance
    )
    return MultiRegionDataset.from_timeseries_wide_dates_df(wide_date_df).add_provenance_series(
        provenance
    )


@dataclasses.dataclass
class DivisionMethod(Method):
    """A method of calculating test positivity by dividing a numerator by a denominator"""

    _name: str
    _numerator: FieldName
    _denominator: FieldName

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Set[FieldName]:
        return {self._numerator, self._denominator}

    def calculate(self, dataset: MultiRegionDataset, diff_days: int) -> MultiRegionDataset:
        delta_df = (
            dataset.timeseries_wide_dates()
            .reorder_levels([PdFields.VARIABLE, CommonFields.LOCATION_ID])
            .diff(periods=diff_days, axis=1)
        )
        assert delta_df.columns.names == [CommonFields.DATE]
        assert delta_df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        # delta_df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label so operators such as `/` are calculated for each
        # region/state and date.
        wide_date_df = delta_df.loc[self._numerator, :] / delta_df.loc[self._denominator, :]
        return _make_output_dataset(dataset, self._name, wide_date_df, CommonFields.TEST_POSITIVITY)


@dataclasses.dataclass
class PassThruMethod(Method):
    """A method of calculating test positivity by passing through a column directly"""

    _name: str
    _column: FieldName

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Set[FieldName]:
        return {self._column}

    def calculate(self, dataset: MultiRegionDataset, diff_days: int) -> MultiRegionDataset:
        df = dataset.timeseries_wide_dates().reorder_levels(
            [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        )
        assert df.columns.names == [CommonFields.DATE]
        assert df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        # df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label
        wide_date_df = df.loc[self._column, :]
        # Optional optimization: The following likely adds the variable/field/column name back in
        # to the index which was just taken out. Consider skipping reindexing.
        return _make_output_dataset(dataset, self._name, wide_date_df, CommonFields.TEST_POSITIVITY)


TEST_POSITIVITY_METHODS = (
    # HACK: For now we assume TEST_POSITIVITY_7D came from CDC numbers while
    # TEST_POSITIVITY_14D came from CMS.
    PassThruMethod("CDCTesting", CommonFields.TEST_POSITIVITY_7D),
    PassThruMethod("CMSTesting", CommonFields.TEST_POSITIVITY_14D),
    DivisionMethod(
        "positiveTestsViral_totalTestsViral",
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.TOTAL_TESTS_VIRAL,
    ),
)


class TestPositivityException(Exception):
    pass


class NoColumnsWithDataException(TestPositivityException):
    pass


class NoMethodsWithRelevantColumns(TestPositivityException):
    pass


class NoRealTimeseriesValuesException(TestPositivityException):
    pass


@dataclasses.dataclass
class AllMethods:
    """The result of calculating all test positivity methods for all regions"""

    # Test positivity calculated in all valid methods for each region
    all_methods_timeseries: pd.DataFrame

    # A MultiRegionDataset with exactly one column, TEST_POSITIVITY, the best available
    # method for each region.
    test_positivity: MultiRegionDataset

    @staticmethod
    def run(
        dataset_in: MultiRegionDataset,
        methods: Sequence[Method] = TEST_POSITIVITY_METHODS,
        diff_days: int = 7,
        recent_days: int = 14,
    ) -> "AllMethods":
        """Runs `methods` on `dataset_in` and returns the results or raises a TestPositivityException."""
        relevant_columns = AllMethods._list_columns(
            AllMethods._methods_with_columns_available(methods, dataset_in.timeseries.columns)
        )
        if not relevant_columns:
            raise NoMethodsWithRelevantColumns()

        input_wide = dataset_in.timeseries_wide_dates()
        if input_wide.empty:
            raise NoRealTimeseriesValuesException()
        dates = input_wide.columns.get_level_values(CommonFields.DATE)
        # The oldest date that is considered recent/not-stale. If recent_days is 1 then this is
        # the most recent day in the input.
        assert recent_days >= 1
        recent_date_cutoff = dates.max() + pd.to_timedelta(1 - recent_days, "D")

        methods_with_data = AllMethods._methods_with_columns_available(
            methods, input_wide.index.get_level_values(PdFields.VARIABLE).unique()
        )
        if not methods_with_data:
            raise NoColumnsWithDataException()

        calculated_dataset_map = {
            timeseries.DatasetName(method.name): method.calculate(dataset_in, diff_days)
            for method in methods_with_data
        }
        calculated_dataset_recent_map = {
            name: ds.drop_stale_timeseries(recent_date_cutoff)
            for name, ds in calculated_dataset_map.items()
        }
        # Make a dataset object with one metric, containing for each region the timeseries
        # from the highest priority dataset that has recent data.
        test_positivity = timeseries.combined_datasets(
            calculated_dataset_recent_map,
            {CommonFields.TEST_POSITIVITY: list(calculated_dataset_map.keys())},
            {},
        )
        # For debugging create a DataFrame with the calculated timeseries of all methods, including
        # timeseries that are not recent.
        all_datasets_df = pd.concat(
            {name: ds.timeseries_wide_dates() for name, ds in calculated_dataset_map.items()},
            names=[PdFields.DATASET, CommonFields.LOCATION_ID, PdFields.VARIABLE],
        )
        all_methods = (
            all_datasets_df.xs(CommonFields.TEST_POSITIVITY, level=PdFields.VARIABLE)
            .sort_index()
            .reindex(columns=dates)
            .reorder_levels([CommonFields.LOCATION_ID, PdFields.DATASET])
        )
        return AllMethods(all_methods_timeseries=all_methods, test_positivity=test_positivity)

    @staticmethod
    def _methods_with_columns_available(
        methods_in: Sequence[Method], available_columns: Sequence[str]
    ) -> Sequence[Method]:
        available_columns_set = set(available_columns)
        return [m for m in methods_in if m.columns <= available_columns_set]

    @staticmethod
    def _list_columns(methods: Iterable[Method]) -> List[FieldName]:
        """Returns unsorted list of columns in given Method objects."""
        return list(set(chain.from_iterable(method.columns for method in methods)))

    def write(self, csv_path: pathlib.Path):
        # TODO(tom): Change all_method_timeseries to be something like Mapping[DatasetName,
        #  MultiRegionDataset] and use shared code to write it.
        df = self.all_methods_timeseries.dropna("columns", "all")
        start_date = df.columns.min()
        end_date = df.columns.max()
        date_range = pd.date_range(start=start_date, end=end_date)
        df = df.reindex(columns=date_range).rename_axis(None, axis="columns")
        df.columns = df.columns.strftime("%Y-%m-%d")

        df.sort_index().to_csv(csv_path, index=True, float_format="%.05g")


def run_and_maybe_join_columns(
    mrts: timeseries.MultiRegionDataset, log
) -> timeseries.MultiRegionDataset:
    """Calculates test positivity and joins it with the input, if successful."""
    try:
        test_positivity_results = AllMethods.run(mrts)
    except TestPositivityException:
        log.exception("test_positivity failed")
        return mrts

    return mrts.join_columns(test_positivity_results.test_positivity)
