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

from libs.datasets.timeseries import MultiRegionTimeseriesDataset


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
    def calculate(self, df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a DataFrame with LOCATION_ID index and DATE columns.

        Args:
            df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Must contain at least one real value for each of self.columns.
            delta_df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Values are 7-day deltas. Must contain at least one real value for each of self.columns.
        """
        pass


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

    def calculate(self, df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
        assert delta_df.columns.names == [CommonFields.DATE]
        assert delta_df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        # delta_df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label so operators such as `/` are calculated for each
        # region/state and date.
        return delta_df.loc[self._numerator, :] / delta_df.loc[self._denominator, :]


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

    def calculate(self, df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
        assert df.columns.names == [CommonFields.DATE]
        assert df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        # df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label
        return df.loc[self._column, :]


TEST_POSITIVITY_METHODS = (
    # HACK: For now we assume TEST_POSITIVITY came from CMS and use that as the name that gets plumbed into provenance.
    PassThruMethod("CMSTesting", CommonFields.TEST_POSITIVITY),
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

    # A MultiRegionTimeseriesDataset with exactly one column, TEST_POSITIVITY, the best available
    # method for each region.
    test_positivity: MultiRegionTimeseriesDataset

    @staticmethod
    def run(
        dataset_in: MultiRegionTimeseriesDataset,
        methods: Sequence[Method] = TEST_POSITIVITY_METHODS,
        diff_days: int = 7,
        recent_days: int = 14,
    ) -> "AllMethods":
        """Runs `methods` on `dataset_in` and returns the results or raises a TestPositivityException."""
        relevant_columns = AllMethods._list_columns(
            AllMethods._methods_with_columns_available(methods, dataset_in.data.columns)
        )
        if not relevant_columns:
            raise NoMethodsWithRelevantColumns()

        input_long = dataset_in.timeseries_long(relevant_columns).set_index(
            [PdFields.VARIABLE, CommonFields.LOCATION_ID, CommonFields.DATE]
        )[PdFields.VALUE]
        dates = input_long.index.get_level_values(CommonFields.DATE)
        if dates.empty:
            raise NoRealTimeseriesValuesException()
        start_date = dates.min()
        end_date = dates.max()
        input_date_range = pd.date_range(start=start_date, end=end_date)
        recent_date_range = pd.date_range(end=end_date, periods=recent_days).intersection(
            input_date_range
        )
        input_wide = (
            input_long.unstack(CommonFields.DATE)
            .reindex(columns=input_date_range)
            .rename_axis(columns=CommonFields.DATE)
        )

        methods_with_data = AllMethods._methods_with_columns_available(
            methods, input_wide.index.get_level_values(PdFields.VARIABLE).unique()
        )
        if not methods_with_data:
            raise NoColumnsWithDataException()

        # This calculates the difference only when the cumulative value is a real value `diff_days` apart.
        # It looks like our input data has few or no holes so this works well enough.
        diff_df = input_wide.diff(periods=diff_days, axis=1)

        all_wide = (
            pd.concat(
                {
                    method.name: method.calculate(input_wide, diff_df)
                    for method in methods_with_data
                },
                names=[PdFields.VARIABLE],
            )
            .reorder_levels([CommonFields.LOCATION_ID, PdFields.VARIABLE])
            # Drop empty timeseries
            .dropna("index", "all")
            .sort_index()
        )

        method_cat_type = pd.CategoricalDtype(
            categories=[method.name for method in methods], ordered=True
        )

        has_recent_data = all_wide.loc[:, recent_date_range].notna().any(axis=1)
        all_recent_data = all_wide.loc[has_recent_data, :].reset_index()
        all_recent_data[PdFields.VARIABLE] = all_recent_data[PdFields.VARIABLE].astype(
            method_cat_type
        )
        first = all_recent_data.groupby(CommonFields.LOCATION_ID).first()
        provenance = first[PdFields.VARIABLE].astype(str).rename(PdFields.PROVENANCE)
        provenance.index = pd.MultiIndex.from_product(
            [provenance.index, [CommonFields.TEST_POSITIVITY]],
            names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
        )
        positivity = first.drop(columns=[PdFields.VARIABLE])

        test_positivity = MultiRegionTimeseriesDataset.from_timeseries_df(
            positivity.stack().rename(CommonFields.TEST_POSITIVITY).reset_index(),
            provenance=provenance,
        )

        return AllMethods(all_methods_timeseries=all_wide, test_positivity=test_positivity)

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
        self.all_methods_timeseries.to_csv(
            csv_path, date_format="%Y-%m-%d", index=True, float_format="%.05g",
        )


def run_and_maybe_join_columns(
    mrts: timeseries.MultiRegionTimeseriesDataset, log
) -> timeseries.MultiRegionTimeseriesDataset:
    """Calculates test positivity and joins it with the input, if successful."""
    try:
        test_positivity_results = AllMethods.run(mrts)
    except TestPositivityException:
        log.exception("test_positivity failed")
        return mrts
    else:
        # We overwrite the test_positivity column if it exists.
        return mrts.drop_column_if_present(CommonFields.TEST_POSITIVITY).join_columns(
            test_positivity_results.test_positivity
        )
