from abc import ABC, abstractmethod
import dataclasses
from itertools import chain
from typing import Collection
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import structlog

import pandas as pd
from datapublic.common_fields import CommonFields, FieldName
from datapublic.common_fields import PdFields
from libs import series_utils
from typing_extensions import final

from libs.datasets import timeseries
from libs.datasets.tail_filter import TagField

from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset

_log = structlog.get_logger()


@final
@dataclasses.dataclass(frozen=True)
class MethodOutput:
    """The values returned by Method.caluclate.

    In the grand glorious future we'll have a more generic way to represent multiple datasets and
    can remove this class.
    """

    # A dataset containing all computed outputs. This is used for debugging.
    all_output: MultiRegionDataset
    # A dataset containing only data considered recent according to the parameters of the method.
    recent: MultiRegionDataset


@dataclasses.dataclass
class Method(ABC):
    """A method of calculating test positivity"""

    # This method needs a timeseries to have at least one real value within recent_days for it to
    # be considered recent / not stale. Stale timeseries are dropped.
    recent_days: int = 14

    @property
    @abstractmethod
    def name(self) -> timeseries.DatasetName:
        pass

    @property
    @abstractmethod
    def columns(self) -> Set[FieldName]:
        pass

    @abstractmethod
    def calculate(
        self, dataset: MultiRegionDataset, diff_days: int, most_recent_date: pd.Timestamp
    ) -> MethodOutput:
        """Calculate a DataFrame with LOCATION_ID index and DATE columns.

        Args:
            df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Must contain at least one real value for each of self.columns.
            delta_df: DataFrame with rows having MultiIndex of [VARIABLE, LOCATION_ID] and columns with DATE
                index. Values are 7-day deltas. Must contain at least one real value for each of self.columns.
            most_recent_date: The most recent date in dataset. Timeseries with a real value for
                at least one of the dates within recent_days of this are considered recent.
        """
        pass

    def _remove_stale_regions(
        self, all_output: MultiRegionDataset, most_recent_date: pd.Timestamp
    ) -> MethodOutput:
        """Filters the output for all regions, returning output for all regions and only those
        with recent data."""
        assert self.recent_days >= 1
        # The oldest date that is considered recent/not-stale. If recent_days is 1 then this is
        # the most recent day in the input.
        recent_date_cutoff = most_recent_date + pd.to_timedelta(1 - self.recent_days, "D")
        return MethodOutput(
            all_output=all_output, recent=all_output.drop_stale_timeseries(recent_date_cutoff)
        )


def _append_variable_index_level(df: Union[pd.DataFrame, pd.Series], field_name: FieldName) -> None:
    """Add level `VARIABLE` to df.index with constant value `field_name`, in-place"""
    assert df.index.names == [CommonFields.LOCATION_ID]
    # From https://stackoverflow.com/a/40636392
    df.index = pd.MultiIndex.from_product(
        [df.index, [field_name]], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )


def _make_output_dataset(
    dataset_in: MultiRegionDataset,
    source_columns: Collection[FieldName],
    wide_date_df: pd.DataFrame,
    output_metric: FieldName,
) -> MultiRegionDataset:
    """Returns a dataset with `wide_date_df` in a column named `output_metric` and provenance
    information copied from `dataset_in`.

    Args:
        dataset_in: original MultiRegionDataset.
        source_columns: columns that were a source for the output. tags are copied from these to
          the output.
        wide_date_df: the timeseries output, copied to the returned dataset
        output_metric: wide_date_df is copied to this metric/column in the returned dataset
    """
    assert wide_date_df.index.names == [CommonFields.LOCATION_ID]
    # Drop all-NA timeseries now, as done in from_timeseries_wide_dates_df. This makes sure
    # `locations` is used to build provenance information for only timeseries in the returned
    # MultiRegionDataset.
    wide_date_df = wide_date_df.dropna(axis="index", how="all")
    locations = wide_date_df.index.unique(CommonFields.LOCATION_ID)
    _append_variable_index_level(wide_date_df, output_metric)

    assert dataset_in.tag_all_bucket.index.names == [
        TagField.LOCATION_ID,
        TagField.VARIABLE,
        TagField.TYPE,
    ]
    dataset_out = MultiRegionDataset.from_timeseries_wide_dates_df(wide_date_df, bucketed=False)
    if source_columns:
        source_tags = dataset_in.tag_all_bucket.loc[locations, list(source_columns)].reset_index()
        source_tags[TagField.VARIABLE] = output_metric
        # When there are two source_columns they usually contain the same provenance content.
        # Only keep one copy of it.
        output_tags = source_tags.drop_duplicates(ignore_index=True).copy()
        timeseries.tag_df_add_all_bucket_in_place(output_tags)
        dataset_out = dataset_out.append_tag_df(output_tags)

    return dataset_out


@dataclasses.dataclass
class _DivisionMethodAttributes:
    """Non-default attributes of DivisionMethod, extracted to avoid TypeError.

    The problem and this somewhat ugly solution are described at
    https://stackoverflow.com/q/51575931
    """

    _name: timeseries.DatasetName
    _numerator: FieldName
    _denominator: FieldName


@dataclasses.dataclass
class DivisionMethod(Method, _DivisionMethodAttributes):
    """A method of calculating test positivity by dividing a numerator by a denominator"""

    @property
    def name(self) -> timeseries.DatasetName:
        return self._name

    @property
    def columns(self) -> Set[FieldName]:
        return {self._numerator, self._denominator}

    def calculate(
        self, dataset: MultiRegionDataset, diff_days: int, most_recent_date: pd.Timestamp
    ) -> MethodOutput:
        numerator_delta = dataset.get_timeseries_not_bucketed_wide_dates(self._numerator).diff(
            periods=diff_days, axis=1
        )
        assert numerator_delta.index.names == [CommonFields.LOCATION_ID]
        assert numerator_delta.columns.names == [CommonFields.DATE]

        denomintaor_delta = dataset.get_timeseries_not_bucketed_wide_dates(self._denominator).diff(
            periods=diff_days, axis=1
        )
        assert denomintaor_delta.index.names == [CommonFields.LOCATION_ID]
        assert denomintaor_delta.columns.names == [CommonFields.DATE]

        # `/` is calculated for each region/state and date.
        wide_date_df = numerator_delta / denomintaor_delta

        all_output = _make_output_dataset(
            dataset, self.columns, wide_date_df, CommonFields.TEST_POSITIVITY,
        )
        return self._remove_stale_regions(all_output, most_recent_date)


@dataclasses.dataclass
class _PassThruMethodAttributes:
    """Non-default attributes of PassThruMethod, extracted to avoid TypeError.

    The problem and this somewhat ugly solution are described at
    https://stackoverflow.com/q/51575931
    """

    _name: timeseries.DatasetName
    _column: FieldName


@dataclasses.dataclass
class PassThruMethod(Method, _PassThruMethodAttributes):
    """A method of calculating test positivity by passing through a column directly"""

    @property
    def name(self) -> timeseries.DatasetName:
        return self._name

    @property
    def columns(self) -> Set[FieldName]:
        return {self._column}

    def calculate(
        self, dataset: MultiRegionDataset, diff_days: int, most_recent_date: pd.Timestamp
    ) -> MethodOutput:
        wide_date_df = dataset.get_timeseries_not_bucketed_wide_dates(self._column)
        assert wide_date_df.index.names == [CommonFields.LOCATION_ID]
        assert wide_date_df.columns.names == [CommonFields.DATE]

        # Optional optimization: The following likely adds the variable/field/column name back in
        # to the index which was just taken out. Consider skipping reindexing.

        all_output = _make_output_dataset(
            dataset, self.columns, wide_date_df, CommonFields.TEST_POSITIVITY
        )
        return self._remove_stale_regions(all_output, most_recent_date)


class SmoothedTests(Method):
    """A method of calculating test positivity using smoothed POSITIVE_TEST and NEGATIVE_TESTS."""

    @property
    def name(self) -> timeseries.DatasetName:
        return timeseries.DatasetName("SmoothedTests")

    @property
    def columns(self) -> Set[FieldName]:
        return {CommonFields.POSITIVE_TESTS, CommonFields.NEGATIVE_TESTS}

    def calculate(
        self, dataset: MultiRegionDataset, diff_days: int, most_recent_date: pd.Timestamp
    ) -> MethodOutput:

        positivity_time_series = {}
        # To replicate the behavior of the old code, a region is considered to have recent
        # positivity when the input timeseries (POSITIVE_TESTS and NEGATIVE_TESTS) are recent. The
        # other subclasses of `Method` filter based on the most recent real value in the output
        # timeseries.
        recent_regions = []
        for region, regional_data in dataset.iter_one_regions():
            data = regional_data.date_indexed
            positive_negative_recent = self._has_recent_data(
                data[CommonFields.POSITIVE_TESTS]
            ) and self._has_recent_data(data[CommonFields.NEGATIVE_TESTS])
            series = calculate_test_positivity(regional_data)
            if not series.empty:
                positivity_time_series[region.location_id] = series
                if positive_negative_recent:
                    recent_regions.append(region)

        # Convert dict[location_id, Series] to rows with key as index and value as the row data.
        # See https://stackoverflow.com/a/21005134
        wide_date_df = pd.concat(positivity_time_series, axis=1).T.rename_axis(
            index=CommonFields.LOCATION_ID, columns=CommonFields.DATE
        )

        # Make a dataset with TEST_POSITIVITY for every region where the calculation finished.
        all_output = _make_output_dataset(
            dataset, self.columns, wide_date_df, CommonFields.TEST_POSITIVITY,
        )
        # Make a dataset with the subset of regions having recent input timeseries.
        ds_recent = all_output.get_regions_subset(recent_regions)
        return MethodOutput(all_output=all_output, recent=ds_recent)

    def _has_recent_data(self, series: pd.Series) -> bool:
        """Returns True iff series has recent data relative to today().
        TODO(tom): Replace with something that uses most_recent_date instead of a global clock.
        """
        return series_utils.has_recent_data(
            series, days_back=self.recent_days, required_non_null_datapoints=1
        )


TEST_POSITIVITY_METHODS = (
    # SmoothedTests(recent_days=10),
    # HACK: For now we assume TEST_POSITIVITY_7D came from CDC numbers while
    # TEST_POSITIVITY_14D came from CMS.
    PassThruMethod("CDCTesting", CommonFields.TEST_POSITIVITY_7D),
    # PassThruMethod("CMSTesting", CommonFields.TEST_POSITIVITY_14D),
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
    all_methods_datasets: Mapping[timeseries.DatasetName, MultiRegionDataset]

    # A MultiRegionDataset with exactly one column, TEST_POSITIVITY, the best available
    # method for each region.
    test_positivity: MultiRegionDataset

    @staticmethod
    def run(
        dataset_in: MultiRegionDataset,
        methods: Sequence[Method] = TEST_POSITIVITY_METHODS,
        *,
        diff_days: int = 7,
    ) -> "AllMethods":
        """Runs `methods` on `dataset_in` and returns the results or raises a TestPositivityException."""
        relevant_columns = AllMethods._list_columns(
            AllMethods._methods_with_columns_available(methods, dataset_in.timeseries.columns)
        )
        if not relevant_columns:
            raise NoMethodsWithRelevantColumns()

        input_wide = dataset_in.timeseries_bucketed_wide_dates
        if input_wide.empty:
            raise NoRealTimeseriesValuesException()
        dates = input_wide.columns.unique(CommonFields.DATE)
        most_recent_date = dates.max()

        methods_with_data = AllMethods._methods_with_columns_available(
            methods, input_wide.index.unique(PdFields.VARIABLE)
        )
        if not methods_with_data:
            raise NoColumnsWithDataException()

        method_map = {method.name: method for method in methods_with_data}
        calculated_dataset_map = {
            method_name: method.calculate(dataset_in, diff_days, most_recent_date)
            for method_name, method in method_map.items()
        }
        calculated_dataset_recent_map = {
            name: method_output.recent for name, method_output in calculated_dataset_map.items()
        }
        calculated_dataset_all_map = {
            name: method_output.all_output for name, method_output in calculated_dataset_map.items()
        }
        # HACK: If SmoothedTests is in calculated_dataset_map (that is the MethodOutput returned by
        # `calculate`) then add it again at the end of the map with the Method.all_output. Remember
        # that dict entries remain in the order inserted. This makes # SmoothedTests the final
        # fallback for a location if no other Method has a timeseries for it.
        old_method_output: Optional[MethodOutput] = calculated_dataset_map.get(
            timeseries.DatasetName("SmoothedTests")
        )
        if old_method_output:
            calculated_dataset_recent_map[
                timeseries.DatasetName("SmoothedTestsAll")
            ] = old_method_output.all_output

        # Make a dataset object with one metric, containing for each region the timeseries
        # from the highest priority dataset that has recent data.
        test_positivity = timeseries.combined_datasets(
            {CommonFields.TEST_POSITIVITY: list(calculated_dataset_recent_map.values())}, {}
        )
        return AllMethods(
            all_methods_datasets=calculated_dataset_all_map, test_positivity=test_positivity
        )

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


def calculate_test_positivity(
    region_dataset: OneRegionTimeseriesDataset, lag_lookback: int = 7
) -> pd.Series:
    """Calculates positive test rate from combined data."""
    data = region_dataset.date_indexed
    positive_tests = series_utils.interpolate_stalled_and_missing_values(
        data[CommonFields.POSITIVE_TESTS]
    )
    negative_tests = series_utils.interpolate_stalled_and_missing_values(
        data[CommonFields.NEGATIVE_TESTS]
    )

    daily_negative_tests = negative_tests.diff()
    daily_positive_tests = positive_tests.diff()
    positive_smoothed = series_utils.smooth_with_rolling_average(daily_positive_tests)
    negative_smoothed = series_utils.smooth_with_rolling_average(
        daily_negative_tests, include_trailing_zeros=False
    )
    last_n_positive = positive_smoothed[-lag_lookback:]
    last_n_negative = negative_smoothed[-lag_lookback:]

    if any(last_n_positive) and last_n_negative.isna().all():
        return pd.Series([], dtype="float64")

    return positive_smoothed / (negative_smoothed + positive_smoothed)
