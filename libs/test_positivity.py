import dataclasses
import pathlib
from itertools import chain
from typing import List
from typing import Sequence
from typing import Set

import structlog

import pandas as pd
import numpy as np

from covidactnow.datapublic.common_fields import CommonFields, FieldName
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets.timeseries import MultiRegionTimeseriesDataset


_log = structlog.get_logger()


@dataclasses.dataclass
class Method:
    """A method of calculating test positivity"""

    name: str
    numerator: FieldName
    denominator: FieldName

    @property
    def columns(self) -> Set[str]:
        return {self.numerator, self.denominator}

    def calculate(self, delta_df: pd.DataFrame) -> pd.DataFrame:
        assert delta_df.columns.names == [CommonFields.DATE]
        assert delta_df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
        if not (self.columns <= set(delta_df.index.get_level_values(PdFields.VARIABLE))):
            return pd.DataFrame([])
        # delta_df has the field name as the first level of the index. delta_df.loc[field, :] returns a
        # DataFrame without the field label so operators such as `/` are calculated for each
        # region/state and date.
        return delta_df.loc[self.numerator, :] / delta_df.loc[self.denominator, :]


TEST_POSITIVITY_METHODS = (
    Method(
        "positiveCasesViral_totalTestEncountersViral",
        CommonFields.POSITIVE_CASES_VIRAL,
        CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL,
    ),
    Method(
        "positiveTestsViral_totalTestsViral",
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.TOTAL_TESTS_VIRAL,
    ),
    Method(
        "positiveCasesViral_totalTestsViral",
        CommonFields.POSITIVE_CASES_VIRAL,
        CommonFields.TOTAL_TESTS_VIRAL,
    ),
    Method(
        "positiveTests_totalTestsViral", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS_VIRAL
    ),
    Method(
        "positiveCasesViral_totalTestsPeopleViral",
        CommonFields.POSITIVE_CASES_VIRAL,
        CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
    ),
    Method(
        "positiveCasesViral_totalTestResults",
        CommonFields.POSITIVE_CASES_VIRAL,
        CommonFields.TOTAL_TESTS,
    ),
)


class NoTestPositivityResultsException(Exception):
    pass


@dataclasses.dataclass
class AllMethods:
    """The result of calculating all test positivity methods for all regions"""

    # Test positivity calculated in all valid methods for each region
    all_methods_timeseries: pd.DataFrame

    # Test positivity using the best available method for each region
    test_positivity: MultiRegionTimeseriesDataset

    @staticmethod
    def run(
        metrics_in: MultiRegionTimeseriesDataset,
        methods: Sequence[Method] = TEST_POSITIVITY_METHODS,
        diff_days: int = 7,
        recent_days: int = 14,
    ) -> "AllMethods":
        metrics_in_column_set = set(metrics_in.data.columns)
        ts_value_columns_set = set()
        for method in methods:
            if method.columns <= metrics_in_column_set:
                ts_value_columns_set.update(method.columns)
        if not ts_value_columns_set:
            raise ValueError(f"No data for test positivity")

        input_long = metrics_in.timeseries_long(list(ts_value_columns_set)).set_index(
            [PdFields.VARIABLE, CommonFields.LOCATION_ID, CommonFields.DATE]
        )[PdFields.VALUE]
        dates = input_long.index.get_level_values(CommonFields.DATE)
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
        # This calculates the difference only when the cumulative value is a real value `diff_days` apart.
        # It looks like our input data has few or no holes so this works well enough.
        diff_df = input_wide.diff(periods=diff_days, axis=1)

        results = {}
        for method in methods:
            result = method.calculate(diff_df)
            if not result.empty:
                results[method.name] = result
        if not results:
            raise NoTestPositivityResultsException()

        all_wide = (
            pd.concat(results, names=[PdFields.VARIABLE],)
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

    def write(self, csv_path: pathlib.Path):
        self.all_methods_timeseries.to_csv(
            csv_path, date_format="%Y-%m-%d", index=True, float_format="%.05g",
        )
