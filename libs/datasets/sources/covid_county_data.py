import dataclasses
from functools import lru_cache
from typing import Tuple

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df

from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset

# 2020/11/01: By manual comparison of test positivity calculated via Covid County Data vs CMS
# and local dashboards where available, these states have data that seems less credible than
# our CMS data source.
DISABLED_TEST_POSITIVITY_STATES = ["AL", "DE", "FL", "IN", "IA", "MD", "ND", "NE", "PA", "WI", "RI"]


class CovidCountyDataDataSource(data_source.DataSource):
    DATA_PATH = "data/cases-covid-county-data/timeseries-common.csv"
    SOURCE_NAME = "Valorum"

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    # Keep in sync with update_covid_county_data.py in the covid-data-public repo.
    # DataSource objects must have a map from CommonFields to fields in the source file.
    # For this source the conversion is done in the covid-data-public repo so the map
    # here doesn't represent any field renaming.
    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.CASES,
            CommonFields.DEATHS,
            CommonFields.CURRENT_ICU,
            CommonFields.CURRENT_ICU_TOTAL,
            CommonFields.NEGATIVE_TESTS,
            CommonFields.POSITIVE_TESTS,
            CommonFields.STAFFED_BEDS,
            CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
            CommonFields.CURRENT_VENTILATED,
            CommonFields.CURRENT_HOSPITALIZED,
            CommonFields.ICU_BEDS,
        }
    }

    @classmethod
    def synthesize_test_metrics(cls, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Synthesize testing metrics where they can be calculated from other metrics.

        This function processes each date independently. The fix applied depends on what testing and cases
        metrics have a real value on that date. The following table lists the number of FIPS-date rows with
        a given combination of metrics and the treatment applied by this function. The table was generated
        with:

        columns_to_check = ["negative_tests", "positive_tests", "total_tests", "cases"]
        df[columns_to_check].notna().groupby(columns_to_check).size().reset_index(name="count").sort_values("count")

        negative_tests  positive_tests  total_tests  cases  count    Treatment / reason for doing nothing
        True            False           False        True   6        Very rare
        False           False           True         False  39       Very rare
        True            True            True         False  245      Testing data already set
        False           False           False        False  361      No data available
        True            False           True         True   627      pos = tests - neg
        False           True            False        True   1079     Testing data not available
        True            True            False        True   7064     Pos and neg tests set, total_tests not used
        False           True            True         True   25769    neg = tests - pos
        False           False           True         True   32684    pos = cases, neg = tests - pos
        True            True            True         True   61951    Have everything already
        False           False           False        True   508238   No test data
        """

        # Make a copy to avoid modifying the argument when using mask with inplace=True.
        df = data.copy()
        tests_and_cases = (
            df["negative_tests"].isna()
            & df["positive_tests"].isna()
            & df["total_tests"].notna()
            & df["cases"].notna()
        )
        missing_neg = (
            df["negative_tests"].isna()
            & df["positive_tests"].notna()
            & df["total_tests"].notna()
            & df["cases"].notna()
        )
        missing_pos = (
            df["negative_tests"].notna()
            & df["positive_tests"].isna()
            & df["total_tests"].notna()
            & df["cases"].notna()
        )
        disabled = df.eval("state in @DISABLED_TEST_POSITIVITY_STATES")

        # Keep the same order of rows in `provenance` as `df` so that the same masks can be used when
        # writing it.
        provenance = df.loc[:, [CommonFields.FIPS, CommonFields.DATE]]
        provenance[CommonFields.POSITIVE_TESTS] = "none"
        provenance[CommonFields.NEGATIVE_TESTS] = "none"

        # Using where/mask is suggested by https://stackoverflow.com/a/36067403/341400
        df[CommonFields.POSITIVE_TESTS].mask(tests_and_cases, df[CommonFields.CASES], inplace=True)
        df[CommonFields.NEGATIVE_TESTS].mask(
            tests_and_cases, df[CommonFields.TOTAL_TESTS] - df[CommonFields.CASES], inplace=True
        )
        provenance[CommonFields.POSITIVE_TESTS].mask(
            tests_and_cases, "tests_and_cases", inplace=True
        )
        provenance[CommonFields.NEGATIVE_TESTS].mask(
            tests_and_cases, "tests_and_cases", inplace=True
        )

        df[CommonFields.NEGATIVE_TESTS].mask(
            missing_neg,
            df[CommonFields.TOTAL_TESTS] - df[CommonFields.POSITIVE_TESTS],
            inplace=True,
        )
        provenance[CommonFields.NEGATIVE_TESTS].mask(missing_neg, "missing_neg", inplace=True)

        df[CommonFields.POSITIVE_TESTS].mask(
            missing_pos,
            df[CommonFields.TOTAL_TESTS] - df[CommonFields.NEGATIVE_TESTS],
            inplace=True,
        )
        provenance[CommonFields.POSITIVE_TESTS].mask(missing_pos, "missing_pos", inplace=True)

        # Remove disabled locations.
        df[CommonFields.POSITIVE_TESTS].mask(disabled, None, inplace=True)
        provenance[CommonFields.POSITIVE_TESTS].mask(disabled, "disabled", inplace=True)
        df[CommonFields.NEGATIVE_TESTS].mask(disabled, None, inplace=True)
        provenance[CommonFields.NEGATIVE_TESTS].mask(disabled, "disabled", inplace=True)

        # preventing a circular import by importing combined datasets here.
        # TODO(chris): Move provenance_wide_metrics_to_series to fix the circular import.
        from libs.datasets import combined_datasets

        provenance_series = combined_datasets.provenance_wide_metrics_to_series(
            provenance.set_index([CommonFields.FIPS, CommonFields.DATE]), structlog.get_logger()
        )

        return df, provenance_series

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        data, provenance = cls.synthesize_test_metrics(data)
        # Column names are already CommonFields so don't need to rename
        return cls(data, provenance=provenance)

    @lru_cache(None)
    def multi_region_dataset(self) -> MultiRegionDataset:
        dataset = super().multi_region_dataset()

        # Add latest ICU_BEDS from timeseries to static.
        # Hacked out of _timeseries_latest_values
        # timeseries is already sorted by DATE with the latest at the bottom.
        long = dataset.timeseries[CommonFields.ICU_BEDS].droplevel(CommonFields.DATE).dropna()
        # `long` Index with LOCATION_ID. Keep only the last
        # row with each index to get the last value for each date.
        unduplicated_and_last_mask = ~long.index.duplicated(keep="last")
        latest_icu_beds = long.loc[unduplicated_and_last_mask]
        static_df = dataset.static.copy()
        static_df.insert(len(static_df.columns), CommonFields.ICU_BEDS, latest_icu_beds)

        return dataclasses.replace(dataset, static=static_df)
