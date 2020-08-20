from typing import Tuple

import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset


class CovidCountyDataDataSource(data_source.DataSource):
    DATA_PATH = "data/cases-covid-county-data/timeseries-common.csv"
    SOURCE_NAME = "Valorum"

    # Covid County Data reports values at both the state and county level. However, state values
    # are not reported until complete values from states are received.  The latest
    # row may contain some county data but not state data - instead of aggregating and returning
    # incomplete state data, we are choosing to not aggregate.
    FILL_MISSING_STATE_LEVEL_DATA = False

    INDEX_FIELD_MAP = {f: f for f in TimeseriesDataset.INDEX_FIELDS}

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
        tests_and_cases = df.eval(
            "negative_tests.isna() & positive_tests.isna() & total_tests.notna() & cases.notna()"
        )
        missing_neg = df.eval(
            "negative_tests.isna() & positive_tests.notna() & total_tests.notna() & cases.notna()"
        )
        missing_pos = df.eval(
            "negative_tests.notna() & positive_tests.isna() & total_tests.notna() & cases.notna()"
        )

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
