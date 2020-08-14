import pandas as pd

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
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        negative_tests  positive_tests  total_tests  cases  count
        True            False           False        True   6
        False           False           True         False  39
        True            True            True         False  245      Do nothing, leave cases blank
        False           False           False        False  361      Do nothing, leave cases blank
        True            False           True         True   627      pos = tests - neg
        False           True            False        True   1079     Do nothing, no test data
        True            True            False        True   7064     Do nothing, total_tests not used
        False           True            True         True   25769    neg = tests - pos
        False           False           True         True   32684    pos = cases, neg = tests - pos
        True            True            True         True   61951    Have everything already
        False           False           False        True   508238   No test data
        """
        # Make a copy to avoid modifying the argument.
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

        df[CommonFields.POSITIVE_TESTS].mask(tests_and_cases, df[CommonFields.CASES], inplace=True)
        df[CommonFields.NEGATIVE_TESTS].mask(
            tests_and_cases, df[CommonFields.TOTAL_TESTS] - df[CommonFields.CASES], inplace=True
        )

        df[CommonFields.NEGATIVE_TESTS].mask(
            missing_neg,
            df[CommonFields.TOTAL_TESTS] - df[CommonFields.POSITIVE_TESTS],
            inplace=True,
        )

        df[CommonFields.POSITIVE_TESTS].mask(
            missing_pos,
            df[CommonFields.TOTAL_TESTS] - df[CommonFields.NEGATIVE_TESTS],
            inplace=True,
        )

        return df

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        data = cls.standardize_data(data)
        # Column names are already CommonFields so don't need to rename
        return cls(data)
