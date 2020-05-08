import logging
import pandas as pd
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields

_logger = logging.getLogger(__name__)


class CovidTrackingDataSource(data_source.DataSource):
    DATA_PATH = "data/covid-tracking/covid_tracking_states.csv"
    SOURCE_NAME = "covid_tracking"

    class Fields(object):
        # ISO 8601 date of when these values were valid.
        DATE_CHECKED = "dateChecked"
        STATE = "state"
        # Total cumulative positive test results.
        POSITIVE_TESTS = "positive"
        # Increase from the day before.
        POSITIVE_INCREASE = "positiveIncrease"
        # Total cumulative negative test results.
        NEGATIVE_TESTS = "negative"
        # Increase from the day before.
        NEGATIVE_INCREASE = "negativeIncrease"
        # Total cumulative number of people hospitalized.
        TOTAL_HOSPITALIZED = "hospitalized"
        # Total cumulative number of people hospitalized.
        CURRENT_HOSPITALIZED = "hospitalizedCurrently"
        # Increase from the day before.
        HOSPITALIZED_INCREASE = "hospitalizedIncrease"
        # Total cumulative number of people that have died.
        DEATHS = "death"
        # Increase from the day before.
        DEATH_INCREASE = "deathIncrease"
        # Tests that have been submitted to a lab but no results have been reported yet.
        PENDING = "pending"
        # Calculated value (positive + negative) of total test results.
        TOTAL_TEST_RESULTS = "totalTestResults"
        # Increase from the day before.
        TOTAL_TEST_RESULTS_INCREASE = "totalTestResultsIncrease"

        IN_ICU_CURRENTLY = "inIcuCurrently"
        IN_ICU_CUMULATIVE = "inIcuCumulative"

        IN_ICU_CURRENTLY = "inIcuCurrently"
        TOTAL_IN_ICU = "inIcuCumulative"

        ON_VENTILATOR_CURRENTLY = "onVentilatorCurrently"
        TOTAL_ON_VENTILATOR = "onVentilatorCumulative"

        COUNTRY = "country"
        COUNTY = "county"
        DATE = "date"
        AGGREGATE_LEVEL = "aggregate_level"
        FIPS = "fips"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.DEATHS: Fields.DEATHS,
        CommonFields.CURRENT_HOSPITALIZED: Fields.CURRENT_HOSPITALIZED,
        CommonFields.CURRENT_ICU: Fields.IN_ICU_CURRENTLY,
        CommonFields.CURRENT_VENTILATED: Fields.ON_VENTILATOR_CURRENTLY,
        CommonFields.CUMULATIVE_HOSPITALIZED: Fields.TOTAL_HOSPITALIZED,
        CommonFields.CUMULATIVE_ICU: Fields.TOTAL_IN_ICU,
        CommonFields.POSITIVE_TESTS: Fields.POSITIVE_TESTS,
        CommonFields.NEGATIVE_TESTS: Fields.NEGATIVE_TESTS,
    }

    TESTS_ONLY_FIELDS = [
        Fields.DATE,
        Fields.POSITIVE_TESTS,
        Fields.NEGATIVE_TESTS,
    ]

    TEST_FIELDS = [
        Fields.DATE,
        Fields.STATE,
        Fields.POSITIVE_TESTS,
        Fields.NEGATIVE_TESTS,
        Fields.POSITIVE_INCREASE,
        Fields.NEGATIVE_INCREASE,
        Fields.AGGREGATE_LEVEL,
    ]

    def __init__(self, input_path):
        data = pd.read_csv(
            input_path,
            parse_dates=[self.Fields.DATE_CHECKED],
            dtype={self.Fields.FIPS: str},
        )
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def local(cls) -> "CovidTrackingDataSource":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data[cls.Fields.COUNTY] = None
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        # Date checked is the time that the data is actually updated.
        # assigning the date field as the date floor of that day.
        data[cls.Fields.DATE] = (
            data[cls.Fields.DATE_CHECKED].dt.tz_localize(None).dt.floor("D")
        )

        dtypes = {
            cls.Fields.POSITIVE_TESTS: "Int64",
            cls.Fields.NEGATIVE_TESTS: "Int64",
            cls.Fields.POSITIVE_INCREASE: "Int64",
            cls.Fields.NEGATIVE_INCREASE: "Int64",
        }

        data = data.astype(dtypes)

        # Dropping PR because of bad data
        # TODO(chris): Handle this in a more sane way.
        data = data.loc[data.state != "PR", :]

        # must stay true: positive + negative  ==  total
        assert (
            data[cls.Fields.POSITIVE_TESTS] + data[cls.Fields.NEGATIVE_TESTS]
            == data[cls.Fields.TOTAL_TEST_RESULTS]
        ).all()

        # must stay true: positive chage + negative change ==  total change
        assert (
            data[cls.Fields.POSITIVE_INCREASE] + data[cls.Fields.NEGATIVE_INCREASE]
            == data[cls.Fields.TOTAL_TEST_RESULTS_INCREASE]
        ).all()

        # TODO implement assertion to check for shift, as sliced by geo
        # df['totalTestResults'] - df['totalTestResultsIncrease']  ==  df['totalTestResults'].shift(-1)
        return data
