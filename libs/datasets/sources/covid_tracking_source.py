import logging
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel

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

    TIMESERIES_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.FIPS: Fields.FIPS,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.CURRENT_HOSPITALIZED: Fields.CURRENT_HOSPITALIZED,
        TimeseriesDataset.Fields.CURRENT_ICU: Fields.IN_ICU_CURRENTLY,
        TimeseriesDataset.Fields.CUMULATIVE_HOSPITALIZED: Fields.TOTAL_HOSPITALIZED,
        TimeseriesDataset.Fields.CUMULATIVE_ICU: Fields.TOTAL_IN_ICU,
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
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
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE_CHECKED])
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
        data[cls.Fields.DATE] = data[cls.Fields.DATE_CHECKED].dt.tz_localize(None).dt.floor("D")

        dtypes = {
            cls.Fields.POSITIVE_TESTS: "Int64",
            cls.Fields.NEGATIVE_TESTS: "Int64",
            cls.Fields.POSITIVE_INCREASE: "Int64",
            cls.Fields.NEGATIVE_INCREASE: "Int64",
        }

        data = data.astype(dtypes)

        # Covid Tracking source has the state level fips, however none of the other
        # data sources have state level fips, and the generic code may implicitly assume
        # it doesn't.  I would like to add a state level fips (maybe for example a state fips code
        # of 45 being 45000), but it's not there, so in the meantime we're setting fips to null so
        # as not to confuse downstream data.
        data[cls.Fields.FIPS] = None

        # must stay true: positive + negative  ==  total
        assert (
            data[cls.Fields.POSITIVE_TESTS]
            + data[cls.Fields.NEGATIVE_TESTS] == data[cls.Fields.TOTAL_TEST_RESULTS]
        ).all()

        # must stay true: positive chage + negative change ==  total change
        assert (
            data[cls.Fields.POSITIVE_INCREASE]
            + data[cls.Fields.NEGATIVE_INCREASE]
            == data[cls.Fields.TOTAL_TEST_RESULTS_INCREASE]
        ).all()

        nevada_data = cls._load_nevada_override_data()
        data = cls._add_nevada_data(data, nevada_data)

        # TODO implement assertion to check for shift, as sliced by geo
        # df['totalTestResults'] - df['totalTestResultsIncrease']  ==  df['totalTestResults'].shift(-1)
        return data

    @classmethod
    def _load_nevada_override_data(cls):
        from libs.datasets import NevadaHospitalAssociationData

        data = NevadaHospitalAssociationData.local().timeseries(fill_na=False).data
        columns_to_include = []
        for timeseries_column, ct_column in cls.TIMESERIES_FIELD_MAP.items():
            if timeseries_column in data.columns:
                columns_to_include.append(ct_column)

        data = data.rename(cls.TIMESERIES_FIELD_MAP, axis=1)
        return data[columns_to_include]

    @classmethod
    def _add_nevada_data(cls, data, nevada_data):
        """Adds nevada data, replacing any state or county level values that match index.

        Args:
            data: Covid tracking data
            nevada_data: Nevada specific override data.

        Returns: Updated dataframe with
        """
        # NOTE(chris): This logic will most likely work as we have more hospitalization data
        # numbers that will override covid tracking data.
        matching_index_group = [
            cls.Fields.DATE,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        data = data.set_index(matching_index_group)
        nevada_data = nevada_data.set_index(matching_index_group)
        # Sort indices so that we have chunks of equal length in the
        # correct order so that we can splice in values from nevada data.
        data = data.sort_index()
        nevada_data = nevada_data.sort_index()
        data_in_nevada = data.index.isin(nevada_data.index)
        nevada_in_data = nevada_data.index.isin(data.index)

        if not sum(data_in_nevada) == sum(nevada_in_data):
            raise ValueError("Number of rows should be the for data to replace")

        # Fill in values with data that matches index in nevada data.
        data.loc[data_in_nevada, nevada_data.columns] = nevada_data.loc[nevada_in_data, :]

        # Combine updated data with rows not present in covid tracking data.
        return pd.concat([
            data,
            nevada_data[~nevada_in_data]
        ]).reset_index()
