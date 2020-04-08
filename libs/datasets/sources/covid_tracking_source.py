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
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

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
        data[cls.Fields.DATE] = data[cls.Fields.DATE_CHECKED].dt.floor("D")

        # Covid Tracking source has the state level fips, however none of the other
        # data sources have state level fips, and the generic code may implicitly assume
        # it doesn't.  I would like to add a state level fips (maybe for example a state fips code
        # of 45 being 45000), but it's not there, so in the meantime we're setting fips to null so
        # as not to confuse downstream data.
        data[cls.Fields.FIPS] = None

        # Since we're using this data for hospitalized data only, only returning
        # values with hospitalization data.  I think as the use cases of this data source
        # expand, we may not want to drop. For context, as of 4/6 144/1620 rows contained
        # hospitalization data.
        return data[data[cls.Fields.CURRENT_HOSPITALIZED].notnull()]
