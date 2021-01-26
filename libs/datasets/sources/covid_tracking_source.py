from functools import lru_cache

from covidactnow.datapublic import common_df

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class CovidTrackingDataSource(data_source.DataSource):
    COMMON_DF_CSV_PATH = "data/covid-tracking/timeseries.csv"
    SOURCE_NAME = "covid_tracking"

    # Keep in sync with update_covid_tracking_data in the covid-data-public repo.
    # DataSource objects must have a map from CommonFields to fields in the source file.
    # For this source the conversion is done in the covid-data-public repo so the map
    # here doesn't represent any field renaming.
    EXPECTED_FIELDS = [
        CommonFields.DEATHS,
        CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.CURRENT_ICU,
        CommonFields.CURRENT_VENTILATED,
        CommonFields.CUMULATIVE_HOSPITALIZED,
        CommonFields.CUMULATIVE_ICU,
        CommonFields.POSITIVE_TESTS,
        CommonFields.NEGATIVE_TESTS,
    ]
