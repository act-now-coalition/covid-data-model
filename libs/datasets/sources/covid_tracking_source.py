from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CovidTrackingDataSource(data_source.DataSource):
    SOURCE_NAME = "covid_tracking"

    COMMON_DF_CSV_PATH = "data/covid-tracking/timeseries.csv"

    # Keep in sync with update_covid_tracking_data in the covid-data-public repo.
    # DataSource objects must have a map from CommonFields to fields in the source file.
    # For this source the conversion is done in the covid-data-public repo so the map
    # here doesn't represent any field renaming.
    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.DEATHS,
            CommonFields.CURRENT_HOSPITALIZED,
            CommonFields.CURRENT_ICU,
            CommonFields.CURRENT_VENTILATED,
            CommonFields.CUMULATIVE_HOSPITALIZED,
            CommonFields.CUMULATIVE_ICU,
            CommonFields.POSITIVE_TESTS,
            CommonFields.NEGATIVE_TESTS,
        }
    }

    @classmethod
    def local(cls) -> "CovidTrackingDataSource":
        data = common_df.read_csv(cls.INPUT_PATH).reset_index()
        # Column names are already CommonFields so don't need to rename
        return cls(data)
