import pandas as pd
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.common_fields import CommonFields, CommonIndexFields
import requests
import structlog
import us

log = structlog.get_logger()


class KinsaDataset(data_source.DataSource):
    """
    Wrapper class for accessing kinsa data via their api. 
    Data is not included in covid-data-public right now and is accessed directly from the kinsa API.

    """

    SOURCE_NAME = "Kinsa"

    class Fields(object):
        DATE = "date"
        COUNTY = "county"
        STATE = "state"
        FIPS = "fips"
        COUNTRY = "country"
        AGGREGATE_LEVEL = "aggregate_level"
        # documentation for the following fields can be found here
        # https://www.kinsahealth.co/kinsa-us-health-weather-public-api-documentation/
        OBSERVED_ILI = "observed_ili"
        ATYPICAL_ILI = "atypical_ili"
        ATYPICAL_ILI = "atypical_ili_delta"
        ANOMALY_FEVERS = "anomaly_fevers"
        FORECAST_EXPECTED = "forecast_expected"
        FORECAST_LOWER = "forecast_lower"
        FORECAST_UPPER = "forecast_upper"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    def __init__(self):
        data = self.load_all_us_kinsa_data()
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def load_all_us_kinsa_data(cls) -> pd.DataFrame:
        log.info("Downloading kinsa data from API.")
        df = pd.DataFrame()
        for state in us.STATES:
            r = requests.get(f"https://static.kinsahealth.com/{state.abbr}_data.json")
            if r.status_code == 200:
                records = r.json()
                df = df.append(
                    pd.DataFrame.from_records(data=records["data"], columns=records["columns"]),
                    ignore_index=True,
                )
            df["date"] = pd.to_datetime(df["date"])
        return df

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = "county"

        data = data.rename(columns={"region_id": cls.Fields.FIPS, "region_name": cls.Fields.COUNTY})
        data = data.drop(columns=["region_type"])
        return data
