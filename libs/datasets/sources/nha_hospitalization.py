import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel


class NevadaHospitalAssociationData(data_source.DataSource):
    DATA_PATH = "data/misc/nha_hospitalization_county.csv"
    SOURCE_NAME = "NHA"

    class Fields(object):
        DATE = "date"
        STATE = "state_code"
        COUNTY = "county_name"
        FIPS = "fips_code"
        ACCUTE_STAFFED = "acute_staffed"
        ACCUTE_OCCUPIED = "acute_occupied"
        ICU_STAFFED = "icu_staffed"
        ICU_OCCUPIED = "icu_occupied"
        VENTILATORS = "ventilators"
        VENTILATORS_OCCUPIED = "ventilators_occupied"
        COVID_CONFIRMED = "covid_confirmed"
        COVID_ACUTE_OCCUPIED = "covid_suspected"
        COVID_ICU_OCCUPIED = "covid_icu"
        COVID_VENTILATOR = "covid_ventilator"

        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    TIMESERIES_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.FIPS: Fields.FIPS,
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
        TimeseriesDataset.Fields.CURRENT_HOSPITALIZED: Fields.COVID_CONFIRMED,
        TimeseriesDataset.Fields.CURRENT_ICU: Fields.COVID_ICU_OCCUPIED,
        TimeseriesDataset.Fields.CURRENT_VENTILATED: Fields.COVID_VENTILATOR,
    }

    @classmethod
    def standardize_data(cls, data):
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.COUNTY.value

        return data

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(
            input_path,
            skiprows=1,
            parse_dates=[cls.Fields.DATE],
            dtype={cls.Fields.FIPS: str}
        )
        data = cls.standardize_data(data)

        return cls(data)
