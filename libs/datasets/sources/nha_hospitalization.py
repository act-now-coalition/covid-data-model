import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.common_fields import CommonIndexFields


class NevadaHospitalAssociationData(data_source.DataSource):
    DATA_PATH = "data/states/nv/nha_hospitalization_county.csv"
    SOURCE_NAME = "NHA"

    class Fields(object):
        FIPS = "fips"
        DATE = "date"
        COUNTY = "county"
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

        CURRENT_HOSPITALIZED_TOTAL = "current_hospitalized_total"
        AGGREGATE_LEVEL = "aggregate_level"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: CommonFields.COUNTRY,
        CommonIndexFields.STATE: CommonFields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CURRENT_HOSPITALIZED: Fields.COVID_CONFIRMED,
        CommonFields.CURRENT_ICU: Fields.COVID_ICU_OCCUPIED,
        CommonFields.CURRENT_VENTILATED: Fields.COVID_VENTILATOR,
        CommonFields.ICU_BEDS: Fields.ICU_STAFFED,
        CommonFields.CURRENT_ICU_TOTAL: Fields.ICU_OCCUPIED,
        CommonFields.CURRENT_HOSPITALIZED_TOTAL: Fields.CURRENT_HOSPITALIZED_TOTAL,
    }

    @classmethod
    def standardize_data(cls, data):
        data[CommonFields.COUNTRY] = "USA"
        data[CommonFields.STATE] = "NV"
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.COUNTY.value
        data[cls.Fields.CURRENT_HOSPITALIZED_TOTAL] = (
            data[cls.Fields.ACCUTE_OCCUPIED] + data[cls.Fields.ICU_OCCUPIED]
        )
        return data

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(input_path, parse_dates=[cls.Fields.DATE], dtype={cls.Fields.FIPS: str})
        data = cls.standardize_data(data)

        return cls(cls._rename_to_common_fields(data))
