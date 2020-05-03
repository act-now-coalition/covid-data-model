import pandas as pd
import requests
from libs.datasets import data_source
from libs.datasets import AggregationLevel
from libs.datasets import CommonFields
from libs.datasets import CommonIndexFields
from libs import us_state_abbrev


class StateInterventions(data_source.DataSource):
    INTERVENTIONS_URL = "https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json"

    class Fields(object):
        FIPS = "fips"
        STATE = "state"
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"
        INTERVENTION = "intervention"

    INDEX_FIELD_MAP = {
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.INTERVENTION: Fields.INTERVENTION,
    }

    @classmethod
    def standardize_data(cls, data) -> pd.DataFrame:
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.FIPS] = data[cls.Fields.STATE].map(
            us_state_abbrev.ABBREV_US_FIPS
        )
        return data

    @classmethod
    def local(cls):
        interventions = requests.get(cls.INTERVENTIONS_URL).json()

        columns = [cls.Fields.STATE, cls.Fields.INTERVENTION]

        data = pd.DataFrame(list(interventions.items()), columns=columns)
        data = cls.standardize_data(data)
        print(data)
        return cls(data)
