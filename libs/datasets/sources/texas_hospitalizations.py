import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils


class TexasHospitalizations(data_source.DataSource):
    DATA_PATH = "data/states/tx/tx_fips_hospitalizations.csv"
    SOURCE_NAME = "tx_hosp"

    # CMDC data reports values at both the state and county level. However, state values
    # are not reported until complete values from states are received.  The latest
    # row may contain some county data but not state data - instead of aggregating and returning
    # incomplete state data, we are choosing to not aggregate.
    FILL_MISSING_STATE_LEVEL_DATA = False

    INDEX_FIELD_MAP = {
        CommonFields.DATE: CommonFields.DATE,
        CommonFields.AGGREGATE_LEVEL: CommonFields.AGGREGATE_LEVEL,
        CommonFields.COUNTRY: CommonFields.COUNTRY,
        CommonFields.STATE: CommonFields.STATE,
        CommonFields.FIPS: CommonFields.FIPS,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CURRENT_HOSPITALIZED: CommonFields.CURRENT_HOSPITALIZED,
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(
            input_path, parse_dates=[CommonFields.DATE], dtype={CommonFields.FIPS: str}
        )
        return cls(data)
