import logging
import pandas as pd

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset

_logger = logging.getLogger(__name__)


def fill_missing_county_with_city(row):
    """Fills in missing county data with city if available.

    """
    if pd.isnull(row.county) and not pd.isnull(row.city):
        if row.city == "New York City":
            return "New York"
        return row.city

    return row.county


class CDSDataset(data_source.DataSource):
    DATA_PATH = "data/cases-cds/timeseries-common.csv"
    SOURCE_NAME = "CDS"

    INDEX_FIELD_MAP = {f: f for f in TimeseriesDataset.INDEX_FIELDS}

    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.CASES,
            CommonFields.NEGATIVE_TESTS,
            CommonFields.POSITIVE_TESTS,
            CommonFields.POPULATION,
            CommonFields.CUMULATIVE_ICU,
            CommonFields.CUMULATIVE_HOSPITALIZED,
        }
    }

    @classmethod
    def local(cls) -> "CDSDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        df = common_df.read_csv(data_root / cls.DATA_PATH).reset_index()
        df[CommonFields.POSITIVE_TESTS] = df[CommonFields.CASES]
        # Column names are already CommonFields so don't need to rename, but do need to drop extra
        # columns that will fail NYC aggregation.
        return cls(cls._drop_unlisted_fields(df))
