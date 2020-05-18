from pathlib import Path
from typing import List

import pandas as pd
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields


class StatesData(data_source.DataSource):
    INPUT_FILES_GLOB = "data/states/*.csv"

    SOURCE_NAME = "states"

    INDEX_FIELD_MAP = {
        f: f
        for f in [
            CommonIndexFields.AGGREGATE_LEVEL,
            CommonIndexFields.COUNTRY,
            CommonIndexFields.STATE,
            CommonIndexFields.FIPS,
            CommonIndexFields.DATE,
        ]
    }

    COMMON_FIELD_MAP = {
        f: f
        for f in [
            CommonFields.CURRENT_ICU_TOTAL,
            CommonFields.CURRENT_VENTILATED,
            CommonFields.CURRENT_ICU,
        ]
    }

    @staticmethod
    def _load_dataframe(input_files: List[FilePath]) -> pd.DataFrame:
        # Consider factoring out with jhu_dataset.JHUDataset.__init__
        loaded_df = []
        for input_filename in input_files:
            df = pd.read_csv(
                input_filename,
                parse_dates=[CommonIndexFields.DATE],
                dtype={CommonIndexFields.FIPS: str},
            )
            # TODO: Check that index fields are always set and unique
            loaded_df.append(df)
        return pd.concat(loaded_df)

    @classmethod
    def local(cls):
        input_files = dataset_utils.LOCAL_PUBLIC_DATA_PATH.glob(cls.INPUT_FILES_GLOB)
        df = StatesData._load_dataframe(input_files)
        return cls(df)
