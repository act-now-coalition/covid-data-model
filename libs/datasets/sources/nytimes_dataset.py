from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from functools import lru_cache
from libs.datasets.custom_aggregations import ALL_NYC_REGIONS


class NYTimesDataset(data_source.DataSource):
    SOURCE_NAME = "NYTimes"

    DATA_PATH = "data/cases-nytimes/timeseries-common.csv"

    HAS_AGGREGATED_NYC_BOROUGH = True

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.CASES, CommonFields.DEATHS,}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls(cls._rename_to_common_fields(data))

    @lru_cache(None)
    def multi_region_dataset(self) -> MultiRegionDataset:
        # NY Times has cases and deaths for all boroughs aggregated into 36061 / New York County.
        # Remove all the NYC data so that USAFacts (which reports each borough separately) is used.
        return super().multi_region_dataset().remove_regions(ALL_NYC_REGIONS)
