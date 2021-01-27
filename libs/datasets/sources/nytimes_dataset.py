from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import timeseries
from functools import lru_cache
from libs.datasets.custom_aggregations import ALL_NYC_REGIONS


class NYTimesDataset(data_source.DataSource):
    SOURCE_NAME = "NYTimes"

    COMMON_DF_CSV_PATH = "data/cases-nytimes/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        # NY Times has cases and deaths for all boroughs aggregated into 36061 / New York County.
        # Remove all the NYC data so that USAFacts (which reports each borough separately) is used.
        return super().make_dataset().remove_regions(ALL_NYC_REGIONS)
