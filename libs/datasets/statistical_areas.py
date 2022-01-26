import collections
from dataclasses import dataclass
from typing import List
from typing import Mapping
import pandas as pd

from libs import pipeline
from libs.datasets.timeseries import MultiRegionDataset
from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import region_aggregation
from libs.pipeline import Region

CBSA_LIST_PATH = "data/census-msa/list1_2020.xls"


CBSA_COLUMN = "CBSA"


@dataclass
class CountyToCBSAAggregator:
    # Map from 5 digit county FIPS to 5 digit CBSA Code
    county_map: Mapping[str, str]

    # Map from 5 digit CBSA code to CBSA title
    cbsa_title_map: Mapping[str, str]

    aggregations: List[
        region_aggregation.StaticWeightedAverageAggregation
    ] = region_aggregation.WEIGHTED_AGGREGATIONS

    @property
    def county_to_cbsa_region_map(self) -> Mapping[Region, Region]:
        return {
            pipeline.Region.from_fips(fips): pipeline.Region.from_cbsa_code(cbsa_code)
            for fips, cbsa_code in self.county_map.items()
        }

    @property
    def cbsa_to_counties_region_map(self) -> Mapping[Region, List[Region]]:
        cbsa_to_counties = collections.defaultdict(list)
        for county, cbsa in self.county_to_cbsa_region_map.items():
            cbsa_to_counties[cbsa].append(county)
        return cbsa_to_counties

    def aggregate(
        self, dataset_in: MultiRegionDataset, reporting_ratio_required_to_aggregate=None
    ) -> MultiRegionDataset:
        """Returns a dataset of CBSA regions, created by aggregating counties in the input data."""
        return region_aggregation.aggregate_regions(
            dataset_in,
            self.county_to_cbsa_region_map,
            self.aggregations,
            reporting_ratio_required_to_aggregate=reporting_ratio_required_to_aggregate,
        )

    @staticmethod
    def from_local_public_data() -> "CountyToCBSAAggregator":
        """Creates a new object using data in the covid-data-public repo."""
        df = pd.read_excel(
            dataset_utils.LOCAL_PUBLIC_DATA_PATH / CBSA_LIST_PATH,
            header=2,
            convert_float=False,
            dtype={"FIPS State Code": str, "FIPS County Code": str},
        )
        df[CommonFields.FIPS] = df["FIPS State Code"] + df["FIPS County Code"]
        df = df.loc[df[CommonFields.FIPS].notna(), :]

        dups = df.duplicated(CommonFields.FIPS, keep=False)
        if dups.any():
            raise ValueError(f"Duplicate FIPS:\n{df.loc[dups, CommonFields.FIPS]}")

        county_map = df.set_index(CommonFields.FIPS)["CBSA Code"].to_dict()

        cbsa_title_map = (
            df.loc[:, ["CBSA Code", "CBSA Title"]]
            .drop_duplicates()
            .set_index("CBSA Code", verify_integrity=True)["CBSA Title"]
            .to_dict()
        )

        return CountyToCBSAAggregator(county_map=county_map, cbsa_title_map=cbsa_title_map)
