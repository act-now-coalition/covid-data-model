from dataclasses import dataclass
from typing import Mapping
import pandas as pd

from libs import pipeline
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils


CBSA_LIST_PATH = "data/census-msa/list1_2020.xls"


CBSA_COLUMN = "CBSA"


@dataclass
class CountyToCBSAAggregator:
    # Map from 5 digit county FIPS to 5 digit CBSA Code
    county_map: Mapping[str, str]

    # Map from 5 digit CBSA code to CBSA title
    cbsa_title_map: Mapping[str, str]

    def aggregate(self, dataset_in: MultiRegionTimeseriesDataset) -> MultiRegionTimeseriesDataset:
        """Returns a dataset of CBSA regions, created by aggregating counties in the input data."""
        # Make a copy to avoid modifying the input. DataFrame.assign is an alternative but the API
        # doesn't work well here.
        df = dataset_in.data.copy()
        df[CBSA_COLUMN] = df[CommonFields.FIPS].map(self.county_map)

        # TODO(tom): Put the title in the data when it is clear where it goes in the returned value
        # TODO(tom): Handle dates with a subset of counties reporting.
        # TODO(tom): Handle data columns that don't make sense aggregated with sum.
        df_cbsa = df.groupby([CBSA_COLUMN, CommonFields.DATE], as_index=False).sum()
        df_cbsa[CommonFields.LOCATION_ID] = df_cbsa[CBSA_COLUMN].apply(pipeline.cbsa_to_location_id)

        return MultiRegionTimeseriesDataset(df_cbsa)

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

        county_map = df.set_index(CommonFields.FIPS)["CBSA Code"].to_dict()

        cbsa_title_map = (
            df.loc[:, ["CBSA Code", "CBSA Title"]]
            .drop_duplicates()
            .set_index("CBSA Code", verify_integrity=True)["CBSA Title"]
            .to_dict()
        )

        return CountyToCBSAAggregator(county_map=county_map, cbsa_title_map=cbsa_title_map)
