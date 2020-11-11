from dataclasses import dataclass
from typing import Mapping
import pandas as pd

from libs import pipeline
from libs.datasets.timeseries import MultiRegionDataset
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

    def aggregate(self, dataset_in: MultiRegionDataset) -> MultiRegionDataset:
        """Returns a dataset of CBSA regions, created by aggregating counties in the input data."""
        return MultiRegionDataset.from_timeseries_df(
            self._aggregate_fips_df(dataset_in.data_with_fips, groupby_date=True)
        ).add_latest_df(
            # No need to reset latest_data_with_fips LOCATION_ID index because FIPS is used.
            self._aggregate_fips_df(dataset_in.latest_data_with_fips, groupby_date=False),
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

    def _aggregate_fips_df(self, df: pd.DataFrame, groupby_date: bool) -> pd.DataFrame:
        # Make a copy to avoid modifying the input. DataFrame.assign is an alternative but the API
        # doesn't work well here.
        df = df.copy()
        df[CBSA_COLUMN] = df[CommonFields.FIPS].map(self.county_map)

        # TODO(tom): Put the title in the data when it is clear where it goes in the returned value
        # TODO(tom): Handle dates with a subset of counties reporting.
        # TODO(tom): Handle data columns that don't make sense aggregated with sum.
        groupby_columns = [CBSA_COLUMN, CommonFields.DATE] if groupby_date else [CBSA_COLUMN]
        df_cbsa = df.groupby(groupby_columns, as_index=False).sum()
        df_cbsa[CommonFields.LOCATION_ID] = df_cbsa[CBSA_COLUMN].apply(pipeline.cbsa_to_location_id)
        df_cbsa = df_cbsa.drop(columns=[CBSA_COLUMN])

        return df_cbsa
