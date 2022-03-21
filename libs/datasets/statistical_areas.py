import collections
from dataclasses import dataclass
from typing import List
from typing import Mapping
import pandas as pd

import enum
import dataclasses
from libs import pipeline
from libs.datasets.timeseries import MultiRegionDataset
from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import region_aggregation
from libs.pipeline import Region
from libs.datasets.sources.fips_population import FIPSPopulation

from datapublic.common_fields import FieldName
from datapublic.common_fields import GetByValueMixin
from datapublic.common_fields import ValueAsStrMixin

from tests import test_helpers

CBSA_LIST_PATH = "data/misc/list1_2020.xls"
HSA_LIST_PATH = "data/cdc_hsa_mapping.csv"


CBSA_COLUMN = "CBSA"


@enum.unique
class HSAFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Fields pertaining to the HSA data"""

    HSA = "hsa"
    HSA_POPULATION = "hsa_population"
    COUNTY_PERCENT_OF_HSA_POPULATION = "county_percent_pop"


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
            dataset_utils.REPO_ROOT / CBSA_LIST_PATH,
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


@dataclass
class CountyHSADisaggregator:
    county_map: Mapping[str, str]

    # Mapping of county location_ids -> hsa codes
    @property
    def county_to_hsa_region_map(self) -> Mapping[str, str]:
        return {
            Region.from_fips(fips).location_id: hsa_code
            for fips, hsa_code in self.county_map.items()
        }

    @staticmethod
    def from_local_data() -> "CountyHSADisaggregator":
        """Creates a new object using the HSA data stored in data/."""
        hsa_raw_map = dict(pd.read_csv(HSA_LIST_PATH, dtype={"HSA": str, "FIPS": str}).values)
        return CountyHSADisaggregator(county_map=hsa_raw_map)

    def spread_data(
        self,
        dataset_in: MultiRegionDataset = None,
        fields_to_disaggregate: List[CommonFields] = [
            CommonFields.STAFFED_BEDS,
            CommonFields.HOSPITAL_ADMISSIONS_COVID_7D,
        ],
    ) -> MultiRegionDataset:

        if not dataset_in:  # For testing only
            dataset_in = MultiRegionDataset.from_csv(
                dataset_utils.DATA_DIRECTORY / "hhs.csv"
            )  # temp file
            # dataset_in = test_helpers.build_dataset(
            #     {
            #         Region.from_fips("01053"): {
            #             CommonFields.HOSPITAL_ADMISSIONS_COVID_7D: [10, 20, 30],
            #             CommonFields.NEW_CASES: [4, 5, 6],
            #         },
            #         Region.from_fips("01035"): {
            #             CommonFields.HOSPITAL_ADMISSIONS_COVID_7D: [10, 20, 30],
            #             CommonFields.NEW_CASES: [1, 2, 3],
            #         },
            #     },
            #     start_date="2022-03-13",
            # )

        fields_to_keep = [
            col for col in dataset_in.timeseries.columns if col not in fields_to_disaggregate
        ]
        ds_in_ts = dataset_in.timeseries.loc[:, fields_to_disaggregate]
        ds_ts_other = dataset_in.timeseries.loc[:, fields_to_keep]

        populations = FIPSPopulation.make_dataset().static
        ds_in_ts: pd.DataFrame = ds_in_ts.join(populations)

        ds_in_ts[HSAFields.HSA] = ds_in_ts.index.get_level_values("location_id").map(
            self.county_to_hsa_region_map
        )

        # Calculate HSA-level data only for selected fields.
        hsa_ts = (
            ds_in_ts.groupby([HSAFields.HSA, CommonFields.DATE])
            .sum()
            .rename(columns={CommonFields.POPULATION: HSAFields.HSA_POPULATION})
            .loc[:, fields_to_disaggregate + [HSAFields.HSA_POPULATION]]
            .reset_index()
        )

        # Map HSA data to each county
        ds_in_ts = ds_in_ts.loc[:, [CommonFields.POPULATION, HSAFields.HSA]].reset_index()
        ds_in_ts = ds_in_ts.merge(hsa_ts, on=[CommonFields.DATE, HSAFields.HSA])
        ds_in_ts[HSAFields.COUNTY_PERCENT_OF_HSA_POPULATION] = (
            ds_in_ts[CommonFields.POPULATION] / ds_in_ts[HSAFields.HSA_POPULATION]
        )

        # Distribute HSA data to counties based on county populations
        for var in fields_to_disaggregate:
            ds_in_ts[var] = ds_in_ts[var] * ds_in_ts[HSAFields.COUNTY_PERCENT_OF_HSA_POPULATION]

        # Set and sort the index to group locations to make index monotonic increasing.
        # Index locations became unordered after unsetting and resetting the index.
        ds_in_ts = ds_in_ts.set_index([CommonFields.LOCATION_ID, CommonFields.DATE]).sort_index()

        # Combine disaggregated fields with untouched fields.
        out_ts = ds_in_ts.join(ds_ts_other)

        # Drop all the columns used in the calculations
        out_ts = out_ts.loc[:, dataset_in.timeseries.columns]
        return dataclasses.replace(dataset_in, timeseries=out_ts, timeseries_bucketed=None)
