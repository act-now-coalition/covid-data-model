import collections
from dataclasses import dataclass
from typing import Dict, List
from typing import Mapping
import pandas as pd
import dataclasses

from libs import pipeline
from libs.datasets.timeseries import MultiRegionDataset
from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import region_aggregation
from libs.pipeline import Region
from libs.datasets.dataset_utils import AggregationLevel

CBSA_LIST_PATH = "data/misc/list1_2020.xls"

CBSA_COLUMN = "CBSA"

HSA_FIELDS_MAPPING = {
    CommonFields.STAFFED_BEDS: CommonFields.STAFFED_BEDS_HSA,
    CommonFields.HOSPITAL_BEDS_IN_USE_ANY: CommonFields.HOSPITAL_BEDS_IN_USE_ANY_HSA,
    CommonFields.CURRENT_HOSPITALIZED: CommonFields.CURRENT_HOSPITALIZED_HSA,
    CommonFields.ICU_BEDS: CommonFields.ICU_BEDS_HSA,
    CommonFields.CURRENT_ICU: CommonFields.CURRENT_ICU_HSA,
    CommonFields.CURRENT_ICU_TOTAL: CommonFields.CURRENT_ICU_TOTAL_HSA,
    CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID_HSA,
}


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
        """Creates a new object using data in the data/ dir."""
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
class CountyToHSAAggregator:
    county_map: Mapping[str, str]

    # Mapping of county regions -> hsa regions
    @property
    def county_to_hsa_region_map(self) -> Mapping[Region, Region]:
        return {
            Region.from_fips(fips): Region.from_hsa_code(hsa_code)
            for fips, hsa_code in self.county_map.items()
        }

    @property
    def hsa_to_counties_region_map(self) -> Mapping[Region, List[Region]]:
        hsa_to_counties = collections.defaultdict(list)
        for county, hsa in self.county_to_hsa_region_map.items():
            hsa_to_counties[hsa].append(county)
        return hsa_to_counties

    @staticmethod
    def from_local_data() -> "CountyToHSAAggregator":
        """Creates a new object using the HSA data stored in data/."""
        hsa_df = pd.read_csv(
            dataset_utils.HSA_LIST_PATH, dtype={CommonFields.HSA: str, CommonFields.FIPS: str}
        )
        hsa_df[CommonFields.HSA] = hsa_df[CommonFields.HSA].str.zfill(3)
        hsa_df = hsa_df[[CommonFields.FIPS, CommonFields.HSA]]
        hsa_raw_map = dict(hsa_df.values)
        return CountyToHSAAggregator(county_map=hsa_raw_map)

    def aggregate(
        self,
        dataset_in: MultiRegionDataset,
        fields_to_aggregate: Dict[CommonFields, CommonFields] = HSA_FIELDS_MAPPING,
        restrict_to_current_state: bool = False,
    ) -> MultiRegionDataset:
        """Create new fields by aggregating county-level data into HSA level data. 
        
        Args:
            dataset_in: MultiRegionDataset with fields to aggregate.
            fields_to_aggregate: Mapping of names of columns to aggregate to names of resulting columns.
        """

        # TODO PARALLEL: make a multiregion_dataset class explicitly for 1 state/place.
        # check that restriction is only used when one state is present
        # if restrict_to_current_state and len(dataset_in.location_ids) != 1:
        # raise ValueError(
        # f"restrict_to_current_state flag can only be used on single-location datasets. {}"
        # )

        # Only aggregate county-level data for specified fields
        counties = dataset_in.get_subset(aggregation_level=AggregationLevel.COUNTY)
        columns_to_aggregate = [
            col for col in counties.timeseries_bucketed.columns if col in fields_to_aggregate.keys()
        ]
        counties_ts: pd.DataFrame = counties.timeseries_bucketed.loc[:, columns_to_aggregate]
        counties_selected_ds = dataclasses.replace(counties, timeseries_bucketed=counties_ts)

        # No special aggregations are needed because all fields track beds or people.
        hsa_ts: pd.DataFrame = region_aggregation.aggregate_regions(
            counties_selected_ds, self.county_to_hsa_region_map, aggregations=[],
        ).timeseries_bucketed

        # Map counties back onto HSAs.
        hsa_to_counties_location_id_map = {
            hsa.location_id: [county.location_id for county in counties]
            for hsa, counties in self.hsa_to_counties_region_map.items()
        }
        hsa_ts[CommonFields.LOCATION_ID] = hsa_ts.index.get_level_values(
            CommonFields.LOCATION_ID
        ).map(hsa_to_counties_location_id_map)

        # Create a row for each county in each HSA using HSA data.
        # NOTE: Every county in an HSA will have data for these HSA fields as long as
        # any other county in the same HSA has data, regardless of whether or
        # not we have actually collected data for that county.
        aggregated_ts = hsa_ts.explode(CommonFields.LOCATION_ID)

        # Here we are dropping the index level "location_id", which are the HSAs,
        # and replacing them with the newly added "location_id" column, which are the counties.
        aggregated_ts = aggregated_ts.droplevel(CommonFields.LOCATION_ID)
        aggregated_ts = aggregated_ts.set_index(CommonFields.LOCATION_ID, append=True).sort_index()
        aggregated_ts = aggregated_ts.rename(columns=fields_to_aggregate)

        assert not set(aggregated_ts.columns) & set(dataset_in.timeseries_bucketed.columns)
        out_ts = dataset_in.timeseries_bucketed.combine_first(aggregated_ts)
        out_ds = dataclasses.replace(dataset_in, timeseries_bucketed=out_ts)

        # TODO PARALLEL: Clean this up. assert that there's only one state, etc...
        if restrict_to_current_state and len(dataset_in.location_ids) != 0:
            state = Region.from_state(Region.from_location_id(dataset_in.location_ids[0]).state)
            out_ds, other_locs = out_ds.partition_by_region(
                include=[
                    state,
                    pipeline.RegionMask(states=[state.state], level=AggregationLevel.COUNTY),
                ]
            )
            return out_ds
        return out_ds
