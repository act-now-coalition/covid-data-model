from functools import lru_cache
import pandas as pd

from datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.statistical_areas import CountyToHSAAggregator
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.dataset_utils import HSA_LIST_PATH


def get_location_level(location_id):
    location = Region.from_location_id(location_id)
    # TODO(sean): Agg. levels for iso2:us#iso2:us-dc#fips:11999 and iso2:us#iso2:us-fl#fips:12999
    # can't be determined. Find out why these counties are in the population files.
    try:
        return location.level
    except NotImplementedError:
        return None


class HSAPopulation(data_source.DataSource):
    """HSA number and HSA population for each US county. 
    
    HSA populations are calculated as the sum of the populations of the contained counties. 

    For more information on HSAs see:
        https://github.com/covid-projections/covid-data-model/blob/main/data/misc/README.md
    """

    EXPECTED_FIELDS = [CommonFields.HSA_POPULATION, CommonFields.HSA]

    SOURCE_TYPE = "HSA"

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        location_map = CountyToHSAAggregator.from_local_data().county_to_hsa_region_map
        location_map = {county.location_id: hsa.location_id for county, hsa in location_map.items()}
        populations = FIPSPopulation.make_dataset().static
        county_index = populations.index.map(get_location_level) == AggregationLevel.COUNTY
        counties = populations[county_index].copy()  # copy to remove SettingWithCopy error
        # hsa_names: pd.DataFrame = (
        #     pd.read_csv(HSA_LIST_PATH, dtype={CommonFields.HSA: str})
        #     .loc[:, [CommonFields.HSA, CommonFields.HSA_NAME]]
        #     .drop_duplicates()
        #     .assign(hsa_name=lambda row: row[CommonFields.HSA_NAME].fillna("Unknown HSA Name"))
        # )

        # Map HSAs to counties
        counties[CommonFields.HSA] = counties.index.map(location_map)
        counties = counties.reset_index()
        return counties

        # Calculate HSA populations and join back onto counties
        hsas = (
            counties.groupby(CommonFields.HSA)
            .sum()
            .reset_index()
            .rename(columns={CommonFields.POPULATION: CommonFields.HSA_POPULATION})
        )
        counties = counties.merge(hsas, on=CommonFields.HSA)

        # Get county and HSA FIPS from location IDs
        counties[CommonFields.FIPS] = counties[CommonFields.LOCATION_ID].map(
            lambda loc: Region.from_location_id(loc).fips
        )
        counties[CommonFields.HSA] = counties[CommonFields.HSA].map(
            lambda loc: Region.from_location_id(loc).fips
        )
        # Add HSA names to dataframe
        counties.merge(hsa_names, how="left", on=CommonFields.HSA).reset_index()

        data = counties.loc[
            :,
            [
                CommonFields.HSA_POPULATION,
                CommonFields.FIPS,
                CommonFields.HSA,
                CommonFields.HSA_NAME,
            ],
        ]
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)
