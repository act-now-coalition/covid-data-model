import pathlib
from functools import lru_cache

import pandas as pd

from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.us_state_abbrev import ABBREV_US_FIPS, ABBREV_US_UNKNOWN_COUNTY_FIPS
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.statistical_areas import CountyToHSAAggregator

CURRENT_FOLDER = pathlib.Path(__file__).parent


def get_location_level(location_id):
    location = Region.from_location_id(location_id)
    # TODO(sean): Weed out locations who's levels cant be determined.
    # In this case it's fine because we only want to select counties.
    try:
        return location.level
    except NotImplementedError:
        return None


class FIPSPopulation(data_source.DataSource):
    """FIPS data from US Gov census predictions + fips list.

    https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
    https://www.census.gov/geographies/reference-files/2018/demo/popest/2018-fips.html

    Puerto Rico-specific county data:
    https://www.census.gov/data/datasets/time-series/demo/popest/2010s-total-puerto-rico-municipios.html
    """

    FILE_PATH = "data/misc/fips_population.csv"

    SOURCE_TYPE = "FIPS"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        FIPS = "fips"
        POPULATION = "population"

        # Added in standardize data.
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    EXPECTED_FIELDS = [CommonFields.POPULATION, CommonFields.COUNTY]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.REPO_ROOT
        data = pd.read_csv(data_root / cls.FILE_PATH, dtype={"fips": str})
        data["fips"] = data.fips.str.zfill(5)
        data = cls.standardize_data(data)
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # Add Missing
        unknown_fips = []

        for state in data.state.unique():
            row = {
                cls.Fields.STATE: state,
                cls.Fields.FIPS: ABBREV_US_UNKNOWN_COUNTY_FIPS[state],
                cls.Fields.POPULATION: None,
                cls.Fields.COUNTY: "Unknown",
            }
            unknown_fips.append(row)

        data = data.append(unknown_fips)
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.COUNTY.value
        data[cls.Fields.COUNTRY] = "USA"

        states_aggregated = dataset_utils.aggregate_and_get_nonmatching(
            data,
            [cls.Fields.COUNTRY, cls.Fields.STATE, cls.Fields.AGGREGATE_LEVEL],
            AggregationLevel.COUNTY,
            AggregationLevel.STATE,
        ).reset_index()
        states_aggregated[cls.Fields.FIPS] = states_aggregated[cls.Fields.STATE].map(ABBREV_US_FIPS)
        states_aggregated[cls.Fields.COUNTY] = None

        us_row = {
            CommonFields.FIPS: "0",
            CommonFields.COUNTRY: "USA",
            CommonFields.POPULATION: states_aggregated[CommonFields.POPULATION].sum(),
        }
        country_aggregated = pd.DataFrame.from_records([us_row])

        common_fields_data = pd.concat([data, states_aggregated, country_aggregated])
        return common_fields_data


class HSAPopulation(data_source.DataSource):
    """HSA number and population for each US county. 
    
    """

    EXPECTED_FIELDS = [CommonFields.HSA_POPULATION, CommonFields.HSA]

    SOURCE_TYPE = "HSA"

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        location_map = CountyToHSAAggregator.from_local_data().county_to_hsa_region_map
        populations = FIPSPopulation.make_dataset().static
        county_index = populations.index.map(get_location_level) == AggregationLevel.COUNTY
        counties = populations[county_index].copy()  # copy to remove SettingWithCopy error

        # Map HSAs to counties
        counties[CommonFields.HSA] = counties.index.map(location_map)
        counties = counties.reset_index()

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

        data = counties.loc[:, [CommonFields.HSA_POPULATION, CommonFields.FIPS, CommonFields.HSA]]
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)
